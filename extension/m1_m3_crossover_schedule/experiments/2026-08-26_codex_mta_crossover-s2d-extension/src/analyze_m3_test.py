#!/usr/bin/env python3
"""One-read official-test confirmation analysis; no reselection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


DATASETS = ("emnist_balanced", "k49", "svhn")
BOOTSTRAP_SEED = 260829
BOOTSTRAP_RESAMPLES = 20_000
MARGIN = 0.005


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def holm(pvalues):
    order = np.argsort(pvalues)
    out = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * float(pvalues[idx]))
        out[idx] = min(1.0, running)
    return out.tolist()


def read_rows(root: Path):
    rows = []
    for path in sorted((root / "raw").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") == "completed":
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--validation-report", type=Path, required=True)
    ap.add_argument("--test-freeze", type=Path, required=True)
    ap.add_argument("--m2-result", type=Path, required=True)
    ap.add_argument("--m3-result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    freeze = json.loads(args.test_freeze.read_text(encoding="utf-8"))
    m2 = json.loads(args.m2_result.read_text(encoding="utf-8"))
    m3 = json.loads(args.m3_result.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("phase") != "m3_test":
        raise RuntimeError("M3 official-test integrity gate not passed")
    if freeze.get("status") != "M3_VALIDATION_FROZEN_OFFICIAL_TEST_AUTHORIZED":
        raise RuntimeError("M3 official-test freeze is not authorized")
    if freeze.get("m2_result_sha256") != sha256(args.m2_result):
        raise RuntimeError("official-test freeze is bound to a different M2 result")
    if freeze.get("validation_result_sha256") != sha256(args.m3_result):
        raise RuntimeError("official-test freeze is bound to a different M3 validation result")
    rows = read_rows(args.root)
    simple = freeze["strongest_simple_schedule"]
    expected_methods = {"softmax", "topk_softmax_025", "mechanistic_switch", simple}
    if set(freeze.get("methods", [])) != expected_methods:
        raise RuntimeError("official-test freeze contains an invalid method family")
    index = {(r["dataset"], r["architecture"], r["tier"], int(r["seed"]), r["method"]): r for r in rows}
    expected_cells = len(DATASETS) * 2 * 3 * len(expected_methods) * 10
    if len(index) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} unique official-test rows, found {len(index)}")
    diffs_by_dataset = {d: [] for d in DATASETS}
    simple_diffs = {d: [] for d in DATASETS}
    pair_rows = []
    for dataset in DATASETS:
        for architecture in ("plain_transformer", "local_hybrid"):
            for tier in ("compact", "deep", "wide"):
                for seed in range(10, 20):
                    key = (dataset, architecture, tier, seed)
                    dense = index[(*key, "softmax")]
                    mech = index[(*key, "mechanistic_switch")]
                    simp = index[(*key, simple)]
                    diff = float(mech["test_accuracy"]) - float(dense["test_accuracy"])
                    sdiff = float(mech["test_accuracy"]) - float(simp["test_accuracy"])
                    diffs_by_dataset[dataset].append(diff)
                    simple_diffs[dataset].append(sdiff)
                    pair_rows.append({
                        "dataset": dataset, "architecture": architecture, "tier": tier, "seed": seed,
                        "mechanistic_minus_dense": diff, "mechanistic_minus_simple": sdiff,
                    })
    means = {d: float(np.mean(v)) for d, v in diffs_by_dataset.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray([means[d] for d in DATASETS])
    draws = values[rng.integers(0, 3, size=(BOOTSTRAP_RESAMPLES, 3))].mean(1)
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    tests, pvalues = [], []
    for d in DATASETS:
        arr = np.asarray(diffs_by_dataset[d])
        t = stats.ttest_1samp(arr, 0.0, alternative="two-sided")
        tests.append({"dataset": d, "mean_difference": float(arr.mean()), "p_raw": float(t.pvalue)})
        pvalues.append(float(t.pvalue))
    for row, p in zip(tests, holm(pvalues)):
        row["p_holm"] = p
        row["supported_harm_over_margin"] = row["mean_difference"] < -MARGIN and p < 0.05
    test_pass = ci[0] > -MARGIN and not any(r["supported_harm_over_margin"] for r in tests)
    combined = (
        m2.get("status") == "M2_DISCOVERY_PASS_PENDING_M3"
        and m3.get("status") == "M3_VALIDATION_PASS_TEST_PENDING"
        and test_pass
    )
    result = {
        "status": "M3_TEST_PASS" if test_pass else "M3_TEST_DISCONFIRMED",
        "final_method_status": "SUPPORTED" if combined else "KILLED",
        "strongest_simple_schedule": simple,
        "test_difference_by_dataset": means,
        "dataset_bootstrap_ci95_test_difference": ci,
        "task_tests": tests,
        "test_freeze_sha256": sha256(args.test_freeze),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "m3_test_pairwise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        w.writeheader(); w.writerows(pair_rows)
    (args.out / "m3_test_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "analysis_source_sha256": sha256(Path(__file__)),
        "validation_report_sha256": sha256(args.validation_report),
        "test_freeze_sha256": sha256(args.test_freeze),
        "m2_result_sha256": sha256(args.m2_result),
        "m3_result_sha256": sha256(args.m3_result),
        "outputs": {
            "m3_test_pairwise.csv": sha256(args.out / "m3_test_pairwise.csv"),
            "m3_test_result.json": sha256(args.out / "m3_test_result.json"),
        },
    }
    (args.out / "m3_test_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if test_pass else 2


if __name__ == "__main__":
    sys.exit(main())
