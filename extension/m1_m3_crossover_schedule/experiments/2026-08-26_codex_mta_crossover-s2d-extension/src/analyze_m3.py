#!/usr/bin/env python3
"""Frozen confirmation-validation analysis and official-test freeze creator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


DATASETS = ("emnist_balanced", "k49", "svhn")
ARCHITECTURES = ("plain_transformer", "local_hybrid")
TIERS = ("compact", "deep", "wide")
BOOTSTRAP_SEED = 260828
BOOTSTRAP_RESAMPLES = 20_000
MARGIN = 0.005


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "raw").glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("status") == "completed":
                    rows.append(row)
    return rows


def ttt(log: list[dict], threshold: float) -> int:
    for row in log:
        if float(row["val_accuracy"]) >= threshold:
            return int(row["epoch"])
    return 31


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(pvalues[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def bootstrap(per_dataset: dict[str, float], seed: int) -> list[float]:
    values = np.asarray([per_dataset[d] for d in DATASETS])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))].mean(1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def write_csv(path: Path, rows: list[dict]):
    cols = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--validation-report", type=Path, required=True)
    ap.add_argument("--m2-result", type=Path, required=True)
    ap.add_argument("--m3-lock", type=Path, required=True)
    ap.add_argument("--switch-map", type=Path, required=True)
    ap.add_argument("--switch-map-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    lock = json.loads(args.m3_lock.read_text(encoding="utf-8"))
    m2 = json.loads(args.m2_result.read_text(encoding="utf-8"))
    switch_manifest = json.loads(args.switch_map_manifest.read_text(encoding="utf-8"))
    simple = lock["strongest_simple_schedule"]
    if validation.get("status") != "PASS" or validation.get("phase") != "m3":
        raise RuntimeError("M3 validation integrity gate not passed")
    if lock.get("status") != "DISCOVERY_FROZEN_FOR_M3":
        raise RuntimeError("M3 discovery lock is not frozen")
    if lock.get("m2_result_sha256") != sha256(args.m2_result):
        raise RuntimeError("M2 result hash does not match the discovery lock")
    if switch_manifest.get("status") != "M3_SWITCH_MAP_FROZEN" \
            or switch_manifest.get("switch_map_sha256") != sha256(args.switch_map):
        raise RuntimeError("M3 switch map does not match its freeze manifest")
    if switch_manifest.get("m3_discovery_lock_sha256") != sha256(args.m3_lock):
        raise RuntimeError("M3 switch-map manifest is bound to a different discovery lock")
    methods = ("softmax", "topk_softmax_025", "mechanistic_switch", simple)
    rows = read_rows(args.root)
    grouped = {
        (r["dataset"], r["architecture"], r["tier"], r["method"], int(r["seed"])): r
        for r in rows
    }
    expected = len(DATASETS) * len(ARCHITECTURES) * len(TIERS) * len(methods) * 10
    if len(grouped) != expected:
        raise RuntimeError(f"expected {expected} unique confirmation rows, found {len(grouped)}")

    pairs = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for tier in TIERS:
                for seed in range(10, 20):
                    dense = grouped[(dataset, architecture, tier, "softmax", seed)]
                    dense_final = float(dense["epoch_log"][-1]["val_accuracy"])
                    threshold = 0.95 * dense_final
                    dense_ttt = ttt(dense["epoch_log"], threshold)
                    for method in methods:
                        row = grouped[(dataset, architecture, tier, method, seed)]
                        method_ttt = ttt(row["epoch_log"], threshold)
                        final = float(row["epoch_log"][-1]["val_accuracy"])
                        pairs.append({
                            "dataset": dataset, "architecture": architecture, "tier": tier,
                            "seed": seed, "method": method, "time_to_target": method_ttt,
                            "dense_time_to_target": dense_ttt,
                            "relative_reduction_vs_dense": (dense_ttt - method_ttt) / dense_ttt,
                            "final_accuracy": final, "dense_final_accuracy": dense_final,
                            "final_difference_vs_dense": final - dense_final,
                        })

    idx = {(r["dataset"], r["architecture"], r["tier"], r["seed"], r["method"]): r for r in pairs}
    mech = [r for r in pairs if r["method"] == "mechanistic_switch"]
    simple_reductions = []
    for row in mech:
        key = (row["dataset"], row["architecture"], row["tier"], row["seed"])
        base = idx[(*key, simple)]
        simple_reductions.append((base["time_to_target"] - row["time_to_target"]) / base["time_to_target"])

    by_dataset_reduction = {
        d: float(np.mean([r["relative_reduction_vs_dense"] for r in mech if r["dataset"] == d]))
        for d in DATASETS
    }
    by_dataset_simple = {}
    for d in DATASETS:
        values = []
        for row in [x for x in mech if x["dataset"] == d]:
            key = (row["dataset"], row["architecture"], row["tier"], row["seed"])
            base = idx[(*key, simple)]
            values.append((base["time_to_target"] - row["time_to_target"]) / base["time_to_target"])
        by_dataset_simple[d] = float(np.mean(values))
    by_dataset_final = {
        d: float(np.mean([r["final_difference_vs_dense"] for r in mech if r["dataset"] == d]))
        for d in DATASETS
    }
    ci_dense = bootstrap(by_dataset_reduction, BOOTSTRAP_SEED)
    ci_simple = bootstrap(by_dataset_simple, BOOTSTRAP_SEED + 1)
    ci_final = bootstrap(by_dataset_final, BOOTSTRAP_SEED + 2)

    by_arch = {
        a: float(np.mean([r["relative_reduction_vs_dense"] for r in mech if r["architecture"] == a]))
        for a in ARCHITECTURES
    }
    favorable_datasets = [d for d, v in by_dataset_reduction.items() if v > 0]
    favorable_architectures = [a for a, v in by_arch.items() if v > 0]

    pvals, task_stats = [], []
    for dataset in DATASETS:
        diffs = np.asarray([r["final_difference_vs_dense"] for r in mech if r["dataset"] == dataset])
        test = stats.ttest_1samp(diffs, popmean=0.0, alternative="two-sided")
        pvals.append(float(test.pvalue))
        task_stats.append({"dataset": dataset, "mean_difference": float(diffs.mean()), "p_raw": float(test.pvalue)})
    adjusted = holm(pvals)
    for row, p in zip(task_stats, adjusted):
        row["p_holm"] = p
        row["supported_harm_over_margin"] = row["mean_difference"] < -MARGIN and p < 0.05
    harmful = [r["dataset"] for r in task_stats if r["supported_harm_over_margin"]]

    criterion_time = ci_dense[0] > 0 and ci_simple[0] > 0
    criterion_noninferiority = ci_final[0] > -MARGIN and not harmful
    criterion_direction = len(favorable_datasets) >= 2 and set(favorable_architectures) == set(ARCHITECTURES)
    m3_pass = all((criterion_time, criterion_noninferiority, criterion_direction))
    combined_pass = m2.get("status") == "M2_DISCOVERY_PASS_PENDING_M3" and m3_pass
    result = {
        "status": "M3_VALIDATION_PASS_TEST_PENDING" if m3_pass else "M3_CONFIRMATION_KILLED",
        "combined_method_status": "DISCOVERY_AND_VALIDATION_PASS_TEST_PENDING" if combined_pass else "METHOD_CLAIM_KILLED",
        "strongest_simple_schedule": simple,
        "mean_relative_reduction_by_dataset": by_dataset_reduction,
        "mean_relative_reduction_by_architecture": by_arch,
        "dataset_bootstrap_ci95_reduction_vs_dense": ci_dense,
        "dataset_bootstrap_ci95_reduction_vs_simple": ci_simple,
        "dataset_bootstrap_ci95_final_difference": ci_final,
        "favorable_datasets": favorable_datasets,
        "favorable_architectures": favorable_architectures,
        "task_final_tests": task_stats,
        "criteria": {
            "time_to_target": criterion_time,
            "noninferiority": criterion_noninferiority,
            "direction_breadth": criterion_direction,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "m3_validation_pairwise.csv", pairs)
    write_csv(args.out / "m3_validation_task_tests.csv", task_stats)
    (args.out / "m3_validation_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "status": "M3_VALIDATION_FROZEN_OFFICIAL_TEST_AUTHORIZED",
        "strongest_simple_schedule": simple,
        "methods": list(methods),
        "validation_report_sha256": sha256(args.validation_report),
        "validation_result_sha256": sha256(args.out / "m3_validation_result.json"),
        "switch_map_sha256": sha256(args.switch_map),
        "switch_map_manifest_sha256": sha256(args.switch_map_manifest),
        "discovery_lock_sha256": sha256(args.m3_lock),
        "m2_result_sha256": sha256(args.m2_result),
        "analysis_source_sha256": sha256(Path(__file__)),
    }
    (args.out / "m3_test_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if m3_pass else 2


if __name__ == "__main__":
    sys.exit(main())
