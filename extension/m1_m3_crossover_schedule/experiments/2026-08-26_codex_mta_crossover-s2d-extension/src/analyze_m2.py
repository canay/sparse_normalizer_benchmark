#!/usr/bin/env python3
"""Frozen M2 time-to-target, safety, and breadth analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASETS = (
    "mnist", "fashion_mnist", "kmnist", "cifar10", "cifar100",
    "twenty_news", "synthetic_marker", "emnist_letters", "emnist_digits", "usps",
)
ARCHITECTURES = ("plain_transformer", "local_hybrid")
TIERS = ("compact", "deep", "wide")
METHODS = (
    "softmax", "topk_softmax_025", "median_switch", "early_accuracy_switch",
    "entropy_switch", "mechanistic_switch", "headwise_adaptive_entmax", "cosine_alpha",
)
SIMPLE_TIE_ORDER = ("median_switch", "early_accuracy_switch", "entropy_switch")
BOOTSTRAP_SEED = 260827
BOOTSTRAP_RESAMPLES = 20_000
NONINFERIORITY_MARGIN = 0.005


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
                    row["_source"] = path.name
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def time_to_target(epoch_log: list[dict], threshold: float) -> int:
    for row in epoch_log:
        if float(row["val_accuracy"]) >= threshold:
            return int(row["epoch"])
    return 31


def transition_instability(epoch_log: list[dict], switch_epoch: int | None) -> float:
    if switch_epoch is None or switch_epoch < 2:
        return float("nan")
    values = [float(r["val_accuracy"]) for r in epoch_log]
    start = max(1, switch_epoch - 1)
    end = min(30, switch_epoch + 2)
    diffs = [abs(values[e] - values[e - 1]) for e in range(start, end)]
    return max(diffs) if diffs else float("nan")


def bootstrap_ci(per_dataset: dict[str, float], seed: int) -> list[float]:
    values = np.asarray([per_dataset[d] for d in DATASETS], dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def require_m1_manifest(manifest_path: Path, result_path: Path, switch_map_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("M1 analysis manifest has no output hash map")
    expected = {
        "m1_result.json": sha256(result_path),
        "m1_switch_map.json": sha256(switch_map_path),
    }
    for name, digest in expected.items():
        if outputs.get(name) != digest:
            raise RuntimeError(f"M1 analysis manifest hash mismatch: {name}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--validation-report", type=Path, required=True)
    ap.add_argument("--m1-result", type=Path, required=True)
    ap.add_argument("--m1-switch-map", type=Path, required=True)
    ap.add_argument("--m1-analysis-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("phase") != "m2":
        raise RuntimeError("M2 integrity validator must pass before analysis")
    require_m1_manifest(args.m1_analysis_manifest, args.m1_result, args.m1_switch_map)
    m1_result = json.loads(args.m1_result.read_text(encoding="utf-8"))
    rows = read_rows(args.root)
    grouped = {
        (r["dataset"], r["architecture"], r["tier"], r["method"], int(r["seed"])): r
        for r in rows
    }
    expected = len(DATASETS) * len(ARCHITECTURES) * len(TIERS) * len(METHODS) * 5
    if len(grouped) != expected:
        raise RuntimeError(f"expected {expected} completed unique rows, found {len(grouped)}")

    pairwise = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for tier in TIERS:
                for seed in range(5, 10):
                    dense = grouped[(dataset, architecture, tier, "softmax", seed)]
                    dense_final = float(dense["epoch_log"][-1]["val_accuracy"])
                    threshold = 0.95 * dense_final
                    dense_ttt = time_to_target(dense["epoch_log"], threshold)
                    for method in METHODS:
                        row = grouped[(dataset, architecture, tier, method, seed)]
                        ttt = time_to_target(row["epoch_log"], threshold)
                        final = float(row["epoch_log"][-1]["val_accuracy"])
                        pairwise.append({
                            "dataset": dataset,
                            "architecture": architecture,
                            "tier": tier,
                            "seed": seed,
                            "method": method,
                            "dense_target_accuracy": threshold,
                            "time_to_target": ttt,
                            "dense_time_to_target": dense_ttt,
                            "relative_ttt_reduction_vs_dense": (dense_ttt - ttt) / dense_ttt,
                            "final_accuracy": final,
                            "dense_final_accuracy": dense_final,
                            "final_difference_vs_dense": final - dense_final,
                            "learning_curve_mean_accuracy": float(np.mean([
                                float(e["val_accuracy"]) for e in row["epoch_log"]
                            ])),
                            "switch_epoch": row.get("switch_epoch"),
                            "transition_instability": transition_instability(
                                row["epoch_log"], row.get("switch_epoch")
                            ),
                        })

    macro_mean_ttt = {}
    for method in METHODS:
        dataset_means = []
        for dataset in DATASETS:
            vals = [r["time_to_target"] for r in pairwise if r["method"] == method and r["dataset"] == dataset]
            dataset_means.append(float(np.mean(vals)))
        macro_mean_ttt[method] = float(np.mean(dataset_means))
    strongest_simple = min(
        SIMPLE_TIE_ORDER,
        key=lambda method: (macro_mean_ttt[method], SIMPLE_TIE_ORDER.index(method)),
    )

    index = {
        (r["dataset"], r["architecture"], r["tier"], r["seed"], r["method"]): r
        for r in pairwise
    }
    mechanistic_rows = [r for r in pairwise if r["method"] == "mechanistic_switch"]
    rel_dense = []
    rel_simple = []
    diff_dense_by_dataset: dict[str, list[float]] = defaultdict(list)
    diff_simple_by_dataset: dict[str, list[float]] = defaultdict(list)
    final_diff_by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in mechanistic_rows:
        key = (row["dataset"], row["architecture"], row["tier"], row["seed"])
        simple = index[(*key, strongest_simple)]
        rd = float(row["relative_ttt_reduction_vs_dense"])
        rs = (float(simple["time_to_target"]) - float(row["time_to_target"])) / float(simple["time_to_target"])
        rel_dense.append(rd)
        rel_simple.append(rs)
        diff_dense_by_dataset[row["dataset"]].append(rd)
        diff_simple_by_dataset[row["dataset"]].append(rs)
        final_diff_by_dataset[row["dataset"]].append(float(row["final_difference_vs_dense"]))

    cluster_dense = {d: float(np.mean(v)) for d, v in diff_dense_by_dataset.items()}
    cluster_simple = {d: float(np.mean(v)) for d, v in diff_simple_by_dataset.items()}
    cluster_final = {d: float(np.mean(v)) for d, v in final_diff_by_dataset.items()}
    ci_dense = bootstrap_ci(cluster_dense, BOOTSTRAP_SEED)
    ci_simple = bootstrap_ci(cluster_simple, BOOTSTRAP_SEED + 1)
    ci_final = bootstrap_ci(cluster_final, BOOTSTRAP_SEED + 2)

    architecture_benefit = {}
    for architecture in ARCHITECTURES:
        vals = [r["relative_ttt_reduction_vs_dense"] for r in mechanistic_rows if r["architecture"] == architecture]
        architecture_benefit[architecture] = float(np.mean(vals))
    tier_benefit = {}
    for tier in TIERS:
        vals = [r["relative_ttt_reduction_vs_dense"] for r in mechanistic_rows if r["tier"] == tier]
        tier_benefit[tier] = float(np.mean(vals))

    median_rel_dense = float(np.median(rel_dense))
    median_rel_simple = float(np.median(rel_simple))
    criterion_1 = median_rel_dense >= 0.10
    criterion_2 = ci_dense[0] > 0 and ci_simple[0] > 0
    criterion_3 = ci_final[0] > -NONINFERIORITY_MARGIN
    criterion_4 = all(v > 0 for v in architecture_benefit.values()) and sum(v > 0 for v in tier_benefit.values()) >= 2
    discovery_pass = all((criterion_1, criterion_2, criterion_3, criterion_4))

    result = {
        "status": "M2_DISCOVERY_PASS_PENDING_M3" if discovery_pass else "M2_METHOD_CLAIM_KILLED",
        "operation_id": "snd-crossover-mechanism-s2d-extension-20260826",
        "m1_status": m1_result.get("status"),
        "strongest_simple_schedule": strongest_simple,
        "macro_mean_time_to_target": macro_mean_ttt,
        "median_relative_reduction_vs_dense": median_rel_dense,
        "median_relative_reduction_vs_strongest_simple": median_rel_simple,
        "dataset_bootstrap_ci95_reduction_vs_dense": ci_dense,
        "dataset_bootstrap_ci95_reduction_vs_strongest_simple": ci_simple,
        "dataset_bootstrap_ci95_final_accuracy_difference": ci_final,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "architecture_mean_relative_benefit": architecture_benefit,
        "tier_mean_relative_benefit": tier_benefit,
        "criteria": {
            "median_reduction_at_least_10_percent": criterion_1,
            "bootstrap_reductions_above_zero": criterion_2,
            "full_budget_noninferiority": criterion_3,
            "architecture_capacity_breadth": criterion_4,
            "confirmation": "pending",
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "m2_pairwise_endpoints.csv", pairwise)
    (args.out / "m2_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lock = {
        "status": "DISCOVERY_FROZEN_FOR_M3",
        "strongest_simple_schedule": strongest_simple,
        "mechanistic_schedule": "mechanistic_switch",
        "m1_switch_map_sha256": sha256(args.m1_switch_map),
        "m1_result_sha256": sha256(args.m1_result),
        "m1_analysis_manifest_sha256": sha256(args.m1_analysis_manifest),
        "m2_validation_sha256": sha256(args.validation_report),
        "m2_result_sha256": sha256(args.out / "m2_result.json"),
        "analysis_source_sha256": sha256(Path(__file__)),
    }
    (args.out / "m3_discovery_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "analysis_source_sha256": sha256(Path(__file__)),
        "validation_report_sha256": sha256(args.validation_report),
        "outputs": {
            name: sha256(args.out / name) for name in (
                "m2_pairwise_endpoints.csv", "m2_result.json", "m3_discovery_lock.json"
            )
        },
    }
    (args.out / "m2_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if discovery_pass else 2


if __name__ == "__main__":
    sys.exit(main())
