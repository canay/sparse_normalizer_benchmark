#!/usr/bin/env python3
"""Frozen M1 crossover-target and leave-one-dataset-out predictor analysis."""

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
METHODS = ("softmax", "topk_softmax_025")
SEEDS = tuple(range(5))
FEATURE_SETS = {
    "mechanistic": (
        "accuracy_level", "accuracy_slope", "entropy_level", "entropy_slope",
        "margin_level", "margin_slope", "tail_level", "tail_slope",
        "gradient_level", "gradient_slope", "log_classes", "log_seq_len",
        "architecture_local", "blocks", "embed_dim",
    ),
    "early_accuracy": ("accuracy_level", "accuracy_slope"),
    "entropy": ("entropy_level", "entropy_slope"),
    "metadata": ("log_classes", "log_seq_len", "architecture_local", "blocks", "embed_dim"),
}
BASELINE_TIE_ORDER = ("constant", "early_accuracy", "entropy", "metadata")
BOOTSTRAP_SEED = 260826
BOOTSTRAP_RESAMPLES = 20_000


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
            for line_no, line in enumerate(f, 1):
                row = json.loads(line)
                if row.get("status") != "completed":
                    continue
                row["_source"] = path.name
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def mean_epoch(rows: list[dict], epoch: int, field: str) -> float:
    vals = [float(r["epoch_log"][epoch - 1][field]) for r in rows]
    if len(vals) != len(SEEDS) or not all(math.isfinite(x) for x in vals):
        raise RuntimeError(f"bad {field} at epoch {epoch}")
    return float(np.mean(vals))


def level_slope(rows: list[dict], field: str) -> tuple[float, float]:
    vals = [mean_epoch(rows, epoch, field) for epoch in (1, 2, 3)]
    return vals[2], (vals[2] - vals[0]) / 2.0


def build_cells(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["architecture"], row["tier"], row["method"])
        grouped[key].append(row)
    cells = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for tier in TIERS:
                dense = grouped[(dataset, architecture, tier, "softmax")]
                sparse = grouped[(dataset, architecture, tier, "topk_softmax_025")]
                if sorted(int(r["seed"]) for r in dense) != list(SEEDS):
                    raise RuntimeError(f"dense seed mismatch: {dataset}/{architecture}/{tier}")
                if sorted(int(r["seed"]) for r in sparse) != list(SEEDS):
                    raise RuntimeError(f"top-k seed mismatch: {dataset}/{architecture}/{tier}")
                dense = sorted(dense, key=lambda r: int(r["seed"]))
                sparse = sorted(sparse, key=lambda r: int(r["seed"]))
                delta = []
                for epoch in range(1, 31):
                    paired = [
                        float(s["epoch_log"][epoch - 1]["val_accuracy"])
                        - float(d["epoch_log"][epoch - 1]["val_accuracy"])
                        for d, s in zip(dense, sparse)
                    ]
                    delta.append(float(np.mean(paired)))
                early_delta = float(np.mean(delta[:3]))
                target = 31
                reversal = False
                if early_delta <= 0:
                    target = 3
                else:
                    for epoch in range(4, 29):
                        if all(delta[e - 1] <= 0 for e in (epoch, epoch + 1, epoch + 2)):
                            target = epoch
                            reversal = True
                            break

                accuracy_level, accuracy_slope = level_slope(sparse, "val_accuracy")
                entropy_level, entropy_slope = level_slope(sparse, "probe_dense_entropy")
                margin_level, margin_slope = level_slope(sparse, "probe_boundary_margin")
                tail_level, tail_slope = level_slope(sparse, "probe_tail_mass")
                gradient_level, gradient_slope = level_slope(sparse, "probe_gradient_topk_share")
                exemplar = sparse[0]
                cell = {
                    "dataset": dataset,
                    "architecture": architecture,
                    "tier": tier,
                    "target_switch_epoch": target,
                    "right_censored": int(target == 31),
                    "persistent_reversal": int(reversal),
                    "early_delta": early_delta,
                    "accuracy_level": accuracy_level,
                    "accuracy_slope": accuracy_slope,
                    "entropy_level": entropy_level,
                    "entropy_slope": entropy_slope,
                    "margin_level": margin_level,
                    "margin_slope": margin_slope,
                    "tail_level": tail_level,
                    "tail_slope": tail_slope,
                    "gradient_level": gradient_level,
                    "gradient_slope": gradient_slope,
                    "log_classes": math.log(float(exemplar["classes"])),
                    "log_seq_len": math.log(float(exemplar["seq_len"])),
                    "architecture_local": int(architecture == "local_hybrid"),
                    "blocks": int(exemplar["layers"]),
                    "embed_dim": int(exemplar["embed_dim"]),
                    "delta_by_epoch": json.dumps(delta, separators=(",", ":")),
                }
                cells.append(cell)
    return cells


def ridge_predict(train: list[dict], test: list[dict], features: tuple[str, ...]) -> np.ndarray:
    x_train = np.asarray([[float(r[f]) for f in features] for r in train], dtype=float)
    x_test = np.asarray([[float(r[f]) for f in features] for r in test], dtype=float)
    y = np.asarray([float(r["target_switch_epoch"]) for r in train], dtype=float)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    y_mean = float(y.mean())
    beta = np.linalg.solve(x_train.T @ x_train + np.eye(x_train.shape[1]), x_train.T @ (y - y_mean))
    raw = y_mean + x_test @ beta
    return np.clip(np.rint(raw), 3, 30).astype(int)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rankdata(np.asarray(a, dtype=float)), rankdata(np.asarray(b, dtype=float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def balanced_accuracy(targets: list[int], preds: list[int]) -> float:
    truth = np.asarray(targets) <= 30
    pred = np.asarray(preds) <= 30
    rates = []
    for label in (False, True):
        mask = truth == label
        if mask.any():
            rates.append(float((pred[mask] == truth[mask]).mean()))
    return float(np.mean(rates))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="outputs/m1 directory")
    ap.add_argument("--validation-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("phase") != "m1":
        raise RuntimeError("M1 integrity validator must pass before analysis")
    rows = read_rows(args.root)
    cells = build_cells(rows)
    if len(cells) != 60:
        raise RuntimeError(f"expected 60 task-architecture-capacity cells, found {len(cells)}")

    predictions = []
    model_errors: dict[str, dict[str, float]] = defaultdict(dict)
    for dataset in DATASETS:
        train = [r for r in cells if r["dataset"] != dataset]
        test = [r for r in cells if r["dataset"] == dataset]
        constant = int(np.clip(np.rint(np.median([r["target_switch_epoch"] for r in train])), 3, 30))
        fold_predictions = {"constant": np.full(len(test), constant, dtype=int)}
        for name, features in FEATURE_SETS.items():
            fold_predictions[name] = ridge_predict(train, test, features)
        for i, cell in enumerate(test):
            out = {
                "dataset": cell["dataset"], "architecture": cell["architecture"],
                "tier": cell["tier"], "target_switch_epoch": cell["target_switch_epoch"],
            }
            for name, values in fold_predictions.items():
                out[f"pred_{name}"] = int(values[i])
                out[f"abs_error_{name}"] = abs(int(values[i]) - int(cell["target_switch_epoch"]))
            predictions.append(out)
        for name, values in fold_predictions.items():
            model_errors[name][dataset] = float(np.mean([
                abs(int(values[i]) - int(test[i]["target_switch_epoch"])) for i in range(len(test))
            ]))

    macro_mae = {name: float(np.mean(list(per_dataset.values()))) for name, per_dataset in model_errors.items()}
    strongest = min(BASELINE_TIE_ORDER, key=lambda name: (macro_mae[name], BASELINE_TIE_ORDER.index(name)))
    relative_improvement = (macro_mae[strongest] - macro_mae["mechanistic"]) / macro_mae[strongest]

    per_dataset_diff = np.asarray([
        model_errors["mechanistic"][d] - model_errors[strongest][d] for d in DATASETS
    ])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = per_dataset_diff[rng.integers(0, len(DATASETS), size=(BOOTSTRAP_RESAMPLES, len(DATASETS)))].mean(axis=1)
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    positive_improvements = {
        d: max(0.0, model_errors[strongest][d] - model_errors["mechanistic"][d]) for d in DATASETS
    }
    influence_denominator = float(sum(positive_improvements.values()))
    max_influence = max(positive_improvements.values()) / influence_denominator \
        if influence_denominator > 0 else 1.0

    reversals = [c for c in cells if c["persistent_reversal"]]
    outside = sorted({c["dataset"] for c in reversals if c["dataset"] != "cifar10"})
    reversal_architectures = sorted({c["architecture"] for c in reversals})
    reversal_tiers = sorted({c["tier"] for c in reversals})
    criterion_1 = len(outside) >= 3 and set(reversal_architectures) == set(ARCHITECTURES) and len(reversal_tiers) >= 2
    criterion_2 = relative_improvement >= 0.10
    criterion_3 = ci[1] < 0
    criterion_4 = influence_denominator > 0 and max_influence <= 0.5
    m1_pass = all((criterion_1, criterion_2, criterion_3, criterion_4))

    proposed_rows = sorted(predictions, key=lambda r: (r["dataset"], r["architecture"], r["tier"]))
    targets = [int(r["target_switch_epoch"]) for r in proposed_rows]
    proposed = [int(r["pred_mechanistic"]) for r in proposed_rows]
    result = {
        "status": "M1_PASS" if m1_pass else "M1_KILLED",
        "operation_id": "snd-crossover-mechanism-s2d-extension-20260826",
        "input_validation_sha256": sha256(args.validation_report),
        "raw_file_sha256": validation.get("raw_files", {}),
        "cell_count": len(cells),
        "persistent_reversal_cells": len(reversals),
        "persistent_reversal_datasets_outside_cifar10": outside,
        "persistent_reversal_architectures": reversal_architectures,
        "persistent_reversal_tiers": reversal_tiers,
        "macro_mae": macro_mae,
        "strongest_baseline": strongest,
        "relative_mae_improvement": relative_improvement,
        "proposed_minus_strongest_bootstrap_ci95": ci,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "positive_influence_by_dataset": positive_improvements,
        "max_single_dataset_influence": max_influence,
        "spearman": spearman(targets, proposed),
        "event_balanced_accuracy": balanced_accuracy(targets, proposed),
        "mean_signed_calibration_error": float(np.mean(np.asarray(proposed) - np.asarray(targets))),
        "criteria": {
            "crossovers_breadth": criterion_1,
            "relative_mae_at_least_10_percent": criterion_2,
            "bootstrap_interval_below_zero": criterion_3,
            "max_dataset_influence_at_most_half": criterion_4,
        },
    }

    switch_map = {
        "mechanistic_switch": {}, "median_switch": {},
        "early_accuracy_switch": {}, "entropy_switch": {},
    }
    pred_key = {
        "mechanistic_switch": "pred_mechanistic",
        "median_switch": "pred_constant",
        "early_accuracy_switch": "pred_early_accuracy",
        "entropy_switch": "pred_entropy",
    }
    for method, key in pred_key.items():
        for row in predictions:
            switch_map[method].setdefault(row["dataset"], {}).setdefault(row["architecture"], {})[row["tier"]] = int(row[key])

    args.out.mkdir(parents=True, exist_ok=True)
    cell_rows = [{k: v for k, v in c.items()} for c in cells]
    write_csv(args.out / "m1_cell_features.csv", cell_rows)
    write_csv(args.out / "m1_lodo_predictions.csv", predictions)
    (args.out / "m1_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out / "m1_switch_map.json").write_text(json.dumps(switch_map, indent=2) + "\n", encoding="utf-8")
    evidence = {
        "analysis_source_sha256": sha256(Path(__file__)),
        "validation_report_sha256": sha256(args.validation_report),
        "outputs": {
            name: sha256(args.out / name) for name in (
                "m1_cell_features.csv", "m1_lodo_predictions.csv", "m1_result.json", "m1_switch_map.json"
            )
        },
    }
    (args.out / "m1_analysis_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if m1_pass else 2


if __name__ == "__main__":
    sys.exit(main())
