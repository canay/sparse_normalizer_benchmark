#!/usr/bin/env python3
"""Apply frozen discovery predictors to confirmation top-k epochs 1--3 only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CONFIRMATION = ("emnist_balanced", "k49", "svhn")
ARCHITECTURES = ("plain_transformer", "local_hybrid")
TIERS = ("compact", "deep", "wide")
FEATURE_SETS = {
    "mechanistic_switch": (
        "accuracy_level", "accuracy_slope", "entropy_level", "entropy_slope",
        "margin_level", "margin_slope", "tail_level", "tail_slope",
        "gradient_level", "gradient_slope", "log_classes", "log_seq_len",
        "architecture_local", "blocks", "embed_dim",
    ),
    "early_accuracy_switch": ("accuracy_level", "accuracy_slope"),
    "entropy_switch": ("entropy_level", "entropy_slope"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_topk(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "raw").glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("status") == "completed" and row.get("method") == "topk_softmax_025":
                    rows.append(row)
    return rows


def level_slope(rows: list[dict], field: str) -> tuple[float, float]:
    # Deliberately access only epochs 1--3; later trajectory values are not read.
    vals = [float(np.mean([float(r["epoch_log"][e - 1][field]) for r in rows])) for e in (1, 2, 3)]
    return vals[2], (vals[2] - vals[0]) / 2.0


def ridge_fit_predict(train: list[dict], test: list[dict], features: tuple[str, ...]) -> np.ndarray:
    x = np.asarray([[float(r[f]) for f in features] for r in train], dtype=float)
    z = np.asarray([[float(r[f]) for f in features] for r in test], dtype=float)
    y = np.asarray([float(r["target_switch_epoch"]) for r in train], dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    x, z = (x - mean) / std, (z - mean) / std
    y_mean = float(y.mean())
    beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ (y - y_mean))
    return np.clip(np.rint(y_mean + z @ beta), 3, 30).astype(int)


def require_m1_feature_manifest(manifest_path: Path, feature_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("m1_cell_features.csv") != sha256(feature_path):
        raise RuntimeError("M1 feature file is not bound to its analysis manifest")
    return manifest


def require_baseline_validation(report_path: Path, root: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("phase") != "m3_baseline":
        raise RuntimeError("M3 baseline integrity validator must pass before prediction")
    actual = {path.name: sha256(path) for path in sorted((root / "raw").glob("*.jsonl"))}
    if report.get("raw_files") != actual:
        raise RuntimeError("M3 baseline raw hashes no longer match the validation report")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m1-features", type=Path, required=True)
    ap.add_argument("--m1-analysis-manifest", type=Path, required=True)
    ap.add_argument("--m3-root", type=Path, required=True)
    ap.add_argument("--m3-baseline-validation", type=Path, required=True)
    ap.add_argument("--m3-lock", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    require_m1_feature_manifest(args.m1_analysis_manifest, args.m1_features)
    require_baseline_validation(args.m3_baseline_validation, args.m3_root)
    discovery = read_csv(args.m1_features)
    lock = json.loads(args.m3_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "DISCOVERY_FROZEN_FOR_M3":
        raise RuntimeError("M3 discovery lock is not frozen")
    strongest = lock["strongest_simple_schedule"]
    rows = read_topk(args.m3_root)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["architecture"], row["tier"])].append(row)

    confirmation = []
    for dataset in CONFIRMATION:
        for architecture in ARCHITECTURES:
            for tier in TIERS:
                group = grouped[(dataset, architecture, tier)]
                if sorted(int(r["seed"]) for r in group) != list(range(10, 20)):
                    raise RuntimeError(f"confirmation top-k seed mismatch: {dataset}/{architecture}/{tier}")
                accuracy_level, accuracy_slope = level_slope(group, "val_accuracy")
                entropy_level, entropy_slope = level_slope(group, "probe_dense_entropy")
                margin_level, margin_slope = level_slope(group, "probe_boundary_margin")
                tail_level, tail_slope = level_slope(group, "probe_tail_mass")
                gradient_level, gradient_slope = level_slope(group, "probe_gradient_topk_share")
                exemplar = group[0]
                confirmation.append({
                    "dataset": dataset, "architecture": architecture, "tier": tier,
                    "accuracy_level": accuracy_level, "accuracy_slope": accuracy_slope,
                    "entropy_level": entropy_level, "entropy_slope": entropy_slope,
                    "margin_level": margin_level, "margin_slope": margin_slope,
                    "tail_level": tail_level, "tail_slope": tail_slope,
                    "gradient_level": gradient_level, "gradient_slope": gradient_slope,
                    "log_classes": math.log(float(exemplar["classes"])),
                    "log_seq_len": math.log(float(exemplar["seq_len"])),
                    "architecture_local": int(architecture == "local_hybrid"),
                    "blocks": int(exemplar["layers"]), "embed_dim": int(exemplar["embed_dim"]),
                })

    maps = {name: {} for name in (*FEATURE_SETS.keys(), "median_switch")}
    predictions = {}
    for name, features in FEATURE_SETS.items():
        predictions[name] = ridge_fit_predict(discovery, confirmation, features)
    median = int(np.clip(np.rint(np.median([float(r["target_switch_epoch"]) for r in discovery])), 3, 30))
    predictions["median_switch"] = np.full(len(confirmation), median, dtype=int)
    for name, values in predictions.items():
        for i, row in enumerate(confirmation):
            maps[name].setdefault(row["dataset"], {}).setdefault(row["architecture"], {})[row["tier"]] = int(values[i])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(maps, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "M3_SWITCH_MAP_FROZEN",
        "strongest_simple_schedule": strongest,
        "used_confirmation_epochs": [1, 2, 3],
        "m1_features_sha256": sha256(args.m1_features),
        "m1_analysis_manifest_sha256": sha256(args.m1_analysis_manifest),
        "m3_discovery_lock_sha256": sha256(args.m3_lock),
        "m3_baseline_validation_sha256": sha256(args.m3_baseline_validation),
        "builder_source_sha256": sha256(Path(__file__)),
        "switch_map_sha256": sha256(args.out),
    }
    (args.out.parent / "m3_switch_map_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
