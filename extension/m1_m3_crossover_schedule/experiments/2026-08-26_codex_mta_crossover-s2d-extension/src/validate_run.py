#!/usr/bin/env python3
"""Integrity validator for MCH-SND-002 pilot and M1 raw records."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path


DISCOVERY = (
    "mnist", "fashion_mnist", "kmnist", "cifar10", "cifar100",
    "twenty_news", "synthetic_marker", "emnist_letters", "emnist_digits", "usps",
)
CONFIRMATION = ("emnist_balanced", "k49", "svhn")
ARCHITECTURES = ("plain_transformer", "local_hybrid")
TIERS = ("compact", "deep", "wide")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument(
        "--phase", choices=["pilot", "m1", "m2", "m3_baseline", "m3", "m3_test"],
        required=True,
    )
    ap.add_argument("--simple-method", choices=["median_switch", "early_accuracy_switch", "entropy_switch"])
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    raw_files = sorted(args.root.glob("raw/*.jsonl"))
    manifest_files = sorted(args.root.glob("manifests/*.json"))
    errors: list[str] = []
    rows = []
    for path in raw_files:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path.name}:{line_no}: invalid JSON: {exc}")
                    continue
                row["_source"] = path.name
                rows.append(row)

    completed = [r for r in rows if r.get("status") == "completed"]
    failed = [r for r in rows if r.get("status") != "completed"]
    keys = [(r.get("dataset"), r.get("architecture"), r.get("tier"), r.get("method"), int(r.get("seed", -1)))
            for r in completed]
    dup = [k for k, n in Counter(keys).items() if n > 1]
    if dup:
        errors.append(f"duplicate completed cells: {dup[:10]}")

    if args.phase == "pilot":
        methods = ("softmax", "topk_softmax_025", "switch_2", "cosine_alpha", "headwise_adaptive_entmax")
        expected = set(itertools.product(("mnist",), ARCHITECTURES, ("compact",), methods, (0,)))
        expected_epochs = 3
    elif args.phase == "m1":
        methods = ("softmax", "topk_softmax_025")
        expected = set(itertools.product(DISCOVERY, ARCHITECTURES, TIERS, methods, range(5)))
        expected_epochs = 30
    elif args.phase == "m2":
        methods = (
            "softmax", "topk_softmax_025", "median_switch", "early_accuracy_switch",
            "entropy_switch", "mechanistic_switch", "headwise_adaptive_entmax", "cosine_alpha",
        )
        expected = set(itertools.product(DISCOVERY, ARCHITECTURES, TIERS, methods, range(5, 10)))
        expected_epochs = 30
    elif args.phase == "m3_baseline":
        methods = ("softmax", "topk_softmax_025")
        expected = set(itertools.product(CONFIRMATION, ARCHITECTURES, TIERS, methods, range(10, 20)))
        expected_epochs = 30
    else:
        if not args.simple_method:
            raise SystemExit("--simple-method is required for m3 and m3_test")
        methods = ("softmax", "topk_softmax_025", "mechanistic_switch", args.simple_method)
        expected = set(itertools.product(CONFIRMATION, ARCHITECTURES, TIERS, methods, range(10, 20)))
        expected_epochs = 30

    actual = set(keys)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing completed cells ({len(missing)}): {missing[:10]}")
    if extra:
        errors.append(f"unexpected completed cells ({len(extra)}): {extra[:10]}")
    completed_key_set = set(keys)
    unresolved_failed = [
        r for r in failed
        if (r.get("dataset"), r.get("architecture"), r.get("tier"), r.get("method"), int(r.get("seed", -1)))
        not in completed_key_set
    ]
    if unresolved_failed:
        errors.append(f"unresolved noncompleted raw rows ({len(unresolved_failed)}): " +
                      ", ".join(f"{r.get('dataset')}/{r.get('method')}/{r.get('seed')}={r.get('status')}"
                                for r in unresolved_failed[:10]))

    metric_keys = (
        "val_accuracy", "val_macro_f1", "val_loss", "attention_density",
        "attention_entropy", "probe_dense_entropy", "probe_boundary_margin",
        "probe_tail_mass", "probe_gradient_topk_share", "probe_loss",
    )
    for row in completed:
        log = row.get("epoch_log")
        if not isinstance(log, list) or len(log) != expected_epochs:
            errors.append(f"{row['_source']}: bad epoch count for {keys[completed.index(row)]}: "
                          f"{len(log) if isinstance(log, list) else None}")
            continue
        epochs = [e.get("epoch") for e in log]
        if epochs != list(range(1, expected_epochs + 1)):
            errors.append(f"{row['_source']}: nonconsecutive epochs for {row.get('dataset')}/{row.get('method')}")
        for epoch in log:
            for metric in metric_keys:
                if not finite(epoch.get(metric)):
                    errors.append(
                        f"{row['_source']}: nonfinite {metric} at "
                        f"{row.get('dataset')}/{row.get('architecture')}/{row.get('tier')}/"
                        f"{row.get('method')}/{row.get('seed')}/e{epoch.get('epoch')}"
                    )
            for metric in ("probe_dense_entropy", "probe_tail_mass", "probe_gradient_topk_share"):
                value = epoch.get(metric)
                if finite(value) and not (0.0 <= float(value) <= 1.0 + 1e-6):
                    errors.append(f"{row['_source']}: range error {metric}={value}")
        if row.get("cfg_batch_size") != 256:
            errors.append(f"{row['_source']}: batch size is {row.get('cfg_batch_size')}, expected 256")
        if row.get("probe_n") != 128:
            errors.append(f"{row['_source']}: probe_n is {row.get('probe_n')}, expected 128")
        if args.phase == "m3_test":
            if row.get("final_test_read") is not True:
                errors.append(f"{row['_source']}: missing final_test_read marker")
            for metric in ("test_accuracy", "test_macro_f1", "test_loss"):
                if not finite(row.get(metric)):
                    errors.append(f"{row['_source']}: nonfinite or missing {metric}")
        elif row.get("final_test_read") is True:
            errors.append(f"{row['_source']}: unexpected official test read in phase {args.phase}")

    source_hashes = set()
    manifest_hashes = {}
    manifest_by_stem = {}
    for path in manifest_files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        manifest_by_stem[path.stem] = obj
        manifest_hashes[path.name] = sha256(path)
        source_hash = obj.get("source_sha256", {}).get("extension_runner.py")
        if source_hash:
            source_hashes.add(source_hash)
    if len(source_hashes) > 1:
        errors.append(f"mixed runner source hashes: {sorted(source_hashes)}")
    if not manifest_files:
        errors.append("no manifests found")
    for raw_path in raw_files:
        manifest = manifest_by_stem.get(raw_path.stem)
        if not isinstance(manifest, dict):
            errors.append(f"{raw_path.name}: matching run manifest missing")
            continue
        expected_config_hash = manifest.get("config_hash")
        with raw_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    raw_row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if raw_row.get("config_hash") != expected_config_hash:
                    errors.append(f"{raw_path.name}:{line_no}: row/manifest config hash mismatch")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "phase": args.phase,
        "validator_source_sha256": sha256(Path(__file__)),
        "root": str(args.root.resolve()),
        "raw_files": {p.name: sha256(p) for p in raw_files},
        "manifest_files": manifest_hashes,
        "runner_source_hashes": sorted(source_hashes),
        "completed_cells": len(completed),
        "expected_cells": len(expected),
        "failed_rows": len(failed),
        "recovered_failed_rows": len(failed) - len(unresolved_failed),
        "unresolved_failed_rows": len(unresolved_failed),
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
