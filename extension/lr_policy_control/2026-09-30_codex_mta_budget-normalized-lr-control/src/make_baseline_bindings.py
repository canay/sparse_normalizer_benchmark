"""Admit the frozen same-runtime fixed-LR traces as a within-path comparator.

This reads the September 28 completed producer. It never edits or reruns it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
OLD = PROJECT / "experiments/2026-09-28_codex_mta_matched-bottomk-policy/runtime_cohorts/post_reboot_001"
CONFIG = json.loads((RUN / "candidate_config.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest().upper()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, obj: dict) -> None:
    if path.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=RUN / "baseline_bindings.json")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(RUN.resolve())
    admission_path = OLD / "admission/DATA_ENV_ADMISSION.json"
    manifest_path = OLD / "RUN_MANIFEST.json"
    decision_path = OLD / "outputs/SCIENTIFIC_DECISION.json"
    if digest(admission_path) != CONFIG["fixed_baseline_admission_sha256"]:
        raise RuntimeError("BASELINE_ADMISSION_HASH_MISMATCH")
    if digest(manifest_path) != CONFIG["fixed_baseline_manifest_sha256"]:
        raise RuntimeError("BASELINE_MANIFEST_HASH_MISMATCH")
    if digest(decision_path) != CONFIG["fixed_baseline_decision_sha256"]:
        raise RuntimeError("BASELINE_DECISION_HASH_MISMATCH")
    admission = read(admission_path)
    manifest = read(manifest_path)
    if admission["status"] != "DATA_ENV_ADMITTED":
        raise RuntimeError("BASELINE_DATA_NOT_ADMITTED")
    if admission["archive_sha256"].upper() != CONFIG["archive_sha256"]:
        raise RuntimeError("BASELINE_ARCHIVE_MISMATCH")
    old_args = manifest["hyperparameters"]
    for key, new_key in (("batch_size", "batch_size"), ("patch", "patch"),
                         ("val_fraction", "val_fraction"), ("patience", "patience"),
                         ("tier", "tier"), ("weight_decay", "weight_decay"),
                         ("grad_clip", "grad_clip"), ("dropout", "dropout")):
        if old_args[key] != CONFIG[new_key]:
            raise RuntimeError(f"BASELINE_HYPERPARAMETER_MISMATCH:{key}")
    if old_args["lr"] != CONFIG["lr_base"] or old_args["max_epochs"] != 30:
        raise RuntimeError("BASELINE_LR_OR_BUDGET_MISMATCH")
    splits = {int(s["seed"]): s for s in admission["splits"]}
    wanted_splits = set(CONFIG["seeds"]) | set(CONFIG["smoke_seeds"])
    if not wanted_splits.issubset(splits):
        raise RuntimeError("BASELINE_SPLIT_CENSUS_MISMATCH")
    splits = {seed: splits[seed] for seed in sorted(wanted_splits)}
    entries = []
    for method in CONFIG["methods"]:
        for seed in CONFIG["seeds"]:
            cell = OLD / "main/matched50/cells" / f"{method}__seed{seed}" / "attempt-001"
            row_path, receipt_path = cell / "row.json", cell / "worker_receipt.json"
            row, receipt = read(row_path), read(receipt_path)
            complete = read(cell.parent / "complete.json")
            if (complete.get("classification") != "VALID_COMPLETE"
                    or complete.get("row_sha256") != digest(row_path)
                    or complete.get("worker_receipt_sha256") != digest(receipt_path)
                    or complete.get("row_path") != "attempt-001/row.json"
                    or complete.get("worker_receipt_path") != "attempt-001/worker_receipt.json"):
                raise RuntimeError(f"BASELINE_PRODUCER_COMPLETE_MISMATCH:{method}:{seed}")
            if (row.get("method"), row.get("seed"), row.get("status"), row.get("epochs_run")) != (method, seed, "completed", 30):
                raise RuntimeError(f"BASELINE_ROW_INVALID:{method}:{seed}")
            expected_cfg = {"tier": CONFIG["tier"], "cfg_max_epochs": 30,
                "cfg_batch_size": CONFIG["batch_size"], "cfg_patch": CONFIG["patch"],
                "cfg_val_fraction": CONFIG["val_fraction"],
                "cfg_patience": CONFIG["patience"], "n_train": 40000,
                "n_val": 10000, "layers": 2, "embed_dim": 64,
                "heads": 4, "seq_len": 65, "classes": 10}
            if any(row.get(k) != v for k, v in expected_cfg.items()):
                raise RuntimeError(f"BASELINE_ROW_CONFIG_MISMATCH:{method}:{seed}")
            if [e.get("epoch") for e in row["epoch_log"]] != list(range(30)):
                raise RuntimeError(f"BASELINE_EPOCH_CENSUS_INVALID:{method}:{seed}")
            if receipt["source_manifest_sha256"] != admission["source_manifest_sha256"]:
                raise RuntimeError(f"BASELINE_SOURCE_MISMATCH:{method}:{seed}")
            if receipt["runtime_identity"] != admission["runtime_identity"]:
                raise RuntimeError(f"BASELINE_RUNTIME_MISMATCH:{method}:{seed}")
            if receipt["final_config_sha256"] != admission["final_config_sha256"]:
                raise RuntimeError(f"BASELINE_CONFIG_MISMATCH:{method}:{seed}")
            epoch_hash = hashlib.sha256(json.dumps(row["epoch_log"], sort_keys=True).encode()).hexdigest().upper()
            if receipt["epoch_log_serialization_sha256"] != epoch_hash:
                raise RuntimeError(f"BASELINE_EPOCH_HASH_MISMATCH:{method}:{seed}")
            if receipt["initial_parameter_sha256"] != read(cell / "scientific_bundle.json")["metadata"]["initial_parameter_sha256"]:
                raise RuntimeError(f"BASELINE_INITIALIZATION_MISMATCH:{method}:{seed}")
            entries.append({"method": method, "seed": seed,
                            "row_project_relative": row_path.relative_to(PROJECT).as_posix(),
                            "row_sha256": digest(row_path),
                            "receipt_project_relative": receipt_path.relative_to(PROJECT).as_posix(),
                            "receipt_sha256": digest(receipt_path),
                            "initial_parameter_sha256": receipt["initial_parameter_sha256"]})
    if len(entries) != 30 or len({(e["method"], e["seed"]) for e in entries}) != 30:
        raise RuntimeError("BASELINE_30_CENSUS_FAILED")
    result = {"status": "BASELINE_FIXED_RATE_WITHIN_PATH_ADMITTED",
              "comparator_is_new_independent_replication": False,
              "source_population": "matched50/post_reboot_001",
              "fixed_rate_reference_uses_30_epoch_prefixes": True,
              "admission_sha256": digest(admission_path),
              "manifest_sha256": digest(manifest_path),
              "decision_sha256": digest(decision_path),
              "runtime_identity": admission["runtime_identity"],
              "train_x_byte_sha256": admission["train_x_byte_sha256"],
              "train_y_byte_sha256": admission["train_y_byte_sha256"],
              "splits": splits, "entries": entries}
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "entries": len(entries),
                      "binding_sha256": digest(output), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
