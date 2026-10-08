"""Outcome-blind resource admission: hashes, census, elapsed time and bytes only."""
import argparse
import json
import time
from pathlib import Path

import training_runner as base
from campaign import plan, source_census


def summarize(root, scientific_config):
    cfg = json.loads((root / "config.json").read_text())
    if cfg["run_kind"] != "resource_pilot" or scientific_config["run_kind"] != "scientific":
        raise RuntimeError("resource-pilot and scientific configs required")
    differing = {key for key in set(cfg) | set(scientific_config)
                 if cfg.get(key) != scientific_config.get(key)}
    allowed = {"run_id", "run_kind", "scope", "seeds", "forbidden_seeds", "max_disk_bytes",
               "maximum_launch_wall_seconds", "manuscript_authority"}
    if differing - allowed or len(cfg["seeds"]) != 1 or set(cfg["seeds"]) & set(scientific_config["seeds"]):
        raise RuntimeError("pilot/science resource extrapolation interface drift")
    base.require_run_open(root)
    status = json.loads((root / "outputs/status.json").read_text())
    validation_path = root / "outputs/validation.json"
    validation = json.loads(validation_path.read_text())
    if (status["status"] != "COMPLETED_VERIFIED" or status["exit_code"] != 0
            or validation["status"] != "PASS" or status["validation_sha256"] != base.sha256(validation_path)
            or validation["config_sha256"] != base.sha256(root / "config.json")
            or validation["units"] != len(plan(cfg))):
        raise RuntimeError("verified full pilot census required")
    for unit in validation["unit_receipts"]:
        directory = "forks" if "_at" in unit["unit_id"] else "main"
        marker = root / "outputs" / directory / unit["unit_id"] / "complete.json"
        if base.sha256(marker) != unit["complete_sha256"]:
            raise RuntimeError("pilot completion receipt drift")
        for item in json.loads(marker.read_text())["artifacts"]:
            if base.sha256(marker.parent / item["path"]) != item["sha256"]:
                raise RuntimeError("pilot artifact drift")
    data_bytes = sum(base.tree_bytes(root / folder) for folder in ("inputs", "outputs"))
    projected = data_bytes * len(scientific_config["seeds"]) + 1_000_000_000
    free = base.shutil.disk_usage(root).free
    peak = status["peak_resident_memory_bytes"]
    passed = (projected <= scientific_config["max_disk_bytes"]
              and free - projected >= scientific_config["minimum_free_disk_bytes"]
              and peak <= scientific_config["maximum_resident_memory_bytes"])
    return {"status": "PASS" if passed else "RESOURCE_INFEASIBLE", "outcomes_read": False,
            "pilot_config_sha256": base.sha256(root / "config.json"),
            "source_sha256": source_census(root), "validation_sha256": base.sha256(validation_path),
            "planned_units": validation["units"], "data_bytes": data_bytes,
            "projected_scientific_bytes_including_1GB_reserve": projected,
            "current_free_disk_bytes": free, "projected_remaining_free_disk_bytes": free-projected,
            "peak_resident_memory_bytes": peak, "pilot_elapsed_seconds": status["elapsed_seconds"],
            "linear_scientific_seconds_excluding_growth_overhead": status["elapsed_seconds"] * len(scientific_config["seeds"]),
            "warning": "Timing is an extrapolation; full-tree fsync/hash/census overhead can grow with output size.",
            "checked_at_unix": time.time()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--scientific-config", required=True)
    args = parser.parse_args()
    root = Path(args.pilot_root).resolve()
    science_path = Path(args.scientific_config).resolve()
    result = summarize(root, json.loads(science_path.read_text()))
    result["scientific_config_sha256"] = base.sha256(science_path)
    base.exclusive_atomic_json(root / "RESOURCE_RECEIPT.json", result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)
