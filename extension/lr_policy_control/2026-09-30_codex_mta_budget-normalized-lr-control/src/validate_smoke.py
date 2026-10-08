"""Validate all 12 engineering-only smoke cells before main release."""
# ruff: noqa: E402 -- bytecode must be disabled before project imports.
from __future__ import annotations

import json
import math
import shutil
import sys

sys.dont_write_bytecode = True

from campaign import RUN, gpu_free_bytes, output_bytes, validate_complete, validate_release
from decay_worker import atomic_json, read, sha


def main() -> None:
    config = read(RUN / "candidate_config.json")
    release = validate_release(config, "smoke")
    release_sha = sha(RUN / "admission/SMOKE_RELEASE.json")
    complete_path = RUN / "SMOKE_COMPLETE.json"
    complete = read(complete_path)
    if complete.get("status") != "SMOKE_COMPLETE" or complete.get("cells") != 12:
        raise RuntimeError("SMOKE_NOT_COMPLETE_TWELVE")
    if complete.get("release_sha256") != release_sha or len(complete.get("results", [])) != 12:
        raise RuntimeError("SMOKE_COMPLETE_RELEASE_OR_CENSUS_INVALID")
    expected = {(budget, method, seed) for budget in (3, 30)
                for method in config["methods"] for seed in config["smoke_seeds"]}
    declared = {(r["budget"], r["method"], r["seed"]): r for r in complete["results"]}
    if set(declared) != expected:
        raise RuntimeError("SMOKE_COMPLETE_IDENTITIES_INVALID")
    rows = []
    for budget, method, seed in sorted(expected):
        cell = RUN / "engineering_smoke" / f"budget{budget:02d}" / f"{method}__seed{seed}"
        if not validate_complete(cell, budget, method, seed, release_sha,
                                 config["expected_minibatches_per_epoch"]):
            raise RuntimeError("SMOKE_CELL_NOT_VALID_COMPLETE")
        marker_path = cell / "COMPLETE.json"
        if sha(marker_path) != declared[(budget, method, seed)]["complete_sha256"]:
            raise RuntimeError("SMOKE_COMPLETE_MARKER_HASH_MISMATCH")
        marker = read(marker_path)
        attempt = cell / marker["attempt"]
        terminal = read(attempt / "TERMINAL.json")
        receipt = read(attempt / "receipt.json")
        row = read(attempt / "row.json")
        if (receipt.get("runtime") != read(RUN / "baseline_bindings.json")["runtime_identity"]
                or receipt.get("train_loader_calls") != ["cifar10"]
                or receipt.get("optimizer_updates") != budget * config["expected_minibatches_per_epoch"]
                or row.get("n_val") != 10000):
            raise RuntimeError("SMOKE_RUNTIME_DATA_OR_UPDATE_PARITY_FAILED")
        rows.append({"budget": budget, "method": method, "seed": seed,
                     "complete_sha256": sha(marker_path),
                     "row_sha256": marker["row_sha256"],
                     "max_rss_bytes": terminal["max_observed_rss_bytes"],
                     "peak_gpu_allocated_mb": row["peak_memory_mb"],
                     "elapsed_monotonic_seconds": terminal["elapsed_monotonic_seconds"]})
    maximum_rss = max(r["max_rss_bytes"] for r in rows)
    maximum_gpu_mb = max(r["peak_gpu_allocated_mb"] for r in rows)
    smoke_seconds = sum(r["elapsed_monotonic_seconds"] for r in rows)
    projected_main_seconds = smoke_seconds * (3060 / 198)
    if not all(math.isfinite(v) and v >= 0 for v in (maximum_rss, maximum_gpu_mb,
                                                    smoke_seconds, projected_main_seconds)):
        raise RuntimeError("SMOKE_RESOURCE_MEASUREMENT_INVALID")
    if (maximum_rss > config["resource_ceiling"]["worker_rss_gib"] * 1024 ** 3
            or maximum_gpu_mb > config["resource_ceiling"]["gpu_allocated_gib"] * 1024
            or output_bytes() > config["resource_ceiling"]["new_output_gib"] * 1024 ** 3
            or projected_main_seconds > config["resource_ceiling"]["wall_hours"] * 3600):
        raise RuntimeError("SMOKE_MEASURED_RESOURCE_OR_PROJECTED_WALL_CEILING_FAILED")
    frozen = {key: value for key, value in release.items() if key.endswith("_sha256")}
    result = {"status": "SMOKE_VALIDATED", "cells": 12,
              "smoke_complete_sha256": sha(complete_path),
              "smoke_release_sha256": release_sha,
              "frozen_inputs": frozen, "cells_bound": rows,
              "maximum_worker_rss_bytes": maximum_rss,
              "maximum_gpu_allocated_mb": maximum_gpu_mb,
              "smoke_elapsed_seconds": smoke_seconds,
              "projected_main_seconds_descriptive": projected_main_seconds,
              "current_gpu_free_bytes": gpu_free_bytes(),
              "current_disk_free_bytes": shutil.disk_usage(RUN).free,
              "current_new_output_bytes": output_bytes(),
              "science_outcomes_not_interpreted": True}
    path = RUN / "admission/SMOKE_VALIDATED.json"
    atomic_json(path, result)
    print(json.dumps({"status": result["status"], "cells": 12,
                      "projected_main_seconds_descriptive": projected_main_seconds,
                      "sha256": sha(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
