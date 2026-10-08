"""Strict local-only analysis of the frozen 210-cell decay policy control.

This script never trains, reads official test data or changes old decisions.
It requires a post-transport local hash certificate before interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
T9_95 = 2.2621571628540993


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest().upper()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def finite_number(value, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"NONFINITE_REQUIRED_ENDPOINT:{key}:{value}")
    return float(value)


def selected(log: list[dict], budget: int) -> dict:
    prefix = log[:budget]
    if len(prefix) != budget or [x.get("epoch") for x in prefix] != list(range(budget)):
        raise RuntimeError("INVALID_ORDERED_PREFIX")
    best = max(prefix, key=lambda x: finite_number(x["val_accuracy"], "val_accuracy"))
    best_accuracy = finite_number(best["val_accuracy"], "best_accuracy")
    count = round(best_accuracy * 10000)
    if not 0 <= count <= 10000 or abs(best_accuracy - count / 10000) > 1e-12:
        raise RuntimeError("BEST_VALIDATION_ACCURACY_NOT_10000_COUNT")
    return {"best_accuracy": best_accuracy, "best_correct_count": count,
            "best_epoch": int(best["epoch"]),
            "best_f1": finite_number(best["val_macro_f1"], "best_f1"),
            "terminal_accuracy": finite_number(prefix[-1]["val_accuracy"], "terminal_accuracy"),
            "terminal_f1": finite_number(prefix[-1]["val_macro_f1"], "terminal_f1")}


def summarize(count_differences: list[int]) -> dict:
    if len(count_differences) != 10 or any(not isinstance(v, int) for v in count_differences):
        raise RuntimeError("PAIRED_N_NOT_10")
    values = [v / 10000 for v in count_differences]
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T9_95 * sd / math.sqrt(10)
    return {"n": 10, "mean": sum(count_differences) / 100000, "sample_sd": sd,
            "nominal95_low": mean - half, "nominal95_high": mean + half,
            "paired_seed_differences": values,
            "paired_seed_correct_count_differences": count_differences,
            "sum_correct_count_differences": sum(count_differences)}


def load_decay(config: dict, transport: dict) -> dict:
    complete_path = RUN / "COMPUTE_COMPLETE.json"
    if sha(complete_path) != transport["compute_complete_sha256"]:
        raise RuntimeError("LOCAL_COMPLETE_MARKER_NOT_TRANSPORT_VERIFIED")
    complete = read(complete_path)
    if complete["status"] != "COMPUTE_COMPLETE" or complete["cells"] != config["expected_cells"]:
        raise RuntimeError("DECAY_CAMPAIGN_INCOMPLETE")
    release_path = RUN / "admission/MAIN_RELEASE.json"
    release = read(release_path)
    if complete["release_sha256"] != sha(release_path) or release.get("status") != "MAIN_RELEASED":
        raise RuntimeError("DECAY_MAIN_RELEASE_CHAIN_INVALID")
    for field, relative in (("candidate_config_sha256", "candidate_config.json"),
                            ("protocol_sha256", "PROTOCOL.md"),
                            ("baseline_bindings_sha256", "baseline_bindings.json"),
                            ("worker_sha256", "src/decay_worker.py"),
                            ("campaign_sha256", "src/campaign.py"),
                            ("analyzer_sha256", "src/analyze_results.py"),
                            ("precompute_checks_sha256", "src/precompute_checks.py"),
                            ("controller_tests_sha256", "src/controller_contract_tests.py"),
                            ("transport_verifier_sha256", "src/verify_transport.py"),
                            ("benchmark_sha256", "src/original/benchmark_r1.py"),
                            ("datasets_sha256", "src/original/datasets_r1.py")):
        if release[field] != sha(RUN / relative):
            raise RuntimeError(f"DECAY_RELEASE_FROZEN_INPUT_DRIFT:{field}")
    if len(complete["results"]) != 210:
        raise RuntimeError("COMPLETE_MARKER_RESULT_CENSUS_INVALID")
    declared = {(r["budget"], r["method"], r["seed"]): r["complete_sha256"]
                for r in complete["results"]}
    if len(declared) != 210:
        raise RuntimeError("COMPLETE_MARKER_DUPLICATE_CELL_ID")
    release_sha = complete["release_sha256"]
    rows = {}
    for budget in config["budgets"]:
        for method in config["methods"]:
            for seed in config["seeds"]:
                cell = RUN / "main" / f"budget{budget:02d}" / f"{method}__seed{seed}"
                marker_path = cell / "COMPLETE.json"
                if sha(marker_path) != declared.get((budget, method, seed)):
                    raise RuntimeError("COMPUTE_COMPLETE_CELL_HASH_CHAIN_FAILED")
                marker = read(marker_path)
                attempt = cell / marker["attempt"]
                row_path, receipt_path = attempt / "row.json", attempt / "receipt.json"
                if (sha(row_path) != marker["row_sha256"] or sha(receipt_path) != marker["receipt_sha256"]
                        or sha(attempt / "TERMINAL.json") != marker["terminal_sha256"]):
                    raise RuntimeError(f"DECAY_CELL_HASH_MISMATCH:{budget}:{method}:{seed}")
                row, receipt = read(row_path), read(receipt_path)
                terminal = read(attempt / "TERMINAL.json")
                if terminal.get("status") != "VALID_COMPLETE" or terminal.get("exit_code") != 0:
                    raise RuntimeError("DECAY_TERMINAL_NOT_VALID_COMPLETE")
                if (row.get("status"), row.get("method"), row.get("seed"), row.get("cfg_budget"),
                        row.get("epochs_run")) != ("completed", method, seed, budget, budget):
                    raise RuntimeError(f"DECAY_CELL_IDENTITY_MISMATCH:{budget}:{method}:{seed}")
                if row.get("optimizer_updates") != budget * config["expected_minibatches_per_epoch"]:
                    raise RuntimeError("DECAY_UPDATE_COUNT_MISMATCH")
                if (receipt["row_sha256"] != sha(row_path)
                        or receipt["release_sha256"] != release_sha
                        or receipt["official_test_extracted"] is not False
                        or receipt["official_test_unpickled"] is not False):
                    raise RuntimeError("DECAY_RECEIPT_OR_TEST_BOUNDARY_FAILED")
                if row.get("n_val") != 10000:
                    raise RuntimeError("DECAY_VALIDATION_DENOMINATOR_NOT_10000")
                if row.get("best_epoch") != selected(row["epoch_log"], budget)["best_epoch"]:
                    raise RuntimeError("DECAY_EARLIEST_BEST_SELECTION_MISMATCH")
                for epoch in row["epoch_log"]:
                    for key in ("train_loss", "val_accuracy", "val_macro_f1", "val_loss",
                                "attention_density", "attention_entropy"):
                        finite_number(epoch[key], key)
                rows[(budget, method, seed)] = row
    if len(rows) != 210:
        raise RuntimeError("DECAY_210_CENSUS_FAILED")
    return rows


def load_fixed(config: dict, binding: dict) -> dict:
    rows = {}
    for entry in binding["entries"]:
        path = PROJECT / entry["row_project_relative"]
        receipt = PROJECT / entry["receipt_project_relative"]
        if sha(path) != entry["row_sha256"] or sha(receipt) != entry["receipt_sha256"]:
            raise RuntimeError("FIXED_RATE_COMPARATOR_DRIFT")
        row = read(path)
        if row["status"] != "completed" or row["epochs_run"] != 30:
            raise RuntimeError("FIXED_RATE_COMPARATOR_INCOMPLETE")
        if row.get("n_val") != 10000:
            raise RuntimeError("FIXED_RATE_VALIDATION_DENOMINATOR_NOT_10000")
        rows[(entry["method"], entry["seed"])] = row
    if len(rows) != 30:
        raise RuntimeError("FIXED_RATE_30_CENSUS_FAILED")
    return rows


def fixed_reference_pattern(config: dict, binding: dict, fixed: dict) -> dict:
    patterns = {}
    for method in config["methods"][1:]:
        ends = {}
        for budget in (3, 30):
            counts = []
            for seed in config["seeds"]:
                sparse = selected(fixed[(method, seed)]["epoch_log"], budget)["best_correct_count"]
                dense = selected(fixed[("softmax", seed)]["epoch_log"], budget)["best_correct_count"]
                counts.append(sparse - dense)
            ends[str(budget)] = {"paired_seed_correct_count_differences": counts,
                                 "sum_correct_count_differences": sum(counts),
                                 "mean_accuracy_difference": sum(counts) / 100000}
        patterns[method] = {"budget3": ends["3"], "budget30": ends["30"],
                            "reference_has_positive_to_negative_reversal":
                            ends["3"]["sum_correct_count_differences"] > 0
                            and ends["30"]["sum_correct_count_differences"] < 0}
    return {"status": "FIXED_RATE_WITHIN_PATH_REFERENCE_PATTERN_VERIFIED",
            "not_independent_replication": True,
            "config_sha256": sha(RUN / "candidate_config.json"),
            "baseline_bindings_sha256": sha(RUN / "baseline_bindings.json"),
            "patterns": patterns}


def observed_terminal_status(transport: dict) -> tuple[str, dict]:
    main = RUN / "main"
    science_paths = sorted(main.rglob("scientific_failure.json")) if main.exists() else []
    science_paths += sorted(main.rglob("SCIENTIFIC_FAILURE_NON_EVIDENCE.json")) if main.exists() else []
    stop = RUN / "CAMPAIGN_STOP.json"
    if stop.exists() and read(stop).get("mode") == "main":
        science_paths.append(stop)
    for path in science_paths:
        relative = path.relative_to(RUN).as_posix()
        if transport.get("file_hashes", {}).get(relative) != sha(path):
            raise RuntimeError(f"SCIENTIFIC_MARKER_NOT_TRANSPORT_VERIFIED:{relative}")
    complete = RUN / "COMPUTE_COMPLETE.json"
    declared = transport.get("campaign_terminal_status")
    if science_paths:
        status = "SCIENTIFIC_FAILURE_NON_EVIDENCE"
    elif complete.exists():
        if transport.get("compute_complete_sha256") != sha(complete):
            raise RuntimeError("COMPLETE_MARKER_NOT_TRANSPORT_VERIFIED")
        status = "COMPUTE_COMPLETE"
    else:
        status = "INCOMPLETE_NON_EVIDENCE"
    if status == "COMPUTE_COMPLETE" and declared != status:
        raise RuntimeError("DECLARED_INCOMPLETE_CANNOT_SUPPRESS_VERIFIED_COMPLETE")
    if status == "INCOMPLETE_NON_EVIDENCE" and declared == "COMPUTE_COMPLETE":
        raise RuntimeError("DECLARED_COMPLETE_WITHOUT_MARKER")
    attempts = list(main.rglob("attempt-[0-9][0-9][0-9]")) if main.exists() else []
    recoveries = list(main.rglob("RECOVERY-[0-9][0-9][0-9].json")) if main.exists() else []
    for path in recoveries:
        relative = path.relative_to(RUN).as_posix()
        if transport.get("file_hashes", {}).get(relative) != sha(path):
            raise RuntimeError(f"RECOVERY_RECORD_NOT_TRANSPORT_VERIFIED:{relative}")
    census = {"main_attempt_directories": len(attempts),
              "main_recovery_records": len(recoveries),
              "main_scientific_markers": len(science_paths),
              "operator_declared_terminal_status": declared}
    return status, census


def policy_labels(validity: str, reference: dict, contrasts: list[dict],
                  methods: list[str]) -> tuple[dict, str]:
    signs = {}
    for method in methods[1:]:
        q = {c["budget"]: c["sum_correct_count_differences"] for c in contrasts
             if c["policy"] == "decay" and c["contrast"] == method + "_minus_softmax"}
        if 3 not in q or 30 not in q:
            raise RuntimeError("MISSING_DECLARED_SIGN_ENDPOINT")
        reference_present = reference["patterns"][method]["reference_has_positive_to_negative_reversal"]
        label = ("NOT_INTERPRETED_INVALID_CONTROL" if validity != "VALID_DESCRIPTIVE_POLICY_CONTROL" else
                 "NOT_APPLICABLE_REFERENCE_NO_REVERSAL" if not reference_present else
                 "persists" if q[3] > 0 and q[30] < 0 else "fails")
        signs[method] = {"budget3_sum_correct_count_difference": q[3],
                         "budget30_sum_correct_count_difference": q[30],
                         "fixed_reference_reversal_present": reference_present, "label": label}
    labels = [value["label"] for value in signs.values()]
    combined = ("NOT_INTERPRETED_INVALID_CONTROL" if validity != "VALID_DESCRIPTIVE_POLICY_CONTROL" else
                "REFERENCE_PREMISE_ABSENT" if "NOT_APPLICABLE_REFERENCE_NO_REVERSAL" in labels else
                "PERSISTS_BOTH" if labels == ["persists", "persists"] else
                "FAILS_BOTH" if labels == ["fails", "fails"] else "MIXED")
    return signs, combined


def main() -> None:
    if os.name != "nt":
        raise RuntimeError("INTERPRETATION_REQUIRES_HASH_VERIFIED_WINDOWS_LOCAL_COPY")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--fixed-reference-only", action="store_true")
    args = parser.parse_args()
    config_path, binding_path = RUN / "candidate_config.json", RUN / "baseline_bindings.json"
    config, binding = read(config_path), read(binding_path)
    if args.fixed_reference_only:
        fixed = load_fixed(config, binding)
        pattern = fixed_reference_pattern(config, binding, fixed)
        path = RUN / "outputs/FIXED_REFERENCE_PATTERN.json"
        atomic_json(path, pattern)
        print(json.dumps({"status": pattern["status"], "patterns": pattern["patterns"],
                          "sha256": sha(path)}, sort_keys=True))
        return
    transport = read(RUN / "admission/LOCAL_TRANSPORT_VERIFIED.json")
    if transport["status"] != "LOCAL_TRANSPORT_HASH_VERIFIED":
        raise RuntimeError("LOCAL_TRANSPORT_NOT_VERIFIED")
    if sha(config_path) != transport["candidate_config_sha256"] or sha(binding_path) != transport["baseline_bindings_sha256"]:
        raise RuntimeError("LOCAL_SCIENTIFIC_INPUT_DRIFT")
    terminal_status, recovery_census = observed_terminal_status(transport)
    if terminal_status in ("SCIENTIFIC_FAILURE_NON_EVIDENCE", "INCOMPLETE_NON_EVIDENCE"):
        if terminal_status == "SCIENTIFIC_FAILURE_NON_EVIDENCE":
            stopped = RUN / "CAMPAIGN_STOP.json"
            if stopped.exists() and read(stopped).get("mode") == "main" and sha(stopped) != transport.get("campaign_stop_sha256"):
                raise RuntimeError("TRANSPORT_SCIENTIFIC_FAILURE_LATCH_UNBOUND")
        result = {"status": terminal_status, "interpretation": "NO_COMPLETE_CASE_ANALYSIS",
                  "exploratory_non_rescue": True, "registered_M1_M2_M3_N3_verdicts_unchanged": True,
                  "recovery_census": recovery_census,
                  "candidate_config_sha256": sha(config_path),
                  "baseline_bindings_sha256": sha(binding_path),
                  "local_transport_receipt_sha256": sha(RUN / "admission/LOCAL_TRANSPORT_VERIFIED.json")}
        path = RUN / "outputs/SCIENTIFIC_DECISION.json"
        atomic_json(path, result)
        print(json.dumps({"status": terminal_status, "decision_sha256": sha(path)}, sort_keys=True))
        return
    if terminal_status != "COMPUTE_COMPLETE":
        raise RuntimeError("TRANSPORT_CAMPAIGN_TERMINAL_STATUS_INVALID")
    decay = load_decay(config, transport)
    fixed = load_fixed(config, binding)
    reference_path = RUN / "outputs/FIXED_REFERENCE_PATTERN.json"
    reference = read(reference_path)
    if reference != fixed_reference_pattern(config, binding, fixed):
        raise RuntimeError("FIXED_REFERENCE_PATTERN_DRIFT")
    metrics, contrasts = [], []
    for policy in ("decay", "fixed_within_path"):
        for budget in config["budgets"]:
            for method in config["methods"]:
                for seed in config["seeds"]:
                    row = decay[(budget, method, seed)] if policy == "decay" else fixed[(method, seed)]
                    result = selected(row["epoch_log"], budget)
                    metrics.append({"policy": policy, "budget": budget, "method": method,
                                    "seed": seed, **result})
            for method in config["methods"][1:]:
                map_by = {(m["policy"], m["budget"], m["method"], m["seed"]): m for m in metrics}
                values = [map_by[(policy, budget, method, seed)]["best_correct_count"]
                          - map_by[(policy, budget, "softmax", seed)]["best_correct_count"]
                          for seed in config["seeds"]]
                contrasts.append({"policy": policy, "budget": budget, "contrast": method + "_minus_softmax",
                                  **summarize(values)})
    dense30_count_sum = sum(m["best_correct_count"] for m in metrics
                            if m["policy"] == "decay" and m["budget"] == 30 and m["method"] == "softmax")
    dense30 = dense30_count_sum / 100000
    validity = "VALID_DESCRIPTIVE_POLICY_CONTROL" if dense30_count_sum >= 60000 else "INVALID_CONTROL_DENSE_FLOOR"
    signs, combined = policy_labels(validity, reference, contrasts, config["methods"])
    result = {"status": validity, "exploratory_non_rescue": True,
              "registered_M1_M2_M3_N3_verdicts_unchanged": True,
              "official_test_access": False, "methods": config["methods"], "seeds": config["seeds"],
              "budgets": config["budgets"], "decay_cells": 210,
              "fixed_reference_is_same_seed_path_not_independent_replication": True,
              "decay_dense_budget30_mean": dense30, "sign_labels": signs,
              "recovery_census": recovery_census,
              "combined_label": combined,
              "fixed_reference_pattern_sha256": sha(reference_path),
              "contrasts": contrasts, "metrics": metrics,
              "candidate_config_sha256": sha(config_path),
              "baseline_bindings_sha256": sha(binding_path),
              "local_transport_receipt_sha256": sha(RUN / "admission/LOCAL_TRANSPORT_VERIFIED.json"),
              "compute_complete_sha256": sha(RUN / "COMPUTE_COMPLETE.json")}
    atomic_json(RUN / "outputs/SCIENTIFIC_DECISION.json", result)
    print(json.dumps({"status": validity, "sign_labels": signs,
                      "decision_sha256": sha(RUN / "outputs/SCIENTIFIC_DECISION.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
