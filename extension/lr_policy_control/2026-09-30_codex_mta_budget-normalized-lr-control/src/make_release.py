"""Create one immutable, hash-bound smoke or main release; no training."""
# ruff: noqa: E402 -- bytecode must be disabled before project imports.
from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True

from decay_worker import RUN, atomic_json, read, sha


SOURCES = {
    "candidate_config_sha256": "candidate_config.json",
    "protocol_sha256": "PROTOCOL.md",
    "baseline_bindings_sha256": "baseline_bindings.json",
    "worker_sha256": "src/decay_worker.py",
    "campaign_sha256": "src/campaign.py",
    "analyzer_sha256": "src/analyze_results.py",
    "precompute_checks_sha256": "src/precompute_checks.py",
    "controller_tests_sha256": "src/controller_contract_tests.py",
    "release_maker_sha256": "src/make_release.py",
    "smoke_validator_sha256": "src/validate_smoke.py",
    "transport_verifier_sha256": "src/verify_transport.py",
    "benchmark_sha256": "src/original/benchmark_r1.py",
    "datasets_sha256": "src/original/datasets_r1.py",
    "precompute_receipt_sha256": "admission/PRECOMPUTE_CHECKS.json",
    "controller_test_receipt_sha256": "admission/CONTROLLER_TESTS.json",
    "review_reclosure_sha256": "reviews/INDEPENDENT_DIFF_REVIEW.json",
    "fixed_reference_pattern_sha256": "outputs/FIXED_REFERENCE_PATTERN.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("smoke", "main"), required=True)
    args = parser.parse_args()
    config = read(RUN / "candidate_config.json")
    if config.get("status") != "PRECOMPUTE_REPAIR_V4_CANDIDATE":
        raise RuntimeError("CONFIG_NOT_FIXED_PRECOMPUTE_REPAIR_V4_CANDIDATE")
    review = read(RUN / "reviews/INDEPENDENT_DIFF_REVIEW.json")
    if review.get("status") != "PRECOMPUTE_READY_FOR_ENGINEERING_SMOKE":
        raise RuntimeError("INDEPENDENT_DIFF_REVIEW_NOT_READY")
    reviewed_manifest = RUN / "reviews/PRECOMPUTE_V4_INPUT_MANIFEST.json"
    if sha(reviewed_manifest) != review.get("reviewed_manifest_sha256"):
        raise RuntimeError("REVIEWED_MANIFEST_SHA_MISMATCH")
    manifest = read(reviewed_manifest)
    reviewed = {item["path"]: item["sha256"] for item in manifest["files"]}
    if len(reviewed) != len(manifest["files"]) or reviewed != review.get("reviewed_files"):
        raise RuntimeError("REVIEWED_FILE_MAP_MISMATCH")
    report_path = RUN / "reviews/PRECOMPUTE_V4_DIFF_REPORT_ORIGINAL.md"
    if sha(report_path) != review.get("review_report_sha256"):
        raise RuntimeError("REVIEW_REPORT_SHA_MISMATCH")
    if any(p.is_dir() for p in (RUN / "src").rglob("__pycache__")):
        raise RuntimeError("PROJECT_BYTECODE_CACHE_FORBIDDEN")
    checks = read(RUN / "admission/PRECOMPUTE_CHECKS.json")
    controller = read(RUN / "admission/CONTROLLER_TESTS.json")
    pattern = read(RUN / "outputs/FIXED_REFERENCE_PATTERN.json")
    if (checks.get("status") != "PRECOMPUTE_CHECKS_PASS"
            or controller.get("status") != "CONTROLLER_TESTS_PASS"
            or pattern.get("status") != "FIXED_RATE_WITHIN_PATH_REFERENCE_PATTERN_VERIFIED"):
        raise RuntimeError("PRECOMPUTE_OR_FIXED_REFERENCE_GATE_FAILED")
    if not all(v["reference_has_positive_to_negative_reversal"]
               for v in pattern["patterns"].values()):
        raise RuntimeError("FIXED_REFERENCE_PREMISE_ABSENT")
    fields = {key: sha(RUN / relative) for key, relative in SOURCES.items()}
    for key, relative in SOURCES.items():
        if key in ("precompute_receipt_sha256", "controller_test_receipt_sha256",
                   "review_reclosure_sha256"):
            continue
        if reviewed.get(relative) != fields[key]:
            raise RuntimeError(f"REVIEWED_BYTES_CHANGED:{relative}")
    if (checks["candidate_config_sha256"] != fields["candidate_config_sha256"]
            or checks["baseline_bindings_sha256"] != fields["baseline_bindings_sha256"]
            or checks["worker_sha256"] != fields["worker_sha256"]
            or checks["campaign_sha256"] != fields["campaign_sha256"]
            or checks["analyzer_sha256"] != fields["analyzer_sha256"]
            or checks["self_sha256"] != fields["precompute_checks_sha256"]):
        raise RuntimeError("PRECOMPUTE_RECEIPT_INPUT_DRIFT")
    if (controller.get("candidate_config_sha256") != fields["candidate_config_sha256"]
            or controller.get("campaign_sha256") != fields["campaign_sha256"]
            or controller.get("self_sha256") != fields["controller_tests_sha256"]):
        raise RuntimeError("CONTROLLER_RECEIPT_INPUT_DRIFT")
    if (pattern.get("config_sha256") != fields["candidate_config_sha256"]
            or pattern.get("baseline_bindings_sha256") != fields["baseline_bindings_sha256"]):
        raise RuntimeError("FIXED_PATTERN_INPUT_DRIFT")
    release = {"schema_version": 2, "status": "SMOKE_RELEASED" if args.mode == "smoke" else "MAIN_RELEASED",
               "mode": args.mode, **fields}
    if args.mode == "main":
        smoke = RUN / "admission/SMOKE_VALIDATED.json"
        if read(smoke).get("status") != "SMOKE_VALIDATED":
            raise RuntimeError("MAIN_REQUIRES_SMOKE_VALIDATED")
        release["smoke_validated_sha256"] = sha(smoke)
    path = RUN / "admission" / ("SMOKE_RELEASE.json" if args.mode == "smoke" else "MAIN_RELEASE.json")
    atomic_json(path, release)
    print(json.dumps({"status": release["status"], "release_sha256": sha(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
