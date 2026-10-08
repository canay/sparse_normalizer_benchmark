"""No-training, bound MTA precompute checks for the LR-policy control."""
# ruff: noqa: E402 -- bytecode must be disabled before project imports.
from __future__ import annotations

import importlib
import json
import math
import pickle
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch

from analyze_results import policy_labels, summarize
from decay_worker import (RUN, atomic_json, budget_adamw_class, byte_sha,
                          check_runtime, classify_science, linear_lr, load_train, parameter_sha,
                          read, sha)


def classifier_cases() -> int:
    epoch = {"epoch": 0, "train_loss": 1.0, "val_accuracy": 0.5,
             "val_macro_f1": 0.5, "val_loss": 1.0,
             "attention_density": 1.0, "attention_entropy": 1.0}
    row = {"status": "completed", "epoch_log": [epoch],
           "val_accuracy": 0.5, "val_macro_f1": 0.5,
           "val_loss": 1.0, "train_seconds": 1.0, "peak_memory_mb": 1.0}
    if classify_science({"status": "failed_nonfinite_loss"}, 1) != "SCIENTIFIC_FAILURE_NON_EVIDENCE":
        raise RuntimeError("CLASSIFIER_FAILED_NONFINITE_STATUS_TEST")
    if classify_science(row, 1) != "VALID_COMPLETE":
        raise RuntimeError("CLASSIFIER_FINITE_ROW_TEST")
    bad = {**row, "epoch_log": [{**epoch, "val_loss": float("nan")}]}
    if classify_science(bad, 1) != "SCIENTIFIC_FAILURE_NON_EVIDENCE":
        raise RuntimeError("CLASSIFIER_NONFINITE_ENDPOINT_TEST")
    malformed = {**row, "epoch_log": [{**epoch, "val_accuracy": "invalid"}]}
    try:
        classify_science(malformed, 1)
    except RuntimeError as exc:
        if "SCIENCE_ENDPOINT_SCHEMA_INVALID" not in str(exc):
            raise
    else:
        raise RuntimeError("CLASSIFIER_SCHEMA_INVALID_TEST")
    return 4


def toy_adamw_equivalence() -> dict:
    before = torch.random.get_rng_state().clone()
    a = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    b = torch.nn.Parameter(a.detach().clone())
    holders = []
    toy_type = budget_adamw_class(7, holders)
    test = toy_type([a], lr=0.001, weight_decay=0.0001)
    reference = torch.optim.AdamW([b], lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(reference, lr_lambda=lambda t: 1 - t / 7)
    observed = []
    for step in range(7):
        grad = torch.tensor([0.25 + step / 10, -0.5], dtype=torch.float64)
        a.grad = grad.clone()
        b.grad = grad.clone()
        test.step()
        reference.step()
        scheduler.step()
        if not torch.equal(a.detach(), b.detach()):
            raise RuntimeError(f"TOY_ADAMW_NOT_BITWISE_EQUAL_AT_STEP:{step}")
        observed.append(test.param_groups[0]["lr"])
    try:
        test.step()
    except RuntimeError as exc:
        if "MORE_UPDATES_THAN_FROZEN_BUDGET" not in str(exc):
            raise
    else:
        raise RuntimeError("TOY_ADAMW_EXTRA_STEP_NOT_REJECTED")
    if not torch.equal(before, torch.random.get_rng_state()):
        raise RuntimeError("TOY_ADAMW_CHANGED_TORCH_RNG_STATE")
    if len(holders) != 1 or observed != [linear_lr(0.001, t, 7) for t in range(7)]:
        raise RuntimeError("TOY_ADAMW_LR_SEQUENCE_MISMATCH")
    return {"steps": 7, "bitwise_parameter_equal": True, "rng_unchanged": True,
            "extra_step_rejected": True}


def main() -> None:
    if sys.platform != "linux":
        raise RuntimeError("MTA_LINUX_RUNTIME_REQUIRED")
    config = read(RUN / "candidate_config.json")
    binding = read(RUN / "baseline_bindings.json")
    runtime = check_runtime(binding["runtime_identity"])
    archive = RUN / "data/cifar-10-python.tar.gz"
    calls = []
    original_pickle_load = pickle.load

    def spy(stream, **kwargs):
        calls.append(getattr(stream, "name", "member"))
        return original_pickle_load(stream, **kwargs)

    pickle.load = spy
    try:
        data = load_train(archive, config)
    finally:
        pickle.load = original_pickle_load
    if len(calls) != 5 or len(data["train_member_order"]) != 5:
        raise RuntimeError("TRAIN_ONLY_PICKLE_DECODE_CENSUS_NOT_FIVE")
    if byte_sha(data["train_x"]) != binding["train_x_byte_sha256"]:
        raise RuntimeError("TRAIN_X_PARITY_FAILED")
    if byte_sha(data["train_y"]) != binding["train_y_byte_sha256"]:
        raise RuntimeError("TRAIN_Y_PARITY_FAILED")
    if sha(RUN / "src/original/benchmark_r1.py") != config["source_benchmark_sha256"]:
        raise RuntimeError("ORIGINAL_BENCHMARK_HASH_FAILED")
    if sha(RUN / "src/original/datasets_r1.py") != config["source_datasets_sha256"]:
        raise RuntimeError("ORIGINAL_DATASET_SOURCE_HASH_FAILED")
    sys.path.insert(0, str(RUN / "src/original"))
    original = importlib.import_module("benchmark_r1")
    for seed in config["seeds"] + config["smoke_seeds"]:
        fit, val = original.split_train_val(50000, config["val_fraction"], seed)
        old = binding["splits"][str(seed)]
        if len(fit) != 40000 or len(val) != 10000:
            raise RuntimeError(f"SPLIT_SIZE_FAILED:{seed}")
        if byte_sha(fit) != old["fit_index_byte_sha256"] or byte_sha(val) != old["validation_index_byte_sha256"]:
            raise RuntimeError(f"SPLIT_INDEX_PARITY_FAILED:{seed}")
        if np.bincount(data["train_y"][fit], minlength=10).tolist() != old["fit_class_counts"]:
            raise RuntimeError(f"FIT_CLASS_PARITY_FAILED:{seed}")
    init_checked = 0
    for entry in binding["entries"]:
        original.set_seed(entry["seed"])
        model = original.SequenceClassifier("patch", 48, 10, entry["method"],
                                            64, 4, 2, 64, 0.1)
        if parameter_sha(model) != entry["initial_parameter_sha256"]:
            raise RuntimeError(f"INIT_PARITY_FAILED:{entry['method']}:{entry['seed']}")
        init_checked += 1
        del model
    if init_checked != 30:
        raise RuntimeError("INIT_CENSUS_NOT_THIRTY")
    for budget in config["budgets"]:
        total = budget * config["expected_minibatches_per_epoch"]
        first = linear_lr(config["lr_base"], 0, total)
        last = linear_lr(config["lr_base"], total - 1, total)
        if first != config["lr_base"] or not math.isclose(last, config["lr_base"] / total, rel_tol=1e-12):
            raise RuntimeError("LR_BOUNDARY_FAILED")
        try:
            linear_lr(config["lr_base"], total, total)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("LR_EXTRA_STEP_NOT_REJECTED")
    toy = toy_adamw_equivalence()
    classifier_count = classifier_cases()
    zero = summarize([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    if zero["sum_correct_count_differences"] != 0 or zero["mean"] != 0:
        raise RuntimeError("EXACT_ZERO_COUNT_SUM_NOT_PRESERVED")
    synthetic_reference = {"patterns": {m: {"reference_has_positive_to_negative_reversal": True}
                                       for m in config["methods"][1:]}}
    synthetic_contrasts = [{"policy": "decay", "budget": budget,
                            "contrast": method + "_minus_softmax",
                            "sum_correct_count_differences": (0 if budget == 3 else -1)}
                           for method in config["methods"][1:] for budget in (3, 30)]
    zero_labels, zero_combined = policy_labels("VALID_DESCRIPTIVE_POLICY_CONTROL",
                                               synthetic_reference, synthetic_contrasts,
                                               config["methods"])
    if zero_combined != "FAILS_BOTH" or any(v["label"] != "fails" for v in zero_labels.values()):
        raise RuntimeError("EXACT_ZERO_FALSE_PERSISTENCE_LABEL")
    with tempfile.TemporaryDirectory(prefix="lr-control-atomic-") as name:
        path = Path(name) / "exclusive.json"
        atomic_json(path, {"a": 1})
        try:
            atomic_json(path, {"a": 2})
        except RuntimeError as exc:
            if "REFUSE_OVERWRITE" not in str(exc):
                raise
        else:
            raise RuntimeError("ATOMIC_JSON_OVERWRITE_NOT_REJECTED")
    result = {"status": "PRECOMPUTE_CHECKS_PASS", "no_training": True,
              "runtime_identity": runtime, "baseline_entries": init_checked,
              "split_seeds": len(config["seeds"] + config["smoke_seeds"]),
              "train_pickle_calls": len(calls), "toy_adamw": toy,
              "classifier_cases": classifier_count,
              "exact_zero_count_rule_tested": True,
              "lr_budgets": config["budgets"], "archive_sha256": sha(archive),
              "candidate_config_sha256": sha(RUN / "candidate_config.json"),
              "baseline_bindings_sha256": sha(RUN / "baseline_bindings.json"),
              "worker_sha256": sha(RUN / "src/decay_worker.py"),
              "campaign_sha256": sha(RUN / "src/campaign.py"),
              "analyzer_sha256": sha(RUN / "src/analyze_results.py"),
              "self_sha256": sha(Path(__file__))}
    atomic_json(RUN / "admission/PRECOMPUTE_CHECKS.json", result)
    print(json.dumps({"status": result["status"], "baseline_entries": init_checked,
                      "split_seeds": result["split_seeds"],
                      "receipt_sha256": sha(RUN / "admission/PRECOMPUTE_CHECKS.json")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
