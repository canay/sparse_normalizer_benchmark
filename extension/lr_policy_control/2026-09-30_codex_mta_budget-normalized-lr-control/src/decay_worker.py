"""One prospectively specified CIFAR-10 decay cell; never reads official test.

The original benchmark numerical core is copied byte-for-byte. This adapter
changes only AdamW's learning rate immediately before each optimizer update.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pickle
import platform
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest().upper()


def byte_sha(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def linear_lr(base: float, update: int, total_updates: int) -> float:
    if total_updates <= 0 or not 0 <= update < total_updates:
        raise RuntimeError("LR_PROGRESS_OUTSIDE_FROZEN_BUDGET")
    return base * (1.0 - update / total_updates)


def budget_adamw_class(total_updates: int, holders: list,
                       expected_parent: dict | None = None):
    """Return the single numerical intervention, importable by toy tests."""
    original_adamw = torch.optim.AdamW

    class BudgetLinearAdamW(original_adamw):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.base_lrs = [float(group["lr"]) for group in self.param_groups]
            self.update_count = 0
            self.lr_by_update: list[list[float]] = []
            holders.append(self)

        def step(self, *positional, **keywords):
            if self.update_count >= total_updates:
                raise RuntimeError("MORE_UPDATES_THAN_FROZEN_BUDGET")
            if expected_parent is not None:
                actual_parent = process_record(os.getppid())
                if (not actual_parent or actual_parent["pid"] != expected_parent["pid"]
                        or actual_parent["start_token"] != expected_parent["start_token"]):
                    raise RuntimeError("CONTROLLER_PARENT_LOST_DURING_TRAINING")
            for group, base in zip(self.param_groups, self.base_lrs, strict=True):
                group["lr"] = linear_lr(base, self.update_count, total_updates)
            self.lr_by_update.append([float(group["lr"]) for group in self.param_groups])
            result = super().step(*positional, **keywords)
            self.update_count += 1
            if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 4 * 1024 ** 3:
                raise RuntimeError("RESOURCE_GPU_ALLOCATED_CEILING_DURING_TRAINING")
            return result

    return BudgetLinearAdamW


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict, *, allow_nan: bool = False) -> None:
    if path.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=allow_nan)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def parameter_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + str(value.dtype).encode()
                      + repr(tuple(value.shape)).encode() + value.numpy().tobytes())
    return digest.hexdigest().upper()


def load_train(archive: Path, config: dict) -> dict:
    if archive.stat().st_size != config["archive_bytes"] or sha(archive) != config["archive_sha256"]:
        raise RuntimeError("CIFAR_ARCHIVE_IDENTITY_MISMATCH_BEFORE_DECODE")
    arrays, labels, members = [], [], []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():  # Preserve the original tar order.
            if not os.path.basename(member.name).startswith("data_batch"):
                continue
            if not member.isfile():
                raise RuntimeError("CIFAR_TRAIN_MEMBER_NOT_FILE")
            members.append(member.name)
            with tar.extractfile(member) as stream:
                item = pickle.load(stream, encoding="latin1")
            arrays.append(item["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
            labels.append(np.asarray(item["labels"], dtype=np.int64))
    if len(members) != 5:
        raise RuntimeError("CIFAR_TRAIN_MEMBER_CENSUS_MISMATCH")
    return {"train_x": np.concatenate(arrays), "train_y": np.concatenate(labels),
            "channels": 3, "classes": 10, "train_member_order": members}


def process_record(pid: int) -> dict | None:
    try:
        content = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError):
        return None
    fields = content[content.rfind(")") + 2:].split()
    return {"pid": pid, "ppid": int(fields[1]), "start_token": fields[19]}


def runtime_identity() -> dict:
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                             "--format=csv,noheader"], capture_output=True, text=True,
                            check=True, timeout=10).stdout.strip()
    return {"host": platform.node(), "os": platform.platform(),
            "python_executable": os.path.realpath(sys.executable), "python": sys.version,
            "torch": torch.__version__, "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": driver, "precision": torch.get_float32_matmul_precision(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}


def check_runtime(expected: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_SAME_RUNTIME_CONTROL")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision("high")
    actual = runtime_identity()
    if actual != expected:
        different = [key for key in sorted(set(actual) | set(expected))
                     if actual.get(key) != expected.get(key)]
        raise RuntimeError(f"SAME_RUNTIME_COMPARATOR_MISMATCH:{different}")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") is not None:
        raise RuntimeError("UNBOUND_NVIDIA_TF32_OVERRIDE")
    return actual


def admitted_inputs(config: dict, mode: str) -> tuple[dict, dict]:
    binding_path = RUN / "baseline_bindings.json"
    binding = read(binding_path)
    release_path = RUN / "admission" / ("MAIN_RELEASE.json" if mode == "main" else "SMOKE_RELEASE.json")
    release = read(release_path)
    if release.get("status") != ("MAIN_RELEASED" if mode == "main" else "SMOKE_RELEASED"):
        raise RuntimeError("MODE_RELEASE_NOT_ADMITTED")
    for field, path in (("candidate_config_sha256", RUN / "candidate_config.json"),
                        ("protocol_sha256", RUN / "PROTOCOL.md"),
                        ("baseline_bindings_sha256", binding_path),
                        ("worker_sha256", Path(__file__)),
                        ("campaign_sha256", RUN / "src/campaign.py"),
                        ("analyzer_sha256", RUN / "src/analyze_results.py"),
                        ("precompute_checks_sha256", RUN / "src/precompute_checks.py"),
                        ("controller_tests_sha256", RUN / "src/controller_contract_tests.py"),
                        ("release_maker_sha256", RUN / "src/make_release.py"),
                        ("smoke_validator_sha256", RUN / "src/validate_smoke.py"),
                        ("transport_verifier_sha256", RUN / "src/verify_transport.py"),
                        ("benchmark_sha256", RUN / "src/original/benchmark_r1.py"),
                        ("datasets_sha256", RUN / "src/original/datasets_r1.py")):
        if sha(path) != release[field]:
            raise RuntimeError(f"FROZEN_INPUT_CHANGED:{field}")
    if sha(RUN / "admission/PRECOMPUTE_CHECKS.json") != release["precompute_receipt_sha256"]:
        raise RuntimeError("PRECOMPUTE_CHECKS_NOT_BOUND")
    for key, path in (("controller_test_receipt_sha256", RUN / "admission/CONTROLLER_TESTS.json"),
                      ("review_reclosure_sha256", RUN / "reviews/INDEPENDENT_DIFF_REVIEW.json"),
                      ("fixed_reference_pattern_sha256", RUN / "outputs/FIXED_REFERENCE_PATTERN.json")):
        if sha(path) != release[key]:
            raise RuntimeError(f"FROZEN_INPUT_CHANGED:{key}")
    if release["benchmark_sha256"] != config["source_benchmark_sha256"]:
        raise RuntimeError("BENCHMARK_SOURCE_NOT_ORIGINAL")
    if release["datasets_sha256"] != config["source_datasets_sha256"]:
        raise RuntimeError("DATASET_SOURCE_NOT_ORIGINAL")
    if binding["status"] != "BASELINE_FIXED_RATE_WITHIN_PATH_ADMITTED" or len(binding["entries"]) != 30:
        raise RuntimeError("FIXED_RATE_BASELINE_NOT_ADMITTED")
    return binding, release


SCIENCE_EPOCH_KEYS = ("train_loss", "val_accuracy", "val_macro_f1", "val_loss",
                      "attention_density", "attention_entropy")


def classify_science(row: dict, budget: int) -> str:
    if row.get("status") == "failed_nonfinite_loss":
        return "SCIENTIFIC_FAILURE_NON_EVIDENCE"
    if row.get("status") != "completed":
        raise RuntimeError(f"INFRASTRUCTURE_FAILED_ROW:{row.get('status')}")
    log = row.get("epoch_log")
    if not isinstance(log, list) or [e.get("epoch") for e in log] != list(range(budget)):
        raise RuntimeError("ORDERED_EPOCH_CENSUS_MISMATCH")
    for epoch in log:
        for key in SCIENCE_EPOCH_KEYS:
            value = epoch.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"SCIENCE_ENDPOINT_SCHEMA_INVALID:{key}")
            if not math.isfinite(value):
                return "SCIENTIFIC_FAILURE_NON_EVIDENCE"
    for key in ("val_accuracy", "val_macro_f1", "val_loss", "train_seconds", "peak_memory_mb"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"SCIENCE_ENDPOINT_SCHEMA_INVALID:{key}")
        if not math.isfinite(value):
            return "SCIENTIFIC_FAILURE_NON_EVIDENCE"
    return "VALID_COMPLETE"


def run_cell(args: argparse.Namespace) -> dict:
    config = read(RUN / "candidate_config.json")
    if args.mode not in ("main", "smoke"):
        raise RuntimeError("INVALID_MODE")
    if args.method not in config["methods"] or args.budget not in config["budgets"]:
        raise RuntimeError("UNREGISTERED_METHOD_OR_BUDGET")
    allowed_seeds = config["seeds"] if args.mode == "main" else config["smoke_seeds"]
    if args.seed not in allowed_seeds:
        raise RuntimeError("UNREGISTERED_SEED_OR_SMOKE_MAIN_MIX")
    attempt = Path(args.attempt_dir).resolve()
    attempt.relative_to(RUN.resolve())
    if (attempt / "row.json").exists():
        raise RuntimeError("ATTEMPT_ALREADY_HAS_ROW")
    launch = read(attempt / "launch.json")
    if (launch.get("mode"), launch.get("method"), launch.get("seed"),
            launch.get("budget")) != (args.mode, args.method, args.seed, args.budget):
        raise RuntimeError("WORKER_LAUNCH_IDENTITY_MISMATCH")
    parent = process_record(os.getppid())
    if (not parent or parent["pid"] != launch.get("parent_pid")
            or parent["start_token"] != launch.get("parent_start_token")):
        raise RuntimeError("WORKER_AUTHENTIC_CONTROLLER_PARENT_REQUIRED")
    binding, release = admitted_inputs(config, args.mode)
    runtime = check_runtime(binding["runtime_identity"])
    archive = RUN / "data/cifar-10-python.tar.gz"
    raw = load_train(archive, config)
    if byte_sha(raw["train_x"]) != binding["train_x_byte_sha256"] or byte_sha(raw["train_y"]) != binding["train_y_byte_sha256"]:
        raise RuntimeError("TRAIN_ARRAY_BYTES_DIFFER_FROM_FIXED_RATE_REFERENCE")
    sys.path.insert(0, str(RUN / "src/original"))
    benchmark = importlib.import_module("benchmark_r1")
    fit, val = benchmark.split_train_val(50000, config["val_fraction"], args.seed)
    original_split = binding["splits"][str(args.seed)]
    if (byte_sha(fit) != original_split["fit_index_byte_sha256"]
            or byte_sha(val) != original_split["validation_index_byte_sha256"]):
        raise RuntimeError("SPLIT_INDEX_BYTES_DIFFER_FROM_FIXED_RATE_REFERENCE")
    train_hashes = {"train_x": byte_sha(raw["train_x"]),
                    "train_y": byte_sha(raw["train_y"])}
    member_order = list(raw["train_member_order"])
    loader_calls = []

    def train_only_loader(name, root):
        if name != "cifar10" or loader_calls:
            raise RuntimeError("UNREGISTERED_OR_REPEAT_DATASET_LOAD")
        loader_calls.append(name)
        # The original loader consumes this return value once; the raw 614 MB
        # arrays no longer remain referenced by this worker after load_bundle.
        return {k: raw.pop(k) for k in ("train_x", "train_y", "channels", "classes")}

    def forbidden_test_or_download(*unused_args, **unused_kwargs):
        raise RuntimeError("OFFICIAL_TEST_OR_NETWORK_PATH_FORBIDDEN")

    benchmark.datasets_r1.load_raw = train_only_loader
    for forbidden in ("load_cifar", "_fetch", "_fetch_any"):
        if hasattr(benchmark.datasets_r1, forbidden):
            setattr(benchmark.datasets_r1, forbidden, forbidden_test_or_download)
    if Path(benchmark.__file__).resolve() != (RUN / "src/original/benchmark_r1.py").resolve():
        raise RuntimeError("ORIGINAL_BENCHMARK_MODULE_PATH_MISMATCH")
    if Path(benchmark.datasets_r1.__file__).resolve() != (RUN / "src/original/datasets_r1.py").resolve():
        raise RuntimeError("ORIGINAL_DATASETS_MODULE_PATH_MISMATCH")
    original_constructor = benchmark.SequenceClassifier
    init: dict[str, str] = {}

    def recorded_constructor(*positional, **keywords):
        model = original_constructor(*positional, **keywords)
        init["initial_parameter_sha256"] = parameter_sha(model)
        if args.mode == "main":
            entry = next(e for e in binding["entries"]
                         if e["method"] == args.method and e["seed"] == args.seed)
            if init["initial_parameter_sha256"] != entry["initial_parameter_sha256"]:
                raise RuntimeError("INITIAL_PARAMETER_BYTES_DIFFER_FROM_FIXED_RATE_REFERENCE")
        return model

    benchmark.SequenceClassifier = recorded_constructor
    original_adamw = torch.optim.AdamW
    total_updates = args.budget * config["expected_minibatches_per_epoch"]
    holders: list = []
    torch.optim.AdamW = budget_adamw_class(total_updates, holders, parent)
    numerical_args = SimpleNamespace(tier=config["tier"], batch_size=config["batch_size"],
        patch=config["patch"], val_fraction=config["val_fraction"], max_epochs=args.budget,
        patience=config["patience"], lr=config["lr_base"], weight_decay=config["weight_decay"],
        grad_clip=config["grad_clip"], dropout=config["dropout"], train_limit=config["train_limit"],
        text_train_limit=0, synthetic_train_limit=20000, seq_len=96, vocab_size=20000,
        data_root=str(RUN / "data"), self_check=True)
    started = time.time()
    try:
        row = benchmark.run_one("cifar10", args.method, args.seed, numerical_args, torch.device("cuda"))
    finally:
        torch.optim.AdamW = original_adamw
    scientific_classification = classify_science(row, args.budget)
    if scientific_classification == "SCIENTIFIC_FAILURE_NON_EVIDENCE":
        observed = holders[0] if len(holders) == 1 else None
        observed_lrs = observed.lr_by_update if observed is not None else []
        lr_blob = b"".join(struct.pack("<d", pair[0]) for pair in observed_lrs)
        atomic_json(attempt / "scientific_failure.json",
                    {"classification": scientific_classification, "row": row,
                     "method": args.method, "seed": args.seed, "budget": args.budget,
                     "runtime": runtime,
                     "initial_parameter_sha256": init.get("initial_parameter_sha256"),
                     "optimizer_updates": observed.update_count if observed is not None else None,
                     "lr_sequence_float64_sha256": hashlib.sha256(lr_blob).hexdigest().upper(),
                     "release_sha256": sha(RUN / "admission" / ("MAIN_RELEASE.json" if args.mode == "main" else "SMOKE_RELEASE.json")),
                     "recorded_unix": time.time()}, allow_nan=True)
        return {"status": scientific_classification, "method": args.method,
                "seed": args.seed, "budget": args.budget}
    final_parent = process_record(os.getppid())
    if (not final_parent or final_parent["pid"] != parent["pid"]
            or final_parent["start_token"] != parent["start_token"]):
        raise RuntimeError("CONTROLLER_PARENT_LOST_BEFORE_SCIENCE_PROMOTION")
    if len(holders) != 1:
        raise RuntimeError("ADAMW_CONSTRUCTOR_CENSUS_MISMATCH")
    optimizer = holders[0]
    if optimizer.base_lrs != [config["lr_base"]] or optimizer.update_count != total_updates:
        raise RuntimeError("LR_BASE_OR_UPDATE_COUNT_MISMATCH")
    if row.get("epochs_run") != args.budget:
        raise RuntimeError("COMPLETED_EPOCH_CENSUS_MISMATCH")
    if [e["epoch"] for e in row["epoch_log"]] != list(range(args.budget)):
        raise RuntimeError("ORDERED_EPOCH_CENSUS_MISMATCH")
    updates_per_epoch = config["expected_minibatches_per_epoch"]
    if len(optimizer.lr_by_update) != args.budget * updates_per_epoch:
        raise RuntimeError("LR_UPDATE_TRACE_CENSUS_MISMATCH")
    lr_trace = [{"epoch": e, "first": optimizer.lr_by_update[e * updates_per_epoch][0],
                 "last": optimizer.lr_by_update[(e + 1) * updates_per_epoch - 1][0]}
                for e in range(args.budget)]
    lr_blob = b"".join(struct.pack("<d", pair[0]) for pair in optimizer.lr_by_update)
    lr_sequence_sha256 = hashlib.sha256(lr_blob).hexdigest().upper()
    row["lr_policy"] = config["lr_policy"]
    row["cfg_base_lr"] = config["lr_base"]
    row["cfg_budget"] = args.budget
    row["optimizer_updates"] = optimizer.update_count
    row["lr_epoch_trace"] = lr_trace
    row["lr_sequence_float64_sha256"] = lr_sequence_sha256
    metadata = {"mode": args.mode, "method": args.method, "seed": args.seed, "budget": args.budget,
                "started_unix": started, "ended_unix": time.time(), "pid": os.getpid(),
                "runtime": runtime, "initial_parameter_sha256": init["initial_parameter_sha256"],
                "train_x_byte_sha256": train_hashes["train_x"],
                "train_y_byte_sha256": train_hashes["train_y"],
                "fit_index_byte_sha256": byte_sha(fit), "validation_index_byte_sha256": byte_sha(val),
                "train_member_order": member_order, "train_loader_calls": loader_calls,
                "official_test_extracted": False,
                "official_test_unpickled": False, "archive_sha256": sha(archive),
                "release_sha256": sha(RUN / "admission" / ("MAIN_RELEASE.json" if args.mode == "main" else "SMOKE_RELEASE.json")),
                "baseline_bindings_sha256": sha(RUN / "baseline_bindings.json"),
                "candidate_config_sha256": sha(RUN / "candidate_config.json"),
                "protocol_sha256": sha(RUN / "PROTOCOL.md"),
                "benchmark_sha256": sha(RUN / "src/original/benchmark_r1.py"),
                "worker_sha256": sha(Path(__file__)),
                "optimizer_updates": optimizer.update_count,
                "lr_sequence_float64_sha256": lr_sequence_sha256,
                "lr_first": optimizer.lr_by_update[0][0],
                "lr_last": optimizer.lr_by_update[-1][0],
                "peak_memory_mb": row["peak_memory_mb"]}
    if row["peak_memory_mb"] > config["resource_ceiling"]["gpu_allocated_gib"] * 1024:
        raise RuntimeError("RESOURCE_GPU_ALLOCATED_CEILING")
    if benchmark.datasets_r1._PROVENANCE or benchmark.datasets_r1._CACHE:
        raise RuntimeError("DATASET_FALLBACK_OR_CACHE_USED")
    if loader_calls != ["cifar10"]:
        raise RuntimeError("TRAIN_ONLY_LOADER_CALL_CENSUS_MISMATCH")
    atomic_json(attempt / "row.json", row, allow_nan=True)
    metadata["row_sha256"] = sha(attempt / "row.json")
    atomic_json(attempt / "receipt.json", metadata)
    return {"status": "VALID_COMPLETE", "method": args.method, "seed": args.seed,
            "budget": args.budget, "row_sha256": metadata["row_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("main", "smoke"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--attempt-dir", required=True)
    args = parser.parse_args()
    try:
        result = run_cell(args)
    except Exception as exc:
        attempt = Path(args.attempt_dir).resolve()
        try:
            attempt.relative_to(RUN.resolve())
            if attempt.is_dir() and not (attempt / "failure.json").exists():
                atomic_json(attempt / "failure.json", {"classification": "INCOMPLETE_NON_EVIDENCE",
                    "error_type": type(exc).__name__, "tag": str(exc).split(":", 1)[0],
                    "message": str(exc)[:1000], "recorded_unix": time.time()})
        finally:
            print(json.dumps({"status": "INCOMPLETE_NON_EVIDENCE",
                              "tag": str(exc).split(":", 1)[0]}, sort_keys=True), flush=True)
        return 30
    print(json.dumps(result, sort_keys=True), flush=True)
    return 20 if result["status"] == "SCIENTIFIC_FAILURE_NON_EVIDENCE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
