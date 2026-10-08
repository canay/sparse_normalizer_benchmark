"""Durable N3 main-trajectory runner.

Pilot and scientific configurations share this implementation but write to
different output roots.  A completed unit is immutable and is skipped only
after its recorded artifact hashes have been verified.
"""
from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import time
import uuid

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.nn import functional as F

from training_models import build_model, generate_split, make_batch_schedule, stable_seed
from training_operators import (
    InvalidControlError,
    NORM_ZERO_TOLERANCE,
    TAIL_LEAK_MASS,
    TOP_K,
    TOP_SCORE_RANGE_CEILING,
    backward_diagnostics,
    entropy,
    weights,
)


def invalid_control(exc: BaseException) -> bool:
    return isinstance(exc, InvalidControlError)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    """Persist directory entries on the Linux execution host."""
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def exclusive_atomic_json(path: Path, obj) -> None:
    """Atomically create a JSON record without ever replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        sync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, tmp)
        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
        sync_directory(destination.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def atomic_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def guarded_tree_bytes(run_root: Path, output_root: Path, cfg: dict) -> int:
    configured = cfg.get("disk_guard_paths")
    if configured is None:
        paths = [run_root / "inputs", output_root]
    else:
        paths = []
        for relative in configured:
            path = (run_root / relative).resolve()
            if not path.is_relative_to(run_root):
                raise ValueError(f"disk guard path escapes run root: {relative}")
            paths.append(path)
        if not any(output_root == path or output_root.is_relative_to(path) for path in paths):
            raise ValueError("active output root is absent from disk_guard_paths")
    unique = {path for path in paths}
    return sum(tree_bytes(path) for path in unique)


def resident_bytes():
    path = Path("/proc/self/statm")
    if not path.is_file():
        return None
    return int(path.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def memory_guard(cfg):
    ceiling = cfg.get("maximum_resident_memory_bytes")
    if ceiling is not None:
        used = resident_bytes()
        if used is None or used > ceiling:
            raise RuntimeError("resident RAM guard unavailable or exceeded")


def producer_binding(run_root: Path) -> dict:
    sources = {}
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.glob("*.py")):
        sources[f"src/{path.name}"] = sha256(path)
    for path in sorted((run_root / "tests").glob("*.py")):
        sources[f"tests/{path.name}"] = sha256(path)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "sources": sources,
        "environment": {
            "os_family": platform.system(),
            "architecture": platform.machine(),
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": gpu_name,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled(),
            "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
        },
    }


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # access denied is not proof of absence
        try:
            status = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
                return True
            return status.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def configure_determinism(threads: int) -> None:
    if not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("invalid registered thread count")
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def require_config_binding(run_root: Path, cfg: dict, cfg_sha: str) -> None:
    path = run_root / "config.json"
    if not path.is_file() or sha256(path) != cfg_sha:
        raise RuntimeError("config file/hash mismatch")
    if json.loads(path.read_text(encoding="utf-8")) != cfg:
        raise RuntimeError("config content/hash mismatch")


def require_run_open(run_root: Path) -> None:
    if (run_root / "INVALID_CONTROL.json").exists():
        raise InvalidControlError("whole campaign already INVALID_CONTROL")


def persist_invalid_control(function):
    @functools.wraps(function)
    def guarded(run_root, *args, **kwargs):
        require_run_open(run_root)
        try:
            return function(run_root, *args, **kwargs)
        except InvalidControlError as exc:
            marker = run_root / "INVALID_CONTROL.json"
            if not marker.exists():
                exclusive_atomic_json(marker, {"status": "INVALID_CONTROL", "error": str(exc),
                    "function": function.__name__, "recorded_at_unix": time.time()})
            raise
    return guarded


class ControllerLock:
    """Single-controller lock that preserves stale and released lock evidence."""

    def __init__(self, output_root: Path, config_sha256: str):
        self.output_root = output_root
        # *.lock is explicitly excluded by the measured CALISMALAR/.stglobalignore.
        self.path = output_root / "controller.lock"
        self.payload = {
            "schema_version": 1,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "config_sha256": config_sha256,
            "acquired_at_unix": time.time(),
            "lock_id": uuid.uuid4().hex,
        }
        self.acquired = False

    def __enter__(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                exclusive_atomic_json(self.path, self.payload)
                self.acquired = True
                return self
            except FileExistsError:
                prior = json.loads(self.path.read_text(encoding="utf-8"))
                same_host = prior.get("host") == socket.gethostname()
                prior_pid = int(prior.get("pid", -1))
                if (same_host and process_is_alive(prior_pid)) or not same_host:
                    raise RuntimeError(f"active controller lock exists: {self.path}")
                stale = self.output_root / "locks" / "stale" / (
                    f"controller.{int(time.time())}.{prior.get('lock_id', 'unknown')}.json"
                )
                stale.parent.mkdir(parents=True, exist_ok=True)
                os.replace(self.path, stale)
        raise RuntimeError(f"could not acquire controller lock: {self.path}")

    def __exit__(self, exc_type, exc, tb):
        if self.acquired and self.path.exists():
            released = self.output_root / "locks" / "released" / (
                f"controller.{int(time.time())}.{self.payload['lock_id']}.json"
            )
            released.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.path, released)
        self.acquired = False


def prepare_run_clock(output_root: Path, cfg_sha: str, binding: dict) -> float:
    path = output_root / "run_clock.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("config_sha256") != cfg_sha or data.get("producer_binding") != binding:
            raise RuntimeError(f"run-clock binding mismatch: {path}")
        return float(data["started_at_unix"])
    started = time.time()
    exclusive_atomic_json(
        path,
        {
            "schema_version": 1,
            "started_at_unix": started,
            "config_sha256": cfg_sha,
            "producer_binding": binding,
        },
    )
    return started


def verify_complete(unit_root: Path, expected: dict) -> bool:
    marker = unit_root / "complete.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text(encoding="utf-8"))
    if data.get("status") != "COMPLETED":
        raise RuntimeError(f"invalid completion marker: {marker}")
    recorded = [item["path"] for item in data["artifacts"]]
    actual = {p.relative_to(unit_root).as_posix()
              for folder in ("metrics", "predictions", "checkpoints", "dynamics")
              for p in (unit_root / folder).glob("*") if p.is_file()}
    if not recorded or len(recorded) != len(set(recorded)) or set(recorded) != actual:
        raise RuntimeError(f"completion artifact census mismatch: {marker}")
    for item in data["artifacts"]:
        artifact = unit_root / item["path"]
        if not artifact.resolve().is_relative_to(unit_root.resolve()):
            raise RuntimeError("completion path escapes unit")
        if not artifact.is_file() or sha256(artifact) != item["sha256"]:
            raise RuntimeError(f"completed artifact mismatch: {artifact}")
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"completed binding mismatch for {key}: {marker}")
    if "evaluation_steps" in expected:
        planned = {f"step_{int(step):04d}" for step in expected["evaluation_steps"]}
        for folder, suffix in (("metrics", ".json"), ("predictions", ".npz"), ("checkpoints", ".pt")):
            observed = {p.stem for p in (unit_root / folder).glob("*" + suffix)}
            if observed != planned:
                raise RuntimeError("planned evaluation census mismatch")
        for step in expected["evaluation_steps"]:
            verify_published_evaluation(unit_root, step, expected["config_sha256"], expected["producer_binding"])
    return True


def prepare_inputs(run_root: Path, cfg: dict, cfg_sha: str, binding: dict, r: int, seed: int) -> Path:
    target = run_root / "inputs" / f"data_r{r}_seed{seed}.npz"
    schedule_path = run_root / "inputs" / f"batches_r{r}_seed{seed}.npy"
    meta_path = run_root / "inputs" / f"input_r{r}_seed{seed}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("config_sha256") != cfg_sha or meta.get("producer_binding") != binding:
            raise RuntimeError(f"input binding mismatch: {meta_path}")
        for item in meta["artifacts"]:
            p = run_root / item["path"]
            if not p.is_file() or sha256(p) != item["sha256"]:
                raise RuntimeError(f"input hash mismatch: {p}")
        return target

    train = generate_split(seed, r, "train", cfg["train_size"], cfg["n"])
    val = generate_split(seed, r, "validation", cfg["validation_size"], cfg["n"])
    atomic_npz(
        target,
        train_features=train.features,
        train_values=train.values,
        train_labels=train.labels,
        train_relevance=train.relevance,
        validation_features=val.features,
        validation_values=val.values,
        validation_labels=val.labels,
        validation_relevance=val.relevance,
    )
    schedule = make_batch_schedule(seed, r, cfg["train_size"], cfg["batch_size"], cfg["updates"])
    tmp = schedule_path.with_name(schedule_path.name + f".{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, schedule, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, schedule_path)
    sync_directory(schedule_path.parent)
    exclusive_atomic_json(
        meta_path,
        {
            "schema_version": 1,
            "r": r,
            "seed": seed,
            "z_semantics": "binary relevance indicator",
            "substreams": ["support", "marker_noise", "values", "nuisance", "batch_schedule"],
            "config_sha256": cfg_sha,
            "producer_binding": binding,
            "artifacts": [
                {"path": target.relative_to(run_root).as_posix(), "sha256": sha256(target), "bytes": target.stat().st_size},
                {"path": schedule_path.relative_to(run_root).as_posix(), "sha256": sha256(schedule_path), "bytes": schedule_path.stat().st_size},
            ],
        },
    )
    return target


def prepare_initialization(run_root: Path, cfg_sha: str, binding: dict, architecture: str, r: int, seed: int) -> Path:
    path = run_root / "inputs" / f"initial_{architecture}_r{r}_seed{seed}.pt"
    meta_path = path.with_suffix(".json")
    init_seed = stable_seed(seed, r, architecture, "initialization")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if sha256(path) != meta["sha256"] or meta["init_seed"] != init_seed or meta.get("config_sha256") != cfg_sha or meta.get("producer_binding") != binding:
            raise RuntimeError(f"initialization hash mismatch: {path}")
        return path
    model = build_model(architecture, init_seed)
    atomic_torch(path, {"init_seed": init_seed, "model_state": model.state_dict()})
    exclusive_atomic_json(meta_path, {"schema_version": 1, "init_seed": init_seed, "sha256": sha256(path), "bytes": path.stat().st_size, "config_sha256": cfg_sha, "producer_binding": binding})
    return path


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def describe_dose(scores, upstream, null_seed=0):
    """Descriptive hypothetical gradients cannot invalidate a valid baseline."""
    try:
        dose = backward_diagnostics(scores, upstream, null_seed)
    except InvalidControlError as exc:
        return {"dose_status": "UNDEFINED_HYPOTHETICAL", "dose_error": str(exc),
                "dose_undefined_rows": scores.numel() // scores.shape[-1]}
    result = {"dose_status": "OBSERVED"}
    for name, values in dose.items():
        finite = values[torch.isfinite(values)]
        result[f"dose_{name}_finite_rows"] = finite.numel()
        result[f"dose_{name}_undefined_rows"] = values.numel() - finite.numel()
        result[f"dose_{name}_mean"] = float(finite.mean()) if finite.numel() else None
        result[f"dose_{name}_max"] = float(finite.max()) if finite.numel() else None
    return result


def evaluate(model, arrays: dict[str, np.ndarray], r: int, mode: str, device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(arrays["validation_features"]).to(device)
        v = torch.from_numpy(arrays["validation_values"]).to(device)
        y = torch.from_numpy(arrays["validation_labels"]).to(device)
        rel = torch.from_numpy(arrays["validation_relevance"]).to(device)
        scores = model.scores(x)
        w = weights(scores, mode)
        gamma = F.softplus(model.raw_gamma).to(w.dtype)
        logits = gamma * np.sqrt(r) * (w * v.to(w.dtype)).sum(-1) + model.bias.to(w.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, y.to(logits.dtype))
        probs = torch.sigmoid(logits)
        relevant_mass = (w * rel.to(w.dtype)).sum(-1)
        support = (w > 0).sum(-1)
        metrics = {
            "validation_cross_entropy": float(loss.item()),
            "validation_accuracy": float(((probs >= 0.5) == (y >= 0.5)).double().mean().item()),
            "mean_relevant_mass": float(relevant_mass.double().mean().item()),
            "mean_entropy": float(entropy(w.double()).mean().item()),
            "mean_support": float(support.double().mean().item()),
            "gamma": float(gamma.item()),
            "class_balance": float(y.double().mean().item()),
        }
        if mode in {"S", "G", "N"}:
            upstream = (
                (torch.sigmoid(logits.double()) - y.double())[:, None]
                * gamma.double()
                * np.sqrt(r)
                * v.double()
                / y.numel()
            )
            metrics.update(describe_dose(scores, upstream))
        prediction = {
            "logits": logits.detach().cpu().numpy(),
            "probabilities": probs.detach().cpu().numpy(),
            "labels": y.detach().cpu().numpy(),
            "relevant_mass": relevant_mass.detach().cpu().numpy(),
            "support": support.detach().cpu().numpy(),
        }
    model.train()
    return metrics, prediction


def optimizer_for(model, cfg: dict):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["betas"]),
        eps=cfg["adam_epsilon"],
    )


def published_evaluation_paths(unit_root: Path, step: int) -> dict[str, Path]:
    return {
        "metric": unit_root / "metrics" / f"step_{step:04d}.json",
        "prediction": unit_root / "predictions" / f"step_{step:04d}.npz",
        "checkpoint": unit_root / "checkpoints" / f"step_{step:04d}.pt",
    }


def verify_published_evaluation(unit_root: Path, step: int, cfg_sha: str, binding: dict) -> None:
    destinations = published_evaluation_paths(unit_root, step)
    if not all(path.is_file() for path in destinations.values()):
        raise RuntimeError(f"incomplete published evaluation triplet at step {step}")
    metric = json.loads(destinations["metric"].read_text(encoding="utf-8"))
    if metric.get("step") != step or metric.get("config_sha256") != cfg_sha or metric.get("producer_binding") != binding:
        raise RuntimeError(f"published evaluation binding mismatch at step {step}")
    if metric.get("prediction_sha256") != sha256(destinations["prediction"]) or metric.get("checkpoint_sha256") != sha256(destinations["checkpoint"]):
        raise RuntimeError(f"published evaluation triplet mismatch at step {step}")
    checkpoint = torch.load(destinations["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("step") != step or checkpoint.get("config_sha256") != cfg_sha or checkpoint.get("producer_binding") != binding:
        raise RuntimeError(f"published evaluation checkpoint mismatch at step {step}")


def retire_evaluation_transaction(transaction_root: Path) -> None:
    """Atomically leave the stable namespace before best-effort cleanup."""
    cleanup_root = transaction_root.with_name(f".published_{transaction_root.name}.{uuid.uuid4().hex}.cleanup")
    os.rename(transaction_root, cleanup_root)
    for child in tuple(cleanup_root.iterdir()):
        if child.is_file():
            child.unlink()
    try:
        cleanup_root.rmdir()
    except OSError:
        # Preserve unknown residue for inspection. It remains under the disk
        # guard but cannot be mistaken for a stable step_* transaction.
        pass


def publish_evaluation_transaction(unit_root: Path, transaction_root: Path, step: int, cfg_sha: str, binding: dict) -> None:
    staged = {
        "metric": transaction_root / "metric.json",
        "prediction": transaction_root / "prediction.npz",
        "checkpoint": transaction_root / "checkpoint.pt",
    }
    if not all(path.is_file() for path in staged.values()):
        raise RuntimeError(f"incomplete stable evaluation transaction: {transaction_root}")
    metric = json.loads(staged["metric"].read_text(encoding="utf-8"))
    if metric.get("step") != step or metric.get("config_sha256") != cfg_sha or metric.get("producer_binding") != binding:
        raise RuntimeError(f"evaluation transaction binding mismatch: {transaction_root}")
    if metric.get("prediction_sha256") != sha256(staged["prediction"]) or metric.get("checkpoint_sha256") != sha256(staged["checkpoint"]):
        raise RuntimeError(f"evaluation transaction artifact mismatch: {transaction_root}")
    checkpoint = torch.load(staged["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("step") != step or checkpoint.get("config_sha256") != cfg_sha or checkpoint.get("producer_binding") != binding:
        raise RuntimeError(f"evaluation transaction checkpoint mismatch: {transaction_root}")
    destinations = published_evaluation_paths(unit_root, step)
    for name in ("prediction", "checkpoint", "metric"):
        destination = destinations[name]
        if destination.exists():
            if sha256(destination) != sha256(staged[name]):
                raise RuntimeError(f"partial evaluation publication mismatch: {destination}")
        else:
            atomic_copy(staged[name], destination)
    verify_published_evaluation(unit_root, step, cfg_sha, binding)
    retire_evaluation_transaction(transaction_root)


def recover_evaluation_transactions(unit_root: Path, cfg_sha: str, binding: dict) -> None:
    transaction_parent = unit_root / "transactions"
    if not transaction_parent.is_dir():
        return
    for transaction_root in sorted(transaction_parent.glob("step_*")):
        if not transaction_root.is_dir():
            raise RuntimeError(f"invalid evaluation transaction path: {transaction_root}")
        step = int(transaction_root.name.split("_")[-1])
        staged = (
            transaction_root / "metric.json",
            transaction_root / "prediction.npz",
            transaction_root / "checkpoint.pt",
        )
        if not all(path.is_file() for path in staged):
            destinations = published_evaluation_paths(unit_root, step)
            if not all(path.is_file() for path in destinations.values()):
                raise RuntimeError(f"incomplete stable evaluation transaction: {transaction_root}")
            verify_published_evaluation(unit_root, step, cfg_sha, binding)
            retire_evaluation_transaction(transaction_root)
            continue
        publish_evaluation_transaction(unit_root, transaction_root, step, cfg_sha, binding)


def save_evaluation(
    unit_root: Path,
    step: int,
    metrics: dict,
    prediction: dict,
    model,
    optimizer,
    cfg_sha: str,
    binding: dict,
) -> None:
    metric_path = unit_root / "metrics" / f"step_{step:04d}.json"
    prediction_path = unit_root / "predictions" / f"step_{step:04d}.npz"
    checkpoint_path = unit_root / "checkpoints" / f"step_{step:04d}.pt"
    if any(p.exists() for p in (metric_path, prediction_path, checkpoint_path)):
        if not all(p.exists() for p in (metric_path, prediction_path, checkpoint_path)):
            transaction_root = unit_root / "transactions" / f"step_{step:04d}"
            if not transaction_root.is_dir():
                raise RuntimeError(f"partial checkpoint triplet without recovery transaction at step {step}")
            publish_evaluation_transaction(unit_root, transaction_root, step, cfg_sha, binding)
            return
        existing = json.loads(metric_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != cfg_sha or existing.get("producer_binding") != binding or existing.get("step") != step:
            raise RuntimeError(f"existing evaluation binding mismatch at step {step}")
        if existing.get("prediction_sha256") != sha256(prediction_path) or existing.get("checkpoint_sha256") != sha256(checkpoint_path):
            raise RuntimeError(f"existing evaluation artifact mismatch at step {step}")
        return

    transaction_parent = unit_root / "transactions"
    transaction_parent.mkdir(parents=True, exist_ok=True)
    transaction_root = transaction_parent / f"step_{step:04d}"
    if not transaction_root.exists():
        staged_root = transaction_parent / f".step_{step:04d}.{uuid.uuid4().hex}.tmp"
        staged_root.mkdir()
        staged_prediction = staged_root / "prediction.npz"
        staged_checkpoint = staged_root / "checkpoint.pt"
        staged_metric = staged_root / "metric.json"
        atomic_npz(staged_prediction, **prediction)
        atomic_torch(
            staged_checkpoint,
            {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "config_sha256": cfg_sha,
                "producer_binding": binding,
            },
        )
        exclusive_atomic_json(
            staged_metric,
            {
                "schema_version": 1,
                "step": step,
                "config_sha256": cfg_sha,
                "producer_binding": binding,
                **metrics,
                "prediction_sha256": sha256(staged_prediction),
                "checkpoint_sha256": sha256(staged_checkpoint),
            },
        )
        os.rename(staged_root, transaction_root)
        sync_directory(transaction_parent)
    publish_evaluation_transaction(unit_root, transaction_root, step, cfg_sha, binding)


def latest_checkpoint(unit_root: Path, cfg_sha: str, binding: dict) -> tuple[int, Path | None]:
    found = []
    for path in (unit_root / "checkpoints").glob("step_*.pt"):
        step = int(path.stem.split("_")[-1])
        metric = json.loads((unit_root / "metrics" / f"step_{step:04d}.json").read_text(encoding="utf-8"))
        if metric.get("config_sha256") != cfg_sha or metric.get("producer_binding") != binding:
            raise RuntimeError(f"metric binding mismatch: {path}")
        if sha256(path) != metric["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        pred = unit_root / "predictions" / f"step_{step:04d}.npz"
        if sha256(pred) != metric["prediction_sha256"]:
            raise RuntimeError(f"prediction hash mismatch: {pred}")
        found.append((step, path))
    if not found:
        return 0, None
    step, path = max(found)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["config_sha256"] != cfg_sha
        or payload.get("producer_binding") != binding
        or payload["step"] != step
    ):
        raise RuntimeError(f"checkpoint binding mismatch: {path}")
    return step, path


@persist_invalid_control
def train_unit(
    run_root: Path,
    output_root: Path,
    cfg: dict,
    cfg_sha: str,
    binding: dict,
    global_deadline: float,
    architecture: str,
    r: int,
    seed: int,
    mode: str,
    interrupt_after_step: int | None,
    sleep_per_step: float,
) -> str:
    require_config_binding(run_root, cfg, cfg_sha)
    if mode not in {"D", "S"}:
        raise ValueError("main trajectories admit D and S only")
    unit_id = f"{architecture}_r{r}_seed{seed}_{mode}"
    unit_root = output_root / "main" / unit_id
    unit_root.mkdir(parents=True, exist_ok=True)
    invalid_marker = unit_root / "invalid_control.json"
    if invalid_marker.exists():
        raise InvalidControlError(f"frozen unit already marked INVALID_CONTROL: {unit_id}")

    data_path = prepare_inputs(run_root, cfg, cfg_sha, binding, r, seed)
    init_path = prepare_initialization(run_root, cfg_sha, binding, architecture, r, seed)
    expected = {
        "unit_id": unit_id,
        "completed_updates": cfg["updates"],
        "config_sha256": cfg_sha,
        "input_sha256": sha256(data_path),
        "initialization_sha256": sha256(init_path),
        "producer_binding": binding,
        "evaluation_steps": cfg["evaluation_steps"],
    }
    if verify_complete(unit_root, expected):
        return "SKIPPED_VERIFIED"
    attempt = unit_root / "attempt.json"
    if not attempt.exists():
        exclusive_atomic_json(
            attempt,
            {
                "schema_version": 1,
                "unit_id": unit_id,
                "architecture": architecture,
                "r": r,
                "seed": seed,
                "mode": mode,
                "config_sha256": cfg_sha,
                "input_sha256": sha256(data_path),
                "initialization_sha256": sha256(init_path),
                "producer_binding": binding,
                "started_at_unix": time.time(),
            },
        )
    else:
        attempt_data = json.loads(attempt.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if key in {"completed_updates", "evaluation_steps"}:
                continue
            if attempt_data.get(key) != value:
                raise RuntimeError(f"attempt binding mismatch for {key}: {unit_id}")

    arrays = load_arrays(data_path)
    schedule = np.load(run_root / "inputs" / f"batches_r{r}_seed{seed}.npy", allow_pickle=False)
    device = torch.device(cfg["device"])
    model = build_model(architecture, stable_seed(seed, r, architecture, "initialization")).to(device)
    initial = torch.load(init_path, map_location="cpu", weights_only=False)
    model.load_state_dict(initial["model_state"])
    trajectory_seed = stable_seed(seed, r, architecture, "trajectory_rng")
    torch.manual_seed(trajectory_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trajectory_seed)
    optimizer = optimizer_for(model, cfg)
    recover_evaluation_transactions(unit_root, cfg_sha, binding)
    start_step, checkpoint_path = latest_checkpoint(unit_root, cfg_sha, binding)
    if checkpoint_path is not None:
        # RNG state tensors are CPU ByteTensors even for CUDA trajectories.
        # Optimizer loading remaps its tensors to the parameter device.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    else:
        metrics, prediction = evaluate(model, arrays, r, mode, device)
        save_evaluation(unit_root, 0, metrics, prediction, model, optimizer, cfg_sha, binding)

    eval_steps = set(cfg["evaluation_steps"])
    phase_started = time.time()
    attempt_id = uuid.uuid4().hex
    x_np, v_np, y_np = arrays["train_features"], arrays["train_values"], arrays["train_labels"]
    for step in range(start_step + 1, cfg["updates"] + 1):
        idx = schedule[step - 1]
        x = torch.from_numpy(x_np[idx]).to(device)
        v = torch.from_numpy(v_np[idx]).to(device)
        y = torch.from_numpy(y_np[idx]).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(x, v, r, mode, weights)
        loss = F.binary_cross_entropy_with_logits(logits, y.to(logits.dtype))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss: {unit_id} step {step}")
        try:
            loss.backward()
            optimizer.step()
        except Exception as exc:
            if invalid_control(exc):
                raise InvalidControlError(str(exc)) from exc
            raise
        if torch.cuda.is_available() and torch.cuda.max_memory_reserved() > cfg["maximum_cuda_memory_bytes"]:
            raise RuntimeError("CUDA reserved-memory ceiling exceeded")
        if step in eval_steps:
            metrics, prediction = evaluate(model, arrays, r, mode, device)
            metrics["training_loss_last_batch"] = float(loss.item())
            save_evaluation(unit_root, step, metrics, prediction, model, optimizer, cfg_sha, binding)
        if step % 16 == 0:
            memory_guard(cfg)
            if time.time() > global_deadline:
                raise TimeoutError(f"global run wall-time ceiling exceeded: {unit_id}")
            if guarded_tree_bytes(run_root, output_root, cfg) > cfg["max_disk_bytes"]:
                raise RuntimeError("run input-plus-output disk ceiling exceeded")
            if shutil.disk_usage(output_root).free < cfg["minimum_free_disk_bytes"]:
                raise RuntimeError("free disk fell below configured guard")
            atomic_json(output_root / "heartbeat.json", {"run_id": cfg["run_id"], "unit_id": unit_id,
                "phase": "main", "pid": os.getpid(), "attempt_id": attempt_id,
                "config_sha256": cfg_sha, "phase_started_at_unix": phase_started,
                "inner_completed": step, "inner_total": cfg["updates"],
                "timestamp_unix": time.time(), "last_durable_step": max(x for x in eval_steps if x <= step)})
        if sleep_per_step:
            time.sleep(sleep_per_step)
        if interrupt_after_step is not None and step == interrupt_after_step:
            exclusive_atomic_json(unit_root / f"interruption_step_{step:04d}.json", {"step": step, "last_durable_step": max(x for x in eval_steps if x <= step), "kind": "controlled_durability_test"})
            return "INTERRUPTED_FOR_TEST"

    artifacts = []
    for subdir in ("metrics", "predictions", "checkpoints"):
        for path in sorted((unit_root / subdir).glob("*")):
            artifacts.append({"path": path.relative_to(unit_root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    exclusive_atomic_json(
        unit_root / "complete.json",
        {
            "schema_version": 1,
            "status": "COMPLETED",
            "unit_id": unit_id,
            "completed_updates": cfg["updates"],
            "config_sha256": cfg_sha,
            "input_sha256": sha256(data_path),
            "initialization_sha256": sha256(init_path),
            "producer_binding": binding,
            "artifacts": artifacts,
            "evaluation_steps": cfg["evaluation_steps"],
            "completed_at_unix": time.time(),
        },
    )
    return "COMPLETED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--unit-id")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--interrupt-after-step", type=int)
    parser.add_argument("--sleep-per-step", type=float, default=0.0)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    config_path = (run_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    output_root = (run_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    if not output_root.is_relative_to(run_root):
        raise ValueError("output root escapes run root")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg_sha = sha256(config_path)
    if cfg.get("run_kind") != "engineering_fixture":
        raise RuntimeError("Use campaign.py with explicit hash-bound admission; legacy partial-unit CLI is engineering-only")
    required_root = cfg.get("required_run_root_name")
    if required_root and run_root.name != required_root:
        raise ValueError(f"run root name {run_root.name!r} does not match required {required_root!r}")
    if float(cfg.get("tail_leak_mass", -1.0)) != TAIL_LEAK_MASS:
        raise ValueError("configured tail_leak_mass does not match frozen operator constant")
    if int(cfg.get("top_k", TOP_K)) != TOP_K:
        raise ValueError("configured top_k does not match operator constant")
    if float(cfg.get("top_score_range_ceiling", TOP_SCORE_RANGE_CEILING)) != TOP_SCORE_RANGE_CEILING:
        raise ValueError("configured top-score range ceiling does not match operator constant")
    if float(cfg.get("score_vjp_zero_tolerance", NORM_ZERO_TOLERANCE)) != NORM_ZERO_TOLERANCE:
        raise ValueError("configured VJP zero tolerance does not match operator constant")
    if int(cfg.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("N3 does not permit gradient accumulation")
    overlap = set(int(x) for x in cfg.get("seeds", [])) & set(int(x) for x in cfg.get("forbidden_seeds", []))
    if overlap:
        raise ValueError(f"forbidden seed reuse: {sorted(overlap)}")
    if (args.interrupt_after_step is not None or args.sleep_per_step) and "pilot" not in cfg.get("scope", "").lower():
        raise ValueError("interruption and artificial delay are pilot-only controls")
    if args.sleep_per_step < 0:
        raise ValueError("sleep-per-step must be non-negative")
    output_root.mkdir(parents=True, exist_ok=True)
    initial_bytes = tree_bytes(output_root)
    if shutil.disk_usage(output_root).free < cfg["minimum_free_disk_bytes"]:
        raise RuntimeError("free disk below configured guard")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    binding = producer_binding(run_root)
    with ControllerLock(output_root, cfg_sha):
        run_started = prepare_run_clock(output_root, cfg_sha, binding)
        global_deadline = run_started + float(cfg["maximum_wall_seconds"])
        launch_id = f"{int(time.time())}_{uuid.uuid4().hex}"
        status_path = output_root / "launches" / f"{launch_id}.json"
        started = time.time()
        results = []
        current_unit_id = None
        exclusive_atomic_json(
            status_path,
            {
                "schema_version": 1,
                "status": "RUNNING",
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "config_sha256": cfg_sha,
                "producer_binding": binding,
                "started_at_unix": started,
            },
        )
        units = [
            (architecture, int(r), int(seed), mode)
            for architecture in cfg["architectures"]
            for r in cfg["r"]
            for seed in cfg["seeds"]
            for mode in cfg["arms"]
        ]
        if args.unit_id:
            units = [u for u in units if f"{u[0]}_r{u[1]}_seed{u[2]}_{u[3]}" == args.unit_id]
            if len(units) != 1:
                raise ValueError("unit-id did not resolve uniquely")
        if args.max_units is not None:
            units = units[: args.max_units]
        try:
            for unit in units:
                current_unit_id = f"{unit[0]}_r{unit[1]}_seed{unit[2]}_{unit[3]}"
                try:
                    verdict = train_unit(
                        run_root,
                        output_root,
                        cfg,
                        cfg_sha,
                        binding,
                        global_deadline,
                        *unit,
                        args.interrupt_after_step,
                        args.sleep_per_step,
                    )
                except Exception as exc:
                    if invalid_control(exc):
                        raise InvalidControlError(str(exc)) from exc
                    raise
                results.append({"unit_id": current_unit_id, "verdict": verdict})
                if verdict == "INTERRUPTED_FOR_TEST":
                    atomic_json(
                        status_path,
                        {
                            "schema_version": 1,
                            "status": "INTERRUPTED_FOR_TEST",
                            "exit_code": 75,
                            "config_sha256": cfg_sha,
                            "producer_binding": binding,
                            "results": results,
                            "elapsed_seconds": time.time() - started,
                        },
                    )
                    return 75
            elapsed = time.time() - started
            new_bytes = tree_bytes(output_root) - initial_bytes
            peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            status = {
                "schema_version": 1,
                "status": "COMPLETED",
                "exit_code": 0,
                "config_sha256": cfg_sha,
                "producer_binding": binding,
                "results": results,
                "elapsed_seconds": elapsed,
                "global_elapsed_seconds": time.time() - run_started,
                "new_output_bytes": new_bytes,
                "peak_cuda_memory_allocated_bytes": peak_allocated,
                "peak_cuda_memory_reserved_bytes": peak_reserved,
                "free_disk_bytes": shutil.disk_usage(output_root).free,
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
            atomic_json(status_path, status)
            print(json.dumps(status, sort_keys=True))
            return 0
        except InvalidControlError as exc:
            if current_unit_id is not None:
                marker = output_root / "main" / current_unit_id / "invalid_control.json"
                if not marker.exists():
                    exclusive_atomic_json(
                        marker,
                        {
                            "schema_version": 1,
                            "status": "INVALID_CONTROL",
                            "unit_id": current_unit_id,
                            "config_sha256": cfg_sha,
                            "producer_binding": binding,
                            "error": str(exc),
                            "recorded_at_unix": time.time(),
                        },
                    )
            status = {
                "schema_version": 1,
                "status": "INVALID_CONTROL",
                "exit_code": 42,
                "unit_id": current_unit_id,
                "error": str(exc),
                "config_sha256": cfg_sha,
                "producer_binding": binding,
                "results": results,
                "elapsed_seconds": time.time() - started,
            }
            atomic_json(status_path, status)
            print(json.dumps(status, sort_keys=True), file=sys.stderr)
            return 42
        except Exception as exc:
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "status": "FAILED",
                    "exit_code": 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "config_sha256": cfg_sha,
                    "producer_binding": binding,
                    "results": results,
                    "elapsed_seconds": time.time() - started,
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
