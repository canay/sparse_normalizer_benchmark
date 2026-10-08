"""One-controller paired assay. No outcome display; explicit admission required."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import torch

import training_runner as base
from fork_runner import run_fork
from training_operators import InvalidControlError


def plan(cfg):
    units = []
    for seed in cfg["seeds"]:
        for architecture in cfg["architectures"]:
            for r in cfg["r"]:
                for arm in ("D", "S"):
                    units.append({"kind": "main", "architecture": architecture,
                        "r": r, "seed": seed, "arm": arm,
                        "id": f"{architecture}_r{r}_seed{seed}_{arm}"})
                for parent in cfg["fork_checkpoints"]:
                    for arm in cfg["fork_arms"]:
                        units.append({"kind": "fork", "architecture": architecture,
                            "r": r, "seed": seed, "arm": arm, "parent": parent,
                            "id": f"{architecture}_r{r}_seed{seed}_S_at{parent}_{arm}"})
    if len({u["id"] for u in units}) != len(units):
        raise ValueError("duplicate planned unit")
    return units


def source_census(root):
    return {p.relative_to(root).as_posix(): base.sha256(p)
            for folder in ("src", "tests") for p in sorted((root / folder).glob("*.py"))}


def preflight(root, cfg, cfg_sha):
    base.require_config_binding(root, cfg, cfg_sha)
    if cfg.get("run_kind") not in {"engineering_fixture", "resource_pilot", "scientific"}:
        raise ValueError("unknown run kind")
    if cfg["arms"] != ["D", "S"] or cfg["fork_arms"] != ["S", "L", "P", "G", "N"]:
        raise ValueError("unregistered main/fork arm census")
    if cfg["n"] != 32 or cfg["top_k"] != 8 or cfg["tail_leak_mass"] != 1/9:
        raise ValueError("N3-A operator constants mismatch")
    if cfg["gradient_accumulation_steps"] != 1:
        raise ValueError("gradient accumulation prohibited")
    if set(cfg["seeds"]) & set(cfg["forbidden_seeds"]):
        raise ValueError("forbidden seed reuse")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("duplicate seeds")
    for parent in cfg["fork_checkpoints"]:
        if not {parent, parent + cfg["fork_updates"]}.issubset(cfg["evaluation_steps"]):
            raise ValueError("S replay endpoint absent from main plan")
    if cfg["fork_evaluation_steps"] != [0, 1, cfg["fork_updates"]]:
        raise ValueError("invalid fork endpoints")
    required = {"inputs", "outputs"}
    if set(cfg["disk_guard_paths"]) != required:
        raise ValueError("disk census must cover all inputs and outputs")
    if cfg["run_kind"] != "engineering_fixture":
        receipt = json.loads((root / "ADMISSION.json").read_text(encoding="utf-8"))
        if receipt.get("status") != "ADMITTED" or receipt.get("run_kind") != cfg["run_kind"]:
            raise RuntimeError("no matching admission")
        if receipt["config_sha256"] != cfg_sha or receipt["source_sha256"] != source_census(root):
            raise RuntimeError("admission source/config mismatch")
        for relative, digest in receipt["document_sha256"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or base.sha256(path) != digest:
                raise RuntimeError("admission document mismatch")
        if not receipt.get("independent_review") or not receipt.get("prelaunch_evidence"):
            raise RuntimeError("missing independent review/prelaunch evidence")
    return plan(cfg)


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def verify_unit(root, cfg, cfg_sha, binding, unit):
    unit_root = root / "outputs" / ("main" if unit["kind"] == "main" else "forks") / unit["id"]
    marker = unit_root / "complete.json"
    if not marker.exists():
        raise RuntimeError("planned unit incomplete: " + unit["id"])
    data = json.loads(marker.read_text(encoding="utf-8"))
    expected_binding = binding
    if unit["kind"] == "fork":
        parent_id = f"{unit['architecture']}_r{unit['r']}_seed{unit['seed']}_S"
        parent = root / "outputs/main" / parent_id
        parent_path = base.published_evaluation_paths(parent, unit["parent"])["checkpoint"]
        expected_binding = {**binding, "fork_parent": {"unit_id": parent_id,
            "step": unit["parent"], "sha256": base.sha256(parent_path), "mode": unit["arm"],
            "batch_rule": "parent_step_plus_relative_step",
            "fork_runner_sha256": base.sha256(root / "src/fork_runner.py")}}
    expected = {"unit_id": unit["id"], "config_sha256": cfg_sha,
        "producer_binding": expected_binding,
        "completed_updates": cfg["updates"] if unit["kind"] == "main" else cfg["fork_updates"],
        "evaluation_steps": cfg["evaluation_steps"] if unit["kind"] == "main" else cfg["fork_evaluation_steps"]}
    if not base.verify_complete(unit_root, expected):
        raise RuntimeError("unit completion failure")
    if unit["kind"] == "fork":
        actual_dynamics = {p.stem for p in (unit_root / "dynamics").glob("*.json")}
        if actual_dynamics != {f"step_{step:04d}" for step in range(1, cfg["fork_updates"] + 1)}:
            raise RuntimeError("planned dynamics census mismatch")
        pairs = [(unit["parent"], 0)]
        if unit["arm"] == "S":
            pairs.append((unit["parent"] + cfg["fork_updates"], cfg["fork_updates"]))
        for parent_step, fork_step in pairs:
            left = torch.load(base.published_evaluation_paths(parent, parent_step)["checkpoint"], map_location="cpu", weights_only=False)
            right = torch.load(base.published_evaluation_paths(unit_root, fork_step)["checkpoint"], map_location="cpu", weights_only=False)
            for key in ("model_state", "optimizer_state", "torch_rng_state", "cuda_rng_state"):
                if not state_equal(left[key], right[key]):
                    raise RuntimeError("WORKFLOW_NONDETERMINISTIC: " + unit["id"] + ":" + key)
    if any((unit_root / "transactions").iterdir()):
        raise RuntimeError("unresolved transaction residue")
    return {"unit_id": unit["id"], "complete_sha256": base.sha256(marker)}


def validate(root, cfg, cfg_sha, binding, units):
    base.require_run_open(root)
    if list((root / "outputs").rglob("invalid_control.json")):
        raise InvalidControlError("invalid unit marker in campaign")
    for kind, directory in (("main", "main"), ("fork", "forks")):
        expected = {u["id"] for u in units if u["kind"] == kind}
        actual = {p.name for p in (root / "outputs" / directory).iterdir() if p.is_dir()}
        if actual != expected:
            raise RuntimeError("run-level planned unit census mismatch")
    receipts = [verify_unit(root, cfg, cfg_sha, binding, u) for u in units]
    return {"status": "PASS", "units": len(receipts), "config_sha256": cfg_sha,
        "producer_binding": binding, "unit_receipts": receipts, "verified_at_unix": time.time()}


def execute(root, cfg, cfg_sha, binding, units, argv):
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    base.require_run_open(root)
    with base.ControllerLock(output, cfg_sha):
        started = time.time()
        attempt = base.uuid.uuid4().hex
        launch = output / "launches" / (attempt + ".json")
        status = {"run_id": cfg["run_id"], "attempt_id": attempt, "argv": argv,
            "pid": os.getpid(), "host": socket.gethostname(), "config_sha256": cfg_sha,
            "producer_binding": binding, "phase_started_at_unix": started,
            "status": "RUNNING", "completed_units": 0, "planned_units": len(units)}
        base.exclusive_atomic_json(launch, status)
        def interrupted(signum, frame):
            raise InterruptedError("signal " + str(signum))
        old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            for index, unit in enumerate(units):
                base.require_run_open(root)
                if time.time() - started > cfg["maximum_launch_wall_seconds"]:
                    raise TimeoutError("launch wall ceiling")
                status.update(unit_id=unit["id"], completed_units=index, timestamp_unix=time.time())
                base.atomic_json(output / "campaign_heartbeat.json", status)
                deadline = time.time() + cfg["maximum_unit_wall_seconds"]
                args = (unit["architecture"], unit["r"], unit["seed"])
                if unit["kind"] == "main":
                    base.train_unit(root, output, cfg, cfg_sha, binding, deadline, *args, unit["arm"], None, 0)
                else:
                    run_fork(root, output, output, cfg, cfg_sha, binding, *args,
                        unit["parent"], unit["arm"], deadline)
                verify_unit(root, cfg, cfg_sha, binding, unit)
                status.update(completed_units=index+1, timestamp_unix=time.time())
                base.atomic_json(output / "campaign_heartbeat.json", status)
            validation = validate(root, cfg, cfg_sha, binding, units)
            base.atomic_json(output / "validation.json", validation)
            status.update(status="COMPLETED_VERIFIED", exit_code=0,
                validation_sha256=base.sha256(output / "validation.json"))
        except BaseException as exc:
            status.update(status="INVALID_CONTROL" if isinstance(exc, InvalidControlError)
                else "INTERRUPTED_UNKNOWN" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "WORKFLOW_FAILED",
                exit_code=42 if isinstance(exc, InvalidControlError) else 75 if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 1,
                error_type=type(exc).__name__, error=str(exc))
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            status.update(finished_at_unix=time.time(), elapsed_seconds=time.time()-started,
                bytes=base.tree_bytes(root), free_disk_bytes=base.shutil.disk_usage(root).free)
            if sys.platform == "linux":
                import resource
                status["peak_resident_memory_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            base.atomic_json(launch, status)
            base.atomic_json(output / "status.json", status)
        # Never expose scientific outcomes in operational stdout.
        print(json.dumps({k: status[k] for k in ("status", "exit_code", "run_id", "completed_units", "planned_units", "elapsed_seconds", "bytes", "free_disk_bytes")}))
        return status["exit_code"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    cfg_path = root / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_sha = base.sha256(cfg_path)
    units = preflight(root, cfg, cfg_sha)
    base.configure_determinism(cfg["torch_threads"])
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    binding = base.producer_binding(root)
    if args.validate_only:
        result = validate(root, cfg, cfg_sha, binding, units)
        base.atomic_json(root / "outputs/validation_recheck.json", result)
        print(json.dumps({"status": result["status"], "units": result["units"]}))
        return 0
    return execute(root, cfg, cfg_sha, binding, units, sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
