"""Paired checkpoint continuations. No CLI or scientific launch admission yet.

Parent evidence is read-only. Relative fork step j uses parent batch t+j and
inherits model, AdamW moments and RNG from the same validated parent triplet.
"""
from __future__ import annotations

import functools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import training_runner as base
from training_models import build_model, stable_seed
from training_operators import InvalidControlError, backward_diagnostics, weights


def restore(payload: dict, model, optimizer) -> None:
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])


def square_norm(tensors) -> float:
    values = [x.detach().double().square().sum() for x in tensors]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def common_state_parameter_dose(model, x, v, y, r, key):
    """Hypothetical G/N parameter gradients at S's shared pre-update state."""
    vectors = {}
    try:
        for mode in ("G", "N"):
            logits = model.logits(x, v, r, mode, functools.partial(weights, null_seed=key))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(logits.dtype))
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            vectors[mode] = torch.cat([g.detach().double().reshape(-1) for g in grads])
        g, n = vectors["G"], vectors["N"]
        if not torch.isfinite(g).all() or not torch.isfinite(n).all():
            return {"status": "UNDEFINED_NONFINITE"}
        gn, nn = g.norm(), n.norm()
        return {"status": "OBSERVED", "G_norm": float(gn), "N_norm": float(nn),
                "G_N_cosine": float(torch.dot(g, n)/(gn*nn)) if gn*nn > 0 else None}
    except InvalidControlError as exc:
        return {"status": "UNDEFINED_HYPOTHETICAL", "reason": str(exc)}


@base.persist_invalid_control
def run_fork(run_root: Path, main_output: Path, fork_output: Path, cfg: dict,
             cfg_sha: str, binding: dict, architecture: str, r: int, seed: int,
             parent_step: int, mode: str, deadline: float,
             interrupt_after_step: int | None = None) -> str:
    """Write one immutable fork unit, or resume its validated last triplet."""
    if parent_step not in cfg["fork_checkpoints"] or mode not in cfg["fork_arms"]:
        raise ValueError("fork absent from prospective plan")
    count = int(cfg["fork_updates"])
    eval_steps = set(cfg["fork_evaluation_steps"])
    if not {0, count}.issubset(eval_steps) or min(eval_steps) < 0 or max(eval_steps) > count:
        raise ValueError("fork endpoints are incomplete or out of range")
    if parent_step + count > int(cfg["updates"]):
        raise ValueError("fork extends beyond stored batch schedule")
    if interrupt_after_step is not None and cfg.get("run_kind") != "engineering_fixture":
        raise ValueError("failure injection requires engineering_fixture")
    base.require_config_binding(run_root, cfg, cfg_sha)
    parent_id = f"{architecture}_r{r}_seed{seed}_S"
    parent_root = main_output / "main" / parent_id
    base.verify_published_evaluation(parent_root, parent_step, cfg_sha, binding)
    parent_path = base.published_evaluation_paths(parent_root, parent_step)["checkpoint"]
    parent_digest = base.sha256(parent_path)
    unit_id = f"{parent_id}_at{parent_step}_{mode}"
    root = fork_output / "forks" / unit_id
    root.mkdir(parents=True, exist_ok=True)
    fork_binding = {**binding, "fork_parent": {
        "unit_id": parent_id, "step": parent_step, "sha256": parent_digest,
        "mode": mode, "batch_rule": "parent_step_plus_relative_step",
        "fork_runner_sha256": base.sha256(Path(__file__)),
    }}
    # Use the canonical input helper to verify data and schedule hashes.
    data_path = base.prepare_inputs(run_root, cfg, cfg_sha, binding, r, seed)
    arrays = base.load_arrays(data_path)
    schedule = np.load(run_root / "inputs" / f"batches_r{r}_seed{seed}.npy", allow_pickle=False)
    expected = {"unit_id": unit_id, "completed_updates": count,
                "config_sha256": cfg_sha, "producer_binding": fork_binding,
                "evaluation_steps": cfg["fork_evaluation_steps"],
                "input_sha256": base.sha256(data_path), "parent_sha256": parent_digest}
    if (root / "invalid_control.json").exists():
        raise InvalidControlError("fork previously marked invalid")
    if base.verify_complete(root, expected):
        return "SKIPPED_VERIFIED"
    identity = root / "identity.json"
    if identity.exists():
        if json.loads(identity.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("fork identity changed")
    else:
        base.exclusive_atomic_json(identity, expected)
    device = torch.device(cfg["device"])
    model = build_model(architecture, stable_seed(seed, r, architecture, "initialization")).to(device)
    optimizer = base.optimizer_for(model, cfg)
    base.recover_evaluation_transactions(root, cfg_sha, fork_binding)
    step, resume_path = base.latest_checkpoint(root, cfg_sha, fork_binding)
    payload = torch.load(resume_path or parent_path, map_location="cpu", weights_only=False)
    parent_state = torch.load(parent_path, map_location="cpu", weights_only=False)["model_state"]
    restore(payload, model, optimizer)
    model.train()
    if resume_path is None:
        metrics, predictions = base.evaluate(model, arrays, r, mode, device)
        base.save_evaluation(root, 0, metrics, predictions, model, optimizer, cfg_sha, fork_binding)
    started = time.monotonic()
    attempt_id = base.uuid.uuid4().hex
    for relative in range(step + 1, count + 1):
        base.memory_guard(cfg)
        if time.time() > deadline:
            raise TimeoutError("fork deadline")
        if base.guarded_tree_bytes(run_root, fork_output, cfg) > cfg["max_disk_bytes"]:
            raise RuntimeError("fork disk ceiling")
        if base.shutil.disk_usage(fork_output).free < cfg["minimum_free_disk_bytes"]:
            raise RuntimeError("fork free-disk guard")
        global_step = parent_step + relative
        idx = schedule[global_step - 1]
        x = torch.from_numpy(arrays["train_features"][idx]).to(device)
        v = torch.from_numpy(arrays["train_values"][idx]).to(device)
        y = torch.from_numpy(arrays["train_labels"][idx]).to(device)
        key = stable_seed(seed, architecture, r, parent_step, relative, "null_permutation")
        operator = functools.partial(weights, null_seed=key)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.zero_grad(set_to_none=True)
        scores = model.scores(x)
        w = operator(scores, mode)
        gamma = F.softplus(model.raw_gamma).to(w.dtype)
        logits = gamma * np.sqrt(r) * (w * v.to(w.dtype)).sum(-1) + model.bias.to(w.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, y.to(logits.dtype))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite fork training loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("nonfinite fork parameter gradient")
        gradient_norm = square_norm(p.grad for p in model.parameters() if p.grad is not None)
        common_dose = common_state_parameter_dose(model, x, v, y, r, key) if mode == "S" else None
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("nonfinite fork parameter update")
        update_norm = square_norm(p.detach() - old for p, old in zip(model.parameters(), before))
        if torch.cuda.is_available() and torch.cuda.max_memory_reserved() > cfg["maximum_cuda_memory_bytes"]:
            raise RuntimeError("fork CUDA memory ceiling")
        diagnostics = {"relative_step": relative, "parent_global_step": global_step,
                       "null_seed": key, "parameter_gradient_norm": gradient_norm,
                       "parameter_update_norm": update_norm, "parent_sha256": parent_digest}
        diagnostics["displacement_from_parent_norm"] = square_norm(
            p.detach() - parent_state[name].to(p.device) for name, p in model.named_parameters())
        if common_dose is not None:
            diagnostics["common_state_hypothetical_parameter_dose"] = common_dose
        # All arms retain actual update norms. Node dose is comparable at the
        # common hard-forward state, not asserted equal in parameter space.
        if mode in {"S", "G", "N"}:
            upstream = ((torch.sigmoid(logits.detach().double()) - y.double())[:, None]
                        * gamma.detach().double() * np.sqrt(r) * v.double() / y.numel())
            diagnostics.update(base.describe_dose(scores.detach(), upstream, null_seed=key))
        # Every assigned update is retained, including steps between endpoints.
        dynamics_path = root / "dynamics" / f"step_{relative:04d}.json"
        if dynamics_path.exists():
            if json.loads(dynamics_path.read_text(encoding="utf-8")) != diagnostics:
                raise RuntimeError("WORKFLOW_NONDETERMINISTIC: replayed update diagnostics changed")
        else:
            base.exclusive_atomic_json(dynamics_path, diagnostics)
        if relative in eval_steps:
            metrics, predictions = base.evaluate(model, arrays, r, mode, device)
            metrics["last_update_diagnostics"] = diagnostics
            base.save_evaluation(root, relative, metrics, predictions, model, optimizer, cfg_sha, fork_binding)
        base.atomic_json(fork_output / "heartbeat.json", {
            "run_id": cfg["run_id"], "unit_id": unit_id, "pid": base.os.getpid(),
            "attempt_id": attempt_id, "config_sha256": cfg_sha,
            "phase_started_at_unix": time.time() - (time.monotonic() - started),
            "phase": "fork", "inner_completed": relative, "inner_total": count,
            "unit_elapsed_seconds": time.monotonic() - started,
            "timestamp_unix": time.time(), "last_durable_step": max(t for t in eval_steps if t <= relative),
        })
        if relative == interrupt_after_step:
            return "INTERRUPTED_FOR_TEST"
    artifacts = [{"path": p.relative_to(root).as_posix(), "sha256": base.sha256(p), "bytes": p.stat().st_size}
                 for folder in ("metrics", "predictions", "checkpoints", "dynamics") for p in sorted((root / folder).glob("*"))]
    if base.sha256(parent_path) != parent_digest:
        raise RuntimeError("parent changed during fork")
    base.exclusive_atomic_json(root / "complete.json", {
        **expected, "status": "COMPLETED", "artifacts": artifacts,
        "elapsed_seconds": time.monotonic() - started, "completed_at_unix": time.time(),
    })
    return "COMPLETED"
