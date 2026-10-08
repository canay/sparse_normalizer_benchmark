#!/usr/bin/env python3
"""Prospectively registered crossover/mechanism experiment runner.

This file is additive. It imports the published r1 implementation and does not
modify benchmark_r1.py or datasets_r1.py. New behavior is limited to the
MCH-SND-002 registration: a second architecture, three capacity tiers,
mechanism probes, and one-time top-k-to-softmax schedules.

Dependencies: torch, numpy, and scipy only when SVHN is requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RUN_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = RUN_DIR.parent
_base_override = os.environ.get("SND_R1_BASE_SRC")
BASE_SRC = Path(_base_override) if _base_override else \
    EXPERIMENTS_DIR / "2026-08-17_claude_mta_r1-validation-split" / "src"
if not BASE_SRC.exists():
    raise RuntimeError(
        f"published r1 source directory not found: {BASE_SRC}; set SND_R1_BASE_SRC on a remote host"
    )
sys.path.insert(0, str(BASE_SRC))

import benchmark_r1 as r1  # noqa: E402
import datasets_e5b  # noqa: E402


CODE_VERSION = "mch-snd-002-v1"
PRIMARY_RATIO = 0.25
PROBE_SIZE = 128
CAPACITY_TIERS = {
    "compact": (2, 64, 4),
    "deep": (4, 64, 4),
    "wide": (2, 128, 4),
}
BASE_IMAGE_DATASETS = set(r1.IMAGE_DATASETS)
E5B_IMAGE_DATASETS = set(datasets_e5b.E5B_DATASETS)
ALL_IMAGE_DATASETS = BASE_IMAGE_DATASETS | E5B_IMAGE_DATASETS
DEFAULT_PATCH = {name: 4 for name in ALL_IMAGE_DATASETS}
DEFAULT_PATCH["usps"] = 2
DISCOVERY_DATASETS = (
    "mnist", "fashion_mnist", "kmnist", "cifar10", "cifar100",
    "twenty_news", "synthetic_marker", "emnist_letters", "emnist_digits", "usps",
)
CONFIRMATION_DATASETS = ("emnist_balanced", "k49", "svhn")
M2_METHODS = (
    "softmax", "topk_softmax_025", "median_switch", "early_accuracy_switch",
    "entropy_switch", "mechanistic_switch", "headwise_adaptive_entmax", "cosine_alpha",
)
SIMPLE_SWITCH_METHODS = {"median_switch", "early_accuracy_switch", "entropy_switch"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_registered_invocation(args) -> dict | None:
    """Reject phase drift before any registered dataset or official test is read."""
    if args.tag.startswith(("smoke", "pilot")):
        return None
    if args.max_epochs != 30 or args.batch_size != 256:
        raise ValueError("registered full runs require 30 epochs and batch size 256")

    datasets = set(args.datasets)
    methods = set(args.methods)
    seeds = set(args.seeds)
    if args.experiment_id == "EXP-SND-101":
        if datasets != set(DISCOVERY_DATASETS) or methods != {"softmax", "topk_softmax_025"} \
                or seeds != set(range(5)):
            raise ValueError("EXP-SND-101 invocation does not match the registered M1 grid")
    elif args.experiment_id == "EXP-SND-102":
        if datasets != set(DISCOVERY_DATASETS) or methods != set(M2_METHODS) \
                or seeds != set(range(5, 10)):
            raise ValueError("EXP-SND-102 invocation does not match the registered M2 grid")
        if not args.switch_map or not Path(args.switch_map).is_file():
            raise ValueError("EXP-SND-102 requires the frozen M1 switch map")
    elif args.experiment_id == "EXP-SND-103":
        if datasets != set(CONFIRMATION_DATASETS) or seeds != set(range(10, 20)):
            raise ValueError("EXP-SND-103 invocation does not match the registered confirmation grid")
        if args.final_test_read:
            if not args.switch_map or not Path(args.switch_map).is_file():
                raise ValueError("official test read requires the frozen M3 switch map")
            freeze_path = Path(args.test_freeze) if args.test_freeze else None
            if freeze_path is None or not freeze_path.is_file():
                raise ValueError("official test read requires the frozen M3 validation certificate")
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if freeze.get("status") != "M3_VALIDATION_FROZEN_OFFICIAL_TEST_AUTHORIZED":
                raise ValueError("official test freeze status is not authorized")
            if freeze.get("switch_map_sha256") != sha256_file(Path(args.switch_map)):
                raise ValueError("official test switch-map hash does not match the validation freeze")
            frozen_methods = freeze.get("methods")
            if not isinstance(frozen_methods, list) or methods != set(frozen_methods):
                raise ValueError("official test methods do not match the validation freeze")
            simple = freeze.get("strongest_simple_schedule")
            if simple not in SIMPLE_SWITCH_METHODS or methods != {
                "softmax", "topk_softmax_025", "mechanistic_switch", simple,
            }:
                raise ValueError("official test method family is invalid")
            return freeze
        if args.test_freeze:
            raise ValueError("--test-freeze is only valid with --final-test-read")
        baseline = {"softmax", "topk_softmax_025"}
        schedule_simple = methods - {"mechanistic_switch"}
        schedule_lane = (
            len(methods) == 2
            and "mechanistic_switch" in methods
            and len(schedule_simple) == 1
            and next(iter(schedule_simple)) in SIMPLE_SWITCH_METHODS
        )
        if methods != baseline and not schedule_lane:
            raise ValueError("EXP-SND-103 validation must be a registered baseline or schedule lane")
        if schedule_lane and (not args.switch_map or not Path(args.switch_map).is_file()):
            raise ValueError("EXP-SND-103 schedule lane requires the frozen M3 switch map")
    return None


def load_bundle(name: str, seed: int, args) -> r1.Bundle:
    """Load every registered dataset under the unchanged r1 split contract."""
    if name in BASE_IMAGE_DATASETS or name in {"twenty_news", "synthetic_marker"}:
        return r1.load_bundle(name, seed, args)
    if name not in E5B_IMAGE_DATASETS:
        raise ValueError(f"unregistered dataset: {name}")

    raw = datasets_e5b.load_raw_e5b(name, Path(args.data_root))
    patch = args.patch if args.patch else DEFAULT_PATCH[name]
    x_all = torch.from_numpy(np.ascontiguousarray(raw["train_x"]))
    y_all = torch.from_numpy(raw["train_y"])
    keep = r1._limit(len(x_all), args.train_limit, seed)
    x_all, y_all = x_all[keep], y_all[keep]
    fit, val = r1.split_train_val(len(x_all), args.val_fraction, seed)
    x_tr = r1.patchify(x_all[fit], patch)
    x_va = r1.patchify(x_all[val], patch)
    return r1.Bundle(
        train=TensorDataset(x_tr, y_all[fit]),
        val=TensorDataset(x_va, y_all[val]),
        input_kind="patch",
        input_dim=int(x_tr.shape[-1]),
        classes=int(raw["classes"]),
        seq_len=int(x_tr.shape[1]),
        source=f"{name}:patch{patch}",
        sizes={"train": len(x_tr), "val": len(x_va)},
    )


def load_official_test(name: str, args) -> tuple[TensorDataset, int]:
    """Load the official test split only for the frozen EXP-SND-103 read."""
    if name in BASE_IMAGE_DATASETS:
        raw = r1.datasets_r1.load_raw(name, Path(args.data_root))
    elif name in E5B_IMAGE_DATASETS:
        raw = datasets_e5b.load_raw_e5b(name, Path(args.data_root))
    else:
        raise ValueError(f"official-test helper is intentionally image-only: {name}")
    patch = args.patch if args.patch else DEFAULT_PATCH[name]
    x = torch.from_numpy(np.ascontiguousarray(raw["test_x"]))
    y = torch.from_numpy(raw["test_y"])
    return TensorDataset(r1.patchify(x, patch), y), int(raw["classes"])


def masked_scores(scores: torch.Tensor, key_valid: torch.Tensor | None) -> torch.Tensor:
    if key_valid is None:
        return scores
    row_min = scores.masked_fill(~key_valid, float("inf")).amin(dim=-1, keepdim=True)
    return torch.where(key_valid, scores, row_min - r1.MASK_MARGIN)


class InstrumentedAttention(r1.NormalizedSelfAttention):
    """r1 attention plus a no-update dense counterfactual mechanism probe."""

    def __init__(self, embed_dim: int, heads: int, method: str):
        init_method = "headwise_adaptive_entmax" if method == "headwise_adaptive_entmax" else method
        if init_method in {"cosine_alpha", "scheduled"}:
            init_method = "softmax"
        super().__init__(embed_dim, heads, init_method)
        self.probe_mode = False
        self.probe_scores: torch.Tensor | None = None
        self.probe_key_valid: torch.Tensor | None = None
        self.probe_query_valid: torch.Tensor | None = None
        self.cosine_alpha: float | None = None

    def normalize(self, scores: torch.Tensor, key_valid: torch.Tensor | None) -> torch.Tensor:
        if self.cosine_alpha is None:
            return super().normalize(scores, key_valid)
        z = masked_scores(scores, key_valid)
        attn = r1.entmax_bisect(z, alpha=float(self.cosine_alpha), dim=-1)
        if self.collect_stats:
            with torch.no_grad():
                if key_valid is None:
                    denom = torch.ones_like(attn, dtype=torch.bool)
                else:
                    denom = key_valid.expand_as(attn)
                self.last["nonzero_ratio"] = float(
                    ((attn > 1e-7) & denom).sum().cpu() / denom.sum().clamp(min=1).cpu()
                )
                p = attn * denom
                q = p.clamp(min=1e-12)
                self.last["entropy"] = float((-(q * q.log()).sum(-1)).mean().cpu())
                self.last["alpha_mean"] = float(self.cosine_alpha)
                self.last["alpha_std"] = 0.0
        return attn

    def forward(self, x, key_valid=None, query_valid=None):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.probe_mode:
            z = masked_scores(scores, key_valid)
            z.retain_grad()
            self.probe_scores = z
            self.probe_key_valid = key_valid
            self.probe_query_valid = query_valid
            attn = torch.softmax(z, dim=-1)
        else:
            attn = self.normalize(scores, key_valid)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        return self.out(out)

    def read_probe(self) -> dict[str, float]:
        if self.probe_scores is None or self.probe_scores.grad is None:
            raise RuntimeError("probe scores/gradients unavailable")
        s = self.probe_scores.detach()
        g = self.probe_scores.grad.detach().abs()
        b, h, q, kdim = s.shape

        if self.probe_key_valid is None:
            key = torch.ones((b, 1, 1, kdim), dtype=torch.bool, device=s.device)
        else:
            key = self.probe_key_valid
        if self.probe_query_valid is None:
            qry = torch.ones((b, 1, q, 1), dtype=torch.bool, device=s.device)
        else:
            qry = self.probe_query_valid
        key_full = key.expand(b, h, q, kdim)
        row_valid = qry.expand(b, h, q, 1).squeeze(-1)

        n_valid = key.sum(dim=-1, keepdim=True).to(s.dtype)
        k_row = torch.ceil(n_valid * PRIMARY_RATIO).clamp(min=1).to(torch.long)
        k_max = int(k_row.max().item())
        vals, idx = s.topk(min(k_max + 1, kdim), dim=-1)
        rank = torch.arange(k_max, device=s.device).view(1, 1, 1, -1)
        keep_rank = rank < k_row
        support = torch.zeros_like(key_full)
        support.scatter_(-1, idx[..., :k_max], keep_rank.expand_as(idx[..., :k_max]))
        support &= key_full

        dense = torch.softmax(s, dim=-1) * key_full
        tail = (dense * ~support).sum(dim=-1)
        entropy = -(dense.clamp(min=1e-12) * dense.clamp(min=1e-12).log()).sum(dim=-1)
        norm = torch.log(n_valid.squeeze(-1).expand(b, h, q).clamp(min=2.0))
        entropy = entropy / norm

        key_float = key_full.to(s.dtype)
        mean = (s * key_float).sum(-1) / key_float.sum(-1).clamp(min=1)
        var = (((s - mean.unsqueeze(-1)) ** 2) * key_float).sum(-1) / key_float.sum(-1).clamp(min=1)
        std = var.sqrt().clamp(min=1e-8)
        kth_index = (k_row - 1).expand(b, h, q, 1)
        next_index = k_row.expand(b, h, q, 1).clamp(max=vals.size(-1) - 1)
        kth = vals.gather(-1, kth_index).squeeze(-1)
        nxt = vals.gather(-1, next_index).squeeze(-1)
        has_next = (k_row.squeeze(-1).expand(b, h, q) < n_valid.squeeze(-1).expand(b, h, q))
        margin = (kth - nxt) / std

        grad_total = (g * key_full).sum(-1)
        grad_top = (g * support).sum(-1)
        grad_share = grad_top / grad_total.clamp(min=1e-12)

        def avg(x: torch.Tensor, valid: torch.Tensor = row_valid) -> float:
            z = x[valid & torch.isfinite(x)]
            return float(z.mean().cpu()) if z.numel() else float("nan")

        result = {
            "probe_dense_entropy": avg(entropy),
            "probe_boundary_margin": avg(margin, row_valid & has_next),
            "probe_tail_mass": avg(tail),
            "probe_gradient_topk_share": avg(grad_share, row_valid & (grad_total > 0)),
        }
        self.probe_scores = None
        self.probe_key_valid = None
        self.probe_query_valid = None
        return result


class LocalMixer(nn.Module):
    def __init__(self, embed_dim: int, input_kind: str, seq_len: int):
        super().__init__()
        self.input_kind = input_kind
        self.seq_len = seq_len
        self.norm = nn.LayerNorm(embed_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))
        if input_kind == "patch":
            side = int(round(math.sqrt(seq_len)))
            if side * side != seq_len:
                raise ValueError(f"image patch sequence is not square: {seq_len}")
            self.grid = (side, side)
            self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
            self.conv1 = None
        else:
            self.grid = None
            self.conv1 = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
            self.conv2 = None

    def forward(self, x: torch.Tensor, content_mask: torch.Tensor | None) -> torch.Tensor:
        cls, content = x[:, :1], x[:, 1:]
        z = self.norm(content)
        if content_mask is not None:
            z = z * content_mask.unsqueeze(-1).to(z.dtype)
        if self.input_kind == "patch":
            gh, gw = self.grid
            z = z.transpose(1, 2).reshape(z.size(0), z.size(2), gh, gw)
            z = self.conv2(z).flatten(2).transpose(1, 2)
        else:
            z = self.conv1(z.transpose(1, 2)).transpose(1, 2)
        z = torch.nn.functional.gelu(z)
        content = content + self.scale * z
        if content_mask is not None:
            content = content * content_mask.unsqueeze(-1).to(content.dtype)
        return torch.cat([cls, content], dim=1)


class InstrumentedBlock(nn.Module):
    def __init__(self, embed_dim, heads, method, dropout, architecture, input_kind, seq_len):
        super().__init__()
        self.local = LocalMixer(embed_dim, input_kind, seq_len) if architecture == "local_hybrid" else None
        self.attn = InstrumentedAttention(embed_dim, heads, method)
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_valid=None, query_valid=None, content_mask=None):
        if self.local is not None:
            x = self.local(x, content_mask)
        x = self.norm1(x + self.drop(self.attn(x, key_valid, query_valid)))
        return self.norm2(x + self.drop(self.ff(x)))


class InstrumentedClassifier(nn.Module):
    def __init__(self, bundle, method_spec, architecture, embed_dim, heads, layers, dropout, max_epochs, switch_epoch):
        super().__init__()
        self.method_spec = method_spec
        self.max_epochs = max_epochs
        self.switch_epoch = switch_epoch
        init_method = self._active_method(1)
        self.input = nn.Linear(bundle.input_dim, embed_dim) if bundle.input_kind == "patch" \
            else nn.Embedding(bundle.input_dim, embed_dim, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, bundle.seq_len + 1, embed_dim))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([
            InstrumentedBlock(embed_dim, heads, init_method, dropout, architecture,
                              bundle.input_kind, bundle.seq_len)
            for _ in range(layers)
        ])
        self.head = nn.Linear(embed_dim, bundle.classes)
        self.active_method = init_method

    def _active_method(self, epoch: int) -> str:
        if self.method_spec.startswith("switch_") or self.method_spec.endswith("_switch"):
            if self.switch_epoch is None:
                raise ValueError(f"switch epoch unavailable for {self.method_spec}")
            return "topk_softmax_025" if epoch < self.switch_epoch else "softmax"
        return self.method_spec

    def set_epoch(self, epoch: int):
        active = self._active_method(epoch)
        self.active_method = active
        for block in self.blocks:
            block.attn.cosine_alpha = None
            if active == "cosine_alpha":
                progress = (epoch - 1) / max(self.max_epochs - 1, 1)
                block.attn.cosine_alpha = 1.0 + 0.25 * (1.0 - math.cos(math.pi * progress))
                block.attn.method = "softmax"
            else:
                block.attn.method = active

    def set_self_check(self, on: bool):
        for b in self.blocks:
            b.attn.self_check = on

    def set_collect_stats(self, on: bool):
        for b in self.blocks:
            b.attn.collect_stats = on

    def set_probe(self, on: bool):
        for b in self.blocks:
            b.attn.probe_mode = on

    def forward(self, x, content_mask=None):
        x = self.input(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos[:, :x.size(1)]
        key_valid = query_valid = None
        if content_mask is not None:
            ones = torch.ones(content_mask.size(0), 1, dtype=torch.bool, device=content_mask.device)
            full = torch.cat([ones, content_mask], dim=1)
            key_valid = full.view(full.size(0), 1, 1, -1)
            query_valid = full.view(full.size(0), 1, -1, 1)
        for blk in self.blocks:
            x = blk(x, key_valid, query_valid, content_mask)
        return self.head(x[:, 0])

    def stats(self) -> dict:
        def agg(key):
            a = np.asarray([b.attn.last[key] for b in self.blocks], dtype=float)
            return float(np.nanmean(a)) if not np.isnan(a).all() else float("nan")
        return {k: agg(k) for k in ("nonzero_ratio", "entropy", "alpha_mean", "alpha_std")}

    def read_probe(self) -> dict[str, float]:
        rows = [b.attn.read_probe() for b in self.blocks]
        return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def make_probe_loader(bundle: r1.Bundle, batch_size: int, device: torch.device):
    x, y = bundle.val.tensors
    n = min(PROBE_SIZE, len(x))
    ds = TensorDataset(x[:n], y[:n])
    mask = bundle.val_mask[:n] if bundle.val_mask is not None else None
    return r1.make_loader(ds, mask, min(batch_size, n), False, device), n


def mechanism_probe(model, loader, device, n_classes) -> dict[str, float]:
    del n_classes
    model.eval()
    model.zero_grad(set_to_none=True)
    model.set_probe(True)
    batch = next(iter(loader))
    xb, yb, mb = r1.unpack(batch, device)
    loss = nn.CrossEntropyLoss()(model(xb, mb), yb)
    loss.backward()
    metrics = model.read_probe()
    metrics["probe_loss"] = float(loss.detach().cpu())
    model.set_probe(False)
    model.zero_grad(set_to_none=True)
    return metrics


def resolve_switch(method: str, dataset: str, architecture: str, tier: str, switch_map: dict | None) -> int | None:
    if method.startswith("switch_") and method.split("_", 1)[1].isdigit():
        return int(method.split("_", 1)[1])
    if method.endswith("_switch"):
        if switch_map is None:
            raise ValueError(f"{method} requires --switch-map")
        try:
            return int(switch_map[method][dataset][architecture][tier])
        except KeyError as exc:
            raise KeyError(f"switch map missing {method}/{dataset}/{architecture}/{tier}") from exc
    return None


def run_one(dataset, method, seed, args, device, switch_map) -> dict:
    r1.set_seed(seed)
    bundle = load_bundle(dataset, seed, args)
    layers, embed_dim, heads = CAPACITY_TIERS[args.tier]
    switch_epoch = resolve_switch(method, dataset, args.architecture, args.tier, switch_map)
    model = InstrumentedClassifier(
        bundle, method, args.architecture, embed_dim, heads, layers,
        args.dropout, args.max_epochs, switch_epoch,
    ).to(device)
    model.set_self_check(args.self_check)
    tr = r1.make_loader(bundle.train, bundle.train_mask, args.batch_size, True, device)
    va = r1.make_loader(bundle.val, bundle.val_mask, args.batch_size, False, device)
    probe, probe_n = make_probe_loader(bundle, args.batch_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    epoch_log = []
    best = {"accuracy": -1.0}
    best_epoch = -1
    t0 = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.set_epoch(epoch)
        model.train()
        loss_sum = torch.zeros((), device=device)
        n_batches = 0
        for batch in tr:
            xb, yb, mb = r1.unpack(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_sum += loss.detach()
            n_batches += 1
        train_loss = float((loss_sum / max(n_batches, 1)).cpu())
        if not math.isfinite(train_loss):
            return {
                "dataset": dataset, "method": method, "seed": seed,
                "status": "failed_nonfinite_loss", "epoch": epoch,
            }

        vm = r1.evaluate(model, va, device, bundle.classes)
        st = model.stats()
        pm = mechanism_probe(model, probe, device, bundle.classes)
        row = {
            "epoch": epoch,
            "active_normalizer": model.active_method,
            "train_loss": train_loss,
            "val_accuracy": vm["accuracy"],
            "val_macro_f1": vm["macro_f1"],
            "val_loss": vm["loss"],
            "attention_density": vm["nonzero_ratio"],
            "attention_entropy": vm["attention_entropy"],
            "alpha_mean": st["alpha_mean"],
            "alpha_std": st["alpha_std"],
            **pm,
        }
        epoch_log.append(row)
        if vm["accuracy"] > best["accuracy"]:
            best, best_epoch = vm, epoch

    final = epoch_log[-1]
    result = {
        "dataset": dataset,
        "dataset_source": bundle.source,
        "architecture": args.architecture,
        "method": method,
        "switch_epoch": switch_epoch,
        "seed": seed,
        "tier": args.tier,
        "status": "completed",
        "layers": layers,
        "embed_dim": embed_dim,
        "heads": heads,
        "cfg_max_epochs": args.max_epochs,
        "cfg_batch_size": args.batch_size,
        "cfg_patch": args.patch,
        "cfg_val_fraction": args.val_fraction,
        "cfg_patience": 0,
        "cfg_lr": args.lr,
        "cfg_weight_decay": args.weight_decay,
        "cfg_dropout": args.dropout,
        "seq_len": bundle.seq_len + 1,
        "classes": bundle.classes,
        "n_train": bundle.sizes["train"],
        "n_val": bundle.sizes["val"],
        "probe_n": probe_n,
        "masked_attention": bundle.train_mask is not None,
        "epochs_run": len(epoch_log),
        "best_epoch": best_epoch,
        "best_val_accuracy": best["accuracy"],
        "val_accuracy": final["val_accuracy"],
        "val_macro_f1": final["val_macro_f1"],
        "val_loss": final["val_loss"],
        "val_nonzero_ratio": final["attention_density"],
        "val_attention_entropy": final["attention_entropy"],
        "train_seconds": time.perf_counter() - t0,
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 ** 2)
        if device.type == "cuda" else None,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epoch_log": epoch_log,
    }
    if args.final_test_read:
        test_ds, test_classes = load_official_test(dataset, args)
        test_loader = r1.make_loader(test_ds, None, args.batch_size, False, device)
        tm = r1.evaluate(model, test_loader, device, test_classes)
        result.update({
            "final_test_read": True,
            "test_accuracy": tm["accuracy"],
            "test_macro_f1": tm["macro_f1"],
            "test_loss": tm["loss"],
            "n_test": len(test_ds),
        })
    else:
        result["final_test_read"] = False
    return result


def cell_config(args) -> dict:
    return {
        "code_version": CODE_VERSION,
        "experiment_id": args.experiment_id,
        "architecture": args.architecture,
        "tier": args.tier,
        "cfg_max_epochs": args.max_epochs,
        "cfg_batch_size": args.batch_size,
        "cfg_patch": args.patch,
        "cfg_val_fraction": args.val_fraction,
        "cfg_patience": 0,
        "cfg_lr": args.lr,
        "cfg_weight_decay": args.weight_decay,
        "cfg_dropout": args.dropout,
        "cfg_train_limit": args.train_limit,
        "cfg_text_train_limit": args.text_train_limit,
        "cfg_synthetic_train_limit": args.synthetic_train_limit,
        "cfg_seq_len": args.seq_len,
        "cfg_vocab_size": args.vocab_size,
        "cfg_switch_map_sha256": sha256_file(Path(args.switch_map)) if args.switch_map else None,
        "cfg_final_test_read": bool(args.final_test_read),
        "cfg_test_freeze_sha256": sha256_file(Path(args.test_freeze)) if args.test_freeze else None,
    }


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_completed(out: Path, expected_hash: str) -> set[tuple[str, str, int]]:
    done = set()
    for path in sorted((out / "raw").glob("*.jsonl")):
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("status") == "completed" and row.get("config_hash") == expected_hash:
                        done.add((row["dataset"], row["method"], int(row["seed"])))
        except OSError:
            continue
    return done


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def finalise(out: Path, run_id: str, rows: list[dict], args, device, cfg, cfg_hash) -> None:
    flat = [{k: v for k, v in r.items() if k not in {"epoch_log", "traceback"}} for r in rows]
    write_csv(out / "processed" / f"{run_id}_runs.csv", flat)
    logs = [
        {
            "dataset": r["dataset"], "architecture": r.get("architecture"),
            "tier": r.get("tier"), "method": r["method"], "seed": r["seed"], **e,
        }
        for r in rows if r.get("status") == "completed" for e in r.get("epoch_log", [])
    ]
    write_csv(out / "processed" / f"{run_id}_epochs.csv", logs)
    manifest = {
        "run_id": run_id,
        "operation_id": "snd-crossover-mechanism-s2d-extension-20260826",
        "experiment_id": args.experiment_id,
        "code_version": CODE_VERSION,
        "config": cfg,
        "config_hash": cfg_hash,
        "args": vars(args),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "host": platform.node(),
        "dataset_sha256": {**r1.datasets_r1.provenance(), **datasets_e5b.provenance()},
        "source_sha256": {
            "extension_runner.py": sha256_file(Path(__file__)),
            "benchmark_r1.py": sha256_file(BASE_SRC / "benchmark_r1.py"),
            "datasets_r1.py": sha256_file(BASE_SRC / "datasets_r1.py"),
            "datasets_e5b.py": sha256_file(BASE_SRC / "datasets_e5b.py"),
        },
        "completed": sum(r.get("status") == "completed" for r in rows),
        "failed": sum(r.get("status") != "completed" for r in rows),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (out / "manifests" / f"{run_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-id", choices=["EXP-SND-101", "EXP-SND-102", "EXP-SND-103"],
                    default="EXP-SND-101")
    ap.add_argument("--datasets", nargs="+", default=["mnist"])
    ap.add_argument("--methods", nargs="+", default=["softmax", "topk_softmax_025"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--architecture", choices=["plain_transformer", "local_hybrid"],
                    default="plain_transformer")
    ap.add_argument("--tier", choices=sorted(CAPACITY_TIERS), default="compact")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patch", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--text-train-limit", type=int, default=0)
    ap.add_argument("--synthetic-train-limit", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--data-root", default="./data_cache")
    ap.add_argument("--out", default=str(RUN_DIR / "outputs"))
    ap.add_argument("--switch-map")
    ap.add_argument("--final-test-read", action="store_true")
    ap.add_argument("--test-freeze")
    ap.add_argument("--cell-timeout-seconds", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--env", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.env:
        print(json.dumps({
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node(), "base_src": str(BASE_SRC), "code_version": CODE_VERSION,
        }, indent=2))
        return 0
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.batch_size != 256 and not args.tag.startswith("smoke"):
        raise ValueError("registered evidence requires batch size 256")
    registered = set(DISCOVERY_DATASETS) | set(CONFIRMATION_DATASETS)
    unknown = set(args.datasets) - registered
    if unknown:
        raise ValueError(f"datasets outside registration: {sorted(unknown)}")
    if args.experiment_id in {"EXP-SND-101", "EXP-SND-102"} and set(args.datasets) & set(CONFIRMATION_DATASETS):
        raise ValueError("confirmation datasets are blocked before EXP-SND-103")
    if args.final_test_read:
        if args.experiment_id != "EXP-SND-103":
            raise ValueError("official test read is allowed only for EXP-SND-103")
        if set(args.datasets) - set(CONFIRMATION_DATASETS):
            raise ValueError("official test read is restricted to confirmation datasets")
        if not args.test_freeze or not Path(args.test_freeze).is_file():
            raise ValueError("official test read requires the frozen M3 validation certificate")
    validate_registered_invocation(args)

    out = Path(args.out)
    for sub in ("raw", "processed", "manifests", "logs", "status"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    switch_map = json.loads(Path(args.switch_map).read_text(encoding="utf-8")) if args.switch_map else None
    cfg = cell_config(args)
    cfg_hash = config_hash(cfg)
    run_id = time.strftime("mch002_%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    raw_path = out / "raw" / f"{run_id}.jsonl"
    total = len(args.datasets) * len(args.methods) * len(args.seeds)
    done = load_completed(out, cfg_hash) if args.skip_existing else set()
    print(
        f"RUN {run_id} experiment={args.experiment_id} device={device} cells={total} "
        f"architecture={args.architecture} tier={args.tier} config={cfg_hash[:12]}",
        flush=True,
    )
    if done:
        print(f"SKIP_EXISTING verified_cells={len(done)}", flush=True)

    rows: list[dict] = []
    n = 0
    with raw_path.open("w", encoding="utf-8") as f:
        for dataset in args.datasets:
            for method in args.methods:
                for seed in args.seeds:
                    key = (dataset, method, int(seed))
                    n += 1
                    if key in done:
                        print(f"[{n}/{total}] {dataset} {method} seed={seed} skipped_verified", flush=True)
                        continue
                    t0 = time.perf_counter()
                    try:
                        old_handler = None
                        if args.cell_timeout_seconds and hasattr(signal, "SIGALRM"):
                            def _timeout_handler(signum, frame):
                                del signum, frame
                                raise TimeoutError(
                                    f"cell exceeded {args.cell_timeout_seconds} registered seconds"
                                )
                            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                            signal.alarm(args.cell_timeout_seconds)
                        row = run_one(dataset, method, seed, args, device, switch_map)
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "dataset": dataset, "architecture": args.architecture,
                            "tier": args.tier, "method": method, "seed": seed,
                            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-4000:],
                        }
                    finally:
                        if args.cell_timeout_seconds and hasattr(signal, "SIGALRM"):
                            signal.alarm(0)
                            if old_handler is not None:
                                signal.signal(signal.SIGALRM, old_handler)
                    row["code_version"] = CODE_VERSION
                    row["config_hash"] = cfg_hash
                    row["wall_seconds"] = time.perf_counter() - t0
                    rows.append(row)
                    f.write(json.dumps(row, allow_nan=True) + "\n")
                    f.flush()
                    print(
                        f"[{n}/{total}] {dataset:18s} {args.architecture:17s} {args.tier:7s} "
                        f"{method:24s} seed={seed} {row.get('status')} "
                        f"val={row.get('val_accuracy')} sec={row['wall_seconds']:.1f}",
                        flush=True,
                    )
                    if row.get("status") != "completed":
                        print(f"ERROR {row.get('error')}", flush=True)
                        if "CUDA" in str(row.get("error")) or "Accelerator" in str(row.get("error")):
                            finalise(out, run_id, rows, args, device, cfg, cfg_hash)
                            return 3
                        if "registered seconds" in str(row.get("error")):
                            finalise(out, run_id, rows, args, device, cfg, cfg_hash)
                            return 4
    finalise(out, run_id, rows, args, device, cfg, cfg_hash)
    failed = sum(r.get("status") != "completed" for r in rows)
    print(f"DONE {run_id} completed={len(rows)-failed} failed={failed}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
