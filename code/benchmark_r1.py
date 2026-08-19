#!/usr/bin/env python3
"""r1 benchmark: validation-split selection, masked text attention, scaled regime.

This is the r0 replication-package benchmark extended to answer the Machine
Learning desk-reject criticisms (issue atoms SND-RJ-001..006). The normalizer
implementations -- sparsemax, the entmax bisection, the HAE parameterisation and
the top-k form -- are carried over UNCHANGED so r1 stays comparable with r0.
Every change is in the protocol around them.

E1  A validation split exists. The official training pool is split per seed into
    train/validation; the official TEST partition is never read by this script.
    Selection therefore cannot see test, which is the r0 defect the editor named.

E2  Text attention is masked. The mask is applied to the SCORES, before the
    normalizer, because sparsemax/entmax normalise over their input support and
    zeroing weights afterwards would leave mass already distributed over padded
    slots. Masked keys are pushed a finite margin below the row minimum rather
    than to -inf: -inf degenerates the entmax bisection range and a huge finite
    value destroys its precision. `--self-check` asserts masked keys receive
    exactly zero mass. Fixed top-k additionally counts k over VALID tokens per
    row, ceil(r * n_valid) rather than ceil(r * T); a single global k on
    variable-length documents silently changes the retained fraction per
    document, which is the same defect in a second form.

E3  Scale is parameterised: full official training partitions by default, a patch
    size that controls sequence length, early stopping on validation instead of a
    fixed epoch budget, and a named capacity tier so capacity dependence becomes
    a reported result rather than an unexamined confound.

E4  Early-training predictors are logged every epoch (attention density, entropy,
    validation trajectory, learned-alpha mean/spread). The selection rule is
    fitted from these logs by a separate script; this file only produces honest
    inputs for it.

Dependencies are deliberately torch + numpy ONLY. See datasets_r1.py for why.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets_r1  # noqa: E402

TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")

# Masked keys sit this far below the row minimum: large enough that
# sparsemax/entmax give them exactly zero mass (tau never falls below
# row_min - 1), small enough that the 18-step bisection keeps its precision.
MASK_MARGIN = 60.0

IMAGE_DATASETS = {"mnist": 4, "fashion_mnist": 4, "kmnist": 4, "cifar10": 4, "cifar100": 4}
CAPACITY_TIERS = {"compact": (2, 64, 4), "medium": (4, 128, 4)}


# ---------------------------------------------------------------- metrics --

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


# ------------------------------------------------------------------- data --

@dataclass
class Bundle:
    train: TensorDataset
    val: TensorDataset
    input_kind: str
    input_dim: int
    classes: int
    seq_len: int
    source: str
    train_mask: torch.Tensor | None = None
    val_mask: torch.Tensor | None = None
    sizes: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash_token(token: str, vocab_size: int) -> int:
    import hashlib
    d = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return 2 + (int.from_bytes(d, "little") % (vocab_size - 2))


def tokenize_text(text: str, seq_len: int, vocab_size: int) -> tuple[list[int], int]:
    ids = [stable_hash_token(t.lower(), vocab_size) for t in TOKEN_RE.findall(text)]
    if len(ids) >= seq_len:
        return ids[:seq_len], seq_len
    n = len(ids)
    return ids + [0] * (seq_len - n), n


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    if images.dim() == 3:
        images = images.unsqueeze(1)
    n, c, h, w = images.shape
    images = images[:, :, : h - h % patch, : w - w % patch]
    x = images.unfold(2, patch, patch).unfold(3, patch, patch)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    return x.view(n, -1, c * patch * patch).float()


def split_train_val(n: int, val_fraction: float, seed: int):
    """Deterministic per-seed split of the official TRAINING pool only."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_fraction))
    return perm[n_val:], perm[:n_val]


def _limit(n: int, limit, seed: int) -> np.ndarray:
    if not limit or limit >= n:
        return np.arange(n)
    return np.random.default_rng(seed).permutation(n)[:limit]


def load_image_bundle(name: str, seed: int, args) -> Bundle:
    raw = datasets_r1.load_raw(name, Path(args.data_root))
    patch = args.patch if args.patch else IMAGE_DATASETS[name]
    x_all = torch.from_numpy(np.ascontiguousarray(raw["train_x"]))
    y_all = torch.from_numpy(raw["train_y"])
    keep = _limit(len(x_all), args.train_limit, seed)
    x_all, y_all = x_all[keep], y_all[keep]
    fit, val = split_train_val(len(x_all), args.val_fraction, seed)
    x_tr = patchify(x_all[fit], patch)
    x_va = patchify(x_all[val], patch)
    return Bundle(
        train=TensorDataset(x_tr, y_all[fit]),
        val=TensorDataset(x_va, y_all[val]),
        input_kind="patch", input_dim=int(x_tr.shape[-1]),
        classes=int(raw["classes"]), seq_len=int(x_tr.shape[1]),
        source=f"{name}:patch{patch}",
        sizes={"train": len(x_tr), "val": len(x_va)},
    )


def load_text_bundle(seed: int, args) -> Bundle:
    raw = datasets_r1.load_raw("twenty_news", Path(args.data_root))
    L, V = args.seq_len, args.vocab_size
    docs, ys = raw["train_docs"], raw["train_y"]
    keep = _limit(len(docs), args.text_train_limit, seed)
    fit, val = split_train_val(len(keep), args.val_fraction, seed)

    def encode(idx):
        seqs, lens = [], []
        for i in idx:
            s, n = tokenize_text(docs[int(keep[int(i)])], L, V)
            seqs.append(s)
            lens.append(n)
        x = torch.tensor(seqs, dtype=torch.long)
        y = torch.tensor([int(ys[int(keep[int(i)])]) for i in idx], dtype=torch.long)
        # True == real token. CLS is prepended by the model and always valid.
        m = torch.arange(L).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)
        return x, y, m

    x_tr, y_tr, m_tr = encode(fit)
    x_va, y_va, m_va = encode(val)
    return Bundle(
        train=TensorDataset(x_tr, y_tr), val=TensorDataset(x_va, y_va),
        input_kind="token", input_dim=V, classes=int(raw["classes"]), seq_len=L,
        source="20newsgroups:masked",
        train_mask=m_tr, val_mask=m_va,
        sizes={"train": len(x_tr), "val": len(x_va)},
    )


def _marker(seed: int, n: int, L: int, V: int, C: int) -> TensorDataset:
    rng = np.random.default_rng(seed)
    x = rng.integers(2, V, size=(n, L), dtype=np.int64)
    pos = rng.integers(0, L - 1, size=n)
    y = np.empty(n, dtype=np.int64)
    for r, p in enumerate(pos):
        x[r, p] = 1
        y[r] = int(x[r, p + 1] % C)
    return TensorDataset(torch.tensor(x), torch.tensor(y))


def load_synthetic_bundle(seed: int, args) -> Bundle:
    C, L, V = 10, args.seq_len, args.vocab_size
    n_tr = args.synthetic_train_limit
    n_va = max(1, int(round(n_tr * args.val_fraction)))
    return Bundle(
        train=_marker(seed, n_tr, L, V, C), val=_marker(seed + 5000, n_va, L, V, C),
        input_kind="token", input_dim=V, classes=C, seq_len=L,
        source="synthetic:marker_following_token",
        sizes={"train": n_tr, "val": n_va},
    )


def load_bundle(name: str, seed: int, args) -> Bundle:
    if name in IMAGE_DATASETS:
        return load_image_bundle(name, seed, args)
    if name == "twenty_news":
        return load_text_bundle(seed, args)
    if name == "synthetic_marker":
        return load_synthetic_bundle(seed, args)
    raise ValueError(name)


# ------------------------------------------------ normalizers (from r0) --

def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, descending=True, dim=dim).values
    rng = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    rng = rng.view(view)
    cumsum = zs.cumsum(dim)
    k = ((1 + rng * zs) > cumsum).sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(dim, k - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def entmax_bisect(logits: torch.Tensor, alpha=1.5, dim: int = -1, n_iter: int = 18) -> torch.Tensor:
    a = torch.tensor(alpha, dtype=logits.dtype, device=logits.device) if isinstance(alpha, float) \
        else alpha.to(dtype=logits.dtype, device=logits.device)
    while a.dim() < logits.dim():
        a = a.unsqueeze(-1)
    a = a.clamp(1.05, 2.0)
    d = logits.size(dim)
    y = logits * (a - 1)
    y = y - y.max(dim=dim, keepdim=True).values
    tau_lo = y.min(dim=dim, keepdim=True).values - 1
    tau_hi = y.max(dim=dim, keepdim=True).values - (1.0 / d) ** (a - 1)
    inv = 1.0 / (a - 1)
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(y - tau, min=0) ** inv
        f = p.sum(dim=dim, keepdim=True) - 1
        tau_lo = torch.where(f >= 0, tau, tau_lo)
        tau_hi = torch.where(f >= 0, tau_hi, tau)
    p = torch.clamp(y - tau_hi, min=0) ** inv
    return p / (p.sum(dim=dim, keepdim=True) + 1e-12)


def parse_topk_ratio(method: str) -> float:
    if method == "topk_softmax":
        return 0.25
    return {"0125": 0.125, "025": 0.25, "05": 0.5}[method.rsplit("_", 1)[-1]]


# `bottomk_softmax_*` is a CONTROL, not a proposed method. It keeps the same
# number of keys as the matching top-k ratio but chooses the LOWEST-scoring ones,
# so it holds support size fixed while destroying selection quality. It exists to
# separate two explanations of the r0 result that the accuracy numbers alone
# cannot distinguish: that restricting attention regularises an underfit model,
# or that top-k genuinely selects informative keys. If bottom-k also beats dense
# softmax in the underfit regime, only the first explanation survives.
# Deterministic on purpose -- a random-k control would add evaluation noise and
# make the comparison harder to read, not easier.


class NormalizedSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int, method: str):
        super().__init__()
        assert embed_dim % heads == 0
        self.embed_dim, self.heads = embed_dim, heads
        self.head_dim = embed_dim // heads
        self.method = method
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.alpha_raw = nn.Parameter(torch.zeros(heads)) if method == "headwise_adaptive_entmax" else None
        self.last = {"nonzero_ratio": float("nan"), "entropy": float("nan"),
                     "alpha_mean": float("nan"), "alpha_std": float("nan")}
        self.self_check = False
        # Every entry in `last` is written with a `.cpu()` call, which forces a
        # device synchronisation on each forward pass. During training that is
        # thousands of stalls per epoch for numbers nobody reads until the epoch
        # ends, so collection is off by default and enabled only for evaluation.
        self.collect_stats = False

    def normalize(self, scores: torch.Tensor, key_valid: torch.Tensor | None) -> torch.Tensor:
        m = self.method
        if key_valid is not None:
            row_min = scores.masked_fill(~key_valid, float("inf")).amin(dim=-1, keepdim=True)
            scores = torch.where(key_valid, scores, row_min - MASK_MARGIN)

        if m == "softmax":
            attn = torch.softmax(scores, dim=-1)
        elif m.startswith("topk_softmax") or m.startswith("bottomk_softmax"):
            ratio = parse_topk_ratio(m)
            worst = m.startswith("bottomk_softmax")
            if key_valid is None:
                k = max(1, int(math.ceil(scores.size(-1) * ratio)))
                vals, idx = scores.topk(k, dim=-1, largest=not worst)
                masked = torch.full_like(scores, -1e9)
                masked.scatter_(-1, idx, vals)
            else:
                n_valid = key_valid.sum(dim=-1, keepdim=True).to(scores.dtype)
                k_row = torch.ceil(n_valid * ratio).clamp(min=1)
                k_max = int(k_row.max().item())
                vals, idx = scores.topk(k_max, dim=-1, largest=not worst)
                rank = torch.arange(k_max, device=scores.device).view(1, 1, 1, -1)
                keep = rank < k_row
                masked = torch.full_like(scores, -1e9)
                masked.scatter_(-1, idx, torch.where(keep, vals, torch.full_like(vals, -1e9)))
            attn = torch.softmax(masked, dim=-1)
        elif m == "sparsemax":
            attn = sparsemax(scores, dim=-1)
        elif m == "entmax15":
            attn = entmax_bisect(scores, alpha=1.5, dim=-1)
        elif m == "headwise_adaptive_entmax":
            alpha = 1.05 + 0.90 * torch.sigmoid(self.alpha_raw)
            if self.collect_stats:
                self.last["alpha_mean"] = float(alpha.detach().mean().cpu())
                self.last["alpha_std"] = float(alpha.detach().std().cpu()) if self.heads > 1 else 0.0
            attn = entmax_bisect(scores, alpha=alpha.view(1, self.heads, 1, 1), dim=-1)
        else:
            raise ValueError(m)

        if self.self_check and key_valid is not None:
            leak = float(attn.masked_fill(key_valid, 0.0).abs().max().detach().cpu())
            if leak > 1e-6:
                raise AssertionError(f"{m}: masked key received attention mass {leak:.3e}")

        if self.collect_stats:
            with torch.no_grad():
                if key_valid is None:
                    self.last["nonzero_ratio"] = float((attn > 1e-7).float().mean().cpu())
                    p = attn
                else:
                    denom = key_valid.expand_as(attn)
                    self.last["nonzero_ratio"] = float(
                        ((attn > 1e-7) & denom).sum().cpu() / denom.sum().clamp(min=1).cpu())
                    p = attn * denom
                q = p.clamp(min=1e-12)
                self.last["entropy"] = float((-(q * q.log()).sum(-1)).mean().cpu())
        return attn

    def forward(self, x, key_valid=None):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.normalize(scores, key_valid)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        return self.out(out)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, heads, method, dropout):
        super().__init__()
        self.attn = NormalizedSelfAttention(embed_dim, heads, method)
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(4 * embed_dim, embed_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_valid=None):
        x = self.norm1(x + self.drop(self.attn(x, key_valid)))
        return self.norm2(x + self.drop(self.ff(x)))


class SequenceClassifier(nn.Module):
    def __init__(self, input_kind, input_dim, classes, method, embed_dim, heads, layers, max_len, dropout):
        super().__init__()
        self.input = nn.Linear(input_dim, embed_dim) if input_kind == "patch" \
            else nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, heads, method, dropout) for _ in range(layers)])
        self.head = nn.Linear(embed_dim, classes)

    def set_self_check(self, on: bool):
        for b in self.blocks:
            b.attn.self_check = on

    def set_collect_stats(self, on: bool):
        for b in self.blocks:
            b.attn.collect_stats = on

    def forward(self, x, content_mask=None):
        x = self.input(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        key_valid = None
        if content_mask is not None:
            ones = torch.ones(content_mask.size(0), 1, dtype=torch.bool, device=content_mask.device)
            key_valid = torch.cat([ones, content_mask], dim=1).view(content_mask.size(0), 1, 1, -1)
        for blk in self.blocks:
            x = blk(x, key_valid)
        return self.head(x[:, 0])

    def stats(self) -> dict:
        def agg(key):
            a = np.asarray([b.attn.last[key] for b in self.blocks], dtype=float)
            return float(np.nanmean(a)) if not np.isnan(a).all() else float("nan")
        return {k: agg(k) for k in ("nonzero_ratio", "entropy", "alpha_mean", "alpha_std")}


# ------------------------------------------------------------ train/eval --

class GpuBatches:
    """Whole-split GPU residency instead of a DataLoader.

    Measured on the pilot: one CIFAR-10 cell took 1072 s with a `DataLoader`
    while the RTX 5060 sat at 31-37% utilisation, i.e. the run was bound by the
    Python batching loop and per-batch host-to-device copies, not by the GPU.
    The model here is tiny and the splits are small (CIFAR-10 train is ~490 MB
    as patches), so the entire split fits in 8 GB VRAM and can be indexed in
    place. Shuffling becomes a `randperm` on device.

    This changes throughput only. The batch contents, the order semantics and
    the arithmetic are the same, which matters because r1 must stay comparable
    with r0.
    """

    def __init__(self, x, y, mask, batch, shuffle, device):
        self.x = x.to(device, non_blocking=True)
        self.y = y.to(device, non_blocking=True)
        self.mask = mask.to(device, non_blocking=True) if mask is not None else None
        self.batch = batch
        self.shuffle = shuffle
        self.device = device
        self.n = self.x.shape[0]

    def __len__(self):
        return (self.n + self.batch - 1) // self.batch

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.device) if self.shuffle \
            else torch.arange(self.n, device=self.device)
        for s in range(0, self.n, self.batch):
            j = idx[s: s + self.batch]
            if self.mask is None:
                yield self.x[j], self.y[j], None
            else:
                yield self.x[j], self.y[j], self.mask[j]


def make_loader(ds, mask, batch, shuffle, device=None):
    if device is not None and device.type == "cuda":
        x, y = ds.tensors
        return GpuBatches(x, y, mask, batch, shuffle, device)
    if mask is None:
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)
    x, y = ds.tensors
    return DataLoader(TensorDataset(x, y, mask), batch_size=batch, shuffle=shuffle, num_workers=0)


def unpack(batch, device):
    """Accept both GpuBatches triples and DataLoader pairs/triples."""
    if len(batch) == 3:
        xb, yb, mb = batch
    else:
        xb, yb = batch
        mb = None
    if xb.device != device:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device) if mb is not None else None
    return xb, yb, mb


@torch.no_grad()
def evaluate(model, loader, device, n_classes) -> dict:
    model.eval()
    model.set_collect_stats(True)
    crit = nn.CrossEntropyLoss()
    yt, yp, losses, dens, ents = [], [], [], [], []
    for batch in loader:
        xb, yb, mb = unpack(batch, device)
        logits = model(xb, mb)
        losses.append(float(crit(logits, yb).cpu()))
        yt.append(yb.cpu().numpy())
        yp.append(logits.argmax(1).cpu().numpy())
        s = model.stats()
        dens.append(s["nonzero_ratio"])
        ents.append(s["entropy"])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    model.set_collect_stats(False)
    return {"accuracy": accuracy_score(yt, yp), "macro_f1": macro_f1(yt, yp, n_classes),
            "loss": float(np.mean(losses)), "nonzero_ratio": float(np.nanmean(dens)),
            "attention_entropy": float(np.nanmean(ents))}


def run_one(dataset, method, seed, args, device) -> dict:
    set_seed(seed)
    bundle = load_bundle(dataset, seed, args)
    layers, embed_dim, heads = CAPACITY_TIERS[args.tier]
    model = SequenceClassifier(bundle.input_kind, bundle.input_dim, bundle.classes, method,
                               embed_dim, heads, layers, bundle.seq_len, args.dropout).to(device)
    model.set_self_check(args.self_check)
    tr = make_loader(bundle.train, bundle.train_mask, args.batch_size, True, device)
    va = make_loader(bundle.val, bundle.val_mask, args.batch_size, False, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, bad, log = {"accuracy": -1.0}, -1, 0, []
    t0 = time.perf_counter()
    for epoch in range(args.max_epochs):
        model.train()
        # Accumulated on device: reading `loss` per step would synchronise the
        # GPU on every batch for a number only used once, at end of epoch.
        loss_sum = torch.zeros((), device=device)
        n_batches = 0
        for batch in tr:
            xb, yb, mb = unpack(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_sum += loss.detach()
            n_batches += 1
        mean_loss = float((loss_sum / max(n_batches, 1)).cpu())
        if not math.isfinite(mean_loss):
            return {"dataset": dataset, "method": method, "seed": seed, "status": "failed_nonfinite_loss"}
        losses = [mean_loss]
        vm = evaluate(model, va, device, bundle.classes)
        st = model.stats()
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                    "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"], "val_loss": vm["loss"],
                    "attention_density": vm["nonzero_ratio"], "attention_entropy": vm["attention_entropy"],
                    "alpha_mean": st["alpha_mean"], "alpha_std": st["alpha_std"]})
        if vm["accuracy"] > best["accuracy"]:
            best, best_epoch, bad = vm, epoch, 0
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                break
    return {"dataset": dataset, "dataset_source": bundle.source, "method": method, "seed": seed,
            "tier": args.tier, "status": "completed", "layers": layers, "embed_dim": embed_dim, "heads": heads,
            "cfg_max_epochs": args.max_epochs, "cfg_batch_size": args.batch_size,
            "cfg_patch": args.patch, "cfg_val_fraction": args.val_fraction,
            "cfg_patience": args.patience,
            "seq_len": bundle.seq_len + 1, "classes": bundle.classes,
            "n_train": bundle.sizes["train"], "n_val": bundle.sizes["val"],
            "masked_attention": bundle.train_mask is not None,
            "epochs_run": len(log), "best_epoch": best_epoch,
            "val_accuracy": best["accuracy"], "val_macro_f1": best["macro_f1"], "val_loss": best["loss"],
            "val_nonzero_ratio": best["nonzero_ratio"], "val_attention_entropy": best["attention_entropy"],
            "train_seconds": time.perf_counter() - t0,
            "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None,
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "epoch_log": log}


def load_completed(out: Path, cfg: dict | None = None) -> set[tuple[str, str, int]]:
    """(dataset, method, seed) triples already completed in any prior raw log.

    The grid runs on a GPU that also drives a display, so a long kernel can trip
    the X watchdog with cudaErrorLaunchTimeout. Recovery is per-process, not
    per-cell, because the CUDA context does not survive that error -- hence a
    resume path rather than a retry loop.
    """
    done: set[tuple[str, str, int]] = set()
    for path in sorted((out / "raw").glob("*.jsonl")):
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("status") != "completed":
                        continue
                    # Identity is not enough. The first resume treated a
                    # 1-epoch smoke cell as a finished grid cell and skipped
                    # the real one, leaving 18 seed-0 cells carrying a
                    # single epoch of training. Match the configuration too;
                    # a row without cfg_* predates this fix and is refused.
                    if cfg is not None:
                        if any(r.get(k) != v for k, v in cfg.items()):
                            continue
                    done.add((r["dataset"], r["method"], int(r["seed"])))
        except OSError:
            continue
    return done


def _finalise(out: Path, run_id: str, rows: list[dict], args, device) -> None:
    write_csv(out / "processed" / f"{run_id}_val_runs.csv",
              [{k: v for k, v in r.items() if k not in ("epoch_log", "traceback")} for r in rows])
    logs = [dict(dataset=r["dataset"], method=r["method"], seed=r["seed"], **e)
            for r in rows if r.get("status") == "completed" for e in r.get("epoch_log", [])]
    write_csv(out / "processed" / f"{run_id}_epoch_predictors.csv", logs)
    (out / "processed" / f"{run_id}_manifest.json").write_text(json.dumps({
        "run_id": run_id, "args": vars(args), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "host": platform.node(),
        "dataset_sha256": datasets_r1.provenance(),
        "completed": sum(1 for r in rows if r.get("status") == "completed"),
        "failed": sum(1 for r in rows if r.get("status") == "failed"),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["cifar10"])
    ap.add_argument("--methods", nargs="+", default=[
        "softmax", "topk_softmax_0125", "topk_softmax_025", "topk_softmax_05",
        "sparsemax", "entmax15", "headwise_adaptive_entmax"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--tier", choices=sorted(CAPACITY_TIERS), default="compact")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=512)
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
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (dataset, method, seed) cells already completed in outputs/raw/*.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--env", action="store_true", help="print environment and exit")
    args = ap.parse_args()

    if args.env:
        print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(),
                          "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          "host": platform.node()}, indent=2))
        return 0

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "processed").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Ampere-and-later tensor cores for fp32 matmuls. The model is small
        # enough that this is not the dominant cost, but it is free and the
        # normalizer comparison is unaffected: every method sees the same
        # precision, and the sparse mappings run in fp32 regardless.
        torch.set_float32_matmul_precision("high")
    run_id = time.strftime("r1_select_%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    raw_path = out / "raw" / f"{run_id}.jsonl"
    total = len(args.datasets) * len(args.methods) * len(args.seeds)
    print(f"RUN {run_id} device={device} cells={total} tier={args.tier}", flush=True)

    # `cfg_patience` belongs here: a fixed-budget ladder point and an
    # early-stopped point can share every other field and still be different
    # experiments, so leaving patience out would let a resume silently reuse the
    # wrong one -- the same class of defect as the smoke-cell reuse.
    run_cfg = {"cfg_max_epochs": args.max_epochs, "cfg_batch_size": args.batch_size,
               "cfg_patch": args.patch, "cfg_val_fraction": args.val_fraction,
               "cfg_patience": args.patience, "tier": args.tier}
    done_cells = load_completed(out, run_cfg) if args.skip_existing else set()
    if done_cells:
        print(f"SKIP_EXISTING {len(done_cells)} cells already completed", flush=True)

    rows, n = [], 0
    with raw_path.open("w", encoding="utf-8") as f:
        for ds in args.datasets:
            for me in args.methods:
                for sd in args.seeds:
                    if (ds, me, int(sd)) in done_cells:
                        n += 1
                        print(f"[{n}/{total}] {ds:14s} {me:26s} seed={sd} skipped_already_done", flush=True)
                        continue
                    t0 = time.perf_counter()
                    try:
                        row = run_one(ds, me, sd, args, device)
                    except Exception as exc:  # noqa: BLE001
                        import traceback
                        row = {"dataset": ds, "method": me, "seed": sd, "status": "failed",
                               "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()[-2000:]}
                        # A cudaErrorLaunchTimeout poisons the CUDA context: every
                        # later cell in this process dies instantly without doing
                        # work. On 2026-08-17 one such trip turned into 296
                        # cascade failures in seconds. Stop the process instead and
                        # let the driver relaunch with --skip-existing.
                        if "CUDA" in str(exc) or "Accelerator" in type(exc).__name__:
                            row["wall_seconds"] = time.perf_counter() - t0
                            f.write(json.dumps(row) + "\n")
                            f.flush()
                            print(f"[{n + 1}/{total}] {ds} {me} seed={sd} FAILED_CUDA -> aborting process "
                                  f"so the context is rebuilt on relaunch", flush=True)
                            print("   ERROR:", row.get("error"), flush=True)
                            rows.append(row)
                            _finalise(out, run_id, rows, args, device)
                            return 3
                    row["wall_seconds"] = time.perf_counter() - t0
                    rows.append(row)
                    n += 1
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"[{n}/{total}] {ds:14s} {me:26s} seed={sd} "
                          f"{row.get('status')} val_acc={row.get('val_accuracy')} "
                          f"ep={row.get('epochs_run')} {row['wall_seconds']:.1f}s", flush=True)
                    if row.get("status") == "failed":
                        print("   ERROR:", row.get("error"), flush=True)

    _finalise(out, run_id, rows, args, device)
    print(f"DONE {run_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
