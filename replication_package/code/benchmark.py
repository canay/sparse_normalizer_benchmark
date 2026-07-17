#!/usr/bin/env python3
"""Reproducible compact benchmark for sparse attention normalizers.

The script preserves the implementation used by the saved evidence, including
the disclosed unmasked fixed-length representation for 20 Newsgroups. Outputs
are written below ``replication_package/outputs`` and dataset caches below
``replication_package/data_cache``. Plotting is intentionally separate from
model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUTS = PACKAGE / "outputs"
RAW = OUTPUTS / "raw"
PROCESSED = OUTPUTS / "processed"
STATS = OUTPUTS / "statistical_tests"
RUN_LOGS = OUTPUTS / "run_logs"
DATA_DIR = PACKAGE / "data_cache"

TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


@dataclass
class DatasetBundle:
    train: TensorDataset
    test: TensorDataset
    input_kind: str
    input_dim: int
    classes: int
    train_n: int
    test_n: int
    source: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash_token(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return 2 + (int.from_bytes(digest, "little") % (vocab_size - 2))


def tokenize_text(text: str, seq_len: int, vocab_size: int) -> list[int]:
    ids = [stable_hash_token(tok.lower(), vocab_size) for tok in TOKEN_RE.findall(text)]
    if len(ids) >= seq_len:
        return ids[:seq_len]
    return ids + [0] * (seq_len - len(ids))


def patchify_images(images: torch.Tensor, patch: int) -> torch.Tensor:
    # Input can be N,H,W or N,C,H,W. Output: N,T,D.
    if images.dim() == 3:
        images = images.unsqueeze(1)
    n, c, h, w = images.shape
    images = images[:, :, : h - h % patch, : w - w % patch]
    x = images.unfold(2, patch, patch).unfold(3, patch, patch)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    return x.view(n, -1, c * patch * patch).float()


def deterministic_subset(length: int, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    take = min(length, limit)
    return rng.permutation(length)[:take]


def load_torchvision_image_dataset(name: str, seed: int, train_limit: int, test_limit: int) -> DatasetBundle | None:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # noqa: BLE001
        print(f"DATASET_SKIP {name} torchvision_import_failed {type(exc).__name__}: {exc}")
        return None

    tfm = transforms.Compose([transforms.ToTensor()])
    root = DATA_DIR / "torchvision"
    try:
        if name == "fashion_mnist":
            train_ds = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
            test_ds = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
            patch = 7
            classes = 10
        elif name == "cifar10":
            train_ds = datasets.CIFAR10(root, train=True, download=True, transform=tfm)
            test_ds = datasets.CIFAR10(root, train=False, download=True, transform=tfm)
            patch = 8
            classes = 10
        elif name == "kmnist":
            train_ds = datasets.KMNIST(root, train=True, download=True, transform=tfm)
            test_ds = datasets.KMNIST(root, train=False, download=True, transform=tfm)
            patch = 7
            classes = 10
        else:
            raise ValueError(name)
    except Exception as exc:  # noqa: BLE001
        print(f"DATASET_SKIP {name} download_or_load_failed {type(exc).__name__}: {exc}")
        return None

    def convert(ds, idx: np.ndarray) -> TensorDataset:
        xs, ys = [], []
        for i in idx:
            img, label = ds[int(i)]
            xs.append(img)
            ys.append(int(label))
        x = patchify_images(torch.stack(xs), patch)
        y = torch.tensor(ys, dtype=torch.long)
        return TensorDataset(x, y)

    train_idx = deterministic_subset(len(train_ds), train_limit, seed)
    test_idx = deterministic_subset(len(test_ds), test_limit, seed + 10_000)
    train = convert(train_ds, train_idx)
    test = convert(test_ds, test_idx)
    return DatasetBundle(
        train=train,
        test=test,
        input_kind="patch",
        input_dim=int(train.tensors[0].shape[-1]),
        classes=classes,
        train_n=len(train),
        test_n=len(test),
        source=f"torchvision:{name}",
    )


def load_twenty_news(seed: int, train_limit: int, test_limit: int, seq_len: int, vocab_size: int) -> DatasetBundle | None:
    try:
        from sklearn.datasets import fetch_20newsgroups

        train_raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), data_home=str(DATA_DIR / "sklearn"))
        test_raw = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), data_home=str(DATA_DIR / "sklearn"))
    except Exception as exc:  # noqa: BLE001
        print(f"DATASET_SKIP twenty_news fetch_failed {type(exc).__name__}: {exc}")
        return None

    train_idx = deterministic_subset(len(train_raw.data), train_limit, seed)
    test_idx = deterministic_subset(len(test_raw.data), test_limit, seed + 10_000)
    x_train = torch.tensor([tokenize_text(train_raw.data[int(i)], seq_len, vocab_size) for i in train_idx], dtype=torch.long)
    y_train = torch.tensor([int(train_raw.target[int(i)]) for i in train_idx], dtype=torch.long)
    x_test = torch.tensor([tokenize_text(test_raw.data[int(i)], seq_len, vocab_size) for i in test_idx], dtype=torch.long)
    y_test = torch.tensor([int(test_raw.target[int(i)]) for i in test_idx], dtype=torch.long)
    return DatasetBundle(
        train=TensorDataset(x_train, y_train),
        test=TensorDataset(x_test, y_test),
        input_kind="token",
        input_dim=vocab_size,
        classes=len(train_raw.target_names),
        train_n=len(x_train),
        test_n=len(x_test),
        source="sklearn:20newsgroups",
    )


def generate_marker_sequence(seed: int, n: int, seq_len: int, vocab_size: int, classes: int) -> TensorDataset:
    rng = np.random.default_rng(seed)
    # 0 padding not used; 1 marks the position. Label is the token following marker modulo classes.
    x = rng.integers(2, vocab_size, size=(n, seq_len), dtype=np.int64)
    marker_pos = rng.integers(0, seq_len - 1, size=n)
    y = np.empty(n, dtype=np.int64)
    for row, pos in enumerate(marker_pos):
        x[row, pos] = 1
        y[row] = int(x[row, pos + 1] % classes)
    return TensorDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))


def load_synthetic_marker(seed: int, train_limit: int, test_limit: int, seq_len: int, vocab_size: int) -> DatasetBundle:
    classes = 10
    train = generate_marker_sequence(seed, train_limit, seq_len, vocab_size, classes)
    test = generate_marker_sequence(seed + 10_000, test_limit, seq_len, vocab_size, classes)
    return DatasetBundle(
        train=train,
        test=test,
        input_kind="token",
        input_dim=vocab_size,
        classes=classes,
        train_n=len(train),
        test_n=len(test),
        source="synthetic:marker_following_token",
    )


def load_dataset(name: str, seed: int, args) -> DatasetBundle | None:
    if name in {"fashion_mnist", "cifar10", "kmnist"}:
        return load_torchvision_image_dataset(name, seed, args.train_limit, args.test_limit)
    if name == "twenty_news":
        return load_twenty_news(seed, args.text_train_limit, args.text_test_limit, args.seq_len, args.vocab_size)
    if name == "synthetic_marker":
        return load_synthetic_marker(seed, args.synthetic_train_limit, args.synthetic_test_limit, args.seq_len, args.vocab_size)
    raise ValueError(name)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, descending=True, dim=dim).values
    range_ = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    range_ = range_.view(view)
    bound = 1 + range_ * zs
    cumsum = zs.cumsum(dim)
    is_gt = bound > cumsum
    k = is_gt.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(dim, k - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def entmax_bisect(logits: torch.Tensor, alpha: torch.Tensor | float = 1.5, dim: int = -1, n_iter: int = 18) -> torch.Tensor:
    if isinstance(alpha, float):
        alpha_t = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
    else:
        alpha_t = alpha.to(dtype=logits.dtype, device=logits.device)
    while alpha_t.dim() < logits.dim():
        alpha_t = alpha_t.unsqueeze(-1)
    alpha_t = alpha_t.clamp(1.05, 2.0)
    d = logits.size(dim)
    y = logits * (alpha_t - 1)
    y = y - y.max(dim=dim, keepdim=True).values
    tau_lo = y.min(dim=dim, keepdim=True).values - 1
    tau_hi = y.max(dim=dim, keepdim=True).values - (1.0 / d) ** (alpha_t - 1)
    inv = 1.0 / (alpha_t - 1)
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
    if method.startswith("topk_softmax_"):
        suffix = method.rsplit("_", 1)[-1]
        if suffix == "0125":
            return 0.125
        if suffix == "025":
            return 0.25
        if suffix == "05":
            return 0.5
    raise ValueError(method)


class NormalizedSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int, method: str):
        super().__init__()
        assert embed_dim % heads == 0
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.method = method
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.alpha_raw = nn.Parameter(torch.zeros(heads)) if method == "headwise_adaptive_entmax" else None
        self.last_nonzero_ratio = float("nan")
        self.last_alpha_mean = float("nan")

    def normalize(self, scores: torch.Tensor) -> torch.Tensor:
        method = self.method
        if method == "softmax":
            attn = torch.softmax(scores, dim=-1)
        elif method.startswith("topk_softmax"):
            ratio = parse_topk_ratio(method)
            k = max(1, int(math.ceil(scores.size(-1) * ratio)))
            values, indices = scores.topk(k, dim=-1)
            masked = torch.full_like(scores, -1e9)
            masked.scatter_(-1, indices, values)
            attn = torch.softmax(masked, dim=-1)
        elif method == "sparsemax":
            attn = sparsemax(scores, dim=-1)
        elif method == "entmax15":
            attn = entmax_bisect(scores, alpha=1.5, dim=-1)
        elif method == "headwise_adaptive_entmax":
            alpha = 1.05 + 0.90 * torch.sigmoid(self.alpha_raw)
            self.last_alpha_mean = float(alpha.detach().mean().cpu())
            attn = entmax_bisect(scores, alpha=alpha.view(1, self.heads, 1, 1), dim=-1)
        else:
            raise ValueError(method)
        self.last_nonzero_ratio = float((attn > 1e-7).float().mean().detach().cpu())
        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.normalize(scores)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out(out)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, method: str, dropout: float):
        super().__init__()
        self.attn = NormalizedSelfAttention(embed_dim, heads, method)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_kind: str,
        input_dim: int,
        classes: int,
        method: str,
        embed_dim: int,
        heads: int,
        layers: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()
        self.input_kind = input_kind
        if input_kind == "patch":
            self.input = nn.Linear(input_dim, embed_dim)
        elif input_kind == "token":
            self.input = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        else:
            raise ValueError(input_kind)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, heads, method, dropout) for _ in range(layers)])
        self.head = nn.Linear(embed_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])

    def attention_stats(self) -> tuple[float, float]:
        nonzero = []
        alpha = []
        for block in self.blocks:
            nonzero.append(block.attn.last_nonzero_ratio)
            alpha.append(block.attn.last_alpha_mean)
        nonzero_arr = np.asarray(nonzero, dtype=float)
        alpha_arr = np.asarray(alpha, dtype=float)
        nonzero_mean = float(np.nanmean(nonzero_arr)) if not np.isnan(nonzero_arr).all() else float("nan")
        alpha_mean = float(np.nanmean(alpha_arr)) if not np.isnan(alpha_arr).all() else float("nan")
        return nonzero_mean, alpha_mean


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: SequenceClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    crit = nn.CrossEntropyLoss()
    y_true, y_pred, losses, nonzeros = [], [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            pred = logits.argmax(dim=1)
            y_true.extend(yb.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
            losses.append(float(loss.detach().cpu()))
            nonzeros.append(model.attention_stats()[0])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "loss": float(np.mean(losses)),
        "inference_seconds": time.perf_counter() - start,
        "nonzero_ratio": float(np.nanmean(nonzeros)),
    }


def run_one(dataset_name: str, method: str, seed: int, args, device: torch.device) -> dict:
    set_seed(seed)
    bundle = load_dataset(dataset_name, seed, args)
    if bundle is None:
        return {"dataset": dataset_name, "method": method, "seed": seed, "status": "skipped_dataset_unavailable"}

    max_len = int(bundle.train.tensors[0].shape[1])
    train_loader = DataLoader(bundle.train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(bundle.test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SequenceClassifier(
        input_kind=bundle.input_kind,
        input_dim=bundle.input_dim,
        classes=bundle.classes,
        method=method,
        embed_dim=args.embed_dim,
        heads=args.heads,
        layers=args.layers,
        max_len=max_len,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    train_losses = []
    start = time.perf_counter()
    for _epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                return {"dataset": dataset_name, "method": method, "seed": seed, "status": "failed_nonfinite_loss"}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    metrics = evaluate(model, test_loader, device)
    nonzero, alpha = model.attention_stats()
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
    return {
        "dataset": dataset_name,
        "dataset_source": bundle.source,
        "method": method,
        "seed": seed,
        "status": "completed",
        "epochs": args.epochs,
        "train_n": bundle.train_n,
        "test_n": bundle.test_n,
        "input_kind": bundle.input_kind,
        "seq_len": max_len,
        "classes": bundle.classes,
        "embed_dim": args.embed_dim,
        "heads": args.heads,
        "layers": args.layers,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "train_seconds": train_seconds,
        "train_loss_last": train_losses[-1] if train_losses else float("nan"),
        "learned_alpha_mean": alpha,
        "attention_nonzero_ratio_last": nonzero,
        "peak_memory_mb": peak_memory_mb,
        **metrics,
    }


def write_summary(results: list[dict], run_id: str) -> None:
    df = pd.DataFrame(results)
    runs_path = PROCESSED / f"{run_id}_advanced_runs.csv"
    summary_path = PROCESSED / f"{run_id}_advanced_summary.csv"
    df.to_csv(runs_path, index=False)
    done = df[df["status"] == "completed"].copy()
    if done.empty:
        return
    summary = done.groupby(["dataset", "method"], as_index=False).agg(
        n=("accuracy", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        loss_mean=("loss", "mean"),
        train_seconds_mean=("train_seconds", "mean"),
        inference_seconds_mean=("inference_seconds", "mean"),
        nonzero_ratio_mean=("nonzero_ratio", "mean"),
        learned_alpha_mean=("learned_alpha_mean", "mean"),
        peak_memory_mb_mean=("peak_memory_mb", "mean"),
    )
    summary.to_csv(summary_path, index=False)

    test_rows = []
    baselines = ["softmax", "sparsemax", "entmax15", "headwise_adaptive_entmax"]
    for dataset, sub in done.groupby("dataset"):
        for method in sorted(sub["method"].unique()):
            for baseline in baselines:
                if method == baseline or baseline not in set(sub["method"]):
                    continue
                a = sub[sub["method"] == method][["seed", "accuracy", "macro_f1"]]
                b = sub[sub["method"] == baseline][["seed", "accuracy", "macro_f1"]]
                merged = pd.merge(a, b, on="seed", suffixes=("_method", "_baseline"))
                if len(merged) < 3:
                    continue
                acc_diff = merged["accuracy_method"] - merged["accuracy_baseline"]
                f1_diff = merged["macro_f1_method"] - merged["macro_f1_baseline"]
                if float(acc_diff.std(ddof=1)) == 0.0:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_test = stats.ttest_rel(merged["accuracy_method"], merged["accuracy_baseline"])
                    t_stat, p_val = float(t_test.statistic), float(t_test.pvalue)
                test_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "baseline": baseline,
                    "n": int(len(merged)),
                    "mean_delta_accuracy": float(acc_diff.mean()),
                    "mean_delta_macro_f1": float(f1_diff.mean()),
                    "paired_t_stat_accuracy": t_stat,
                    "paired_p_value_accuracy": p_val,
                })
    pd.DataFrame(test_rows).to_csv(STATS / f"{run_id}_advanced_paired_tests.csv", index=False)

    # Stable alias files for downstream plotting and manuscript scripts.
    df.to_csv(PROCESSED / "advanced_normalizer_runs_latest.csv", index=False)
    summary.to_csv(PROCESSED / "advanced_normalizer_summary_latest.csv", index=False)
    pd.DataFrame(test_rows).to_csv(STATS / "advanced_paired_tests_latest.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["fashion_mnist", "cifar10", "twenty_news", "synthetic_marker"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["softmax", "topk_softmax_0125", "topk_softmax_025", "topk_softmax_05", "sparsemax", "entmax15", "headwise_adaptive_entmax"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=3000)
    parser.add_argument("--text-train-limit", type=int, default=8000)
    parser.add_argument("--text-test-limit", type=int, default=3000)
    parser.add_argument("--synthetic-train-limit", type=int, default=12000)
    parser.add_argument("--synthetic-test-limit", type=int, default=3000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=20000)
    args = parser.parse_args()

    for path in [RAW, PROCESSED, STATS, RUN_LOGS, DATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = time.strftime("advanced_normalizer_%Y%m%d_%H%M%S")
    raw_path = RAW / f"{run_id}.jsonl"
    results = []
    started = time.strftime("%Y-%m-%d %H:%M:%S %z")
    with raw_path.open("w", encoding="utf-8") as f:
        for dataset in args.datasets:
            for method in args.methods:
                for seed in args.seeds:
                    t0 = time.perf_counter()
                    try:
                        row = run_one(dataset, method, seed, args, device)
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    row["wall_seconds_total"] = time.perf_counter() - t0
                    results.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(json.dumps(row))
    write_summary(results, run_id)
    manifest = {
        "run_id": run_id,
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "raw": str(raw_path),
        "advanced_runs": str(PROCESSED / f"{run_id}_advanced_runs.csv"),
        "advanced_summary": str(PROCESSED / f"{run_id}_advanced_summary.csv"),
        "advanced_paired_tests": str(STATS / f"{run_id}_advanced_paired_tests.csv"),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": vars(args),
    }
    (RUN_LOGS / f"{run_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
