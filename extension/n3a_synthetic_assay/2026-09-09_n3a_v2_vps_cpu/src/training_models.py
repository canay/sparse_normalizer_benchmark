"""Deterministic synthetic routing data and N3 router models.

The routing features never contain the scalar value path.  ``z`` is an
explicit binary relevance indicator, not a signed coefficient; this keeps the
classification target representable by non-negative attention weights.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class SplitArrays:
    features: np.ndarray
    values: np.ndarray
    labels: np.ndarray
    relevance: np.ndarray


def generate_split(master_seed: int, r: int, split: str, size: int, n: int = 32) -> SplitArrays:
    if not 0 < r <= n:
        raise ValueError("r must lie in [1,n]")
    # Independent, named PCG64 substreams make draw order changes detectable.
    support_rng = np.random.Generator(np.random.PCG64(stable_seed(master_seed, r, split, "support")))
    noise_rng = np.random.Generator(np.random.PCG64(stable_seed(master_seed, r, split, "marker_noise")))
    value_rng = np.random.Generator(np.random.PCG64(stable_seed(master_seed, r, split, "values")))
    nuisance_rng = np.random.Generator(np.random.PCG64(stable_seed(master_seed, r, split, "nuisance")))

    # The r smallest independent uniforms form a uniform subset without replacement.
    support_u = support_rng.random((size, n))
    support = np.argpartition(support_u, r - 1, axis=1)[:, :r]
    relevance = np.zeros((size, n), dtype=np.float32)
    relevance[np.arange(size)[:, None], support] = 1.0

    marker = relevance + noise_rng.standard_normal((size, n)).astype(np.float32)
    nuisance = nuisance_rng.standard_normal((size, n, 6)).astype(np.float32)
    features = np.concatenate((marker[..., None], nuisance), axis=-1)
    values = value_rng.standard_normal((size, n)).astype(np.float32)
    latent = (relevance * values).sum(axis=1) / math.sqrt(r)
    labels = (latent >= 0).astype(np.float32)
    return SplitArrays(features=features, values=values, labels=labels, relevance=relevance)


def make_batch_schedule(master_seed: int, r: int, train_size: int, batch_size: int, updates: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(stable_seed(master_seed, r, "batch_schedule")))
    chunks: list[np.ndarray] = []
    needed = batch_size * updates
    have = 0
    while have < needed:
        p = rng.permutation(train_size)
        chunks.append(p)
        have += len(p)
    return np.concatenate(chunks)[:needed].reshape(updates, batch_size).astype(np.int64)


class QueryPoolRouter(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bnd,d->bn", self.ff(x), self.query) / math.sqrt(self.query.numel())


class TransformerRouter(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.input = nn.Linear(7, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.input(x))
        return torch.einsum("bnd,d->bn", h, self.query) / math.sqrt(self.query.numel())


class RoutingClassifier(nn.Module):
    def __init__(self, architecture: str):
        super().__init__()
        if architecture == "query":
            self.router = QueryPoolRouter()
        elif architecture == "transformer":
            self.router = TransformerRouter()
        else:
            raise ValueError(architecture)
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def scores(self, features: torch.Tensor) -> torch.Tensor:
        return self.router(features)

    def logits(self, features: torch.Tensor, values: torch.Tensor, r: int, mode: str, operator) -> torch.Tensor:
        w = operator(self.scores(features), mode)
        gamma = F.softplus(self.raw_gamma).to(w.dtype)
        return gamma * math.sqrt(r) * (w * values.to(w.dtype)).sum(-1) + self.bias.to(w.dtype)


def build_model(architecture: str, init_seed: int) -> RoutingClassifier:
    # fork_rng prevents construction from perturbing the controller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        return RoutingClassifier(architecture)
