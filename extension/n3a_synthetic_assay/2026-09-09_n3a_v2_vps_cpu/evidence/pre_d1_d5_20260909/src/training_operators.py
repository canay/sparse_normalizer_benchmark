"""N3 fixed-tail-charge and same-forward backward-access assay operators.

These are experimental controls, not proposed production attention methods.
Top-k membership is treated as locally constant, as in ordinary hard routing.
"""
from __future__ import annotations

import torch


TOP_K = 8
TAIL_LEAK_MASS = 1.0 / (TOP_K + 1.0)
MASS_TOLERANCE = 1e-10
TOP_SCORE_RANGE_CEILING = 700.0
NORM_ZERO_TOLERANCE = 1e-14


class InvalidControlError(RuntimeError):
    """A prospective numerical invariant of an N3 assay control failed."""


def entropy(p: torch.Tensor) -> torch.Tensor:
    """Descriptive forward diagnostic; never an N3 selection endpoint."""
    return -(p * torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))).sum(-1)


def covariance_product(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return p * (x - (p * x).sum(-1, keepdim=True))


def split_weights(scores: torch.Tensor):
    """Return top-k/tail softmax rows, masks, and within-block score ranges."""
    if scores.shape[-1] != 32:
        raise ValueError("N3 configuration requires exactly 32 positions")
    if not torch.isfinite(scores).all():
        raise RuntimeError("nonfinite N3 input scores")
    s = scores.double() - scores.double().mean(-1, keepdim=True).detach()
    order = torch.argsort(s, dim=-1, descending=True, stable=True)
    top_ix = order[..., :TOP_K]
    tail_ix = order[..., TOP_K:]
    top_scores = s.gather(-1, top_ix)
    tail_scores = s.gather(-1, tail_ix)
    top_range = top_scores.max(-1).values - top_scores.min(-1).values
    tail_range = tail_scores.max(-1).values - tail_scores.min(-1).values
    if torch.any(top_range > TOP_SCORE_RANGE_CEILING):
        raise RuntimeError("NUMERICAL_ENVELOPE: top-block score range")
    q = torch.zeros_like(s).scatter(-1, top_ix, torch.softmax(top_scores, -1))
    t = torch.zeros_like(s).scatter(-1, tail_ix, torch.softmax(tail_scores, -1))
    top_mask = torch.zeros_like(s, dtype=torch.bool).scatter(-1, top_ix, True)
    tail_mask = ~top_mask
    return q, t, top_mask, tail_mask, tail_ix, top_range, tail_range


def validate_sparse(q: torch.Tensor, top_mask: torch.Tensor) -> None:
    finite = torch.isfinite(q).all()
    sums = torch.all((q.sum(-1) - 1).abs() <= MASS_TOLERANCE)
    support = torch.all((q > 0) == top_mask)
    if not bool(finite and sums and support):
        raise InvalidControlError("N3 sparse mass/support invariant")


def validate_charged(
    q: torch.Tensor,
    charged: torch.Tensor,
    top_mask: torch.Tensor,
    tail_mask: torch.Tensor,
    require_tail_charge: bool,
) -> None:
    finite = torch.isfinite(charged).all()
    sums = torch.all((charged.sum(-1) - 1).abs() <= MASS_TOLERANCE)
    top_mass = (charged * top_mask).sum(-1)
    expected_top = 1.0 - TAIL_LEAK_MASS if require_tail_charge else 1.0
    top_ok = torch.all((top_mass - expected_top).abs() <= MASS_TOLERANCE)
    tail_mass = (charged * tail_mask).sum(-1)
    expected_tail = TAIL_LEAK_MASS if require_tail_charge else 0.0
    tail_ok = torch.all((tail_mass - expected_tail).abs() <= MASS_TOLERANCE)
    if not bool(finite and sums and top_ok and tail_ok):
        raise InvalidControlError("N3 charged mass/nonfinite invariant")


def component_vjps(q: torch.Tensor, t: torch.Tensor, upstream: torch.Tensor):
    sparse_vjp = covariance_product(q, upstream)
    tail_vjp = covariance_product(t, upstream)
    leaky_vjp = (1.0 - TAIL_LEAK_MASS) * sparse_vjp + TAIL_LEAK_MASS * tail_vjp
    return sparse_vjp, tail_vjp, leaky_vjp


def shuffled_tail_vjp(tail_vjp: torch.Tensor, tail_ix: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Seeded row-wise random permutation, not a guaranteed orthogonal null.

    A private generator avoids modifying the model/data RNG stream. The caller
    supplies a prospectively fixed key for each paired checkpoint/update.
    """
    ranked = tail_vjp.gather(-1, tail_ix)
    generator = torch.Generator(device=tail_vjp.device).manual_seed(seed)
    order = torch.rand(ranked.shape, device=ranked.device, generator=generator,
                       dtype=torch.float64).argsort(dim=-1, stable=True)
    permuted = ranked.gather(-1, order)
    return torch.zeros_like(tail_vjp).scatter(-1, tail_ix, permuted)


def norm_match(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference_norm = reference.norm(dim=-1, keepdim=True)
    candidate_norm = candidate.norm(dim=-1, keepdim=True)
    reference_active = reference_norm > NORM_ZERO_TOLERANCE
    invalid = reference_active & (candidate_norm <= torch.finfo(candidate_norm.dtype).tiny)
    if torch.any(invalid):
        raise InvalidControlError("N3 backward norm denominator")
    scale = torch.where(
            reference_active,
        reference_norm / candidate_norm.clamp_min(torch.finfo(candidate_norm.dtype).tiny),
        torch.zeros_like(reference_norm),
    )
    matched = candidate * scale
    if not torch.isfinite(matched).all():
        raise InvalidControlError("N3 nonfinite backward VJP")
    return matched, scale


class SameForwardAlternateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, shuffled: bool, null_seed: int) -> torch.Tensor:
        q, t, top_mask, tail_mask, tail_ix, _, _ = split_weights(scores)
        validate_sparse(q, top_mask)
        charged = (1.0 - TAIL_LEAK_MASS) * q + TAIL_LEAK_MASS * t
        validate_charged(q, charged, top_mask, tail_mask, require_tail_charge=True)
        ctx.save_for_backward(q, t, tail_ix)
        ctx.shuffled = bool(shuffled)
        ctx.input_dtype = scores.dtype
        ctx.null_seed = int(null_seed)
        return q

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        q, t, tail_ix = ctx.saved_tensors
        g = grad.double()
        sparse_vjp, tail_vjp, _ = component_vjps(q, t, g)
        if ctx.shuffled:
            tail_vjp = shuffled_tail_vjp(tail_vjp, tail_ix, ctx.null_seed)
        candidate = (1.0 - TAIL_LEAK_MASS) * sparse_vjp + TAIL_LEAK_MASS * tail_vjp
        substituted, _ = norm_match(sparse_vjp, candidate)
        return substituted.to(ctx.input_dtype), None, None


def backward_diagnostics(scores: torch.Tensor, upstream: torch.Tensor, null_seed: int = 0) -> dict[str, torch.Tensor]:
    """Per-row realized intervention dose at a common hard-forward state."""
    q, t, _, _, tail_ix, top_range, tail_range = split_weights(scores)
    sparse_vjp, tail_vjp, leaky_vjp = component_vjps(q, t, upstream.double())
    null_tail = shuffled_tail_vjp(tail_vjp, tail_ix, null_seed)
    shuffled = (1.0 - TAIL_LEAK_MASS) * sparse_vjp + TAIL_LEAK_MASS * null_tail
    matched, scale = norm_match(sparse_vjp, leaky_vjp)
    matched_null, null_scale = norm_match(sparse_vjp, shuffled)
    tiny = torch.finfo(torch.float64).tiny
    sparse_norm = sparse_vjp.norm(dim=-1)
    tail_norm = tail_vjp.norm(dim=-1)
    matched_norm = matched.norm(dim=-1)
    cosine = (matched * sparse_vjp).sum(-1) / (matched_norm * sparse_norm).clamp_min(tiny)
    return {
        "rho_tail_to_sparse": tail_norm / sparse_norm.clamp_min(tiny),
        "cosine_matched_to_sparse": cosine,
        "on_support_attenuation": scale.squeeze(-1) * (1.0 - TAIL_LEAK_MASS),
        "null_on_support_attenuation": null_scale.squeeze(-1) * (1.0 - TAIL_LEAK_MASS),
        "top_score_range": top_range,
        "tail_score_range": tail_range,
        "matched_norm": matched_norm,
        "sparse_norm": sparse_norm,
        "null_norm": matched_null.norm(dim=-1),
        "cosine_tail_to_null_tail": (tail_vjp * null_tail).sum(-1) / (tail_norm * null_tail.norm(dim=-1)).clamp_min(tiny),
    }


def weights(scores: torch.Tensor, mode: str, null_seed: int = 0) -> torch.Tensor:
    centered = scores.double() - scores.double().mean(-1, keepdim=True).detach()
    if mode == "D":
        split_weights(scores)  # identical numerical-domain guard in both main arms
        dense = torch.softmax(centered, -1)
        if not torch.isfinite(dense).all() or not torch.all((dense.sum(-1) - 1).abs() <= MASS_TOLERANCE):
            raise RuntimeError("invalid dense baseline row")
        return dense
    if mode == "G":
        return SameForwardAlternateBackward.apply(scores, False, null_seed)
    if mode == "N":
        return SameForwardAlternateBackward.apply(scores, True, null_seed)
    q, t, top_mask, tail_mask, _, _, _ = split_weights(scores)
    validate_sparse(q, top_mask)
    if mode == "S":
        return q
    if mode == "L":
        charged = (1.0 - TAIL_LEAK_MASS) * q + TAIL_LEAK_MASS * t
        validate_charged(q, charged, top_mask, tail_mask, require_tail_charge=True)
        return charged
    if mode == "P":
        uniform_top = top_mask.to(q.dtype) / TOP_K
        charged = (1.0 - TAIL_LEAK_MASS) * q + TAIL_LEAK_MASS * uniform_top
        validate_charged(q, charged, top_mask, tail_mask, require_tail_charge=False)
        return charged
    raise ValueError(mode)
