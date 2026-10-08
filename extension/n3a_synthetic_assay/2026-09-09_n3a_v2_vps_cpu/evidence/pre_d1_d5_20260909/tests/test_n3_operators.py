from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from training_operators import (  # noqa: E402
    InvalidControlError,
    TAIL_LEAK_MASS,
    TOP_K,
    backward_diagnostics,
    component_vjps,
    covariance_product,
    norm_match,
    split_weights,
    weights,
)


def row_vjp(scores: torch.Tensor, mode: str, upstream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = scores.detach().clone().requires_grad_(True)
    out = weights(x, mode)
    (out * upstream).sum().backward()
    return out.detach(), x.grad.detach()


def test_forward_mass_support_and_controls() -> None:
    torch.manual_seed(301)
    scores = torch.randn(13, 32, dtype=torch.float64)
    sparse = weights(scores, "S")
    leaky = weights(scores, "L")
    selected_null = weights(scores, "P")
    top = sparse > 0
    assert torch.all((sparse > 0).sum(-1) == TOP_K)
    assert torch.all((leaky > 0).sum(-1) == 32)
    assert torch.all((selected_null > 0).sum(-1) == TOP_K)
    assert torch.equal(weights(scores, "G"), sparse)
    assert torch.equal(weights(scores, "N"), sparse)
    assert torch.allclose((leaky * top).sum(-1), torch.full((13,), 1 - TAIL_LEAK_MASS, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose((leaky * ~top).sum(-1), torch.full((13,), TAIL_LEAK_MASS, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(selected_null.sum(-1), torch.ones(13, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.all(selected_null[~top] == 0)


def test_analytic_vjps_and_same_forward_directions() -> None:
    torch.manual_seed(302)
    scores = torch.randn(11, 32, dtype=torch.float64)
    upstream = torch.randn(11, 32, dtype=torch.float64)
    q, t, *_ = split_weights(scores)
    out_s, grad_s = row_vjp(scores, "S", upstream)
    out_l, grad_l = row_vjp(scores, "L", upstream)
    out_g, grad_g = row_vjp(scores, "G", upstream)
    out_n, grad_n = row_vjp(scores, "N", upstream)
    sparse_expected, _, leaky_expected = component_vjps(q, t, upstream)
    assert torch.allclose(grad_s, sparse_expected, atol=1e-11, rtol=1e-10)
    assert torch.allclose(grad_l, leaky_expected, atol=1e-11, rtol=1e-10)
    assert torch.equal(out_s, out_g)
    assert torch.equal(out_s, out_n)
    assert torch.allclose(grad_s.norm(dim=-1), grad_g.norm(dim=-1), atol=1e-11, rtol=1e-10)
    assert torch.allclose(grad_s.norm(dim=-1), grad_n.norm(dim=-1), atol=1e-11, rtol=1e-10)
    assert torch.allclose(
        grad_g / grad_g.norm(dim=-1, keepdim=True),
        grad_l / grad_l.norm(dim=-1, keepdim=True),
        atol=1e-11,
        rtol=1e-10,
    )
    tail = out_s == 0
    assert torch.all(grad_s[tail] == 0)
    assert torch.any(grad_g[tail] != 0)
    assert torch.allclose(grad_g[out_s > 0], grad_n[out_s > 0], atol=1e-11, rtol=1e-10)


def test_dose_matched_null_and_diagnostics() -> None:
    torch.manual_seed(303)
    scores = torch.randn(7, 32, dtype=torch.float64)
    upstream = torch.randn(7, 32, dtype=torch.float64)
    sparse, grad_s = row_vjp(scores, "S", upstream)
    _, grad_g = row_vjp(scores, "G", upstream)
    _, grad_n = row_vjp(scores, "N", upstream)
    dose = backward_diagnostics(scores, upstream)
    assert torch.allclose(grad_g.norm(dim=-1), grad_s.norm(dim=-1), atol=1e-11, rtol=1e-10)
    assert torch.allclose(grad_n.norm(dim=-1), grad_s.norm(dim=-1), atol=1e-11, rtol=1e-10)
    assert torch.allclose(dose["matched_norm"], dose["sparse_norm"], atol=1e-11, rtol=1e-10)
    assert torch.allclose(dose["null_norm"], dose["sparse_norm"], atol=1e-11, rtol=1e-10)
    top = sparse > 0
    assert torch.allclose(grad_g[top], grad_n[top], atol=1e-11, rtol=1e-10)
    for value in dose.values():
        assert torch.isfinite(value).all()


def test_gradcheck_away_from_selection_boundary() -> None:
    base = torch.linspace(3, -3, 32, dtype=torch.float64).repeat(2, 1)
    base = base + torch.tensor([[0.0], [0.01]], dtype=torch.float64)
    for mode in ("D", "S", "L", "P"):
        x = base.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(lambda z: weights(z, mode), (x,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_zero_sparse_branch_and_row_independence() -> None:
    torch.manual_seed(304)
    scores = torch.randn(4, 32, dtype=torch.float64)
    sparse = weights(scores, "S")
    upstream = torch.where(sparse > 0, torch.ones_like(sparse), torch.randn_like(sparse))
    _, grad_s = row_vjp(scores, "S", upstream)
    _, grad_g = row_vjp(scores, "G", upstream)
    assert torch.all(grad_s.abs() < 1e-14)
    assert torch.all(grad_g == 0)

    random_upstream = torch.randn_like(scores)
    _, grad_before = row_vjp(scores, "G", random_upstream)
    changed = scores.clone()
    changed[0] += torch.linspace(-0.3, 0.3, 32, dtype=torch.float64)
    _, grad_after = row_vjp(changed, "G", random_upstream)
    assert torch.equal(grad_before[1:], grad_after[1:])


def test_ties_repeatability_and_underflow_policy() -> None:
    tied = torch.zeros(2, 32, dtype=torch.float64)
    upstream = torch.arange(32, dtype=torch.float64).repeat(2, 1)
    sparse_1, grad_1 = row_vjp(tied, "G", upstream)
    sparse_2, grad_2 = row_vjp(tied, "G", upstream)
    expected = torch.zeros_like(sparse_1, dtype=torch.bool)
    expected[..., :TOP_K] = True
    assert torch.equal(sparse_1 > 0, expected)
    assert torch.equal(sparse_1, sparse_2)
    assert torch.equal(grad_1, grad_2)

    top = torch.arange(0, -8, -1, dtype=torch.float64)
    tail = torch.linspace(-1000, -2000, 24, dtype=torch.float64)
    underflow_row = torch.cat((top, tail))[None, :]
    leaky = weights(underflow_row, "L")
    assert torch.isfinite(leaky).all()
    assert torch.allclose(leaky[..., TOP_K:].sum(-1), torch.tensor([TAIL_LEAK_MASS], dtype=torch.float64), atol=1e-12, rtol=0)
    assert int((leaky[..., TOP_K:] > 0).sum()) < 24


def test_guard_classification() -> None:
    with torch.no_grad():
        try:
            weights(torch.zeros(2, 31, dtype=torch.float64), "S")
        except ValueError as exc:
            assert not isinstance(exc, InvalidControlError)
        else:
            raise AssertionError("shape error did not fire")

        bad = torch.zeros(2, 32, dtype=torch.float64)
        bad[0, 0] = float("nan")
        try:
            weights(bad, "L")
        except RuntimeError as exc:
            assert not isinstance(exc, InvalidControlError)
        else:
            raise AssertionError("nonfinite input error did not fire")

        top_range = torch.cat((torch.arange(0, -808, -101, dtype=torch.float64), torch.full((24,), -2000.0)))[None, :]
        try:
            weights(top_range, "S")
        except RuntimeError as exc:
            assert not isinstance(exc, InvalidControlError)
            assert "NUMERICAL_ENVELOPE" in str(exc)
        else:
            raise AssertionError("top-score-range guard did not fire")

        try:
            norm_match(torch.ones(1, 32, dtype=torch.float64), torch.zeros(1, 32, dtype=torch.float64))
        except InvalidControlError:
            pass
        else:
            raise AssertionError("denominator guard did not fire")

        try:
            candidate = torch.zeros(1, 32, dtype=torch.float64)
            candidate[0, 0] = float("nan")
            norm_match(torch.ones(1, 32, dtype=torch.float64), candidate)
        except InvalidControlError:
            pass
        else:
            raise AssertionError("nonfinite matched-VJP guard did not fire")


if __name__ == "__main__":
    test_forward_mass_support_and_controls()
    test_analytic_vjps_and_same_forward_directions()
    test_dose_matched_null_and_diagnostics()
    test_gradcheck_away_from_selection_boundary()
    test_zero_sparse_branch_and_row_independence()
    test_ties_repeatability_and_underflow_policy()
    test_guard_classification()
    print("PASS: 7 N3 operator test groups")
