#!/usr/bin/env python3
"""Fast, download-free contract tests for MCH-SND-002."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import extension_runner as ext  # noqa: E402


def image_bundle():
    x = torch.randn(12, 4, 8)
    y = torch.arange(12) % 3
    return ext.r1.Bundle(
        train=TensorDataset(x[:8], y[:8]), val=TensorDataset(x[8:], y[8:]),
        input_kind="patch", input_dim=8, classes=3, seq_len=4,
        source="unit:image", sizes={"train": 8, "val": 4},
    )


def text_bundle():
    x = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9], [2, 0, 0, 0]])
    y = torch.tensor([0, 1, 0, 1])
    mask = x.ne(0)
    return ext.r1.Bundle(
        train=TensorDataset(x, y), val=TensorDataset(x, y),
        input_kind="token", input_dim=16, classes=2, seq_len=4,
        source="unit:text", train_mask=mask, val_mask=mask,
        sizes={"train": 4, "val": 4},
    )


def test_architectures_and_schedule():
    bundle = image_bundle()
    plain = ext.InstrumentedClassifier(bundle, "switch_4", "plain_transformer", 16, 4, 2, 0.0, 6, 4)
    local = ext.InstrumentedClassifier(bundle, "switch_4", "local_hybrid", 16, 4, 2, 0.0, 6, 4)
    assert sum(p.numel() for p in local.parameters()) > sum(p.numel() for p in plain.parameters())
    plain.set_epoch(3)
    assert plain.active_method == "topk_softmax_025"
    plain.set_epoch(4)
    assert plain.active_method == "softmax"
    assert plain(bundle.val.tensors[0]).shape == (4, 3)
    assert local(bundle.val.tensors[0]).shape == (4, 3)


def test_mask_and_probe_are_finite():
    bundle = text_bundle()
    model = ext.InstrumentedClassifier(
        bundle, "topk_softmax_025", "local_hybrid", 16, 4, 2, 0.0, 3, None,
    )
    model.set_epoch(1)
    model.set_self_check(True)
    loader, n = ext.make_probe_loader(bundle, 256, torch.device("cpu"))
    assert n == 4
    metrics = ext.mechanism_probe(model, loader, torch.device("cpu"), 2)
    for key in (
        "probe_dense_entropy", "probe_boundary_margin", "probe_tail_mass",
        "probe_gradient_topk_share", "probe_loss",
    ):
        assert math.isfinite(metrics[key]), (key, metrics[key])
    assert 0.0 <= metrics["probe_tail_mass"] <= 1.0
    assert 0.0 <= metrics["probe_gradient_topk_share"] <= 1.0


def test_cosine_alpha_endpoints():
    bundle = image_bundle()
    model = ext.InstrumentedClassifier(bundle, "cosine_alpha", "plain_transformer", 16, 4, 1, 0.0, 30, None)
    model.set_epoch(1)
    assert abs(model.blocks[0].attn.cosine_alpha - 1.0) < 1e-12
    model.set_epoch(30)
    assert abs(model.blocks[0].attn.cosine_alpha - 1.5) < 1e-12


def invocation(**overrides):
    values = {
        "tag": "registered", "max_epochs": 30, "batch_size": 256,
        "experiment_id": "EXP-SND-102", "datasets": list(ext.DISCOVERY_DATASETS),
        "methods": list(ext.M2_METHODS), "seeds": list(range(5, 10)),
        "switch_map": __file__, "final_test_read": False, "test_freeze": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registered_phase_guards(tmp_path):
    assert ext.validate_registered_invocation(invocation()) is None
    try:
        ext.validate_registered_invocation(invocation(seeds=[5, 6]))
    except ValueError as exc:
        assert "M2 grid" in str(exc)
    else:
        raise AssertionError("partial M2 grid was accepted")

    switch_map = tmp_path / "m3_switch_map.json"
    switch_map.write_text("{}\n", encoding="utf-8")
    freeze = tmp_path / "m3_test_freeze.json"
    methods = ["softmax", "topk_softmax_025", "mechanistic_switch", "median_switch"]
    freeze.write_text(json.dumps({
        "status": "M3_VALIDATION_FROZEN_OFFICIAL_TEST_AUTHORIZED",
        "switch_map_sha256": ext.sha256_file(switch_map),
        "methods": methods,
        "strongest_simple_schedule": "median_switch",
    }) + "\n", encoding="utf-8")
    args = invocation(
        experiment_id="EXP-SND-103", datasets=list(ext.CONFIRMATION_DATASETS),
        methods=methods, seeds=list(range(10, 20)), switch_map=str(switch_map),
        final_test_read=True, test_freeze=str(freeze),
    )
    assert ext.validate_registered_invocation(args)["strongest_simple_schedule"] == "median_switch"
    freeze.write_text(
        freeze.read_text(encoding="utf-8").replace(ext.sha256_file(switch_map), "0" * 64),
        encoding="utf-8",
    )
    try:
        ext.validate_registered_invocation(args)
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("mismatched official-test switch map was accepted")


def main():
    test_architectures_and_schedule()
    test_mask_and_probe_are_finite()
    test_cosine_alpha_endpoints()
    print("PASS test_extension_runner")


if __name__ == "__main__":
    main()
