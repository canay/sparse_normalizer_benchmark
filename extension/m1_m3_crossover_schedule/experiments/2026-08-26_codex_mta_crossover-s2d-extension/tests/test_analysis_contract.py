#!/usr/bin/env python3
"""Fast integrity and endpoint tests for the frozen MCH-SND-002 analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import analyze_m1 as m1  # noqa: E402
import analyze_m2 as m2  # noqa: E402
import analyze_m3 as m3  # noqa: E402
import build_m3_switch_map as builder  # noqa: E402


def test_time_to_target_and_transition_window():
    values = [0.1, 0.2, 0.4, 0.3, 0.35] + [0.35] * 25
    log = [{"epoch": e, "val_accuracy": v} for e, v in enumerate(values, 1)]
    assert m2.time_to_target(log, 0.35) == 3
    assert m2.time_to_target(log, 0.5) == 31
    assert abs(m2.transition_instability(log, 3) - 0.2) < 1e-12


def test_holm_is_monotone_in_sorted_order():
    raw = [0.04, 0.001, 0.02]
    adjusted = m3.holm(raw)
    order = np.argsort(raw)
    sorted_adjusted = [adjusted[i] for i in order]
    assert sorted_adjusted == sorted(sorted_adjusted)
    assert all(a >= p for a, p in zip(adjusted, raw))


def test_m1_manifest_bindings_fail_closed(tmp_path):
    result = tmp_path / "m1_result.json"
    switch = tmp_path / "m1_switch_map.json"
    result.write_text("{}\n", encoding="utf-8")
    switch.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "m1_analysis_manifest.json"
    manifest.write_text(json.dumps({"outputs": {
        result.name: m2.sha256(result), switch.name: m2.sha256(switch),
    }}) + "\n", encoding="utf-8")
    m2.require_m1_manifest(manifest, result, switch)
    switch.write_text('{"changed": true}\n', encoding="utf-8")
    try:
        m2.require_m1_manifest(manifest, result, switch)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered M1 switch map was accepted")


def test_m1_feature_binding_and_confirmation_epoch_boundary(tmp_path):
    features = tmp_path / "m1_cell_features.csv"
    features.write_text("dataset,target_switch_epoch\nmnist,4\n", encoding="utf-8")
    manifest = tmp_path / "m1_analysis_manifest.json"
    manifest.write_text(json.dumps({"outputs": {
        features.name: builder.sha256(features),
    }}) + "\n", encoding="utf-8")
    builder.require_m1_feature_manifest(manifest, features)

    rows = []
    for offset in (0.0, 0.2):
        rows.append({"epoch_log": [
            {"metric": 1.0 + offset}, {"metric": 2.0 + offset},
            {"metric": 5.0 + offset}, {"metric": 10_000.0},
        ]})
    level, slope = builder.level_slope(rows, "metric")
    assert abs(level - 5.1) < 1e-12
    assert abs(slope - 2.0) < 1e-12


def test_m3_baseline_validation_binds_raw_files(tmp_path):
    root = tmp_path / "m3"
    (root / "raw").mkdir(parents=True)
    raw = root / "raw" / "segment.jsonl"
    raw.write_text('{"status":"completed"}\n', encoding="utf-8")
    report = tmp_path / "m3_baseline_validation.json"
    report.write_text(json.dumps({
        "status": "PASS", "phase": "m3_baseline",
        "raw_files": {raw.name: builder.sha256(raw)},
    }) + "\n", encoding="utf-8")
    builder.require_baseline_validation(report, root)
    raw.write_text('{"status":"completed","changed":true}\n', encoding="utf-8")
    try:
        builder.require_baseline_validation(report, root)
    except RuntimeError as exc:
        assert "raw hashes" in str(exc)
    else:
        raise AssertionError("tampered M3 baseline raw file was accepted")


def test_ridge_prediction_is_integer_clipped():
    features = ("x",)
    train = [
        {"x": 0.0, "target_switch_epoch": 3},
        {"x": 1.0, "target_switch_epoch": 31},
    ]
    pred = m1.ridge_predict(train, [{"x": -100.0}, {"x": 100.0}], features)
    assert pred.dtype.kind in "iu"
    assert pred.tolist() == [3, 30]
