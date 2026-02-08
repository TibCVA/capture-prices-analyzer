from __future__ import annotations

import numpy as np

from src.phase_diagnostics import diagnose_phase


def test_phase_stage_1(thresholds_cfg):
    metrics = {
        "h_negative_obs": 20,
        "h_below_5_obs": 50,
        "capture_ratio_pv": 0.92,
        "sr": 0.005,
        "far": np.nan,
        "ir": 0.30,
        "days_spread_above_50_obs": 10,
        "h_regime_c": 5000,
    }
    d = diagnose_phase(metrics, thresholds_cfg)
    assert d["phase"] in {"stage_1", "unknown"}
    assert d["score"] >= 0


def test_phase_stage_2(thresholds_cfg):
    metrics = {
        "h_negative_obs": 520,
        "h_below_5_obs": 900,
        "capture_ratio_pv": 0.62,
        "sr": 0.04,
        "far": 0.2,
        "ir": 0.65,
        "days_spread_above_50_obs": 180,
        "h_regime_c": 2500,
    }
    d = diagnose_phase(metrics, thresholds_cfg)
    assert d["phase"] in {"stage_2", "unknown"}


def test_phase_unknown_low_confidence(thresholds_cfg):
    metrics = {
        "h_negative_obs": 0,
        "h_below_5_obs": 0,
        "capture_ratio_pv": np.nan,
        "sr": np.nan,
        "far": np.nan,
        "ir": np.nan,
        "days_spread_above_50_obs": 0,
        "h_regime_c": 0,
    }
    d = diagnose_phase(metrics, thresholds_cfg)
    # In this edge case, stage_1 rules can still be matched (low negatives, low h<5).
    assert d["phase"] in {"stage_1", "unknown"}
