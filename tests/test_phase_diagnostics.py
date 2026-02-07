import numpy as np
import yaml
from src.phase_diagnostics import diagnose_phase


def _load_thresholds():
    with open('config/thresholds.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_stage_1():
    thresholds = _load_thresholds()
    metrics = {
        'h_negative': 50, 'h_below_5': 100, 'capture_ratio_pv': 0.92,
        'sr': 0.005, 'far_structural': np.nan, 'far_observed': np.nan,
        'h_regime_c': 7000, 'h_regime_d_tail': 200,
        'days_spread_above_50': 30, 'vre_share': 0.15, 'ir': 0.40,
        'is_outlier': False,
    }
    result = diagnose_phase(metrics, thresholds)
    assert result['phase'] == 'stage_1'


def test_stage_2():
    thresholds = _load_thresholds()
    metrics = {
        'h_negative': 500, 'h_below_5': 800, 'capture_ratio_pv': 0.65,
        'sr': 0.04, 'far_structural': 0.25, 'far_observed': 0.20,
        'h_regime_c': 4500, 'h_regime_d_tail': 500,
        'days_spread_above_50': 250, 'vre_share': 0.35, 'ir': 0.65,
        'is_outlier': False,
    }
    result = diagnose_phase(metrics, thresholds)
    assert result['phase'] == 'stage_2'
