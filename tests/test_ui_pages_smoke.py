from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.constants import (
    COL_BESS_CHARGE,
    COL_BESS_DISCHARGE,
    COL_BESS_SOC,
    COL_FLEX_EFFECTIVE,
    COL_LOAD,
    COL_MUST_RUN,
    COL_NET_POSITION,
    COL_NRL,
    COL_PRICE_DA,
    COL_PRICE_SYNTH,
    COL_PRICE_USED,
    COL_PSH_PUMP,
    COL_REGIME,
    COL_REGIME_COHERENT,
    COL_SINK_NON_BESS,
    COL_SOLAR,
    COL_SURPLUS,
    COL_SURPLUS_UNABS,
    COL_TCA,
    COL_VRE,
    COL_WIND_OFF,
    COL_WIND_ON,
)


def _synthetic_state() -> dict:
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")

    df = pd.DataFrame(
        {
            COL_LOAD: 50000.0,
            COL_SOLAR: 7000.0,
            COL_WIND_ON: 8000.0,
            COL_WIND_OFF: 1000.0,
            COL_VRE: 16000.0,
            COL_MUST_RUN: 20000.0,
            COL_NRL: 14000.0,
            COL_SURPLUS: 0.0,
            COL_SINK_NON_BESS: 1000.0,
            COL_BESS_CHARGE: 0.0,
            COL_BESS_DISCHARGE: 0.0,
            COL_BESS_SOC: 1000.0,
            COL_FLEX_EFFECTIVE: 1000.0,
            COL_SURPLUS_UNABS: 0.0,
            COL_REGIME: "C",
            COL_PRICE_DA: 70.0,
            COL_TCA: 75.0,
            COL_PRICE_SYNTH: 75.0,
            COL_PRICE_USED: 70.0,
            COL_REGIME_COHERENT: True,
            COL_PSH_PUMP: 0.0,
            COL_NET_POSITION: 0.0,
        },
        index=idx,
    )

    metrics = {
        "country": "FR",
        "year": 2024,
        "baseload_price_used": 70.0,
        "peakload_price_used": 72.0,
        "offpeak_price_used": 68.0,
        "price_used_p05": 60.0,
        "price_used_p25": 66.0,
        "price_used_median": 70.0,
        "price_used_p75": 74.0,
        "price_used_p95": 80.0,
        "price_used_stddev": 5.0,
        "baseload_price_obs": 70.0,
        "h_negative_obs": 0,
        "h_below_5_obs": 0,
        "h_above_100_obs": 0,
        "h_above_200_obs": 0,
        "days_spread_above_50_obs": 0,
        "avg_daily_spread_obs": 8.0,
        "max_daily_spread_obs": 12.0,
        "capture_rate_pv": 68.0,
        "capture_rate_wind": 69.0,
        "capture_ratio_pv": 0.97,
        "capture_ratio_wind": 0.99,
        "h_regime_a": 0,
        "h_regime_b": 0,
        "h_regime_c": 72,
        "h_regime_d": 0,
        "sr": 0.0,
        "sr_hours": 0.0,
        "far": float("nan"),
        "ir": 0.35,
        "ttl": 80.0,
        "pv_penetration_pct_gen": 12.0,
        "wind_penetration_pct_gen": 18.0,
        "vre_penetration_pct_gen": 30.0,
        "total_generation_twh": 1.0,
        "total_load_twh": 1.1,
        "total_vre_twh": 0.3,
        "total_surplus_twh": 0.0,
        "total_surplus_unabs_twh": 0.0,
        "bess_cycles_est": 0.0,
        "bess_charge_twh": 0.0,
        "bess_discharge_twh": 0.0,
        "data_completeness": 1.0,
        "regime_coherence": 0.9,
        "is_outlier": False,
    }

    return {
        "data_loaded": True,
        "raw": {("FR", 2024): df.copy()},
        "processed": {("FR", 2024, "observed", "observed", "observed"): df.copy()},
        "metrics": {("FR", 2024, "observed"): metrics},
        "diagnostics": {("FR", 2024): {"phase": "stage_2", "score": 4, "confidence": 0.8, "matched_rules": [], "alerts": []}},
        "countries_selected": ["FR"],
        "year_range": (2024, 2024),
        "exclude_2022": True,
        "must_run_mode": "observed",
        "flex_model_mode": "observed",
        "price_mode": "observed",
        "scenario_price_mode": "synthetic",
        "commodities": {
            "gas_daily": pd.Series([35.0], index=pd.date_range("2024-01-01", periods=1, freq="D")),
            "co2_daily": pd.Series([80.0], index=pd.date_range("2024-01-01", periods=1, freq="D")),
            "coal_daily": None,
            "bess_capacity": None,
        },
        "countries_cfg": {
            "FR": {
                "name": "France",
                "entsoe_code": "FR",
                "timezone": "Europe/Paris",
                "must_run": {"mode": "observed", "observed_components": ["nuclear"], "floor_params": {}},
                "flex": {
                    "model_mode": "observed",
                    "historical_proxy": {"use_psh_pumping": True, "use_positive_net_position": True},
                    "capacity_defaults": {
                        "export_max_gw": 10.0,
                        "psh_pump_gw": 1.0,
                        "dsm_gw": 1.0,
                        "bess_power_gw_default": 0.5,
                        "bess_energy_gwh_default": 2.0,
                    },
                },
                "thermal": {"marginal_tech": "CCGT"},
            }
        },
        "thresholds": {
            "model_params": {"regime_d": {"method": "quantile", "positive_nrl_quantile": 0.90, "absolute_nrl_mw": None}},
            "coherence_params": {
                "price_low_threshold": 5.0,
                "b_price_min": -10.0,
                "b_price_max_frac_tca_median": 0.5,
                "c_price_min_frac_tca_median": 0.3,
                "c_price_max_frac_tca_median": 2.0,
            },
            "phase_thresholds": {
                "stage_1": {"h_negative_max": 100, "h_below_5_max": 200, "capture_ratio_pv_min": 0.85, "sr_max": 0.01},
                "stage_2": {
                    "h_negative_min": 200,
                    "h_negative_strong": 300,
                    "h_below_5_min": 500,
                    "capture_ratio_pv_max": 0.80,
                    "capture_ratio_pv_crisis": 0.70,
                    "days_spread_50_min": 150,
                },
                "stage_3": {"far_min": 0.60, "far_strong": 0.80, "require_h_neg_declining": True},
                "stage_4": {"far_min": 0.90, "h_regime_c_max": 1500},
            },
            "alerts": {},
        },
        "scenarios": {"accelerated_pv": {"name": "Acceleration PV", "params": {"delta_pv_gw": 2.0}}},
        "ui_overrides": {},
    }


def test_app_and_pages_smoke() -> None:
    state = _synthetic_state()

    files = [Path("app.py")] + sorted(Path("pages").glob("*.py"))
    for file in files:
        at = AppTest.from_file(str(file))
        at.session_state["state"] = state
        at.run(timeout=30)
        assert not at.exception, f"Unexpected exception in {file}: {at.exception}"


def test_excesum_static_page_shows_quality_sections() -> None:
    page = Path("pages/9_\U0001F9FE_ExceSum_des_conclusions.py")
    at = AppTest.from_file(str(page))
    at.run(timeout=30)
    assert not at.exception, f"Unexpected exception in {page}: {at.exception}"

    markdown_values = [getattr(x, "value", "") for x in at.markdown]
    all_text = "\n".join(str(v) for v in markdown_values)
    assert "Controle qualite" in all_text
