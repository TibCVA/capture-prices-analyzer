from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.excesum_engine import (
    BaselineRunConfig,
    _q1_threshold_table,
    _q2_slopes,
    _q3_transition_status,
    _q6_heat_cold_scope,
    build_country_conclusions,
    run_excesum_baseline,
    write_excesum_docs,
)


def _metrics_df() -> pd.DataFrame:
    rows = []
    for country in ["FR", "DE"]:
        for year, pv, cr, hneg, h5, far in [
            (2020, 12.0, 0.95, 120, 300, 0.45),
            (2021, 14.0, 0.88, 180, 450, 0.58),
            (2022, 16.0, 0.82, 260, 520, 0.62),
            (2023, 18.0, 0.78, 210, 530, 0.67),
            (2024, 20.0, 0.74, 170, 510, 0.72),
        ]:
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "sr": 0.02 + 0.001 * (year - 2020),
                    "pv_penetration_pct_gen": pv,
                    "h_negative_obs": hneg,
                    "h_below_5_obs": h5,
                    "capture_ratio_pv": cr,
                    "far": far,
                    "h_regime_a": max(0, 300 - 40 * (year - 2020)),
                    "ttl": 95 + 2 * (year - 2020),
                    "phase": "stage_2",
                    "data_completeness": 0.99,
                    "regime_coherence": 0.7,
                    "is_outlier": year == 2022,
                }
            )
    return pd.DataFrame(rows)


def test_q_tables_and_country_conclusions(tmp_path: Path) -> None:
    m = _metrics_df()
    thresholds = {
        "phase_thresholds": {
            "stage_2": {"h_negative_min": 200, "h_below_5_min": 500, "capture_ratio_pv_max": 0.80},
            "stage_3": {"far_min": 0.60},
        }
    }

    q1_detail, q1_country = _q1_threshold_table(m, thresholds)
    q2_df = _q2_slopes(m, exclude_outlier_years=(2022,))
    q3_df = _q3_transition_status(m, thresholds)
    q4_df = pd.DataFrame(
        [
            {"country": "FR", "plateau_baseline": True, "stress_found": True},
            {"country": "DE", "plateau_baseline": False, "stress_found": True},
        ]
    )
    q5_df = pd.DataFrame(
        [
            {"country": "FR", "delta_ttl_high_co2": 12.0, "delta_ttl_high_gas": 18.0},
            {"country": "DE", "delta_ttl_high_co2": 11.0, "delta_ttl_high_gas": 17.5},
        ]
    )
    q6_df = _q6_heat_cold_scope(m)

    out = build_country_conclusions(m, q1_country, q2_df, q3_df, q4_df, q5_df, q6_df)

    assert not q1_detail.empty
    assert not q2_df.empty
    assert not q3_df.empty
    assert not q6_df.empty
    assert not out.empty
    assert {"country", "q2_slope", "q3_status", "q6_status"}.issubset(out.columns)

    c_path, v_path = write_excesum_docs(
        {
            "metrics_df": m,
            "verification_rows": [
                {"check": "verification_data", "status": "PASS", "detail": "ok"},
                {"check": "verification_calc", "status": "PASS", "detail": "ok"},
            ],
        },
        docs_dir=str(tmp_path),
    )
    assert Path(c_path).exists()
    assert Path(v_path).exists()


def test_q2_slopes_excludes_outlier_year() -> None:
    m = _metrics_df()
    slopes = _q2_slopes(m, exclude_outlier_years=(2022,))
    assert (slopes["n_points"] == 4).all()
    assert np.isfinite(slopes["slope"]).all()


def test_run_excesum_baseline_force_recompute_yields_regime_a(monkeypatch) -> None:
    import src.excesum_engine as ex
    from src.constants import (
        COL_LOAD,
        COL_NET_POSITION,
        COL_NUCLEAR,
        COL_PRICE_DA,
        COL_PSH_PUMP,
        COL_SOLAR,
        COL_WIND_ON,
    )

    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    solar = [0.0] * 8 + [2500.0] * 8 + [0.0] * 8 + [0.0] * 24
    raw = pd.DataFrame(
        {
            COL_LOAD: 3000.0,
            COL_SOLAR: solar,
            COL_WIND_ON: 200.0,
            COL_NUCLEAR: 1300.0,
            COL_PRICE_DA: 60.0,
            COL_PSH_PUMP: 0.0,
            COL_NET_POSITION: 0.0,
        },
        index=idx,
    )

    countries_cfg = {
        "FR": {
            "name": "France",
            "entsoe_code": "FR",
            "timezone": "Europe/Paris",
            "must_run": {
                "mode": "observed",
                "observed_components": ["nuclear"],
                "floor_params": {"nuclear_floor_gw": 0.0, "nuclear_min_output_pct": 0.0},
            },
            "flex": {
                "model_mode": "observed",
                "historical_proxy": {"use_psh_pumping": True, "use_positive_net_position": True},
                "capacity_defaults": {
                    "export_max_gw": 0.0,
                    "psh_pump_gw": 0.0,
                    "dsm_gw": 0.0,
                    "bess_power_gw_default": 0.0,
                    "bess_energy_gwh_default": 0.0,
                },
            },
            "thermal": {"marginal_tech": "CCGT"},
        }
    }
    thresholds = {
        "model_params": {"regime_d": {"method": "quantile", "positive_nrl_quantile": 0.90, "absolute_nrl_mw": None}},
        "coherence_params": {
            "price_low_threshold": 5.0,
            "b_price_min": -10.0,
            "b_price_max_frac_tca_median": 0.50,
            "c_price_min_frac_tca_median": 0.30,
            "c_price_max_frac_tca_median": 2.00,
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
    }
    scenarios = {"dummy": {"name": "dummy", "params": {}}}
    commodities = {
        "gas_daily": pd.Series([30.0], index=pd.date_range("2024-01-01", periods=1, freq="D")),
        "co2_daily": pd.Series([80.0], index=pd.date_range("2024-01-01", periods=1, freq="D")),
        "coal_daily": None,
        "bess_capacity": None,
    }

    monkeypatch.setattr(ex, "load_countries_config", lambda: countries_cfg)
    monkeypatch.setattr(ex, "load_thresholds", lambda: thresholds)
    monkeypatch.setattr(ex, "load_scenarios", lambda: scenarios)
    monkeypatch.setattr(ex, "load_commodity_prices", lambda: commodities)
    monkeypatch.setattr(ex, "load_raw", lambda country, year: raw.copy())
    monkeypatch.setattr(ex, "load_processed", lambda *args, **kwargs: None)
    monkeypatch.setattr(ex, "save_processed", lambda *args, **kwargs: None)

    cfg = BaselineRunConfig(countries=("FR",), years=(2024,), force_recompute=True)
    out = run_excesum_baseline(run_cfg=cfg, force_recompute=True)
    assert not out["metrics_df"].empty
    row = out["metrics_df"].iloc[0]
    assert float(row["h_regime_a"]) > 0
    assert float(row["total_surplus_unabs_twh"]) > 0
    assert "h_negative_declining" in out["metrics_df"].columns
    assert "phase_blocked_rules" in out["metrics_df"].columns
    cc = out["country_conclusions"].iloc[0]
    assert cc["phase_latest"] == row["phase"]
