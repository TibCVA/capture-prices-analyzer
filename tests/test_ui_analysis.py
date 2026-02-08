from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    COL_BESS_CHARGE,
    COL_LOAD,
    COL_NRL,
    COL_PRICE_DA,
    COL_REGIME,
    COL_SINK_NON_BESS,
    COL_SOLAR,
    COL_SURPLUS,
    COL_SURPLUS_UNABS,
    COL_WIND_OFF,
    COL_WIND_ON,
)
from src.ui_analysis import (
    compute_nrl_price_link_stats,
    compute_q4_bess_sweep,
    compute_q4_plateau_diagnostics,
    find_q4_stress_reference,
)


def _country_cfg() -> dict:
    return {
        "name": "France",
        "entsoe_code": "FR",
        "timezone": "Europe/Paris",
        "must_run": {
            "mode": "observed",
            "observed_components": [],
            "floor_params": {
                "nuclear_floor_gw": 0.0,
                "nuclear_min_output_pct": 0.0,
                "hydro_ror_floor_gw": 0.0,
                "biomass_floor_gw": 0.0,
            },
        },
        "flex": {
            "model_mode": "observed",
            "historical_proxy": {"use_psh_pumping": False, "use_positive_net_position": False},
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


def _thresholds() -> dict:
    return {
        "model_params": {"regime_d": {"method": "quantile", "positive_nrl_quantile": 0.90, "absolute_nrl_mw": None}},
        "coherence_params": {
            "price_low_threshold": 5.0,
            "b_price_min": -10.0,
            "b_price_max_frac_tca_median": 0.50,
            "c_price_min_frac_tca_median": 0.30,
            "c_price_max_frac_tca_median": 2.00,
        },
    }


def _commodities() -> dict:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    return {
        "gas_daily": pd.Series(35.0, index=idx),
        "co2_daily": pd.Series(80.0, index=idx),
        "coal_daily": None,
        "bess_capacity": None,
    }


def _base_processed_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    h = np.arange(len(idx))
    solar = np.maximum(0.0, np.sin((h % 24 - 6) / 24 * np.pi * 2)) * 200.0
    wind_on = np.full(len(idx), 200.0)
    load = np.full(len(idx), 2500.0)
    return pd.DataFrame(
        {
            COL_LOAD: load,
            COL_SOLAR: solar,
            COL_WIND_ON: wind_on,
            COL_WIND_OFF: 0.0,
            COL_PRICE_DA: 60.0,
            COL_NRL: load - (solar + wind_on),
            COL_SURPLUS: 0.0,
            COL_SURPLUS_UNABS: 0.0,
            COL_SINK_NON_BESS: 0.0,
            COL_BESS_CHARGE: 0.0,
            COL_REGIME: "C",
        },
        index=idx,
    )


def test_compute_nrl_price_link_stats_returns_core_fields() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            COL_NRL: np.linspace(-1000, 1000, 10),
            COL_PRICE_DA: np.linspace(20, 120, 10),
        },
        index=idx,
    )
    metrics = {"regime_coherence": 0.72}

    out = compute_nrl_price_link_stats(df, metrics)

    assert set(out.keys()) == {
        "pearson_r",
        "pearson_r_pct",
        "regime_coherence",
        "regime_coherence_pct",
        "n_valid",
        "corr_status",
        "coherence_status",
    }
    assert out["n_valid"] == 10
    assert out["pearson_r"] > 0.99
    assert np.isclose(out["pearson_r_pct"], out["pearson_r"] * 100.0)
    assert np.isclose(out["regime_coherence"], 0.72)
    assert out["coherence_status"] in {"weak", "medium", "strong"}


def test_compute_nrl_price_link_stats_handles_missing_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)

    out = compute_nrl_price_link_stats(df, None)

    assert out["n_valid"] == 0
    assert np.isnan(out["pearson_r"])
    assert out["corr_status"] == "unknown"


def test_compute_q4_plateau_diagnostics_detects_full_absorption() -> None:
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            COL_SURPLUS: 100.0,
            COL_SURPLUS_UNABS: 0.0,
            COL_SINK_NON_BESS: 120.0,
            COL_BESS_CHARGE: 0.0,
            COL_REGIME: "B",
        },
        index=idx,
    )
    out = compute_q4_plateau_diagnostics(df)
    assert out["h_regime_a"] == 0
    assert np.isclose(out["far"], 1.0)
    assert out["total_surplus_unabs_twh"] == 0.0


def test_find_q4_stress_reference_identifies_reference() -> None:
    out = find_q4_stress_reference(
        df_base_processed=_base_processed_df(),
        country_key="FR",
        year=2024,
        country_cfg=_country_cfg(),
        thresholds=_thresholds(),
        commodities=_commodities(),
        max_delta_pv_gw=20,
        step_gw=2,
    )
    assert out["found"] is True
    assert float(out["delta_pv_gw"]) >= 0.0
    assert isinstance(out["df_reference"], pd.DataFrame)
    assert isinstance(out["tested_grid"], pd.DataFrame)
    assert not out["tested_grid"].empty


def test_compute_q4_bess_sweep_returns_expected_columns() -> None:
    df_sweep = compute_q4_bess_sweep(
        df_base_processed=_base_processed_df(),
        country_key="FR",
        year=2024,
        country_cfg=_country_cfg(),
        thresholds=_thresholds(),
        commodities=_commodities(),
        sweep_gw=[0.0, 2.0, 4.0, 6.0],
        reference_overrides={"delta_pv_gw": 8.0},
    )
    assert not df_sweep.empty
    assert {"delta_bess_power_gw", "far", "h_regime_a", "total_surplus_unabs_twh"}.issubset(df_sweep.columns)
    assert len(df_sweep) == 4
    assert np.isfinite(df_sweep["delta_bess_power_gw"]).all()

