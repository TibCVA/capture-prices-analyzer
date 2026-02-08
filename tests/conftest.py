from __future__ import annotations

import pandas as pd
import pytest

from src.constants import (
    COL_BIOMASS,
    COL_COAL,
    COL_GAS,
    COL_HAS_GAP,
    COL_HYDRO_RES,
    COL_HYDRO_ROR,
    COL_LIGNITE,
    COL_LOAD,
    COL_NET_POSITION,
    COL_NUCLEAR,
    COL_OTHER,
    COL_PRICE_DA,
    COL_PSH_GEN,
    COL_PSH_PUMP,
    COL_SOLAR,
    COL_WIND_OFF,
    COL_WIND_ON,
)


@pytest.fixture
def make_raw_df():
    def _make(
        n: int = 48,
        load: float = 50000.0,
        solar: float = 8000.0,
        wind_on: float = 7000.0,
        wind_off: float = 1000.0,
        nuclear: float = 20000.0,
        lignite: float = 0.0,
        coal: float = 2000.0,
        gas: float = 5000.0,
        hydro_ror: float = 3000.0,
        hydro_res: float = 1000.0,
        psh_gen: float = 0.0,
        psh_pump: float = 0.0,
        biomass: float = 1500.0,
        other: float = 500.0,
        price_da: float = 60.0,
        net_position: float = 0.0,
    ) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                COL_LOAD: load,
                COL_SOLAR: solar,
                COL_WIND_ON: wind_on,
                COL_WIND_OFF: wind_off,
                COL_NUCLEAR: nuclear,
                COL_LIGNITE: lignite,
                COL_COAL: coal,
                COL_GAS: gas,
                COL_HYDRO_ROR: hydro_ror,
                COL_HYDRO_RES: hydro_res,
                COL_PSH_GEN: psh_gen,
                COL_PSH_PUMP: psh_pump,
                COL_BIOMASS: biomass,
                COL_OTHER: other,
                COL_PRICE_DA: price_da,
                COL_NET_POSITION: net_position,
                COL_HAS_GAP: False,
            },
            index=idx,
        )

    return _make


@pytest.fixture
def fr_cfg() -> dict:
    return {
        "name": "France",
        "entsoe_code": "FR",
        "timezone": "Europe/Paris",
        "must_run": {
            "mode": "observed",
            "observed_components": ["nuclear", "hydro_ror", "biomass"],
            "floor_params": {
                "nuclear_floor_gw": 20.0,
                "nuclear_min_output_pct": 0.5,
                "hydro_ror_floor_gw": 0.0,
                "biomass_floor_gw": 0.0,
            },
        },
        "flex": {
            "model_mode": "observed",
            "historical_proxy": {"use_psh_pumping": True, "use_positive_net_position": True},
            "capacity_defaults": {
                "export_max_gw": 17.0,
                "psh_pump_gw": 4.5,
                "dsm_gw": 2.0,
                "bess_power_gw_default": 0.5,
                "bess_energy_gwh_default": 2.0,
            },
        },
        "thermal": {"marginal_tech": "CCGT"},
    }


@pytest.fixture
def thresholds_cfg() -> dict:
    return {
        "model_params": {
            "regime_d": {
                "method": "quantile",
                "positive_nrl_quantile": 0.90,
                "absolute_nrl_mw": None,
            }
        },
        "coherence_params": {
            "price_low_threshold": 5.0,
            "b_price_min": -10.0,
            "b_price_max_frac_tca_median": 0.50,
            "c_price_min_frac_tca_median": 0.30,
            "c_price_max_frac_tca_median": 2.00,
        },
        "phase_thresholds": {
            "stage_1": {
                "h_negative_max": 100,
                "h_below_5_max": 200,
                "capture_ratio_pv_min": 0.85,
                "sr_max": 0.01,
            },
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
        "alerts": {
            "approaching_stage_2": {
                "h_negative_range": [150, 300],
                "capture_ratio_pv_range": [0.75, 0.85],
                "label": "Approche Stage 2",
            },
            "deep_stage_2": {"h_negative_min": 500, "capture_ratio_pv_max": 0.65, "label": "Stage 2 severe"},
            "high_inflexibility": {"ir_min": 0.60, "label": "IR eleve"},
            "low_flex": {"far_max": 0.30, "sr_min": 0.02, "label": "Flex insuffisante"},
        },
    }


@pytest.fixture
def commodities_cfg() -> dict:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    gas = pd.Series(35.0, index=idx)
    co2 = pd.Series(80.0, index=idx)
    return {"gas_daily": gas, "co2_daily": co2, "coal_daily": None, "bess_capacity": None}
