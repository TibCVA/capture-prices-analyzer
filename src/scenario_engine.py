"""Scenario engine v3.0: deterministic profile perturbations + full recompute."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    COL_LOAD,
    COL_MUST_RUN,
    COL_PRICE_DA,
    COL_REGIME,
    COL_SOLAR,
    COL_WIND_OFF,
    COL_WIND_ON,
    Q995,
)
from src.nrl_engine import compute_nrl
from src.time_utils import to_utc_index


def _scale_profile(series: pd.Series, delta_cap_gw: float, label: str) -> pd.Series:
    if delta_cap_gw == 0:
        return series

    cap_est = float(series.fillna(0.0).quantile(Q995))
    delta_mw = float(delta_cap_gw) * 1000.0

    if cap_est < 100.0 and delta_mw > 0:
        raise NotImplementedError(
            f"Profil quasi-nul : fournir un profil de reference externe ({label})"
        )

    if cap_est <= 0:
        return series

    scale_factor = (cap_est + delta_mw) / cap_est
    return (series.fillna(0.0) * scale_factor).clip(lower=0.0)


def _apply_demand_shape(df: pd.DataFrame, country_cfg: dict, params: dict) -> None:
    delta_pct = float(params.get("delta_demand_pct", 0.0))
    if delta_pct != 0.0:
        df[COL_LOAD] = df[COL_LOAD] * (1.0 + delta_pct / 100.0)

    tz = country_cfg["timezone"]
    idx_local = df.index.tz_convert(tz)

    midday_delta_mw = float(params.get("delta_demand_midday_gw", 0.0)) * 1000.0
    evening_delta_mw = float(params.get("delta_demand_evening_gw", 0.0)) * 1000.0

    if midday_delta_mw != 0.0:
        mask_mid = (idx_local.hour >= 11) & (idx_local.hour <= 15)
        df.loc[mask_mid, COL_LOAD] = df.loc[mask_mid, COL_LOAD] + midday_delta_mw

    if evening_delta_mw != 0.0:
        mask_eve = (idx_local.hour >= 18) & (idx_local.hour <= 21)
        df.loc[mask_eve, COL_LOAD] = df.loc[mask_eve, COL_LOAD] + evening_delta_mw


def apply_scenario(
    df_base_processed: pd.DataFrame,
    country_key: str,
    year: int,
    country_cfg: dict,
    thresholds: dict,
    commodities: dict,
    scenario_params: dict,
    price_mode: str = "synthetic",
    must_run_mode: str | None = None,
    flex_model_mode: str | None = None,
) -> pd.DataFrame:
    """Apply deterministic scenario perturbations and recompute full pipeline."""

    if COL_LOAD not in df_base_processed.columns:
        raise ValueError("df_base_processed invalide: load_mw manquant")

    df = df_base_processed.copy()
    df.index = to_utc_index(pd.DatetimeIndex(df.index))

    # Build raw-like frame for recompute.
    # We keep physical input columns and observed price (used for coherence/obs metrics).
    raw = pd.DataFrame(index=df.index)
    for col in [
        COL_LOAD,
        COL_SOLAR,
        COL_WIND_ON,
        COL_WIND_OFF,
        COL_PRICE_DA,
    ]:
        if col in df.columns:
            raw[col] = df[col]

    # Preserve commonly used generation components when present
    for extra in [
        "nuclear_mw",
        "lignite_mw",
        "coal_mw",
        "gas_mw",
        "hydro_ror_mw",
        "hydro_reservoir_mw",
        "psh_generation_mw",
        "psh_pumping_mw",
        "biomass_mw",
        "other_mw",
        "net_position_mw",
    ]:
        if extra in df.columns:
            raw[extra] = df[extra]

    # VRE scaling
    raw[COL_SOLAR] = _scale_profile(raw.get(COL_SOLAR, pd.Series(0.0, index=raw.index)), float(scenario_params.get("delta_pv_gw", 0.0)), "PV")
    raw[COL_WIND_ON] = _scale_profile(
        raw.get(COL_WIND_ON, pd.Series(0.0, index=raw.index)),
        float(scenario_params.get("delta_wind_onshore_gw", 0.0)),
        "Wind Onshore",
    )
    raw[COL_WIND_OFF] = _scale_profile(
        raw.get(COL_WIND_OFF, pd.Series(0.0, index=raw.index)),
        float(scenario_params.get("delta_wind_offshore_gw", 0.0)),
        "Wind Offshore",
    )

    # Demand perturbations
    _apply_demand_shape(raw, country_cfg, scenario_params)

    # Ensure observed price remains available (for observable metrics)
    if COL_PRICE_DA not in raw.columns:
        raw[COL_PRICE_DA] = np.nan

    # Full recompute in scenario modes.
    # By spec default: must_run=floor and flex=capacity.
    mr_mode = must_run_mode or "floor"
    fx_mode = flex_model_mode or "capacity"

    result = compute_nrl(
        df_raw=raw,
        country_key=country_key,
        year=year,
        country_cfg=country_cfg,
        thresholds=thresholds,
        commodities=commodities,
        must_run_mode=mr_mode,
        flex_model_mode=fx_mode,
        scenario_overrides=scenario_params,
        price_mode=price_mode,
    )

    return result
