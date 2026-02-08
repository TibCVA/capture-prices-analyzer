"""NRL engine v3.0: physical regimes, BESS dispatch, and price linkage."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.constants import (
    BESS_ETA_CHARGE,
    BESS_ETA_DISCHARGE,
    BESS_SOC_INIT_FRAC,
    COL_BESS_CHARGE,
    COL_BESS_DISCHARGE,
    COL_BESS_SOC,
    COL_BIOMASS,
    COL_COAL,
    COL_FLEX_EFFECTIVE,
    COL_GAS,
    COL_HAS_GAP,
    COL_HYDRO_ROR,
    COL_LIGNITE,
    COL_LOAD,
    COL_MUST_RUN,
    COL_NET_POSITION,
    COL_NRL,
    COL_NUCLEAR,
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
    Q75,
)
from src.price_model import compute_price_synth, compute_tca, select_price_used
from src.time_utils import to_utc_index

logger = logging.getLogger("capture_prices.nrl_engine")


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")


def _get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(0.0)
    return pd.Series(default, index=df.index, dtype=float)


def _must_run_observed(df: pd.DataFrame, country_cfg: dict) -> pd.Series:
    mapping = {
        "nuclear": COL_NUCLEAR,
        "lignite": COL_LIGNITE,
        "coal": COL_COAL,
        "hydro_ror": COL_HYDRO_ROR,
        "biomass": COL_BIOMASS,
    }
    comps = country_cfg["must_run"].get("observed_components", [])
    mr = pd.Series(0.0, index=df.index)
    for c in comps:
        if c in mapping:
            mr = mr.add(_get_series(df, mapping[c]), fill_value=0.0)
    return mr


def _must_run_floor(df: pd.DataFrame, country_cfg: dict) -> pd.Series:
    mapping = {
        "nuclear": COL_NUCLEAR,
        "lignite": COL_LIGNITE,
        "coal": COL_COAL,
        "hydro_ror": COL_HYDRO_ROR,
        "biomass": COL_BIOMASS,
    }
    floor_params = country_cfg["must_run"].get("floor_params", {})

    mr = pd.Series(0.0, index=df.index)
    for tech, col in mapping.items():
        obs = _get_series(df, col)
        floor_gw = floor_params.get(f"{tech}_floor_gw", 0.0)
        min_pct = floor_params.get(f"{tech}_min_output_pct", 1.0)

        floor_abs_mw = float(floor_gw) * 1000.0
        min_pct = float(min_pct)
        if not 0.0 <= min_pct <= 1.0:
            raise ValueError(f"floor min_output_pct hors borne [0,1] pour {tech}")

        mr_tech = np.maximum(floor_abs_mw, obs.values * min_pct)
        mr_tech = np.minimum(mr_tech, obs.values)  # clamp historique
        mr = mr.add(pd.Series(mr_tech, index=df.index), fill_value=0.0)

    return mr


def _compute_sink_non_bess(df: pd.DataFrame, country_cfg: dict, flex_mode: str, scenario_overrides: dict) -> pd.Series:
    if flex_mode == "observed":
        hist = country_cfg["flex"].get("historical_proxy", {})
        sink = pd.Series(0.0, index=df.index)
        if hist.get("use_psh_pumping", False):
            sink = sink.add(_get_series(df, COL_PSH_PUMP).clip(lower=0.0), fill_value=0.0)
        if hist.get("use_positive_net_position", False):
            if COL_NET_POSITION in df.columns:
                sink = sink.add(df[COL_NET_POSITION].fillna(0.0).clip(lower=0.0), fill_value=0.0)
        return sink

    if flex_mode == "capacity":
        cap = country_cfg["flex"].get("capacity_defaults", {})
        export_max = float(cap.get("export_max_gw", 0.0)) + float(
            scenario_overrides.get("delta_export_max_gw", 0.0)
        )
        psh_pump = float(cap.get("psh_pump_gw", 0.0)) + float(
            scenario_overrides.get("delta_psh_pump_gw", 0.0)
        )
        dsm = float(cap.get("dsm_gw", 0.0)) + float(scenario_overrides.get("delta_dsm_gw", 0.0))
        sink_mw = max(0.0, (export_max + psh_pump + dsm) * 1000.0)
        return pd.Series(sink_mw, index=df.index, dtype=float)

    raise ValueError(f"flex_model_mode invalide: {flex_mode}")


def _resolve_bess_capacity(
    country_key: str,
    year: int,
    country_cfg: dict,
    commodities: dict | None,
    scenario_overrides: dict,
) -> tuple[float, float]:
    cap = country_cfg["flex"].get("capacity_defaults", {})
    power_mw = float(cap.get("bess_power_gw_default", 0.0)) * 1000.0
    energy_mwh = float(cap.get("bess_energy_gwh_default", 0.0)) * 1000.0

    bess_hist = None
    if commodities:
        bess_hist = commodities.get("bess_capacity")

    if isinstance(bess_hist, pd.DataFrame):
        match = bess_hist[(bess_hist["country"] == country_key) & (bess_hist["year"] == year)]
        if not match.empty:
            power_mw = float(match.iloc[0]["power_mw"])
            energy_mwh = float(match.iloc[0]["energy_mwh"])

    power_mw += float(scenario_overrides.get("delta_bess_power_gw", 0.0)) * 1000.0
    energy_mwh += float(scenario_overrides.get("delta_bess_energy_gwh", 0.0)) * 1000.0

    return max(0.0, power_mw), max(0.0, energy_mwh)


def _dispatch_bess(
    nrl: pd.Series,
    surplus: pd.Series,
    sink_non_bess: pd.Series,
    bess_power_mw: float,
    bess_energy_mwh: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    charge = np.zeros(len(nrl), dtype=float)
    discharge = np.zeros(len(nrl), dtype=float)
    soc_series = np.zeros(len(nrl), dtype=float)

    soc = BESS_SOC_INIT_FRAC * bess_energy_mwh

    for i in range(len(nrl)):
        surplus_after_non_bess = max(0.0, float(surplus.iloc[i]) - float(sink_non_bess.iloc[i]))

        charge_limit_power = bess_power_mw
        charge_limit_energy = (
            (bess_energy_mwh - soc) / BESS_ETA_CHARGE if BESS_ETA_CHARGE > 0 else 0.0
        )
        charge_i = min(surplus_after_non_bess, charge_limit_power, max(0.0, charge_limit_energy))
        soc += charge_i * BESS_ETA_CHARGE

        deficit = max(0.0, float(nrl.iloc[i]))
        discharge_limit_power = bess_power_mw
        discharge_limit_energy = soc * BESS_ETA_DISCHARGE
        discharge_i = min(deficit, discharge_limit_power, max(0.0, discharge_limit_energy))
        soc -= discharge_i / BESS_ETA_DISCHARGE if BESS_ETA_DISCHARGE > 0 else 0.0

        charge[i] = charge_i
        discharge[i] = discharge_i
        soc_series[i] = soc

    return (
        pd.Series(charge, index=nrl.index, name=COL_BESS_CHARGE),
        pd.Series(discharge, index=nrl.index, name=COL_BESS_DISCHARGE),
        pd.Series(soc_series, index=nrl.index, name=COL_BESS_SOC),
    )


def _compute_tca_fallback(price_da: pd.Series) -> pd.Series:
    valid_count = int(price_da.notna().sum())
    window = 24 * 30
    if valid_count < window:
        return pd.Series(np.nan, index=price_da.index, name=COL_TCA)
    return price_da.rolling(window=window, min_periods=window).quantile(Q75).rename(COL_TCA)


def _compute_regime(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    regime = pd.Series("C", index=df.index)

    mask_b = (df[COL_SURPLUS] > 0.0) & (df[COL_SURPLUS_UNABS] == 0.0)
    mask_a = df[COL_SURPLUS_UNABS] > 0.0
    regime.loc[mask_b] = "B"
    regime.loc[mask_a] = "A"

    model_params = thresholds.get("model_params", {})
    regime_d = model_params.get("regime_d", {})
    method = regime_d.get("method", "quantile")

    nrl_pos = df.loc[df[COL_NRL] > 0.0, COL_NRL]

    if method == "quantile":
        q = float(regime_d.get("positive_nrl_quantile", 0.90))
        if len(nrl_pos) == 0:
            return regime
        nrl_threshold = float(nrl_pos.quantile(q))
    elif method == "absolute":
        nrl_threshold = regime_d.get("absolute_nrl_mw")
        if nrl_threshold is None:
            raise NotImplementedError(
                "Spec ambigue : method='absolute' sans absolute_nrl_mw (section E.2)"
            )
        nrl_threshold = float(nrl_threshold)
    else:
        raise NotImplementedError(f"Spec ambigue : regime_d.method inconnu ({method})")

    mask_d = df[COL_NRL] > nrl_threshold
    regime.loc[mask_d] = "D"
    return regime


def _coherence_flags(df: pd.DataFrame, thresholds: dict, price_mode: str) -> tuple[pd.Series, float]:
    if price_mode != "observed":
        return pd.Series(pd.NA, index=df.index, dtype="boolean", name=COL_REGIME_COHERENT), np.nan

    if COL_PRICE_DA not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="boolean", name=COL_REGIME_COHERENT), np.nan

    coh = thresholds.get("coherence_params", {})
    price_low = float(coh.get("price_low_threshold", 5.0))
    b_min = float(coh.get("b_price_min", -10.0))
    b_max_frac = float(coh.get("b_price_max_frac_tca_median", 0.50))
    c_min_frac = float(coh.get("c_price_min_frac_tca_median", 0.30))
    c_max_frac = float(coh.get("c_price_max_frac_tca_median", 2.00))

    valid = df[[COL_REGIME, COL_PRICE_DA, COL_TCA]].dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=df.index, dtype="boolean", name=COL_REGIME_COHERENT), np.nan

    tca_median = float(valid[COL_TCA].median())
    out = pd.Series(pd.NA, index=df.index, dtype="boolean", name=COL_REGIME_COHERENT)

    mask_a = valid[COL_REGIME] == "A"
    mask_b = valid[COL_REGIME] == "B"
    mask_c = valid[COL_REGIME] == "C"
    mask_d = valid[COL_REGIME] == "D"

    out.loc[valid.index[mask_a]] = valid.loc[mask_a, COL_PRICE_DA] <= price_low
    out.loc[valid.index[mask_b]] = (valid.loc[mask_b, COL_PRICE_DA] >= b_min) & (
        valid.loc[mask_b, COL_PRICE_DA] <= (b_max_frac * tca_median)
    )
    out.loc[valid.index[mask_c]] = (valid.loc[mask_c, COL_PRICE_DA] >= (c_min_frac * tca_median)) & (
        valid.loc[mask_c, COL_PRICE_DA] <= (c_max_frac * tca_median)
    )
    out.loc[valid.index[mask_d]] = valid.loc[mask_d, COL_PRICE_DA] > tca_median

    out_num = out.dropna().astype(float)
    score = float(out_num.mean()) if out_num.size > 0 else np.nan
    return out, score


def compute_nrl(
    df_raw: pd.DataFrame,
    country_key: str,
    year: int,
    country_cfg: dict,
    thresholds: dict,
    commodities: dict | None = None,
    must_run_mode: str | None = None,
    flex_model_mode: str | None = None,
    scenario_overrides: dict | None = None,
    price_mode: str = "observed",
) -> pd.DataFrame:
    """Compute full v3.0 physical + price pipeline."""

    scenario_overrides = scenario_overrides or {}

    if bool(df_raw.attrs.get("derived_net_position_from_generation_minus_load", False)):
        raise NotImplementedError(
            "Spec ambigue : approximation exports via generation-load interdite (section I.1)"
        )

    required = [COL_LOAD, COL_SOLAR, COL_WIND_ON, COL_PRICE_DA]
    _require_columns(df_raw, required)

    if not isinstance(df_raw.index, pd.DatetimeIndex):
        raise TypeError("df_raw doit avoir un DatetimeIndex")

    df = df_raw.copy()
    df.index = to_utc_index(pd.DatetimeIndex(df.index))

    if df[COL_LOAD].isna().all():
        raise RuntimeError("Absence totale de load (spec I.1)")

    # Ensure optional columns
    for col in [
        COL_WIND_OFF,
        COL_NUCLEAR,
        COL_LIGNITE,
        COL_COAL,
        COL_GAS,
        COL_HYDRO_ROR,
        COL_BIOMASS,
        COL_PSH_PUMP,
        COL_NET_POSITION,
    ]:
        if col not in df.columns:
            df[col] = 0.0 if col != COL_NET_POSITION else np.nan

    # Step 1 — VRE
    df[COL_VRE] = _get_series(df, COL_SOLAR) + _get_series(df, COL_WIND_ON) + _get_series(df, COL_WIND_OFF)

    # Step 2 — Must-run
    mr_mode = must_run_mode or country_cfg["must_run"]["mode"]
    if mr_mode == "observed":
        df[COL_MUST_RUN] = _must_run_observed(df, country_cfg)
    elif mr_mode == "floor":
        df[COL_MUST_RUN] = _must_run_floor(df, country_cfg)
    else:
        raise ValueError(f"must_run_mode invalide: {mr_mode}")

    # Optional scenario delta on must-run
    delta_mr = float(scenario_overrides.get("delta_must_run_gw", 0.0)) * 1000.0
    if delta_mr != 0.0:
        df[COL_MUST_RUN] = (df[COL_MUST_RUN] + delta_mr).clip(lower=0.0)

    # Step 3 — NRL
    df[COL_NRL] = _get_series(df, COL_LOAD) - df[COL_VRE] - df[COL_MUST_RUN]

    # Step 4 — Surplus
    df[COL_SURPLUS] = (-df[COL_NRL]).clip(lower=0.0)

    # Step 5 — Sink non-BESS
    flex_mode = flex_model_mode or country_cfg["flex"]["model_mode"]
    df[COL_SINK_NON_BESS] = _compute_sink_non_bess(df, country_cfg, flex_mode, scenario_overrides)

    # Step 6 — BESS capacities
    bess_power_mw, bess_energy_mwh = _resolve_bess_capacity(
        country_key, year, country_cfg, commodities, scenario_overrides
    )

    # Step 7 — Sequential BESS dispatch
    bess_charge, bess_discharge, bess_soc = _dispatch_bess(
        nrl=df[COL_NRL],
        surplus=df[COL_SURPLUS],
        sink_non_bess=df[COL_SINK_NON_BESS],
        bess_power_mw=bess_power_mw,
        bess_energy_mwh=bess_energy_mwh,
    )
    df[COL_BESS_CHARGE] = bess_charge
    df[COL_BESS_DISCHARGE] = bess_discharge
    df[COL_BESS_SOC] = bess_soc

    # Step 8 — Flex effective + unabsorbed surplus
    df[COL_FLEX_EFFECTIVE] = df[COL_SINK_NON_BESS] + df[COL_BESS_CHARGE]
    df[COL_SURPLUS_UNABS] = (df[COL_SURPLUS] - df[COL_FLEX_EFFECTIVE]).clip(lower=0.0)

    # Step 9 — Regimes (strict anti-circularity)
    df[COL_REGIME] = _compute_regime(df, thresholds)

    # Step 10 — TCA + synthetic price
    if commodities and commodities.get("gas_daily") is not None and commodities.get("co2_daily") is not None:
        df[COL_TCA] = compute_tca(
            df=df,
            country_cfg=country_cfg,
            commodities=commodities,
            scenario_overrides=scenario_overrides,
        )
    else:
        df[COL_TCA] = _compute_tca_fallback(df[COL_PRICE_DA])

    df[COL_PRICE_SYNTH] = compute_price_synth(df)

    # Step 11 — price_used
    df[COL_PRICE_USED] = select_price_used(df, price_mode=price_mode)

    # Step 12 — coherence observed regime/price
    coherence_flags, coherence_score = _coherence_flags(df, thresholds, price_mode)
    df[COL_REGIME_COHERENT] = coherence_flags
    df.attrs["coherence_score"] = coherence_score
    df.attrs["country_key"] = country_key
    df.attrs["year"] = year
    df.attrs["must_run_mode"] = mr_mode
    df.attrs["flex_model_mode"] = flex_mode
    df.attrs["price_mode"] = price_mode

    if COL_HAS_GAP not in df.columns:
        df[COL_HAS_GAP] = df[COL_LOAD].isna() | df[COL_PRICE_DA].isna()

    return df
