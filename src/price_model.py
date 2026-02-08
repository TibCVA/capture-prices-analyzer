"""Thermal anchor and synthetic price model (v3.0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    COAL_GAS_INDEX_RATIO,
    COL_PRICE_DA,
    COL_PRICE_SYNTH,
    COL_PRICE_USED,
    COL_REGIME,
    COL_TCA,
    EF_COAL,
    EF_GAS,
    ETA_CCGT,
    ETA_COAL,
    PRICE_SYNTH_A,
    PRICE_SYNTH_A_MIN,
    PRICE_SYNTH_B,
    PRICE_SYNTH_B_MAX,
    PRICE_SYNTH_B_MIN,
    PRICE_SYNTH_C_ADDER,
    PRICE_SYNTH_D_MAX,
    PRICE_SYNTH_D_MULTIPLIER,
    VOM_CCGT,
    VOM_COAL,
)


def _to_naive_daily_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _as_daily_series(values: pd.Series) -> pd.Series:
    s = values.copy()
    s.index = pd.to_datetime(s.index)
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s.sort_index()


def _build_hourly_from_daily(
    base_index: pd.DatetimeIndex,
    daily_series: pd.Series | None,
    override_constant: float | None,
    label: str,
) -> np.ndarray:
    days = _to_naive_daily_index(base_index.floor("D"))
    unique_days = pd.DatetimeIndex(days.unique())

    if override_constant is not None:
        daily = pd.Series(float(override_constant), index=unique_days)
    else:
        if daily_series is None:
            raise ValueError(f"Commodite manquante pour TCA: {label}")
        daily = _as_daily_series(daily_series)

    aligned = daily.reindex(unique_days, method="ffill")
    if aligned.isna().any():
        aligned = aligned.reindex(unique_days, method="bfill")
    mapped = aligned.reindex(days, method="ffill")
    return mapped.values.astype(float)


def compute_tca(
    df: pd.DataFrame,
    country_cfg: dict,
    commodities: dict,
    scenario_overrides: dict | None,
) -> pd.Series:
    """Calcule la Thermal Cost Anchor horaire."""

    scenario_overrides = scenario_overrides or {}
    tech = str(country_cfg["thermal"]["marginal_tech"]).lower()

    gas_h = _build_hourly_from_daily(
        base_index=df.index,
        daily_series=commodities.get("gas_daily") if commodities else None,
        override_constant=scenario_overrides.get("gas_price_eur_mwh"),
        label="gas_daily",
    )
    co2_h = _build_hourly_from_daily(
        base_index=df.index,
        daily_series=commodities.get("co2_daily") if commodities else None,
        override_constant=scenario_overrides.get("co2_price_eur_t"),
        label="co2_daily",
    )

    if tech == "ccgt":
        tca = gas_h / ETA_CCGT + (EF_GAS / ETA_CCGT) * co2_h + VOM_CCGT
    elif tech == "coal":
        coal_daily = commodities.get("coal_daily") if commodities else None
        if coal_daily is None:
            coal_h = gas_h * COAL_GAS_INDEX_RATIO
        else:
            coal_h = _build_hourly_from_daily(
                base_index=df.index,
                daily_series=coal_daily,
                override_constant=None,
                label="coal_daily",
            )
        tca = coal_h / ETA_COAL + (EF_COAL / ETA_COAL) * co2_h + VOM_COAL
    else:
        raise NotImplementedError(
            f"Spec ambigue : thermal.marginal_tech non supporte ({country_cfg['thermal']['marginal_tech']})"
        )

    return pd.Series(tca, index=df.index, name=COL_TCA)


def compute_price_synth(df: pd.DataFrame) -> pd.Series:
    """Prix synthetique affine par regime A/B/C/D."""

    if COL_REGIME not in df.columns or COL_TCA not in df.columns:
        raise ValueError("compute_price_synth requiert les colonnes regime et tca_eur_mwh")

    out = pd.Series(np.nan, index=df.index, name=COL_PRICE_SYNTH)

    mask_a = df[COL_REGIME] == "A"
    mask_b = df[COL_REGIME] == "B"
    mask_c = df[COL_REGIME] == "C"
    mask_d = df[COL_REGIME] == "D"

    out.loc[mask_a] = max(PRICE_SYNTH_A_MIN, PRICE_SYNTH_A)
    out.loc[mask_b] = min(max(PRICE_SYNTH_B, PRICE_SYNTH_B_MIN), PRICE_SYNTH_B_MAX)
    out.loc[mask_c] = df.loc[mask_c, COL_TCA] + PRICE_SYNTH_C_ADDER
    out.loc[mask_d] = np.minimum(PRICE_SYNTH_D_MAX, df.loc[mask_d, COL_TCA] * PRICE_SYNTH_D_MULTIPLIER)

    return out


def select_price_used(df: pd.DataFrame, price_mode: str) -> pd.Series:
    """Selectionne la colonne de prix a utiliser dans les metriques."""

    if price_mode == "observed":
        if COL_PRICE_DA not in df.columns:
            raise ValueError("price_mode='observed' mais price_da_eur_mwh absent")
        return df[COL_PRICE_DA].rename(COL_PRICE_USED)

    if price_mode == "synthetic":
        if COL_PRICE_SYNTH not in df.columns:
            raise ValueError("price_mode='synthetic' mais price_synth_eur_mwh absent")
        return df[COL_PRICE_SYNTH].rename(COL_PRICE_USED)

    raise ValueError(f"price_mode invalide: {price_mode}")
