"""Annual metrics computation aligned to v3.0 specification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    COL_BESS_CHARGE,
    COL_BESS_DISCHARGE,
    COL_BESS_SOC,
    COL_BIOMASS,
    COL_COAL,
    COL_FLEX_EFFECTIVE,
    COL_GAS,
    COL_HAS_GAP,
    COL_HYDRO_RES,
    COL_HYDRO_ROR,
    COL_LIGNITE,
    COL_LOAD,
    COL_MUST_RUN,
    COL_NRL,
    COL_NUCLEAR,
    COL_OTHER,
    COL_PRICE_DA,
    COL_PRICE_USED,
    COL_REGIME,
    COL_REGIME_COHERENT,
    COL_SOLAR,
    COL_SURPLUS,
    COL_SURPLUS_UNABS,
    COL_VRE,
    COL_WIND_OFF,
    COL_WIND_ON,
    OUTLIER_YEARS,
    PRICE_HIGH_THRESHOLD,
    PRICE_NEGATIVE_THRESHOLD,
    PRICE_VERY_HIGH_THRESHOLD,
    PRICE_VERY_LOW_THRESHOLD,
    Q05,
    Q10,
    Q25,
    Q50,
    Q75,
    Q95,
    SPREAD_DAILY_THRESHOLD,
)
from src.time_utils import peak_mask


def _safe_float(v) -> float:
    if v is None:
        return float("nan")
    return float(v)


def _ov_float(df: pd.DataFrame, key: str, default: float) -> float:
    overrides = df.attrs.get("ui_overrides", {}) if hasattr(df, "attrs") else {}
    val = overrides.get(key, default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _capture_rate(price_used: pd.Series, production: pd.Series) -> float:
    prod = production.fillna(0.0)
    total = float(prod.sum())
    if total <= 0.0:
        return float("nan")
    return float((price_used * prod).sum() / total)


def _price_stats(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {
            "mean": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "std": float("nan"),
        }
    return {
        "mean": float(s.mean()),
        "p05": float(s.quantile(Q05)),
        "p25": float(s.quantile(Q25)),
        "p50": float(s.quantile(Q50)),
        "p75": float(s.quantile(Q75)),
        "p95": float(s.quantile(Q95)),
        "std": float(s.std(ddof=0)),
    }


def _daily_spreads_observed(price_obs: pd.Series, tz: str) -> pd.Series:
    s = price_obs.dropna()
    if s.empty:
        return pd.Series(dtype=float)

    idx_local = s.index.tz_convert(tz)
    by_day = pd.DataFrame({"price": s.values, "date": idx_local.date})
    spread = by_day.groupby("date")["price"].agg(lambda x: float(np.max(x) - np.min(x)))
    return spread


def compute_annual_metrics(df: pd.DataFrame, country_key: str, year: int, country_cfg: dict) -> dict:
    """Compute annual metrics dictionary as specified in section G.7."""

    if COL_PRICE_USED not in df.columns:
        raise ValueError("compute_annual_metrics requiert la colonne price_used_eur_mwh")

    tz = country_cfg["timezone"]

    price_used = df[COL_PRICE_USED]
    price_obs = df[COL_PRICE_DA] if COL_PRICE_DA in df.columns else pd.Series(np.nan, index=df.index)

    price_negative_threshold = _ov_float(df, "price_negative_threshold", PRICE_NEGATIVE_THRESHOLD)
    price_very_low_threshold = _ov_float(df, "price_very_low_threshold", PRICE_VERY_LOW_THRESHOLD)
    price_high_threshold = _ov_float(df, "price_high_threshold", PRICE_HIGH_THRESHOLD)
    price_very_high_threshold = _ov_float(df, "price_very_high_threshold", PRICE_VERY_HIGH_THRESHOLD)
    spread_daily_threshold = _ov_float(df, "spread_daily_threshold", SPREAD_DAILY_THRESHOLD)

    used_stats = _price_stats(price_used)
    obs_stats = _price_stats(price_obs)

    peak = peak_mask(df.index, tz)
    peak_used = price_used[peak].dropna()
    offpeak_used = price_used[~peak].dropna()

    baseload_price_used = used_stats["mean"]
    peakload_price_used = float(peak_used.mean()) if not peak_used.empty else float("nan")
    offpeak_price_used = float(offpeak_used.mean()) if not offpeak_used.empty else float("nan")

    baseload_price_obs = obs_stats["mean"]

    # Observable counts strictly on observed price
    h_negative_obs = int((price_obs < price_negative_threshold).sum())
    h_below_5_obs = int((price_obs < price_very_low_threshold).sum())
    h_above_100_obs = int((price_obs > price_high_threshold).sum())
    h_above_200_obs = int((price_obs > price_very_high_threshold).sum())

    daily_spread_obs = _daily_spreads_observed(price_obs, tz)
    days_spread_above_50_obs = int((daily_spread_obs > spread_daily_threshold).sum())
    avg_daily_spread_obs = float(daily_spread_obs.mean()) if not daily_spread_obs.empty else float("nan")
    max_daily_spread_obs = float(daily_spread_obs.max()) if not daily_spread_obs.empty else float("nan")

    # Capture rates/ratios on price_used
    solar = df[COL_SOLAR] if COL_SOLAR in df.columns else pd.Series(0.0, index=df.index)
    wind_total = (
        (df[COL_WIND_ON] if COL_WIND_ON in df.columns else pd.Series(0.0, index=df.index))
        + (df[COL_WIND_OFF] if COL_WIND_OFF in df.columns else pd.Series(0.0, index=df.index))
    )

    capture_rate_pv = _capture_rate(price_used, solar)
    capture_rate_wind = _capture_rate(price_used, wind_total)
    capture_ratio_pv = (
        float(capture_rate_pv / baseload_price_used)
        if np.isfinite(capture_rate_pv) and np.isfinite(baseload_price_used) and baseload_price_used != 0
        else float("nan")
    )
    capture_ratio_wind = (
        float(capture_rate_wind / baseload_price_used)
        if np.isfinite(capture_rate_wind) and np.isfinite(baseload_price_used) and baseload_price_used != 0
        else float("nan")
    )

    # Regime hours
    regime = df[COL_REGIME] if COL_REGIME in df.columns else pd.Series("C", index=df.index)
    h_regime_a = int((regime == "A").sum())
    h_regime_b = int((regime == "B").sum())
    h_regime_c = int((regime == "C").sum())
    h_regime_d = int((regime == "D").sum())

    # Structural ratios
    surplus = df[COL_SURPLUS] if COL_SURPLUS in df.columns else pd.Series(0.0, index=df.index)
    flex_effective = (
        df[COL_FLEX_EFFECTIVE] if COL_FLEX_EFFECTIVE in df.columns else pd.Series(0.0, index=df.index)
    )

    gen_cols = [
        COL_SOLAR,
        COL_WIND_ON,
        COL_WIND_OFF,
        COL_NUCLEAR,
        COL_LIGNITE,
        COL_COAL,
        COL_GAS,
        COL_HYDRO_ROR,
        COL_HYDRO_RES,
        COL_BIOMASS,
        COL_OTHER,
    ]
    present_gen_cols = [c for c in gen_cols if c in df.columns]
    total_generation_mwh = float(df[present_gen_cols].fillna(0.0).sum().sum()) if present_gen_cols else 0.0

    total_load_mwh = float(df[COL_LOAD].fillna(0.0).sum()) if COL_LOAD in df.columns else float("nan")
    total_vre_mwh = float(df[COL_VRE].fillna(0.0).sum()) if COL_VRE in df.columns else float("nan")
    total_surplus_mwh = float(surplus.fillna(0.0).sum())
    total_surplus_unabs_mwh = (
        float(df[COL_SURPLUS_UNABS].fillna(0.0).sum()) if COL_SURPLUS_UNABS in df.columns else float("nan")
    )

    sr = float(total_surplus_mwh / total_generation_mwh) if total_generation_mwh > 0 else float("nan")

    valid_struct = df[[COL_SURPLUS]].dropna().index if COL_SURPLUS in df.columns else df.index
    total_hours_valid = len(valid_struct)
    sr_hours = (
        float((df.loc[valid_struct, COL_SURPLUS] > 0.0).sum() / total_hours_valid)
        if total_hours_valid > 0
        else float("nan")
    )

    if total_surplus_mwh == 0:
        far = float("nan")
    else:
        absorbed = float(np.minimum(surplus.fillna(0.0), flex_effective.fillna(0.0)).sum())
        far = absorbed / total_surplus_mwh

    valid_ir = df[[COL_MUST_RUN, COL_LOAD]].dropna() if COL_MUST_RUN in df.columns else pd.DataFrame()
    if valid_ir.empty:
        ir = float("nan")
    else:
        mr_p10 = float(valid_ir[COL_MUST_RUN].quantile(Q10))
        load_p10 = float(valid_ir[COL_LOAD].quantile(Q10))
        ir = float(mr_p10 / load_p10) if load_p10 != 0 else float("nan")

    mask_ttl = regime.isin(["C", "D"]) & price_used.notna()
    ttl = float(price_used.loc[mask_ttl].quantile(Q95)) if int(mask_ttl.sum()) > 0 else float("nan")

    # Penetration (% generation)
    solar_mwh = float(solar.fillna(0.0).sum())
    wind_mwh = float(wind_total.fillna(0.0).sum())
    vre_mwh = solar_mwh + wind_mwh

    pv_penetration_pct_gen = (
        float(100.0 * solar_mwh / total_generation_mwh) if total_generation_mwh > 0 else float("nan")
    )
    wind_penetration_pct_gen = (
        float(100.0 * wind_mwh / total_generation_mwh) if total_generation_mwh > 0 else float("nan")
    )
    vre_penetration_pct_gen = (
        float(100.0 * vre_mwh / total_generation_mwh) if total_generation_mwh > 0 else float("nan")
    )

    # BESS metrics
    bess_charge_mwh = float(df[COL_BESS_CHARGE].fillna(0.0).sum()) if COL_BESS_CHARGE in df.columns else 0.0
    bess_discharge_mwh = (
        float(df[COL_BESS_DISCHARGE].fillna(0.0).sum()) if COL_BESS_DISCHARGE in df.columns else 0.0
    )

    energy_cap_mwh = float(df[COL_BESS_SOC].max()) if COL_BESS_SOC in df.columns else 0.0
    bess_cycles_est = (
        float(bess_discharge_mwh / energy_cap_mwh) if energy_cap_mwh > 0 else float("nan")
    )

    # Quality
    if COL_HAS_GAP in df.columns:
        data_completeness = float((~df[COL_HAS_GAP].fillna(True)).mean())
    else:
        data_completeness = float((df[COL_LOAD].notna() & price_obs.notna()).mean())

    if COL_REGIME_COHERENT in df.columns and df[COL_REGIME_COHERENT].dropna().size > 0:
        regime_coherence = float(df[COL_REGIME_COHERENT].dropna().mean())
    else:
        regime_coherence = float("nan")

    return {
        "country": country_key,
        "year": int(year),
        "baseload_price_used": baseload_price_used,
        "peakload_price_used": peakload_price_used,
        "offpeak_price_used": offpeak_price_used,
        "price_used_p05": used_stats["p05"],
        "price_used_p25": used_stats["p25"],
        "price_used_median": used_stats["p50"],
        "price_used_p75": used_stats["p75"],
        "price_used_p95": used_stats["p95"],
        "price_used_stddev": used_stats["std"],
        "baseload_price_obs": baseload_price_obs,
        "h_negative_obs": h_negative_obs,
        "h_below_5_obs": h_below_5_obs,
        "h_above_100_obs": h_above_100_obs,
        "h_above_200_obs": h_above_200_obs,
        "days_spread_above_50_obs": days_spread_above_50_obs,
        "avg_daily_spread_obs": avg_daily_spread_obs,
        "max_daily_spread_obs": max_daily_spread_obs,
        "capture_rate_pv": capture_rate_pv,
        "capture_rate_wind": capture_rate_wind,
        "capture_ratio_pv": capture_ratio_pv,
        "capture_ratio_wind": capture_ratio_wind,
        "h_regime_a": h_regime_a,
        "h_regime_b": h_regime_b,
        "h_regime_c": h_regime_c,
        "h_regime_d": h_regime_d,
        "sr": sr,
        "sr_hours": sr_hours,
        "far": far,
        "ir": ir,
        "ttl": ttl,
        "pv_penetration_pct_gen": pv_penetration_pct_gen,
        "wind_penetration_pct_gen": wind_penetration_pct_gen,
        "vre_penetration_pct_gen": vre_penetration_pct_gen,
        "total_generation_twh": total_generation_mwh * 1e-6,
        "total_load_twh": total_load_mwh * 1e-6,
        "total_vre_twh": total_vre_mwh * 1e-6,
        "total_surplus_twh": total_surplus_mwh * 1e-6,
        "total_surplus_unabs_twh": total_surplus_unabs_mwh * 1e-6,
        "bess_cycles_est": bess_cycles_est,
        "bess_charge_twh": bess_charge_mwh * 1e-6,
        "bess_discharge_twh": bess_discharge_mwh * 1e-6,
        "data_completeness": data_completeness,
        "regime_coherence": regime_coherence,
        "is_outlier": year in OUTLIER_YEARS,
    }
