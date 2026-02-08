"""UI-level analytical helpers (robust stats + Q4 diagnostics)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.constants import (
    COL_BESS_CHARGE,
    COL_PRICE_DA,
    COL_NRL,
    COL_REGIME,
    COL_SINK_NON_BESS,
    COL_SURPLUS,
    COL_SURPLUS_UNABS,
)
from src.metrics import compute_annual_metrics
from src.scenario_engine import apply_scenario

_CORR_THRESH = {"weak": 0.20, "medium": 0.45}
_COH_THRESH = {"weak": 0.55, "medium": 0.70}


def _metric_lookup(metrics_row: Mapping | pd.Series | None, key: str) -> float:
    if metrics_row is None:
        return float("nan")
    try:
        value = metrics_row.get(key)  # type: ignore[attr-defined]
    except Exception:
        value = None
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _corr_status(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    v = abs(float(value))
    if v < float(_CORR_THRESH["weak"]):
        return "weak"
    if v < float(_CORR_THRESH["medium"]):
        return "medium"
    return "strong"


def _coh_status(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    v = float(value)
    if v < float(_COH_THRESH["weak"]):
        return "weak"
    if v < float(_COH_THRESH["medium"]):
        return "medium"
    return "strong"


def _numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def compute_nrl_price_link_stats(
    df_proc: pd.DataFrame,
    metrics_row: Mapping | pd.Series | None,
) -> dict:
    """Compute robust UI stats for the NRL-vs-observed-price validation block."""

    if not isinstance(df_proc, pd.DataFrame) or COL_NRL not in df_proc.columns or COL_PRICE_DA not in df_proc.columns:
        pearson_r = float("nan")
        n_valid = 0
    else:
        pairs = df_proc[[COL_NRL, COL_PRICE_DA]].dropna()
        n_valid = int(len(pairs))
        if n_valid >= 2:
            x = pairs[COL_NRL].to_numpy(dtype=float)
            y = pairs[COL_PRICE_DA].to_numpy(dtype=float)
            if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
                pearson_r = float("nan")
            else:
                pearson_r = float(np.corrcoef(x, y)[0, 1])
        else:
            pearson_r = float("nan")

    regime_coherence = _metric_lookup(metrics_row, "regime_coherence")
    pearson_r_pct = pearson_r * 100.0 if np.isfinite(pearson_r) else float("nan")
    regime_coherence_pct = regime_coherence * 100.0 if np.isfinite(regime_coherence) else float("nan")

    return {
        "pearson_r": pearson_r,
        "pearson_r_pct": pearson_r_pct,
        "regime_coherence": regime_coherence,
        "regime_coherence_pct": regime_coherence_pct,
        "n_valid": n_valid,
        "corr_status": _corr_status(pearson_r),
        "coherence_status": _coh_status(regime_coherence),
    }


def compute_q4_plateau_diagnostics(df_scenario_base: pd.DataFrame) -> dict:
    """Summarize the physical state that can explain a flat Q4 battery sweep."""

    if not isinstance(df_scenario_base, pd.DataFrame) or df_scenario_base.empty:
        return {
            "total_surplus_twh": float("nan"),
            "total_surplus_unabs_twh": float("nan"),
            "sink_non_bess_mean_mw": float("nan"),
            "bess_charge_twh": float("nan"),
            "h_regime_a": float("nan"),
            "far": float("nan"),
            "n_hours": 0,
        }

    surplus = _numeric_series(df_scenario_base, COL_SURPLUS)
    surplus_unabs = _numeric_series(df_scenario_base, COL_SURPLUS_UNABS)
    sink_non_bess = _numeric_series(df_scenario_base, COL_SINK_NON_BESS)
    bess_charge = _numeric_series(df_scenario_base, COL_BESS_CHARGE)
    regime = (
        df_scenario_base[COL_REGIME]
        if COL_REGIME in df_scenario_base.columns
        else pd.Series("C", index=df_scenario_base.index)
    )

    absorbed = float(np.minimum(surplus.values, (sink_non_bess + bess_charge).values).sum())
    surplus_total = float(surplus.sum())
    far = float("nan") if surplus_total <= 0.0 else float(absorbed / surplus_total)

    return {
        "total_surplus_twh": surplus_total * 1e-6,
        "total_surplus_unabs_twh": float(surplus_unabs.sum()) * 1e-6,
        "sink_non_bess_mean_mw": float(sink_non_bess.mean()),
        "bess_charge_twh": float(bess_charge.sum()) * 1e-6,
        "h_regime_a": int((regime == "A").sum()),
        "far": far,
        "n_hours": int(len(df_scenario_base)),
    }


def _q4_effect_identifiable(metrics: Mapping[str, float | int | None]) -> bool:
    far = metrics.get("far", np.nan)
    h_regime_a = metrics.get("h_regime_a", np.nan)
    try:
        far_f = float(far)
    except Exception:
        far_f = np.nan
    try:
        h_a = float(h_regime_a)
    except Exception:
        h_a = np.nan
    return bool((np.isfinite(h_a) and h_a > 0.0) or (np.isfinite(far_f) and far_f < 0.995))


def find_q4_stress_reference(
    df_base_processed: pd.DataFrame,
    country_key: str,
    year: int,
    country_cfg: dict,
    thresholds: dict,
    commodities: dict,
    max_delta_pv_gw: float = 40,
    step_gw: float = 2,
    base_overrides: dict | None = None,
) -> dict:
    """Find the minimum deterministic PV stress where BESS effect becomes identifiable."""

    tested_rows: list[dict] = []

    overrides = dict(base_overrides or {})

    def _run(delta_pv_gw: float) -> tuple[pd.DataFrame, dict]:
        scenario_params = dict(overrides)
        scenario_params["delta_pv_gw"] = float(scenario_params.get("delta_pv_gw", 0.0)) + float(delta_pv_gw)
        df_s = apply_scenario(
            df_base_processed=df_base_processed,
            country_key=country_key,
            year=year,
            country_cfg=country_cfg,
            thresholds=thresholds,
            commodities=commodities,
            scenario_params=scenario_params,
            price_mode="synthetic",
        )
        m = compute_annual_metrics(df_s, country_key, year, country_cfg)
        return df_s, m

    max_delta = max(0.0, float(max_delta_pv_gw))
    step = max(0.5, float(step_gw))
    grid = np.arange(0.0, max_delta + 1e-9, step)

    selected_df = None
    selected_metrics: dict | None = None
    selected_delta: float | None = None

    for delta in grid:
        df_s, m = _run(float(delta))
        tested_rows.append(
            {
                "delta_pv_gw": float(delta),
                "far": float(m.get("far", np.nan)),
                "h_regime_a": float(m.get("h_regime_a", np.nan)),
                "sr": float(m.get("sr", np.nan)),
                "total_surplus_twh": float(m.get("total_surplus_twh", np.nan)),
                "total_surplus_unabs_twh": float(m.get("total_surplus_unabs_twh", np.nan)),
            }
        )
        if _q4_effect_identifiable(m):
            selected_df = df_s
            selected_metrics = m
            selected_delta = float(delta)
            break

    tested_df = pd.DataFrame(tested_rows)
    if selected_df is None or selected_metrics is None or selected_delta is None:
        return {
            "found": False,
            "delta_pv_gw": float("nan"),
            "df_reference": None,
            "metrics": None,
            "diagnostics": None,
            "tested_grid": tested_df,
        }

    return {
        "found": True,
        "delta_pv_gw": selected_delta,
        "df_reference": selected_df,
        "metrics": selected_metrics,
        "diagnostics": compute_q4_plateau_diagnostics(selected_df),
        "tested_grid": tested_df,
    }


def compute_q4_bess_sweep(
    df_base_processed: pd.DataFrame,
    country_key: str,
    year: int,
    country_cfg: dict,
    thresholds: dict,
    commodities: dict,
    sweep_gw: Sequence[float],
    reference_overrides: dict | None,
) -> pd.DataFrame:
    """Run deterministic Q4 battery sweep on a chosen reference case."""

    rows: list[dict] = []
    base_overrides = dict(reference_overrides or {})

    for gw in sweep_gw:
        delta = float(gw)
        params = dict(base_overrides)
        params["delta_bess_power_gw"] = delta
        if "delta_bess_energy_gwh" not in params:
            params["delta_bess_energy_gwh"] = delta * 4.0

        df_s = apply_scenario(
            df_base_processed=df_base_processed,
            country_key=country_key,
            year=year,
            country_cfg=country_cfg,
            thresholds=thresholds,
            commodities=commodities,
            scenario_params=params,
            price_mode="synthetic",
        )
        m = compute_annual_metrics(df_s, country_key, year, country_cfg)
        rows.append(
            {
                "delta_bess_power_gw": delta,
                "delta_bess_energy_gwh": float(params["delta_bess_energy_gwh"]),
                "far": float(m.get("far", np.nan)),
                "h_regime_a": float(m.get("h_regime_a", np.nan)),
                "sr": float(m.get("sr", np.nan)),
                "ttl": float(m.get("ttl", np.nan)),
                "capture_ratio_pv": float(m.get("capture_ratio_pv", np.nan)),
                "total_surplus_twh": float(m.get("total_surplus_twh", np.nan)),
                "total_surplus_unabs_twh": float(m.get("total_surplus_unabs_twh", np.nan)),
                "bess_charge_twh": float(m.get("bess_charge_twh", np.nan)),
                "bess_discharge_twh": float(m.get("bess_discharge_twh", np.nan)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "delta_bess_power_gw",
                "delta_bess_energy_gwh",
                "far",
                "h_regime_a",
                "sr",
                "ttl",
                "capture_ratio_pv",
                "total_surplus_twh",
                "total_surplus_unabs_twh",
                "bess_charge_twh",
                "bess_discharge_twh",
            ]
        )
    return out.sort_values("delta_bess_power_gw").reset_index(drop=True)
