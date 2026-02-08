"""UI state schema adapter and legacy compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def normalize_metrics_record(m: Mapping | None) -> dict:
    """Normalize legacy/v3 metric records to the canonical v3 UI schema."""

    if m is None:
        return {}

    out = dict(m)

    # Legacy aliases -> canonical v3 keys
    if "h_negative_obs" not in out and "h_negative" in out:
        out["h_negative_obs"] = out["h_negative"]
    if "h_below_5_obs" not in out and "h_below_5" in out:
        out["h_below_5_obs"] = out["h_below_5"]
    if "h_above_100_obs" not in out and "h_above_100" in out:
        out["h_above_100_obs"] = out["h_above_100"]
    if "h_above_200_obs" not in out and "h_above_200" in out:
        out["h_above_200_obs"] = out["h_above_200"]

    if "h_regime_d" not in out and "h_regime_d_tail" in out:
        out["h_regime_d"] = out["h_regime_d_tail"]

    if "far" not in out and "far_structural" in out:
        out["far"] = out["far_structural"]

    if "pv_penetration_pct_gen" not in out and "pv_share" in out:
        out["pv_penetration_pct_gen"] = _as_float(out["pv_share"]) * 100.0
    if "wind_penetration_pct_gen" not in out and "wind_share" in out:
        out["wind_penetration_pct_gen"] = _as_float(out["wind_share"]) * 100.0
    if "vre_penetration_pct_gen" not in out and "vre_share" in out:
        out["vre_penetration_pct_gen"] = _as_float(out["vre_share"]) * 100.0

    if "phase" not in out and "phase_number" in out:
        try:
            out["phase"] = f"stage_{int(out['phase_number'])}"
        except Exception:
            out["phase"] = "unknown"

    return out


def ensure_plot_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> pd.DataFrame:
    """Guarantee required plotting columns exist, filling missing with NaN."""

    out = df.copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def metrics_to_dataframe(state: Mapping | None, price_mode: str | None) -> pd.DataFrame:
    """Build a normalized metrics dataframe from `st.session_state.state`."""

    if not state:
        return pd.DataFrame()

    metrics = state.get("metrics", {})
    diagnostics = state.get("diagnostics", {})

    rows: list[dict] = []
    for key, val in metrics.items():
        if not isinstance(key, tuple):
            continue

        if len(key) >= 3:
            country, year, mode = key[0], key[1], key[2]
        elif len(key) == 2:
            country, year = key
            mode = price_mode or "observed"
        else:
            continue

        if price_mode is not None and mode != price_mode:
            continue

        rec = normalize_metrics_record(val if isinstance(val, Mapping) else {})
        diag = diagnostics.get((country, year), {}) if isinstance(diagnostics, Mapping) else {}

        row = {
            "country": country,
            "year": int(year),
            "price_mode": mode,
            "phase": diag.get("phase", rec.get("phase", "unknown")),
            "phase_confidence": diag.get("confidence", np.nan),
            "phase_score": diag.get("score", np.nan),
        }
        row.update(rec)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["country", "year", "price_mode"]).reset_index(drop=True)
    return out


def normalize_state_metrics(state: Mapping | None) -> None:
    """In-place normalization for state['metrics'] records."""

    if not state or "metrics" not in state or not isinstance(state["metrics"], Mapping):
        return

    normalized = {}
    for key, val in state["metrics"].items():
        if isinstance(val, Mapping):
            normalized[key] = normalize_metrics_record(val)
        else:
            normalized[key] = val
    state["metrics"] = normalized
