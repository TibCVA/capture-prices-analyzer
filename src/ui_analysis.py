"""UI analytical helpers (non-mutating statistics used in visual commentary)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.constants import (
    COHERENCE_STATUS_THRESHOLDS,
    COL_NRL,
    COL_PRICE_DA,
    CORRELATION_STATUS_THRESHOLDS,
)


def _get_metric(metrics_row: Mapping | pd.Series | None, key: str) -> float:
    if metrics_row is None:
        return float("nan")
    try:
        if isinstance(metrics_row, pd.Series):
            return float(metrics_row.get(key, np.nan))
        return float(metrics_row.get(key, np.nan))
    except Exception:
        return float("nan")


def _classify_correlation(r: float) -> str:
    if not np.isfinite(r):
        return "unknown"
    a = abs(float(r))
    if a < CORRELATION_STATUS_THRESHOLDS["weak"]:
        return "weak"
    if a < CORRELATION_STATUS_THRESHOLDS["medium"]:
        return "medium"
    return "strong"


def _classify_coherence(score: float) -> str:
    if not np.isfinite(score):
        return "unknown"
    if score < COHERENCE_STATUS_THRESHOLDS["weak"]:
        return "weak"
    if score < COHERENCE_STATUS_THRESHOLDS["medium"]:
        return "medium"
    return "strong"


def compute_nrl_price_link_stats(df_proc: pd.DataFrame, metrics_row: Mapping | pd.Series | None) -> dict:
    """Centralized NRL vs observed-price link stats for UI.

    Returns keys:
    - pearson_r
    - pearson_r_pct
    - regime_coherence
    - regime_coherence_pct
    - n_valid
    - corr_status
    - coherence_status
    """

    if COL_NRL not in df_proc.columns or COL_PRICE_DA not in df_proc.columns:
        return {
            "pearson_r": float("nan"),
            "pearson_r_pct": float("nan"),
            "regime_coherence": float("nan"),
            "regime_coherence_pct": float("nan"),
            "n_valid": 0,
            "corr_status": "unknown",
            "coherence_status": "unknown",
        }

    valid = df_proc[[COL_NRL, COL_PRICE_DA]].dropna()
    n_valid = int(len(valid))
    pearson_r = float("nan")

    if n_valid >= 3:
        x = valid[COL_NRL].astype(float).values
        y = valid[COL_PRICE_DA].astype(float).values
        if np.nanstd(x) > 0 and np.nanstd(y) > 0:
            pearson_r = float(np.corrcoef(x, y)[0, 1])

    regime_coherence = _get_metric(metrics_row, "regime_coherence")
    if not np.isfinite(regime_coherence):
        regime_coherence = float(df_proc.attrs.get("coherence_score", np.nan))

    return {
        "pearson_r": pearson_r,
        "pearson_r_pct": pearson_r * 100.0 if np.isfinite(pearson_r) else float("nan"),
        "regime_coherence": regime_coherence,
        "regime_coherence_pct": regime_coherence * 100.0 if np.isfinite(regime_coherence) else float("nan"),
        "n_valid": n_valid,
        "corr_status": _classify_correlation(pearson_r),
        "coherence_status": _classify_coherence(regime_coherence),
    }

