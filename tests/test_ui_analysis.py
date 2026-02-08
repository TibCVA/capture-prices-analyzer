from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import COL_NRL, COL_PRICE_DA
from src.ui_analysis import compute_nrl_price_link_stats


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
