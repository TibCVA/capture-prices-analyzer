from __future__ import annotations

import numpy as np
import pandas as pd

from src.state_adapter import (
    coerce_numeric_columns,
    ensure_plot_columns,
    metrics_to_dataframe,
    normalize_metrics_record,
)


def test_normalize_metrics_record_maps_legacy_keys() -> None:
    legacy = {
        "h_negative": 12,
        "h_below_5": 34,
        "h_regime_d_tail": 56,
        "far_structural": 0.42,
        "pv_share": 0.11,
        "wind_share": 0.22,
        "vre_share": 0.33,
    }

    out = normalize_metrics_record(legacy)

    assert out["h_negative_obs"] == 12
    assert out["h_below_5_obs"] == 34
    assert out["h_regime_d"] == 56
    assert out["far"] == 0.42
    assert np.isclose(out["pv_penetration_pct_gen"], 11.0)
    assert np.isclose(out["wind_penetration_pct_gen"], 22.0)
    assert np.isclose(out["vre_penetration_pct_gen"], 33.0)


def test_metrics_to_dataframe_builds_phase_and_filters_price_mode() -> None:
    state = {
        "metrics": {
            ("FR", 2024, "observed"): {"sr": 0.1, "h_negative": 10},
            ("FR", 2024, "synthetic"): {"sr": 0.2, "h_negative": 20},
        },
        "diagnostics": {("FR", 2024): {"phase": "stage_2", "confidence": 0.75, "score": 4}},
    }

    out = metrics_to_dataframe(state, "observed")

    assert len(out) == 1
    assert out.iloc[0]["country"] == "FR"
    assert int(out.iloc[0]["year"]) == 2024
    assert out.iloc[0]["phase"] == "stage_2"
    assert np.isclose(float(out.iloc[0]["sr"]), 0.1)
    assert int(out.iloc[0]["h_negative_obs"]) == 10


def test_ensure_plot_columns_adds_missing_with_nan() -> None:
    df = pd.DataFrame({"a": [1, 2]})

    out = ensure_plot_columns(df, ["a", "b", "c"], with_notice=True)

    assert list(out.columns) == ["a", "b", "c"]
    assert out["b"].isna().all()
    assert out["c"].isna().all()
    assert out.attrs["_missing_plot_columns"] == ["b", "c"]


def test_coerce_numeric_columns_handles_invalid_values() -> None:
    df = pd.DataFrame({"a": ["1", "x"], "b": [2, 3], "c": ["4.5", None]})

    out = coerce_numeric_columns(df, ["a", "c"])

    assert np.isclose(out.loc[0, "a"], 1.0)
    assert np.isnan(out.loc[1, "a"])
    assert np.isclose(out.loc[0, "c"], 4.5)
