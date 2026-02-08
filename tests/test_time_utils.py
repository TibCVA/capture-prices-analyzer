from __future__ import annotations

import pandas as pd

from src.time_utils import as_local, is_weekday_local, peak_mask, to_utc_index


def test_to_utc_index_from_naive():
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    out = to_utc_index(idx)
    assert out.tz is not None
    assert str(out.tz) == "UTC"


def test_as_local_converts_timezone():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    local = as_local(df, "Europe/Paris")
    assert str(local.index.tz) == "Europe/Paris"


def test_peak_mask_weekday_hours():
    idx = pd.date_range("2024-01-02", periods=24, freq="h", tz="UTC")
    mask = peak_mask(idx, "Europe/Paris")
    assert mask.dtype == bool
    assert mask.sum() >= 10


def test_is_weekday_local_true_false():
    ts_weekday = pd.Timestamp("2024-01-03T12:00:00+01:00")
    ts_weekend = pd.Timestamp("2024-01-06T12:00:00+01:00")
    assert is_weekday_local(ts_weekday)
    assert not is_weekday_local(ts_weekend)
