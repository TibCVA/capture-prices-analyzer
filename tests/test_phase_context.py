from __future__ import annotations

import pandas as pd

from src.phase_context import compute_h_negative_declining_flags


def test_compute_h_negative_declining_flags_three_year_rule() -> None:
    df = pd.DataFrame(
        [
            {"country": "FR", "year": 2022, "h_negative_obs": 300},
            {"country": "FR", "year": 2023, "h_negative_obs": 260},
            {"country": "FR", "year": 2024, "h_negative_obs": 220},
        ]
    )
    out = compute_h_negative_declining_flags(df)
    row_2024 = out[(out["country"] == "FR") & (out["year"] == 2024)].iloc[0]
    assert bool(row_2024["h_negative_declining"]) is True
    assert float(row_2024["h_negative_recent_peak_3y"]) == 300.0


def test_compute_h_negative_declining_flags_two_year_rule() -> None:
    df = pd.DataFrame(
        [
            {"country": "DE", "year": 2023, "h_negative_obs": 90},
            {"country": "DE", "year": 2024, "h_negative_obs": 120},
        ]
    )
    out = compute_h_negative_declining_flags(df)
    row_2024 = out[(out["country"] == "DE") & (out["year"] == 2024)].iloc[0]
    assert bool(row_2024["h_negative_declining"]) is False
    assert float(row_2024["h_negative_recent_peak_3y"]) == 120.0


def test_compute_h_negative_declining_flags_single_year_defaults_false() -> None:
    df = pd.DataFrame([{"country": "ES", "year": 2024, "h_negative_obs": 50}])
    out = compute_h_negative_declining_flags(df)
    assert bool(out.iloc[0]["h_negative_declining"]) is False
