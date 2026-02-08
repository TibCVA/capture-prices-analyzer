from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.excesum_engine import (
    _q1_threshold_table,
    _q2_slopes,
    _q3_transition_status,
    _q6_heat_cold_scope,
    build_country_conclusions,
    write_excesum_docs,
)


def _metrics_df() -> pd.DataFrame:
    rows = []
    for country in ["FR", "DE"]:
        for year, pv, cr, hneg, h5, far in [
            (2020, 12.0, 0.95, 120, 300, 0.45),
            (2021, 14.0, 0.88, 180, 450, 0.58),
            (2022, 16.0, 0.82, 260, 520, 0.62),
            (2023, 18.0, 0.78, 210, 530, 0.67),
            (2024, 20.0, 0.74, 170, 510, 0.72),
        ]:
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "sr": 0.02 + 0.001 * (year - 2020),
                    "pv_penetration_pct_gen": pv,
                    "h_negative_obs": hneg,
                    "h_below_5_obs": h5,
                    "capture_ratio_pv": cr,
                    "far": far,
                    "h_regime_a": max(0, 300 - 40 * (year - 2020)),
                    "ttl": 95 + 2 * (year - 2020),
                    "phase": "stage_2",
                    "data_completeness": 0.99,
                    "regime_coherence": 0.7,
                    "is_outlier": year == 2022,
                }
            )
    return pd.DataFrame(rows)


def test_q_tables_and_country_conclusions(tmp_path: Path) -> None:
    m = _metrics_df()
    thresholds = {
        "phase_thresholds": {
            "stage_2": {"h_negative_min": 200, "h_below_5_min": 500, "capture_ratio_pv_max": 0.80},
            "stage_3": {"far_min": 0.60},
        }
    }

    q1_detail, q1_country = _q1_threshold_table(m, thresholds)
    q2_df = _q2_slopes(m, exclude_outlier_years=(2022,))
    q3_df = _q3_transition_status(m, thresholds)
    q4_df = pd.DataFrame(
        [
            {"country": "FR", "plateau_baseline": True, "stress_found": True},
            {"country": "DE", "plateau_baseline": False, "stress_found": True},
        ]
    )
    q5_df = pd.DataFrame(
        [
            {"country": "FR", "delta_ttl_high_co2": 12.0, "delta_ttl_high_gas": 18.0},
            {"country": "DE", "delta_ttl_high_co2": 11.0, "delta_ttl_high_gas": 17.5},
        ]
    )
    q6_df = _q6_heat_cold_scope(m)

    out = build_country_conclusions(m, q1_country, q2_df, q3_df, q4_df, q5_df, q6_df)

    assert not q1_detail.empty
    assert not q2_df.empty
    assert not q3_df.empty
    assert not q6_df.empty
    assert not out.empty
    assert {"country", "q2_slope", "q3_status", "q6_status"}.issubset(out.columns)

    c_path, v_path = write_excesum_docs(
        {
            "metrics_df": m,
            "verification_rows": [
                {"check": "verification_data", "status": "PASS", "detail": "ok"},
                {"check": "verification_calc", "status": "PASS", "detail": "ok"},
            ],
        },
        docs_dir=str(tmp_path),
    )
    assert Path(c_path).exists()
    assert Path(v_path).exists()


def test_q2_slopes_excludes_outlier_year() -> None:
    m = _metrics_df()
    slopes = _q2_slopes(m, exclude_outlier_years=(2022,))
    assert (slopes["n_points"] == 4).all()
    assert np.isfinite(slopes["slope"]).all()
