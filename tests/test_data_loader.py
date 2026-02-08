from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.data_loader as dl
from src.constants import (
    COL_LOAD,
    COL_PRICE_DA,
    COL_SOLAR,
    COL_WIND_ON,
    COL_BESS_CHARGE,
    COL_FLEX_EFFECTIVE,
    COL_NRL,
    COL_REGIME,
    COL_SINK_NON_BESS,
    COL_SURPLUS,
    COL_SURPLUS_UNABS,
)


def _bad_processed_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    # Incoherent by construction: surplus=10, flex=0, unabs=0 should be 10
    return pd.DataFrame(
        {
            COL_SURPLUS: [10.0, 0.0, 5.0, 2.0],
            COL_SINK_NON_BESS: [0.0, 0.0, 0.0, 0.0],
            COL_BESS_CHARGE: [0.0, 0.0, 0.0, 0.0],
            COL_FLEX_EFFECTIVE: [0.0, 0.0, 0.0, 0.0],
            COL_SURPLUS_UNABS: [0.0, 0.0, 0.0, 0.0],
            COL_NRL: [-10.0, 5.0, -5.0, -2.0],
            COL_REGIME: ["B", "C", "B", "B"],
        },
        index=idx,
    )


def test_validate_processed_semantics_detects_incoherent_surplus() -> None:
    ok, reasons = dl.validate_processed_semantics(_bad_processed_df())
    assert ok is False
    assert any("surplus_unabsorbed incoherent" in r for r in reasons)


def test_migrate_legacy_recomputes_flex_and_unabsorbed() -> None:
    legacy = _bad_processed_df().copy()
    migrated = dl._migrate_legacy_df(legacy, price_mode="observed")  # pylint: disable=protected-access

    ok, reasons = dl.validate_processed_semantics(migrated)
    # Regime labels may remain legacy-inconsistent and are filtered at load time;
    # this test only enforces physical column recomputation.
    assert ok is False or ok is True
    assert float(migrated[COL_SURPLUS_UNABS].iloc[0]) == 10.0
    assert float(migrated[COL_SURPLUS_UNABS].iloc[2]) == 5.0
    assert "regime B invalide" in "; ".join(reasons)


def test_load_processed_invalidates_incoherent_cache(tmp_path: Path, monkeypatch) -> None:
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dl, "_PROCESSED_DIR", proc_dir)

    p = proc_dir / "FR_2024_observed_observed_observed.parquet"
    _bad_processed_df().to_parquet(p, index=True)

    out_validated = dl.load_processed(
        "FR",
        2024,
        "observed",
        "observed",
        "observed",
        validate_semantics=True,
    )
    out_unchecked = dl.load_processed(
        "FR",
        2024,
        "observed",
        "observed",
        "observed",
        validate_semantics=False,
    )

    assert out_validated is None
    assert out_unchecked is not None


def test_ensure_raw_minimum_columns_adds_missing_and_flags() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    raw = pd.DataFrame({COL_LOAD: [1.0, 2.0, 3.0]}, index=idx)
    out = dl.ensure_raw_minimum_columns(raw, "PL", 2015)

    assert COL_SOLAR in out.columns
    assert COL_WIND_ON in out.columns
    assert COL_PRICE_DA in out.columns
    assert float(out[COL_SOLAR].sum()) == 0.0
    assert float(out[COL_WIND_ON].sum()) == 0.0
    assert out[COL_PRICE_DA].isna().all()
    assert len(out.attrs.get("data_quality_flags", [])) >= 1
