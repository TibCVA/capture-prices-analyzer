"""Data loading, cache handling, and legacy migration utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.constants import (
    COL_BESS_CHARGE,
    COL_BESS_DISCHARGE,
    COL_BESS_SOC,
    COL_FLEX_EFFECTIVE,
    COL_HAS_GAP,
    COL_LOAD,
    COL_MUST_RUN,
    COL_NET_POSITION,
    COL_NRL,
    COL_PRICE_DA,
    COL_PRICE_SYNTH,
    COL_PRICE_USED,
    COL_PSH_PUMP,
    COL_REGIME,
    COL_REGIME_COHERENT,
    COL_SINK_NON_BESS,
    COL_SOLAR,
    COL_SURPLUS,
    COL_SURPLUS_UNABS,
    COL_TCA,
    COL_VRE,
    COL_WIND_ON,
    OPTIONAL_COLUMNS,
)
from src.config_loader import load_countries_config, load_scenarios, load_thresholds

logger = logging.getLogger("capture_prices.data_loader")

_RAW_DIR = Path("data/raw")
_PROCESSED_DIR = Path("data/processed")
_METRICS_DIR = Path("data/metrics")
_DIAGNOSTICS_DIR = Path("data/diagnostics")
_EXTERNAL_DIR = Path("data/external")

_LEGACY_WARNING_EMITTED = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def raw_cache_path(country_key: str, year: int) -> Path:
    return _RAW_DIR / f"{country_key}_{year}.parquet"


def processed_cache_path(
    country_key: str,
    year: int,
    must_run_mode: str,
    flex_model_mode: str,
    price_mode: str,
) -> Path:
    return _PROCESSED_DIR / (
        f"{country_key}_{year}_{must_run_mode}_{flex_model_mode}_{price_mode}.parquet"
    )


def load_raw(country_key: str, year: int) -> pd.DataFrame:
    """Lit `data/raw/{country}_{year}.parquet` ou leve FileNotFoundError."""

    path = raw_cache_path(country_key, year)
    if not path.exists():
        raise FileNotFoundError(f"Raw cache introuvable: {path}")
    return pd.read_parquet(path)


def import_csv_to_raw(filepath: str, country_key: str, year: int) -> pd.DataFrame:
    """Importe un CSV manuel et le sauvegarde dans le cache raw parquet."""

    csv_path = Path(filepath)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"timestamp", COL_LOAD, COL_SOLAR, COL_WIND_ON, COL_PRICE_DA}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV invalide: colonnes manquantes {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("CSV invalide: timestamps non parseables")

    df = df.set_index("timestamp").sort_index()
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            if col == COL_NET_POSITION:
                df[col] = np.nan
            else:
                df[col] = 0.0

    path = raw_cache_path(country_key, year)
    _ensure_dir(path.parent)
    df.to_parquet(path, index=True)
    return df


def _read_price_series(path: Path, value_candidates: Iterable[str]) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns:
        logger.warning("Fichier externe sans colonne date: %s", path)
        return None
    value_col = next((c for c in value_candidates if c in df.columns), None)
    if value_col is None:
        logger.warning("Fichier externe sans colonne prix reconnue: %s", path)
        return None
    s = pd.to_datetime(df["date"], errors="coerce")
    out = pd.Series(df[value_col].values, index=s, name=value_col).sort_index()
    out = out[~out.index.isna()]
    return out


def load_commodity_prices() -> dict:
    """Charge TTF/EUA/Coal/BESS depuis `data/external/`.

    Retourne:
      - gas_daily: Series daily EUR/MWh_th
      - co2_daily: Series daily EUR/tCO2
      - coal_daily: Series daily EUR/MWh_th ou None
      - bess_capacity: DataFrame(country, year, power_mw, energy_mwh) ou None
    """

    gas_daily = _read_price_series(
        _EXTERNAL_DIR / "ttf_daily.csv", ["price_eur_mwh", "ttf_eur_mwh", "value"]
    )
    if gas_daily is None:
        logger.warning("ttf_daily.csv absent/invalide: gas_daily=None")

    co2_daily = _read_price_series(
        _EXTERNAL_DIR / "eua_daily.csv", ["price_eur_t", "eua_eur_t", "value"]
    )
    if co2_daily is None:
        logger.warning("eua_daily.csv absent/invalide: co2_daily=None")

    coal_daily = _read_price_series(
        _EXTERNAL_DIR / "coal_daily.csv", ["price_eur_mwh", "coal_eur_mwh", "value"]
    )
    if coal_daily is None:
        logger.warning("coal_daily.csv absent/invalide: coal_daily=None")

    bess_path = _EXTERNAL_DIR / "bess_capacity.csv"
    if bess_path.exists():
        bess_capacity = pd.read_csv(bess_path)
        expected = {"country", "year", "power_mw", "energy_mwh"}
        if not expected.issubset(set(bess_capacity.columns)):
            logger.warning("bess_capacity.csv incomplet: bess_capacity=None")
            bess_capacity = None
    else:
        logger.warning("bess_capacity.csv absent: bess_capacity=None")
        bess_capacity = None

    return {
        "gas_daily": gas_daily,
        "co2_daily": co2_daily,
        "coal_daily": coal_daily,
        "bess_capacity": bess_capacity,
    }


def _legacy_candidates(country_key: str, year: int) -> list[Path]:
    """Trouve les parquets legacy `country_year_mode.parquet`."""

    pattern = f"{country_key}_{year}_*.parquet"
    candidates: list[Path] = []
    for p in _PROCESSED_DIR.glob(pattern):
        stem_parts = p.stem.split("_")
        if len(stem_parts) == 3:
            candidates.append(p)
    return sorted(candidates)


def _migrate_legacy_df(df: pd.DataFrame, price_mode: str) -> pd.DataFrame:
    """Mappe les anciennes conventions de colonnes/valeurs vers v3."""

    out = df.copy()

    # Regimes legacy
    if COL_REGIME in out.columns:
        out[COL_REGIME] = out[COL_REGIME].replace({"D_tail": "D"})

    # Colonnes flex legacy
    if COL_SINK_NON_BESS not in out.columns:
        if "flex_used_mw" in out.columns:
            out[COL_SINK_NON_BESS] = out["flex_used_mw"].fillna(0.0)
        elif "flex_capacity_mw" in out.columns:
            out[COL_SINK_NON_BESS] = out["flex_capacity_mw"].fillna(0.0)
        else:
            out[COL_SINK_NON_BESS] = 0.0

    if COL_BESS_CHARGE not in out.columns:
        out[COL_BESS_CHARGE] = 0.0
    if COL_BESS_DISCHARGE not in out.columns:
        out[COL_BESS_DISCHARGE] = 0.0
    if COL_BESS_SOC not in out.columns:
        out[COL_BESS_SOC] = 0.0

    if COL_FLEX_EFFECTIVE not in out.columns:
        out[COL_FLEX_EFFECTIVE] = out[COL_SINK_NON_BESS] + out[COL_BESS_CHARGE]

    if COL_SURPLUS_UNABS not in out.columns:
        if COL_SURPLUS in out.columns:
            out[COL_SURPLUS_UNABS] = (out[COL_SURPLUS] - out[COL_FLEX_EFFECTIVE]).clip(lower=0.0)
        else:
            out[COL_SURPLUS_UNABS] = 0.0

    if COL_PRICE_SYNTH not in out.columns:
        out[COL_PRICE_SYNTH] = np.nan

    if COL_PRICE_USED not in out.columns:
        if price_mode == "observed":
            out[COL_PRICE_USED] = out.get(COL_PRICE_DA)
        elif price_mode == "synthetic":
            out[COL_PRICE_USED] = out.get(COL_PRICE_SYNTH)
        else:
            raise ValueError(f"price_mode invalide: {price_mode}")

    if COL_REGIME_COHERENT not in out.columns:
        out[COL_REGIME_COHERENT] = np.nan

    if COL_HAS_GAP not in out.columns:
        has_load_na = out[COL_LOAD].isna() if COL_LOAD in out.columns else True
        has_price_na = out[COL_PRICE_DA].isna() if COL_PRICE_DA in out.columns else True
        out[COL_HAS_GAP] = has_load_na | has_price_na

    return out


def load_processed(
    country_key: str,
    year: int,
    must_run_mode: str,
    flex_model_mode: str,
    price_mode: str,
) -> pd.DataFrame | None:
    """Charge le cache process v3, avec migration progressive du legacy."""

    global _LEGACY_WARNING_EMITTED

    new_path = processed_cache_path(country_key, year, must_run_mode, flex_model_mode, price_mode)
    if new_path.exists():
        return pd.read_parquet(new_path)

    legacy = _legacy_candidates(country_key, year)
    if not legacy:
        return None

    if not _LEGACY_WARNING_EMITTED:
        logger.warning(
            "Migration cache legacy activee: conversion vers format v3 au premier chargement."
        )
        _LEGACY_WARNING_EMITTED = True

    old_path = legacy[0]
    df = pd.read_parquet(old_path)
    migrated = _migrate_legacy_df(df, price_mode=price_mode)
    save_processed(migrated, country_key, year, must_run_mode, flex_model_mode, price_mode)
    return migrated


def save_processed(
    df: pd.DataFrame,
    country_key: str,
    year: int,
    must_run_mode: str,
    flex_model_mode: str,
    price_mode: str,
) -> None:
    path = processed_cache_path(country_key, year, must_run_mode, flex_model_mode, price_mode)
    _ensure_dir(path.parent)
    df.to_parquet(path, index=True)


def list_processed_keys(
    must_run_mode: str | None = None,
    flex_model_mode: str | None = None,
    price_mode: str | None = None,
) -> list[tuple[str, int, str, str, str]]:
    """Retourne la liste des caches process v3 disponibles."""

    keys: list[tuple[str, int, str, str, str]] = []
    for path in sorted(_PROCESSED_DIR.glob("*.parquet")):
        parts = path.stem.split("_")
        if len(parts) != 5:
            continue
        country, year_s, mr, flex, price = parts
        try:
            year = int(year_s)
        except ValueError:
            continue

        if must_run_mode and mr != must_run_mode:
            continue
        if flex_model_mode and flex != flex_model_mode:
            continue
        if price_mode and price != price_mode:
            continue
        keys.append((country, year, mr, flex, price))
    return keys


def _sanitize_json_value(v):
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, dict):
        return {k: _sanitize_json_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_json_value(x) for x in v]
    return v


def save_metrics(metrics: dict, country: str, year: int, price_mode: str) -> None:
    _ensure_dir(_METRICS_DIR)
    path = _METRICS_DIR / f"{country}_{year}_{price_mode}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json_value(metrics), f, ensure_ascii=False, indent=2)


def load_metrics(country: str, year: int, price_mode: str) -> dict | None:
    path = _METRICS_DIR / f"{country}_{year}_{price_mode}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_diagnostics(diag: dict, country: str, year: int, price_mode: str) -> None:
    _ensure_dir(_DIAGNOSTICS_DIR)
    path = _DIAGNOSTICS_DIR / f"{country}_{year}_{price_mode}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json_value(diag), f, ensure_ascii=False, indent=2)


def load_diagnostics(country: str, year: int, price_mode: str) -> dict | None:
    path = _DIAGNOSTICS_DIR / f"{country}_{year}_{price_mode}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
