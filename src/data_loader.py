import os
import glob
import json
import logging
import math
import yaml
import pandas as pd
from src.constants import *

logger = logging.getLogger("capture_prices.data_loader")


def load_raw(country_key: str, year: int) -> pd.DataFrame | None:
    """Charge depuis data/raw/{country}_{year}.parquet. Retourne None si absent."""
    path = os.path.join("data", "raw", f"{country_key}_{year}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    logger.warning(f"Pas de donnees pour {country_key}/{year}")
    return None


def load_processed(country_key: str, year: int, mode: str) -> pd.DataFrame | None:
    """Charge depuis data/processed/{country}_{year}_{mode}.parquet."""
    path = os.path.join("data", "processed", f"{country_key}_{year}_{mode}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_processed(df: pd.DataFrame, country_key: str, year: int, mode: str):
    """Sauvegarde un DataFrame processe."""
    path = os.path.join("data", "processed", f"{country_key}_{year}_{mode}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=True)


def load_country_config(country_key: str) -> dict:
    """Charge la config d'un pays depuis countries.yaml."""
    config_path = os.path.join("config", "countries.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        all_config = yaml.safe_load(f)
    countries = all_config.get('countries', {})
    if country_key not in countries:
        raise ValueError(f"Pays inconnu: {country_key}. Disponibles: {list(countries.keys())}")
    return countries[country_key]


def load_all_countries_config() -> dict:
    """Charge toute la config pays."""
    config_path = os.path.join("config", "countries.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        all_config = yaml.safe_load(f)
    return all_config.get('countries', {})


def load_thresholds() -> dict:
    """Charge les seuils de diagnostic."""
    path = os.path.join("config", "thresholds.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_scenarios_config() -> dict:
    """Charge les scenarios predefinis."""
    path = os.path.join("config", "scenarios.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ==================== PERSISTANCE METRICS & DIAGNOSTICS ====================

def _sanitize_for_json(obj):
    """Convertit les types numpy/pandas en types Python natifs pour JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, (float,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if hasattr(obj, 'item'):  # numpy scalar
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


def save_metrics(metrics: dict, country: str, year: int, mode: str):
    """Sauvegarde les metriques annuelles en JSON."""
    dirpath = os.path.join("data", "metrics")
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"{country}_{year}_{mode}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(metrics), f, ensure_ascii=False)


def load_metrics(country: str, year: int, mode: str) -> dict | None:
    """Charge les metriques annuelles depuis JSON."""
    path = os.path.join("data", "metrics", f"{country}_{year}_{mode}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_diagnostics(diag: dict, country: str, year: int, mode: str):
    """Sauvegarde le diagnostic de phase en JSON."""
    dirpath = os.path.join("data", "diagnostics")
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"{country}_{year}_{mode}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(diag), f, ensure_ascii=False)


def load_diagnostics(country: str, year: int, mode: str) -> dict | None:
    """Charge le diagnostic de phase depuis JSON."""
    path = os.path.join("data", "diagnostics", f"{country}_{year}_{mode}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def scan_cached_processed(mode: str = "observed") -> list[tuple[str, int]]:
    """Scanne data/processed/ et retourne les (country, year) disponibles pour le mode donne."""
    pattern = os.path.join("data", "processed", f"*_{mode}.parquet")
    results = []
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        # Format: {COUNTRY}_{YEAR}_{MODE}.parquet
        parts = fname.replace(".parquet", "").rsplit("_", 2)
        if len(parts) == 3:
            country, year_str, _ = parts
            try:
                results.append((country, int(year_str)))
            except ValueError:
                continue
    return sorted(results)
