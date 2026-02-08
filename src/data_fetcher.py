"""ENTSO-E data fetcher with v3.0 harmonization and caching."""

from __future__ import annotations

import calendar
import inspect
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from entsoe import EntsoePandasClient

from src.config_loader import resolve_entsoe_code
from src.constants import (
    COL_BIOMASS,
    COL_COAL,
    COL_GAS,
    COL_HAS_GAP,
    COL_HYDRO_RES,
    COL_HYDRO_ROR,
    COL_LIGNITE,
    COL_LOAD,
    COL_NET_POSITION,
    COL_NUCLEAR,
    COL_OTHER,
    COL_PRICE_DA,
    COL_PSH_GEN,
    COL_PSH_PUMP,
    COL_SOLAR,
    COL_WIND_OFF,
    COL_WIND_ON,
    ENTSOE_GEN_ALIASES,
    ENTSOE_GEN_MAPPING,
    HOURS_LEAP,
    HOURS_YEAR,
    OPTIONAL_COLUMNS,
)
from src.time_utils import to_utc_index

logger = logging.getLogger("capture_prices.data_fetcher")

_RAW_DIR = Path("data/raw")

_GENERATION_COLS = [
    COL_SOLAR,
    COL_WIND_ON,
    COL_WIND_OFF,
    COL_NUCLEAR,
    COL_LIGNITE,
    COL_COAL,
    COL_GAS,
    COL_HYDRO_ROR,
    COL_HYDRO_RES,
    COL_PSH_GEN,
    COL_PSH_PUMP,
    COL_BIOMASS,
    COL_OTHER,
]


class _MissingDataError(RuntimeError):
    pass


def _api_call_with_retry(func, *args, **kwargs):
    delays = [5, 15, 30]
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt >= len(delays):
                raise
            wait_s = delays[attempt]
            attempt += 1
            logger.warning(
                "ENTSO-E call failed (attempt %s/%s), retry in %ss",
                attempt,
                len(delays),
                wait_s,
            )
            time.sleep(wait_s)


def _resolve_api_key() -> str:
    env_key = os.getenv("ENTSOE_API_KEY")
    if env_key:
        return env_key
    dotenv_key = dotenv_values(".env").get("ENTSOE_API_KEY")
    if dotenv_key:
        return str(dotenv_key)
    raise RuntimeError("ENTSOE_API_KEY manquante (env ou .env)")


def _extract_load_series(raw_load: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(raw_load, pd.Series):
        s = raw_load.copy()
        s.name = COL_LOAD
        return s

    if isinstance(raw_load, pd.DataFrame):
        candidates = [c for c in raw_load.columns if "Actual Load" in str(c)]
        if not candidates:
            raise NotImplementedError(
                "Spec ambigue : format load ENTSO-E inattendu (section G.3 Etape 3A)"
            )
        s = raw_load[candidates[0]].copy()
        s.name = COL_LOAD
        return s

    raise NotImplementedError("Spec ambigue : type load ENTSO-E non supporte (section G.3 Etape 3A)")


def _normalize_gen_label(label: str) -> str:
    if label in ENTSOE_GEN_ALIASES:
        return ENTSOE_GEN_ALIASES[label]
    return label


def _extract_generation_columns(raw_gen: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}

    def _accumulate(key: str, values: pd.Series) -> None:
        if key not in out:
            out[key] = values.astype(float)
        else:
            out[key] = out[key].add(values.astype(float), fill_value=0.0)

    if isinstance(raw_gen.columns, pd.MultiIndex):
        for gen_type, flow_type in raw_gen.columns:
            gen_type = _normalize_gen_label(str(gen_type))
            mapped = ENTSOE_GEN_MAPPING.get(gen_type)
            if mapped is None:
                logger.warning("Generation type ignore (non mappe): %s", gen_type)
                continue

            series = raw_gen[(gen_type, flow_type)]
            flow_type = str(flow_type)

            if mapped == "_psh_dispatch":
                if "Actual Aggregated" in flow_type:
                    _accumulate(COL_PSH_GEN, series.abs())
                elif "Actual Consumption" in flow_type:
                    _accumulate(COL_PSH_PUMP, series.abs())
                continue

            if "Actual Aggregated" in flow_type:
                _accumulate(mapped, series)
    else:
        for col in raw_gen.columns:
            col_s = str(col)
            mapped = ENTSOE_GEN_MAPPING.get(_normalize_gen_label(col_s))
            if mapped is None:
                logger.warning("Generation type ignore (non mappe): %s", col_s)
                continue
            if mapped == "_psh_dispatch":
                if "Consumption" in col_s:
                    _accumulate(COL_PSH_PUMP, raw_gen[col].abs())
                else:
                    _accumulate(COL_PSH_GEN, raw_gen[col].abs())
            else:
                _accumulate(mapped, raw_gen[col])

    return out


def _is_local_auction_issue(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "local auction" in msg or "query_day_ahead_prices_local" in msg


def _fetch_prices(client: EntsoePandasClient, entsoe_code: str, start, end) -> pd.Series:
    try:
        prices = _api_call_with_retry(client.query_day_ahead_prices, entsoe_code, start=start, end=end)
        if prices is None or len(prices) == 0:
            raise _MissingDataError("query_day_ahead_prices vide")
        return prices.rename(COL_PRICE_DA)
    except Exception as exc:
        if isinstance(exc, _MissingDataError) or _is_local_auction_issue(exc):
            if hasattr(client, "query_day_ahead_prices_local"):
                prices = _api_call_with_retry(
                    client.query_day_ahead_prices_local, entsoe_code, start=start, end=end
                )
                if prices is None or len(prices) == 0:
                    raise _MissingDataError("query_day_ahead_prices_local vide")
                return prices.rename(COL_PRICE_DA)
            raise NotImplementedError(
                "Prix DA local auction non supporte par la version entsoe-py installee"
            ) from exc
        raise


def _fetch_net_position(client: EntsoePandasClient, entsoe_code: str, start, end) -> pd.Series | None:
    try:
        sig = inspect.signature(client.query_net_position)
        if "dayahead" in sig.parameters:
            series = _api_call_with_retry(
                client.query_net_position,
                entsoe_code,
                start=start,
                end=end,
                dayahead=True,
            )
        else:
            series = _api_call_with_retry(client.query_net_position, entsoe_code, start=start, end=end)
        if series is None or len(series) == 0:
            return None
        return series.rename(COL_NET_POSITION)
    except Exception as exc:  # graceful fallback per spec
        logger.warning("Net position indisponible (%s): %s", entsoe_code, exc)
        return None


def fetch_country_year(
    country_key: str,
    year: int,
    countries_cfg: dict,
    force: bool = False,
) -> pd.DataFrame:
    """Recupere et harmonise les donnees ENTSO-E pour un pays/annee."""

    cache_path = _RAW_DIR / f"{country_key}_{year}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    if country_key not in countries_cfg:
        raise KeyError(f"Pays inconnu: {country_key}")

    country_cfg = countries_cfg[country_key]
    country_tz = country_cfg["timezone"]
    entsoe_code = resolve_entsoe_code(country_key, year, countries_cfg)

    api_key = _resolve_api_key()
    client = EntsoePandasClient(api_key=api_key)

    start = pd.Timestamp(f"{year}0101", tz=country_tz)
    end = pd.Timestamp(f"{year + 1}0101", tz=country_tz)

    # A) Load
    raw_load = _api_call_with_retry(client.query_load, entsoe_code, start=start, end=end)
    load = _extract_load_series(raw_load)

    # B) Generation
    raw_gen = _api_call_with_retry(client.query_generation, entsoe_code, start=start, end=end, psr_type=None)
    gen_cols = _extract_generation_columns(raw_gen)

    # C) Prices
    prices = _fetch_prices(client, entsoe_code, start, end)

    # D) Net position
    net_position = _fetch_net_position(client, entsoe_code, start, end)

    # Assemble union index
    series_list = [load, prices, *gen_cols.values()]
    if net_position is not None:
        series_list.append(net_position)

    idx = None
    for s in series_list:
        idx = s.index if idx is None else idx.union(s.index)
    if idx is None:
        raise RuntimeError(f"Aucune serie recuperee pour {country_key}/{year}")

    df = pd.DataFrame(index=idx)
    df[COL_LOAD] = load.reindex(idx)
    for col, s in gen_cols.items():
        df[col] = s.reindex(idx)
    df[COL_PRICE_DA] = prices.reindex(idx)
    if net_position is not None:
        df[COL_NET_POSITION] = net_position.reindex(idx)

    # Missing optionals
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            if col == COL_NET_POSITION:
                df[col] = np.nan
            else:
                df[col] = 0.0

    # Keep load/price as-is; fill other generation columns to 0 if absent
    for col in _GENERATION_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Normalize timezone + hourly
    df.index = to_utc_index(pd.DatetimeIndex(df.index))

    physical_cols = [COL_LOAD, *_GENERATION_COLS]
    physical_resampled = df[physical_cols].resample("h").mean()
    price_resampled = df[[COL_PRICE_DA]].resample("h").mean()
    net_resampled = df[[COL_NET_POSITION]].resample("h").mean()

    df = physical_resampled.join(price_resampled, how="outer").join(net_resampled, how="left")

    # Gaps management
    df[physical_cols] = df[physical_cols].interpolate(limit=3)
    df[COL_HAS_GAP] = df[COL_LOAD].isna() | df[COL_PRICE_DA].isna()

    # Validations (warnings only)
    if df[COL_LOAD].isna().all():
        raise RuntimeError("Absence totale de load (spec I.1)")

    n_neg_load = int((df[COL_LOAD] < 0).sum())
    if n_neg_load > 0:
        logger.warning("%s/%s: %s heures avec load < 0", country_key, year, n_neg_load)

    completeness_load = float(df[COL_LOAD].notna().mean())
    completeness_price = float(df[COL_PRICE_DA].notna().mean())
    if completeness_load < 0.90:
        logger.warning("%s/%s: completude load %.1f%% < 90%%", country_key, year, 100 * completeness_load)
    if completeness_price < 0.90:
        logger.warning("%s/%s: completude prix %.1f%% < 90%%", country_key, year, 100 * completeness_price)

    expected_hours = HOURS_LEAP if calendar.isleap(year) else HOURS_YEAR
    if len(df) < 0.95 * expected_hours:
        logger.warning(
            "%s/%s: %s heures (attendu %s)", country_key, year, len(df), expected_hours
        )

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=True)
    return df
