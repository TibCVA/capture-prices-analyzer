import os
import calendar
import logging
import time
import pandas as pd
import numpy as np
from entsoe import EntsoePandasClient
from src.constants import *

logger = logging.getLogger("capture_prices.data_fetcher")


def _api_call_with_retry(func, *args, retries=4, delay=15, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"API attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


def fetch_country_year(country_key: str, year: int,
                       api_key: str, force: bool = False) -> pd.DataFrame:
    """
    Telecharge load, generation, prix, net position pour un pays/annee.
    Retourne un DataFrame horaire en UTC avec colonnes harmonisees.
    Stocke le resultat en Parquet dans data/raw/.
    """
    # Etape 1 — Cache check
    cache_path = os.path.join("data", "raw", f"{country_key}_{year}.parquet")
    if os.path.exists(cache_path) and not force:
        logger.info(f"Cache hit: {cache_path}")
        return pd.read_parquet(cache_path)

    # Etape 2 — Resolution du code ENTSO-E
    entsoe_code = COUNTRY_ENTSOE[country_key]
    if country_key in COUNTRY_CODE_PERIODS:
        for period in COUNTRY_CODE_PERIODS[country_key]:
            if period['start'] <= f"{year}-01-01" <= period['end']:
                entsoe_code = period['code']
                break

    # Etape 3 — Bornes temporelles
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    client = EntsoePandasClient(api_key=api_key)

    # Etape 4a — Load
    raw_load = _api_call_with_retry(client.query_load, entsoe_code, start=start, end=end)
    if isinstance(raw_load, pd.DataFrame):
        load_series = raw_load['Actual Load'] if 'Actual Load' in raw_load.columns else raw_load.iloc[:, 0]
    else:
        load_series = raw_load

    # Etape 4b — Generation par type
    raw_gen = _api_call_with_retry(client.query_generation, entsoe_code, start=start, end=end, psr_type=None)

    gen_dict = {}

    if isinstance(raw_gen.columns, pd.MultiIndex):
        for gen_type, col_type in raw_gen.columns:
            target_col = ENTSOE_GEN_MAPPING.get(gen_type)
            if target_col is None:
                continue

            if gen_type == "Hydro Pumped Storage":
                if col_type == "Actual Aggregated":
                    gen_dict[COL_PSH_GEN] = raw_gen[(gen_type, col_type)].abs()
                elif col_type == "Actual Consumption":
                    gen_dict[COL_PSH_PUMP] = raw_gen[(gen_type, col_type)].abs()
            else:
                if col_type == "Actual Aggregated":
                    if target_col in gen_dict:
                        gen_dict[target_col] = gen_dict[target_col] + raw_gen[(gen_type, col_type)].fillna(0)
                    else:
                        gen_dict[target_col] = raw_gen[(gen_type, col_type)]
    else:
        for col in raw_gen.columns:
            target_col = ENTSOE_GEN_MAPPING.get(col)
            if target_col:
                if target_col in gen_dict:
                    gen_dict[target_col] = gen_dict[target_col] + raw_gen[col].fillna(0)
                else:
                    gen_dict[target_col] = raw_gen[col]

    # Etape 4c — Prix Day-Ahead
    raw_price = _api_call_with_retry(client.query_day_ahead_prices, entsoe_code, start=start, end=end)
    if hasattr(raw_price.index, 'freq') and raw_price.index.freq is not None:
        freq_min = raw_price.index.freq.n if hasattr(raw_price.index.freq, 'n') else 60
    else:
        diffs = raw_price.index.to_series().diff().dt.total_seconds().dropna()
        freq_min = int(diffs.median() / 60)
        logger.info(f"Prix DA: resolution inferee = {freq_min} min")

    price_series = raw_price

    # Etape 4d — Net Position (cross-border)
    try:
        raw_net_pos = _api_call_with_retry(
            client.query_net_position, entsoe_code, start=start, end=end, dayahead=True
        )
        net_position = raw_net_pos
        logger.info(f"Net position chargee via API pour {country_key}/{year}")
    except Exception as e:
        logger.warning(f"query_net_position failed for {country_key}/{year}: {e}. "
                       f"Fallback: net_position = NaN (flex export non disponible).")
        net_position = None

    # Etape 5 — Assemblage et harmonisation
    df = pd.DataFrame(gen_dict)

    df[COL_LOAD] = load_series
    df[COL_PRICE_DA] = price_series
    if net_position is not None:
        df[COL_NET_POSITION] = net_position

    # Convertir en UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Resample hourly
    price_col = df[COL_PRICE_DA].copy() if COL_PRICE_DA in df.columns else None
    cols_no_price = [c for c in df.columns if c != COL_PRICE_DA]
    df_resampled = df[cols_no_price].resample('h').mean()

    if price_col is not None:
        df_resampled[COL_PRICE_DA] = price_col.resample('h').mean()

    df = df_resampled

    # Colonnes manquantes -> 0
    # Inclure solar et wind_on car certains pays (PL, DK early years) n'ont pas de donnees
    all_gen_optional = OPTIONAL_COLUMNS | {COL_SOLAR, COL_WIND_ON, COL_COAL, COL_GAS,
                                            COL_HYDRO_ROR, COL_BIOMASS, COL_OTHER}
    for col in all_gen_optional:
        if col not in df.columns:
            df[col] = 0.0

    if COL_NET_POSITION not in df.columns:
        df[COL_NET_POSITION] = np.nan

    # Total generation
    gen_cols = [COL_SOLAR, COL_WIND_ON, COL_WIND_OFF, COL_NUCLEAR, COL_LIGNITE,
                COL_COAL, COL_GAS, COL_HYDRO_ROR, COL_HYDRO_RES, COL_PSH_GEN,
                COL_BIOMASS, COL_OTHER]
    df[COL_TOTAL_GEN] = df[[c for c in gen_cols if c in df.columns]].sum(axis=1)

    # Interpolation des trous courts — SEULEMENT sur load et generation
    # JAMAIS sur les prix
    cols_to_interpolate = [c for c in df.columns if c != COL_PRICE_DA]
    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(
        method='linear', limit=3, limit_direction='forward'
    )

    # Flag gaps
    df[COL_HAS_GAP] = df[COL_LOAD].isna() | df[COL_PRICE_DA].isna()

    # Etape 6 — Validation
    expected = HOURS_LEAP if calendar.isleap(year) else HOURS_YEAR
    n = len(df)

    if (df[COL_LOAD] < 0).any():
        n_neg = (df[COL_LOAD] < 0).sum()
        logger.warning(f"{country_key}/{year}: {n_neg} heures avec load negatif -> mis a NaN")
        df.loc[df[COL_LOAD] < 0, COL_LOAD] = np.nan

    if n < expected * 0.95:
        logger.warning(f"{country_key}/{year}: {n}/{expected} heures seulement")

    completeness_load = df[COL_LOAD].notna().mean()
    completeness_price = df[COL_PRICE_DA].notna().mean()
    logger.info(f"Fetched {country_key}/{year}: {n} rows, "
                f"load={completeness_load:.1%}, price={completeness_price:.1%}")

    # Etape 7 — Sauvegarde
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=True)
    return df


def load_commodity_prices() -> dict:
    """
    Charge les prix gaz (TTF) et CO2 (EUA) depuis data/external/.
    Retourne {'gas': pd.Series, 'co2': pd.Series, 'bess': pd.DataFrame}
    Les Series ont un DatetimeIndex daily.
    Les cles manquantes sont None.
    """
    result = {'gas': None, 'co2': None, 'bess': None}

    gas_path = os.path.join("data", "external", "ttf_daily.csv")
    if os.path.exists(gas_path):
        gas_df = pd.read_csv(gas_path, parse_dates=['date'], index_col='date')
        result['gas'] = gas_df['price_eur_mwh']
    else:
        logger.warning("ttf_daily.csv absent -- TCA sera approxime")

    co2_path = os.path.join("data", "external", "eua_daily.csv")
    if os.path.exists(co2_path):
        co2_df = pd.read_csv(co2_path, parse_dates=['date'], index_col='date')
        result['co2'] = co2_df['price_eur_t']
    else:
        logger.warning("eua_daily.csv absent -- TCA sera approxime")

    bess_path = os.path.join("data", "external", "bess_capacity.csv")
    if os.path.exists(bess_path):
        result['bess'] = pd.read_csv(bess_path)

    return result
