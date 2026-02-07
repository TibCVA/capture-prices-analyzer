import logging
import numpy as np
import pandas as pd
from src.constants import *
from src.constants import get_constants

logger = logging.getLogger("capture_prices.metrics")


def compute_annual_metrics(df: pd.DataFrame, year: int,
                           country_key: str,
                           constants_override: dict | None = None) -> dict:
    """
    Calcule toutes les metriques annuelles a partir d'un DataFrame processe.
    Retourne un dict exhaustif de ~50 metriques.
    """
    consts = get_constants(constants_override)
    # Exclure les heures avec donnees manquantes
    valid = df[~df[COL_HAS_GAP]].copy()
    n_valid = len(valid)

    if n_valid < 1000:
        logger.warning(f"{country_key}/{year}: seulement {n_valid} heures valides")

    # === PRIX ===
    tz_local = COUNTRY_TZ.get(country_key, "UTC")
    local_idx = valid.index.tz_convert(tz_local)
    is_weekday = local_idx.weekday < 5
    local_hour = local_idx.hour
    is_peak = is_weekday & (local_hour >= 8) & (local_hour < 20)

    baseload_price = valid[COL_PRICE_DA].mean()
    peakload_price = valid.loc[is_peak, COL_PRICE_DA].mean() if is_peak.any() else np.nan
    offpeak_price = valid.loc[~is_peak, COL_PRICE_DA].mean() if (~is_peak).any() else np.nan

    # === CAPTURE RATES ===
    def _capture_rate(production_col: str) -> float:
        prod = valid[production_col].fillna(0)
        total = prod.sum()
        if total < 1e-3:
            return np.nan
        return (valid[COL_PRICE_DA] * prod).sum() / total

    cr_pv = _capture_rate(COL_SOLAR)
    cr_wind = _capture_rate(COL_WIND_ON)
    cr_wind_off = _capture_rate(COL_WIND_OFF)
    wind_total = valid[COL_WIND_ON].fillna(0) + valid[COL_WIND_OFF].fillna(0)
    cr_wind_total = (valid[COL_PRICE_DA] * wind_total).sum() / wind_total.sum() if wind_total.sum() > 1e-3 else np.nan

    # Capture ratios
    cratio_pv = cr_pv / baseload_price if baseload_price and baseload_price != 0 else np.nan
    cratio_wind = cr_wind_total / baseload_price if baseload_price and baseload_price != 0 else np.nan

    # === PENETRATION RES ===
    # CONVENTION : % of total generation (PAS % of demand)
    total_gen_energy = valid[COL_TOTAL_GEN].sum()
    vre_share = valid[COL_VRE].sum() / total_gen_energy if total_gen_energy > 0 else 0
    pv_share = valid[COL_SOLAR].fillna(0).sum() / total_gen_energy if total_gen_energy > 0 else 0
    wind_share = (valid[COL_WIND_ON].fillna(0) + valid[COL_WIND_OFF].fillna(0)).sum() / total_gen_energy if total_gen_energy > 0 else 0

    # === HEURES PAR REGIME ===
    h_a = (valid[COL_REGIME] == 'A').sum()
    h_b = (valid[COL_REGIME] == 'B').sum()
    h_c = (valid[COL_REGIME] == 'C').sum()
    h_d = (valid[COL_REGIME] == 'D_tail').sum()

    # === HEURES DE PRIX ===
    h_negative = (valid[COL_PRICE_DA] < consts["PRICE_NEGATIVE"]).sum()
    h_below_5 = (valid[COL_PRICE_DA] < consts["PRICE_VERY_LOW"]).sum()
    h_above_100 = (valid[COL_PRICE_DA] > consts["PRICE_HIGH"]).sum()
    h_above_200 = (valid[COL_PRICE_DA] > consts["PRICE_VERY_HIGH"]).sum()

    # === SPREADS ===
    valid_local = valid.copy()
    valid_local['_date_local'] = local_idx.date
    daily = valid_local.groupby('_date_local')[COL_PRICE_DA].agg(['max', 'min'])
    daily['spread'] = daily['max'] - daily['min']

    spread_p95_p05 = valid[COL_PRICE_DA].quantile(Q95) - valid[COL_PRICE_DA].quantile(Q05)
    days_spread_50 = (daily['spread'] > consts["SPREAD_DAILY_THRESHOLD"]).sum()
    avg_daily_spread = daily['spread'].mean()
    max_daily_spread = daily['spread'].max()

    # === RATIOS STRUCTURELS ===
    sr = valid[COL_SURPLUS].sum() / valid[COL_LOAD].sum() if valid[COL_LOAD].sum() > 0 else 0
    sr_hours = (valid[COL_SURPLUS] > 0).sum() / n_valid if n_valid > 0 else 0

    # FAR_structural — basé sur la flex DOMESTIQUE (PSH + BESS + DSM, sans exports)
    # Les exports ne sont pas une flex structurelle fiable (correlated VRE across Europe)
    surplus_mask = valid[COL_SURPLUS] > 0
    if surplus_mask.sum() == 0:
        far_structural = np.nan
    else:
        flex_dom_col = COL_FLEX_DOMESTIC if COL_FLEX_DOMESTIC in valid.columns else COL_FLEX_CAPACITY
        absorbed_cap = np.minimum(
            valid.loc[surplus_mask, COL_SURPLUS],
            valid.loc[surplus_mask, flex_dom_col]
        ).sum()
        far_structural = absorbed_cap / valid.loc[surplus_mask, COL_SURPLUS].sum()

    # FAR_observed
    if surplus_mask.sum() == 0:
        far_observed = np.nan
    else:
        absorbed_used = np.minimum(
            valid.loc[surplus_mask, COL_SURPLUS],
            valid.loc[surplus_mask, COL_FLEX_USED]
        ).sum()
        far_observed = absorbed_used / valid.loc[surplus_mask, COL_SURPLUS].sum()

    # IR (Inflexibility Ratio)
    mr_load = valid[[COL_MUST_RUN, COL_LOAD]].dropna()
    if len(mr_load) < 100:
        ir = np.nan
    else:
        mr_p10 = mr_load[COL_MUST_RUN].quantile(Q10)
        load_p10 = mr_load[COL_LOAD].quantile(Q10)
        ir = mr_p10 / load_p10 if load_p10 > 0 else np.nan

    # TTL (Thermal Tail Level)
    mask_cd = valid[COL_REGIME].isin(['C', 'D_tail'])
    ttl = valid.loc[mask_cd, COL_PRICE_DA].quantile(Q95) if mask_cd.sum() > 50 else np.nan

    # === VOLUMES ===
    mwh_to_twh = 1e-6
    total_load_twh = valid[COL_LOAD].sum() * mwh_to_twh
    total_vre_twh = valid[COL_VRE].sum() * mwh_to_twh
    total_surplus_twh = valid[COL_SURPLUS].sum() * mwh_to_twh
    total_surplus_unabs_twh = valid[COL_SURPLUS_UNABS].sum() * mwh_to_twh

    # === QUALITE ===
    data_completeness = 1.0 - df[COL_HAS_GAP].mean()
    regime_coherence = valid[COL_REGIME_COHERENT].mean() if len(valid) > 0 else 0

    return {
        'country': country_key,
        'year': year,
        # Prix
        'baseload_price': round(baseload_price, 2) if not np.isnan(baseload_price) else np.nan,
        'peakload_price': round(peakload_price, 2) if not np.isnan(peakload_price) else np.nan,
        'offpeak_price': round(offpeak_price, 2) if not np.isnan(offpeak_price) else np.nan,
        'price_p05': round(valid[COL_PRICE_DA].quantile(Q05), 2),
        'price_p25': round(valid[COL_PRICE_DA].quantile(Q25), 2),
        'price_median': round(valid[COL_PRICE_DA].quantile(Q50), 2),
        'price_p75': round(valid[COL_PRICE_DA].quantile(Q75), 2),
        'price_p95': round(valid[COL_PRICE_DA].quantile(Q95), 2),
        'price_stddev': round(valid[COL_PRICE_DA].std(), 2),
        # Capture
        'capture_rate_pv': round(cr_pv, 2) if not np.isnan(cr_pv) else np.nan,
        'capture_rate_wind': round(cr_wind_total, 2) if not np.isnan(cr_wind_total) else np.nan,
        'capture_rate_wind_on': round(cr_wind, 2) if not np.isnan(cr_wind) else np.nan,
        'capture_rate_wind_off': round(cr_wind_off, 2) if not np.isnan(cr_wind_off) else np.nan,
        'capture_ratio_pv': round(cratio_pv, 4) if cratio_pv is not None and not np.isnan(cratio_pv) else np.nan,
        'capture_ratio_wind': round(cratio_wind, 4) if cratio_wind is not None and not np.isnan(cratio_wind) else np.nan,
        # Heures regime
        'h_regime_a': int(h_a), 'h_regime_b': int(h_b),
        'h_regime_c': int(h_c), 'h_regime_d_tail': int(h_d),
        # Heures prix
        'h_negative': int(h_negative), 'h_below_5': int(h_below_5),
        'h_above_100': int(h_above_100), 'h_above_200': int(h_above_200),
        # Spreads
        'spread_p95_p05': round(spread_p95_p05, 2),
        'days_spread_above_50': int(days_spread_50),
        'avg_daily_spread': round(avg_daily_spread, 2),
        'max_daily_spread': round(max_daily_spread, 2),
        # Ratios
        'sr': round(sr, 4), 'sr_hours': round(sr_hours, 4),
        'far_structural': round(far_structural, 4) if not np.isnan(far_structural) else np.nan,
        'far_observed': round(far_observed, 4) if not np.isnan(far_observed) else np.nan,
        'ir': round(ir, 4) if ir is not None and not np.isnan(ir) else np.nan,
        'ttl': round(ttl, 2) if ttl is not None and not np.isnan(ttl) else np.nan,
        # Penetration (% of total generation)
        'vre_share': round(vre_share, 4),
        'pv_share': round(pv_share, 4),
        'wind_share': round(wind_share, 4),
        # Volumes
        'total_load_twh': round(total_load_twh, 2),
        'total_vre_twh': round(total_vre_twh, 2),
        'total_surplus_twh': round(total_surplus_twh, 3),
        'total_surplus_unabs_twh': round(total_surplus_unabs_twh, 3),
        # Qualite
        'data_completeness': round(data_completeness, 4),
        'regime_coherence': round(regime_coherence, 4),
        'is_outlier': year in OUTLIER_YEARS,
    }
