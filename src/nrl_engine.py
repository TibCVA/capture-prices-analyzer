import logging
import numpy as np
import pandas as pd
from src.constants import *
from src.constants import get_constants

logger = logging.getLogger("capture_prices.nrl_engine")


def compute_nrl(df: pd.DataFrame,
                country_config: dict,
                country_key: str,
                year: int,
                commodity_prices: dict | None = None,
                must_run_mode: str | None = None,
                constants_override: dict | None = None) -> pd.DataFrame:
    """
    Calcule NRL, surplus, flex, regimes sur un DataFrame horaire.

    Args:
        df: DataFrame horaire avec colonnes harmonisees (COL_LOAD, COL_SOLAR, etc.)
        country_config: dict de config pays (section du countries.yaml)
        country_key: str, code pays interne ('FR', 'DE', etc.)
        year: int, annee (pour lookup BESS capacity)
        commodity_prices: dict retourne par load_commodity_prices()
        must_run_mode: override du mode config ('observed' ou 'floor')

    Returns:
        DataFrame enrichi avec COL_VRE, COL_MUST_RUN, COL_NRL, COL_SURPLUS,
        COL_FLEX_CAPACITY, COL_FLEX_USED, COL_SURPLUS_UNABS, COL_REGIME,
        COL_TCA, COL_REGIME_COHERENT.
    """
    df = df.copy()
    config = country_config
    mode = must_run_mode or config['must_run']['mode']
    consts = get_constants(constants_override)

    # Colonnes VRE/generation manquantes -> 0 (ex: PL sans solaire avant 2018)
    for col in [COL_SOLAR, COL_WIND_ON, COL_WIND_OFF, COL_NUCLEAR, COL_LIGNITE,
                COL_COAL, COL_GAS, COL_HYDRO_ROR, COL_HYDRO_RES, COL_BIOMASS,
                COL_OTHER, COL_PSH_GEN, COL_PSH_PUMP]:
        if col not in df.columns:
            df[col] = 0.0

    # Etape 1 — VRE total
    df[COL_VRE] = (df[COL_SOLAR].fillna(0)
                   + df[COL_WIND_ON].fillna(0)
                   + df[COL_WIND_OFF].fillna(0))

    # Etape 2 — Must-Run
    component_map = {
        'nuclear': COL_NUCLEAR,
        'lignite': COL_LIGNITE,
        'coal': COL_COAL,
        'hydro_ror': COL_HYDRO_ROR,
        'biomass': COL_BIOMASS,
    }

    if mode == "observed":
        components = config['must_run']['observed_components']
        df[COL_MUST_RUN] = sum(
            df[component_map[c]].fillna(0) for c in components
            if c in component_map
        )

    elif mode == "floor":
        floor = config['must_run'].get('floor_params', {})
        mr = pd.Series(0.0, index=df.index)

        for filiere, col in component_map.items():
            floor_key = f"{filiere}_floor_gw"
            mod_key = f"{filiere}_modulation_pct"

            if floor_key in floor:
                floor_mw = floor[floor_key] * 1000  # GW -> MW
                mod_pct = floor.get(mod_key, 1.0)
                observed = df[col].fillna(0)

                # REGLE PHYSIQUE : MR ne depasse JAMAIS la production observee
                floor_value = np.maximum(floor_mw, observed * mod_pct)
                mr += np.minimum(observed, floor_value)

            elif filiere in config['must_run'].get('observed_components', []):
                mr += df[col].fillna(0)

        df[COL_MUST_RUN] = mr
    else:
        raise ValueError(f"Mode must-run inconnu: {mode}")

    # Etape 3 — NRL
    df[COL_NRL] = df[COL_LOAD] - df[COL_VRE] - df[COL_MUST_RUN]

    # Etape 4 — Surplus brut
    df[COL_SURPLUS] = np.maximum(0.0, -df[COL_NRL])

    # Etape 5 — Flex : trois colonnes distinctes
    # --- FLEX DOMESTIC (PSH + BESS + DSM, sans exports — pour FAR_structural) ---
    cap = config['flex']['capacity']
    bess_mw = cap.get('bess_power_gw', 0) * 1000

    # Enrichir avec BESS observe si disponible
    if commodity_prices and commodity_prices.get('bess') is not None:
        bess_df = commodity_prices['bess']
        match = bess_df.loc[
            (bess_df['country'] == country_key) & (bess_df['year'] == year)
        ]
        if not match.empty:
            bess_mw = match['power_mw'].values[0]

    flex_domestic = (
        cap.get('psh_pump_capacity_gw', 0) * 1000
        + bess_mw
        + cap.get('dsm_gw', 0) * 1000
    )
    # --- FLEX CAPACITY (totale, avec exports — pour regime A/B) ---
    export_max_mw = cap.get('export_max_gw', 0) * 1000
    flex_cap = flex_domestic + export_max_mw

    df[COL_FLEX_DOMESTIC] = flex_domestic
    df[COL_FLEX_CAPACITY] = flex_cap

    # --- FLEX USED (observee, pour FAR_observed) ---
    flex_used = pd.Series(0.0, index=df.index)

    if COL_PSH_PUMP in df.columns:
        flex_used += df[COL_PSH_PUMP].fillna(0)

    if COL_NET_POSITION in df.columns and df[COL_NET_POSITION].notna().any():
        flex_used += df[COL_NET_POSITION].clip(lower=0)

    df[COL_FLEX_USED] = flex_used

    # Etape 6 — Surplus non absorbe
    df[COL_SURPLUS_UNABS] = np.maximum(0.0, df[COL_SURPLUS] - df[COL_FLEX_CAPACITY])

    # Etape 7 — TCA (Thermal Cost Anchor)
    tech = config['thermal']['marginal_tech']

    if (commodity_prices
        and commodity_prices.get('gas') is not None
        and commodity_prices.get('co2') is not None):

        # Floor hourly index to daily, strip tz for alignment with tz-naive CSV data
        date_index = df.index.floor('D')
        date_index_naive = date_index.tz_localize(None) if date_index.tz else date_index

        gas_daily = commodity_prices['gas']
        co2_daily = commodity_prices['co2']

        dates_needed = pd.DatetimeIndex(date_index_naive.unique())
        gas_aligned = gas_daily.reindex(dates_needed, method='ffill').reindex(date_index_naive).values
        co2_aligned = co2_daily.reindex(dates_needed, method='ffill').reindex(date_index_naive).values

        if tech == 'CCGT':
            df[COL_TCA] = gas_aligned / consts["ETA_CCGT"] + (consts["EF_GAS"] / consts["ETA_CCGT"]) * co2_aligned + consts["VOM_CCGT"]
        elif tech == 'coal':
            coal_price = gas_aligned * 0.4
            df[COL_TCA] = coal_price / consts["ETA_COAL"] + (consts["EF_COAL"] / consts["ETA_COAL"]) * co2_aligned + consts["VOM_COAL"]
        elif tech == 'OCGT':
            df[COL_TCA] = gas_aligned / consts["ETA_OCGT"] + (consts["EF_GAS"] / consts["ETA_OCGT"]) * co2_aligned + consts["VOM_OCGT"]
        else:
            raise ValueError(f"Tech marginale inconnue: {tech}")
    else:
        logger.warning("Commodities absentes -- TCA fallback sur P75 prix DA (circularite partielle)")
        df[COL_TCA] = df[COL_PRICE_DA].rolling(24 * 30, min_periods=24 * 7).quantile(Q75)
        df[COL_TCA] = df[COL_TCA].bfill().ffill()

    # Etape 8 — Classification en regimes
    # REGLE ANTI-CIRCULARITE : classification basee UNIQUEMENT sur variables physiques.
    mask_a = df[COL_SURPLUS_UNABS] > 0
    mask_b = (df[COL_SURPLUS] > 0) & (~mask_a)

    mask_positive = df[COL_NRL] > 0
    nrl_positive = df.loc[mask_positive, COL_NRL]

    if len(nrl_positive) > 100:
        nrl_p90 = nrl_positive.quantile(Q90)
    else:
        nrl_p90 = np.inf

    mask_d = mask_positive & (df[COL_NRL] > nrl_p90)
    mask_c = mask_positive & ~mask_d

    df[COL_REGIME] = 'C'
    df.loc[mask_b, COL_REGIME] = 'B'
    df.loc[mask_a, COL_REGIME] = 'A'
    df.loc[mask_d, COL_REGIME] = 'D_tail'

    # Etape 9 — Validation croisee (post-classification)
    tca_median = df[COL_TCA].median()

    df[COL_REGIME_COHERENT] = False

    m_a = df[COL_REGIME] == 'A'
    df.loc[m_a, COL_REGIME_COHERENT] = df.loc[m_a, COL_PRICE_DA] <= consts["PRICE_VERY_LOW"]

    m_b = df[COL_REGIME] == 'B'
    df.loc[m_b, COL_REGIME_COHERENT] = (
        (df.loc[m_b, COL_PRICE_DA] >= -10) &
        (df.loc[m_b, COL_PRICE_DA] <= max(tca_median * 0.5, 30))
    )

    m_c = df[COL_REGIME] == 'C'
    df.loc[m_c, COL_REGIME_COHERENT] = (
        (df.loc[m_c, COL_PRICE_DA] >= max(tca_median * 0.3, 10)) &
        (df.loc[m_c, COL_PRICE_DA] <= tca_median * 2.0)
    )

    m_d = df[COL_REGIME] == 'D_tail'
    df.loc[m_d, COL_REGIME_COHERENT] = df.loc[m_d, COL_PRICE_DA] > tca_median

    coherence = df[COL_REGIME_COHERENT].mean()
    if coherence < 0.55:
        logger.warning(f"Coherence regime/prix = {coherence:.1%} (< 55%) -- "
                       f"verifier config must-run pour {country_key}/{year}")

    return df
