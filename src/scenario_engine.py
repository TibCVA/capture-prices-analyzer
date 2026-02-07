import logging
import numpy as np
import pandas as pd
from src.constants import *
from src.constants import get_constants

logger = logging.getLogger("capture_prices.scenario_engine")


def _scale_vre_profile(series: pd.Series, delta_gw: float,
                       profile_type: str = 'solar') -> pd.Series:
    """
    Ajoute delta_gw de capacite a un profil VRE existant.
    Si profil quasi-nul, genere un profil synthetique.
    """
    installed_estimate = series.quantile(Q99) if len(series) > 0 else 0

    if installed_estimate < 100:
        if delta_gw <= 0:
            return series

        logger.warning(f"Profil {profile_type} quasi-nul -- utilisation profil synthetique")
        hour = series.index.hour
        month = series.index.month

        if profile_type == 'solar':
            profile = np.maximum(0, np.cos(np.pi * (hour - 12) / 8)) ** 2
            season = 0.5 + 0.5 * np.cos(2 * np.pi * (month - 6) / 12)
            synthetic = profile * season * delta_gw * 1000
        elif profile_type == 'wind_onshore':
            base = 0.25
            season = 1.0 + 0.3 * np.cos(2 * np.pi * (month - 1) / 12)
            diurnal = 1.0 - 0.1 * np.cos(2 * np.pi * (hour - 3) / 24)
            noise = np.random.RandomState(42).uniform(0.5, 1.5, len(series))
            synthetic = base * season * diurnal * noise * delta_gw * 1000
        elif profile_type == 'wind_offshore':
            base = 0.40
            season = 1.0 + 0.2 * np.cos(2 * np.pi * (month - 1) / 12)
            noise = np.random.RandomState(42).uniform(0.7, 1.3, len(series))
            synthetic = base * season * noise * delta_gw * 1000
        else:
            raise ValueError(f"profile_type inconnu: {profile_type}")

        return pd.Series(np.maximum(0, synthetic), index=series.index)

    scale_factor = 1.0 + (delta_gw * 1000) / installed_estimate
    return series * max(scale_factor, 0)


def _compute_bess_absorption(surplus: pd.Series,
                              power_mw: float,
                              energy_mwh: float,
                              consts: dict | None = None) -> pd.Series:
    """
    Calcule l'absorption BESS heure par heure avec contrainte de SoC.
    """
    if consts is None:
        consts = get_constants()
    soc = energy_mwh * consts["BESS_MIN_SOC"]
    soc_max = energy_mwh * consts["BESS_MAX_SOC"]
    soc_min = energy_mwh * consts["BESS_MIN_SOC"]

    absorbed = np.zeros(len(surplus))

    for i in range(len(surplus)):
        s = surplus.iloc[i]
        if s > 0:
            space = soc_max - soc
            charge = min(s, power_mw, space)
            soc += charge * np.sqrt(consts["BESS_ROUND_TRIP_EFF"])
            absorbed[i] = charge
        else:
            discharge = min(power_mw, soc - soc_min)
            soc -= discharge

        # Reset journalier
        if i > 0 and surplus.index[i].hour == 0:
            soc = energy_mwh * consts["BESS_MIN_SOC"]

    return pd.Series(absorbed, index=surplus.index)


def apply_scenario(baseline_df: pd.DataFrame,
                   scenario_params: dict,
                   country_config: dict,
                   country_key: str,
                   year: int,
                   commodity_prices: dict | None = None,
                   constants_override: dict | None = None) -> pd.DataFrame:
    """
    Applique un scenario au DataFrame baseline et recalcule NRL + regimes.
    """
    df = baseline_df.copy()
    p = scenario_params
    consts = get_constants(constants_override)

    # --- 1. Modifications VRE ---
    if 'delta_pv_gw' in p and p['delta_pv_gw'] != 0:
        df[COL_SOLAR] = _scale_vre_profile(df[COL_SOLAR], p['delta_pv_gw'], 'solar')
    if 'delta_wind_onshore_gw' in p and p['delta_wind_onshore_gw'] != 0:
        df[COL_WIND_ON] = _scale_vre_profile(df[COL_WIND_ON], p['delta_wind_onshore_gw'], 'wind_onshore')
    if 'delta_wind_offshore_gw' in p and p['delta_wind_offshore_gw'] != 0:
        df[COL_WIND_OFF] = _scale_vre_profile(df[COL_WIND_OFF], p['delta_wind_offshore_gw'], 'wind_offshore')

    # --- 2. Modifications demande ---
    if 'delta_demand_pct' in p and p['delta_demand_pct'] != 0:
        df[COL_LOAD] = df[COL_LOAD] * (1 + p['delta_demand_pct'] / 100)
    if 'delta_demand_midday_gw' in p and p['delta_demand_midday_gw'] != 0:
        midday_mask = (df.index.hour >= 10) & (df.index.hour < 17)
        df.loc[midday_mask, COL_LOAD] += p['delta_demand_midday_gw'] * 1000
    if 'delta_demand_evening_gw' in p and p['delta_demand_evening_gw'] != 0:
        evening_mask = (df.index.hour >= 17) & (df.index.hour < 22)
        df.loc[evening_mask, COL_LOAD] += p['delta_demand_evening_gw'] * 1000

    # --- 3. Modifications must-run ---
    if 'delta_must_run_gw' in p and p['delta_must_run_gw'] != 0:
        df[COL_MUST_RUN] = np.maximum(0, df[COL_MUST_RUN] + p['delta_must_run_gw'] * 1000)

    # --- 4. Recalcul VRE + NRL + Surplus ---
    df[COL_VRE] = df[COL_SOLAR].fillna(0) + df[COL_WIND_ON].fillna(0) + df[COL_WIND_OFF].fillna(0)
    df[COL_NRL] = df[COL_LOAD] - df[COL_VRE] - df[COL_MUST_RUN]
    df[COL_SURPLUS] = np.maximum(0.0, -df[COL_NRL])

    # --- 5. Modifications flex ---
    cap = country_config['flex']['capacity']
    bess_power = cap.get('bess_power_gw', 0) * 1000 + p.get('delta_bess_power_gw', 0) * 1000
    bess_energy = cap.get('bess_energy_gwh', 0) * 1000 + p.get('delta_bess_energy_gwh', 0) * 1000

    bess_absorption = _compute_bess_absorption(df[COL_SURPLUS], bess_power, bess_energy, consts)

    psh_mw = cap.get('psh_pump_capacity_gw', 0) * 1000
    dsm_mw = cap.get('dsm_gw', 0) * 1000
    export_max_mw = cap.get('export_max_gw', 0) * 1000

    # Flex domestique (PSH + BESS + DSM) — pour FAR structural
    flex_domestic = psh_mw + dsm_mw + bess_power
    # Flex totale (avec exports) — pour regime A/B classification
    flex_total = flex_domestic + export_max_mw

    df[COL_FLEX_DOMESTIC] = flex_domestic
    df[COL_FLEX_CAPACITY] = flex_total
    df[COL_SURPLUS_UNABS] = np.maximum(0.0, df[COL_SURPLUS] - flex_total - bess_absorption)

    # --- 6. TCA scenario + prix mecaniste ---
    tech = country_config['thermal']['marginal_tech']

    if 'gas_price_eur_mwh' in p:
        gas_scenario = p['gas_price_eur_mwh']
    elif commodity_prices and commodity_prices.get('gas') is not None:
        gas_scenario = commodity_prices['gas'].median()
    else:
        gas_scenario = 30.0

    if 'co2_price_eur_t' in p:
        co2_scenario = p['co2_price_eur_t']
    elif commodity_prices and commodity_prices.get('co2') is not None:
        co2_scenario = commodity_prices['co2'].median()
    else:
        co2_scenario = 65.0

    if tech == 'CCGT':
        tca_scenario = gas_scenario / consts["ETA_CCGT"] + (consts["EF_GAS"] / consts["ETA_CCGT"]) * co2_scenario + consts["VOM_CCGT"]
    elif tech == 'coal':
        coal_price = gas_scenario * 0.4
        tca_scenario = coal_price / consts["ETA_COAL"] + (consts["EF_COAL"] / consts["ETA_COAL"]) * co2_scenario + consts["VOM_COAL"]
    else:
        tca_scenario = gas_scenario / consts["ETA_CCGT"] + (consts["EF_GAS"] / consts["ETA_CCGT"]) * co2_scenario + consts["VOM_CCGT"]

    df[COL_TCA] = tca_scenario

    # PRIX MECANISTE AFFINE PAR MORCEAUX
    price_mech = pd.Series(tca_scenario, index=df.index)

    # Regime A : prix negatif proportionnel au surplus non absorbe
    mask_a = df[COL_SURPLUS_UNABS] > 0
    price_mech.loc[mask_a] = -df.loc[mask_a, COL_SURPLUS_UNABS] / df.loc[mask_a, COL_LOAD].clip(lower=1) * 100

    # Regime B : cout marginal flex (~BESS cycling cost)
    mask_b = (df[COL_SURPLUS] > 0) & ~mask_a
    price_mech.loc[mask_b] = consts["BESS_CYCLING_COST"]

    # Regime D_tail : TCA x 1.5 (proxy scarcity)
    nrl_pos = df.loc[df[COL_NRL] > 0, COL_NRL]
    if len(nrl_pos) > 100:
        nrl_p90 = nrl_pos.quantile(Q90)
        mask_d = (df[COL_NRL] > nrl_p90) & (df[COL_NRL] > 0)
        price_mech.loc[mask_d] = tca_scenario * 1.5 + (
            (df.loc[mask_d, COL_NRL] - nrl_p90) / nrl_p90 * tca_scenario * 0.5
        ).clip(upper=tca_scenario * 3)

    df['price_mechanistic'] = price_mech

    # --- 7. Reclassification regimes ---
    df[COL_REGIME] = 'C'
    df.loc[mask_b, COL_REGIME] = 'B'
    df.loc[mask_a, COL_REGIME] = 'A'
    if len(nrl_pos) > 100:
        df.loc[mask_d, COL_REGIME] = 'D_tail'

    return df
