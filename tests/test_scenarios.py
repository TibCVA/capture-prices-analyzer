import numpy as np
import pandas as pd
from src.nrl_engine import compute_nrl
from src.scenario_engine import apply_scenario
from src.constants import *


def test_adding_bess_reduces_regime_a(make_raw_df, fr_config):
    """Ajouter du BESS reduit les heures de Regime A"""
    df = make_raw_df(n=48, load=30000, solar=20000, wind_on=15000, nuclear=30000)
    baseline = compute_nrl(df, fr_config, 'FR', 2024)
    h_a_baseline = (baseline[COL_REGIME] == 'A').sum()

    scenario = apply_scenario(baseline, {'delta_bess_power_gw': 20, 'delta_bess_energy_gwh': 80},
                              fr_config, 'FR', 2024)
    h_a_scenario = (scenario[COL_REGIME] == 'A').sum()

    assert h_a_scenario <= h_a_baseline


def test_gas_co2_scenario_changes_tca(make_raw_df, fr_config):
    """Les scenarios CO2/gaz modifient le TCA et le prix mecaniste"""
    # n=200 pour que le rolling P75 (min_periods=168) fonctionne dans le TCA fallback
    df = make_raw_df(n=200, load=50000, solar=5000, wind_on=3000, nuclear=30000)
    baseline = compute_nrl(df, fr_config, 'FR', 2024)
    baseline_tca = baseline[COL_TCA].median()

    scenario = apply_scenario(baseline, {'gas_price_eur_mwh': 60, 'co2_price_eur_t': 150},
                              fr_config, 'FR', 2024)
    scenario_tca = scenario[COL_TCA].median()

    assert scenario_tca > baseline_tca
    assert scenario['price_mechanistic'].median() != baseline.get(COL_PRICE_DA, pd.Series([0])).median()
