import numpy as np
from src.nrl_engine import compute_nrl
from src.metrics import compute_annual_metrics
from src.constants import *


def test_capture_rate_formula(make_raw_df, fr_config):
    """Capture rate = prix pondere par production"""
    df = make_raw_df(n=3)
    df[COL_PRICE_DA] = [10, 50, 80]
    df[COL_SOLAR] = [100, 200, 0]
    processed = compute_nrl(df, fr_config, 'FR', 2024)
    metrics = compute_annual_metrics(processed, 2024, 'FR')
    # capture = (10*100 + 50*200 + 80*0) / (100+200) = 11000/300 ~ 36.67
    assert abs(metrics['capture_rate_pv'] - 36.67) < 0.1


def test_far_no_surplus(make_raw_df, fr_config):
    """FAR = NaN quand surplus = 0 partout"""
    df = make_raw_df(load=80000, solar=5000, wind_on=2000, nuclear=30000)
    # NRL = 80000 - 7000 - 34000 = 39000 > 0 -> pas de surplus
    processed = compute_nrl(df, fr_config, 'FR', 2024)
    metrics = compute_annual_metrics(processed, 2024, 'FR')
    assert metrics['far_structural'] is None or np.isnan(metrics['far_structural'])


def test_vre_share_is_of_generation(make_raw_df, fr_config):
    """VRE share = % of total generation, PAS % of demand"""
    df = make_raw_df(load=50000, solar=10000, wind_on=5000, nuclear=30000,
                     gas=5000, hydro_ror=3000, biomass=1000)
    # total_gen = 10000+5000+30000+5000+3000+1000 = 54000
    # vre = 15000 -> vre_share = 15000/54000 ~ 0.2778
    processed = compute_nrl(df, fr_config, 'FR', 2024)
    metrics = compute_annual_metrics(processed, 2024, 'FR')
    assert abs(metrics['vre_share'] - 15000/54000) < 0.01
