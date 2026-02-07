import numpy as np
from src.nrl_engine import compute_nrl
from src.constants import *


def test_nrl_basic(make_raw_df, fr_config):
    """NRL = load - VRE - must_run"""
    df = make_raw_df(load=50000, solar=10000, wind_on=5000, nuclear=30000)
    result = compute_nrl(df, fr_config, 'FR', 2024)
    # VRE = 15000, MR = 30000 + 3000 + 1000 = 34000, NRL = 50000 - 15000 - 34000 = 1000
    assert (result[COL_NRL] == 1000).all()
    assert (result[COL_SURPLUS] == 0).all()
    assert (result[COL_REGIME] == 'C').all()


def test_surplus_created(make_raw_df, fr_config):
    """NRL < 0 -> surplus"""
    df = make_raw_df(load=30000, solar=20000, wind_on=15000, nuclear=30000)
    result = compute_nrl(df, fr_config, 'FR', 2024)
    # VRE = 35000, MR = 34000, NRL = 30000 - 35000 - 34000 = -39000
    assert (result[COL_SURPLUS] == 39000).all()


def test_regime_a_surplus_exceeds_flex(make_raw_df, fr_config):
    """Surplus > flex_capacity -> Regime A"""
    # flex_capacity = (4.5 + 0.5 + 2.0 + 17.0) * 1000 = 24000 MW
    df = make_raw_df(load=30000, solar=20000, wind_on=15000, nuclear=30000)
    # surplus = 39000 > flex_capacity = 24000
    result = compute_nrl(df, fr_config, 'FR', 2024)
    assert (result[COL_REGIME] == 'A').all()
    assert (result[COL_SURPLUS_UNABS] == 39000 - 24000).all()


def test_regime_b_surplus_within_flex(make_raw_df, fr_config):
    """Surplus <= flex_capacity -> Regime B"""
    # Petit surplus < 24000
    df = make_raw_df(load=50000, solar=15000, wind_on=8000, nuclear=30000)
    # VRE = 23000, MR = 34000, NRL = 50000 - 23000 - 34000 = -7000
    # surplus = 7000 < flex 24000
    result = compute_nrl(df, fr_config, 'FR', 2024)
    assert (result[COL_REGIME] == 'B').all()


def test_floor_mode_capped_by_observed(make_raw_df, fr_config):
    """Floor mode : MR ne depasse jamais la production observee"""
    df = make_raw_df(nuclear=10000)
    # floor = max(20000, 10000 * 0.50) = 20000
    # mais min(observed=10000, floor=20000) = 10000 -> MR nucl = 10000
    result = compute_nrl(df, fr_config, 'FR', 2024, must_run_mode='floor')
    # MR_total = 10000 + 3000 + 1000 = 14000
    expected_mr = 14000
    assert (result[COL_MUST_RUN] == expected_mr).all()
