import pytest
import pandas as pd
import numpy as np
from src.constants import *


@pytest.fixture
def make_raw_df():
    """Factory pour creer un DataFrame brut (pre-NRL)."""
    def _make(n=24, load=50000, solar=10000, wind_on=5000, wind_off=0,
              nuclear=30000, lignite=0, coal=0, gas=5000,
              hydro_ror=3000, biomass=1000, price=50.0,
              psh_pump=0, net_position=0):
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({
            COL_LOAD: load,
            COL_SOLAR: solar,
            COL_WIND_ON: wind_on,
            COL_WIND_OFF: wind_off,
            COL_NUCLEAR: nuclear,
            COL_LIGNITE: lignite,
            COL_COAL: coal,
            COL_GAS: gas,
            COL_HYDRO_ROR: hydro_ror,
            COL_HYDRO_RES: 0,
            COL_PSH_GEN: 0,
            COL_PSH_PUMP: psh_pump,
            COL_BIOMASS: biomass,
            COL_OTHER: 0,
            COL_PRICE_DA: price,
            COL_NET_POSITION: net_position,
            COL_TOTAL_GEN: solar + wind_on + wind_off + nuclear + lignite + coal + gas + hydro_ror + biomass,
            COL_HAS_GAP: False,
        }, index=idx)
    return _make


@pytest.fixture
def fr_config():
    return {
        'must_run': {
            'mode': 'observed',
            'observed_components': ['nuclear', 'hydro_ror', 'biomass'],
            'floor_params': {'nuclear_floor_gw': 20.0, 'nuclear_modulation_pct': 0.50}
        },
        'flex': {
            'capacity': {
                'psh_pump_capacity_gw': 4.5,
                'bess_power_gw': 0.5,
                'bess_energy_gwh': 2.0,
                'dsm_gw': 2.0,
                'export_max_gw': 17.0,
            }
        },
        'thermal': {'marginal_tech': 'CCGT'},
    }
