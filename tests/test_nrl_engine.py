from __future__ import annotations

import numpy as np

from src.constants import COL_BESS_CHARGE, COL_BESS_DISCHARGE, COL_BESS_SOC, COL_NRL, COL_REGIME, COL_SURPLUS
from src.nrl_engine import compute_nrl


def test_nrl_engine_basic_observed(make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg):
    df = make_raw_df(n=24, load=50000, solar=10000, wind_on=8000, nuclear=20000)
    out = compute_nrl(
        df_raw=df,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        must_run_mode="observed",
        flex_model_mode="observed",
        price_mode="observed",
    )
    assert COL_NRL in out.columns
    assert COL_SURPLUS in out.columns
    assert COL_REGIME in out.columns
    assert out.index.tz is not None


def test_no_generation_minus_load_export_rule(make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg):
    df = make_raw_df(n=12)
    df.attrs["derived_net_position_from_generation_minus_load"] = True

    try:
        compute_nrl(
            df_raw=df,
            country_key="FR",
            year=2024,
            country_cfg=fr_cfg,
            thresholds=thresholds_cfg,
            commodities=commodities_cfg,
            price_mode="observed",
        )
    except NotImplementedError as exc:
        assert "generation-load" in str(exc)
    else:
        raise AssertionError("NotImplementedError attendu pour la regle anti generation-load")


def test_bess_soc_dispatch_respects_energy(make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg):
    # Force strong recurring surplus and deficits to stress charge/discharge
    df = make_raw_df(n=72, load=35000, solar=18000, wind_on=12000, nuclear=22000, psh_pump=0)
    out = compute_nrl(
        df_raw=df,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        must_run_mode="floor",
        flex_model_mode="capacity",
        scenario_overrides={"delta_bess_power_gw": 0.0, "delta_bess_energy_gwh": 0.0},
        price_mode="synthetic",
    )

    soc = out[COL_BESS_SOC]
    charge = out[COL_BESS_CHARGE]
    discharge = out[COL_BESS_DISCHARGE]

    assert np.isfinite(soc).all()
    assert (charge >= 0).all()
    assert (discharge >= 0).all()
    # Energy capacity default FR = 2 GWh = 2000 MWh
    assert float(soc.max()) <= 2000.0 + 1e-6
    assert float(soc.min()) >= -1e-6
