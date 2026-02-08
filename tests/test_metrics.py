from __future__ import annotations

import numpy as np

from src.metrics import compute_annual_metrics
from src.nrl_engine import compute_nrl


def test_metrics_penetration_definition_matches_generation_share(
    make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg
):
    df = make_raw_df(n=24, solar=10000, wind_on=5000, wind_off=2000, nuclear=20000, gas=3000, biomass=2000)
    out = compute_nrl(
        df_raw=df,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        price_mode="observed",
    )
    m = compute_annual_metrics(out, "FR", 2024, fr_cfg)

    total_gen = (
        df["solar_mw"]
        + df["wind_onshore_mw"]
        + df["wind_offshore_mw"]
        + df["nuclear_mw"]
        + df["coal_mw"]
        + df["gas_mw"]
        + df["lignite_mw"]
        + df["hydro_ror_mw"]
        + df["hydro_reservoir_mw"]
        + df["biomass_mw"]
        + df["other_mw"]
    ).sum()
    pv_expected = 100 * df["solar_mw"].sum() / total_gen
    assert abs(m["pv_penetration_pct_gen"] - pv_expected) < 1e-6


def test_metrics_observables_use_observed_price_not_price_used(
    make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg
):
    df = make_raw_df(n=24, price_da=-5.0)
    out = compute_nrl(
        df_raw=df,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        price_mode="synthetic",
    )
    m = compute_annual_metrics(out, "FR", 2024, fr_cfg)

    assert m["h_negative_obs"] == 24
    assert np.isfinite(m["baseload_price_used"])


def test_far_nan_when_no_surplus(make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg):
    df = make_raw_df(n=24, load=90000, solar=1000, wind_on=1000, nuclear=10000)
    out = compute_nrl(
        df_raw=df,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        price_mode="observed",
    )
    m = compute_annual_metrics(out, "FR", 2024, fr_cfg)
    assert np.isnan(m["far"])
