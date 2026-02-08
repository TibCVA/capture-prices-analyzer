from __future__ import annotations

from src.metrics import compute_annual_metrics
from src.nrl_engine import compute_nrl
from src.scenario_engine import apply_scenario


def test_scenario_plus_bess_increases_far_and_reduces_h_regime_a(
    make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg
):
    df = make_raw_df(n=168, load=32000, solar=17000, wind_on=12000, nuclear=22000)
    base = compute_nrl(
        df_raw=df,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        must_run_mode="floor",
        flex_model_mode="capacity",
        price_mode="synthetic",
    )
    m_base = compute_annual_metrics(base, "FR", 2024, fr_cfg)

    scen = apply_scenario(
        df_base_processed=base,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        scenario_params={"delta_bess_power_gw": 8, "delta_bess_energy_gwh": 32},
        price_mode="synthetic",
    )
    m_s = compute_annual_metrics(scen, "FR", 2024, fr_cfg)

    assert m_s["far"] >= m_base["far"]
    assert m_s["h_regime_a"] <= m_base["h_regime_a"]


def test_scenario_co2_gas_increases_ttl_synth(make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg):
    df = make_raw_df(n=168)
    base = compute_nrl(
        df_raw=df,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        must_run_mode="floor",
        flex_model_mode="capacity",
        price_mode="synthetic",
    )
    m_base = compute_annual_metrics(base, "FR", 2024, fr_cfg)

    high = apply_scenario(
        df_base_processed=base,
        country_key="FR",
        year=2024,
        country_cfg=fr_cfg,
        thresholds=thresholds_cfg,
        commodities=commodities_cfg,
        scenario_params={"gas_price_eur_mwh": 70, "co2_price_eur_t": 160},
        price_mode="synthetic",
    )
    m_high = compute_annual_metrics(high, "FR", 2024, fr_cfg)

    assert m_high["ttl"] >= m_base["ttl"]
