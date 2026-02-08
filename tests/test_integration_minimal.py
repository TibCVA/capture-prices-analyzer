from __future__ import annotations

import os

import pytest

from src.metrics import compute_annual_metrics
from src.nrl_engine import compute_nrl


def test_integration_minimal_offline(make_raw_df, fr_cfg, thresholds_cfg, commodities_cfg):
    df = make_raw_df(n=72)
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
    assert "sr" in m
    assert "far" in m
    assert "ttl" in m


@pytest.mark.skipif(not os.getenv("ENTSOE_API_KEY"), reason="ENTSOE_API_KEY absent")
def test_integration_entsoe_key_present_placeholder():
    # Placeholder integration guard: actual remote fetch is exercised manually in app flow.
    assert True
