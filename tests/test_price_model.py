from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import COL_PRICE_SYNTH, COL_REGIME, COL_TCA
from src.price_model import compute_price_synth, compute_tca, select_price_used


def test_compute_tca_ccgt_with_overrides(fr_cfg, commodities_cfg, make_raw_df):
    df = make_raw_df(n=24)
    tca = compute_tca(
        df=df,
        country_cfg=fr_cfg,
        commodities=commodities_cfg,
        scenario_overrides={"gas_price_eur_mwh": 50.0, "co2_price_eur_t": 120.0},
    )
    assert tca.notna().all()
    assert float(tca.mean()) > 0


def test_compute_price_synth_by_regime(make_raw_df):
    df = make_raw_df(n=4)
    df[COL_REGIME] = ["A", "B", "C", "D"]
    df[COL_TCA] = [70.0, 70.0, 70.0, 70.0]
    out = compute_price_synth(df)
    assert out.name == COL_PRICE_SYNTH
    assert out.iloc[0] <= 0
    assert 0 <= out.iloc[1] <= 30
    assert np.isclose(out.iloc[2], 70.0)
    assert out.iloc[3] > out.iloc[2]


def test_select_price_used_modes(make_raw_df):
    df = make_raw_df(n=3)
    df[COL_PRICE_SYNTH] = [1.0, 2.0, 3.0]

    obs = select_price_used(df, "observed")
    syn = select_price_used(df, "synthetic")

    assert obs.notna().all()
    assert list(syn.values) == [1.0, 2.0, 3.0]
