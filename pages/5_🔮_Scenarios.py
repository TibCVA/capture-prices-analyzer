"""Page 5 - Scenarios."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import comment_scenario_delta, commentary_block
from src.metrics import compute_annual_metrics
from src.scenario_engine import apply_scenario
from src.ui_helpers import guard_no_data, inject_global_css, render_commentary, section

st.set_page_config(page_title="Scenarios", page_icon="🔮", layout="wide")
inject_global_css()

st.title("🔮 Scenarios")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Scenarios")

proc = state["processed"]
metrics_dict = state["metrics"]
if not proc:
    guard_no_data("la page Scenarios")

countries = sorted({k[0] for k in proc.keys()})
country = st.selectbox("Pays", countries)
years = sorted({k[1] for k in proc.keys() if k[0] == country})
year = st.selectbox("Annee baseline", years, index=len(years) - 1)

scenario_names = list(state["scenarios"].keys())
sc_name = st.selectbox("Scenario predefini", scenario_names)
base_params = state["scenarios"][sc_name].get("params", {})

with st.expander("Ajustements manuels", expanded=True):
    delta_pv_gw = st.slider("delta_pv_gw", -10.0, 40.0, float(base_params.get("delta_pv_gw", 0.0)), 0.5)
    delta_wind_onshore_gw = st.slider(
        "delta_wind_onshore_gw", -10.0, 40.0, float(base_params.get("delta_wind_onshore_gw", 0.0)), 0.5
    )
    delta_bess_power_gw = st.slider(
        "delta_bess_power_gw", 0.0, 25.0, float(base_params.get("delta_bess_power_gw", 0.0)), 0.5
    )
    delta_bess_energy_gwh = st.slider(
        "delta_bess_energy_gwh", 0.0, 120.0, float(base_params.get("delta_bess_energy_gwh", 0.0)), 1.0
    )
    gas_price_eur_mwh = st.slider("gas_price_eur_mwh", 10.0, 90.0, float(base_params.get("gas_price_eur_mwh", 35.0)), 1.0)
    co2_price_eur_t = st.slider("co2_price_eur_t", 20.0, 200.0, float(base_params.get("co2_price_eur_t", 80.0)), 1.0)

params = {
    "delta_pv_gw": delta_pv_gw,
    "delta_wind_onshore_gw": delta_wind_onshore_gw,
    "delta_bess_power_gw": delta_bess_power_gw,
    "delta_bess_energy_gwh": delta_bess_energy_gwh,
    "gas_price_eur_mwh": gas_price_eur_mwh,
    "co2_price_eur_t": co2_price_eur_t,
}

base_key = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
if base_key not in proc:
    guard_no_data("la baseline scenario")

if st.button("Executer scenario", type="primary"):
    with st.spinner("Calcul scenario..."):
        df_s = apply_scenario(
            df_base_processed=proc[base_key],
            country_key=country,
            year=year,
            country_cfg=state["countries_cfg"][country],
            thresholds=state["thresholds"],
            commodities=state["commodities"],
            scenario_params=params,
            price_mode=state["scenario_price_mode"],
        )

        m_base = metrics_dict[(country, year, state["price_mode"])]
        m_s = compute_annual_metrics(df_s, country, year, state["countries_cfg"][country])

    section("Resultats scenario", "Prix synthetique: affine par regimes, ancre sur TCA")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SR", f"{m_s['sr']:.3f}", f"{(m_s['sr'] - m_base['sr']):+.3f}")
    c2.metric("FAR", f"{m_s['far']:.3f}", f"{(m_s['far'] - m_base['far']):+.3f}")
    c3.metric("h_regime_a", f"{m_s['h_regime_a']:.0f}", f"{(m_s['h_regime_a'] - m_base['h_regime_a']):+.0f}")
    c4.metric("TTL", f"{m_s['ttl']:.1f}", f"{(m_s['ttl'] - m_base['ttl']):+.1f}")
    c5.metric(
        "capture_ratio_pv",
        f"{m_s['capture_ratio_pv']:.3f}",
        f"{(m_s['capture_ratio_pv'] - m_base['capture_ratio_pv']):+.3f}",
    )

    render_commentary(comment_scenario_delta(m_base, m_s))

    section("Graphique 1 - Heures par regime", "Baseline vs Scenario")
    chart_df = pd.DataFrame(
        {
            "regime": ["A", "B", "C", "D"],
            "baseline": [m_base["h_regime_a"], m_base["h_regime_b"], m_base["h_regime_c"], m_base["h_regime_d"]],
            "scenario": [m_s["h_regime_a"], m_s["h_regime_b"], m_s["h_regime_c"], m_s["h_regime_d"]],
        }
    )
    fig1 = px.bar(chart_df, x="regime", y=["baseline", "scenario"], barmode="group")
    fig1.update_layout(height=350)
    st.plotly_chart(fig1, use_container_width=True)

    render_commentary(
        commentary_block(
            title="Impact sur les regimes",
            n_label="regimes",
            n_value=4,
            observed={"h_A_delta": float(m_s["h_regime_a"] - m_base["h_regime_a"]), "h_D_delta": float(m_s["h_regime_d"] - m_base["h_regime_d"])},
            method_link="Regimes recalcules apres modifications physiques et dispatch BESS deterministe.",
            limits="Sans modelisation explicite des congestions reseau intra-zonales.",
        )
    )

    section("Graphique 2 - Prix utilises (distribution)", "Scenario")
    fig2 = px.histogram(df_s, x="price_used_eur_mwh", nbins=80)
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    render_commentary(
        commentary_block(
            title="Distribution du prix utilise",
            n_label="heures",
            n_value=len(df_s),
            observed={"p05": float(m_s["price_used_p05"]), "p95": float(m_s["price_used_p95"])},
            method_link="price_used=price_synth en mode scenario par defaut.",
            limits="Prix synthetique indicatif, non previsionnel pour le spot reel.",
        )
    )

    section("Tableau comparatif", "Baseline vs Scenario")
    table = pd.DataFrame(
        {
            "metric": ["sr", "far", "h_regime_a", "ttl", "capture_ratio_pv"],
            "baseline": [m_base["sr"], m_base["far"], m_base["h_regime_a"], m_base["ttl"], m_base["capture_ratio_pv"]],
            "scenario": [m_s["sr"], m_s["far"], m_s["h_regime_a"], m_s["ttl"], m_s["capture_ratio_pv"]],
        }
    )
    table["delta"] = table["scenario"] - table["baseline"]
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_commentary(
        commentary_block(
            title="Comparatif scenario",
            n_label="metriques",
            n_value=len(table),
            observed={"delta_sr": float(table.loc[table.metric == "sr", "delta"].iloc[0]), "delta_far": float(table.loc[table.metric == "far", "delta"].iloc[0])},
            method_link="Les metriques scenario suivent strictement les memes formules que l'historique.",
            limits="Resultat conditionnel aux hypotheses scenario; pas une projection probabiliste.",
        )
    )
