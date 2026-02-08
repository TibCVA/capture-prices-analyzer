"""Page 5 - Scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import comment_scenario_delta, so_what_block
from src.metrics import compute_annual_metrics
from src.scenario_engine import apply_scenario
from src.state_adapter import coerce_numeric_columns, metrics_to_dataframe, normalize_metrics_record
from src.ui_theme import PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS
from src.ui_helpers import (
    challenge_block,
    guard_no_data,
    info_card,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
    section_header,
)

st.set_page_config(page_title="Scenarios", page_icon="🔮", layout="wide")
inject_global_css()
st.title("🔮 Scenarios")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Scenarios")
normalize_state_metrics(state)

proc = state.get("processed", {})
df_all = metrics_to_dataframe(state, state.get("price_mode"))
if not proc or df_all.empty or "country" not in df_all.columns:
    guard_no_data("la page Scenarios")

countries = sorted({k[0] for k in proc.keys()})
country = st.selectbox("Pays", countries)
years = sorted({k[1] for k in proc.keys() if k[0] == country})
year = st.selectbox("Annee baseline", years, index=len(years) - 1)

scenario_names = list(state.get("scenarios", {}).keys())
if not scenario_names:
    guard_no_data("la page Scenarios")
sc_name = st.selectbox("Scenario predefini", scenario_names)
base_params = state["scenarios"][sc_name].get("params", {})

narrative(
    "Le moteur scenario recalcule tout le pipeline (NRL -> regimes -> TCA -> prix synth -> metriques). "
    "So what: on mesure des deltas structurels, pas une prediction precise du spot reel."
)

left, right = st.columns([1, 1])
with left:
    info_card("Cadre", "Prix scenario = synthetic par defaut, affine par regimes et ancre TCA.")
with right:
    info_card("Lecture", "Chercher d'abord delta FAR, delta h_regime_a, puis delta capture_ratio_pv.")

with st.expander("Ajustements manuels", expanded=True):
    delta_pv_gw = st.slider(
        "delta_pv_gw",
        -10.0,
        40.0,
        float(base_params.get("delta_pv_gw", 0.0)),
        0.5,
        help="Variation de capacité PV installée par rapport à la baseline (en GW).",
    )
    delta_wind_onshore_gw = st.slider(
        "delta_wind_onshore_gw",
        -10.0,
        40.0,
        float(base_params.get("delta_wind_onshore_gw", 0.0)),
        0.5,
        help="Variation de capacité éolienne onshore installée par rapport à la baseline (en GW).",
    )
    delta_bess_power_gw = st.slider(
        "delta_bess_power_gw",
        0.0,
        25.0,
        float(base_params.get("delta_bess_power_gw", 0.0)),
        0.5,
        help="Ajout de puissance batterie (GW): limite de charge/décharge instantanée.",
    )
    delta_bess_energy_gwh = st.slider(
        "delta_bess_energy_gwh",
        0.0,
        120.0,
        float(base_params.get("delta_bess_energy_gwh", 0.0)),
        1.0,
        help="Ajout de capacité d'énergie batterie (GWh): volume stockable total.",
    )
    delta_demand_pct = st.slider(
        "delta_demand_pct",
        -20.0,
        30.0,
        float(base_params.get("delta_demand_pct", 0.0)),
        1.0,
        help="Variation proportionnelle de toute la courbe de charge annuelle (en %).",
    )
    delta_demand_midday_gw = st.slider(
        "delta_demand_midday_gw",
        0.0,
        20.0,
        float(base_params.get("delta_demand_midday_gw", 0.0)),
        0.5,
        help="Ajout absolu de charge locale sur les heures 11h-15h (heure locale), en GW.",
    )
    gas_price_eur_mwh = st.slider(
        "gas_price_eur_mwh",
        10.0,
        90.0,
        float(base_params.get("gas_price_eur_mwh", 35.0)),
        1.0,
        help="Override du prix gaz TTF (EUR/MWh_th), constant sur toute l'année de scénario.",
    )
    co2_price_eur_t = st.slider(
        "co2_price_eur_t",
        20.0,
        200.0,
        float(base_params.get("co2_price_eur_t", 80.0)),
        1.0,
        help="Override du prix CO2 EUA (EUR/tCO2), constant sur toute l'année de scénario.",
    )

st.markdown("#### Ce que signifie chaque curseur")
st.markdown(
    f"""
- `delta_pv_gw = {delta_pv_gw:+.1f}`: la capacité PV installée est augmentée/réduite de **{abs(delta_pv_gw):.1f} GW** vs baseline.
  Le profil horaire PV est **redimensionné** (même forme, amplitude modifiée).  
  Exemple: passer de `0` à `+27` signifie **+27 GW de capacité PV**.
- `delta_wind_onshore_gw = {delta_wind_onshore_gw:+.1f}`: même logique pour l'éolien onshore (capacité installée).
- `delta_bess_power_gw = {delta_bess_power_gw:+.1f}`: ajoute de la **puissance** batterie (vitesse max de charge/décharge).
- `delta_bess_energy_gwh = {delta_bess_energy_gwh:+.1f}`: ajoute de la **capacité d'énergie** batterie (stock total disponible).
- `delta_demand_pct = {delta_demand_pct:+.1f}%`: multiplie toute la demande horaire par `(1 + delta/100)`.
- `delta_demand_midday_gw = {delta_demand_midday_gw:+.1f}`: ajoute un bloc de charge sur 11h-15h (heure locale), en GW.
- `gas_price_eur_mwh = {gas_price_eur_mwh:.1f}`: force un prix gaz constant pour le calcul TCA/prix synth.
- `co2_price_eur_t = {co2_price_eur_t:.1f}`: force un prix CO2 constant pour le calcul TCA/prix synth.
"""
)
st.caption(
    "Tous les deltas sont appliqués par rapport à la baseline sélectionnée (pays/année), "
    "puis le pipeline est recalculé en entier."
)

params = {
    "delta_pv_gw": delta_pv_gw,
    "delta_wind_onshore_gw": delta_wind_onshore_gw,
    "delta_bess_power_gw": delta_bess_power_gw,
    "delta_bess_energy_gwh": delta_bess_energy_gwh,
    "delta_demand_pct": delta_demand_pct,
    "delta_demand_midday_gw": delta_demand_midday_gw,
    "gas_price_eur_mwh": gas_price_eur_mwh,
    "co2_price_eur_t": co2_price_eur_t,
}

ui_overrides = state.get("ui_overrides", {}) if isinstance(state.get("ui_overrides", {}), dict) else {}
scenario_payload = {**ui_overrides, **params}

base_key = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
if base_key not in proc:
    fallback = [k for k in proc.keys() if k[0] == country and k[1] == year]
    if fallback:
        base_key = sorted(fallback)[0]
    else:
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
            scenario_params=scenario_payload,
            price_mode=state["scenario_price_mode"],
        )

        base_row = df_all[(df_all["country"] == country) & (df_all["year"] == year)]
        if base_row.empty:
            guard_no_data("metriques baseline")
        m_base = normalize_metrics_record(base_row.iloc[0].to_dict())
        m_s = compute_annual_metrics(df_s, country, year, state["countries_cfg"][country])

    section_header("Resultats scenario", "Prix synthetic: affine par regimes, ancre sur TCA")
    row1 = st.columns(3)
    row1[0].metric("SR", f"{m_s['sr']:.3f}", f"{(m_s['sr'] - m_base['sr']):+.3f}")
    row1[1].metric("FAR", f"{m_s['far']:.3f}", f"{(m_s['far'] - m_base['far']):+.3f}")
    row1[2].metric("h_regime_a", f"{m_s['h_regime_a']:.0f}", f"{(m_s['h_regime_a'] - m_base['h_regime_a']):+.0f}")
    row2 = st.columns(3)
    row2[0].metric("TTL", f"{m_s['ttl']:.1f}", f"{(m_s['ttl'] - m_base['ttl']):+.1f}")
    row2[1].metric("capture_ratio_pv", f"{m_s['capture_ratio_pv']:.3f}", f"{(m_s['capture_ratio_pv'] - m_base['capture_ratio_pv']):+.3f}")

    render_commentary(comment_scenario_delta(m_base, m_s))

    if m_s["far"] < m_base["far"] and (params["delta_bess_power_gw"] > 0 or params["delta_bess_energy_gwh"] > 0):
        challenge_block(
            "Resultat contre-intuitif",
            "Le FAR baisse malgre ajout BESS. So what: verifier hypothese de surplus et autres deltas actifs qui peuvent dominer l'effet stockage.",
        )

    section_header("Graphique 1 - Heures par regime", "Baseline vs Scenario")
    chart_df = pd.DataFrame(
        {
            "regime": ["A", "B", "C", "D"],
            "baseline": [m_base["h_regime_a"], m_base["h_regime_b"], m_base["h_regime_c"], m_base["h_regime_d"]],
            "scenario": [m_s["h_regime_a"], m_s["h_regime_b"], m_s["h_regime_c"], m_s["h_regime_d"]],
        }
    )
    chart_df = coerce_numeric_columns(chart_df, ["baseline", "scenario"])
    fig1 = px.bar(chart_df, x="regime", y=["baseline", "scenario"], barmode="group")
    fig1.update_layout(height=420, title="Heures par regime — baseline vs scenario", **PLOTLY_LAYOUT_DEFAULTS)
    fig1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.caption("Comparaison du nombre d'heures par regime avant/apres scenario.")
    st.plotly_chart(fig1, use_container_width=True)

    render_commentary(
        so_what_block(
            title="Impact regimes",
            purpose="Mesurer si le scenario reduit les heures A (surplus non absorbe) et deplace vers des regimes plus soutenables",
            observed={
                "h_A_delta": float(m_s["h_regime_a"] - m_base["h_regime_a"]),
                "h_D_delta": float(m_s["h_regime_d"] - m_base["h_regime_d"]),
            },
            method_link="Reclassement regimes apres perturbation physique et dispatch BESS deterministe.",
            limits="Ne modelise pas explicitement les congestions intra-zonales.",
            n=4,
            decision_use="Verifier que le scenario cible reduit effectivement les heures A avant de communiquer un benefice.",
        )
    )

    section_header("Graphique 2 - Distribution price_used", "Scenario")
    fig2 = px.histogram(df_s, x="price_used_eur_mwh", nbins=80)
    fig2.update_layout(height=420, title="Distribution des prix mecanistes (scenario)", **PLOTLY_LAYOUT_DEFAULTS)
    fig2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.caption("Distribution des prix synthetiques calcules par le scenario.")
    st.plotly_chart(fig2, use_container_width=True)

    render_commentary(
        so_what_block(
            title="Signal prix scenario",
            purpose="Observer le deplacement de la distribution de prix utilises sous nouvelles contraintes systeme",
            observed={"p05": float(m_s["price_used_p05"]), "p95": float(m_s["price_used_p95"])},
            method_link="price_used = price_synth en mode scenario par defaut.",
            limits="Prix synthetique indicatif; ne pas interpreter comme forecast spot transactionnel.",
            n=len(df_s),
            decision_use="Tester rapidement la direction de variation de la queue de prix sous hypothese de scenario.",
        )
    )

    section_header("Tableau comparatif", "Baseline vs Scenario")
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
        so_what_block(
            title="Conclusion scenario",
            purpose="Traduire les deltas en decision de pilotage (priorite flex, rythme PV, exposition prix)",
            observed={
                "delta_sr": float(table.loc[table.metric == "sr", "delta"].iloc[0]),
                "delta_far": float(table.loc[table.metric == "far", "delta"].iloc[0]),
                "delta_capture_ratio_pv": float(table.loc[table.metric == "capture_ratio_pv", "delta"].iloc[0]),
            },
            method_link="Memes formules metriques que l'historique, comparaison coherent baseline/scenario.",
            limits="Resultat conditionnel aux hypotheses scenario actives.",
            n=len(table),
            decision_use="Selectionner les scenarios qui ameliorent FAR et capture ratio sans aggraver SR.",
        )
    )
