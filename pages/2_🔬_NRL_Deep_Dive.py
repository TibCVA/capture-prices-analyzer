"""Page 2 - NRL deep dive."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import commentary_block
from src.constants import COL_FLEX_EFFECTIVE, COL_LOAD, COL_NRL, COL_PRICE_DA, COL_REGIME, COL_SURPLUS, COL_SURPLUS_UNABS, COL_VRE
from src.ui_helpers import guard_no_data, inject_global_css, render_commentary, section

st.set_page_config(page_title="NRL Deep Dive", page_icon="🔬", layout="wide")
inject_global_css()

st.title("🔬 NRL Deep Dive")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page NRL Deep Dive")

proc = state["processed"]
if not proc:
    guard_no_data("la page NRL Deep Dive")

pairs = sorted({(k[0], k[1]) for k in proc.keys()})
country = st.selectbox("Pays", sorted({p[0] for p in pairs}))
year = st.selectbox("Annee", sorted({p[1] for p in pairs if p[0] == country}))
key = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
if key not in proc:
    guard_no_data("la page NRL Deep Dive")

metrics = state["metrics"].get((country, year, state["price_mode"]), {})
df = proc[key]

section("Traces horaires", "Load, VRE, NRL, surplus et flex")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_LOAD], name="load"))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_VRE], name="vre"))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_NRL], name="nrl"))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_SURPLUS], name="surplus"))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_FLEX_EFFECTIVE], name="flex_effective"))
fig1.update_layout(height=420, xaxis_title="Heure", yaxis_title="MW")
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    commentary_block(
        title="Lecture temporelle NRL",
        n_label="heures",
        n_value=len(df),
        observed={
            "nrl_min": float(df[COL_NRL].min()),
            "nrl_max": float(df[COL_NRL].max()),
            "surplus_total_mwh": float(df[COL_SURPLUS].sum()),
        },
        method_link="Surplus=max(0,-NRL), puis absorption par flex_effective selon G.6.",
        limits="Visualisation annuelle dense; utiliser des zooms temporels pour l'analyse operationnelle.",
    )
)

section("Distribution NRL et surplus non absorbe", "Histogrammes")
col1, col2 = st.columns(2)
with col1:
    fig2 = px.histogram(df, x=COL_NRL, nbins=80)
    fig2.update_layout(height=330)
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = px.histogram(df, x=COL_SURPLUS_UNABS, nbins=60)
    fig3.update_layout(height=330)
    st.plotly_chart(fig3, use_container_width=True)

render_commentary(
    commentary_block(
        title="Distribution physique",
        n_label="heures",
        n_value=len(df),
        observed={
            "h_surplus": int((df[COL_SURPLUS] > 0).sum()),
            "h_surplus_unabs": int((df[COL_SURPLUS_UNABS] > 0).sum()),
        },
        method_link="Regime A correspond exactement aux heures surplus_unabsorbed > 0.",
        limits="La distribution depend des hypothese must-run/flex actives dans la session.",
    )
)

section("Scatter NRL vs prix observe", "Validation coherence regime/prix")
fig4 = px.scatter(df.dropna(subset=[COL_NRL, COL_PRICE_DA]), x=COL_NRL, y=COL_PRICE_DA, color=COL_REGIME, opacity=0.35)
fig4.update_layout(height=420)
st.plotly_chart(fig4, use_container_width=True)

coh = metrics.get("regime_coherence")
render_commentary(
    commentary_block(
        title="Coherence regime/prix",
        n_label="points",
        n_value=int(df[[COL_NRL, COL_PRICE_DA]].dropna().shape[0]),
        observed={"regime_coherence": coh},
        method_link="Cohérence calculee uniquement en mode observed selon thresholds.coherence_params.",
        limits="Un score <55% signale une divergence modele/marche a investiguer (configuration ou donnees).",
    )
)

if coh is not None and coh == coh and coh < 0.55:
    st.warning(f"Coherence regime/prix basse: {coh:.1%} (<55%).")
