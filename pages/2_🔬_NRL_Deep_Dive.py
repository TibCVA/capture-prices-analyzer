"""Page 2 - NRL Deep Dive."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import so_what_block
from src.constants import (
    COL_FLEX_EFFECTIVE,
    COL_LOAD,
    COL_NRL,
    COL_PRICE_DA,
    COL_REGIME,
    COL_SURPLUS,
    COL_SURPLUS_UNABS,
    COL_VRE,
)
from src.state_adapter import ensure_plot_columns, metrics_to_dataframe
from src.ui_helpers import (
    challenge_block,
    dynamic_narrative,
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
    section_header,
)

st.set_page_config(page_title="NRL Deep Dive", page_icon="??", layout="wide")
inject_global_css()
st.title("?? NRL Deep Dive")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page NRL Deep Dive")
normalize_state_metrics(state)

proc = state.get("processed", {})
metrics_df = metrics_to_dataframe(state, state.get("price_mode"))
if not proc or metrics_df.empty:
    guard_no_data("la page NRL Deep Dive")

pairs = sorted({(k[0], k[1]) for k in proc.keys()})
country = st.selectbox("Pays", sorted({p[0] for p in pairs}))
year = st.selectbox("Annee", sorted({p[1] for p in pairs if p[0] == country}))

key = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
if key not in proc:
    fallback = [k for k in proc.keys() if k[0] == country and k[1] == year]
    if not fallback:
        guard_no_data("la page NRL Deep Dive")
    key = sorted(fallback)[0]

df = ensure_plot_columns(
    proc[key],
    [COL_LOAD, COL_VRE, COL_NRL, COL_SURPLUS, COL_FLEX_EFFECTIVE, COL_SURPLUS_UNABS, COL_REGIME, COL_PRICE_DA],
)

narrative(
    "Objectif de cette page: ouvrir la boite physique heure par heure pour comprendre d'ou viennent les regimes, "
    "puis verifier si la structure NRL est coherente avec les prix observes."
)

section_header("Traces horaires", "Load, VRE, NRL, surplus et flexibilite")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_LOAD], name="load", line=dict(color="#1B2A4A", width=1.7)))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_VRE], name="vre", line=dict(color="#27AE60", width=1.7)))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_NRL], name="nrl", line=dict(color="#E74C3C", width=1.7)))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_SURPLUS], name="surplus", line=dict(color="#F39C12", width=1.2)))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_FLEX_EFFECTIVE], name="flex_effective", line=dict(color="#2980B9", width=1.2)))
fig1.update_layout(height=430, xaxis_title="Heure", yaxis_title="MW")
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Lecture temporelle",
        purpose="Identifier les episodes de surplus et juger si la flexibilite suit la dynamique du surplus",
        observed={
            "nrl_min": float(df[COL_NRL].min()),
            "nrl_max": float(df[COL_NRL].max()),
            "surplus_total_mwh": float(df[COL_SURPLUS].sum()),
        },
        method_link="Pipeline physique: NRL -> surplus -> flex_effective -> surplus_unabsorbed.",
        limits="Graphique dense sur annee complete; utiliser zoom temporel pour diagnostic operationnel.",
        n=len(df),
    )
)

section_header("Distribution NRL et surplus non absorbe", "Histogrammes")
c1, c2 = st.columns(2)
with c1:
    fig2 = px.histogram(df, x=COL_NRL, nbins=80, color=COL_REGIME)
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    fig3 = px.histogram(df, x=COL_SURPLUS_UNABS, nbins=60)
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

render_commentary(
    so_what_block(
        title="Distribution physique",
        purpose="Quantifier la masse d'heures exposees au regime A (surplus non absorbe)",
        observed={
            "h_surplus": int((df[COL_SURPLUS] > 0).sum()),
            "h_surplus_unabs": int((df[COL_SURPLUS_UNABS] > 0).sum()),
            "h_regime_a": int((df[COL_REGIME] == "A").sum()),
        },
        method_link="Regime A <=> surplus_unabsorbed > 0 par definition.",
        limits="Depend fortement des hypotheses must-run/flex actives dans la session.",
        n=len(df),
    )
)

section_header("NRL vs prix observe", "Test de coherence regime/prix")
scatter_df = df[[COL_NRL, COL_PRICE_DA, COL_REGIME]].dropna().copy()
if scatter_df.empty:
    st.info("Pas assez de points valides (NRL + prix observe).")
else:
    fig4 = px.scatter(scatter_df, x=COL_NRL, y=COL_PRICE_DA, color=COL_REGIME, opacity=0.35)
    fig4.update_layout(height=430, xaxis_title="NRL (MW)", yaxis_title="Prix observe (EUR/MWh)")
    st.plotly_chart(fig4, use_container_width=True)

row = metrics_df[(metrics_df["country"] == country) & (metrics_df["year"] == year)]
coh = float("nan")
if not row.empty:
    coh = float(row.iloc[0].get("regime_coherence", np.nan))

render_commentary(
    so_what_block(
        title="Coherence regime/prix",
        purpose="Verifier si la segmentation physique des heures reste predictive du niveau de prix observe",
        observed={"regime_coherence": coh},
        method_link="Coherence calculee en observed selon thresholds.coherence_params.",
        limits="Un score <55% peut signaler hypotheses incompletes (must-run, net position, qualite donnees).",
        n=len(scatter_df),
    )
)

if coh == coh and coh < 0.55:
    challenge_block(
        "Coherence faible",
        f"Score {coh:.1%} < 55%. So what: prudence sur l'interpretation causale avant recalibration des hypotheses.",
    )
else:
    dynamic_narrative(
        "Coherence regime/prix au niveau attendu ou superieur. So what: base solide pour interpretation structurelle.",
        severity="success",
    )
