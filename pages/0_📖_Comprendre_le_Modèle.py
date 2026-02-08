"""Page 0 - Comprendre le modele (version detaillee)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import commentary_block
from src.constants import (
    COL_FLEX_EFFECTIVE,
    COL_LOAD,
    COL_MUST_RUN,
    COL_NRL,
    COL_PRICE_DA,
    COL_REGIME,
    COL_SURPLUS,
    COL_SURPLUS_UNABS,
    COL_VRE,
)
from src.ui_helpers import guard_no_data, inject_global_css, normalize_state_metrics, render_commentary, section

st.set_page_config(page_title="Comprendre le modele", page_icon="📖", layout="wide")
inject_global_css()

st.title("📖 Comprendre le Modele")
st.caption("Cadre methodologique v3.0: mecanismes physiques, ratios pivots et lecture analytique")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Comprendre le modele")
normalize_state_metrics(state)

proc = state.get("processed", {})
if not proc:
    guard_no_data("la page Comprendre le modele")

section("1) Pourquoi ce modele", "Du signal prix au mecanisme physique")
st.markdown(
    "Le modele relie directement les prix a la physique du systeme electrique.\n\n"
    "- Point de depart: `NRL = load - VRE - must-run`.\n"
    "- Si `NRL < 0`: surplus, pression sur les prix, apparition de prix tres bas/negatifs.\n"
    "- Si `NRL > 0`: besoin thermique, ancrage des prix sur le cout marginal (TCA).\n"
    "- Les regimes `A/B/C/D` sont classes **sans utiliser le prix** (anti-circularite)."
)

render_commentary(
    commentary_block(
        title="Logique de construction",
        n_label="etapes", n_value=4,
        observed={"regimes": 4, "ratios_pivots": 4},
        method_link="Pipeline: NRL -> surplus/flex -> regimes -> TCA/prix -> metriques.",
        limits="Cadre structurel: utile pour expliquer les mecanismes, pas pour prevoir finement le spot horaire.",
    )
)

# pick a sample year/country from loaded data
pairs = sorted({(k[0], k[1]) for k in proc.keys()})
country = st.selectbox("Pays (exemple didactique)", sorted({p[0] for p in pairs}), key="model_country")
year = st.selectbox("Annee (exemple didactique)", sorted({p[1] for p in pairs if p[0] == country}), key="model_year")
proc_key = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
if proc_key not in proc:
    guard_no_data("la page Comprendre le modele")

df = proc[proc_key]

section("2) Exemple temporel 48h", "Load, VRE, Must-run, NRL, surplus")
df48 = df.head(48).copy()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_LOAD], name="Load", line=dict(color="#111827", width=2)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_VRE], name="VRE", line=dict(color="#16a34a", width=2)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_MUST_RUN], name="Must-run", line=dict(color="#6b7280", width=2)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_NRL], name="NRL", line=dict(color="#dc2626", width=2, dash="dash")))
fig1.add_hline(y=0, line_dash="dot", line_color="#334155")
fig1.update_layout(height=430, xaxis_title="Heure", yaxis_title="MW")
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    commentary_block(
        title="Lecture 48h",
        n_label="heures", n_value=len(df48),
        observed={
            "nrl_min_mw": float(df48[COL_NRL].min()),
            "nrl_max_mw": float(df48[COL_NRL].max()),
            "h_nrl_neg": int((df48[COL_NRL] < 0).sum()),
        },
        method_link="Le surplus brut est `max(0, -NRL)` avant absorption par la flex.",
        limits="Fenetre 48h illustrative; la robustesse des conclusions se lit sur l'annee complete.",
    )
)

section("3) Distribution annuelle", "Comment lire les zones de regime")
col_left, col_right = st.columns(2)
with col_left:
    fig2 = px.histogram(df, x=COL_NRL, nbins=100, color=COL_REGIME)
    fig2.update_layout(height=360, xaxis_title="NRL (MW)", yaxis_title="Nombre d'heures")
    st.plotly_chart(fig2, use_container_width=True)
with col_right:
    fig3 = px.histogram(df, x=COL_SURPLUS_UNABS, nbins=80)
    fig3.update_layout(height=360, xaxis_title="Surplus non absorbe (MW)", yaxis_title="Nombre d'heures")
    st.plotly_chart(fig3, use_container_width=True)

render_commentary(
    commentary_block(
        title="Structure annuelle des regimes",
        n_label="heures", n_value=len(df),
        observed={
            "h_regime_A": int((df[COL_REGIME] == "A").sum()),
            "h_regime_B": int((df[COL_REGIME] == "B").sum()),
            "h_regime_C": int((df[COL_REGIME] == "C").sum()),
            "h_regime_D": int((df[COL_REGIME] == "D").sum()),
        },
        method_link="A: surplus non absorbe, B: surplus absorbe, C/D: heures a NRL positif.",
        limits="La frontiere C/D depend du parametre `thresholds.model_params.regime_d`.",
    )
)

section("4) Prix observes et ancre thermique", "Validation regime/prix")
fig4 = px.scatter(
    df.dropna(subset=[COL_NRL, COL_PRICE_DA]),
    x=COL_NRL,
    y=COL_PRICE_DA,
    color=COL_REGIME,
    opacity=0.35,
)
fig4.update_layout(height=420, xaxis_title="NRL (MW)", yaxis_title="Prix observe (EUR/MWh)")
st.plotly_chart(fig4, use_container_width=True)

coh = np.nan
if "regime_coherence" in state.get("metrics", {}).get((country, year, state["price_mode"]), {}):
    coh = state["metrics"][(country, year, state["price_mode"])]["regime_coherence"]

render_commentary(
    commentary_block(
        title="Cohérence regime/prix",
        n_label="points valides",
        n_value=int(df[[COL_NRL, COL_PRICE_DA]].dropna().shape[0]),
        observed={"coherence": coh},
        method_link="Le score mesure l'alignement des prix observes avec les regimes physiques classes ex ante.",
        limits="Seuil indicatif >55%; en dessous, revoir configuration/flex/donnees avant interpretation forte.",
    )
)

section("5) Ratios pivots a retenir", "SR / FAR / IR / TTL")
metric_key = (country, year, state["price_mode"])
m = state.get("metrics", {}).get(metric_key)
if m:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SR", f"{m.get('sr', float('nan')):.3f}")
    c2.metric("FAR", f"{m.get('far', float('nan')):.3f}")
    c3.metric("IR", f"{m.get('ir', float('nan')):.3f}")
    c4.metric("TTL", f"{m.get('ttl', float('nan')):.1f} EUR/MWh")

    render_commentary(
        commentary_block(
            title="Synthese des ratios",
            n_label="annee", n_value=1,
            observed={
                "SR": m.get("sr"),
                "FAR": m.get("far"),
                "IR": m.get("ir"),
                "TTL": m.get("ttl"),
            },
            method_link="Ratios calcules strictement selon les definitions de la spec G.7.",
            limits="Les ratios ne remplacent pas une analyse causale complete; ils structurent le diagnostic.",
        )
    )
