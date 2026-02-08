"""Page 1 - Analyse historique."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.commentary_engine import so_what_block
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

st.set_page_config(page_title="Analyse historique", page_icon="??", layout="wide")
inject_global_css()
st.title("?? Analyse Historique")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Analyse Historique")
normalize_state_metrics(state)

df_all = metrics_to_dataframe(state, state.get("price_mode"))
if df_all.empty:
    guard_no_data("la page Analyse Historique")

countries = sorted(df_all["country"].dropna().unique())
country = st.selectbox("Pays", countries)

df = df_all[df_all["country"] == country].copy().sort_values("year")
if df.empty:
    guard_no_data("la page Analyse Historique")

df = ensure_plot_columns(
    df,
    [
        "capture_ratio_pv",
        "capture_ratio_wind",
        "pv_penetration_pct_gen",
        "h_regime_a",
        "h_regime_b",
        "h_regime_c",
        "h_regime_d",
        "h_negative_obs",
        "h_below_5_obs",
        "sr",
        "far",
        "ir",
        "ttl",
    ],
)

if state.get("exclude_2022", True):
    df_vis = df[df["year"] != 2022].copy()
else:
    df_vis = df.copy()

narrative(
    "Cette page sert a lire la trajectoire d'un marche dans le temps: "
    "penetration VRE, degradation capture ratio, pression surplus et evolution des regimes."
)

section_header(
    "Heures negatives / tres basses et capture ratio PV",
    "Barres: heures observees | Lignes: capture ratio PV et penetration PV",
)

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(
    go.Bar(x=df_vis["year"], y=df_vis["h_negative_obs"], name="h_negative_obs", marker_color="#E74C3C"),
    secondary_y=False,
)
fig1.add_trace(
    go.Bar(x=df_vis["year"], y=df_vis["h_below_5_obs"], name="h_below_5_obs", marker_color="#F39C12", opacity=0.75),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(
        x=df_vis["year"],
        y=df_vis["capture_ratio_pv"],
        name="capture_ratio_pv",
        mode="lines+markers",
        line=dict(color="#27AE60", width=2),
    ),
    secondary_y=True,
)
fig1.add_trace(
    go.Scatter(
        x=df_vis["year"],
        y=df_vis["pv_penetration_pct_gen"] / 100.0,
        name="pv_penetration_pct_gen",
        mode="lines+markers",
        line=dict(color="#2980B9", width=2, dash="dash"),
    ),
    secondary_y=True,
)
fig1.update_layout(height=480, barmode="group", xaxis_title="Annee")
fig1.update_yaxes(title_text="Heures observees", secondary_y=False)
fig1.update_yaxes(title_text="Ratio / part", secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Signal de cannibalisation",
        purpose="Verifier si la hausse de penetration PV coincide avec une baisse du capture ratio et une hausse des heures contraintes",
        observed={
            "capture_ratio_start": float(df_vis["capture_ratio_pv"].iloc[0]) if not df_vis.empty else np.nan,
            "capture_ratio_end": float(df_vis["capture_ratio_pv"].iloc[-1]) if not df_vis.empty else np.nan,
            "h_negative_end": float(df_vis["h_negative_obs"].iloc[-1]) if not df_vis.empty else np.nan,
        },
        method_link="Observables prix sur price_da; capture ratio sur price_used, selon G.7.",
        limits="Association descriptive: la causalite complete exige une analyse multivariee complementaire.",
        n=len(df_vis),
    )
)

section_header("Repartition des regimes A/B/C/D", "Heures annuelles par regime")
fig2 = px.bar(
    df_vis,
    x="year",
    y=["h_regime_a", "h_regime_b", "h_regime_c", "h_regime_d"],
    labels={"value": "Heures", "variable": "Regime"},
)
fig2.update_layout(height=390)
st.plotly_chart(fig2, use_container_width=True)

render_commentary(
    so_what_block(
        title="Bascule structurelle",
        purpose="Suivre le poids des heures de surplus non absorbe (A) versus heures thermiques (C/D)",
        observed={
            "h_A_latest": float(df_vis["h_regime_a"].iloc[-1]) if not df_vis.empty else np.nan,
            "h_D_latest": float(df_vis["h_regime_d"].iloc[-1]) if not df_vis.empty else np.nan,
            "sr_latest": float(df_vis["sr"].iloc[-1]) if not df_vis.empty else np.nan,
        },
        method_link="Classification regimes strictement physique (NRL/surplus/flex).",
        limits="Depend du mode must-run/flex choisi pour la session.",
        n=len(df_vis),
    )
)

section_header("Tableau annuel", "SR/FAR/IR/TTL, capture et observables")
show_cols = [
    "year",
    "capture_ratio_pv",
    "capture_ratio_wind",
    "sr",
    "far",
    "ir",
    "ttl",
    "h_negative_obs",
    "h_below_5_obs",
    "phase",
]
st.dataframe(df_vis[show_cols], use_container_width=True, hide_index=True)

if not df_vis.empty and float(df_vis["capture_ratio_pv"].iloc[-1]) < 0.80:
    dynamic_narrative(
        f"{country} est sous le seuil de 0.80 sur le capture ratio PV (valeur recente {float(df_vis['capture_ratio_pv'].iloc[-1]):.2f}). "
        "So what: pression economique accrue sur les revenus merchant PV.",
        severity="warning",
    )

if not df_vis.empty and float(df_vis["far"].iloc[-1]) < 0.30 and float(df_vis["sr"].iloc[-1]) > 0.02:
    challenge_block(
        "Flexibilite insuffisante",
        "FAR faible avec SR deja materialise: le systeme absorbe mal les surplus, risque de regime A durable.",
    )

render_commentary(
    so_what_block(
        title="Synthese annuelle",
        purpose="Fournir une base objective pour le diagnostic de phase et la priorisation des leviers",
        observed={
            "sr_moy": float(df_vis["sr"].mean()) if not df_vis.empty else np.nan,
            "far_moy": float(df_vis["far"].mean()) if not df_vis.empty else np.nan,
            "ttl_moy": float(df_vis["ttl"].mean()) if not df_vis.empty else np.nan,
        },
        method_link="Toutes les metriques suivent les definitions v3 harmonisees.",
        limits="Les valeurs de crise (ex: 2022) peuvent biaiser les tendances si non exclues.",
        n=len(df_vis),
    )
)
