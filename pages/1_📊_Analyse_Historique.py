"""Page 1 - Analyse historique."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import commentary_block
from src.ui_helpers import guard_no_data, inject_global_css, render_commentary, section

st.set_page_config(page_title="Analyse historique", page_icon="📊", layout="wide")
inject_global_css()

st.title("📊 Analyse historique")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Analyse historique")

metrics_dict = state["metrics"]
if not metrics_dict:
    guard_no_data("la page Analyse historique")

countries = sorted({k[0] for k in metrics_dict.keys()})
country = st.selectbox("Pays", countries)
years = sorted([k[1] for k in metrics_dict if k[0] == country])

rows = []
for y in years:
    m = metrics_dict.get((country, y, state["price_mode"]))
    if m:
        rows.append(m)

df = pd.DataFrame(rows).sort_values("year")
if df.empty:
    guard_no_data("la page Analyse historique")

section("Capture ratio vs penetration PV", "Serie 2015-2024")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["year"], y=df["capture_ratio_pv"], name="capture_ratio_pv", mode="lines+markers"))
fig1.add_trace(
    go.Scatter(
        x=df["year"],
        y=df["pv_penetration_pct_gen"] / 100.0,
        name="pv_penetration_pct_gen",
        mode="lines+markers",
        line=dict(dash="dot"),
    )
)
fig1.update_layout(height=380, xaxis_title="Annee", yaxis_title="Ratio")
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    commentary_block(
        title="Evolution annuelle capture vs penetration",
        n_label="annees",
        n_value=len(df),
        observed={
            "capture_ratio_pv_start": float(df["capture_ratio_pv"].iloc[0]),
            "capture_ratio_pv_end": float(df["capture_ratio_pv"].iloc[-1]),
            "pv_penetration_start_pct": float(df["pv_penetration_pct_gen"].iloc[0]),
            "pv_penetration_end_pct": float(df["pv_penetration_pct_gen"].iloc[-1]),
        },
        method_link="Capture calculee sur price_used; penetration calculee en % de generation totale.",
        limits="Lecture descriptive; la causalite exige une analyse multivariee complementaire.",
    )
)

section("Regimes A/B/C/D", "Heures annuelles")
fig2 = px.bar(
    df,
    x="year",
    y=["h_regime_a", "h_regime_b", "h_regime_c", "h_regime_d"],
    labels={"value": "Heures", "variable": "Regime"},
)
fig2.update_layout(height=380)
st.plotly_chart(fig2, use_container_width=True)

render_commentary(
    commentary_block(
        title="Repartition des regimes",
        n_label="annees",
        n_value=len(df),
        observed={
            "h_A_latest": float(df["h_regime_a"].iloc[-1]),
            "h_C_latest": float(df["h_regime_c"].iloc[-1]),
            "h_D_latest": float(df["h_regime_d"].iloc[-1]),
        },
        method_link="Regimes derives de NRL/surplus/flex uniquement (anti-circularite).",
        limits="Depend de la disponibilite des donnees physiques (notamment net_position et psh_pumping).",
    )
)

section("Tableau annuel", "Metriques clefs")
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
]
st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

render_commentary(
    commentary_block(
        title="Tableau de synthese",
        n_label="lignes",
        n_value=len(df),
        observed={"sr_moy": float(df["sr"].mean()), "far_moy": float(df["far"].mean())},
        method_link="Toutes les metriques suivent les formules G.7; observables prix sur price_da uniquement.",
        limits="Toute comparaison inter-annee doit tenir compte des outliers et de la completude donnees.",
    )
)
