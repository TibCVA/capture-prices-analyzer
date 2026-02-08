"""Page 4 - Comparaison pays."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import commentary_block
from src.ui_helpers import guard_no_data, inject_global_css, normalize_state_metrics, render_commentary, section

st.set_page_config(page_title="Comparaison pays", page_icon="🗺️", layout="wide")
inject_global_css()

st.title("🗺️ Comparaison pays")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Comparaison pays")
normalize_state_metrics(state)

metrics = state["metrics"]
if not metrics:
    guard_no_data("la page Comparaison pays")

year = st.selectbox("Annee", sorted({k[1] for k in metrics.keys()}), index=len(sorted({k[1] for k in metrics.keys()})) - 1)
countries = st.multiselect("Pays", sorted({k[0] for k in metrics.keys()}), default=state["countries_selected"])

rows = []
for c in countries:
    key = (c, year, state["price_mode"])
    if key in metrics:
        m = metrics[key]
        d = state["diagnostics"].get((c, year), {})
        rows.append(
            {
                "country": c,
                "sr": m.get("sr"),
                "far": m.get("far"),
                "ir": m.get("ir"),
                "ttl": m.get("ttl"),
                "capture_ratio_pv": m.get("capture_ratio_pv"),
                "h_negative_obs": m.get("h_negative_obs"),
                "phase": d.get("phase", "unknown"),
            }
        )

df = pd.DataFrame(rows)
if df.empty:
    guard_no_data("la page Comparaison pays")

section("Radar structurel", "SR/FAR/IR/TTL/CRPV/Hneg normalises")
radar = df.copy()
for col in ["sr", "far", "ir", "ttl", "capture_ratio_pv", "h_negative_obs"]:
    vals = radar[col].astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if np.isclose(vmin, vmax):
        radar[col] = 0.5
    else:
        radar[col] = (vals - vmin) / (vmax - vmin)

axes = ["sr", "far", "ir", "ttl", "capture_ratio_pv", "h_negative_obs"]
fig1 = go.Figure()
for _, r in radar.iterrows():
    values = [r[a] for a in axes]
    fig1.add_trace(
        go.Scatterpolar(r=values + [values[0]], theta=axes + [axes[0]], fill="toself", name=r["country"])
    )
fig1.update_layout(height=450)
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    commentary_block(
        title="Profil multi-dimensionnel",
        n_label="pays",
        n_value=len(df),
        observed={"sr_mean": float(df["sr"].mean()), "far_mean": float(df["far"].mean())},
        method_link="Comparaison sur les 4 ratios pivots + observables, normalisation min-max annuelle.",
        limits="La normalisation depend de l'echantillon de pays selectionne.",
    )
)

section("Scatter VRE vs capture ratio PV", "Taille = h_negative_obs")
plot_df = df.copy()
plot_df["vre_penetration_pct_gen"] = [
    metrics[(c, year, state["price_mode"])].get("vre_penetration_pct_gen") for c in plot_df["country"]
]
fig2 = px.scatter(
    plot_df,
    x="vre_penetration_pct_gen",
    y="capture_ratio_pv",
    color="country",
    size="h_negative_obs",
    hover_data=["phase", "sr", "far", "ir", "ttl"],
)
fig2.update_layout(height=420)
st.plotly_chart(fig2, use_container_width=True)

render_commentary(
    commentary_block(
        title="Positionnement relatif des pays",
        n_label="points",
        n_value=len(plot_df),
        observed={
            "vre_penetration_pct_min": float(plot_df["vre_penetration_pct_gen"].min()),
            "vre_penetration_pct_max": float(plot_df["vre_penetration_pct_gen"].max()),
        },
        method_link="Capture ratio calcule sur price_used et penetration en % generation (G.7).",
        limits="Comparaison instantanee (une annee); ne capture pas la dynamique temporelle.",
    )
)

section("Tableau compare", "Colonnes limitees aux indicateurs pivots")
st.dataframe(df, use_container_width=True, hide_index=True)

render_commentary(
    commentary_block(
        title="Tableau comparatif",
        n_label="lignes",
        n_value=len(df),
        observed={"ttl_median": float(df["ttl"].median()), "h_negative_total": float(df["h_negative_obs"].sum())},
        method_link="Agrégation annuelle harmonisee par pays sur le meme schema de colonnes.",
        limits="Les differences de mix national et de donnees manquantes peuvent biaiser les comparaisons directes.",
    )
)
