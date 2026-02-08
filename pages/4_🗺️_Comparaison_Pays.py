"""Page 4 - Comparaison pays."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import so_what_block
from src.export_utils import export_to_excel, export_to_gsheets
from src.state_adapter import ensure_plot_columns, metrics_to_dataframe
from src.ui_helpers import (
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
    section_header,
)

st.set_page_config(page_title="Comparaison pays", page_icon="???", layout="wide")
inject_global_css()
st.title("??? Comparaison Pays")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Comparaison pays")
normalize_state_metrics(state)

df_all = metrics_to_dataframe(state, state.get("price_mode"))
if df_all.empty:
    guard_no_data("la page Comparaison pays")

years = sorted(df_all["year"].dropna().unique())
year = st.selectbox("Annee", years, index=len(years) - 1)
countries = st.multiselect(
    "Pays",
    sorted(df_all["country"].dropna().unique()),
    default=state.get("countries_selected", []),
)

if not countries:
    guard_no_data("la page Comparaison pays")

df = df_all[(df_all["country"].isin(countries)) & (df_all["year"] == year)].copy()
if df.empty:
    guard_no_data("la page Comparaison pays")

df = ensure_plot_columns(
    df,
    ["sr", "far", "ir", "ttl", "capture_ratio_pv", "h_negative_obs", "vre_penetration_pct_gen", "phase"],
)

narrative(
    "Objectif: comparer les profils structurels pays a annee donnee pour prioriser les leviers "
    "(flex, must-run, gestion du surplus) et situer les marches sur une meme grille de lecture."
)

section_header("Radar structurel", "SR/FAR/IR/TTL/CRPV/Hneg normalises")
radar = df.copy()
axes = ["sr", "far", "ir", "ttl", "capture_ratio_pv", "h_negative_obs"]
for col in axes:
    vals = radar[col].astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if np.isclose(vmin, vmax):
        radar[col] = 0.5
    else:
        radar[col] = (vals - vmin) / (vmax - vmin)

fig1 = go.Figure()
for _, r in radar.iterrows():
    values = [r[a] for a in axes]
    fig1.add_trace(go.Scatterpolar(r=values + [values[0]], theta=axes + [axes[0]], fill="toself", name=r["country"]))
fig1.update_layout(height=470)
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Profil relatif",
        purpose="Identifier quels pays combinent niveau de surplus eleve et faible capacite d'absorption",
        observed={
            "sr_mean": float(df["sr"].mean()),
            "far_mean": float(df["far"].mean()),
            "ttl_mean": float(df["ttl"].mean()),
        },
        method_link="Normalisation min-max intra-echantillon sur l'annee selectionnee.",
        limits="Lecture relative: depend du panier de pays selectionnes.",
        n=len(df),
    )
)

section_header("VRE vs capture ratio PV", "Taille bulle = h_negative_obs")
fig2 = px.scatter(
    df,
    x="vre_penetration_pct_gen",
    y="capture_ratio_pv",
    color="country",
    size="h_negative_obs",
    hover_data=["phase", "sr", "far", "ir", "ttl"],
)
fig2.update_layout(height=430)
st.plotly_chart(fig2, use_container_width=True)

render_commentary(
    so_what_block(
        title="Positionnement concurrentiel",
        purpose="Visualiser les marches ou la penetration VRE est deja elevee avec une valeur captee encore resiliente (ou non)",
        observed={
            "vre_min_pct": float(df["vre_penetration_pct_gen"].min()),
            "vre_max_pct": float(df["vre_penetration_pct_gen"].max()),
            "capture_ratio_min": float(df["capture_ratio_pv"].min()),
        },
        method_link="Penetration = % generation; capture ratio calcule sur price_used.",
        limits="Photo a un instant t: ne montre pas la trajectoire temporelle.",
        n=len(df),
    )
)

section_header("Tableau comparatif", "Indicateurs pivots + phase")
show_cols = ["country", "sr", "far", "ir", "ttl", "capture_ratio_pv", "h_negative_obs", "phase"]
st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

render_commentary(
    so_what_block(
        title="Lecture tabulaire",
        purpose="Fournir une base de decision rapide pour arbitrage multi-pays",
        observed={
            "ttl_median": float(df["ttl"].median()),
            "h_negative_total": float(df["h_negative_obs"].sum()),
        },
        method_link="Schema harmonise v3 et mapping legacy force.",
        limits="Ne remplace pas l'analyse intra-annuelle des regimes (voir NRL Deep Dive).",
        n=len(df),
    )
)

section_header("Export", "Excel toujours dispo, Google Sheets optionnel")
metrics_rows = df.to_dict("records")
diag_rows = []
for c in countries:
    diag = state.get("diagnostics", {}).get((c, year), {})
    if diag:
        diag_rows.append({"country": c, "year": int(year), **diag})

col1, col2 = st.columns(2)
with col1:
    if st.button("Exporter Excel", type="primary"):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        path = f"data/exports/comparaison_{year}_{ts}.xlsx"
        export_to_excel(metrics_rows, diag_rows, [], path)
        with open(path, "rb") as f:
            st.download_button(
                label="Telecharger le fichier",
                data=f.read(),
                file_name=f"comparaison_{year}_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with col2:
    if st.button("Exporter Google Sheets"):
        url = export_to_gsheets(metrics_rows, diag_rows, [], f"Comparaison_{year}")
        if url:
            st.success(f"Export cree: {url}")
        else:
            st.warning("Credentials absentes ou export indisponible.")
