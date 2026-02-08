"""Page 4 - Comparaison pays."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import so_what_block
from src.export_utils import export_to_excel, export_to_gsheets
from src.state_adapter import coerce_numeric_columns, ensure_plot_columns, metrics_to_dataframe
from src.ui_theme import COUNTRY_PALETTE, PHASE_COLORS, PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS
from src.ui_helpers import (
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
    section_header,
)

st.set_page_config(page_title="Comparaison pays", page_icon="🗺️", layout="wide")
inject_global_css()
st.title("🗺️ Comparaison Pays")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Comparaison pays")
normalize_state_metrics(state)

df_all = metrics_to_dataframe(state, state.get("price_mode"))
if df_all.empty or "country" not in df_all.columns:
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
    with_notice=True,
)
df = coerce_numeric_columns(df, ["sr", "far", "ir", "ttl", "capture_ratio_pv", "h_negative_obs", "vre_penetration_pct_gen"])

missing_cols = df.attrs.get("_missing_plot_columns", [])
if missing_cols:
    st.info("Colonnes manquantes completees en NaN pour robustesse d'affichage: " + ", ".join(missing_cols))

narrative(
    "Objectif: comparer la structure de stress des pays sur une meme annee. "
    "Le radar est oriente en mode 'stress': plus la valeur est elevee, plus le pays est sous pression structurelle."
)

section_header("Radar structurel (profil de stress)", "Axes normalises de 0 a 1, orientation unique stress")

radar_raw = df.copy()
radar_raw["far_stress"] = 1.0 - radar_raw["far"]
radar_raw["capture_ratio_pv_stress"] = 1.0 - radar_raw["capture_ratio_pv"]

axes_map = {
    "SR": "sr",
    "FAR (stress)": "far_stress",
    "IR": "ir",
    "TTL": "ttl",
    "Capture PV (stress)": "capture_ratio_pv_stress",
    "Heures negatives": "h_negative_obs",
}

radar = radar_raw.copy()
for col in axes_map.values():
    vals = radar[col].astype(float)
    finite = vals[np.isfinite(vals)]
    if finite.empty:
        radar[col] = 0.0
        continue
    vmin, vmax = float(finite.min()), float(finite.max())
    if np.isclose(vmin, vmax):
        radar[col] = 0.5
    else:
        radar[col] = (vals - vmin) / (vmax - vmin)

fig1 = go.Figure()
axes_labels = list(axes_map.keys())
for _, r in radar.iterrows():
    values = [float(r[axes_map[a]]) for a in axes_labels]
    country = str(r["country"])
    fig1.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=axes_labels + [axes_labels[0]],
            fill="toself",
            name=country,
            line=dict(color=COUNTRY_PALETTE.get(country, "#64748b"), width=2.0),
            opacity=0.35,
        )
    )

fig1.update_layout(
    height=500,
    polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10), gridcolor="#dbe5f1")),
    **PLOTLY_LAYOUT_DEFAULTS,
)
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Lecture radar orientee stress",
        purpose="Le radar permet de voir en un coup d'oeil les pays ou le stress est concentre sur surplus, rigidite ou cannibalisation.",
        observed={
            "n_pays": len(radar),
            "sr_mean": float(df["sr"].mean()),
            "far_mean": float(df["far"].mean()),
            "capture_ratio_pv_mean": float(df["capture_ratio_pv"].mean()),
        },
        method_link="Normalisation min-max intra-echantillon; inversion FAR/capture ratio pour une orientation stress uniforme.",
        limits="Comparaison relative au panier selectionne; ajouter/retirer un pays change la normalisation.",
        n=len(radar),
        decision_use="Prioriser les pays a traiter en premier selon la nature dominante du stress.",
    )
)

section_header("Valeurs brutes derriere le radar", "Transparence des valeurs non normalisees")
raw_cols = ["country", "sr", "far", "ir", "ttl", "capture_ratio_pv", "h_negative_obs", "phase"]
st.dataframe(df[raw_cols], use_container_width=True, hide_index=True)

section_header("Penetration VRE vs capture ratio PV", "Taille bulle = heures negatives")
fig2 = px.scatter(
    df,
    x="vre_penetration_pct_gen",
    y="capture_ratio_pv",
    color="country",
    size="h_negative_obs",
    color_discrete_map=COUNTRY_PALETTE,
    hover_data=["phase", "sr", "far", "ir", "ttl"],
)
fig2.update_layout(height=430, xaxis_title="Penetration VRE (% generation)", yaxis_title="Capture ratio PV", **PLOTLY_LAYOUT_DEFAULTS)
fig2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.plotly_chart(fig2, use_container_width=True)

render_commentary(
    so_what_block(
        title="Positionnement competitif",
        purpose="Comparer les pays a penetration elevee selon leur capacite a maintenir un capture ratio defendable.",
        observed={
            "vre_min_pct": float(df["vre_penetration_pct_gen"].min()),
            "vre_max_pct": float(df["vre_penetration_pct_gen"].max()),
            "capture_ratio_min": float(df["capture_ratio_pv"].min()),
            "h_negative_total": float(df["h_negative_obs"].sum()),
        },
        method_link="Penetration en % generation (v3); capture ratio sur price_used.",
        limits="Photographie annuelle: completer avec trajectoires temporelles (pages Historique et Capture Rates).",
        n=len(df),
        decision_use="Identifier les pays ou accelerer la flexibilite avant d'augmenter encore la penetration VRE.",
    )
)

section_header("Comparaison phase vs TTL", "Diagnostic de stade et queue haute")
phase_df = df.copy()
phase_df["phase"] = phase_df["phase"].fillna("unknown")
fig3 = px.bar(
    phase_df.sort_values("ttl", ascending=False),
    x="country",
    y="ttl",
    color="phase",
    color_discrete_map=PHASE_COLORS,
    hover_data=["sr", "far", "ir", "capture_ratio_pv"],
)
fig3.update_layout(height=380, xaxis_title="Pays", yaxis_title="TTL (EUR/MWh)", **PLOTLY_LAYOUT_DEFAULTS)
fig3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.plotly_chart(fig3, use_container_width=True)

render_commentary(
    so_what_block(
        title="Queue thermique et stade",
        purpose="Un TTL eleve peut renforcer la valeur de flexibilite, mais signale aussi une queue de prix plus risquee.",
        observed={"ttl_median": float(df["ttl"].median()), "ttl_max": float(df["ttl"].max())},
        method_link="TTL = P95(price_used) sur regimes C+D; phase issue du score thresholds.yaml.",
        limits="TTL peut etre influence par chocs commodites independamment de la penetration VRE.",
        n=len(df),
        decision_use="Arbitrer entre leviers de couverture prix et leviers de flexibilite physique.",
    )
)

section_header("Export", "Excel toujours disponible, Google Sheets optionnel")
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
