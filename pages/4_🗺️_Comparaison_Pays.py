"""
Page 4 -- Comparaison Pays
Radar multi-axes, scatter VRE vs capture, tableau comparatif, exports.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import io
import datetime

from src.constants import *
from src.export_utils import export_to_excel, export_to_gsheets
from src.ui_helpers import inject_global_css, narrative, guard_no_data

st.set_page_config(page_title="Comparaison Pays", page_icon="🗺️", layout="wide")
inject_global_css()
st.title("🗺️ Comparaison Pays")

# ── Validation session state ──────────────────────────────────────────────
required_keys = ["annual_metrics", "diagnostics", "selected_countries"]
if (not all(k in st.session_state for k in required_keys)
        or not st.session_state.get("annual_metrics")):
    guard_no_data("la Comparaison Pays")

annual_metrics: dict = st.session_state["annual_metrics"]
diagnostics: dict = st.session_state["diagnostics"]
selected_countries: list = st.session_state["selected_countries"]
year_range: tuple = st.session_state.get("year_range", (2015, 2024))

# ── Selecteurs ────────────────────────────────────────────────────────────
available_years = sorted({k[1] for k in annual_metrics.keys()})
col_sel1, col_sel2 = st.columns([1, 2])

with col_sel1:
    selected_year = st.selectbox("Annee de comparaison", available_years,
                                 index=len(available_years) - 1 if available_years else 0)

all_metric_names = [
    "capture_ratio_pv", "capture_ratio_wind", "h_negative", "h_below_5",
    "h_above_100", "far_structural", "ir", "spread_p95_p05", "vre_share",
    "pv_share", "wind_share", "sr", "baseload_price", "ttl",
    "h_regime_a", "h_regime_b", "h_regime_c", "h_regime_d_tail",
    "regime_coherence", "data_completeness",
]

# Preset views avec radio, fallback vers multiselect si colonnes absentes
preset_views = {
    "Vue synthetique": ["capture_ratio_pv", "h_negative", "far_structural", "vre_share"],
    "Vue detaillee": ["capture_ratio_pv", "capture_ratio_wind", "h_negative", "h_below_5",
                      "far_structural", "ir", "spread_p95_p05", "vre_share"],
    "Vue regimes": ["h_regime_a", "h_regime_b", "h_regime_c", "h_regime_d_tail", "regime_coherence"],
}

with col_sel2:
    view_choice = st.radio("Vue", list(preset_views.keys()), horizontal=True)
    selected_metrics = preset_views[view_choice]

    # Fallback : verifier que les metriques du preset existent dans les donnees
    # Collecter toutes les cles presentes dans les metriques de l'annee selectionnee
    available_metric_keys = set()
    for c in selected_countries:
        if (c, selected_year) in annual_metrics:
            available_metric_keys.update(annual_metrics[(c, selected_year)].keys())

    valid_preset_metrics = [m for m in selected_metrics if m in available_metric_keys]

    if not valid_preset_metrics:
        # Aucune metrique du preset n'est disponible -> fallback multiselect
        st.caption("Les metriques du preset ne sont pas disponibles. Selection manuelle :")
        selected_metrics = st.multiselect(
            "Metriques a afficher dans le tableau",
            all_metric_names,
            default=["capture_ratio_pv", "h_negative", "far_structural", "ir",
                     "spread_p95_p05", "vre_share", "baseload_price"],
            key="fallback_multiselect",
        )
    else:
        selected_metrics = valid_preset_metrics

# ── Filtrer les donnees pour l'annee selectionnee ─────────────────────────
countries_with_data = [c for c in selected_countries if (c, selected_year) in annual_metrics]

if not countries_with_data:
    st.info(f"Aucun pays avec des donnees pour {selected_year}.")
    st.stop()

metrics_year = {c: annual_metrics[(c, selected_year)] for c in countries_with_data}
diag_year = {c: diagnostics.get((c, selected_year), {}) for c in countries_with_data}

# ══════════════════════════════════════════════════════════════════════════
# CHART 1 -- Radar multi-axes
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Radar structurel")

narrative("Le radar compare le profil structurel de chaque marche sur 6 axes normalises. "
          "Le marche ideal pour les VRE est compact sur 'H negatives' et 'IR', et etendu "
          "sur 'Capture Ratio' et 'FAR'. Le FAR structural mesure la flex domestique "
          "(PSH + BESS + DSM, hors exports) — un FAR faible signale un deficit d'absorption.")

radar_axes = [
    ("capture_ratio_pv", "Capture Ratio PV", 1.0),
    ("h_negative", "H negatives (/1000)", 1000.0),
    ("far_structural", "FAR structural", 1.0),
    ("ir", "IR (inflexibilite)", 1.0),
    ("spread_p95_p05", "Spread P95-P05 (/200)", 200.0),
    ("vre_share", "VRE share", 1.0),
]
axis_labels = [a[1] for a in radar_axes]

fig_radar = go.Figure()

for idx, country in enumerate(countries_with_data):
    m = metrics_year[country]
    values = []
    for key, _, normalizer in radar_axes:
        raw = m.get(key, 0)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raw = 0
        values.append(raw / normalizer if normalizer != 1.0 else raw)
    # Fermer le polygone
    values_closed = values + [values[0]]
    labels_closed = axis_labels + [axis_labels[0]]

    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        name=country,
        opacity=0.55,
        line=dict(color=COUNTRY_PALETTE.get(country, "#999999")),
    ))

fig_radar.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])),
    showlegend=True,
    height=520,
    title=f"Profil structurel par pays -- {selected_year}",
)
st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# CHART 2 -- Scatter VRE share vs Capture Ratio PV
# ══════════════════════════════════════════════════════════════════════════
st.subheader("VRE share vs Capture Ratio PV")

narrative("Ce graphique positionne chaque marche sur la matrice VRE share vs Capture Ratio. "
          "La taille des bulles = nombre d'heures negatives. Couleur = pays, forme = phase.")

PHASE_SYMBOLS = {1: "circle", 2: "diamond", 3: "square", 4: "star"}

scatter_data = []
for country in countries_with_data:
    m = metrics_year[country]
    d = diag_year[country]
    scatter_data.append({
        "country": country,
        "vre_share": m.get("vre_share", 0),
        "capture_ratio_pv": m.get("capture_ratio_pv", np.nan),
        "h_negative": max(m.get("h_negative", 10), 10),  # min size = 10
        "phase_number": d.get("phase_number", 1),
    })

df_scatter = pd.DataFrame(scatter_data).dropna(subset=["capture_ratio_pv"])

if not df_scatter.empty:
    fig_scatter = go.Figure()
    max_h_neg = max(df_scatter["h_negative"].max(), 1)

    for _, row in df_scatter.iterrows():
        c = row["country"]
        phase = int(row["phase_number"])
        bubble_size = max(12, 50 * row["h_negative"] / max_h_neg)
        fig_scatter.add_trace(go.Scatter(
            x=[row["vre_share"]],
            y=[row["capture_ratio_pv"]],
            mode="markers+text",
            text=[f"{c} (S{phase})"],
            textposition="top center",
            marker=dict(
                size=bubble_size,
                color=COUNTRY_PALETTE.get(c, "#999999"),
                symbol=PHASE_SYMBOLS.get(phase, "circle"),
                line=dict(width=1, color="white"),
            ),
            name=f"{c} — Phase {phase}",
            hovertemplate=(
                f"<b>{c}</b> (Phase {phase})<br>"
                f"VRE: {row['vre_share']:.1%}<br>"
                f"Capture Ratio PV: {row['capture_ratio_pv']:.3f}<br>"
                f"H negatives: {int(row['h_negative'])}<extra></extra>"
            ),
        ))

    fig_scatter.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="Part VRE (% generation)",
        yaxis_title="Capture Ratio PV",
        title=f"Positionnement des marches -- {selected_year}",
        height=500,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Pas assez de donnees pour le scatter plot (capture_ratio_pv manquant).")

# ══════════════════════════════════════════════════════════════════════════
# TABLEAU COMPARATIF
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Tableau comparatif")

if selected_metrics:
    table_rows = []
    for country in countries_with_data:
        m = metrics_year[country]
        row = {"Pays": country}
        for metric in selected_metrics:
            row[metric] = m.get(metric, np.nan)
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows).set_index("Pays")
    st.dataframe(df_table, use_container_width=True)
else:
    st.info("Selectionnez au moins une metrique ci-dessus.")

# ══════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Export")

st.caption("Telecharger les donnees pour utilisation dans Excel ou Google Sheets.")

col_exp1, col_exp2 = st.columns(2)

# Preparer les donnees d'export
all_metrics_list = [metrics_year[c] for c in countries_with_data]
diag_list = [{"country": c, **diag_year[c]} for c in countries_with_data]
slopes_list = []  # Pas de slopes sur cette page

with col_exp1:
    if st.button("Telecharger Excel", type="primary"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filepath = f"data/exports/comparaison_{selected_year}_{timestamp}.xlsx"
        try:
            export_to_excel(all_metrics_list, diag_list, slopes_list, filepath)
            # Lire le fichier pour le download
            with open(filepath, "rb") as f:
                st.download_button(
                    label="Cliquer pour telecharger",
                    data=f.read(),
                    file_name=f"comparaison_{selected_year}_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.success(f"Export genere: {filepath}")
        except Exception as e:
            st.error(f"Erreur export Excel: {e}")

with col_exp2:
    if st.button("Exporter vers Google Sheets"):
        sheet_name = f"Comparaison_{selected_year}_{datetime.datetime.now().strftime('%Y%m%d')}"
        url = export_to_gsheets(all_metrics_list, diag_list, slopes_list, sheet_name)
        if url:
            st.success(f"Export Google Sheets: [{sheet_name}]({url})")
        else:
            st.warning("Credentials Google Sheets non configurees. Voir docs/SECRETS_REGISTER_LOCAL.md.")
