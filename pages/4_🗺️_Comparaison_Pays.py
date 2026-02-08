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
from src.phase_context import compute_h_negative_declining_flags
from src.phase_diagnostics import diagnose_phase
from src.state_adapter import coerce_numeric_columns, ensure_plot_columns, metrics_to_dataframe
from src.ui_helpers import (
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
    section_header,
)
from src.ui_theme import COUNTRY_PALETTE, PHASE_COLORS, PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS

st.set_page_config(page_title="Comparaison pays", page_icon="🌍", layout="wide")
inject_global_css()
st.title("🗺️ Comparaison Pays")

PHASE_ORDER = ["stage_1", "stage_2", "stage_3", "stage_4", "unknown"]


def _canonical_phase(value) -> str:
    s = str(value).strip().lower()
    return s if s in set(PHASE_ORDER) else "unknown"


def _phase_order_value(phase: str) -> int:
    mapping = {"stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4}
    return mapping.get(_canonical_phase(phase), 0)


def _phase_from_value(value: int) -> str:
    rev = {1: "stage_1", 2: "stage_2", 3: "stage_3", 4: "stage_4"}
    return rev.get(int(value), "unknown")


def _smooth_phase_latest(df_hist: pd.DataFrame, country: str, year: int) -> str:
    sub = df_hist[(df_hist["country"] == country) & (df_hist["year"] <= year)].sort_values("year")
    if sub.empty:
        return "unknown"
    tail = sub.tail(3)
    vals = tail["phase"].astype(str).map(_phase_order_value)
    vals = vals[vals > 0]
    if vals.empty:
        return "unknown"
    med = int(round(float(vals.median())))
    return _phase_from_value(med)


def _recompute_phase_context(df_hist: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df_hist.copy()
    required = {"country", "year", "h_negative_obs"}
    if not required.issubset(set(out.columns)):
        return out

    out = out.drop(columns=["h_negative_declining"], errors="ignore")
    ctx = compute_h_negative_declining_flags(out[["country", "year", "h_negative_obs"]])
    out = out.merge(ctx, on=["country", "year"], how="left")
    out["h_negative_declining"] = out["h_negative_declining"].astype("boolean").fillna(False).astype(bool)

    phases: list[str] = []
    confidences: list[float] = []
    scores: list[float] = []
    blocked: list[str] = []
    for _, row in out.iterrows():
        diag = diagnose_phase(row.to_dict(), thresholds)
        phases.append(_canonical_phase(diag.get("phase", "unknown")))
        confidences.append(float(diag.get("confidence", np.nan)))
        scores.append(float(diag.get("score", np.nan)))
        blocked.append("; ".join(diag.get("blocked_rules", [])))

    out["phase"] = phases
    out["phase_confidence"] = confidences
    out["phase_score"] = scores
    out["phase_blocked_rules"] = blocked
    return out


state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Comparaison pays")
normalize_state_metrics(state)

df_all = metrics_to_dataframe(state, state.get("price_mode"))
if df_all.empty or "country" not in df_all.columns:
    guard_no_data("la page Comparaison pays")
df_all = _recompute_phase_context(df_all, state.get("thresholds", {}))

years = sorted(df_all["year"].dropna().unique())
year = st.selectbox("Année", years, index=len(years) - 1)
countries = st.multiselect(
    "Pays",
    sorted(df_all["country"].dropna().unique()),
    default=state.get("countries_selected", []),
)
show_smoothed_phase = st.toggle(
    "Afficher phase lissée 3 ans (lecture pédagogique)",
    value=False,
    help="La phase officielle reste annuelle; ce lissage sert uniquement à lire les tendances.",
)
if not countries:
    guard_no_data("la page Comparaison pays")

df = df_all[(df_all["country"].isin(countries)) & (df_all["year"] == year)].copy()
if df.empty:
    guard_no_data("la page Comparaison pays")

df = ensure_plot_columns(
    df,
    [
        "sr",
        "far",
        "ir",
        "ttl",
        "capture_ratio_pv",
        "h_negative_obs",
        "vre_penetration_pct_gen",
        "phase",
        "phase_confidence",
        "phase_score",
        "phase_blocked_rules",
    ],
    with_notice=True,
)
df = coerce_numeric_columns(
    df,
    [
        "sr",
        "far",
        "ir",
        "ttl",
        "capture_ratio_pv",
        "h_negative_obs",
        "vre_penetration_pct_gen",
        "phase_confidence",
        "phase_score",
    ],
)

missing_cols = df.attrs.get("_missing_plot_columns", [])
if missing_cols:
    st.info("Colonnes manquantes complétées en NaN pour robustesse d'affichage: " + ", ".join(missing_cols))

narrative(
    "Objectif: comparer la structure de stress des pays sur une même année. "
    "Le radar est orienté en mode stress: plus la valeur est élevée, plus le pays est sous pression structurelle."
)

section_header("Radar structurel (profil de stress)", "Axes normalisés de 0 à 1, orientation unique stress")

radar_raw = df.copy()
radar_raw["far_stress"] = 1.0 - radar_raw["far"]
radar_raw["capture_ratio_pv_stress"] = 1.0 - radar_raw["capture_ratio_pv"]

axes_map = {
    "SR": "sr",
    "FAR (stress)": "far_stress",
    "IR": "ir",
    "TTL": "ttl",
    "Capture PV (stress)": "capture_ratio_pv_stress",
    "Heures négatives": "h_negative_obs",
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
            opacity=0.5,
        )
    )

fig1.update_layout(
    height=520,
    title="Radar structurel de stress",
    polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10), gridcolor="#dbe5f1")),
    **PLOTLY_LAYOUT_DEFAULTS,
)
st.caption("Axes normalisés 0-1. Plus la surface est large, plus le stress est élevé.")
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Lecture radar orientée stress",
        purpose="Le radar permet de voir en un coup d'œil les pays où le stress est concentré sur surplus, rigidité ou cannibalisation.",
        observed={
            "n_pays": len(radar),
            "sr_mean": float(df["sr"].mean()),
            "far_mean": float(df["far"].mean()),
            "capture_ratio_pv_mean": float(df["capture_ratio_pv"].mean()),
        },
        method_link="Normalisation min-max intra-échantillon; inversion FAR/capture ratio pour une orientation stress uniforme.",
        limits="Comparaison relative au panier sélectionné; ajouter/retirer un pays change la normalisation.",
        n=len(radar),
        decision_use="Prioriser les pays à traiter en premier selon la nature dominante du stress.",
    )
)

section_header("Valeurs brutes derrière le radar", "Transparence des valeurs non normalisées")
raw_cols = [
    "country",
    "sr",
    "far",
    "ir",
    "ttl",
    "capture_ratio_pv",
    "h_negative_obs",
    "phase",
    "phase_confidence",
]
st.dataframe(df[raw_cols], use_container_width=True, hide_index=True)

section_header("Pénétration VRE vs capture ratio PV", "Taille bulle = heures négatives")
fig2 = px.scatter(
    df,
    x="vre_penetration_pct_gen",
    y="capture_ratio_pv",
    color="country",
    size="h_negative_obs",
    color_discrete_map=COUNTRY_PALETTE,
    hover_data=["phase", "phase_confidence", "sr", "far", "ir", "ttl"],
)
fig2.update_layout(
    height=480,
    title="Pénétration VRE vs capture ratio PV",
    xaxis_title="Pénétration VRE (% génération)",
    yaxis_title="Capture ratio PV",
    **PLOTLY_LAYOUT_DEFAULTS,
)
fig2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.caption("Taille des bulles = nombre d'heures à prix négatif.")
st.plotly_chart(fig2, use_container_width=True)

render_commentary(
    so_what_block(
        title="Positionnement comparatif",
        purpose="Comparer les pays à pénétration élevée selon leur capacité à maintenir un capture ratio défendable.",
        observed={
            "vre_min_pct": float(df["vre_penetration_pct_gen"].min()),
            "vre_max_pct": float(df["vre_penetration_pct_gen"].max()),
            "capture_ratio_min": float(df["capture_ratio_pv"].min()),
            "h_negative_total": float(df["h_negative_obs"].sum()),
        },
        method_link="Pénétration en % génération (v3); capture ratio sur price_used.",
        limits="Photographie annuelle: compléter avec trajectoires temporelles (pages Historique et Capture Rates).",
        n=len(df),
        decision_use="Identifier les pays où accélérer la flexibilité avant d'augmenter encore la pénétration VRE.",
    )
)

section_header("Comparaison phase vs TTL", "Phase annuelle (non monotone) et queue haute")
phase_df = df.copy()
phase_df["phase_officielle"] = phase_df["phase"].apply(_canonical_phase)
phase_df["phase_affichee"] = phase_df["phase_officielle"]
if show_smoothed_phase:
    phase_df["phase_affichee"] = phase_df["country"].apply(
        lambda c: _canonical_phase(_smooth_phase_latest(df_all, c, int(year)))
    )

fig3 = px.bar(
    phase_df.sort_values("ttl", ascending=False),
    x="country",
    y="ttl",
    color="phase_affichee",
    category_orders={"phase_affichee": PHASE_ORDER},
    color_discrete_map=PHASE_COLORS,
    hover_data=["phase_officielle", "phase_confidence", "phase_score", "phase_blocked_rules", "sr", "far", "ir", "capture_ratio_pv"],
)
fig3.update_layout(
    height=420,
    title="Phase de marché vs TTL",
    xaxis_title="Pays",
    yaxis_title="TTL (EUR/MWh)",
    **PLOTLY_LAYOUT_DEFAULTS,
)
fig3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.caption(
    "Classement par TTL décroissant. Couleur = phase annuelle. "
    "La phase peut changer d'une année à l'autre selon les indicateurs observés."
)
if show_smoothed_phase:
    st.caption("Mode lissé actif: la couleur affiche la médiane des 3 dernières phases annuelles (lecture pédagogique).")
st.plotly_chart(fig3, use_container_width=True)

render_commentary(
    so_what_block(
        title="Queue thermique et stade annuel",
        purpose="Un TTL élevé peut renforcer la valeur de flexibilité, mais signale aussi une queue de prix plus risquée.",
        observed={
            "ttl_median": float(df["ttl"].median()),
            "ttl_max": float(df["ttl"].max()),
            "phase_confidence_median": float(df["phase_confidence"].median()) if df["phase_confidence"].notna().any() else float("nan"),
        },
        method_link="TTL = P95(price_used) sur régimes C+D; phase = score annuel issu de thresholds.yaml.",
        limits="La phase est annuelle et non monotone; un pays peut passer de stage_3 à stage_2 selon FAR/h_negative et autres règles.",
        n=len(df),
        decision_use="Lire la phase comme un état observé annuel, pas comme un chemin irréversible.",
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
                label="Télécharger le fichier",
                data=f.read(),
                file_name=f"comparaison_{year}_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with col2:
    if st.button("Exporter Google Sheets"):
        url = export_to_gsheets(metrics_rows, diag_rows, [], f"Comparaison_{year}")
        if url:
            st.success(f"Export créé: {url}")
        else:
            st.warning("Credentials absentes ou export indisponible.")
