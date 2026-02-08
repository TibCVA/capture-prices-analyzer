"""Page 2 - NRL Deep Dive."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import so_what_block
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
from src.state_adapter import coerce_numeric_columns, ensure_plot_columns, metrics_to_dataframe
from src.ui_analysis import compute_nrl_price_link_stats
from src.ui_theme import PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS, REGIME_COLORS
from src.ui_helpers import (
    challenge_block,
    dynamic_narrative,
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
    render_kpi_banner,
    section_header,
)

st.set_page_config(page_title="NRL Deep Dive", page_icon="🔬", layout="wide")
inject_global_css()
st.title("🔬 NRL Deep Dive")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page NRL Deep Dive")
normalize_state_metrics(state)

proc = state.get("processed", {})
metrics_df = metrics_to_dataframe(state, state.get("price_mode"))
if not proc or metrics_df.empty or "country" not in metrics_df.columns:
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
    [
        COL_LOAD,
        COL_VRE,
        COL_NRL,
        COL_SURPLUS,
        COL_FLEX_EFFECTIVE,
        COL_SURPLUS_UNABS,
        COL_REGIME,
        COL_PRICE_DA,
    ],
    with_notice=True,
)
df = coerce_numeric_columns(
    df,
    [COL_LOAD, COL_VRE, COL_NRL, COL_SURPLUS, COL_FLEX_EFFECTIVE, COL_SURPLUS_UNABS, COL_PRICE_DA],
)
missing_cols = df.attrs.get("_missing_plot_columns", [])
if missing_cols:
    dynamic_narrative(
        "Certaines colonnes requises pour les graphiques etaient absentes et ont ete ajoutees en NaN: "
        + ", ".join(missing_cols),
        severity="warning",
    )

narrative(
    "Objectif de cette page: ouvrir la logique physique heure par heure (load, VRE, NRL, surplus, flex), "
    "puis verifier en externe si cette lecture physique explique effectivement les niveaux de prix observes."
)

section_header("Traces horaires", "Load, VRE, NRL, surplus et flexibilite")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_LOAD], name="Load", line=dict(color="#1B2A4A", width=1.9)))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_VRE], name="VRE", line=dict(color="#16a34a", width=1.8)))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_NRL], name="NRL", line=dict(color="#dc2626", width=2.0)))
fig1.add_trace(go.Scatter(x=df.index, y=df[COL_SURPLUS], name="Surplus", line=dict(color="#f59e0b", width=1.3)))
fig1.add_trace(
    go.Scatter(x=df.index, y=df[COL_FLEX_EFFECTIVE], name="Flex effective", line=dict(color="#2563eb", width=1.3))
)
fig1.update_layout(height=440, xaxis_title="Heure", yaxis_title="MW", **PLOTLY_LAYOUT_DEFAULTS)
fig1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Lecture temporelle du systeme",
        purpose="Quand NRL devient negatif, le systeme entre en surplus brut et la question cle devient la capacite d'absorption.",
        observed={
            "nrl_min_mw": float(df[COL_NRL].min()),
            "nrl_max_mw": float(df[COL_NRL].max()),
            "surplus_total_mwh": float(df[COL_SURPLUS].sum()),
            "flex_total_mwh": float(df[COL_FLEX_EFFECTIVE].sum()),
        },
        method_link="Pipeline physique: NRL -> surplus -> flex_effective -> surplus_unabsorbed (G.6).",
        limits="Serie annuelle dense: completer avec zoom sur episodes critiques pour un diagnostic operationnel.",
        n=len(df),
        decision_use="Qualifier si le probleme dominant est la generation en surplus ou le manque de flexibilite disponible.",
    )
)

section_header("Distribution NRL et surplus non absorbe", "Histogrammes")
c1, c2 = st.columns(2)
with c1:
    fig2 = px.histogram(df, x=COL_NRL, nbins=80, color=COL_REGIME, color_discrete_map=REGIME_COLORS)
    fig2.update_layout(height=360, xaxis_title="NRL (MW)", yaxis_title="Heures", **PLOTLY_LAYOUT_DEFAULTS)
    fig2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    fig3 = px.histogram(df, x=COL_SURPLUS_UNABS, nbins=60)
    fig3.update_layout(height=360, xaxis_title="Surplus non absorbe (MW)", yaxis_title="Heures", **PLOTLY_LAYOUT_DEFAULTS)
    fig3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig3, use_container_width=True)

render_commentary(
    so_what_block(
        title="Structure des heures contraintes",
        purpose="Le poids des heures A mesure un stress structurel; un poids B eleve indique une absorption efficace du surplus.",
        observed={
            "h_surplus": int((df[COL_SURPLUS] > 0).sum()),
            "h_surplus_unabs": int((df[COL_SURPLUS_UNABS] > 0).sum()),
            "h_regime_a": int((df[COL_REGIME] == "A").sum()),
            "h_regime_b": int((df[COL_REGIME] == "B").sum()),
        },
        method_link="Regime A <=> surplus_unabsorbed > 0; regime B <=> surplus absorbe integralement.",
        limits="Lecture sensible aux hypotheses must-run/flex actives dans la session.",
        n=len(df),
        decision_use="Prioriser les investissements qui reduisent d'abord les heures A avant de viser des gains marginaux sur B/C.",
    )
)

section_header("NRL vs prix observe", "Validation externe de la segmentation physique")
scatter_df = df[[COL_NRL, COL_PRICE_DA, COL_REGIME]].dropna().copy()
row = metrics_df[(metrics_df["country"] == country) & (metrics_df["year"] == year)]
stats = compute_nrl_price_link_stats(scatter_df if not scatter_df.empty else df, row.iloc[0] if not row.empty else None)

k1, k2 = st.columns(2)
with k1:
    corr_txt = f"{stats['pearson_r_pct']:+.1f}%" if np.isfinite(stats["pearson_r_pct"]) else "N/A"
    render_kpi_banner(
        "Correlation NRL / prix observe",
        corr_txt,
        f"n={stats['n_valid']} points valides",
        status=stats["corr_status"],
    )
with k2:
    coh_txt = f"{stats['regime_coherence_pct']:.1f}%" if np.isfinite(stats["regime_coherence_pct"]) else "N/A"
    render_kpi_banner(
        "Coherence regime / prix observe",
        coh_txt,
        "Seuil de reference: >55%",
        status=stats["coherence_status"],
    )

if scatter_df.empty:
    st.info("Pas assez de points valides pour tracer NRL vs prix observe.")
else:
    fig4 = px.scatter(
        scatter_df,
        x=COL_NRL,
        y=COL_PRICE_DA,
        color=COL_REGIME,
        color_discrete_map=REGIME_COLORS,
        opacity=0.38,
    )
    fig4.update_layout(
        height=450,
        xaxis_title="NRL (MW)",
        yaxis_title="Prix observe (EUR/MWh)",
        legend_title="Regime",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig4.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig4.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    fig4.add_annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            f"Correlation: {stats['pearson_r_pct']:+.1f}%"
            if np.isfinite(stats["pearson_r_pct"])
            else "Correlation: N/A"
        )
        + "<br>"
        + (
            f"Coherence: {stats['regime_coherence_pct']:.1f}%"
            if np.isfinite(stats["regime_coherence_pct"])
            else "Coherence: N/A"
        ),
        font=dict(size=12, color="#1f2937"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    st.plotly_chart(fig4, use_container_width=True)

render_commentary(
    so_what_block(
        title="Validation NRL vs prix observe",
        purpose="La correlation capture le lien monotone global; la coherence regime/prix teste l'alignement avec les seuils de regimes.",
        observed={
            "corr_pct": stats["pearson_r_pct"],
            "coherence_pct": stats["regime_coherence_pct"],
            "n_valid": stats["n_valid"],
        },
        method_link="Correlation de Pearson sur (NRL, prix observe) + score de coherence selon thresholds.coherence_params.",
        limits="Une bonne correlation ne prouve pas la causalite; une coherence faible peut venir d'hypotheses incompltes ou de donnees degradees.",
        n=stats["n_valid"],
        decision_use="Decider si le cadre physique est assez robuste pour servir de base aux scenarios et aux messages business.",
    ),
    variant="analysis",
)

coh = stats["regime_coherence"]
if np.isfinite(coh) and coh < 0.55:
    challenge_block(
        "Coherence faible",
        f"Score {coh:.1%} < 55%. Recalibrer hypotheses must-run/flex et verifier qualite donnees avant conclusion forte.",
    )
elif np.isfinite(coh):
    dynamic_narrative(
        "Coherence regime/prix au niveau attendu. La lecture physique est exploitable pour interpretation et scenarios.",
        severity="success",
    )
