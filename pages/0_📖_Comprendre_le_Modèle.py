"""Page 0 - Comprendre le modele (didactique approfondi)."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import so_what_block
from src.constants import (
    COL_LOAD,
    COL_MUST_RUN,
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
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
    render_kpi_banner,
    section_header,
)

st.set_page_config(page_title="Comprendre le modele", page_icon="📖", layout="wide")
inject_global_css()
st.title("📖 Comprendre le Modele")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Comprendre le modele")
normalize_state_metrics(state)

proc = state.get("processed", {})
metrics_df = metrics_to_dataframe(state, state.get("price_mode"))
if not proc or metrics_df.empty or "country" not in metrics_df.columns:
    guard_no_data("la page Comprendre le modele")

narrative(
    "Cette page donne la logique complete, de la donnee ENTSO-E brute jusqu'aux conclusions business. "
    "Le but est de comprendre ce que disent les chiffres, ce qu'ils ne disent pas, et pourquoi l'outil reste auditables."
)

with st.expander("Comment lire cette page en 3 minutes", expanded=True):
    st.markdown(
        """
1. **Commencer par la formule NRL**: `NRL = load - VRE - must-run`.
2. **Regarder les regimes**: A (surplus non absorbe), B (surplus absorbe), C (thermique), D (queue haute).
3. **Verifier la validation externe**: correlation NRL/prix observe + coherence regime/prix.
4. **Conclure avec les 4 ratios pivots**: SR, FAR, IR, TTL.
"""
    )

with st.expander("1) Donnees et conventions ENTSO-E", expanded=True):
    st.markdown(
        """
- Source horaire: load, generation, prix DA, net position.
- Convention critique: **interdit** d'approximer les echanges via `generation - load`.
- Le load ENTSO-E inclut l'energie absorbee (stockage/pompage): le pompage est traite explicitement comme flex.
- Les observables marche (heures negatives, spreads) se calculent sur prix **observe** uniquement.
"""
    )

with st.expander("2) Formule physique centrale: NRL", expanded=True):
    st.latex(r"NRL = Load - VRE - Must\text{-}Run")
    st.markdown(
        """
- `NRL < 0`: surplus brut, exposition aux regimes de prix bas.
- `NRL > 0`: besoin thermique, prix ancres par le cout marginal (TCA).
- Le must-run est essentiel: sans lui, on surestime artificiellement la flexibilite.
"""
    )

with st.expander("3) Regimes A/B/C/D sans circularite", expanded=True):
    st.markdown(
        """
Classification strictement physique:
- **A**: surplus non absorbe (`surplus_unabsorbed > 0`)
- **B**: surplus absorbe (`surplus > 0` et `surplus_unabsorbed = 0`)
- **C**: NRL positif hors queue haute
- **D**: queue haute des NRL positifs

Les prix ne servent pas a definir les regimes; ils servent ensuite a valider la coherence externe.
"""
    )

with st.expander("4) Ratios pivots et lecture business", expanded=True):
    st.markdown(
        """
- **SR**: part du surplus brut dans la generation annuelle.
- **FAR**: part du surplus absorbee par les flexibilites.
- **IR**: rigidite de creux (must-run vs load bas quantiles).
- **TTL**: niveau de queue haute de prix en regimes C+D.

Lecture courte:
- SR monte + FAR bas -> risque de saturation.
- FAR monte a SR constant -> absorption structurelle progresse.
- TTL monte avec TCA (gaz/CO2) -> ancre thermique plus chere.
"""
    )

pairs = sorted({(k[0], k[1]) for k in proc.keys()})
country = st.selectbox("Pays exemple", sorted({p[0] for p in pairs}), key="model_country_v3")
year = st.selectbox("Annee exemple", sorted({p[1] for p in pairs if p[0] == country}), key="model_year_v3")

proc_key = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
if proc_key not in proc:
    fallback = [k for k in proc.keys() if k[0] == country and k[1] == year]
    if not fallback:
        guard_no_data("la page Comprendre le modele")
    proc_key = sorted(fallback)[0]

df = ensure_plot_columns(
    proc[proc_key],
    [COL_LOAD, COL_VRE, COL_MUST_RUN, COL_NRL, COL_SURPLUS, COL_SURPLUS_UNABS, COL_REGIME, COL_PRICE_DA],
    with_notice=True,
)
df = coerce_numeric_columns(
    df,
    [COL_LOAD, COL_VRE, COL_MUST_RUN, COL_NRL, COL_SURPLUS, COL_SURPLUS_UNABS, COL_PRICE_DA],
)
missing_cols = df.attrs.get("_missing_plot_columns", [])
if missing_cols:
    st.info("Colonnes ajoutees pour robustesse de rendu: " + ", ".join(missing_cols))

st.markdown("### Decomposition physique 48h")
df48 = df.head(48).copy()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_LOAD], name="Load", line=dict(color="#1B2A4A", width=2.1)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_VRE], name="VRE", line=dict(color="#16a34a", width=1.9)))
fig1.add_trace(
    go.Scatter(x=df48.index, y=df48[COL_MUST_RUN], name="Must-run", line=dict(color="#64748b", width=1.8))
)
fig1.add_trace(
    go.Scatter(x=df48.index, y=df48[COL_NRL], name="NRL", line=dict(color="#dc2626", width=2.1, dash="dash"))
)
fig1.add_hline(y=0, line_dash="dot", line_color="#475569")
fig1.update_layout(height=430, xaxis_title="Heure", yaxis_title="MW", **PLOTLY_LAYOUT_DEFAULTS)
fig1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Lecture 48h: du signal physique a l'interpretation",
        purpose="Les heures ou NRL passe sous zero sont les heures de risque principal pour les prix bas.",
        observed={
            "nrl_min_mw": float(df48[COL_NRL].min()),
            "nrl_max_mw": float(df48[COL_NRL].max()),
            "h_nrl_neg": int((df48[COL_NRL] < 0).sum()),
            "surplus_mwh_48h": float(df48[COL_SURPLUS].sum()),
        },
        method_link="Surplus brut = max(0, -NRL), puis absorption via flex_effective dans le pipeline G.6.",
        limits="Fenetre illustrative: confirmer les ordres de grandeur sur l'annee complete.",
        n=len(df48),
        decision_use="Discuter rapidement si le marche est dans un regime de stress episodique ou recurrent.",
    )
)

st.markdown("### Distribution annuelle des regimes")
col1, col2 = st.columns(2)
with col1:
    fig2 = px.histogram(df, x=COL_NRL, nbins=90, color=COL_REGIME, color_discrete_map=REGIME_COLORS)
    fig2.update_layout(height=365, xaxis_title="NRL (MW)", yaxis_title="Heures", **PLOTLY_LAYOUT_DEFAULTS)
    fig2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = px.histogram(df, x=COL_SURPLUS_UNABS, nbins=80)
    fig3.update_layout(
        height=365,
        xaxis_title="Surplus non absorbe (MW)",
        yaxis_title="Heures",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig3, use_container_width=True)

render_commentary(
    so_what_block(
        title="Pourquoi les regimes importent",
        purpose="Le volume d'heures A indique la part du stress qui n'est pas absorbee par la flexibilite disponible.",
        observed={
            "h_A": int((df[COL_REGIME] == "A").sum()),
            "h_B": int((df[COL_REGIME] == "B").sum()),
            "h_C": int((df[COL_REGIME] == "C").sum()),
            "h_D": int((df[COL_REGIME] == "D").sum()),
        },
        method_link="Regimes classes sur variables physiques uniquement (anti-circularite stricte).",
        limits="Le seuil C/D depend de `thresholds.model_params.regime_d`.",
        n=len(df),
        decision_use="Prioriser les leviers qui reduisent d'abord les heures A avant d'optimiser les regimes C/D.",
    )
)

st.markdown("### Validation externe: NRL vs prix observe")
scatter_df = df[[COL_NRL, COL_PRICE_DA, COL_REGIME]].dropna().copy()
row = metrics_df[(metrics_df["country"] == country) & (metrics_df["year"] == year)]
stats = compute_nrl_price_link_stats(scatter_df if not scatter_df.empty else df, row.iloc[0] if not row.empty else None)

b1, b2 = st.columns(2)
with b1:
    render_kpi_banner(
        "Correlation NRL / prix observe",
        f"{stats['pearson_r_pct']:+.1f}%" if np.isfinite(stats["pearson_r_pct"]) else "N/A",
        f"n={stats['n_valid']} points valides",
        status=stats["corr_status"],
    )
with b2:
    render_kpi_banner(
        "Coherence regime / prix observe",
        f"{stats['regime_coherence_pct']:.1f}%" if np.isfinite(stats["regime_coherence_pct"]) else "N/A",
        "Reference: >55%",
        status=stats["coherence_status"],
    )

if not scatter_df.empty:
    fig4 = px.scatter(
        scatter_df,
        x=COL_NRL,
        y=COL_PRICE_DA,
        color=COL_REGIME,
        color_discrete_map=REGIME_COLORS,
        opacity=0.35,
    )
    fig4.update_layout(
        height=430,
        xaxis_title="NRL (MW)",
        yaxis_title="Prix observe (EUR/MWh)",
        legend_title="Regime",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig4.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig4.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Pas assez de points valides pour la validation NRL/prix observe.")

render_commentary(
    so_what_block(
        title="Validation du cadre methodologique",
        purpose="Une correlation elevee seule ne suffit pas: il faut aussi une coherence regime/prix robuste.",
        observed={
            "corr_pct": stats["pearson_r_pct"],
            "coherence_pct": stats["regime_coherence_pct"],
            "n_valid": stats["n_valid"],
        },
        method_link="Correlation de Pearson + coherence selon thresholds.coherence_params (mode observed).",
        limits="Resultat sensible a la completude donnees, aux hypotheses must-run/flex et aux ruptures de marche.",
        n=stats["n_valid"],
        decision_use="Valider que les analyses historiques et scenarios s'appuient sur une base physique defendable.",
    )
)

st.markdown("### Limites explicites du modele")
st.markdown(
    """
- Le prix scenario est un **prix synthetique structurel**, pas une prevision spot transactionnelle.
- Le dispatch BESS est deterministe (pas d'optimisation economique complete).
- Le diagnostic de phase est un score interpretable, pas un modele causal econometrique.
- Les conclusions doivent toujours etre relues avec la qualite de donnees (`data_completeness`) et la coherence regime/prix.
"""
)
