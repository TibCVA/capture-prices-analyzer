"""Page 0 - Comprendre le modele (version didactique complete)."""

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
from src.state_adapter import ensure_plot_columns, metrics_to_dataframe
from src.ui_helpers import (
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
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
if not proc or metrics_df.empty:
    guard_no_data("la page Comprendre le modele")

narrative(
    "Cette page explique le cadre analytique complet: pourquoi la NRL est la variable centrale, "
    "comment les regimes A/B/C/D sont classes sans circularite, et comment lire les ratios pivots "
    "SR/FAR/IR/TTL pour prendre des decisions business."
)

with st.expander("Pourquoi cette analyse ?", expanded=True):
    st.markdown(
        """
Le probleme business est simple: quand la penetration VRE augmente, les producteurs VRE produisent souvent ensemble, ce qui comprime les prix aux memes heures.

Objectif de l'outil:
1. expliquer les mecanismes physiques derriere les capture prices,
2. quantifier des seuils auditables (pas de boite noire),
3. tester des scenarios deterministes (BESS, demande, gaz/CO2).

Le cadre retenu est volontairement structurel: on explique les ordres de grandeur et les bascules, pas une prevision spot fine.
        """
    )

with st.expander("Merit order et capture ratio"):
    st.markdown(
        """
Le prix spot est fixe par la derniere technologie necessaire a l'equilibre. Quand la production solaire/eolienne monte,
la courbe d'offre se deplace vers les technologies moins couteuses, et les prix baissent surtout aux heures VRE.

Definitions:
- Capture price PV = somme(prix * production PV) / somme(production PV)
- Capture ratio PV = capture price PV / prix baseload

Lecture:
- capture ratio proche de 1.0: peu de cannibalisation
- capture ratio < 0.8: cannibalisation significative
- capture ratio < 0.7: degradation severe
        """
    )

with st.expander("Pourquoi la NRL ?", expanded=True):
    st.latex(r"NRL = Load - VRE - Must\text{-}Run")
    st.markdown(
        """
La NRL traduit directement la pression physique sur le systeme:
- NRL < 0: surplus brut, risque de prix tres bas/negatifs
- NRL > 0: besoin thermique, prix ancres sur le cout marginal (TCA)

Le must-run est indispensable dans la formule: ignorer cette rigidite surestime la flexibilite et sous-estime les surplus.
        """
    )

with st.expander("Regimes A/B/C/D et anti-circularite", expanded=True):
    st.markdown(
        """
Classification strictement physique (pas de prix dans la regle):
- A: surplus non absorbe
- B: surplus absorbe
- C: NRL positif hors queue haute
- D: queue haute des NRL positifs

Ensuite seulement, on verifie la coherence regime/prix observe. Cette separation evite la circularite.
        """
    )

with st.expander("Stades et regles de bascule"):
    st.markdown(
        """
Le diagnostic de phase se base sur un scoring (thresholds.yaml):
- stage_1: faible stress (peu d'heures negatives, capture ratio eleve)
- stage_2: premiers surplus marquants
- stage_3: absorption structurelle (FAR eleve)
- stage_4: saturation avancee

Ce n'est pas un test causal; c'est un classement interpretable et reproductible.
        """
    )

with st.expander("Ratios pivots SR / FAR / IR / TTL", expanded=True):
    st.markdown(
        """
- SR (Surplus Ratio): volume de surplus brut rapporte a la generation annuelle
- FAR (Flex Absorption Ratio): part du surplus absorbee
- IR (Inflexibility Ratio): rigidite systeme au creux (P10 must-run / P10 load)
- TTL (Thermal Tail Level): queue haute des prix en regimes C+D

Ces 4 ratios sont la colonne vertebrale de l'interpretation business.
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
)

st.markdown("### Exemple temporel 48h")
df48 = df.head(48).copy()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_LOAD], name="Load", line=dict(color="#1B2A4A", width=2)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_VRE], name="VRE", line=dict(color="#27AE60", width=2)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_MUST_RUN], name="Must-run", line=dict(color="#7F8C8D", width=2)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_NRL], name="NRL", line=dict(color="#E74C3C", width=2, dash="dash")))
fig1.add_hline(y=0, line_dash="dot", line_color="#555")
fig1.update_layout(height=420, xaxis_title="Heure", yaxis_title="MW")
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Lecture 48h",
        purpose="Verifier visuellement quand le systeme bascule en surplus et donc en risque de pression prix",
        observed={
            "nrl_min_mw": float(df48[COL_NRL].min()),
            "nrl_max_mw": float(df48[COL_NRL].max()),
            "h_nrl_neg": int((df48[COL_NRL] < 0).sum()),
        },
        method_link="Surplus brut = max(0, -NRL), puis absorption par flex_effective.",
        limits="Fenetre illustrative: la robustesse de diagnostic se lit sur l'annee complete.",
        n=len(df48),
    )
)

st.markdown("### Distribution annuelle")
col1, col2 = st.columns(2)
with col1:
    fig2 = px.histogram(df, x=COL_NRL, nbins=100, color=COL_REGIME)
    fig2.update_layout(height=360, xaxis_title="NRL (MW)", yaxis_title="Heures")
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = px.histogram(df, x=COL_SURPLUS_UNABS, nbins=80)
    fig3.update_layout(height=360, xaxis_title="Surplus non absorbe (MW)", yaxis_title="Heures")
    st.plotly_chart(fig3, use_container_width=True)

render_commentary(
    so_what_block(
        title="Structure des regimes",
        purpose="Mesurer le poids relatif des heures de surplus non absorbe (regime A), cle pour les risques de cannibalisation",
        observed={
            "h_A": int((df[COL_REGIME] == "A").sum()),
            "h_B": int((df[COL_REGIME] == "B").sum()),
            "h_C": int((df[COL_REGIME] == "C").sum()),
            "h_D": int((df[COL_REGIME] == "D").sum()),
        },
        method_link="Regimes classes par variables physiques uniquement (anti-circularite).",
        limits="Le seuil C/D depend du parametrage regime_d dans thresholds.yaml.",
        n=len(df),
    )
)

st.markdown("### Coherence regime/prix observe")
scatter_df = df[[COL_NRL, COL_PRICE_DA, COL_REGIME]].dropna().copy()
if not scatter_df.empty:
    fig4 = px.scatter(scatter_df, x=COL_NRL, y=COL_PRICE_DA, color=COL_REGIME, opacity=0.35)
    fig4.update_layout(height=420, xaxis_title="NRL (MW)", yaxis_title="Prix observe (EUR/MWh)")
    st.plotly_chart(fig4, use_container_width=True)

row = metrics_df[(metrics_df["country"] == country) & (metrics_df["year"] == year)]
coh = float("nan")
if not row.empty:
    coh = float(row.iloc[0].get("regime_coherence", np.nan))

render_commentary(
    so_what_block(
        title="Validation externe",
        purpose="Evaluer si la lecture physique des regimes explique correctement les niveaux de prix observes",
        observed={"coherence": coh},
        method_link="Score calcule uniquement en mode observed selon coherence_params.",
        limits="Un score faible peut venir de donnees incompletes ou d'hypotheses non calibrees, pas uniquement d'un bug.",
        n=len(scatter_df),
    )
)

st.markdown("### Ce que le modele ne capture pas")
st.markdown(
    """
- Pas de prevision spot fine; le prix synth est un proxy structurel.
- Pas de dispatch unitaire complet ni de congestion intra-zone.
- Les scenarios restent deterministes (pas d'incertitude probabiliste).
- Le diagnostic phase est heuristique, pas econometrique causal.
    """
)
