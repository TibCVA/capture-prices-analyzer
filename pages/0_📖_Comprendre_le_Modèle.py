"""Page 0 - Comprendre le modele (parcours didactique complet)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import so_what_block
from src.constants import (
    COL_FLEX_EFFECTIVE,
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
st.title("📖 Comprendre le modele")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Comprendre le modele")
normalize_state_metrics(state)

proc = state.get("processed", {})
metrics_df = metrics_to_dataframe(state, state.get("price_mode"))
if not proc or metrics_df.empty or "country" not in metrics_df.columns:
    guard_no_data("la page Comprendre le modele")

narrative(
    "Fil rouge de cette page: partir des donnees ENTSO-E, construire NRL heure par heure, "
    "classifier les regimes A/B/C/D, puis verifier si les prix observes restent coherents avec ce cadre physique."
)

mode = st.radio(
    "Mode de lecture",
    ["Parcours complet", "Parcours rapide (2 min)"],
    horizontal=True,
    index=0,
)
full_mode = mode == "Parcours complet"

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
    [
        COL_LOAD,
        COL_VRE,
        COL_MUST_RUN,
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
    [COL_LOAD, COL_VRE, COL_MUST_RUN, COL_NRL, COL_SURPLUS, COL_FLEX_EFFECTIVE, COL_SURPLUS_UNABS, COL_PRICE_DA],
)
if df.attrs.get("_missing_plot_columns", []):
    st.info("Colonnes completees en NaN pour robustesse: " + ", ".join(df.attrs.get("_missing_plot_columns", [])))

row = metrics_df[(metrics_df["country"] == country) & (metrics_df["year"] == year)]
metrics_row = row.iloc[0].to_dict() if not row.empty else {}

if full_mode:
    with st.expander("Etape 1 - Problematique business", expanded=True):
        st.markdown(
            """
- Objectif: expliquer la dynamique des capture prices sans boite noire.
- Question centrale: a quel moment la penetration VRE cree un surplus que la flexibilite ne peut plus absorber.
- Resultat attendu: des regles auditables (SR/FAR/IR/TTL) pour discuter seuils, pentes et leviers.
"""
        )
        st.caption("Base de calcul: metriques annuelles par couple pays/annee, mode historique actif dans la session.")

with st.expander("Etape 2 - Donnees et conventions ENTSO-E", expanded=full_mode):
    st.markdown(
        """
- Sources: load, generation par filiere, prix day-ahead, net position.
- Convention critique: `generation - load` ne doit jamais servir a approximer les echanges.
- Le load ENTSO-E integre l'energie absorbee: le pompage est traite comme un usage de flexibilite, pas comme generation.
- Les observables marche (heures negatives, spreads) restent calcules sur prix observe.
"""
    )
    st.caption(
        "Base de calcul: colonnes harmonisees v3 (`load_mw`, `price_da_eur_mwh`, `net_position_mw`, filieres generation)."
    )

with st.expander("Etape 3 - Construction de NRL (exemple horaire reel)", expanded=True):
    st.latex(r"NRL = Load - VRE - Must\text{-}Run")
    valid = df[[COL_LOAD, COL_VRE, COL_MUST_RUN, COL_NRL, COL_SURPLUS]].dropna()
    if valid.empty:
        st.info("Impossible d'afficher un exemple horaire: donnees insuffisantes.")
    else:
        ts = valid[COL_NRL].idxmin()
        line = valid.loc[ts]
        ex = pd.DataFrame(
            [
                {"variable": "load_mw", "valeur": float(line[COL_LOAD])},
                {"variable": "vre_mw", "valeur": float(line[COL_VRE])},
                {"variable": "must_run_mw", "valeur": float(line[COL_MUST_RUN])},
                {"variable": "nrl_mw", "valeur": float(line[COL_NRL])},
                {"variable": "surplus_mw", "valeur": float(line[COL_SURPLUS])},
            ]
        )
        st.caption(f"Heure exemple: {ts}")
        st.dataframe(ex, use_container_width=True, hide_index=True)
        render_commentary(
            so_what_block(
                title="Lecture de l'exemple horaire",
                purpose="Un NRL negatif ne signifie pas un prix negatif automatique, mais un risque de surplus a absorber.",
                observed={
                    "nrl_mw": float(line[COL_NRL]),
                    "surplus_mw": float(line[COL_SURPLUS]),
                },
                method_link="Surplus = max(0, -NRL), avant absorption par flex_effective.",
                limits="Exemple ponctuel: confirmer la dynamique a l'echelle annuelle.",
                n=1,
                decision_use="Verifier si le probleme est ponctuel ou structurel avant de dimensionner des leviers.",
            )
        )

with st.expander("Etape 4 - De NRL aux regimes A/B/C/D", expanded=True):
    table_regimes = pd.DataFrame(
        [
            {"Regime": "A", "Condition": "surplus_unabsorbed > 0", "Interpretation": "surplus non absorbe"},
            {"Regime": "B", "Condition": "surplus > 0 et surplus_unabsorbed = 0", "Interpretation": "surplus absorbe"},
            {"Regime": "C", "Condition": "NRL >= 0 hors queue haute", "Interpretation": "thermique marginal"},
            {"Regime": "D", "Condition": "queue haute des NRL positifs", "Interpretation": "heures de pointe"},
        ]
    )
    st.dataframe(table_regimes, use_container_width=True, hide_index=True)

    regime_counts = (
        df[COL_REGIME]
        .value_counts(dropna=False)
        .rename_axis("regime")
        .reset_index(name="heures")
        .sort_values("regime")
    )
    st.dataframe(regime_counts, use_container_width=True, hide_index=True)
    st.caption("Base de calcul: classification strictement physique, sans utiliser le prix (anti-circularite).")

section_header("Visualisation 48h", "Load, VRE, Must-run, NRL, surplus et flex")
df48 = df.head(48).copy()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_LOAD], name="Load", line=dict(color="#1b2a4a", width=2.0)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_VRE], name="VRE", line=dict(color="#16a34a", width=1.9)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_MUST_RUN], name="Must-run", line=dict(color="#64748b", width=1.7)))
fig1.add_trace(go.Scatter(x=df48.index, y=df48[COL_NRL], name="NRL", line=dict(color="#dc2626", width=2.1, dash="dash")))
fig1.add_trace(
    go.Scatter(x=df48.index, y=df48[COL_FLEX_EFFECTIVE], name="Flex effective", line=dict(color="#2563eb", width=1.5))
)
fig1.add_hline(y=0, line_dash="dot", line_color="#475569")
fig1.update_layout(height=430, xaxis_title="Heure", yaxis_title="MW", **PLOTLY_LAYOUT_DEFAULTS)
fig1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.plotly_chart(fig1, use_container_width=True)

render_commentary(
    so_what_block(
        title="Lecture temporelle 48h",
        purpose="Identifier quand la flexibilite suffit et quand un surplus non absorbe apparait.",
        observed={
            "nrl_min_mw": float(df48[COL_NRL].min()),
            "h_nrl_neg": int((df48[COL_NRL] < 0).sum()),
            "surplus_48h_mwh": float(df48[COL_SURPLUS].sum()),
            "surplus_unabs_48h_mwh": float(df48[COL_SURPLUS_UNABS].sum()),
        },
        method_link="Pipeline physique G.6: NRL -> surplus -> flex_effective -> surplus_unabsorbed.",
        limits="Fenetre courte; la conclusion strategique doit etre confirmee sur l'annee.",
        n=len(df48),
        decision_use="Verifier rapidement si l'enjeu principal est un manque de flex ou un excs de generation residuelle.",
    )
)

section_header("Etape 5 - Validation externe", "Lien NRL/prix observe et coherence regime/prix")
scatter_df = df[[COL_NRL, COL_PRICE_DA, COL_REGIME]].dropna().copy()
stats = compute_nrl_price_link_stats(scatter_df if not scatter_df.empty else df, metrics_row if metrics_row else None)

c1, c2 = st.columns(2)
with c1:
    render_kpi_banner(
        "Correlation NRL / prix observe",
        f"{stats['pearson_r_pct']:+.1f}%" if np.isfinite(stats["pearson_r_pct"]) else "N/A",
        f"n={stats['n_valid']} points valides",
        status=stats["corr_status"],
    )
with c2:
    render_kpi_banner(
        "Coherence regime / prix observe",
        f"{stats['regime_coherence_pct']:.1f}%" if np.isfinite(stats["regime_coherence_pct"]) else "N/A",
        "Reference: >55%",
        status=stats["coherence_status"],
    )

if scatter_df.empty:
    st.info("Pas assez de points valides pour la validation NRL/prix observe.")
else:
    fig2 = px.scatter(
        scatter_df,
        x=COL_NRL,
        y=COL_PRICE_DA,
        color=COL_REGIME,
        color_discrete_map=REGIME_COLORS,
        opacity=0.35,
    )
    fig2.update_layout(
        height=430,
        xaxis_title="NRL (MW)",
        yaxis_title="Prix observe (EUR/MWh)",
        legend_title="Regime",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig2, use_container_width=True)

render_commentary(
    so_what_block(
        title="Difference entre correlation et coherence",
        purpose="La correlation mesure un lien global; la coherence teste la compatibilite avec la logique des regimes.",
        observed={
            "corr_pct": stats["pearson_r_pct"],
            "coherence_pct": stats["regime_coherence_pct"],
            "n_valid": stats["n_valid"],
        },
        method_link="Pearson sur (NRL, prix observe) + score regime_coherent base sur thresholds.coherence_params.",
        limits="Correlation elevee != preuve causale. Coherence faible peut aussi signaler des hypotheses incompletes.",
        n=stats["n_valid"],
        decision_use="Valider le cadre avant d'interpreter les scenarios ou de communiquer des seuils business.",
    )
)

if full_mode:
    with st.expander("Etape 6 - Ratios pivots SR/FAR/IR/TTL", expanded=True):
        pivots = pd.DataFrame(
            [
                {"ratio": "SR", "valeur": metrics_row.get("sr"), "lecture": "part du surplus brut dans la generation"},
                {"ratio": "FAR", "valeur": metrics_row.get("far"), "lecture": "part du surplus absorbee par la flex"},
                {"ratio": "IR", "valeur": metrics_row.get("ir"), "lecture": "rigidite en creux (P10 must-run / P10 load)"},
                {"ratio": "TTL", "valeur": metrics_row.get("ttl"), "lecture": "queue haute de prix sur regimes C+D"},
            ]
        )
        st.dataframe(pivots, use_container_width=True, hide_index=True)
        st.caption("Base de calcul: definitions G.7, penetration VRE en % de generation (pas en % de demande).")

    with st.expander("Etape 7 - Diagnostic de phase", expanded=True):
        phase = metrics_row.get("phase", "unknown")
        conf = metrics_row.get("phase_confidence", np.nan)
        score = metrics_row.get("phase_score", np.nan)
        conf_txt = f"{(float(conf) * 100.0):.1f}%" if np.isfinite(conf) else "N/A"
        st.markdown(
            f"- Phase estimee: **{phase}**\n"
            f"- Score: **{score if np.isfinite(score) else 'N/A'}**\n"
            f"- Confiance: **{conf_txt}**"
        )
        st.caption("Base de calcul: scoring sur thresholds.yaml, avec regles et alertes explicites.")

with st.expander("Etape 8 - Limites et bon usage", expanded=True):
    st.markdown(
        """
- Le prix scenario est un prix synthetique structurel, pas une prevision transactionnelle du spot.
- Le dispatch BESS est deterministe (pas d'optimisation economique complete).
- Les conclusions fortes exigent un `n` suffisant, une qualite de donnees correcte et une coherence regime/prix acceptable.
- Si les indicateurs sont plats, il faut verifier la physique de base avant d'inferer un "effet nul".
"""
    )
    st.caption(
        "Base de calcul: conventions v3 strictes, sans random, sans approximation interdite des echanges."
    )

section_header("Distribution annuelle", "NRL et surplus non absorbe")
d1, d2 = st.columns(2)
with d1:
    fig3 = px.histogram(df, x=COL_NRL, nbins=90, color=COL_REGIME, color_discrete_map=REGIME_COLORS)
    fig3.update_layout(height=350, xaxis_title="NRL (MW)", yaxis_title="Heures", **PLOTLY_LAYOUT_DEFAULTS)
    fig3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig3, use_container_width=True)
with d2:
    fig4 = px.histogram(df, x=COL_SURPLUS_UNABS, nbins=80)
    fig4.update_layout(height=350, xaxis_title="Surplus non absorbe (MW)", yaxis_title="Heures", **PLOTLY_LAYOUT_DEFAULTS)
    fig4.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig4.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig4, use_container_width=True)

render_commentary(
    so_what_block(
        title="Synthese de lecture",
        purpose="Le cadre est utile si la physique est lisible et si la validation externe reste coherente.",
        observed={
            "h_A": int((df[COL_REGIME] == "A").sum()),
            "h_B": int((df[COL_REGIME] == "B").sum()),
            "h_C": int((df[COL_REGIME] == "C").sum()),
            "h_D": int((df[COL_REGIME] == "D").sum()),
            "coherence_pct": stats["regime_coherence_pct"],
        },
        method_link="Regimes classes sur NRL/surplus/flex uniquement, puis confrontation au prix observe.",
        limits="Les seuils de phase sont des regles de diagnostic, pas une preuve causale stricte.",
        n=len(df),
        decision_use="Fonder les decisions sur des chiffres reproductibles et explicables de bout en bout.",
    )
)
