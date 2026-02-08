"""Page 0 - Comprendre le modele (parcours didactique complet)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import so_what_block
from src.constants import (
    COL_BESS_CHARGE,
    COL_FLEX_EFFECTIVE,
    COL_LOAD,
    COL_MUST_RUN,
    COL_NET_POSITION,
    COL_NRL,
    COL_PRICE_DA,
    COL_PSH_PUMP,
    COL_REGIME,
    COL_SINK_NON_BESS,
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
        COL_SINK_NON_BESS,
        COL_BESS_CHARGE,
        COL_NET_POSITION,
        COL_PSH_PUMP,
    ],
    with_notice=True,
)
df = coerce_numeric_columns(
    df,
    [
        COL_LOAD,
        COL_VRE,
        COL_MUST_RUN,
        COL_NRL,
        COL_SURPLUS,
        COL_FLEX_EFFECTIVE,
        COL_SURPLUS_UNABS,
        COL_PRICE_DA,
        COL_SINK_NON_BESS,
        COL_BESS_CHARGE,
        COL_NET_POSITION,
        COL_PSH_PUMP,
    ],
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

section_header("Point physique critique", "Que signifie exactement 'surplus non absorbe'")
surplus_total_mwh = float(df[COL_SURPLUS].fillna(0.0).sum())
absorbed_model_mwh = float(np.minimum(df[COL_SURPLUS].fillna(0.0), df[COL_FLEX_EFFECTIVE].fillna(0.0)).sum())
unabsorbed_mwh = float(df[COL_SURPLUS_UNABS].fillna(0.0).sum())
exports_pos_mwh = (
    float(df[COL_NET_POSITION].fillna(0.0).clip(lower=0.0).sum()) if COL_NET_POSITION in df.columns else float("nan")
)
psh_pump_mwh = (
    float(df[COL_PSH_PUMP].fillna(0.0).clip(lower=0.0).sum()) if COL_PSH_PUMP in df.columns else float("nan")
)
render_commentary(
    so_what_block(
        title="Interpretation physique sans ambiguite",
        purpose="Verifier que 'surplus non absorbe' est un residuel de modele, pas une energie qui disparait physiquement.",
        observed={
            "surplus_total_twh": surplus_total_mwh * 1e-6,
            "absorbed_model_twh": absorbed_model_mwh * 1e-6,
            "surplus_unabsorbed_twh": unabsorbed_mwh * 1e-6,
            "exports_pos_twh": exports_pos_mwh * 1e-6 if np.isfinite(exports_pos_mwh) else np.nan,
            "psh_pump_twh": psh_pump_mwh * 1e-6 if np.isfinite(psh_pump_mwh) else np.nan,
        },
        method_link=(
            "Identites du modele: surplus_unabsorbed = max(0, surplus - flex_effective), "
            "avec flex_effective = sink_non_bess + bess_charge. En mode observed, sink_non_bess "
            "inclut PSH pumping et net_position>0 (exports)."
        ),
        limits=(
            "Le residuel non absorbe indique une energie qui doit etre geree hors des sinks modelises "
            "(curtailment, baisse forcee, flexibilites non observees, ajustements infra-horaires)."
        ),
        n=len(df),
        decision_use="Ne pas lire ce residuel comme une violation physique; le lire comme signal de contrainte systeme.",
    )
)

if country == "FR":
    render_commentary(
        "<strong>Lecture specifique France : surplus d'origine nucleaire</strong><br>"
        "Le surplus non absorbe francais est principalement d'origine <strong>nucleaire</strong>, "
        "pas VRE. Le modele classe toute la production nucleaire observee comme must-run "
        "(IR &gt; 1.0 en 2024 : le must-run seul depasse la demande minimale). "
        "Ce surplus existait deja en 2015 (4043 h en regime A) avec seulement 5% de VRE.<br><br>"
        "<strong>Pourquoi la coherence regime/prix est faible pour FR (~28%)</strong> : "
        "le modele ne compte que PSH + exports + BESS comme flexibilite. "
        "Il ne modelise pas l'hydro barrage (~10 GW dispatchable en France), "
        "le DSM industriel ni les ajustements operationnels nucleaires reels. "
        "En consequence, le modele detecte 4437 h de regime A en 2024, "
        "mais seules ~350 h affichent des prix negatifs sur le marche.<br><br>"
        "<em>Consequence pour l'analyse</em> : les metriques FR (SR, FAR, h_regime_a) "
        "mesurent correctement le stress <strong>dans le perimetre modelise</strong> "
        "(PSH + exports + BESS), mais surestiment le stress reel du marche. "
        "Toujours croiser avec h_negative_obs et regime_coherence pour calibrer l'interpretation.",
        variant="warning",
    )

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
fig1.add_hline(y=0, line_dash="solid", line_width=1.5, line_color="#94a3b8")
fig1.update_layout(title="Profil horaire 48h — Load, VRE, NRL", height=480, xaxis_title="Heure", yaxis_title="MW", **PLOTLY_LAYOUT_DEFAULTS)
fig1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.caption("Deux jours consecutifs pour observer le cycle diurne du NRL.")
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
        opacity=0.5,
    )
    fig2.update_layout(
        title="NRL vs prix observe par regime",
        height=480,
        xaxis_title="NRL (MW)",
        yaxis_title="Prix observe (EUR/MWh)",
        legend_title="Regime",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.caption("Chaque point = 1 heure. Couleur = regime physique.")
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

        st.markdown("---")
        st.markdown("#### Comment lire chaque ratio : exemples concrets")

        st.markdown(
            """
##### SR — Surplus Ratio

**Definition** : SR = surplus VRE total (MWh) / generation totale (MWh).
Il mesure la **part de l'energie produite qui excede ce que le systeme peut consommer a l'instant t**.

**Formule** : quand VRE + Must-run > Load, le NRL devient negatif, et surplus = |NRL negatif|.
Le SR est la somme annuelle de ces surplus rapportee a la generation totale.

| Exemple | SR | Signification concrete |
|---------|-----|----------------------|
| Espagne 2015 | 0.0000 | Surplus quasi-nul. Le PV et l'eolien sont encore marginaux (< 5% de la generation). Le systeme absorbe tout sans difficulte. |
| Allemagne 2020 | 0.0007 | Surplus faible (0.07% de la generation). Quelques heures de surplus existent (typiquement midi en ete), mais elles restent marginales. |
| France 2024 | 0.107 | Surplus eleve (10.7% de la generation). Plus d'une heure sur dix produit un excedent. Le nucleaire (must-run rigide) combine au PV cree des surplus frequents. |
| Danemark 2024 | 0.128 | Surplus tres eleve (12.8%). Le Danemark a ~75% de VRE dans un petit systeme, ce qui sollicite fortement la flex modelisee. |

**Regle de lecture** : SR < 0.01 = surplus marginal. SR entre 0.01 et 0.05 = surplus significatif. SR > 0.05 = surplus structurel, la flex est sollicitee en permanence.
"""
        )

        st.markdown(
            """
##### FAR — Flex Absorption Ratio

**Definition** : FAR = energie absorbee par la flexibilite modelisee / surplus total.
Dans ce modele, la flexibilite modelisee = `flex_effective = sink_non_bess + bess_charge`.
En mode `observed`, `sink_non_bess` inclut le pompage PSH et `net_position>0` (exports nets positifs).

**Formule** : FAR = min(surplus, flex_effective).sum() / surplus.sum(). Un FAR de 1.0 signifie que toute l'energie excedentaire est absorbee localement.

| Exemple | FAR | Signification concrete |
|---------|------|----------------------|
| Allemagne 2024 | 0.979 | La flexibilite modelisee absorbe 97.9% du surplus. Seuls 2.1% restent non absorbes dans le perimetre du modele. |
| Espagne 2024 | 0.922 | 92.2% absorbe. Le deficit de flex commence a se voir : 7.8% de surplus residuel, correspondant a des heures ou ni le PSH ni les batteries ne suffisent. |
| Danemark 2024 | 0.847 | 84.7% absorbe. Le residuel (15.3%) signale une contrainte de flex dans le perimetre modelise. |
| France 2024 | 0.769 | 76.9% absorbe. Avec un surplus eleve (SR=0.107), le residuel non absorbe reste important dans le perimetre modelise. |

**Regle de lecture** : FAR > 0.95 = flex suffisante. FAR entre 0.80 et 0.95 = flex sous tension. FAR < 0.80 = deficit de flex, le systeme depend des exports ou des curtailments.

**Attention** : FAR = 1.0 ne signifie pas "tout va bien" — cela signifie juste que le surplus (qui peut etre faible) est absorbe. Il faut toujours lire FAR **conjointement avec SR** : un FAR de 1.0 avec un SR de 0.0001 n'a pas la meme signification qu'un FAR de 1.0 avec un SR de 0.10.
"""
        )

        st.markdown(
            """
##### IR — Inflexibility Ratio

**Definition** : IR = P10(must-run) / P10(load).
Il mesure la **rigidite structurelle du systeme en creux de charge** : quelle part de la demande minimale est deja occupee par des centrales qui ne peuvent pas s'arreter (nucleaire, CHP, biomasse).

**Formule** : P10 = 10e percentile (les 10% d'heures les plus basses). Si P10(must-run) = 20 GW et P10(load) = 25 GW, alors IR = 0.80 → il ne reste que 5 GW pour absorber le VRE en creux.

| Exemple | IR | Signification concrete |
|---------|-----|----------------------|
| Danemark 2024 | 0.001 | Quasiment aucune rigidite. Le DK n'a ni nucleaire ni gros CHP obligatoire. Toute la place est libre pour le VRE, meme en creux de nuit. |
| Espagne 2024 | 0.295 | Rigidite moderee. Environ 30% de la demande minimale est occupee par du must-run (nucleaire + CHP). Il reste 70% de marge pour le VRE. |
| Pologne 2024 | 0.331 | Rigidite moderee-haute. Le charbon en baseload et le CHP occupent un tiers de la demande minimale. Le VRE doit se glisser dans les 67% restants. |
| France 2024 | 1.063 | Rigidite extreme (> 1.0 !). Le must-run nucleaire **depasse** la demande minimale en creux. Cela signifie que meme sans aucun VRE, le systeme produit deja trop en creux de nuit. L'ajout de VRE ne fait qu'aggraver le surplus. |

**Regle de lecture** : IR < 0.30 = systeme flexible. IR entre 0.30 et 0.70 = rigidite moderee. IR > 0.70 = systeme rigide (la marge pour le VRE est faible). IR > 1.0 = le must-run excede la demande minimale, surplus structurel garanti.

**Cas FR** : IR > 1 explique pourquoi la France a un SR eleve (0.107) malgré une penetration PV moderee (< 10%). Le nucleaire occupe toute la place en creux → le moindre MW de VRE cree du surplus.
"""
        )

        st.markdown(
            """
##### TTL — Thermal Tail Level

**Definition** : TTL = quantile 95% de `price_used` sur les heures de regimes C et D.
Il mesure la **queue haute des prix** quand le systeme est en regimes thermiques/pointe.

**Formule** : TTL = Q95(`price_used`) sur heures C+D. Les heures de surplus (A, B) sont exclues.

| Exemple | TTL | Signification concrete |
|---------|------|----------------------|
| Allemagne 2015 | 52 €/MWh | Gaz bon marche, CO2 a ~8 €/t. Le marginal thermique (CCGT) ne coute pas cher → les prix hors surplus sont moderes. |
| Espagne 2024 | 140 €/MWh | Gaz a ~35 €/MWh_th, CO2 a ~65 €/t. Le cout du CCGT marginal s'est beaucoup alourdi depuis 2020. |
| France 2024 | 158 €/MWh | Plus eleve que ES malgré le nucleaire, car les heures C/D en France sont souvent des heures de pointe hivernale ou le gaz est au marginal. |
| Pologne 2024 | 173 €/MWh | Le plus eleve du panel. Le charbon (facteur d'emission eleve) subit de plein fouet la hausse du CO2. Le marginal thermique polonais est structurellement cher. |

**Regle de lecture** : TTL < 80 €/MWh = commodites bon marche. TTL entre 80 et 150 €/MWh = niveau standard post-2020. TTL > 150 €/MWh = systeme ou le marginal thermique est cher (charbon+CO2 eleve ou gaz en tension).

**Importance pour l'analyse** : le TTL est "l'ancre" des prix hors surplus. Quand le surplus augmente (SR↑), les prix baissent en dessous du TTL. La difference entre le TTL et le prix moyen PV (capture price) mesure exactement la cannibalisation. Si TTL = 150 et que le PV capture 100, le capture ratio PV = 100/150 = 0.67 → le PV ne capte que 67% de la valeur thermique.
"""
        )

        st.markdown(
            """
##### Lecture croisee : pourquoi il faut les 4 ratios ensemble

Un seul ratio ne dit rien. C'est leur **combinaison** qui raconte l'histoire :

| Situation | SR | FAR | IR | TTL | Diagnostic |
|-----------|-----|------|-----|------|-----------|
| Systeme pre-transition (ES 2015) | ~0 | ~1.0 | 0.29 | 52 | Pas de surplus, flex inutile, marginal thermique bon marche. Aucun stress. |
| Transition debutante (DE 2020) | 0.001 | 1.0 | 0.35 | 119 | Surplus faible mais reel. La flex absorbe tout. Le TTL a monte (gaz/CO2). Le systeme tient mais la pression monte. |
| Stress avance (FR 2024) | 0.107 | 0.77 | 1.06 | 158 | Surplus massif (IR > 1 !). La flex ne suit plus (FAR < 0.80). Le TTL est eleve. Le capture ratio PV se degrade vite. C'est le profil le plus stresse du panel. |
| Stress VRE massif (DK 2024) | 0.128 | 0.85 | 0.001 | 153 | Surplus tres eleve malgré un IR quasi-nul. Le probleme n'est pas la rigidite mais le volume : 75% de VRE depasse la capacite d'absorption locale (0.8 GW de flex). |

**Conclusion** : SR mesure le probleme, FAR mesure la reponse, IR explique la cause structurelle, TTL donne le contexte economique.
"""
        )

    with st.expander("Etape 7 - Diagnostic de phase", expanded=True):
        phase = metrics_row.get("phase", "unknown")
        conf = metrics_row.get("phase_confidence", np.nan)
        score = metrics_row.get("phase_score", np.nan)
        blocked_rules_txt = str(metrics_row.get("phase_blocked_rules", "") or "").strip()
        conf_txt = f"{(float(conf) * 100.0):.1f}%" if np.isfinite(conf) else "N/A"

        phase_status = "strong" if np.isfinite(conf) and conf >= 0.70 else ("medium" if np.isfinite(conf) and conf >= 0.50 else "weak")
        render_kpi_banner(
            f"Diagnostic de phase — {country} {year}",
            str(phase),
            f"Score = {score if np.isfinite(score) else 'N/A'} | Confiance = {conf_txt}",
            status=phase_status,
        )
        if blocked_rules_txt:
            st.caption(f"Regles bloquees: {blocked_rules_txt}")

        st.markdown(
            """
#### Systeme de scoring par phase

L'outil classe chaque couple pays/annee dans une des 4 phases en attribuant des **points** pour chaque critere rempli.
La phase avec le score le plus eleve l'emporte (minimum 2 points requis). La **confiance** = score obtenu / score total possible.

---

##### Stage 1 — Pre-transition (systeme confortable)

Le VRE est marginal, le systeme n'a pas de surplus significatif.

| Critere | Seuil | Points | Signification |
|---------|-------|--------|---------------|
| Heures negatives | ≤ 100 h/an | +1 | Tres peu d'heures a prix negatif — le VRE ne perturbe pas le marche |
| Heures sous 5 €/MWh | ≤ 200 h/an | +1 | Tres peu d'heures a bas prix — pas de pression sur les revenus |
| Capture ratio PV | ≥ 0.85 | +1 | Le PV capte au moins 85% de la valeur moyenne — pas de cannibalisation significative |
| SR | ≤ 0.01 | +1 | Le surplus est inferieur a 1% de la generation — negligeable |

**Exemple type** : Espagne 2015-2017 — VRE < 25%, 0 heures negatives, capture ratio PV > 0.99, SR ≈ 0. Le systeme est dans un equilibre confortable, le thermique (gaz + charbon) fixe le prix sans interference du VRE.

**Score max** : 4 points.

---

##### Stage 2 — Stress de penetration (cannibalisation active)

Le surplus VRE devient frequent, les prix sont sous pression, la valeur captee par le PV se degrade.

| Critere | Seuil | Points | Signification |
|---------|-------|--------|---------------|
| Heures negatives | ≥ 200 h/an | +1 | Le surplus deprime les prix pendant plus de 200 heures |
| Heures negatives (seuil fort) | ≥ 300 h/an | +2 (bonus) | Signal de stress severe — les prix negatifs deviennent frequents |
| Heures sous 5 €/MWh | ≥ 500 h/an | +1 | La zone de bas prix s'elargit considerablement |
| Capture ratio PV | ≤ 0.80 | +1 | Le PV perd au moins 20% de la valeur thermique |
| Capture ratio PV (seuil crise) | ≤ 0.70 | +2 (bonus) | Cannibalisation severe — le PV ne capte que 70% ou moins |
| Jours avec spread > 50 €/MWh | ≥ 150 j/an | +1 | Volatilite intra-journaliere elevee (ecart min/max de prix) |

**Exemple type** : Allemagne 2024 — 457 heures negatives (≥ 300, bonus +2), 756 heures sous 5 € (≥ 500, +1), capture ratio PV = 0.59 (≤ 0.70, bonus +2). Score stage_2 tres eleve. Le PV ne capte que 59% de la valeur thermique → cannibalisation severe.

**Exemple intermediaire** : Pologne 2024 — 197 heures negatives (< 200, pas de point), 350 h sous 5 € (< 500, pas de point), capture ratio PV = 0.75 (≤ 0.80, +1). Score stage_2 faible → le classement stage_2 est incertain (confiance = 50%).

**Score max** : 8 points (avec les 2 bonus).

---

##### Stage 3 ??? Absorption structurelle (la flex repond)

La flexibilite absorbe une part significative du surplus **apres** un episode de stress stage 2.

| Critere | Seuil | Points | Signification |
|---------|-------|--------|---------------|
| FAR | ??? 0.60 | +1 | Au moins 60% du surplus est absorbe par la flex modelisee |
| FAR (seuil fort) | ??? 0.80 | +2 (bonus) | Absorption elevee ??? la flex est dimensionnee pour le surplus actuel |
| Tendance heures negatives | Baissiere | (bloquant) | Les heures negatives doivent **diminuer** dans le temps ??? pas juste un FAR eleve ponctuellement |
| Historique stage 2 recent | max(h_neg sur 3 ans) ??? 200 | (bloquant) | Le systeme doit avoir connu un niveau de stress stage 2 recent avant d'etre classe en stage 3 |

**Pourquoi aucun pays n'est en stage_3 en 2024** : les FAR sont eleves (0.77 a 1.00), mais la tendance des heures negatives n'est pas baissiere sur 3 ans. Certains pays sont aussi bloques faute d'historique stage 2 recent.

**Exemple 2021** : l'Allemagne peut etre stage_3 (FAR eleve + tendance baissiere + pic recent >= 200), alors que le Danemark est bloque (`require_stage2_history`) malgre FAR eleve, car son pic recent reste < 200.

**Score max** : 3 points.

---

##### Stage 4 — Equilibre post-transition (VRE domine le prix)

Le systeme a integre le VRE : la flex absorbe quasi-tout, le thermique recule structurellement.

| Critere | Seuil | Points | Signification |
|---------|-------|--------|---------------|
| FAR | ≥ 0.90 | +1 | 90%+ du surplus absorbe localement |
| Heures regime C | ≤ 1500 h/an | +1 | Le thermique ne fixe plus le prix que < 1500 heures par an (< 17% du temps) |

**Signification** : en stage_4, le VRE + flex dominent le merit order. Le thermique ne sert plus que pour les pointes extremes. Le TTL perd de sa pertinence comme indicateur de prix moyen.

**Aucun pays n'approche stage_4** dans la base actuelle : les heures regime C depassent largement 1500 dans tous les pays (7000-8000 h typiques).

**Score max** : 2 points.

---

##### Comment la confiance est calculee

```
confiance = meilleur_score / somme_de_tous_les_scores
```

Si le meilleur score est de 5 et la somme totale est de 8, la confiance = 62.5%.
- **Confiance ≥ 70%** : classification robuste (vert)
- **50% ≤ Confiance < 70%** : classification probable mais fragile (bleu)
- **Confiance < 50%** : classification incertaine (orange)
- **Confiance < 30%** : classification rejetee → `unknown`

##### Alertes automatiques

En complement du diagnostic de phase, des alertes se declenchent quand certaines combinaisons sont detectees :

| Alerte | Condition | Signification |
|--------|-----------|---------------|
| Approche Stage 2 | 150 ≤ h_neg ≤ 300 ET 0.75 ≤ CR_PV ≤ 0.85 | Le systeme est en zone de transition — surveiller |
| Stage 2 severe | h_neg ≥ 500 ET CR_PV ≤ 0.65 | Cannibalisation profonde — intervention necessaire |
| IR eleve | IR ≥ 0.60 | Rigidite structurelle — le must-run limite la marge VRE |
| Flex insuffisante | FAR ≤ 0.30 ET SR ≥ 0.02 | Le surplus n'est pas absorbe — deficit de flexibilite |
"""
        )
        st.caption("Base de calcul: scoring sur thresholds.yaml, avec regles et alertes explicites.")

with st.expander("Etape 8 - Limites et bon usage", expanded=True):
    st.markdown(
        """
- Le prix scenario est un prix synthetique structurel, pas une prevision transactionnelle du spot.
- Le dispatch BESS est deterministe (pas d'optimisation economique complete).
- Les conclusions fortes exigent un `n` suffisant, une qualite de donnees correcte et une coherence regime/prix acceptable.
- Si les indicateurs sont plats, il faut verifier la physique de base avant d'inferer un "effet nul".
- **Must-run = production observee** : le modele traite toute la production nucleaire/charbon/biomasse/hydro RoR comme inflexible. Pour la France (IR > 1), cela surestime le surplus car le nucleaire est en realite partiellement modulable. La coherence regime/prix pour FR (~28%) reflete cette limitation. Pour les pays sans gros must-run nucleaire (DE, DK, PL), le modele est bien calibre (coherence > 90%).
- **Flex modelisee limitee** : le modele ne compte que PSH + exports + BESS. L'hydro barrage, le DSM et les ajustements thermiques ne sont pas modelises comme absorption de surplus.
"""
    )
    st.caption(
        "Base de calcul: conventions v3 strictes, sans random, sans approximation interdite des echanges."
    )

section_header("Distribution annuelle", "NRL et surplus non absorbe")
st.caption("NRL colore par regime (gauche), surplus residuel (droite).")
d1, d2 = st.columns(2)
with d1:
    fig3 = px.histogram(df, x=COL_NRL, nbins=90, color=COL_REGIME, color_discrete_map=REGIME_COLORS)
    fig3.update_layout(title="Distribution du NRL", height=420, xaxis_title="NRL (MW)", yaxis_title="Heures", **PLOTLY_LAYOUT_DEFAULTS)
    fig3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig3, use_container_width=True)
with d2:
    fig4 = px.histogram(df, x=COL_SURPLUS_UNABS, nbins=80)
    fig4.update_layout(title="Distribution du surplus non absorbe", height=420, xaxis_title="Surplus non absorbe (MW)", yaxis_title="Heures", **PLOTLY_LAYOUT_DEFAULTS)
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
