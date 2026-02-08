"""Page 9 - ExceSum statique (rapport final, sans moteur runtime)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.commentary_bridge import so_what_block
from src.ui_helpers import (
    challenge_block,
    dynamic_narrative,
    inject_global_css,
    narrative,
    render_commentary,
    render_kpi_banner,
    section_header,
)
from src.ui_theme import COUNTRY_PALETTE, PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS


st.set_page_config(page_title="ExceSum des conclusions", page_icon="🧾", layout="wide")
inject_global_css()

st.title("🧾 ExceSum des conclusions")
st.caption(
    "Rapport statique figé sur la baseline: FR / DE / ES / PL / DK, 2015-2024, "
    "modes observed / observed / observed. Aucun recalcul n'est exécuté à l'ouverture."
)

# ── Load static data ────────────────────────────────────────────────
json_path = Path("docs") / "EXCESUM_STATIC_REPORT.json"
if not json_path.exists():
    st.error(
        "Fichier statique manquant: `docs/EXCESUM_STATIC_REPORT.json`. "
        "La page ExceSum est volontairement non-dynamique et ne recalcule rien."
    )
    st.stop()

payload = json.loads(json_path.read_text(encoding="utf-8"))
meta = payload["meta"]
gm = payload["global_medians"]

df_means = pd.DataFrame(payload["by_country_means"])
df_latest = pd.DataFrame(payload["latest_year"]).sort_values("country")
df_q1 = pd.DataFrame(payload["q1_country"]).sort_values("country")
df_q1_detail = pd.DataFrame(payload["q1_detail"])
df_q2 = pd.DataFrame(payload["q2_slopes"]).sort_values("slope")
df_q3 = pd.DataFrame(payload["q3_transition"]).sort_values("country")
df_q4 = pd.DataFrame(payload["q4_summary"]).sort_values("country")
df_q5 = pd.DataFrame(payload["q5_commodity"]).sort_values("country")
df_q6 = pd.DataFrame(payload["q6_scope"]).sort_values("country")
df_country = pd.DataFrame(payload["country_conclusions"]).sort_values("country")
df_annex = pd.DataFrame(payload["metrics_annex"]).sort_values(["country", "year"])
verification = pd.DataFrame(meta.get("verification", []))

# ── Narrative d'introduction ────────────────────────────────────────
narrative(
    "Ce document est le rapport final statique du Capture Prices Analyzer. "
    "Toutes les conclusions ci-dessous sont figées sur un périmètre unique "
    "(5 pays, 10 années, 50 couples pays/année) et sur des conventions "
    "méthodologiques identiques. Aucun recalcul n'est exécuté à l'ouverture."
)

# ── KPI banner ──────────────────────────────────────────────────────
cols = st.columns(4)
cols[0].metric("Couverture", f"{len(df_annex)} couples pays/année")
cols[1].metric("Période", f"{meta['baseline']['years'][0]}-{meta['baseline']['years'][1]}")
cols[2].metric("Pays", ", ".join(meta["baseline"]["countries"]))
cols[3].metric("Outlier exclu pour pentes", ", ".join(str(y) for y in meta["baseline"]["exclude_outlier_for_slopes"]))

# =====================================================================
#   RÉSUMÉ EXÉCUTIF GLOBAL
# =====================================================================
section_header("Résumé exécutif global")

st.markdown(
    f"""
**Articulation logique des conclusions.**

Le rapport répond aux 6 questions de S. Michel dans une logique intégrée.
Les conclusions s'enchaînent comme suit :

1. **Le cadre physique est validé** (n=50 couples, complétude moyenne 98.6%).
   Le pipeline NRL → surplus → absorption → 4 régimes (A/B/C/D) produit des indicateurs
   cohérents sur l'ensemble du périmètre. La cohérence régime/prix médiane atteint
   `{gm['regime_coherence'] * 100:.1f}%`, ce qui confirme que la classification horaire
   est alignée avec les prix observés.

2. **Tous les pays sont classés `stage_2` en 2024** (Q1), mais avec des niveaux de confiance
   différents : FR (80%), DK (73%), DE (62%), ES (60%), PL (50%).
   La bascule vers stage_2 apparaît quand trois seuils sont simultanément franchis :
   heures négatives ≥ 200, heures sous 5 €/MWh ≥ 500, capture ratio PV ≤ 0.80.

3. **La dégradation du capture ratio PV est universelle** (Q2). Les 5 pentes sont négatives,
   avec une intensité variable : FR (-7.08 pp/pp de pénétration PV) est la plus rapide,
   ES (-2.13) la plus lente. 4 pays sur 5 sont statistiquement significatifs (p ≤ 0.05) ;
   PL est fragile (n=4, p=0.095).

4. **Aucun pays n'a atteint stage_3** (Q3). Tous sont en `transition_partielle` :
   le FAR est élevé mais les heures négatives continuent d'augmenter (+13 à +27 h/an
   selon les pays). La flexibilité domestique ne suffit pas encore à inverser la tendance.

5. **Le surplus baseline est déjà absorbé** (Q4) : surplus non absorbé = 0 TWh dans les 5 pays.
   Le sweep BESS est plat en baseline, ce qui est physiquement cohérent (pas de résidu à traiter).
   Sous stress PV additionnel, l'effet BESS redevient identifiable (DK dès +4 GW, DE seulement à +24 GW).

6. **Les commodités amplifient le TTL** (Q5) : CO₂↑ et gaz↑ augmentent systématiquement
   le coût thermique. L'asymétrie est marquée : PL (+86 €/MWh sous stress CO₂) est 3.5×
   plus sensible que FR (+25 €/MWh), reflet direct de la composition du mix thermique.

7. **Q6 (chaleur/froid) : pas de conclusion causale possible** avec les données actuelles.
   Un proxy qualitatif (BESS η=0.88 vs thermique η=0.50) est disponible dans l'onglet Q6
   de la page dynamique, mais il ne constitue pas une preuve de synergie ou de compétition.

**Niveaux médians sur l'ensemble de la base (n=50).**
"""
)

kpi_cols = st.columns(4)
kpi_cols[0].metric("SR médian", f"{gm['sr']:.4f}", help="Surplus Ratio — très faible, surplus marginal en baseline")
kpi_cols[1].metric("FAR médian", f"{gm['far']:.4f}", help="Flex Absorption Ratio — proche de 1 = quasi-total")
kpi_cols[2].metric("Capture ratio PV", f"{gm['capture_ratio_pv']:.4f}", help="Valeur captée par le PV vs prix moyen")
kpi_cols[3].metric("TTL médian", f"{gm['ttl']:.1f} €/MWh", help="Thermal Tail Level — coût marginal thermique")

kpi_cols2 = st.columns(3)
kpi_cols2[0].metric("IR médian", f"{gm['ir']:.4f}", help="Integration Ratio")
kpi_cols2[1].metric("Heures négatives", f"{gm['h_negative_obs']:.0f}", help="Médianes observées sur la base")
kpi_cols2[2].metric("Cohérence régime/prix", f"{gm['regime_coherence'] * 100:.1f}%", help="Alignement classification vs prix")

st.markdown("**Hiérarchisation des pays par urgence (2024).**")
risk_df = df_latest[["country", "phase_confidence", "capture_ratio_pv", "h_negative_obs", "sr"]].copy()
risk_df = risk_df.sort_values("capture_ratio_pv", ascending=True)
risk_df.columns = ["Pays", "Confiance phase", "Capture ratio PV", "Heures négatives", "SR"]
st.dataframe(risk_df.round(4), use_container_width=True, hide_index=True)

render_commentary(
    so_what_block(
        title="Synthèse globale (fixée)",
        purpose="Fournir une lecture consolidée rigoureuse avant le détail question par question.",
        observed={
            "n_couples": len(df_annex),
            "sr_median": gm["sr"],
            "far_median": gm["far"],
            "coherence_median_pct": gm["regime_coherence"] * 100.0,
        },
        method_link="Rapport figé sur baseline unique (5 pays × 10 ans), sans recalcul runtime.",
        limits="Conclusions valables pour ce périmètre précis uniquement ; elles ne se substituent pas à une causalité expérimentale.",
        n=len(df_annex),
        decision_use="Fournir un socle commun de décision avant arbitrage pays par pays.",
    )
)

# =====================================================================
#   TABS
# =====================================================================
tabs = st.tabs(
    [
        "Méthode et vérifications",
        "Q1 — Seuils stage_2",
        "Q2 — Pente de dégradation",
        "Q3 — Transition stage_3",
        "Q4 — Batteries",
        "Q5 — CO₂ et gaz",
        "Q6 — Chaleur/froid",
        "Conclusions pays",
        "Annexes",
    ]
)

# ── Tab 0 : Méthode ─────────────────────────────────────────────────
with tabs[0]:
    section_header("Méthode figée et contrôles de qualité")
    st.markdown(
        f"""
**Protocole figé du rapport**
- **Pays** : `{", ".join(meta['baseline']['countries'])}`
- **Période** : `{meta['baseline']['years'][0]}-{meta['baseline']['years'][1]}` (10 années × 5 pays = 50 couples)
- **Modes** : `{meta['baseline']['modes']}` (prix observés, production observée, must-run observé)
- **Outlier exclu pour les régressions** : `{", ".join(str(y) for y in meta['baseline']['exclude_outlier_for_slopes'])}` (crise énergétique)

**Conventions méthodologiques**
- **Seuils stage_2** : heures négatives ≥ 200, heures sous 5 €/MWh ≥ 500, capture ratio PV ≤ 0.80 (les 3 conditions doivent être remplies simultanément)
- **Seuil de significativité statistique** : p-value ≤ 0.05 (régression linéaire)
- **FAR** (Flex Absorption Ratio) : énergie absorbée par la flexibilité domestique (PSH + BESS + DSM, hors exports) / surplus total
- **SR** (Surplus Ratio) : surplus VRE / consommation totale
- **Régimes** : A = surplus non absorbé, B = surplus partiellement absorbé, C = pas de surplus, D = pénurie
- **Prix synthétique** : TCA (Thermal Cost Anchor) = gaz/η_ccgt + (ef_gas/η_ccgt) × CO₂ + VOM_ccgt

**Règle de gouvernance**
- Page statique : aucun recalcul à l'ouverture.
- Tous les résultats proviennent d'une extraction unique et figée.
- Les chiffres affichés sont identiques d'une session à l'autre.
"""
    )

    if not verification.empty:
        section_header("Contrôles automatiques", "Résultats des vérifications de cohérence")
        for _, row in verification.iterrows():
            status_str = str(row["status"]).upper()
            detail = str(row["detail"])
            check_name = str(row["check"])
            if status_str == "PASS":
                render_kpi_banner(check_name, "PASS", detail, status="strong")
            elif status_str == "WARN":
                render_kpi_banner(check_name, "WARN", detail, status="medium")
            else:
                render_kpi_banner(check_name, status_str, detail, status="weak")

    st.markdown("#### Moyennes par pays (2015-2024)")
    st.dataframe(df_means.round(4), use_container_width=True, hide_index=True)

# ── Tab 1 : Q1 ──────────────────────────────────────────────────────
with tabs[1]:
    section_header("Q1 — À quels niveaux observe-t-on la bascule vers stage_2 ?")
    st.markdown(
        """
**Réponse objective.**
En 2024, **4 pays sur 5** franchissent simultanément les 3 seuils de bascule stage_2 :
`DE`, `DK`, `ES` et `FR`. `PL` ne remplit pas encore l'ensemble des conditions.

**Critères de bascule** (les 3 doivent être remplis simultanément) :
- Heures à prix négatif ≥ 200
- Heures sous 5 €/MWh ≥ 500
- Capture ratio PV ≤ 0.80

**Première année de franchissement par pays :**
| Pays | Année | Commentaire |
|------|-------|-------------|
| DE | 2023 | Franchissement franc, confirmé en 2024 (457 h négatives, CR_PV=0.59) |
| DK | 2023 | Franchissement franc, confirmé en 2024 (375 h négatives, CR_PV=0.67) |
| ES | 2024 | Franchissement récent (247 h négatives, 1642 h sous 5€, CR_PV=0.67) |
| FR | 2024 | Franchissement récent (352 h négatives, 1018 h sous 5€, CR_PV=0.68) |
| PL | — | Non franchi (197 h négatives, 350 h sous 5€, CR_PV=0.75) |

**Lecture** : DE et DK ont basculé un an plus tôt que FR et ES. PL reste en deçà des seuils,
principalement à cause d'un nombre d'heures sous 5 €/MWh encore insuffisant (350 vs seuil de 500),
cohérent avec un mix encore dominé par le charbon et une pénétration VRE plus faible.
"""
    )

    st.caption("Chaque point = 1 pays/année. Le seuil horizontal marque h_neg=200.")
    fig_q1 = px.scatter(
        df_q1_detail,
        x="sr",
        y="h_negative_obs",
        color="country",
        color_discrete_map=COUNTRY_PALETTE,
        opacity=0.5,
        hover_data=["year", "capture_ratio_pv", "cross_all"],
    )
    fig_q1.add_hline(y=200, line_dash="dash", line_color="#e11d48", annotation_text="h_neg=200")
    fig_q1.update_layout(
        title="Q1 — SR vs heures négatives observées (points annuels)",
        height=480,
        xaxis_title="SR (surplus ratio)",
        yaxis_title="Heures à prix négatif",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig_q1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q1, use_container_width=True)
    st.dataframe(df_q1, use_container_width=True, hide_index=True)

    render_commentary(
        so_what_block(
            title="Q1 — Seuils de bascule observés",
            purpose="Le passage stage_2 correspond à un système qui ne digère plus facilement les surplus VRE.",
            observed={
                "pays_stage2_2024": 4,
                "pays_total": 5,
                "h_neg_max_2024": int(df_latest["h_negative_obs"].max()),
                "h_neg_min_2024": int(df_latest["h_negative_obs"].min()),
            },
            method_link="Lecture conjointe de 3 indicateurs (h_neg, h_below_5, capture_ratio_pv) avec seuils combinés.",
            limits="Seuils de diagnostic, pas de modèle causal. La chronologie pays par pays reste essentielle.",
            n=len(df_q1_detail),
            decision_use="Fixer des seuils d'alerte pour anticiper la bascule avant dégradation sévère du capture ratio.",
        )
    )

# ── Tab 2 : Q2 ──────────────────────────────────────────────────────
with tabs[2]:
    section_header("Q2 — Quelle est la pente de dégradation du capture ratio PV en phase 2 ?")
    st.markdown(
        """
**Réponse objective.**
Les 5 pentes sont négatives. La dégradation du capture ratio PV est **universelle** mais d'intensité variable.

**Unité** : variation du capture ratio PV par point de pourcentage de pénétration PV
(régression linéaire, hors 2022).

**Lecture comparée** (triée de la plus rapide à la plus lente) :

| Pays | Pente | R² | p-value | n | Robustesse |
|------|-------|----|---------|---|------------|
| FR | -0.0708 | 0.681 | 0.006 | 9 | Significatif |
| DE | -0.0397 | 0.746 | 0.003 | 9 | Significatif |
| DK | -0.0367 | 0.851 | 0.0004 | 9 | Significatif |
| PL | -0.0293 | 0.819 | 0.095 | 4 | Fragile |
| ES | -0.0213 | 0.932 | 0.00002 | 9 | Significatif |

**Interprétation** :
- **FR** perd 7.08 points de capture ratio PV par point de pénétration PV supplémentaire —
  la dégradation la plus rapide du panel, malgré un mix nucléaire important. Cela peut refléter
  la corrélation temporelle entre production PV et surplus dans un système à forte base nucléaire.
- **ES** a la pente la plus faible (-2.13) mais le R² le plus élevé (0.932), ce qui indique
  une dégradation régulière et prévisible.
- **PL** est le seul pays non significatif (p=0.095 > 0.05) avec seulement 4 points utiles.
  Le résultat est fragile et doit être interprété avec prudence.
- **4 pays sur 5** sont significatifs au seuil p ≤ 0.05.
"""
    )

    st.caption("Barres = pente de régression par pays. Couleur = significativité statistique (p ≤ 0.05).")
    fig_q2 = px.bar(
        df_q2,
        x="country",
        y="slope",
        color="robustesse",
        color_discrete_map={"forte": "#16a34a", "fragile": "#f59e0b"},
        hover_data=["r_squared", "p_value", "n_points"],
    )
    fig_q2.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
    fig_q2.update_layout(
        title="Q2 — Pentes de régression capture ratio PV vs pénétration PV (hors 2022)",
        height=420,
        xaxis_title="Pays",
        yaxis_title="Pente (Δ capture_ratio / Δ pénétration PV en pp)",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig_q2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q2, use_container_width=True)
    st.dataframe(df_q2.round(6), use_container_width=True, hide_index=True)

    render_commentary(
        so_what_block(
            title="Q2 — Intensité de cannibalisation",
            purpose="Plus la pente est négative, plus la valeur captée par le PV se dégrade vite quand la pénétration augmente.",
            observed={
                "slope_min_FR": float(df_q2[df_q2["country"] == "FR"]["slope"].iloc[0]) if "FR" in df_q2["country"].values else np.nan,
                "slope_max_ES": float(df_q2[df_q2["country"] == "ES"]["slope"].iloc[0]) if "ES" in df_q2["country"].values else np.nan,
                "n_significatifs": int((df_q2["p_value"] <= 0.05).sum()),
            },
            method_link="Régression linéaire (linregress) pays par pays sur séries annuelles normalisées, 2022 exclu.",
            limits="n souvent limité (9 points max) ; association statistique uniquement, pas de causalité.",
            n=int(df_q2["n_points"].sum()),
            decision_use="Comparer les vitesses de dégradation pour prioriser les leviers pays par pays.",
        )
    )

# ── Tab 3 : Q3 ──────────────────────────────────────────────────────
with tabs[3]:
    section_header("Q3 — Quelles conditions marquent le passage stage_2 → stage_3 ?")
    st.markdown(
        """
**Réponse objective.**
**Aucun pays n'est en transition effective vers stage_3.** Tous sont classés `transition_partielle`.

**Critères de passage stage_2 → stage_3** :
- FAR durablement élevé (la flexibilité domestique absorbe le surplus)
- Tendance des heures négatives **baissière** (le surplus résiduel diminue dans le temps)

**Constat par pays (2024)** :

| Pays | FAR | h_neg | Pente h_neg/an | h_regime_A | Statut |
|------|-----|-------|----------------|------------|--------|
| DE | 0.979 | 457 | +26.3 | 0 | transition_partielle |
| DK | 0.847 | 375 | +25.8 | 0 | transition_partielle |
| ES | 0.922 | 247 | +13.5 | 0 | transition_partielle |
| FR | 0.769 | 352 | +26.8 | 0 | transition_partielle |
| PL | 0.788 | 197 | +12.6 | 0 | transition_partielle |

**Lecture** :
- Les FAR sont élevés (0.77 à 0.98), ce qui signifie que la flexibilité domestique absorbe
  déjà 77% à 98% du surplus VRE. Cependant, les pentes d'heures négatives sont **toutes positives**
  (+13 à +27 h/an), ce qui invalide le critère de "détente" nécessaire au passage en stage_3.
- **h_regime_A = 0** pour tous les pays : aucune heure de surplus totalement non absorbé en 2024
  dans le cadre de cette modélisation. Le surplus existe mais il est intégralement redirigé
  vers la flexibilité domestique (PSH, BESS, DSM).
- La **hausse continue des heures négatives** montre que malgré un FAR élevé, le volume de
  surplus augmente plus vite que la capacité d'absorption. Le système n'est pas encore stabilisé.
"""
    )

    st.caption("Chaque point = dernier point annuel par pays. Position = FAR vs heures négatives.")
    fig_q3 = px.scatter(
        df_q3,
        x="far_latest",
        y="h_negative_latest",
        color="country",
        color_discrete_map=COUNTRY_PALETTE,
        opacity=0.5,
        hover_data=["h_negative_slope_per_year", "h_regime_a_latest", "status_transition_2_to_3"],
        text="country",
    )
    fig_q3.add_vline(x=0.60, line_dash="dash", line_color="#2563eb", annotation_text="FAR=0.60")
    fig_q3.update_traces(textposition="top center")
    fig_q3.update_layout(
        title="Q3 — FAR vs heures négatives (2024)",
        height=480,
        xaxis_title="FAR (Flex Absorption Ratio)",
        yaxis_title="Heures à prix négatif (2024)",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig_q3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q3, use_container_width=True)
    st.dataframe(df_q3.round(4), use_container_width=True, hide_index=True)

    render_commentary(
        so_what_block(
            title="Q3 — Conditions de transition vers l'absorption structurelle",
            purpose="Un FAR élevé seul ne suffit pas : les heures négatives doivent aussi baisser pour valider une transition robuste.",
            observed={
                "far_median": float(df_q3["far_latest"].median()),
                "h_neg_slope_median": float(df_q3["h_negative_slope_per_year"].median()),
                "n_transition_effective": 0,
            },
            method_link="Règles stage_3 basées sur FAR et dynamique temporelle des heures négatives.",
            limits="Les chocs commodités annuels (2022) peuvent masquer la tendance structurelle de flexibilité.",
            n=len(df_q3),
            decision_use="Valider si le système est prêt pour une accélération VRE ou s'il faut d'abord renforcer la flex.",
        )
    )

# ── Tab 4 : Q4 ──────────────────────────────────────────────────────
with tabs[4]:
    section_header("Q4 — Combien de batteries pour freiner la dégradation ?")
    st.markdown(
        """
**Réponse objective.**
Le sweep BESS baseline est **plat** dans les 5 pays (`plateau_baseline=True`,
`surplus_unabs_twh_baseline=0`). Ce résultat est **physiquement normal** :
en 2024, la flexibilité non-BESS (PSH, DSM, interconnexions domestiques) absorbe
déjà 100% du surplus dans les 5 pays. Il n'y a donc pas de résidu à traiter par du BESS additionnel.

**Sous stress PV additionnel**, l'effet BESS redevient identifiable dans tous les pays :

| Pays | Stress PV min (GW) | Interprétation |
|------|-------------------|----------------|
| DK | +4 GW | Très sensible — petit système, flex domestique limitée (0.8 GW) |
| ES | +4 GW | Sensible — pénétration PV déjà élevée, seuil de saturation proche |
| PL | +8 GW | Modéré — surplus encore faible, flex charbon partielle |
| FR | +20 GW | Résilient — large base nucléaire absorbe beaucoup de surplus |
| DE | +24 GW | Très résilient — flex domestique importante (PSH + gaz + interconnexions) |

**Explication physique du plateau** :
- Surplus total > 0 dans les 5 pays, mais FAR ≈ 0.77-0.98 → quasi-intégralement absorbé.
- Surplus non absorbé = 0 TWh → pas de matière à traiter par du BESS.
- Ajouter du BESS dans ce contexte ne change rien : il n'y a rien à charger.
- Le stress PV crée artificiellement du surplus non absorbé, rendant l'effet BESS mesurable.
"""
    )

    st.caption("Stress PV minimal (GW additionnels) pour que l'ajout de BESS devienne mesurable.")
    q4_ch = df_q4[["country", "stress_delta_pv_gw"]].copy().sort_values("stress_delta_pv_gw")
    fig_q4 = px.bar(
        q4_ch,
        x="country",
        y="stress_delta_pv_gw",
        color="country",
        color_discrete_map=COUNTRY_PALETTE,
    )
    fig_q4.update_layout(
        title="Q4 — Stress PV minimal pour rendre l'effet BESS identifiable",
        height=420,
        xaxis_title="Pays",
        yaxis_title="Delta PV additionnel (GW)",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig_q4.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q4.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q4, use_container_width=True)
    st.dataframe(df_q4.round(4), use_container_width=True, hide_index=True)

    challenge_block(
        "Interprétation méthodologique Q4",
        "Un plateau baseline n'est pas un bug de calcul. Il est physiquement cohérent "
        "avec un surplus non absorbé nul. Le stress PV permet de créer la contrainte "
        "nécessaire pour mesurer l'effet marginal des batteries. "
        "DK est le plus sensible (+4 GW) car son système est petit avec peu de flex domestique ; "
        "DE est le moins sensible (+24 GW) car sa flex domestique est déjà importante."
    )

    render_commentary(
        so_what_block(
            title="Q4 — Dimensionnement BESS et signal de surplus",
            purpose="Le BESS n'a d'effet mesurable que s'il existe un surplus résiduel non absorbé par la flex existante.",
            observed={
                "plateau_tous_pays": True,
                "stress_min_gw": float(df_q4["stress_delta_pv_gw"].min()),
                "stress_max_gw": float(df_q4["stress_delta_pv_gw"].max()),
                "surplus_unabs_baseline": 0.0,
            },
            method_link="Sweep déterministe +BESS sur baseline puis sous stress PV, avec recalcul complet du pipeline.",
            limits="Sensibilité dépendante des hypothèses de flex domestique et de must-run. Prix synthétiques.",
            n=len(df_q4),
            decision_use="Ne pas conclure à un 'effet batterie nul' sans vérifier la contrainte physique de départ.",
        )
    )

# ── Tab 5 : Q5 ──────────────────────────────────────────────────────
with tabs[5]:
    section_header("Q5 — Quel est l'impact du CO₂ et du gaz sur le coût thermique (TTL) ?")
    st.markdown(
        f"""
**Réponse objective.**
Les deux stress (`CO₂↑`, `gaz↑`) augmentent systématiquement le TTL synthétique dans tous les pays.

**Amplitudes observées (2024)** :

| Pays | TTL baseline | Δ TTL (CO₂↑) | Δ TTL (gaz↑) | Commentaire |
|------|-------------|---------------|---------------|-------------|
| DE | {df_q5[df_q5['country']=='DE']['ttl_baseline'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='DE']['delta_ttl_high_co2'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='DE']['delta_ttl_high_gas'].iloc[0]:.1f} | Sensibilité gaz > CO₂ |
| DK | {df_q5[df_q5['country']=='DK']['ttl_baseline'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='DK']['delta_ttl_high_co2'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='DK']['delta_ttl_high_gas'].iloc[0]:.1f} | Équilibre CO₂/gaz |
| ES | {df_q5[df_q5['country']=='ES']['ttl_baseline'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='ES']['delta_ttl_high_co2'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='ES']['delta_ttl_high_gas'].iloc[0]:.1f} | Sensibilité gaz > CO₂ |
| FR | {df_q5[df_q5['country']=='FR']['ttl_baseline'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='FR']['delta_ttl_high_co2'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='FR']['delta_ttl_high_gas'].iloc[0]:.1f} | Sensibilité la plus faible (mix nucléaire) |
| PL | {df_q5[df_q5['country']=='PL']['ttl_baseline'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='PL']['delta_ttl_high_co2'].iloc[0]:.1f} | +{df_q5[df_q5['country']=='PL']['delta_ttl_high_gas'].iloc[0]:.1f} | Très sensible au CO₂ (mix charbon) |

**Analyse de l'asymétrie** :
- **Sensibilité CO₂** (classement décroissant) : PL (+86) >> ES (+57) > DK (+52) > DE (+49) > FR (+25).
  La Pologne est 3.5× plus sensible que la France au stress CO₂, reflet direct d'un mix thermique
  encore dominé par le charbon (facteur d'émission élevé).
- **Sensibilité gaz** (classement décroissant) : ES (+64) > DE (+58) > DK (+51) > FR (+45) > PL (+31).
  L'Espagne est la plus sensible au gaz car son marginal thermique est un cycle combiné gaz.
  La Pologne, paradoxalement, est la moins sensible au gaz car son marginal est davantage charbon.
- **FR** est systématiquement le moins sensible aux deux stress, grâce à la base nucléaire
  qui réduit la dépendance aux combustibles fossiles pour le price-setting.
"""
    )

    q5m = df_q5.melt(
        id_vars=["country", "year"],
        value_vars=["delta_ttl_high_co2", "delta_ttl_high_gas"],
        var_name="scenario",
        value_name="delta_ttl",
    )
    q5m["scenario"] = q5m["scenario"].map({
        "delta_ttl_high_co2": "Stress CO₂",
        "delta_ttl_high_gas": "Stress gaz",
    })

    st.caption("Variation du TTL synthétique par pays sous stress CO₂ et gaz (2024).")
    fig_q5 = px.bar(
        q5m,
        x="country",
        y="delta_ttl",
        color="scenario",
        barmode="group",
    )
    fig_q5.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
    fig_q5.update_layout(
        title="Q5 — Variation du TTL par pays sous stress commodités",
        height=420,
        xaxis_title="Pays",
        yaxis_title="Δ TTL (€/MWh)",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig_q5.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q5.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q5, use_container_width=True)
    st.dataframe(df_q5.round(2), use_container_width=True, hide_index=True)

    render_commentary(
        so_what_block(
            title="Q5 — Sensibilité de l'ancre thermique aux commodités",
            purpose="Le gaz et le CO₂ déplacent le TCA et donc le TTL. L'asymétrie entre pays reflète la composition du mix thermique.",
            observed={
                "delta_co2_max_PL": float(df_q5[df_q5["country"] == "PL"]["delta_ttl_high_co2"].iloc[0]),
                "delta_co2_min_FR": float(df_q5[df_q5["country"] == "FR"]["delta_ttl_high_co2"].iloc[0]),
                "ratio_PL_sur_FR": float(
                    df_q5[df_q5["country"] == "PL"]["delta_ttl_high_co2"].iloc[0]
                    / max(df_q5[df_q5["country"] == "FR"]["delta_ttl_high_co2"].iloc[0], 1e-9)
                ),
            },
            method_link="Formule TCA du modèle prix synthétique v3 avec scénarios CO₂ et gaz.",
            limits="Ne capture pas les primes de rareté ni la microstructure du marché journalier.",
            n=len(df_q5),
            decision_use="Construire des stress tests commodités cohérents avant interprétation des variations de TTL.",
        )
    )

# ── Tab 6 : Q6 ──────────────────────────────────────────────────────
with tabs[6]:
    section_header("Q6 — Stockage chaleur/froid : synergie ou compétition avec les BESS ?")
    st.markdown(
        """
**Réponse objective et prudente.**
Avec les données actuellement présentes dans l'outil, **une conclusion causale robuste
sur la synergie ou la compétition chaleur-froid n'est pas possible**. Le statut est
`non_identifiable_sans_donnees_dediees` pour les 5 pays.

**Raison** : le périmètre de données couvre les prix de marché day-ahead, la production VRE,
la consommation et la flexibilité domestique (PSH, BESS, DSM). Il ne contient pas de variable
dédiée sur le stockage thermique (chaleur, froid, RFNBO, power-to-heat) ni sur les profils
de demande chaleur/froid.

**Proxy disponible dans l'outil** (page 6, onglet Q6) :
L'outil propose néanmoins une comparaison indicative entre BESS (rendement round-trip η = 0.88)
et stockage thermique (η = 0.50) sur une grille de durées de 2h à 24h :
- Sur les durées courtes (2-8h), le BESS est **nettement plus efficace** (énergie utile restituée
  supérieure à capacité équivalente).
- Sur les durées longues (12-24h), le stockage thermique **peut devenir compétitif** en termes
  de coût de capacité (CAPEX/MWh inférieur), mais avec un rendement de restitution plus faible.
- Le ratio thermique/BESS converge vers ~0.57 quelle que soit la durée (rapport des rendements).

**Ce proxy ne constitue pas une preuve** de synergie ou de compétition. Il structure la discussion
sur la segmentation des usages : court terme (BESS) vs longue durée (thermique), sans prétendre
à une conclusion causale.
"""
    )
    st.dataframe(df_q6, use_container_width=True, hide_index=True)

    dynamic_narrative(
        "Conclusion Q6 : pas d'invention au-delà des données. La réponse est volontairement "
        "prudente et méthodologiquement stricte. Un proxy indicatif (η BESS vs η thermique) est "
        "disponible dans la page dynamique (onglet Q6) pour structurer la réflexion, sans valeur causale.",
        severity="warning",
    )

    render_commentary(
        so_what_block(
            title="Q6 — Limite méthodologique assumée",
            purpose="Reconnaître explicitement les frontières de l'analyse plutôt que de forcer une conclusion non fondée.",
            observed={
                "heat_cold_data_available": False,
                "n_countries_assessed": 5,
                "proxy_available": True,
            },
            method_link="Absence de variable dédiée dans le périmètre actuel. Proxy η BESS vs η thermique en page 6.",
            limits="Sans données de demande chaleur/froid et de profils de stockage thermique, toute conclusion serait spéculative.",
            n=5,
            decision_use="Identifier le besoin de données complémentaires avant de statuer sur la complémentarité chaleur/BESS.",
        )
    )

# ── Tab 7 : Conclusions pays ────────────────────────────────────────
with tabs[7]:
    section_header("Conclusions détaillées pays par pays")

    st.markdown(
        "Le tableau ci-dessous synthétise les 5 pays sur les dimensions clés. "
        "Les fiches détaillées suivent, avec interprétation pour chaque question."
    )

    # Tableau comparatif synthétique
    comp_df = df_country[["country", "phase_latest", "sr_latest", "far_latest", "capture_ratio_pv_latest",
                           "q1_first_stage2_year", "q2_slope", "q3_status"]].copy()
    comp_df.columns = ["Pays", "Phase 2024", "SR 2024", "FAR 2024", "CR PV 2024",
                        "1ère année stage_2", "Pente Q2", "Statut Q3"]
    st.dataframe(comp_df.round(4), use_container_width=True, hide_index=True)

    # Fiches pays détaillées
    for _, row in df_country.iterrows():
        c = row["country"]
        lat = df_latest[df_latest["country"] == c].iloc[0] if c in df_latest["country"].values else None

        with st.expander(f"{c} — Fiche détaillée", expanded=False):
            # KPI banners
            if lat is not None:
                conf = float(lat.get("phase_confidence", 0))
                conf_status = "strong" if conf >= 0.70 else ("medium" if conf >= 0.55 else "weak")
                render_kpi_banner(
                    f"{c} — Phase 2024",
                    str(row["phase_latest"]),
                    f"Confiance : {conf:.0%}",
                    status=conf_status,
                )

            st.markdown(
                f"""
**Indicateurs clés 2024**
| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| SR | `{row['sr_latest']:.6f}` | {"Très élevé — surplus important" if row['sr_latest'] > 0.05 else "Faible — surplus marginal" if row['sr_latest'] < 0.01 else "Modéré"} |
| FAR | `{row['far_latest']:.4f}` | {"Élevé — bonne absorption" if row['far_latest'] > 0.90 else "Modéré — absorption partielle" if row['far_latest'] > 0.75 else "Faible — absorption insuffisante"} |
| Capture ratio PV | `{row['capture_ratio_pv_latest']:.4f}` | {"Sévèrement dégradé" if row['capture_ratio_pv_latest'] < 0.65 else "Dégradé" if row['capture_ratio_pv_latest'] < 0.75 else "Correct"} |
| h_neg observées | `{int(lat['h_negative_obs']) if lat is not None else '—'}` | {"Élevé (>300h)" if lat is not None and lat['h_negative_obs'] > 300 else "Modéré (200-300h)" if lat is not None and lat['h_negative_obs'] >= 200 else "Faible (<200h)" if lat is not None else "—"} |
| TTL | `{lat['ttl']:.1f} €/MWh` if lat is not None else '—' | Coût marginal thermique |

**Réponses aux 6 questions**

**Q1** — Première année de franchissement stage_2 : """
                + (f"`{int(row['q1_first_stage2_year'])}`" if not np.isnan(row["q1_first_stage2_year"]) else "**Non franchi**")
                + f"""
{"Franchissement confirmé en 2024." if not np.isnan(row['q1_first_stage2_year']) else "Les 3 seuils ne sont pas encore simultanément atteints."}

**Q2** — Pente de dégradation : `{row['q2_slope']:.4f}` par point de pénétration PV.
{"Dégradation rapide — pente parmi les plus négatives du panel." if row['q2_slope'] < -0.05 else "Dégradation modérée." if row['q2_slope'] < -0.03 else "Dégradation lente — pente la moins négative du panel." if row['q2_slope'] > -0.025 else "Dégradation modérée."}

**Q3** — Statut transition : `{row['q3_status']}`.
FAR élevé mais heures négatives en hausse → transition non effective.

**Q4** — Plateau baseline : `{bool(row['q4_plateau_baseline'])}`. Stress BESS trouvé : `{bool(row['q4_stress_found'])}`.
Surplus déjà absorbé en baseline ; BESS n'apporte un gain que sous stress PV additionnel.

**Q5** — Δ TTL sous stress CO₂ : `+{row['q5_delta_ttl_co2']:.1f} €/MWh`. Δ TTL sous stress gaz : `+{row['q5_delta_ttl_gas']:.1f} €/MWh`.
{"Très sensible au CO₂ (mix charbon)." if row['q5_delta_ttl_co2'] > 70 else "Sensibilité modérée." if row['q5_delta_ttl_co2'] > 40 else "Peu sensible au CO₂ (mix nucléaire/renouvelable)."}

**Q6** — `{row['q6_status']}`. Pas de conclusion causale possible avec les données actuelles.
"""
            )

    st.markdown("#### Indicateurs 2024 — vue complète")
    st.dataframe(df_latest.round(4), use_container_width=True, hide_index=True)

# ── Tab 8 : Annexes ─────────────────────────────────────────────────
with tabs[8]:
    section_header("Annexes chiffrées exhaustives")
    st.markdown(
        "Toutes les séries annuelles utilisées pour le rapport sont listées ci-dessous. "
        "La table couvre les 5 pays, 10 années, et les dimensions clés de la méthode."
    )
    st.dataframe(df_annex.round(6), use_container_width=True, hide_index=True)

    corr = float(np.nanmedian(df_annex["regime_coherence"].to_numpy()))
    render_commentary(
        so_what_block(
            title="Traçabilité finale",
            purpose="Garantir la reproductibilité et l'auditabilité des conclusions statiques.",
            observed={
                "n_rows_annex": len(df_annex),
                "n_countries": df_annex["country"].nunique(),
                "n_years": df_annex["year"].nunique(),
                "median_regime_coherence_pct": corr * 100.0,
            },
            method_link="Table figée exportée après calcul unique ; aucune mutation runtime.",
            limits="Ce rapport est une photographie méthodologiquement cohérente du périmètre actuel.",
            n=len(df_annex),
            decision_use="Utiliser cette base comme référence commune avant tout approfondissement ad hoc.",
        )
    )
