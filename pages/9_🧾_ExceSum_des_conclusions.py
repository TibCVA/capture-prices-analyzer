"""Static ExceSum report page (frozen dataset, no runtime recompute)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui_helpers import (
    challenge_block,
    dynamic_narrative,
    inject_global_css,
    narrative,
    question_banner,
    render_kpi_banner,
    section_header,
)
from src.ui_theme import COUNTRY_PALETTE, PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS


st.set_page_config(page_title="ExceSum des conclusions", page_icon="🧾", layout="wide")
inject_global_css()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_static_payload() -> dict:
    path = Path("docs") / "EXCESUM_STATIC_REPORT.json"
    if not path.exists():
        st.error("Fichier manquant: docs/EXCESUM_STATIC_REPORT.json")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_df(payload: dict, key: str) -> pd.DataFrame:
    value = payload.get(key, [])
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _sf(v, fmt=".4f") -> str:
    """Safe format a numeric value."""
    try:
        f = float(v)
        if not np.isfinite(f):
            return "n/a"
        return f"{f:{fmt}}"
    except Exception:
        return "n/a"


payload = _load_static_payload()
meta = payload.get("meta", {})
baseline = meta.get("baseline", {})
global_medians = payload.get("global_medians", {})
rebuild_matrix = payload.get("rebuild_matrix", {})
by_country = payload.get("by_country_means", [])
latest_year = payload.get("latest_year", [])

df_q1_detail = _safe_df(payload, "q1_detail")
df_q1_country = _safe_df(payload, "q1_country")
df_q2 = _safe_df(payload, "q2_slopes")
df_q3 = _safe_df(payload, "q3_transition")
df_q4 = _safe_df(payload, "q4_summary")
df_q5 = _safe_df(payload, "q5_commodity")
df_q6 = _safe_df(payload, "q6_scope")
df_country = _safe_df(payload, "country_conclusions")
df_annex = _safe_df(payload, "metrics_annex")
df_quality = _safe_df(payload, "data_quality_flags")
df_verification = _safe_df({"x": meta.get("verification", [])}, "x")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ExceSum des conclusions")
st.caption(
    "Rapport statique fige sur baseline unique. "
    "Aucun recalcul n'est execute a l'ouverture."
)

narrative(
    "Cette page est une synthese executive statique, fondee sur un jeu de resultats fixe "
    "et reproductible. Toutes les valeurs sont tracees vers le JSON source "
    "<code>docs/EXCESUM_STATIC_REPORT.json</code>."
)

# --- KPI banner row ---
cols = st.columns(5)
cols[0].metric("Pays", len(baseline.get("countries", [])))
years = baseline.get("years", [2015, 2024])
cols[1].metric("Periode", f"{years[0]}-{years[-1]}")
n_pairs = int(rebuild_matrix.get("pairs_total", len(df_annex)))
cols[2].metric("Couples pays-annee", n_pairs)
n_ha = int(rebuild_matrix.get("pairs_h_regime_a_gt_0", 0))
cols[3].metric("h_regime_a > 0", f"{n_ha}/{n_pairs}")
cols[4].metric("Coherence regime/prix", f"{100 * global_medians.get('regime_coherence', 0):.1f}%")

# --- Global synthesis ---
section_header("Synthese globale", "Medianes panel 5 pays x 10 ans (2015-2024)")

narrative(
    "L'ensemble du panel se situe en debut de phase 2 en 2024. "
    "Le surplus structurel (SR) reste tres faible en mediane, mais la degradation du capture ratio PV "
    "est significative dans 4 pays sur 5. "
    "La flex domestique (PSH + BESS, hors exports) n'absorbe pas la totalite du surplus dans les pays a fort VRE : "
    f"{n_ha} couples sur {n_pairs} presentent des heures en regime A (surplus non absorbe)."
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_kpi_banner("SR median", _sf(global_medians.get("sr"), ".4f"),
                      "Surplus structurel tres faible", "medium")
with kpi_cols[1]:
    far_med = global_medians.get("far", 0)
    render_kpi_banner("FAR median", _sf(far_med, ".3f"),
                      "Absorption domestique elevee en mediane", "strong" if far_med > 0.95 else "medium")
with kpi_cols[2]:
    render_kpi_banner("IR median", _sf(global_medians.get("ir"), ".3f"),
                      "Part must-run / charge P10", "medium")
with kpi_cols[3]:
    render_kpi_banner("TTL median", f"{_sf(global_medians.get('ttl'), '.1f')} EUR/MWh",
                      "Ancre thermique longue duree", "medium")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "Methode",
    "Q1 - Seuils stage_2",
    "Q2 - Pentes CR_PV",
    "Q3 - Transition 2->3",
    "Q4 - Batteries",
    "Q5 - Commodites",
    "Q6 - Chaleur/froid",
    "Conclusions pays",
    "Annexes",
])

# ===== TAB 0 : METHODE =====
with tabs[0]:
    section_header("Conventions methodologiques")
    narrative(
        "Ce rapport utilise une baseline unique : 5 pays (FR, DE, ES, PL, DK) x 10 ans (2015-2024), "
        "modes observed/observed/observed. L'annee 2022 (crise energetique) est exclue des regressions Q2."
    )

    st.markdown("""
**Seuils stage_2** (page 0, etape 7) :
- `h_negative_obs >= 200` (+1 pt), `>= 300` (+2 pts)
- `h_below_5_obs >= 500` (+1 pt)
- `capture_ratio_pv <= 0.80` (+1 pt), `<= 0.70` (+2 pts)

**Seuil de significativite des pentes Q2** : p-value <= 0.05. Au-dessus : "fragile".

**Seuil FAR pour transition Q3** : FAR >= 0.60 (+ tendance h_neg declinante).

**Outlier exclu** : 2022 (crise energetique, prix hors normes).

**Flex domestique** : PSH + BESS (hors exports/net_position). Les exports ne sont pas comptes
comme de la flexibilite domestique car ils dependent de la capacite d'interconnexion des voisins.
""")

    section_header("Controle qualite et verifications")
    if not df_verification.empty:
        st.dataframe(df_verification, use_container_width=True, hide_index=True)

    if rebuild_matrix:
        rm = rebuild_matrix
        st.markdown(f"""
**Rebuild matrix** : {rm.get('pairs_total', '?')} couples recalcules,
{rm.get('pairs_h_regime_a_gt_0', 0)} avec h_regime_a > 0,
{rm.get('cache_semantic_invalid_pairs', 0)} caches invalides corriges.
Phases : stage_1={rm.get('phase_distribution', {}).get('stage_1', '?')},
stage_2={rm.get('phase_distribution', {}).get('stage_2', '?')},
stage_3={rm.get('phase_distribution', {}).get('stage_3', '?')}.
""")


# ===== TAB 1 : Q1 =====
with tabs[1]:
    question_banner("Q1 - A quels niveaux observe-t-on la bascule vers stage_2 ?")

    dynamic_narrative(
        "Reponse courte : la bascule vers stage_2 apparait quand h_negative_obs, h_below_5_obs "
        "et capture_ratio_pv franchissent simultanement leurs seuils respectifs "
        "(h_neg >= 200, h_below_5 >= 500, CR_PV <= 0.80).",
        severity="info",
    )

    narrative(
        "Le graphique ci-dessous montre l'evolution de chaque pays dans l'espace "
        "(h_negative_obs, capture_ratio_pv). Les lignes de seuil indiquent les zones de bascule. "
        "Un pays qui traverse les deux lignes simultanement entre en zone stage_2."
    )

    if not df_q1_detail.empty:
        st.caption("Scatter : capture_ratio_pv vs h_negative_obs par pays et annee (2015-2024)")
        fig_q1 = px.scatter(
            df_q1_detail,
            x="h_negative_obs",
            y="capture_ratio_pv",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
            hover_data=["year", "sr", "h_below_5_obs"],
            opacity=0.5,
        )
        fig_q1.add_hline(y=0.80, line_dash="dash", line_color="#94a3b8",
                         annotation_text="CR_PV=0.80", annotation_position="top left")
        fig_q1.add_vline(x=200, line_dash="dash", line_color="#94a3b8",
                         annotation_text="h_neg=200", annotation_position="top right")
        fig_q1.update_layout(
            height=480,
            xaxis_title="Heures negatives observees",
            yaxis_title="Capture ratio PV",
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        fig_q1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q1, use_container_width=True)

    if not df_q1_country.empty:
        section_header("Resume par pays")
        for _, row in df_q1_country.iterrows():
            c = row["country"]
            first_yr = row.get("first_stage2_cross_year")
            crossed = row.get("latest_cross_all", False)
            yr_str = f"{int(first_yr)}" if pd.notna(first_yr) else "jamais"
            status = "Franchi" if crossed else "Non franchi"
            st.markdown(
                f"- **{c}** : premier franchissement simultane = **{yr_str}** | "
                f"2024 = **{status}** | CR_PV = {_sf(row.get('latest_capture_ratio_pv'), '.3f')} | "
                f"h_neg = {_sf(row.get('latest_h_negative_obs'), '.0f')}"
            )

        challenge_block(
            "PL : seuils non franchis simultanement",
            "La Pologne ne franchit pas les 3 seuils en meme temps sur la periode 2015-2024. "
            "Seul le capture_ratio_pv passe sous 0.80 en 2024, mais h_neg (197) reste sous 200 et "
            "h_below_5 (350) reste sous 500.",
        )


# ===== TAB 2 : Q2 =====
with tabs[2]:
    question_banner("Q2 - Quelle est la pente de degradation du capture ratio PV en phase 2 ?")

    dynamic_narrative(
        "Reponse courte : la pente mesure la variation du capture_ratio_pv par point de pourcentage "
        "de penetration PV (regression lineaire, hors 2022). Une pente de -0.07 signifie que chaque "
        "point supplementaire de PV dans le mix fait baisser le capture ratio de 7 points de base.",
        severity="info",
    )

    if not df_q2.empty:
        # Map JSON "forte" → "significatif" for consistency with page 6
        df_q2_display = df_q2.copy().sort_values("slope")
        if "robustesse" in df_q2_display.columns:
            df_q2_display["robustesse"] = df_q2_display["robustesse"].replace(
                {"forte": "significatif"}
            )

        st.caption("Pentes capture_ratio_pv vs penetration PV par pays (hors 2022)")
        color_map = {"significatif": "#16a34a", "fragile": "#f59e0b"}
        fig_q2 = px.bar(
            df_q2_display,
            x="country",
            y="slope",
            color="robustesse",
            color_discrete_map=color_map,
            hover_data=["r_squared", "p_value", "n_points"],
        )
        fig_q2.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig_q2.update_layout(
            height=420,
            xaxis_title="Pays",
            yaxis_title="Pente (variation CR_PV / pp penetration PV)",
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        fig_q2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q2, use_container_width=True)

        narrative(
            "**Lecture comparative** : FR presente la degradation la plus rapide "
            f"(pente = {_sf(df_q2_display[df_q2_display['country'] == 'FR']['slope'].iloc[0] if len(df_q2_display[df_q2_display['country'] == 'FR']) > 0 else float('nan'), '.4f')}), "
            "suivie de DE, DK et PL. ES affiche la pente la plus lente. "
            "4 pays sur 5 sont significatifs a p <= 0.05. "
            "PL est fragile (n=4 points seulement, p=0.095) : la tendance est suggestive mais non confirmee."
        )


# ===== TAB 3 : Q3 =====
with tabs[3]:
    question_banner("Q3 - Quelles conditions marquent le passage stage_2 -> stage_3 ?")

    dynamic_narrative(
        "Reponse courte : le passage vers stage_3 exige un FAR durablement eleve (>= 0.60) "
        "et une detente des heures negatives (tendance declinante), pas seulement un FAR ponctuellement bon.",
        severity="info",
    )

    if not df_q3.empty:
        st.caption("Scatter : FAR latest vs h_negative latest par pays (2024)")
        fig_q3 = px.scatter(
            df_q3,
            x="far_latest",
            y="h_negative_latest",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
            size="h_regime_a_latest",
            hover_data=["status_transition_2_to_3", "h_negative_slope_per_year", "sr_latest"],
            opacity=0.5,
        )
        fig_q3.add_vline(x=0.60, line_dash="dash", line_color="#94a3b8",
                         annotation_text="FAR=0.60", annotation_position="top left")
        fig_q3.update_layout(
            height=480,
            xaxis_title="FAR (Flex Absorption Ratio) - 2024",
            yaxis_title="Heures negatives observees - 2024",
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        fig_q3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q3, use_container_width=True)

        narrative(
            "**Statut 2024** : tous les pays sont en 'transition partielle'. "
            "Le FAR depasse 0.60 partout, mais les h_neg sont en hausse (pas en detente). "
            "La taille des bulles represente h_regime_a (surplus non absorbe par la flex domestique)."
        )

        st.markdown("**Detail par pays** :")
        for _, row in df_q3.iterrows():
            c = row["country"]
            ha = row.get("h_regime_a_latest", 0)
            ha_str = f"{int(ha)}" if pd.notna(ha) else "0"
            st.markdown(
                f"- **{c}** : FAR = {_sf(row.get('far_latest'), '.3f')} | "
                f"h_neg = {_sf(row.get('h_negative_latest'), '.0f')} | "
                f"h_regime_a = {ha_str} | "
                f"pente h_neg = {_sf(row.get('h_negative_slope_per_year'), '.1f')} h/an | "
                f"statut = {row.get('status_transition_2_to_3', 'n/a')}"
            )

        challenge_block(
            "h_neg en hausse partout",
            "La tendance h_negative_obs est en hausse dans tous les pays (pentes positives de +13 a +27 h/an). "
            "Cela signifie que la condition de detente des h_neg pour atteindre stage_3 n'est remplie nulle part. "
            "Le passage effectif a stage_3 n'est pas imminent.",
        )


# ===== TAB 4 : Q4 =====
with tabs[4]:
    question_banner("Q4 - Combien de batteries pour freiner la degradation ?")

    dynamic_narrative(
        "Reponse courte : le diagnostic de plateau teste si ajouter du BESS modifie les metriques structurelles. "
        "Un plateau signifie que le surplus est deja absorbe par la flex existante (PSH). "
        "L'ajout de BESS supplementaire n'a pas d'effet visible.",
        severity="info",
    )

    if not df_q4.empty:
        st.caption("Stress PV minimal pour identifier un effet BESS par pays (2024)")
        df_q4_sorted = df_q4.sort_values("stress_delta_pv_gw")
        fig_q4 = px.bar(
            df_q4_sorted,
            x="country",
            y="stress_delta_pv_gw",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
        )
        fig_q4.update_layout(
            height=420,
            xaxis_title="Pays",
            yaxis_title="Delta PV additionnel (GW) pour effet BESS",
            showlegend=False,
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        fig_q4.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q4.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q4, use_container_width=True)

        section_header("Diagnostic par pays")
        for _, row in df_q4.iterrows():
            c = row["country"]
            surplus = row.get("surplus_unabs_twh_baseline", 0)
            far_b = row.get("far_baseline", 0)
            ha_b = row.get("h_regime_a_baseline", 0)
            stress_gw = row.get("stress_delta_pv_gw", 0)
            plateau = row.get("plateau_baseline", True)

            if surplus > 1.0:
                interpretation = (
                    f"Surplus non absorbe = {_sf(surplus, '.2f')} TWh. "
                    f"La flex domestique ne couvre que {_sf(far_b, '.1%')} du surplus. "
                    f"Le BESS a un role structurel potentiel pour absorber les {int(ha_b)} heures en regime A."
                )
            elif surplus > 0.01:
                interpretation = (
                    f"Petit surplus residuel = {_sf(surplus, '.3f')} TWh ({int(ha_b)} heures en regime A). "
                    f"Le BESS a un role marginal en baseline."
                )
            else:
                interpretation = (
                    "Aucun surplus residuel en baseline (FAR = 1.0). "
                    "La flex domestique absorbe deja tout le surplus. "
                    f"Il faut ajouter {int(stress_gw)} GW de PV pour que le BESS devienne utile."
                )

            st.markdown(f"**{c}** : {interpretation}")

        challenge_block(
            "Plateau BESS = resultat physiquement normal",
            "Dans le sweep baseline, le FAR reste a 1.0 et h_regime_a a 0 "
            "quelle que soit la capacite BESS ajoutee (quand la flex existante suffit). "
            "Ce n'est pas un bug : si le surplus est deja absorbe, ajouter du BESS ne change rien. "
            "Il faut un stress PV additionnel pour creer du surplus residuel.",
        )


# ===== TAB 5 : Q5 =====
with tabs[5]:
    question_banner("Q5 - Quel est l'impact CO2/Gaz sur l'ancre thermique ?")

    dynamic_narrative(
        "Reponse courte : le TTL (ancre thermique longue duree) augmente significativement sous stress "
        "CO2 ou gaz. L'ampleur depend du mix electrique de chaque pays.",
        severity="info",
    )

    if not df_q5.empty:
        melt = df_q5.melt(
            id_vars=["country", "year"],
            value_vars=["delta_ttl_high_co2", "delta_ttl_high_gas"],
            var_name="scenario",
            value_name="delta_ttl",
        )
        melt["scenario"] = melt["scenario"].replace({
            "delta_ttl_high_co2": "CO2 eleve",
            "delta_ttl_high_gas": "Gaz eleve",
        })

        st.caption("Impact sur le TTL des scenarios CO2 eleve et gaz eleve par pays (2024)")
        fig_q5 = px.bar(
            melt,
            x="country",
            y="delta_ttl",
            color="scenario",
            barmode="group",
        )
        fig_q5.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig_q5.update_layout(
            height=420,
            xaxis_title="Pays",
            yaxis_title="Delta TTL (EUR/MWh)",
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        fig_q5.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q5.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q5, use_container_width=True)

        narrative(
            "**Sensibilite CO2** : PL est tres sensible au CO2 "
            f"(+{_sf(df_q5[df_q5['country'] == 'PL']['delta_ttl_high_co2'].iloc[0] if len(df_q5[df_q5['country'] == 'PL']) > 0 else 0, '.0f')} EUR/MWh) "
            "en raison de son mix charbon. FR est le moins sensible "
            f"(+{_sf(df_q5[df_q5['country'] == 'FR']['delta_ttl_high_co2'].iloc[0] if len(df_q5[df_q5['country'] == 'FR']) > 0 else 0, '.0f')} EUR/MWh) "
            "grace au nucleaire."
        )

        st.markdown("""
**Classement sensibilite CO2** : PL > ES > DK > DE > FR

**Classement sensibilite gaz** : ES > DE > DK > FR > PL
""")

        challenge_block(
            "Asymetrie FR/PL notable",
            "L'ecart de sensibilite CO2 entre PL (+86 EUR/MWh) et FR (+25 EUR/MWh) est un facteur 3.5x. "
            "Cela reflete la dependance au charbon de la Pologne vs le mix nucleaire de la France. "
            "Pour la Pologne, toute hausse du prix CO2 a un impact direct et massif sur le TTL.",
        )


# ===== TAB 6 : Q6 =====
with tabs[6]:
    question_banner("Q6 - Stockage chaleur/froid : synergie ou competition avec BESS ?")

    dynamic_narrative(
        "Reponse courte : le BESS est plus efficace en aller-retour court (2-8h), "
        "le thermique peut etre pertinent sur durees longues (12-24h). "
        "Les deux sont complementaires si les usages sont bien segmentes.",
        severity="info",
    )

    narrative(
        "Le perimetre actuel du modele ne contient pas de variable chaleur/froid dediee. "
        "Aucune conclusion causale n'est possible sans dataset specifique."
    )

    if not df_q6.empty:
        st.dataframe(df_q6, use_container_width=True, hide_index=True)

    narrative(
        "La page 6 (onglet Q6) propose neanmoins une comparaison indicative : "
        "le BESS (rendement aller-retour eta = 0.88) est plus efficient que le stockage thermique "
        "(eta = 0.50) en cycle court. En duree longue (>12h), le thermique pourrait devenir "
        "pertinent grace a des couts de capacite plus faibles, mais cette hypothese n'est pas "
        "validable avec les donnees actuelles."
    )

    challenge_block(
        "Perimetre limite",
        "Sans donnees de consommation chaleur/froid par pays, toute conclusion sur la synergie "
        "BESS/thermique reste qualitative. L'integration d'un dataset CHP/district heating "
        "permettrait de quantifier le potentiel de couplage sectoriel.",
    )


# ===== TAB 7 : CONCLUSIONS PAYS =====
with tabs[7]:
    section_header("Conclusions par pays", "Synthese des reponses Q1-Q6 pour chaque pays")

    # --- Comparative table ---
    if not df_country.empty:
        st.markdown("#### Tableau comparatif")
        for _, row in df_country.iterrows():
            c = row["country"]
            cols = st.columns([1, 1, 1, 1, 1])
            with cols[0]:
                render_kpi_banner("Pays", c, row.get("phase_latest", "?"), "medium")
            with cols[1]:
                far_v = row.get("far_latest", 0)
                render_kpi_banner("FAR", _sf(far_v, ".3f"), "",
                                  "strong" if far_v >= 0.95 else "medium" if far_v >= 0.80 else "weak")
            with cols[2]:
                cr = row.get("capture_ratio_pv_latest", 0)
                render_kpi_banner("CR_PV", _sf(cr, ".3f"), "",
                                  "strong" if cr >= 0.85 else "medium" if cr >= 0.70 else "weak")
            with cols[3]:
                slope = row.get("q2_slope", 0)
                render_kpi_banner("Pente Q2", _sf(slope, ".4f"), "",
                                  "weak" if abs(slope) > 0.05 else "medium")
            with cols[4]:
                d_co2 = row.get("q5_delta_ttl_co2", 0)
                render_kpi_banner("Delta CO2", f"+{_sf(d_co2, '.0f')} EUR", "",
                                  "weak" if d_co2 > 60 else "medium")
            st.markdown("---")

        # --- Country fiches ---
        section_header("Fiches pays detaillees")
        for _, row in df_country.iterrows():
            c = row["country"]
            with st.expander(f"**{c}** — {row.get('phase_latest', '?')} (2024)", expanded=False):
                fc = st.columns(3)
                with fc[0]:
                    render_kpi_banner(
                        "Phase 2024", row.get("phase_latest", "?"),
                        f"SR = {_sf(row.get('sr_latest'), '.4f')}",
                        "medium",
                    )
                with fc[1]:
                    far_v = row.get("far_latest", 0)
                    render_kpi_banner(
                        "FAR", _sf(far_v, ".3f"),
                        "Absorption domestique",
                        "strong" if far_v >= 0.95 else "medium" if far_v >= 0.80 else "weak",
                    )
                with fc[2]:
                    cr = row.get("capture_ratio_pv_latest", 0)
                    render_kpi_banner(
                        "Capture ratio PV", _sf(cr, ".3f"),
                        "Degradation vs baseload",
                        "strong" if cr >= 0.85 else "medium" if cr >= 0.70 else "weak",
                    )

                first_yr = row.get("q1_first_stage2_year")
                yr_str = f"{int(first_yr)}" if pd.notna(first_yr) else "jamais"

                # Q3 data for this country
                q3_row = df_q3[df_q3["country"] == c].iloc[0] if len(df_q3[df_q3["country"] == c]) > 0 else {}
                ha_latest = q3_row.get("h_regime_a_latest", 0) if isinstance(q3_row, dict) or hasattr(q3_row, "get") else (q3_row["h_regime_a_latest"] if "h_regime_a_latest" in q3_row.index else 0)

                # Q4 data for this country
                q4_row = df_q4[df_q4["country"] == c].iloc[0] if len(df_q4[df_q4["country"] == c]) > 0 else {}
                surplus_val = q4_row.get("surplus_unabs_twh_baseline", 0) if isinstance(q4_row, dict) or hasattr(q4_row, "get") else (q4_row["surplus_unabs_twh_baseline"] if "surplus_unabs_twh_baseline" in q4_row.index else 0)

                st.markdown(f"""
**Q1** : Premier franchissement simultane des 3 seuils stage_2 = **{yr_str}**.

**Q2** : Pente = **{_sf(row.get('q2_slope'), '.4f')}** par pp de penetration PV.{' (fragile, n=4, p>0.05)' if c == 'PL' else ' (significatif, p<=0.05)'}

**Q3** : Statut transition = **{row.get('q3_status', 'n/a')}**. FAR = {_sf(row.get('far_latest'), '.3f')}, h_regime_a = {int(ha_latest) if pd.notna(ha_latest) else 0}.

**Q4** : Surplus non absorbe = **{_sf(surplus_val, '.2f')} TWh**. Stress PV = {_sf(row.get('q4_stress_found', ''), '')} → delta = {_sf(df_q4[df_q4['country'] == c]['stress_delta_pv_gw'].iloc[0] if len(df_q4[df_q4['country'] == c]) > 0 else 0, '.0f')} GW.

**Q5** : Sensibilite CO2 = +{_sf(row.get('q5_delta_ttl_co2'), '.0f')} EUR/MWh | Sensibilite gaz = +{_sf(row.get('q5_delta_ttl_gas'), '.0f')} EUR/MWh.

**Q6** : {row.get('q6_status', 'n/a').replace('_', ' ')}.
""")


# ===== TAB 8 : ANNEXES =====
with tabs[8]:
    section_header("Annexes chiffrees", "Metriques completes 5 pays x 10 ans")

    if not df_annex.empty:
        st.dataframe(df_annex, use_container_width=True, hide_index=True)

    if not df_quality.empty:
        section_header("Qualite des donnees")
        st.dataframe(df_quality, use_container_width=True, hide_index=True)

        caveat_price = df_quality[df_quality.get("price_completeness", pd.Series(dtype=float)).lt(0.90)] if "price_completeness" in df_quality.columns else pd.DataFrame()
        if not caveat_price.empty:
            dynamic_narrative(
                "Completude prix < 90% detectee sur certaines paires (notamment PL 2015-2016).",
                severity="warning",
            )
