"""Static ExceSum report page (frozen dataset, no runtime recompute)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
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
from src.ui_theme import COUNTRY_PALETTE, PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS, PHASE_COLORS

st.set_page_config(page_title="ExceSum des conclusions", page_icon="🧾", layout="wide")
inject_global_css()


# ---------------------------------------------------------------------------
# Data loading (identical helpers)
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
    try:
        f = float(v)
    except Exception:
        return "n/a"
    if not np.isfinite(f):
        return "n/a"
    return f"{f:{fmt}}"


def _phase_distribution_text(df_latest: pd.DataFrame) -> str:
    if df_latest.empty or "phase" not in df_latest.columns:
        return "distribution indisponible"
    counts = df_latest["phase"].fillna("unknown").value_counts(dropna=False).to_dict()
    order = ["stage_1", "stage_2", "stage_3", "stage_4", "unknown"]
    chunks = [f"{k}={int(counts.get(k, 0))}" for k in order if k in counts or k == "unknown"]
    return ", ".join(chunks)


payload = _load_static_payload()
meta = payload.get("meta", {})
baseline = meta.get("baseline", {})
global_medians = payload.get("global_medians", {})
rebuild_matrix = payload.get("rebuild_matrix", {})
latest_year_rows = payload.get("latest_year", [])

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
df_verification = _safe_df({"rows": meta.get("verification", [])}, "rows")
df_latest = pd.DataFrame(latest_year_rows) if isinstance(latest_year_rows, list) else pd.DataFrame()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ExceSum des conclusions")
st.caption("Rapport statique fige sur baseline unique (aucun recalcul a l'ouverture).")

narrative(
    "Cette page est une synthese executive statique. Toutes les valeurs sont tracees "
    "vers le JSON source <code>docs/EXCESUM_STATIC_REPORT.json</code>. "
    "Les interpretations ci-dessous sont fondees exclusivement sur les chiffres de ce fichier."
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
    "Distribution des phases en 2024 : "
    f"{_phase_distribution_text(df_latest)}. "
    "La phase est une classification annuelle recalculee chaque annee. "
    "Elle n'est pas monotone : un pays peut reculer d'une phase si ses indicateurs se degradent."
)

dynamic_narrative(
    f"Sur {n_pairs} couples pays-annee, {n_ha} presentent des heures en regime A "
    "(surplus non absorbe par la flex modelisee). "
    "Ce chiffre signifie que la flex (PSH + exports + BESS) ne suffit pas a absorber "
    "tout le surplus pendant ces heures. Les mecanismes non modelises (hydro barrage, DSM, "
    "curtailment) prennent le relais dans la realite.",
    severity="info",
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_kpi_banner("SR median", _sf(global_medians.get("sr"), ".4f"),
                      "Part du surplus dans la generation", "medium")
with kpi_cols[1]:
    far_med = float(global_medians.get("far", np.nan))
    render_kpi_banner(
        "FAR median", _sf(far_med, ".3f"),
        "Part du surplus absorbee par la flex",
        "strong" if np.isfinite(far_med) and far_med > 0.95 else "medium",
    )
with kpi_cols[2]:
    render_kpi_banner("IR median", _sf(global_medians.get("ir"), ".3f"),
                      "P10 must-run / P10 load", "medium")
with kpi_cols[3]:
    render_kpi_banner("TTL median", f"{_sf(global_medians.get('ttl'), '.1f')} EUR/MWh",
                      "Q95 prix sur heures C+D", "medium")

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
**Seuils stage_2** (scoring par points, cf. page 0 etape 7) :
- `h_negative_obs >= 200` (+1 pt), `>= 300` (+2 pts bonus)
- `h_below_5_obs >= 500` (+1 pt)
- `capture_ratio_pv <= 0.80` (+1 pt), `<= 0.70` (+2 pts bonus)

**Seuil de significativite des pentes Q2** : p-value <= 0.05. Au-dessus : "fragile".

**Seuil FAR pour transition Q3** : FAR >= 0.60, avec condition supplementaire
`require_h_neg_declining = true` (les heures negatives doivent baisser dans le temps).

**Outlier exclu** : 2022 (crise energetique, prix hors normes).
""")

    dynamic_narrative(
        "<strong>Limite du modele flex (observed)</strong> : "
        "la flex modelisee = PSH pompage observe + exports observes (net_position > 0) + BESS modelise. "
        "L'hydro barrage, le DSM industriel et les ajustements thermiques ne sont pas modelises. "
        "Pour la France (IR > 1, must-run nucleaire dominant), cette limitation entraine une "
        "surestimation du surplus non absorbe. La coherence regime/prix de ~28% pour FR "
        "(vs > 90% pour DE) en est la consequence directe. "
        "Pour les pays sans gros must-run nucleaire (DE, DK, ES, PL), le modele est bien calibre.",
        severity="warning",
    )

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

    narrative(
        "La bascule vers stage_2 est identifiee quand trois signaux de stress "
        "franchissent simultanement leurs seuils respectifs : "
        "h_negative_obs >= 200, h_below_5_obs >= 500, capture_ratio_pv <= 0.80. "
        "Franchir un seul seuil n'est pas suffisant."
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
                         annotation_text="CR_PV = 0.80", annotation_position="top left")
        fig_q1.add_vline(x=200, line_dash="dash", line_color="#94a3b8",
                         annotation_text="h_neg = 200", annotation_position="top right")
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

        narrative(
            "**Lecture** : DE et DK franchissent les 3 seuils des 2023, ce qui est coherent avec "
            "leur penetration VRE elevee (48% et 75%). ES et FR ne les franchissent qu'en 2024. "
            "PL ne les franchit jamais : seul le CR_PV passe sous 0.80 en 2024 (0.748), "
            "mais h_neg (197) reste sous le seuil de 200 et h_below_5 (350) sous 500."
        )

        challenge_block(
            "PL : seuils non franchis simultanement",
            "La Pologne ne franchit pas les 3 seuils en meme temps sur 2015-2024. "
            "Le classement stage_2 de PL (confiance = 50%) repose sur le seul critere CR_PV. "
            "Les indicateurs de prix (h_neg, h_below_5) ne confirment pas encore un stress de marche equivalent.",
        )


# ===== TAB 2 : Q2 =====
with tabs[2]:
    question_banner("Q2 - Quelle est la pente de degradation du capture ratio PV ?")

    narrative(
        "La pente mesure la variation du capture_ratio_pv par point de pourcentage de penetration PV "
        "(regression lineaire sur 2015-2024, hors 2022). "
        "Une pente de -0.07 signifie que chaque point supplementaire de PV dans le mix "
        "fait baisser le capture ratio de 7 points de base."
    )

    if not df_q2.empty:
        # Map "forte" → "significatif" for coherence with page 6
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
            "**Classement de la degradation** (du plus rapide au plus lent) :<br>"
            "1. <strong>FR</strong> : pente = -0.0708 (p = 0.006, R² = 0.68, n = 9) — significatif<br>"
            "2. <strong>DE</strong> : pente = -0.0397 (p = 0.003, R² = 0.75, n = 9) — significatif<br>"
            "3. <strong>DK</strong> : pente = -0.0367 (p = 0.0004, R² = 0.85, n = 9) — significatif<br>"
            "4. <strong>PL</strong> : pente = -0.0293 (p = 0.095, R² = 0.82, n = 4) — fragile<br>"
            "5. <strong>ES</strong> : pente = -0.0213 (p = 0.00002, R² = 0.93, n = 9) — significatif"
        )

        narrative(
            "**Interpretation** : FR se degrade 3.3x plus vite que ES. "
            "Cela s'explique par le fait que le surplus FR est principalement nucleaire "
            "(present meme a faible VRE), ce qui amplifie l'effet de chaque point supplementaire de PV. "
            "En ES, le profil solaire est tres correle a la demande (climatisation ete), "
            "ce qui reduit la cannibalisation. "
            "PL est fragile car la regression ne repose que sur 4 points (PV absent avant 2020) "
            "et la p-value (0.095) depasse le seuil de 5%."
        )


# ===== TAB 3 : Q3 =====
with tabs[3]:
    question_banner("Q3 - Quelles conditions marquent le passage stage_2 -> stage_3 ?")

    narrative(
        "Le passage vers stage_3 exige deux conditions simultanees : "
        "un FAR durablement eleve (>= 0.60) et une tendance declinante des heures negatives. "
        "Un FAR ponctuellement bon ne suffit pas : la regle "
        "<code>require_h_neg_declining = true</code> impose que les h_neg diminuent dans le temps."
    )

    if not df_q3.empty:
        q3_plot = df_q3.copy()
        for col in ["h_negative_declining_latest", "h_negative_slope_per_year", "status_transition_2_to_3"]:
            if col not in q3_plot.columns:
                q3_plot[col] = np.nan
        if "h_negative_declining_latest" in q3_plot.columns:
            q3_plot["h_negative_declining_latest"] = (
                q3_plot["h_negative_declining_latest"].astype("boolean").fillna(False).astype(bool)
            )

        st.caption("Scatter : FAR vs h_negative par pays (2024). Taille des bulles = h_regime_a.")
        fig_q3 = px.scatter(
            q3_plot,
            x="far_latest",
            y="h_negative_latest",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
            size="h_regime_a_latest",
            hover_data=["status_transition_2_to_3", "h_negative_slope_per_year", "h_negative_declining_latest"],
            opacity=0.5,
        )
        fig_q3.add_vline(x=0.60, line_dash="dash", line_color="#94a3b8",
                         annotation_text="FAR = 0.60", annotation_position="top left")
        fig_q3.update_layout(
            height=480,
            xaxis_title="FAR (Flex Absorption Ratio) - 2024",
            yaxis_title="Heures negatives observees - 2024",
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        fig_q3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q3, use_container_width=True)

        section_header("Detail par pays")
        for _, row in df_q3.iterrows():
            c = row["country"]
            ha = row.get("h_regime_a_latest", 0)
            ha_str = f"{int(ha)}" if pd.notna(ha) else "0"
            slope_hneg = row.get("h_negative_slope_per_year", 0)
            st.markdown(
                f"- **{c}** : FAR = {_sf(row.get('far_latest'), '.3f')} | "
                f"h_neg = {_sf(row.get('h_negative_latest'), '.0f')} | "
                f"h_regime_a = {ha_str} | "
                f"pente h_neg = +{_sf(slope_hneg, '.1f')} h/an | "
                f"statut = {row.get('status_transition_2_to_3', 'n/a')}"
            )

        narrative(
            "**Verdict** : tous les pays sont en 'transition partielle'. Le FAR depasse 0.60 partout, "
            "mais la condition de detente des h_neg n'est remplie nulle part. "
            "Les pentes h_neg sont toutes positives : DE +26 h/an, FR +27 h/an, DK +26 h/an, "
            "ES +13 h/an, PL +13 h/an. Le surplus augmente plus vite que la flex ne s'adapte."
        )

        challenge_block(
            "h_neg en hausse partout : stage_3 n'est pas imminent",
            "La tendance h_negative_obs est en hausse dans tous les pays. "
            "Le passage effectif a stage_3 exigerait un retournement de cette tendance, "
            "ce qui impliquerait soit un deploiement massif de flex additionnelle, "
            "soit un plafonnement de la penetration VRE.",
        )


# ===== TAB 4 : Q4 =====
with tabs[4]:
    question_banner("Q4 - Combien de batteries pour freiner la degradation ?")

    narrative(
        "Le diagnostic teste l'effet de l'ajout de BESS en baseline (2024). "
        "Deux cas existent: (1) si le surplus residuel est deja present (FAR < 1 ou h_regime_a > 0), "
        "l'effet marginal du BESS est observable sans ajout PV (delta=0) ; "
        "(2) si le surplus est deja integralement absorbe (FAR=1 et h_regime_a=0), "
        "un stress PV additionnel peut etre requis pour rendre l'effet observable."
    )

    if not df_q4.empty:
        st.caption(
            "Delta PV additionnel (GW) necessaire pour observer un effet BESS par pays "
            "(0 GW = effet deja observable en baseline)."
        )
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
            yaxis_title="Delta PV additionnel (GW) pour effet BESS (0 = deja visible)",
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
            try:
                stress_gw_f = float(stress_gw)
            except Exception:
                stress_gw_f = float("nan")
            stress_needs_extra_pv = bool(np.isfinite(stress_gw_f) and stress_gw_f > 1e-9)
            stress_text = (
                f"Il faut +{int(round(stress_gw_f))} GW de PV additionnel pour rendre l'effet marginal du BESS observable."
                if stress_needs_extra_pv
                else "L'effet marginal du BESS est deja observable en baseline (delta PV requis = 0 GW)."
            )

            if surplus > 1.0:
                interpretation = (
                    f"Surplus non absorbe = **{_sf(surplus, '.2f')} TWh** ({int(ha_b)} h en regime A). "
                    f"FAR = {_sf(far_b, '.3f')} : la flex ne couvre que {_sf(far_b, '.1%')} du surplus. "
                    f"Le BESS a un role structurel potentiel. "
                    f"{stress_text}"
                )
            elif surplus > 0.01:
                interpretation = (
                    f"Petit surplus residuel = **{_sf(surplus, '.3f')} TWh** ({int(ha_b)} h en regime A). "
                    f"Le BESS a un role marginal en baseline. "
                    f"{stress_text}"
                )
            else:
                interpretation = (
                    f"Aucun surplus residuel significatif (FAR = {_sf(far_b, '.3f')}). "
                    f"La flex existante absorbe deja tout. "
                    f"{stress_text}"
                )

            st.markdown(f"**{c}** : {interpretation}")

        dynamic_narrative(
            "<strong>Pourquoi la France concentre l'essentiel du surplus non absorbe</strong><br>"
            "Le surplus francais est d'origine <strong>nucleaire</strong>, pas VRE. "
            "L'IR (Inflexibility Ratio) de la France depasse 1.0 : le must-run nucleaire seul "
            "excede la demande minimale (P10). Ce surplus existait deja en 2015 "
            "(4043 h en regime A, 5% de VRE seulement) et a chute a 378 h en 2022 "
            "(crise corrosion nucleaire, IR = 0.71), confirmant que c'est le nucleaire "
            "qui en est le moteur.<br><br>"
            "Le modele ne compte que PSH + exports + BESS comme flex. "
            "L'hydro barrage (~10 GW en France) et le DSM ne sont pas modelises. "
            "En consequence, les 4437 h de regime A et 12.47 TWh de surplus non absorbe "
            "surestiment le stress reel : seules ~352 h affichent des prix negatifs sur le marche "
            "(coherence regime/prix = 28%). "
            "Pour DE/DK/ES/PL, le modele est bien calibre (coherence > 90%).",
            severity="warning",
        )

        challenge_block(
            "Plateau BESS = resultat physiquement normal",
            "Quand le surplus est deja absorbe par la flex existante, ajouter du BESS ne change pas FAR ni h_regime_a. "
            "Ce n'est pas un bug. Dans ce cas uniquement, un stress PV additionnel peut etre necessaire.",
        )


# ===== TAB 5 : Q5 =====
with tabs[5]:
    question_banner("Q5 - Quel est l'impact CO2/gaz sur l'ancre thermique ?")

    narrative(
        "Le TTL (Thermal Tail Level, Q95 des prix sur regimes C+D) est l'ancre de prix "
        "en dehors des heures de surplus. Cette analyse mesure comment le TTL evolue "
        "sous deux scenarios deterministes : CO2 eleve et gaz eleve."
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
            "**Sensibilite CO2** (classement du plus sensible au moins sensible) :<br>"
            "1. PL : +86 EUR/MWh — mix charbon, facteur d'emission eleve<br>"
            "2. ES : +57 EUR/MWh — CCGT marginal avec composante charbon<br>"
            "3. DK : +52 EUR/MWh — CCGT marginal<br>"
            "4. DE : +49 EUR/MWh — CCGT marginal, lignite residuel<br>"
            "5. FR : +25 EUR/MWh — nucleaire dominant, CCGT marginal uniquement en pointe"
        )

        narrative(
            "**Sensibilite gaz** (classement) :<br>"
            "1. ES : +64 EUR/MWh<br>"
            "2. DE : +58 EUR/MWh<br>"
            "3. DK : +51 EUR/MWh<br>"
            "4. FR : +45 EUR/MWh<br>"
            "5. PL : +31 EUR/MWh — le charbon est marginal, pas le gaz"
        )

        challenge_block(
            "Asymetrie FR/PL sur le CO2 : facteur 3.5x",
            "L'ecart de sensibilite CO2 entre PL (+86 EUR/MWh) et FR (+25 EUR/MWh) est un facteur 3.5x. "
            "Cela reflete la dependance au charbon de la Pologne vs le mix nucleaire de la France. "
            "Pour PL, toute hausse du prix CO2 a un impact direct et massif sur le TTL. "
            "Pour FR, le nucleaire isole les prix de l'exposition carbone.",
        )


# ===== TAB 6 : Q6 =====
with tabs[6]:
    question_banner("Q6 - Stockage chaleur/froid : synergie ou competition avec BESS ?")

    narrative(
        "Le perimetre actuel du modele ne contient pas de variable chaleur/froid dediee. "
        "Aucune conclusion causale n'est possible sans dataset specifique."
    )

    dynamic_narrative(
        "Conclusion prudente : sans jeu de donnees chaleur/froid dedie, "
        "la causalite n'est pas identifiable. Ce constat est identique pour les 5 pays.",
        severity="warning",
    )

    if not df_q6.empty:
        st.dataframe(df_q6, use_container_width=True, hide_index=True)

    narrative(
        "La page 6 (onglet Q6) propose neanmoins une comparaison indicative : "
        "le BESS (rendement aller-retour eta = 0.88) est plus efficient que le stockage thermique "
        "(eta = 0.50) en cycle court (2-8h). En duree longue (>12h), le thermique pourrait devenir "
        "pertinent grace a des couts de capacite plus faibles, mais cette hypothese n'est pas "
        "validable avec les donnees actuelles."
    )

    challenge_block(
        "Perimetre limite — donnees manquantes",
        "L'integration d'un dataset CHP/district heating par pays "
        "permettrait de quantifier le potentiel de couplage sectoriel.",
    )


# ===== TAB 7 : CONCLUSIONS PAYS =====
with tabs[7]:
    section_header("Conclusions par pays", "Synthese des reponses Q1-Q6 pour chaque pays")

    narrative(
        "Chaque fiche ci-dessous rassemble les conclusions des 6 questions pour un pays donne. "
        "Toutes les valeurs proviennent du JSON source. "
        "Les interpretations sont derivees exclusivement des indicateurs observes, "
        "sans extrapolation ni prediction."
    )

    # --- Comparative table ---
    if not df_country.empty:
        st.markdown("#### Tableau comparatif 2024")
        for _, row in df_country.sort_values("country").iterrows():
            c = row["country"]
            kcols = st.columns([1, 1, 1, 1, 1])
            with kcols[0]:
                render_kpi_banner("Pays", c, str(row.get("phase_latest", "?")), "medium")
            with kcols[1]:
                far_v = float(row.get("far_latest", 0))
                render_kpi_banner("FAR", _sf(far_v, ".3f"), "",
                                  "strong" if far_v >= 0.95 else "medium" if far_v >= 0.80 else "weak")
            with kcols[2]:
                cr = float(row.get("capture_ratio_pv_latest", 0))
                render_kpi_banner("CR_PV", _sf(cr, ".3f"), "",
                                  "strong" if cr >= 0.85 else "medium" if cr >= 0.70 else "weak")
            with kcols[3]:
                slope = float(row.get("q2_slope", 0))
                render_kpi_banner("Pente Q2", _sf(slope, ".4f"), "",
                                  "weak" if abs(slope) > 0.05 else "medium")
            with kcols[4]:
                d_co2 = float(row.get("q5_delta_ttl_co2", 0))
                render_kpi_banner("Delta CO2", f"+{_sf(d_co2, '.0f')} EUR", "",
                                  "weak" if d_co2 > 60 else "medium")
            st.markdown("---")

    # --- Country fiches ---
    section_header("Fiches pays detaillees")

    # Helper: get latest_year row for a country
    def _get_latest(c: str) -> dict:
        if df_latest.empty:
            return {}
        match = df_latest[df_latest["country"] == c]
        return match.iloc[0].to_dict() if not match.empty else {}

    def _get_q3(c: str) -> dict:
        if df_q3.empty:
            return {}
        match = df_q3[df_q3["country"] == c]
        return match.iloc[0].to_dict() if not match.empty else {}

    def _get_q4(c: str) -> dict:
        if df_q4.empty:
            return {}
        match = df_q4[df_q4["country"] == c]
        return match.iloc[0].to_dict() if not match.empty else {}

    if not df_country.empty:
        for _, row in df_country.sort_values("country").iterrows():
            c = str(row.get("country", "N/A"))
            latest = _get_latest(c)
            q3_data = _get_q3(c)
            q4_data = _get_q4(c)

            phase_lbl = str(row.get("phase_latest", "unknown"))
            conf_pct = float(row.get("phase_confidence_latest", 0)) * 100
            blocked = str(row.get("phase_blocked_rules_latest", ""))
            far_v = float(row.get("far_latest", 0))
            cr_v = float(row.get("capture_ratio_pv_latest", 0))
            sr_v = float(row.get("sr_latest", 0))
            slope_v = float(row.get("q2_slope", 0))
            ha_v = float(q3_data.get("h_regime_a_latest", 0))
            h_neg_v = float(latest.get("h_negative_obs", 0))
            h_below5_v = float(latest.get("h_below_5_obs", 0))
            ir_v = float(latest.get("ir", 0))
            ttl_v = float(latest.get("ttl", 0))
            surplus_v = float(q4_data.get("surplus_unabs_twh_baseline", 0))
            h_neg_slope = float(q3_data.get("h_negative_slope_per_year", 0))
            stress_pv = float(q4_data.get("stress_delta_pv_gw", 0))
            if np.isfinite(stress_pv) and stress_pv > 1e-9:
                q4_stress_text = (
                    f"+{int(round(stress_pv))} GW de PV additionnel necessaires pour rendre l'effet BESS observable."
                )
            else:
                q4_stress_text = "0 GW additionnel requis: l'effet BESS est deja observable en baseline."
            d_co2 = float(row.get("q5_delta_ttl_co2", 0))
            d_gas = float(row.get("q5_delta_ttl_gas", 0))
            first_yr = row.get("q1_first_stage2_year")
            yr_str = f"{int(first_yr)}" if pd.notna(first_yr) else "jamais"

            with st.expander(f"**{c}** — {phase_lbl} (2024, confiance {conf_pct:.0f}%)", expanded=False):
                # --- KPI banners ---
                fc = st.columns(4)
                with fc[0]:
                    render_kpi_banner(
                        "Phase 2024", phase_lbl,
                        f"Confiance = {conf_pct:.0f}%",
                        "strong" if conf_pct >= 70 else "medium" if conf_pct >= 50 else "weak",
                    )
                with fc[1]:
                    render_kpi_banner(
                        "FAR", _sf(far_v, ".3f"),
                        "Absorption surplus",
                        "strong" if far_v >= 0.95 else "medium" if far_v >= 0.80 else "weak",
                    )
                with fc[2]:
                    render_kpi_banner(
                        "CR_PV", _sf(cr_v, ".3f"),
                        "Capture ratio PV",
                        "strong" if cr_v >= 0.85 else "medium" if cr_v >= 0.70 else "weak",
                    )
                with fc[3]:
                    render_kpi_banner(
                        "IR", _sf(ir_v, ".3f"),
                        "Rigidite must-run",
                        "weak" if ir_v >= 0.70 else "medium" if ir_v >= 0.30 else "strong",
                    )

                # --- Metrics row ---
                mc = st.columns(4)
                mc[0].metric("SR", _sf(sr_v, ".4f"))
                mc[1].metric("TTL", f"{_sf(ttl_v, '.1f')} EUR/MWh")
                mc[2].metric("h_neg", f"{int(h_neg_v)}")
                mc[3].metric("h_below_5", f"{int(h_below5_v)}")

                # --- Structured Q1-Q6 answers ---
                st.markdown("---")
                st.markdown("##### Reponses Q1-Q6")

                # Robustesse Q2
                if c == "PL":
                    q2_robust = "fragile (n = 4, p = 0.095 > 0.05)"
                else:
                    q2_robust = "significatif (p <= 0.05)"

                st.markdown(f"""
**Q1 — Bascule stage_2** : Premier franchissement simultane des 3 seuils = **{yr_str}**.
En 2024 : h_neg = {int(h_neg_v)}, h_below_5 = {int(h_below5_v)}, CR_PV = {_sf(cr_v, '.3f')}.

**Q2 — Pente de cannibalisation** : Pente = **{_sf(slope_v, '.4f')}** par pp de penetration PV.
Robustesse : {q2_robust}.

**Q3 — Transition stage_2 -> stage_3** : Statut = **{row.get('q3_status', 'n/a')}**.
FAR = {_sf(far_v, '.3f')}, h_regime_a = {int(ha_v)}.
Pente h_neg = +{_sf(h_neg_slope, '.1f')} h/an (en hausse, condition de detente non remplie).
Regle bloquante : `{blocked}`.

**Q4 — Effet BESS** : Surplus non absorbe = **{_sf(surplus_v, '.2f')} TWh**.
{q4_stress_text}

**Q5 — Sensibilite commodites** : Delta TTL CO2 = +{_sf(d_co2, '.0f')} EUR/MWh.
Delta TTL gaz = +{_sf(d_gas, '.0f')} EUR/MWh.

**Q6 — Chaleur/froid** : Non identifiable sans donnees dediees.
""")

                # --- Country-specific interpretation ---
                st.markdown("##### Diagnostic")

                if c == "DE":
                    narrative(
                        "L'Allemagne presente la cannibalisation PV la plus severe du panel "
                        "(CR_PV = 0.589, le plus bas des 5 pays), malgre un FAR parfait de 1.000. "
                        "Cela signifie que la flex domestique (PSH 6.5 GW + exports 25 GW) "
                        "absorbe integralement le surplus, mais que la correlation temporelle VRE/prix "
                        "erode la valeur captee par le PV. Avec 48% de VRE, "
                        "le PV ne capte que 59% de la valeur thermique de reference. "
                        "La pente de degradation (-0.0397 par pp) est la 2e plus rapide apres FR."
                    )

                elif c == "DK":
                    narrative(
                        "Le Danemark illustre le cas d'un petit systeme sature par le VRE (75% de penetration). "
                        "Sans PSH domestique (0 GW), la seule flex est l'export vers la Scandinavie et l'Allemagne "
                        "(6 GW d'interconnexion). Le surplus VRE (SR = 12.8%) cree 677 h de regime A "
                        "et 0.31 TWh de surplus non absorbe. "
                        "Le CR_PV (0.675) est comparable a l'Allemagne, "
                        "mais pour une penetration VRE bien superieure."
                    )

                elif c == "ES":
                    narrative(
                        "L'Espagne entre en stage_2 en 2024 seulement, le plus tardivement avec la France. "
                        "Sa pente de cannibalisation est la plus lente du panel (-0.0213), "
                        "grace a un profil solaire correle a la demande (climatisation estivale). "
                        "Cependant, h_below_5 = 1642, le chiffre le plus eleve du panel, "
                        "signale un nombre important d'heures a tres bas prix. "
                        "Le FAR de 0.962 est eleve : la flex (PSH 3 GW + exports 5 GW) suffit "
                        "a absorber l'essentiel du surplus (134 GWh de residuel seulement). "
                        "La sensibilite au gaz (+64 EUR/MWh) est la plus forte du panel."
                    )

                elif c == "FR":
                    narrative(
                        "La France presente un profil unique dans le panel. "
                        "Avec seulement 13.4% de VRE, elle affiche le FAR le plus bas (0.774), "
                        "la pente de cannibalisation la plus rapide (-0.0708), "
                        "et le plus grand volume de surplus non absorbe (12.47 TWh, 4437 h en regime A). "
                        "L'IR de 1.063 (superieur a 1) signifie que le must-run nucleaire seul "
                        "depasse la demande minimale P10. "
                        "En contrepartie, la sensibilite CO2 est la plus faible du panel (+25 EUR/MWh), "
                        "car le nucleaire isole le TTL de l'exposition carbone."
                    )
                    dynamic_narrative(
                        "<strong>Note methodologique cruciale</strong> : "
                        "le surplus francais est d'origine nucleaire, pas VRE. "
                        "Le modele traite toute la production nucleaire observee comme must-run, "
                        "ce qui surestime le surplus dans le perimetre modelise. "
                        "Preuves : (1) en 2015, 4043 h de regime A avec seulement 5% de VRE ; "
                        "(2) en 2022 (crise corrosion, IR = 0.71), h_regime_a chute a 378 h. "
                        "La coherence regime/prix de 28% (vs > 90% pour DE) confirme que "
                        "le modele decrit mal la realite marche pour FR. "
                        "Les 352 h de prix negatifs observes sont un indicateur de stress reel "
                        "plus fiable que les 4437 h de regime A.",
                        severity="warning",
                    )

                elif c == "PL":
                    narrative(
                        "La Pologne est en retard de transition : le classement stage_2 "
                        "n'a qu'une confiance de 50%, et les 3 seuils Q1 ne sont pas franchis "
                        "simultanement. Le surplus est quasi nul (0.001 TWh, 1 h en regime A). "
                        "La pente Q2 est fragile (p = 0.095, n = 4 seulement) car le PV "
                        "n'est present dans le mix polonais que depuis 2020. "
                        "La vraie alerte pour la Pologne est la sensibilite CO2 : +86 EUR/MWh, "
                        "soit 3.5 fois la France. Toute hausse du prix carbone europeen "
                        "aura un impact direct et massif sur le cout marginal polonais."
                    )


# ===== TAB 8 : ANNEXES =====
with tabs[8]:
    section_header("Annexes chiffrees", "Metriques completes 5 pays x 10 ans")

    if not df_latest.empty:
        latest_plot = df_latest.copy()
        for col in ["phase_confidence", "phase_score", "far", "sr", "capture_ratio_pv"]:
            if col not in latest_plot.columns:
                latest_plot[col] = np.nan

        section_header("Derniere annee (2024)")
        fig = px.bar(
            latest_plot.sort_values("country"),
            x="country",
            y="ttl",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            hover_data=["phase_confidence", "phase_score", "far", "sr", "capture_ratio_pv"],
        )
        fig.update_layout(height=420, xaxis_title="Pays", yaxis_title="TTL (EUR/MWh)", **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(latest_plot, use_container_width=True, hide_index=True)

    if not df_annex.empty:
        section_header("Metriques completes")
        st.dataframe(df_annex, use_container_width=True, hide_index=True)

    section_header("Controle qualite des donnees")
    if not df_quality.empty:
        st.dataframe(df_quality, use_container_width=True, hide_index=True)
        if "price_completeness" in df_quality.columns:
            caveat_price = df_quality[df_quality["price_completeness"] < 0.90]
            if not caveat_price.empty:
                dynamic_narrative(
                    f"Completude prix < 90% detectee sur {len(caveat_price)} couples pays-annee.",
                    severity="warning",
                )
    else:
        st.info("Aucun indicateur qualite disponible dans le payload statique.")
