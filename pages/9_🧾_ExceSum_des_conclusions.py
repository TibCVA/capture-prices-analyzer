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

st.title("ExceSum des conclusions")
st.caption("Rapport statique figé sur baseline unique (aucun recalcul à l'ouverture).")

narrative(
    "Cette page synthétise les résultats d'une baseline fixe et traçable. "
    "Toutes les conclusions ci-dessous sont générées depuis les tableaux du fichier "
    "<code>docs/EXCESUM_STATIC_REPORT.json</code>."
)

# KPI header
cols = st.columns(5)
cols[0].metric("Pays", len(baseline.get("countries", [])))
years = baseline.get("years", [2015, 2024])
cols[1].metric("Période", f"{years[0]}-{years[-1]}")
n_pairs = int(rebuild_matrix.get("pairs_total", len(df_annex)))
cols[2].metric("Couples pays-année", n_pairs)
n_ha = int(rebuild_matrix.get("pairs_h_regime_a_gt_0", 0))
cols[3].metric("h_regime_a > 0", f"{n_ha}/{n_pairs}")
cols[4].metric("Cohérence régime/prix", f"{100 * global_medians.get('regime_coherence', 0):.1f}%")

section_header("Synthèse globale", "Médianes panel 5 pays x 10 ans (2015-2024)")
narrative(
    "Distribution des phases en 2024: "
    f"{_phase_distribution_text(df_latest)}. "
    "La phase est une classification annuelle, non monotone: un pays peut évoluer "
    "d'une année à l'autre selon FAR, heures négatives, capture ratio et spreads."
)
dynamic_narrative(
    "Important: la narration ExceSum est alignée automatiquement sur les tables source. "
    "Aucun texte de phase n'est hardcodé.",
    severity="info",
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_kpi_banner("SR médian", _sf(global_medians.get("sr"), ".4f"), "Surplus structurel", "medium")
with kpi_cols[1]:
    far_med = float(global_medians.get("far", np.nan))
    render_kpi_banner(
        "FAR médian",
        _sf(far_med, ".3f"),
        "Absorption du surplus",
        "strong" if np.isfinite(far_med) and far_med > 0.95 else "medium",
    )
with kpi_cols[2]:
    render_kpi_banner("IR médian", _sf(global_medians.get("ir"), ".3f"), "Rigidité", "medium")
with kpi_cols[3]:
    render_kpi_banner("TTL médian", f"{_sf(global_medians.get('ttl'), '.1f')} EUR/MWh", "Queue haute", "medium")

tabs = st.tabs(
    [
        "Méthode",
        "Q1 - Seuils stage_2",
        "Q2 - Pentes CR_PV",
        "Q3 - Transition 2→3",
        "Q4 - Batteries",
        "Q5 - Commodités",
        "Q6 - Chaleur/froid",
        "Conclusions pays",
        "Annexes",
    ]
)

with tabs[0]:
    section_header("Conventions méthodologiques")
    narrative(
        "Baseline unique: FR/DE/ES/PL/DK, période 2015-2024, modes observed/observed/observed. "
        "L'année 2022 est exclue des pentes Q2."
    )
    st.markdown(
        """
**Règles de phase (extrait)**:
- Stage 2: seuils sur `h_negative_obs`, `h_below_5_obs`, `capture_ratio_pv`, `days_spread_above_50_obs`
- Stage 3: FAR élevé + condition `require_h_neg_declining=true`
- Stage 4: FAR élevé + plafond sur `h_regime_c`

**Lecture correcte des phases**:
- La phase est annuelle.
- Elle n'est pas un état irréversible.
- ExceSum et les pages UI doivent partager exactement la même logique.

**Modèle flex (observed)**:
- `sink_non_bess = PSH_pumping + net_position positive (exports)`
- Le modèle actuel inclut donc les exports dans la flex observée.
"""
    )
    section_header("Contrôle qualité")
    if not df_verification.empty:
        st.dataframe(df_verification, use_container_width=True, hide_index=True)
    if rebuild_matrix:
        st.markdown(
            f"""
**Rebuild matrix**: {rebuild_matrix.get('pairs_total', '?')} couples,
{rebuild_matrix.get('pairs_h_regime_a_gt_0', 0)} avec `h_regime_a > 0`,
{rebuild_matrix.get('cache_semantic_invalid_pairs', 0)} caches invalides corrigés.
"""
        )

with tabs[1]:
    question_banner("Q1 - À quels niveaux observe-t-on la bascule vers stage_2 ?")
    dynamic_narrative(
        "Réponse courte: la bascule vers stage_2 apparaît quand les trois signaux "
        "(`h_negative_obs`, `h_below_5_obs`, `capture_ratio_pv`) franchissent les seuils.",
        severity="info",
    )
    if not df_q1_detail.empty:
        fig = px.scatter(
            df_q1_detail,
            x="h_negative_obs",
            y="capture_ratio_pv",
            color="country",
            hover_data=["year", "h_below_5_obs", "sr"],
            color_discrete_map=COUNTRY_PALETTE,
            opacity=0.55,
        )
        fig.add_hline(y=0.80, line_dash="dash", line_color="#94a3b8")
        fig.add_vline(x=200, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(height=470, **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
    if not df_q1_country.empty:
        st.dataframe(df_q1_country, use_container_width=True, hide_index=True)

with tabs[2]:
    question_banner("Q2 - Quelle est la pente de capture ratio PV vs pénétration PV ?")
    dynamic_narrative(
        "Réponse courte: la pente est estimée par régression linéaire (hors 2022). "
        "Le signe et la p-value indiquent direction et robustesse.",
        severity="info",
    )
    if not df_q2.empty:
        fig = px.bar(
            df_q2.sort_values("slope"),
            x="country",
            y="slope",
            color="robustesse",
            hover_data=["r_squared", "p_value", "n_points"],
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig.update_layout(height=430, **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_q2, use_container_width=True, hide_index=True)

with tabs[3]:
    question_banner("Q3 - Conditions de transition stage_2 → stage_3")
    dynamic_narrative(
        "Deux lectures sont affichées séparément: "
        "1) phase annuelle officielle, 2) statut de transition 2→3.",
        severity="info",
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

        fig = px.scatter(
            q3_plot,
            x="far_latest",
            y="h_negative_latest",
            color="country",
            size="h_regime_a_latest",
            hover_data=["status_transition_2_to_3", "h_negative_slope_per_year", "h_negative_declining_latest"],
            color_discrete_map=COUNTRY_PALETTE,
            opacity=0.6,
        )
        fig.add_vline(x=0.60, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(height=470, **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(q3_plot, use_container_width=True, hide_index=True)
    challenge_block(
        "Pourquoi une phase peut reculer d'une année à l'autre ?",
        "Parce que la phase est recalculée chaque année. Si les règles actives changent "
        "(ex: FAR ou condition de baisse des heures négatives), la phase peut évoluer dans les deux sens.",
    )

with tabs[4]:
    question_banner("Q4 - Effet des batteries")
    dynamic_narrative(
        "Le résultat peut être plat si la flex existante absorbe déjà le surplus. "
        "Dans ce cas, l'effet BESS n'est visible qu'après stress de référence.",
        severity="info",
    )
    if not df_q4.empty:
        fig = px.bar(
            df_q4.sort_values("country"),
            x="country",
            y="stress_delta_pv_gw",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
        )
        fig.update_layout(height=420, showlegend=False, **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_q4, use_container_width=True, hide_index=True)

with tabs[5]:
    question_banner("Q5 - Impact CO2/gaz sur TTL")
    dynamic_narrative(
        "Le test compare le TTL baseline à des scénarios déterministes CO2 élevé et gaz élevé.",
        severity="info",
    )
    if not df_q5.empty:
        melt = df_q5.melt(
            id_vars=["country", "year"],
            value_vars=["delta_ttl_high_co2", "delta_ttl_high_gas"],
            var_name="scenario",
            value_name="delta_ttl",
        )
        fig = px.bar(melt, x="country", y="delta_ttl", color="scenario", barmode="group")
        fig.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig.update_layout(height=430, **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_q5, use_container_width=True, hide_index=True)

with tabs[6]:
    question_banner("Q6 - Stockage chaleur/froid")
    dynamic_narrative(
        "Conclusion prudente: sans jeu de données chaleur/froid dédié, la causalité n'est pas identifiable.",
        severity="warning",
    )
    if not df_q6.empty:
        st.dataframe(df_q6, use_container_width=True, hide_index=True)

with tabs[7]:
    section_header("Conclusions par pays")
    if not df_country.empty:
        st.dataframe(df_country, use_container_width=True, hide_index=True)
        for _, row in df_country.sort_values("country").iterrows():
            country = str(row.get("country", "N/A"))
            with st.expander(f"{country} — phase {row.get('phase_latest', 'unknown')} ({int(row.get('latest_year', 0))})"):
                cols = st.columns(4)
                cols[0].metric("SR", _sf(row.get("sr_latest"), ".4f"))
                cols[1].metric("FAR", _sf(row.get("far_latest"), ".3f"))
                cols[2].metric("CR_PV", _sf(row.get("capture_ratio_pv_latest"), ".3f"))
                cols[3].metric("Q3 transition", str(row.get("q3_status", "n/a")))
                st.markdown(
                    f"- Phase officielle (annuelle): `{row.get('phase_latest', 'unknown')}`\n"
                    f"- Statut transition 2→3: `{row.get('q3_status', 'n/a')}`\n"
                    f"- Sensibilité TTL CO2: `{_sf(row.get('q5_delta_ttl_co2'), '.2f')}` EUR/MWh\n"
                    f"- Sensibilité TTL gaz: `{_sf(row.get('q5_delta_ttl_gas'), '.2f')}` EUR/MWh"
                )

with tabs[8]:
    section_header("Annexes chiffrées")
    if not df_latest.empty:
        latest_plot = df_latest.copy()
        for col in ["phase_confidence", "phase_score", "far", "sr", "capture_ratio_pv"]:
            if col not in latest_plot.columns:
                latest_plot[col] = np.nan

        section_header("Dernière année (cohérence narrative)")
        fig = px.bar(
            latest_plot.sort_values("country"),
            x="country",
            y="ttl",
            color="phase",
            color_discrete_map=PHASE_COLORS,
            hover_data=["phase_confidence", "phase_score", "far", "sr", "capture_ratio_pv"],
        )
        fig.update_layout(height=420, **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(latest_plot, use_container_width=True, hide_index=True)

    if not df_annex.empty:
        section_header("Métriques complètes")
        st.dataframe(df_annex, use_container_width=True, hide_index=True)

    section_header("Controle qualite des donnees")
    if not df_quality.empty:
        st.dataframe(df_quality, use_container_width=True, hide_index=True)
        if "price_completeness" in df_quality.columns:
            caveat_price = df_quality[df_quality["price_completeness"] < 0.90]
            if not caveat_price.empty:
                dynamic_narrative(
                    f"Completude prix < 90% détectée sur {len(caveat_price)} couples pays-année.",
                    severity="warning",
                )
    else:
        st.info("Aucun indicateur qualité disponible dans le payload statique.")
