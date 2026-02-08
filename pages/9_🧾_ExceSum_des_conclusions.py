"""Page 9 - ExceSum des conclusions (baseline tool-first, mirror UI/UX)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.commentary_bridge import so_what_block
from src.excesum_engine import run_excesum_baseline, write_excesum_docs
from src.excesum_narrative import (
    country_conclusion_markdown,
    executive_summary,
    q1_text,
    q2_text,
    q3_text,
    q4_text,
    q5_text,
    q6_text,
)
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


def _status_to_banner(status: str) -> str:
    return {
        "PASS": "strong",
        "WARN": "medium",
        "FAIL": "weak",
    }.get(str(status).upper(), "unknown")


def _find_docx_path() -> str | None:
    direct = Path("Question S Michel") / "Synthèse travail à faire.docx"
    if direct.exists():
        return str(direct)
    candidates = sorted(Path("Question S Michel").glob("*.docx"))
    if candidates:
        return str(candidates[0])
    return None


@st.cache_data(show_spinner=True)
def _load_excesum(docx_path: str | None) -> dict:
    return run_excesum_baseline(objectives_docx_path=docx_path)


st.title("🧾 ExceSum des conclusions")
st.caption("Baseline figee: observed / observed / observed, FR-DE-ES-PL-DK, periode 2015-2024")

docx_path = _find_docx_path()
if "excesum_results" not in st.session_state:
    st.session_state["excesum_results"] = None

is_test_mode = "PYTEST_CURRENT_TEST" in os.environ

col_run, col_reset = st.columns([2, 1])
with col_run:
    run_clicked = st.button("Calculer / Recalculer ExceSum", type="primary", use_container_width=True)
with col_reset:
    reset_clicked = st.button("Effacer cache ExceSum", use_container_width=True)

if reset_clicked:
    st.session_state["excesum_results"] = None

if run_clicked or (not is_test_mode and st.session_state["excesum_results"] is None):
    st.session_state["excesum_results"] = _load_excesum(docx_path)

results = st.session_state["excesum_results"]
if not isinstance(results, dict):
    st.info("Cliquez sur `Calculer / Recalculer ExceSum` pour lancer la synthese complete.")
    st.stop()
metrics_df = results.get("metrics_df", pd.DataFrame())

if metrics_df.empty:
    st.error("ExceSum indisponible: aucune donnee baseline chargee.")
    for issue in results.get("issues", []):
        st.caption(f"- {issue}")
    st.stop()

narrative(executive_summary(results))

tabs = st.tabs(
    [
        "Résumé exécutif",
        "Méthode & Vérifications x3",
        "Q1-Q2",
        "Q3-Q4",
        "Q5-Q6",
        "Conclusions par pays",
        "Annexes chiffrées",
    ]
)

# ------------------------------------------------------------------
# Tab 1
# ------------------------------------------------------------------
with tabs[0]:
    section_header("Vue d'ensemble", "Synthese des resultats robustes sur 5 pays x 10 ans")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pays couverts", int(metrics_df["country"].nunique()))
    c2.metric("Annees", f"{int(metrics_df['year'].min())}-{int(metrics_df['year'].max())}")
    c3.metric("Couples pays/annee", int(len(metrics_df)))
    c4.metric("Coherence mediane", f"{float(metrics_df['regime_coherence'].median()) * 100.0:.1f}%")

    trends = (
        metrics_df.groupby("year", as_index=False)[["sr", "far", "capture_ratio_pv", "ttl"]]
        .median()
        .sort_values("year")
    )
    fig = px.line(
        trends,
        x="year",
        y=["sr", "far", "capture_ratio_pv"],
        markers=True,
        title="Tendances medianes des ratios structurels",
    )
    fig.update_layout(height=430, yaxis_title="Valeur", **PLOTLY_LAYOUT_DEFAULTS)
    fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)

    render_commentary(
        so_what_block(
            title="Resume executif - tendances globales",
            purpose="Resumer la trajectoire structurelle sans sur-interpreter les ecarts annuels.",
            observed={
                "sr_median": float(metrics_df["sr"].median()),
                "far_median": float(metrics_df["far"].median()),
                "capture_ratio_pv_median": float(metrics_df["capture_ratio_pv"].median()),
                "ttl_median": float(metrics_df["ttl"].median()),
            },
            method_link="Baseline fixe FR/DE/ES/PL/DK (2015-2024), metriques v3 harmonisees.",
            limits="Mediane multi-pays: la priorisation operative doit rester pays-specifique.",
            n=len(metrics_df),
            decision_use="Identifier les points d'attention globaux avant lecture detaillee Q1..Q6.",
        )
    )

    q_txt_cols = st.columns(3)
    q_txt_cols[0].info(q1_text(results.get("q1_country", pd.DataFrame())))
    q_txt_cols[1].info(q2_text(results.get("q2_slopes", pd.DataFrame())))
    q_txt_cols[2].info(q3_text(results.get("q3_transition", pd.DataFrame())))
    q_txt_cols = st.columns(3)
    q_txt_cols[0].info(q4_text(results.get("q4_summary", pd.DataFrame())))
    q_txt_cols[1].info(q5_text(results.get("q5_commodity", pd.DataFrame())))
    q_txt_cols[2].info(q6_text(results.get("q6_scope", pd.DataFrame())))

# ------------------------------------------------------------------
# Tab 2
# ------------------------------------------------------------------
with tabs[1]:
    section_header("Verification x3", "Donnees, calculs, et coherence narrative")
    ver_rows = results.get("verification_rows", [])
    if not ver_rows:
        st.warning("Aucune verification disponible.")
    else:
        cols = st.columns(3)
        for idx, row in enumerate(ver_rows):
            with cols[idx % 3]:
                render_kpi_banner(
                    row["check"],
                    row["status"],
                    row["detail"],
                    status=_status_to_banner(row["status"]),
                )

    consistency_report = results.get("consistency_report", pd.DataFrame())
    if isinstance(consistency_report, pd.DataFrame) and not consistency_report.empty:
        st.markdown("#### Controle de coherence inter-pages")
        st.dataframe(consistency_report, use_container_width=True, hide_index=True)
        fails = int((consistency_report["status"] == "FAIL").sum())
        warns = int((consistency_report["status"] == "WARN").sum())
        if fails > 0:
            challenge_block(
                "Divergences a corriger",
                f"{fails} checks en FAIL et {warns} en WARN. Les commentaires sources doivent etre ajustes.",
            )
        else:
            dynamic_narrative(
                f"Controle inter-pages sans contradiction bloquante (WARN={warns}).",
                severity="success" if warns == 0 else "warning",
            )

    objectives = results.get("objectives")
    if isinstance(objectives, dict) and objectives.get("paragraphs"):
        st.markdown("#### Synthese objectifs (DOCX)")
        st.caption(f"Source: {objectives.get('path')}")
        with st.expander("Extraits objectifs / questions", expanded=False):
            for line in objectives.get("objective_lines", [])[:20]:
                st.markdown(f"- {line}")

# ------------------------------------------------------------------
# Tab 3
# ------------------------------------------------------------------
with tabs[2]:
    section_header("Q1 - Seuils de bascule", "Distances aux seuils stage_2 et annees de franchissement")
    q1_detail = results.get("q1_detail", pd.DataFrame())
    q1_country = results.get("q1_country", pd.DataFrame())
    if isinstance(q1_detail, pd.DataFrame) and not q1_detail.empty:
        fig_q1 = px.scatter(
            q1_detail,
            x="sr",
            y="h_negative_obs",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
            hover_data=["year", "capture_ratio_pv", "cross_all"],
            title="Q1 - SR vs heures negatives observees",
        )
        fig_q1.update_layout(height=480, xaxis_title="SR", yaxis_title="h_negative_obs", **PLOTLY_LAYOUT_DEFAULTS)
        fig_q1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q1, use_container_width=True)
        st.dataframe(q1_country, use_container_width=True, hide_index=True)

        render_commentary(
            so_what_block(
                title="Q1 - Lecture des seuils",
                purpose="Objectiver quand la combinaison SR + h_neg + capture PV devient compatible avec stage_2.",
                observed={
                    "countries_cross_latest": int(q1_country["latest_cross_all"].fillna(False).sum()),
                    "countries_total": int(len(q1_country)),
                    "first_cross_min_year": float(q1_country["first_stage2_cross_year"].min(skipna=True)),
                },
                method_link="Seuils stage_2 issus de thresholds.yaml, sans ajustement ad hoc.",
                limits="Le franchissement ne prouve pas a lui seul la causalite; il signale un regime de risque.",
                n=len(q1_detail),
                decision_use="Declencher plus tot les leviers de flex dans les pays proches du franchissement.",
            )
        )

    section_header("Q2 - Pentes phase 2", "capture_ratio_pv ~ pv_penetration_pct_gen (hors 2022)")
    q2_df = results.get("q2_slopes", pd.DataFrame())
    if isinstance(q2_df, pd.DataFrame) and not q2_df.empty:
        fig_q2 = px.bar(
            q2_df,
            x="country",
            y="slope",
            color="robustesse",
            hover_data=["r_squared", "p_value", "n_points"],
            title="Q2 - Pente capture ratio PV par pays",
        )
        fig_q2.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig_q2.update_layout(height=430, **PLOTLY_LAYOUT_DEFAULTS)
        fig_q2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q2, use_container_width=True)
        st.dataframe(q2_df, use_container_width=True, hide_index=True)
        render_commentary(
            so_what_block(
                title="Q2 - Vitesse de degradation",
                purpose="Comparer la sensibilite du capture ratio PV a la penetration selon les pays.",
                observed={
                    "slope_min": float(q2_df["slope"].min()),
                    "slope_max": float(q2_df["slope"].max()),
                    "sig_count": int((q2_df["p_value"] <= 0.05).sum()),
                },
                method_link="linregress sur series annuelles, exclusion outlier 2022.",
                limits="n_points limite par pays; la pente doit etre lue avec r2 et p_value.",
                n=len(q2_df),
                decision_use="Prioriser les pays ou la degradation est la plus rapide et statistiquement robuste.",
            )
        )

# ------------------------------------------------------------------
# Tab 4
# ------------------------------------------------------------------
with tabs[3]:
    section_header("Q3 - Conditions de transition 2->3", "FAR, heures negatives et regimes")
    q3_df = results.get("q3_transition", pd.DataFrame())
    if isinstance(q3_df, pd.DataFrame) and not q3_df.empty:
        fig_q3 = px.scatter(
            q3_df,
            x="far_latest",
            y="h_negative_latest",
            color="status_transition_2_to_3",
            hover_data=["country", "h_negative_slope_per_year", "h_regime_a_latest", "sr_latest"],
            title="Q3 - FAR vs heures negatives (dernier point annuel)",
        )
        fig_q3.update_layout(height=450, xaxis_title="FAR latest", yaxis_title="h_negative latest", **PLOTLY_LAYOUT_DEFAULTS)
        fig_q3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q3, use_container_width=True)
        st.dataframe(q3_df, use_container_width=True, hide_index=True)

    section_header("Q4 - Effet batteries", "Plateau baseline et sweep sous stress")
    q4_df = results.get("q4_summary", pd.DataFrame())
    if isinstance(q4_df, pd.DataFrame) and not q4_df.empty:
        st.dataframe(q4_df, use_container_width=True, hide_index=True)
        plateau_count = int(q4_df["plateau_baseline"].fillna(False).sum())
        if plateau_count > 0:
            challenge_block(
                "Plateau Q4 detecte",
                f"{plateau_count}/{len(q4_df)} pays presentent un sweep baseline plat. "
                "Ce signal est interpretable physiquement et n'indique pas automatiquement un bug.",
            )
        render_commentary(
            so_what_block(
                title="Q4 - Identification de l'effet batterie",
                purpose="Distinguer un vrai effet marginal BESS d'un cas deja absorbe sans contrainte residuelle.",
                observed={
                    "plateau_count": plateau_count,
                    "stress_found_count": int(q4_df["stress_found"].fillna(False).sum()),
                },
                method_link="Diagnostic plateau baseline puis stress de reference deterministe (delta PV minimal).",
                limits="Effet sensible aux hypotheses de flex structurelle et au perimetre scenario.",
                n=len(q4_df),
                decision_use="Dimensionner BESS uniquement dans une zone de sensibilite observable.",
            )
        )

        country_options = sorted(q4_df["country"].dropna().unique().tolist())
        country_sel = st.selectbox("Pays detail Q4", country_options, key="excesum_q4_country")
        base_sweeps = results.get("q4_baseline_sweeps", {})
        stress_sweeps = results.get("q4_stress_sweeps", {})
        b = base_sweeps.get(country_sel, pd.DataFrame())
        s = stress_sweeps.get(country_sel, pd.DataFrame())
        if isinstance(b, pd.DataFrame) and not b.empty:
            fig_b = px.line(
                b,
                x="delta_bess_power_gw",
                y=["far", "h_regime_a"],
                markers=True,
                title=f"Q4 baseline - {country_sel}",
            )
            fig_b.update_layout(height=420, **PLOTLY_LAYOUT_DEFAULTS)
            fig_b.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
            fig_b.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
            st.plotly_chart(fig_b, use_container_width=True)
        if isinstance(s, pd.DataFrame) and not s.empty:
            fig_s = px.line(
                s,
                x="delta_bess_power_gw",
                y=["far", "h_regime_a"],
                markers=True,
                title=f"Q4 stress - {country_sel}",
            )
            fig_s.update_layout(height=420, **PLOTLY_LAYOUT_DEFAULTS)
            fig_s.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
            fig_s.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
            st.plotly_chart(fig_s, use_container_width=True)

# ------------------------------------------------------------------
# Tab 5
# ------------------------------------------------------------------
with tabs[4]:
    section_header("Q5 - Impact CO2/Gaz", "Sensibilite de TTL sous scenarios synthetiques")
    q5_df = results.get("q5_commodity", pd.DataFrame())
    if isinstance(q5_df, pd.DataFrame) and not q5_df.empty:
        chart = q5_df.melt(
            id_vars=["country", "year"],
            value_vars=["delta_ttl_high_co2", "delta_ttl_high_gas"],
            var_name="scenario",
            value_name="delta_ttl",
        )
        fig_q5 = px.bar(
            chart,
            x="country",
            y="delta_ttl",
            color="scenario",
            barmode="group",
            title="Q5 - Delta TTL sous stress CO2 / gaz",
        )
        fig_q5.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig_q5.update_layout(height=430, **PLOTLY_LAYOUT_DEFAULTS)
        fig_q5.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_q5.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_q5, use_container_width=True)
        st.dataframe(q5_df, use_container_width=True, hide_index=True)

    section_header("Q6 - Chaleur/Froid", "Conclusion prudente sur le perimetre actuel")
    q6_df = results.get("q6_scope", pd.DataFrame())
    if isinstance(q6_df, pd.DataFrame) and not q6_df.empty:
        st.dataframe(q6_df, use_container_width=True, hide_index=True)
        render_commentary(
            so_what_block(
                title="Q6 - Portee des conclusions",
                purpose="Distinguer ce qui est mesure de ce qui necessite des donnees additionnelles.",
                observed={
                    "countries_without_heat_data": int((~q6_df["heat_cold_dataset_available"]).sum()),
                    "countries_total": int(len(q6_df)),
                },
                method_link="Audit explicite des variables disponibles dans le pipeline actuel.",
                limits="Sans donnees chaleur/froid dediees, aucune attribution causale robuste n'est possible.",
                n=len(q6_df),
                decision_use="Definir un plan de collecte data avant conclusion technologique forte sur Q6.",
            )
        )

# ------------------------------------------------------------------
# Tab 6
# ------------------------------------------------------------------
with tabs[5]:
    section_header("Conclusions par pays", "Synthese detaillee et traceable")
    country_df = results.get("country_conclusions", pd.DataFrame())
    if isinstance(country_df, pd.DataFrame) and not country_df.empty:
        country = st.selectbox("Pays", sorted(country_df["country"].tolist()), key="excesum_country_select")
        row = country_df[country_df["country"] == country].iloc[0]
        st.markdown(country_conclusion_markdown(row))

        history = metrics_df[metrics_df["country"] == country].sort_values("year")
        fig_c = px.line(
            history,
            x="year",
            y=["sr", "far", "capture_ratio_pv", "h_negative_obs"],
            markers=True,
            title=f"Historique structurel - {country}",
        )
        fig_c.update_layout(height=440, **PLOTLY_LAYOUT_DEFAULTS)
        fig_c.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig_c.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Aucune conclusion pays disponible.")

# ------------------------------------------------------------------
# Tab 7
# ------------------------------------------------------------------
with tabs[6]:
    section_header("Annexes chiffrées", "Tables de preuve complètes")
    for key, label in [
        ("metrics_df", "Metriques baseline"),
        ("q1_detail", "Q1 detail seuils"),
        ("q2_slopes", "Q2 pentes"),
        ("q3_transition", "Q3 transition"),
        ("q4_summary", "Q4 synthese"),
        ("q5_commodity", "Q5 commodites"),
        ("q6_scope", "Q6 perimetre"),
        ("country_conclusions", "Conclusions par pays"),
        ("consistency_report", "Rapport coherence commentaires"),
    ]:
        df = results.get(key, pd.DataFrame())
        with st.expander(label, expanded=False):
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("Tableau vide.")

    if st.button("Regenerer docs ExceSum", use_container_width=True):
        c_path, v_path = write_excesum_docs(results)
        st.success(f"Docs ecrits: {c_path} et {v_path}")
