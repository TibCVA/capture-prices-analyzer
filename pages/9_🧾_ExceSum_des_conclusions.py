"""Static ExceSum report page (frozen dataset, no runtime recompute)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.commentary_bridge import so_what_block
from src.ui_helpers import (
    dynamic_narrative,
    inject_global_css,
    narrative,
    render_commentary,
    section_header,
)
from src.ui_theme import COUNTRY_PALETTE, PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS


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


payload = _load_static_payload()
meta = payload.get("meta", {})
baseline = meta.get("baseline", {})
global_medians = payload.get("global_medians", {})
rebuild_matrix = payload.get("rebuild_matrix", {})

df_q1 = _safe_df(payload, "q1_country")
df_q2 = _safe_df(payload, "q2_slopes")
df_q3 = _safe_df(payload, "q3_transition")
df_q4 = _safe_df(payload, "q4_summary")
df_q5 = _safe_df(payload, "q5_commodity")
df_q6 = _safe_df(payload, "q6_scope")
df_country = _safe_df(payload, "country_conclusions")
df_annex = _safe_df(payload, "metrics_annex")
df_quality = _safe_df(payload, "data_quality_flags")
df_verification = _safe_df({"x": meta.get("verification", [])}, "x")

st.title("🧾 ExceSum des conclusions")
st.caption(
    "Rapport statique fige sur baseline unique. "
    "Aucun recalcul n est execute a l ouverture."
)

narrative(
    "Cette page est une synthese executive statique, fondee sur un jeu de resultats fixe. "
    "Toutes les valeurs affichees sont reproductibles et tracees."
)

if rebuild_matrix:
    dynamic_narrative(
        "Correction appliquee: les caches historiques incoherents ont ete detectes et "
        "la baseline a ete recalculée depuis raw avant generation du rapport statique.",
        severity="warning",
    )

cols = st.columns(5)
cols[0].metric("Pays", ", ".join(baseline.get("countries", [])))
years = baseline.get("years", ["?", "?"])
cols[1].metric("Periode", f"{years[0]}-{years[1]}")
cols[2].metric("Couples", int(rebuild_matrix.get("pairs_total", len(df_annex))))
cols[3].metric("h_regime_a > 0", int(rebuild_matrix.get("pairs_h_regime_a_gt_0", 0)))
cols[4].metric("Caches invalides", int(rebuild_matrix.get("cache_semantic_invalid_pairs", 0)))

section_header("Synthese globale")
st.markdown(
    f"""
- `SR median`: `{global_medians.get('sr', float('nan')):.4f}`
- `FAR median`: `{global_medians.get('far', float('nan')):.4f}`
- `IR median`: `{global_medians.get('ir', float('nan')):.4f}`
- `TTL median`: `{global_medians.get('ttl', float('nan')):.2f}`
- `capture_ratio_pv median`: `{global_medians.get('capture_ratio_pv', float('nan')):.4f}`
- `coherence regime/prix mediane`: `{100.0 * global_medians.get('regime_coherence', float('nan')):.1f}%`
"""
)

render_commentary(
    so_what_block(
        title="Lecture globale",
        purpose="Poser une base commune de lecture avant detail par question.",
        observed={
            "n_couples": int(rebuild_matrix.get("pairs_total", len(df_annex))),
            "sr_median": global_medians.get("sr"),
            "far_median": global_medians.get("far"),
            "hA_positive_pairs": rebuild_matrix.get("pairs_h_regime_a_gt_0"),
        },
        method_link="Baseline figee 5 pays x 10 ans, regeneree apres correction semantique des caches.",
        limits="Resultats valables pour ce perimetre, non extrapolables sans recalcul explicite.",
        n=int(rebuild_matrix.get("pairs_total", len(df_annex))),
        decision_use="Prioriser les analyses pays et les leviers Q1..Q6 sur une base assainie.",
    )
)

tabs = st.tabs(
    [
        "Methode & qualite",
        "Q1-Q2",
        "Q3-Q4",
        "Q5-Q6",
        "Conclusions pays",
        "Annexes",
    ]
)

with tabs[0]:
    section_header("Controle qualite et correction caches")
    if not df_verification.empty:
        st.dataframe(df_verification, use_container_width=True, hide_index=True)
    if rebuild_matrix:
        st.json(rebuild_matrix)

    if not df_quality.empty:
        st.markdown("#### Table de controle qualite")
        st.dataframe(df_quality, use_container_width=True, hide_index=True)

        caveat_price = df_quality[df_quality.get("price_completeness", 1.0) < 0.90]
        caveat_impute = df_quality[df_quality.get("raw_imputation_count", 0) > 0]
        if not caveat_price.empty:
            st.warning("Completude prix < 90% detectee sur certaines paires (notamment PL 2015-2016).")
            st.dataframe(
                caveat_price[["country", "year", "price_completeness"]],
                use_container_width=True,
                hide_index=True,
            )
        if not caveat_impute.empty:
            st.warning("Imputations raw detectees (ex: solar_mw absent sur PL 2015-2018, imputee a 0.0).")
            st.dataframe(
                caveat_impute[["country", "year", "raw_imputation_count", "raw_imputation_flags"]],
                use_container_width=True,
                hide_index=True,
            )

with tabs[1]:
    section_header("Q1 - Seuils stage_2")
    if not df_q1.empty:
        st.dataframe(df_q1, use_container_width=True, hide_index=True)

    section_header("Q2 - Pentes capture ratio PV")
    if not df_q2.empty:
        df_q2 = df_q2.sort_values("slope")
        fig = px.bar(
            df_q2,
            x="country",
            y="slope",
            color="robustesse",
            hover_data=["r_squared", "p_value", "n_points"],
            title="Q2 - Pentes capture_ratio_pv vs penetration PV",
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig.update_layout(height=420, **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_q2, use_container_width=True, hide_index=True)

with tabs[2]:
    section_header("Q3 - Conditions 2->3")
    if not df_q3.empty:
        fig = px.scatter(
            df_q3,
            x="far_latest",
            y="h_negative_latest",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
            hover_data=["status_transition_2_to_3", "h_negative_slope_per_year"],
            title="Q3 - FAR vs h_negative latest",
        )
        fig.update_layout(height=420, **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_q3, use_container_width=True, hide_index=True)

    section_header("Q4 - Batteries")
    if not df_q4.empty:
        st.dataframe(df_q4, use_container_width=True, hide_index=True)
        fig4 = px.bar(
            df_q4.sort_values("stress_delta_pv_gw"),
            x="country",
            y="stress_delta_pv_gw",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
            title="Q4 - Stress PV minimal pour effet BESS identifiable",
        )
        fig4.update_layout(height=400, **PLOTLY_LAYOUT_DEFAULTS)
        fig4.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig4.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig4, use_container_width=True)

with tabs[3]:
    section_header("Q5 - Impact CO2/Gaz")
    if not df_q5.empty:
        melt = df_q5.melt(
            id_vars=["country", "year"],
            value_vars=["delta_ttl_high_co2", "delta_ttl_high_gas"],
            var_name="scenario",
            value_name="delta_ttl",
        )
        fig5 = px.bar(
            melt,
            x="country",
            y="delta_ttl",
            color="scenario",
            barmode="group",
            title="Q5 - Delta TTL sous stress commodites",
        )
        fig5.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
        fig5.update_layout(height=420, **PLOTLY_LAYOUT_DEFAULTS)
        fig5.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig5.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig5, use_container_width=True)
        st.dataframe(df_q5, use_container_width=True, hide_index=True)

    section_header("Q6 - Chaleur / froid")
    if not df_q6.empty:
        st.dataframe(df_q6, use_container_width=True, hide_index=True)
        dynamic_narrative(
            "Q6 reste prudent: pas de conclusion causale possible sans dataset chaleur/froid dedie.",
            severity="warning",
        )

with tabs[4]:
    section_header("Conclusions par pays")
    if not df_country.empty:
        st.dataframe(df_country, use_container_width=True, hide_index=True)
        country = st.selectbox("Pays", sorted(df_country["country"].unique().tolist()))
        row = df_country[df_country["country"] == country].iloc[0]
        st.markdown(
            f"""
### {country}
- Phase recente: `{row.get('phase_latest', 'n/a')}`
- SR/FAR: `{row.get('sr_latest', float('nan')):.4f}` / `{row.get('far_latest', float('nan')):.4f}`
- Capture ratio PV: `{row.get('capture_ratio_pv_latest', float('nan')):.4f}`
- Q1 first stage_2 year: `{row.get('q1_first_stage2_year', 'n/a')}`
- Q2 slope: `{row.get('q2_slope', float('nan')):.4f}`
- Q3 status: `{row.get('q3_status', 'n/a')}`
- Q4 plateau/stress: `{row.get('q4_plateau_baseline', 'n/a')}` / `{row.get('q4_stress_found', 'n/a')}`
- Q5 deltas TTL (CO2/Gas): `{row.get('q5_delta_ttl_co2', float('nan')):.2f}` / `{row.get('q5_delta_ttl_gas', float('nan')):.2f}`
- Q6: `{row.get('q6_status', 'n/a')}`
"""
        )

with tabs[5]:
    section_header("Annexes chiffrees")
    st.dataframe(df_annex, use_container_width=True, hide_index=True)
