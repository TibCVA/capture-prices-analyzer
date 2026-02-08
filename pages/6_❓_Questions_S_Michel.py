"""Page 6 - Questions S. Michel (answer-first)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import so_what_block
from src.metrics import compute_annual_metrics
from src.scenario_engine import apply_scenario
from src.slope_analysis import compute_slope
from src.state_adapter import ensure_plot_columns, metrics_to_dataframe, normalize_metrics_record
from src.ui_helpers import (
    challenge_block,
    dynamic_narrative,
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    question_banner,
    render_commentary,
)

st.set_page_config(page_title="Questions S. Michel", page_icon="❓", layout="wide")
inject_global_css()
st.title("❓ Questions S. Michel")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Questions S. Michel")
normalize_state_metrics(state)

metrics_dict = state.get("metrics", {})
proc = state.get("processed", {})

df_all = metrics_to_dataframe(state, state.get("price_mode"))
if df_all.empty:
    guard_no_data("la page Questions S. Michel")

required = ["sr", "h_negative_obs", "capture_ratio_pv", "far", "pv_penetration_pct_gen", "phase", "country", "year"]
df_all = ensure_plot_columns(df_all, required)

if state.get("exclude_2022", True):
    df_reg = df_all[df_all["year"] != 2022].copy()
else:
    df_reg = df_all.copy()

narrative(
    "Chaque onglet repond d'abord a la question business, puis montre les preuves chiffrees. "
    "Les commentaires restent objectifs: constats, lien methode, limites."
)

tabs = st.tabs(
    [
        "Q1 Seuil 1->2",
        "Q2 Pente phase 2",
        "Q3 Conditions 2->3",
        "Q4 Effet batteries",
        "Q5 CO2/Gaz",
        "Q6 Stockage chaleur/froid",
    ]
)

with tabs[0]:
    question_banner("Q1 - A quels niveaux observe-t-on la bascule vers stage_2 ?")
    dynamic_narrative(
        "Reponse courte: la bascule stage_2 est associee a la combinaison SR en hausse, "
        "augmentation des heures negatives observees, et degradation du capture ratio PV.",
        severity="info",
    )

    q1_df = ensure_plot_columns(df_reg.copy(), ["sr", "h_negative_obs", "capture_ratio_pv", "phase", "country", "year"])
    fig = px.scatter(
        q1_df,
        x="sr",
        y="h_negative_obs",
        color="country",
        hover_data=["year", "capture_ratio_pv", "phase"],
    )
    fig.add_hline(y=200, line_dash="dash")
    fig.update_layout(height=390)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        q1_df[["country", "year", "sr", "h_negative_obs", "capture_ratio_pv", "phase"]],
        use_container_width=True,
        hide_index=True,
    )

    render_commentary(
        so_what_block(
            title="Q1 - Seuils observes",
            purpose="Objectiver le moment ou le systeme quitte une integration facile pour une phase de stress",
            observed={
                "sr_median": float(q1_df["sr"].median()),
                "h_negative_median": float(q1_df["h_negative_obs"].median()),
                "capture_ratio_pv_median": float(q1_df["capture_ratio_pv"].median()),
            },
            method_link="Seuils stage_2 issus de thresholds.yaml (h_negative, capture_ratio_pv, spreads).",
            limits="Seuils heuristiques de diagnostic, pas de causalite econometrique stricte.",
            n=len(q1_df),
        )
    )

with tabs[1]:
    question_banner("Q2 - Quelle est la pente de degradation du capture ratio PV en phase 2 ?")

    slope_rows = []
    for c in sorted(df_reg["country"].unique()):
        country_metrics = []
        for (cc, yy, p), m in metrics_dict.items():
            if cc == c and p == state.get("price_mode"):
                rec = normalize_metrics_record(m)
                rec["is_outlier"] = bool(rec.get("is_outlier", yy == 2022))
                country_metrics.append(rec)

        slope = compute_slope(
            country_metrics,
            "pv_penetration_pct_gen",
            "capture_ratio_pv",
            exclude_outliers=state.get("exclude_2022", True),
        )
        slope_rows.append({"country": c, **slope})

    slope_df = pd.DataFrame(slope_rows)
    slope_df = ensure_plot_columns(slope_df, ["country", "slope", "intercept", "r_squared", "p_value", "n_points"])

    fig = px.bar(slope_df, x="country", y="slope", color="country", text="r_squared")
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(slope_df[["country", "slope", "intercept", "r_squared", "p_value", "n_points"]], use_container_width=True, hide_index=True)

    if slope_df["slope"].dropna().gt(0).any():
        bad = ", ".join(slope_df.loc[slope_df["slope"] > 0, "country"].astype(str).tolist())
        challenge_block(
            "Pente positive detectee",
            f"Pays concernes: {bad}. So what: resultat contre-intuitif, verifier signal statistique et outliers.",
        )

    slope_min = float(slope_df["slope"].min()) if slope_df["slope"].notna().any() else float("nan")
    slope_max = float(slope_df["slope"].max()) if slope_df["slope"].notna().any() else float("nan")
    r2_median = (
        float(slope_df["r_squared"].median()) if slope_df["r_squared"].notna().any() else float("nan")
    )

    render_commentary(
        so_what_block(
            title="Q2 - Vitesse de degradation",
            purpose="Comparer la sensibilite relative des pays a la cannibalisation PV",
            observed={
                "slope_min": slope_min,
                "slope_max": slope_max,
                "r2_median": r2_median,
            },
            method_link="linregress pays par pays sur series annuelles, exclusion optionnelle 2022.",
            limits="n points limite; significance a confirmer via p-value.",
            n=len(slope_df),
        )
    )

with tabs[2]:
    question_banner("Q3 - Quelles conditions marquent le passage stage_2 -> stage_3 ?")
    dynamic_narrative(
        "Reponse courte: FAR doit monter durablement et la dynamique des heures negatives doit se detendre. "
        "Un FAR eleve seul ne suffit pas si le surplus est marginal ou si la tendance prix reste degradee.",
        severity="info",
    )

    q3_df = ensure_plot_columns(df_reg.copy(), ["far", "h_negative_obs", "phase", "sr", "country", "year"])
    fig = px.scatter(q3_df, x="far", y="h_negative_obs", color="country", hover_data=["year", "phase", "sr"])
    fig.add_vline(x=0.60, line_dash="dash")
    fig.update_layout(height=390)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(q3_df[["country", "year", "far", "h_negative_obs", "sr", "phase"]], use_container_width=True, hide_index=True)

    far_median = float(q3_df["far"].median()) if q3_df["far"].notna().any() else float("nan")
    h_negative_median = (
        float(q3_df["h_negative_obs"].median()) if q3_df["h_negative_obs"].notna().any() else float("nan")
    )
    sr_median = float(q3_df["sr"].median()) if q3_df["sr"].notna().any() else float("nan")

    render_commentary(
        so_what_block(
            title="Q3 - Conditions d'absorption",
            purpose="Mesurer si la flexibilite absorbe vraiment le surplus a un niveau compatible stage_3",
            observed={
                "far_median": far_median,
                "h_negative_median": h_negative_median,
                "sr_median": sr_median,
            },
            method_link="Regles stage_3 basees sur far_min/far_strong et dynamique des heures negatives.",
            limits="L'effet interannuel peut etre brouille par des chocs externes de commodites.",
            n=len(q3_df),
        )
    )

with tabs[3]:
    question_banner("Q4 - Combien de batteries pour freiner la degradation ?")

    countries_q4 = sorted(df_reg["country"].dropna().unique())
    country = st.selectbox("Pays (Q4)", countries_q4, key="q4_country")
    year = int(df_reg[df_reg["country"] == country]["year"].max())

    base_key = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
    if base_key not in proc:
        fallback = [k for k in proc.keys() if k[0] == country and k[1] == year]
        if fallback:
            base_key = sorted(fallback)[0]

    if base_key in proc:
        bess_grid = np.arange(0, 21, 2)
        out = []
        for x in bess_grid:
            payload = {**state.get("ui_overrides", {}), "delta_bess_power_gw": float(x), "delta_bess_energy_gwh": float(x) * 4.0}
            df_s = apply_scenario(
                df_base_processed=proc[base_key],
                country_key=country,
                year=year,
                country_cfg=state["countries_cfg"][country],
                thresholds=state["thresholds"],
                commodities=state["commodities"],
                scenario_params=payload,
                price_mode="synthetic",
            )
            m = compute_annual_metrics(df_s, country, year, state["countries_cfg"][country])
            out.append({"delta_bess_power_gw": x, "far": m["far"], "h_regime_a": m["h_regime_a"]})

        out_df = pd.DataFrame(out)
        fig = px.line(out_df, x="delta_bess_power_gw", y="far", markers=True)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(out_df, use_container_width=True, hide_index=True)

        render_commentary(
            so_what_block(
                title="Q4 - Rendement marginal BESS",
                purpose="Identifier le niveau de stockage a partir duquel les gains FAR deviennent marginaux",
                observed={
                    "far_start": float(out_df["far"].iloc[0]),
                    "far_end": float(out_df["far"].iloc[-1]),
                    "h_A_end": float(out_df["h_regime_a"].iloc[-1]),
                },
                method_link="Chaque point = scenario complet recalcule sur meme base annuelle.",
                limits="BESS simplifie (SoC deterministe), pas d'optimisation economique.",
                n=len(out_df),
            )
        )

with tabs[4]:
    question_banner("Q5 - Quel est l'impact CO2/Gaz sur l'ancre thermique ?")

    gas = np.arange(15, 81, 2)
    co2 = np.arange(20, 181, 5)
    g_grid, c_grid = np.meshgrid(gas, co2)

    ov = state.get("ui_overrides", {}) if isinstance(state.get("ui_overrides", {}), dict) else {}
    eta_ccgt = float(ov.get("eta_ccgt", 0.57))
    ef_gas = float(ov.get("ef_gas", 0.202))
    vom_ccgt = float(ov.get("vom_ccgt", 3.0))

    tca = g_grid / eta_ccgt + (ef_gas / eta_ccgt) * c_grid + vom_ccgt

    fig = go.Figure(data=go.Heatmap(x=gas, y=co2, z=tca, colorscale="YlOrRd"))
    fig.update_layout(height=430, xaxis_title="Gaz EUR/MWh_th", yaxis_title="CO2 EUR/t")
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame({"gas": [20, 30, 50, 70], "co2": [40, 80, 120, 160]})
    table["tca_ccgt"] = table["gas"] / eta_ccgt + (ef_gas / eta_ccgt) * table["co2"] + vom_ccgt
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_commentary(
        so_what_block(
            title="Q5 - Sensibilite TCA",
            purpose="Quantifier la transmission des commodites vers le niveau de queue thermique et donc TTL",
            observed={"tca_min": float(tca.min()), "tca_max": float(tca.max())},
            method_link="Formule CCGT conforme au modele prix avec parametres actifs de session.",
            limits="N'integre pas explicitement la prime de rarete ni les contraintes unitaires detaillees.",
            n=int(tca.size),
        )
    )

with tabs[5]:
    question_banner("Q6 - Stockage chaleur/froid: synergie ou competition avec BESS ?")
    dynamic_narrative(
        "Reponse courte: le stockage thermique peut completer le BESS sur des durees longues, "
        "avec rendement plus faible mais potentiel systeme utile pour absorber des surplus prolonges.",
        severity="info",
    )

    duration = np.array([2, 4, 6, 8, 12, 24])
    bess_eff = 0.88
    thermal_eff = 0.50
    bess_value = duration * bess_eff
    thermal_value = duration * thermal_eff

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=duration, y=bess_value, mode="lines+markers", name="BESS (eta=0.88)"))
    fig.add_trace(go.Scatter(x=duration, y=thermal_value, mode="lines+markers", name="Thermique (eta=0.50)"))
    fig.update_layout(height=360, xaxis_title="Duree (h)", yaxis_title="Energie utile (relative)")
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame({"duree_h": duration, "bess_relative": bess_value, "thermal_relative": thermal_value})
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_commentary(
        so_what_block(
            title="Q6 - Complementarite technologique",
            purpose="Poser un cadre quantifie pour discuter l'articulation BESS (court terme) et thermique (longue duree)",
            observed={"bess_24h": float(bess_value[-1]), "thermal_24h": float(thermal_value[-1])},
            method_link="Comparaison normative de rendement round-trip sur grille de durees.",
            limits="Proxy simplifie: pas de modele techno-economique complet chaleur/froid.",
            n=len(duration),
        )
    )
