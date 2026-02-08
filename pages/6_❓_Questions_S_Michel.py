"""Page 6 - Questions S. Michel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import commentary_block
from src.metrics import compute_annual_metrics
from src.scenario_engine import apply_scenario
from src.slope_analysis import compute_slope
from src.ui_helpers import guard_no_data, inject_global_css, normalize_state_metrics, render_commentary, section

st.set_page_config(page_title="Questions S. Michel", page_icon="❓", layout="wide")
inject_global_css()

st.title("❓ Questions S. Michel")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Questions S. Michel")
normalize_state_metrics(state)

metrics = state["metrics"]
proc = state["processed"]
if not metrics:
    guard_no_data("la page Questions S. Michel")


def _normalize_metrics_schema(m: dict) -> dict:
    out = dict(m)
    # Legacy compatibility mappings (Cloud/session cache safety)
    if "h_negative_obs" not in out and "h_negative" in out:
        out["h_negative_obs"] = out["h_negative"]
    if "h_below_5_obs" not in out and "h_below_5" in out:
        out["h_below_5_obs"] = out["h_below_5"]
    if "h_regime_d" not in out and "h_regime_d_tail" in out:
        out["h_regime_d"] = out["h_regime_d_tail"]
    if "far" not in out and "far_structural" in out:
        out["far"] = out["far_structural"]
    if "pv_penetration_pct_gen" not in out and "pv_share" in out:
        out["pv_penetration_pct_gen"] = float(out["pv_share"]) * 100.0
    if "wind_penetration_pct_gen" not in out and "wind_share" in out:
        out["wind_penetration_pct_gen"] = float(out["wind_share"]) * 100.0
    if "vre_penetration_pct_gen" not in out and "vre_share" in out:
        out["vre_penetration_pct_gen"] = float(out["vre_share"]) * 100.0
    return out


rows = []
for (c, y, p), m in metrics.items():
    if p != state["price_mode"]:
        continue
    d = state["diagnostics"].get((c, y), {})
    rows.append({"country": c, "year": y, **_normalize_metrics_schema(m), "phase": d.get("phase", "unknown")})

df_all = pd.DataFrame(rows)
if df_all.empty:
    guard_no_data("la page Questions S. Michel")

required_cols = [
    "sr",
    "h_negative_obs",
    "capture_ratio_pv",
    "far",
    "pv_penetration_pct_gen",
]
for col in required_cols:
    if col not in df_all.columns:
        df_all[col] = np.nan

if state["exclude_2022"]:
    df_reg = df_all[df_all["year"] != 2022]
else:
    df_reg = df_all

tabs = st.tabs([
    "Q1 Seuil 1→2",
    "Q2 Pente phase 2",
    "Q3 Conditions 2→3",
    "Q4 Effet batteries",
    "Q5 CO2/Gaz",
    "Q6 Stockage chaleur/froid",
])

with tabs[0]:
    section("Q1 - Seuils de bascule vers stage_2", "SR / h_negative_obs / capture_ratio_pv")
    fig = px.scatter(df_reg, x="sr", y="h_negative_obs", color="country", hover_data=["year", "capture_ratio_pv", "phase"])
    fig.add_hline(y=200, line_dash="dash")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_reg[["country", "year", "sr", "h_negative_obs", "capture_ratio_pv", "phase"]], use_container_width=True, hide_index=True)

    render_commentary(
        commentary_block(
            title="Q1 - Constat de seuil",
            n_label="points pays-annee",
            n_value=len(df_reg),
            observed={"sr_median": float(df_reg["sr"].median()), "h_negative_median": float(df_reg["h_negative_obs"].median())},
            method_link="Seuils stage_2 issus de thresholds.yaml (h_negative_min, capture_ratio_pv_max...).",
            limits="Seuils heuristiques de classification; ils ne constituent pas un test causal.",
        )
    )

with tabs[1]:
    section("Q2 - Pentes capture_ratio_pv vs pv_penetration_pct_gen", "Comparaison pays")
    slope_rows = []
    for c in sorted(df_reg["country"].unique()):
        metrics_list = [m for (cc, yy, p), m in metrics.items() if cc == c and p == state["price_mode"]]
        s = compute_slope(metrics_list, "pv_penetration_pct_gen", "capture_ratio_pv", exclude_outliers=state["exclude_2022"])
        slope_rows.append({"country": c, **s})

    slope_df = pd.DataFrame(slope_rows)
    fig = px.bar(slope_df, x="country", y="slope", color="country", text="r_squared")
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        slope_df[["country", "slope", "intercept", "r_squared", "p_value", "n_points"]],
        use_container_width=True,
        hide_index=True,
    )

    render_commentary(
        commentary_block(
            title="Q2 - Lecture des pentes",
            n_label="pays",
            n_value=len(slope_df),
            observed={"slope_min": float(slope_df["slope"].min()), "slope_max": float(slope_df["slope"].max())},
            method_link="Estimation linregress sur series annuelles, exclusion optionnelle de 2022.",
            limits="n faible par pays; interpretation statistique conditionnee par p-value et outliers.",
        )
    )

with tabs[2]:
    section("Q3 - Passage stage_2 -> stage_3", "Rôle de FAR")
    fig = px.scatter(df_reg, x="far", y="h_negative_obs", color="country", hover_data=["year", "phase", "sr"])
    fig.add_vline(x=0.60, line_dash="dash")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_reg[["country", "year", "far", "h_negative_obs", "sr", "phase"]], use_container_width=True, hide_index=True)

    render_commentary(
        commentary_block(
            title="Q3 - Conditions d'absorption",
            n_label="points",
            n_value=len(df_reg),
            observed={"far_median": float(df_reg["far"].median()), "h_neg_median": float(df_reg["h_negative_obs"].median())},
            method_link="Stage_3 s'appuie sur far_min/far_strong et dynamique h_negative selon thresholds.",
            limits="L'effet interannuel de declin h_negative depend du contexte macro et du mode prix.",
        )
    )

with tabs[3]:
    section("Q4 - Combien de batteries pour ameliorer FAR ?", "Sensibilite deterministic")
    country = st.selectbox("Pays (Q4)", sorted(df_reg["country"].unique()), key="q4_country")
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
            df_s = apply_scenario(
                df_base_processed=proc[base_key],
                country_key=country,
                year=year,
                country_cfg=state["countries_cfg"][country],
                thresholds=state["thresholds"],
                commodities=state["commodities"],
                scenario_params={"delta_bess_power_gw": float(x), "delta_bess_energy_gwh": float(x) * 4.0},
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
            commentary_block(
                title="Q4 - Rendement marginal BESS",
                n_label="points de grille",
                n_value=len(out_df),
                observed={"far_start": float(out_df["far"].iloc[0]), "far_end": float(out_df["far"].iloc[-1])},
                method_link="Scenarios recalcules completement; FAR mesure la part de surplus absorbee.",
                limits="Modele BESS simplifie (SoC deterministic); pas d'optimisation economique sur prix.",
            )
        )

with tabs[4]:
    section("Q5 - Sensibilite CO2 et gaz", "Ancre thermique TCA")
    gas = np.arange(15, 81, 2)
    co2 = np.arange(20, 181, 5)
    G, C = np.meshgrid(gas, co2)
    # CCGT convention
    tca = G / 0.57 + (0.202 / 0.57) * C + 3.0
    fig = go.Figure(data=go.Heatmap(x=gas, y=co2, z=tca, colorscale="YlOrRd"))
    fig.update_layout(height=420, xaxis_title="Gaz EUR/MWh_th", yaxis_title="CO2 EUR/t")
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame(
        {
            "gas": [20, 30, 50, 70],
            "co2": [40, 80, 120, 160],
        }
    )
    table["tca_ccgt"] = table["gas"] / 0.57 + (0.202 / 0.57) * table["co2"] + 3.0
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_commentary(
        commentary_block(
            title="Q5 - Ancre thermique",
            n_label="combinaisons",
            n_value=int(tca.size),
            observed={"tca_min": float(tca.min()), "tca_max": float(tca.max())},
            method_link="Formule TCA CCGT conforme aux constantes ETA/EF/VOM.",
            limits="N'integre pas les primes de rarete ni les contraintes d'unite reelles.",
        )
    )

with tabs[5]:
    section("Q6 - Synergies stockage chaleur/froid", "Proxy comparatif avec BESS")
    duration = np.array([2, 4, 6, 8, 12, 24])
    # Simple comparative proxy: effective recovered energy per unit power
    bess_eff = 0.88
    thermal_eff = 0.50
    bess_value = duration * bess_eff
    thermal_value = duration * thermal_eff
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=duration, y=bess_value, mode="lines+markers", name="BESS (eta=0.88)"))
    fig.add_trace(go.Scatter(x=duration, y=thermal_value, mode="lines+markers", name="Thermique (eta=0.50)"))
    fig.update_layout(height=360, xaxis_title="Duree (h)", yaxis_title="Energie utile (unite relative)")
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame({"duree_h": duration, "bess_relative": bess_value, "thermal_relative": thermal_value})
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_commentary(
        commentary_block(
            title="Q6 - Complementarite technologique",
            n_label="durees",
            n_value=len(duration),
            observed={"bess_24h": float(bess_value[-1]), "thermal_24h": float(thermal_value[-1])},
            method_link="Comparaison normative des rendements round-trip pour illustrer la portee systeme.",
            limits="Proxy simplifie; ne remplace pas un module techno-economique dedie chaleur/froid.",
        )
    )
