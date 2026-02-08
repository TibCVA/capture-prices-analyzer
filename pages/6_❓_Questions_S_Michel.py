"""Page 6 - Questions S. Michel (answer-first, rigorous and didactic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import so_what_block
from src.constants import (
    COUNTRY_PALETTE,
    OUTLIER_YEARS,
    PLOTLY_AXIS_DEFAULTS,
    PLOTLY_LAYOUT_DEFAULTS,
)
from src.metrics import compute_annual_metrics
from src.scenario_engine import apply_scenario
from src.slope_analysis import compute_slope
from src.state_adapter import coerce_numeric_columns, ensure_plot_columns, metrics_to_dataframe, normalize_metrics_record
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
thresholds = state.get("thresholds", {})

df_all = metrics_to_dataframe(state, state.get("price_mode"))
if df_all.empty or "country" not in df_all.columns:
    guard_no_data("la page Questions S. Michel")

required = [
    "sr",
    "h_negative_obs",
    "capture_ratio_pv",
    "far",
    "pv_penetration_pct_gen",
    "phase",
    "country",
    "year",
]
df_all = ensure_plot_columns(df_all, required, with_notice=True)
df_all = coerce_numeric_columns(
    df_all,
    ["sr", "h_negative_obs", "capture_ratio_pv", "far", "pv_penetration_pct_gen", "ttl", "ir", "vre_penetration_pct_gen"],
)
if df_all.attrs.get("_missing_plot_columns", []):
    st.info("Colonnes manquantes completees en NaN: " + ", ".join(df_all.attrs.get("_missing_plot_columns", [])))

if state.get("exclude_2022", True):
    df_reg = df_all[~df_all["year"].isin(OUTLIER_YEARS)].copy()
else:
    df_reg = df_all.copy()

narrative(
    "Chaque onglet repond a une question business avec la meme logique: "
    "reponse courte, preuve graphique, tableau de chiffres, interpretation methodologique et limites."
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

# ------------------------------------------------------------------
# Q1
# ------------------------------------------------------------------
with tabs[0]:
    question_banner("Q1 - A quels niveaux observe-t-on la bascule vers stage_2 ?")
    dynamic_narrative(
        "Reponse courte: la bascule vers stage_2 apparait quand SR, heures negatives observees et degradation "
        "du capture ratio PV se combinent durablement.",
        severity="info",
    )

    q1_df = ensure_plot_columns(df_reg.copy(), ["sr", "h_negative_obs", "capture_ratio_pv", "phase", "country", "year"])
    q1_df = coerce_numeric_columns(q1_df, ["sr", "h_negative_obs", "capture_ratio_pv"])
    q1_df = q1_df.dropna(subset=["sr", "h_negative_obs", "capture_ratio_pv"])
    if q1_df.empty:
        st.info("Q1 indisponible: donnees insuffisantes.")
    else:
        fig = px.scatter(
            q1_df,
            x="sr",
            y="h_negative_obs",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
            hover_data=["year", "capture_ratio_pv", "phase"],
        )
        stage2 = thresholds.get("phase_thresholds", {}).get("stage_2", {})
        hneg_ref = float(stage2.get("h_negative_min", 200))
        fig.add_hline(y=hneg_ref, line_dash="dash", line_color="#e11d48", annotation_text=f"h_neg={hneg_ref:.0f}")
        fig.update_layout(
            height=410,
            xaxis_title="SR (surplus ratio)",
            yaxis_title="Heures negatives observees",
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            q1_df[["country", "year", "sr", "h_negative_obs", "capture_ratio_pv", "phase"]].sort_values(["country", "year"]),
            use_container_width=True,
            hide_index=True,
        )

        render_commentary(
            so_what_block(
                title="Q1 - Seuils de bascule observes",
                purpose="Le passage stage_2 correspond a un systeme qui ne digere plus facilement les surplus VRE.",
                observed={
                    "sr_median": float(q1_df["sr"].median()),
                    "h_negative_median": float(q1_df["h_negative_obs"].median()),
                    "capture_ratio_pv_median": float(q1_df["capture_ratio_pv"].median()),
                },
                method_link="Lecture conjointe des indicateurs utilises dans thresholds.stage_2.",
                limits="Seuils de diagnostic, pas modele causal; la chronologie pays par pays reste essentielle.",
                n=len(q1_df),
                decision_use="Fixer des seuils d'alerte pour anticiper la phase de stress avant degradation severe.",
            )
        )

# ------------------------------------------------------------------
# Q2
# ------------------------------------------------------------------
with tabs[1]:
    question_banner("Q2 - Quelle est la pente de degradation du capture ratio PV en phase 2 ?")

    slope_rows = []
    for c in sorted(df_reg["country"].dropna().unique()):
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
    slope_df = coerce_numeric_columns(slope_df, ["slope", "intercept", "r_squared", "p_value", "n_points"])

    if slope_df.empty:
        st.info("Q2 indisponible: donnees insuffisantes.")
    else:
        slope_df["sig"] = np.where(slope_df["p_value"] <= 0.05, "significatif", "fragile")
        fig = px.bar(
            slope_df,
            x="country",
            y="slope",
            color="sig",
            color_discrete_map={"significatif": "#16a34a", "fragile": "#f59e0b"},
            hover_data=["r_squared", "p_value", "n_points"],
            text="r_squared",
        )
        fig.add_hline(y=0.0, line_dash="dot", line_color="#64748b")
        fig.update_layout(height=390, xaxis_title="Pays", yaxis_title="Slope capture_ratio_pv vs penetration PV", **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            slope_df[["country", "slope", "intercept", "r_squared", "p_value", "n_points"]].sort_values("slope"),
            use_container_width=True,
            hide_index=True,
        )

        bad = slope_df[slope_df["slope"] > 0]["country"].tolist()
        if bad:
            challenge_block(
                "Pentes contre-intuitives",
                "Pays avec pente positive: " + ", ".join(map(str, bad)) + ". Verifier n, p-value et effets exogenes.",
            )

        render_commentary(
            so_what_block(
                title="Q2 - Intensite de cannibalisation",
                purpose="Plus la pente est negative, plus la valeur captee se degrade vite quand la penetration PV augmente.",
                observed={
                    "slope_min": float(slope_df["slope"].min()) if slope_df["slope"].notna().any() else np.nan,
                    "slope_max": float(slope_df["slope"].max()) if slope_df["slope"].notna().any() else np.nan,
                    "r2_median": float(slope_df["r_squared"].median()) if slope_df["r_squared"].notna().any() else np.nan,
                },
                method_link="linregress pays par pays sur series annuelles normalisees v3.",
                limits="n souvent limite; interpretation forte seulement si p-value et r2 coherents.",
                n=len(slope_df),
                decision_use="Comparer les vitesses de degradation pour prioriser les plans pays et l'ordre des leviers.",
            )
        )

# ------------------------------------------------------------------
# Q3
# ------------------------------------------------------------------
with tabs[2]:
    question_banner("Q3 - Quelles conditions marquent le passage stage_2 -> stage_3 ?")
    dynamic_narrative(
        "Reponse courte: le passage vers stage_3 exige un FAR durablement eleve et une detente des heures negatives, "
        "pas seulement un FAR ponctuellement bon.",
        severity="info",
    )

    q3_df = ensure_plot_columns(df_reg.copy(), ["far", "h_negative_obs", "phase", "sr", "country", "year"])
    q3_df = coerce_numeric_columns(q3_df, ["far", "h_negative_obs", "sr"])
    q3_df = q3_df.dropna(subset=["far", "h_negative_obs"])
    if q3_df.empty:
        st.info("Q3 indisponible: donnees insuffisantes.")
    else:
        fig = px.scatter(
            q3_df,
            x="far",
            y="h_negative_obs",
            color="country",
            color_discrete_map=COUNTRY_PALETTE,
            hover_data=["year", "phase", "sr"],
        )
        stage3 = thresholds.get("phase_thresholds", {}).get("stage_3", {})
        far_ref = float(stage3.get("far_min", 0.60))
        fig.add_vline(x=far_ref, line_dash="dash", line_color="#2563eb", annotation_text=f"FAR={far_ref:.2f}")
        fig.update_layout(height=405, xaxis_title="FAR", yaxis_title="Heures negatives observees", **PLOTLY_LAYOUT_DEFAULTS)
        fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
        fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            q3_df[["country", "year", "far", "h_negative_obs", "sr", "phase"]].sort_values(["country", "year"]),
            use_container_width=True,
            hide_index=True,
        )

        render_commentary(
            so_what_block(
                title="Q3 - Conditions de transition vers l'absorption structurelle",
                purpose="Un FAR eleve doit s'accompagner d'une baisse des heures negatives pour valider une transition robuste.",
                observed={
                    "far_median": float(q3_df["far"].median()),
                    "h_negative_median": float(q3_df["h_negative_obs"].median()),
                    "sr_median": float(q3_df["sr"].median()) if q3_df["sr"].notna().any() else np.nan,
                },
                method_link="Regles stage_3 basees sur FAR et dynamique des observables dans thresholds.yaml.",
                limits="Les chocs commodites annuels peuvent masquer partiellement l'effet de la flexibilite.",
                n=len(q3_df),
                decision_use="Valider si le systeme est pret pour une acceleration VRE ou s'il faut d'abord renforcer la flex.",
            )
        )

# ------------------------------------------------------------------
# Q4
# ------------------------------------------------------------------
with tabs[3]:
    question_banner("Q4 - Combien de batteries pour freiner la degradation ?")

    countries_q4 = sorted(df_reg["country"].dropna().unique())
    if not countries_q4:
        st.info("Q4 indisponible: aucun pays valide.")
    else:
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
                payload = {
                    **state.get("ui_overrides", {}),
                    "delta_bess_power_gw": float(x),
                    "delta_bess_energy_gwh": float(x) * 4.0,
                }
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
            out_df = coerce_numeric_columns(out_df, ["delta_bess_power_gw", "far", "h_regime_a"])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=out_df["delta_bess_power_gw"],
                    y=out_df["far"],
                    mode="lines+markers",
                    name="FAR",
                    line=dict(color="#2563eb", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=out_df["delta_bess_power_gw"],
                    y=out_df["h_regime_a"],
                    mode="lines+markers",
                    name="h_regime_a",
                    yaxis="y2",
                    line=dict(color="#dc2626", width=2, dash="dash"),
                )
            )
            fig.update_layout(
                height=400,
                xaxis_title="BESS supplementaire (GW)",
                yaxis=dict(title="FAR"),
                yaxis2=dict(title="h_regime_a", overlaying="y", side="right"),
                **PLOTLY_LAYOUT_DEFAULTS,
            )
            fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
            fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out_df, use_container_width=True, hide_index=True)

            render_commentary(
                so_what_block(
                    title="Q4 - Effet marginal du BESS",
                    purpose="Les premiers GW de BESS reduisent en general plus vite les heures A; ensuite les gains marginaux diminuent.",
                    observed={
                        "far_start": float(out_df["far"].iloc[0]),
                        "far_end": float(out_df["far"].iloc[-1]),
                        "h_A_start": float(out_df["h_regime_a"].iloc[0]),
                        "h_A_end": float(out_df["h_regime_a"].iloc[-1]),
                    },
                    method_link="Chaque point est un scenario complet recalcule (NRL->regimes->prix synth->metriques).",
                    limits="Modele BESS simplifie (SoC deterministe, sans optimisation economique intra-jour detaillee).",
                    n=len(out_df),
                    decision_use="Identifier une zone de dimensionnement cible avant saturation du rendement marginal.",
                )
            )
        else:
            st.info("Q4 indisponible: baseline process absente.")

# ------------------------------------------------------------------
# Q5
# ------------------------------------------------------------------
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
    dtca_dgas = 1.0 / eta_ccgt
    dtca_dco2 = ef_gas / eta_ccgt

    fig = go.Figure(data=go.Heatmap(x=gas, y=co2, z=tca, colorscale="YlOrRd"))
    fig.update_layout(
        height=430,
        xaxis_title="Gaz EUR/MWh_th",
        yaxis_title="CO2 EUR/t",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame({"gas": [20, 30, 50, 70], "co2": [40, 80, 120, 160]})
    table["tca_ccgt"] = table["gas"] / eta_ccgt + (ef_gas / eta_ccgt) * table["co2"] + vom_ccgt
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_commentary(
        so_what_block(
            title="Q5 - Sensibilite de l'ancre thermique",
            purpose="Le gaz et le CO2 deplacent le niveau de TCA et donc la queue thermique des prix (TTL).",
            observed={
                "tca_min": float(tca.min()),
                "tca_max": float(tca.max()),
                "dTCA_dGas": float(dtca_dgas),
                "dTCA_dCO2": float(dtca_dco2),
            },
            method_link="Formule CCGT du modele prix v3 avec parametres de session.",
            limits="Ne capture pas toutes les primes de rarete ni la micro-structure du marche journalier.",
            n=int(tca.size),
            decision_use="Construire des stress tests gaz/CO2 coherents avant interpretation de variations de TTL.",
        )
    )

# ------------------------------------------------------------------
# Q6
# ------------------------------------------------------------------
with tabs[5]:
    question_banner("Q6 - Stockage chaleur/froid: synergie ou competition avec BESS ?")
    dynamic_narrative(
        "Reponse courte: le BESS est plus efficace en aller-retour court, le thermique peut etre pertinent sur durees longues; "
        "les deux sont complementaires si les usages sont bien segmentes.",
        severity="info",
    )

    duration = np.array([2, 4, 6, 8, 12, 24], dtype=float)
    bess_eff = 0.88
    thermal_eff = 0.50
    bess_value = duration * bess_eff
    thermal_value = duration * thermal_eff

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=duration, y=bess_value, mode="lines+markers", name="BESS (eta=0.88)", line=dict(color="#2563eb", width=2)))
    fig.add_trace(
        go.Scatter(
            x=duration,
            y=thermal_value,
            mode="lines+markers",
            name="Thermique (eta=0.50)",
            line=dict(color="#ea580c", width=2, dash="dash"),
        )
    )
    fig.update_layout(height=365, xaxis_title="Duree (h)", yaxis_title="Energie utile relative", **PLOTLY_LAYOUT_DEFAULTS)
    fig.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame(
        {
            "duree_h": duration,
            "bess_relative": bess_value,
            "thermal_relative": thermal_value,
            "ratio_thermal_sur_bess": thermal_value / np.where(bess_value == 0, np.nan, bess_value),
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    render_commentary(
        so_what_block(
            title="Q6 - Complementarite technologique",
            purpose="Le thermique stocke potentiellement plus longtemps, mais restitue moins d'energie utile a capacite equivalente.",
            observed={
                "bess_24h": float(bess_value[-1]),
                "thermal_24h": float(thermal_value[-1]),
                "ratio_24h": float(thermal_value[-1] / bess_value[-1]),
            },
            method_link="Comparaison normative de rendement round-trip sur une meme grille de durees.",
            limits="Proxy simplifie; n'integre ni CAPEX/OPEX ni contraintes reseau/detail techno.",
            n=len(duration),
            decision_use="Structurer la discussion sur le bon mix court-terme (BESS) vs longue duree (thermique).",
        )
    )

