"""Page 3 - Capture Rates and slope analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_bridge import comment_regression, so_what_block
from src.slope_analysis import compute_slope
from src.state_adapter import coerce_numeric_columns, ensure_plot_columns, metrics_to_dataframe, normalize_metrics_record
from src.ui_theme import COUNTRY_PALETTE, PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS
from src.ui_helpers import (
    challenge_block,
    dynamic_narrative,
    guard_no_data,
    inject_global_css,
    narrative,
    normalize_state_metrics,
    render_commentary,
    section_header,
)

st.set_page_config(page_title="Capture Rates", page_icon="📈", layout="wide")
inject_global_css()
st.title("📈 Capture Rates")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Capture Rates")
normalize_state_metrics(state)

metrics_dict = state.get("metrics", {})
proc = state.get("processed", {})

df_all = metrics_to_dataframe(state, state.get("price_mode"))
if df_all.empty or "country" not in df_all.columns:
    guard_no_data("la page Capture Rates")

countries = sorted(df_all["country"].dropna().unique())
selected = st.multiselect("Pays", countries, default=state.get("countries_selected", countries[:3]))
tech = st.radio("Technologie", ["PV", "Wind"], horizontal=True)

if not selected:
    guard_no_data("la page Capture Rates")

x_key = "pv_penetration_pct_gen" if tech == "PV" else "wind_penetration_pct_gen"
y_key = "capture_ratio_pv" if tech == "PV" else "capture_ratio_wind"

plot_df = df_all[df_all["country"].isin(selected)].copy()
plot_df = ensure_plot_columns(plot_df, [x_key, y_key, "year", "country", "is_outlier"], with_notice=True)
plot_df = coerce_numeric_columns(plot_df, [x_key, y_key, "year"])
if plot_df.attrs.get("_missing_plot_columns", []):
    st.info("Colonnes completees en NaN pour robustesse: " + ", ".join(plot_df.attrs.get("_missing_plot_columns", [])))
plot_df = plot_df.dropna(subset=[x_key, y_key])
if plot_df.empty:
    guard_no_data("la page Capture Rates")

narrative(
    "Le coeur de cette page est la pente capture ratio vs penetration: c'est le rythme de cannibalisation. "
    "La question So what est simple: la valeur marginale du MW VRE se degrade-t-elle, et a quelle vitesse ?"
)

section_header("Scatter penetration vs capture ratio", "Regression par pays")
fig1 = px.scatter(
    plot_df,
    x=x_key,
    y=y_key,
    color="country",
    color_discrete_map=COUNTRY_PALETTE,
    hover_data=["year"],
)

for c in sorted(plot_df["country"].unique()):
    metrics_list = []
    for (cc, yy, p), m in metrics_dict.items():
        if cc == c and p == state.get("price_mode"):
            rec = normalize_metrics_record(m)
            rec["is_outlier"] = bool(rec.get("is_outlier", yy == 2022))
            metrics_list.append(rec)

    slope = compute_slope(metrics_list, x_key=x_key, y_key=y_key, exclude_outliers=state.get("exclude_2022", True))
    if slope["n_points"] >= 3 and np.isfinite(slope["slope"]):
        x_vals = np.array(slope["x_values"])
        x_line = np.linspace(x_vals.min(), x_vals.max(), 20)
        y_line = slope["slope"] * x_line + slope["intercept"]
        fig1.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"{c} reg", showlegend=False))

fig1.update_layout(height=430, **PLOTLY_LAYOUT_DEFAULTS)
fig1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.plotly_chart(fig1, use_container_width=True)

slope_all = compute_slope(
    [
        normalize_metrics_record(m)
        for (c, y, p), m in metrics_dict.items()
        if c in selected and p == state.get("price_mode")
    ],
    x_key=x_key,
    y_key=y_key,
    exclude_outliers=state.get("exclude_2022", True),
)
render_commentary(comment_regression(slope_all, x_name=x_key, y_name=y_key))

if slope_all["n_points"] >= 3 and np.isfinite(slope_all["slope"]):
    if slope_all["slope"] > 0:
        challenge_block(
            "Pente positive detectee",
            "Le capture ratio augmente avec la penetration sur l'echantillon. So what: verifier si l'effet commodites domine la dynamique VRE.",
        )
    elif np.isfinite(slope_all.get("p_value", np.nan)) and float(slope_all["p_value"]) > 0.05:
        dynamic_narrative(
            "Pente non significative au seuil de 5%. So what: prudence, signal de cannibalisation statistiquement fragile.",
            severity="warning",
        )

section_header("Price duration curve", "Prix observes par annee")
country = st.selectbox("Pays PDC", selected, key="pdc_country_v3")
pairs = sorted([k for k in proc.keys() if k[0] == country and k[4] == state.get("price_mode")])

fig2 = go.Figure()
for k in pairs:
    if "price_da_eur_mwh" not in proc[k].columns:
        continue
    yy = proc[k]["price_da_eur_mwh"].dropna().sort_values(ascending=False).values
    if len(yy) == 0:
        continue
    xx = np.linspace(0, 100, len(yy))
    fig2.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=str(k[1])))
fig2.update_layout(height=400, xaxis_title="% du temps", yaxis_title="EUR/MWh")
fig2.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
fig2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
fig2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
st.plotly_chart(fig2, use_container_width=True)

if pairs:
    latest_metrics = df_all[(df_all["country"] == country) & (df_all["year"] == pairs[-1][1])]
    p95 = float(latest_metrics.iloc[0].get("price_used_p95", np.nan)) if not latest_metrics.empty else np.nan
else:
    p95 = np.nan

render_commentary(
    so_what_block(
        title="Queue de prix",
        purpose="Mesurer l'amplitude des heures extremes et leur impact potentiel sur TTL et la valeur flex",
        observed={"price_used_p95_latest": p95},
        method_link="Courbe de duree construite sur prix DA observes; quantiles price_used selon G.7.",
                limits="Tres sensible aux episodes de crise et aux anomalies de donnees.",
                n=len(pairs),
                decision_use="Quantifier le risque de queue de prix avant de fixer la valeur cible de flexibilite.",
            )
        )

section_header("Heatmap prix observe (mois x heure locale)", "Structure intra-annuelle")
years = sorted({k[1] for k in pairs})
if years:
    year = st.selectbox("Annee Heatmap", years, key="hm_year_v3")
    k = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
    if k in proc and "price_da_eur_mwh" in proc[k].columns:
        d = proc[k]
        local = d.index.tz_convert(state["countries_cfg"][country]["timezone"])
        tmp = pd.DataFrame({"price": d["price_da_eur_mwh"].values, "month": local.month, "hour": local.hour})
        pivot = tmp.pivot_table(index="month", columns="hour", values="price", aggfunc="mean")
        fig3 = px.imshow(pivot, aspect="auto", labels={"x": "Heure", "y": "Mois", "color": "EUR/MWh"})
        fig3.update_layout(height=420, **PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig3, use_container_width=True)

        render_commentary(
            so_what_block(
                title="Pattern saisonnier",
                purpose="Identifier les zones horaires/saisonnieres de compression de prix utiles pour calibration scenario",
                observed={"price_min": float(np.nanmin(pivot.values)), "price_max": float(np.nanmax(pivot.values))},
                method_link="Aggregation locale mois x heure sur prix observes.",
                limits="Moyennes locales: les extremes intramensuels sont lisses.",
                n=int(pivot.size),
                decision_use="Repérer les plages horaires saisonnieres les plus exposees a la compression de prix.",
            )
        )
