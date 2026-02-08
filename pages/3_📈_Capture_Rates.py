"""Page 3 - Capture rates and slope analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import comment_regression, commentary_block
from src.slope_analysis import compute_slope
from src.ui_helpers import guard_no_data, inject_global_css, normalize_state_metrics, render_commentary, section

st.set_page_config(page_title="Capture Rates", page_icon="📈", layout="wide")
inject_global_css()

st.title("📈 Capture Rates")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Capture Rates")
normalize_state_metrics(state)

metrics_dict = state["metrics"]
proc = state["processed"]

countries = sorted({k[0] for k in metrics_dict.keys()})
selected = st.multiselect("Pays", countries, default=state["countries_selected"])
tech = st.radio("Technologie", ["PV", "Wind"], horizontal=True)

if not selected:
    guard_no_data("la page Capture Rates")

x_key = "pv_penetration_pct_gen" if tech == "PV" else "wind_penetration_pct_gen"
y_key = "capture_ratio_pv" if tech == "PV" else "capture_ratio_wind"

rows = []
for (c, y, p), m in metrics_dict.items():
    if c in selected and p == state["price_mode"]:
        rows.append({"country": c, "year": y, x_key: m.get(x_key), y_key: m.get(y_key), "is_outlier": m.get("is_outlier", False)})

df = pd.DataFrame(rows).dropna(subset=[x_key, y_key])
if df.empty:
    guard_no_data("la page Capture Rates")

section("Scatter penetration vs capture ratio", "Avec regression par pays")
fig1 = px.scatter(df, x=x_key, y=y_key, color="country", hover_data=["year"])
for c in sorted(df["country"].unique()):
    data = [m for (cc, yy, pp), m in metrics_dict.items() if cc == c and pp == state["price_mode"]]
    slope = compute_slope(data, x_key=x_key, y_key=y_key, exclude_outliers=state["exclude_2022"])
    if slope["n_points"] >= 3 and np.isfinite(slope["slope"]):
        x = np.array(slope["x_values"])
        xx = np.linspace(x.min(), x.max(), 20)
        yy = slope["slope"] * xx + slope["intercept"]
        fig1.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=f"{c} reg", showlegend=False))
fig1.update_layout(height=430)
st.plotly_chart(fig1, use_container_width=True)

slope_all = compute_slope(
    [m for (c, y, p), m in metrics_dict.items() if c in selected and p == state["price_mode"]],
    x_key=x_key,
    y_key=y_key,
    exclude_outliers=state["exclude_2022"],
)
render_commentary(comment_regression(slope_all, x_name=x_key, y_name=y_key))

section("Price duration curve", "Prix observe")
country = st.selectbox("Pays PDC", selected, key="pdc_country_v3")
pairs = sorted([k for k in proc.keys() if k[0] == country and k[4] == state["price_mode"]])
fig2 = go.Figure()
for k in pairs:
    yy = proc[k]["price_da_eur_mwh"].dropna().sort_values(ascending=False).values
    if len(yy) == 0:
        continue
    xx = np.linspace(0, 100, len(yy))
    fig2.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=str(k[1])))
fig2.update_layout(height=400, xaxis_title="% du temps", yaxis_title="EUR/MWh")
st.plotly_chart(fig2, use_container_width=True)

render_commentary(
    commentary_block(
        title="Courbe de duree des prix",
        n_label="annees",
        n_value=len(pairs),
        observed={"price_p95_latest": float(metrics_dict[(country, pairs[-1][1], state["price_mode"])]["price_used_p95"])},
        method_link="Les quantiles de prix_used sont calcules annuellement (G.7).",
        limits="La courbe est sensible aux episodes extremes (crises commodites, outliers).",
    )
)

section("Heatmap prix observe (mois x heure)", "Structure intra-annuelle")
year = st.selectbox("Annee Heatmap", [k[1] for k in pairs], key="hm_year_v3")
k = (country, year, state["must_run_mode"], state["flex_model_mode"], state["price_mode"])
if k in proc:
    d = proc[k]
    local = d.index.tz_convert(state["countries_cfg"][country]["timezone"])
    tmp = pd.DataFrame({"price": d["price_da_eur_mwh"].values, "month": local.month, "hour": local.hour})
    pivot = tmp.pivot_table(index="month", columns="hour", values="price", aggfunc="mean")
    fig3 = px.imshow(pivot, aspect="auto", labels={"x": "Heure", "y": "Mois", "color": "EUR/MWh"})
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    render_commentary(
        commentary_block(
            title="Saisonnalite horaire des prix",
            n_label="cellules",
            n_value=int(pivot.size),
            observed={"price_min": float(np.nanmin(pivot.values)), "price_max": float(np.nanmax(pivot.values))},
            method_link="Aggregation locale mois x heure sur prix observes.",
            limits="Moyennes horaires masquent la variabilite intra-heure et les evenements rares.",
        )
    )
