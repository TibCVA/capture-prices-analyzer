"""
Page 3 - Capture Rates
Analyse des taux de capture PV/Wind : scatter avec regression,
Price Duration Curve, et heatmap prix mois x heure.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.constants import *
from src.slope_analysis import compute_slope
from src.ui_helpers import inject_global_css, narrative, guard_no_data, dynamic_narrative, challenge_block

st.set_page_config(page_title="Capture Rates", page_icon="📈", layout="wide")
inject_global_css()
st.title("📈 Capture Rates")

# ---------- Guard: donnees chargees ? ----------
if "annual_metrics" not in st.session_state or not st.session_state.annual_metrics:
    guard_no_data("les Capture Rates")

annual_metrics: dict = st.session_state.annual_metrics
processed_data: dict = st.session_state.get("processed_data", {})
exclude_2022: bool = st.session_state.get("exclude_2022", True)

# ---------- Selecteurs ----------
col_s1, col_s2, col_s3 = st.columns([3, 2, 2])

with col_s1:
    available_countries = sorted({k[0] for k in annual_metrics.keys()})
    if not available_countries:
        st.warning("Aucun pays disponible.")
        st.stop()
    default_countries = [c for c in st.session_state.get("selected_countries", [])
                         if c in available_countries] or available_countries[:3]
    countries = st.multiselect(
        "Pays",
        available_countries,
        default=default_countries,
    )

with col_s2:
    tech_type = st.radio("Technologie", ["PV", "Wind"], horizontal=True)

with col_s3:
    show_regression = st.toggle("Afficher regression", value=True)

if not countries:
    st.warning("Selectionnez au moins un pays.")
    st.stop()

# Definir les cles selon la techno
if tech_type == "PV":
    share_key = "pv_share"
    capture_key = "capture_ratio_pv"
    share_label = "Part PV (%)"
    capture_label = "Capture Ratio PV"
else:
    share_key = "wind_share"
    capture_key = "capture_ratio_wind"
    share_label = "Part Eolien (%)"
    capture_label = "Capture Ratio Wind"

# ============================================================
# Chart 1 — Scatter : share vs capture ratio, colore par pays
# ============================================================
st.subheader(f"{share_label} vs {capture_label}")

narrative("Ce graphique est la piece maitresse de l'analyse. La theorie (Hirth 2013) predit "
          "une relation negative entre penetration VRE et capture ratio : plus il y a de VRE, "
          "plus le capture ratio baisse. La pente de la regression quantifie la vitesse de "
          "cannibalisation. Un R2 eleve confirme que la penetration est le facteur dominant.")

scatter_rows = []
for (c, y), m in annual_metrics.items():
    if c not in countries:
        continue
    share_val = m.get(share_key)
    capture_val = m.get(capture_key)
    if share_val is None or capture_val is None:
        continue
    if isinstance(share_val, float) and np.isnan(share_val):
        continue
    if isinstance(capture_val, float) and np.isnan(capture_val):
        continue
    is_outlier = (y in OUTLIER_YEARS and exclude_2022)
    scatter_rows.append({
        "country": c,
        "year": y,
        "share": share_val,
        "capture": capture_val,
        "label": f"{c} {y}",
        "is_outlier": is_outlier,
    })

if scatter_rows:
    df_scatter = pd.DataFrame(scatter_rows)

    # Palette de couleurs par pays (cohérente sur toutes les pages)
    country_colors = {c: COUNTRY_PALETTE.get(c, "#999999") for c in sorted(countries)}

    fig1 = go.Figure()

    for c in sorted(countries):
        df_c = df_scatter[df_scatter["country"] == c]
        if df_c.empty:
            continue

        # Points normaux
        df_normal = df_c[~df_c["is_outlier"]]
        df_outlier = df_c[df_c["is_outlier"]]

        if not df_normal.empty:
            fig1.add_trace(
                go.Scatter(
                    x=df_normal["share"],
                    y=df_normal["capture"],
                    mode="markers+text",
                    name=c,
                    text=df_normal["year"].astype(str),
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(size=10, color=country_colors[c]),
                )
            )

        # Points outlier (2022) — croix grise
        if not df_outlier.empty:
            fig1.add_trace(
                go.Scatter(
                    x=df_outlier["share"],
                    y=df_outlier["capture"],
                    mode="markers+text",
                    name=f"{c} (2022)",
                    text=df_outlier["year"].astype(str),
                    textposition="top center",
                    textfont=dict(size=9, color="gray"),
                    marker=dict(size=10, color="lightgray", symbol="x"),
                    showlegend=False,
                )
            )

        # Regression
        if show_regression:
            metrics_list = [
                {**annual_metrics[(c, y)], "is_outlier": y in OUTLIER_YEARS}
                for y in sorted({k[1] for k in annual_metrics if k[0] == c})
                if (c, y) in annual_metrics
            ]
            reg = compute_slope(metrics_list, share_key, capture_key, exclude_outliers=exclude_2022)

            if reg["n_points"] >= 3 and not np.isnan(reg["slope"]):
                x_reg = np.array(reg["x_values"])
                x_line = np.linspace(x_reg.min(), x_reg.max(), 50)
                y_line = reg["slope"] * x_line + reg["intercept"]

                fig1.add_trace(
                    go.Scatter(
                        x=x_line, y=y_line,
                        mode="lines",
                        name=f"{c} reg (R²={reg['r_squared']:.3f})",
                        line=dict(color=country_colors[c], dash="dash", width=1.5),
                    )
                )

    fig1.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title=share_label,
        yaxis_title=capture_label,
        height=500,
    )

    # Ligne horizontale a capture = 1
    fig1.add_hline(y=1.0, line_dash="dot", line_color="gray",
                   annotation_text="Capture = 1", annotation_position="bottom right")

    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Aucune donnee de scatter disponible pour la selection.")

# ============================================================
# Chart 2 — Price Duration Curve
# ============================================================
st.subheader("Price Duration Curve")

narrative("La courbe de duree classe les 8760 heures de l'annee de la plus chere a la moins chere. "
          "Comparer les courbes d'annees differentes montre comment la structure des prix evolue.")

col_pdc1, col_pdc2 = st.columns([2, 4])
with col_pdc1:
    pdc_country = st.selectbox("Pays (PDC)", countries, index=0, key="pdc_country")

if pdc_country:
    pdc_years = sorted({k[1] for k in processed_data if k[0] == pdc_country})

    if pdc_years:
        fig2 = go.Figure()

        year_colors = px.colors.sample_colorscale(
            "Viridis", [i / max(len(pdc_years) - 1, 1) for i in range(len(pdc_years))]
        )

        for i, y in enumerate(pdc_years):
            key = (pdc_country, y)
            if key not in processed_data:
                continue
            df_y = processed_data[key]
            if COL_PRICE_DA not in df_y.columns:
                continue

            prices_sorted = df_y[COL_PRICE_DA].dropna().sort_values(ascending=False).reset_index(drop=True)
            # X = rang en pourcentage du temps
            x_pct = np.linspace(0, 100, len(prices_sorted))

            fig2.add_trace(
                go.Scatter(
                    x=x_pct, y=prices_sorted.values,
                    mode="lines",
                    name=str(y),
                    line=dict(color=year_colors[i], width=1.5),
                )
            )

        fig2.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            xaxis_title="% du temps (heures triees par prix decroissant)",
            yaxis_title="Prix DA (EUR/MWh)",
            height=450,
        )

        # Limiter y pour lisibilite (couper les extremes)
        fig2.update_yaxes(range=[-50, 300])

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"Aucune donnee horaire pour {pdc_country}.")
else:
    st.info("Selectionnez un pays pour la Price Duration Curve.")

# ============================================================
# Chart 3 — Heatmap : mois x heure, prix moyen
# ============================================================
st.subheader("Heatmap des prix moyens (mois x heure, heure locale)")

narrative("Le creux bleu (prix bas) visible en milieu de journee en ete est la signature du "
          "solaire. S'il s'etend au printemps et a l'automne, le PV devient dominant plus longtemps. "
          "Les pointes rouges en hiver aux heures 8h et 18h signalent la tension demande/offre.")

col_hm1, col_hm2 = st.columns([2, 4])
with col_hm1:
    hm_country = st.selectbox("Pays (Heatmap)", countries, index=0, key="hm_country")
    hm_years = sorted({k[1] for k in processed_data if k[0] == hm_country})
    hm_year = st.selectbox("Annee (Heatmap)", hm_years, index=len(hm_years) - 1 if hm_years else 0, key="hm_year")

hm_key = (hm_country, hm_year)
if hm_key in processed_data:
    df_hm = processed_data[hm_key].copy()

    if COL_PRICE_DA in df_hm.columns:
        # Convertir en heure locale
        tz_local = COUNTRY_TZ.get(hm_country, "UTC")

        if COL_TS in df_hm.columns:
            ts = pd.to_datetime(df_hm[COL_TS])
        else:
            ts = pd.to_datetime(df_hm.index)

        # Normaliser en Series pour acces .dt uniforme
        if isinstance(ts, pd.DatetimeIndex):
            ts = ts.to_series()
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        ts_local = ts.dt.tz_convert(tz_local)

        df_hm["month"] = ts_local.dt.month.values
        df_hm["hour"] = ts_local.dt.hour.values

        # Pivot : mois en lignes, heures en colonnes
        heatmap_data = df_hm.pivot_table(
            values=COL_PRICE_DA,
            index="month",
            columns="hour",
            aggfunc="mean",
        )

        # Labels mois
        month_labels = ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
                        "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"]
        heatmap_data.index = [month_labels[m - 1] for m in heatmap_data.index]

        fig3 = px.imshow(
            heatmap_data,
            labels=dict(x="Heure (locale)", y="Mois", color="EUR/MWh"),
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=heatmap_data.values[~np.isnan(heatmap_data.values)].mean()
            if not np.all(np.isnan(heatmap_data.values))
            else 50,
            aspect="auto",
        )

        fig3.update_layout(
            **PLOTLY_LAYOUT_DEFAULTS,
            height=450,
            xaxis=dict(dtick=1),
        )

        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning(f"Colonne {COL_PRICE_DA} absente.")
else:
    st.info(f"Pas de donnees pour {hm_country} {hm_year}.")

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("**Pour aller plus loin** : Comparaison Pays (radar multi-marche) | Scenarios (prospectif)")
