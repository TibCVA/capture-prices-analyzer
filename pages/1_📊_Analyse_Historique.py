"""
Page 1 - Analyse Historique
Visualisation des metriques annuelles : heures negatives, capture ratio,
repartition des regimes, et tableau recapitulatif.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.constants import *
from src.ui_helpers import inject_global_css, narrative, guard_no_data, section_header, dynamic_narrative, challenge_block

st.set_page_config(page_title="Analyse Historique", page_icon="📊", layout="wide")
inject_global_css()
st.title("📊 Analyse Historique")

# ---------- Guard: donnees chargees ? ----------
if "annual_metrics" not in st.session_state or not st.session_state.annual_metrics:
    guard_no_data("l'Analyse Historique")

annual_metrics: dict = st.session_state.annual_metrics
selected_countries: list = st.session_state.get("selected_countries", [])
year_range: tuple = st.session_state.get("year_range", (2015, 2024))

# ---------- Selecteurs ----------
col_sel1, col_sel2, col_sel3 = st.columns([2, 3, 2])

with col_sel1:
    available_countries = sorted({k[0] for k in annual_metrics.keys()})
    if not available_countries:
        guard_no_data("l'Analyse Historique")
    country = st.selectbox("Pays", available_countries, index=0)

with col_sel2:
    all_years = sorted({k[1] for k in annual_metrics.keys() if k[0] == country})
    if len(all_years) >= 2:
        period = st.slider(
            "Periode",
            min_value=min(all_years),
            max_value=max(all_years),
            value=(min(all_years), max(all_years)),
        )
    else:
        period = (all_years[0], all_years[0]) if all_years else year_range
        st.write(f"Periode : {period[0]}")

with col_sel3:
    exclude_2022 = st.toggle("Exclure 2022 (crise gaziere)", value=st.session_state.get("exclude_2022", True))

# ---------- Filtrer les metriques ----------
years_in_range = [y for y in all_years if period[0] <= y <= period[1]]
metrics_series = []
for y in years_in_range:
    key = (country, y)
    if key in annual_metrics:
        m = annual_metrics[key].copy() if isinstance(annual_metrics[key], dict) else annual_metrics[key]
        m["year"] = y
        m["is_excluded"] = (y == 2022 and exclude_2022)
        metrics_series.append(m)

if not metrics_series:
    guard_no_data("l'Analyse Historique")

df_metrics = pd.DataFrame(metrics_series)

# ============================================================
# Chart 1 — Combo : barres empilees + lignes sur axe secondaire
# ============================================================
section_header(
    "Heures negatives / basses & Capture Ratio PV + Part VRE",
    "Axe gauche : nombre d'heures | Axe droit : ratios et pourcentages",
)
st.caption("Barres = nombre d'heures | Lignes = ratios sur axe droit")

narrative(
    "Ce graphique teste la these centrale du modele : la penetration VRE croissante "
    "entraine-t-elle une augmentation des heures a prix negatif et une baisse du capture "
    "ratio ? Si les barres rouges montent tandis que la ligne verte (capture ratio) descend, "
    "la cannibalisation est confirmee. L'acceleration des barres = indicateur de seuil de saturation."
)

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# Barres : heures negatives et heures < 5 EUR
colors_neg = ["#e74c3c" if not exc else "#d5d5d5" for exc in df_metrics["is_excluded"]]
colors_low = ["#f39c12" if not exc else "#d5d5d5" for exc in df_metrics["is_excluded"]]

fig1.add_trace(
    go.Bar(
        x=df_metrics["year"],
        y=df_metrics.get("h_negative", pd.Series(dtype=float)),
        name="Heures prix < 0",
        marker_color=colors_neg,
        offsetgroup=0,
    ),
    secondary_y=False,
)
fig1.add_trace(
    go.Bar(
        x=df_metrics["year"],
        y=df_metrics.get("h_below_5", pd.Series(dtype=float)),
        name="Heures prix < 5 EUR",
        marker_color=colors_low,
        offsetgroup=0,
        base=df_metrics.get("h_negative", pd.Series(dtype=float)),
    ),
    secondary_y=False,
)

# Lignes : capture ratio PV + VRE share
fig1.add_trace(
    go.Scatter(
        x=df_metrics["year"],
        y=df_metrics.get("capture_ratio_pv", pd.Series(dtype=float)),
        name="Capture Ratio PV",
        mode="lines+markers",
        line=dict(color="#2ecc71", width=2),
        marker=dict(size=7),
    ),
    secondary_y=True,
)
fig1.add_trace(
    go.Scatter(
        x=df_metrics["year"],
        y=df_metrics.get("vre_share", pd.Series(dtype=float)),
        name="Part VRE (%)",
        mode="lines+markers",
        line=dict(color="#3498db", width=2, dash="dash"),
        marker=dict(size=7),
    ),
    secondary_y=True,
)

fig1.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    barmode="stack",
    xaxis=dict(title="Annee", dtick=1),
    height=480,
)
fig1.update_yaxes(title_text="Heures", secondary_y=False)
fig1.update_yaxes(title_text="Ratio / %", secondary_y=True)

st.plotly_chart(fig1, use_container_width=True)

# Dynamic interpretation
if country:
    last_yr = max([y for (cc, y) in annual_metrics if cc == country], default=None)
    if last_yr:
        m = annual_metrics.get((country, last_yr), {})
        cr = m.get('capture_ratio_pv')
        if cr and cr == cr and cr < 0.80:
            dynamic_narrative(
                f"<strong>{country} ({last_yr})</strong> : Capture ratio PV = {cr:.2f} -- "
                f"le seuil de cannibalisation significative (0.80) est franchi.",
                "warning")

# ============================================================
# Chart 2 — Barres empilees 100 % : regimes A / B / C / D_tail
# ============================================================
section_header(
    "Repartition des regimes de prix (% des heures)",
    "Proportion annuelle de chaque regime (A = surplus, D = tension)",
)

narrative(
    "La progression du regime A (rouge, surplus non absorbe) est le signe le plus direct "
    "de la saturation du systeme. Les criteres de Phase 2 incluent : h negatives >= 200, "
    "capture ratio PV < 0.80, h prix < 5 EUR >= 500 (voir page Comprendre le Modele)."
)

REGIME_COLORS = {
    "A": "#e74c3c",
    "B": "#f39c12",
    "C": "#3498db",
    "D_tail": "#9b59b6",
}
REGIME_KEYS = {
    "A": "h_regime_a",
    "B": "h_regime_b",
    "C": "h_regime_c",
    "D_tail": "h_regime_d_tail",
}

# Calculer les totaux et pourcentages
regime_data = []
for _, row in df_metrics.iterrows():
    total = sum(row.get(v, 0) or 0 for v in REGIME_KEYS.values())
    total = max(total, 1)  # eviter div/0
    entry = {"year": row["year"], "is_excluded": row["is_excluded"]}
    for regime_name, col_key in REGIME_KEYS.items():
        entry[regime_name] = ((row.get(col_key, 0) or 0) / total) * 100
    regime_data.append(entry)

df_regimes = pd.DataFrame(regime_data)

fig2 = go.Figure()
for regime_name in ["A", "B", "C", "D_tail"]:
    bar_colors = [
        REGIME_COLORS[regime_name] if not exc else "#d5d5d5"
        for exc in df_regimes["is_excluded"]
    ]
    fig2.add_trace(
        go.Bar(
            x=df_regimes["year"],
            y=df_regimes[regime_name],
            name=f"Regime {regime_name}",
            marker_color=bar_colors,
        )
    )

fig2.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    barmode="stack",
    xaxis=dict(title="Annee", dtick=1),
    yaxis=dict(title="% des heures", range=[0, 100]),
    height=420,
)

st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# Chart 3 — Tableau recapitulatif des metriques annuelles
# ============================================================
section_header(
    "Tableau des metriques annuelles",
    "Toutes les valeurs cles par annee",
)

narrative(
    "Toutes les metriques annuelles en un coup d'oeil. Les colonnes ratio et "
    "coherence sont colorees : vert = valeur favorable, rouge = signal d'alerte."
)

# Colonnes a afficher (filtrer celles qui existent)
display_cols = [
    "year", "h_negative", "h_below_5", "h_regime_a", "h_regime_b",
    "h_regime_c", "h_regime_d_tail", "capture_ratio_pv", "capture_ratio_wind",
    "vre_share", "pv_share", "wind_share", "mean_price", "median_price",
    "p10_price", "p90_price", "regime_coherence",
]
available_cols = [c for c in display_cols if c in df_metrics.columns]

df_display = df_metrics[available_cols].copy()
df_display = df_display.set_index("year") if "year" in available_cols else df_display

# Formatage
format_dict = {}
for col in df_display.columns:
    if "ratio" in col or "share" in col or "coherence" in col:
        format_dict[col] = "{:.2%}"
    elif "price" in col:
        format_dict[col] = "{:.1f}"
    elif col.startswith("h_"):
        format_dict[col] = "{:.0f}"

st.dataframe(
    df_display.style.format(format_dict, na_rep="—").background_gradient(
        subset=[c for c in df_display.columns if "ratio" in c or "coherence" in c],
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
    ),
    use_container_width=True,
    height=400,
)

# ---------- Navigation hint ----------
st.divider()
st.caption("**Pour aller plus loin** : NRL Deep Dive (zoom horaire) | Capture Rates (regressions)")
