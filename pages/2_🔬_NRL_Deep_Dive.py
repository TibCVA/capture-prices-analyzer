"""
Page 2 - NRL Deep Dive
Analyse detaillee de la Net Residual Load : courbe temporelle,
distribution, scatter NRL vs prix, et score de coherence des regimes.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.constants import *
from src.ui_helpers import inject_global_css, narrative, guard_no_data, section_header, dynamic_narrative, challenge_block

st.set_page_config(page_title="NRL Deep Dive", page_icon="🔬", layout="wide")
inject_global_css()
st.title("🔬 NRL Deep Dive")

# ---------- Guard: donnees chargees ? ----------
if "processed_data" not in st.session_state or not st.session_state.processed_data:
    guard_no_data("le NRL Deep Dive")

processed_data: dict = st.session_state.processed_data
annual_metrics: dict = st.session_state.get("annual_metrics", {})

# ---------- Selecteurs ----------
col_s1, col_s2, col_s3 = st.columns([2, 2, 2])

with col_s1:
    available_countries = sorted({k[0] for k in processed_data.keys()})
    if not available_countries:
        guard_no_data("le NRL Deep Dive")
    country = st.selectbox("Pays", available_countries, index=0)

with col_s2:
    available_years = sorted({k[1] for k in processed_data.keys() if k[0] == country})
    if not available_years:
        guard_no_data("le NRL Deep Dive")
    year = st.selectbox("Annee", available_years, index=len(available_years) - 1)

with col_s3:
    month_options = ["Annee complete"] + [
        f"{m:02d} - {name}"
        for m, name in enumerate(
            ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
             "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"],
            start=1,
        )
    ]
    zoom = st.selectbox("Zoom", month_options, index=0)

# ---------- Charger les donnees ----------
key = (country, year)
if key not in processed_data:
    st.error(f"Pas de donnees pour {country} {year}.")
    st.stop()

df: pd.DataFrame = processed_data[key].copy()

# Assurer que le timestamp est en datetime
if COL_TS in df.columns and not pd.api.types.is_datetime64_any_dtype(df[COL_TS]):
    df[COL_TS] = pd.to_datetime(df[COL_TS])

# Filtrage par mois si necessaire
if zoom != "Annee complete":
    month_num = int(zoom.split(" - ")[0])
    if COL_TS in df.columns:
        df = df[df[COL_TS].dt.month == month_num]
    else:
        df = df[df.index.month == month_num] if hasattr(df.index, "month") else df

if df.empty:
    guard_no_data("le NRL Deep Dive")

# Axe x : timestamp ou index
x_col = df[COL_TS] if COL_TS in df.columns else df.index

# ============================================================
# Chart 1 — Area chart NRL decomposition + prix secondaire
# ============================================================
section_header(
    "Decomposition de la charge et NRL",
    "Axe gauche : MW (zones empilees) | Axe droit : EUR/MWh (prix DA)",
)

narrative(
    "Ce graphique decompose la question fondamentale : a chaque heure, qui fournit "
    "l'electricite ? Quand les zones verte (VRE) et grise (must-run) depassent la ligne "
    "noire (demande), le NRL devient negatif et les prix s'effondrent. C'est le mecanisme "
    "physique de la cannibalisation. Cherchez : les creux en milieu de journee (effet solaire) "
    "et les periodes prolongees de NRL negatif (rigidite du systeme)."
)

show_price = st.toggle("Superposer le prix DA", value=False, key="nrl_price_toggle")

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# Must-run (gris)
if COL_MUST_RUN in df.columns:
    fig1.add_trace(
        go.Scatter(
            x=x_col, y=df[COL_MUST_RUN],
            name="Must-Run",
            fill="tozeroy",
            fillcolor="rgba(180,180,180,0.4)",
            line=dict(color="rgba(150,150,150,0.6)", width=0.5),
        ),
        secondary_y=False,
    )

# VRE (vert)
if COL_VRE in df.columns:
    fig1.add_trace(
        go.Scatter(
            x=x_col, y=df[COL_VRE],
            name="VRE",
            fill="tozeroy",
            fillcolor="rgba(46,204,113,0.35)",
            line=dict(color="rgba(39,174,96,0.6)", width=0.5),
        ),
        secondary_y=False,
    )

# Load (noir)
if COL_LOAD in df.columns:
    fig1.add_trace(
        go.Scatter(
            x=x_col, y=df[COL_LOAD],
            name="Load",
            line=dict(color="black", width=1.2),
        ),
        secondary_y=False,
    )

# NRL (bleu > 0, rouge < 0)
if COL_NRL in df.columns:
    nrl = df[COL_NRL]
    nrl_pos = nrl.clip(lower=0)
    nrl_neg = nrl.clip(upper=0)

    fig1.add_trace(
        go.Scatter(
            x=x_col, y=nrl_pos,
            name="NRL > 0",
            line=dict(color="#2980b9", width=1.5),
        ),
        secondary_y=False,
    )
    fig1.add_trace(
        go.Scatter(
            x=x_col, y=nrl_neg,
            name="NRL < 0",
            line=dict(color="#e74c3c", width=1.5),
        ),
        secondary_y=False,
    )

# Prix DA (axe secondaire, rouge) — conditionne par le toggle
if show_price and COL_PRICE_DA in df.columns:
    fig1.add_trace(
        go.Scatter(
            x=x_col, y=df[COL_PRICE_DA],
            name="Prix DA (EUR/MWh)",
            line=dict(color="#c0392b", width=1, dash="dot"),
            opacity=0.7,
        ),
        secondary_y=True,
    )

fig1.update_layout(
    **PLOTLY_LAYOUT_DEFAULTS,
    height=520,
    hovermode="x unified",
)
fig1.update_yaxes(title_text="MW", secondary_y=False)
fig1.update_yaxes(title_text="EUR/MWh", secondary_y=True)

st.plotly_chart(fig1, use_container_width=True)

# ============================================================
# Chart 2 — Histogramme NRL colore par regime
# ============================================================
section_header(
    "Distribution de la NRL",
    "Axe x : NRL en MW | Axe y : nombre d'heures",
)

narrative(
    "La distribution de la NRL revele la structure du marche. A gauche du zero "
    "(rouge/orange) = heures de surplus. A droite (bleu/violet) = heures ou "
    "les centrales thermiques sont necessaires."
)

if COL_NRL in df.columns:
    # Calculer les bins a 1000 MW
    nrl_min = df[COL_NRL].min()
    nrl_max = df[COL_NRL].max()
    bin_start = int(np.floor(nrl_min / 1000) * 1000)
    bin_end = int(np.ceil(nrl_max / 1000) * 1000)
    nbins = max((bin_end - bin_start) // 1000, 10)

    regime_color_map = {
        "A": "#e74c3c",
        "B": "#f39c12",
        "C": "#3498db",
        "D_tail": "#9b59b6",
    }

    if COL_REGIME in df.columns:
        fig2 = px.histogram(
            df, x=COL_NRL,
            color=COL_REGIME,
            color_discrete_map=regime_color_map,
            nbins=nbins,
            labels={COL_NRL: "NRL (MW)", COL_REGIME: "Regime"},
            category_orders={COL_REGIME: ["A", "B", "C", "D_tail"]},
        )
    else:
        fig2 = px.histogram(df, x=COL_NRL, nbins=nbins, labels={COL_NRL: "NRL (MW)"})

    # Ligne verticale NRL = 0
    fig2.add_vline(x=0, line_dash="dash", line_color="black", line_width=2,
                   annotation_text="NRL = 0")

    fig2.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=400,
        xaxis_title="NRL (MW)",
        yaxis_title="Nombre d'heures",
        bargap=0.05,
    )

    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning(f"Colonne {COL_NRL} absente du jeu de donnees.")

# ============================================================
# Chart 3 — Scatter NRL vs Prix, colore par regime
# ============================================================
section_header(
    "NRL vs Prix Day-Ahead",
    "Axe x : NRL en MW | Axe y : prix DA en EUR/MWh",
)

narrative(
    "Ce scatter est le test empirique du modele : si les regimes sont correctement "
    "calibres, les couleurs doivent etre spatialement separees (rouge a gauche = surplus, "
    "bleu/violet a droite = thermique). Le chevauchement des couleurs revele les zones "
    "ou le modele et la realite divergent."
)

if COL_NRL in df.columns and COL_PRICE_DA in df.columns:
    if COL_REGIME in df.columns:
        fig3 = px.scatter(
            df, x=COL_NRL, y=COL_PRICE_DA,
            color=COL_REGIME,
            color_discrete_map={
                "A": "#e74c3c",
                "B": "#f39c12",
                "C": "#3498db",
                "D_tail": "#9b59b6",
            },
            opacity=0.3,
            labels={COL_NRL: "NRL (MW)", COL_PRICE_DA: "Prix DA (EUR/MWh)", COL_REGIME: "Regime"},
            category_orders={COL_REGIME: ["A", "B", "C", "D_tail"]},
        )
    else:
        fig3 = px.scatter(
            df, x=COL_NRL, y=COL_PRICE_DA,
            opacity=0.3,
            labels={COL_NRL: "NRL (MW)", COL_PRICE_DA: "Prix DA (EUR/MWh)"},
        )

    fig3.update_traces(marker=dict(size=4))
    fig3.update_layout(**PLOTLY_LAYOUT_DEFAULTS, height=450)

    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Colonnes NRL ou Prix DA absentes.")

# ============================================================
# Card — Score de coherence des regimes
# ============================================================
section_header(
    "Coherence des regimes",
    "Accord entre le regime physique (NRL) et le prix observe",
)

metrics_key = (country, year)
if metrics_key in annual_metrics:
    coherence = annual_metrics[metrics_key].get("regime_coherence")
    if coherence is not None and not (isinstance(coherence, float) and np.isnan(coherence)):
        pct = coherence * 100 if coherence <= 1 else coherence  # gerer % ou ratio

        if pct > 70:
            color = "🟢"
            status = "Bonne coherence"
            delta_color = "normal"
        elif pct >= 55:
            color = "🟠"
            status = "Coherence moyenne"
            delta_color = "off"
        else:
            color = "🔴"
            status = "Coherence faible"
            delta_color = "inverse"

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(
                label="Regime Coherence Score",
                value=f"{pct:.1f} %",
                delta=status,
                delta_color=delta_color,
                help="Pourcentage d'heures ou le regime physique (base sur la NRL) est coherent avec le prix observe. Cible > 55%.",
            )
        with col_m2:
            st.markdown(f"**Interpretation** : {color} {status}")
            st.caption(
                "Ce score mesure le pourcentage d'heures ou le regime de prix "
                "(A/B/C/D) est coherent avec le niveau de NRL observe."
            )
        with col_m3:
            # Nombre d'heures par regime
            regime_counts = {}
            if COL_REGIME in df.columns:
                regime_counts = df[COL_REGIME].value_counts().to_dict()
            for regime_name in ["A", "B", "C", "D_tail"]:
                count = regime_counts.get(regime_name, 0)
                st.write(f"**Regime {regime_name}** : {count:,} h")

        # Challenge block for low coherence
        if pct < 55:
            challenge_block(
                "Coherence faible",
                f"La coherence regime/prix est de {pct:.1f}%, en dessous du seuil de 55%. "
                f"Causes possibles : must-run mal calibre, exports non comptabilises, "
                f"mecanismes de capacite. Voir page Guide Utilisateur, section calibration.")
    else:
        st.info("Score de coherence non disponible pour cette annee.")
else:
    st.info("Metriques annuelles non disponibles pour cette selection.")

# ---------- Navigation hint ----------
st.divider()
st.caption("**Pour aller plus loin** : Capture Rates (regressions) | Comprendre le Modele (explications)")
