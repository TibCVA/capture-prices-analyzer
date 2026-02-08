"""Page 0 - Comprendre le modele."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.commentary_engine import commentary_block
from src.constants import COL_LOAD, COL_MUST_RUN, COL_NRL, COL_VRE
from src.ui_helpers import guard_no_data, inject_global_css, render_commentary, section

st.set_page_config(page_title="Comprendre le modele", page_icon="📖", layout="wide")
inject_global_css()

st.title("📖 Comprendre le modele")

state = st.session_state.get("state")
if not state or not state.get("data_loaded"):
    guard_no_data("la page Comprendre le modele")

section("Cadre methodologique", "Definitions v3.0")
st.markdown(
    "- La penetration RES est definie en % de generation annuelle totale (pas en % de demande).\n"
    "- NRL = load - VRE - must-run.\n"
    "- Regimes classes sur variables physiques uniquement (anti-circularite).\n"
    "- La variable load ENTSO-E exclut l'energie absorbee pour stockage/pompage; le pompage est donc traite comme flex."
)

render_commentary(
    commentary_block(
        title="Definition du cadre",
        n_label="regles",
        n_value=4,
        observed={"penetration_base": 100, "regimes": 4},
        method_link="Conventions A/B/C/D, SR/FAR/IR/TTL et definitions de la spec v3.0.",
        limits="Le cadre explique la logique; il ne remplace pas une calibration des donnees pays.",
    )
)

section("Exemple 48h", "Load, VRE, Must-run, NRL")

proc = state["processed"]
if not proc:
    guard_no_data("la page Comprendre le modele")

first_key = sorted(proc.keys())[0]
df = proc[first_key].head(48)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[COL_LOAD], name="Load", line=dict(color="#111827")))
fig.add_trace(go.Scatter(x=df.index, y=df[COL_VRE], name="VRE", line=dict(color="#16a34a")))
fig.add_trace(go.Scatter(x=df.index, y=df[COL_MUST_RUN], name="Must-run", line=dict(color="#6b7280")))
fig.add_trace(go.Scatter(x=df.index, y=df[COL_NRL], name="NRL", line=dict(color="#dc2626", dash="dash")))
fig.add_hline(y=0, line_dash="dot", line_color="#334155")
fig.update_layout(height=420, xaxis_title="Heure", yaxis_title="MW")
st.plotly_chart(fig, use_container_width=True)

n_hours = len(df)
n_neg = int((df[COL_NRL] < 0).sum())
render_commentary(
    commentary_block(
        title="Lecture exemple 48h",
        n_label="heures",
        n_value=n_hours,
        observed={"h_nrl_negatives": n_neg, "nrl_min_mw": float(df[COL_NRL].min())},
        method_link="Le surplus brut est defini par max(0, -NRL), puis absorbe via flex non-BESS + BESS.",
        limits="Illustration locale (48h) non representative de la distribution annuelle complete.",
    )
)
