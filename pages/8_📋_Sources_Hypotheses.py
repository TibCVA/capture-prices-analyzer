"""Page 8 - Sources et hypotheses (editable)."""

from __future__ import annotations

import streamlit as st

from src.commentary_engine import so_what_block
from src.constants import (
    EF_COAL,
    EF_GAS,
    EF_LIGNITE,
    ETA_CCGT,
    ETA_COAL,
    ETA_LIGNITE,
    ETA_OCGT,
    PRICE_HIGH_THRESHOLD,
    PRICE_NEGATIVE_THRESHOLD,
    PRICE_VERY_HIGH_THRESHOLD,
    PRICE_VERY_LOW_THRESHOLD,
    SPREAD_DAILY_THRESHOLD,
    VOM_CCGT,
    VOM_COAL,
    VOM_LIGNITE,
    VOM_OCGT,
)
from src.ui_helpers import inject_global_css, narrative, render_commentary, section_header

st.set_page_config(page_title="Sources & hypotheses", page_icon="📋", layout="wide")
inject_global_css()

st.title("📋 Sources & Hypotheses")

state = st.session_state.get("state")
if state is None:
    st.session_state.state = {}
    state = st.session_state.state

if "ui_overrides" not in state or not isinstance(state["ui_overrides"], dict):
    state["ui_overrides"] = {}

ov = state["ui_overrides"]

narrative(
    "Cette page recense les sources et permet de modifier des hypotheses de travail. "
    "Les modifications s'appliquent au prochain clic 'Charger donnees' depuis l'accueil."
)

section_header("Sources de donnees", "Tracabilite du pipeline")
st.markdown(
    """
| Source | Donnees | Granularite | Fichier/Origine |
|---|---|---|---|
| ENTSO-E Transparency | load, generation, DA price, net position | horaire | API + cache `data/raw/` |
| TTF | prix gaz | journalier | `data/external/ttf_daily.csv` |
| EUA | prix CO2 | journalier | `data/external/eua_daily.csv` |
| Coal (optionnel) | prix charbon | journalier | `data/external/coal_daily.csv` |
| BESS capacities | puissance/energie par pays | annuel | `data/external/bess_capacity.csv` |
| Configurations | pays/scenarios/seuils | statique | `config/*.yaml` |
    """
)

section_header("Hypotheses modifiables", "Utiles pour sensibilite et calibration")

tab_therm, tab_bess, tab_price = st.tabs(["Thermique", "BESS", "Seuils prix"])

with tab_therm:
    st.markdown("#### Rendements thermiques")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ov["eta_ccgt"] = st.number_input("eta_ccgt", min_value=0.40, max_value=0.70, value=float(ov.get("eta_ccgt", ETA_CCGT)), step=0.01)
    with c2:
        ov["eta_ocgt"] = st.number_input("eta_ocgt", min_value=0.20, max_value=0.50, value=float(ov.get("eta_ocgt", ETA_OCGT)), step=0.01)
    with c3:
        ov["eta_coal"] = st.number_input("eta_coal", min_value=0.20, max_value=0.50, value=float(ov.get("eta_coal", ETA_COAL)), step=0.01)
    with c4:
        ov["eta_lignite"] = st.number_input("eta_lignite", min_value=0.20, max_value=0.45, value=float(ov.get("eta_lignite", ETA_LIGNITE)), step=0.01)

    st.markdown("#### Facteurs d'emission")
    c1, c2, c3 = st.columns(3)
    with c1:
        ov["ef_gas"] = st.number_input("ef_gas", min_value=0.10, max_value=0.30, value=float(ov.get("ef_gas", EF_GAS)), step=0.001)
    with c2:
        ov["ef_coal"] = st.number_input("ef_coal", min_value=0.20, max_value=0.45, value=float(ov.get("ef_coal", EF_COAL)), step=0.001)
    with c3:
        ov["ef_lignite"] = st.number_input("ef_lignite", min_value=0.20, max_value=0.50, value=float(ov.get("ef_lignite", EF_LIGNITE)), step=0.001)

    st.markdown("#### VOM")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ov["vom_ccgt"] = st.number_input("vom_ccgt", min_value=0.0, max_value=15.0, value=float(ov.get("vom_ccgt", VOM_CCGT)), step=0.5)
    with c2:
        ov["vom_ocgt"] = st.number_input("vom_ocgt", min_value=0.0, max_value=15.0, value=float(ov.get("vom_ocgt", VOM_OCGT)), step=0.5)
    with c3:
        ov["vom_coal"] = st.number_input("vom_coal", min_value=0.0, max_value=15.0, value=float(ov.get("vom_coal", VOM_COAL)), step=0.5)
    with c4:
        ov["vom_lignite"] = st.number_input("vom_lignite", min_value=0.0, max_value=15.0, value=float(ov.get("vom_lignite", VOM_LIGNITE)), step=0.5)

with tab_bess:
    st.markdown("#### Parametres BESS de simulation")
    c1, c2, c3 = st.columns(3)
    with c1:
        ov["bess_eta_charge"] = st.number_input(
            "bess_eta_charge", min_value=0.50, max_value=1.00, value=float(ov.get("bess_eta_charge", 0.95)), step=0.01
        )
    with c2:
        ov["bess_eta_discharge"] = st.number_input(
            "bess_eta_discharge", min_value=0.50, max_value=1.00, value=float(ov.get("bess_eta_discharge", 0.95)), step=0.01
        )
    with c3:
        ov["bess_soc_init_frac"] = st.number_input(
            "bess_soc_init_frac", min_value=0.00, max_value=1.00, value=float(ov.get("bess_soc_init_frac", 0.50)), step=0.01
        )

with tab_price:
    st.markdown("#### Seuils observables prix")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        ov["price_negative_threshold"] = st.number_input(
            "price_negative_threshold", min_value=-500.0, max_value=100.0, value=float(ov.get("price_negative_threshold", PRICE_NEGATIVE_THRESHOLD)), step=1.0
        )
    with c2:
        ov["price_very_low_threshold"] = st.number_input(
            "price_very_low_threshold", min_value=-100.0, max_value=100.0, value=float(ov.get("price_very_low_threshold", PRICE_VERY_LOW_THRESHOLD)), step=1.0
        )
    with c3:
        ov["price_high_threshold"] = st.number_input(
            "price_high_threshold", min_value=0.0, max_value=500.0, value=float(ov.get("price_high_threshold", PRICE_HIGH_THRESHOLD)), step=5.0
        )
    with c4:
        ov["price_very_high_threshold"] = st.number_input(
            "price_very_high_threshold", min_value=0.0, max_value=5000.0, value=float(ov.get("price_very_high_threshold", PRICE_VERY_HIGH_THRESHOLD)), step=10.0
        )
    with c5:
        ov["spread_daily_threshold"] = st.number_input(
            "spread_daily_threshold", min_value=0.0, max_value=500.0, value=float(ov.get("spread_daily_threshold", SPREAD_DAILY_THRESHOLD)), step=5.0
        )

state["ui_overrides"] = ov

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Reinitialiser hypotheses"):
        state["ui_overrides"] = {}
        st.rerun()
with c2:
    st.info("Les changements sont pris en compte au prochain 'Charger donnees' depuis la page d'accueil.")

section_header("Conventions fixes (non modifiables)", "Pour garder la comparabilite v3")
st.markdown(
    """
- Regimes classes uniquement sur variables physiques (anti-circularite).
- Interdiction stricte de l'approximation exports = generation - load.
- Penetration VRE mesuree en % de generation (pas % demande).
- Scenarios deterministes (pas de random, pas de Monte Carlo).
    """
)

render_commentary(
    so_what_block(
        title="Cadre des hypotheses",
        purpose="Permettre des sensibilites explicites sans casser la trame methodologique v3",
        observed={"nb_overrides_actifs": len(state.get("ui_overrides", {})), "regles_fixes": 4},
        method_link="Overrides session-scopes appliques au recalcul; defaults constants/YAML restent references.",
        limits="Les overrides sont exploratoires; ils ne remplacent pas une calibration gouvernee.",
        n=1,
    )
)
