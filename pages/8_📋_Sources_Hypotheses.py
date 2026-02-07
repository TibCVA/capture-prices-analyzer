"""
Page 8 -- Sources de Donnees & Hypotheses du Modele
Inventaire des donnees, parametres modifiables, transparence.
"""
import streamlit as st
from src.constants import *
from src.ui_helpers import inject_global_css, narrative

st.set_page_config(page_title="Sources & Hypotheses", page_icon="📋", layout="wide")
inject_global_css()
st.title("📋 Sources de Donnees & Hypotheses")

narrative("Cette page recense les sources de donnees utilisees et les hypotheses standard du modele. "
          "Vous pouvez modifier les parametres ci-dessous ; les changements prendront effet "
          "au prochain chargement des donnees (bouton Charger sur la page d'accueil).")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 : SOURCES DE DONNEES
# ══════════════════════════════════════════════════════════════════════════
st.markdown("### Sources de donnees")

st.markdown("""
<table style="width:100%; border-collapse: collapse; font-size: 0.88rem;">
<tr style="background:#0066CC; color:white;">
    <th style="padding:8px; text-align:left;">Source</th>
    <th style="padding:8px; text-align:left;">Donnee</th>
    <th style="padding:8px; text-align:left;">Granularite</th>
    <th style="padding:8px; text-align:left;">Periode</th>
    <th style="padding:8px; text-align:left;">Reference</th>
</tr>
<tr style="background:#F8F9FB;">
    <td style="padding:8px;"><strong>ENTSO-E Transparency</strong></td>
    <td style="padding:8px;">Charge, generation par filiere, prix day-ahead, net position</td>
    <td style="padding:8px;">Horaire</td>
    <td style="padding:8px;">2015-2024</td>
    <td style="padding:8px;">transparency.entsoe.eu</td>
</tr>
<tr>
    <td style="padding:8px;"><strong>TTF (ICE/PEGAS)</strong></td>
    <td style="padding:8px;">Prix spot gaz naturel (Title Transfer Facility)</td>
    <td style="padding:8px;">Journalier</td>
    <td style="padding:8px;">2015-2024</td>
    <td style="padding:8px;">data/external/ttf_daily.csv</td>
</tr>
<tr style="background:#F8F9FB;">
    <td style="padding:8px;"><strong>EUA (ICE ECX)</strong></td>
    <td style="padding:8px;">Prix CO2 EU ETS (European Union Allowances)</td>
    <td style="padding:8px;">Journalier</td>
    <td style="padding:8px;">2015-2024</td>
    <td style="padding:8px;">data/external/eua_daily.csv</td>
</tr>
<tr>
    <td style="padding:8px;"><strong>BESS Capacity</strong></td>
    <td style="padding:8px;">Capacite installee batteries par pays</td>
    <td style="padding:8px;">Annuel</td>
    <td style="padding:8px;">2015-2024</td>
    <td style="padding:8px;">data/external/bess_capacity.csv</td>
</tr>
<tr style="background:#F8F9FB;">
    <td style="padding:8px;"><strong>Configuration pays</strong></td>
    <td style="padding:8px;">Must-run, flex capacities, tech marginale</td>
    <td style="padding:8px;">Statique</td>
    <td style="padding:8px;">--</td>
    <td style="padding:8px;">config/countries.yaml</td>
</tr>
</table>
""", unsafe_allow_html=True)

# Completude des donnees chargees
if st.session_state.get("annual_metrics"):
    st.markdown("")
    st.markdown("**Donnees actuellement chargees :**")
    loaded = st.session_state["annual_metrics"]
    countries = sorted({c for c, y in loaded.keys()})
    for c in countries:
        years = sorted([y for cc, y in loaded.keys() if cc == c])
        completeness = [loaded[(c, y)].get("data_completeness", 1.0) for y in years]
        avg_compl = sum(completeness) / len(completeness) if completeness else 0
        st.caption(f"{c}: {years[0]}-{years[-1]} ({len(years)} ans, completude moy. {avg_compl:.1%})")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 : HYPOTHESES MODIFIABLES
# ══════════════════════════════════════════════════════════════════════════
st.markdown("### Hypotheses du modele")

# Init custom_hypotheses if not present
if "custom_hypotheses" not in st.session_state:
    st.session_state.custom_hypotheses = {}

hyp = st.session_state.custom_hypotheses

tab_therm, tab_bess, tab_prix = st.tabs([
    "Rendements & Emissions",
    "BESS & Stockage",
    "Seuils de prix",
])

# ── Tab 1 : Rendements & Emissions ───────────────────────────────────────
with tab_therm:
    st.markdown("#### Rendements thermiques (PCI, net)")
    st.caption("Source : donnees constructeurs, moyennes du parc installe europeen.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hyp["ETA_CCGT"] = st.number_input(
            "η CCGT", value=hyp.get("ETA_CCGT", ETA_CCGT),
            min_value=0.40, max_value=0.65, step=0.01, format="%.2f",
            help="Rendement net CCGT moderne. Plage typique : 0.55-0.62")
    with col2:
        hyp["ETA_OCGT"] = st.number_input(
            "η OCGT", value=hyp.get("ETA_OCGT", ETA_OCGT),
            min_value=0.25, max_value=0.45, step=0.01, format="%.2f",
            help="Rendement net turbine gaz simple cycle. Plage : 0.35-0.42")
    with col3:
        hyp["ETA_COAL"] = st.number_input(
            "η Charbon", value=hyp.get("ETA_COAL", ETA_COAL),
            min_value=0.30, max_value=0.45, step=0.01, format="%.2f",
            help="Rendement net charbon pulverise. Plage : 0.35-0.43")
    with col4:
        hyp["ETA_LIGNITE"] = st.number_input(
            "η Lignite", value=hyp.get("ETA_LIGNITE", ETA_LIGNITE),
            min_value=0.28, max_value=0.40, step=0.01, format="%.2f",
            help="Rendement net lignite. Plage : 0.32-0.38 (tres variable)")

    st.markdown("")
    st.markdown("#### Facteurs d'emission (tCO2 / MWh thermique)")
    st.caption("Source : IPCC 2006, Tier 1 (combustion directe, hors upstream).")
    col1, col2, col3 = st.columns(3)
    with col1:
        hyp["EF_GAS"] = st.number_input(
            "EF Gaz", value=hyp.get("EF_GAS", EF_GAS),
            min_value=0.180, max_value=0.230, step=0.001, format="%.3f",
            help="56.1 kgCO2/GJ = 0.202 tCO2/MWh_th. Hors fuites methane amont.")
    with col2:
        hyp["EF_COAL"] = st.number_input(
            "EF Charbon", value=hyp.get("EF_COAL", EF_COAL),
            min_value=0.300, max_value=0.370, step=0.001, format="%.3f",
            help="94.6 kgCO2/GJ = 0.335 tCO2/MWh_th. Charbon bitumineux.")
    with col3:
        hyp["EF_LIGNITE"] = st.number_input(
            "EF Lignite", value=hyp.get("EF_LIGNITE", EF_LIGNITE),
            min_value=0.330, max_value=0.400, step=0.001, format="%.3f",
            help="101 kgCO2/GJ = 0.364 tCO2/MWh_th. Lignite (brown coal).")

    st.markdown("")
    st.markdown("#### Couts variables (EUR/MWh electrique)")
    st.caption("VOM : main d'oeuvre, maintenance, consommables. Hors combustible.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hyp["VOM_CCGT"] = st.number_input(
            "VOM CCGT", value=hyp.get("VOM_CCGT", VOM_CCGT),
            min_value=1.0, max_value=8.0, step=0.5, format="%.1f")
    with col2:
        hyp["VOM_OCGT"] = st.number_input(
            "VOM OCGT", value=hyp.get("VOM_OCGT", VOM_OCGT),
            min_value=2.0, max_value=10.0, step=0.5, format="%.1f")
    with col3:
        hyp["VOM_COAL"] = st.number_input(
            "VOM Charbon", value=hyp.get("VOM_COAL", VOM_COAL),
            min_value=2.0, max_value=8.0, step=0.5, format="%.1f")
    with col4:
        hyp["VOM_LIGNITE"] = st.number_input(
            "VOM Lignite", value=hyp.get("VOM_LIGNITE", VOM_LIGNITE),
            min_value=2.0, max_value=10.0, step=0.5, format="%.1f")

# ── Tab 2 : BESS ────────────────────────────────────────────────────────
with tab_bess:
    st.markdown("#### Parametres batteries Li-ion")
    st.caption("Les simulations BESS utilisent ces parametres pour le stockage/destockage horaire.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hyp["BESS_ROUND_TRIP_EFF"] = st.number_input(
            "Rendement round-trip", value=hyp.get("BESS_ROUND_TRIP_EFF", BESS_ROUND_TRIP_EFF),
            min_value=0.70, max_value=0.95, step=0.01, format="%.2f",
            help="Efficacite aller-retour Li-ion. Typique 2024 : 0.85-0.92")
    with col2:
        hyp["BESS_MIN_SOC"] = st.number_input(
            "SoC minimum (%)", value=hyp.get("BESS_MIN_SOC", BESS_MIN_SOC),
            min_value=0.0, max_value=0.20, step=0.01, format="%.2f",
            help="Plancher d'etat de charge pour proteger la batterie.")
    with col3:
        hyp["BESS_MAX_SOC"] = st.number_input(
            "SoC maximum (%)", value=hyp.get("BESS_MAX_SOC", BESS_MAX_SOC),
            min_value=0.80, max_value=1.00, step=0.01, format="%.2f",
            help="Plafond d'etat de charge (evite le stress cellulaire).")
    with col4:
        hyp["BESS_CYCLING_COST"] = st.number_input(
            "Cout cycling (EUR/MWh)", value=hyp.get("BESS_CYCLING_COST", BESS_CYCLING_COST),
            min_value=0.0, max_value=15.0, step=0.5, format="%.1f",
            help="Proxy de degradation par cycle de decharge. Inclut l'amortissement CapEx partiel.")

# ── Tab 3 : Seuils de prix ──────────────────────────────────────────────
with tab_prix:
    st.markdown("#### Seuils de classification des prix")
    st.caption("Utilises pour le comptage des heures negatives, tres basses, elevees.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hyp["PRICE_VERY_LOW"] = st.number_input(
            "Prix tres bas (EUR/MWh)", value=hyp.get("PRICE_VERY_LOW", PRICE_VERY_LOW),
            min_value=0.0, max_value=20.0, step=1.0, format="%.1f",
            help="Seuil sous lequel on compte les heures a prix tres bas.")
    with col2:
        hyp["PRICE_HIGH"] = st.number_input(
            "Prix eleve (EUR/MWh)", value=hyp.get("PRICE_HIGH", PRICE_HIGH),
            min_value=50.0, max_value=300.0, step=10.0, format="%.0f")
    with col3:
        hyp["PRICE_VERY_HIGH"] = st.number_input(
            "Prix tres eleve (EUR/MWh)", value=hyp.get("PRICE_VERY_HIGH", PRICE_VERY_HIGH),
            min_value=100.0, max_value=500.0, step=10.0, format="%.0f")
    with col4:
        hyp["SPREAD_DAILY_THRESHOLD"] = st.number_input(
            "Seuil spread journalier", value=hyp.get("SPREAD_DAILY_THRESHOLD", SPREAD_DAILY_THRESHOLD),
            min_value=10.0, max_value=200.0, step=5.0, format="%.0f",
            help="Ecart min-max intra-journalier a partir duquel on considere le marche volatile.")

st.session_state.custom_hypotheses = hyp

# ── Bouton reset ─────────────────────────────────────────────────────────
st.markdown("")
col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    if st.button("Restaurer les valeurs par defaut", use_container_width=True):
        st.session_state.custom_hypotheses = {}
        st.rerun()

st.info("Les modifications prendront effet au prochain clic sur **Charger les donnees** "
        "(page d'accueil). Les donnees deja en cache ne sont pas recalculees automatiquement.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 : HYPOTHESES NON MODIFIABLES
# ══════════════════════════════════════════════════════════════════════════
st.markdown("### Conventions fixes du modele")
st.caption("Ces parametres ne sont pas modifiables car ils affecteraient la comparabilite des resultats.")

st.markdown("""
| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| **Annees outliers** | 2022 | Crise gaziere systemique en Europe. Exclu des regressions par defaut. |
| **Seuil D_tail** | P90 de la NRL positive | Convention statistique standard (queue de distribution). |
| **Signe net position** | Positif = export | Convention ENTSO-E Transparency. |
| **Reset SoC BESS** | Quotidien (h=0) | Simplification : chaque jour demarre a SoC moyen. |
| **Interpolation gaps** | limit=3 heures | Les donnees manquantes sont interpolees jusqu'a 3h max. Au-dela, NaN preserve. |
| **Prix : jamais interpoles** | -- | Les prix DA ne sont jamais interpoles pour eviter les biais. |
""")
