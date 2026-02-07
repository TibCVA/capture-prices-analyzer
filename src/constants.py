"""
Constantes physiques et conventions du modele Capture Prices.
Chaque constante est commentee avec sa source.
"""
import numpy as np

# ==================== RENDEMENTS THERMIQUES (PCI, net) ====================
ETA_CCGT = 0.57        # CCGT moderne, rendement net (source : donnees constructeurs)
ETA_OCGT = 0.38        # OCGT / turbine gaz simple cycle
ETA_COAL = 0.38        # Charbon pulverise moderne
ETA_LIGNITE = 0.33     # Lignite

# ==================== FACTEURS D'EMISSION (tCO2 / MWh_thermique entrant) ====================
EF_GAS = 0.202         # Gaz naturel (IPCC 2006, 56.1 kgCO2/GJ)
EF_COAL = 0.335        # Charbon bitumineux (IPCC 2006, 94.6 kgCO2/GJ)
EF_LIGNITE = 0.364     # Lignite (IPCC 2006, 101 kgCO2/GJ)

# ==================== COUTS VARIABLES (EUR/MWh_e produit) ====================
VOM_CCGT = 3.0
VOM_OCGT = 5.0
VOM_COAL = 4.0
VOM_LIGNITE = 5.0

# ==================== SEUILS DE PRIX (EUR/MWh) ====================
PRICE_NEGATIVE = 0.0
PRICE_VERY_LOW = 5.0
PRICE_HIGH = 100.0
PRICE_VERY_HIGH = 200.0
SPREAD_DAILY_THRESHOLD = 50.0

# ==================== QUANTILES ====================
Q05 = 0.05
Q10 = 0.10
Q25 = 0.25
Q50 = 0.50
Q75 = 0.75
Q90 = 0.90
Q95 = 0.95
Q99 = 0.99

# ==================== PERIODES ====================
HOURS_YEAR = 8760
HOURS_LEAP = 8784
ANALYSIS_YEARS = range(2015, 2025)     # 2015 a 2024 inclus
OUTLIER_YEARS = {2022}                  # Crise gaziere — exclure des regressions
TZ_UTC = "UTC"

# ==================== SENSIBILITES ANALYTIQUES TCA ====================
DTCA_DGAS_CCGT = 1.0 / ETA_CCGT         # ~= 1.754 EUR/MWh_e par EUR/MWh_th
DTCA_DCO2_CCGT = EF_GAS / ETA_CCGT      # ~= 0.354 EUR/MWh_e par EUR/tCO2
DTCA_DCO2_COAL = EF_COAL / ETA_COAL      # ~= 0.882
DTCA_DCO2_LIGNITE = EF_LIGNITE / ETA_LIGNITE  # ~= 1.103

# ==================== BESS SoC PARAMETERS ====================
BESS_ROUND_TRIP_EFF = 0.88              # Round-trip efficiency Li-ion
BESS_MIN_SOC = 0.05                     # Plancher SoC (% of energy capacity)
BESS_MAX_SOC = 0.95                     # Plafond SoC (% of energy capacity)
BESS_CYCLING_COST = 5.0                 # EUR/MWh_discharged (degradation proxy)

# ==================== NOMS DE COLONNES HARMONISES ====================
# --- Donnees brutes ---
COL_TS = "timestamp"
COL_LOAD = "load_mw"
COL_SOLAR = "solar_mw"
COL_WIND_ON = "wind_onshore_mw"
COL_WIND_OFF = "wind_offshore_mw"
COL_NUCLEAR = "nuclear_mw"
COL_LIGNITE = "lignite_mw"
COL_COAL = "coal_mw"
COL_GAS = "gas_mw"
COL_HYDRO_ROR = "hydro_ror_mw"
COL_HYDRO_RES = "hydro_reservoir_mw"
COL_PSH_GEN = "psh_generation_mw"       # Turbinage, >= 0
COL_PSH_PUMP = "psh_pumping_mw"         # Pompage, >= 0
COL_BIOMASS = "biomass_mw"
COL_OTHER = "other_mw"
COL_PRICE_DA = "price_da_eur_mwh"
COL_NET_POSITION = "net_position_mw"     # Positif = export net, negatif = import net
COL_TOTAL_GEN = "total_generation_mw"

# --- Colonnes calculees par nrl_engine ---
COL_VRE = "vre_mw"
COL_MUST_RUN = "must_run_mw"
COL_NRL = "nrl_mw"
COL_SURPLUS = "surplus_mw"               # max(0, -NRL)
COL_FLEX_CAPACITY = "flex_capacity_mw"   # Capacite totale de flex (BESS+PSH+DSM+export_max)
COL_FLEX_DOMESTIC = "flex_domestic_mw"   # Flex domestique (PSH+BESS+DSM, sans exports)
COL_FLEX_USED = "flex_used_mw"           # Flex effectivement activee (pompage+exports obs.)
COL_SURPLUS_UNABS = "surplus_unabsorbed_mw"
COL_REGIME = "regime"                    # 'A', 'B', 'C', 'D_tail'
COL_TCA = "tca_eur_mwh"
COL_REGIME_COHERENT = "regime_coherent"  # bool
COL_HAS_GAP = "has_gap"                 # bool

# ==================== CODES PAYS ENTSO-E ====================
COUNTRY_ENTSOE = {
    "FR": "FR",
    "DE": "DE_LU",
    "ES": "ES",
    "PL": "PL",
    "DK": "DK_1",
}

# Zones ayant change de code dans la periode d'analyse
COUNTRY_CODE_PERIODS = {
    "DE": [
        {"code": "DE_AT_LU", "start": "2015-01-01", "end": "2018-09-30"},
        {"code": "DE_LU", "start": "2018-10-01", "end": "2099-12-31"},
    ],
}

# Timezones locales par pays (pour peak/offpeak et spreads journaliers)
COUNTRY_TZ = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "ES": "Europe/Madrid",
    "PL": "Europe/Warsaw",
    "DK": "Europe/Copenhagen",
}

# ==================== MAPPING COLONNES ENTSO-E ====================
ENTSOE_GEN_MAPPING = {
    "Solar": COL_SOLAR,
    "Wind Onshore": COL_WIND_ON,
    "Wind Offshore": COL_WIND_OFF,
    "Nuclear": COL_NUCLEAR,
    "Fossil Brown coal/Lignite": COL_LIGNITE,
    "Fossil Hard coal": COL_COAL,
    "Fossil Gas": COL_GAS,
    "Hydro Run-of-river and poundage": COL_HYDRO_ROR,
    "Hydro Water Reservoir": COL_HYDRO_RES,
    "Biomass": COL_BIOMASS,
    "Other": COL_OTHER,
    "Other renewable": COL_OTHER,
    "Waste": COL_OTHER,
    "Geothermal": COL_OTHER,
    "Marine": COL_OTHER,
    "Fossil Oil": COL_OTHER,
    "Fossil Oil shale": COL_OTHER,
    "Fossil Peat": COL_OTHER,
}

# Colonnes optionnelles (mises a 0 si absentes pour le pays)
OPTIONAL_COLUMNS = {COL_WIND_OFF, COL_NUCLEAR, COL_LIGNITE, COL_HYDRO_RES,
                    COL_PSH_GEN, COL_PSH_PUMP}

# ==================== PALETTE UI ====================
REGIME_PALETTE = {
    "A": "#E74C3C",
    "B": "#F39C12",
    "C": "#3498DB",
    "D_tail": "#8E44AD",
}
COUNTRY_PALETTE = {
    "FR": "#0066CC",
    "DE": "#E67E22",
    "ES": "#27AE60",
    "PL": "#E74C3C",
    "DK": "#8E44AD",
}
PHASE_PALETTE = {
    1: "#27AE60",
    2: "#F39C12",
    3: "#3498DB",
    4: "#2C3E50",
}
PLOTLY_LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1B2A4A"),
    plot_bgcolor="#FAFBFC",
    paper_bgcolor="#FAFBFC",
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="center", x=0.5, font=dict(size=11),
    ),
)

# ==================== CONVENTION DE SIGNE : NET POSITION ====================
# Selon ENTSO-E : net position = generation - load - absorbed_energy
# Positif = le pays est exportateur net sur la periode
# Negatif = le pays est importateur net
# Source : ENTSO-E Transparency FAQ, section "Net Position"
NET_POSITION_SIGN_CONVENTION = "positive = export, negative = import"


def get_constants(overrides: dict | None = None) -> dict:
    """Retourne toutes les constantes modifiables dans un dict, avec overrides optionnels."""
    defaults = {
        "ETA_CCGT": ETA_CCGT, "ETA_OCGT": ETA_OCGT,
        "ETA_COAL": ETA_COAL, "ETA_LIGNITE": ETA_LIGNITE,
        "EF_GAS": EF_GAS, "EF_COAL": EF_COAL, "EF_LIGNITE": EF_LIGNITE,
        "VOM_CCGT": VOM_CCGT, "VOM_OCGT": VOM_OCGT,
        "VOM_COAL": VOM_COAL, "VOM_LIGNITE": VOM_LIGNITE,
        "BESS_ROUND_TRIP_EFF": BESS_ROUND_TRIP_EFF,
        "BESS_MIN_SOC": BESS_MIN_SOC, "BESS_MAX_SOC": BESS_MAX_SOC,
        "BESS_CYCLING_COST": BESS_CYCLING_COST,
        "PRICE_NEGATIVE": PRICE_NEGATIVE, "PRICE_VERY_LOW": PRICE_VERY_LOW,
        "PRICE_HIGH": PRICE_HIGH, "PRICE_VERY_HIGH": PRICE_VERY_HIGH,
        "SPREAD_DAILY_THRESHOLD": SPREAD_DAILY_THRESHOLD,
    }
    if overrides:
        defaults.update(overrides)
    return defaults
