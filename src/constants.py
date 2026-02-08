"""
Constantes et conventions — Capture Prices Analyzer v3.0
Aucune valeur numerique ne doit etre hardcodee ailleurs.
"""

# ==================== RENDEMENTS THERMIQUES (net) ====================
ETA_CCGT = 0.57
ETA_OCGT = 0.38
ETA_COAL = 0.38
ETA_LIGNITE = 0.33

# ==================== FACTEURS D'EMISSION (tCO2 / MWh_th) ====================
EF_GAS = 0.202
EF_COAL = 0.335
EF_LIGNITE = 0.364

# ==================== COUTS VARIABLES (EUR/MWh_elec) ====================
VOM_CCGT = 3.0
VOM_OCGT = 5.0
VOM_COAL = 4.0
VOM_LIGNITE = 5.0

# ==================== PRIX & SEUILS OBSERVABLES ====================
PRICE_NEGATIVE_THRESHOLD = 0.0
PRICE_VERY_LOW_THRESHOLD = 5.0
PRICE_HIGH_THRESHOLD = 100.0
PRICE_VERY_HIGH_THRESHOLD = 200.0
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
Q995 = 0.995

# ==================== ANNEES & TIME ====================
HOURS_YEAR = 8760
HOURS_LEAP = 8784
ANALYSIS_YEARS = range(2015, 2025)  # 2015..2024 inclus
OUTLIER_YEARS = {2022}
TZ_UTC = "UTC"

# ==================== PRIX COMMODITES — CONVENTIONS ====================
# TTF en EUR/MWh_th (PCI)
# EUA en EUR/tCO2
# Coal (optionnel) en EUR/MWh_th ; sinon approximation via ratio gaz
COAL_GAS_INDEX_RATIO = 0.40

# ==================== SENSIBILITES ANALYTIQUES TCA ====================
DTCA_DGAS_CCGT = 1.0 / ETA_CCGT
DTCA_DCO2_CCGT = EF_GAS / ETA_CCGT
DTCA_DGAS_COAL = 1.0 / ETA_COAL
DTCA_DCO2_COAL = EF_COAL / ETA_COAL

# ==================== BESS — DISPATCH SIMPLIFIE ====================
# Modele SoC sequentiel deterministe (aucun prix utilise)
BESS_ETA_CHARGE = 0.95
BESS_ETA_DISCHARGE = 0.95
BESS_SOC_INIT_FRAC = 0.50  # SoC initial = 50% (reduit effets de bord)

# ==================== PRIX SYNTHETIQUE — PARAMETRES (SCENARIOS) ====================
# Regle : le prix synthetique est une sortie du modele, pas un input.
# Il est utilise pour les metriques en scenario si price_mode="synthetic".

# Regime A (surplus non absorbe) : prix constant (floor), cap negatif
PRICE_SYNTH_A = -5.0
PRICE_SYNTH_A_MIN = -250.0

# Regime B (surplus absorbe) : prix bas
PRICE_SYNTH_B = 10.0
PRICE_SYNTH_B_MIN = 0.0
PRICE_SYNTH_B_MAX = 30.0

# Regime C (thermique marginal) : ancre sur TCA
PRICE_SYNTH_C_ADDER = 0.0

# Regime D (pointe) : multiplicateur sur TCA
PRICE_SYNTH_D_MULTIPLIER = 1.8
PRICE_SYNTH_D_MAX = 5000.0

# ==================== NOMS DE COLONNES ====================
COL_TS = "timestamp"

# Inputs (harmonises)
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
COL_PSH_GEN = "psh_generation_mw"
COL_PSH_PUMP = "psh_pumping_mw"
COL_BIOMASS = "biomass_mw"
COL_OTHER = "other_mw"
COL_PRICE_DA = "price_da_eur_mwh"

# Echanges : NET POSITION (positif = net export)
COL_NET_POSITION = "net_position_mw"

# Calculs (NRL & flex)
COL_VRE = "vre_mw"
COL_MUST_RUN = "must_run_mw"
COL_NRL = "nrl_mw"
COL_SURPLUS = "surplus_mw"
COL_SINK_NON_BESS = "sink_non_bess_mw"
COL_BESS_CHARGE = "bess_charge_mw"
COL_BESS_DISCHARGE = "bess_discharge_mw"
COL_BESS_SOC = "bess_soc_mwh"
COL_FLEX_EFFECTIVE = "flex_effective_mw"
COL_SURPLUS_UNABS = "surplus_unabsorbed_mw"

# Regimes
COL_REGIME = "regime"  # 'A','B','C','D'
COL_REGIME_COHERENT = "regime_coherent"  # bool (historique seulement)

# Prix modele
COL_TCA = "tca_eur_mwh"
COL_PRICE_SYNTH = "price_synth_eur_mwh"
COL_PRICE_USED = "price_used_eur_mwh"

# Qualite donnees
COL_HAS_GAP = "has_gap"

# ==================== COLONNES OPTIONNELLES ====================
OPTIONAL_COLUMNS = {
    COL_WIND_OFF,
    COL_NUCLEAR,
    COL_LIGNITE,
    COL_HYDRO_RES,
    COL_PSH_GEN,
    COL_PSH_PUMP,
    COL_NET_POSITION,
}

# ==================== MAPPINGS ENTSO-E ====================
ENTSOE_GEN_MAPPING = {
    "Solar": COL_SOLAR,
    "Wind Onshore": COL_WIND_ON,
    "Wind Offshore": COL_WIND_OFF,
    "Nuclear": COL_NUCLEAR,
    "Fossil Gas": COL_GAS,
    "Fossil Hard coal": COL_COAL,
    "Fossil Brown coal/Lignite": COL_LIGNITE,
    "Hydro Run-of-river and poundage": COL_HYDRO_ROR,
    "Hydro Water Reservoir": COL_HYDRO_RES,
    "Hydro Pumped Storage": "_psh_dispatch",
    "Biomass": COL_BIOMASS,
}

# Alias robustes ENTSO-E observes dans la pratique
ENTSOE_GEN_ALIASES = {
    "Hydro Run-of-river and poundage": "Hydro Run-of-river and poundage",
    "Hydro Run-of-river and poundage ": "Hydro Run-of-river and poundage",
    "Fossil Brown coal/Lignite": "Fossil Brown coal/Lignite",
    "Fossil Brown coal/lignite": "Fossil Brown coal/Lignite",
}

# ==================== VISUELS ====================
REGIME_COLORS = {
    "A": "#d73027",
    "B": "#fc8d59",
    "C": "#4575b4",
    "D": "#313695",
}

PHASE_COLORS = {
    "stage_1": "#1b9e77",
    "stage_2": "#d95f02",
    "stage_3": "#7570b3",
    "stage_4": "#e7298a",
    "unknown": "#666666",
}

COUNTRY_PALETTE = {
    "FR": "#003399",
    "DE": "#FFCC00",
    "ES": "#CC0000",
    "PL": "#DC143C",
    "DK": "#C8102E",
    "NL": "#FF6F00",
    "BE": "#111111",
    "IT": "#008C45",
    "AT": "#ED2939",
    "CH": "#D52B1E",
}

PLOTLY_LAYOUT_DEFAULTS = {
    "template": "plotly_white",
    "font": {"family": "Segoe UI, Arial, sans-serif", "size": 13, "color": "#1f2937"},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
    "margin": {"l": 52, "r": 28, "t": 48, "b": 46},
    "plot_bgcolor": "#ffffff",
    "paper_bgcolor": "#ffffff",
    "hovermode": "closest",
}

PLOTLY_AXIS_DEFAULTS = {
    "showgrid": True,
    "gridcolor": "#e5e7eb",
    "zeroline": False,
    "linecolor": "#cbd5e1",
    "mirror": False,
}

# Statut interpretable des liens NRL/prix observes
CORRELATION_STATUS_THRESHOLDS = {
    "weak": 0.20,
    "medium": 0.45,
}

COHERENCE_STATUS_THRESHOLDS = {
    "weak": 0.55,
    "medium": 0.70,
}
