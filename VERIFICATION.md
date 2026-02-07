# VERIFICATION SPEC v3.0 — Capture Prices Analyzer

Date de verification : 2026-02-06
Tests : **12/12 PASSED**

---

## A. ARBORESCENCE

| Fichier / Dossier | Statut | Notes |
|---|---|---|
| `.env.example` | OK | ENTSOE_API_KEY + GOOGLE_SHEETS_CREDS |
| `.gitignore` | OK | .env, data/raw/, data/processed/, data/exports/, __pycache__, capture_prices.log |
| `.streamlit/config.toml` | OK | primaryColor=#1a5276, maxUploadSize=200, runOnSave=true, gatherUsageStats=false |
| `requirements.txt` | OK | 13 deps (streamlit, pandas, numpy, plotly, entsoe-py, pyarrow, pyyaml, python-dotenv, scikit-learn, scipy, openpyxl, gspread, google-auth) |
| `README.md` | OK | 13 sections |
| `config/countries.yaml` | OK | 5 pays (FR, DE, ES, PL, DK) |
| `config/scenarios.yaml` | OK | 7 scenarios predefinis |
| `config/thresholds.yaml` | OK | 4 stages + 4 alertes |
| `data/raw/.gitkeep` | OK | |
| `data/processed/.gitkeep` | OK | |
| `data/external/.gitkeep` | OK | |
| `data/exports/.gitkeep` | OK | |
| `src/__init__.py` | OK | Fichier vide |
| `src/constants.py` | OK | ~152 lignes |
| `src/data_fetcher.py` | OK | |
| `src/data_loader.py` | OK | + fonctions utilitaires ajoutees (load_country_config, etc.) |
| `src/nrl_engine.py` | OK | |
| `src/metrics.py` | OK | |
| `src/phase_diagnostics.py` | OK | |
| `src/scenario_engine.py` | OK | |
| `src/slope_analysis.py` | OK | |
| `src/export_utils.py` | OK | |
| `app.py` | OK | |
| `pages/0_Comprendre_le_Modele.py` | OK | 16.6 KB |
| `pages/1_Analyse_Historique.py` | OK | 7.7 KB |
| `pages/2_NRL_Deep_Dive.py` | OK | 9.9 KB |
| `pages/3_Capture_Rates.py` | OK | 10.5 KB |
| `pages/4_Comparaison_Pays.py` | OK | 9.6 KB |
| `pages/5_Scenarios.py` | OK | 11.9 KB |
| `pages/6_Questions_S_Michel.py` | OK | 18.4 KB |
| `pages/7_Guide_Utilisateur.py` | OK | 10.7 KB |
| `tests/conftest.py` | OK | |
| `tests/test_nrl_engine.py` | OK | 5 tests |
| `tests/test_metrics.py` | OK | 3 tests |
| `tests/test_phase_diagnostics.py` | OK | 2 tests |
| `tests/test_scenarios.py` | OK | 2 tests |

**Total : 36 fichiers** (31 spec + 4 .gitkeep + 1 VERIFICATION.md)

---

## B. CONSTANTES PHYSIQUES (Partie C de la spec)

| Constante | Spec | Code | OK? |
|---|---|---|---|
| ETA_CCGT | 0.57 | 0.57 | OK |
| ETA_OCGT | 0.38 | 0.38 | OK |
| ETA_COAL | 0.38 | 0.38 | OK |
| ETA_LIGNITE | 0.33 | 0.33 | OK |
| EF_GAS | 0.202 | 0.202 | OK |
| EF_COAL | 0.335 | 0.335 | OK |
| EF_LIGNITE | 0.364 | 0.364 | OK |
| VOM_CCGT | 3.0 | 3.0 | OK |
| VOM_OCGT | 5.0 | 5.0 | OK |
| VOM_COAL | 4.0 | 4.0 | OK |
| VOM_LIGNITE | 5.0 | 5.0 | OK |
| PRICE_NEGATIVE | 0.0 | 0.0 | OK |
| PRICE_VERY_LOW | 5.0 | 5.0 | OK |
| PRICE_HIGH | 100.0 | 100.0 | OK |
| PRICE_VERY_HIGH | 200.0 | 200.0 | OK |
| SPREAD_DAILY_THRESHOLD | 50.0 | 50.0 | OK |
| BESS_ROUND_TRIP_EFF | 0.88 | 0.88 | OK |
| BESS_MIN_SOC | 0.05 | 0.05 | OK |
| BESS_MAX_SOC | 0.95 | 0.95 | OK |
| BESS_CYCLING_COST | 5.0 | 5.0 | OK |
| HOURS_YEAR | 8760 | 8760 | OK |
| HOURS_LEAP | 8784 | 8784 | OK |
| ANALYSIS_YEARS | 2015-2024 | range(2015, 2025) | OK |
| OUTLIER_YEARS | {2022} | {2022} | OK |
| DTCA_DGAS_CCGT | 1/ETA_CCGT | 1.0/ETA_CCGT | OK |
| DTCA_DCO2_CCGT | EF_GAS/ETA_CCGT | EF_GAS/ETA_CCGT | OK |
| Q05..Q99 | 0.05..0.99 | 0.05..0.99 | OK |

---

## C. COLONNES HARMONISEES (Partie C)

| Colonne | Spec | Code | OK? |
|---|---|---|---|
| COL_LOAD | "load_mw" | "load_mw" | OK |
| COL_SOLAR | "solar_mw" | "solar_mw" | OK |
| COL_WIND_ON | "wind_onshore_mw" | "wind_onshore_mw" | OK |
| COL_WIND_OFF | "wind_offshore_mw" | "wind_offshore_mw" | OK |
| COL_NUCLEAR | "nuclear_mw" | "nuclear_mw" | OK |
| COL_LIGNITE | "lignite_mw" | "lignite_mw" | OK |
| COL_COAL | "coal_mw" | "coal_mw" | OK |
| COL_GAS | "gas_mw" | "gas_mw" | OK |
| COL_HYDRO_ROR | "hydro_ror_mw" | "hydro_ror_mw" | OK |
| COL_HYDRO_RES | "hydro_reservoir_mw" | "hydro_reservoir_mw" | OK |
| COL_PSH_GEN | "psh_generation_mw" | "psh_generation_mw" | OK |
| COL_PSH_PUMP | "psh_pumping_mw" | "psh_pumping_mw" | OK |
| COL_BIOMASS | "biomass_mw" | "biomass_mw" | OK |
| COL_OTHER | "other_mw" | "other_mw" | OK |
| COL_PRICE_DA | "price_da_eur_mwh" | "price_da_eur_mwh" | OK |
| COL_NET_POSITION | "net_position_mw" | "net_position_mw" | OK |
| COL_TOTAL_GEN | "total_generation_mw" | "total_generation_mw" | OK |
| COL_VRE | "vre_mw" | "vre_mw" | OK |
| COL_MUST_RUN | "must_run_mw" | "must_run_mw" | OK |
| COL_NRL | "nrl_mw" | "nrl_mw" | OK |
| COL_SURPLUS | "surplus_mw" | "surplus_mw" | OK |
| COL_FLEX_CAPACITY | "flex_capacity_mw" | "flex_capacity_mw" | OK |
| COL_FLEX_USED | "flex_used_mw" | "flex_used_mw" | OK |
| COL_SURPLUS_UNABS | "surplus_unabsorbed_mw" | "surplus_unabsorbed_mw" | OK |
| COL_REGIME | "regime" | "regime" | OK |
| COL_TCA | "tca_eur_mwh" | "tca_eur_mwh" | OK |
| COL_REGIME_COHERENT | "regime_coherent" | "regime_coherent" | OK |
| COL_HAS_GAP | "has_gap" | "has_gap" | OK |

---

## D. MAPPING ENTSO-E (Partie C)

| ENTSO-E type | Colonne cible | OK? |
|---|---|---|
| Solar | COL_SOLAR | OK |
| Wind Onshore | COL_WIND_ON | OK |
| Wind Offshore | COL_WIND_OFF | OK |
| Nuclear | COL_NUCLEAR | OK |
| Fossil Brown coal/Lignite | COL_LIGNITE | OK |
| Fossil Hard coal | COL_COAL | OK |
| Fossil Gas | COL_GAS | OK |
| Hydro Run-of-river and poundage | COL_HYDRO_ROR | OK |
| Hydro Water Reservoir | COL_HYDRO_RES | OK |
| Biomass | COL_BIOMASS | OK |
| Other / Other renewable / Waste / Geothermal / Marine / Fossil Oil / Fossil Oil shale / Fossil Peat | COL_OTHER | OK |

---

## E. CODES PAYS ENTSO-E (Partie C)

| Pays | Code | OK? |
|---|---|---|
| FR | FR | OK |
| DE | DE_LU (+ DE_AT_LU pre-2018-10-01) | OK |
| ES | ES | OK |
| PL | PL | OK |
| DK | DK_1 | OK |

---

## F. CONFIG PAYS (Partie D — countries.yaml)

### FR
- [x] must_run.mode = "observed"
- [x] must_run.observed_components = [nuclear, hydro_ror, biomass]
- [x] must_run.floor_params.nuclear_floor_gw = 20.0
- [x] must_run.floor_params.nuclear_modulation_pct = 0.50
- [x] flex.capacity.psh_pump_capacity_gw = 4.5
- [x] flex.capacity.bess_power_gw = 0.5
- [x] flex.capacity.bess_energy_gwh = 2.0
- [x] flex.capacity.dsm_gw = 2.0
- [x] flex.capacity.export_max_gw = 17.0
- [x] thermal.marginal_tech = "CCGT"

### DE
- [x] must_run.observed_components = [lignite, hydro_ror, biomass]
- [x] lignite_floor_gw = 5.0, lignite_modulation_pct = 0.40
- [x] flex: psh=6.5, bess=3.0/12.0, dsm=3.0, export=25.0
- [x] marginal_tech = "CCGT"

### ES
- [x] must_run.observed_components = [nuclear, hydro_ror, biomass]
- [x] nuclear_floor_gw = 6.5, nuclear_modulation_pct = 0.90
- [x] flex: psh=3.0, bess=0.5/2.0, dsm=1.5, export=5.0
- [x] marginal_tech = "CCGT"

### PL
- [x] must_run.observed_components = [coal, hydro_ror, biomass]
- [x] coal_floor_gw = 8.0, coal_modulation_pct = 0.50
- [x] flex: psh=1.5, bess=0.2/0.8, dsm=1.0, export=8.0
- [x] marginal_tech = "coal"

### DK
- [x] must_run.observed_components = [biomass]
- [x] biomass_floor_gw = 0.5
- [x] flex: psh=0.0, bess=0.3/1.2, dsm=0.5, export=6.0
- [x] marginal_tech = "CCGT"

---

## G. SCENARIOS (Partie E — scenarios.yaml)

| Scenario | Parametres | OK? |
|---|---|---|
| accelerated_pv | delta_pv_gw=20, delta_bess_power_gw=5, delta_bess_energy_gwh=20 | OK |
| electrification | delta_demand_pct=15, delta_demand_midday_gw=8, delta_demand_evening_gw=3 | OK |
| high_co2 | co2_price_eur_t=120 | OK |
| nuclear_reduction_fr | delta_must_run_gw=-10, delta_pv_gw=15, delta_wind_onshore_gw=10 | OK |
| flex_massive | delta_bess_power_gw=15, delta_bess_energy_gwh=60 | OK |
| high_gas | gas_price_eur_mwh=50 | OK |
| combined_stress | delta_pv_gw=25, delta_demand_pct=0, delta_bess_power_gw=1, delta_bess_energy_gwh=4 | OK |

---

## H. THRESHOLDS (Partie F — thresholds.yaml)

### Stages
- [x] stage_1: h_negative_max=100, h_below_5_max=200, capture_ratio_pv_min=0.85, sr_max=0.01
- [x] stage_2: h_negative_min=200, h_negative_strong=300, h_below_5_min=500, capture_ratio_pv_max=0.80, capture_ratio_pv_crisis=0.70, days_spread_50_min=150
- [x] stage_3: far_structural_min=0.60, far_structural_strong=0.80, require_h_neg_declining=true
- [x] stage_4: far_structural_min=0.90, h_regime_c_max=1500

### Alertes
- [x] approaching_stage_2: h_negative_range=[150,300], capture_ratio_pv_range=[0.75,0.85]
- [x] deep_stage_2: h_negative_min=500, capture_ratio_pv_max=0.65
- [x] high_inflexibility: ir_min=0.60
- [x] low_flex: far_structural_max=0.30, sr_min=0.02

---

## I. BACKEND — REGLES METIER CRITIQUES

### I.1 NRL Engine (src/nrl_engine.py)
- [x] NRL = Load - VRE - MustRun (ligne 81)
- [x] Surplus = max(0, -NRL) (ligne 84)
- [x] VRE = solar + wind_onshore + wind_offshore (ligne 36-38)
- [x] Must-run mode "observed" : somme des composants observes (ligne 49-53)
- [x] Must-run mode "floor" : min(observed, max(floor, observed*mod_pct)) (ligne 56-76)
- [x] REGLE PHYSIQUE : MR ne depasse jamais la production observee (ligne 71)
- [x] Flex capacity = PSH + BESS + DSM + Export_max (lignes 88-105)
- [x] Flex used = pompage + export net positif (lignes 108-116)
- [x] Surplus non absorbe = max(0, surplus - flex_capacity) (ligne 119)
- [x] TCA avec commodity prices : gas/ETA + (EF/ETA)*CO2 + VOM (lignes 122-149)
- [x] TCA fallback P75 rolling si commodities absentes (ligne 148)
- [x] REGLE ANTI-CIRCULARITE : classification basee UNIQUEMENT sur variables physiques (ligne 152)
- [x] Regime A : surplus_unabs > 0 (ligne 153)
- [x] Regime B : surplus > 0 ET pas A (ligne 154)
- [x] Regime D_tail : NRL > 0 ET NRL > P90(NRL positif) (lignes 156-164)
- [x] Regime C : NRL > 0 ET pas D_tail (ligne 165)
- [x] Hierarchie : A > B > C > D_tail (lignes 167-170)
- [x] Validation croisee regime/prix post-classification (lignes 172-198)

### I.2 Metrics (src/metrics.py)
- [x] ~50 metriques retournees dans un dict
- [x] VRE penetration = % of total generation (PAS % of demand) (ligne 53-54)
- [x] Peak/offpeak calcule en heure locale (lignes 23-27)
- [x] Capture rate = prix pondere par production (lignes 34-39)
- [x] Capture ratio = capture_rate / baseload_price (lignes 48-49)
- [x] FAR_structural : capacity-based (lignes 86-94)
- [x] FAR_observed : flex reellement activee (lignes 97-104)
- [x] IR = MR_P10 / Load_P10 (lignes 107-113)
- [x] TTL = P95 des prix en regime C+D_tail (lignes 116-117)
- [x] SR = surplus_total / load_total (ligne 82)
- [x] is_outlier = year in OUTLIER_YEARS (ligne 179)

### I.3 Phase Diagnostics (src/phase_diagnostics.py)
- [x] Scoring par stade (s1, s2, s3, s4)
- [x] Score minimum de 2 pour etre candidat (ligne 92)
- [x] Confidence = max_score / total (ligne 99)
- [x] Critere inter-annuel Stage 3 : h_neg en baisse malgre VRE en hausse (lignes 64-70)
- [x] 4 alertes implementees avec conditions multiples (lignes 106-144)

### I.4 Scenario Engine (src/scenario_engine.py)
- [x] 7 etapes : VRE mods, demand mods, must-run mods, NRL recalc, flex+BESS SoC, TCA+prix mecaniste, reclassification
- [x] _scale_vre_profile : profil synthetique si quasi-nul (lignes 10-46)
- [x] _compute_bess_absorption : SoC-constrained avec reset journalier (lignes 49-76)
- [x] Prix mecaniste affine par morceaux : A=negatif, B=BESS_CYCLING_COST, C=TCA, D=TCA*1.5+ (lignes 160-178)
- [x] Fallback gas=30, co2=65 si pas de commodity prices (lignes 141, 148)

### I.5 Slope Analysis (src/slope_analysis.py)
- [x] scipy.stats.linregress
- [x] Exclusion outliers (is_outlier)
- [x] Minimum 3 points requis

### I.6 Export Utils (src/export_utils.py)
- [x] Excel 3 onglets (Metriques, Diagnostics, Slopes) avec header bold
- [x] Google Sheets avec graceful fail si creds absentes

### I.7 Data Fetcher (src/data_fetcher.py)
- [x] 7 etapes : cache check, code resolution, bornes, API calls, assemblage, validation, sauvegarde
- [x] Retry 3x avec backoff (lignes 13-22)
- [x] DE code period : DE_AT_LU pre-2018-10 → DE_LU post-2018-10 (lignes 40-44)
- [x] Resample hourly avec mean (lignes 127-134)
- [x] JAMAIS interpoler les prix (lignes 151-155)
- [x] Flag HAS_GAP (ligne 158)
- [x] load_commodity_prices() : gas, co2, bess (lignes 183-210)

### I.8 Data Loader (src/data_loader.py)
- [x] load_raw, load_processed, save_processed
- [x] load_country_config, load_all_countries_config (ajout vs spec, necessaire pour app.py)
- [x] load_thresholds, load_scenarios_config

---

## J. UI — STREAMLIT

### app.py (H.0)
- [x] Session state init (12 cles)
- [x] Sidebar : API key, multiselect pays, slider annees, checkbox 2022, radio must-run
- [x] Boutons Charger + Forcer refresh
- [x] Pipeline : load raw → compute_nrl → save processed → metrics → diagnostics
- [x] KPI cards (4 colonnes : h_neg, CR_PV, FAR, Phase)
- [x] Schema ASCII du modele
- [x] Tableau synthetique multi-pays

### Pages (8 pages)
- [x] Page 0 : Comprendre le Modele (6 expanders didactiques)
- [x] Page 1 : Analyse Historique (combo chart prix+VRE, barre regimes, metriques)
- [x] Page 2 : NRL Deep Dive (area chart, histogram, scatter NRL vs prix, coherence)
- [x] Page 3 : Capture Rates (scatter+regression, duration curve, heatmap)
- [x] Page 4 : Comparaison Pays (radar, bubble scatter, export buttons)
- [x] Page 5 : Scenarios (2 colonnes, sliders, delta cards, duration curve)
- [x] Page 6 : Questions S. Michel (6 tabs pour 6 questions)
- [x] Page 7 : Guide Utilisateur (6 expanders)

---

## K. TESTS

| Test | Resultat | Description |
|---|---|---|
| test_nrl_basic | PASSED | NRL = load - VRE - must_run |
| test_surplus_created | PASSED | NRL < 0 → surplus |
| test_regime_a_surplus_exceeds_flex | PASSED | surplus > flex → Regime A |
| test_regime_b_surplus_within_flex | PASSED | surplus <= flex → Regime B |
| test_floor_mode_capped_by_observed | PASSED | MR floor ne depasse pas observed |
| test_capture_rate_formula | PASSED | capture = prix pondere par production |
| test_far_no_surplus | PASSED | FAR = NaN si pas de surplus |
| test_vre_share_is_of_generation | PASSED | VRE share = % of generation |
| test_stage_1 | PASSED | Phase diagnostics stage 1 |
| test_stage_2 | PASSED | Phase diagnostics stage 2 |
| test_adding_bess_reduces_regime_a | PASSED | BESS reduit heures regime A |
| test_gas_co2_scenario_changes_tca | PASSED | Gas/CO2 scenario modifie TCA |

**12/12 PASSED**

---

## L. CONVENTIONS (Partie N)

- [x] VRE penetration = % of total generation (pas % of demand)
- [x] Prix en EUR/MWh
- [x] Puissances en MW dans le code, GW dans les configs YAML
- [x] Index temporel en UTC
- [x] Peak/offpeak en heure locale (via COUNTRY_TZ)
- [x] 2022 = outlier exclu par defaut des regressions
- [x] Net position : positif = export, negatif = import

---

## M. LIMITES CONNUES (Partie N)

- [x] BESS SoC simplifie (reset journalier) — documente dans README
- [x] Regime D_tail = P90 NRL positif (pas proxy de rarete) — documente
- [x] Prix mecaniste = proxy deterministe (pas projection de marche) — documente
- [x] 2022 outlier — documente
- [x] Net position fallback si API echoue — documente

---

## N. DEVIATIONS PAR RAPPORT A LA SPEC

| # | Deviation | Raison | Impact |
|---|---|---|---|
| 1 | `data_loader.py` contient `load_country_config()`, `load_all_countries_config()`, `load_thresholds()`, `load_scenarios_config()` | Fonctions referencees dans app.py (J.1) mais pas definies dans la spec G.2. Necessaires au fonctionnement. | Aucun — additions compatibles |
| 2 | `test_gas_co2_scenario_changes_tca` : n=200 au lieu de n=24 | Le TCA fallback P75 rolling (min_periods=168) produit NaN avec seulement 24 heures. n=200 corrige. | Aucun — le test verifie la meme logique |
| 3 | `src/__init__.py` vide | La spec ne precise pas le contenu. | Aucun |
| 4 | `test_phase_diagnostics.py` : `encoding='utf-8'` ajoute a `open()` | Spec n'a pas l'encoding, mais Windows cp1252 ne lit pas les emojis UTF-8 sans (piege #12 CLAUDE.md). | Aucun — compatibilite Windows |

## N.bis ECARTS CORRIGES LORS DU 2e AUDIT (2026-02-06)

| # | Ecart detecte | Correction |
|---|---|---|
| 1 | `thresholds.yaml` labels sans emojis | Labels corriges : `"⚠️ Approche Stage 2"`, `"🔴 Stage 2 sévère"`, `"🔴 IR élevé"`, `"🔴 Flex insuffisante"` |
| 2 | `scenarios.yaml` noms/descriptions sans accents | Corriges : `"Accélération PV"`, `"Électrification forte"`, `"CO2 élevé"`, `"Réduction nucléaire FR"`, `"Gaz élevé"`, `"Stress combiné"` + descriptions avec `€`, `→`, `×` |
| 3 | `export_utils.py` imports incomplets | Ajoute `PatternFill` et `get_column_letter` (spec G.8 ligne 1824) |
| 4 | `export_utils.py` noms onglets sans accents | Corriges : `'Métriques'` au lieu de `'Metriques'` (Excel + GSheets) |
| 5 | `phase_diagnostics.py` import yaml manquant | Ajoute `import yaml` (spec G.5 ligne 1362) |
| 6 | `app.py` fonction `load_and_process` manquante | Ajoutee avec `@st.cache_data(ttl=3600)` (spec J.1) |

---

## O. CHECKLIST FINALE

- [x] 36 fichiers crees
- [x] 12/12 tests passent
- [x] Arborescence conforme a la spec
- [x] Constantes physiques conformes
- [x] Colonnes harmonisees conformes
- [x] 5 pays configures
- [x] 7 scenarios predefinis
- [x] 4 stages + 4 alertes
- [x] 9 modules backend
- [x] 9 pages Streamlit (app.py + 8 pages)
- [x] Regle anti-circularite respectee
- [x] VRE = % generation (pas % demand)
- [x] Floor mode MR ne depasse jamais observed
- [x] Prix JAMAIS interpoles
- [x] BESS SoC avec reset journalier
- [x] Logging vers capture_prices.log + console
- [ ] Lancement `streamlit run app.py` a tester manuellement
