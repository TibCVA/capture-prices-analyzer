# Capture Prices Analyzer — CVA

Outil d'analyse empirique des capture prices des energies renouvelables (VRE) en Europe.
Developpe par CVA pour TotalEnergies.

## Deploiement

**Streamlit Cloud** : L'app est deployee automatiquement depuis ce repo.
Push sur `main` → redeploy automatique (~2 min).

**Local** :
```bash
git clone https://github.com/TibCVA/capture-prices-analyzer.git
cd capture-prices-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
app.py                          # Entree principale + dashboard
pages/
  0_📖_Comprendre_le_Modèle.py  # Explication pedagogique du modele NRL
  1_📊_Analyse_Historique.py     # Series temporelles, regimes, prix
  2_🔬_NRL_Deep_Dive.py         # NRL scatter, heatmap, distributions
  3_📈_Capture_Rates.py         # Capture rates, cannibalisation, TTL
  4_🗺️_Comparaison_Pays.py     # Comparaison multi-pays
  5_🔮_Scenarios.py             # Moteur de scenarios prospectifs
  6_❓_Questions_S_Michel.py    # Q&A analyste (6 questions TotalEnergies)
  7_📘_Guide_Utilisateur.py     # Guide utilisateur
  8_📋_Sources_Hypotheses.py    # Sources, hypotheses, constantes
src/
  constants.py        # Noms colonnes, constantes physiques, palettes
  data_fetcher.py     # Client API ENTSO-E (entsoe-py)
  data_loader.py      # Load/save parquet, configs YAML, scan cache
  export_utils.py     # Export Google Sheets (optionnel)
  metrics.py          # compute_annual_metrics() → ~50 indicateurs
  nrl_engine.py       # compute_nrl() → NRL, regimes, surplus, flex
  phase_diagnostics.py # diagnose_phase() → Stage 1-4
  scenario_engine.py  # apply_scenario() → delta VRE/demande/flex → recalcul
  slope_analysis.py   # Regression lineaire sur capture prices
  ui_helpers.py       # CSS global, narrative boxes, metric cards
config/
  countries.yaml      # Capacites flex, config thermique, listes must-run par pays
  scenarios.yaml      # 7 scenarios predefinits
  thresholds.yaml     # Seuils des phases (vre_share, sr, far...)
data/
  processed/          # 50 parquets (FR/DE/ES/PL/DK × 2015-2024, commites)
  raw/                # Donnees brutes ENTSO-E (gitignore)
tests/                # 12 tests (pytest)
```

## Modele

```
Donnees ENTSO-E horaires (load, generation par filiere, prix day-ahead)
        |
NRL = Load - VRE - MustRun     (Net Residual Load)
        |
4 regimes :
  A = surplus > flex (prix negatifs)
  B = surplus absorbe par flex (prix bas ~BESS cycling cost)
  C = thermique marginal (prix ~ TCA)
  D_tail = queue haute NRL (prix scarcity)
        |
~50 metriques annuelles (capture rates, FAR, heures negatives, TTL...)
        |
Diagnostic de phase (Stage 1-4) + Scenarios prospectifs
```

## Donnees

- **Source** : ENTSO-E Transparency Platform via `entsoe-py`
- **Pays** : FR, DE, ES, PL, DK (Ouest)
- **Periode** : 2015-2024 (horaire)
- **50 fichiers processed** deja commites → pas besoin de cle API pour utiliser l'app
- **2022 exclu** par defaut des regressions (crise gaziere = outlier)

## Configuration pays

Fichier `config/countries.yaml`. Deux modes must-run :
- **observed** : somme des productions observees des filieres must-run
- **floor** : planchers techniques (pour scenarios)

## Scenarios predefinits

7 scenarios dans `config/scenarios.yaml` :
1. Acceleration PV (+20 GW PV, +5 GW BESS)
2. Boom eolien offshore (+15 GW offshore, +10 GW onshore)
3. Flexibilite massive (+10 GW BESS, +40 GWh stockage)
4. Electrification demande (+15% demande, +5 GW midi, +3 GW soir)
5. Mix equilibre (+10 GW PV, +5 GW wind, +3 GW BESS, +5% demande)
6. Reduction nucleaire (-5 GW must-run, +5 GW PV, +3 GW wind)
7. Stress test (prix gaz 80€, CO2 120€, -3 GW must-run, +15 GW PV)

## Tests

```bash
python -m pytest tests/ -v
```
12 tests couvrant : metriques, NRL engine, phase diagnostics, scenarios.

## Conventions

- Penetration VRE = % of total generation (pas % of demand)
- Prix en EUR/MWh
- Puissances en MW dans le code, GW dans les configs YAML
- Index temporel en UTC
- FAR structural = surplus / flex_domestic (PSH+BESS+DSM, sans exports)

## Limites connues

- BESS SoC simplifie (reset journalier, pas d'optimisation inter-day)
- Regime D_tail = queue statistique P90 du NRL positif, pas un vrai proxy de rarete
- Prix mecaniste en scenario = proxy deterministe, pas une projection de marche
- 2022 exclu par defaut des regressions (crise gaziere)
- PL slope p=0.095 (non significatif au seuil 5%)
