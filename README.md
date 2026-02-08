# Capture Prices Analyzer (v3.0)

Outil d'analyse outside-in des capture prices renouvelables en Europe.

## Scope
- Historique 2015-2024 (FR/DE/ES/PL/DK)
- Ratios pivots: `SR`, `FAR`, `IR`, `TCA/TTL`
- Scenarios deterministes (sans Monte Carlo)
- UI Streamlit multi-pages avec commentaires analytiques dynamiques

## Defs principales
- `NRL = load - VRE - must-run`
- `SR = surplus annuel / generation annuelle totale`
- `FAR = surplus absorbe / surplus brut` (`NaN` si surplus nul)
- `IR = P10(must_run) / P10(load)`
- `TTL = P95(price_used)` sur regimes `C + D`

## Conventions critiques
- Penetration RES en **% de generation annuelle totale**.
- Les observables marche (`h_negative_obs`, spreads) sont calcules sur prix observes `price_da`.
- La classification regime `A/B/C/D` utilise uniquement des variables physiques (anti-circularite).
- Interdit: approximer exports par `generation - load`.

## Note ENTSO-E (load)
`Actual Total Load` ENTSO-E incorpore `absorbed energy` (stockage/pompage).
Consequence: le pompage est traite explicitement comme composante de flex et ne doit pas etre double-compte via `generation-load`.

## Structure
- `src/config_loader.py`: validation YAML + resolution code ENTSO-E variable dans le temps
- `src/data_fetcher.py`: fetch ENTSO-E + retry + fallback local auction
- `src/nrl_engine.py`: pipeline physique complet + BESS SoC deterministe + regimes + prix
- `src/metrics.py`: metriques annuelles v3
- `src/scenario_engine.py`: recalcul complet sous perturbations scenario
- `src/phase_diagnostics.py`: classification stage_1..stage_4|unknown

## Limits
- Prix scenario: `price_synth` indicatif (affine par regimes, ancre TCA), pas prevision spot reelle.
- BESS simplifie (SoC deterministe, sans optimisation economique).
- Cohérence regime/prix est un score de validation, pas une preuve causale.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests
```bash
python -m pytest -q
$env:PYTHONWARNINGS='error::RuntimeWarning'; python -m pytest -q
```
