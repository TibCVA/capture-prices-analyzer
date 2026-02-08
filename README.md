# Capture Prices Analyzer (v3.0)

Outil d'analyse outside-in des capture prices renouvelables en Europe.

## Scope
- Historique 2015-2024 (FR/DE/ES/PL/DK)
- Ratios pivots: `SR`, `FAR`, `IR`, `TCA/TTL`
- Scenarios deterministes (sans Monte Carlo)
- UI Streamlit multi-pages avec commentaires analytiques "So what"
- Hypotheses session-modifiables depuis `Sources & Hypotheses`

## Definitions principales
- `NRL = load - VRE - must-run`
- `SR = surplus annuel / generation annuelle totale`
- `FAR = surplus absorbe / surplus brut` (`NaN` si surplus nul)
- `IR = P10(must_run) / P10(load)`
- `TTL = P95(price_used)` sur regimes `C + D`

## Conventions critiques
- Penetration RES en **% de generation annuelle totale**.
- Observables marche (`h_negative_obs`, spreads, etc.) calcules sur `price_da` observe.
- Classification regime `A/B/C/D` strictement physique (anti-circularite).
- Interdit: approximer exports par `generation - load`.

## Note ENTSO-E (load)
`Actual Total Load` ENTSO-E inclut `absorbed energy` (stockage/pompage).
Consequence: le pompage doit etre traite explicitement comme composante de flex; pas de double comptage via `generation-load`.

## Hypotheses modifiables UI
Page `?? Sources & Hypotheses`:
- Rendements/emissions/VOM thermiques
- Parametres BESS de simulation
- Seuils observables prix

Ces hypotheses sont stockees dans `st.session_state.state['ui_overrides']` et appliquees au prochain `Charger donnees`.
Les references par defaut restent `constants.py` et `config/*.yaml`.

## Comment lire les commentaires analytiques
Chaque ecran/graphique suit le format:
1. Constat chiffre
2. So what (implication business)
3. Lien methode
4. Limites/portee

Objectif: interpretation rigoureuse, objective, traçable.

## Structure
- `src/config_loader.py`: validation YAML + resolution code ENTSO-E variable dans le temps
- `src/data_fetcher.py`: fetch ENTSO-E + retry + fallback local auction
- `src/nrl_engine.py`: pipeline physique complet + BESS SoC deterministe + regimes + prix
- `src/metrics.py`: metriques annuelles v3
- `src/scenario_engine.py`: recalcul complet sous perturbations scenario
- `src/phase_diagnostics.py`: classification stage_1..stage_4|unknown
- `src/state_adapter.py`: contrat de schema UI + mapping legacy/v3

## Limites
- Prix scenario: `price_synth` indicatif (affine par regimes, ancre TCA), pas prevision spot reelle.
- BESS simplifie (SoC deterministe, sans optimisation economique).
- Cohérence regime/prix = score de validation, pas preuve causale.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests
```bash
python -m pytest -q
$env:PYTHONWARNINGS='error::RuntimeWarning'; python -m pytest -q
python -m compileall -q app.py src pages tests
```
