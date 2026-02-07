# Capture Prices Analyzer -- CVA

Outil d'analyse empirique des capture prices des energies renouvelables (VRE) en Europe.
Developpe par CVA pour TotalEnergies.

## Prerequis

- Python 3.11+
- Cle API ENTSO-E (optionnelle si donnees CSV pre-chargees)

## Installation

```bash
git clone <repo_url>
cd capture-prices-analyzer
pip install -r requirements.txt
```

## Configuration

Creer un fichier `.env` a la racine :

```
ENTSOE_API_KEY=votre_cle_api
# Optionnel :
# GOOGLE_SHEETS_CREDS=path/to/service_account.json
```

## Lancement

```bash
streamlit run app.py
```

## Modele

```
Donnees ENTSO-E horaires (load, generation, prix)
        |
NRL = Load - VRE - MustRun
        |
4 regimes : A (surplus > flex) | B (absorbe) | C (thermique) | D_tail (queue haute)
        |
Metriques annuelles (~50 indicateurs)
        |
Diagnostic de phase (Stage 1-4) + Scenarios prospectifs
```

## Configuration pays

Fichier `config/countries.yaml`. Deux modes de calcul du must-run :

- **observed** : somme directe des productions observees des filieres must-run
- **floor** : planchers techniques (pour les scenarios)

## Import CSV

Placer les fichiers dans `data/raw/` au format `{PAYS}_{ANNEE}.parquet` ou utiliser
l'API ENTSO-E via l'interface.

Colonnes obligatoires : `load_mw`, `solar_mw`, `wind_onshore_mw`, `price_da_eur_mwh`

Colonnes optionnelles : `wind_offshore_mw`, `nuclear_mw`, `lignite_mw`, `coal_mw`,
`gas_mw`, `hydro_ror_mw`, `hydro_reservoir_mw`, `psh_generation_mw`, `psh_pumping_mw`,
`biomass_mw`, `other_mw`, `net_position_mw`

## Export Google Sheets

Necessite un fichier de credentials service account JSON. Chemin dans `.env` :
`GOOGLE_SHEETS_CREDS=path/to/creds.json`

## Tests

```bash
python -m pytest tests/ -v
```

## Conventions

- Penetration RES = % of total generation (pas % of demand)
- Prix en EUR/MWh
- Puissances en MW dans le code, GW dans les configs YAML
- Index temporel en UTC

## Limites connues

- BESS SoC simplifie (reset journalier, pas d'optimisation inter-day)
- Regime D_tail = queue statistique P90 du NRL positif, pas un vrai proxy de rarete
- Prix mecaniste en scenario = proxy deterministe, pas une projection de marche
- 2022 = outlier (crise gaziere) -- exclu par defaut des regressions
- Net position : si l'API `query_net_position` echoue, flex export non disponible
- Peak/offpeak calcule en heure locale, stockage en UTC
