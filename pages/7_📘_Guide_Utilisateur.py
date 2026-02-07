"""
Page 7 -- Guide Utilisateur
Documentation integree pour la prise en main de l'outil Capture Prices Analyzer.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.constants import *
from src.ui_helpers import inject_global_css

st.set_page_config(page_title="Guide Utilisateur", page_icon="📘", layout="wide")
inject_global_css()

st.title("📘 Guide Utilisateur")
st.caption("Tout ce qu'il faut pour demarrer et utiliser l'outil.")

# Quick links
col_q1, col_q2, col_q3 = st.columns(3)
with col_q1:
    st.markdown('''<div class="info-card">
        <h4>Premier usage ?</h4>
        <p>Ouvrez la section 1 ci-dessous.</p>
    </div>''', unsafe_allow_html=True)
with col_q2:
    st.markdown('''<div class="info-card">
        <h4>Import CSV ?</h4>
        <p>Section 2 : format et colonnes attendus.</p>
    </div>''', unsafe_allow_html=True)
with col_q3:
    st.markdown('''<div class="info-card">
        <h4>Coherence faible ?</h4>
        <p>Section 3 : ajuster le must-run.</p>
    </div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. PREMIER LANCEMENT
# ══════════════════════════════════════════════════════════════════════════
with st.expander("🚀 1. Premier lancement", expanded=False):
    st.markdown("""
### Installation

```bash
# 1. Cloner le repo
git clone <url-du-repo> capture-prices-analyzer
cd capture-prices-analyzer

# 2. Creer un environnement virtuel
python -m venv .venv
# Windows :
.venv\\Scripts\\activate
# Linux/Mac :
source .venv/bin/activate

# 3. Installer les dependances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

### Configuration de la cle API ENTSO-E

L'application a besoin d'une cle API ENTSO-E Transparency Platform pour telecharger
les donnees de generation et de prix.

1. Creer un compte sur [transparency.entsoe.eu](https://transparency.entsoe.eu/)
2. Demander une cle API dans les parametres du compte
3. Configurer la variable d'environnement :

```bash
# Windows (PowerShell)
$env:ENTSOE_API_KEY = "votre-cle-ici"

# Linux / Mac
export ENTSOE_API_KEY="votre-cle-ici"
```

Ou creer un fichier `.env` a la racine du projet (jamais commite) :

```
ENTSOE_API_KEY=votre-cle-ici
```

### Premiere execution

1. Selectionner les pays et annees dans le panneau de gauche
2. Cliquer sur **Charger les donnees**
3. Explorer les pages d'analyse via le menu
""")

# ══════════════════════════════════════════════════════════════════════════
# 2. IMPORT CSV
# ══════════════════════════════════════════════════════════════════════════
with st.expander("📄 2. Import CSV", expanded=False):
    st.markdown(f"""
### Format attendu

Les fichiers CSV doivent etre places dans `data/raw/` avec le nommage :
`<COUNTRY>_<YEAR>.csv` (ex: `FR_2023.csv`).

### Colonnes obligatoires

| Colonne | Description | Unite |
|---------|-------------|-------|
| `{COL_TS}` | Horodatage UTC (ISO 8601) | datetime |
| `{COL_LOAD}` | Charge totale | MW |
| `{COL_SOLAR}` | Production solaire | MW |
| `{COL_WIND_ON}` | Production eolien onshore | MW |
| `{COL_PRICE_DA}` | Prix day-ahead | EUR/MWh |

### Colonnes optionnelles

| Colonne | Description |
|---------|-------------|
| `{COL_WIND_OFF}` | Eolien offshore (0 si absent) |
| `{COL_NUCLEAR}` | Nucleaire |
| `{COL_LIGNITE}` | Lignite |
| `{COL_COAL}` | Charbon |
| `{COL_GAS}` | Gaz |
| `{COL_HYDRO_ROR}` | Hydraulique fil de l'eau |
| `{COL_HYDRO_RES}` | Hydraulique reservoir |
| `{COL_PSH_GEN}` | Turbinage PSH |
| `{COL_PSH_PUMP}` | Pompage PSH |
| `{COL_BIOMASS}` | Biomasse |
| `{COL_NET_POSITION}` | Position nette (export - import) |

### Frequence

- **Horaire** (une ligne par heure UTC). 8760 lignes pour une annee standard, 8784 pour une annee bissextile.
- Les donnees infra-horaires seront resamplees a l'heure (moyenne).
- Les trous de donnees sont marques automatiquement (`has_gap = True`).
""")

# ══════════════════════════════════════════════════════════════════════════
# 3. AJUSTER MUST-RUN
# ══════════════════════════════════════════════════════════════════════════
with st.expander("🔧 3. Ajuster le must-run", expanded=False):
    st.markdown("""
### Fichier de configuration

Le must-run de chaque pays est defini dans `config/countries.yaml` :

```yaml
countries:
  FR:
    must_run:
      nuclear_pct: 0.75       # % du nucleaire considere must-run
      hydro_ror_pct: 1.0      # Hydraulique fil de l'eau = 100% must-run
      biomass_pct: 1.0
      lignite_pct: 0.5
      other_pct: 0.5
    thermal:
      marginal_tech: CCGT
    flex:
      capacity:
        psh_pump_capacity_gw: 4.5
        bess_power_gw: 0.5
        bess_energy_gwh: 2.0
        dsm_gw: 2.0
        export_max_gw: 17.0
```

### Workflow de calibration

1. Lancer l'analyse avec les parametres par defaut
2. Verifier la **coherence de regime** sur la page NRL Deep Dive (cible > 55%)
3. Si coherence faible :
   - Must-run trop haut → trop de surplus → trop de Regime A → baisser `nuclear_pct`
   - Must-run trop bas → pas assez de surplus → Regime A sous-estime → augmenter les pcts
4. Ajuster `config/countries.yaml` et relancer
5. Iterer jusqu'a obtenir une coherence > 55% sur les annees recentes

### Conseil

Commencez par la France (nucleaire dominant, parametres bien documentes) puis ajustez
les autres pays par analogie.
""")

# ══════════════════════════════════════════════════════════════════════════
# 4. WORKFLOWS TYPIQUES
# ══════════════════════════════════════════════════════════════════════════
with st.expander("📋 4. Workflows typiques", expanded=False):
    st.markdown("""
### Comprendre un pays

1. **Accueil** : selectionner pays et annees, cliquer Charger
2. **Analyse Historique** : observer les tendances multi-annees
3. **NRL Deep Dive** : examiner la structure horaire
4. **Capture Rates** : mesurer la cannibalisation

### Comparer des marches

1. Charger 3+ pays sur l'accueil
2. **Comparaison Pays** : radar + tableau comparatif
3. **Questions S. Michel > Q2** : comparer les regressions entre pays

### Tester un scenario

1. Charger un pays baseline
2. **Scenarios** : ajuster les curseurs (VRE, stockage, commodites)
3. Cliquer Calculer et observer les deltas de metriques
4. Comparer la courbe de duree NRL baseline vs scenario

### Repondre a Stephane Michel

1. **Questions S. Michel** : 6 onglets prepares
2. Chaque onglet contient la question, une reponse synthetique et un graphique
3. Utiliser les tableaux sous-jacents pour documenter les reponses
""")

# ══════════════════════════════════════════════════════════════════════════
# 5. LIMITES CONNUES
# ══════════════════════════════════════════════════════════════════════════
with st.expander("⚠️ 5. Limites connues", expanded=False):
    st.markdown("""
### Simplifications du modele

| Limite | Impact | Mitigation |
|--------|--------|------------|
| **BESS SoC simplifie** | Le modele de stockage utilise un reset journalier du SoC et ne modelise pas les cycles multi-jours | Acceptable pour des analyses annuelles ; sous-estime legerement le FAR domestique |
| **FAR = flex domestique** | Le FAR structural exclut les exports (interconnexions) car lors de surplus VRE correles, les voisins ne peuvent pas absorber. Inclut uniquement PSH + BESS + DSM | Mesure conservatrice ; les exports sont pris en compte pour la classification des regimes A/B |
| **D_tail statistique** | Le regime D_tail est defini par le P90 de la NRL positive, pas par un modele de scarcity | Proxy correct pour identifier les heures de tension ; ne remplace pas un modele de adequacy |
| **Prix mecaniste = proxy** | Le prix affine par morceaux ne modelise pas le merit order complet | Utile pour les tendances et comparaisons ; ne pas utiliser pour du pricing |
| **2022 = outlier** | La crise gaziere de 2022 perturbe les regressions | Exclure 2022 des regressions (option cochee par defaut) |
| **Net position API-dependant** | La net position depend des donnees ENTSO-E qui peuvent etre incompletes | Verifier la completude des donnees |

### Donnees manquantes

- L'eolien offshore est absent pour certains pays/annees (mis a 0)
- Le pompage PSH n'est pas toujours disponible (impact sur FAR_observed)
- Les prix negatifs avant 2016 sont rares dans les donnees ENTSO-E (possible biais)
""")

# ══════════════════════════════════════════════════════════════════════════
# 6. RACCOURCIS
# ══════════════════════════════════════════════════════════════════════════
with st.expander("⌨️ 6. Raccourcis clavier", expanded=False):
    st.markdown("""
### Raccourcis Streamlit

| Raccourci | Action |
|-----------|--------|
| **R** | Rerun l'application (relance le script) |
| **C** | Vider le cache Streamlit |
| **Ctrl+C** | Arreter le serveur Streamlit (dans le terminal) |

### Raccourcis navigateur

| Raccourci | Action |
|-----------|--------|
| **Ctrl+F** | Rechercher dans la page |
| **Ctrl+Shift+I** | Ouvrir la console developpeur |
| **F5** | Rafraichir la page (equivalent a R) |

### Astuces

- En cas de comportement etrange, lancez un **C** (clear cache) puis **R** (rerun).
- Pour debugger un scenario, verifiez les metriques baseline avant de comparer.
""")
