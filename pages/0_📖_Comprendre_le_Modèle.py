"""
Page 0 -- Comprendre le Modele
Framework analytique : merit order, NRL, regimes, phases, ratios.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.constants import *
from src.ui_helpers import inject_global_css, narrative

st.set_page_config(page_title="Comprendre le Modele", page_icon="📖", layout="wide")
inject_global_css()
st.title("📖 Comprendre le Modele")

narrative("Cette page presente le cadre analytique complet de l'outil. "
          "Chaque section explique non seulement QUOI est mesure, "
          "mais POURQUOI cette approche a ete choisie.")

# ══════════════════════════════════════════════════════════════════════════
# Section 0 : Pourquoi cette analyse ?
# ══════════════════════════════════════════════════════════════════════════
with st.expander("**Pourquoi cette analyse ?**", expanded=True):
    st.markdown("""
Les investisseurs en energies renouvelables (solaire, eolien) dependent des **revenus marchands** :
le prix auquel ils vendent leur electricite sur le marche spot. Or, plus les VRE se deployent,
plus elles produisent simultanement, ce qui **comprime les prix** aux heures de forte production.

Ce phenomene, appele **cannibalisation**, reduit progressivement la rentabilite des nouveaux projets.
La question centrale est : **a quel rythme cette degradation se produit-elle, et quels leviers
(stockage, flexibilite, interconnexions) peuvent la ralentir ?**

Cet outil quantifie la cannibalisation sur 5 marches europeens (2015-2024) a partir de donnees
publiques (ENTSO-E), en utilisant un framework analytique base sur la **Net Residual Load (NRL)**.
    """)

# ══════════════════════════════════════════════════════════════════════════
# Section 1 : Le merit order et le capture price
# ══════════════════════════════════════════════════════════════════════════
with st.expander("**Le merit order et le capture price**"):
    st.markdown("""
#### Le mecanisme du merit order

Sur un marche de l'electricite, les centrales sont appelees par **ordre de cout marginal croissant** :
d'abord les VRE (cout ~0), puis le nucleaire, le charbon, le gaz, et enfin les turbines de pointe.
Le **prix spot** est fixe par la derniere centrale necessaire pour satisfaire la demande.

Quand la production VRE augmente :
1. Elle se substitue aux centrales plus couteuses (gaz, charbon)
2. Le prix spot baisse aux heures de forte production VRE
3. Les VRE captent un prix moyen **inferieur** au prix moyen du marche
    """)

    # Schema merit order simplifie
    st.markdown("""
<div style="display:flex; align-items:flex-end; gap:4px; margin:1rem 0; padding:1rem; background:#FAFBFC; border-radius:8px;">
    <div style="width:15%; height:40px; background:#27AE60; border-radius:4px 4px 0 0; text-align:center; font-size:0.7rem; color:white; padding-top:12px;">VRE<br>~0€</div>
    <div style="width:15%; height:60px; background:#3498DB; border-radius:4px 4px 0 0; text-align:center; font-size:0.7rem; color:white; padding-top:20px;">Nucleaire<br>~12€</div>
    <div style="width:15%; height:90px; background:#8E44AD; border-radius:4px 4px 0 0; text-align:center; font-size:0.7rem; color:white; padding-top:35px;">Lignite<br>~35€</div>
    <div style="width:15%; height:120px; background:#E67E22; border-radius:4px 4px 0 0; text-align:center; font-size:0.7rem; color:white; padding-top:50px;">Charbon<br>~55€</div>
    <div style="width:15%; height:160px; background:#E74C3C; border-radius:4px 4px 0 0; text-align:center; font-size:0.7rem; color:white; padding-top:70px;">CCGT<br>~75€</div>
    <div style="width:15%; height:200px; background:#C0392B; border-radius:4px 4px 0 0; text-align:center; font-size:0.7rem; color:white; padding-top:90px;">OCGT<br>~120€</div>
</div>
<p style="text-align:center; font-size:0.8rem; color:#7F8C8D; margin-top:-0.5rem;">
    Schema simplifie du merit order. Le prix spot = cout marginal de la derniere unite appelee.
</p>
    """, unsafe_allow_html=True)

    st.markdown("""
#### Le capture price et le capture ratio

- **Capture price PV** = prix moyen pondere par la production solaire = `sum(prix × solar) / sum(solar)`
- **Capture ratio PV** = capture price / prix moyen baseload

Un capture ratio de **0.85** signifie que le solaire capte 85% du prix moyen.
En dessous de **0.80**, on parle de cannibalisation significative.
En dessous de **0.60**, le modele merchant est en difficulte structurelle.

Le capture ratio est l'indicateur central car il est **comparable** entre pays et entre annees,
independamment du niveau absolu des prix.
    """)

# ══════════════════════════════════════════════════════════════════════════
# Section 2 : Pourquoi le NRL ?
# ══════════════════════════════════════════════════════════════════════════
with st.expander("**Pourquoi la Net Residual Load (NRL) ?**"):
    st.markdown("""
#### La variable qui structure le marche

La **Net Residual Load** est le concept central de l'analyse. Elle est utilisee par l'IEA,
l'ACER (regulateur europeen de l'energie) et dans la litterature academique
(Hirth 2013, *The Market Value of Variable Renewables*).

**Formule :**
    """)
    st.latex(r"NRL = \text{Demande} - \text{VRE} - \text{Must-Run}")
    st.markdown("""
**Pourquoi chaque terme ?**

| Terme | Definition | Justification |
|-------|-----------|---------------|
| **Demande** (Load) | Consommation electrique totale | Le besoin a satisfaire |
| **VRE** | Solaire + Eolien onshore + offshore | Production fatale a cout marginal ~0 |
| **Must-Run** | Nucleaire + Hydro fil-de-l'eau + Biomasse | Production **incompressible** : ne peut pas etre reduite a court terme |

#### Pourquoi pas simplement "Load - VRE" ?

La demande residuelle simple (`Load - VRE`) ignorerait le must-run. Or, dans un pays comme la
France avec ~40 GW de nucleaire, cette production incompressible **comprime mecaniquement la NRL**
et accelere l'apparition de surplus. C'est un facteur structurel majeur.

#### Ce que revele la NRL

- **NRL > 0** : Il reste de la demande a satisfaire par des centrales thermiques dispatchables (gaz, charbon).
  Le prix suit le cout marginal de ces centrales.
- **NRL < 0** : La VRE + le must-run depassent la demande. Il y a **surplus**.
  Le prix s'effondre, souvent en territoire negatif.
- **NRL = 0** : Point d'equilibre exact. En pratique, zone de transition avec prix tres volatils.
    """)

    # Graphique 48h si donnees disponibles
    if st.session_state.get("processed_data"):
        first_key = next(iter(st.session_state["processed_data"]))
        df_sample = st.session_state["processed_data"][first_key].head(48)
        if COL_LOAD in df_sample.columns and COL_VRE in df_sample.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df_sample[COL_LOAD], mode='lines', name='Demande',
                line=dict(color='#1B2A4A', width=2)))
            fig.add_trace(go.Scatter(
                y=df_sample[COL_VRE] + df_sample[COL_MUST_RUN],
                mode='lines', name='VRE + Must-Run',
                line=dict(color='#27AE60', width=2)))
            fig.add_trace(go.Scatter(
                y=df_sample[COL_NRL], mode='lines', name='NRL',
                line=dict(color='#E74C3C', width=2, dash='dash')))
            fig.add_hline(y=0, line_dash="dot", line_color="grey")
            fig.update_layout(
                title=f"Exemple 48h -- {first_key[0]} {first_key[1]}",
                yaxis_title="MW", height=350,
                **PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# Section 3 : Les 4 regimes
# ══════════════════════════════════════════════════════════════════════════
with st.expander("**Les 4 regimes de prix -- construction et justification**"):
    st.markdown("""
#### Pourquoi 4 regimes ?

Le prix de l'electricite ne reagit pas de la meme facon selon la zone de la courbe de merit order :

- Quand il y a surplus (NRL < 0), le prix depend de la **capacite d'absorption** du systeme
- Quand la NRL est positive, le prix depend du **cout marginal** des centrales thermiques
- Les queues de distribution (extremes) ont un comportement specifique

D'ou 4 regimes, classes par **variables physiques** (pas par le prix, pour eviter la circularite) :

| Regime | Condition | Prix attendu | Explication physique |
|--------|-----------|-------------|---------------------|
| **A** (Surplus non absorbe) | `surplus_unabsorbed > 0` | Negatif ou ~0 | Le surplus depasse la flexibilite : les producteurs paient pour ecouler |
| **B** (Surplus absorbe) | `surplus > 0` ET `surplus_unabsorbed = 0` | Bas positif (0-30€) | Le surplus existe mais est absorbe (stockage, exports) |
| **C** (Thermique) | `NRL > 0` ET `NRL <= P90` | ~TCA (cout marginal) | Les centrales thermiques fixent le prix selon gaz/CO2 |
| **D_tail** (Queue haute) | `NRL > P90(NRL positive)` | Eleve (> TCA) | Pointes de demande, rarete, prix de stress |

#### Le principe d'anti-circularite

**Point methodologique crucial** : la classification en regimes est basee UNIQUEMENT sur des
variables physiques (NRL, surplus, flex capacity). Le prix n'intervient **jamais** dans la
classification. La coherence prix/regime est un **test de validation** a posteriori, pas un
critere de construction.

Si la coherence est faible (< 55%), cela signifie que le modele et le marche divergent.
Les causes possibles : must-run mal calibre, exports non comptabilises, mecanismes de capacite.
    """)

    # Tableau des conditions
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Formules exactes :**
- `VRE = solar + wind_on + wind_off`
- `NRL = load - VRE - must_run`
- `surplus = max(0, -NRL)`
- `surplus_unabs = max(0, surplus - flex_total)` (flex_total = domestique + exports)
        """)
    with col2:
        st.markdown("""
**Seuils de classification :**
- Regime A : `surplus_unabs > 0`
- Regime B : `surplus > 0` ET PAS A
- Regime C : `NRL > 0` ET `NRL <= P90`
- Regime D_tail : `NRL > P90(NRL+)`
        """)

# ══════════════════════════════════════════════════════════════════════════
# Section 4 : Les 4 stades
# ══════════════════════════════════════════════════════════════════════════
with st.expander("**Les 4 stades d'un marche**"):
    st.markdown("""
#### De l'integration facile a la saturation

Ce framework s'inspire des **phases d'integration VRE** decrites par l'IEA (WEO 2014, 2019)
et adapte au contexte europeen. Chaque stade correspond a un niveau de stress du systeme :

| Stade | Nom | Part VRE (ref. IEA) | Criteres cles du scoring |
|-------|-----|---------------------|--------------------------|
| **1** | Integration facile | ~ 0-15% | H negatives < 100, capture ratio > 0.85, SR < 1% |
| **2** | Premiers surplus | variable | H negatives >= 200, capture ratio < 0.80, H below 5 >= 500 |
| **3** | Absorption structurelle | VRE >= 20% ET SR >= 0.5% requis | FAR domestique >= 0.60, h negatives en baisse malgre VRE croissante |
| **4** | Saturation | VRE >= 35% requis | FAR domestique >= 0.90, h regime C < 1500 |

*Les fourchettes VRE sont indicatives. Le diagnostic repose sur un scoring multi-criteres, PAS sur le niveau de VRE seul. Exemple : le Danemark a 75% de VRE en 2024 mais est classe Stage 2 (flex domestique = 0.8 GW seulement).*

**Exemples (2024)** : France ~ Stage 2 (14.7% VRE), Allemagne ~ Stage 2 (47.7% VRE), Espagne ~ Stage 2 (43.9% VRE), Pologne ~ Stage 2 (26.3% VRE), Danemark ~ Stage 2 (75.1% VRE, flex domestique insuffisante)

#### Comment le diagnostic est-il calcule ?

Le diagnostic utilise un **systeme de scoring par points** (0-10 par stade).
Pour chaque annee et pays, des points sont attribues selon des criteres mesurables :

**Stage 1** (max 10 pts) :
- +3 si h_negatives < 100
- +3 si capture_ratio_pv > 0.85
- +2 si surplus_ratio < 0.01
- +2 si h_below_5 < 200

**Stage 2** (max 10 pts) :
- +2 si h_negatives >= 200 | +2 supplementaire si >= 300
- +2 si h_below_5 >= 500
- +2 si capture_ratio_pv < 0.80 | +1 supplementaire si < 0.70
- +1 si days_spread_50 >= 150

**Stage 3** (max ~11 pts) — *requis : VRE >= 20% ET surplus ratio >= 0.5%* :
- +3 si FAR domestique >= 0.60 | +2 supplementaire si >= 0.80
- +3 si VRE en hausse ET h_negatives en baisse (inter-annuel)
- +2 si VRE > 40% ET h_negatives < 200

**Stage 4** (max 8 pts) — *requis : VRE >= 35%* :
- +3 si FAR domestique >= 0.90
- +3 si h_regime_c < 1500
- +2 si h_regime_d_tail < 500

Le stade avec le **score le plus eleve** est retenu. La **confiance** = score max / somme des scores.
Une confiance > 70% indique un diagnostic clair ; < 50% signale une ambiguite.

**Limitation** : ce scoring est heuristique, pas un modele econometrique. Il est concu pour donner
une orientation structurelle, pas un verdict precis.
    """)

# ══════════════════════════════════════════════════════════════════════════
# Section 5 : Les ratios du modele
# ══════════════════════════════════════════════════════════════════════════
with st.expander("**Les ratios du modele**"):
    st.markdown("""
Le modele calcule ~50 metriques par pays/annee. Les 4 ratios structurels sont :
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
#### SR (Surplus Ratio)
**Formule** : `SR = sum(surplus) / sum(load)`

**Pourquoi** : Mesure la fraction de la demande qui est "en trop" (couverte par un exces de VRE + must-run).
Un SR de 2% signifie que l'equivalent de 2% de la consommation annuelle est produit en surplus.

**Interpretation** : SR < 1% = Stage 1. SR > 5% = saturation avancee.

---

#### FAR (Flex Absorption Ratio)
**Formule structurel** : `FAR = sum(min(surplus, flex_domestic)) / sum(surplus)`

**Pourquoi** : Mesure la capacite **domestique** du systeme a absorber les surplus.
La flex domestique = PSH (pompage-turbinage) + BESS (batteries) + DSM (effacement).
Les **exports sont exclus** car lors de surplus VRE correles a l'echelle europeenne,
les interconnexions ne sont pas une flex fiable.

Un FAR de 0.80 signifie que 80% du surplus peut etre absorbe par les moyens domestiques.

**Interpretation** : FAR > 0.60 = capacite d'absorption correcte. FAR < 0.30 = goulot structurel.
        """)
    with col2:
        st.markdown("""
#### IR (Inflexibility Ratio)
**Formule** : `IR = P10(must_run) / P10(load)`

**Pourquoi** : Mesure la rigidite du systeme. Si le must-run represente une part elevee de la
demande minimale, le systeme est mecaniquement contraint de produire des surplus des que les VRE arrivent.

**Interpretation** : IR > 0.60 = systeme tres rigide (ex: France avec nucleaire).

---

#### TTL (Thermal Tail Level)
**Formule** : `TTL = P95(prix | regime in {C, D_tail})`

**Pourquoi** : Estime le cout de la "queue thermique" -- les heures les plus cheres du dispatch
thermique. C'est un proxy du prix de stress du systeme.

**Interpretation** : TTL depend fortement du prix du gaz et du CO2.
        """)

# ══════════════════════════════════════════════════════════════════════════
# Section 6 : Comment lire les graphiques
# ══════════════════════════════════════════════════════════════════════════
with st.expander("**Comment lire les graphiques**"):
    st.markdown("""
| Type de graphique | Pages | Ce qu'il montre |
|------------------|-------|----------------|
| **Area chart (empile)** | NRL Deep Dive | Decomposition horaire : qui produit quoi, quand le surplus apparait |
| **Scatter (nuage de points)** | Capture Rates, NRL Deep Dive | Relations entre variables (VRE vs capture, NRL vs prix) |
| **Barres empilees 100%** | Analyse Historique | Repartition des heures entre les 4 regimes |
| **Radar** | Comparaison Pays | Profil multi-dimensionnel d'un marche (6 axes normalises) |
| **Heatmap (mois x heure)** | Capture Rates | Structure temporelle des prix : effet solaire en ete, pointes hiver |
| **Duration curve** | Capture Rates | Classement des 8760 prix du plus cher au moins cher |
| **Courbe parametrique** | Questions S. Michel | Impact d'un parametre (BESS, CO2) sur une metrique (FAR, TCA) |
    """)

# ══════════════════════════════════════════════════════════════════════════
# Section 7 : Ce que le modele ne capture PAS
# ══════════════════════════════════════════════════════════════════════════
with st.expander("**Ce que le modele ne capture PAS**"):
    st.markdown("""
Aucun modele n'est complet. Voici les **limitations explicites** de cet outil :

| Limitation | Impact | Alternative possible |
|-----------|--------|---------------------|
| **Pas de merit order complet** | Le prix n'est pas modelise par dispatch economique | Les prix mecanistes du scenario engine sont un proxy affine par morceaux |
| **BESS SoC reset journalier** | Sous-estime legerement le FAR (pas d'arbitrage inter-jours) | Extension possible avec SoC continu |
| **D_tail = P90 (statistique)** | Ne modelise pas la rarete physique (capacity adequacy) | Necessiterait un modele de probabilite de defaillance |
| **Interconnexions statiques** | L'export max est un plafond fixe, pas un dispatch cross-border | Necessiterait un modele de couplage de marches |
| **2022 = outlier** | La crise gaziere est exclue des regressions par defaut | L'exclusion est configurable (case a cocher) |
| **Pas de mecanismes de capacite** | Les capacity markets / strategic reserves ne sont pas modelises | Affecte surtout la coherence regime/prix sur certains marches |
| **Pas de curtailment explicite** | Le surplus est calcule, pas le curtailment reel | Le surplus est un majorant du curtailment |

**En resume** : L'outil est concu pour l'**analyse structurelle** (tendances, comparaisons,
ordres de grandeur), pas pour le **pricing precis** ou la **prevision**.
    """)
