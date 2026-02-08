"""Page 7 - Guide utilisateur."""

from __future__ import annotations

import streamlit as st

from src.commentary_bridge import so_what_block
from src.ui_helpers import info_card, inject_global_css, render_commentary, section_header

st.set_page_config(page_title="Guide utilisateur", page_icon="📘", layout="wide")
inject_global_css()

st.title("📘 Guide Utilisateur")
st.caption("Guide d'usage, interpretation et bonnes pratiques")

c1, c2, c3 = st.columns(3)
with c1:
    info_card("Premier usage", "Section 1: installation et lancement.")
with c2:
    info_card("Importer vos donnees", "Section 2: format CSV attendu.")
with c3:
    info_card("Interpretation", "Sections 3-5: lecture correcte des analyses.")

with st.expander("1) Demarrage rapide", expanded=True):
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.markdown(
        """
1. Renseigner la cle ENTSO-E si le cache raw n'est pas deja present.
2. Choisir pays/periode/modes depuis la page d'accueil.
3. Cliquer "Charger donnees".
4. Naviguer sur les pages d'analyse.
        """
    )

with st.expander("2) Definitions metriques pivots", expanded=True):
    st.markdown(
        """
- SR = surplus annuel / generation annuelle totale
- FAR = surplus absorbe / surplus brut (NaN si surplus nul)
- IR = P10(must-run) / P10(load)
- TCA = ancre thermique (gaz/CO2/eta/VOM)
- TTL = P95(price_used) sur regimes C + D
        """
    )

with st.expander("3) Comment interpreter les ecrans"):
    st.markdown(
        """
- Analyse Historique: trajectoire annuelle (bascule, degradation, pression prix)
- NRL Deep Dive: mecanisme physique horaire (surplus/flex/regimes)
- Capture Rates: pente de cannibalisation (sensibilite penetration -> valeur)
- Comparaison Pays: positionnement relatif multi-pays
- Scenarios: deltas structurels sous hypotheses deterministes
- Questions S. Michel: reponses business explicites + preuves chiffrees
        """
    )

with st.expander("4) Bonnes pratiques de rigueur"):
    st.markdown(
        """
- Toujours verifier data_completeness et regime_coherence avant conclusion forte.
- Distinguer price_used (analyse) et price_obs (validation observables marche).
- Ne pas inferrer de causalite forte depuis une simple correlation.
- Documenter les hypotheses actives (page Sources/Hypotheses).
        """
    )

with st.expander("5) Limites du modele"):
    st.markdown(
        """
- Le prix scenario est synthetique (affine par regimes), pas un forecast spot transactionnel.
- Le BESS est simplifie (SoC deterministe, sans optimisation economique).
- Le diagnostic de phase est un score interpretable, pas un modele causal econometrique.
        """
    )

section_header("Cadrage methodologique", "Ce que disent et ne disent pas les resultats")
render_commentary(
    so_what_block(
        title="Guide d'interpretation",
        purpose="Assurer une lecture homogene, objective et defendable des analyses",
        observed={"ecrans_analyse": 6, "ratios_pivots": 4, "regimes": 4},
        method_link="Conventions v3: separation stricte structurel vs observable, anti-circularite des regimes.",
        limits="Le modele structure la discussion de decision; il ne remplace pas une etude de dispatch detaillee.",
        n=1,
    )
)
