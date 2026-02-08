"""Page 7 - Guide utilisateur."""

from __future__ import annotations

import streamlit as st

from src.commentary_engine import commentary_block
from src.ui_helpers import inject_global_css, render_commentary, section

st.set_page_config(page_title="Guide utilisateur", page_icon="📘", layout="wide")
inject_global_css()

st.title("📘 Guide utilisateur")

section("Demarrage", "Installation et lancement")
st.code(
    "pip install -r requirements.txt\n"
    "streamlit run app.py",
    language="bash",
)
st.markdown(
    "1. Renseigner `ENTSOE_API_KEY` si le cache raw est absent.\n"
    "2. Choisir pays/periode/modes en page d'accueil.\n"
    "3. Charger les donnees puis naviguer dans les pages d'analyse."
)

render_commentary(
    commentary_block(
        title="Guide demarrage",
        n_label="etapes",
        n_value=3,
        observed={"python_min": 3.11, "streamlit_min": 1.30},
        method_link="Le pipeline backend suit la spec v3.0 (pure functions + I/O isole).",
        limits="Sans API key, seules les donnees deja en cache local sont utilisables.",
    )
)

section("Definitions metriques", "SR / FAR / IR / TCA / TTL")
st.markdown(
    "- SR = surplus annuel / generation annuelle totale.\n"
    "- FAR = surplus absorbe / surplus brut (NaN si pas de surplus).\n"
    "- IR = P10(must_run) / P10(load).\n"
    "- TCA = ancre thermique (gaz/CO2/eta/VOM).\n"
    "- TTL = P95(price_used) sur regimes C+D."
)

render_commentary(
    commentary_block(
        title="Coherence methodologique",
        n_label="metriques pivots",
        n_value=5,
        observed={"price_modes": 2, "regimes": 4},
        method_link="Separation stricte structurel vs observable conforme section E.1.",
        limits="L'interpretation doit distinguer price_used (analyse) et price_da observe (validation).",
    )
)

section("Bonnes pratiques", "Interpretation rigoureuse")
st.markdown(
    "- Eviter toute causalite speculative; prioriser des relations mesurees.\n"
    "- Verifier `data_completeness` et `regime_coherence` avant conclusion.\n"
    "- Documenter les hypotheses scenario et le mode de prix utilise."
)

render_commentary(
    commentary_block(
        title="Cadre d'interpretation",
        n_label="regles",
        n_value=3,
        observed={"coherence_target_pct": 55.0},
        method_link="Le seuil de coherence >55% est un critere de stabilite narrative (thresholds.yaml).",
        limits="Un score eleve n'implique pas une preuve causale, seulement une adequation regime/prix.",
    )
)
