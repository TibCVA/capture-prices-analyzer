"""Page 8 - Sources et hypotheses."""

from __future__ import annotations

import streamlit as st

from src.commentary_engine import commentary_block
from src.ui_helpers import inject_global_css, render_commentary, section

st.set_page_config(page_title="Sources & hypotheses", page_icon="📋", layout="wide")
inject_global_css()

st.title("📋 Sources & hypotheses")

section("Sources de donnees", "Entree du pipeline")
st.markdown(
    "- ENTSO-E: load, generation par filiere, prix day-ahead, net position.\n"
    "- TTF/EUA/Coal (optionnel): ancrage TCA.\n"
    "- BESS capacity historique: baseline capacitaire si disponible."
)

render_commentary(
    commentary_block(
        title="Traçabilite des sources",
        n_label="familles",
        n_value=3,
        observed={"horizon": 2015.0, "horizon_end": 2024.0},
        method_link="I/O isole dans data_fetcher/data_loader, transformations deterministes dans nrl_engine.",
        limits="Qualite depend de la disponibilite ENTSO-E et des fichiers externes locaux.",
    )
)

section("Hypotheses fixes", "Conventions v3.0")
st.markdown(
    "- Interdiction explicite de l'approximation exports = generation - load.\n"
    "- Regimes A/B/C/D classes sans utiliser le prix.\n"
    "- Scenarios deterministes (aucun aleatoire).\n"
    "- Hardcoded numeriques centralises constants.py / YAML uniquement."
)

render_commentary(
    commentary_block(
        title="Regles de gouvernance",
        n_label="regles",
        n_value=4,
        observed={"random_used": 0.0},
        method_link="Fail-fast sur cas interdits et ambiguities (NotImplementedError explicite).",
        limits="Les simplifications de modelisation (BESS, flex capacities) restent des abstractions systeme.",
    )
)

section("Parametrage interpretation", "Points de vigilance")
st.markdown(
    "- Toujours expliciter `must_run_mode`, `flex_model_mode`, `price_mode`.\n"
    "- Verifier la migration cache legacy -> v3 lors de la premiere execution.\n"
    "- Utiliser le rapport `docs/VERIFICATION_V3_AUDIT.md` comme preuve de conformite."
)

render_commentary(
    commentary_block(
        title="Portee des conclusions",
        n_label="parametres",
        n_value=3,
        observed={"price_modes": 2.0, "cache_formats": 2.0},
        method_link="Comparabilite assuree par schemas de colonnes et conventions unifiees.",
        limits="Toute interpretation hors cadre (forecast spot, causalite forte) est hors scope du modele.",
    )
)
