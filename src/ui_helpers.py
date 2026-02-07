"""
Fonctions utilitaires UI pour Capture Prices Analyzer.
Fournit des composants visuels reutilisables (narratifs, guards, headers).
"""
import streamlit as st


# ==================== CSS GLOBAL ====================
GLOBAL_CSS = """
<style>
    /* --- Hierarchie typographique --- */
    h1 { font-size: 1.8rem !important; font-weight: 700 !important; color: #1B2A4A !important; }
    h2 { font-size: 1.4rem !important; font-weight: 600 !important; color: #1B2A4A !important; }
    h3 { font-size: 1.15rem !important; font-weight: 600 !important; color: #2C3E6B !important; }

    /* --- Spacing coherent --- */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background: #F0F4FA;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #0066CC;
    }

    /* --- Sidebar plus aeree --- */
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

    /* --- Containers/Cards --- */
    .info-card {
        background: #F0F4FA;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #D6E4F0;
    }
    .info-card h4 { margin-top: 0; color: #0066CC; font-size: 0.95rem; }
    .info-card p { margin-bottom: 0.3rem; color: #3A4A6B; font-size: 0.88rem; }

    /* --- Narratif contextuel --- */
    .narrative-box {
        background: #EBF5FB;
        border-left: 4px solid #0066CC;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem;
        color: #2C3E50;
        line-height: 1.5;
    }

    /* --- Guard messages --- */
    .guard-message {
        text-align: center;
        padding: 3rem 2rem;
        background: #F8F9FB;
        border-radius: 12px;
        border: 2px dashed #CBD5E1;
    }
    .guard-message h3 { color: #64748B; font-weight: 500; }
    .guard-message p { color: #94A3B8; }

    /* --- Tableaux plus lisibles --- */
    [data-testid="stDataFrame"] { font-size: 0.85rem; }

    /* --- Question banner (page Q&A) --- */
    .question-banner {
        background: #0066CC;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 1.05rem;
        font-weight: 600;
    }

    .dynamic-narrative-info {
        background: #EBF5FB; border-left: 4px solid #0066CC;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #2C3E50; line-height: 1.5;
    }
    .dynamic-narrative-warning {
        background: #FFF3E0; border-left: 4px solid #E65100;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #3E2723; line-height: 1.5;
    }
    .dynamic-narrative-alert {
        background: #FFEBEE; border-left: 4px solid #C62828;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #3E2723; line-height: 1.5;
    }
    .dynamic-narrative-success {
        background: #E8F5E9; border-left: 4px solid #2E7D32;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #1B5E20; line-height: 1.5;
    }
    .challenge-block {
        background: #FFF3E0; border-left: 4px solid #E65100;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #3E2723; line-height: 1.5;
    }
    .challenge-block strong { color: #E65100; }
</style>
"""


def inject_global_css():
    """Injecter le CSS global. Appeler une seule fois par page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def narrative(text: str):
    """Affiche un encadre narratif bleu clair avant un graphique."""
    st.markdown(f'<div class="narrative-box">{text}</div>',
                unsafe_allow_html=True)


def guard_no_data(page_name: str = "cette page"):
    """Affiche un message de garde et arrete la page."""
    st.markdown(f'''
    <div class="guard-message">
        <h3>Donnees non chargees</h3>
        <p>Pour utiliser {page_name}, retournez sur la <strong>page d'accueil</strong>
        (menu a gauche), selectionnez vos pays et annees,
        puis cliquez sur <strong>Charger les donnees</strong>.</p>
    </div>
    ''', unsafe_allow_html=True)
    st.stop()


def section_header(title: str, subtitle: str = ""):
    """Titre de section avec sous-titre optionnel."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def info_card(title: str, body: str):
    """Affiche une carte info stylisee."""
    st.markdown(f'''<div class="info-card">
        <h4>{title}</h4>
        <p>{body}</p>
    </div>''', unsafe_allow_html=True)


def question_banner(text: str):
    """Affiche un bandeau de question (page Q&A)."""
    st.markdown(f'<div class="question-banner">{text}</div>',
                unsafe_allow_html=True)


def dynamic_narrative(text: str, severity: str = "info"):
    """Bloc interpretatif colore selon la severite (info/warning/alert/success)."""
    css_class = f"dynamic-narrative-{severity}"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def challenge_block(title: str, body: str):
    """Bloc d'alerte quand les resultats contredisent les attentes du modele."""
    st.markdown(f'''<div class="challenge-block">
        <strong>⚠ {title}</strong><br>
        <span>{body}</span>
    </div>''', unsafe_allow_html=True)
