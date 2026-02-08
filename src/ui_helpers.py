"""Shared Streamlit UI helpers (visual system + narrative components)."""

from __future__ import annotations

import streamlit as st

from src.state_adapter import normalize_metrics_record, normalize_state_metrics as _normalize_state_metrics


GLOBAL_CSS = """
<style>
    h1 { font-size: 1.85rem !important; font-weight: 700 !important; color: #1b2a4a !important; }
    h2 { font-size: 1.35rem !important; font-weight: 650 !important; color: #1b2a4a !important; }
    h3 { font-size: 1.15rem !important; font-weight: 650 !important; color: #2c3e6b !important; }

    .block-container { padding-top: 1.8rem; padding-bottom: 1.8rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

    [data-testid="stMetric"] {
        background: #f0f4fa;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #0066cc;
    }

    .info-card {
        background: #f0f4fa;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        border: 1px solid #d6e4f0;
    }
    .info-card h4 { margin-top: 0; color: #0066cc; font-size: 0.95rem; }
    .info-card p { margin-bottom: 0.2rem; color: #3a4a6b; font-size: 0.88rem; }

    .narrative-box {
        background: #ebf5fb;
        border-left: 4px solid #0066cc;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.9rem;
        color: #2c3e50;
        line-height: 1.5;
    }

    .commentary-box {
        background: #f8fafc;
        border-left: 4px solid #0f766e;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0 1rem 0;
        color: #1f2937;
        line-height: 1.45;
    }

    .guard-message {
        text-align: center;
        padding: 2.6rem 2rem;
        background: #f8f9fb;
        border-radius: 12px;
        border: 2px dashed #cbd5e1;
    }
    .guard-message h3 { color: #64748b; font-weight: 560; }
    .guard-message p { color: #94a3b8; }

    .question-banner {
        background: #0066cc;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 1.05rem;
        font-weight: 620;
    }

    .dynamic-narrative-info {
        background: #ebf5fb; border-left: 4px solid #0066cc;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #2c3e50; line-height: 1.5;
    }
    .dynamic-narrative-warning {
        background: #fff3e0; border-left: 4px solid #e65100;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #3e2723; line-height: 1.5;
    }
    .dynamic-narrative-alert {
        background: #ffebee; border-left: 4px solid #c62828;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #3e2723; line-height: 1.5;
    }
    .dynamic-narrative-success {
        background: #e8f5e9; border-left: 4px solid #2e7d32;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #1b5e20; line-height: 1.5;
    }

    .challenge-block {
        background: #fff3e0; border-left: 4px solid #e65100;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0; font-size: 0.9rem; color: #3e2723; line-height: 1.5;
    }
    .challenge-block strong { color: #e65100; }

    [data-testid="stDataFrame"] { font-size: 0.86rem; }
</style>
"""


def inject_global_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def guard_no_data(page_name: str = "cette page") -> None:
    st.markdown(
        f"""
        <div class="guard-message">
            <h3>Donnees non chargees</h3>
            <p>Pour utiliser {page_name}, revenez sur la page d'accueil,
            selectionnez vos pays/annees puis cliquez <strong>Charger donnees</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def section(title: str, subtitle: str | None = None) -> None:
    section_header(title, subtitle or "")


def info_card(title: str, body: str) -> None:
    st.markdown(
        f"""<div class="info-card">
        <h4>{title}</h4>
        <p>{body}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def narrative(text: str) -> None:
    st.markdown(f'<div class="narrative-box">{text}</div>', unsafe_allow_html=True)


def render_commentary(md_text: str) -> None:
    st.markdown(f'<div class="commentary-box">{md_text}</div>', unsafe_allow_html=True)


def question_banner(text: str) -> None:
    st.markdown(f'<div class="question-banner">{text}</div>', unsafe_allow_html=True)


def dynamic_narrative(text: str, severity: str = "info") -> None:
    css_class = f"dynamic-narrative-{severity}"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def challenge_block(title: str, body: str) -> None:
    st.markdown(
        f"""<div class="challenge-block">
        <strong>{title}</strong><br>
        <span>{body}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def normalize_state_metrics(state: dict | None) -> None:
    _normalize_state_metrics(state)


__all__ = [
    "inject_global_css",
    "guard_no_data",
    "section_header",
    "section",
    "info_card",
    "narrative",
    "render_commentary",
    "question_banner",
    "dynamic_narrative",
    "challenge_block",
    "normalize_metrics_record",
    "normalize_state_metrics",
]
