"""Shared Streamlit UI helpers (visual system + narrative components)."""

from __future__ import annotations

import html

import streamlit as st

from src.state_adapter import normalize_metrics_record, normalize_state_metrics as _normalize_state_metrics


GLOBAL_CSS = """
<style>
    h1 { font-size: 1.86rem !important; font-weight: 700 !important; color: #1b2a4a !important; letter-spacing: 0.1px; }
    h2 { font-size: 1.38rem !important; font-weight: 650 !important; color: #1b2a4a !important; letter-spacing: 0.1px; }
    h3 { font-size: 1.16rem !important; font-weight: 650 !important; color: #2c3e6b !important; letter-spacing: 0.1px; }
    .block-container { padding-top: 1.7rem; padding-bottom: 1.9rem; max-width: 1320px; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #f0f4fa 0%, #f6f8fc 100%);
        border-radius: 11px;
        padding: 12px 16px;
        border: 1px solid #d6e4f0;
        border-left: 4px solid #0066cc;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stDataFrame"] { font-size: 0.86rem; }

    .info-card {
        background: linear-gradient(180deg, #f0f4fa 0%, #f6f8fc 100%);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.95rem;
        border: 1px solid #d6e4f0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    .info-card h4 { margin-top: 0; margin-bottom: 0.45rem; color: #005bb2; font-size: 0.95rem; }
    .info-card p { margin-bottom: 0.1rem; color: #334155; font-size: 0.89rem; line-height: 1.45; }

    .narrative-box {
        background: #ebf5fb;
        border-left: 4px solid #0066cc;
        border-radius: 0 8px 8px 0;
        padding: 0.82rem 1rem;
        margin: 0.45rem 0 1rem 0;
        font-size: 0.91rem;
        color: #1f3a56;
        line-height: 1.5;
    }

    .commentary-box-analysis {
        background: #f8fafc;
        border-left: 4px solid #0f766e;
        border-radius: 0 8px 8px 0;
        padding: 0.92rem 1.06rem;
        margin: 0.44rem 0 1rem 0;
        color: #1f2937;
        line-height: 1.5;
    }
    .commentary-box-method {
        background: #eef6ff;
        border-left: 4px solid #1d4ed8;
        border-radius: 0 8px 8px 0;
        padding: 0.86rem 1.02rem;
        margin: 0.44rem 0 1rem 0;
        color: #1e3a8a;
        line-height: 1.5;
    }
    .commentary-box-warning {
        background: #fff7ed;
        border-left: 4px solid #ea580c;
        border-radius: 0 8px 8px 0;
        padding: 0.86rem 1.02rem;
        margin: 0.44rem 0 1rem 0;
        color: #7c2d12;
        line-height: 1.5;
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
        padding: 0.82rem 1.16rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 1.03rem;
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
    .challenge-block strong { color: #b54708; }

    .kpi-banner {
        border-radius: 12px;
        border: 1px solid #d6e4f0;
        padding: 0.8rem 1rem 0.7rem 1rem;
        margin: 0.3rem 0 0.9rem 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    .kpi-banner-strong { background: linear-gradient(180deg, #ecfdf3 0%, #f4fff8 100%); border-left: 4px solid #15803d; }
    .kpi-banner-medium { background: linear-gradient(180deg, #eff6ff 0%, #f7fbff 100%); border-left: 4px solid #1d4ed8; }
    .kpi-banner-weak { background: linear-gradient(180deg, #fff7ed 0%, #fffaf3 100%); border-left: 4px solid #ea580c; }
    .kpi-banner-unknown { background: linear-gradient(180deg, #f8fafc 0%, #fbfdff 100%); border-left: 4px solid #64748b; }
    .kpi-banner .label { font-size: 0.84rem; color: #475569; font-weight: 620; letter-spacing: 0.08px; }
    .kpi-banner .value { font-size: 1.55rem; color: #0f172a; font-weight: 720; line-height: 1.18; margin-top: 0.08rem; }
    .kpi-banner .subtitle { font-size: 0.83rem; color: #334155; margin-top: 0.18rem; line-height: 1.35; }

    .analysis-note p {
        margin: 0.22rem 0;
        line-height: 1.5;
    }
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
        <h4>{html.escape(title)}</h4>
        <p>{html.escape(body)}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def narrative(text: str) -> None:
    st.markdown(f'<div class="narrative-box">{text}</div>', unsafe_allow_html=True)


def render_commentary(md_text: str, variant: str = "analysis") -> None:
    css = {
        "analysis": "commentary-box-analysis",
        "method": "commentary-box-method",
        "warning": "commentary-box-warning",
    }.get(variant, "commentary-box-analysis")
    if "<" in md_text and ">" in md_text:
        body = md_text
    else:
        body = html.escape(md_text).replace("\n", "<br>")
    st.markdown(f'<div class="{css}">{body}</div>', unsafe_allow_html=True)


def render_analysis_note(md_text: str, variant: str = "analysis") -> None:
    render_commentary(md_text, variant=variant)


def render_kpi_banner(title: str, value: str, subtitle: str = "", status: str = "unknown") -> None:
    safe_status = status if status in {"strong", "medium", "weak", "unknown"} else "unknown"
    st.markdown(
        f"""
        <div class="kpi-banner kpi-banner-{safe_status}">
            <div class="label">{html.escape(title)}</div>
            <div class="value">{html.escape(value)}</div>
            <div class="subtitle">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def question_banner(text: str) -> None:
    st.markdown(f'<div class="question-banner">{text}</div>', unsafe_allow_html=True)


def dynamic_narrative(text: str, severity: str = "info") -> None:
    css_class = f"dynamic-narrative-{severity}"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def challenge_block(title: str, body: str) -> None:
    st.markdown(
        f"""<div class="challenge-block">
        <strong>{html.escape(title)}</strong><br>
        <span>{html.escape(body)}</span>
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
    "render_analysis_note",
    "render_kpi_banner",
    "question_banner",
    "dynamic_narrative",
    "challenge_block",
    "normalize_metrics_record",
    "normalize_state_metrics",
]
