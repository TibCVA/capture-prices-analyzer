"""Shared Streamlit UI helpers."""

from __future__ import annotations

import streamlit as st


GLOBAL_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
.small-note { color: #4b5563; font-size: 0.9rem; }
.commentary-box {
  background: #f8fafc;
  border-left: 4px solid #0f766e;
  padding: 0.7rem 0.9rem;
  margin: 0.4rem 0 1rem 0;
  border-radius: 6px;
}
.guard-box {
  border: 1px dashed #94a3b8;
  border-radius: 8px;
  padding: 1rem;
  color: #334155;
  background: #f8fafc;
}
</style>
"""


def inject_global_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def guard_no_data(page_name: str) -> None:
    st.markdown(
        f"<div class='guard-box'><strong>Donnees non chargees.</strong><br>"
        f"Chargez les donnees depuis la page d'accueil pour utiliser {page_name}.</div>",
        unsafe_allow_html=True,
    )
    st.stop()


def render_commentary(md_text: str) -> None:
    st.markdown(f"<div class='commentary-box'>{md_text}</div>", unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)
