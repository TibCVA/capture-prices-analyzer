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


def normalize_metric_record(m: dict) -> dict:
    out = dict(m)
    if "h_negative_obs" not in out and "h_negative" in out:
        out["h_negative_obs"] = out["h_negative"]
    if "h_below_5_obs" not in out and "h_below_5" in out:
        out["h_below_5_obs"] = out["h_below_5"]
    if "h_regime_d" not in out and "h_regime_d_tail" in out:
        out["h_regime_d"] = out["h_regime_d_tail"]
    if "far" not in out and "far_structural" in out:
        out["far"] = out["far_structural"]
    if "pv_penetration_pct_gen" not in out and "pv_share" in out:
        out["pv_penetration_pct_gen"] = float(out["pv_share"]) * 100.0
    if "wind_penetration_pct_gen" not in out and "wind_share" in out:
        out["wind_penetration_pct_gen"] = float(out["wind_share"]) * 100.0
    if "vre_penetration_pct_gen" not in out and "vre_share" in out:
        out["vre_penetration_pct_gen"] = float(out["vre_share"]) * 100.0
    return out


def normalize_state_metrics(state: dict | None) -> None:
    if not state or "metrics" not in state or not isinstance(state["metrics"], dict):
        return
    normalized = {}
    for key, val in state["metrics"].items():
        if isinstance(val, dict):
            normalized[key] = normalize_metric_record(val)
        else:
            normalized[key] = val
    state["metrics"] = normalized
