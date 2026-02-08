"""Dynamic analytical commentary generation (French, objective, quantified)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _fmt(value, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "NaN"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if np.isnan(v):
        return "NaN"
    return f"{v:.{digits}f}{unit}"


def so_what_block(
    title: str,
    purpose: str,
    observed: Mapping[str, float | int | None],
    method_link: str,
    limits: str,
    n: int,
) -> str:
    """Standardized commentary block for every chart/screen."""

    vals = ", ".join(f"{k}={_fmt(v)}" for k, v in observed.items()) if observed else "-"
    return (
        f"**{title}**\n"
        f"- Constat chiffre: n={n}; {vals}.\n"
        f"- So what: {purpose}.\n"
        f"- Lien methode: {method_link}.\n"
        f"- Limites/portee: {limits}."
    )


def commentary_block(
    title: str,
    n_label: str,
    n_value: int,
    observed: Mapping[str, float | int | None],
    method_link: str,
    limits: str,
) -> str:
    """Backward-compatible wrapper."""

    return so_what_block(
        title=title,
        purpose=f"Lecture sur {n_label}",
        observed=observed,
        method_link=method_link,
        limits=limits,
        n=n_value,
    )


def comment_kpi(metrics: dict, label: str = "KPI") -> str:
    return so_what_block(
        title=label,
        purpose="Qualification rapide du stade de stress et de la capacite d'absorption du systeme",
        observed={
            "SR": metrics.get("sr"),
            "FAR": metrics.get("far"),
            "IR": metrics.get("ir"),
            "TTL": metrics.get("ttl"),
            "capture_ratio_pv": metrics.get("capture_ratio_pv"),
        },
        method_link="Ratios calcules selon G.7 (SR/FAR/IR/TTL) et capture sur price_used.",
        limits="Interpretation valable sous reserve de completude des donnees et du mode de prix selectionne.",
        n=1,
    )


def comment_regression(slope: dict, x_name: str, y_name: str) -> str:
    return so_what_block(
        title=f"Regression {y_name} vs {x_name}",
        purpose="Quantifier la pente de degradation ou d'amelioration et sa robustesse statistique",
        observed={
            "slope": slope.get("slope"),
            "r_squared": slope.get("r_squared"),
            "p_value": slope.get("p_value"),
        },
        method_link="Regression lineaire scipy.stats.linregress, exclusion optionnelle des outliers.",
        limits="Association statistique uniquement; pas d'inference causale sans identification complementaire.",
        n=int(slope.get("n_points", 0) or 0),
    )


def comment_distribution(metrics: dict, title: str = "Distribution") -> str:
    n_hours = int(
        (metrics.get("h_regime_a") or 0)
        + (metrics.get("h_regime_b") or 0)
        + (metrics.get("h_regime_c") or 0)
        + (metrics.get("h_regime_d") or 0)
    )
    return so_what_block(
        title=title,
        purpose="Lire la structure physique des heures et le niveau de saturation du systeme",
        observed={
            "h_A": metrics.get("h_regime_a"),
            "h_B": metrics.get("h_regime_b"),
            "h_C": metrics.get("h_regime_c"),
            "h_D": metrics.get("h_regime_d"),
            "coherence": metrics.get("regime_coherence"),
        },
        method_link="Regimes classes uniquement sur variables physiques (anti-circularite G.6).",
        limits="Un score de coherence faible signale une divergence modele/marche, pas necessairement une erreur de code.",
        n=n_hours,
    )


def comment_scenario_delta(base: dict, scen: dict) -> str:
    obs = {
        "d_SR": (scen.get("sr") or np.nan) - (base.get("sr") or np.nan),
        "d_FAR": (scen.get("far") or np.nan) - (base.get("far") or np.nan),
        "d_h_A": (scen.get("h_regime_a") or np.nan) - (base.get("h_regime_a") or np.nan),
        "d_TTL": (scen.get("ttl") or np.nan) - (base.get("ttl") or np.nan),
        "d_capture_ratio_pv": (scen.get("capture_ratio_pv") or np.nan)
        - (base.get("capture_ratio_pv") or np.nan),
    }
    return so_what_block(
        title="Impact scenario",
        purpose="Identifier si le scenario reduit la saturation (A) et ameliore l'absorption (FAR)",
        observed=obs,
        method_link="Recalcul complet NRL -> regimes -> TCA -> price_synth -> metrics (F.1, G.9).",
        limits="Resultats indicatifs sur prix synthetique; pas de prevision spot DA reelle.",
        n=1,
    )
