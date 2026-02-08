"""Dynamic analytical commentary generation (French, objective, quantified)."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _fmt(value, digits=2, unit="") -> str:
    if value is None:
        return "NaN"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if np.isnan(v):
        return "NaN"
    return f"{v:.{digits}f}{unit}"


def commentary_block(
    title: str,
    n_label: str,
    n_value: int,
    observed: Mapping[str, float | int | None],
    method_link: str,
    limits: str,
) -> str:
    """Return a normalized 3-part commentary block."""

    vals = ", ".join(f"{k}={_fmt(v)}" for k, v in observed.items())
    return (
        f"**{title}**\n"
        f"- Constat chiffre: n={n_value} ({n_label}); {vals}.\n"
        f"- Lien methode: {method_link}.\n"
        f"- Limites/portee: {limits}."
    )


def comment_kpi(metrics: dict, label: str = "KPI") -> str:
    return commentary_block(
        title=label,
        n_label="annee",
        n_value=1,
        observed={
            "SR": metrics.get("sr"),
            "FAR": metrics.get("far"),
            "IR": metrics.get("ir"),
            "TTL": metrics.get("ttl"),
            "capture_ratio_pv": metrics.get("capture_ratio_pv"),
        },
        method_link="Ratios calcules selon G.7 (SR/FAR/IR/TTL) et capture sur price_used.",
        limits="Interpretation valable sous reserve de completude des donnees et du mode de prix selectionne.",
    )


def comment_regression(slope: dict, x_name: str, y_name: str) -> str:
    return commentary_block(
        title=f"Regression {y_name} vs {x_name}",
        n_label="points",
        n_value=int(slope.get("n_points", 0) or 0),
        observed={
            "slope": slope.get("slope"),
            "r_squared": slope.get("r_squared"),
            "p_value": slope.get("p_value"),
        },
        method_link="Regression lineaire scipy.stats.linregress, exclusion optionnelle des outliers.",
        limits="Association statistique uniquement; pas d'inference causale sans identification complementaire.",
    )


def comment_distribution(metrics: dict, title: str = "Distribution") -> str:
    return commentary_block(
        title=title,
        n_label="heures annuelles",
        n_value=int(
            (metrics.get("h_regime_a") or 0)
            + (metrics.get("h_regime_b") or 0)
            + (metrics.get("h_regime_c") or 0)
            + (metrics.get("h_regime_d") or 0)
        ),
        observed={
            "h_A": metrics.get("h_regime_a"),
            "h_B": metrics.get("h_regime_b"),
            "h_C": metrics.get("h_regime_c"),
            "h_D": metrics.get("h_regime_d"),
            "coherence": metrics.get("regime_coherence"),
        },
        method_link="Regimes classes uniquement sur variables physiques (anti-circularite G.6).",
        limits="Un score de coherence faible signale une divergence modele/marche, pas necessairement une erreur de code.",
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
    return commentary_block(
        title="Impact scenario",
        n_label="comparaison",
        n_value=1,
        observed=obs,
        method_link="Recalcul complet NRL->regimes->TCA->price_synth->metrics (F.1, G.9).",
        limits="Resultats indicatifs sur prix synthetique; pas de prevision spot DA reelle.",
    )
