"""Dynamic analytical commentary generation (French, objective, quantified)."""

from __future__ import annotations

import html
from collections.abc import Mapping

import numpy as np


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "NaN"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if np.isnan(v):
        return "NaN"
    return f"{v:.{digits}f}"


def _format_observed(observed: Mapping[str, float | int | None]) -> str:
    if not observed:
        return "-"
    return ", ".join(f"{k}={_fmt(v)}" for k, v in observed.items())


def analysis_note(
    title: str,
    objective: str,
    reading: str,
    findings: str,
    implication: str,
    method: str,
    limits: str,
    n: int,
    confidence: float | None = None,
    decision_use: str | None = None,
) -> str:
    """Return a structured HTML note for clean and readable UI commentary."""

    decision_txt = decision_use or "Prioriser les leviers selon les effets mesures."
    confidence_block = ""
    if confidence is not None:
        try:
            conf = float(confidence)
            if np.isfinite(conf):
                confidence_block = (
                    "<p><strong>Niveau de confiance.</strong> "
                    f"{html.escape(f'{conf:.2f}')}.</p>"
                )
        except Exception:
            confidence_block = ""

    return (
        "<div class='analysis-note'>"
        f"<p><strong>{html.escape(title)}</strong></p>"
        f"<p><strong>Objectif de l'analyse.</strong> {html.escape(objective)}</p>"
        f"<p><strong>Comment lire le graphique.</strong> {html.escape(reading)}</p>"
        f"<p><strong>Constat chiffre.</strong> {html.escape(findings)}</p>"
        f"<p><strong>Ce que cela signifie.</strong> {html.escape(implication)}</p>"
        f"<p><strong>Pourquoi cette analyse sert a decider.</strong> {html.escape(decision_txt)}</p>"
        f"<p><strong>Lien methode.</strong> {html.escape(method)}</p>"
        f"<p><strong>Limites.</strong> {html.escape(limits)}</p>"
        f"<p><strong>Base de calcul.</strong> n={int(n)} observations.</p>"
        f"{confidence_block}"
        "</div>"
    )


def so_what_block(
    title: str,
    purpose: str,
    observed: Mapping[str, float | int | None],
    method_link: str,
    limits: str,
    n: int,
    implication: str | None = None,
    decision_use: str | None = None,
    confidence: float | None = None,
) -> str:
    """Standardized commentary block used on all charts/screens."""

    observed_str = _format_observed(observed)
    implication_txt = implication or purpose
    decision_txt = decision_use or "Prioriser les leviers a activer et calibrer les hypotheses de scenario."

    return analysis_note(
        title=title,
        objective=purpose,
        reading="Lire les variables affichees, verifier leur niveau, puis les relier aux seuils de la methode.",
        findings=f"n={int(n)}; {observed_str}.",
        implication=implication_txt,
        method=method_link,
        limits=limits,
        n=n,
        confidence=confidence,
        decision_use=decision_txt,
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
        purpose="Le systeme combine niveau de surplus, capacite d'absorption et rigidite de creux.",
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
        decision_use="Qualifier rapidement le stade de stress et choisir le prochain levier (flex, must-run, cadence VRE).",
    )


def comment_regression(slope: dict, x_name: str, y_name: str) -> str:
    return so_what_block(
        title=f"Regression {y_name} vs {x_name}",
        purpose="La pente mesure le rythme de degradation/ameliorement; r2 et p-value mesurent la robustesse.",
        observed={
            "slope": slope.get("slope"),
            "r_squared": slope.get("r_squared"),
            "p_value": slope.get("p_value"),
        },
        method_link="Regression lineaire scipy.stats.linregress, exclusion optionnelle des outliers.",
        limits="Association statistique uniquement; pas d'inference causale sans identification complementaire.",
        n=int(slope.get("n_points", 0) or 0),
        decision_use="Comparer la sensibilite des pays et calibrer les hypotheses de transition entre phases.",
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
        purpose="La repartition des heures indique si le stress est transitoire (B/C) ou structurel (A).",
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
        decision_use="Prioriser les leviers de flexibilite sur les regimes dominants plutot que sur la moyenne annuelle seule.",
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
        purpose="Les deltas indiquent si le scenario reduit la saturation et restaure la valeur captee.",
        observed=obs,
        method_link="Recalcul complet NRL -> regimes -> TCA -> price_synth -> metrics (F.1, G.9).",
        limits="Resultats indicatifs sur prix synthetique; pas de prevision spot DA reelle.",
        n=1,
        decision_use="Arbitrer entre leviers (BESS, demande, must-run, commodites) sur des effets mesurables.",
    )

