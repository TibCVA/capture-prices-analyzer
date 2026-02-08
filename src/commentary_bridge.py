"""Compatibility bridge for commentary imports across UI versions."""

from __future__ import annotations

from collections.abc import Mapping


def _fmt_value(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


try:
    from src.commentary_engine import (
        analysis_note,
        comment_kpi,
        comment_regression,
        comment_scenario_delta,
        so_what_block,
    )
except Exception:
    try:
        from src.commentary_engine import commentary_block as _legacy_commentary_block
    except Exception:
        _legacy_commentary_block = None

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
        if _legacy_commentary_block is not None:
            return _legacy_commentary_block(
                title=title,
                n_label="points",
                n_value=n,
                observed=observed,
                method_link=f"{method_link}. So what: {purpose}",
                limits=limits,
            )
        vals = ", ".join(f"{k}={_fmt_value(v)}" for k, v in observed.items()) if observed else "-"
        implication_txt = implication or purpose
        decision_txt = decision_use or "Utiliser ce resultat pour calibrer les decisions de pilotage."
        return (
            f"**{title}**\n"
            f"- Constat chiffre: n={n}; {vals}.\n"
            f"- Ce que cela signifie: {implication_txt}.\n"
            f"- Pourquoi cette analyse sert a decider: {decision_txt}.\n"
            f"- Lien methode: {method_link}.\n"
            f"- Limites: {limits}."
        )

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
        decision_txt = decision_use or "Utiliser ce resultat pour la priorisation des leviers."
        suffix = f" Niveau de confiance={_fmt_value(confidence)}." if confidence is not None else ""
        return (
            f"**{title}**\n"
            f"- Objectif de l'analyse: {objective}.\n"
            f"- Comment lire le graphique: {reading}.\n"
            f"- Constat chiffre: {findings}.\n"
            f"- Ce que cela signifie: {implication}.\n"
            f"- Pourquoi cette analyse sert a decider: {decision_txt}.\n"
            f"- Lien methode: {method}.\n"
            f"- Limites: {limits}.\n"
            f"- Base de calcul: n={n}.{suffix}"
        )

    def comment_kpi(metrics: dict, label: str = "KPI") -> str:
        return so_what_block(
            title=label,
            purpose="Qualification rapide du niveau de stress systeme",
            observed={
                "SR": metrics.get("sr"),
                "FAR": metrics.get("far"),
                "IR": metrics.get("ir"),
                "TTL": metrics.get("ttl"),
                "capture_ratio_pv": metrics.get("capture_ratio_pv"),
            },
            method_link="Ratios pivots SR/FAR/IR/TTL",
            limits="Interpretation sous reserve de qualite donnees.",
            n=1,
        )

    def comment_regression(slope: dict, x_name: str, y_name: str) -> str:
        return so_what_block(
            title=f"Regression {y_name} vs {x_name}",
            purpose="Quantifier la pente et la robustesse statistique",
            observed={
                "slope": slope.get("slope"),
                "r_squared": slope.get("r_squared"),
                "p_value": slope.get("p_value"),
            },
            method_link="linregress",
            limits="Association statistique uniquement.",
            n=int(slope.get("n_points", 0) or 0),
        )

    def comment_scenario_delta(base: dict, scen: dict) -> str:
        return so_what_block(
            title="Impact scenario",
            purpose="Lire les deltas structurels baseline vs scenario",
            observed={
                "d_SR": (scen.get("sr") or 0.0) - (base.get("sr") or 0.0),
                "d_FAR": (scen.get("far") or 0.0) - (base.get("far") or 0.0),
                "d_h_A": (scen.get("h_regime_a") or 0.0) - (base.get("h_regime_a") or 0.0),
            },
            method_link="Recalcul complet scenario",
            limits="Prix scenario indicatif.",
            n=1,
        )
