"""Narrative builders for the ExceSum page (French, strict, evidence-first)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def executive_summary(results: dict) -> str:
    metrics = results.get("metrics_df", pd.DataFrame())
    if metrics.empty:
        return (
            "Objectif: produire une synthese robuste des 10 ans sur 5 pays.\n\n"
            "Constat: aucune donnee exploitable n'a ete chargee pour la baseline ExceSum."
        )

    countries = sorted(metrics["country"].dropna().unique())
    years_min = int(metrics["year"].min())
    years_max = int(metrics["year"].max())
    sr_med = _safe_float(metrics["sr"].median())
    far_med = _safe_float(metrics["far"].median())
    coh_med = _safe_float(metrics["regime_coherence"].median()) * 100.0
    n = int(len(metrics))
    return (
        f"Perimetre couvert: {len(countries)} pays ({', '.join(countries)}) sur {years_min}-{years_max}.\n\n"
        f"Constat global (n={n} couples pays/annee): SR median={sr_med:.3f}, FAR median={far_med:.3f}, "
        f"coherence regime/prix mediane={coh_med:.1f}%.\n\n"
        "Lecture: SR mesure la pression de surplus, FAR la capacite d'absorption, et la coherence valide la "
        "compatibilite entre classification physique et prix observes.\n\n"
        "Portee: ces conclusions decrivent des regularites structurelles et ne constituent pas une prevision spot."
    )


def q_block_title(q_code: str) -> str:
    labels = {
        "Q1": "Seuils de bascule vers stage_2",
        "Q2": "Pente capture ratio PV vs penetration",
        "Q3": "Conditions de transition stage_2 -> stage_3",
        "Q4": "Effet marginal des batteries",
        "Q5": "Sensibilite CO2/gaz sur TTL",
        "Q6": "Stockage chaleur/froid: perimetre de preuve",
    }
    return labels.get(q_code, q_code)


def q1_text(q1_country: pd.DataFrame) -> str:
    if q1_country.empty:
        return "Q1: impossible de conclure faute de points valides."
    crossed = int(q1_country["latest_cross_all"].fillna(False).sum())
    total = int(len(q1_country))
    return (
        f"Q1: {crossed}/{total} pays franchissent simultanement les seuils stage_2 sur la derniere annee disponible. "
        "La lecture doit rester pays-par-pays via l'annee de premier franchissement."
    )


def q2_text(q2_df: pd.DataFrame) -> str:
    if q2_df.empty:
        return "Q2: aucune regression exploitable."
    neg = int((q2_df["slope"] < 0).sum())
    sig = int((q2_df["p_value"] <= 0.05).sum())
    total = int(len(q2_df))
    return (
        f"Q2: {neg}/{total} pays ont une pente negative (degradation de capture ratio quand la penetration PV augmente). "
        f"{sig}/{total} pentes sont statistiquement significatives (p<=0.05)."
    )


def q3_text(q3_df: pd.DataFrame) -> str:
    if q3_df.empty:
        return "Q3: statut de transition indisponible."
    counts = q3_df["status_transition_2_to_3"].value_counts(dropna=False).to_dict()
    return (
        "Q3: statut de transition observe - "
        f"validee={counts.get('transition_validee', 0)}, "
        f"partielle={counts.get('transition_partielle', 0)}, "
        f"non_validee={counts.get('transition_non_validee', 0)}."
    )


def q4_text(q4_df: pd.DataFrame) -> str:
    if q4_df.empty:
        return "Q4: analyse batterie indisponible."
    plateau = int(q4_df["plateau_baseline"].fillna(False).sum())
    stress = int(q4_df["stress_found"].fillna(False).sum())
    delta = pd.to_numeric(q4_df.get("stress_delta_pv_gw"), errors="coerce")
    zero_delta = int(((delta.fillna(np.nan).abs() <= 1e-9) & q4_df["stress_found"].fillna(False)).sum())
    positive_delta = int(((delta.fillna(np.nan) > 1e-9) & q4_df["stress_found"].fillna(False)).sum())
    total = int(len(q4_df))
    return (
        f"Q4: plateau baseline sur {plateau}/{total} pays. Effet BESS deja identifiable en baseline "
        f"(delta PV=0) sur {zero_delta}/{total} pays; stress PV additionnel requis sur {positive_delta}/{total}. "
        f"Reference exploitable au total sur {stress}/{total} pays."
    )


def q5_text(q5_df: pd.DataFrame) -> str:
    if q5_df.empty:
        return "Q5: sensibilites CO2/gaz indisponibles."
    d_co2 = _safe_float(q5_df["delta_ttl_high_co2"].median())
    d_gas = _safe_float(q5_df["delta_ttl_high_gas"].median())
    return (
        f"Q5: variation mediane de TTL sous stress CO2={d_co2:.2f} EUR/MWh et sous stress gaz={d_gas:.2f} EUR/MWh. "
        "La direction attendue est une hausse de TTL lorsque CO2 ou gaz augmentent."
    )


def q6_text(q6_df: pd.DataFrame) -> str:
    if q6_df.empty:
        return "Q6: perimetre de preuve non evalue."
    missing = int((~q6_df["heat_cold_dataset_available"].fillna(False)).sum())
    return (
        f"Q6: {missing}/{len(q6_df)} pays sans jeu de donnees chaleur/froid dedie. "
        "Conclusion causale non identifiable avec le perimetre actuel; seule une conclusion prudente est valide."
    )


def country_conclusion_markdown(row: pd.Series) -> str:
    return (
        f"**{row['country']}**\n\n"
        f"- Phase recente: `{row['phase_latest']}` ({int(row['latest_year'])}).\n"
        f"- Stress structurel: `SR={_safe_float(row['sr_latest']):.3f}`, `FAR={_safe_float(row['far_latest']):.3f}`, "
        f"`capture_ratio_pv={_safe_float(row['capture_ratio_pv_latest']):.3f}`.\n"
        f"- Q1: premier franchissement stage_2 = `{row['q1_first_stage2_year']}`.\n"
        f"- Q2: pente capture PV = `{_safe_float(row['q2_slope']):.4f}`.\n"
        f"- Q3: statut transition = `{row['q3_status']}`.\n"
        f"- Q4: plateau baseline={bool(row['q4_plateau_baseline'])}, stress trouvé={bool(row['q4_stress_found'])}.\n"
        f"- Q5: delta TTL CO2={_safe_float(row['q5_delta_ttl_co2']):.2f}, delta TTL gaz={_safe_float(row['q5_delta_ttl_gas']):.2f}.\n"
        f"- Q6: `{row['q6_status']}`."
    )


__all__ = [
    "executive_summary",
    "q_block_title",
    "q1_text",
    "q2_text",
    "q3_text",
    "q4_text",
    "q5_text",
    "q6_text",
    "country_conclusion_markdown",
]
