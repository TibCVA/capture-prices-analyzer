"""Phase diagnostics v3.0."""

from __future__ import annotations

import math


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _score_stage_1(m: dict, t: dict, rules: list[str]) -> int:
    score = 0
    if m.get("h_negative_obs", 0) <= t["h_negative_max"]:
        score += 1
        rules.append("stage_1:h_negative_max")
    if m.get("h_below_5_obs", 0) <= t["h_below_5_max"]:
        score += 1
        rules.append("stage_1:h_below_5_max")
    if _is_finite(m.get("capture_ratio_pv")) and m["capture_ratio_pv"] >= t["capture_ratio_pv_min"]:
        score += 1
        rules.append("stage_1:capture_ratio_pv_min")
    if _is_finite(m.get("sr")) and m["sr"] <= t["sr_max"]:
        score += 1
        rules.append("stage_1:sr_max")
    return score


def _score_stage_2(m: dict, t: dict, rules: list[str]) -> int:
    score = 0
    if m.get("h_negative_obs", 0) >= t["h_negative_min"]:
        score += 1
        rules.append("stage_2:h_negative_min")
    if m.get("h_negative_obs", 0) >= t["h_negative_strong"]:
        score += 2
        rules.append("stage_2:h_negative_strong")
    if m.get("h_below_5_obs", 0) >= t["h_below_5_min"]:
        score += 1
        rules.append("stage_2:h_below_5_min")
    if _is_finite(m.get("capture_ratio_pv")) and m["capture_ratio_pv"] <= t["capture_ratio_pv_max"]:
        score += 1
        rules.append("stage_2:capture_ratio_pv_max")
    if _is_finite(m.get("capture_ratio_pv")) and m["capture_ratio_pv"] <= t["capture_ratio_pv_crisis"]:
        score += 2
        rules.append("stage_2:capture_ratio_pv_crisis")
    if m.get("days_spread_above_50_obs", 0) >= t["days_spread_50_min"]:
        score += 1
        rules.append("stage_2:days_spread_50_min")
    return score


def _score_stage_3(m: dict, t: dict, rules: list[str], blocked_rules: list[str]) -> int:
    far = m.get("far")
    if not _is_finite(far):
        return 0

    require_declining = bool(t.get("require_h_neg_declining", False))
    if require_declining and m.get("h_negative_declining") is not True:
        blocked_rules.append("stage_3:require_h_neg_declining")
        return 0

    score = 0
    if far >= t["far_min"]:
        score += 1
        rules.append("stage_3:far_min")
    if far >= t["far_strong"]:
        score += 2
        rules.append("stage_3:far_strong")
    return score


def _score_stage_4(m: dict, t: dict, rules: list[str]) -> int:
    far = m.get("far")
    if not _is_finite(far):
        return 0

    score = 0
    if far >= t["far_min"]:
        score += 1
        rules.append("stage_4:far_min")
    if m.get("h_regime_c", 0) <= t["h_regime_c_max"]:
        score += 1
        rules.append("stage_4:h_regime_c_max")
    return score


def _compute_alerts(metrics: dict, alerts_cfg: dict) -> list[dict]:
    alerts = []
    for key, cfg in alerts_cfg.items():
        triggered = True

        if "h_negative_range" in cfg:
            lo, hi = cfg["h_negative_range"]
            hn = metrics.get("h_negative_obs", 0)
            triggered &= lo <= hn <= hi
        if "h_negative_min" in cfg:
            triggered &= metrics.get("h_negative_obs", 0) >= cfg["h_negative_min"]
        if "capture_ratio_pv_range" in cfg:
            cr = metrics.get("capture_ratio_pv")
            if not _is_finite(cr):
                triggered = False
            else:
                lo, hi = cfg["capture_ratio_pv_range"]
                triggered &= lo <= cr <= hi
        if "capture_ratio_pv_max" in cfg:
            cr = metrics.get("capture_ratio_pv")
            triggered &= _is_finite(cr) and cr <= cfg["capture_ratio_pv_max"]
        if "ir_min" in cfg:
            ir = metrics.get("ir")
            triggered &= _is_finite(ir) and ir >= cfg["ir_min"]
        if "far_max" in cfg:
            far = metrics.get("far")
            triggered &= _is_finite(far) and far <= cfg["far_max"]
        if "sr_min" in cfg:
            sr = metrics.get("sr")
            triggered &= _is_finite(sr) and sr >= cfg["sr_min"]

        if triggered:
            alerts.append({"key": key, "label": cfg.get("label", key)})

    return alerts


def diagnose_phase(metrics: dict, thresholds: dict) -> dict:
    """Classify annual metrics into stage_1..stage_4 or unknown per v3.0 rules."""

    phase_cfg = thresholds["phase_thresholds"]

    rules_1: list[str] = []
    rules_2: list[str] = []
    rules_3: list[str] = []
    rules_4: list[str] = []
    blocked_rules: list[str] = []

    s1 = _score_stage_1(metrics, phase_cfg["stage_1"], rules_1)
    s2 = _score_stage_2(metrics, phase_cfg["stage_2"], rules_2)
    s3 = _score_stage_3(metrics, phase_cfg["stage_3"], rules_3, blocked_rules)
    s4 = _score_stage_4(metrics, phase_cfg["stage_4"], rules_4)

    far = metrics.get("far")
    if not _is_finite(far):
        s3 = 0
        s4 = 0
        rules_3 = []
        rules_4 = []

    scores = {
        "stage_1": s1,
        "stage_2": s2,
        "stage_3": s3,
        "stage_4": s4,
    }

    score_max_theorique = max(1, s1 + s2 + s3 + s4)
    best_score = max(scores.values())

    # Candidate requires score >= 2
    candidates = [k for k, v in scores.items() if v >= 2 and v == best_score]

    chosen = "unknown"
    if len(candidates) == 1:
        chosen = candidates[0]
    elif len(candidates) > 1:
        # tie-break: highest stage only if FAR/IR/SR coherent
        order = ["stage_4", "stage_3", "stage_2", "stage_1"]
        candidate_sorted = [s for s in order if s in candidates]
        tentative = candidate_sorted[0]

        # coherence checks
        sr = metrics.get("sr")
        ir = metrics.get("ir")

        coherent = True
        if tentative in {"stage_3", "stage_4"}:
            coherent &= _is_finite(far)
        if tentative == "stage_4":
            coherent &= _is_finite(sr)
            coherent &= _is_finite(ir)

        chosen = tentative if coherent else "unknown"

    confidence = float(best_score / score_max_theorique)
    if confidence < 0.30:
        chosen = "unknown"

    matched = []
    if chosen == "stage_1":
        matched = rules_1
    elif chosen == "stage_2":
        matched = rules_2
    elif chosen == "stage_3":
        matched = rules_3
    elif chosen == "stage_4":
        matched = rules_4

    alerts = _compute_alerts(metrics, thresholds.get("alerts", {}))

    return {
        "phase": chosen,
        "score": int(best_score),
        "confidence": float(confidence),
        "matched_rules": matched,
        "blocked_rules": blocked_rules,
        "alerts": alerts,
    }
