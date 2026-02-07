import logging
import numpy as np
import yaml

logger = logging.getLogger("capture_prices.phase_diagnostics")


def diagnose_phase(metrics: dict, thresholds: dict,
                   prev_year_metrics: dict | None = None) -> dict:
    """
    Diagnostique le stade d'un marche sur la base de ses metriques annuelles.

    Args:
        metrics: dict retourne par compute_annual_metrics
        thresholds: dict charge depuis thresholds.yaml
        prev_year_metrics: metriques N-1 (pour critere inter-annuel Stage 3)

    Returns:
        {
            'phase': str,         # 'stage_1', 'stage_2', 'stage_3', 'stage_4'
            'phase_number': int,  # 1, 2, 3, 4
            'scores': dict,       # scores par stade (0-10)
            'confidence': float,  # 0-1
            'alerts': list[dict], # liste des alertes declenchees
        }
    """
    t = thresholds['phase_thresholds']

    # ---- Scoring Stage 1 ----
    s1 = 0
    if metrics['h_negative'] < t['stage_1']['h_negative_max']:
        s1 += 3
    if metrics.get('capture_ratio_pv') and metrics['capture_ratio_pv'] > t['stage_1']['capture_ratio_pv_min']:
        s1 += 3
    if metrics['sr'] < t['stage_1']['sr_max']:
        s1 += 2
    if metrics['h_below_5'] < t['stage_1']['h_below_5_max']:
        s1 += 2

    # ---- Scoring Stage 2 ----
    s2 = 0
    if metrics['h_negative'] >= t['stage_2']['h_negative_min']:
        s2 += 2
    if metrics['h_negative'] >= t['stage_2']['h_negative_strong']:
        s2 += 2
    if metrics['h_below_5'] >= t['stage_2']['h_below_5_min']:
        s2 += 2
    if metrics.get('capture_ratio_pv') and metrics['capture_ratio_pv'] < t['stage_2']['capture_ratio_pv_max']:
        s2 += 2
    if metrics.get('capture_ratio_pv') and metrics['capture_ratio_pv'] < t['stage_2']['capture_ratio_pv_crisis']:
        s2 += 1
    if metrics['days_spread_above_50'] >= t['stage_2']['days_spread_50_min']:
        s2 += 1

    # ---- Scoring Stage 3 ----
    # REGLE : Stage 3 = "absorption structurelle" requiert :
    #   1) Un niveau VRE significatif (>= 20%) — sinon c'est juste du Stage 1 tranquille
    #   2) Un surplus significatif (SR >= 0.5%) — sinon le FAR est trivial (rien a absorber)
    s3 = 0
    vre_share = metrics.get('vre_share', 0)
    vre_floor_s3 = t['stage_3'].get('vre_share_min', 0.20)
    sr_floor_s3 = t['stage_3'].get('sr_min', 0.005)
    far = metrics.get('far_structural')

    has_meaningful_surplus = metrics.get('sr', 0) >= sr_floor_s3
    if vre_share >= vre_floor_s3 and has_meaningful_surplus and far is not None and not np.isnan(far):
        if far >= t['stage_3']['far_structural_min']:
            s3 += 3
        if far >= t['stage_3']['far_structural_strong']:
            s3 += 2

        # Critere inter-annuel : H_neg en baisse malgre VRE en hausse
        if prev_year_metrics and t['stage_3'].get('require_h_neg_declining', False):
            vre_up = metrics['vre_share'] > prev_year_metrics.get('vre_share', 0)
            h_neg_down = metrics['h_negative'] < prev_year_metrics.get('h_negative', 0)
            if vre_up and h_neg_down:
                s3 += 3
            elif vre_up and metrics['h_negative'] <= prev_year_metrics.get('h_negative', 0):
                s3 += 1

        # VRE eleve mais h_neg modere -> signe d'absorption
        if metrics['vre_share'] > 0.40 and metrics['h_negative'] < 200:
            s3 += 2

    # ---- Scoring Stage 4 ----
    # REGLE : Stage 4 = "saturation" requiert VRE >= 35%
    s4 = 0
    vre_floor_s4 = t['stage_4'].get('vre_share_min', 0.35)
    if vre_share >= vre_floor_s4 and far is not None and not np.isnan(far):
        if far >= t['stage_4']['far_structural_min']:
            s4 += 3
    if vre_share >= vre_floor_s4 and metrics['h_regime_c'] < t['stage_4']['h_regime_c_max']:
        s4 += 3
    if vre_share >= vre_floor_s4 and metrics.get('h_regime_d_tail', 0) < 500:
        s4 += 2

    # ---- Determination de la phase ----
    scores = {'stage_1': s1, 'stage_2': s2, 'stage_3': s3, 'stage_4': s4}
    max_score = max(scores.values())
    total = sum(scores.values())

    # Score minimum de 2 pour etre candidat
    candidates = {k: v for k, v in scores.items() if v >= 2}
    if not candidates:
        phase = 'stage_1'
    else:
        phase = max(candidates, key=candidates.get)

    phase_number = int(phase.split('_')[1])
    confidence = max_score / max(total, 1)
    confidence = min(max(confidence, 0.0), 1.0)

    # ---- Alertes ----
    alerts = []
    alert_defs = t.get('alerts', {})

    for alert_key, alert_def in alert_defs.items():
        triggered = True

        if 'h_negative_range' in alert_def:
            lo, hi = alert_def['h_negative_range']
            if not (lo <= metrics['h_negative'] <= hi):
                triggered = False
        if 'h_negative_min' in alert_def:
            if metrics['h_negative'] < alert_def['h_negative_min']:
                triggered = False
        if 'capture_ratio_pv_range' in alert_def:
            cr = metrics.get('capture_ratio_pv')
            if cr is None or np.isnan(cr):
                triggered = False
            else:
                lo, hi = alert_def['capture_ratio_pv_range']
                if not (lo <= cr <= hi):
                    triggered = False
        if 'capture_ratio_pv_max' in alert_def:
            cr = metrics.get('capture_ratio_pv')
            if cr is None or np.isnan(cr) or cr > alert_def['capture_ratio_pv_max']:
                triggered = False
        if 'ir_min' in alert_def:
            ir_val = metrics.get('ir')
            if ir_val is None or np.isnan(ir_val) or ir_val < alert_def['ir_min']:
                triggered = False
        if 'far_structural_max' in alert_def:
            if far is None or np.isnan(far) or far > alert_def['far_structural_max']:
                triggered = False
        if 'sr_min' in alert_def:
            if metrics['sr'] < alert_def['sr_min']:
                triggered = False

        if triggered:
            alerts.append({
                'key': alert_key,
                'label': alert_def.get('label', alert_key),
                'color': alert_def.get('color', '#ff9800'),
            })

    return {
        'phase': phase,
        'phase_number': phase_number,
        'scores': scores,
        'confidence': round(confidence, 3),
        'alerts': alerts,
    }
