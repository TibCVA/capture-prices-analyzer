import numpy as np
from scipy import stats
import logging

logger = logging.getLogger("capture_prices.slope_analysis")


def compute_slope(metrics_list: list[dict], x_key: str, y_key: str,
                  exclude_outliers: bool = True) -> dict:
    """
    Regression lineaire entre deux metriques sur la serie temporelle.

    Args:
        metrics_list: liste de dicts (un par annee)
        x_key: cle de la metrique X (ex: 'pv_share')
        y_key: cle de la metrique Y (ex: 'capture_ratio_pv')
        exclude_outliers: exclure les annees dans OUTLIER_YEARS

    Returns:
        dict avec slope, intercept, r_squared, p_value, n_points, x_values, y_values
    """
    # Filtrer les points valides
    points = []
    for m in metrics_list:
        x_val = m.get(x_key)
        y_val = m.get(y_key)
        if x_val is None or y_val is None:
            continue
        if np.isnan(x_val) or np.isnan(y_val):
            continue
        if exclude_outliers and m.get('is_outlier', False):
            continue
        points.append((float(x_val), float(y_val)))

    if len(points) < 3:
        return {
            'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan,
            'p_value': np.nan, 'n_points': len(points),
            'x_values': [], 'y_values': [],
        }

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    result = stats.linregress(x, y)

    return {
        'slope': round(float(result.slope), 6),
        'intercept': round(float(result.intercept), 4),
        'r_squared': round(float(result.rvalue ** 2), 4),
        'p_value': round(float(result.pvalue), 6),
        'n_points': len(points),
        'x_values': x.tolist(),
        'y_values': y.tolist(),
    }
