"""Slope analysis utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress


def compute_slope(
    metrics_list: list[dict],
    x_key: str,
    y_key: str,
    exclude_outliers: bool = True,
) -> dict:
    """Linear slope between annual metrics (scipy linregress)."""

    x_vals = []
    y_vals = []

    for m in metrics_list:
        if exclude_outliers and bool(m.get("is_outlier", False)):
            continue
        x = m.get(x_key)
        y = m.get(y_key)
        if x is None or y is None:
            continue
        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x_vals.append(x)
        y_vals.append(y)

    n = len(x_vals)
    if n < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "p_value": float("nan"),
            "n_points": n,
            "x_values": x_vals,
            "y_values": y_vals,
        }

    res = linregress(x_vals, y_vals)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue**2),
        "p_value": float(res.pvalue),
        "n_points": n,
        "x_values": list(x_vals),
        "y_values": list(y_vals),
    }
