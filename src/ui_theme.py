"""UI theme compatibility layer.

This module prevents hard failures when deploying across mixed code versions
where some visual constants may be missing in ``src.constants``.
"""

from __future__ import annotations

from src import constants as C


REGIME_COLORS = getattr(
    C,
    "REGIME_COLORS",
    {
        "A": "#d73027",
        "B": "#fc8d59",
        "C": "#4575b4",
        "D": "#313695",
    },
)

PHASE_COLORS = getattr(
    C,
    "PHASE_COLORS",
    {
        "stage_1": "#1b9e77",
        "stage_2": "#d95f02",
        "stage_3": "#7570b3",
        "stage_4": "#e7298a",
        "unknown": "#666666",
    },
)

COUNTRY_PALETTE = getattr(
    C,
    "COUNTRY_PALETTE",
    {
        "FR": "#003399",
        "DE": "#FFCC00",
        "ES": "#CC0000",
        "PL": "#DC143C",
        "DK": "#C8102E",
    },
)

PLOTLY_LAYOUT_DEFAULTS = getattr(
    C,
    "PLOTLY_LAYOUT_DEFAULTS",
    {
        "template": "plotly_white",
        "font": {"family": "Segoe UI, Arial, sans-serif", "size": 13, "color": "#1f2937"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        "margin": {"l": 52, "r": 28, "t": 48, "b": 46},
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
        "hovermode": "closest",
    },
)

PLOTLY_AXIS_DEFAULTS = getattr(
    C,
    "PLOTLY_AXIS_DEFAULTS",
    {
        "showgrid": True,
        "gridcolor": "#e5e7eb",
        "zeroline": False,
        "linecolor": "#cbd5e1",
        "mirror": False,
    },
)

