"""Context features for annual phase diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _declining_flag_for_country(df_country: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    ordered = df_country.sort_values("year").reset_index(drop=True)

    for i in range(len(ordered)):
        slice_i = ordered.iloc[: i + 1]
        years = _safe_series(slice_i["year"])
        h_neg = _safe_series(slice_i["h_negative_obs"])

        valid = pd.DataFrame({"year": years, "h_negative_obs": h_neg}).dropna()
        recent_peak_3y = float("nan")
        if len(valid) >= 1:
            recent_peak_3y = float(valid.tail(3)["h_negative_obs"].max())

        if len(valid) >= 3:
            tail = valid.tail(3)
            x = tail["year"].to_numpy(dtype=float)
            y = tail["h_negative_obs"].to_numpy(dtype=float)
            if np.nanstd(y) <= 0:
                declining = False
            else:
                slope = float(np.polyfit(x, y, 1)[0])
                declining = bool(slope < 0.0)
        elif len(valid) == 2:
            y_prev = float(valid["h_negative_obs"].iloc[-2])
            y_curr = float(valid["h_negative_obs"].iloc[-1])
            declining = bool(y_curr < y_prev)
        else:
            declining = False

        rows.append(
            {
                "country": ordered.iloc[i]["country"],
                "year": int(ordered.iloc[i]["year"]),
                "h_negative_declining": bool(declining),
                "h_negative_recent_peak_3y": recent_peak_3y,
            }
        )

    return pd.DataFrame(rows)


def compute_h_negative_declining_flags(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic h_negative declining flag by country/year.

    Rules:
    - >=3 available years: slope on last 3 years, declining iff slope < 0
    - 2 years: current < previous
    - 1 year: False
    """

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=["country", "year", "h_negative_declining", "h_negative_recent_peak_3y"])

    required = {"country", "year", "h_negative_obs"}
    missing = sorted(required - set(metrics_df.columns))
    if missing:
        raise ValueError(f"compute_h_negative_declining_flags: colonnes manquantes {missing}")

    rows: list[pd.DataFrame] = []
    for country, chunk in metrics_df.groupby("country"):
        _ = country  # explicit for readability in group loop
        rows.append(_declining_flag_for_country(chunk))

    if not rows:
        return pd.DataFrame(columns=["country", "year", "h_negative_declining", "h_negative_recent_peak_3y"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["country", "year"]).reset_index(drop=True)
    return out


__all__ = ["compute_h_negative_declining_flags"]
