"""Outside-in vs client reconciliation utilities."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_client_curves(path: str = "data/external/client_curves.csv") -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    required = {"country", "year"}
    if not required.issubset(df.columns):
        raise ValueError("client_curves.csv invalide: colonnes country/year requises")
    return df


def reconcile_against_client(
    model_metrics: list[dict],
    client_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-country/year deltas with heuristic explanatory tags."""

    mm = pd.DataFrame(model_metrics)
    if mm.empty:
        return pd.DataFrame()

    merged = mm.merge(client_df, on=["country", "year"], suffixes=("_model", "_client"), how="inner")
    if merged.empty:
        return merged

    if "capture_ratio_pv_client" in merged.columns and "capture_ratio_pv" in merged.columns:
        merged["delta_capture_ratio_pv"] = (
            merged["capture_ratio_pv_model"] if "capture_ratio_pv_model" in merged.columns else merged["capture_ratio_pv"]
        ) - merged["capture_ratio_pv_client"]

    def _diagnostic_row(row):
        # Simple transparent attribution rules
        sr = row.get("sr", row.get("sr_model", float("nan")))
        ir = row.get("ir", row.get("ir_model", float("nan")))
        ttl = row.get("ttl", row.get("ttl_model", float("nan")))

        if pd.notna(sr) and sr > 0.03:
            return "Ecart explique par SR"
        if pd.notna(ir) and ir > 0.55:
            return "Ecart explique par IR"
        if pd.notna(ttl) and ttl > 120:
            return "Ecart explique par TTL (TCA)"
        return "Ecart inexplique -> a investiguer"

    merged["diagnostic"] = merged.apply(_diagnostic_row, axis=1)
    return merged
