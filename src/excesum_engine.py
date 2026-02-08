"""ExceSum engine: baseline build, Q1..Q6 analytics, and verification matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.commentary_consistency import validate_claims_against_baseline
from src.config_loader import load_countries_config, load_scenarios, load_thresholds
from src.constants import OUTLIER_YEARS
from src.data_loader import load_commodity_prices, load_metrics, load_processed, load_raw
from src.metrics import compute_annual_metrics
from src.nrl_engine import compute_nrl
from src.objectives_loader import load_objectives_docx
from src.phase_diagnostics import diagnose_phase
from src.scenario_engine import apply_scenario
from src.slope_analysis import compute_slope
from src.ui_analysis import (
    compute_q4_bess_sweep,
    compute_q4_plateau_diagnostics,
    find_q4_stress_reference,
)


BASELINE_COUNTRIES = ("FR", "DE", "ES", "PL", "DK")
BASELINE_YEARS = tuple(range(2015, 2025))


@dataclass(frozen=True)
class BaselineRunConfig:
    countries: tuple[str, ...] = BASELINE_COUNTRIES
    years: tuple[int, ...] = BASELINE_YEARS
    must_run_mode: str = "observed"
    flex_model_mode: str = "observed"
    price_mode: str = "observed"
    exclude_outlier_years: tuple[int, ...] = (2022,)


def _safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _load_or_compute_processed(
    country: str,
    year: int,
    countries_cfg: dict,
    thresholds: dict,
    commodities: dict,
    must_run_mode: str,
    flex_model_mode: str,
    price_mode: str,
) -> pd.DataFrame:
    df = load_processed(country, year, must_run_mode, flex_model_mode, price_mode)
    if df is not None:
        return df

    raw = load_raw(country, year)
    return compute_nrl(
        df_raw=raw,
        country_key=country,
        year=year,
        country_cfg=countries_cfg[country],
        thresholds=thresholds,
        commodities=commodities,
        must_run_mode=must_run_mode,
        flex_model_mode=flex_model_mode,
        scenario_overrides=None,
        price_mode=price_mode,
    )


def _build_baseline_records(
    cfg: BaselineRunConfig,
    countries_cfg: dict,
    thresholds: dict,
    commodities: dict,
) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame], list[str]]:
    rows: list[dict] = []
    processed: dict[tuple[str, int], pd.DataFrame] = {}
    issues: list[str] = []

    for country in cfg.countries:
        if country not in countries_cfg:
            issues.append(f"Pays absent de countries.yaml: {country}")
            continue

        for year in cfg.years:
            try:
                df_proc = _load_or_compute_processed(
                    country=country,
                    year=year,
                    countries_cfg=countries_cfg,
                    thresholds=thresholds,
                    commodities=commodities,
                    must_run_mode=cfg.must_run_mode,
                    flex_model_mode=cfg.flex_model_mode,
                    price_mode=cfg.price_mode,
                )
                processed[(country, year)] = df_proc

                metrics = compute_annual_metrics(df_proc, country, year, countries_cfg[country])
                diag = diagnose_phase(metrics, thresholds)
                record = dict(metrics)
                record["phase"] = diag.get("phase", "unknown")
                record["phase_confidence"] = diag.get("confidence", np.nan)
                record["phase_score"] = diag.get("score", np.nan)
                rows.append(record)
            except FileNotFoundError:
                issues.append(f"Donnees absentes: {country} {year}")
            except Exception as exc:
                issues.append(f"Echec {country} {year}: {exc}")

    if not rows:
        return pd.DataFrame(), processed, issues

    df = pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)
    return df, processed, issues


def _q1_threshold_table(metrics_df: pd.DataFrame, thresholds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage2 = thresholds["phase_thresholds"]["stage_2"]
    h_negative_min = float(stage2["h_negative_min"])
    h_below_5_min = float(stage2["h_below_5_min"])
    capture_ratio_max = float(stage2["capture_ratio_pv_max"])

    q1 = metrics_df[["country", "year", "sr", "h_negative_obs", "h_below_5_obs", "capture_ratio_pv"]].copy()
    q1["dist_h_negative"] = q1["h_negative_obs"] - h_negative_min
    q1["dist_h_below_5"] = q1["h_below_5_obs"] - h_below_5_min
    q1["dist_capture_ratio"] = capture_ratio_max - q1["capture_ratio_pv"]
    q1["cross_h_negative"] = q1["h_negative_obs"] >= h_negative_min
    q1["cross_h_below_5"] = q1["h_below_5_obs"] >= h_below_5_min
    q1["cross_capture_ratio"] = q1["capture_ratio_pv"] <= capture_ratio_max
    q1["cross_all"] = q1["cross_h_negative"] & q1["cross_h_below_5"] & q1["cross_capture_ratio"]

    summary_rows: list[dict] = []
    for country, chunk in q1.groupby("country"):
        chunk_sorted = chunk.sort_values("year")
        crossed = chunk_sorted[chunk_sorted["cross_all"]]
        first_cross_year = int(crossed.iloc[0]["year"]) if not crossed.empty else np.nan
        latest = chunk_sorted.iloc[-1]
        summary_rows.append(
            {
                "country": country,
                "first_stage2_cross_year": first_cross_year,
                "latest_year": int(latest["year"]),
                "latest_sr": _safe_float(latest["sr"]),
                "latest_h_negative_obs": _safe_float(latest["h_negative_obs"]),
                "latest_h_below_5_obs": _safe_float(latest["h_below_5_obs"]),
                "latest_capture_ratio_pv": _safe_float(latest["capture_ratio_pv"]),
                "latest_cross_all": bool(latest["cross_all"]),
            }
        )

    return q1.sort_values(["country", "year"]).reset_index(drop=True), pd.DataFrame(summary_rows)


def _q2_slopes(metrics_df: pd.DataFrame, exclude_outlier_years: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict] = []
    outlier_set = set(int(y) for y in exclude_outlier_years)

    for country, chunk in metrics_df.groupby("country"):
        records: list[dict] = []
        for _, rec in chunk.iterrows():
            item = rec.to_dict()
            item["is_outlier"] = bool(int(item.get("year", -1)) in outlier_set)
            records.append(item)
        slope = compute_slope(records, "pv_penetration_pct_gen", "capture_ratio_pv", exclude_outliers=True)
        rows.append(
            {
                "country": country,
                "slope": _safe_float(slope.get("slope")),
                "intercept": _safe_float(slope.get("intercept")),
                "r_squared": _safe_float(slope.get("r_squared")),
                "p_value": _safe_float(slope.get("p_value")),
                "n_points": int(slope.get("n_points", 0) or 0),
                "robustesse": "forte"
                if _safe_float(slope.get("p_value")) <= 0.05 and _safe_float(slope.get("r_squared")) >= 0.4
                else "fragile",
            }
        )

    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def _q3_transition_status(metrics_df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    far_min = float(thresholds["phase_thresholds"]["stage_3"]["far_min"])
    h_negative_min = float(thresholds["phase_thresholds"]["stage_2"]["h_negative_min"])
    rows: list[dict] = []

    for country, chunk in metrics_df.groupby("country"):
        c = chunk.sort_values("year").copy()
        if c.empty:
            continue
        latest = c.iloc[-1]
        x = c["year"].to_numpy(dtype=float)
        y = c["h_negative_obs"].to_numpy(dtype=float)
        if len(x) >= 2 and np.nanstd(y) > 0:
            slope_hneg = float(np.polyfit(x, y, 1)[0])
        else:
            slope_hneg = float("nan")
        far_latest = _safe_float(latest["far"])
        hneg_latest = _safe_float(latest["h_negative_obs"])
        h_a_latest = _safe_float(latest["h_regime_a"])

        if np.isfinite(far_latest) and far_latest >= far_min and np.isfinite(slope_hneg) and slope_hneg < 0 and hneg_latest <= h_negative_min:
            status = "transition_validee"
        elif np.isfinite(far_latest) and far_latest >= far_min:
            status = "transition_partielle"
        else:
            status = "transition_non_validee"

        rows.append(
            {
                "country": country,
                "far_latest": far_latest,
                "h_negative_latest": hneg_latest,
                "h_negative_slope_per_year": slope_hneg,
                "h_regime_a_latest": h_a_latest,
                "sr_latest": _safe_float(latest["sr"]),
                "status_transition_2_to_3": status,
            }
        )

    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def _q4_battery_analysis(
    metrics_df: pd.DataFrame,
    processed: dict[tuple[str, int], pd.DataFrame],
    countries_cfg: dict,
    thresholds: dict,
    commodities: dict,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    summary_rows: list[dict] = []
    baseline_sweeps: dict[str, pd.DataFrame] = {}
    stress_sweeps: dict[str, pd.DataFrame] = {}

    for country in sorted(metrics_df["country"].dropna().unique()):
        subset = metrics_df[metrics_df["country"] == country]
        if subset.empty:
            continue
        year = int(subset["year"].max())
        df_base = processed.get((country, year))
        if df_base is None:
            continue

        baseline_diag = compute_q4_plateau_diagnostics(df_base)
        sweep_grid = [float(x) for x in np.arange(0, 21, 2)]
        sweep_base = compute_q4_bess_sweep(
            df_base_processed=df_base,
            country_key=country,
            year=year,
            country_cfg=countries_cfg[country],
            thresholds=thresholds,
            commodities=commodities,
            sweep_gw=sweep_grid,
            reference_overrides={},
        )
        baseline_sweeps[country] = sweep_base
        plateau = (not sweep_base.empty) and (
            sweep_base["far"].nunique(dropna=False) == 1 and sweep_base["h_regime_a"].nunique(dropna=False) == 1
        )

        stress = find_q4_stress_reference(
            df_base_processed=df_base,
            country_key=country,
            year=year,
            country_cfg=countries_cfg[country],
            thresholds=thresholds,
            commodities=commodities,
            max_delta_pv_gw=40,
            step_gw=2,
            base_overrides={},
        )

        stress_found = bool(stress.get("found", False))
        stress_delta = _safe_float(stress.get("delta_pv_gw"))
        stress_start_far = float("nan")
        stress_end_far = float("nan")
        stress_start_ha = float("nan")
        stress_end_ha = float("nan")

        if stress_found and isinstance(stress.get("df_reference"), pd.DataFrame):
            sweep_stress = compute_q4_bess_sweep(
                df_base_processed=stress["df_reference"],
                country_key=country,
                year=year,
                country_cfg=countries_cfg[country],
                thresholds=thresholds,
                commodities=commodities,
                sweep_gw=sweep_grid,
                reference_overrides={},
            )
            stress_sweeps[country] = sweep_stress
            if not sweep_stress.empty:
                stress_start_far = _safe_float(sweep_stress["far"].iloc[0])
                stress_end_far = _safe_float(sweep_stress["far"].iloc[-1])
                stress_start_ha = _safe_float(sweep_stress["h_regime_a"].iloc[0])
                stress_end_ha = _safe_float(sweep_stress["h_regime_a"].iloc[-1])

        summary_rows.append(
            {
                "country": country,
                "year": year,
                "plateau_baseline": bool(plateau),
                "surplus_unabs_twh_baseline": _safe_float(baseline_diag.get("total_surplus_unabs_twh")),
                "far_baseline": _safe_float(baseline_diag.get("far")),
                "h_regime_a_baseline": _safe_float(baseline_diag.get("h_regime_a")),
                "stress_found": stress_found,
                "stress_delta_pv_gw": stress_delta,
                "far_stress_start": stress_start_far,
                "far_stress_end": stress_end_far,
                "h_regime_a_stress_start": stress_start_ha,
                "h_regime_a_stress_end": stress_end_ha,
            }
        )

    return (
        pd.DataFrame(summary_rows).sort_values("country").reset_index(drop=True),
        baseline_sweeps,
        stress_sweeps,
    )


def _q5_commodity_stress(
    metrics_df: pd.DataFrame,
    processed: dict[tuple[str, int], pd.DataFrame],
    countries_cfg: dict,
    thresholds: dict,
    commodities: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    for country in sorted(metrics_df["country"].dropna().unique()):
        subset = metrics_df[metrics_df["country"] == country]
        if subset.empty:
            continue
        year = int(subset["year"].max())
        df_base = processed.get((country, year))
        if df_base is None:
            continue

        base_m = compute_annual_metrics(df_base, country, year, countries_cfg[country])
        df_co2 = apply_scenario(
            df_base_processed=df_base,
            country_key=country,
            year=year,
            country_cfg=countries_cfg[country],
            thresholds=thresholds,
            commodities=commodities,
            scenario_params={"co2_price_eur_t": 120.0},
            price_mode="synthetic",
        )
        df_gas = apply_scenario(
            df_base_processed=df_base,
            country_key=country,
            year=year,
            country_cfg=countries_cfg[country],
            thresholds=thresholds,
            commodities=commodities,
            scenario_params={"gas_price_eur_mwh": 50.0},
            price_mode="synthetic",
        )
        m_co2 = compute_annual_metrics(df_co2, country, year, countries_cfg[country])
        m_gas = compute_annual_metrics(df_gas, country, year, countries_cfg[country])
        rows.append(
            {
                "country": country,
                "year": year,
                "ttl_baseline": _safe_float(base_m.get("ttl")),
                "ttl_high_co2": _safe_float(m_co2.get("ttl")),
                "ttl_high_gas": _safe_float(m_gas.get("ttl")),
                "delta_ttl_high_co2": _safe_float(m_co2.get("ttl")) - _safe_float(base_m.get("ttl")),
                "delta_ttl_high_gas": _safe_float(m_gas.get("ttl")) - _safe_float(base_m.get("ttl")),
            }
        )
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def _q6_heat_cold_scope(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, chunk in metrics_df.groupby("country"):
        rows.append(
            {
                "country": country,
                "years_covered": int(chunk["year"].nunique()),
                "heat_cold_dataset_available": False,
                "conclusion_status": "non_identifiable_sans_donnees_dediees",
                "comment": (
                    "Le perimetre actuel ne contient pas de variable chaleur/froid dedicatee; "
                    "conclusion causale impossible."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def _verification_data(metrics_df: pd.DataFrame, cfg: BaselineRunConfig, issues: list[str]) -> dict:
    expected = len(cfg.countries) * len(cfg.years)
    loaded = int(len(metrics_df))
    status = "PASS" if loaded == expected and not issues else ("WARN" if loaded > 0 else "FAIL")
    completeness = _safe_float(metrics_df["data_completeness"].mean()) if not metrics_df.empty else float("nan")
    return {
        "check": "verification_data",
        "status": status,
        "detail": f"couples={loaded}/{expected}, issues={len(issues)}, completeness_mean={completeness:.3f}",
    }


def _verification_calc(metrics_df: pd.DataFrame, cfg: BaselineRunConfig) -> dict:
    if metrics_df.empty:
        return {"check": "verification_calc", "status": "FAIL", "detail": "aucune metrique calculee"}

    failures = 0
    cache_mismatch = 0
    for _, row in metrics_df.iterrows():
        if _safe_float(row.get("sr")) < -1e-12:
            failures += 1
        far = _safe_float(row.get("far"))
        if np.isfinite(far) and (far < -1e-12 or far > 1 + 1e-12):
            failures += 1
        if _safe_float(row.get("pv_penetration_pct_gen")) < -1e-12:
            failures += 1
        if _safe_float(row.get("wind_penetration_pct_gen")) < -1e-12:
            failures += 1
        if _safe_float(row.get("vre_penetration_pct_gen")) < -1e-12:
            failures += 1

        cached = load_metrics(str(row["country"]), int(row["year"]), cfg.price_mode)
        if cached is not None:
            for key in ("sr", "far", "ir", "ttl", "capture_ratio_pv"):
                a = _safe_float(cached.get(key))
                b = _safe_float(row.get(key))
                if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 1e-9:
                    cache_mismatch += 1
                    break

    status = "PASS" if failures == 0 and cache_mismatch == 0 else ("WARN" if failures == 0 else "FAIL")
    return {
        "check": "verification_calc",
        "status": status,
        "detail": f"invariants_failures={failures}, cache_mismatch={cache_mismatch}",
    }


def _verification_narrative() -> tuple[dict, pd.DataFrame]:
    report = pd.DataFrame(validate_claims_against_baseline())
    if report.empty:
        return {"check": "verification_narrative", "status": "WARN", "detail": "aucun signal de narration"}, report
    fail_count = int((report["status"] == "FAIL").sum())
    warn_count = int((report["status"] == "WARN").sum())
    status = "PASS" if fail_count == 0 and warn_count == 0 else ("WARN" if fail_count == 0 else "FAIL")
    return {
        "check": "verification_narrative",
        "status": status,
        "detail": f"rows={len(report)}, warn={warn_count}, fail={fail_count}",
    }, report


def build_country_conclusions(
    metrics_df: pd.DataFrame,
    q1_country: pd.DataFrame,
    q2_df: pd.DataFrame,
    q3_df: pd.DataFrame,
    q4_df: pd.DataFrame,
    q5_df: pd.DataFrame,
    q6_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for country in sorted(metrics_df["country"].dropna().unique()):
        latest = metrics_df[metrics_df["country"] == country].sort_values("year").iloc[-1]
        q1 = q1_country[q1_country["country"] == country]
        q2 = q2_df[q2_df["country"] == country]
        q3 = q3_df[q3_df["country"] == country]
        q4 = q4_df[q4_df["country"] == country]
        q5 = q5_df[q5_df["country"] == country]
        q6 = q6_df[q6_df["country"] == country]
        rows.append(
            {
                "country": country,
                "latest_year": int(latest["year"]),
                "phase_latest": latest.get("phase", "unknown"),
                "sr_latest": _safe_float(latest.get("sr")),
                "far_latest": _safe_float(latest.get("far")),
                "capture_ratio_pv_latest": _safe_float(latest.get("capture_ratio_pv")),
                "q1_first_stage2_year": _safe_float(q1["first_stage2_cross_year"].iloc[0]) if not q1.empty else np.nan,
                "q2_slope": _safe_float(q2["slope"].iloc[0]) if not q2.empty else np.nan,
                "q3_status": str(q3["status_transition_2_to_3"].iloc[0]) if not q3.empty else "n/a",
                "q4_plateau_baseline": bool(q4["plateau_baseline"].iloc[0]) if not q4.empty else False,
                "q4_stress_found": bool(q4["stress_found"].iloc[0]) if not q4.empty else False,
                "q5_delta_ttl_co2": _safe_float(q5["delta_ttl_high_co2"].iloc[0]) if not q5.empty else np.nan,
                "q5_delta_ttl_gas": _safe_float(q5["delta_ttl_high_gas"].iloc[0]) if not q5.empty else np.nan,
                "q6_status": str(q6["conclusion_status"].iloc[0]) if not q6.empty else "n/a",
            }
        )
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def write_excesum_docs(results: dict, docs_dir: str = "docs") -> tuple[str, str]:
    root = Path(docs_dir)
    root.mkdir(parents=True, exist_ok=True)
    conclusions_path = root / "EXCESUM_CONCLUSIONS_BASELINE.md"
    verification_path = root / "EXCESUM_VERIFICATION_MATRIX.md"

    metrics = results.get("metrics_df", pd.DataFrame())
    countries = ", ".join(sorted(metrics["country"].dropna().unique())) if not metrics.empty else "n/a"
    years = f"{int(metrics['year'].min())}-{int(metrics['year'].max())}" if not metrics.empty else "n/a"

    lines = [
        "# ExceSum Conclusions Baseline",
        "",
        f"- Pays: {countries}",
        f"- Periode: {years}",
        "- Modes: observed / observed / observed",
        "",
        "## Q1..Q6",
        "",
        "Les tableaux de preuve sont produits dans la page ExceSum et dans les objets de sortie du moteur.",
    ]
    conclusions_path.write_text("\n".join(lines), encoding="utf-8")

    verif_rows = results.get("verification_rows", [])
    v_lines = ["# ExceSum Verification Matrix", "", "| Check | Status | Detail |", "|---|---|---|"]
    for row in verif_rows:
        v_lines.append(f"| {row['check']} | {row['status']} | {row['detail']} |")
    verification_path.write_text("\n".join(v_lines), encoding="utf-8")

    return str(conclusions_path), str(verification_path)


def run_excesum_baseline(
    objectives_docx_path: str | None = None,
    run_cfg: BaselineRunConfig | None = None,
) -> dict:
    cfg = run_cfg or BaselineRunConfig()
    countries_cfg = load_countries_config()
    thresholds = load_thresholds()
    scenarios = load_scenarios()
    commodities = load_commodity_prices()

    metrics_df, processed, issues = _build_baseline_records(cfg, countries_cfg, thresholds, commodities)

    if metrics_df.empty:
        return {
            "config": cfg,
            "metrics_df": pd.DataFrame(),
            "issues": issues or ["Aucune donnee baseline disponible."],
        }

    q1_detail, q1_country = _q1_threshold_table(metrics_df, thresholds)
    q2_df = _q2_slopes(metrics_df, cfg.exclude_outlier_years)
    q3_df = _q3_transition_status(metrics_df, thresholds)
    q4_df, q4_baseline_sweeps, q4_stress_sweeps = _q4_battery_analysis(
        metrics_df=metrics_df,
        processed=processed,
        countries_cfg=countries_cfg,
        thresholds=thresholds,
        commodities=commodities,
    )
    q5_df = _q5_commodity_stress(
        metrics_df=metrics_df,
        processed=processed,
        countries_cfg=countries_cfg,
        thresholds=thresholds,
        commodities=commodities,
    )
    q6_df = _q6_heat_cold_scope(metrics_df)

    country_conclusions = build_country_conclusions(
        metrics_df=metrics_df,
        q1_country=q1_country,
        q2_df=q2_df,
        q3_df=q3_df,
        q4_df=q4_df,
        q5_df=q5_df,
        q6_df=q6_df,
    )

    ver_data = _verification_data(metrics_df, cfg, issues)
    ver_calc = _verification_calc(metrics_df, cfg)
    ver_narrative, consistency_report = _verification_narrative()
    verification_rows = [ver_data, ver_calc, ver_narrative]

    objectives = None
    if objectives_docx_path:
        try:
            objectives = load_objectives_docx(objectives_docx_path)
        except Exception as exc:
            issues.append(f"Objectifs DOCX non charges: {exc}")
            objectives = {"path": objectives_docx_path, "paragraphs": [], "questions": {}, "objective_lines": []}

    return {
        "config": cfg,
        "thresholds": thresholds,
        "scenarios": scenarios,
        "countries_cfg": countries_cfg,
        "objectives": objectives,
        "issues": issues,
        "metrics_df": metrics_df,
        "processed_map": processed,
        "q1_detail": q1_detail,
        "q1_country": q1_country,
        "q2_slopes": q2_df,
        "q3_transition": q3_df,
        "q4_summary": q4_df,
        "q4_baseline_sweeps": q4_baseline_sweeps,
        "q4_stress_sweeps": q4_stress_sweeps,
        "q5_commodity": q5_df,
        "q6_scope": q6_df,
        "country_conclusions": country_conclusions,
        "consistency_report": consistency_report,
        "verification_rows": verification_rows,
    }


__all__ = [
    "BASELINE_COUNTRIES",
    "BASELINE_YEARS",
    "BaselineRunConfig",
    "run_excesum_baseline",
    "write_excesum_docs",
    "build_country_conclusions",
]
