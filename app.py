"""Capture Prices Analyzer v3.0 - Streamlit entrypoint."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.commentary_bridge import comment_kpi, so_what_block
from src.config_loader import load_countries_config, load_scenarios, load_thresholds
from src.data_fetcher import fetch_country_year
from src.data_loader import (
    list_processed_keys,
    load_commodity_prices,
    load_metrics,
    load_processed,
    load_raw,
    save_diagnostics,
    save_metrics,
    save_processed,
)
from src.metrics import compute_annual_metrics
from src.nrl_engine import compute_nrl
from src.phase_diagnostics import diagnose_phase
from src.state_adapter import metrics_to_dataframe, normalize_metrics_record, normalize_state_metrics
from src.ui_helpers import dynamic_narrative, info_card, inject_global_css, narrative, render_commentary, section_header

load_dotenv()

st.set_page_config(page_title="Capture Prices Analyzer", page_icon="⚡", layout="wide")
inject_global_css()


_REQUIRED_METRICS_KEYS = {
    "sr",
    "far",
    "ir",
    "ttl",
    "capture_ratio_pv",
    "capture_ratio_wind",
    "h_regime_a",
    "h_regime_b",
    "h_regime_c",
    "h_regime_d",
    "h_negative_obs",
    "h_below_5_obs",
    "pv_penetration_pct_gen",
    "wind_penetration_pct_gen",
    "vre_penetration_pct_gen",
    "price_used_p95",
}

_DEFAULT_COUNTRY_SELECTION = ["FR", "DE", "ES", "PL", "DK"]


def _init_state() -> None:
    if "state" not in st.session_state:
        st.session_state.state = {
            "data_loaded": False,
            "raw": {},
            "processed": {},
            "metrics": {},
            "diagnostics": {},
            "countries_selected": _DEFAULT_COUNTRY_SELECTION.copy(),
            "year_range": (2015, 2024),
            "exclude_2022": True,
            "must_run_mode": "observed",
            "flex_model_mode": "observed",
            "price_mode": "observed",
            "scenario_price_mode": "synthetic",
            "commodities": None,
            "countries_cfg": None,
            "thresholds": None,
            "scenarios": None,
            "ui_overrides": {},
        }


@st.cache_data(show_spinner=False)
def _cached_configs():
    return load_countries_config(), load_thresholds(), load_scenarios()


@st.cache_data(show_spinner=False)
def _cached_commodities():
    return load_commodity_prices()


def _metrics_schema_ok(metrics: dict | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    normalized = normalize_metrics_record(metrics)
    return _REQUIRED_METRICS_KEYS.issubset(set(normalized.keys()))


def _metrics_df_schema_ok(df: pd.DataFrame) -> bool:
    required = {"country", "year"}
    return required.issubset(set(df.columns))


def _runtime_overrides(s: dict) -> dict:
    raw = s.get("ui_overrides", {})
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        if v is None:
            continue
        out[k] = v
    return out


def _load_one(country: str, year: int, s: dict):
    countries_cfg = s["countries_cfg"]
    thresholds = s["thresholds"]
    commodities = s["commodities"]

    mr = s["must_run_mode"]
    flex = s["flex_model_mode"]
    price_mode = s["price_mode"]

    runtime_overrides = _runtime_overrides(s)
    use_runtime_overrides = bool(runtime_overrides)

    df_proc = None if use_runtime_overrides else load_processed(country, year, mr, flex, price_mode)
    df_raw = None

    if df_proc is None:
        try:
            df_raw = load_raw(country, year)
        except FileNotFoundError:
            df_raw = fetch_country_year(country, year, countries_cfg=countries_cfg, force=False)

        df_proc = compute_nrl(
            df_raw=df_raw,
            country_key=country,
            year=year,
            country_cfg=countries_cfg[country],
            thresholds=thresholds,
            commodities=commodities,
            must_run_mode=mr,
            flex_model_mode=flex,
            scenario_overrides=runtime_overrides or None,
            price_mode=price_mode,
        )

        if not use_runtime_overrides:
            save_processed(df_proc, country, year, mr, flex, price_mode)
    else:
        try:
            df_raw = load_raw(country, year)
        except FileNotFoundError:
            df_raw = None

    metrics = None if use_runtime_overrides else load_metrics(country, year, price_mode)
    if (not _metrics_schema_ok(metrics)) or use_runtime_overrides:
        metrics = compute_annual_metrics(df_proc, country, year, countries_cfg[country])
        if not use_runtime_overrides:
            save_metrics(metrics, country, year, price_mode)
    else:
        metrics = normalize_metrics_record(metrics)

    diag = diagnose_phase(metrics, thresholds)
    if not use_runtime_overrides:
        save_diagnostics(diag, country, year, price_mode)

    return country, year, df_raw, df_proc, metrics, diag


def _load_selected_data(s: dict) -> None:
    countries = s["countries_selected"]
    y0, y1 = s["year_range"]
    tasks = [(c, y) for c in countries for y in range(y0, y1 + 1)]

    progress = st.progress(0.0, text="Chargement des donnees...")
    done = 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_load_one, c, y, s) for (c, y) in tasks]
        for fut in as_completed(futs):
            country, year, df_raw, df_proc, metrics, diag = fut.result()

            if df_raw is not None:
                s["raw"][(country, year)] = df_raw

            proc_key = (country, year, s["must_run_mode"], s["flex_model_mode"], s["price_mode"])
            s["processed"][proc_key] = df_proc
            s["metrics"][(country, year, s["price_mode"])] = normalize_metrics_record(metrics)
            s["diagnostics"][(country, year)] = diag

            done += 1
            progress.progress(done / max(1, len(tasks)), text=f"{country} {year} ({done}/{len(tasks)})")

    s["data_loaded"] = True
    progress.progress(1.0, text="Termine")


_init_state()
state = st.session_state.state

countries_cfg, thresholds, scenarios = _cached_configs()
commodities = _cached_commodities()

state["countries_cfg"] = countries_cfg
state["thresholds"] = thresholds
state["scenarios"] = scenarios
state["commodities"] = commodities

normalize_state_metrics(state)
all_countries = sorted([k for k in countries_cfg.keys() if not k.startswith("__")])
country_labels = {c: countries_cfg[c].get("name", c) for c in all_countries}

# Migration douce: si l'utilisateur est encore sur l'ancien défaut FR/DE/ES,
# étendre automatiquement à FR/DE/ES/PL/DK tant que les données ne sont pas chargées.
if (
    not state.get("data_loaded")
    and set(state.get("countries_selected", [])) == {"FR", "DE", "ES"}
):
    state["countries_selected"] = [c for c in _DEFAULT_COUNTRY_SELECTION if c in all_countries]

with st.sidebar:
    st.markdown("#### Capture Prices Analyzer")

    st.markdown("**Selection**")
    state["countries_selected"] = st.multiselect(
        "Pays",
        options=all_countries,
        default=(
            [c for c in state["countries_selected"] if c in all_countries]
            or [c for c in _DEFAULT_COUNTRY_SELECTION if c in all_countries]
            or all_countries[: min(5, len(all_countries))]
        ),
        format_func=lambda c: f"{c} - {country_labels.get(c, c)}",
    )

    state["year_range"] = st.slider("Periode", 2015, 2024, state["year_range"])

    st.divider()

    with st.expander("Modes de calcul", expanded=True):
        state["must_run_mode"] = st.radio(
            "Must-run",
            ["observed", "floor"],
            index=0 if state["must_run_mode"] == "observed" else 1,
        )
        state["flex_model_mode"] = st.radio(
            "Flex",
            ["observed", "capacity"],
            index=0 if state["flex_model_mode"] == "observed" else 1,
        )
        state["price_mode"] = st.radio(
            "Price mode historique",
            ["observed", "synthetic"],
            index=0 if state["price_mode"] == "observed" else 1,
        )
        state["scenario_price_mode"] = st.radio(
            "Price mode scenario",
            ["synthetic", "observed"],
            index=0 if state["scenario_price_mode"] == "synthetic" else 1,
        )

    overrides_count = len(_runtime_overrides(state))
    if overrides_count > 0:
        st.warning(f"Hypotheses personnalisees actives: {overrides_count}")

    if st.button("Charger donnees", type="primary", use_container_width=True):
        if not state["countries_selected"]:
            st.error("Selectionnez au moins un pays.")
        else:
            _load_selected_data(state)
            st.rerun()

    st.divider()
    st.markdown("**Statut**")
    if state["data_loaded"]:
        st.success(f"{len(state['metrics'])} couples pays/annee charges")
        available = list_processed_keys(
            must_run_mode=state["must_run_mode"],
            flex_model_mode=state["flex_model_mode"],
            price_mode=state["price_mode"],
        )
        st.caption(f"Caches process disponibles: {len(available)}")
    else:
        st.info("Aucune donnee chargee")

st.title("Capture Prices Analyzer")
st.caption("Framework v3.0 - NRL, regimes A/B/C/D, SR/FAR/IR/TTL, scenarios deterministes")

if not state["data_loaded"]:
    section_header("Bienvenue", "Analyse rigoureuse des capture prices renouvelables")
    narrative(
        "L'outil relie les prix de marche aux mecanismes physiques du systeme: "
        "NRL, surplus, flexibilite, ancre thermique."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("1. Selectionnez", "Choisissez pays, periode et modes dans la sidebar.")
    with c2:
        info_card("2. Chargez", "Cliquez sur Charger donnees pour lancer le pipeline complet.")
    with c3:
        info_card("3. Interpretez", "Lisez les pages avec commentaires So what traces sur les chiffres.")

    st.markdown("### Architecture du modele")
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center; gap:6px; margin:0.8rem 0 1.6rem 0;">
            <div style="background:#ebf5fb; border:2px solid #3498db; border-radius:10px; padding:10px 24px; width:78%; max-width:560px; text-align:center;"><strong style="color:#2980b9;">Niveau 1</strong><br><span style="font-size:0.88rem;">Donnees ENTSO-E horaires (load, generation, prix, net position)</span></div>
            <div style="color:#95a5a6;">&#x25BC;</div>
            <div style="background:#e8f8f5; border:2px solid #27ae60; border-radius:10px; padding:10px 24px; width:78%; max-width:560px; text-align:center;"><strong style="color:#27ae60;">Niveau 2</strong><br><span style="font-size:0.88rem;">NRL = load - VRE - must-run</span></div>
            <div style="color:#95a5a6;">&#x25BC;</div>
            <div style="background:#fef9e7; border:2px solid #f39c12; border-radius:10px; padding:10px 24px; width:78%; max-width:560px; text-align:center;"><strong style="color:#e67e22;">Niveau 3</strong><br><span style="font-size:0.88rem;">Regimes physiques A / B / C / D (anti-circularite)</span></div>
            <div style="color:#95a5a6;">&#x25BC;</div>
            <div style="background:#f4ecf7; border:2px solid #8e44ad; border-radius:10px; padding:10px 24px; width:78%; max-width:560px; text-align:center;"><strong style="color:#8e44ad;">Niveau 4</strong><br><span style="font-size:0.88rem;">Metriques annuelles et diagnostic de phase</span></div>
            <div style="color:#95a5a6;">&#x25BC;</div>
            <div style="background:#fdecec; border:2px solid #e74c3c; border-radius:10px; padding:10px 24px; width:78%; max-width:560px; text-align:center;"><strong style="color:#c0392b;">Niveau 5</strong><br><span style="font-size:0.88rem;">Scenarios deterministes et interpretation business</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

metrics_df = metrics_to_dataframe(state, state["price_mode"])
if metrics_df.empty or not _metrics_df_schema_ok(metrics_df):
    st.info("Aucune metrique disponible pour les filtres actifs.")
    st.stop()
    # Bare-mode fallback: `st.stop()` may not interrupt plain Python imports.
    metrics_df = pd.DataFrame(
        [
            {
                "country": "N/A",
                "year": 0,
                "sr": float("nan"),
                "far": float("nan"),
                "ir": float("nan"),
                "ttl": float("nan"),
                "capture_ratio_pv": float("nan"),
                "h_negative_obs": float("nan"),
                "phase": "unknown",
                "regime_coherence": float("nan"),
            }
        ]
    )

active_overrides = _runtime_overrides(state)
if active_overrides:
    dynamic_narrative(
        f"Hypotheses personnalisees actives ({len(active_overrides)}). "
        "Les resultats affiches incluent ces overrides et restent comparables uniquement dans ce meme cadre.",
        severity="warning",
    )

selected_country = st.selectbox(
    "Pays dashboard",
    options=sorted(metrics_df["country"].dropna().unique()),
    format_func=lambda c: f"{c} - {country_labels.get(c, c)}",
)

df_country = metrics_df[metrics_df["country"] == selected_country].copy()
latest_year = int(df_country["year"].max())
latest = df_country[df_country["year"] == latest_year].iloc[0].to_dict()

section_header("Tableau de bord", f"{selected_country} {latest_year}")
cols = st.columns(5)
cols[0].metric("SR", f"{float(latest.get('sr', float('nan'))):.3f}", help="Part du surplus brut sur la generation annuelle.")
cols[1].metric("FAR", f"{float(latest.get('far', float('nan'))):.3f}", help="Part du surplus absorbee par la flexibilite.")
cols[2].metric("IR", f"{float(latest.get('ir', float('nan'))):.3f}", help="Rigidite systeme: P10(must-run)/P10(load).")
cols[3].metric("TTL", f"{float(latest.get('ttl', float('nan'))):.1f} EUR/MWh", help="Queue haute price_used sur regimes C+D.")
cols[4].metric("Phase", str(latest.get("phase", "unknown")), help="Diagnostic de phase issu de thresholds.yaml.")

render_commentary(comment_kpi(latest, label="Lecture KPI"))

if float(latest.get("regime_coherence", float("nan"))) == float(latest.get("regime_coherence", float("nan"))):
    coh = float(latest.get("regime_coherence", float("nan")))
    if coh < 0.55:
        dynamic_narrative(
            f"Coherence regime/prix observee faible ({coh:.1%} < 55%). "
            "Avant interpretation forte, verifier hypotheses must-run/flex et completude donnees.",
            severity="warning",
        )

section_header("Comparaison multi-pays", f"Annee {latest_year}")
rows = []
for c in state["countries_selected"]:
    d = metrics_df[(metrics_df["country"] == c) & (metrics_df["year"] == latest_year)]
    if d.empty:
        continue
    r = d.iloc[0]
    rows.append(
        {
            "country": c,
            "sr": r.get("sr"),
            "far": r.get("far"),
            "ir": r.get("ir"),
            "ttl": r.get("ttl"),
            "capture_ratio_pv": r.get("capture_ratio_pv"),
            "h_negative_obs": r.get("h_negative_obs"),
            "phase": r.get("phase", "unknown"),
        }
    )

if rows:
    df_cmp = pd.DataFrame(rows)
    styled = df_cmp.style.format(
        {
            "sr": "{:.3f}",
            "far": "{:.3f}",
            "ir": "{:.3f}",
            "ttl": "{:.1f}",
            "capture_ratio_pv": "{:.3f}",
            "h_negative_obs": "{:.0f}",
        },
        na_rep="--",
    )
    if df_cmp["far"].notna().any():
        styled = styled.background_gradient(subset=["far"], cmap="RdYlGn", vmin=0.0, vmax=1.0)
    if df_cmp["capture_ratio_pv"].notna().any():
        styled = styled.background_gradient(subset=["capture_ratio_pv"], cmap="RdYlGn", vmin=0.5, vmax=1.0)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    sr_median = float(df_cmp["sr"].median()) if df_cmp["sr"].notna().any() else float("nan")
    far_median = float(df_cmp["far"].median()) if df_cmp["far"].notna().any() else float("nan")
    h_negative_total = (
        float(df_cmp["h_negative_obs"].sum()) if df_cmp["h_negative_obs"].notna().any() else float("nan")
    )

    render_commentary(
        so_what_block(
            title="Lecture comparative",
            purpose="Identifier rapidement les pays les plus exposes a la cannibalisation et au surplus",
            observed={
                "n_pays": len(df_cmp),
                "sr_median": sr_median,
                "far_median": far_median,
                "h_negative_total": h_negative_total,
            },
            method_link="Metriques harmonisees v3, price_mode unique, mapping legacy normalise.",
            limits="Comparaison sensible a la completude des donnees et au contexte systeme propre a chaque pays.",
            n=len(df_cmp),
        )
    )
