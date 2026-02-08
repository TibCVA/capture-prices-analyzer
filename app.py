"""Capture Prices Analyzer v3.0 - Streamlit entrypoint."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.commentary_engine import comment_kpi
from src.config_loader import load_countries_config, load_scenarios, load_thresholds
from src.data_fetcher import fetch_country_year
from src.data_loader import (
    list_processed_keys,
    load_commodity_prices,
    load_diagnostics,
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
from src.ui_helpers import guard_no_data, inject_global_css, render_commentary, section

load_dotenv()

st.set_page_config(page_title="Capture Prices Analyzer", page_icon="⚡", layout="wide")
inject_global_css()


def _init_state() -> None:
    if "state" not in st.session_state:
        st.session_state.state = {
            "data_loaded": False,
            "raw": {},
            "processed": {},
            "metrics": {},
            "diagnostics": {},
            "countries_selected": ["FR", "DE", "ES"],
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
        }


@st.cache_data(show_spinner=False)
def _cached_configs():
    countries_cfg = load_countries_config()
    thresholds = load_thresholds()
    scenarios = load_scenarios()
    return countries_cfg, thresholds, scenarios


@st.cache_data(show_spinner=False)
def _cached_commodities():
    return load_commodity_prices()


def _load_one(country: str, year: int, s: dict):
    countries_cfg = s["countries_cfg"]
    thresholds = s["thresholds"]
    commodities = s["commodities"]

    mr = s["must_run_mode"]
    flex = s["flex_model_mode"]
    price_mode = s["price_mode"]

    df_proc = load_processed(country, year, mr, flex, price_mode)
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
            scenario_overrides=None,
            price_mode=price_mode,
        )
        save_processed(df_proc, country, year, mr, flex, price_mode)
    else:
        try:
            df_raw = load_raw(country, year)
        except FileNotFoundError:
            df_raw = None

    metrics = load_metrics(country, year, price_mode)
    if metrics is None:
        metrics = compute_annual_metrics(df_proc, country, year, countries_cfg[country])
        save_metrics(metrics, country, year, price_mode)

    diag = load_diagnostics(country, year, price_mode)
    if diag is None:
        diag = diagnose_phase(metrics, thresholds)
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
            s["metrics"][(country, year, s["price_mode"])] = metrics
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

all_countries = sorted([k for k in countries_cfg.keys() if not k.startswith("__")])

with st.sidebar:
    st.markdown("### Parametres")
    _ = st.text_input(
        "API key ENTSO-E",
        value=os.getenv("ENTSOE_API_KEY", ""),
        type="password",
        help="Optionnel si le cache raw est deja present.",
    )

    state["countries_selected"] = st.multiselect(
        "Pays",
        options=all_countries,
        default=[c for c in state["countries_selected"] if c in all_countries] or all_countries[:3],
    )

    state["year_range"] = st.slider("Periode", 2015, 2024, state["year_range"])
    state["exclude_2022"] = st.checkbox("Exclure 2022 (regressions)", value=state["exclude_2022"])

    state["must_run_mode"] = st.radio(
        "Must-run mode",
        ["observed", "floor"],
        index=0 if state["must_run_mode"] == "observed" else 1,
    )
    state["flex_model_mode"] = st.radio(
        "Flex mode",
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

    if st.button("Charger donnees", type="primary", use_container_width=True):
        if not state["countries_selected"]:
            st.error("Selectionnez au moins un pays.")
        else:
            _load_selected_data(state)
            st.rerun()

    st.divider()
    st.markdown("### Statut")
    if state["data_loaded"]:
        st.success(f"{len(state['metrics'])} couples pays/annee charges")
    else:
        st.info("Aucune donnee chargee")

st.title("⚡ Capture Prices Analyzer")
st.caption("Modele v3.0 — SR/FAR/IR/TTL, regimes physiques A/B/C/D, scenarios deterministes")

if not state["data_loaded"]:
    section("Demarrage", "Chargez les donnees depuis la barre laterale pour activer toutes les pages.")
    st.markdown(
        "- Le pipeline est: donnees ENTSO-E -> NRL -> regimes A/B/C/D -> TCA/prix synth -> metriques annuelles.\n"
        "- Les observables marche (prix negatifs, spreads) restent calcules sur prix observes.\n"
        "- Les scenarios utilisent par defaut le prix synthetique ancre TCA."
    )
    st.stop()

# Dashboard selection
selected_country = state["countries_selected"][0]
min_y, max_y = state["year_range"]
latest_year = max_y
metric_key = (selected_country, latest_year, state["price_mode"])

if metric_key not in state["metrics"]:
    guard_no_data("le tableau de bord")

m = state["metrics"][metric_key]
diag = state["diagnostics"].get((selected_country, latest_year), {})

section("KPI principaux", f"{selected_country} {latest_year}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("SR", f"{m.get('sr', float('nan')):.3f}")
col2.metric("FAR", f"{m.get('far', float('nan')):.3f}")
col3.metric("IR", f"{m.get('ir', float('nan')):.3f}")
col4.metric("TTL", f"{m.get('ttl', float('nan')):.1f} EUR/MWh")
col5.metric("Phase", diag.get("phase", "unknown"))

render_commentary(comment_kpi(m, label="Lecture KPI"))

section("Comparaison rapide pays", "Derniere annee chargee")
rows = []
for c in state["countries_selected"]:
    key = (c, latest_year, state["price_mode"])
    if key not in state["metrics"]:
        continue
    mm = state["metrics"][key]
    dd = state["diagnostics"].get((c, latest_year), {})
    rows.append(
        {
            "country": c,
            "sr": mm.get("sr"),
            "far": mm.get("far"),
            "ir": mm.get("ir"),
            "ttl": mm.get("ttl"),
            "capture_ratio_pv": mm.get("capture_ratio_pv"),
            "h_negative_obs": mm.get("h_negative_obs"),
            "phase": dd.get("phase", "unknown"),
        }
    )

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

render_commentary(
    "**Comparaison pays**\n"
    f"- Constat chiffre: n={len(rows)} pays, annee={latest_year}.\n"
    "- Lien methode: indicateurs calcules via G.7 sur un schema de donnees homogene.\n"
    "- Limites/portee: comparaison sensible a la completude et au mode de prix selectionne."
)
