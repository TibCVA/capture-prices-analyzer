"""
Capture Prices Analyzer -- CVA
Page d'accueil et configuration globale.
"""
import os
import sys
import logging
import streamlit as st
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('capture_prices.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("capture_prices")

# Load dotenv
from dotenv import load_dotenv
load_dotenv()

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_fetcher import fetch_country_year, load_commodity_prices
from src.data_loader import (load_raw, load_processed, save_processed,
                              load_country_config, load_all_countries_config,
                              load_thresholds, load_metrics, save_metrics,
                              load_diagnostics, save_diagnostics,
                              scan_cached_processed)
from src.nrl_engine import compute_nrl
from src.metrics import compute_annual_metrics
from src.phase_diagnostics import diagnose_phase
from src.ui_helpers import inject_global_css, narrative


def _get_secret(key: str, default: str = "") -> str:
    """Read a secret from st.secrets (Streamlit Cloud) with safe fallback."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


@st.cache_data(ttl=3600, show_spinner="Chargement des données...")
def load_and_process(country_key: str, year: int, must_run_mode: str,
                     api_key: str | None = None) -> tuple:
    """Cache load + process pendant 1h."""
    # 1. Tenter cache processed
    processed = load_processed(country_key, year, must_run_mode)
    if processed is not None:
        metrics = compute_annual_metrics(processed, year, country_key)
        return processed, metrics

    # 2. Tenter cache raw
    raw = load_raw(country_key, year)
    if raw is None and api_key:
        raw = fetch_country_year(country_key, year, api_key)

    if raw is None:
        return None, None

    # 3. Process
    config = load_country_config(country_key)
    commodities = load_commodity_prices()
    processed = compute_nrl(raw, config, country_key, year, commodities, must_run_mode)

    # 4. Save processed
    save_processed(processed, country_key, year, must_run_mode)

    # 5. Metrics
    metrics = compute_annual_metrics(processed, year, country_key)
    return processed, metrics


# ============================================================
# Session State initialization
# ============================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.raw_data = {}
    st.session_state.processed_data = {}
    st.session_state.annual_metrics = {}
    st.session_state.diagnostics = {}
    st.session_state.selected_countries = ['FR', 'DE', 'ES']
    st.session_state.year_range = (2015, 2024)
    st.session_state.exclude_2022 = True
    st.session_state.must_run_mode = 'observed'
    st.session_state.commodity_prices = None
    st.session_state.thresholds = None
    st.session_state.custom_hypotheses = {}

    # --- Auto-chargement depuis le cache disque ---
    mode = 'observed'
    cached = scan_cached_processed(mode)
    if cached:
        thresholds = load_thresholds()
        st.session_state.thresholds = thresholds
        commodity_prices = load_commodity_prices()
        st.session_state.commodity_prices = commodity_prices
        countries_found = set()
        years_found = set()

        for country_key, year in cached:
            # 1. Essayer les JSON metrics/diagnostics (instantane)
            metrics = load_metrics(country_key, year, mode)
            diag = load_diagnostics(country_key, year, mode)

            if metrics and diag:
                # Pas besoin de charger le parquet pour le dashboard
                st.session_state.annual_metrics[(country_key, year)] = metrics
                st.session_state.diagnostics[(country_key, year)] = diag
            else:
                # Fallback : charger le parquet et calculer
                processed = load_processed(country_key, year, mode)
                if processed is None:
                    continue
                st.session_state.processed_data[(country_key, year)] = processed
                metrics = compute_annual_metrics(processed, year, country_key)
                st.session_state.annual_metrics[(country_key, year)] = metrics
                prev_metrics = st.session_state.annual_metrics.get((country_key, year - 1))
                diag = diagnose_phase(metrics, thresholds, prev_metrics)
                st.session_state.diagnostics[(country_key, year)] = diag
                # Persister pour la prochaine fois
                save_metrics(metrics, country_key, year, mode)
                save_diagnostics(diag, country_key, year, mode)

            countries_found.add(country_key)
            years_found.add(year)

        if countries_found:
            st.session_state.data_loaded = True
            st.session_state.selected_countries = sorted(countries_found)
            st.session_state.year_range = (min(years_found), max(years_found))
            logger.info(f"Auto-loaded {len(st.session_state.annual_metrics)} pays/annees depuis le cache")

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Capture Prices Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS
inject_global_css()

# ============================================================
# Sidebar
# ============================================================
all_countries = load_all_countries_config()
country_options = {k: v['name'] for k, v in all_countries.items()}

with st.sidebar:
    st.markdown("#### ⚡ Capture Prices Analyzer")

    # --- Selection (toujours visible) ---
    st.markdown("**Selection**")

    selected = st.multiselect(
        "Pays",
        options=list(country_options.keys()),
        default=st.session_state.selected_countries,
        format_func=lambda x: f"{x} - {country_options[x]}",
    )
    st.session_state.selected_countries = selected

    year_range = st.slider(
        "Periode",
        min_value=2015, max_value=2024,
        value=st.session_state.year_range,
    )
    st.session_state.year_range = year_range

    st.divider()

    btn_load = st.button("Charger les donnees", type="primary", use_container_width=True)

    # --- Options avancees (expander) ---
    with st.expander("Options avancees"):
        api_key = st.text_input(
            "Cle API ENTSO-E",
            value=os.environ.get('ENTSOE_API_KEY', '') or _get_secret("ENTSOE_API_KEY"),
            type="password",
            help="Obtenir gratuitement sur transparency.entsoe.eu"
        )

        exclude_2022 = st.checkbox("Exclure 2022 (crise gaz)", value=st.session_state.exclude_2022)
        st.session_state.exclude_2022 = exclude_2022

        must_run_mode = st.radio(
            "Mode must-run",
            options=['observed', 'floor'],
            index=0 if st.session_state.must_run_mode == 'observed' else 1,
            help="'observed' = donnees reelles, 'floor' = planchers techniques"
        )
        st.session_state.must_run_mode = must_run_mode

        btn_force = st.button("Forcer refresh", use_container_width=True)

    # --- Chargement ---
    if btn_load or btn_force:
        force = btn_force
        thresholds = load_thresholds()
        st.session_state.thresholds = thresholds
        commodity_prices = load_commodity_prices()
        st.session_state.commodity_prices = commodity_prices

        # Precharger les configs pays
        configs = {ck: load_country_config(ck) for ck in selected}
        years = list(range(year_range[0], year_range[1] + 1))

        # Separer les taches en cache-hit (rapides) et api-needed (lentes)
        tasks_cached = []
        tasks_api = []
        for ck in selected:
            for y in years:
                processed = load_processed(ck, y, must_run_mode)
                if processed is not None and not force:
                    tasks_cached.append((ck, y, processed))
                else:
                    tasks_api.append((ck, y))

        total = len(tasks_cached) + len(tasks_api)
        done = 0
        progress = st.progress(0, text="Chargement du cache...")

        # Hypotheses custom de la page Sources & Hypotheses
        hyp_override = st.session_state.get("custom_hypotheses") or None

        # 1. Traiter les donnees en cache (instantane)
        for ck, y, processed in tasks_cached:
            metrics = load_metrics(ck, y, must_run_mode)
            if metrics is None or hyp_override:
                metrics = compute_annual_metrics(processed, y, ck,
                                                 constants_override=hyp_override)
                save_metrics(metrics, ck, y, must_run_mode)

            st.session_state.processed_data[(ck, y)] = processed
            st.session_state.annual_metrics[(ck, y)] = metrics

            prev_m = st.session_state.annual_metrics.get((ck, y - 1))
            diag = load_diagnostics(ck, y, must_run_mode)
            if diag is None:
                diag = diagnose_phase(metrics, thresholds, prev_m)
                save_diagnostics(diag, ck, y, must_run_mode)
            st.session_state.diagnostics[(ck, y)] = diag

            done += 1
            progress.progress(done / max(total, 1),
                              text=f"Cache: {ck} {y} ({done}/{total})")

        st.session_state.data_loaded = True

        # 2. Telecharger les donnees manquantes (API) en parallele
        if tasks_api and api_key:
            errors = []

            def _fetch_one(ck, y):
                raw = load_raw(ck, y)
                if raw is None:
                    raw = fetch_country_year(ck, y, api_key, force=force)
                processed = compute_nrl(raw, configs[ck], ck, y,
                                        commodity_prices, must_run_mode,
                                        constants_override=hyp_override)
                save_processed(processed, ck, y, must_run_mode)
                metrics = compute_annual_metrics(processed, y, ck,
                                                 constants_override=hyp_override)
                save_metrics(metrics, ck, y, must_run_mode)
                return ck, y, processed, metrics

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(_fetch_one, ck, y): (ck, y)
                    for ck, y in tasks_api
                }
                for future in as_completed(futures):
                    ck, y = futures[future]
                    try:
                        ck, y, processed, metrics = future.result()
                        st.session_state.processed_data[(ck, y)] = processed
                        st.session_state.annual_metrics[(ck, y)] = metrics

                        prev_m = st.session_state.annual_metrics.get((ck, y - 1))
                        diag = diagnose_phase(metrics, thresholds, prev_m)
                        save_diagnostics(diag, ck, y, must_run_mode)
                        st.session_state.diagnostics[(ck, y)] = diag
                    except Exception as e:
                        errors.append(f"{ck}/{y}: {e}")

                    done += 1
                    progress.progress(done / max(total, 1),
                                      text=f"API: {ck} {y} ({done}/{total})")

            for err in errors:
                st.warning(err)

        elif tasks_api:
            st.info(f"{len(tasks_api)} pays/annees sans cache et sans cle API -- ignorees.")

        progress.progress(1.0, text="Termine !")
        st.rerun()

    # Status
    if st.session_state.data_loaded:
        st.success(f"{len(st.session_state.annual_metrics)} pays/annees charges")
        for ck in st.session_state.selected_countries:
            loaded_years = sorted([y for (c, y) in st.session_state.annual_metrics if c == ck])
            if loaded_years:
                st.caption(f"{ck}: {loaded_years[0]}-{loaded_years[-1]} ({len(loaded_years)} ans)")

# ============================================================
# Main page
# ============================================================
st.title("⚡ Capture Prices Analyzer")
st.caption("Analyse des capture prices VRE en Europe -- CVA pour TotalEnergies")

if not st.session_state.data_loaded:
    # ---- Accueil premier usage ----
    st.markdown("")
    st.markdown("### Bienvenue")
    st.markdown("Cet outil analyse les **capture prices** des energies renouvelables "
                "sur 5 marches europeens (2015-2024). Pour commencer :")
    st.markdown("")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown('''<div class="info-card">
            <h4>1. Selectionnez</h4>
            <p>Choisissez vos pays et la periode d'analyse dans le panneau de gauche.</p>
        </div>''', unsafe_allow_html=True)
    with col_s2:
        st.markdown('''<div class="info-card">
            <h4>2. Chargez</h4>
            <p>Cliquez sur <strong>Charger les donnees</strong>. Le telechargement prend quelques minutes.</p>
        </div>''', unsafe_allow_html=True)
    with col_s3:
        st.markdown('''<div class="info-card">
            <h4>3. Explorez</h4>
            <p>Naviguez entre les pages d'analyse via le menu a gauche.</p>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # ---- Schema du modele (HTML) ----
    st.markdown("### Architecture du modele")
    st.markdown('''
    <div style="display: flex; flex-direction: column; align-items: center; gap: 6px; margin: 1rem 0 2rem 0;">
        <div style="background: #EBF5FB; border: 2px solid #3498DB; border-radius: 10px;
                    padding: 10px 24px; text-align: center; width: 80%; max-width: 520px;">
            <strong style="color: #2980B9;">Niveau 1</strong><br>
            <span style="font-size: 0.88rem;">Donnees ENTSO-E horaires (load, generation, prix)</span>
        </div>
        <div style="font-size: 1.2rem; color: #95A5A6;">&#x25BC;</div>
        <div style="background: #E8F8F5; border: 2px solid #27AE60; border-radius: 10px;
                    padding: 10px 24px; text-align: center; width: 80%; max-width: 520px;">
            <strong style="color: #27AE60;">Niveau 2</strong><br>
            <span style="font-size: 0.88rem;">NRL = Load &minus; VRE &minus; Must-Run</span>
        </div>
        <div style="font-size: 1.2rem; color: #95A5A6;">&#x25BC;</div>
        <div style="background: #FEF9E7; border: 2px solid #F39C12; border-radius: 10px;
                    padding: 10px 24px; text-align: center; width: 80%; max-width: 520px;">
            <strong style="color: #E67E22;">Niveau 3</strong><br>
            <span style="font-size: 0.88rem;">4 regimes : Surplus | Absorbe | Thermique | Queue haute</span>
        </div>
        <div style="font-size: 1.2rem; color: #95A5A6;">&#x25BC;</div>
        <div style="background: #F4ECF7; border: 2px solid #8E44AD; border-radius: 10px;
                    padding: 10px 24px; text-align: center; width: 80%; max-width: 520px;">
            <strong style="color: #8E44AD;">Niveau 4</strong><br>
            <span style="font-size: 0.88rem;">~50 metriques annuelles (capture rates, FAR, IR, TTL...)</span>
        </div>
        <div style="font-size: 1.2rem; color: #95A5A6;">&#x25BC;</div>
        <div style="background: #FDEDEC; border: 2px solid #E74C3C; border-radius: 10px;
                    padding: 10px 24px; text-align: center; width: 80%; max-width: 520px;">
            <strong style="color: #C0392B;">Niveau 5</strong><br>
            <span style="font-size: 0.88rem;">Diagnostic de phase (Stage 1&rarr;4) + Scenarios prospectifs</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

else:
    # ---- KPI cards ----
    last_year = st.session_state.year_range[1]
    default_country = st.session_state.selected_countries[0] if st.session_state.selected_countries else 'FR'

    m = st.session_state.annual_metrics.get((default_country, last_year))
    d = st.session_state.diagnostics.get((default_country, last_year))

    if m and d:
        st.markdown(f"### Tableau de bord -- {country_options.get(default_country, default_country)} {last_year}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Heures a prix negatif", f"{m['h_negative']}",
                      help="Nombre d'heures dans l'annee ou le prix spot est tombe sous 0 EUR/MWh. "
                           "Plus ce nombre est eleve, plus le marche souffre de surplus VRE.")
        with col2:
            cr_pv = m.get('capture_ratio_pv')
            val = f"{cr_pv:.1%}" if cr_pv and cr_pv == cr_pv else "N/A"
            st.metric("Capture Ratio PV", val,
                      help="Ratio entre le prix moyen capte par le solaire et le prix moyen du marche. "
                           "En dessous de 0.80, on parle de cannibalisation significative.")
        with col3:
            far = m.get('far_structural')
            val = f"{far:.1%}" if far and far == far else "N/A"
            st.metric("FAR structural", val,
                      help="Capacite theorique du systeme a absorber les surplus VRE "
                           "(stockage, exports, flexibilite). 1.0 = absorption totale.")
        with col4:
            pn = d['phase_number']
            phase_info = {
                1: "Stage 1",
                2: "Stage 2",
                3: "Stage 3",
                4: "Stage 4",
            }
            st.metric("Phase du marche", f"Stage {pn}",
                      help="Diagnostic base sur un faisceau d'indicateurs (SR, FAR, IR, heures negatives). "
                           "Stage 1 = integration facile, Stage 4 = saturation structurelle.")

    st.divider()

    # ---- Tableau synthetique multi-pays ----
    st.markdown(f"### Vue comparee {last_year}")
    narrative("Ce tableau resume les indicateurs-clefs de chaque marche pour l'annee la plus recente. "
              "Les colonnes colorees indiquent les valeurs favorables (vert) ou les signaux d'alerte (rouge).")

    rows = []
    for ck in st.session_state.selected_countries:
        m_ck = st.session_state.annual_metrics.get((ck, last_year))
        d_ck = st.session_state.diagnostics.get((ck, last_year))
        if m_ck and d_ck:
            cr = m_ck.get('capture_ratio_pv')
            far_val = m_ck.get('far_structural')
            rows.append({
                'Pays': all_countries[ck]['name'],
                'Part VRE': m_ck['vre_share'],
                'H neg.': int(m_ck['h_negative']),
                'Capture PV': cr if cr and cr == cr else None,
                'FAR': far_val if far_val and far_val == far_val else None,
                'Phase': f"Stage {d_ck['phase_number']}",
            })

    if rows:
        df_synth = pd.DataFrame(rows)
        styler = df_synth.style.format({
            'Part VRE': '{:.1%}',
            'Capture PV': '{:.2f}',
            'FAR': '{:.2f}',
        }, na_rep='--')

        # Gradient conditionnel sur les colonnes numeriques
        if 'Capture PV' in df_synth.columns and df_synth['Capture PV'].notna().any():
            styler = styler.background_gradient(
                subset=['Capture PV'], cmap='RdYlGn', vmin=0.5, vmax=1.0)
        if 'FAR' in df_synth.columns and df_synth['FAR'].notna().any():
            styler = styler.background_gradient(
                subset=['FAR'], cmap='RdYlGn', vmin=0.0, vmax=1.0)

        st.dataframe(styler, use_container_width=True, hide_index=True, height=200)

    st.divider()

    # ---- Schema du modele (HTML, version compacte) ----
    st.markdown("### Architecture du modele")
    st.markdown('''
    <div style="display: flex; flex-direction: column; align-items: center; gap: 4px; margin: 0.5rem 0 1rem 0;">
        <div style="background: #EBF5FB; border: 1px solid #3498DB; border-radius: 8px;
                    padding: 6px 16px; text-align: center; width: 70%; max-width: 480px; font-size: 0.85rem;">
            <strong style="color: #2980B9;">N1</strong> Donnees ENTSO-E
        </div>
        <div style="color: #95A5A6;">&#x25BC;</div>
        <div style="background: #E8F8F5; border: 1px solid #27AE60; border-radius: 8px;
                    padding: 6px 16px; text-align: center; width: 70%; max-width: 480px; font-size: 0.85rem;">
            <strong style="color: #27AE60;">N2</strong> NRL = Load &minus; VRE &minus; Must-Run
        </div>
        <div style="color: #95A5A6;">&#x25BC;</div>
        <div style="background: #FEF9E7; border: 1px solid #F39C12; border-radius: 8px;
                    padding: 6px 16px; text-align: center; width: 70%; max-width: 480px; font-size: 0.85rem;">
            <strong style="color: #E67E22;">N3</strong> 4 regimes de prix
        </div>
        <div style="color: #95A5A6;">&#x25BC;</div>
        <div style="background: #F4ECF7; border: 1px solid #8E44AD; border-radius: 8px;
                    padding: 6px 16px; text-align: center; width: 70%; max-width: 480px; font-size: 0.85rem;">
            <strong style="color: #8E44AD;">N4</strong> Metriques annuelles
        </div>
        <div style="color: #95A5A6;">&#x25BC;</div>
        <div style="background: #FDEDEC; border: 1px solid #E74C3C; border-radius: 8px;
                    padding: 6px 16px; text-align: center; width: 70%; max-width: 480px; font-size: 0.85rem;">
            <strong style="color: #C0392B;">N5</strong> Diagnostic + Scenarios
        </div>
    </div>
    ''', unsafe_allow_html=True)
