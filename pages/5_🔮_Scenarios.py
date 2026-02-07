"""
Page 5 -- Scenarios
Simuler l'impact de capacites VRE, BESS, demande et commodity prices sur les metriques.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.constants import *
from src.scenario_engine import apply_scenario
from src.metrics import compute_annual_metrics
from src.data_loader import load_scenarios_config
from src.ui_helpers import inject_global_css, narrative, guard_no_data, dynamic_narrative, challenge_block

st.set_page_config(page_title="Scenarios", page_icon="🔮", layout="wide")
inject_global_css()
st.title("🔮 Scenarios")

# ── Validation session state ──────────────────────────────────────────────
required_keys = ["processed_data", "annual_metrics", "selected_countries"]
if (not all(k in st.session_state for k in required_keys)
        or not st.session_state.get("annual_metrics")):
    guard_no_data("les Scenarios")

narrative("Le moteur de scenarios modifie les profils VRE, la demande, le stockage et les prix "
          "de commodites sur la base de l'annee la plus recente, puis recalcule l'ensemble du pipeline "
          "NRL → regimes → metriques. Les prix affiches sont des prix mecanistes (proxy affine "
          "par morceaux), pas un merit order complet. Limites : les profils VRE sont mis a l'echelle "
          "lineairement, le BESS est simule avec un SoC reset journalier, les interconnexions ne "
          "sont pas recalculees.")

processed_data: dict = st.session_state["processed_data"]
annual_metrics: dict = st.session_state["annual_metrics"]
selected_countries: list = st.session_state["selected_countries"]
commodity_prices: dict = st.session_state.get("commodity_prices", {})

# ── Charger scenarios predefinis ──────────────────────────────────────────
try:
    scenarios_config = load_scenarios_config()
    predefined_scenarios = scenarios_config.get("scenarios", {})
except Exception:
    predefined_scenarios = {}

# ══════════════════════════════════════════════════════════════════════════
# LAYOUT : 35% gauche (parametres) | 65% droite (resultats)
# ══════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([35, 65])

# ── Colonne gauche : selecteurs + sliders ─────────────────────────────────
with col_left:
    st.subheader("Parametres")

    # Baseline
    available_keys = sorted(processed_data.keys())
    available_countries = sorted({k[0] for k in available_keys})
    baseline_country = st.selectbox("Pays baseline", available_countries,
                                    index=0 if available_countries else 0)
    available_years_for_country = sorted([k[1] for k in available_keys if k[0] == baseline_country])
    baseline_year = st.selectbox("Annee baseline", available_years_for_country,
                                 index=len(available_years_for_country) - 1 if available_years_for_country else 0)

    st.divider()

    # Scenario predefini OU manuel — noms lisibles
    _sc_key_by_label = {}
    _sc_labels = ["-- Manuel --"]
    for _k, _v in predefined_scenarios.items():
        _label = _v.get("name", _k)
        _sc_labels.append(_label)
        _sc_key_by_label[_label] = _k

    scenario_label = st.selectbox("Scenario predefini", _sc_labels)
    scenario_key = _sc_key_by_label.get(scenario_label)

    if scenario_key and scenario_key in predefined_scenarios:
        preset_raw = predefined_scenarios[scenario_key]
        preset = preset_raw.get("params", preset_raw)  # supporte les 2 formats YAML
        delta_pv = preset.get("delta_pv_gw", 0.0)
        delta_wind_on = preset.get("delta_wind_onshore_gw", 0.0)
        delta_wind_off = preset.get("delta_wind_offshore_gw", 0.0)
        delta_bess_power = preset.get("delta_bess_power_gw", 0.0)
        delta_bess_energy = preset.get("delta_bess_energy_gwh", 0.0)
        delta_demand_pct = preset.get("delta_demand_pct", 0.0)
        delta_demand_midday = preset.get("delta_demand_midday_gw", 0.0)
        gas_price = preset.get("gas_price_eur_mwh", 30.0)
        co2_price = preset.get("co2_price_eur_t", 65.0)
        st.caption(f"Predefini : {preset_raw.get('description', scenario_key)}")
    else:
        delta_pv = 0.0
        delta_wind_on = 0.0
        delta_wind_off = 0.0
        delta_bess_power = 0.0
        delta_bess_energy = 0.0
        delta_demand_pct = 0.0
        delta_demand_midday = 0.0
        gas_price = 30.0
        co2_price = 65.0

    # ── Sliders organises en 3 onglets ────────────────────────────────────
    tab_vre, tab_flex, tab_prix = st.tabs(["Capacites VRE", "Stockage & Demande", "Commodites"])

    with tab_vre:
        st.markdown("**Delta GW**")
        delta_pv = st.slider("PV", 0.0, 50.0, float(delta_pv), 0.5, key="sl_pv")
        delta_wind_on = st.slider("Eolien onshore", 0.0, 30.0, float(delta_wind_on), 0.5, key="sl_won")
        delta_wind_off = st.slider("Eolien offshore", 0.0, 20.0, float(delta_wind_off), 0.5, key="sl_woff")

    with tab_flex:
        st.markdown("**Stockage BESS (delta)**")
        delta_bess_power = st.slider("Puissance (GW)", 0.0, 30.0, float(delta_bess_power), 0.5, key="sl_bess_p")
        delta_bess_energy = st.slider("Energie (GWh)", 0.0, 120.0, float(delta_bess_energy), 1.0, key="sl_bess_e")
        st.markdown("**Demande**")
        delta_demand_pct = st.slider("Variation demande (%)", -20.0, 30.0, float(delta_demand_pct), 1.0, key="sl_dem")
        delta_demand_midday = st.slider("Demande midday (GW)", 0.0, 20.0, float(delta_demand_midday), 0.5, key="sl_mid")

    with tab_prix:
        st.markdown("**Commodity prices**")
        gas_price = st.slider("Gaz (EUR/MWh_th)", 10.0, 80.0, float(gas_price), 1.0, key="sl_gas")
        co2_price = st.slider("CO2 (EUR/t)", 20.0, 200.0, float(co2_price), 5.0, key="sl_co2")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_scenario = st.button("Calculer le scenario", type="primary")
    with col_btn2:
        reset = st.button("Reset")

    if reset:
        for key in ["sl_pv", "sl_won", "sl_woff", "sl_bess_p", "sl_bess_e",
                     "sl_dem", "sl_mid", "sl_gas", "sl_co2"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Colonne droite : resultats ────────────────────────────────────────────
with col_right:
    if not run_scenario:
        st.markdown('''
        <div class="guard-message">
            <h3>Parametrez votre scenario</h3>
            <p>Ajustez les curseurs a gauche (capacites VRE, stockage, prix des commodites)
            puis cliquez <strong>Calculer le scenario</strong> pour voir l'impact.</p>
        </div>
        ''', unsafe_allow_html=True)
        st.stop()

    baseline_key = (baseline_country, baseline_year)
    if baseline_key not in processed_data:
        st.error(f"Pas de donnees pour {baseline_country}/{baseline_year}.")
        st.stop()

    baseline_df = processed_data[baseline_key]
    baseline_metrics = annual_metrics.get(baseline_key, {})

    # Charger la config pays
    from src.data_loader import load_country_config
    try:
        country_config = load_country_config(baseline_country)
    except Exception as e:
        st.error(f"Config pays introuvable: {e}")
        st.stop()

    # Construire le dict de parametres scenario
    scenario_params = {
        "delta_pv_gw": delta_pv,
        "delta_wind_onshore_gw": delta_wind_on,
        "delta_wind_offshore_gw": delta_wind_off,
        "delta_bess_power_gw": delta_bess_power,
        "delta_bess_energy_gwh": delta_bess_energy,
        "delta_demand_pct": delta_demand_pct,
        "delta_demand_midday_gw": delta_demand_midday,
        "gas_price_eur_mwh": gas_price,
        "co2_price_eur_t": co2_price,
    }

    hyp_override = st.session_state.get("custom_hypotheses") or None
    with st.spinner("Calcul du scenario..."):
        scenario_df = apply_scenario(
            baseline_df, scenario_params, country_config,
            baseline_country, baseline_year, commodity_prices,
            constants_override=hyp_override,
        )
        scenario_metrics = compute_annual_metrics(scenario_df, baseline_year, baseline_country,
                                                   constants_override=hyp_override)

    # ── 5 delta metric cards ──────────────────────────────────────────────
    st.subheader("Impact du scenario")

    def _delta_card(label: str, baseline_val, scenario_val, unit: str = "",
                    inverse: bool = False, help: str = ""):
        """Affiche une metrique avec delta colore."""
        if baseline_val is None or (isinstance(baseline_val, float) and np.isnan(baseline_val)):
            baseline_val = 0
        if scenario_val is None or (isinstance(scenario_val, float) and np.isnan(scenario_val)):
            scenario_val = 0
        delta = scenario_val - baseline_val
        direction = "inverse" if inverse else "normal"
        st.metric(label, f"{scenario_val:.2f} {unit}", f"{delta:+.2f} {unit}",
                  delta_color=direction, help=help)

    card_cols = st.columns(5)
    with card_cols[0]:
        _delta_card("H regime A (surplus)",
                    baseline_metrics.get("h_regime_a", 0),
                    scenario_metrics.get("h_regime_a", 0), "h", inverse=True,
                    help="Heures de surplus non absorbe. Une baisse = le stockage absorbe mieux.")
    with card_cols[1]:
        _delta_card("FAR structural",
                    baseline_metrics.get("far_structural", 0),
                    scenario_metrics.get("far_structural", 0),
                    help="Capacite d'absorption des surplus (PSH+BESS+DSM). 1.0 = absorption totale.")
    with card_cols[2]:
        _delta_card("Prix moyen meca.",
                    baseline_metrics.get("baseload_price", 0),
                    scenario_metrics.get("baseload_price", 0), "EUR/MWh",
                    help="Prix moyen mecaniste (proxy). Baseline = prix observe, Scenario = prix recalcule.")
    with card_cols[3]:
        _delta_card("TTL (tension)",
                    baseline_metrics.get("ttl", 0),
                    scenario_metrics.get("ttl", 0), "EUR/MWh",
                    help="Prix P95 en heures thermiques (regimes C+D). Indicateur de pointe.")
    with card_cols[4]:
        _delta_card("Capture Ratio PV",
                    baseline_metrics.get("capture_ratio_pv", 0),
                    scenario_metrics.get("capture_ratio_pv", 0),
                    help="Prix capte par le solaire / prix moyen. Sous 0.80 = cannibalisation.")

    # ── Resume narratif dynamique ───────────────────────────────────────
    _changes = []
    if scenario_params.get("delta_pv_gw", 0) != 0:
        _changes.append(f"PV {scenario_params['delta_pv_gw']:+.0f} GW")
    if scenario_params.get("delta_wind_onshore_gw", 0) != 0:
        _changes.append(f"Eolien onshore {scenario_params['delta_wind_onshore_gw']:+.0f} GW")
    if scenario_params.get("delta_wind_offshore_gw", 0) != 0:
        _changes.append(f"Eolien offshore {scenario_params['delta_wind_offshore_gw']:+.0f} GW")
    if scenario_params.get("delta_bess_power_gw", 0) != 0:
        _changes.append(f"BESS {scenario_params['delta_bess_power_gw']:+.0f} GW / {scenario_params.get('delta_bess_energy_gwh', 0):+.0f} GWh")
    if scenario_params.get("delta_demand_pct", 0) != 0:
        _changes.append(f"Demande {scenario_params['delta_demand_pct']:+.0f}%")
    if scenario_params.get("gas_price_eur_mwh", 30) != 30:
        _changes.append(f"Gaz {scenario_params['gas_price_eur_mwh']:.0f} EUR/MWh")
    if scenario_params.get("co2_price_eur_t", 65) != 65:
        _changes.append(f"CO2 {scenario_params['co2_price_eur_t']:.0f} EUR/t")

    _ha_b = baseline_metrics.get("h_regime_a", 0)
    _ha_s = scenario_metrics.get("h_regime_a", 0)
    _far_b = baseline_metrics.get("far_structural", 0) or 0
    _far_s = scenario_metrics.get("far_structural", 0) or 0
    _cr_b = baseline_metrics.get("capture_ratio_pv", 0) or 0
    _cr_s = scenario_metrics.get("capture_ratio_pv", 0) or 0

    _summary_parts = []
    if _changes:
        _summary_parts.append(f"<strong>Hypotheses :</strong> {', '.join(_changes)}.")
    if _ha_b != _ha_s:
        _pct = ((_ha_s - _ha_b) / max(_ha_b, 1)) * 100
        _summary_parts.append(f"Heures regime A : {_ha_b} &rarr; {_ha_s} ({_pct:+.0f}%).")
    if _far_b != _far_s:
        _summary_parts.append(f"FAR structural : {_far_b:.2f} &rarr; {_far_s:.2f} ({_far_s - _far_b:+.2f}).")
    if _cr_b != _cr_s:
        _direction = "cannibalisation accrue" if _cr_s < _cr_b else "amelioration"
        _summary_parts.append(f"Capture ratio PV : {_cr_b:.2f} &rarr; {_cr_s:.2f} ({_direction}).")

    if _summary_parts:
        dynamic_narrative(" ".join(_summary_parts), "info")

    # ── Tableau comparatif Baseline vs Scenario ─────────────────────────
    _compare_keys = [
        ("H regime A (surplus)", "h_regime_a", "h", True),
        ("H regime B (absorbe)", "h_regime_b", "h", False),
        ("H regime C (thermique)", "h_regime_c", "h", False),
        ("H regime D (tension)", "h_regime_d_tail", "h", False),
        ("H prix negatifs", "h_negative", "h", True),
        ("FAR structural", "far_structural", "", False),
        ("Capture Ratio PV", "capture_ratio_pv", "", False),
        ("Capture Ratio Wind", "capture_ratio_wind", "", False),
        ("Prix moyen", "baseload_price", "EUR/MWh", False),
        ("TTL", "ttl", "EUR/MWh", False),
        ("Surplus (TWh)", "total_surplus_twh", "TWh", True),
    ]
    _rows_cmp = []
    for _lbl, _key, _unit, _inv in _compare_keys:
        _bv = baseline_metrics.get(_key)
        _sv = scenario_metrics.get(_key)
        if _bv is None or (isinstance(_bv, float) and np.isnan(_bv)):
            _bv = 0
        if _sv is None or (isinstance(_sv, float) and np.isnan(_sv)):
            _sv = 0
        _rows_cmp.append({
            "Metrique": _lbl,
            "Baseline": round(_bv, 2),
            "Scenario": round(_sv, 2),
            "Delta": round(_sv - _bv, 2),
        })
    with st.expander("Tableau comparatif detaille", expanded=False):
        st.dataframe(pd.DataFrame(_rows_cmp), use_container_width=True, hide_index=True)

    st.divider()

    # ── Grouped bars : regimes baseline vs scenario ───────────────────────
    st.subheader("Repartition des regimes")

    regimes = ["A", "B", "C", "D_tail"]
    regime_keys = ["h_regime_a", "h_regime_b", "h_regime_c", "h_regime_d_tail"]
    baseline_hours = [baseline_metrics.get(k, 0) for k in regime_keys]
    scenario_hours = [scenario_metrics.get(k, 0) for k in regime_keys]

    fig_regimes = go.Figure()
    fig_regimes.add_trace(go.Bar(
        name="Baseline", x=regimes, y=baseline_hours,
        marker_color="#636EFA", opacity=0.7,
    ))
    fig_regimes.add_trace(go.Bar(
        name="Scenario", x=regimes, y=scenario_hours,
        marker_color="#EF553B", opacity=0.7,
    ))
    fig_regimes.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        barmode="group", height=380,
        yaxis_title="Heures",
        title="Heures par regime : Baseline vs Scenario",
    )
    st.plotly_chart(fig_regimes, use_container_width=True)

    # ── NRL Duration Curve ────────────────────────────────────────────────
    st.subheader("Courbe de duree NRL")

    nrl_baseline = baseline_df[COL_NRL].sort_values(ascending=False).reset_index(drop=True)
    nrl_scenario = scenario_df[COL_NRL].sort_values(ascending=False).reset_index(drop=True)

    fig_nrl = go.Figure()
    fig_nrl.add_trace(go.Scatter(
        x=list(range(len(nrl_baseline))), y=nrl_baseline,
        mode="lines", name="Baseline",
        line=dict(color="gray", width=1.5),
    ))
    fig_nrl.add_trace(go.Scatter(
        x=list(range(len(nrl_scenario))), y=nrl_scenario,
        mode="lines", name="Scenario",
        line=dict(color="#EF553B", width=2),
    ))
    fig_nrl.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig_nrl.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=420,
        xaxis_title="Heures (triees par NRL decroissant)",
        yaxis_title="NRL (MW)",
        title="Net Residual Load -- Duration Curve",
    )
    st.plotly_chart(fig_nrl, use_container_width=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.warning(
        "**Disclaimer** : Les prix mecanistes sont un proxy simplifie. "
        "Ils ne modelisent pas le merit order complet. "
        "Utiliser pour detecter des tendances, pas pour du pricing precis.",
        icon="⚠️"
    )
