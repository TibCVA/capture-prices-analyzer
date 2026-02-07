"""
Page 6 -- Questions Stephane Michel
6 onglets repondant aux questions cles sur la transition energetique.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.constants import *
from src.slope_analysis import compute_slope
from src.scenario_engine import apply_scenario
from src.ui_helpers import inject_global_css, narrative, guard_no_data, question_banner, info_card, dynamic_narrative, challenge_block

st.set_page_config(page_title="Questions S. Michel", page_icon="❓", layout="wide")
inject_global_css()
st.title("❓ Questions Stephane Michel")

# ── Validation session state ──────────────────────────────────────────────
required_keys = ["annual_metrics", "diagnostics", "selected_countries", "processed_data"]
if (not all(k in st.session_state for k in required_keys)
        or not st.session_state.get("annual_metrics")):
    guard_no_data("les Questions S. Michel")

annual_metrics: dict = st.session_state["annual_metrics"]
diagnostics: dict = st.session_state["diagnostics"]
selected_countries: list = st.session_state["selected_countries"]
processed_data: dict = st.session_state["processed_data"]
exclude_2022: bool = st.session_state.get("exclude_2022", True)
thresholds: dict = st.session_state.get("thresholds", {})
commodity_prices: dict = st.session_state.get("commodity_prices", {})

narrative("Cette page repond aux 6 questions-clefs posees par Stephane Michel, "
          "chacune avec une reponse synthetique et un graphique interactif.")

# ── Helpers ───────────────────────────────────────────────────────────────

def _get_metrics_series(country: str) -> list[dict]:
    """Retourne la liste des metriques annuelles pour un pays, triee par annee."""
    series = []
    for (c, y), m in sorted(annual_metrics.items()):
        if c == country:
            m_copy = dict(m)
            m_copy["_diag"] = diagnostics.get((c, y), {})
            series.append(m_copy)
    return series


def _all_points() -> pd.DataFrame:
    """DataFrame de tous les (country, year, metrics) disponibles."""
    rows = []
    for (c, y), m in annual_metrics.items():
        d = diagnostics.get((c, y), {})
        row = {"country": c, "year": y, "phase_number": d.get("phase_number", 1)}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


all_df = _all_points()
if exclude_2022:
    all_df_reg = all_df[~all_df["year"].isin(OUTLIER_YEARS)]
else:
    all_df_reg = all_df.copy()

# ══════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "Q1 : Phase 1→2",
    "Q2 : Slope Phase 2",
    "Q3 : Phase 2→3",
    "Q4 : Batteries",
    "Q5 : CO2/Gaz",
    "Q6 : Stockage thermique",
])

# ── Q1 : Transition Phase 1 -> 2 ─────────────────────────────────────────
with tabs[0]:
    question_banner("A partir de quel niveau de VRE un marche bascule-t-il en Phase 2 ?")

    # Reponse dynamique basee sur les donnees
    transitions_12 = []
    for country in selected_countries:
        series = _get_metrics_series(country)
        for i in range(1, len(series)):
            prev = series[i-1].get("_diag", {}).get("phase_number", 1)
            curr = series[i].get("_diag", {}).get("phase_number", 1)
            if prev == 1 and curr >= 2:
                transitions_12.append((country, series[i]["year"], series[i].get("vre_share", 0)))

    if transitions_12:
        vre_vals = [t[2] for t in transitions_12]
        details = ", ".join(f"{c} en {y} a {v:.1%} VRE" for c, y, v in transitions_12)
        dynamic_narrative(
            f"<strong>Reponse basee sur les donnees chargees :</strong> "
            f"{len(transitions_12)} transition(s) Phase 1→2 observee(s) : {details}. "
            f"Seuil VRE moyen a la transition : <strong>{np.mean(vre_vals):.1%}</strong>. "
            f"Ce seuil varie selon le niveau de flexibilite et d'inflexibilite du systeme.",
            "info")
    else:
        dynamic_narrative(
            "Aucune transition Phase 1→2 observee dans les donnees chargees. "
            "Cela peut signifier que tous les pays sont deja en Phase 2+ sur la periode analysee, "
            "ou que les donnees ne couvrent pas la periode de transition.",
            "warning")

    if not all_df.empty:
        fig_q1 = px.scatter(
            all_df,
            x="vre_share",
            y="h_negative",
            color="country",
            symbol="country",
            color_discrete_map=COUNTRY_PALETTE,
            hover_data=["year", "phase_number"],
            labels={
                "vre_share": "Part VRE (% generation)",
                "h_negative": "Heures a prix negatif",
            },
            title="Heures negatives vs Part VRE -- tous pays/annees",
            height=520,
        )
        # Annoter les transitions (phase change d'une annee a l'autre)
        for country in selected_countries:
            series = _get_metrics_series(country)
            for i in range(1, len(series)):
                prev_phase = series[i - 1].get("_diag", {}).get("phase_number", 1)
                curr_phase = series[i].get("_diag", {}).get("phase_number", 1)
                if prev_phase == 1 and curr_phase >= 2:
                    fig_q1.add_annotation(
                        x=series[i]["vre_share"],
                        y=series[i]["h_negative"],
                        text=f"{country} {series[i]['year']}",
                        showarrow=True, arrowhead=2,
                        font=dict(size=10, color="red"),
                    )

        fig_q1.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_q1, use_container_width=True)

        # Tableau sous-jacent
        with st.expander("Donnees sous-jacentes"):
            st.dataframe(
                all_df[["country", "year", "vre_share", "h_negative", "phase_number"]]
                .sort_values(["country", "year"]),
                use_container_width=True,
            )

# ── Q2 : Slope de degradation en Phase 2 ─────────────────────────────────
with tabs[1]:
    question_banner("Quelle est la vitesse de degradation du capture ratio en Phase 2 ?")
    st.markdown(
        "**Reponse synthetique** : La pente capture_ratio_pv vs pv_share est negative "
        "pour tous les marches en Phase 2. Plus la pente est raide, plus le marche "
        "cannibalise rapidement la valeur solaire."
    )

    slopes_data = []
    fig_q2 = go.Figure()

    for idx, country in enumerate(selected_countries):
        c_color = COUNTRY_PALETTE.get(country, "#999999")
        metrics_list = _get_metrics_series(country)
        slope_result = compute_slope(metrics_list, "pv_share", "capture_ratio_pv",
                                     exclude_outliers=exclude_2022)

        slopes_data.append({
            "Pays": country,
            "Slope": slope_result["slope"],
            "R2": slope_result["r_squared"],
            "p-value": slope_result["p_value"],
            "N points": slope_result["n_points"],
        })

        if slope_result["x_values"]:
            x_vals = np.array(slope_result["x_values"])
            y_vals = np.array(slope_result["y_values"])
            fig_q2.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers", name=country,
                marker=dict(color=c_color, size=10),
            ))
            # Ligne de regression
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            y_line = slope_result["slope"] * x_line + slope_result["intercept"]
            fig_q2.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines", name=f"{country} (reg.)",
                line=dict(color=c_color, dash="dash"),
                showlegend=False,
            ))

    fig_q2.update_layout(
        xaxis_title="Part PV (% generation)",
        yaxis_title="Capture Ratio PV",
        title="Regression : Capture Ratio PV vs Part PV par pays",
        height=500,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig_q2, use_container_width=True)

    df_slopes = pd.DataFrame(slopes_data)
    st.dataframe(df_slopes, use_container_width=True, hide_index=True)

    # Interpretation dynamique des pentes
    for _, row in df_slopes.iterrows():
        pays = row["Pays"]
        slope = row["Slope"]
        r2 = row["R2"]
        pval = row["p-value"]
        if slope is not None and slope == slope:  # not NaN
            if slope > 0:
                challenge_block(
                    f"Pente positive pour {pays}",
                    f"Le capture ratio augmente avec la part PV (slope={slope:.4f}). "
                    f"Cela contredit la theorie de cannibalisation. Causes possibles : "
                    f"trop peu de points, effet prix gaz dominant, ou marche encore en Stage 1.")
            elif pval is not None and pval == pval and pval > 0.05:
                challenge_block(
                    f"Regression non significative pour {pays}",
                    f"p-value = {pval:.3f} > 0.05 : la pente n'est pas statistiquement significative "
                    f"au seuil conventionnel de 5%. Les donnees sont insuffisantes pour conclure.")
            elif r2 is not None and r2 == r2 and r2 > 0.7:
                dynamic_narrative(
                    f"<strong>{pays}</strong> : relation forte (R2={r2:.2f}, p={pval:.3f}). "
                    f"La penetration PV explique {r2:.0%} de la variance du capture ratio.",
                    "success")

# ── Q3 : Transition Phase 2 -> 3 ─────────────────────────────────────────
with tabs[2]:
    question_banner("Quand un marche entre-t-il en Phase 3 (absorption structurelle) ?")
    st.markdown(
        "**Reponse synthetique** : La Phase 3 (absorption structurelle) est diagnostiquee "
        "quand **trois conditions** sont reunies :\n\n"
        "1. **Prerequis** : VRE >= 20% ET surplus ratio >= 0.5% (sinon le FAR est trivial)\n"
        "2. **Capacite d'absorption** : FAR domestique >= 0.60 (PSH + BESS + DSM, **hors exports**)\n"
        "3. **Amelioration inter-annuelle** : VRE en hausse ET h negatives en baisse\n\n"
        "Un FAR proche de 1.0 ne suffit **PAS** si le surplus est negligeable "
        "ou si les h negatives continuent d'augmenter. Le BESS et le PSH sont les principaux leviers."
    )

    if not all_df.empty:
        fig_q3 = px.scatter(
            all_df,
            x="far_structural",
            y="h_negative",
            color="country",
            symbol="country",
            color_discrete_map=COUNTRY_PALETTE,
            hover_data=["year", "vre_share"],
            labels={
                "far_structural": "FAR structural (domestique)",
                "h_negative": "Heures negatives",
            },
            title="Heures negatives vs FAR structural (domestique) -- evolution",
            height=500,
        )
        # Relier les points par pays dans l'ordre chronologique
        for country in selected_countries:
            series = _get_metrics_series(country)
            if len(series) < 2:
                continue
            x_vals = [m.get("far_structural", np.nan) for m in series]
            y_vals = [m.get("h_negative", 0) for m in series]
            fig_q3.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines", name=f"{country} (traj.)",
                line=dict(dash="dot", width=1),
                showlegend=False,
                opacity=0.5,
            ))
        # Ligne de reference FAR = 0.60 (seuil Stage 3)
        fig_q3.add_vline(x=0.60, line_dash="dash", line_color="gray", line_width=1,
                         annotation_text="FAR = 0.60", annotation_position="top left")

        # Annoter les points Phase 3
        for (c, y), d in diagnostics.items():
            if d.get("phase_number") == 3 and c in selected_countries:
                m = annual_metrics.get((c, y), {})
                far_val = m.get("far_structural", np.nan)
                h_neg_val = m.get("h_negative", 0)
                if far_val == far_val:  # not NaN
                    fig_q3.add_annotation(
                        x=far_val, y=h_neg_val,
                        text=f"{c} {y} (S3)",
                        showarrow=True, arrowhead=2, arrowcolor="#27AE60",
                        font=dict(size=9, color="#27AE60"),
                        ax=30, ay=-20,
                    )

        fig_q3.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_q3, use_container_width=True)

        # Narrative dynamique : lister les points Phase 3 et expliquer les absences
        s3_points = [(c, y) for (c, y), d in diagnostics.items()
                     if d.get("phase_number") == 3 and c in selected_countries]
        if s3_points:
            details = ", ".join(f"{c} {y}" for c, y in sorted(s3_points))
            dynamic_narrative(
                f"<strong>Points Phase 3 identifies :</strong> {details}. "
                f"Ces annees combinent VRE >= 20%, surplus significatif (SR >= 0.5%), "
                f"FAR domestique >= 0.60, et amelioration inter-annuelle.",
                "success")
        else:
            dynamic_narrative(
                "Aucun point Phase 3 dans les donnees chargees. Tous les marches "
                "sont en Phase 1 ou 2 sur la periode analysee.",
                "info")

        # Expliquer pourquoi certains pays a FAR ~1.0 ne sont pas S3
        high_far_not_s3 = []
        for c in selected_countries:
            latest_y = max([y for (cc, y) in annual_metrics if cc == c], default=None)
            if latest_y:
                m = annual_metrics.get((c, latest_y), {})
                d = diagnostics.get((c, latest_y), {})
                far_val = m.get("far_structural", 0)
                if far_val and far_val > 0.90 and d.get("phase_number", 1) != 3:
                    high_far_not_s3.append((c, latest_y, far_val, m.get("sr", 0), m.get("h_negative", 0)))
        if high_far_not_s3:
            lines = []
            for c, y, far_v, sr_v, h_neg in high_far_not_s3:
                reason = "surplus ratio trop faible" if sr_v < 0.005 else "h negatives en hausse (pas d'amelioration inter-annuelle)"
                lines.append(f"{c} ({y}) : FAR={far_v:.2f} mais {reason}")
            dynamic_narrative(
                "<strong>Pourquoi certains pays a FAR eleve ne sont pas Phase 3 ?</strong><br>"
                + "<br>".join(lines),
                "info")

    # Slider : GW BESS pour atteindre Stage 3
    st.markdown("#### Estimation BESS pour atteindre Stage 3")
    bess_target_country = st.selectbox("Pays cible", selected_countries, key="q3_country")
    bess_slider_gw = st.slider("GW BESS supplementaires", 0.0, 30.0, 5.0, 0.5, key="q3_bess")

    latest_key = None
    for y in sorted({k[1] for k in processed_data.keys()}, reverse=True):
        if (bess_target_country, y) in processed_data:
            latest_key = (bess_target_country, y)
            break

    if latest_key:
        from src.data_loader import load_country_config
        try:
            cfg = load_country_config(bess_target_country)
            test_df = apply_scenario(
                processed_data[latest_key],
                {"delta_bess_power_gw": bess_slider_gw,
                 "delta_bess_energy_gwh": bess_slider_gw * 4},
                cfg, bess_target_country, latest_key[1], commodity_prices,
            )
            from src.metrics import compute_annual_metrics as cam
            test_metrics = cam(test_df, latest_key[1], bess_target_country)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("FAR structural (scenario)",
                          f"{test_metrics.get('far_structural', 0):.3f}")
            with col_b:
                st.metric("H negatives (scenario)",
                          f"{test_metrics.get('h_negative', 0)}")
            with col_c:
                st.metric("H regime A (scenario)",
                          f"{test_metrics.get('h_regime_a', 0)}")
        except Exception as e:
            st.error(f"Erreur simulation: {e}")

# ── Q4 : Impact des batteries ────────────────────────────────────────────
with tabs[3]:
    question_banner("Quel est l'impact marginal des batteries sur le FAR ?")
    st.markdown(
        "**Reponse synthetique** : Le FAR *domestique* (hors exports) augmente de facon concave "
        "avec la capacite BESS. Les premiers GW ont le plus d'impact, puis la courbe sature "
        "lorsque le surplus est totalement absorbe par les moyens domestiques (PSH + BESS + DSM)."
    )

    q4_country = st.selectbox("Pays", selected_countries, key="q4_country")

    # Trouver la derniere annee disponible
    q4_key = None
    for y in sorted({k[1] for k in processed_data.keys()}, reverse=True):
        if (q4_country, y) in processed_data:
            q4_key = (q4_country, y)
            break

    if q4_key:
        bess_range = np.arange(0, 32, 2)
        far_values = []

        from src.data_loader import load_country_config as lcc
        try:
            cfg_q4 = lcc(q4_country)
        except Exception:
            cfg_q4 = None

        if cfg_q4:
            with st.spinner("Simulation BESS en cours..."):
                for bess_gw in bess_range:
                    sc_df = apply_scenario(
                        processed_data[q4_key],
                        {"delta_bess_power_gw": float(bess_gw),
                         "delta_bess_energy_gwh": float(bess_gw) * 4},
                        cfg_q4, q4_country, q4_key[1], commodity_prices,
                    )
                    from src.metrics import compute_annual_metrics as cam2
                    sc_m = cam2(sc_df, q4_key[1], q4_country)
                    far_values.append(sc_m.get("far_structural", np.nan))

            fig_q4 = go.Figure()
            fig_q4.add_trace(go.Scatter(
                x=bess_range.tolist(), y=far_values,
                mode="lines+markers",
                line=dict(color="#636EFA", width=2),
                marker=dict(size=8),
            ))
            fig_q4.update_layout(
                xaxis_title="BESS supplementaire (GW)",
                yaxis_title="FAR structural",
                title=f"FAR vs capacite BESS -- {q4_country} ({q4_key[1]})",
                height=450,
                **PLOTLY_LAYOUT_DEFAULTS,
            )
            st.plotly_chart(fig_q4, use_container_width=True)

            # Point de saturation
            if len(far_values) >= 3:
                increments = [far_values[i+1] - far_values[i] for i in range(len(far_values)-1)
                              if far_values[i] == far_values[i] and far_values[i+1] == far_values[i+1]]
                for idx, delta in enumerate(increments):
                    if delta < 0.005 and idx > 0:
                        sat_gw = bess_range[idx + 1]
                        dynamic_narrative(
                            f"Le rendement marginal du BESS sature apres environ "
                            f"<strong>{sat_gw:.0f} GW</strong> supplementaires. Au-dela, chaque GW "
                            f"n'apporte que {delta:.4f} de FAR.",
                            "info")
                        break

            # Tableau
            with st.expander("Donnees sous-jacentes"):
                st.dataframe(
                    pd.DataFrame({"BESS_GW": bess_range, "FAR_structural": far_values}),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.error(f"Config introuvable pour {q4_country}.")
    else:
        st.info(f"Pas de donnees pour {q4_country}.")

# ── Q5 : Sensibilite CO2 / Gaz ───────────────────────────────────────────
with tabs[4]:
    question_banner("Comment le TCA evolue-t-il en fonction du prix du gaz et du CO2 ?")
    st.markdown(
        "**Reponse synthetique** : Le TCA (CCGT) suit la formule lineaire "
        "`gaz/eta + EF_gas/eta * CO2 + VOM`. Le gaz a un levier ~1.75x, "
        "le CO2 ~0.35x. La heatmap ci-dessous montre les iso-TCA."
    )

    # Grille de prix
    gas_range = np.arange(10, 82, 2)
    co2_range = np.arange(20, 205, 5)
    gas_grid, co2_grid = np.meshgrid(gas_range, co2_range)

    tca_grid = gas_grid / ETA_CCGT + (EF_GAS / ETA_CCGT) * co2_grid + VOM_CCGT

    fig_q5 = go.Figure(data=go.Heatmap(
        z=tca_grid,
        x=gas_range,
        y=co2_range,
        colorscale="YlOrRd",
        colorbar=dict(title="TCA<br>(EUR/MWh)"),
        hovertemplate="Gaz: %{x} EUR/MWh<br>CO2: %{y} EUR/t<br>TCA: %{z:.1f} EUR/MWh<extra></extra>",
    ))
    fig_q5.update_layout(
        xaxis_title="Prix Gaz (EUR/MWh_th)",
        yaxis_title="Prix CO2 (EUR/tCO2)",
        title="TCA CCGT = f(Gaz, CO2)",
        height=550,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig_q5, use_container_width=True)

    # TCA actuel observe
    if st.session_state.get("annual_metrics"):
        for c in selected_countries:
            latest_y = max([y for (cc, y) in annual_metrics if cc == c], default=None)
            if latest_y:
                m = annual_metrics.get((c, latest_y), {})
                tca_med = m.get("tca_median")
                if tca_med and tca_med == tca_med:
                    dynamic_narrative(
                        f"<strong>{c} ({latest_y})</strong> : TCA median observe = "
                        f"<strong>{tca_med:.1f} EUR/MWh</strong>.",
                        "info")
                    break  # Only show first country

    # Formule explicite
    st.latex(
        r"TCA_{CCGT} = \frac{P_{gaz}}{\eta_{CCGT}} "
        r"+ \frac{EF_{gaz}}{\eta_{CCGT}} \times P_{CO_2} + VOM_{CCGT}"
    )
    st.markdown(
        f"Avec : eta_CCGT = {ETA_CCGT}, EF_gaz = {EF_GAS} tCO2/MWh_th, VOM = {VOM_CCGT} EUR/MWh"
    )

    # Tableau numerique pour quelques combos
    with st.expander("Tableau numerique (extraits)"):
        sample_gas = [15, 25, 35, 50, 70]
        sample_co2 = [30, 50, 80, 120, 180]
        rows_q5 = []
        for g in sample_gas:
            for c in sample_co2:
                tca = g / ETA_CCGT + (EF_GAS / ETA_CCGT) * c + VOM_CCGT
                rows_q5.append({"Gaz (EUR/MWh)": g, "CO2 (EUR/t)": c,
                                "TCA CCGT (EUR/MWh)": round(tca, 1)})
        st.dataframe(pd.DataFrame(rows_q5), use_container_width=True, hide_index=True)

# ── Q6 : Stockage thermique ──────────────────────────────────────────────
with tabs[5]:
    question_banner("Le stockage thermique peut-il jouer un role dans la transition ?")
    st.markdown(
        "**Reponse synthetique** : Le stockage thermique (sels fondus, Carnot batteries, stockage "
        "par chaleur latente) est un complement au BESS pour le stockage longue duree (6-24h+). "
        "Son impact sur les metriques du modele serait similaire a celui du BESS mais avec un "
        "rendement plus faible (40-60% round-trip vs 88% pour le Li-ion)."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        info_card("Sels fondus (CSP)", "Temperature : 290-565°C. Duree : 6h a plusieurs jours. Technologie eprouvee dans le solaire thermodynamique.")
    with col2:
        info_card("Batteries Carnot", "Rendement : 40-60%. Principe : electricite → chaleur → electricite. Stockage longue duree potentiel.")
    with col3:
        info_card("Power-to-Heat-to-Power", "Cycles thermochimiques ou mecaniques. Encore experimental. Cout en baisse rapide.")

    st.markdown("""
---

#### Principes

- **Stockage par sels fondus** : Technologie eprouvee dans le CSP (Concentrated Solar Power).
  Temperatures de 290-565 degres C. Duree de stockage de 6h a plusieurs jours.

- **Carnot batteries** : Conversion electricite -> chaleur -> electricite. Rendement round-trip
  de 40-60%. Cout capital potentiellement inferieur au Li-ion pour des durees > 8h.

- **Power-to-Heat-to-Power** : Resistance electrique + stockage thermique + turbine ORC ou vapeur.

#### Pertinence pour le modele Capture Prices

1. **Absorption du surplus (Regime A)** : Le stockage thermique pourrait absorber le surplus
   VRE de facon similaire au BESS, mais avec un cout de cycling plus eleve et un rendement
   moindre.

2. **Flexibilite longue duree** : Contrairement au BESS (typiquement 2-4h), le stockage
   thermique peut stocker sur 12-72h, couvrant ainsi des periodes de faible VRE plus longues.

3. **Impact sur FAR** : L'ajout de stockage thermique augmenterait le FAR structural mais
   avec un effet marginal moindre que le BESS par MWh installe (rendement inferieur).

4. **Limites de la modelisation actuelle** : Le modele simplifie le SoC et ne differencie pas
   les technologies de stockage. Pour integrer le stockage thermique, il faudrait :
   - Ajouter un parametre de rendement round-trip specifique (~0.50)
   - Augmenter la duree de stockage (ratio energie/puissance > 8)
   - Modeliser la rampe de demarrage (inertie thermique)

#### Conclusion

Le stockage thermique est un levier sous-represente dans les analyses actuelles. Son
inclusion dans le modele necesserait une extension du `scenario_engine` avec un deuxieme
vecteur de stockage a rendement et duree parametrables.
""")
