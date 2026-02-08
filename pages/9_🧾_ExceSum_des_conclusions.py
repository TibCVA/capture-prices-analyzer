"""Page 9 - ExceSum statique (rapport final, sans moteur runtime)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.commentary_bridge import so_what_block
from src.ui_helpers import (
    challenge_block,
    dynamic_narrative,
    inject_global_css,
    narrative,
    render_commentary,
    render_kpi_banner,
    section_header,
)
from src.ui_theme import COUNTRY_PALETTE, PLOTLY_AXIS_DEFAULTS, PLOTLY_LAYOUT_DEFAULTS


st.set_page_config(page_title="ExceSum des conclusions", page_icon="🧾", layout="wide")
inject_global_css()

st.title("🧾 ExceSum des conclusions")
st.caption(
    "Version statique, figée une fois pour toutes sur la baseline: FR/DE/ES/PL/DK, 2015-2024, "
    "modes observed/observed/observed."
)

json_path = Path("docs") / "EXCESUM_STATIC_REPORT.json"
if not json_path.exists():
    st.error(
        "Fichier statique manquant: `docs/EXCESUM_STATIC_REPORT.json`. "
        "La page ExceSum est volontairement non-dynamique et ne recalcule rien."
    )
    st.stop()

payload = json.loads(json_path.read_text(encoding="utf-8"))
meta = payload["meta"]
global_medians = payload["global_medians"]

df_means = pd.DataFrame(payload["by_country_means"])
df_latest = pd.DataFrame(payload["latest_year"]).sort_values("country")
df_q1 = pd.DataFrame(payload["q1_country"]).sort_values("country")
df_q2 = pd.DataFrame(payload["q2_slopes"]).sort_values("slope")
df_q3 = pd.DataFrame(payload["q3_transition"]).sort_values("country")
df_q4 = pd.DataFrame(payload["q4_summary"]).sort_values("country")
df_q5 = pd.DataFrame(payload["q5_commodity"]).sort_values("country")
df_q6 = pd.DataFrame(payload["q6_scope"]).sort_values("country")
df_country = pd.DataFrame(payload["country_conclusions"]).sort_values("country")
df_annex = pd.DataFrame(payload["metrics_annex"]).sort_values(["country", "year"])
verification = pd.DataFrame(meta.get("verification", []))

narrative(
    "Ce document est un rapport final statique. Toutes les conclusions ci-dessous sont figées sur le même périmètre "
    "de données et sur les mêmes conventions méthodologiques. Aucun recalcul n'est exécuté à l'ouverture de la page."
)

cols = st.columns(4)
cols[0].metric("Couverture", f"{len(df_annex)} couples pays/année")
cols[1].metric("Période", f"{meta['baseline']['years'][0]}-{meta['baseline']['years'][1]}")
cols[2].metric("Pays", ", ".join(meta["baseline"]["countries"]))
cols[3].metric("Outlier exclu pour pentes", str(meta["baseline"]["exclude_outlier_for_slopes"]))

section_header("Résumé exécutif global")
st.markdown(
    f"""
**Constat global principal.**  
Le cadre physique (NRL -> surplus -> absorption -> régimes) est opérationnel sur 50 couples pays/année avec une complétude moyenne de haut niveau.

**Niveaux médians sur l'ensemble de la base (n=50).**
- SR médian = `{global_medians['sr']:.4f}`
- FAR médian = `{global_medians['far']:.4f}`
- IR médian = `{global_medians['ir']:.4f}`
- TTL médian = `{global_medians['ttl']:.2f} €/MWh`
- Capture ratio PV médian = `{global_medians['capture_ratio_pv']:.4f}`
- Heures négatives observées médianes = `{global_medians['h_negative_obs']:.0f}`
- Cohérence régime/prix médiane = `{global_medians['regime_coherence'] * 100.0:.1f}%`

**Lecture de synthèse.**
1. La dégradation du capture ratio PV avec la pénétration PV est présente dans tous les pays, avec intensité variable.
2. Tous les pays sont classés `stage_2` en dernière année disponible (2024), avec des confiances différentes.
3. Le signal Q4 est plat en baseline dans tous les pays, ce qui est cohérent avec un surplus non absorbé déjà nul en référence.
4. Les stress CO2 et gaz augmentent systématiquement le TTL synthétique, dans les deux cas sur tous les pays.
5. La question chaleur/froid (Q6) ne peut pas être tranchée causalement avec les données actuelles.
"""
)

render_commentary(
    so_what_block(
        title="Synthèse globale (fixée)",
        purpose="Donner une lecture consolidée rigoureuse avant le détail question par question.",
        observed={
            "n_couples": len(df_annex),
            "sr_median": global_medians["sr"],
            "far_median": global_medians["far"],
            "coherence_median_pct": global_medians["regime_coherence"] * 100.0,
        },
        method_link="Rapport figé sur baseline unique (5 pays x 10 ans), sans recalcul runtime.",
        limits="Conclusions valables pour ce périmètre précis uniquement; elles ne se substituent pas à une causalité expérimentale.",
        n=len(df_annex),
        decision_use="Fournir un socle commun de décision avant arbitrage pays par pays.",
    )
)

tabs = st.tabs(
    [
        "Méthode et vérifications",
        "Q1-Q2",
        "Q3-Q4",
        "Q5-Q6",
        "Conclusions pays",
        "Annexes",
    ]
)

with tabs[0]:
    section_header("Méthode figée et contrôles")
    st.markdown(
        f"""
**Protocole figé du rapport**
- Pays: `{", ".join(meta['baseline']['countries'])}`
- Période: `{meta['baseline']['years'][0]}-{meta['baseline']['years'][1]}`
- Modes: `{meta['baseline']['modes']}`
- Exclusion outlier pour pentes: `{meta['baseline']['exclude_outlier_for_slopes']}`

**Règle de gouvernance**
- Page statique: pas de recalcul.
- Tous les résultats proviennent d'une extraction unique et figée.
- Les chiffres affichés sont identiques d'une session à l'autre.
"""
    )
    if not verification.empty:
        st.dataframe(verification, use_container_width=True, hide_index=True)
        for _, row in verification.iterrows():
            status = str(row["status"]).upper()
            if status == "PASS":
                render_kpi_banner(str(row["check"]), status, str(row["detail"]), status="strong")
            elif status == "WARN":
                render_kpi_banner(str(row["check"]), status, str(row["detail"]), status="medium")
            else:
                render_kpi_banner(str(row["check"]), status, str(row["detail"]), status="weak")

    st.markdown("#### Moyennes par pays (2015-2024)")
    st.dataframe(df_means.round(4), use_container_width=True, hide_index=True)

with tabs[1]:
    section_header("Q1 - Seuils de bascule vers stage_2")
    st.markdown(
        """
**Réponse objective.**  
En dernière année (2024), `DE`, `DK`, `ES` et `FR` franchissent la combinaison de seuils stage_2; `PL` ne franchit pas l'ensemble des seuils.
"""
    )
    fig_q1 = px.scatter(
        pd.DataFrame(payload["q1_detail"]),
        x="sr",
        y="h_negative_obs",
        color="country",
        color_discrete_map=COUNTRY_PALETTE,
        hover_data=["year", "capture_ratio_pv", "cross_all"],
        title="Q1 - SR vs heures négatives observées (points annuels)",
    )
    fig_q1.update_layout(height=460, xaxis_title="SR", yaxis_title="h_negative_obs", **PLOTLY_LAYOUT_DEFAULTS)
    fig_q1.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q1.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q1, use_container_width=True)
    st.dataframe(df_q1, use_container_width=True, hide_index=True)

    section_header("Q2 - Pente de phase 2 (capture_ratio_pv ~ pénétration PV)")
    st.markdown(
        """
**Réponse objective.**  
Les pentes sont négatives pour les 5 pays. La pente la plus négative est observée en France (`-0.0708`), la moins négative en Espagne (`-0.0213`).  
La robustesse statistique est forte sur `FR/DE/DK/ES`; elle est fragile sur `PL` (n faible, p-value élevée).
"""
    )
    fig_q2 = px.bar(
        df_q2,
        x="country",
        y="slope",
        color="robustesse",
        hover_data=["r_squared", "p_value", "n_points"],
        title="Q2 - Pentes par pays (hors 2022)",
    )
    fig_q2.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
    fig_q2.update_layout(height=430, **PLOTLY_LAYOUT_DEFAULTS)
    fig_q2.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q2.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q2, use_container_width=True)
    st.dataframe(df_q2.round(6), use_container_width=True, hide_index=True)

with tabs[2]:
    section_header("Q3 - Conditions de passage stage_2 -> stage_3")
    st.markdown(
        """
**Réponse objective.**  
Tous les pays sont classés `transition_partielle`: FAR élevé mais pas de baisse observée robuste des heures négatives dans cette lecture annuelle.
"""
    )
    fig_q3 = px.scatter(
        df_q3,
        x="far_latest",
        y="h_negative_latest",
        color="country",
        color_discrete_map=COUNTRY_PALETTE,
        hover_data=["h_negative_slope_per_year", "h_regime_a_latest", "status_transition_2_to_3"],
        title="Q3 - FAR vs heures négatives (dernier point annuel)",
    )
    fig_q3.update_layout(height=430, **PLOTLY_LAYOUT_DEFAULTS)
    fig_q3.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q3.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q3, use_container_width=True)
    st.dataframe(df_q3.round(6), use_container_width=True, hide_index=True)

    section_header("Q4 - Batteries")
    st.markdown(
        """
**Réponse objective.**  
Le sweep baseline est plat dans les 5 pays (`plateau_baseline=True` partout), avec `surplus_unabs_twh_baseline=0`.  
Donc, sur la référence actuelle, il n'y a pas de surplus résiduel à arbitrer.  
Sous stress PV minimal, un effet marginal redevient identifiable dans les 5 pays (stress trouvé partout).
"""
    )
    st.dataframe(df_q4.round(6), use_container_width=True, hide_index=True)

    q4_ch = df_q4[["country", "stress_delta_pv_gw"]].copy().sort_values("stress_delta_pv_gw")
    fig_q4 = px.bar(
        q4_ch,
        x="country",
        y="stress_delta_pv_gw",
        color="country",
        color_discrete_map=COUNTRY_PALETTE,
        title="Q4 - Stress PV minimal pour rendre l'effet BESS identifiable",
    )
    fig_q4.update_layout(height=420, xaxis_title="Pays", yaxis_title="Delta PV (GW)", **PLOTLY_LAYOUT_DEFAULTS)
    fig_q4.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q4.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q4, use_container_width=True)
    challenge_block(
        "Interprétation méthodologique Q4",
        "Un plateau baseline n'est pas un bug de calcul. Ici, il est cohérent avec un surplus non absorbé nul sur la référence."
    )

with tabs[3]:
    section_header("Q5 - Impact CO2 et gaz sur TTL")
    st.markdown(
        """
**Réponse objective.**  
Les deux stress (`CO2↑`, `gaz↑`) augmentent le TTL synthétique dans tous les pays.  
Amplitude médiane observée: `+52.36 €/MWh` sous stress CO2 et `+50.97 €/MWh` sous stress gaz.
"""
    )
    q5m = df_q5.melt(
        id_vars=["country", "year"],
        value_vars=["delta_ttl_high_co2", "delta_ttl_high_gas"],
        var_name="scenario",
        value_name="delta_ttl",
    )
    fig_q5 = px.bar(
        q5m,
        x="country",
        y="delta_ttl",
        color="scenario",
        barmode="group",
        title="Q5 - Variation TTL par pays",
    )
    fig_q5.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
    fig_q5.update_layout(height=430, **PLOTLY_LAYOUT_DEFAULTS)
    fig_q5.update_xaxes(**PLOTLY_AXIS_DEFAULTS)
    fig_q5.update_yaxes(**PLOTLY_AXIS_DEFAULTS)
    st.plotly_chart(fig_q5, use_container_width=True)
    st.dataframe(df_q5.round(6), use_container_width=True, hide_index=True)

    section_header("Q6 - Stockage chaleur/froid")
    st.markdown(
        """
**Réponse objective et prudente.**  
Avec les données actuellement présentes dans l'outil, une conclusion causale robuste sur la synergie/compétition chaleur-froid n'est pas identifiable.  
Le statut est `non_identifiable_sans_donnees_dediees` dans les 5 pays.
"""
    )
    st.dataframe(df_q6, use_container_width=True, hide_index=True)
    dynamic_narrative(
        "Conclusion Q6: pas d'invention au-delà des données. La réponse reste volontairement prudente et méthodologiquement stricte.",
        severity="warning",
    )

with tabs[4]:
    section_header("Conclusions détaillées pays par pays")
    for _, row in df_country.iterrows():
        with st.expander(f"{row['country']} — conclusion détaillée", expanded=False):
            st.markdown(
                f"""
**Statut 2024**
- Phase: `{row['phase_latest']}` (année `{int(row['latest_year'])}`)
- SR: `{row['sr_latest']:.6f}`
- FAR: `{row['far_latest']:.6f}`
- Capture ratio PV: `{row['capture_ratio_pv_latest']:.6f}`

**Réponses Q1..Q6**
- Q1: première année de franchissement stage_2 = `{row['q1_first_stage2_year']}`
- Q2: pente = `{row['q2_slope']:.6f}`
- Q3: statut = `{row['q3_status']}`
- Q4: plateau baseline = `{bool(row['q4_plateau_baseline'])}`, stress trouvé = `{bool(row['q4_stress_found'])}`
- Q5: delta TTL CO2 = `{row['q5_delta_ttl_co2']:.6f}`, delta TTL gaz = `{row['q5_delta_ttl_gas']:.6f}`
- Q6: `{row['q6_status']}`
"""
            )
    st.markdown("#### Dernière année par pays")
    st.dataframe(df_latest.round(6), use_container_width=True, hide_index=True)

with tabs[5]:
    section_header("Annexes chiffrées exhaustives")
    st.markdown(
        "Toutes les séries annuelles utilisées pour le rapport sont listées ci-dessous. "
        "La table couvre les 5 pays, 10 années, et les dimensions clés de la méthode."
    )
    st.dataframe(df_annex.round(6), use_container_width=True, hide_index=True)

    corr = float(np.nanmedian(df_annex["regime_coherence"].to_numpy()))
    render_commentary(
        so_what_block(
            title="Traçabilité finale",
            purpose="Garantir la reproductibilité et l'auditabilité des conclusions statiques.",
            observed={
                "n_rows_annex": len(df_annex),
                "n_countries": df_annex["country"].nunique(),
                "n_years": df_annex["year"].nunique(),
                "median_regime_coherence_pct": corr * 100.0,
            },
            method_link="Table figée exportée après calcul unique; aucune mutation runtime.",
            limits="Ce rapport est une photographie méthodologiquement cohérente du périmètre actuel.",
            n=len(df_annex),
            decision_use="Utiliser cette base comme référence commune avant tout approfondissement ad hoc.",
        )
    )
