from __future__ import annotations

from src.commentary_engine import analysis_note, comment_kpi, so_what_block


def test_so_what_block_format() -> None:
    txt = so_what_block(
        title="Titre",
        purpose="Implication",
        observed={"a": 1.234, "b": 2},
        method_link="Methode",
        limits="Limites",
        n=12,
        decision_use="Decision",
    )

    assert "Constat chiffre" in txt
    assert "Ce que cela signifie" in txt
    assert "Pourquoi cette analyse sert a decider" in txt
    assert "Lien methode" in txt
    assert "Limites" in txt
    assert "n=12" in txt
    assert "Decision" in txt


def test_comment_kpi_contains_core_ratios() -> None:
    txt = comment_kpi({"sr": 0.1, "far": 0.2, "ir": 0.3, "ttl": 80, "capture_ratio_pv": 0.75})

    assert "SR" in txt
    assert "FAR" in txt
    assert "IR" in txt
    assert "TTL" in txt


def test_analysis_note_contains_structured_sections() -> None:
    txt = analysis_note(
        title="Bloc",
        objective="Objectif test",
        reading="Lecture test",
        findings="n=8; x=1.2",
        implication="Implication test",
        method="Methode test",
        limits="Limites test",
        n=8,
        confidence=0.6,
        decision_use="Decision test",
    )

    assert "Objectif de l'analyse" in txt
    assert "Comment lire le graphique" in txt
    assert "Constat chiffre" in txt
    assert "Ce que cela signifie" in txt
    assert "Pourquoi cette analyse sert a decider" in txt
    assert "Lien methode" in txt
    assert "Limites" in txt
    assert "n=8" in txt
