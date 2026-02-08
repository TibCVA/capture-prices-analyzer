from __future__ import annotations

from src.commentary_engine import comment_kpi, so_what_block


def test_so_what_block_format() -> None:
    txt = so_what_block(
        title="Titre",
        purpose="Implication",
        observed={"a": 1.234, "b": 2},
        method_link="Methode",
        limits="Limites",
        n=12,
    )

    assert "Constat chiffre" in txt
    assert "So what" in txt
    assert "Lien methode" in txt
    assert "Limites/portee" in txt
    assert "n=12" in txt


def test_comment_kpi_contains_core_ratios() -> None:
    txt = comment_kpi({"sr": 0.1, "far": 0.2, "ir": 0.3, "ttl": 80, "capture_ratio_pv": 0.75})

    assert "SR" in txt
    assert "FAR" in txt
    assert "IR" in txt
    assert "TTL" in txt
