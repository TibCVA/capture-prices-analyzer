from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.commentary_consistency import build_page_claim_registry, validate_claims_against_baseline


def test_build_page_claim_registry_reads_python_pages(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "2_test.py").write_text(
        "def x():\n"
        "    txt='Correlation NRL / prix observe'\n"
        "    txt2='Coherence regime / prix observe'\n"
        "    return txt + txt2\n",
        encoding="utf-8",
    )

    reg = build_page_claim_registry(str(pages))

    assert "2_test.py" in reg
    assert reg["2_test.py"]["mentions_corr"] is True
    assert reg["2_test.py"]["mentions_coherence"] is True


def test_validate_claims_against_baseline_returns_checks() -> None:
    thresholds = {
        "phase_thresholds": {
            "stage_2": {"capture_ratio_pv_max": 0.8},
            "stage_3": {"far_min": 0.6},
        }
    }
    registry = {
        "2_test.py": {
            "so_what_count": 1,
            "render_commentary_count": 1,
            "has_d_tail": False,
            "mentions_corr": True,
            "mentions_coherence": True,
            "mentions_q4_plateau": False,
            "mentions_price_synth_disclaimer": False,
            "mentions_forbidden_export_approx": False,
            "numeric_literals": [0.8, 0.6],
        },
        "6_test.py": {
            "so_what_count": 1,
            "render_commentary_count": 1,
            "has_d_tail": False,
            "mentions_corr": False,
            "mentions_coherence": False,
            "mentions_q4_plateau": True,
            "mentions_price_synth_disclaimer": False,
            "mentions_forbidden_export_approx": False,
            "numeric_literals": [0.8, 0.6],
        },
    }

    out = pd.DataFrame(validate_claims_against_baseline(registry=registry, thresholds=thresholds))

    assert not out.empty
    assert {"page", "check", "status", "detail"}.issubset(out.columns)
    assert (out["check"] == "no_legacy_d_tail").any()
