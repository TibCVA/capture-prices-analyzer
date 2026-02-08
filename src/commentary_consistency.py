"""Cross-page commentary consistency checks against ExceSum baseline."""

from __future__ import annotations

import re
from pathlib import Path

from src.config_loader import load_thresholds


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def build_page_claim_registry(pages_dir: str = "pages") -> dict:
    """Extract lightweight claim signals from page source files."""

    registry: dict[str, dict] = {}
    for path in sorted(Path(pages_dir).glob("*.py")):
        text = _read_text(path)
        numeric = [float(x) for x in re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w.])", text)]

        registry[path.name] = {
            "path": str(path),
            "so_what_count": text.count("so_what_block("),
            "render_commentary_count": text.count("render_commentary("),
            "has_d_tail": "D_tail" in text,
            "mentions_corr": "Correlation NRL / prix observe" in text,
            "mentions_coherence": "Coherence regime / prix observe" in text,
            "mentions_q4_plateau": "Resultat plat physiquement normal" in text,
            "mentions_price_synth_disclaimer": "prix synthetique" in text.lower(),
            "mentions_forbidden_export_approx": "generation - load" in text,
            "numeric_literals": numeric,
        }
    return registry


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "PASS"
    if warn:
        return "WARN"
    return "FAIL"


def validate_claims_against_baseline(
    registry: dict | None = None,
    thresholds: dict | None = None,
) -> list[dict]:
    """Validate key UI claims against expected methodological anchors."""

    reg = registry or build_page_claim_registry()
    thr = thresholds or load_thresholds()
    out: list[dict] = []

    stage2_capture = float(thr["phase_thresholds"]["stage_2"]["capture_ratio_pv_max"])
    stage3_far = float(thr["phase_thresholds"]["stage_3"]["far_min"])

    for name, meta in sorted(reg.items()):
        if name.startswith(("0_", "1_", "2_", "3_", "4_", "5_", "6_", "8_")):
            so_what_ok = int(meta["so_what_count"]) >= 1
            out.append(
                {
                    "page": name,
                    "check": "commentary_presence",
                    "status": _status(so_what_ok),
                    "detail": f"so_what_block={meta['so_what_count']}",
                }
            )

        out.append(
            {
                "page": name,
                "check": "no_legacy_d_tail",
                "status": _status(not bool(meta["has_d_tail"])),
                "detail": "legacy regime token D_tail absent",
            }
        )

    page2 = next((k for k in reg if k.startswith("2_")), None)
    if page2:
        corr_ok = bool(reg[page2]["mentions_corr"]) and bool(reg[page2]["mentions_coherence"])
        out.append(
            {
                "page": page2,
                "check": "nrl_price_validation_dual_signal",
                "status": _status(corr_ok, warn=True),
                "detail": "correlation+coherence visible",
            }
        )

    page6 = next((k for k in reg if k.startswith("6_")), None)
    if page6:
        plateau_ok = bool(reg[page6]["mentions_q4_plateau"])
        out.append(
            {
                "page": page6,
                "check": "q4_plateau_explanation",
                "status": _status(plateau_ok, warn=True),
                "detail": "explicit plateau explanation",
            }
        )

        literals = reg[page6]["numeric_literals"]
        has_stage2_capture_literal = any(abs(v - stage2_capture) < 1e-6 for v in literals)
        has_stage3_far_literal = any(abs(v - stage3_far) < 1e-6 for v in literals)
        out.append(
            {
                "page": page6,
                "check": "threshold_literal_stage2_capture",
                "status": _status(has_stage2_capture_literal, warn=True),
                "detail": f"capture_ratio_pv_max={stage2_capture:.2f}",
            }
        )
        out.append(
            {
                "page": page6,
                "check": "threshold_literal_stage3_far",
                "status": _status(has_stage3_far_literal, warn=True),
                "detail": f"far_min={stage3_far:.2f}",
            }
        )

    page5 = next((k for k in reg if k.startswith("5_")), None)
    if page5:
        out.append(
            {
                "page": page5,
                "check": "scenario_price_disclaimer",
                "status": _status(bool(reg[page5]["mentions_price_synth_disclaimer"]), warn=True),
                "detail": "prix synthetique disclaimer present",
            }
        )

    for name, meta in sorted(reg.items()):
        if meta["mentions_forbidden_export_approx"]:
            # Mention is acceptable if stated as forbidden. We mark WARN to force review.
            out.append(
                {
                    "page": name,
                    "check": "forbidden_export_formula_mention_review",
                    "status": "WARN",
                    "detail": "contains 'generation - load' text, verify wording is prohibition only",
                }
            )

    return out

