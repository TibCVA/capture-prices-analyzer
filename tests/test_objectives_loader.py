from __future__ import annotations

import zipfile
from pathlib import Path

from src.objectives_loader import load_objectives_docx


def _build_minimal_docx(path: Path) -> None:
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Objectif principal du projet</w:t></w:r></w:p>
    <w:p><w:r><w:t>Q1 — Seuil de bascule</w:t></w:r></w:p>
    <w:p><w:r><w:t>Approche outside-in rigoureuse</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml)


def test_load_objectives_docx_extracts_questions_and_objectives(tmp_path: Path) -> None:
    docx_path = tmp_path / "synthese.docx"
    _build_minimal_docx(docx_path)

    out = load_objectives_docx(str(docx_path))

    assert out["line_count"] == 3
    assert out["questions"]["Q1"] == "Seuil de bascule"
    assert any("Objectif principal" in line for line in out["objective_lines"])
    assert any("Approche outside-in" in line for line in out["objective_lines"])
