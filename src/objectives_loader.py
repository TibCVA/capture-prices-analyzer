"""Load and structure project objectives from a DOCX synthesis file."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


_WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ").strip())


def _extract_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        data = archive.read("word/document.xml")
    root = ET.fromstring(data)

    paragraphs: list[str] = []
    for para in root.findall(".//w:p", _WORD_NS):
        texts = [node.text for node in para.findall(".//w:t", _WORD_NS) if node.text]
        if not texts:
            continue
        value = _clean_text("".join(texts))
        if value:
            paragraphs.append(value)
    return paragraphs


def _extract_question_lines(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^(Q[1-6])\s*[—:-]\s*(.+)$", line)
        if match:
            out[match.group(1)] = match.group(2)
    return out


def _extract_objective_lines(lines: list[str]) -> list[str]:
    keep: list[str] = []
    for line in lines:
        lower = line.lower()
        if "objectif" in lower or "demande" in lower or "approche" in lower:
            keep.append(line)
    return keep


def load_objectives_docx(path: str) -> dict:
    """Read a DOCX and return structured objectives/questions metadata."""

    docx_path = Path(path)
    if not docx_path.exists():
        raise FileNotFoundError(f"Fichier DOCX introuvable: {docx_path}")

    paragraphs = _extract_docx_paragraphs(docx_path)
    questions = _extract_question_lines(paragraphs)
    objectives = _extract_objective_lines(paragraphs)

    return {
        "path": str(docx_path),
        "paragraphs": paragraphs,
        "line_count": len(paragraphs),
        "questions": questions,
        "objective_lines": objectives,
    }

