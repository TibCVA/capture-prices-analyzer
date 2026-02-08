"""Export helpers for Excel and Google Sheets."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger("capture_prices.export_utils")


def export_to_excel(
    metrics_rows: list[dict],
    diagnostics_rows: list[dict],
    slopes_rows: list[dict],
    filepath: str,
) -> str:
    """Export to xlsx under data/exports and return path."""

    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame(diagnostics_rows).to_excel(writer, sheet_name="diagnostics", index=False)
        pd.DataFrame(slopes_rows).to_excel(writer, sheet_name="slopes", index=False)

    logger.info("Export Excel genere: %s", out)
    return str(out)


def export_to_gsheets(
    metrics_rows: list[dict],
    diagnostics_rows: list[dict],
    slopes_rows: list[dict],
    spreadsheet_name: str,
    creds_path: str | None = None,
) -> str | None:
    """Optional Google Sheets export, fail gracefully when creds are missing."""

    if creds_path is None:
        creds_path = os.getenv("GOOGLE_SHEETS_CREDS")

    if not creds_path or not Path(creds_path).exists():
        logger.warning("Google Sheets creds absentes; export ignore")
        return None

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.create(spreadsheet_name)

        ws1 = sh.sheet1
        ws1.update_title("metrics")
        df1 = pd.DataFrame(metrics_rows)
        ws1.update([df1.columns.tolist()] + df1.fillna("").astype(str).values.tolist())

        ws2 = sh.add_worksheet(title="diagnostics", rows=200, cols=50)
        df2 = pd.DataFrame(diagnostics_rows)
        ws2.update([df2.columns.tolist()] + df2.fillna("").astype(str).values.tolist())

        ws3 = sh.add_worksheet(title="slopes", rows=200, cols=50)
        df3 = pd.DataFrame(slopes_rows)
        ws3.update([df3.columns.tolist()] + df3.fillna("").astype(str).values.tolist())

        return sh.url
    except Exception as exc:  # graceful by design
        logger.error("Google Sheets export echoue: %s", exc)
        return None
