from __future__ import annotations

from collections.abc import Iterable
import unicodedata

import pandas as pd

from .validator import QualityIssue


ISSUE_REPORT_COLUMNS = (
    "csv_row",
    "image_id",
    "image_path",
    "label",
    "severity",
    "issue_code",
    "reason",
)
FORMULA_PREFIXES = frozenset("=+-@")


def _spreadsheet_safe(value: object) -> object:
    if not isinstance(value, str):
        return value

    first_significant = next(
        (
            character
            for character in value
            if not character.isspace()
            and unicodedata.category(character) not in {"Cc", "Cf"}
        ),
        "",
    )
    if first_significant in FORMULA_PREFIXES:
        return "'" + value
    return value


def _record_value(record: pd.Series | None, column: str) -> object:
    if record is None or column not in record.index:
        return ""
    value = record[column]
    return "" if pd.isna(value) else value


def build_issue_report(
    df: pd.DataFrame, issues: Iterable[QualityIssue]
) -> pd.DataFrame:
    """Build a flat, one-row-per-issue report from existing validation results."""
    rows = []
    for issue in issues:
        record = None
        if issue.row is not None:
            index = issue.row - 2
            if 0 <= index < len(df):
                record = df.iloc[index]

        rows.append(
            {
                "csv_row": "" if issue.row is None else issue.row,
                "image_id": _spreadsheet_safe(_record_value(record, "image_id")),
                "image_path": _spreadsheet_safe(_record_value(record, "image_path")),
                "label": _spreadsheet_safe(_record_value(record, "label")),
                "severity": _spreadsheet_safe(issue.severity),
                "issue_code": _spreadsheet_safe(issue.code),
                "reason": _spreadsheet_safe(issue.message),
            }
        )

    return pd.DataFrame(rows, columns=ISSUE_REPORT_COLUMNS)
