from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("image_id", "image_path", "label", "x_min", "y_min", "x_max", "y_max")


@dataclass(frozen=True)
class QualityIssue:
    row: int | None
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    source: str
    rows: int
    issues: list[QualityIssue]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "rows": self.rows,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_annotations(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("V0.1 currently accepts CSV annotation files only.")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The annotation CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse annotation CSV: {exc}") from exc


def validate_dataframe(df: pd.DataFrame, source: str = "<dataframe>") -> ValidationResult:
    issues: list[QualityIssue] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        issues.append(
            QualityIssue(
                row=None,
                code="missing_columns",
                message=f"Missing required columns: {', '.join(missing_columns)}",
            )
        )
        return ValidationResult(source=source, rows=len(df), issues=issues)

    exact_duplicates = df.duplicated(subset=list(REQUIRED_COLUMNS), keep=False)
    for index in df.index[exact_duplicates]:
        issues.append(
            QualityIssue(
                row=int(index) + 2,
                code="duplicate_annotation",
                message="This annotation row is duplicated exactly.",
                severity="warning",
            )
        )

    for index, row in df.iterrows():
        csv_row = int(index) + 2

        for column in ("image_id", "image_path", "label"):
            value = row[column]
            if pd.isna(value) or not str(value).strip():
                issues.append(
                    QualityIssue(
                        row=csv_row,
                        code=f"missing_{column}",
                        message=f"{column} is empty.",
                    )
                )

        coordinates: dict[str, float] = {}
        for column in ("x_min", "y_min", "x_max", "y_max"):
            try:
                value = float(row[column])
                if pd.isna(value) or not pd.notna(value) or value in (float("inf"), float("-inf")):
                    raise ValueError
                coordinates[column] = value
            except (TypeError, ValueError):
                issues.append(
                    QualityIssue(
                        row=csv_row,
                        code="invalid_coordinate",
                        message=f"{column} must be a finite number.",
                    )
                )

        if len(coordinates) == 4:
            if min(coordinates.values()) < 0:
                issues.append(
                    QualityIssue(
                        row=csv_row,
                        code="negative_coordinate",
                        message="Bounding-box coordinates cannot be negative.",
                    )
                )
            if coordinates["x_max"] <= coordinates["x_min"] or coordinates["y_max"] <= coordinates["y_min"]:
                issues.append(
                    QualityIssue(
                        row=csv_row,
                        code="invalid_box_geometry",
                        message="Bounding box must have positive width and height.",
                    )
                )

    return ValidationResult(source=source, rows=len(df), issues=issues)


def validate_csv(path: str | Path) -> ValidationResult:
    path = Path(path)
    return validate_dataframe(load_annotations(path), source=str(path))
