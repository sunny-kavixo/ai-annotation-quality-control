from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

REQUIRED_COLUMNS = ("image_id", "image_path", "label", "x_min", "y_min", "x_max", "y_max")
OPTIONAL_IMAGE_SIZE_COLUMNS = ("image_width", "image_height")


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
        return {"source": self.source, "rows": self.rows, "passed": self.passed,
                "issue_count": len(self.issues), "issues": [i.to_dict() for i in self.issues]}


def load_annotations(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("CSV annotation files are currently supported.")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The annotation CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse annotation CSV: {exc}") from exc


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and number not in (float("inf"), float("-inf")) else None


def validate_dataframe(df: pd.DataFrame, source: str = "<dataframe>", image_root: str | Path | None = None) -> ValidationResult:
    issues: list[QualityIssue] = []
    root = Path(image_root) if image_root is not None else None
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        return ValidationResult(source, len(df), [
            QualityIssue(None, "missing_columns", f"Missing required columns: {', '.join(missing_columns)}")
        ])

    has_width = "image_width" in df.columns
    has_height = "image_height" in df.columns
    if has_width != has_height:
        issues.append(QualityIssue(
            None, "incomplete_image_dimensions",
            "image_width and image_height must be provided together to enable image-bound checks."
        ))

    duplicate_subset = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_IMAGE_SIZE_COLUMNS if c in df.columns]
    exact_duplicates = df.duplicated(subset=duplicate_subset, keep=False)
    for index in df.index[exact_duplicates]:
        issues.append(QualityIssue(int(index) + 2, "duplicate_annotation",
                                   "This annotation row is duplicated exactly.", "warning"))

    for index, row in df.iterrows():
        csv_row = int(index) + 2
        for column in ("image_id", "image_path", "label"):
            if pd.isna(row[column]) or not str(row[column]).strip():
                issues.append(QualityIssue(csv_row, f"missing_{column}", f"{column} is empty."))

        coordinates = {c: _finite_number(row[c]) for c in ("x_min", "y_min", "x_max", "y_max")}
        for column, value in coordinates.items():
            if value is None:
                issues.append(QualityIssue(csv_row, "invalid_coordinate", f"{column} must be a finite number."))

        if all(v is not None for v in coordinates.values()):
            x_min, y_min = coordinates["x_min"], coordinates["y_min"]
            x_max, y_max = coordinates["x_max"], coordinates["y_max"]
            if min(coordinates.values()) < 0:
                issues.append(QualityIssue(csv_row, "negative_coordinate",
                                           "Bounding-box coordinates cannot be negative."))
            if x_max <= x_min or y_max <= y_min:
                issues.append(QualityIssue(csv_row, "invalid_box_geometry",
                                           "Bounding box must have positive width and height."))

            if has_width and has_height:
                width = _finite_number(row["image_width"])
                height = _finite_number(row["image_height"])
                if width is None or height is None or width <= 0 or height <= 0:
                    issues.append(QualityIssue(csv_row, "invalid_image_dimensions",
                                               "image_width and image_height must be positive finite numbers."))
                elif x_max > width or y_max > height:
                    issues.append(QualityIssue(
                        csv_row, "box_outside_image",
                        f"Bounding box ends at ({x_max:g}, {y_max:g}) outside image size {width:g}x{height:g}."
                    ))

                if root is not None:
                    image_path = root / str(row["image_path"])
                    if not image_path.is_file():
                        issues.append(QualityIssue(csv_row, "image_file_missing",
                                                   f"Image file not found: {image_path}"))
                    else:
                        try:
                            with Image.open(image_path) as image:
                                actual_width, actual_height = image.size
                        except (UnidentifiedImageError, OSError):
                            issues.append(QualityIssue(csv_row, "image_file_unreadable",
                                                       f"Could not read image file: {image_path}"))
                        else:
                            if actual_width != width or actual_height != height:
                                issues.append(QualityIssue(
                                    csv_row, "image_dimension_mismatch",
                                    f"CSV declares {width:g}x{height:g}, but image is {actual_width}x{actual_height}."
                                ))

    return ValidationResult(source, len(df), issues)


def validate_csv(path: str | Path, image_root: str | Path | None = None) -> ValidationResult:
    path = Path(path)
    return validate_dataframe(load_annotations(path), source=str(path), image_root=image_root)
