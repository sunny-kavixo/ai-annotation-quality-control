from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .validator import QualityIssue


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


@dataclass(frozen=True)
class LabelQualitySummary:
    counts: dict[str, int]
    normalized_variants: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        return {"counts": self.counts, "normalized_variants": self.normalized_variants}


def inspect_labels(df: pd.DataFrame, allowed_labels: Iterable[str] | None = None) -> tuple[list[QualityIssue], LabelQualitySummary]:
    issues: list[QualityIssue] = []
    raw_labels = ["" if pd.isna(v) else str(v) for v in df["label"]]
    normalized = [normalize_label(v) for v in raw_labels]

    variants: dict[str, set[str]] = {}
    for raw, canonical in zip(raw_labels, normalized):
        if canonical:
            variants.setdefault(canonical, set()).add(raw)

    for canonical, raw_variants in variants.items():
        cleaned = {v.strip() for v in raw_variants}
        if len(cleaned) > 1 or any(v != canonical for v in cleaned):
            rows = [int(i) + 2 for i, v in enumerate(normalized) if v == canonical]
            issues.append(QualityIssue(
                None,
                "inconsistent_label_format",
                f"Label '{canonical}' appears with inconsistent formatting at CSV rows {rows}: {sorted(raw_variants)}.",
                "warning",
            ))

    if allowed_labels is not None:
        allowed = {normalize_label(v) for v in allowed_labels}
        for index, canonical in enumerate(normalized):
            if canonical and canonical not in allowed:
                issues.append(QualityIssue(
                    index + 2,
                    "unknown_label",
                    f"Label '{raw_labels[index]}' is not in the allowed label set.",
                ))

    counts = dict(sorted(Counter(v for v in normalized if v).items()))
    return issues, LabelQualitySummary(
        counts=counts,
        normalized_variants={k: sorted(v) for k, v in sorted(variants.items())},
    )
