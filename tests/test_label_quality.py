import pandas as pd

from annotation_qc.label_quality import inspect_labels
from annotation_qc.validator import REQUIRED_COLUMNS


def frame(labels):
    rows = [
        [f"img_{i}", f"images/{i}.jpg", label, 10, 20, 100, 120]
        for i, label in enumerate(labels, start=1)
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_label_format_variants_are_reported_as_warning():
    issues, summary = inspect_labels(frame(["car", "Car", " car "]))
    assert summary.counts == {"car": 3}
    assert any(issue.code == "inconsistent_label_format" for issue in issues)


def test_unknown_label_fails_allowed_taxonomy_check():
    issues, _ = inspect_labels(frame(["car", "forklift"]), allowed_labels=["car", "person"])
    unknown = [issue for issue in issues if issue.code == "unknown_label"]
    assert len(unknown) == 1
    assert unknown[0].row == 3


def test_valid_labels_produce_distribution_without_errors():
    issues, summary = inspect_labels(frame(["car", "person", "car"]), allowed_labels=["car", "person"])
    assert not any(issue.severity == "error" for issue in issues)
    assert summary.counts == {"car": 2, "person": 1}
