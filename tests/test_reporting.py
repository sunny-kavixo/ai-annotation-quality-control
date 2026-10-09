import csv

import pandas as pd
import pytest

from annotation_qc.reporting import ISSUE_REPORT_COLUMNS, build_issue_report
from annotation_qc.validator import QualityIssue, REQUIRED_COLUMNS, validate_dataframe


def annotation_frame(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_report_maps_validation_issues_to_source_annotations():
    df = annotation_frame(
        [
            ["img_1", "images/1.jpg", "", 10, 20, 5, 120],
            ["img_2", "images/2.jpg", "car", 10, 20, 100, 120],
            ["img_2", "images/2.jpg", "car", 10, 20, 100, 120],
        ]
    )

    result = validate_dataframe(df)
    report = build_issue_report(df, result.issues)

    first_row = report[report["csv_row"] == 2]
    assert set(first_row["issue_code"]) == {"missing_label", "invalid_box_geometry"}
    assert set(first_row["image_id"]) == {"img_1"}
    assert set(first_row["label"]) == {""}

    duplicates = report[report["issue_code"] == "duplicate_annotation"]
    assert list(duplicates["csv_row"]) == [3, 4]
    assert set(duplicates["severity"]) == {"warning"}
    assert all(duplicates["reason"] == "This annotation row is duplicated exactly.")


def test_dataset_level_issue_has_blank_annotation_identity():
    df = pd.DataFrame([{"image_id": "img_1"}])
    result = validate_dataframe(df)

    report = build_issue_report(df, result.issues)

    assert report.to_dict("records") == [
        {
            "csv_row": "",
            "image_id": "",
            "image_path": "",
            "label": "",
            "severity": "error",
            "issue_code": "missing_columns",
            "reason": result.issues[0].message,
        }
    ]


def test_clean_dataset_produces_empty_report_with_headers(tmp_path):
    df = annotation_frame(
        [["img_1", "images/1.jpg", "car", 10, 20, 100, 120]]
    )
    result = validate_dataframe(df)
    report = build_issue_report(df, result.issues)
    output = tmp_path / "issues.csv"

    report.to_csv(output, index=False)

    assert report.empty
    assert list(report.columns) == list(ISSUE_REPORT_COLUMNS)
    assert output.read_text(encoding="utf-8") == ",".join(ISSUE_REPORT_COLUMNS) + "\n"


def test_csv_export_quotes_commas_quotes_and_embedded_newlines(tmp_path):
    label = 'vehicle, "large"\nneeds review'
    reason = 'Expected "car, truck".\nUse the approved taxonomy.'
    df = annotation_frame(
        [["img_1", "images/1.jpg", label, 10, 20, 100, 120]]
    )
    issue = QualityIssue(2, "example_issue", reason)
    output = tmp_path / "issues.csv"

    build_issue_report(df, [issue]).to_csv(output, index=False)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["label"] == label
    assert rows[0]["reason"] == reason


@pytest.mark.parametrize(
    "dangerous",
    [
        "=SUM(1, 1)",
        "+SUM(1, 1)",
        "-2+3",
        "@SUM(A1:A2)",
        "  =HYPERLINK(\"https://example.invalid\", \"click\")",
        "\t+SUM(1, 1)",
        "\r\n-2+3",
        "\x0b@SUM(A1:A2)",
        "\ufeff=SUM(1, 1)",
    ],
)
def test_formula_like_text_is_escaped_without_changing_source(dangerous):
    df = annotation_frame(
        [[dangerous, dangerous, dangerous, 10, 20, 100, 120]]
    )
    issue = QualityIssue(2, "example_issue", dangerous)

    report = build_issue_report(df, [issue])

    assert report.loc[0, "csv_row"] == 2
    for column in ("image_id", "image_path", "label", "reason"):
        assert report.loc[0, column] == "'" + dangerous
    assert df.loc[0, "image_id"] == dangerous
    assert df.loc[0, "image_path"] == dangerous
    assert df.loc[0, "label"] == dangerous


@pytest.mark.parametrize(
    "safe",
    [
        "car",
        "42",
        "text = formula",
        "'=already text",
        "  ordinary text",
        'vehicle, "large"\nneeds review',
        "",
    ],
)
def test_ordinary_text_is_not_changed(safe):
    df = annotation_frame(
        [[safe, safe, safe, 10, 20, 100, 120]]
    )
    issue = QualityIssue(2, "example_issue", safe)

    report = build_issue_report(df, [issue])

    for column in ("image_id", "image_path", "label", "reason"):
        assert report.loc[0, column] == safe


def test_escaped_formula_with_csv_special_characters_round_trips(tmp_path):
    dangerous = '\t=HYPERLINK("https://example.invalid", "click, me")\nnext line'
    df = annotation_frame(
        [["img_1", "images/1.jpg", dangerous, 10, 20, 100, 120]]
    )
    issue = QualityIssue(2, "example_issue", dangerous)
    output = tmp_path / "issues.csv"

    build_issue_report(df, [issue]).to_csv(output, index=False)

    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["label"] == "'" + dangerous
    assert row["reason"] == "'" + dangerous
