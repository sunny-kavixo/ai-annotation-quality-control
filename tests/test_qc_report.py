import csv
import json
import subprocess
import sys


def test_cli_report_includes_dataset_label_quality(tmp_path):
    report_path = tmp_path / "report.json"
    issues_path = tmp_path / "annotation_issues.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "run_qc.py",
            "data/label_quality_exercise.csv",
            "--allowed-labels",
            "config/allowed_labels.txt",
            "--output",
            str(report_path),
            "--issues-csv",
            str(issues_path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["label_quality"]["counts"]["car"] == 3
    assert report["label_quality"]["counts"]["forklift"] == 1
    assert any(issue["code"] == "inconsistent_label_format" for issue in report["issues"])
    assert any(
        issue["code"] == "unknown_label" and issue["row"] == 5
        for issue in report["issues"]
    )

    with issues_path.open(encoding="utf-8", newline="") as handle:
        issues = list(csv.DictReader(handle))
    assert any(
        issue["issue_code"] == "unknown_label" and issue["csv_row"] == "5"
        for issue in issues
    )


def test_clean_cli_run_writes_empty_issue_report_and_returns_zero(tmp_path):
    report_path = tmp_path / "report.json"
    issues_path = tmp_path / "annotation_issues.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "run_qc.py",
            "data/sample_annotations.csv",
            "--output",
            str(report_path),
            "--issues-csv",
            str(issues_path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["issues"] == []
    assert issues_path.read_text(encoding="utf-8") == (
        "csv_row,image_id,image_path,label,severity,issue_code,reason\n"
    )
