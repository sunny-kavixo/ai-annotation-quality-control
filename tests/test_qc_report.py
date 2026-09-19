import json
import subprocess
import sys


def test_cli_report_includes_dataset_label_quality(tmp_path):
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "run_qc.py",
            "data/label_quality_exercise.csv",
            "--allowed-labels",
            "config/allowed_labels.txt",
            "--output",
            str(report_path),
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
