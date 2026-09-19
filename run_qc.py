from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation_qc.label_quality import inspect_labels
from annotation_qc.validator import load_annotations, validate_dataframe


def _load_allowed_labels(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("Allowed-label file does not contain any labels.")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an image-annotation CSV dataset.")
    parser.add_argument("dataset", type=Path, help="Path to an annotation CSV file")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON QC report")
    parser.add_argument("--image-root", type=Path, help="Optional root directory used to verify real image dimensions")
    parser.add_argument("--allowed-labels", type=Path, help="Optional newline-delimited approved label taxonomy")
    args = parser.parse_args()

    try:
        df = load_annotations(args.dataset)
        result = validate_dataframe(df, source=str(args.dataset), image_root=args.image_root)
        allowed_labels = _load_allowed_labels(args.allowed_labels)
        label_issues, label_summary = inspect_labels(df, allowed_labels=allowed_labels)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    result.issues.extend(label_issues)
    report = result.summary()
    report["label_quality"] = label_summary.to_dict()

    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
