from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation_qc.validator import validate_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an image-annotation CSV dataset.")
    parser.add_argument("dataset", type=Path, help="Path to an annotation CSV file")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON QC report")
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Optional root directory used to verify real image dimensions",
    )
    args = parser.parse_args()

    try:
        result = validate_csv(args.dataset, image_root=args.image_root)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    report = result.summary()
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
