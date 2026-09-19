# AI Annotation Quality Control System

A working quality-control and human-review application for image annotation datasets. It validates annotation CSVs, checks bounding-box geometry and real image dimensions, inspects label consistency/taxonomy, exports machine-readable QC reports, and gives reviewers a visual dashboard for resolving failures.

## Product workflow

1. Run QC against an annotation CSV (optionally with real image files and an allowed-label taxonomy).
2. Open the Streamlit reviewer dashboard with the exported JSON report, source CSV, and an optional ZIP preserving image paths.
3. Filter QC failures and inspect the exact source row.
4. Render the source image with its bounding box and label.
5. Record **Approve**, **Reject**, or **Needs Fix** plus a reviewer comment.
6. Export reviewer decisions as JSON/CSV or a reviewed dataset CSV.

## Run locally

```bash
python -m pip install -r requirements.txt
python run_qc.py data/sample_annotations.csv --output reports/qc.json
streamlit run dashboard.py
```

For real-image verification, use the CLI's `--image-root` option. For taxonomy validation, provide `--allowed-labels`. Run `python run_qc.py --help` for the current CLI contract.

## Docker

```bash
docker build -t annotation-qc .
docker run --rm -p 8501:8501 annotation-qc
```

Then open the Streamlit application on port 8501.

## What the system checks

The project evolved through exercised failure cases rather than claiming universal annotation quality. Current checks cover structural CSV requirements, bounding-box validity, optional declared image bounds, optional real-image dimension verification, label normalization/format consistency, optional allowed-label taxonomy checks, safe ZIP image resolution, and reviewer/report row alignment.

Class imbalance is deliberately **not** automatically classified as an error: whether a rare class is problematic depends on project policy and dataset intent.

## Review artifacts

Reviewer state is explicit and exportable. Decisions are keyed to CSV data-row numbers and can be exported as `review_decisions.json`, `review_decisions.csv`, or joined back onto the uploaded annotation dataset as `reviewed_annotations.csv`. The application does not silently overwrite the source dataset.

## Quality gates

```bash
python -m pytest -q
```

GitHub Actions runs regression tests and CLI smoke checks on pull requests and `main`.

## Development history

The repository intentionally keeps the progression visible: ingestion prototype → malformed-label/duplicate assumption correction → image-boundary metadata → verification against real image dimensions → label/taxonomy QC → visual reviewer dashboard → reliable ZIP image resolution → reviewer decisions and exports.

See the `docs/` directory and pull-request history for the exercised assumptions, regressions, and milestone boundaries.
