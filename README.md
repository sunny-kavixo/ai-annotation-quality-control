# AI Annotation Quality Control

A work-in-progress quality-control tool for image annotation datasets.

## Current milestone: V0.1

The first prototype focuses on the ingestion layer: load a CSV annotation file, check its basic structure and bounding-box values, and return a machine-readable QC report.

### Try it

```bash
python -m pip install -r requirements.txt
python run_qc.py data/sample_annotations.csv
python run_qc.py data/messy_annotations.csv --output reports/messy_report.json
```

The clean sample should pass. The messy sample is intentionally expected to fail QC and print the issues it finds.

See [docs/v0.1.md](docs/v0.1.md) for the input contract, current checks, and known limitations.

## Product direction

Later milestones are intended to add image-aware checks and visual review only after the V0.1 data model has been exercised. The repository does not claim those features yet.
