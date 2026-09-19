# Complete product milestone

This milestone closes the prototype loop by adding an explicit human-review outcome to QC findings.

## Exercise that drove the design

The V0.4 dashboard could show a failed annotation, but a reviewer could not record what should happen next. Keeping those decisions in an external spreadsheet would split the audit trail and make row alignment easy to lose.

Reviewer decisions are therefore modeled separately from source annotations and keyed by CSV data-row number. Saving a later decision for the same row replaces the earlier decision rather than creating contradictory states.

## Outputs

- JSON decision artifact with schema version and audit fields.
- CSV decision artifact for operational handoff.
- Reviewed annotation CSV containing decision/comment columns.
- Source data remains unchanged unless the user explicitly downloads the derived reviewed dataset.

## Boundaries

This is not an annotation editor and does not rewrite bounding boxes or labels in place. It is a QC + review product. Editing/correction can be built later against the explicit review decisions rather than hidden mutation.

The dashboard uses Streamlit session state for an interactive review session; durable multi-user storage/authentication is intentionally outside this local product milestone.
