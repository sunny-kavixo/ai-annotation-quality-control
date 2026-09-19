from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from annotation_qc.review import filter_issues, load_image_from_zip, record_for_csv_row, render_annotation
from annotation_qc.decisions import ReviewDecision, apply_decisions_to_dataset, decisions_csv, decisions_json, upsert_decision

st.set_page_config(page_title="Annotation QC Review", layout="wide")
st.title("AI Annotation Quality Control")
st.caption("Reviewer workspace — inspect QC failures, record decisions, and export review artifacts.")
if "decisions" not in st.session_state:
    st.session_state.decisions = []

report_file = st.file_uploader("QC report (JSON)", type=["json"])
dataset_file = st.file_uploader("Annotation dataset (CSV)", type=["csv"])
image_bundle = st.file_uploader("Dataset images (ZIP, optional)", type=["zip"])

if not report_file or not dataset_file:
    st.info("Upload a QC JSON report and its annotation CSV to begin review.")
    st.stop()

try:
    report = json.load(report_file)
    df = pd.read_csv(dataset_file)
except (ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load review artifacts: {exc}")
    st.stop()

raw_issues = report.get("issues", [])
c1, c2, c3 = st.columns(3)
c1.metric("Annotations", len(df))
c2.metric("QC issues", len(raw_issues))
c3.metric("Status", "PASS" if report.get("passed") else "FAIL")

st.subheader("Label distribution")
counts = report.get("label_quality", {}).get("counts", {})
if counts:
    st.bar_chart(pd.Series(counts, name="annotations"))
else:
    st.caption("No label-quality counts in this report.")

st.subheader("Problem annotations")
if not raw_issues:
    st.success("No QC issues were reported.")
    st.stop()

codes = sorted({issue.get("code") for issue in raw_issues if issue.get("code")})
selected = st.multiselect("Filter by issue type", codes, default=codes)
filtered = filter_issues(raw_issues, selected)
st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)

row_values = sorted({int(issue["row"]) for issue in filtered if issue.get("row") is not None})
if not row_values:
    st.caption("Selected issues are dataset-level and are not tied to one CSV row.")
    st.stop()

csv_row = st.selectbox("Review CSV row", row_values)
try:
    record = record_for_csv_row(df, csv_row)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.json(record.to_dict())

if image_bundle:
    try:
        image = load_image_from_zip(image_bundle.getvalue(), str(record["image_path"]))
        rendered = render_annotation(image, record)
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
    else:
        st.image(rendered, caption=f"CSV row {csv_row}: {record['label']}", use_container_width=True)
else:
    st.info("Upload a ZIP preserving the CSV image paths (for example images/301.jpg) to render annotations.")


st.divider()
st.subheader("Reviewer decision")
existing = next((item for item in st.session_state.decisions if int(item["row"]) == int(csv_row)), None)
options = ["approve", "reject", "needs_fix"]
default = options.index(existing["decision"]) if existing and existing.get("decision") in options else 2
decision = st.radio("Decision", options, index=default, horizontal=True)
comment = st.text_area("Reviewer comment", value=existing.get("comment", "") if existing else "")
reviewer = st.text_input("Reviewer", value=existing.get("reviewer", "reviewer") if existing else "reviewer")
if st.button("Save decision", type="primary"):
    try:
        saved = ReviewDecision(int(csv_row), decision, comment, reviewer)
        st.session_state.decisions = upsert_decision(st.session_state.decisions, saved)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.success(f"Saved {decision} for CSV row {csv_row}.")

if st.session_state.decisions:
    st.subheader("Review progress")
    decision_df = pd.DataFrame(st.session_state.decisions)
    st.dataframe(decision_df, use_container_width=True, hide_index=True)
    reviewed = len(decision_df)
    st.progress(min(reviewed / max(len(row_values), 1), 1.0), text=f"{reviewed} decisions saved")

    enriched = apply_decisions_to_dataset(df, st.session_state.decisions)
    st.download_button("Download decisions.json", decisions_json(st.session_state.decisions),
                       file_name="review_decisions.json", mime="application/json")
    st.download_button("Download decisions.csv", decisions_csv(st.session_state.decisions),
                       file_name="review_decisions.csv", mime="text/csv")
    st.download_button("Download reviewed dataset CSV", enriched.to_csv(index=False),
                       file_name="reviewed_annotations.csv", mime="text/csv")
