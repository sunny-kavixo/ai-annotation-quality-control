from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="Annotation QC Review", layout="wide")
st.title("AI Annotation Quality Control")
st.caption("V0.4 reviewer dashboard prototype — inspect QC failures against source annotations.")

report_file = st.file_uploader("QC report (JSON)", type=["json"])
dataset_file = st.file_uploader("Annotation dataset (CSV)", type=["csv"])

if not report_file or not dataset_file:
    st.info("Upload a QC JSON report and its annotation CSV to begin review.")
    st.stop()

report = json.load(report_file)
df = pd.read_csv(dataset_file)
issues = pd.DataFrame(report.get("issues", []))

c1, c2, c3 = st.columns(3)
c1.metric("Annotations", len(df))
c2.metric("QC issues", len(issues))
c3.metric("Status", "PASS" if report.get("passed") else "FAIL")

st.subheader("Label distribution")
counts = report.get("label_quality", {}).get("counts", {})
if counts:
    st.bar_chart(pd.Series(counts, name="annotations"))
else:
    st.caption("No label-quality counts in this report.")

st.subheader("Problem annotations")
if issues.empty:
    st.success("No QC issues were reported.")
    st.stop()

codes = sorted(issues["code"].dropna().unique())
selected = st.multiselect("Filter by issue type", codes, default=codes)
filtered = issues[issues["code"].isin(selected)]
st.dataframe(filtered, use_container_width=True, hide_index=True)

row_values = sorted(int(v) for v in filtered["row"].dropna().unique())
if not row_values:
    st.caption("Selected issues are dataset-level and are not tied to one CSV row.")
    st.stop()

csv_row = st.selectbox("Review CSV row", row_values)
record_index = csv_row - 2
if record_index < 0 or record_index >= len(df):
    st.error("QC report points to a row outside the uploaded dataset.")
    st.stop()

record = df.iloc[record_index]
st.json(record.to_dict())

image_upload = st.file_uploader(
    f"Optional source image for {record['image_path']}",
    type=["jpg", "jpeg", "png", "webp"],
)
if image_upload:
    image = Image.open(image_upload).convert("RGB")
    draw = ImageDraw.Draw(image)
    box = tuple(float(record[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
    draw.rectangle(box, outline="red", width=max(2, image.width // 300))
    draw.text((box[0] + 4, box[1] + 4), str(record["label"]), fill="red")
    st.image(image, caption=f"CSV row {csv_row}: {record['label']}", use_container_width=True)
