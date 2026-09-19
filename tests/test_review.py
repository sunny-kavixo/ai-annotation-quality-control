from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest
from PIL import Image

from annotation_qc.review import filter_issues, load_image_from_zip, record_for_csv_row, render_annotation


def make_zip(path="images/1.png"):
    image_bytes = BytesIO()
    Image.new("RGB", (100, 80), "white").save(image_bytes, format="PNG")
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr(path, image_bytes.getvalue())
    return archive_bytes.getvalue()


def test_issue_filtering_and_row_mapping():
    issues = [{"code": "unknown_label", "row": 3}, {"code": "invalid_box", "row": 4}]
    assert filter_issues(issues, ["unknown_label"]) == [issues[0]]
    df = pd.DataFrame([{"label": "car"}, {"label": "person"}])
    assert record_for_csv_row(df, 3)["label"] == "person"


def test_row_mapping_rejects_report_dataset_mismatch():
    with pytest.raises(ValueError):
        record_for_csv_row(pd.DataFrame([{"label": "car"}]), 99)


def test_zip_loader_resolves_dataset_image_path():
    image = load_image_from_zip(make_zip(), "images/1.png")
    assert image.size == (100, 80)


def test_zip_loader_rejects_traversal_and_missing_image():
    with pytest.raises(ValueError):
        load_image_from_zip(make_zip(), "../secret.png")
    with pytest.raises(FileNotFoundError):
        load_image_from_zip(make_zip(), "images/missing.png")


def test_annotation_renderer_draws_without_mutating_source():
    image = Image.new("RGB", (100, 80), "white")
    record = pd.Series({"label": "car", "x_min": 10, "y_min": 10, "x_max": 50, "y_max": 40})
    rendered = render_annotation(image, record)
    assert rendered.getpixel((10, 10)) != image.getpixel((10, 10))
