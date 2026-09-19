import pandas as pd
import pytest

from annotation_qc.validator import REQUIRED_COLUMNS, load_annotations, validate_dataframe


def clean_frame():
    return pd.DataFrame(
        [["img_1", "images/1.jpg", "car", 10, 20, 100, 120]],
        columns=REQUIRED_COLUMNS,
    )


def test_clean_annotation_passes():
    result = validate_dataframe(clean_frame())
    assert result.passed
    assert result.issues == []


def test_same_image_can_have_multiple_objects_without_duplicate_warning():
    df = pd.DataFrame(
        [
            ["img_1", "images/1.jpg", "car", 10, 20, 100, 120],
            ["img_1", "images/1.jpg", "person", 130, 30, 170, 150],
        ],
        columns=REQUIRED_COLUMNS,
    )
    result = validate_dataframe(df)
    assert result.passed
    assert not any(issue.code == "duplicate_annotation" for issue in result.issues)


def test_exact_duplicate_annotation_is_warning():
    df = pd.concat([clean_frame(), clean_frame()], ignore_index=True)
    result = validate_dataframe(df)
    assert result.passed
    assert sum(issue.code == "duplicate_annotation" for issue in result.issues) == 2


def test_bad_geometry_fails():
    df = clean_frame()
    df.loc[0, "x_max"] = 5
    result = validate_dataframe(df)
    assert not result.passed
    assert any(issue.code == "invalid_box_geometry" for issue in result.issues)


def test_infinite_coordinate_fails():
    df = clean_frame()
    df.loc[0, "x_min"] = float("inf")
    result = validate_dataframe(df)
    assert not result.passed
    assert any(issue.code == "invalid_coordinate" for issue in result.issues)


def test_empty_csv_has_readable_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_annotations(path)
