import pandas as pd

from annotation_qc.validator import REQUIRED_COLUMNS, validate_dataframe


def frame_with_size(box=(10, 20, 100, 120), size=(640, 480)):
    columns = list(REQUIRED_COLUMNS) + ["image_width", "image_height"]
    row = ["img_1", "images/1.jpg", "car", *box, *size]
    return pd.DataFrame([row], columns=columns)


def test_box_inside_image_passes():
    assert validate_dataframe(frame_with_size()).passed


def test_box_outside_image_is_detected():
    result = validate_dataframe(frame_with_size(box=(600, 20, 700, 120)))
    assert not result.passed
    assert any(issue.code == "box_outside_image" for issue in result.issues)


def test_invalid_image_size_is_detected():
    result = validate_dataframe(frame_with_size(size=(0, 480)))
    assert not result.passed
    assert any(issue.code == "invalid_image_dimensions" for issue in result.issues)


def test_dimensions_are_optional_for_v01_compatibility():
    df = pd.DataFrame(
        [["img_1", "images/1.jpg", "car", 10, 20, 100, 120]],
        columns=REQUIRED_COLUMNS,
    )
    assert validate_dataframe(df).passed


def test_only_one_dimension_column_is_configuration_error():
    df = frame_with_size().drop(columns=["image_height"])
    result = validate_dataframe(df)
    assert not result.passed
    assert any(issue.code == "incomplete_image_dimensions" for issue in result.issues)


def test_real_image_dimension_mismatch_is_detected(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (320, 240)).save(image_dir / "1.jpg")

    df = frame_with_size(size=(640, 480))
    result = validate_dataframe(df, image_root=tmp_path)

    assert not result.passed
    assert any(issue.code == "image_dimension_mismatch" for issue in result.issues)


def test_matching_real_image_dimensions_pass(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (640, 480)).save(image_dir / "1.jpg")

    result = validate_dataframe(frame_with_size(), image_root=tmp_path)
    assert result.passed
