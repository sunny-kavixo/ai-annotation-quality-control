import pandas as pd
import pytest

from annotation_qc.validator import REQUIRED_COLUMNS, validate_dataframe


def frame_with_size(box=(10, 20, 100, 120), size=(640, 480), image_path="images/1.jpg"):
    columns = list(REQUIRED_COLUMNS) + ["image_width", "image_height"]
    row = ["img_1", image_path, "car", *box, *size]
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


def test_nested_relative_image_path_passes(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "nested" / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (640, 480)).save(image_dir / "1.jpg")

    result = validate_dataframe(
        frame_with_size(image_path="nested/images/1.jpg"), image_root=tmp_path
    )
    assert result.passed


def test_parent_directory_traversal_is_rejected(tmp_path):
    from PIL import Image

    image_root = tmp_path / "dataset"
    image_root.mkdir()
    Image.new("RGB", (640, 480)).save(tmp_path / "outside.jpg")

    result = validate_dataframe(
        frame_with_size(image_path="../outside.jpg"), image_root=image_root
    )

    assert not result.passed
    assert any(issue.code == "image_path_outside_root" for issue in result.issues)
    assert str(tmp_path) not in result.issues[0].message


def test_absolute_image_path_is_rejected(tmp_path):
    from PIL import Image

    image_root = tmp_path / "dataset"
    image_root.mkdir()
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (640, 480)).save(outside)

    result = validate_dataframe(
        frame_with_size(image_path=str(outside)), image_root=image_root
    )

    assert not result.passed
    assert any(issue.code == "image_path_outside_root" for issue in result.issues)
    assert str(outside) not in result.issues[0].message


def test_symlink_escape_is_rejected(tmp_path):
    from PIL import Image

    image_root = tmp_path / "dataset"
    image_root.mkdir()
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (640, 480)).save(outside)
    link = image_root / "escape.jpg"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    result = validate_dataframe(
        frame_with_size(image_path="escape.jpg"), image_root=image_root
    )

    assert not result.passed
    assert any(issue.code == "image_path_outside_root" for issue in result.issues)


def test_missing_image_inside_root_preserves_existing_issue(tmp_path):
    result = validate_dataframe(frame_with_size(), image_root=tmp_path)

    assert not result.passed
    assert any(issue.code == "image_file_missing" for issue in result.issues)


def test_unreadable_image_inside_root_preserves_existing_issue(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "1.jpg").write_text("not an image", encoding="utf-8")

    result = validate_dataframe(frame_with_size(), image_root=tmp_path)

    assert not result.passed
    assert any(issue.code == "image_file_unreadable" for issue in result.issues)


def test_malformed_image_path_is_reported_and_later_rows_are_validated(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (640, 480)).save(image_dir / "valid.jpg")
    malformed = frame_with_size(image_path="bad\x00name.jpg")
    valid = frame_with_size(image_path="images/valid.jpg")
    valid.loc[0, "image_id"] = "img_2"

    result = validate_dataframe(
        pd.concat([malformed, valid], ignore_index=True), image_root=tmp_path
    )

    assert [(issue.row, issue.code) for issue in result.issues] == [
        (2, "invalid_image_path")
    ]
    assert result.issues[0].message == "image_path is not a valid filesystem path."
    assert "bad" not in result.issues[0].message
    assert str(tmp_path) not in result.issues[0].message
