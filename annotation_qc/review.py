from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import pandas as pd
from PIL import Image, ImageDraw, UnidentifiedImageError


def filter_issues(issues: list[dict], codes: list[str] | None = None) -> list[dict]:
    if not codes:
        return issues
    wanted = set(codes)
    return [issue for issue in issues if issue.get("code") in wanted]


def record_for_csv_row(df: pd.DataFrame, csv_row: int) -> pd.Series:
    index = csv_row - 2
    if index < 0 or index >= len(df):
        raise ValueError(f"CSV row {csv_row} is outside the uploaded dataset.")
    return df.iloc[index]


def load_image_from_zip(zip_bytes: bytes, image_path: str) -> Image.Image:
    safe_path = PurePosixPath(str(image_path).replace("\\", "/"))
    if safe_path.is_absolute() or ".." in safe_path.parts:
        raise ValueError("Unsafe image path in annotation dataset.")

    try:
        with ZipFile(BytesIO(zip_bytes)) as archive:
            names = {PurePosixPath(name).as_posix(): name for name in archive.namelist()}
            key = safe_path.as_posix()
            if key not in names:
                raise FileNotFoundError(f"Image '{key}' is not present in the uploaded ZIP.")
            payload = archive.read(names[key])
    except BadZipFile as exc:
        raise ValueError("Uploaded image bundle is not a valid ZIP file.") from exc

    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Image '{safe_path.as_posix()}' cannot be decoded.") from exc


def render_annotation(image: Image.Image, record: pd.Series) -> Image.Image:
    rendered = image.copy()
    box = tuple(float(record[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
    if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Annotation has invalid bounding-box geometry.")
    draw = ImageDraw.Draw(rendered)
    width = max(2, rendered.width // 300)
    draw.rectangle(box, outline="red", width=width)
    draw.text((box[0] + 4, box[1] + 4), str(record["label"]), fill="red")
    return rendered
