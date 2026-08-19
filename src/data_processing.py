import re
import unicodedata

import numpy as np
import pandas as pd

from .config import RESIDENTIAL_TYPES


def _normalize_text(value: object) -> str:
    """Chuẩn hóa văn bản để nhận diện địa danh ổn định hơn."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value)).lower()
    return re.sub(r"\s+", " ", text).strip()


def extract_area(location: object) -> str:
    """Rút gọn địa chỉ về quận, huyện hoặc TP. Thủ Đức."""
    text = _normalize_text(location)
    if any(name in text for name in ("quận 2", "quận 9", "thủ đức")):
        return "TP. Thủ Đức"
    numbered = re.search(r"quận\s+(1[0-2]|[1-8])(?:\D|$)", text)
    if numbered:
        return f"Quận {numbered.group(1)}"
    named_areas = {
        "bình tân": "Quận Bình Tân",
        "bình thạnh": "Quận Bình Thạnh",
        "gò vấp": "Quận Gò Vấp",
        "phú nhuận": "Quận Phú Nhuận",
        "tân bình": "Quận Tân Bình",
        "tân phú": "Quận Tân Phú",
        "bình chánh": "Huyện Bình Chánh",
        "cần giờ": "Huyện Cần Giờ",
        "củ chi": "Huyện Củ Chi",
        "hóc môn": "Huyện Hóc Môn",
        "nhà bè": "Huyện Nhà Bè",
    }
    return next(
        (area for keyword, area in named_areas.items() if keyword in text),
        "Unknown",
    )


def _make_property_group_id(df: pd.DataFrame) -> pd.Series:
    """Tạo khóa nhóm để các tin trùng không rơi vào hai tập dữ liệu khác nhau."""
    signature_columns = [
        "Location",
        "Property Type",
        "Area",
        "Price",
        "Bedrooms",
        "Bathrooms",
    ]
    signature = (
        df[signature_columns]
        .fillna("")
        .astype(str)
        .agg("|".join, axis=1)
    )
    if "Listing ID" not in df:
        return "signature:" + signature

    listing_id = (
        df["Listing ID"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .str.replace(r"\.0$", "", regex=True)
    )
    return listing_id.where(listing_id.notna(), "signature:" + signature)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra, lọc và loại bản ghi trùng trước khi chia dữ liệu."""
    df = raw.copy()
    required = {"Price", "Area", "Property Type", "Location"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")
    df = df[df["Property Type"].isin(RESIDENTIAL_TYPES)].copy()
    df["location_area"] = df["Location"].map(extract_area)
    numeric_columns = [
        "Price", "Area", "Bedrooms", "Bathrooms", "Floors",
        "Width", "Length", "Alley Width",
    ]
    for column in numeric_columns:
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    unit_price = df["Price"] / df["Area"].replace(0, np.nan)
    valid = (
        df["Price"].between(100, 50_000, inclusive="left")
        & df["Area"].between(5, 500, inclusive="left")
        & unit_price.ge(10)
        & (df["Bedrooms"].isna() | df["Bedrooms"].between(1, 10))
        & (df["Bathrooms"].isna() | df["Bathrooms"].between(0, 20))
        & (df["Floors"].isna() | df["Floors"].between(0, 100))
        & (df["Width"].isna() | df["Width"].between(0.1, 100))
        & (df["Length"].isna() | df["Length"].between(0.1, 200))
        & (df["Alley Width"].isna() | df["Alley Width"].between(0, 30))
    )
    df = df.loc[valid].copy()
    df["property_group_id"] = _make_property_group_id(df)
    df = df.drop_duplicates("property_group_id")
    return df.reset_index(drop=True)
