"""Mô-đun tiền xử lý và làm sạch dữ liệu bất động sản TP.HCM.

Tệp này thực hiện chuẩn hóa văn bản, trích xuất thông tin quận/huyện,
tạo khóa nhóm (property_group_id) để chống Data Leakage, lọc các bản ghi
ngoại lệ (outliers) và loại bỏ bản ghi trùng lặp trước khi huấn luyện mô hình.
"""

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from .config import RESIDENTIAL_TYPES, logger

# Mốc thời gian mặc định khi dữ liệu thiếu ngày cập nhật
CRAWL_DATE: pd.Timestamp = pd.Timestamp("2025-09-30 23:59:59")


def _normalize_text(value: Any) -> str:
    """Chuẩn hóa văn bản tiếng Việt Unicode NFC và loại bỏ khoảng trắng thừa.

    Args:
        value: Chuỗi đầu vào hoặc giá trị thiếu (NaN).

    Returns:
        Chuỗi đã được chuẩn hóa chữ thường, loại bỏ dấu khoảng trắng thừa.
    """
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value)).lower()
    return re.sub(r"\s+", " ", text).strip()


def extract_area(location: Any) -> str:
    """Rút gọn và gán nhãn địa chỉ về đúng đơn vị hành chính TP.HCM.

    Nhận diện các quận từ Quận 1 đến Quận 12, các quận đặt tên (Bình Thạnh, Gò Vấp...),
    TP. Thủ Đức (gộp Quận 2, Quận 9, Thủ Đức), và các huyện ngoại thành.

    Args:
        location: Chuỗi thông tin địa chỉ thô từ tin đăng.

    Returns:
        Tên quận/huyện tiêu chuẩn hoặc "Unknown" nếu không trích xuất được.
    """
    text = _normalize_text(location)

    # Quy hoạch TP. Thủ Đức (bao gồm Quận 2, Quận 9, Thủ Đức cũ)
    if any(name in text for name in ("quận 2", "quận 9", "thủ đức")):
        return "TP. Thủ Đức"

    # Trích xuất các quận bằng số (Quận 1 - 12)
    numbered = re.search(r"quận\s+(1[0-2]|[1-8])(?:\D|$)", text)
    if numbered:
        return f"Quận {numbered.group(1)}"

    # Ánh xạ các quận/huyện có tên riêng.
    named_areas: dict[str, str] = {
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


def make_property_signature(df: pd.DataFrame) -> pd.Series:
    """Tạo chữ ký bất động sản độc lập với tin đăng, giá, ngày đăng và môi giới.

    Các yếu tố chữ ký:
    - location_normalized
    - property_type
    - area_rounded (round 0)
    - width_rounded (round 1)
    - length_rounded (round 1)
    - bedrooms
    - bathrooms
    - latitude_rounded (round 4)
    - longitude_rounded (round 4)

    Args:
        df: DataFrame chứa thông tin bất động sản thô.

    Returns:
        pd.Series chứa chuỗi hash chữ ký đại diện cho từng căn nhà/bất động sản.
    """
    location = df["Location"].map(_normalize_text) if "Location" in df else pd.Series("", index=df.index)
    property_type = (
        df["Property Type"].map(_normalize_text)
        if "Property Type" in df
        else pd.Series("", index=df.index)
    )

    area = pd.to_numeric(df.get("Area"), errors="coerce").round(0)
    width = pd.to_numeric(df.get("Width"), errors="coerce").round(1)
    length = pd.to_numeric(df.get("Length"), errors="coerce").round(1)

    latitude = pd.to_numeric(df.get("Latitude"), errors="coerce").round(4)
    longitude = pd.to_numeric(df.get("Longitude"), errors="coerce").round(4)

    bedrooms = pd.to_numeric(df.get("Bedrooms"), errors="coerce")
    bathrooms = pd.to_numeric(df.get("Bathrooms"), errors="coerce")

    signature = pd.DataFrame(
        {
            "location": location,
            "property_type": property_type,
            "area": area,
            "width": width,
            "length": length,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "latitude": latitude,
            "longitude": longitude,
        }
    )

    return pd.util.hash_pandas_object(
        signature.fillna("missing"),
        index=False,
    ).astype(str)


def _make_property_group_id(df: pd.DataFrame) -> pd.Series:
    """Tạo khóa nhóm duy nhất cho từng bất động sản dựa trên chữ ký đặc trưng."""
    return make_property_signature(df)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Tiền xử lý, lọc nhiễu, kiểm tra hợp lệ và deduplicate dữ liệu thô.

    Quy trình làm sạch:
    1. Kiểm tra sự tồn tại của các cột bắt buộc (`Price`, `Area`, `Property Type`, `Location`).
    2. Lọc loại hình bất động sản thuộc danh mục nhà ở dân dụng (`RESIDENTIAL_TYPES`).
    3. Chuẩn hóa khu vực địa lý (`location_area`).
    4. Ép kiểu số cho các trường dữ liệu đo lường và loại bỏ ngoại lệ phi thực tế
       (Ví dụ: Giá trong khoảng 100M - 50 tỷ, diện tích 5m² - 500m²).
    5. Kiểm tra tọa độ địa lý hợp lệ trong phạm vi khu vực TP.HCM.
    6. Tạo trường ngày tin đăng `listing_date` và nhóm bản ghi trùng lặp theo `property_group_id`.

    Args:
        raw: DataFrame chứa dữ liệu thô từ file CSV.

    Returns:
        DataFrame đã được làm sạch hoàn chỉnh, sẵn sàng chia tập dữ liệu.

    Raises:
        ValueError: Nếu dữ liệu thô thiếu các cột dữ liệu bắt buộc.
    """
    logger.info("Bắt đầu quy trình làm sạch dữ liệu thô...")
    df = raw.copy()
    rows_raw = len(df)

    # 1. Kiểm tra cột bắt buộc
    required = {"Price", "Area", "Property Type", "Location"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc trong dữ liệu đầu vào: {sorted(missing)}")

    # 2. Lọc loại bất động sản dân dụng được hỗ trợ
    df = df[df["Property Type"].isin(RESIDENTIAL_TYPES)].copy()
    rows_after_type_filter = len(df)

    # 3. Trích xuất khu vực quận/huyện
    df["location_area"] = df["Location"].map(extract_area)

    # 4. Chuyển đổi và làm sạch các thuộc tính số
    numeric_columns: list[str] = [
        "Price",
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floors",
        "Width",
        "Length",
        "Alley Width",
        "Latitude",
        "Longitude",
    ]
    for column in numeric_columns:
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Lọc các giá trị vô lý và ngoại lệ thực địa.
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
    rows_after_numeric_filter = len(df)

    # 6. Kiểm tra hợp lệ tọa độ GPS (Vĩ độ: 10.3 - 11.2, Kinh độ: 106.3 - 107.0 cho TP.HCM)
    valid_coordinates = (
        df["Latitude"].between(10.3, 11.2)
        & df["Longitude"].between(106.3, 107.0)
    )
    df.loc[~valid_coordinates, ["Latitude", "Longitude"]] = np.nan

    # Xử lý ngày đăng tin và tạo khóa nhóm chống rò rỉ dữ liệu.
    if "Last Updated Date" in df:
        df["listing_date"] = pd.to_datetime(
            df["Last Updated Date"],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )
    else:
        df["listing_date"] = pd.NaT

    earliest_date = df["listing_date"].min()
    df["listing_date"] = df["listing_date"].fillna(
        earliest_date if pd.notna(earliest_date) else CRAWL_DATE
    )

    df["property_group_id"] = _make_property_group_id(df)

    # 8. Loại bỏ tin trùng lặp dựa trên property_group_id
    rows_before_dedup = len(df)
    df = df.drop_duplicates("property_group_id").reset_index(drop=True)
    rows_final = len(df)

    unsupported_type_removed = rows_raw - rows_after_type_filter
    numeric_outlier_removed = rows_after_type_filter - rows_after_numeric_filter
    duplicates_removed = rows_before_dedup - rows_final

    df.attrs["data_audit"] = {
        "rows_raw": rows_raw,
        "rows_after_cleaning": rows_final,
        "rows_removed_by_reason": {
            "unsupported_property_type": unsupported_type_removed,
            "numeric_or_price_outliers": numeric_outlier_removed,
            "duplicate_property_group": duplicates_removed,
        },
        "duplicate_group_percent": float(round(duplicates_removed / max(rows_raw, 1) * 100, 2)),
    }

    logger.info(
        "Hoàn tất làm sạch dữ liệu. Số bản ghi từ %d còn %d (đã loại %d loại hình không hỗ trợ, %d ngoại lệ số, %d tin trùng).",
        rows_raw,
        rows_final,
        unsupported_type_removed,
        numeric_outlier_removed,
        duplicates_removed,
    )
    return df
