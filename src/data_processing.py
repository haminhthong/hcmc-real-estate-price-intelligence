"""Mô-đun tiền xử lý và làm sạch dữ liệu bất động sản TP.HCM.

Tệp này thực hiện chuẩn hóa văn bản, trích xuất thông tin quận/huyện,
tạo khóa nhóm (property_group_id) để chống Data Leakage, lọc các bản ghi
ngoại lệ (outliers) và loại bỏ bản ghi trùng lặp trước khi huấn luyện mô hình.
"""

import re
import unicodedata
from typing import Any, Dict, List

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

    # Ánh ánh các quận/huyện bằng tên
    named_areas: Dict[str, str] = {
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
    """Tạo khóa nhóm (group key) duy nhất cho từng bất động sản.

    NGUYÊN LÝ CHỐNG DATA LEAKAGE:
    Trên các trang bất động sản, một nhà đất thường được đăng lặp lại nhiều lần
    hoặc có nhiều môi giới cùng đăng một sản phẩm. Nếu chia dữ liệu ngẫu nhiên,
    tin trùng có thể rơi vào cả tập Train và Test.
    Khóa nhóm này kết hợp Listing ID hoặc chữ ký đặc trưng (Địa điểm + Loại hình +
    Diện tích + Giá + Phòng ngủ + Phòng tắm) để đảm bảo toàn bộ tin trùng của 1 nhà
    chỉ nằm trong đúng 1 tập dữ liệu (Train, Calibration hoặc Test).

    Args:
        df: DataFrame chứa thông tin bất động sản thô.

    Returns:
        pd.Series chứa chuỗi khóa nhóm `property_group_id`.
    """
    signature_columns: List[str] = [
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

    # 1. Kiểm tra cột bắt buộc
    required = {"Price", "Area", "Property Type", "Location"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc trong dữ liệu đầu vào: {sorted(missing)}")

    # 2. Lọc loại bất động sản dân dụng được hỗ trợ
    df = df[df["Property Type"].isin(RESIDENTIAL_TYPES)].copy()

    # 3. Trích xuất khu vực quận/huyện
    df["location_area"] = df["Location"].map(extract_area)

    # 4. Chuyển đổi và làm sạch các thuộc tính số
    numeric_columns: List[str] = [
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

    # 5. Lọc các giá trị vô lý / ngoại lệ thực địa (Outlier filtering)
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

    # 6. Kiểm tra hợp lệ tọa độ GPS (Vĩ độ: 10.3 - 11.2, Kinh độ: 106.3 - 107.0 cho TP.HCM)
    valid_coordinates = (
        df["Latitude"].between(10.3, 11.2)
        & df["Longitude"].between(106.3, 107.0)
    )
    df.loc[~valid_coordinates, ["Latitude", "Longitude"]] = np.nan

    # 7. Xử lý ngày đăng tin và tạo khóa nhóm chống Data Leakage
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
    initial_len = len(df)
    df = df.drop_duplicates("property_group_id").reset_index(drop=True)
    logger.info(
        "Hoàn tất làm sạch dữ liệu. Số bản ghi từ %d còn %d (đã loại %d bản ghi trùng/lỗi).",
        initial_len,
        len(df),
        initial_len - len(df),
    )
    return df

