"""Mô-đun trích xuất đặc trưng (Feature Engineering) cho mô hình định giá bất động sản.

Tệp này tính toán các thuộc tính khoảng cách địa lý (Haversine distance tới CBD),
điểm chất lượng dữ liệu (data completeness score), tuổi của tin đăng,
và trích xuất các cờ tiện ích (binary flags) từ văn bản mô tả.
"""

import numpy as np
import pandas as pd

from .config import FLAG_FEATURES, MODEL_FEATURES

# Từ khóa dùng để tạo các cờ nhị phân
KEYWORDS: dict[str, list[str]] = {
    "has_furniture": ["nội thất", "full nội thất"],
    "car_alley": ["hẻm xe hơi", "ô tô vào", "oto vào", "hxh"],
    "near_market": ["gần chợ", "sát chợ"],
    "near_school": ["gần trường", "đại học", "trường học"],
    "is_urgent_sale": ["cần bán gấp", "bán gấp", "chính chủ"],
}

# Tọa độ địa lý trung tâm TP.HCM (Chợ Bến Thành / Quận 1)
CBD_LATITUDE: float = 10.7769
CBD_LONGITUDE: float = 106.7009


def _distance_to_cbd(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    """Tính khoảng cách đường chim bay (km) tới trung tâm Quận 1 bằng công thức Haversine.

    Công thức Haversine tính khoảng cách giữa 2 điểm trên mặt cầu Trái Đất:
        a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        c = 2 * arcsin(√a)
        d = R * c  (với bán kính Trái Đất R ≈ 6371 km)

    Args:
        latitude: Chuỗi tọa độ Vĩ độ (deg).
        longitude: Chuỗi tọa độ Kinh độ (deg).

    Returns:
        pd.Series chứa khoảng cách theo kilômét.
    """
    lat1 = np.radians(latitude.astype(float))
    lon1 = np.radians(longitude.astype(float))
    lat2 = np.radians(CBD_LATITUDE)
    lon2 = np.radians(CBD_LONGITUDE)

    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2

    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(haversine))


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán tuổi tin đăng và điểm đầy đủ dữ liệu (Data Quality Score) 0 - 100%.

    Điểm đầy đủ dữ liệu được tính dựa trên tỷ lệ các trường thông tin quan trọng
    không bị khuyết (NaN hoặc 'Không rõ').

    Args:
        df: DataFrame đầu vào.

    Returns:
        DataFrame được bổ sung hai cột `listing_age_days` và `data_quality_score`.
    """
    out = df.copy()

    # Tính số ngày kể từ mốc thời gian tin đăng gần nhất
    if "listing_date" in out:
        newest_date = out["listing_date"].max()
        out["listing_age_days"] = (newest_date - out["listing_date"]).dt.days
    else:
        out["listing_age_days"] = np.nan

    # Các thuộc tính xem xét điểm chất lượng
    quality_columns: list[str] = [
        "Bedrooms", "Bathrooms", "Floors", "Width", "Length",
        "Alley Width", "Direction", "Position", "Latitude", "Longitude",
    ]
    quality_frame = out.reindex(columns=quality_columns)
    quality_frame = quality_frame.mask(quality_frame.eq("Không rõ"))

    # Tỷ lệ % điền đầy đủ các thông tin
    out["data_quality_score"] = quality_frame.notna().mean(axis=1) * 100
    return out


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất cờ tiện ích từ tiêu đề và nội dung tin đăng bằng NLP Regex đơn giản.

    Nếu người dùng chưa truyền trực tiếp các giá trị nhị phân này, hệ thống sẽ tự động
    quét các từ khóa xuất hiện trong tiêu đề (Title) và mô tả (Description).

    Args:
        df: DataFrame đầu vào.

    Returns:
        DataFrame đã cập nhật các cột cờ nhị phân (0 hoặc 1).
    """
    out = df.copy()
    title = out["Title"] if "Title" in out else pd.Series("", index=out.index)
    description = (
        out["Description"]
        if "Description" in out
        else pd.Series("", index=out.index)
    )

    text = (
        title.fillna("").astype(str)
        + " "
        + description.fillna("").astype(str)
    ).str.lower()

    for flag, keywords in KEYWORDS.items():
        extracted = text.str.contains("|".join(keywords), regex=True).astype(int)
        if flag in out:
            supplied = pd.to_numeric(out[flag], errors="coerce")
            out[flag] = supplied.where(supplied.notna(), extracted).clip(0, 1)
        else:
            out[flag] = extracted
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp và tạo hoàn chỉnh đúng tập đặc trưng (MODEL_FEATURES) mô hình yêu cầu.

    Quy trình:
    1. Bổ sung cờ tiện ích (`add_text_flags`).
    2. Bổ sung đặc trưng chất lượng (`add_quality_features`).
    3. Tính khoảng cách Haversine tới trung tâm Quận 1 (`distance_to_cbd_km`).
    4. Bổ sung các cột thiếu với giá trị 0 (với cờ) hoặc NaN (với thuộc tính số/hạng mục).
    5. Đảm bảo thứ tự cột hoàn toàn khớp với `MODEL_FEATURES`.

    Args:
        df: DataFrame chứa thông tin đã qua bước làm sạch ban đầu.

    Returns:
        DataFrame chỉ chứa các cột đặc trưng mô hình dùng để fit/predict.
    """
    out = add_quality_features(add_text_flags(df))

    latitude = pd.to_numeric(
        (
            out["Latitude"]
            if "Latitude" in out
            else pd.Series(np.nan, index=out.index)
        ),
        errors="coerce",
    )
    longitude = pd.to_numeric(
        (
            out["Longitude"]
            if "Longitude" in out
            else pd.Series(np.nan, index=out.index)
        ),
        errors="coerce",
    )
    out["distance_to_cbd_km"] = _distance_to_cbd(latitude, longitude)

    # Đảm bảo đầy đủ tất cả cột trong danh sách đặc trưng mô hình
    for column in MODEL_FEATURES:
        if column not in out:
            out[column] = 0 if column in FLAG_FEATURES else np.nan

    # Thay thế các giá trị vô cực nếu có
    return out[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
