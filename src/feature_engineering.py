"""Mô-đun trích xuất đặc trưng (Feature Engineering) cho mô hình định giá bất động sản.

Tệp này tính toán các thuộc tính khoảng cách địa lý (Haversine distance tới CBD),
điểm chất lượng dữ liệu (data completeness score), tuổi của tin đăng,
và trích xuất các cờ tiện ích (binary flags) từ văn bản mô tả.
"""

import re
import numpy as np
import pandas as pd

from .config import FLAG_FEATURES, MODEL_FEATURES

# Từ khóa dùng để tạo các cờ nhị phân
KEYWORDS: dict[str, list[str]] = {
    "has_furniture": ["nội thất", "full nội thất"],
    "car_alley": ["hẻm xe hơi", "hẻm ô tô", "hẻm oto", "ô tô vào", "oto vào", "hxh"],
    "near_market": ["gần chợ", "sát chợ"],
    "near_school": ["gần trường", "đại học", "trường học"],
    "is_urgent_sale": ["cần bán gấp", "bán gấp", "chính chủ"],
}

# Tiền tố phủ định tiếng Việt
NEGATION_PATTERN: str = r"(?:không|chưa|chẳng|ko|chua|khong)\s+(?:có\s+|được\s+)?"

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


def add_quality_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Tính toán độ chênh lệch ngày so với train reference date và điểm đầy đủ dữ liệu (0 - 100%).

    Nguyên lý:
    - days_from_train_reference: listing_date - train_reference_date (không clip lower=0 để tránh
      collapse toàn bộ listing mới về 0 khi inference).
    - input_completeness_score: Tỷ lệ các thuộc tính đo lường không bị khuyết (NaN hoặc 'Không rõ').

    Args:
        df: DataFrame đầu vào.
        reference_date: Mốc thời gian tham chiếu tính tuổi tin đăng từ tập Train.

    Returns:
        DataFrame được bổ sung `days_from_train_reference` và `input_completeness_score`.
    """
    out = df.copy()

    # Tính độ lệch thời gian tương đối so với train reference date
    if "listing_date" in out and out["listing_date"].notna().any():
        if reference_date is None:
            ref_date = out["listing_date"].max()
        else:
            ref_date = pd.to_datetime(reference_date)
        out["days_from_train_reference"] = (out["listing_date"] - ref_date).dt.days
    else:
        out["days_from_train_reference"] = 0 if reference_date is not None else np.nan

    # Giữ alias listing_age_days cho khả năng tương thích ngược
    out["listing_age_days"] = out["days_from_train_reference"]

    # Các thuộc tính xem xét điểm hoàn thiện dữ liệu
    quality_columns: list[str] = [
        "Bedrooms", "Bathrooms", "Floors", "Width", "Length",
        "Alley Width", "Direction", "Position", "Latitude", "Longitude",
    ]
    quality_frame = out.reindex(columns=quality_columns)
    quality_frame = quality_frame.mask(quality_frame.eq("Không rõ"))

    # Điểm hoàn thiện dữ liệu đầu vào (%)
    out["input_completeness_score"] = quality_frame.notna().mean(axis=1) * 100
    out["data_quality_score"] = out["input_completeness_score"]
    return out


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất cờ tiện ích từ tiêu đề và nội dung tin đăng, có xử lý phủ định.

    Quy trình:
    1. Loại trừ các cụm từ bị phủ định (ví dụ: 'không có nội thất', 'chưa có nội thất').
    2. Quét các từ khóa khẳng định còn lại để gán nhãn 1, ngược lại 0.
    3. Nếu người dùng đã chỉ định cờ trong df thì ưu tiên giá trị được cấp.

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

    raw_text = (
        title.fillna("").astype(str)
        + " "
        + description.fillna("").astype(str)
    ).str.lower()

    for flag, keywords in KEYWORDS.items():
        kw_pattern = "|".join(re.escape(k) for k in keywords)
        neg_kw_pattern = rf"{NEGATION_PATTERN}(?:{kw_pattern})"

        # Xóa các cụm phủ định trước khi kiểm tra từ khóa khẳng định
        sanitized_text = raw_text.str.replace(neg_kw_pattern, "", regex=True)
        extracted = sanitized_text.str.contains(kw_pattern, regex=True).astype(int)

        if flag in out:
            supplied = pd.to_numeric(out[flag], errors="coerce")
            out[flag] = supplied.where(supplied.notna(), extracted).clip(0, 1)
        else:
            out[flag] = extracted
    return out


def make_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Tổng hợp và tạo hoàn chỉnh đúng tập đặc trưng (MODEL_FEATURES) mô hình yêu cầu.

    Quy trình:
    1. Bổ sung cờ tiện ích có xử lý phủ định (`add_text_flags`).
    2. Bổ sung đặc trưng hoàn thiện dữ liệu và thời gian (`add_quality_features`).
    3. Tính khoảng cách Haversine tới trung tâm Quận 1 (`distance_to_cbd_km`).
    4. Bổ sung các cột thiếu với giá trị 0 (với cờ) hoặc NaN (với thuộc tính số/hạng mục).
    5. Đảm bảo thứ tự cột hoàn toàn khớp với `MODEL_FEATURES`.

    Args:
        df: DataFrame chứa thông tin đã qua bước làm sạch ban đầu.
        reference_date: Mốc thời gian tham chiếu chuẩn hóa từ tập Train.

    Returns:
        DataFrame chỉ chứa các cột đặc trưng mô hình dùng để fit/predict.
    """
    out = add_quality_features(add_text_flags(df), reference_date=reference_date)

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

