import numpy as np
import pandas as pd

from .config import FLAG_FEATURES, MODEL_FEATURES

KEYWORDS = {
    "has_furniture": ["nội thất", "full nội thất"],
    "car_alley": ["hẻm xe hơi", "ô tô vào", "oto vào", "hxh"],
    "near_market": ["gần chợ", "sát chợ"],
    "near_school": ["gần trường", "đại học", "trường học"],
    "is_urgent_sale": ["cần bán gấp", "bán gấp", "chính chủ"],
}

CBD_LATITUDE = 10.7769
CBD_LONGITUDE = 106.7009


def _distance_to_cbd(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    """Tính khoảng cách đường chim bay tới trung tâm Quận 1 bằng Haversine."""
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
    """Tính tuổi tin và điểm đầy đủ dữ liệu trong thang 0-100."""
    out = df.copy()
    if "listing_date" in out:
        newest_date = out["listing_date"].max()
        out["listing_age_days"] = (newest_date - out["listing_date"]).dt.days
    else:
        out["listing_age_days"] = np.nan

    quality_columns = [
        "Bedrooms", "Bathrooms", "Floors", "Width", "Length",
        "Alley Width", "Direction", "Position", "Latitude", "Longitude",
    ]
    quality_frame = out.reindex(columns=quality_columns)
    quality_frame = quality_frame.mask(quality_frame.eq("Không rõ"))
    out["data_quality_score"] = quality_frame.notna().mean(axis=1) * 100
    return out


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung cờ tiện ích từ nội dung tin khi chưa có giá trị nhập trực tiếp."""
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
    """Tạo đúng tập đặc trưng mà mô hình sử dụng."""
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
    for column in MODEL_FEATURES:
        if column not in out:
            out[column] = 0 if column in FLAG_FEATURES else np.nan
    return out[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
