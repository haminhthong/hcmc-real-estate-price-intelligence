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


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung cờ tiện ích từ nội dung tin khi chưa có giá trị nhập trực tiếp."""
    out = df.copy()
    title = out["Title"] if "Title" in out else pd.Series("", index=out.index)
    description = out["Description"] if "Description" in out else pd.Series("", index=out.index)
    text = (title.fillna("").astype(str) + " " + description.fillna("").astype(str)).str.lower()
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
    out = add_text_flags(df)
    for column in MODEL_FEATURES:
        if column not in out:
            out[column] = 0 if column in FLAG_FEATURES else np.nan
    return out[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
