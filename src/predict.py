from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH
from .feature_engineering import make_features


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    """Đọc gói mô hình một lần và dùng lại cho các yêu cầu tiếp theo."""
    if not path.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy: python -m src.train")
    model_package = joblib.load(path)
    required_keys = {"pipeline", "version", "features"}
    if not required_keys.issubset(model_package):
        raise ValueError("Tệp mô hình không đúng cấu trúc hoặc đã bị hỏng.")
    return model_package


def predict_one(values: dict) -> dict:
    """Dự báo một bất động sản và trả thông tin hỗ trợ diễn giải."""
    model_package = load_model()
    row = pd.DataFrame([values])
    feature_frame = make_features(row)
    data_quality_score = float(feature_frame.iloc[0]["data_quality_score"])
    log_prediction = float(model_package["pipeline"].predict(feature_frame)[0])
    predicted_price = max(float(np.expm1(log_prediction)), 0.0)

    error_quantile = float(model_package.get("residual_log_quantile", 0.25))
    lower_bound = max(
        float(np.expm1(log_prediction - error_quantile)),
        0.0,
    )
    upper_bound = float(np.expm1(log_prediction + error_quantile))

    warnings: list[str] = []
    for feature, bounds in model_package.get("training_ranges", {}).items():
        value = values.get(feature)
        if value is not None and not bounds[0] <= float(value) <= bounds[1]:
            warnings.append(
                f"{feature} nằm ngoài phạm vi huấn luyện "
                f"({bounds[0]:g}–{bounds[1]:g})."
            )

    location_area = values.get("location_area")
    if location_area not in model_package.get("supported_areas", []):
        warnings.append("Khu vực này không xuất hiện trong tập huấn luyện.")

    property_type = values.get("Property Type")
    if property_type not in model_package.get("supported_property_types", []):
        warnings.append("Loại bất động sản này không xuất hiện trong tập huấn luyện.")
    if values.get("Latitude") is None or values.get("Longitude") is None:
        warnings.append("Thiếu tọa độ nên mô hình không dùng được khoảng cách tới trung tâm.")
    if data_quality_score < 60:
        warnings.append(
            f"Dữ liệu đầu vào chưa đầy đủ (điểm chất lượng {data_quality_score:.0f}/100)."
        )

    relative_width = (upper_bound - lower_bound) / max(predicted_price, 1)
    if warnings or relative_width > 0.8:
        confidence = "low"
    elif relative_width > 0.4:
        confidence = "medium"
    else:
        confidence = "high"

    segment_unit_price = model_package.get("segment_unit_prices", {}).get(
        (property_type, location_area)
    )
    contributions = explain_top_features(model_package, feature_frame)
    return {
        "predicted_price_million": round(predicted_price, 1),
        "lower_bound_million": round(lower_bound, 1),
        "upper_bound_million": round(upper_bound, 1),
        "confidence": confidence,
        "model_version": model_package["version"],
        "warnings": warnings,
        "data_quality_score": round(data_quality_score, 1),
        "top_contributions": contributions,
        "segment_median_unit_price_million_m2": (
            round(float(segment_unit_price), 1)
            if segment_unit_price is not None
            else None
        ),
        "disclaimer": (
            "Giá đăng tham khảo, không phải giá giao dịch hoặc kết quả "
            "thẩm định chuyên nghiệp."
        ),
    }


def explain_top_features(
    model_package: dict,
    feature_frame: pd.DataFrame,
) -> list[dict[str, str | float]]:
    """Tính năm đóng góp SHAP có trị tuyệt đối lớn nhất."""
    pipeline = model_package["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(feature_frame)
    feature_names = preprocessor.get_feature_names_out()
    model = pipeline.named_steps["model"]

    try:
        import shap

        shap_values = shap.TreeExplainer(model).shap_values(transformed)
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện SHAP. Hãy cài lại requirements.txt."
        ) from exc

    scores = np.asarray(shap_values)[0]
    top_indices = np.argsort(np.abs(scores))[-5:][::-1]
    return [
        {
            "feature": str(feature_names[index]),
            "shap_value": round(float(scores[index]), 4),
        }
        for index in top_indices
    ]
