"""Mô-đun dự báo và giải thích kết quả cho bất động sản TP.HCM.

Tệp này phục vụ dự báo giá cho một bất động sản cụ thể, kiểm tra điều kiện
dữ liệu đầu vào, cảnh báo vượt phạm vi huấn luyện (training ranges), tính toán
khoảng tin cậy conformal, mức độ tin cậy (low, medium, high) và trích xuất
5 đặc trưng ảnh hưởng nhiều nhất bằng thư viện SHAP.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, logger
from .feature_engineering import make_features


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Tải và lưu vào bộ nhớ tạm (LRU cache) gói mô hình đã được huấn luyện.

    Args:
        path: Đường dẫn tới file `.joblib` chứa gói mô hình.

    Returns:
        Dict chứa pipeline, phiên bản mô hình và thông số liên quan.

    Raises:
        FileNotFoundError: Nếu file mô hình chưa tồn tại.
        ValueError: Nếu file mô hình hỏng hoặc thiếu các khóa cấu trúc bắt buộc.
    """
    if not path.exists():
        logger.error("Không tìm thấy file mô hình tại đường dẫn: %s", path)
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy: python -m src.train")

    model_package: Dict[str, Any] = joblib.load(path)
    required_keys = {"pipeline", "version", "features"}
    if not required_keys.issubset(model_package):
        logger.error("File mô hình không đúng cấu trúc: thiếu %s", required_keys - set(model_package))
        raise ValueError("Tệp mô hình không đúng cấu trúc hoặc đã bị hỏng.")

    return model_package


def predict_one(values: Dict[str, Any]) -> Dict[str, Any]:
    """Dự báo giá cho một bất động sản và trả về thông tin bổ trợ diễn giải.

    Quy trình:
    1. Tải gói mô hình (đã cache).
    2. Chuyển đổi dict dữ liệu đầu vào thành DataFrame và trích xuất đặc trưng (`make_features`).
    3. Dự báo giá điểm trung tâm (point prediction) và quy đổi từ log-scale.
    4. Xác định khoảng tin cậy conformal `[lower_bound, upper_bound]` (Target Coverage 80%).
    5. Quét kiểm tra cảnh báo (Out of bounds, missing coordinates, low quality score).
    6. Trích xuất 5 đặc trưng ảnh hưởng mạnh nhất qua SHAP (`explain_top_features`).
    7. Tra cứu đơn giá trung vị phân khúc cùng loại hình x khu vực.

    Args:
        values: Dictionary chứa các trường thuộc tính của bất động sản.

    Returns:
        Dict phản hồi đầy đủ các thông tin dự báo, khoảng tin cậy, mức độ tin cậy,
        cảnh báo, điểm chất lượng và phân tích SHAP.
    """
    model_package = load_model()
    row = pd.DataFrame([values])
    feature_frame = make_features(row)

    data_quality_score = float(feature_frame.iloc[0]["data_quality_score"])
    log_prediction = float(model_package["pipeline"].predict(feature_frame)[0])
    predicted_price = max(float(np.expm1(log_prediction)), 0.0)

    # Tính toán khoảng tin cậy Conformal Prediction (Lower Bound & Upper Bound)
    error_quantile = float(model_package.get("residual_log_quantile", 0.25))
    lower_bound = max(
        float(np.expm1(log_prediction - error_quantile)),
        0.0,
    )
    upper_bound = float(np.expm1(log_prediction + error_quantile))

    # Quét cảnh báo tính hợp lệ và phạm vi huấn luyện
    warnings: List[str] = []
    for feature, bounds in model_package.get("training_ranges", {}).items():
        value = values.get(feature)
        if value is not None and not bounds[0] <= float(value) <= bounds[1]:
            warnings.append(
                f"CẢNH BÁO PHẠM VI: Đặc trưng '{feature}'={value} nằm ngoài ngưỡng huấn luyện "
                f"({bounds[0]:g}–{bounds[1]:g})."
            )

    location_area = values.get("location_area")
    if location_area not in model_package.get("supported_areas", []):
        warnings.append("CẢNH BÁO KHU VỰC: Khu vực này chưa xuất hiện trong tập huấn luyện.")

    property_type = values.get("Property Type")
    if property_type not in model_package.get("supported_property_types", []):
        warnings.append("CẢNH BÁO LOẠI HÌNH: Loại bất động sản này chưa xuất hiện trong tập huấn luyện.")

    if values.get("Latitude") is None or values.get("Longitude") is None:
        warnings.append("CẢNH BÁO TỌA ĐỘ: Thiếu GPS (vĩ độ/kinh độ) nên mô hình không dùng được khoảng cách CBD.")

    if data_quality_score < 60:
        warnings.append(
            f"CẢNH BÁO DỮ LIỆU THIẾU: Dữ liệu đầu vào chưa đầy đủ (Điểm chất lượng {data_quality_score:.0f}/100)."
        )

    # Đánh giá mức độ tin cậy (Confidence Level: Low / Medium / High)
    relative_width = (upper_bound - lower_bound) / max(predicted_price, 1)
    if warnings or relative_width > 0.8:
        confidence = "low"
    elif relative_width > 0.4:
        confidence = "medium"
    else:
        confidence = "high"

    # Tra cứu đơn giá trung vị cùng phân khúc
    segment_unit_price = model_package.get("segment_unit_prices", {}).get(
        (property_type, location_area)
    )

    # Tính toán SHAP values
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
            "Kết quả là giá đăng tham khảo từ mô hình Machine Learning, "
            "không phải giá giao dịch thực tế hoặc văn bản thẩm định giá chuyên nghiệp."
        ),
    }


def explain_top_features(
    model_package: Dict[str, Any],
    feature_frame: pd.DataFrame,
) -> List[Dict[str, Union[str, float]]]:
    """Sử dụng SHAP TreeExplainer trích xuất 5 đặc trưng ảnh hưởng lớn nhất tới quyết định định giá.

    Args:
        model_package: Gói mô hình đã lưu.
        feature_frame: DataFrame 1 dòng chứa đặc trưng của bất động sản.

    Returns:
        Danh sách 5 dict chứa tên đặc trưng và giá trị SHAP tương ứng:
        `[{"feature": str, "shap_value": float}, ...]`
    """
    pipeline = model_package["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(feature_frame)
    feature_names = preprocessor.get_feature_names_out()
    model = pipeline.named_steps["model"]

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
    except ImportError as exc:
        logger.error("Thiếu thư viện 'shap'. Vui lòng kiểm tra lại môi trường venv.")
        raise RuntimeError(
            "Thiếu thư viện SHAP để giải thích mô hình. Vui lòng cài lại requirements.txt."
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

