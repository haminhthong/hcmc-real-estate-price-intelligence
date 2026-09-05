"""Mô-đun dự báo giá, hiệu chuẩn khoảng dự báo (Prediction Interval), cảnh báo OOD,

giải thích SHAP và tìm kiếm bất động sản tương đồng (Comparable Properties).
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, logger
from .feature_engineering import make_features


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
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

    model_package: dict[str, Any] = joblib.load(path)
    required_keys = {"pipeline", "version", "features"}
    if not required_keys.issubset(model_package):
        logger.error("File mô hình không đúng cấu trúc: thiếu %s", required_keys - set(model_package))
        raise ValueError("Tệp mô hình không đúng cấu trúc hoặc đã bị hỏng.")

    return model_package


def friendly_feature_name(raw_name: str) -> str:
    """Ánh xạ tên đặc trưng thô từ scikit-learn sang nhãn tiếng Việt thân thiện."""
    mapping = {
        "num__Area": "Diện tích đất (Area)",
        "num__Bedrooms": "Số phòng ngủ",
        "num__Bathrooms": "Số phòng vệ sinh",
        "num__Floors": "Số tầng",
        "num__Width": "Chiều rộng mặt tiền",
        "num__Length": "Chiều dài",
        "num__Alley Width": "Độ rộng hẻm",
        "num__distance_to_cbd_km": "Khoảng cách tới Quận 1 (CBD)",
        "num__days_from_train_reference": "Thời gian đăng tin",
        "num__listing_age_days": "Thời gian đăng tin",
        "num__input_completeness_score": "Độ đầy đủ thông tin",
        "num__data_quality_score": "Độ đầy đủ thông tin",
        "num__has_furniture": "Nội thất",
        "num__car_alley": "Hẻm xe hơi",
        "num__near_market": "Gần chợ",
        "num__near_school": "Gần trường học",
        "num__is_urgent_sale": "Cần bán gấp",
    }
    if raw_name in mapping:
        return mapping[raw_name]

    if raw_name.startswith("cat__Property Type_"):
        return f"Loại hình: {raw_name.replace('cat__Property Type_', '')}"
    if raw_name.startswith("cat__location_area_"):
        return f"Khu vực: {raw_name.replace('cat__location_area_', '')}"
    if raw_name.startswith("cat__Direction_"):
        return f"Hướng nhà: {raw_name.replace('cat__Direction_', '')}"
    if raw_name.startswith("cat__Position_"):
        return f"Vị trí: {raw_name.replace('cat__Position_', '')}"

    return raw_name.replace("num__", "").replace("cat__", "")


def find_comparables(
    model_package: dict[str, Any],
    values: dict[str, Any],
    n_matches: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Tìm 3-5 bất động sản tương đồng nhất từ tập dữ liệu tham chiếu lịch sử (Train).

    Args:
        model_package: Gói mô hình chứa danh sách `reference_listings`.
        values: Thuộc tính của bất động sản cần tra cứu.
        n_matches: Số lượng bất động sản tương đồng cần trích xuất.

    Returns:
        Tuple gồm danh sách các bất động sản tương đồng và thống kê trung vị (giá, đơn giá).
    """
    references = model_package.get("reference_listings", [])
    if not references:
        return [], {"median_price_million": None, "median_unit_price_million_m2": None}

    target_type = values.get("Property Type")
    target_area_name = values.get("location_area")
    target_area = float(values.get("Area", 80.0)) if pd.notna(values.get("Area")) and float(values.get("Area")) > 0 else 80.0
    target_beds = float(values.get("Bedrooms", 3)) if pd.notna(values.get("Bedrooms")) else 3.0

    # Lọc ứng viên: Ưu tiên cùng loại hình & cùng khu vực
    candidates = [
        r for r in references
        if r.get("property_type") == target_type and r.get("location_area") == target_area_name
    ]
    # Nếu không đủ ứng viên, mở rộng sang cùng loại hình trên toàn TP.HCM
    if len(candidates) < n_matches:
        candidates = [r for r in references if r.get("property_type") == target_type]
    if not candidates:
        candidates = references

    scored = []
    for c in candidates:
        c_area = float(c["area"]) if c.get("area") else target_area
        c_beds = float(c["bedrooms"]) if c.get("bedrooms") else target_beds
        # Khoảng cách khoảng cách hình học chuẩn hóa
        dist = abs(c_area - target_area) / max(target_area, 10.0) + 0.3 * abs(c_beds - target_beds)
        similarity = float(round(1.0 / (1.0 + dist), 2))
        scored.append((dist, similarity, c))

    scored.sort(key=lambda item: item[0])
    selected = scored[:n_matches]

    comparable_list = []
    prices = []
    unit_prices = []
    for _, sim, item in selected:
        record = dict(item)
        record["similarity_score"] = sim
        comparable_list.append(record)
        if record.get("price_million"):
            prices.append(record["price_million"])
        if record.get("unit_price_million_m2"):
            unit_prices.append(record["unit_price_million_m2"])

    summary = {
        "median_price_million": round(float(np.median(prices)), 1) if prices else None,
        "median_unit_price_million_m2": round(float(np.median(unit_prices)), 1) if unit_prices else None,
    }
    return comparable_list, summary


def predict_one(
    values: dict[str, Any],
    include_explanation: bool = False,
) -> dict[str, Any]:
    """Dự báo giá cho một bất động sản và trả về gói Price Intelligence Response hoàn chỉnh.

    Quy trình 7 bước Serving:
    1. Tải gói mô hình (LRU cached).
    2. Shared Feature Engineering với reference_date cố định từ tập Train.
    3. Dự báo điểm trung tâm (point estimate) và quy đổi từ log-scale.
    4. Xác định khoảng dự báo Conformal Prediction (Prediction Interval) không phụ thuộc phân phối.
    5. Kiểm tra cảnh báo Out-Of-Distribution (OOD) bằng phân vị robust P01–P99 và cảnh báo phân khúc hạng sang.
    6. Đánh giá độ tin cậy phân rã (Decomposed Reliability: Overall, Data Completeness, Domain Support, Interval Risk).
    7. Trích xuất top 5 đặc trưng SHAP (nếu yêu cầu) và tra cứu bất động sản tương đồng (Comparable Properties).

    Args:
        values: Dictionary chứa các trường thuộc tính của bất động sản.
        include_explanation: Tùy chọn tính toán đặc trưng SHAP.

    Returns:
        Dict phản hồi chuẩn mực có cấu trúc phân tầng (`valuation`, `market_context`, `reliability`, `comparables`, `model`).
    """
    model_package = load_model()
    reference_date = model_package.get("reference_date")
    row = pd.DataFrame([values])
    feature_frame = make_features(row, reference_date=reference_date)

    data_quality_score = float(
        feature_frame.iloc[0].get("input_completeness_score", feature_frame.iloc[0].get("data_quality_score", 100.0))
    )
    target_formulation = model_package.get("target_formulation", "total_price")
    area_val = float(values.get("Area", 1.0)) if pd.notna(values.get("Area")) and float(values.get("Area")) > 0 else 1.0

    raw_pred = float(model_package["pipeline"].predict(feature_frame)[0])
    if target_formulation == "price_per_m2":
        predicted_price = max(float(np.expm1(raw_pred) * area_val), 0.0)
    else:
        predicted_price = max(float(np.expm1(raw_pred)), 0.0)

    # 4. Xác định khoảng dự báo Conformal Prediction (Prediction Interval)
    error_quantile = float(model_package.get("residual_log_quantile", 0.25))
    target_coverage = float(model_package.get("target_coverage", 0.8))
    if target_formulation == "price_per_m2":
        lower_bound = max(float(np.expm1(raw_pred - error_quantile) * area_val), 0.0)
        upper_bound = float(np.expm1(raw_pred + error_quantile) * area_val)
    else:
        lower_bound = max(float(np.expm1(raw_pred - error_quantile)), 0.0)
        upper_bound = float(np.expm1(raw_pred + error_quantile))

    # 5. Kiểm tra OOD bằng phân vị huấn luyện robust (P01 - P99)
    warnings: list[str] = []
    quantiles_dict = model_package.get("training_quantiles", model_package.get("training_ranges", {}))
    for feature, bounds in quantiles_dict.items():
        value = feature_frame.iloc[0].get(feature)
        if pd.notna(value) and np.isfinite(float(value)) and not (
            bounds[0] <= float(value) <= bounds[1]
        ):
            warnings.append(
                f"CẢNH BÁO PHẠM VI (OOD): Đặc trưng '{feature}'={value} nằm ngoài phân vị huấn luyện P01–P99 "
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
            f"CẢNH BÁO DỮ LIỆU THIẾU: Dữ liệu đầu vào chưa đầy đủ (Điểm hoàn thiện {data_quality_score:.0f}/100)."
        )

    # Cảnh báo ngoại suy phân khúc cao cấp (> 15 tỷ VND)
    if predicted_price > 15_000:
        warnings.append(
            "CẢNH BÁO NGOẠI SUY (LUXURY): Giá dự báo > 15 tỷ VND thuộc vùng phân khúc cao cấp dữ liệu thưa; "
            "khoảng dự báo mở rộng và mức độ hiệu chuẩn hạn chế."
        )

    # 6. Đánh giá độ tin cậy phân rã (Decomposed Reliability)
    relative_width = (upper_bound - lower_bound) / max(predicted_price, 1)
    interval_risk = "wide_interval" if relative_width > 0.8 else ("moderate" if relative_width > 0.4 else "tight")
    domain_support = "warning_ood" if any("CẢNH BÁO" in w for w in warnings) else "in_domain"

    if domain_support == "warning_ood" or interval_risk == "wide_interval" or data_quality_score < 60:
        reliability_level = "low"
    elif interval_risk == "moderate":
        reliability_level = "medium"
    else:
        reliability_level = "high"

    # Tra cứu đơn giá trung vị cùng phân khúc
    segment_unit_price = model_package.get("segment_unit_prices", {}).get(
        (property_type, location_area)
    )

    # Tìm kiếm bất động sản tương đồng (Comparable Properties)
    comparables, comp_summary = find_comparables(model_package, values, n_matches=4)

    # Chỉ tính SHAP khi người dùng hoặc API chủ động yêu cầu.
    should_explain = include_explanation or values.get("include_explanation", False)
    contributions = (
        explain_top_features(model_package, feature_frame)
        if should_explain
        else []
    )

    return {
        # Cấu trúc phân tầng hiện đại
        "valuation": {
            "point_estimate_million": round(predicted_price, 1),
            "prediction_interval": {
                "lower_bound_million": round(lower_bound, 1),
                "upper_bound_million": round(upper_bound, 1),
                "target_coverage": target_coverage,
            },
        },
        "market_context": {
            "segment_median_unit_price_million_m2": (
                round(float(segment_unit_price), 1)
                if segment_unit_price is not None
                else None
            ),
            "comparable_median_price_million": comp_summary["median_price_million"],
            "comparable_median_unit_price_million_m2": comp_summary["median_unit_price_million_m2"],
        },
        "reliability": {
            "overall": reliability_level,
            "reliability_level": reliability_level,
            "input_completeness_score": round(data_quality_score, 1),
            "domain_support": domain_support,
            "interval_risk": interval_risk,
            "warnings": warnings,
        },
        "comparables": comparables,
        "explanation": {
            "top_contributions": contributions,
            "explanation_space": "log_target_space",
            "note": "Giá trị SHAP thể hiện mức độ đóng góp của đặc trưng trên thang log-price của mô hình, không đại diện cho quan hệ nhân quả.",
        },
        "model": {
            "version": model_package["version"],
            "model_type": model_package.get("model_type", "ExtraTreesRegressor"),
            "target_formulation": target_formulation,
        },
        # Các trường tương thích ngược (Flat aliases)
        "predicted_price_million": round(predicted_price, 1),
        "lower_bound_million": round(lower_bound, 1),
        "upper_bound_million": round(upper_bound, 1),
        "confidence": reliability_level,
        "reliability_level": reliability_level,
        "model_version": model_package["version"],
        "warnings": warnings,
        "data_quality_score": round(data_quality_score, 1),
        "input_completeness_score": round(data_quality_score, 1),
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
    model_package: dict[str, Any],
    feature_frame: pd.DataFrame,
) -> list[dict[str, str | float]]:
    """Sử dụng SHAP TreeExplainer trích xuất 5 đặc trưng ảnh hưởng lớn nhất trong không gian log-target.

    Args:
        model_package: Gói mô hình đã lưu.
        feature_frame: DataFrame 1 dòng chứa đặc trưng của bất động sản.

    Returns:
        Danh sách 5 dict chứa tên đặc trưng (đã chuẩn hóa nhãn thân thiện) và giá trị SHAP log-space:
        `[{"feature": str, "friendly_name": str, "shap_value": float}, ...]`
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
        raise RuntimeError(
            "Thiếu thư viện SHAP để giải thích mô hình. Hãy cài requirements.txt."
        ) from exc

    scores = np.asarray(shap_values)[0]
    top_indices = np.argsort(np.abs(scores))[-5:][::-1]

    return [
        {
            "feature": str(feature_names[index]),
            "friendly_name": friendly_feature_name(str(feature_names[index])),
            "shap_value": round(float(scores[index]), 4),
        }
        for index in top_indices
    ]
