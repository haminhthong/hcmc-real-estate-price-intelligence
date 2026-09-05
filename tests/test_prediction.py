from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.predict import load_model, predict_one
from src.train import conformal_quantile


def test_conformal_quantile_is_conservative():
    residuals = np.arange(1, 11, dtype=float)
    assert conformal_quantile(residuals, coverage=0.8) == 9.0


def test_empty_calibration_residuals_raises_value_error():
    with pytest.raises(ValueError, match="không được rỗng"):
        conformal_quantile(np.array([]), coverage=0.8)


def test_coverage_out_of_bounds_raises_value_error():
    residuals = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="khoảng \\(0, 1\\)"):
        conformal_quantile(residuals, coverage=0.0)
    with pytest.raises(ValueError, match="khoảng \\(0, 1\\)"):
        conformal_quantile(residuals, coverage=1.0)


def test_missing_model_artifact_raises_file_not_found():
    load_model.cache_clear()
    fake_path = Path("artifacts/scratch_tmp/non_existent_model.joblib")
    with pytest.raises(FileNotFoundError):
        load_model(fake_path)


import tempfile


def test_malformed_model_artifact_raises_value_error():
    load_model.cache_clear()
    with tempfile.TemporaryDirectory() as temp_dir:
        bad_path = Path(temp_dir) / "bad_model.joblib"
        joblib.dump({"invalid_key": "data"}, bad_path)
        with pytest.raises(ValueError, match="không đúng cấu trúc"):
            load_model(bad_path)


def test_real_model_prediction_schema():
    load_model.cache_clear()
    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input, include_explanation=False)

    assert result["predicted_price_million"] >= 0
    assert result["lower_bound_million"] >= 0
    assert result["upper_bound_million"] >= result["lower_bound_million"]
    assert result["confidence"] in ("low", "medium", "high")
    assert result["top_contributions"] == []


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_shap_explanation_enabled_returns_top_contributions():
    load_model.cache_clear()
    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input, include_explanation=True)
    assert isinstance(result["top_contributions"], list)
    assert len(result["top_contributions"]) <= 5


def test_input_at_min_max_bounds():
    min_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 5.0,
        "Bedrooms": 1,
        "Width": 0.1,
        "Length": 0.1,
    }
    max_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 500.0,
        "Bedrooms": 10,
        "Width": 100.0,
        "Length": 200.0,
    }
    result_min = predict_one(min_input)
    result_max = predict_one(max_input)
    assert result_min["predicted_price_million"] >= 0
    assert result_max["predicted_price_million"] >= 0


def test_unseen_category_in_train_generates_warning():
    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận Chưa Có",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input)
    assert any("CẢNH BÁO KHU VỰC" in w for w in result["warnings"])


def test_optional_fields_all_missing():
    minimal_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 50.0,
        "Bedrooms": 2,
    }
    result = predict_one(minimal_input)
    assert result["predicted_price_million"] > 0
    assert result["data_quality_score"] > 0


def test_conformal_residual_space_matches_inference_space():
    """P0 TEST: Kiểm chứng không gian conformal residual khớp với inference log-space.

    1. Mô hình dự báo trên log1p(Price).
    2. Conformal residual được tính trên thang log: e_i = |log1p(y_i) - y_hat_log|.
    3. Phân vị quantile q được áp dụng đối xứng trên thang log: [y_hat_log - q, y_hat_log + q].
    4. Ánh xạ ngược expm1 tạo ra khoảng dự báo bất đối xứng trên thang tiền tệ VND:
       upper - predicted != predicted - lower.
    5. Điều kiện bao phủ trên thang giá [lower, upper] tương đương toán học với thang log.
    """
    load_model.cache_clear()
    pkg = load_model()
    q = pkg["residual_log_quantile"]
    assert q > 0, "Quantile conformal residual trong log-space phải dương"

    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input)
    pred_price = result["predicted_price_million"]
    lower_price = result["lower_bound_million"]
    upper_price = result["upper_bound_million"]

    # Khoảng tiền tệ là bất đối xứng (Asymmetric Monetary Interval)
    diff_upper = upper_price - pred_price
    diff_lower = pred_price - lower_price
    assert diff_upper > diff_lower, "Do tính lồi của hàm mũ expm1, khoảng cách cận trên phải lớn hơn cận dưới"

    # Kiểm tra tính tương đương toán học giữa log-space và price-space
    log_pred = np.log1p(pred_price)
    expected_lower = np.expm1(log_pred - q)
    expected_upper = np.expm1(log_pred + q)
    assert np.isclose(lower_price, expected_lower, rtol=1e-2)
    assert np.isclose(upper_price, expected_upper, rtol=1e-2)


def test_comparable_engine_returns_valid_matches():
    """P1 TEST: Kiểm tra động cơ tìm kiếm bất động sản tương đồng."""
    load_model.cache_clear()
    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input)
    assert "comparables" in result
    assert isinstance(result["comparables"], list)
    assert len(result["comparables"]) >= 1
    top_comp = result["comparables"][0]
    assert "property_type" in top_comp
    assert "similarity_score" in top_comp
    assert 0.0 <= top_comp["similarity_score"] <= 1.0
    assert result["market_context"]["comparable_median_price_million"] is not None


def test_days_from_reference_no_negative_collapse():
    """P0 TEST: Đảm bảo listing_date mới hơn mốc tham chiếu không bị collapse về 0."""
    from src.feature_engineering import make_features
    ref_date = pd.Timestamp("2025-01-01")
    # Tin đăng mới hơn 100 ngày
    future_listing = pd.DataFrame([{"listing_date": pd.Timestamp("2025-04-11")}])
    features = make_features(future_listing, reference_date=ref_date)
    assert features["days_from_train_reference"].iloc[0] == 100
    assert features["days_from_train_reference"].iloc[0] != 0


def test_text_flag_negation_handling():
    """P1 TEST: Kiểm tra xử lý từ phủ định cho các cờ nhị phân."""
    from src.feature_engineering import make_features
    # Nhà không có nội thất
    row_no_furniture = pd.DataFrame([{"Title": "Nhà đẹp", "Description": "nhà trống không có nội thất, hẻm ô tô"}])
    feats_no = make_features(row_no_furniture)
    assert feats_no["has_furniture"].iloc[0] == 0
    assert feats_no["car_alley"].iloc[0] == 1

    # Nhà có nội thất
    row_furniture = pd.DataFrame([{"Title": "Nhà đẹp", "Description": "full nội thất cao cấp"}])
    feats_yes = make_features(row_furniture)
    assert feats_yes["has_furniture"].iloc[0] == 1


def test_decomposed_reliability_structure():
    """P0 TEST: Kiểm tra cấu trúc độ tin cậy phân rã đa chiều."""
    sample_input = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 80.0,
        "Bedrooms": 3,
    }
    result = predict_one(sample_input)
    assert "reliability" in result
    rel = result["reliability"]
    assert rel["overall"] in ("low", "medium", "high")
    assert rel["reliability_level"] in ("low", "medium", "high")
    assert "input_completeness_score" in rel
    assert rel["domain_support"] in ("in_domain", "warning_ood")
    assert rel["interval_risk"] in ("tight", "moderate", "wide_interval")
