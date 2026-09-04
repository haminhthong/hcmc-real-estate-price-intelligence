from pathlib import Path

import joblib
import numpy as np
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


def test_malformed_model_artifact_raises_value_error(tmp_path):
    load_model.cache_clear()
    bad_path = tmp_path / "bad_model.joblib"
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
