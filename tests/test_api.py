from fastapi.testclient import TestClient

import api.main as api_module
from api.main import app

client = TestClient(app)


def test_health_schema():
    response = client.get("/health")
    assert response.status_code == 200
    assert set(response.json()) == {"status", "model_loaded"}


def test_invalid_area_is_rejected():
    payload = {"Property Type": "Nhà riêng", "location_area": "Hà Nội", "Area": 50, "Bedrooms": 2}
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_required_value_is_rejected():
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Bedrooms": 2,
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_nonfinite_area_is_rejected():
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": "NaN",
        "Bedrooms": 2,
    }
    assert client.post("/predict", json=payload).status_code == 422


def test_prediction_response_schema(monkeypatch):
    expected = {
        "predicted_price_million": 7850.0,
        "lower_bound_million": 6400.0,
        "upper_bound_million": 9300.0,
        "confidence": "medium",
        "model_version": "1.0.0",
        "warnings": [],
        "top_contributions": [],
        "segment_median_unit_price_million_m2": 98.0,
        "disclaimer": "Giá tham khảo.",
    }
    monkeypatch.setattr(api_module, "predict_one", lambda _: expected)
    payload = {
        "Property Type": "Nhà riêng",
        "location_area": "Quận 1",
        "Area": 50,
        "Bedrooms": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == expected
