from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import MODEL_PATH, MODEL_VERSION, RESIDENTIAL_TYPES, SUPPORTED_AREAS
from src.predict import load_model, predict_one

app = FastAPI(title="HCMC Real Estate Price Intelligence", version=MODEL_VERSION)


class PredictionRequest(BaseModel):
    """Dữ liệu đầu vào cho một lần dự báo."""

    model_config = ConfigDict(populate_by_name=True, allow_inf_nan=False)

    property_type: str = Field(alias="Property Type")
    location_area: str
    area: float = Field(gt=0, le=2000, alias="Area")
    bedrooms: int = Field(ge=1, le=10, alias="Bedrooms")
    bathrooms: int | None = Field(default=None, ge=0, le=20, alias="Bathrooms")
    floors: int | None = Field(default=None, ge=0, le=100, alias="Floors")
    width: float | None = Field(default=None, gt=0, le=100, alias="Width")
    length: float | None = Field(default=None, gt=0, le=200, alias="Length")
    alley_width: float | None = Field(default=None, ge=0, le=30, alias="Alley Width")
    latitude: float | None = Field(default=None, ge=10.3, le=11.2, alias="Latitude")
    longitude: float | None = Field(default=None, ge=106.3, le=107.0, alias="Longitude")
    direction: str = Field(default="Không rõ", alias="Direction")
    position: str = Field(default="Không rõ", alias="Position")
    has_furniture: bool = False
    car_alley: bool = False
    near_market: bool = False
    near_school: bool = False
    is_urgent_sale: bool = False

    @field_validator("property_type")
    @classmethod
    def supported_type(cls, value: str) -> str:
        if value not in RESIDENTIAL_TYPES:
            raise ValueError("Loại bất động sản chưa được hỗ trợ")
        return value

    @field_validator("location_area")
    @classmethod
    def supported_area(cls, value: str) -> str:
        if value not in SUPPORTED_AREAS:
            raise ValueError("Khu vực chưa được hỗ trợ")
        return value


class PredictionResponse(BaseModel):
    """Kết quả dự báo trả về cho ứng dụng khách."""

    predicted_price_million: float
    lower_bound_million: float
    upper_bound_million: float
    confidence: Literal["low", "medium", "high"]
    model_version: str
    warnings: list[str]
    data_quality_score: float = Field(ge=0, le=100)
    top_contributions: list[dict]
    segment_median_unit_price_million_m2: float | None
    disclaimer: str


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info() -> dict:
    artifact = load_model()
    return {
        "model_version": artifact["version"],
        "model_type": "RandomForestRegressor",
        "supported_areas": artifact.get("supported_areas", SUPPORTED_AREAS),
        "supported_property_types": artifact.get(
            "supported_property_types",
            RESIDENTIAL_TYPES,
        ),
        "split_protocol": artifact.get("split_protocol"),
        "prediction_interval_target_coverage": artifact.get("target_coverage"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    try:
        return predict_one(request.model_dump(by_alias=True))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
