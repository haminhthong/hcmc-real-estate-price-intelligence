"""REST API cho hệ thống định giá bất động sản TP.HCM (FastAPI).

Ứng dụng cung cấp các HTTP Endpoints:
- GET /health: Kiểm tra trạng thái hoạt động của máy chủ và mô hình.
- GET /model-info: Tra cứu thông tin cấu hình, phiên bản và vùng hỗ trợ.
- POST /predict: Tiếp nhận thông tin bất động sản và trả về dự báo giá, khoảng tin cậy conformal (SHAP tắt mặc định).
- POST /explain: Tiếp nhận thông tin bất động sản và trả về dự báo giá kèm giải thích SHAP.
"""

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import (
    MODEL_PATH,
    MODEL_VERSION,
    RESIDENTIAL_TYPES,
    SUPPORTED_AREAS,
    logger,
)
from src.predict import load_model, predict_one

# Khởi tạo ứng dụng FastAPI và tài liệu OpenAPI.
app = FastAPI(
    title="HCMC Real Estate Price Intelligence API",
    description=(
        "API ước lượng giá đăng tham khảo cho bất động sản dân dụng tại TP.HCM "
        "kết hợp Conformal Prediction (Khoảng tin cậy 80%) và SHAP Explainer."
    ),
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

class PredictionRequest(BaseModel):
    """Dữ liệu yêu cầu đầu vào cho một bất động sản cần định giá."""

    model_config = ConfigDict(populate_by_name=True, allow_inf_nan=False)

    property_type: str = Field(
        ...,
        alias="Property Type",
        description="Loại hình bất động sản (Nhà riêng, Nhà mặt tiền, Căn hộ chung cư, ...)",
        examples=["Nhà riêng"],
    )
    location_area: str = Field(
        ...,
        description="Quận/huyện hoặc khu vực thuộc TP.HCM (Ví dụ: Quận 1, Quận 7, TP. Thủ Đức)",
        examples=["Quận 7"],
    )
    area: float = Field(
        ...,
        gt=0,
        le=2000,
        alias="Area",
        description="Diện tích đất/sử dụng (m²)",
        examples=[80.0],
    )
    bedrooms: int = Field(
        ...,
        ge=1,
        le=10,
        alias="Bedrooms",
        description="Số phòng ngủ",
        examples=[3],
    )
    bathrooms: int | None = Field(
        default=None,
        ge=0,
        le=20,
        alias="Bathrooms",
        description="Số phòng vệ sinh / nhà tắm",
        examples=[2],
    )
    floors: int | None = Field(
        default=None,
        ge=0,
        le=100,
        alias="Floors",
        description="Số tầng",
        examples=[2],
    )
    width: float | None = Field(
        default=None,
        gt=0,
        le=100,
        alias="Width",
        description="Chiều rộng mặt tiền (m)",
        examples=[4.0],
    )
    length: float | None = Field(
        default=None,
        gt=0,
        le=200,
        alias="Length",
        description="Chiều dài / chiều sâu (m)",
        examples=[20.0],
    )
    alley_width: float | None = Field(
        default=None,
        ge=0,
        le=30,
        alias="Alley Width",
        description="Độ rộng hẻm trước nhà (m)",
        examples=[3.0],
    )
    latitude: float | None = Field(
        default=None,
        ge=10.3,
        le=11.2,
        alias="Latitude",
        description="Tọa độ vĩ độ (GPS Latitude tại TP.HCM)",
        examples=[10.7300],
    )
    longitude: float | None = Field(
        default=None,
        ge=106.3,
        le=107.0,
        alias="Longitude",
        description="Tọa độ kinh độ (GPS Longitude tại TP.HCM)",
        examples=[106.7000],
    )
    direction: str = Field(
        default="Không rõ",
        alias="Direction",
        description="Hướng nhà (Đông, Tây, Nam, Bắc, Đông Nam, ...)",
        examples=["Đông Nam"],
    )
    position: str = Field(
        default="Không rõ",
        alias="Position",
        description="Vị trí nhà (Trong hẻm, Đường chính, ...)",
        examples=["Trong hẻm"],
    )
    has_furniture: bool = Field(default=False, description="Cờ tiện ích: Đã có nội thất")
    car_alley: bool = Field(default=False, description="Cờ tiện ích: Hẻm xe hơi / ô tô vào được")
    near_market: bool = Field(default=False, description="Cờ tiện ích: Gần chợ / siêu thị")
    near_school: bool = Field(default=False, description="Cờ tiện ích: Gần trường học / đại học")
    is_urgent_sale: bool = Field(default=False, description="Cờ tiện ích: Chính chủ cần bán gấp")
    include_explanation: bool = Field(
        default=False,
        description="Cờ yêu cầu giải thích 5 đặc trưng SHAP quan trọng nhất (Mặc định False để tiết kiệm tài nguyên CPU)",
    )

    @field_validator("property_type")
    @classmethod
    def supported_type(cls, value: str) -> str:
        """Kiểm tra loại bất động sản có nằm trong danh mục hỗ trợ."""
        if value not in RESIDENTIAL_TYPES:
            raise ValueError(f"Loại bất động sản '{value}' chưa được hỗ trợ. Danh mục: {RESIDENTIAL_TYPES}")
        return value

    @field_validator("location_area")
    @classmethod
    def supported_area(cls, value: str) -> str:
        """Kiểm tra khu vực địa lý có nằm trong danh mục hỗ trợ."""
        if value not in SUPPORTED_AREAS:
            raise ValueError(f"Khu vực '{value}' chưa được hỗ trợ.")
        return value


class PredictionInterval(BaseModel):
    """Khoảng dự báo Conformal Prediction."""

    lower_bound_million: float = Field(..., description="Cận dưới khoảng dự báo conformal (triệu VND)")
    upper_bound_million: float = Field(..., description="Cận trên khoảng dự báo conformal (triệu VND)")
    target_coverage: float = Field(0.8, description="Mức độ bao phủ mục tiêu (0.8 = 80%)")


class ValuationResponse(BaseModel):
    """Thông tin định giá điểm trung tâm và khoảng dự báo."""

    point_estimate_million: float = Field(..., description="Giá dự báo điểm trung tâm (triệu VND)")
    prediction_interval: PredictionInterval


class MarketContextResponse(BaseModel):
    """Bối cảnh thị trường và đơn giá phân khúc cùng loại."""

    segment_median_unit_price_million_m2: float | None = Field(
        None, description="Trung vị đơn giá cùng phân khúc loại hình x quận/huyện (triệu VND/m²)"
    )
    comparable_median_price_million: float | None = Field(
        None, description="Trung vị giá của các bất động sản tương đồng (triệu VND)"
    )
    comparable_median_unit_price_million_m2: float | None = Field(
        None, description="Trung vị đơn giá của các bất động sản tương đồng (triệu VND/m²)"
    )


class ReliabilityResponse(BaseModel):
    """Đánh giá độ tin cậy phân rã đa chiều (Decomposed Reliability)."""

    overall: Literal["low", "medium", "high"] = Field(..., description="Độ tin cậy tổng thể")
    reliability_level: Literal["low", "medium", "high"] = Field(..., description="Mức độ tin cậy chuẩn hóa")
    input_completeness_score: float = Field(..., ge=0, le=100, description="Điểm hoàn thiện dữ liệu đầu vào (%)")
    domain_support: str = Field(..., description="Đánh giá thuộc phân phối huấn luyện (in_domain hoặc warning_ood)")
    interval_risk: str = Field(..., description="Mức độ rủi ro độ rộng khoảng dự báo (tight, moderate, wide_interval)")
    warnings: list[str] = Field(default_factory=list, description="Danh sách các cảnh báo OOD hoặc dữ liệu")


class ComparableProperty(BaseModel):
    """Thông tin bất động sản tương đồng từ tập dữ liệu tham chiếu."""

    property_type: str
    location_area: str
    area: float | None = None
    price_million: float | None = None
    unit_price_million_m2: float | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    floors: int | None = None
    distance_to_cbd_km: float | None = None
    similarity_score: float = 1.0


class ModelMetaResponse(BaseModel):
    """Thông số siêu dữ liệu của mô hình phục vụ."""

    version: str
    model_type: str = "ExtraTreesRegressor"
    target_formulation: str = "total_price"


class PredictionResponse(BaseModel):
    """Dữ liệu kết quả phản hồi định giá Price Intelligence hoàn chỉnh."""

    # Schema phân tầng hiện đại
    valuation: ValuationResponse | None = None
    market_context: MarketContextResponse | None = None
    reliability: ReliabilityResponse | None = None
    comparables: list[ComparableProperty] = Field(default_factory=list, description="Bất động sản tương đồng")
    explanation: dict[str, Any] | None = None
    model: ModelMetaResponse | None = None

    # Các trường phẳng tương thích ngược (Flat aliases)
    predicted_price_million: float = Field(..., description="Giá dự báo điểm trung tâm (triệu VND)")
    lower_bound_million: float = Field(..., description="Cận dưới khoảng dự báo conformal (triệu VND)")
    upper_bound_million: float = Field(..., description="Cận trên khoảng dự báo conformal (triệu VND)")
    confidence: Literal["low", "medium", "high"] = Field(
        ...,
        description="Chỉ báo heuristic độ tin cậy (low / medium / high)",
    )
    reliability_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Chỉ báo độ tin cậy chuẩn hóa",
    )
    model_version: str = Field(..., description="Phiên bản mô hình đang phục vụ")
    warnings: list[str] = Field(default_factory=list, description="Danh sách các cảnh báo")
    data_quality_score: float = Field(..., ge=0, le=100, description="Điểm hoàn thiện dữ liệu (0-100%)")
    input_completeness_score: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Điểm hoàn thiện dữ liệu đầu vào",
    )
    top_contributions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top 5 đặc trưng ảnh hưởng nhiều nhất (SHAP values)",
    )
    segment_median_unit_price_million_m2: float | None = Field(
        None, description="Trung vị đơn giá cùng phân khúc loại hình x quận/huyện (triệu VND/m²)"
    )
    disclaimer: str = Field(..., description="Cảnh báo pháp lý và miễn trừ trách nhiệm")


@app.get("/health", summary="Kiểm tra sức khỏe dịch vụ API", tags=["System"])
def health() -> dict[str, Any]:
    """Trả về trạng thái hoạt động của server và sự tồn tại của file mô hình."""
    model_loaded = MODEL_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "service": "HCMC Real Estate Price Intelligence API",
    }


@app.get("/model-info", summary="Xem thông tin chi tiết của mô hình", tags=["Model"])
def model_info() -> dict[str, Any]:
    """Trả về phiên bản mô hình, thuật toán, danh mục quận/huyện và phương pháp chia tập."""
    try:
        artifact = load_model()
        return {
            "model_version": artifact["version"],
            "model_type": artifact.get("model_type", "ExtraTreesRegressor"),
            "supported_areas": artifact.get("supported_areas", SUPPORTED_AREAS),
            "supported_property_types": artifact.get(
                "supported_property_types",
                RESIDENTIAL_TYPES,
            ),
            "split_protocol": artifact.get("split_protocol", "grouped temporal 60/15/10/15"),
            "prediction_interval_target_coverage": artifact.get("target_coverage", 0.8),
        }
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Không thể tải thông tin mô hình: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse, summary="Dự báo giá bất động sản", tags=["Prediction"])
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Tiếp nhận thông tin chi tiết bất động sản và trả về giá dự báo cùng khoảng tin cậy."""
    try:
        payload = request.model_dump(by_alias=True)
        include_explanation = payload.pop("include_explanation", False)
        return predict_one(payload, include_explanation=include_explanation)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Lỗi khi xử lý dự báo giá: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/explain", response_model=PredictionResponse, summary="Dự báo giá kèm giải thích SHAP", tags=["Prediction"])
def explain(request: PredictionRequest) -> dict[str, Any]:
    """Tiếp nhận thông tin bất động sản và trả về giá dự báo kèm top 5 đặc trưng SHAP (yêu cầu xử lý CPU cao hơn)."""
    try:
        payload = request.model_dump(by_alias=True)
        return predict_one(payload, include_explanation=True)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Lỗi khi xử lý dự báo giá và SHAP: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
