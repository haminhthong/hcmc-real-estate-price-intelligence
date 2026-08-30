"""REST API cho hệ thống định giá bất động sản TP.HCM (FastAPI).

Ứng dụng cung cấp các HTTP Endpoints:
- GET /health: Kiểm tra trạng thái hoạt động của máy chủ và mô hình.
- GET /model-info: Tra cứu thông tin cấu hình, phiên bản và vùng hỗ trợ.
- POST /predict: Tiếp nhận thông tin bất động sản và trả về dự báo giá, khoảng tin cậy conformal, và phân tích SHAP.
"""

from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import MODEL_PATH, MODEL_VERSION, RESIDENTIAL_TYPES, SUPPORTED_AREAS, logger
from src.predict import load_model, predict_one

# Khởi tạo ứng dụng FastAPI với mô tả chuẩn Swagger OpenAPI
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

# Thêm middleware CORS cho phép kết nối từ web front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class PredictionResponse(BaseModel):
    """Dữ liệu kết quả phản hồi dự báo từ API."""

    predicted_price_million: float = Field(..., description="Giá dự báo điểm trung tâm (triệu VND)")
    lower_bound_million: float = Field(..., description="Cận dưới khoảng tin cậy conformal (triệu VND)")
    upper_bound_million: float = Field(..., description="Cận trên khoảng tin cậy conformal (triệu VND)")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Mức độ tin cậy của dự báo (low/medium/high)")
    model_version: str = Field(..., description="Phiên bản mô hình đang phục vụ")
    warnings: List[str] = Field(..., description="Danh sách các cảnh báo về dữ liệu hoặc phạm vi huấn luyện")
    data_quality_score: float = Field(..., ge=0, le=100, description="Điểm chất lượng dữ liệu đầu vào (0-100%)")
    top_contributions: List[Dict[str, Any]] = Field(..., description="Top 5 đặc trưng ảnh hưởng nhiều nhất (SHAP values)")
    segment_median_unit_price_million_m2: float | None = Field(
        None, description="Trung vị đơn giá cùng phân khúc loại hình x quận/huyện (triệu VND/m²)"
    )
    disclaimer: str = Field(..., description="Cảnh báo pháp lý và miễn trừ trách nhiệm")


@app.get("/health", summary="Kiểm tra sức khỏe dịch vụ API", tags=["System"])
def health() -> Dict[str, Any]:
    """Trả về trạng thái hoạt động của server và sự tồn tại của file mô hình."""
    model_loaded = MODEL_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "service": "HCMC Real Estate Price Intelligence API",
    }


@app.get("/model-info", summary="Xem thông tin chi tiết của mô hình", tags=["Model"])
def model_info() -> Dict[str, Any]:
    """Trả về phiên bản mô hình, thuật toán, danh mục quận/huyện và phương pháp chia tập."""
    try:
        artifact = load_model()
        return {
            "model_version": artifact["version"],
            "model_type": "RandomForestRegressor",
            "supported_areas": artifact.get("supported_areas", SUPPORTED_AREAS),
            "supported_property_types": artifact.get(
                "supported_property_types",
                RESIDENTIAL_TYPES,
            ),
            "split_protocol": artifact.get("split_protocol", "grouped temporal 64/16/20"),
            "prediction_interval_target_coverage": artifact.get("target_coverage", 0.8),
        }
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Không thể tải thông tin mô hình: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse, summary="Dự báo giá bất động sản", tags=["Prediction"])
def predict(request: PredictionRequest) -> Dict[str, Any]:
    """Tiếp nhận thông tin chi tiết bất động sản và trả về giá dự báo cùng các phân tích bổ trợ."""
    try:
        payload = request.model_dump(by_alias=True)
        return predict_one(payload)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Lỗi khi xử lý dự báo giá: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

