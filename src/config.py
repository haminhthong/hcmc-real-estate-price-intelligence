"""Cấu hình chung cho toàn bộ dự án HCMC Real Estate Price Intelligence.

Tệp này quản lý các đường dẫn hệ thống, phiên bản mô hình, danh mục bất động sản,
tập đặc trưng (features) và cấu hình logging chuẩn hóa.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình ghi nhật ký
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hcmc_price_intelligence")

# ---------------------------------------------------------------------------
# Đường dẫn thư mục, dữ liệu và mô hình
# ---------------------------------------------------------------------------
# Đường dẫn thư mục gốc của dự án
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Đường dẫn dữ liệu đầu vào
DATA_PATH: Path = ROOT_DIR / "data" / "sample" / "data_public_sample.csv"

# Đường dẫn lưu mô hình và báo cáo
MODEL_PATH: Path = ROOT_DIR / "models" / "price_model.joblib"
METRICS_PATH: Path = ROOT_DIR / "artifacts" / "metrics.json"
MODEL_COMPARISON_PATH: Path = ROOT_DIR / "artifacts" / "model_comparison.json"
ERROR_ANALYSIS_PATH: Path = ROOT_DIR / "artifacts" / "error_analysis.json"
DATA_CARD_PATH: Path = ROOT_DIR / "artifacts" / "data_card.json"

# Phiên bản mô hình, schema artifact và hạt giống ngẫu nhiên để tái lập kết quả
MODEL_VERSION: str = "1.1.0"
ARTIFACT_SCHEMA_VERSION: int = 2
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Danh mục loại bất động sản và khu vực được hỗ trợ tại TP.HCM
# ---------------------------------------------------------------------------
RESIDENTIAL_TYPES: list[str] = [
    "Nhà riêng",
    "Nhà mặt tiền",
    "Căn hộ chung cư",
    "Biệt thự liền kề",
    "Nhà biệt thự",
]

SUPPORTED_AREAS: list[str] = [
    "Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8", "Quận 10",
    "Quận 11", "Quận 12", "Quận Bình Tân", "Quận Bình Thạnh", "Quận Gò Vấp",
    "Quận Phú Nhuận", "Quận Tân Bình", "Quận Tân Phú", "TP. Thủ Đức",
    "Huyện Bình Chánh", "Huyện Cần Giờ", "Huyện Củ Chi", "Huyện Hóc Môn", "Huyện Nhà Bè", "Unknown",
]

# ---------------------------------------------------------------------------
# Danh sách đặc trưng phục vụ huấn luyện
# ---------------------------------------------------------------------------
# Các đặc trưng dạng số
NUMERIC_FEATURES: list[str] = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Width",
    "Length",
    "Alley Width",
    "Latitude",
    "Longitude",
    "distance_to_cbd_km",
    "days_from_train_reference",
    "input_completeness_score",
]

# Các đặc trưng phân loại
CATEGORICAL_FEATURES: list[str] = [
    "Property Type",
    "location_area",
    "Direction",
    "Position",
]

# Các cờ nhị phân trích xuất từ tiện ích và nội dung tin
FLAG_FEATURES: list[str] = [
    "has_furniture",
    "car_alley",
    "near_market",
    "near_school",
    "is_urgent_sale",
]

# Tổng hợp toàn bộ danh sách đặc trưng mô hình sử dụng
MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES
