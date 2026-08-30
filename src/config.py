"""Cấu hình chung cho toàn bộ dự án HCMC Real Estate Price Intelligence.

Tệp này quản lý các đường dẫn hệ thống, phiên bản mô hình, danh mục bất động sản,
tập đặc trưng (features) và cấu hình logging chuẩn hóa.
"""

import logging
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Cấu hình Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hcmc_price_intelligence")

# ---------------------------------------------------------------------------
# Đường dẫn Thư mục và File Dữ liệu / Mô hình
# ---------------------------------------------------------------------------
# Đường dẫn thư mục gốc của dự án
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Đường dẫn dữ liệu đầu vào
DATA_PATH: Path = ROOT_DIR / "data" / "sample" / "data_public_sample.csv"

# Đường dẫn lưu trữ mô hình và các artifacts
MODEL_PATH: Path = ROOT_DIR / "models" / "price_model.joblib"
METRICS_PATH: Path = ROOT_DIR / "artifacts" / "metrics.json"
MODEL_COMPARISON_PATH: Path = ROOT_DIR / "artifacts" / "model_comparison.json"
ERROR_ANALYSIS_PATH: Path = ROOT_DIR / "artifacts" / "error_analysis.json"
DATA_CARD_PATH: Path = ROOT_DIR / "artifacts" / "data_card.json"

# Phiên bản mô hình và Random Seed cho tính lặp lại (reproducibility)
MODEL_VERSION: str = "1.0.0"
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Danh mục Loại Bất Động Sản & Khu Vực Được Hỗ Trợ Tại TP.HCM
# ---------------------------------------------------------------------------
RESIDENTIAL_TYPES: List[str] = [
    "Nhà riêng",
    "Nhà mặt tiền",
    "Căn hộ chung cư",
    "Biệt thự liền kề",
    "Nhà biệt thự",
]

SUPPORTED_AREAS: List[str] = [
    "Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8", "Quận 10",
    "Quận 11", "Quận 12", "Quận Bình Tân", "Quận Bình Thạnh", "Quận Gò Vấp",
    "Quận Phú Nhuận", "Quận Tân Bình", "Quận Tân Phú", "TP. Thủ Đức",
    "Huyện Bình Chánh", "Huyện Cần Giờ", "Huyện Củ Chi", "Huyện Hóc Môn", "Huyện Nhà Bè", "Unknown",
]

# ---------------------------------------------------------------------------
# Danh Sách Đặc Trưng (Features) Phục Vụ Huấn Luyện Mô Hình
# ---------------------------------------------------------------------------
# Các đặc trưng dạng số
NUMERIC_FEATURES: List[str] = [
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
    "listing_age_days",
    "data_quality_score",
]

# Các đặc trưng dạng phân loại (Categorical)
CATEGORICAL_FEATURES: List[str] = [
    "Property Type",
    "location_area",
    "Direction",
    "Position",
]

# Các đặc trưng cờ nhị phân (Binary Flags) trích xuất từ tiện ích/nội dung tin
FLAG_FEATURES: List[str] = [
    "has_furniture",
    "car_alley",
    "near_market",
    "near_school",
    "is_urgent_sale",
]

# Tổng hợp toàn bộ danh sách đặc trưng mô hình sử dụng
MODEL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES

