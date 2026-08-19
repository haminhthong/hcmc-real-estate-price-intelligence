from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample" / "data_public_sample.csv"
MODEL_PATH = ROOT / "models" / "price_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42

RESIDENTIAL_TYPES = ["Nhà riêng", "Nhà mặt tiền", "Căn hộ chung cư", "Biệt thự liền kề", "Nhà biệt thự"]
SUPPORTED_AREAS = [
    "Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8", "Quận 10",
    "Quận 11", "Quận 12", "Quận Bình Tân", "Quận Bình Thạnh", "Quận Gò Vấp",
    "Quận Phú Nhuận", "Quận Tân Bình", "Quận Tân Phú", "TP. Thủ Đức",
    "Huyện Bình Chánh", "Huyện Cần Giờ", "Huyện Củ Chi", "Huyện Hóc Môn", "Huyện Nhà Bè", "Unknown",
]

NUMERIC_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "Width", "Length", "Alley Width"]
CATEGORICAL_FEATURES = ["Property Type", "location_area", "Direction", "Position"]
FLAG_FEATURES = ["has_furniture", "car_alley", "near_market", "near_school", "is_urgent_sale"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES
