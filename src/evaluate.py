"""Mô-đun hiển thị báo cáo đánh giá mô hình định giá bất động sản.

Tệp này đọc file `artifacts/metrics.json` được tạo ra trong quá trình huấn luyện
và hiển thị toàn bộ các chỉ số MAE, RMSE, R2, MAPE, Coverage trên màn hình console.
"""

import json
from .config import METRICS_PATH, MODEL_PATH, logger


def main() -> None:
    """Đọc và in báo cáo kết quả đánh giá đã lưu trong lượt huấn luyện gần nhất."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        logger.error("Chưa tìm thấy file mô hình hoặc file metrics.")
        raise SystemExit("Chưa có kết quả đánh giá. Hãy chạy huấn luyện trước: python -m src.train")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    logger.info("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (TRÊN TẬP TEST ĐỘC LẬP):")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

