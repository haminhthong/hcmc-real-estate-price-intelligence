import json
from .config import METRICS_PATH, MODEL_PATH


def main():
    """Hiển thị kết quả đánh giá đã tạo trong lần huấn luyện gần nhất."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise SystemExit("Chưa có kết quả. Hãy chạy: python -m src.train")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
