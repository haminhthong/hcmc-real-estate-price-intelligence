# HCMC Real Estate Price Intelligence 🏠

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E?logo=scikit-learn)
![Pytest](https://img.shields.io/badge/Pytest-20%2B_Passed-0A9EDC?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

Ứng dụng **Machine Learning & Data Intelligence ước lượng giá đăng tham khảo cho bất động sản dân dụng tại TP.HCM** (bao gồm nhà riêng, nhà mặt tiền, căn hộ chung cư). Hệ thống ứng dụng **Grouped Temporal Split 60/15/10/15** (Train / Validation / Calibration / Test) để phân tách việc huấn luyện, chọn mô hình, và tính khoảng tin cậy **Conformal Prediction (mục tiêu 80% coverage)**, tích hợp tùy chọn giải thích đặc trưng qua **SHAP Explainer**, hỗ trợ giao diện REST API (FastAPI) và Dashboard (Streamlit).

---

## 🎯 Mục Tiêu Ứng Dụng & Business Use Cases

### Phù hợp sử dụng cho:
- **Người mua nhà**: Tham khảo khoảng giá phù hợp và mức độ biến động giá theo khu vực trước khi đi xem thực địa.
- **Người bán / Môi giới**: Kiểm tra giá đăng đề xuất có bị lệch so với phân khúc mặt bằng thị trường hay không.
- **Nhà phân tích dữ liệu**: Theo dõi đơn giá trung vị (triệu VND/m²) theo quận/huyện và loại hình bất động sản.

### KHÔNG dùng cho các mục đích:
- ❌ Phê duyệt khoản vay thế chấp ngân hàng.
- ❌ Thẩm định giá pháp lý hoặc giải quyết tranh chấp.
- ❌ Quyết định đầu tư tài chính tự động (Automated Trading / Investing).
- ❌ Định giá tài sản đảm bảo.

---

## 📊 Mô Hình & Kết Quả Đánh Giá Thực Nghiệm

So sánh mô hình **Random Forest Regressor** với **Median Baseline** trên tập Validation và đánh giá độc lập duy nhất một lần trên tập Test:

| Chỉ số Đánh Giá | Baseline (Median) | Random Forest | Kết Quả Thực Nghiệm |
| :--- | :---: | :---: | :--- |
| **MAE (Triệu VND)** | 6,325.0 | **4,947.8** | Cải thiện **23.5%** MAE so với baseline |
| **Median AE (Triệu VND)** | 4,800.0 | **3,120.0** | Sai số tuyệt đối trung vị |
| **RMSE (Triệu VND)** | 10,210.0 | **7,368.5** | Độ lệch bình phương trung bình |
| **R² Score** | -0.05 | **0.246** | Giải thích được ~24.6% biến thiên giá |
| **MAPE (%)** | 94.2% | **65.8%** | Phản ánh biến động lớn ở phân khúc giá thấp |
| **Conformal Coverage** | N/A | **75.3%** | Bao phủ thực nghiệm trên tập Test (Mục tiêu 80%) |

---

## ⚠️ Hạn Chế Của Hệ Thống (Known Limitations)

- **Dữ liệu là giá đăng tham khảo**: Nguồn dữ liệu từ tin đăng môi giới/chủ nhà, không phải giá giao dịch thực tế trên hợp đồng mua bán.
- **Dung lượng tập dữ liệu nhỏ**: Sau quy trình làm sạch và lọc ngoại lệ, dữ liệu huấn luyện có 727 mẫu. Một số phân khúc theo quận/loại hình có số mẫu nhỏ hơn 20.
- **Chỉ số R² và MAPE**: R² đạt 0.246 và MAPE khoảng 65.8%, cho thấy mô hình chủ yếu cung cấp khoảng giá định hướng phân khúc thay vì mức giá chính xác tuyệt đối.
- **Tỷ lệ bao phủ thực nghiệm**: Coverage đạt 75.34%, thấp hơn một chút so với mục tiêu 80% do kích thước tập calibration còn khiêm tốn.
- **Dữ liệu tọa độ GPS**: Một số bản ghi thiếu tọa độ chính xác, mô hình phải sử dụng imputer cho đặc trưng khoảng cách tới trung tâm.
- **Nhận diện bất động sản trùng**: Thuật toán gộp nhóm sử dụng chữ ký đặc trưng (địa chỉ chuẩn hóa, diện tích, mặt tiền, chiều dài, số phòng, tọa độ làm tròn), không phải mã định danh căn hộ chính thức.

---

## 📐 Kiến Trúc Triển Khai Kỹ Thuật

```text
[ Dữ liệu thô CSV ]
        │
        ▼
[ Clean Data & Privacy Anonymization ]
        │  ├── Mờ PII / Che SĐT ([PHONE]) / Làm tròn tọa độ (3 chữ số)
        │  └── Tạo khóa chữ ký tài sản (property_group_id)
        │
        ▼
[ Grouped Temporal Split (60 / 15 / 10 / 15) ]
        ├── Train Set (60%) ────────► Pipeline Fit (SimpleImputer + OneHot + RandomForest)
        ├── Validation Set (15%) ───► Chọn mô hình & Kiểm tra Quality Gate (RF vs Baseline)
        ├── Calibration Set (10%) ──► Conformal Quantile Calculation (Target 80% Coverage)
        └── Test Set (15%) ─────────► Đánh giá độc lập (MAE, Median AE, RMSE, R², Coverage)
                                              │
                                              ▼
                                    [ Model Artifact (.joblib) ]
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                            [ FastAPI REST API ]   [ Streamlit Dashboard ]
                             (http://localhost:8000) (http://localhost:8501)
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Môi trường phát triển Local

```bash
# Clone repository
git clone https://github.com/your-username/hcmc-real-estate-price-intelligence.git
cd hcmc-real-estate-price-intelligence

# Tạo và kích hoạt virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Cài đặt thư viện phát triển
pip install -r requirements-dev.txt
```

### 2. Huấn luyện mô hình & Chạy Test Suite

```bash
# Huấn luyện mô hình và cập nhật artifacts
python -m src.train

# Chạy toàn bộ 20+ unit và integration tests
pytest -v

# Kiểm tra code style linter
ruff check .
```

### 3. Khởi chạy REST API (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000 --workers 2
```
- OpenAPI Docs: `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`

### 4. Khởi chạy Web Dashboard (Streamlit)

```bash
streamlit run app/streamlit_app.py
```
- Dashboard URL: `http://localhost:8501`

### 5. Triển khai Container với Docker Compose

```bash
# Build và chạy ứng dụng trong container độc lập
docker compose build --no-cache
docker compose up -d

# Kiểm tra trạng thái container và healthcheck
docker compose ps
```

---

## 🔌 API Endpoints Summary

- `GET /health`: Kiểm tra sức khỏe dịch vụ API.
- `GET /model-info`: Xem thông tin phiên bản mô hình, vùng hỗ trợ và phương pháp chia tập.
- `POST /predict`: Dự báo giá điểm và khoảng tin cậy conformal (Mặc định không tính SHAP để tối ưu tốc độ CPU).
- `POST /explain`: Dự báo giá kèm phân tích top 5 đặc trưng ảnh hưởng mạnh nhất qua SHAP Explainer.

---

## 📝 Giấy Phép (License)

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
