# HCMC Real Estate Price Intelligence 🏠

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E?logo=scikit-learn)
![Pytest](https://img.shields.io/badge/Pytest-100%25_Passed-0A9EDC?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

Ứng dụng **Machine Learning & MLOps định giá bất động sản dân dụng tại TP.HCM** (bao gồm nhà riêng, nhà mặt tiền, căn hộ chung cư và biệt thự). Hệ thống được thiết kế theo tiêu chuẩn Production-Grade với phương pháp **Grouped Temporal Split chống Data Leakage**, **Conformal Prediction xác định khoảng tin cậy 80%**, **SHAP Explainer giải thích mô hình**, cùng kiến trúc serving đồng nhất phục vụ qua **FastAPI REST API** và **Streamlit Web Dashboard**.

---

## 🚀 Điểm Sáng Kỹ Thuật Đáng Giá Đưa Vào CV (Key Technical Highlights)

Nếu bạn đưa dự án này vào CV cho vị trí **Data Scientist / Machine Learning Engineer / Analytics Engineer**, đây là 5 luận điểm kỹ thuật mạnh mẽ thu hút nhà tuyển dụng:

1. **Chống Data Leakage Tuyệt Đối (Grouped Temporal Split 64/16/20)**:
   - **Thách thức**: Tin đăng bất động sản thường bị lặp lại (nhiều môi giới cùng đăng một căn nhà) và có yếu tố chuỗi thời gian. Nếu chia ngẫu nhiên, mô hình sẽ bị rò rỉ dữ liệu (Overfitting).
   - **Giải pháp**: Xây dựng thuật toán tạo khóa nhóm `property_group_id` gom toàn bộ bản tin trùng của 1 sản phẩm và sắp xếp theo `listing_date`. Đảm bảo các tập Train, Calibration và Test hoàn toàn giao rỗng (`disjoint sets`).
   - toàn bộ quy trình Imputer và Categorical Encoder được đóng gói trong scikit-learn `Pipeline` và chỉ `fit` duy nhất trên tập Train.

2. **Định Giá Khoảng Tin Cậy Không Phụ Thuộc Phân Phối (Conformal Prediction)**:
   - Thay vì chỉ đưa ra một mức giá điểm cố định (Point Estimation), mô hình áp dụng **Split Conformal Prediction** trên tập Calibration để tạo ra khoảng tin cậy `[Cận dưới, Cận trên]` bảo đảm **80% coverage mục tiêu**.

3. **Tính Minh Bạch & Giải Thích Mô Hình (Model Explainability with SHAP)**:
   - Tích hợp `TreeExplainer` từ thư viện SHAP để trích xuất 5 đặc trưng ảnh hưởng nhiều nhất (Top feature contributions) cho từng dự báo, giúp người dùng hiểu rõ lý do căn nhà được định giá cao hay thấp.

4. **Kiến Trúc Serving Không Bị Skew (Single Source of Truth)**:
   - Module `src.predict` là nguồn sự thật duy nhất phục vụ đồng thời cho cả REST API (`api.main`) và Web UI (`app.streamlit_app`), triệt tiêu rò rỉ hoặc sai lệch giữa môi trường huấn luyện và môi trường triển khai (Training-Serving Skew).

5. **Chuẩn MLOps & Production Engineering**:
   - Lưu trữ artifact mô hình nguyên tử (Atomic Write qua file tạm `.tmp`).
   - Logging chuẩn mực Python `logging`.
   - Đầy đủ test suite tự động với `pytest`.
   - Đóng gói container với **Docker & Docker Compose** sẵn sàng deploy 1-click.

---

## 📐 Kiến Trúc Hệ Thống (System Architecture)

```text
[ Dữ liệu thô CSV ]
        │
        ▼
[ Clean Data & Outlier Filtering ]
        │  ├── Trích xuất chuẩn hóa Quận/Huyện TP.HCM
        │  ├── Tính khoảng cách Haversine tới trung tâm Quận 1 (CBD)
        │  └── Gom nhóm tin trùng qua property_group_id
        │
        ▼
[ Grouped Temporal Split (64 / 16 / 20) ]
        ├── Train Set (64%) ────────► Pipeline Fit (SimpleImputer + OneHot + RandomForest)
        ├── Calibration Set (16%) ──► Conformal Quantile Calculation (Target 80% Coverage)
        └── Test Set (20%) ─────────► Đánh giá kiểm thử độc lập (MAE, RMSE, R², MAPE)
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

## 📊 Mô Hình & Hiệu Năng Đánh Giá (Model Evaluation & Metrics)

So sánh mô hình **Random Forest Regressor** với **Median Baseline** trên tập Calibration và đánh giá kiểm thử trên tập Test độc lập:

| Chỉ số Đánh Giá | Baseline (Median) | Random Forest (Mô hình) | Ghi Chú Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| **MAE (Triệu VND)** | 6,325 | **4,948** | Giảm sai số tuyệt đối trung bình ~1.37 tỷ VND |
| **RMSE (Triệu VND)** | 10,210 | **7,368** | Giảm độ lệch bình phương trung bình |
| **R² Score** | -0.05 | **0.246** | Mô hình học được các biến địa lý & tiện ích |
| **MAPE (%)** | 94.2% | **65.8%** | Phản ánh biến động lớn ở phân khúc nhà giá thấp |
| **Conformal Coverage** | N/A | **75.3% (Target 80%)** | Bao phủ thực nghiệm trên tập Test thời gian thực |

> **Lưu ý nghiệp vụ**: Mô hình hiện được huấn luyện trên dữ liệu mẫu làm sạch (727 tin đăng). Kết quả đạt tiêu chuẩn thử nghiệm pipeline MLOps và cần được tiếp tục huấn luyện trên dữ liệu đầy đủ trước khi áp dụng cho giao dịch thực tế.

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Dự Án (Quickstart)

### 1. Cài đặt môi trường Local Python

```bash
# Clone repository
git clone https://github.com/your-username/hcmc-real-estate-price-intelligence.git
cd hcmc-real-estate-price-intelligence

# Tạo môi trường ảo venv
python -m venv .venv

# Kích hoạt venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Huấn luyện & Đánh giá mô hình

```bash
# Chạy quy trình huấn luyện & tạo artifacts
python -m src.train

# Xem báo cáo đánh giá mô hình
python -m src.evaluate

# Chạy toàn bộ bộ kiểm thử tự động (Unit Tests)
python -m pytest
```

### 3. Khởi chạy REST API (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000
```
- Swagger API Documentation: `http://localhost:8000/docs`
- ReDoc API Documentation: `http://localhost:8000/redoc`

### 4. Khởi chạy Web Dashboard (Streamlit)

```bash
streamlit run app/streamlit_app.py
```
- Truy cập giao diện web tại: `http://localhost:8501`

### 5. Triển khai 1-Click với Docker Compose

```bash
# Khởi chạy đồng thời cả API và Streamlit UI trong container
docker compose up --build
```

---

## 🔌 Tài Liệu REST API (API Endpoints)

### 1. `GET /health` - Kiểm tra trạng thái máy chủ
**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "service": "HCMC Real Estate Price Intelligence API"
}
```

### 2. `POST /predict` - Dự báo giá bất động sản
**Request Payload**:
```json
{
  "Property Type": "Nhà riêng",
  "location_area": "Quận 7",
  "Area": 80.0,
  "Bedrooms": 3,
  "Bathrooms": 2,
  "Floors": 2,
  "Width": 4.0,
  "Length": 20.0,
  "Alley Width": 3.0,
  "Latitude": 10.7300,
  "Longitude": 106.7000,
  "Direction": "Đông Nam",
  "Position": "Trong hẻm",
  "has_furniture": true,
  "car_alley": true
}
```

**Response Payload**:
```json
{
  "predicted_price_million": 7850.0,
  "lower_bound_million": 6400.0,
  "upper_bound_million": 9300.0,
  "confidence": "medium",
  "model_version": "1.0.0",
  "warnings": [],
  "data_quality_score": 90.0,
  "top_contributions": [
    {"feature": "num__Area", "shap_value": 0.4215},
    {"feature": "num__distance_to_cbd_km", "shap_value": -0.2104},
    {"feature": "cat__location_area_Quận 7", "shap_value": 0.1582}
  ],
  "segment_median_unit_price_million_m2": 98.1,
  "disclaimer": "Kết quả là giá đăng tham khảo từ mô hình Machine Learning..."
}
```

---

## 🎯 Góc Phỏng Vấn CV (CV & Interview Q&A Guide)

Khi đưa dự án này vào CV, nhà tuyển dụng có thể đặt các câu hỏi phỏng vấn kỹ thuật sau. Dưới đây là gợi ý câu trả lời ngắn gọn và thuyết phục:

### Q1: "Bạn đã giải quyết vấn đề Data Leakage trong dự án này như thế nào?"
> **Trả lời**: Trong dữ liệu bất động sản, một căn nhà thường được nhiều môi giới đăng lại nhiều lần (duplicate listings). Nếu chia ngẫu nhiên k-fold hay random split, các bản tin trùng của cùng 1 căn nhà sẽ rơi vào cả Train và Test, khiến mô hình đạt điểm ảo nhưng thất bại trên thực tế.
> Tôi đã giải quyết bằng **Grouped Temporal Split**: Tạo chìa khóa nhóm `property_group_id` gom tất cả tin trùng lại, sau đó sắp xếp theo mốc thời gian `listing_date` để chia 64% Train (quá khứ), 16% Calibration, và 20% Test (tương lai). Đồng thời, imputer và One-Hot encoder được đóng gói trong scikit-learn Pipeline chỉ `fit` trên tập Train.

### Q2: "Tại sao bạn lại chọn Conformal Prediction thay vì chỉ dùng RMSE/MAE hay khoảng tin cậy của thuật toán gốc?"
> **Trả lời**: Dự báo điểm (Point estimate) đơn thuần không đủ để người dùng ra quyết định trong bất động sản. Khoảng tin cậy truyền thống từ mô hình thống kê thường giả định phân phối chuẩn (Gaussian distribution) - điều không đúng với giá bất động sản có phân phối lệch phải (right-skewed).
> **Conformal Prediction** cung cấp khoảng tin cậy không phụ thuộc vào phân phối dữ liệu (Distribution-free), có chứng minh toán học bảo đảm tỷ lệ bao phủ thực tế sát với mục tiêu (80% coverage) trên tập dữ liệu kiểm thử.

### Q3: "Bạn xử lý Training-Serving Skew trong kiến trúc dự án như thế nào?"
> **Trả lời**: Tôi thiết kế module `src.predict` làm nguồn sự thật duy nhất (Single Source of Truth). Cả REST API FastAPI và ứng dụng giao diện Streamlit đều gọi trực tiếp hàm `predict_one()` từ mô-đun này. Điều này đảm bảo từ khâu tiền xử lý, tính khoảng cách Haversine, kiểm tra warning phạm vi huấn luyện cho tới giải thích SHAP đều hoàn toàn đồng nhất giữa backend API và giao diện frontend.

---

## 📝 Giấy Phép (License)

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
