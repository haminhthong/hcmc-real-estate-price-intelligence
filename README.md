# HCMC Real Estate Price Intelligence 🏠

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-F7931E?logo=scikit-learn)
![Pytest](https://img.shields.io/badge/Pytest-30_Passed-0A9EDC?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

> **Core Philosophy**: *Reliable housing price intelligence under limited & noisy listing data.*  
> Dự án không cố làm đẹp các chỉ số dự báo mà công khai minh bạch các giới hạn của dữ liệu tin đăng, đồng thời ứng dụng **Split Conformal Prediction** để cung cấp khoảng tin cậy kèm chỉ báo mức độ rủi ro (Low / Medium / High Confidence) cho từng khoản định giá.

---

## 🎯 Context & Positioning

Hệ thống định giá bất động sản dân dụng (Nhà riêng, Nhà mặt tiền, Căn hộ chung cư) tại TP.HCM trên dữ liệu thực tế:

- **Thực tế dữ liệu**: Giá rao tin trên website không phải là giá giao dịch thực tế (transaction price), dữ liệu nhỏ (707 mẫu hợp lệ sau làm sạch), phân phối lệch mạnh với biến động cực lớn ở phân khúc cao cấp (> 15 tỷ VND).
- **Mục tiêu hệ thống**: Cung cấp khoảng định giá tham khảo kèm phân tích rủi ro và các yếu tố ảnh hưởng chính (SHAP values) giúp người dùng đưa ra quyết định thực địa thận trọng.

### 📌 Business Use Cases & Boundary Limits

| Phù hợp sử dụng | KHÔNG dùng cho |
| :--- | :--- |
| ✅ **Người mua nhà**: Tham khảo khoảng giá hợp lý và khung dao động theo khu vực. | ❌ Thẩm định giá vay thế chấp ngân hàng |
| ✅ **Người bán / Môi giới**: Kiểm tra mức lệch giá đăng so với phân khúc cùng loại. | ❌ Thẩm định pháp lý / Tranh chấp tài sản |
| ✅ **Nhà phân tích**: Đánh giá đơn giá trung vị (triệu VND/m²) theo quận/huyện. | ❌ Tự động hóa quyết định đầu tư tài chính |

---

## 🔬 ML Protocol, Data Leakage & Serving Skew Fixes

### 1. Khắc Phục Lỗi Training-Serving Skew & Temporal Data Leakage

- **Vấn đề cũ**: Hàm tạo đặc trưng cũ dùng `out["listing_date"].max()` làm mốc tính `listing_age_days`. Khi train trên dataset lớn có sự biến thiên tuổi tin đăng, nhưng khi phục vụ API (gửi 1 bản ghi), `max()` bằng chính ngày của bản ghi đó $\Rightarrow$ `listing_age_days = 0` trên mọi request (Training-Serving Skew). Ngoài ra, tính `.max()` toàn bộ dataset trước khi split rò rỉ thông tin tương lai (Data Leakage).
- **Giải pháp triệt để**: Xác định mốc `reference_date` cố định **chỉ từ tập Train** (`df_train["listing_date"].max()`) và lưu vào artifact mô hình (`models/price_model.joblib`). Tất cả quy trình Feature Engineering (kể cả train, validation, test và inference API) đều dùng chung mốc `reference_date` này.

### 2. Protocol Chia Tập Dữ Liệu

- **Grouped Temporal Split based on latest listing date per property group (60/15/10/15)**:
  1. Nhóm bản ghi trùng lặp theo `property_group_id` (chữ ký bất động sản).
  2. Sắp xếp các nhóm theo ngày rao tin muộn nhất (`listing_date.max()`).
  3. Chia nhóm theo thứ tự thời gian: **Train (60%)** $\rightarrow$ **Validation (15%)** $\rightarrow$ **Calibration (10%)** $\rightarrow$ **Test (15%)**.
  4. Kiểm tra đảm bảo 4 tập hoàn toàn giao rỗng (Disjoint sets), chống rò rỉ tin đăng trùng giữa các tập.

---

## 📊 Benchmark Multi-Model & Target Formulation Experiments

Hệ thống tiến hành so sánh 5 thuật toán học máy cùng 2 baseline trung vị trên **tập Validation**, kiểm thử cả 2 dạng đặt bài toán (Target Formulations):
- **Formulation A (Total Price)**: $y = \ln(1 + \text{Price})$
- **Formulation B (Price / m²)**: $y = \ln(1 + \text{Price} / \text{Area})$, quy đổi lại $y_{\text{total}} = \hat{y}_{\text{m2}} \times \text{Area}$

### 1. Kết Quả Benchmark Trên Tập Validation (Model Selection Split)

| Algorithm / Baseline | Target Formulation | Val MAE (Triệu VND) | Val Median AE | Val RMSE | Val R² | Val MAPE | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Median** | Total Price | 4,149.1 | 2,200.0 | 7,687.4 | -0.136 | 54.6% | Baseline |
| **District x Type Segment Median** | Segment Unit | 4,117.8 | 2,350.0 | 7,342.4 | -0.036 | 59.1% | Baseline |
| **Ridge Linear Regression** | Total Price | 6,010.7 | 1,556.5 | 20,247.8 | -6.878 | 67.2% | Candidate |
| **Random Forest Regressor** | Total Price | 3,718.2 | 1,452.0 | 6,482.7 | 0.192 | 63.4% | Candidate |
| **HistGradientBoosting** | Total Price | 5,216.9 | 3,886.9 | 7,380.9 | -0.047 | 98.6% | Candidate |
| **ExtraTrees Regressor 🏆** | **Total Price** | **3,711.7** | **2,155.5** | **6,010.5** | **0.306** | **63.6%** | **Deployed** |
| *ExtraTrees (Price/m²)* | Price / m² | 5,180.4 | 2,181.8 | 8,965.8 | -0.545 | 81.7% | Target Exp B |

> 📌 **Kết luận Model Selection**: Mô hình **ExtraTrees Regressor** trên **Total Price formulation** đạt MAE nhỏ nhất trên tập Validation (3,711.7 triệu VND, R² = 0.306), vượt trội hơn so với baseline trung vị và bài toán Price/m².

### 2. Đánh Giá Báo Cáo Độc Lập Trên Tập Test (Test Report Only)

Mô hình được chọn được đánh giá kiểm thử duy nhất một lần trên tập **Test (107 mẫu)**:

| Chỉ số Đánh Giá | Naive Baseline | ExtraTrees (Deployed) | Ý Nghĩa Thực Nghiệm |
| :--- | :---: | :---: | :--- |
| **MAE (Triệu VND)** | 6,305.6 | **5,034.6** | Cải thiện **20.2%** MAE so với baseline |
| **Median AE (Triệu VND)** | 3,015.0 | **2,838.1** | Sai số tuyệt đối trung vị |
| **RMSE (Triệu VND)** | 10,139.8 | **7,583.0** | Độ lệch chuẩn bình phương sai số |
| **R² Score** | -0.407 | **0.2129** | Giải thích được **21.3%** biến thiên giá tin rao |
| **MAPE (%)** | 54.0% | **53.34%** | Tỷ lệ sai số phần trăm tuyệt đối trung bình |
| **Conformal Target Coverage** | N/A | **80.0%** | Ngưỡng bao phủ mục tiêu cài đặt trên Calibration |
| **Actual Test Coverage** | N/A | **68.22%** | Tỷ lệ thực tế mẫu Test nằm trong khoảng dự báo |
| **Median Interval Width** | N/A | **10,626.9M** | Độ rộng trung vị của khoảng tin cậy (10.6 tỷ VND) |

---

## 🔍 In-Depth Conformal Coverage & Error Analysis by Slice

Phân tích chi tiết độ bao phủ (Coverage) và độ rộng khoảng tin cậy Conformal Prediction trên tập Test theo từng phân khúc:

### 1. Theo Khoảng Giá (Price Range Slice)

| Khoảng Giá | Số Mẫu | Mean MAE (Triệu VND) | Median MAE | Coverage (Target 80%) | Median Interval Width | Insight Miền |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **< 3 tỷ VND** | 8 | 2,805.1 | 2,250.4 | 25.0% | 5,773.7 triệu | Mẫu nhỏ, biến động giá/m² lớn |
| **3 – 7 tỷ VND** | 40 | **2,573.8** | **1,550.4** | **80.0%** | **8,844.0 triệu** | **Phân khúc ổn định nhất (Đạt target coverage)** |
| **7 – 15 tỷ VND** | 38 | 3,712.6 | 3,419.8 | **84.2%** | **11,386.5 triệu** | **Bao phủ tốt (Vượt target coverage 80%)** |
| **> 15 tỷ VND** | 21 | 12,963.5 | 11,748.8 | 33.3% | 17,505.2 triệu | Biến động cực cao ở phân khúc biệt thự/hạng sang |

### 2. Theo Loại Hình Bất Động Sản (Property Type Slice)

| Loại Hình | Số Mẫu | Mean MAE (Triệu VND) | Median MAE | Coverage | Median Interval Width |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Căn hộ chung cư** | 3 | 1,289.2 | 1,745.1 | 66.7% | 4,202.8 triệu |
| **Nhà riêng** | 104 | 5,142.7 | 2,887.7 | 68.3% | 10,643.6 triệu |

---

## 🏗️ Streamlit Interactive UI & Architecture Flow

```text
               ┌──────────────────────────────────────────────┐
               │    Input Property Parameters (Streamlit/API) │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │ Feature Engineering (Fixed Reference Date)   │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │  ExtraTrees Model Point Estimate Prediction  │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │  Conformal Interval [Lower Bound, Upper]     │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │  SHAP TreeExplainer Top 5 Feature Factors    │
               └──────────────────────────────────────────────┘
```

---

## ⚠️ Hạn Chế Của Dự Án (Known Limitations)

1. **Giá đăng rao tin $\neq$ Giá giao dịch thực tế**: Dữ liệu thu thập từ các bài đăng trên cổng thông tin bất động sản, chưa phản ánh mức chiết khấu thương lượng thực tế.
2. **Quy mô tập dữ liệu khiêm tốn**: Sau khi lọc ngoại lệ số và gộp nhóm trùng lặp, dataset có **707 mẫu**. Phân khúc căn hộ chung cư và một số huyện ngoại thành có số mẫu ít.
3. **Phân phối lệch đuôi dài (Long-tail skew)**: Bất động sản cao cấp (> 15 tỷ) có phương sai lớn khiến độ rộng khoảng tin cậy mở rộng (~10-17 tỷ VND) và giảm coverage ở phân khúc này.
4. **Coverage Thực Nghiệm**: Đạt **68.22%** trên tập test (mục tiêu 80%) do quy mô tập calibration 70 mẫu.

---

## 🛠️ Hướng Dẫn Cài Đặt & Khởi Chạy Dự Án

### 1. Môi trường phát triển Local

```bash
# Clone repository chính thức
git clone https://github.com/haminhthong/hcmc-real-estate-price-intelligence.git
cd hcmc-real-estate-price-intelligence

# Tạo virtual environment
python -m venv .venv

# Activate venv:
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements-dev.txt
```

### 2. Huấn Luyện Mô Hình & Chạy Test Suite

```bash
# Huấn luyện mô hình, thực nghiệm target & xuất artifacts
python -m src.train

# Kiểm tra báo cáo đánh giá trên tập test
python -m src.evaluate

# Chạy toàn bộ test suite (30 passed cleanly)
python -m pytest -v
```

### 3. Khởi Chạy Services (API & Dashboard)

- **FastAPI REST API**:
  ```bash
  uvicorn api.main:app --reload --port 8000
  ```
  OpenAPI Docs: `http://localhost:8000/docs`

- **Streamlit Web Dashboard**:
  ```bash
  streamlit run app/streamlit_app.py
  ```
  Dashboard URL: `http://localhost:8501`

- **Docker Compose**:
  ```bash
  docker compose build --no-cache
  docker compose up -d
  ```

---

## 🔌 API Endpoints Summary

- `GET /health`: Kiểm tra trạng thái hoạt động dịch vụ và nạp mô hình.
- `GET /model-info`: Xem thông số mô hình, mốc `reference_date`, phương pháp chia tập và vùng hỗ trợ.
- `POST /predict`: Dự báo giá điểm, khoảng tin cậy Conformal Prediction và chỉ báo độ tin cậy.
- `POST /explain`: Dự báo giá kèm trích xuất top 5 đặc trưng ảnh hưởng mạnh nhất qua SHAP Explainer.

---

## 📝 License

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
