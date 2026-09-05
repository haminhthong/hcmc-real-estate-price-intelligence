# HCMC Real Estate Price Intelligence Platform 🏠

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-F7931E?logo=scikit-learn)
![Pytest](https://img.shields.io/badge/Pytest-37_Passed-0A9EDC?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

> **Platform Positioning**:  
> **HCMC Real Estate Price Intelligence Platform** là hệ thống định giá bất động sản chống rò rỉ dữ liệu (leakage-aware), kết hợp kiểm chuẩn chia nhóm theo thời gian (grouped temporal validation), đặc trưng không gian địa lý và kết cấu, khoảng dự báo hiệu chuẩn phân phối tự do (conformal prediction intervals), rào chắn kiểm soát ngoại lai phân phối (OOD guardrails), giải thích đóng góp đặc trưng (SHAP) và đối chiếu bất động sản tương đồng (comparable-property context).

---

## 1. Problem & Business Boundaries

Hệ thống được thiết kế để giải quyết bài toán định giá tham khảo cho bất động sản nhà ở dân dụng tại TP.HCM (Nhà riêng, Nhà mặt tiền, Căn hộ chung cư, Biệt thự liền kề) dựa trên tin đăng trực tuyến thực tế:

- **Thực tế dữ liệu tin rao**: Dữ liệu thu thập từ các website môi giới là giá đăng (asking/listing price), **không phải giá giao dịch thực tế** (transacted price). Thị trường mang tính phân tán cao, nhiều tin đăng trùng lặp từ nhiều môi giới, phân phối giá lệch phải nặng (long-tail skew) và phương sai cực lớn ở phân khúc cao cấp (> 15 tỷ VND).
- **Mục tiêu nền tảng**: Cung cấp thông tin định giá tham khảo đa chiều (Price Intelligence) thay vì chỉ đưa ra một con số dự báo điểm đơn thuần. Hệ thống đồng thời tính toán **Khoảng Dự Báo (Prediction Interval)** có bảo đảm thống kê, cảnh báo dữ liệu ngoài phân phối huấn luyện (OOD) và trích xuất các bất động sản tương đồng đã niêm yết trong quá khứ.

### 📌 Business Use Cases & Phân Định Biên Giới Sử Dụng

| Phù Hợp Sử Dụng (In-Scope) | KHÔNG Dùng Cho (Out-of-Scope) |
| :--- | :--- |
| ✅ **Người mua nhà**: Tham khảo khoảng dao động giá hợp lý theo khu vực và kết cấu để chuẩn bị đàm phán. | ❌ Thẩm định giá pháp lý để cấp tín dụng / thế chấp ngân hàng |
| ✅ **Người bán / Môi giới**: Kiểm tra mức lệch giá niêm yết so với mặt bằng trung vị phân khúc và các căn tương đồng. | ❌ Giám định tranh chấp tài sản, phân chia thừa kế trước tòa án |
| ✅ **Nhà phân tích dữ liệu**: Đánh giá biến động đơn giá trung vị (triệu VND/m²) theo quận/huyện và cự ly tới CBD. | ❌ Thuật toán tự động hóa đặt lệnh đầu tư tài chính phái sinh |

---

## 2. Data Source & Limitations

- **Nguồn dữ liệu**: Bộ dữ liệu tin đăng bất động sản công khai tại TP.HCM (`data/sample/data_public_sample.csv`).
- **Quy mô mẫu**: 2,500 bản ghi thô $\rightarrow$ **707 bất động sản độc lập hợp lệ** sau quy trình lọc nhiễu thực địa và khử trùng nhóm chữ ký tài sản.
- **Thuộc tính quan sát**: Loại hình, địa chỉ, tọa độ GPS (Vĩ độ, Kinh độ), diện tích, số phòng ngủ, số phòng vệ sinh, số tầng, chiều rộng mặt tiền, chiều dài, độ rộng hẻm, hướng nhà, vị trí, tiện ích và ngày cập nhật tin đăng.
- **Giới hạn cố hữu**:
  1. *Thiếu nhãn giá giao dịch chốt hợp đồng*: Giá niêm yết thường cao hơn giá thanh toán thực tế từ 5% – 15% tùy thời điểm thị trường.
  2. *Độ thưa dữ liệu ở phân khúc hạng sang*: Số lượng căn có giá trị > 15 tỷ VND chiếm tỷ trọng nhỏ nhưng có biên độ dao động cực rộng.
  3. *Tọa độ GPS khuyết thiếu*: Một phần tin đăng không cung cấp tọa độ chính xác, cần thuật toán điền khuyết tự động.

---

## 3. System Architecture

Hệ thống được tách bạch hoàn toàn giữa **Offline Model Development** và **Online Serving Layer** nhằm triệt tiêu Data Leakage và Training-Serving Skew.

### A. Offline Model Development Pipeline

```text
               ┌────────────────────────────────────────┐
               │    Raw Property Listing Data (CSV)     │
               └───────────────────┬────────────────────┘
                                   │
                              Data Audit
                                   │
                      Cleaning & Property Grouping
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │         Feature Engineering            │
               │  Structural | Geospatial | Text Signals│
               └───────────────────┬────────────────────┘
                                   │
                        Grouped Temporal Split
          ┌────────────────┬───────┴────────┬────────────────┐
          │ (60%)          │ (15%)          │ (10%)          │ (15%)
          ▼                ▼                ▼                ▼
     Train Set       Validation Set   Calibration Set     Test Set
          │                │                │                │
          │ (Train Only)   │                │                │
          ├──────────────┐ │                │                │
          ▼              ▼ ▼                │                │
     Feature Fit     Model Selection        │                │
    & Reference     (Champion: ExtraTrees)  │                │
          │              │                  │                │
          └──────────────┼──────────────────┘                │
                         │                                   │
                         ▼                                   │
              Independent Conformal Calibration              │
              (Residual Quantile in Log-Space)               │
                         │                                   │
                         ▼                                   ▼
              Freeze Artifact (v1.1.0) ───────────► Final Test & Slices
```

### B. Online Serving Architecture

```text
               User / Web UI (Streamlit) / External Client
                                   │
                                   ▼
                        FastAPI Serving Layer
                                   │
                                   ▼
                      Input Schema Validation (Pydantic)
                                   │
                                   ▼
            Shared Feature Builder (Train Fixed Reference Date)
                                   │
                                   ▼
            OOD & Completeness Guardrails (P01–P99 Quantiles)
                                   │
                                   ▼
             ExtraTrees Regressor ──► Point Estimate (Price)
                                   │
                                   ▼
             Split Conformal Bounds ──► Asymmetric Prediction Interval
                                   │
                                   ▼
             Nearest Comparables Engine ──► 3–5 Similar Properties
                                   │
                                   ▼
             SHAP TreeExplainer ──► Top Feature Contributions (Log-Scale)
                                   │
                                   ▼
              Structured Price Intelligence Response (JSON)
```

---

## 4. Data Cleaning & Property Grouping Audit

### A. Bảng Truy Nguyên Kiểm Toán Dữ Liệu (Data Audit Trail)

Dữ liệu thô trải qua bộ lọc tuần tự có chính sách ngưỡng xác định trước (Domain-Driven Rules) độc lập với nhãn:

```text
Raw Rows: 2,500 bản ghi thô
   │
   ├─► Loại bỏ loại hình không hỗ trợ (Đất nền, kho xưởng...): -1,587 rows (63.48%)
   │
   ├─► Lọc bỏ ngoại lệ đo lường và giá phi thực tế: -186 rows (7.44%)
   │   • Price ∉ [100 triệu, 50 tỷ]
   │   • Area ∉ [5 m², 500 m²]
   │   • Đơn giá < 10 triệu/m²
   │   • Kích thước / phòng ngủ / tầng không khả thi
   │
   ├─► Kiểm tra và chuẩn hóa tọa độ GPS TP.HCM (Lat: 10.3–11.2, Lon: 106.3–107.0)
   │
   └─► Khử trùng theo Chữ Ký Bất Động Sản (Property Group Dedup): -20 rows (0.80%)
       │
       ▼
   707 Bản Ghi Hợp Lệ Sẵn Sàng Huấn Luyện (100% Unique Property Groups)
```

### B. Cơ Chế Nhận Diện & Tạo Khóa Nhóm `property_group_id`

Để giải quyết câu hỏi: *"Làm sao biết hai tin đăng là cùng một căn nhà thực tế?"*, hệ thống xây dựng chữ ký tài sản đa thuộc tính:

$$\text{property\_group\_id} = \text{Hash}\Big(\text{Location}_{\text{norm}} + \text{PropertyType}_{\text{norm}} + \lfloor\text{Area}\rceil_0 + \lfloor\text{Width}\rceil_1 + \lfloor\text{Length}\rceil_1 + \text{Beds} + \text{Baths} + \lfloor\text{Lat}\rceil_4 + \lfloor\text{Lon}\rceil_4\Big)$$

- Tọa độ GPS làm tròn 4 chữ số thập phân ($\approx 11\,\text{m}$) cho phép gộp các tin đăng cùng vị trí nhưng có sai số định vị nhỏ.
- Kích thước dài, rộng và diện tích làm tròn khử biến động do môi giới làm tròn số lẻ.

---

## 5. Feature Engineering

Toàn bộ đặc trưng được tính toán thông qua module dùng chung `src/feature_engineering.py:make_features(df, reference_date)`:

1. **Structural Features**:
   - `Area`, `Bedrooms`, `Bathrooms`, `Floors`, `Width`, `Length`, `Alley Width`.
2. **Geospatial Features**:
   - `Latitude`, `Longitude`.
   - `distance_to_cbd_km`: Khoảng cách Haversine đường chim bay từ tọa độ nhà tới Chợ Bến Thành (Quận 1: $10.7769^\circ\text{N}, 106.7009^\circ\text{E}$).
3. **Temporal Features & Khắc Phục Lỗi Serving Skew**:
   - `days_from_train_reference`: Độ chênh lệch ngày đăng so với mốc thời gian tin đăng muộn nhất của tập Train (`reference_date`).
   - Không áp dụng hàm `.clip(lower=0)` để tránh hiện tượng mọi listing mới phát sinh trong tương lai bị collapse về 0 khi inference.
4. **Input Quality Features**:
   - `input_completeness_score`: Tỷ lệ phần trăm (0 – 100%) các thuộc tính đo lường then chốt không bị khuyết (NaN hoặc "Không rõ").
5. **Text Signals (NLP Heuristic có xử lý phủ định)**:
   - Các cờ nhị phân: `has_furniture`, `car_alley` (hỗ trợ cả "hẻm xe hơi", "hẻm ô tô"), `near_market`, `near_school`, `is_urgent_sale`.
   - **Xử lý từ phủ định (Negation Handling)**: Sử dụng biểu thức chính quy nhận diện tiền tố phủ định tiếng Việt (`không|chưa|chẳng|ko|chua|khong` + `có|được`). Ví dụ: cụm từ *"nhà trống không có nội thất"* sẽ **không** bị kích hoạt cờ `has_furniture=1`.

---

## 6. Grouped Temporal Split Semantics

Dữ liệu được phân chia theo tỷ lệ **Train (60%) $\rightarrow$ Validation (15%) $\rightarrow$ Calibration (10%) $\rightarrow$ Test (15%)**:

```text
               Grouped Temporal Split (424 / 106 / 70 / 107)
 ┌───────────────────────────┬──────────────┬──────────────┬──────────────┐
 │ Train (60%)               │ Val (15%)    │ Calib (10%)  │ Test (15%)   │
 │ 424 mẫu                   │ 106 mẫu      │ 70 mẫu       │ 107 mẫu      │
 └───────────────────────────┴──────────────┴──────────────┴──────────────┘
 ◄────────────────────── Sắp xếp theo latest listing date per group ─────►
```

> ⚠️ **Làm Rõ Ngữ Nghĩa (Crucial Semantic Distinction)**:  
> Phân chia này **ưu tiên tính cô lập nhóm bất động sản (Group Isolation)** dựa trên ngày đăng muộn nhất quan sát được của nhóm (`latest listing_date per group`).  
> Đây **không phải là phân chia dòng thuần túy theo thời gian tuyệt đối (pure chronological row split)**. Nếu một căn nhà có các tin đăng trong quá khứ (ví dụ: tháng 1, tháng 3) và cập nhật gần nhất vào tháng 7, toàn bộ các tin đăng của căn nhà đó sẽ cùng đi vào tập dữ liệu theo mốc tháng 7. Thiết kế này giải quyết triệt để vấn đề rò rỉ chéo giữa các tập mà vẫn bảo đảm dữ liệu mới hơn nằm về phía kiểm chuẩn.

---

## 7. Baselines

Hệ thống thiết lập 2 mô hình đường cơ sở làm mốc đối chuẩn trước khi sử dụng các thuật toán học máy phức tạp:

1. **Naive Median Baseline**: Dự báo giá bằng trung vị toàn cục của tập Train ($\hat{y} = \text{Median}(Y_{\text{train}})$).
2. **Segment Appraisal Baseline**: Dự báo giá dựa trên bảng tra cứu trung vị đơn giá theo từng phân khúc cụ thể:
   $$\hat{y} = \text{Median}\big(\text{Price} \mid \text{Property Type} \times \text{District}\big)_{\text{train}}$$
   Nếu phân khúc chưa từng xuất hiện trong Train, hệ thống fallback về trung vị toàn cục.

---

## 8. Model & Target Benchmark

Hệ thống so sánh đồng thời 5 thuật toán Machine Learning cùng 2 baseline trên cả 2 cách đặt bài toán Target trên **tập Validation**:
- **Target Formulation A (Total Price)**: Huấn luyện hồi quy trên $\ln(1 + \text{Price})$, chuyển đổi lại $\hat{y} = \exp(\hat{y}_{\text{pred}}) - 1$.
- **Target Formulation B (Price / m²)**: Huấn luyện trên $\ln(1 + \text{Price} / \text{Area})$, sau đó nhân ngược lại diện tích $\hat{y} = (\exp(\hat{y}_{\text{pred}}) - 1) \times \text{Area}$.

### Kết Quả Benchmark Trên Tập Validation (Model Selection Split)

| Mô hình / Baseline | Target Formulation | Val MAE (Triệu VND) | Val Median AE | Val RMSE | Val R² | Val MAPE | Val WAPE | Vai trò |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Median** | Total Price | 4,149.1 | 2,200.0 | 7,687.4 | -0.136 | 54.6% | 52.05% | Baseline |
| **District x Type Segment Appraisal** | Segment Unit | 4,117.8 | 2,350.0 | 7,342.4 | -0.036 | 59.1% | 51.66% | Baseline |
| **Ridge Linear (Hedonic)** | Total Price | 6,010.7 | 1,556.5 | 20,247.8 | -6.878 | 67.2% | 75.40% | Candidate |
| **Random Forest** | Total Price | 3,745.8 | 1,424.0 | 6,498.1 | 0.188 | 63.8% | 47.00% | Candidate |
| **HistGradientBoosting** | Total Price | 5,216.9 | 3,886.9 | 7,380.9 | -0.047 | 98.6% | 65.45% | Candidate |
| **ExtraTrees Regressor 🏆** | **Total Price** | **3,760.5** | **2,155.5** | **6,010.5** | **0.306** | **63.6%** | **47.18%** | **Selected** |
| *ExtraTrees (Price / m²)* | Price / m² | 5,190.5 | 2,147.0 | 8,866.8 | -0.511 | 83.7% | 65.12% | Formulation B |

> 💡 **Thấu Cảm Kỹ Thuật (Target Formulation Insight)**:  
> Tại sao dự báo đơn giá Price/m² lại thất bại (MAE 5,190 triệu vs 3,760 triệu)?  
> Khi dự báo đơn giá theo mét vuông rồi nhân ngược lại với diện tích $y = \hat{y}_{\text{m2}} \times \text{Area}$, sai số dự báo của mô hình bị khuếch đại theo cấp số nhân đối với các bất động sản có diện tích lớn. Huấn luyện trực tiếp trên $\ln(1 + \text{Total Price})$ ổn định hơn rất nhiều.

---

## 9. Model Selection

Mô hình **ExtraTrees Regressor** trên bài toán **Total Price** được lựa chọn làm Champion Model triển khai phục vụ vì:
1. Đạt MAE nhỏ nhất trên tập Validation độc lập ($3,760.5$ triệu VND), cải thiện hơn $9.4\%$ so với Naive Median.
2. Giữ chỉ số $R^2$ dương cao nhất ($0.306$) trong toàn bộ các ứng viên.
3. Cấu trúc Ensembling đa cây ngẫu nhiên cực đại (Extremely Randomized Trees) có khả năng chống overfitting và phương sai tốt hơn trên tập dữ liệu kích thước nhỏ (707 mẫu).

---

## 10. Uncertainty Calibration (Split Conformal Prediction)

Hệ thống không giả định phân phối chuẩn của phần dư mà ứng dụng lý thuyết **Split Conformal Prediction** trên tập **Calibration (70 mẫu độc lập)**:

1. **Không gian tính toán phần dư**: Phần dư được tính toán chuẩn mực trong không gian log-target:
   $$e_i = \big|\ln(1 + y_i) - \ln(1 + \hat{y}_i)\big|, \quad \forall i \in \mathcal{D}_{\text{calib}}$$
2. **Phân vị mục tiêu (Coverage 80%)**: Xác định phân vị bảo thủ $q$:
   $$q = \text{Quantile}\left(e, \left\lceil \frac{(N_{\text{calib}} + 1) \times 0.8}{N_{\text{calib}}} \right\rceil\right)$$
3. **Khoảng Dự Báo Bất Đối Xứng (Asymmetric Monetary Prediction Interval)**:  
   Do tính lồi của hàm chuyển đổi $\exp(\cdot) - 1$, khoảng dự báo quy đổi về đơn vị tiền tệ VND là **bất đối xứng**:
   $$\text{Lower Bound} = \max\Big(\exp\big(\hat{y}_{\text{log}} - q\big) - 1, 0\Big)$$
   $$\text{Upper Bound} = \exp\big(\hat{y}_{\text{log}} + q\big) - 1$$
   *(Cận trên cách xa giá dự báo hơn so với cận dưới, phản ánh chính xác rủi ro phân phối giá nhà lệch phải).*

---

## 11. Final Test Evaluation

Đánh giá duy nhất một lần trên tập **Test độc lập (107 mẫu)** chưa từng tham gia fit tham số hay chọn mô hình:

| Chỉ số Đo Lường | Naive Median Baseline | ExtraTrees Model | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| **MAE (Triệu VND)** | 6,305.6 | **5,081.8** | Giảm sai số tuyệt đối trung bình **19.4%** |
| **Median AE (Triệu VND)** | 3,015.0 | **2,745.1** | Sai số trung vị thực địa |
| **RMSE (Triệu VND)** | 10,139.8 | **7,562.8** | Giảm đáng kể độ lệch bình phương |
| **$R^2$ Score** | -0.407 | **0.2170** | Giải thích được 21.7% biến thiên giá rao |
| **MAPE (%)** | 54.0% | **53.66%** | Tỷ lệ sai số phần trăm trung bình |
| **WAPE (%)** | 58.54% | **47.18%** | Sai số phần trăm có trọng số theo quy mô giá |
| **sMAPE (%)** | 60.62% | **46.59%** | Sai số phần trăm đối xứng |
| **Target Coverage** | N/A | **80.0%** | Mức độ bao phủ danh nghĩa đã hiệu chuẩn |
| **Actual Test Coverage** | N/A | **68.22%** | Tỷ lệ thực tế mẫu test nằm trong khoảng dự báo |
| **Median Interval Width** | N/A | **9,856.9M** | Độ rộng khoảng dự báo trung vị (~9.8 tỷ VND) |

---

## 12. Slice & Error Analysis

Phân tích hiệu năng chi tiết trên tập Test theo các phân khúc dữ liệu:

### 1. Theo Tầm Giá (Price Range Slice)

| Phân Khúc Giá | Số Mẫu | Mean MAE (Triệu VND) | Median MAE | Coverage (Target 80%) | Median Interval Width | Nhận Định |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **< 3 tỷ VND** | 8 | 2,612.3 | 2,041.1 | 25.0% | 5,558.1 triệu | Mẫu nhỏ, biến động diện tích lớn |
| **3 – 7 tỷ VND** | 40 | **2,596.0** | **1,379.3** | **82.5%** | **8,879.4 triệu** | **Ổn định nhất (Vượt target coverage)** |
| **7 – 15 tỷ VND** | 38 | 3,829.3 | 3,489.2 | **84.2%** | **11,715.9 triệu** | **Bao phủ tốt (Vượt target coverage)** |
| **> 15 tỷ VND** | 21 | 12,852.7 | 11,951.3 | 33.3% | 17,850.7 triệu | Biến động cực lớn, cần cảnh báo OOD |

### 2. Theo Độ Hoàn Thiện Dữ Liệu (`input_completeness_score`)

| Điểm Hoàn Thiện | Số Mẫu | Mean MAE (Triệu VND) | Median MAE | Coverage (Target 80%) | Ý Nghĩa Thực Nghiệm |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **>= 80% (Đầy đủ)** | 16 | **2,204.9** | **1,379.3** | **93.75%** | **Dữ liệu đầy đủ cho độ tin cậy và coverage rất cao** |
| **60 – 80% (Khá)** | 55 | 4,831.8 | 3,239.5 | 69.09% | Sai số ở mức trung bình |
| **< 60% (Thiếu nhiều)** | 35 | 6,785.2 | 4,275.1 | 60.00% | Sai số tăng vọt, kích hoạt cảnh báo tin cậy thấp |

---

## 13. OOD & Reliability Guardrails

Hệ thống triển khai cơ chế kiểm soát rủi ro đa tầng trước khi trả kết quả:

1. **Phân Vị Robust P01 – P99**: Thay vì min/max dễ nhạy cảm với outlier, hệ thống kiểm tra các thuộc tính số có nằm trong khoảng phân vị P01 đến P99 của tập Train hay không. Nếu vượt ngưỡng, hệ thống kích hoạt cảnh báo OOD.
2. **Cảnh Báo Ngoại Suy Hạng Sang (Luxury Segment Extrapolation)**: Nếu giá dự báo $> 15$ tỷ VND, hệ thống tự động cảnh báo mức độ hỗ trợ hiệu chuẩn hạn chế và gán mức tin cậy thấp.
3. **Độ Tin Cậy Phân Rã (Decomposed Reliability)**:
   - `overall`: `low` | `medium` | `high`
   - `input_completeness_score`: Điểm 0 - 100%
   - `domain_support`: `in_domain` hoặc `warning_ood`
   - `interval_risk`: `tight`, `moderate` hoặc `wide_interval`

---

## 14. SHAP Explanations

- Sử dụng **SHAP TreeExplainer** trích xuất 5 yếu tố có đóng góp lớn nhất vào dự báo giá.
- **Quy chuẩn hiển thị**: Tên đặc trưng thô được ánh xạ sang nhãn tiếng Việt thân thiện (ví dụ: `Diện tích đất (Area)`, `Khoảng cách tới Quận 1 (CBD)`, `Hẻm xe hơi`).
- **Ngữ nghĩa toán học**: Giá trị SHAP biểu thị đóng góp thống kê của đặc trưng trong **không gian log-target**, không biểu thị phép cộng tuyến tính số tiền VND trực tiếp và không đại diện cho mối quan hệ nhân quả tuyệt đối.

---

## 15. Comparable Properties Context

Hệ thống tích hợp **Comparable Properties Engine** tra cứu 3 – 5 bất động sản tương đồng đã từng niêm yết trong quá khứ từ tập tham chiếu Train:
- Tìm kiếm dựa trên khoảng cách hình học chuẩn hóa trên không gian kết cấu (Diện tích, Số phòng ngủ, Số phòng vệ sinh, Khoảng cách CBD), ưu tiên cùng loại hình và cùng quận/huyện.
- Trả về độ tương đồng (`similarity_score`), giá niêm yết, đơn giá và tính toán `comparable_median_price_million` giúp người dùng có góc nhìn so sánh thực tế tại địa bàn.

---

## 16. API & Web Dashboard

### A. Cấu Trúc Phản Hồi Chuẩn Mực (API Response Schema)

```json
{
  "valuation": {
    "point_estimate_million": 7850.0,
    "prediction_interval": {
      "lower_bound_million": 5420.0,
      "upper_bound_million": 11350.0,
      "target_coverage": 0.80
    }
  },
  "market_context": {
    "segment_median_unit_price_million_m2": 98.0,
    "comparable_median_price_million": 7600.0,
    "comparable_median_unit_price_million_m2": 95.0
  },
  "reliability": {
    "overall": "medium",
    "reliability_level": "medium",
    "input_completeness_score": 85.0,
    "domain_support": "in_domain",
    "interval_risk": "moderate",
    "warnings": []
  },
  "comparables": [
    {
      "property_type": "Nhà riêng",
      "location_area": "Quận 1",
      "area": 78.0,
      "price_million": 7500.0,
      "similarity_score": 0.94
    }
  ],
  "model": {
    "version": "1.1.0",
    "model_type": "extra_trees",
    "target_formulation": "total_price"
  }
}
```

### B. Khởi Chạy Dịch Vụ

- **FastAPI**:
  ```bash
  uvicorn api.main:app --reload --port 8000
  ```
  Tài liệu Swagger OpenAPI: `http://localhost:8000/docs`
- **Streamlit Web Dashboard**:
  ```bash
  streamlit run app/streamlit_app.py
  ```
  Giao diện trực quan: `http://localhost:8501`

---

## 17. Reproducibility & Docker Deployment

### 1. Cài Đặt Môi Trường Phát Triển Local

```bash
git clone https://github.com/haminhthong/hcmc-real-estate-price-intelligence.git
cd hcmc-real-estate-price-intelligence

# Khởi tạo virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Trên Windows PowerShell
# source .venv/bin/activate       # Trên Linux/macOS

pip install -r requirements-dev.txt
```

### 2. Huấn Luyện & Chạy Bộ Kiểm Thử (37 Tests Passed)

```bash
# Huấn luyện mô hình và kết xuất artifacts
python -m src.train

# Xem báo cáo đánh giá trên tập test độc lập
python -m src.evaluate

# Chạy toàn bộ 37 tests kiểm định chất lượng
python -m pytest -v
```

### 3. Docker Compose Khởi Chạy Song Song

```bash
docker compose build --no-cache
docker compose up -d
```

---

## 18. Known Limitations & Temporal Drift Considerations

1. **Dữ liệu tin rao $\neq$ Giá chốt giao dịch**: Cần nhân tử chiết khấu đàm phán khi áp dụng thực địa.
2. **Biến động trượt giá theo thời gian (Market Temporal Drift)**: Dữ liệu bất động sản phụ thuộc vào chu kỳ lãi suất và chính sách quy hoạch hạ tầng. Cần giám sát PSI (Population Stability Index) định kỳ hàng quý.
3. **Extrapolation Risk**: Phân khúc biệt thự > 15 tỷ VND có độ phân tán lớn, người dùng cần kiểm tra kỹ thực địa và tham khảo ý kiến chuyên gia thẩm định giá được cấp phép.

---

## 19. Project Roadmap

| Cấp Độ | Hạng Mục Đã Hoàn Thành & Kế Hoạch Tiếp Theo | Trạng Thái |
| :--- | :--- | :---: |
| 🔴 **P0** | Chuẩn hóa thuật ngữ **Prediction Interval**, loại bỏ hoàn toàn "Confidence Interval" | ✅ Hoàn thành |
| 🔴 **P0** | Unit test `test_conformal_residual_space_matches_inference_space` kiểm chứng log space | ✅ Hoàn thành |
| 🔴 **P0** | Đổi `listing_age_days` sang `days_from_train_reference` không clip lower=0 | ✅ Hoàn thành |
| 🔴 **P0** | Chuyển `confidence` sang `reliability_level` và phân rã Decomposed Reliability | ✅ Hoàn thành |
| 🔴 **P0** | Thuyết minh ngữ nghĩa Grouped Temporal Split ưu tiên cô lập nhóm | ✅ Hoàn thành |
| 🟠 **P1** | Xây dựng động cơ tìm kiếm bất động sản tương đồng (**Comparable Properties Engine**) | ✅ Hoàn thành |
| 🟠 **P1** | Bổ sung rào chắn **Robust Quantile OOD Bounds (P01–P99)** và cảnh báo Luxury | ✅ Hoàn thành |
| 🟠 **P1** | Thêm chỉ số chuẩn hóa WAPE, sMAPE và mở rộng lát cắt phân tích sai số (Completeness, CBD) | ✅ Hoàn thành |
| 🟠 **P1** | Tái cấu trúc **API Response Schema** phân tầng chuẩn Price Intelligence Platform | ✅ Hoàn thành |
| 🟠 **P1** | Xử lý từ phủ định (Negation Handling) cho cờ tiện ích văn bản | ✅ Hoàn thành |
| 🟡 **P2** | Xây dựng chỉ số giá quận/huyện theo thời gian lịch sử (District Price Lag Index) | 📋 Kế hoạch |
| 🟡 **P2** | Kiểm chuẩn địa lý Leave-One-District-Out đo lường khả năng chuyển giao vùng | 📋 Kế hoạch |
| 🟡 **P2** | Hiệu chuẩn Mondrian Conformal riêng biệt cho từng cụm giá | 📋 Kế hoạch |
| 🟡 **P3** | Tích hợp lớp dữ liệu GIS / POI không gian địa lý (trường học, metro, bệnh viện) | 📋 Kế hoạch |

---

## 📝 License

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
