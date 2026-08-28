# HCMC Real Estate Price Intelligence

Ứng dụng ước lượng **giá đăng tham khảo** cho nhà riêng, nhà mặt tiền, căn hộ và biệt thự tại TP.HCM. Dự án phục vụ người học/phân tích thị trường; kết quả không thay thế thẩm định chuyên nghiệp hoặc giá giao dịch thực tế.

## Chạy nhanh

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
pytest
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

API docs: `http://localhost:8000/docs`. Streamlit: `http://localhost:8501`.

## Kiến trúc

```text
CSV → kiểm tra chất lượng → deduplicate → grouped temporal split
                                      → baseline vs Random Forest
                                      → conformal interval
                                      ├→ FastAPI
                                      └→ Streamlit + SHAP
```

`src/` là nguồn sự thật duy nhất cho preprocessing, train và inference. `api/` và `app/` cùng gọi `src.predict`, tránh training-serving skew. Notebook trong `notebooks/` chỉ dùng khám phá/thí nghiệm, không cần mở để chạy sản phẩm.

## Chống data leakage

- Lọc và deduplicate trước khi chia dữ liệu.
- Chia train/calibration/test theo thời gian và `property_group_id`; test tự động xác nhận ba tập không giao nhau.
- Imputer và one-hot encoder chỉ `fit` trên train thông qua `Pipeline`.
- `Price` không thuộc danh sách feature; test bảo vệ điều kiện này.
- Calibration dùng để chọn mô hình và hiệu chỉnh khoảng giá; test chỉ dùng báo cáo cuối.

## Mô hình và sai số

Source code so sánh median baseline với Random Forest trên calibration set và chọn Random Forest theo MAE. Test set mới nhất theo thời gian đạt R² `0,246`, MAE khoảng `4.948` triệu VND và RMSE khoảng `7.368` triệu VND. MAPE còn cao (`65,83%`), cho thấy mô hình chưa ổn định ở các bất động sản giá thấp.

Khoảng giá dùng split conformal prediction trên calibration set với coverage mục tiêu 80%; coverage quan sát trên test là `75,34%`. Đây là kết quả thực nghiệm trên sample, chưa phải bảo đảm cho dữ liệu tương lai. Kết quả chi tiết nằm trong `artifacts/metrics.json`, `artifacts/model_comparison.json` và `artifacts/error_analysis.json`.

Kết quả hiện tại trên sample đi kèm còn yếu (xem chính xác trong `artifacts/metrics.json`), vì vậy model này chỉ chứng minh pipeline triển khai. Không dùng kết quả để ra quyết định mua/bán; cần huấn luyện lại trên dữ liệu đầy đủ và chỉ phát hành khi vượt baseline đã định trước.

## Giải thích và cảnh báo

Mỗi dự đoán trả 5 đóng góp SHAP lớn nhất, điểm chất lượng dữ liệu, trung vị đơn giá cùng loại hình/khu vực, độ tin cậy và cảnh báo khi biến số nằm ngoài phạm vi train hoặc thiếu tọa độ. Spatial features gồm latitude, longitude và khoảng cách Haversine tới trung tâm Quận 1.

## API

- `GET /health`
- `GET /model-info`
- `POST /predict`

Ví dụ:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"Property Type":"Nhà riêng","location_area":"Quận 7","Area":80,"Bedrooms":3}'
```

## Docker

```bash
docker build -t hcmc-price-api .
docker run --rm -p 8000:8000 hcmc-price-api
```

## Triển khai demo

Đẩy repository lên GitHub, tạo app trên Streamlit Community Cloud và chọn entrypoint `app/streamlit_app.py`. Có thể triển khai image Docker lên một dịch vụ container cho API rồi đặt link `/docs` tại đây.

- Demo Streamlit: _bổ sung sau khi deploy_
- API documentation: _bổ sung sau khi deploy_
- Ảnh/GIF demo: _bổ sung sau khi chụp giao diện_

## Hạn chế

Dữ liệu mẫu sau làm sạch có 727 tin nhà riêng và căn hộ; 25,58% chưa xác định được khu vực và chỉ 27,10% có tọa độ hợp lệ. Dữ liệu có thiên lệch của nguồn tin đăng và không phải giá giao dịch. Mô hình chưa dùng quy hoạch, pháp lý chi tiết, POI hay dữ liệu giao dịch. Dự đoán ngoài phạm vi train cần được xem là độ tin cậy thấp.
