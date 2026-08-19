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
CSV → làm sạch/deduplicate → group split → feature pipeline → Random Forest
                                                ├→ FastAPI
                                                └→ Streamlit + SHAP
```

`src/` là nguồn sự thật duy nhất cho preprocessing, train và inference. `api/` và `app/` cùng gọi `src.predict`, tránh training-serving skew. Notebook trong `notebooks/` chỉ dùng khám phá/thí nghiệm, không cần mở để chạy sản phẩm.

## Chống data leakage

- Lọc và deduplicate trước khi chia dữ liệu.
- Chia theo `property_group_id`; test tự động xác nhận train/test không giao nhau.
- Imputer và one-hot encoder chỉ `fit` trên train thông qua `Pipeline`.
- `Price` không thuộc danh sách feature; test bảo vệ điều kiện này.
- Test set chỉ dùng báo cáo cuối, không dùng chọn mô hình.

## Mô hình và sai số

Notebook gốc so sánh Linear Regression, Random Forest và XGBoost. Source code dùng Random Forest làm baseline triển khai gọn, tái lập được. Sau `python -m src.train`, số liệu test (MAE, RMSE, R², MAPE) được ghi vào `artifacts/metrics.json`. Khoảng giá lấy từ phân vị 80% của sai số tuyệt đối trên log-price; đây là khoảng thực nghiệm, không phải khoảng tin cậy thống kê chính thức.

Kết quả hiện tại trên sample đi kèm còn yếu (xem chính xác trong `artifacts/metrics.json`), vì vậy model này chỉ chứng minh pipeline triển khai. Không dùng kết quả để ra quyết định mua/bán; cần huấn luyện lại trên dữ liệu đầy đủ và chỉ phát hành khi vượt baseline đã định trước.

## Giải thích và cảnh báo

Mỗi dự đoán trả 5 đóng góp SHAP lớn nhất, trung vị đơn giá cùng loại hình/khu vực, độ tin cậy và cảnh báo khi biến số nằm ngoài phạm vi train. Nếu SHAP không tương thích ở runtime, mã có fallback an toàn để API vẫn phục vụ; môi trường chuẩn trong `requirements.txt` dùng SHAP.

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

Dữ liệu mẫu nhỏ và có thiên lệch của nguồn tin đăng; địa chỉ thiếu/không đồng nhất, giá có thể là giá chào. Mô hình chưa dùng biến động thời gian, quy hoạch, pháp lý chi tiết hay dữ liệu giao dịch. Dự đoán ngoài phạm vi train cần được xem là độ tin cậy thấp.
