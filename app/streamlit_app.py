from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from src.config import RESIDENTIAL_TYPES, SUPPORTED_AREAS
from src.predict import predict_one

st.set_page_config(
    page_title="HCMC Real Estate Price Intelligence",
    page_icon="🏠",
    layout="wide",
)
st.title("🏠 HCMC Real Estate Price Intelligence")
st.caption("Ước lượng giá đăng bất động sản dân dụng tại TP.HCM")

with st.form("prediction"):
    c1, c2, c3 = st.columns(3)
    with c1:
        property_type = st.selectbox("Loại bất động sản", RESIDENTIAL_TYPES)
        area_name = st.selectbox(
            "Quận/khu vực",
            [area for area in SUPPORTED_AREAS if area != "Unknown"],
        )
        area = st.number_input("Diện tích (m²)", 5.0, 2000.0, 80.0)
        bedrooms = st.number_input("Phòng ngủ", 1, 10, 3)
    with c2:
        bathrooms = st.number_input("Phòng tắm", 0, 20, 2)
        floors = st.number_input("Số tầng", 0, 100, 2)
        width = st.number_input("Chiều rộng (m)", 0.1, 100.0, 4.0)
        length = st.number_input("Chiều dài (m)", 0.1, 200.0, 20.0)
    with c3:
        alley = st.number_input("Độ rộng hẻm (m)", 0.0, 30.0, 3.0)
        direction = st.selectbox(
            "Hướng",
            [
                "Không rõ", "Đông", "Tây", "Nam", "Bắc",
                "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc",
            ],
        )
        position = st.selectbox("Vị trí", ["Không rõ", "Trong hẻm", "Đường chính"])
    amenities = st.multiselect(
        "Tiện ích/đặc điểm",
        ["Có nội thất", "Hẻm ô tô", "Gần chợ", "Gần trường", "Bán gấp"],
    )
    submitted = st.form_submit_button("Ước tính giá", type="primary")

if submitted:
    payload = {
        "Property Type": property_type,
        "location_area": area_name,
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Width": width,
        "Length": length,
        "Alley Width": alley,
        "Direction": direction,
        "Position": position,
        "has_furniture": "Có nội thất" in amenities,
        "car_alley": "Hẻm ô tô" in amenities,
        "near_market": "Gần chợ" in amenities,
        "near_school": "Gần trường" in amenities,
        "is_urgent_sale": "Bán gấp" in amenities,
    }
    try:
        result = predict_one(payload)
        price_column, range_column, confidence_column = st.columns(3)
        price_column.metric(
            "Giá ước tính",
            f"{result['predicted_price_million'] / 1000:,.2f} tỷ",
        )
        range_column.metric(
            "Khoảng dự kiến",
            f"{result['lower_bound_million'] / 1000:,.2f}–"
            f"{result['upper_bound_million'] / 1000:,.2f} tỷ",
        )
        confidence_labels = {"low": "THẤP", "medium": "TRUNG BÌNH", "high": "CAO"}
        confidence_column.metric(
            "Độ tin cậy",
            confidence_labels[result["confidence"]],
        )
        if result["warnings"]:
            st.warning("\n".join(result["warnings"]))
        segment_price = result["segment_median_unit_price_million_m2"]
        if segment_price is not None:
            st.info(
                f"Trung vị cùng phân khúc: {segment_price:,.1f} triệu/m²"
            )
        chart = pd.DataFrame(result["top_contributions"]).set_index("feature")
        st.subheader("5 đặc trưng đóng góp nhiều nhất (SHAP)")
        st.bar_chart(chart["shap_value"])
        st.caption(result["disclaimer"])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
