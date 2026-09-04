"""Ứng dụng Web Dashboard Định Giá Bất Động Sản TP.HCM (Streamlit).

Được thiết kế hiện đại với bố cục tab, hiển thị dự báo giá điểm trung tâm,
khoảng tin cậy Conformal Prediction (coverage 80%), cảnh báo tính toàn vẹn dữ liệu,
và đồ thị giải thích SHAP cho từng kết quả định giá.
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import RESIDENTIAL_TYPES, SUPPORTED_AREAS
from src.predict import load_model, predict_one

# ---------------------------------------------------------------------------
# Cấu hình trang Streamlit và kiểu hiển thị
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HCMC Real Estate Price Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Kiểu CSS riêng cho giao diện
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
    </style>
    """,
    unsafe_allow_kwargs={"allow_html": True},
)

# Tiêu đề chính
st.markdown('<div class="main-header">🏠 HCMC Real Estate Price Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hệ thống ước lượng giá đăng bất động sản dân dụng tại TP.HCM dựa trên Machine Learning & Conformal Prediction</div>',
    unsafe_allow_html=True,
)

# Tạo các thẻ chức năng
tab_predict, tab_shap, tab_info = st.tabs([
    "🏠 Dự Báo Giá",
    "📊 Phân Tích SHAP & Thị Trường",
    "ℹ️ Thông Tin Mô Hình & MLOps",
])

# ---------------------------------------------------------------------------
# TAB 1: DỰ BÁO GIÁ
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Nhập thông tin bất động sản cần định giá")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Thông tin cơ bản**")
            property_type = st.selectbox("Loại bất động sản (*)", RESIDENTIAL_TYPES)
            area_name = st.selectbox(
                "Quận/huyện khu vực (*)",
                [area for area in SUPPORTED_AREAS if area != "Unknown"],
            )
            area = st.number_input("Diện tích đất/sử dụng (m²) (*)", 5.0, 2000.0, 80.0, step=5.0)
            bedrooms = st.number_input("Số phòng ngủ (*)", 1, 10, 3)

        with col2:
            st.markdown("**Thông số kích thước & kết cấu**")
            bathrooms = st.number_input("Số phòng vệ sinh", 0, 20, 2)
            floors = st.number_input("Số tầng", 0, 100, 2)
            width = st.number_input("Chiều rộng mặt tiền (m)", 0.1, 100.0, 4.0, step=0.5)
            length = st.number_input("Chiều dài / chiều sâu (m)", 0.1, 200.0, 20.0, step=1.0)

        with col3:
            st.markdown("**Vị trí & Tiện ích**")
            alley = st.number_input("Độ rộng hẻm trước nhà (m)", 0.0, 30.0, 3.0, step=0.5)
            direction = st.selectbox(
                "Hướng nhà",
                [
                    "Không rõ", "Đông", "Tây", "Nam", "Bắc",
                    "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc",
                ],
            )
            position = st.selectbox(
                "Vị trí",
                ["Không rõ", "Trong hẻm", "Đường chính"],
            )

        st.markdown("**Đặc điểm nổi bật / Cờ tiện ích**")
        amenities = st.multiselect(
            "Chọn các tiện ích đi kèm:",
            ["Có nội thất", "Hẻm ô tô", "Gần chợ", "Gần trường", "Bán gấp"],
            default=["Hẻm ô tô"],
        )

        submitted = st.form_submit_button("🔍 Thực Hiện Định Giá", type="primary", use_container_width=True)

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
            result = predict_one(payload, include_explanation=True)
            st.session_state["last_result"] = result

            st.divider()
            st.subheader("🎯 Kết Quả Ước Lượng Giá")

            res_c1, res_c2, res_c3 = st.columns(3)

            price_billion = result["predicted_price_million"] / 1000
            lower_billion = result["lower_bound_million"] / 1000
            upper_billion = result["upper_bound_million"] / 1000

            res_c1.metric(
                label="Giá Ước Tính (Dự Báo Điểm)",
                value=f"{price_billion:,.2f} tỷ VND",
                help="Giá trị trung tâm được dự báo từ mô hình Extra Trees.",
            )

            res_c2.metric(
                label="Khoảng Tin Cậy (Coverage 80%)",
                value=f"{lower_billion:,.2f} – {upper_billion:,.2f} tỷ",
                help="Khoảng giá dự kiến chứa 80% trường hợp thực tế nhờ Conformal Prediction.",
            )

            confidence_labels = {
                "low": "🔴 THẤP (Cần thận trọng)",
                "medium": "🟡 TRUNG BÌNH",
                "high": "🟢 CAO",
            }
            res_c3.metric(
                label="Mức Độ Tin Cậy Mô Hình",
                value=confidence_labels[result["confidence"]],
                help="Dựa trên biên độ khoảng tin cậy và sự đầy đủ của dữ liệu đầu vào.",
            )

            # Thanh điểm chất lượng dữ liệu
            st.progress(
                result["data_quality_score"] / 100,
                text=f"Điểm Đầy Đủ Dữ Liệu Đầu Vào: {result['data_quality_score']:.0f}/100%",
            )

            # Cảnh báo nếu có
            if result["warnings"]:
                with st.expander("⚠️ Danh Sách Cảnh Báo Tính Hợp Lệ Dữ Liệu", expanded=True):
                    for warning in result["warnings"]:
                        st.warning(warning)

            # Thông tin đơn giá phân khúc
            segment_price = result["segment_median_unit_price_million_m2"]
            if segment_price is not None:
                st.info(
                    f"💡 **Tham khảo thị trường**: Trung vị đơn giá cùng phân khúc "
                    f"({property_type} tại {area_name}) là **{segment_price:,.1f} triệu VND/m²**."
                )

            st.caption(f"📌 *{result['disclaimer']}*")

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(f"Đã xảy ra lỗi trong quá trình dự báo: {exc}")

# ---------------------------------------------------------------------------
# TAB 2: PHÂN TÍCH SHAP & THỊ TRƯỜNG
# ---------------------------------------------------------------------------
with tab_shap:
    st.subheader("Giải Thích Mô Hình & Mức Độ Đóng Góp Đặc Trưng (SHAP Values)")
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        contributions = result.get("top_contributions", [])

        if contributions:
            chart_df = pd.DataFrame(contributions)
            chart_df.rename(
                columns={"feature": "Đặc Trưng", "shap_value": "Tác Động SHAP (Log-Scale)"},
                inplace=True,
            )

            st.markdown(
                "Đồ thị dưới đây hiển thị 5 yếu tố có ảnh hưởng mạnh nhất đến giá trị định giá bất động sản của bạn. "
                "Giá trị SHAP dương thể hiện yếu tố làm **tăng giá**, giá trị âm thể hiện yếu tố làm **giảm giá**."
            )

            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                st.bar_chart(
                    chart_df.set_index("Đặc Trưng")["Tác Động SHAP (Log-Scale)"],
                    color="#2563EB",
                )

            with col_table:
                st.dataframe(chart_df, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có dữ liệu SHAP.")
    else:
        st.info("Vui lòng thực hiện một lần định giá tại Tab '🏠 Dự Báo Giá' để xem phân tích SHAP chi tiết.")

# ---------------------------------------------------------------------------
# TAB 3: THÔNG TIN MÔ HÌNH & MLOPS
# ---------------------------------------------------------------------------
with tab_info:
    st.subheader("Thông Tin Kiến Trúc Mô Hình & MLOps Pipeline")
    try:
        model_package = load_model()
        c_info1, c_info2 = st.columns(2)

        with c_info1:
            st.markdown("### 🛠️ Cấu Hướng Mô Hình")
            st.write(f"- **Phiên bản mô hình**: `{model_package.get('version', '1.0.0')}`")
            st.write(
                f"- **Thuật toán chính**: `{model_package.get('model_type', 'ExtraTreesRegressor')}`"
            )
            st.write(
                f"- **Giao thức phân chia dữ liệu**: "
                f"`{model_package.get('split_protocol', 'grouped temporal 60/15/10/15')}`"
            )
            st.write(f"- **Mục tiêu bao phủ Conformal**: `{model_package.get('target_coverage', 0.8) * 100:.0f}%`")

        with c_info2:
            st.markdown("### 🛡️ Nguyên Lý Chống Data Leakage")
            st.markdown(
                """
                1. **Grouped Temporal Split**: Nhóm tin đăng trùng theo `property_group_id` và chia dữ liệu theo dòng thời gian.
                2. **Pipeline Categorical & Imputer**: Chi `fit` trên tập Train, loại trừ khả năng rò rỉ thông tin từ Calibration/Test.
                3. **Conformal Uncertainty**: Đảm bảo khoảng bao phủ tin cậy không phụ thuộc vào giả định phân phối chuẩn.
                """
            )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.error(f"Chưa thể tải thông tin mô hình: {exc}")
