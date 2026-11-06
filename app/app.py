
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail AI - Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "output" / "ml_models"

MODEL_DIR = PROJECT_DIR / "model"

METRICS_FILE = OUTPUT_DIR / "model_metrics.csv"
DETAIL_FILE = OUTPUT_DIR / "forecast_detail.csv"
MODEL_FILE = MODEL_DIR / "random_forest.pkl"


# ============================================================
# 2. STYLE
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F4F6FA;
    }

    [data-testid="stSidebar"] {
        background-color: #102A43;
    }

    [data-testid="stSidebar"] * {
        color: dark;
    }

    .main-title {
        color: #102A43;
        font-size: 30px;
        font-weight: bold;
    }

    .sub-title {
        color: #627D98;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E0E6ED;
        padding: 14px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD RESULTS
# ============================================================

@st.cache_data
def load_metrics():
    if not METRICS_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(METRICS_FILE)


@st.cache_data
def load_forecast_detail():
    if not DETAIL_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(DETAIL_FILE)
    df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data
def load_data_preview():
    data_file = DATA_DIR / "model_data.csv"

    if not data_file.exists():
        return pd.DataFrame()

    return pd.read_csv(data_file, nrows=20)


metrics_df = load_metrics()
forecast_df = load_forecast_detail()


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.title("📦 RETAIL AI")
st.sidebar.caption("Demand Forecasting System")

menu = st.sidebar.radio(
    "ĐIỀU HƯỚNG",
    [
        "Tổng quan",
        "Quản lý dữ liệu",
        "Dự báo nhu cầu",
        "So sánh mô hình",
        "Đề xuất bổ sung hàng",
        "Mô phỏng tồn kho"
    ]
)

st.sidebar.divider()
st.sidebar.caption("M5 Forecasting Dataset")
st.sidebar.caption("Store: CA_1 | Category: FOODS")


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def show_title(title, subtitle):
    st.markdown(
        f'<p class="main-title">{title}</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="sub-title">{subtitle}</p>',
        unsafe_allow_html=True
    )

    st.divider()


def get_period_data(period):
    """
    Lọc dữ liệu theo hai giai đoạn đã đánh giá.
    """

    if forecast_df.empty:
        return pd.DataFrame()

    if period == "Validation":
        start = pd.Timestamp("2016-02-29")
        end = pd.Timestamp("2016-03-27")
    else:
        start = pd.Timestamp("2016-03-28")
        end = pd.Timestamp("2016-04-24")

    return forecast_df[
        (forecast_df["date"] >= start)
        & (forecast_df["date"] <= end)
    ].copy()


# ============================================================
# 6. DASHBOARD
# ============================================================

if menu == "Tổng quan":

    show_title(
        "TỔNG QUAN HỆ THỐNG",
        "Hệ thống dự báo nhu cầu và hỗ trợ bổ sung hàng hóa cho nhà bán lẻ."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Bộ dữ liệu", "M5 Forecasting")
    col2.metric("Cửa hàng", "CA_1")
    col3.metric("Nhóm sản phẩm", "FOODS")

    st.subheader("Trạng thái mô hình")

    if MODEL_FILE.exists():
        st.success("Random Forest đã được lưu và sẵn sàng sử dụng.")
    else:
        st.warning("Chưa tìm thấy file Random Forest.")

    if metrics_df.empty:
        st.warning("Chưa tìm thấy bảng kết quả đánh giá mô hình.")
    else:
        test_metrics = metrics_df[
            metrics_df["Period"] == "Test"
        ]

        if not test_metrics.empty:
            best = test_metrics.sort_values(
                "WAPE (%)"
            ).iloc[0]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Mô hình tốt nhất",
                best["Model"]
            )

            c2.metric(
                "MAE Test",
                f"{best['MAE']:.4f}"
            )

            c3.metric(
                "WAPE Test",
                f"{best['WAPE (%)']:.2f}%"
            )

    st.subheader("Các chức năng chính")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Dự báo nhu cầu")
        st.write(
            "Hiển thị kết quả dự báo doanh số theo từng sản phẩm "
            "bằng mô hình Random Forest."
        )

        st.markdown("### 📦 Đề xuất bổ sung hàng")
        st.write(
            "Kết hợp nhu cầu dự báo với tồn kho hiện tại "
            "để tính lượng hàng cần bổ sung."
        )

    with col2:
        st.markdown("### 📈 So sánh mô hình")
        st.write(
            "So sánh Random Forest, LightGBM và Seasonal Naive "
            "thông qua MAE, RMSE và WAPE."
        )

        st.markdown("### 📉 Mô phỏng tồn kho")
        st.write(
            "Đánh giá tình trạng thiếu hàng và lượng tồn kho "
            "trong giai đoạn mô phỏng."
        )


# ============================================================
# 7. DATA MANAGEMENT
# ============================================================

elif menu == "Quản lý dữ liệu":

    show_title(
        "QUẢN LÝ DỮ LIỆU",
        "Kiểm tra dữ liệu đã tiền xử lý."
    )

    data_file = DATA_DIR / "model_data.csv"

    if not data_file.exists():
        st.error(f"Không tìm thấy file: {data_file}")

    else:
        st.success("Đã tìm thấy model_data.csv")

        file_size = data_file.stat().st_size / (1024 * 1024)

        col1, col2 = st.columns(2)

        col1.metric(
            "Dung lượng dữ liệu",
            f"{file_size:.2f} MB"
        )

        col2.metric(
            "Số dòng mẫu đang xem",
            len(load_data_preview())
        )

        st.subheader("Xem trước dữ liệu")

        preview = load_data_preview()

        if not preview.empty:
            st.dataframe(
                preview,
                use_container_width=True
            )


# ============================================================
# 8. DEMAND FORECASTING
# ============================================================

elif menu == "Dự báo nhu cầu":

    show_title(
        "DỰ BÁO NHU CẦU",
        "Xem kết quả dự báo theo sản phẩm và giai đoạn."
    )

    if forecast_df.empty:
        st.error(
            "Không tìm thấy forecast_detail.csv. "
            "Hãy chạy train_models.py trước."
        )

    else:
        period = st.selectbox(
            "Chọn giai đoạn đánh giá",
            ["Validation", "Test"]
        )

        period_df = get_period_data(period)

        products = sorted(
            period_df["item_id"].dropna().unique().tolist()
        )

        selected_product = st.selectbox(
            "Chọn sản phẩm",
            products
        )

        product_df = period_df[
            period_df["item_id"] == selected_product
        ].copy()

        if product_df.empty:
            st.warning("Không có dữ liệu cho sản phẩm này.")

        else:
            # Lấy doanh số thực tế và dự báo của từng mô hình.
            chart_df = product_df.pivot_table(
                index="date",
                columns="model",
                values="forecast",
                aggfunc="first"
            )

            actual_df = (
                product_df[["date", "actual"]]
                .drop_duplicates("date")
                .set_index("date")
            )

            chart_df = actual_df.join(chart_df)
            chart_df = chart_df.rename(
                columns={"actual": "Doanh số thực tế"}
            )

            st.subheader(
                f"Sản phẩm: {selected_product}"
            )

            st.line_chart(
                chart_df,
                use_container_width=True
            )

            # Chọn mô hình cần xem chi tiết.
            selected_model = st.selectbox(
                "Chọn mô hình xem chi tiết",
                [
                    "Random Forest",
                    "LightGBM",
                    "Seasonal Naive"
                ]
            )

            detail = product_df[
                product_df["model"] == selected_model
            ].copy()

            detail = detail.sort_values("date")

            total_actual = detail["actual"].sum()
            total_forecast = detail["forecast"].sum()

            mae = (
                detail["actual"] - detail["forecast"]
            ).abs().mean()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Tổng doanh số thực tế",
                f"{total_actual:,.0f}"
            )

            c2.metric(
                "Tổng nhu cầu dự báo",
                f"{total_forecast:,.0f}"
            )

            c3.metric(
                "MAE sản phẩm",
                f"{mae:.4f}"
            )

            st.subheader("Chi tiết dự báo")

            display_df = detail[
                ["date", "actual", "forecast"]
            ].rename(
                columns={
                    "date": "Ngày",
                    "actual": "Doanh số thực tế",
                    "forecast": "Dự báo"
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True
            )

            st.download_button(
                "⬇️ Tải kết quả dự báo CSV",
                data=display_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name=f"forecast_{selected_product}_{period}.csv",
                mime="text/csv"
            )


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

elif menu == "So sánh mô hình":

    show_title(
        "SO SÁNH MÔ HÌNH",
        "Đánh giá hiệu quả của Random Forest, LightGBM và Seasonal Naive."
    )

    if metrics_df.empty:
        st.error("Không tìm thấy model_metrics.csv.")

    else:
        period = st.selectbox(
            "Chọn giai đoạn",
            ["Validation", "Test"]
        )

        compare_df = metrics_df[
            metrics_df["Period"] == period
        ].copy()

        compare_df = compare_df.sort_values(
            "WAPE (%)"
        )

        st.subheader("Bảng kết quả")

        st.dataframe(
            compare_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("So sánh WAPE")

        wape_chart = compare_df[
            ["Model", "WAPE (%)"]
        ].set_index("Model")

        st.bar_chart(
            wape_chart,
            use_container_width=True
        )

        st.subheader("So sánh MAE và RMSE")

        error_chart = compare_df[
            ["Model", "MAE", "RMSE"]
        ].set_index("Model")

        st.bar_chart(
            error_chart,
            use_container_width=True
        )

        best = compare_df.iloc[0]

        st.success(
            f"Mô hình có WAPE thấp nhất trên {period}: "
            f"{best['Model']} ({best['WAPE (%)']:.2f}%)."
        )


# ============================================================
# 10. REPLENISHMENT
# ============================================================

elif menu == "Đề xuất bổ sung hàng":

    show_title(
        "ĐỀ XUẤT BỔ SUNG HÀNG",
        "Chức năng tính lượng hàng cần bổ sung sẽ được triển khai tiếp theo."
    )

    st.info(
        "Phần này sẽ sử dụng dự báo Random Forest kết hợp "
        "với tồn kho hiện tại, hàng đang về, lead time "
        "và tồn kho an toàn."
    )

    st.write("Các thông tin cần nhập:")

    st.write("- Sản phẩm cần bổ sung")
    st.write("- Tồn kho hiện tại")
    st.write("- Lượng hàng đang về")
    st.write("- Thời gian cung ứng (Lead time)")
    st.write("- Mức tồn kho an toàn")


# ============================================================
# 11. INVENTORY SIMULATION
# ============================================================

elif menu == "Mô phỏng tồn kho":

    show_title(
        "MÔ PHỎNG TỒN KHO",
        "Đánh giá khả năng đáp ứng nhu cầu dựa trên dự báo."
    )

    st.info(
        "Chức năng mô phỏng sẽ được phát triển sau khi hoàn thiện "
        "phần đề xuất lượng hàng bổ sung."
    )

    st.write(
        "Các chỉ số dự kiến: tỷ lệ thiếu hàng, "
        "tồn kho cuối kỳ và tồn kho bình quân."
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "RETAIL AI | M5 Forecasting | CA_1 | FOODS "
    "| Random Forest Demand Forecasting"
)
