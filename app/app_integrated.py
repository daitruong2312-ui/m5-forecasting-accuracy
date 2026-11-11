from pathlib import Path
import sys

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Retail AI - Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "output" / "ml_models"
MODEL_DIR = PROJECT_DIR / "model"

METRICS_FILE = OUTPUT_DIR / "model_metrics.csv"
DETAIL_FILE = OUTPUT_DIR / "forecast_detail.csv"
MODEL_FILE = MODEL_DIR / "random_forest.pkl"

try:
    from inventory_tools import (
        load_project_data,
        build_recommendations,
        simulate_inventory,
    )
    INVENTORY_TOOLS_ERROR = None
except Exception as exc:
    INVENTORY_TOOLS_ERROR = exc


# ----------------------------- STYLE -----------------------------
st.markdown("""
<style>
    .stApp { background-color: #F4F6FA; color: #243B53; }
    [data-testid="stSidebar"] { background-color: #102A43; }
    [data-testid="stSidebar"] * { color: #F4F7FB !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #F4F7FB !important; }
    .main-title { color: #102A43; font-size: 30px; font-weight: 750; margin-bottom: 4px; }
    .sub-title { color: #627D98; font-size: 15px; margin-top: 0; }
    .stMarkdown, .stText, p, label { color: #243B53; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #D9E2EC;
        padding: 14px; border-radius: 10px;
    }
    div[data-testid="stAlert"] p { color: inherit; }
    .section-note {
        background: #E8F1FB; border-left: 4px solid #2F80ED;
        padding: 12px 16px; border-radius: 7px; color: #243B53;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------- DATA LOADERS -----------------------------
@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_FILE)


@st.cache_data(show_spinner=False)
def load_forecast_detail():
    if not DETAIL_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(DETAIL_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_data_preview():
    data_file = DATA_DIR / "model_data.csv"
    if not data_file.exists():
        return pd.DataFrame()
    return pd.read_csv(data_file, nrows=20)


@st.cache_data(show_spinner=False)
def load_inventory_inputs():
    if INVENTORY_TOOLS_ERROR:
        raise RuntimeError(f"Không import được inventory_tools.py: {INVENTORY_TOOLS_ERROR}")
    return load_project_data(PROJECT_DIR)


metrics_df = load_metrics()
forecast_df = load_forecast_detail()


# ----------------------------- HELPERS -----------------------------
def show_title(title, subtitle):
    st.markdown(f'<p class="main-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{subtitle}</p>', unsafe_allow_html=True)
    st.divider()


def get_period_data(period):
    if forecast_df.empty:
        return pd.DataFrame()
    if period == "Validation":
        start, end = pd.Timestamp("2016-02-29"), pd.Timestamp("2016-03-27")
    else:
        start, end = pd.Timestamp("2016-03-28"), pd.Timestamp("2016-04-24")
    return forecast_df[
        (forecast_df["date"] >= start) & (forecast_df["date"] <= end)
    ].copy()


def money_number(value):
    return f"{float(value):,.0f}"


def inventory_data_or_error():
    try:
        return load_inventory_inputs()
    except Exception as exc:
        st.error("Không thể nạp dữ liệu phục vụ đề xuất/mô phỏng tồn kho.")
        st.code(str(exc))
        st.info(
            "Kiểm tra các file data/processed/train.csv, "
            "data/processed/test.csv và output/ml_models/random_forest_test.csv "
            "(hoặc forecast_detail.csv) theo đúng cấu trúc mà inventory_tools.py yêu cầu."
        )
        return None


# ----------------------------- SIDEBAR -----------------------------
st.sidebar.title("📦 RETAIL AI")
st.sidebar.caption("Demand Forecasting System")
menu = st.sidebar.radio(
    "ĐIỀU HƯỚNG",
    [
        "Tổng quan", "Quản lý dữ liệu", "Dự báo nhu cầu",
        "So sánh mô hình", "Đề xuất bổ sung hàng", "Mô phỏng tồn kho"
    ]
)
st.sidebar.divider()
st.sidebar.caption("M5 Forecasting Dataset")
st.sidebar.caption("Store: CA_1 | Category: FOODS")



# ----------------------------- 1. OVERVIEW -----------------------------
if menu == "Tổng quan":
    show_title(
        "TỔNG QUAN HỆ THỐNG",
        "Hệ thống dự báo nhu cầu và hỗ trợ bổ sung hàng hóa cho nhà bán lẻ."
    )

    # Thông tin tổng quan
    c1, c2, c3 = st.columns(3)
    c1.metric("Bộ dữ liệu", "M5 Forecasting")
    c2.metric("Cửa hàng", "CA_1")
    c3.metric("Nhóm sản phẩm", "FOODS")

    # Trạng thái mô hình
    st.subheader("Trạng thái mô hình")

    if MODEL_FILE.exists():
        st.success("Đã tìm thấy mô hình Random Forest.")
    else:
        st.warning(
            f"Chưa tìm thấy file mô hình tại {MODEL_FILE}. "
            "Ứng dụng vẫn có thể hiển thị kết quả nếu đã có file dự báo xuất sẵn."
        )

    st.info(
        "Ứng dụng hiện sử dụng file dự báo đã xuất sẵn, "
        "không tự huấn luyện mô hình khi khởi chạy."
    )

    # Kết quả đánh giá mô hình
    if metrics_df.empty:
        st.warning(f"Chưa tìm thấy bảng kết quả: {METRICS_FILE}")

    elif {"Period", "WAPE (%)", "Model", "MAE"}.issubset(metrics_df.columns):
        test_metrics = metrics_df[
            metrics_df["Period"].astype(str).str.lower() == "test"
        ]

        if not test_metrics.empty:
            best = test_metrics.sort_values("WAPE (%)").iloc[0]

            a, b, c = st.columns(3)
            a.metric("Mô hình WAPE thấp nhất", str(best["Model"]))
            b.metric("MAE Test", f"{best['MAE']:.4f}")
            c.metric("WAPE Test", f"{best['WAPE (%)']:.2f}%")

    # Giới thiệu chức năng
    st.subheader("Các chức năng chính")

    a, b = st.columns(2)

    with a:
        st.markdown("### 📊 Dự báo nhu cầu")
        st.write(
            "Xem kết quả dự báo doanh số theo từng sản phẩm và giai đoạn."
        )

        st.markdown("### 📦 Đề xuất bổ sung hàng")
        st.write(
            "Kết hợp dự báo với tồn kho, hàng đang về, "
            "lead time và tồn kho an toàn."
        )

    with b:
        st.markdown("### 📈 So sánh mô hình")
        st.write(
            "So sánh Random Forest, LightGBM và Seasonal Naive "
            "qua MAE, RMSE và WAPE."
        )

        st.markdown("### 📉 Mô phỏng tồn kho")
        st.write(
            "Đánh giá mức đáp ứng nhu cầu, thiếu hàng "
            "và tồn kho trong kỳ mô phỏng."
        )

# ----------------------------- 2. DATA MANAGEMENT -----------------------------
elif menu == "Quản lý dữ liệu":
    show_title("QUẢN LÝ DỮ LIỆU", "Kiểm tra dữ liệu đã tiền xử lý.")
    data_file = DATA_DIR / "model_data.csv"
    if not data_file.exists():
        st.error(f"Không tìm thấy file: {data_file}")
    else:
        st.success("Đã tìm thấy model_data.csv")
        file_size = data_file.stat().st_size / (1024 * 1024)
        preview = load_data_preview()
        a, b = st.columns(2)
        a.metric("Dung lượng dữ liệu", f"{file_size:.2f} MB")
        b.metric("Số dòng mẫu đang xem", len(preview))
        st.subheader("Xem trước dữ liệu")
        st.dataframe(preview, use_container_width=True, hide_index=True)


# ----------------------------- 3. FORECAST -----------------------------
elif menu == "Dự báo nhu cầu":
    show_title("DỰ BÁO NHU CẦU", "Xem kết quả dự báo theo sản phẩm và giai đoạn.")
    required = {"date", "item_id", "model", "actual", "forecast"}
    if forecast_df.empty:
        st.error(f"Không tìm thấy hoặc không đọc được {DETAIL_FILE}.")
    elif not required.issubset(forecast_df.columns):
        st.error(f"forecast_detail.csv cần có các cột: {sorted(required)}")
    else:
        period = st.selectbox("Chọn giai đoạn đánh giá", ["Validation", "Test"])
        period_df = get_period_data(period)
        products = sorted(period_df["item_id"].dropna().astype(str).unique().tolist())
        if not products:
            st.warning("Không có sản phẩm trong giai đoạn đã chọn.")
        else:
            selected_product = st.selectbox("Chọn sản phẩm", products)
            product_df = period_df[period_df["item_id"].astype(str) == selected_product].copy()
            chart_df = product_df.pivot_table(
                index="date", columns="model", values="forecast", aggfunc="first"
            )
            actual_df = product_df[["date", "actual"]].drop_duplicates("date").set_index("date")
            chart_df = actual_df.join(chart_df).rename(columns={"actual": "Doanh số thực tế"})
            st.subheader(f"Sản phẩm: {selected_product}")
            st.line_chart(chart_df, use_container_width=True)

            models = product_df["model"].dropna().astype(str).unique().tolist()
            default_model = next((m for m in models if "random forest" in m.lower()), models[0])
            selected_model = st.selectbox(
                "Chọn mô hình xem chi tiết", models,
                index=models.index(default_model)
            )
            detail = product_df[product_df["model"].astype(str) == selected_model].sort_values("date")
            total_actual = detail["actual"].sum()
            total_forecast = detail["forecast"].sum()
            mae = (detail["actual"] - detail["forecast"]).abs().mean()
            a, b, c = st.columns(3)
            a.metric("Tổng doanh số thực tế", money_number(total_actual))
            b.metric("Tổng nhu cầu dự báo", money_number(total_forecast))
            c.metric("MAE sản phẩm", f"{mae:.4f}")
            display_df = detail[["date", "actual", "forecast"]].rename(
                columns={"date": "Ngày", "actual": "Doanh số thực tế", "forecast": "Dự báo"}
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Tải kết quả dự báo CSV",
                data=display_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"forecast_{selected_product}_{period}.csv",
                mime="text/csv"
            )


# ----------------------------- 4. MODEL COMPARISON -----------------------------
elif menu == "So sánh mô hình":
    show_title("SO SÁNH MÔ HÌNH", "Đánh giá hiệu quả của Random Forest, LightGBM và Seasonal Naive.")
    required = {"Period", "Model", "WAPE (%)", "MAE", "RMSE"}
    if metrics_df.empty:
        st.error(f"Không tìm thấy {METRICS_FILE}.")
    elif not required.issubset(metrics_df.columns):
        st.error(f"model_metrics.csv cần có các cột: {sorted(required)}")
    else:
        period = st.selectbox("Chọn giai đoạn", ["Validation", "Test"])
        compare_df = metrics_df[metrics_df["Period"].astype(str).str.lower() == period.lower()].copy()
        compare_df = compare_df.sort_values("WAPE (%)")
        if compare_df.empty:
            st.warning(f"Không có kết quả cho giai đoạn {period}.")
        else:
            st.subheader("Bảng kết quả")
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            st.subheader("So sánh WAPE")
            st.bar_chart(compare_df[["Model", "WAPE (%)"]].set_index("Model"))
            st.subheader("So sánh MAE và RMSE")
            st.bar_chart(compare_df[["Model", "MAE", "RMSE"]].set_index("Model"))
            best = compare_df.iloc[0]
            st.success(
                f"Mô hình có WAPE thấp nhất trên {period}: "
                f"{best['Model']} ({best['WAPE (%)']:.2f}%)."
            )


# ----------------------------- 5. REPLENISHMENT -----------------------------
elif menu == "Đề xuất bổ sung hàng":
    show_title(
        "ĐỀ XUẤT BỔ SUNG HÀNG",
        "Tính lượng hàng cần đặt dựa trên dự báo Random Forest và vị thế tồn kho."
    )
    st.markdown(
        '<div class="section-note">Công thức: Lượng cần đặt = max(0, nhu cầu dự báo trong kỳ bảo vệ + tồn kho an toàn − tồn kho hiện tại − hàng đang về).</div>',
        unsafe_allow_html=True
    )
    loaded = inventory_data_or_error()
    if loaded is not None:
        train, test, predictions, sigma = loaded
        products = sorted(predictions["item_id"].dropna().astype(str).unique().tolist())
        if not products:
            st.warning("Không có sản phẩm trong dữ liệu dự báo.")
        else:
            with st.form("replenishment_form"):
                left, right = st.columns(2)
                with left:
                    item = st.selectbox("Sản phẩm cần bổ sung", products)
                    on_hand = st.number_input("Tồn kho hiện tại (đơn vị)", min_value=0, value=0, step=1)
                    on_order = st.number_input("Lượng hàng đang về (đơn vị)", min_value=0, value=0, step=1)
                with right:
                    lead_time = st.number_input("Lead time – thời gian cung ứng (ngày)", min_value=1, max_value=60, value=6, step=1)
                    review_interval = st.number_input("Chu kỳ xem xét đặt hàng (ngày)", min_value=1, max_value=30, value=1, step=1)
                    safety_stock_input = st.number_input("Tồn kho an toàn (đơn vị)", min_value=0, value=0, step=1)
                submitted = st.form_submit_button("🧮 Tính lượng hàng cần bổ sung", use_container_width=True)

            if submitted:
                custom_stock = pd.DataFrame([{"item_id": item, "on_hand": int(on_hand)}])
                custom_order = pd.DataFrame([{"item_id": item, "on_order": int(on_order)}])
                try:
                    recommendations = build_recommendations(
                        train=train,
                        predictions=predictions,
                        sigma=sigma,
                        lead_time=int(lead_time),
                        review_interval=int(review_interval),
                        stock_cover_days=0,
                        custom_stock=custom_stock,
                        custom_on_order=custom_order,
                    )
                    selected = recommendations[recommendations["item_id"].astype(str) == item].copy()
                    if selected.empty:
                        st.error(
                            f"Sản phẩm {item} không có đủ {int(lead_time + review_interval)} ngày dự báo liên tiếp. "
                            "Hãy giảm lead time/chu kỳ xem xét hoặc kiểm tra file dự báo."
                        )
                    else:
                        # Dùng mức tồn kho an toàn do người dùng nhập, thay vì ước lượng tự động.
                        selected["safety_stock"] = int(safety_stock_input)
                        selected["inventory_position"] = int(on_hand) + int(on_order)
                        selected["suggested_qty"] = (
                            selected["forecast_demand"] + selected["safety_stock"]
                            - selected["inventory_position"]
                        ).clip(lower=0).apply(lambda x: int(__import__("math").ceil(x)))
                        selected["recommendation"] = selected["suggested_qty"].apply(
                            lambda q: "Cần bổ sung" if q > 0 else "Chưa cần bổ sung"
                        )
                        row = selected.iloc[0]
                        st.subheader(f"Kết quả cho sản phẩm {item}")
                        a, b, c, d = st.columns(4)
                        a.metric("Nhu cầu dự báo", money_number(row["forecast_demand"]))
                        b.metric("Tồn kho hiện tại", money_number(on_hand))
                        c.metric("Hàng đang về", money_number(on_order))
                        d.metric("Lượng đề xuất đặt", money_number(row["suggested_qty"]))
                        if row["suggested_qty"] > 0:
                            st.warning(f"Cần đặt bổ sung **{int(row['suggested_qty']):,} đơn vị**.")
                        else:
                            st.success("Tồn kho hiện tại và hàng đang về đủ đáp ứng nhu cầu dự báo trong kỳ.")
                        detail = pd.DataFrame([{
                            "Sản phẩm": item,
                            "Ngày lập kế hoạch": pd.Timestamp(row["planning_date"]).date(),
                            "Lead time (ngày)": int(lead_time),
                            "Chu kỳ xem xét (ngày)": int(review_interval),
                            "Số ngày dự báo": int(row["coverage_days"]),
                            "Nhu cầu dự báo": round(float(row["forecast_demand"]), 2),
                            "Tồn kho hiện tại": int(on_hand),
                            "Hàng đang về": int(on_order),
                            "Tồn kho an toàn": int(safety_stock_input),
                            "Lượng cần đặt": int(row["suggested_qty"]),
                            "Khuyến nghị": row["recommendation"],
                        }])
                        st.subheader("Chi tiết đề xuất")
                        st.dataframe(detail, use_container_width=True, hide_index=True)
                        st.download_button(
                            "⬇️ Tải đề xuất bổ sung CSV",
                            data=detail.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"replenishment_{item}.csv",
                            mime="text/csv"
                        )
                        with st.expander("Thông tin kiểm tra dự báo và sai số"):
                            st.write(f"Nguồn ước lượng sai số: {row['sigma_source']}")
                            st.write(f"Độ lệch chuẩn sai số (sigma): {float(row['sigma_e']):.4f}")
                            st.write("Dữ liệu dự báo dùng cho tính toán:")
                            future_dates = predictions[
                                (predictions["item_id"].astype(str) == item)
                                & (predictions["date"] > pd.Timestamp(row["planning_date"]))
                                & (predictions["date"] <= pd.Timestamp(row["planning_date"]) + pd.Timedelta(days=int(row["coverage_days"])))
                            ].sort_values("date")
                            st.dataframe(future_dates, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error("Không thể tính đề xuất. Chi tiết lỗi:")
                    st.exception(exc)


# ----------------------------- 6. INVENTORY SIMULATION -----------------------------
elif menu == "Mô phỏng tồn kho":
    show_title("MÔ PHỎNG TỒN KHO", "Mô phỏng chính sách đặt hàng trong giai đoạn Test.")
    loaded = inventory_data_or_error()
    if loaded is not None:
        train, test, predictions, sigma = loaded
        with st.form("simulation_form"):
            a, b, c = st.columns(3)
            with a:
                lead_time = st.number_input("Lead time (ngày)", min_value=1, max_value=60, value=6)
            with b:
                review_interval = st.number_input("Chu kỳ xem xét (ngày)", min_value=1, max_value=30, value=1)
            with c:
                stock_cover_days = st.number_input("Số ngày tồn kho đầu kỳ", min_value=0, max_value=60, value=3)
            run_sim = st.form_submit_button("▶ Chạy mô phỏng tồn kho", use_container_width=True)
        if run_sim:
            try:
                detail, summary = simulate_inventory(
                    train=train,
                    test=test,
                    predictions=predictions,
                    sigma=sigma,
                    lead_time=int(lead_time),
                    review_interval=int(review_interval),
                    stock_cover_days=int(stock_cover_days),
                )
                row = summary.iloc[0]
                a, b, c, d = st.columns(4)
                a.metric("Tổng nhu cầu", money_number(row["total_demand"]))
                b.metric("Đáp ứng nhu cầu", f"{row['service_level_pct']:.2f}%")
                c.metric("Doanh số mất do thiếu hàng", money_number(row["lost_sales"]))
                d.metric("Tổng lượng đặt", money_number(row["total_order_qty"]))
                e, f, g = st.columns(3)
                e.metric("Tồn kho cuối ngày bình quân", money_number(row["average_ending_stock"]))
                f.metric("Số ngày-sản phẩm hết hàng", money_number(row["stockout_item_days"]))
                g.metric("Lượng hàng đã nhận", money_number(row["total_received_qty"]))
                st.subheader("Tóm tắt mô phỏng")
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.subheader("Diễn biến tồn kho")
                daily = detail.groupby("date", as_index=True)[
                    ["demand", "sold", "lost_sales", "ending_stock"]
                ].sum()
                st.line_chart(daily)
                st.subheader("Chi tiết mô phỏng")
                st.dataframe(detail, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Tải chi tiết mô phỏng CSV",
                    data=detail.to_csv(index=False).encode("utf-8-sig"),
                    file_name="inventory_simulation_detail.csv",
                    mime="text/csv"
                )
                st.download_button(
                    "⬇️ Tải tổng kết mô phỏng CSV",
                    data=summary.to_csv(index=False).encode("utf-8-sig"),
                    file_name="inventory_simulation_summary.csv",
                    mime="text/csv"
                )
            except Exception as exc:
                st.error("Không thể chạy mô phỏng. Chi tiết lỗi:")
                st.exception(exc)


# ----------------------------- FOOTER -----------------------------
st.divider()
st.caption("RETAIL AI | M5 Forecasting | CA_1 | FOODS | Random Forest Demand Forecasting")
