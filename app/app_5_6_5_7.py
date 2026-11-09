from pathlib import Path
import sys
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from inventory_tools import (
    load_project_data,
    build_recommendations,
    simulate_inventory,
)

st.set_page_config(
    page_title="WMS - Đề xuất bổ sung hàng",
    page_icon="📦",
    layout="wide",
)

st.title("Hệ thống đề xuất bổ sung hàng và mô phỏng tồn kho")
st.caption(
    "Mô phỏng trên dữ liệu M5 – cửa hàng CA_1, nhóm FOODS. "
    "Tồn kho ban đầu là giả định mô phỏng, không phải số liệu tồn kho thực tế."
)

@st.cache_data(show_spinner="Đang tải dữ liệu M5...")
def load_data(root):
    return load_project_data(root)

try:
    train_df, test_df, predictions, sigma_df = load_data(str(PROJECT_DIR))
except Exception as exc:
    st.error(f"Chưa tải được dữ liệu dự án: {exc}")
    st.info(
        "Kiểm tra các file: data/processed/train.csv, "
        "data/processed/test.csv, output/ml_models/random_forest_test.csv "
        "và output/ml_models/forecast_detail.csv."
    )
    st.stop()

with st.sidebar:
    st.header("Thiết lập mô phỏng")
    lead_time = st.number_input("Thời gian giao hàng (ngày)", 1, 28, 6)
    review_interval = st.number_input("Chu kỳ xem xét (ngày)", 1, 14, 1)
    stock_cover = st.number_input("Tồn kho giả định (số ngày bán)", 0, 30, 3)
    z_value = st.select_slider(
        "Hệ số tồn kho an toàn (Z)",
        options=[1.28, 1.45, 1.65, 1.96, 2.33],
        value=1.65,
    )

page = st.sidebar.radio(
    "Chức năng",
    ["Tổng quan", "5.6 – Đề xuất hàng bổ sung", "5.7 – Mô phỏng tồn kho"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Nguồn: M5 Forecasting – CA_1 / FOODS")
st.sidebar.caption("Tồn kho đầu kỳ được giả lập từ doanh số trung bình.")

try:
    recommendations = build_recommendations(
        train_df, predictions, sigma_df,
        lead_time=int(lead_time),
        review_interval=int(review_interval),
        stock_cover_days=int(stock_cover),
        z_value=float(z_value),
    )
except Exception as exc:
    recommendations = None
    st.error(f"Không tính được đề xuất: {exc}")

if page == "Tổng quan":
    st.subheader("Tổng quan dữ liệu và mô hình")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sản phẩm trong dự báo", f"{predictions['item_id'].nunique():,}")
    c2.metric("Số dòng dự báo", f"{len(predictions):,}")
    c3.metric("Ngày cuối tập train", train_df["date"].max().strftime("%d/%m/%Y"))
    c4.metric("Ngày dự báo đầu tiên", predictions["date"].min().strftime("%d/%m/%Y"))
    st.markdown("### Dữ liệu dự báo Random Forest")
    st.dataframe(predictions.head(20), use_container_width=True)
    st.info(
        "Ứng dụng bổ sung này tập trung minh họa hai nhiệm vụ 5.6 và 5.7. "
        "Có thể tích hợp các trang tổng quan, dữ liệu và so sánh mô hình "
        "của app.py hiện tại sau khi đối chiếu mã nguồn gốc."
    )

elif page == "5.6 – Đề xuất hàng bổ sung":
    st.subheader("5.6. Xây dựng chức năng đề xuất lượng hàng bổ sung")
    st.write(
        "Ứng dụng cộng nhu cầu dự báo trong thời gian giao hàng và chu kỳ xem xét "
        "với tồn kho an toàn, sau đó trừ vị thế tồn kho."
    )
    st.latex(r"Q = \max(0, D + SS - I)")
    st.caption(
        "D: nhu cầu dự báo; SS: tồn kho an toàn; I: tồn kho hiện có + hàng đang về."
    )
    st.caption(
        f"Ngày lập kế hoạch: {train_df['date'].max().strftime('%d/%m/%Y')} "
        f"· Khoảng đáp ứng: {int(lead_time + review_interval)} ngày."
    )

    if recommendations is not None:
        total_items = len(recommendations)
        need_order = int((recommendations["suggested_qty"] > 0).sum())
        total_qty = int(recommendations["suggested_qty"].sum())
        a, b, c = st.columns(3)
        a.metric("Sản phẩm được tính", f"{total_items:,}")
        b.metric("Sản phẩm cần bổ sung", f"{need_order:,}")
        c.metric("Tổng lượng đề xuất", f"{total_qty:,}")

        view = recommendations.copy()
        view["planning_date"] = view["planning_date"].dt.strftime("%d/%m/%Y")
        st.markdown("### Danh sách đề xuất")
        st.dataframe(
            view[[
                "item_id", "avg_daily_sales", "forecast_demand",
                "safety_stock", "on_hand", "on_order",
                "inventory_position", "suggested_qty", "recommendation"
            ]],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Tải bảng đề xuất CSV",
            data=recommendations.to_csv(index=False).encode("utf-8-sig"),
            file_name="replenishment_recommendations.csv",
            mime="text/csv",
        )
        selected = st.selectbox(
            "Xem chi tiết sản phẩm",
            recommendations["item_id"].tolist(),
        )
        detail = recommendations[recommendations["item_id"] == selected].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Nhu cầu dự báo", f"{detail['forecast_demand']:.1f}")
        d2.metric("Tồn kho an toàn", f"{detail['safety_stock']:.1f}")
        d3.metric("Tồn kho giả định", f"{int(detail['on_hand'])}")
        d4.metric("Cần bổ sung", f"{int(detail['suggested_qty'])}")
        st.caption(
            "Tồn kho ban đầu được giả định bằng doanh số trung bình 28 ngày "
            f"nhân {int(stock_cover)} ngày bao phủ. Có thể thay đổi tham số ở thanh bên."
        )

else:
    st.subheader("5.7. Tích hợp hệ thống và mô phỏng tồn kho")
    st.write(
        "So sánh kết quả tồn kho khi áp dụng chính sách đặt hàng theo dự báo "
        "Random Forest. Mô phỏng dùng nhu cầu thực tế trong tập test để đánh giá "
        "lại chính sách trên giai đoạn lịch sử."
    )
    try:
        simulation, summary = simulate_inventory(
            train_df, test_df, predictions, sigma_df,
            lead_time=int(lead_time),
            review_interval=int(review_interval),
            stock_cover_days=int(stock_cover),
            z_value=float(z_value),
        )
        row = summary.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng nhu cầu", f"{int(row['total_demand']):,}")
        m2.metric("Lượng bán đáp ứng", f"{int(row['fulfilled_demand']):,}")
        m3.metric("Lượng thiếu hàng", f"{int(row['lost_sales']):,}")
        m4.metric("Tỷ lệ đáp ứng", f"{row['service_level_pct']:.2f}%")

        st.markdown("### Kết quả mô phỏng")
        st.dataframe(summary, use_container_width=True)
        daily = (
            simulation.groupby("date", as_index=False)[
                ["demand", "sold", "lost_sales", "ending_stock", "received_stock"]
            ].sum()
        ).set_index("date")
        st.markdown("### Diễn biến tồn kho theo ngày")
        st.line_chart(daily[["ending_stock"]])
        st.markdown("### Nhu cầu, lượng bán và hàng thiếu")
        st.bar_chart(daily[["demand", "sold", "lost_sales"]])
        st.markdown("### Chi tiết mô phỏng")
        st.dataframe(simulation.head(500), use_container_width=True)
        st.download_button(
            "Tải kết quả mô phỏng CSV",
            data=simulation.to_csv(index=False).encode("utf-8-sig"),
            file_name="inventory_simulation_detail.csv",
            mime="text/csv",
        )
        st.download_button(
            "Tải bảng tổng hợp CSV",
            data=summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="inventory_simulation_summary.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Chưa chạy được mô phỏng: {exc}")

st.markdown("---")
st.caption(
    "Lưu ý phương pháp: tồn kho đầu kỳ là dữ liệu giả lập; kết quả mô phỏng "
    "không đại diện cho tồn kho thực tế của Walmart/M5."
)
