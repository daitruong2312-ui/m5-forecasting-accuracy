from pathlib import Path
import pandas as pd
import streamlit as st

# =========================
# 1. Cấu hình ứng dụng
# =========================
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

# app.py nằm trong thư mục app/, vì vậy thư mục gốc dự án là thư mục cha
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = PROJECT_DIR / "data_processed"

# =========================
# 2. Giao diện navy - trắng
# =========================
st.markdown("""
<style>
    .stApp {
        background-color: #F4F6FA;
    }
    [data-testid="stSidebar"] {
        background-color: #102A43;
    }
    [data-testid="stSidebar"] * {
        color: white;
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


def get_csv_files():
    """Tìm các file CSV trong thư mục dữ liệu của dự án."""
    files = []
    for folder in (DATA_DIR, PROCESSED_DIR, PROJECT_DIR):
        if folder.exists():
            for path in folder.glob("*.csv"):
                if path.is_file() and path not in files:
                    files.append(path)
    return sorted(files, key=lambda p: (p.parent.name, p.name.lower()))


def read_csv_preview(path, nrows=1000):
    """Đọc một phần dữ liệu để xem trước, tránh nạp file M5 lớn vào RAM."""
    return pd.read_csv(path, nrows=nrows, low_memory=False)


# =========================
# 3. Thanh điều hướng
# =========================
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


# =========================
# 4. Trang tổng quan
# =========================
if menu == "Tổng quan":
    st.markdown(
        '<p class="main-title">TỔNG QUAN HỆ THỐNG</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-title">Hệ thống dự báo nhu cầu và hỗ trợ bổ sung hàng hóa cho nhà bán lẻ.</p>',
        unsafe_allow_html=True
    )
    st.divider()

    st.subheader("Thông tin dự án")
    col1, col2, col3 = st.columns(3)
    col1.info("**Bộ dữ liệu**\n\nM5 Forecasting")
    col2.info("**Cửa hàng**\n\nCA_1")
    col3.info("**Nhóm sản phẩm**\n\nFOODS")

    st.subheader("Các chức năng chính")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Dự báo nhu cầu")
        st.write("Sử dụng Baseline và Random Forest để dự báo lượng hàng tiêu thụ.")
        st.markdown("### 📦 Đề xuất bổ sung hàng")
        st.write("Tính toán lượng hàng cần bổ sung dựa trên nhu cầu dự báo và thông tin tồn kho.")
    with col2:
        st.markdown("### 📈 So sánh mô hình")
        st.write("Đánh giá kết quả dự báo thông qua MAE và WAPE.")
        st.markdown("### 📉 Mô phỏng tồn kho")
        st.write("Đánh giá tình trạng thiếu hàng và tồn kho bình quân.")

    st.divider()
    st.caption(
        "Hệ thống hỗ trợ ra quyết định dự báo nhu cầu "
        "và quản lý hàng tồn kho cho nhà bán lẻ."
    )


# =========================
# 5. Quản lý dữ liệu
# =========================
elif menu == "Quản lý dữ liệu":
    st.markdown(
        '<p class="main-title">QUẢN LÝ DỮ LIỆU</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-title">Kiểm tra, xem trước và tải xuống các tệp dữ liệu M5 đang có trong dự án.</p>',
        unsafe_allow_html=True
    )
    st.divider()

    csv_files = get_csv_files()

    if not csv_files:
        st.error("Không tìm thấy file CSV trong thư mục data, data_processed hoặc thư mục gốc.")
        st.write("Hãy kiểm tra cấu trúc thư mục dự án:")
        st.code(
            "m5-forecasting-accuracy/\n"
            "├── app/app.py\n"
            "├── data/\n"
            "└── data_processed/"
        )
    else:
        # Hiển thị đường dẫn tương đối để người dùng dễ nhận biết file
        file_options = {
            f"{path.parent.name}/{path.name}" if path.parent != PROJECT_DIR else path.name: path
            for path in csv_files
        }
        default_label = next(
            (label for label in file_options if label.endswith("model_data.csv")),
            next(iter(file_options))
        )

        selected_label = st.selectbox(
            "Chọn tệp dữ liệu",
            options=list(file_options.keys()),
            index=list(file_options.keys()).index(default_label)
        )
        selected_path = file_options[selected_label]

        file_size_mb = selected_path.stat().st_size / (1024 * 1024)
        st.caption(f"Vị trí: `{selected_path.relative_to(PROJECT_DIR)}`")

        try:
            preview_limit = st.selectbox(
                "Số dòng xem trước",
                options=[10, 20, 50, 100, 500, 1000],
                index=2
            )
            preview_df = read_csv_preview(selected_path, nrows=preview_limit)

            # Thông tin tổng quan của file và phần dữ liệu đang xem
            c1, c2, c3 = st.columns(3)
            c1.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
            c2.metric("Số cột", f"{len(preview_df.columns):,}")
            c3.metric("Số dòng đang xem", f"{len(preview_df):,}")

            st.subheader("Xem trước dữ liệu")
            st.dataframe(preview_df, use_container_width=True, height=360)

            st.caption(
                f"Bảng trên hiển thị tối đa {preview_limit:,} dòng đầu tiên. "
                "Số dòng đang xem không phải tổng số dòng của toàn bộ tệp."
            )

            st.subheader("Thông tin cấu trúc dữ liệu")
            schema_df = pd.DataFrame({
                "Tên thuộc tính": preview_df.columns,
                "Kiểu dữ liệu": [str(dtype) for dtype in preview_df.dtypes],
                "Số giá trị thiếu (trong phần xem trước)": [
                    int(preview_df[col].isna().sum()) for col in preview_df.columns
                ]
            })
            st.dataframe(schema_df, use_container_width=True, height=300)

            st.subheader("Kiểm tra chất lượng dữ liệu mẫu")
            q1, q2 = st.columns(2)
            q1.metric(
                "Ô dữ liệu thiếu",
                f"{int(preview_df.isna().sum().sum()):,}"
            )
            q2.metric(
                "Dòng trùng lặp",
                f"{int(preview_df.duplicated().sum()):,}"
            )
            st.caption(
                "Các chỉ số chất lượng được tính trên phần dữ liệu xem trước, "
                "không đại diện cho toàn bộ tệp nếu tệp có nhiều hơn số dòng đã đọc."
            )

            # Chỉ tạo nút tải xuống khi tệp không quá lớn để tránh tiêu tốn bộ nhớ
            if file_size_mb <= 100:
                with open(selected_path, "rb") as file:
                    file_bytes = file.read()
                st.download_button(
                    label="⬇️ Tải xuống tệp CSV",
                    data=file_bytes,
                    file_name=selected_path.name,
                    mime="text/csv"
                )
            else:
                st.info(
                    "Tệp lớn hơn 100 MB. Để tránh nạp toàn bộ tệp vào bộ nhớ "
                    "của ứng dụng, chức năng tải xuống trực tiếp đang được ẩn."
                )

        except Exception as error:
            st.error(f"Không thể đọc tệp dữ liệu: {error}")
            st.info(
                "Hãy kiểm tra định dạng CSV, dấu phân cách và mã hóa của tệp."
            )


# =========================
# 6. Các chức năng sẽ triển khai ở bước tiếp theo
# =========================
else:
    st.markdown(
        f'<p class="main-title">{menu.upper()}</p>',
        unsafe_allow_html=True
    )

    messages = {
        "Dự báo nhu cầu": "Bước tiếp theo sẽ kết nối dữ liệu đã xử lý với mô hình Random Forest và Baseline.",
        "So sánh mô hình": "Chức năng sẽ trình bày kết quả MAE và WAPE của các mô hình trên cùng tập kiểm tra.",
        "Đề xuất bổ sung hàng": "Chức năng sẽ tính lượng hàng cần bổ sung dựa trên nhu cầu dự báo và giả định tồn kho.",
        "Mô phỏng tồn kho": "Chức năng sẽ mô phỏng diễn biến tồn kho, tỷ lệ thiếu hàng và tồn kho trung bình."
    }
    st.info(messages.get(menu, "Chức năng đang được phát triển."))
