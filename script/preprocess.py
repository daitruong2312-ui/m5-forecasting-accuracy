

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

# ========================= CẤU HÌNH =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_processed"

SALES_FILE = DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE = DATA_DIR / "calendar.csv"
PRICES_FILE = DATA_DIR / "sell_prices.csv"

STORE_ID = "CA_1"       # Các lựa chọn thường gặp: CA_1, CA_2, CA_3, CA_4, TX_1...
CATEGORY = "FOODS"      # Các lựa chọn: FOODS, HOBBIES, HOUSEHOLD
TEST_DAYS = 28          # Số ngày cuối dùng để kiểm tra
LAGS = [1, 7, 14, 28]   # Lượng bán của 1, 7, 14, 28 ngày trước
ROLLING_WINDOWS = [7, 28]
# ============================================================


def require_files():
    missing = [str(p) for p in [SALES_FILE, CALENDAR_FILE, PRICES_FILE] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Không tìm thấy các file sau:\n- "
            + "\n- ".join(missing)
            + "\n\nHãy đặt 3 file CSV vào thư mục data/ nằm cạnh preprocess.py."
        )


def load_and_filter_sales():
    print("1/6 Đọc và lọc dữ liệu bán hàng...")
    # Chỉ đọc các cột định danh trước để lọc nhanh, tránh đọc toàn bộ dữ liệu lớn.
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_ids = pd.read_csv(SALES_FILE, usecols=id_cols)
    mask = (sales_ids["store_id"] == STORE_ID) & (sales_ids["cat_id"] == CATEGORY)
    selected_ids = sales_ids.loc[mask, id_cols].copy()

    if selected_ids.empty:
        stores = sorted(sales_ids["store_id"].dropna().unique().tolist())
        cats = sorted(sales_ids["cat_id"].dropna().unique().tolist())
        raise ValueError(
            f"Không có dữ liệu cho STORE_ID={STORE_ID}, CATEGORY={CATEGORY}.\n"
            f"Cửa hàng có sẵn: {stores}\nNhóm sản phẩm có sẵn: {cats}"
        )

    # Chỉ đọc các dòng được chọn và các cột ngày. Đọc bảng sales một lần với usecols,
    # sau đó lọc theo id để không cần chuyển toàn bộ dataset thành dạng dài.
    print(f"   Số sản phẩm được chọn: {len(selected_ids):,}")
    sales = pd.read_csv(SALES_FILE)
    sales = sales[sales["id"].isin(selected_ids["id"])].copy()

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if not day_cols:
        raise ValueError("Không tìm thấy các cột ngày dạng d_1, d_2, ... trong file sales.")

    # Chuyển bảng rộng sang bảng dài: một dòng cho một sản phẩm/ngày.
    sales_long = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales"
    )
    sales_long["sales"] = pd.to_numeric(sales_long["sales"], errors="coerce")
    sales_long["sales"] = sales_long["sales"].fillna(0).clip(lower=0)
    sales_long["d_num"] = sales_long["d"].str.replace("d_", "", regex=False).astype(int)
    print(f"   Số dòng sau khi chuyển dạng dài: {len(sales_long):,}")
    return sales_long


def load_calendar():
    print("2/6 Đọc và làm sạch lịch...")
    calendar = pd.read_csv(CALENDAR_FILE)
    needed = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2"
    ]
    keep = [c for c in needed if c in calendar.columns]
    calendar = calendar[keep].copy()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    calendar = calendar.drop_duplicates(subset=["d"])
    calendar["is_event"] = (
        calendar.get("event_name_1", pd.Series(index=calendar.index, dtype="object")).notna()
        | calendar.get("event_name_2", pd.Series(index=calendar.index, dtype="object")).notna()
    ).astype("int8")
    calendar["is_weekend"] = calendar["date"].dt.dayofweek.isin([5, 6]).astype("int8")
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    return calendar


def load_prices(selected_sales_ids):
    print("3/6 Đọc và làm sạch giá bán...")
    # Chỉ lấy giá của cửa hàng và các sản phẩm đã lọc.
    prices = pd.read_csv(
        PRICES_FILE,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"]
    )
    item_ids = set(selected_sales_ids["item_id"].unique())
    prices = prices[
        (prices["store_id"] == STORE_ID) & (prices["item_id"].isin(item_ids))
    ].copy()
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="coerce")
    prices.loc[prices["sell_price"] <= 0, "sell_price"] = np.nan
    prices = prices.drop_duplicates(subset=["store_id", "item_id", "wm_yr_wk"])
    return prices


def make_features():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sales_long = load_and_filter_sales()
    selected_sales_ids = sales_long[["item_id", "store_id"]].drop_duplicates()

    calendar = load_calendar()
    prices = load_prices(selected_sales_ids)

    print("4/6 Kết hợp bảng bán hàng, lịch và giá...")
    df = sales_long.merge(calendar, on="d", how="left", validate="many_to_one")
    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one"
    )

    # Sắp xếp theo sản phẩm và ngày để tạo đặc trưng lịch sử đúng thứ tự.
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

    # Giá thiếu: nội suy theo từng sản phẩm trong chuỗi thời gian, sau đó dùng trung vị
    # giá của sản phẩm; nếu vẫn thiếu thì dùng trung vị toàn bộ giá có sẵn.
    df["sell_price"] = df.groupby("item_id")["sell_price"].transform(
        lambda s: s.ffill().bfill()
    )
    global_price_median = df["sell_price"].median()
    if pd.isna(global_price_median):
        global_price_median = 0.0
    df["sell_price"] = df["sell_price"].fillna(global_price_median)

    # Các trường lịch có thể thiếu ở một số bản ghi; giữ event là 0 nếu không có sự kiện.
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Đặc trưng thời gian.
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["is_weekend"].fillna(0).astype("int8")
    df["is_event"] = df["is_event"].fillna(0).astype("int8")

    # Mã hóa các biến phân loại thành số để thuận tiện khi dùng nhiều mô hình.
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if col in df.columns:
            df[col + "_code"] = pd.Categorical(df[col]).codes.astype("int32")

    print("5/6 Tạo đặc trưng lịch sử bán hàng...")
    # Shift trước khi rolling để không dùng doanh số của chính ngày cần dự báo.
    grouped_sales = df.groupby("item_id", sort=False)["sales"]
    for lag in LAGS:
        df[f"sales_lag_{lag}"] = grouped_sales.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"sales_roll_mean_{window}"] = df.groupby("item_id", sort=False)["sales"].transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
        )

    # Các dòng đầu chuỗi chưa có đủ lịch sử. Giữ lại sau khi loại các dòng thiếu lag lớn nhất
    # để tập dữ liệu có cùng tiêu chuẩn đặc trưng.
    max_lag = max(LAGS)
    df = df[df["d_num"] > max_lag].copy()

    # Không đưa cột ngày gốc dạng chuỗi vào mô hình; giữ ngày để phân chia theo thời gian.
    feature_cols = [
        "item_id_code", "dept_id_code", "cat_id_code", "store_id_code", "state_id_code",
        "event_name_1_code", "event_type_1_code", "event_name_2_code", "event_type_2_code",
        "sell_price", "month", "year", "day_of_month", "day_of_week",
        "week_of_year", "is_weekend", "is_event"
    ]
    feature_cols += [f"sales_lag_{lag}" for lag in LAGS]
    feature_cols += [f"sales_roll_mean_{w}" for w in ROLLING_WINDOWS]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Thay giá trị thiếu còn lại ở các đặc trưng số bằng 0; doanh số mục tiêu không thiếu.
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["sales"] = df["sales"].fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    unique_dates = np.sort(df["date"].unique())
    if len(unique_dates) <= TEST_DAYS:
        raise ValueError(
            f"Chỉ có {len(unique_dates)} ngày hợp lệ, không đủ để giữ {TEST_DAYS} ngày kiểm tra."
        )
    test_start = pd.Timestamp(unique_dates[-TEST_DAYS])

    # Chia theo thời gian: các ngày trước test_start để huấn luyện, TEST_DAYS ngày cuối để kiểm tra.
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()

    # Lưu cả bảng đầy đủ và các tập train/test. Giữ cột date/item_id/sales để diễn giải kết quả.
    metadata_cols = ["date", "d", "item_id", "store_id", "cat_id", "sales"]
    metadata_cols = [c for c in metadata_cols if c in df.columns]
    output_cols = metadata_cols + feature_cols
    output_cols = list(dict.fromkeys(output_cols))

    full_path = OUTPUT_DIR / "model_data.csv"
    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    df[output_cols].to_csv(full_path, index=False)
    train_df[output_cols].to_csv(train_path, index=False)
    test_df[output_cols].to_csv(test_path, index=False)

    # Ghi tóm tắt để tiện đưa số liệu vào báo cáo.
    summary = [
        ("Cửa hàng", STORE_ID),
        ("Nhóm sản phẩm", CATEGORY),
        ("Số sản phẩm", int(df["item_id"].nunique())),
        ("Số dòng dữ liệu sau xử lý", int(len(df))),
        ("Số ngày", int(df["date"].nunique())),
        ("Ngày bắt đầu tập kiểm tra", str(test_start.date())),
        ("Số dòng tập huấn luyện", int(len(train_df))),
        ("Số dòng tập kiểm tra", int(len(test_df))),
        ("Số đặc trưng đầu vào", int(len(feature_cols))),
        ("Tỷ lệ thiếu giá sau xử lý", f"{df['sell_price'].isna().mean():.2%}"),
    ]
    pd.DataFrame(summary, columns=["Thông tin", "Giá trị"]).to_csv(
        OUTPUT_DIR / "preprocessing_summary.csv", index=False, encoding="utf-8-sig"
    )

    print("\nHoàn tất tiền xử lý!")
    print(f"- Dữ liệu đầy đủ: {full_path}")
    print(f"- Tập huấn luyện: {train_path}")
    print(f"- Tập kiểm tra: {test_path}")
    print(f"- Tóm tắt: {OUTPUT_DIR / 'preprocessing_summary.csv'}")
    print(f"\nCửa hàng: {STORE_ID} | Nhóm sản phẩm: {CATEGORY}")
    print(f"Số sản phẩm: {df['item_id'].nunique():,}")
    print(f"Số dòng train: {len(train_df):,} | test: {len(test_df):,}")
    print(f"Khoảng kiểm tra: {test_df['date'].min().date()} đến {test_df['date'].max().date()}")
    print("\nLưu ý: Đây là thiết lập đánh giá dự báo một bước trước (one-step-ahead).")
    print("Khi dự báo nhiều ngày liên tiếp mà không biết doanh số thực tế từng ngày,")
    print("cần cập nhật lag bằng dự báo hoặc dùng chiến lược dự báo nhiều bước riêng.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    require_files()
    make_features()
