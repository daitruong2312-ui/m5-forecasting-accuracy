
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. CAU HINH DU AN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULT_DIR = os.path.join(BASE_DIR, "output", "seasonal_naive")

SALES_FILE = os.path.join(
    DATA_DIR, "sales_train_validation.csv"
)

CALENDAR_FILE = os.path.join(
    DATA_DIR, "calendar.csv"
)

STORE_ID = "CA_1"
CATEGORY_ID = "FOODS"

FORECAST_HORIZON = 28
SEASON_LENGTH = 7

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 2. KIEM TRA FILE
# ============================================================

print("\n" + "=" * 60)
print("SEASONAL NAIVE - M5 FORECASTING")
print("=" * 60)

for file_path in [SALES_FILE, CALENDAR_FILE]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Khong tim thay file: {file_path}"
        )

print("Da tim thay file ban hang va Calendar.")


# ============================================================
# 3. DOC DU LIEU BAN HANG
# ============================================================

print("\n[1/7] Dang doc du lieu ban hang...")

header = pd.read_csv(
    SALES_FILE,
    nrows=0,
    encoding="utf-8-sig"
)

metadata_cols = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

missing_cols = [
    col for col in metadata_cols
    if col not in header.columns
]

if missing_cols:
    raise ValueError(
        f"Thieu cot trong sales file: {missing_cols}"
    )

day_cols = sorted(
    [col for col in header.columns if col.startswith("d_")],
    key=lambda x: int(x.split("_")[1])
)

if len(day_cols) <= 2 * FORECAST_HORIZON:
    raise ValueError("Khong du du lieu de chia Train/Validation/Test.")

use_cols = metadata_cols + day_cols

parts = []

for chunk in pd.read_csv(
    SALES_FILE,
    usecols=use_cols,
    chunksize=300,
    encoding="utf-8-sig"
):
    selected = chunk[
        (chunk["store_id"] == STORE_ID) &
        (chunk["cat_id"] == CATEGORY_ID)
    ]

    if not selected.empty:
        parts.append(selected)

if not parts:
    raise ValueError(
        f"Khong tim thay du lieu {CATEGORY_ID} tai {STORE_ID}."
    )

sales = pd.concat(parts, ignore_index=True)

sales[day_cols] = sales[day_cols].apply(
    pd.to_numeric, errors="coerce"
).fillna(0)

print("So san pham:", len(sales))
print("So ngay:", len(day_cols))
print("Cua hang:", STORE_ID)
print("Nhom hang:", CATEGORY_ID)


# ============================================================
# 4. DOC CALENDAR
# ============================================================

print("\n[2/7] Dang doc Calendar...")

calendar = pd.read_csv(
    CALENDAR_FILE,
    encoding="utf-8-sig"
)

calendar.columns = (
    calendar.columns.astype(str)
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

required_calendar_cols = ["d", "date"]

missing_calendar = [
    col for col in required_calendar_cols
    if col not in calendar.columns
]

if missing_calendar:
    raise ValueError(
        f"Calendar thieu cot: {missing_calendar}. "
        f"Cac cot hien co: {calendar.columns.tolist()}"
    )

calendar = calendar[["d", "date"]].copy()

calendar["d"] = calendar["d"].astype(str).str.strip()
calendar["date"] = pd.to_datetime(
    calendar["date"], errors="coerce"
)

calendar = calendar.dropna(subset=["date"])

date_map = dict(zip(calendar["d"], calendar["date"]))

missing_days = [
    d for d in day_cols
    if d not in date_map
]

if missing_days:
    raise ValueError(
        f"Calendar khong co cac ngay: {missing_days[:5]}"
    )

dates = pd.DatetimeIndex(
    [date_map[d] for d in day_cols]
)

print("Khoang thoi gian:", dates.min().date(), "-", dates.max().date())


# ============================================================
# 5. CHIA TRAIN - VALIDATION - TEST
# ============================================================

print("\n[3/7] Dang chia du lieu...")

n_days = len(day_cols)

test_start = n_days - FORECAST_HORIZON
validation_start = test_start - FORECAST_HORIZON

if validation_start <= SEASON_LENGTH:
    raise ValueError("Khong du so ngay Train.")

train_cols = day_cols[:validation_start]
validation_cols = day_cols[
    validation_start:test_start
]
test_cols = day_cols[test_start:]

print("Train:", len(train_cols), "ngay")
print("Validation:", len(validation_cols), "ngay")
print("Test:", len(test_cols), "ngay")

print(
    "Train:",
    dates[0].date(),
    "den",
    dates[validation_start - 1].date()
)

print(
    "Validation:",
    dates[validation_start].date(),
    "den",
    dates[test_start - 1].date()
)

print(
    "Test:",
    dates[test_start].date(),
    "den",
    dates[-1].date()
)


# ============================================================
# 6. HAM DU BAO SEASONAL NAIVE DE QUY
# ============================================================

def recursive_seasonal_naive(history, horizon=28, season=7):
    """
    Du bao nhieu ngay lien tiep.

    Moi ngay:
    - Lay gia tri cach ngay du bao 7 ngay.
    - Neu ngay do nam trong khoang du bao,
      su dung gia tri da du bao truoc do.
    """

    history = np.asarray(history, dtype=float)

    if history.ndim != 2:
        raise ValueError("History phai la ma tran (san pham, ngay).")

    if history.shape[1] < season:
        raise ValueError("Khong du lich su cho Seasonal Naive.")

    n_products = history.shape[0]

    predictions = np.zeros(
        (n_products, horizon),
        dtype=float
    )

    # Chi su dung lich su truoc thoi diem du bao.
    # Cac ngay tiep theo se duoc noi bang du bao.
    extended_history = history.copy()

    for step in range(horizon):

        source_index = extended_history.shape[1] - season

        daily_prediction = extended_history[:, source_index]

        predictions[:, step] = daily_prediction

        extended_history = np.column_stack(
            [extended_history, daily_prediction]
        )

    return predictions


# ============================================================
# 7. HAM TINH CHI SO
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    total_actual = np.sum(np.abs(actual))

    if total_actual > 0:
        wape = (
            np.sum(np.abs(actual - predicted))
            / total_actual
        ) * 100
    else:
        wape = np.nan

    return mae, rmse, wape


# ============================================================
# 8. DU BAO VALIDATION
# ============================================================

print("\n[4/7] Dang du bao Validation...")

y_train = sales[train_cols].to_numpy(dtype=float)

y_validation = sales[
    validation_cols
].to_numpy(dtype=float)

validation_prediction = recursive_seasonal_naive(
    y_train,
    horizon=FORECAST_HORIZON,
    season=SEASON_LENGTH
)

val_mae, val_rmse, val_wape = calculate_metrics(
    y_validation,
    validation_prediction
)

print("\n===== VALIDATION =====")
print(f"MAE  : {val_mae:.4f}")
print(f"RMSE : {val_rmse:.4f}")
print(f"WAPE : {val_wape:.2f}%")


# ============================================================
# 9. DU BAO TEST
# ============================================================

print("\n[5/7] Dang du bao Test...")

# Khi du bao Test, duoc phep su dung doanh so thuc te
# cua Train + Validation.
# Tuyet doi khong su dung doanh so Test de tao du bao.

history_test = sales[
    train_cols + validation_cols
].to_numpy(dtype=float)

y_test = sales[
    test_cols
].to_numpy(dtype=float)

test_prediction = recursive_seasonal_naive(
    history_test,
    horizon=FORECAST_HORIZON,
    season=SEASON_LENGTH
)

test_mae, test_rmse, test_wape = calculate_metrics(
    y_test,
    test_prediction
)

print("\n===== TEST =====")
print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"WAPE : {test_wape:.2f}%")


# ============================================================
# 10. LUU BANG CHI SO
# ============================================================

print("\n[6/7] Dang luu ket qua...")

metrics_df = pd.DataFrame([
    {
        "Model": "Seasonal Naive",
        "Dataset": "Validation",
        "Store": STORE_ID,
        "Category": CATEGORY_ID,
        "Forecast_Days": FORECAST_HORIZON,
        "MAE": val_mae,
        "RMSE": val_rmse,
        "WAPE_Percent": val_wape
    },
    {
        "Model": "Seasonal Naive",
        "Dataset": "Test",
        "Store": STORE_ID,
        "Category": CATEGORY_ID,
        "Forecast_Days": FORECAST_HORIZON,
        "MAE": test_mae,
        "RMSE": test_rmse,
        "WAPE_Percent": test_wape
    }
])

metrics_path = os.path.join(
    RESULT_DIR,
    "seasonal_naive_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. LUU DU BAO CHI TIET THEO SAN PHAM
# ============================================================

forecast_records = []

for step, day in enumerate(validation_cols):
    for product_index in range(len(sales)):

        forecast_records.append({
            "Dataset": "Validation",
            "Date": dates[validation_start + step],
            "d": day,
            "item_id": sales.iloc[product_index]["item_id"],
            "Actual": y_validation[product_index, step],
            "Forecast": validation_prediction[product_index, step]
        })

for step, day in enumerate(test_cols):
    for product_index in range(len(sales)):

        forecast_records.append({
            "Dataset": "Test",
            "Date": dates[test_start + step],
            "d": day,
            "item_id": sales.iloc[product_index]["item_id"],
            "Actual": y_test[product_index, step],
            "Forecast": test_prediction[product_index, step]
        })

forecast_detail = pd.DataFrame(forecast_records)

forecast_detail.to_csv(
    os.path.join(RESULT_DIR, "forecast_detail.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. VE BIEU DO TEST
# ============================================================

print("\n[7/7] Dang ve bieu do...")

actual_daily = y_test.sum(axis=0)
pred_daily = test_prediction.sum(axis=0)

test_dates = dates[test_start:]

plt.figure(figsize=(14, 6))

plt.plot(
    test_dates,
    actual_daily,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    test_dates,
    pred_daily,
    marker="x",
    linestyle="--",
    label="Seasonal Naive Forecast"
)

plt.title(
    f"Seasonal Naive - {STORE_ID} {CATEGORY_ID} - Test 28 Days"
)

plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "actual_vs_forecast.png"),
    dpi=300
)

plt.close()


# ============================================================
# 13. VE PHAN PHOI SAI SO
# ============================================================

errors = (
    y_test - test_prediction
).ravel()

plt.figure(figsize=(10, 5))

plt.hist(
    errors,
    bins=50,
    edgecolor="black"
)

plt.title("Seasonal Naive - Test Error Distribution")
plt.xlabel("Error (Actual - Forecast)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "error_distribution.png"),
    dpi=300
)

plt.close()


# ============================================================
# 14. LUU DU BAO TONG THEO NGAY
# ============================================================

daily_forecast = pd.DataFrame({
    "Date": test_dates,
    "Actual_Total_Sales": actual_daily,
    "Forecast_Total_Sales": pred_daily
})

daily_forecast.to_csv(
    os.path.join(RESULT_DIR, "daily_forecast.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. HOAN THANH
# ============================================================

print("\n" + "=" * 60)
print("HOAN THANH SEASONAL NAIVE")
print("=" * 60)

print("\nKet qua duoc luu tai:")
print(RESULT_DIR)

print("\nCac file da tao:")
for filename in os.listdir(RESULT_DIR):
    print("-", filename)

print("\n===== TONG HOP CHI SO =====")
print(metrics_df.to_string(index=False))

print("\n===== DU BAO TONG THE TEST =====")
print(daily_forecast.to_string(index=False))
