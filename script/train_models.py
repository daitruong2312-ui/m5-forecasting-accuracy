
# ============================================================
# TRAIN RANDOM FOREST & LIGHTGBM - M5 FORECASTING
# Store: CA_1
# Category: FOODS
#
# Forecasting strategy:
# - Validation: forecast 28 consecutive days recursively
# - Test: forecast 28 consecutive days recursively
# - Never use actual sales inside the forecast horizon
# - Compare with recursive Seasonal Naive
# ============================================================

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "model_data.csv"

MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output" / "ml_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STORE = "CA_1"
CATEGORY = "FOODS"

HORIZON = 28
TRAIN_DAYS = 364
RANDOM_STATE = 42


# ============================================================
# 2. FEATURES
# ============================================================

FEATURES = [
    "item_id_code",
    "dept_id_code",
    "cat_id_code",
    "store_id_code",
    "state_id_code",
    "event_name_1_code",
    "event_type_1_code",
    "event_name_2_code",
    "event_type_2_code",
    "sell_price",
    "month",
    "year",
    "day_of_month",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "is_event",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_mean_28",
]

DYNAMIC_FEATURES = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_mean_28",
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("RANDOM FOREST & LIGHTGBM - M5 FORECASTING")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Khong tim thay file:\n{DATA_FILE}\n"
        "Hay kiem tra lai data_processed/model_data.csv."
    )

print("\nDang doc du lieu...")
df = pd.read_csv(DATA_FILE)

required_columns = [
    "date",
    "item_id",
    "sales",
] + FEATURES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Du lieu thieu cac cot: "
        + ", ".join(missing_columns)
    )

df["date"] = pd.to_datetime(df["date"])

# Lọc đúng cửa hàng và nhóm hàng nếu dữ liệu có các cột này.
if "store_id" in df.columns:
    df = df[df["store_id"] == STORE].copy()

if "cat_id" in df.columns:
    df = df[df["cat_id"] == CATEGORY].copy()

df = df.sort_values(
    ["date", "item_id"]
).reset_index(drop=True)

df["sales"] = pd.to_numeric(
    df["sales"], errors="coerce"
).fillna(0).clip(lower=0)

print(f"So dong: {len(df):,}")
print(f"So san pham: {df['item_id'].nunique():,}")
print(f"So ngay: {df['date'].nunique():,}")
print(f"Cua hang: {STORE}")
print(f"Nhom hang: {CATEGORY}")


# ============================================================
# 4. SPLIT TRAIN / VALIDATION / TEST
# ============================================================

all_dates = sorted(df["date"].unique())

if len(all_dates) <= HORIZON * 2:
    raise ValueError("Khong du so ngay de chia Train/Validation/Test.")

last_date = pd.Timestamp(all_dates[-1])

test_start = last_date - pd.Timedelta(days=HORIZON - 1)
validation_end = test_start - pd.Timedelta(days=1)
validation_start = validation_end - pd.Timedelta(days=HORIZON - 1)
train_end = validation_start - pd.Timedelta(days=1)

train_df = df[df["date"] <= train_end].copy()

validation_df = df[
    (df["date"] >= validation_start)
    & (df["date"] <= validation_end)
].copy()

test_df = df[
    (df["date"] >= test_start)
    & (df["date"] <= last_date)
].copy()

print("\n" + "-" * 65)
print("CHIA DU LIEU")
print("-" * 65)

print(
    f"Train:      {train_df['date'].min().date()} "
    f"-> {train_df['date'].max().date()} "
    f"({train_df['date'].nunique()} ngay)"
)

print(
    f"Validation: {validation_df['date'].min().date()} "
    f"-> {validation_df['date'].max().date()} "
    f"({validation_df['date'].nunique()} ngay)"
)

print(
    f"Test:       {test_df['date'].min().date()} "
    f"-> {test_df['date'].max().date()} "
    f"({test_df['date'].nunique()} ngay)"
)

if validation_df["date"].nunique() != HORIZON:
    raise ValueError("Validation khong du 28 ngay.")

if test_df["date"].nunique() != HORIZON:
    raise ValueError("Test khong du 28 ngay.")


# ============================================================
# 5. PREPARE TRAINING DATA
# ============================================================

# Chỉ dùng 364 ngày gần nhất trong giai đoạn Train.
training_start = train_end - pd.Timedelta(days=TRAIN_DAYS - 1)

fit_df = train_df[
    train_df["date"] >= training_start
].copy()

if fit_df.empty:
    raise ValueError("Khong co du lieu de huan luyen.")

X_train = (
    fit_df[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

y_train = fit_df["sales"].to_numpy(dtype=np.float32)

print("\n" + "-" * 65)
print("DU LIEU HUAN LUYEN")
print("-" * 65)
print(f"So dong huan luyen: {len(fit_df):,}")
print(f"So dac trung: {len(FEATURES)}")


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 65)
print("HUAN LUYEN RANDOM FOREST")
print("=" * 65)

start_time = time.time()

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    max_features=0.5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

rf.fit(X_train, y_train)

rf_train_time = time.time() - start_time

joblib.dump(
    rf,
    MODEL_DIR / "random_forest.pkl"
)

print(f"Thoi gian huan luyen: {rf_train_time:.2f} giay")
print("Da luu: model/random_forest.pkl")


# ============================================================
# 7. TRAIN LIGHTGBM
# ============================================================

print("\n" + "=" * 65)
print("HUAN LUYEN LIGHTGBM")
print("=" * 65)

start_time = time.time()

gbm = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=-1,
)

gbm.fit(X_train, y_train)

gbm_train_time = time.time() - start_time

joblib.dump(
    gbm,
    MODEL_DIR / "lightgbm.pkl"
)

print(f"Thoi gian huan luyen: {gbm_train_time:.2f} giay")
print("Da luu: model/lightgbm.pkl")


# ============================================================
# 8. RECURSIVE FORECAST
# ============================================================

def create_history(origin_df):
    """
    Lưu lịch sử doanh số của từng sản phẩm.
    Chỉ lấy doanh số đã xảy ra trước ngày bắt đầu dự báo.
    """
    history = {}

    for item_id, group in origin_df.groupby("item_id", sort=False):
        group = group.sort_values("date")
        history[item_id] = group["sales"].to_numpy(dtype=float).tolist()

    return history


def calculate_dynamic_features(history):
    """
    Tính lag và rolling từ lịch sử hiện có.
    Lịch sử gồm doanh số thực tế trước thời điểm dự báo
    và dự báo của những ngày trước đó.
    """

    values = np.asarray(history, dtype=float)

    if len(values) < 28:
        raise ValueError("Lich su san pham khong du 28 ngay.")

    return {
        "sales_lag_1": values[-1],
        "sales_lag_7": values[-7],
        "sales_lag_14": values[-14],
        "sales_lag_28": values[-28],
        "sales_roll_mean_7": values[-7:].mean(),
        "sales_roll_mean_28": values[-28:].mean(),
    }


def recursive_forecast(model, origin_df, future_df, model_name):
    """
    Dự báo lần lượt từng ngày.

    model:
        Random Forest hoặc LightGBM.
        Nếu model=None thì dùng Seasonal Naive.

    Không đọc cột sales của future_df trong quá trình dự báo.
    """

    history = create_history(origin_df)

    future_dates = sorted(future_df["date"].unique())
    result = []

    for step, forecast_date in enumerate(future_dates, start=1):

        # Các dòng của đúng ngày đang cần dự báo.
        day_df = future_df[
            future_df["date"] == forecast_date
        ].copy()

        day_df = day_df.sort_values("item_id").reset_index(drop=True)

        predictions = []

        if model is None:
            # Seasonal Naive: lấy doanh số cùng thứ của tuần trước.
            for item_id in day_df["item_id"]:
                values = history[item_id]
                prediction = values[-7]
                predictions.append(max(0, float(prediction)))

        else:
            # Tính lại các đặc trưng động cho từng sản phẩm.
            X_day = day_df[FEATURES].copy()

            for i, item_id in enumerate(day_df["item_id"]):
                dynamic_values = calculate_dynamic_features(
                    history[item_id]
                )

                for feature_name, value in dynamic_values.items():
                    X_day.loc[i, feature_name] = value

            X_day = (
                X_day
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )

            predictions = np.maximum(
                model.predict(X_day),
                0
            ).tolist()

        # Lưu kết quả và cập nhật lịch sử bằng dự báo.
        # Tuyệt đối không cập nhật bằng sales thực tế của ngày tương lai.
        for item_id, prediction in zip(
            day_df["item_id"], predictions
        ):
            history[item_id].append(float(prediction))

            result.append({
                "date": forecast_date,
                "item_id": item_id,
                "forecast": float(prediction),
                "model": model_name,
            })

        if step == 1 or step % 7 == 0 or step == len(future_dates):
            print(
                f"{model_name}: da du bao "
                f"{step}/{len(future_dates)} ngay"
            )

    return pd.DataFrame(result)


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_forecast(actual_df, prediction_df, period, model_name):

    actual = actual_df[
        ["date", "item_id", "sales"]
    ].copy()

    actual = actual.rename(
        columns={"sales": "actual"}
    )

    merged = actual.merge(
        prediction_df,
        on=["date", "item_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(actual):
        raise ValueError(
            f"{model_name}: so dong du bao khong khop "
            f"voi du lieu thuc te trong {period}."
        )

    y_true = merged["actual"].to_numpy(dtype=float)
    y_pred = merged["forecast"].to_numpy(dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    total_actual = np.abs(y_true).sum()

    if total_actual == 0:
        wape = np.nan
    else:
        wape = (
            np.abs(y_true - y_pred).sum()
            / total_actual
            * 100
        )

    return {
        "Period": period,
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE (%)": wape,
        "Train time (seconds)": np.nan,
    }, merged


# ============================================================
# 10. RUN VALIDATION AND TEST
# ============================================================

models = {
    "Random Forest": rf,
    "LightGBM": gbm,
    "Seasonal Naive": None,
}

train_times = {
    "Random Forest": rf_train_time,
    "LightGBM": gbm_train_time,
    "Seasonal Naive": np.nan,
}

all_metrics = []
all_details = []

periods = [
    ("Validation", validation_df),
    ("Test", test_df),
]

for period_name, target_df in periods:

    print("\n" + "=" * 65)
    print(f"DU BAO GIAI DOAN {period_name.upper()}")
    print("=" * 65)

    # Validation chỉ được dùng lịch sử Train.
    # Test được dùng lịch sử Train + Validation thực tế.
    if period_name == "Validation":
        origin_df = train_df.copy()
    else:
        origin_df = pd.concat(
            [train_df, validation_df],
            ignore_index=True
        )

    for model_name, model in models.items():

        print(f"\nDang du bao: {model_name}")

        predictions = recursive_forecast(
            model=model,
            origin_df=origin_df,
            future_df=target_df,
            model_name=model_name,
        )

        metrics, detail = evaluate_forecast(
            actual_df=target_df,
            prediction_df=predictions,
            period=period_name,
            model_name=model_name,
        )

        metrics["Train time (seconds)"] = train_times[model_name]

        all_metrics.append(metrics)
        all_details.append(detail)

        # Lưu kết quả riêng từng mô hình và giai đoạn.
        safe_name = model_name.lower().replace(" ", "_")

        predictions.to_csv(
            OUTPUT_DIR / f"{safe_name}_{period_name.lower()}.csv",
            index=False,
        )

        print(
            f"{model_name} | "
            f"MAE={metrics['MAE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"WAPE={metrics['WAPE (%)']:.2f}%"
        )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)

details_df = pd.concat(
    all_details,
    ignore_index=True
)

metrics_path = OUTPUT_DIR / "model_metrics.csv"
details_path = OUTPUT_DIR / "forecast_detail.csv"

metrics_df.to_csv(metrics_path, index=False)
details_df.to_csv(details_path, index=False)

print("\n" + "=" * 65)
print("KET QUA SO SANH MO HINH")
print("=" * 65)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nDa luu ket qua:")
print(metrics_path)
print(details_path)

print("\nHoan tat huan luyen va danh gia.")
