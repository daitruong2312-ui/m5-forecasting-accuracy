from pathlib import Path
import numpy as np
import pandas as pd


def _read_csv(path, required=None, parse_dates=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path, parse_dates=parse_dates or [])
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"File {path.name} thiếu cột: {missing}")
    return df


def load_project_data(project_dir):
    """Đọc lịch sử train, actual test và dự báo Random Forest từ dự án M5."""
    root = Path(project_dir)
    data_dir = root / "data" / "processed"
    output_dir = root / "output" / "ml_models"

    train = _read_csv(
        data_dir / "train.csv",
        ["date", "item_id", "sales"],
        ["date"],
    )
    test = _read_csv(
        data_dir / "test.csv",
        ["date", "item_id", "sales"],
        ["date"],
    )

    # Ưu tiên file dự báo riêng đã có trong output/ml_models.
    pred_path = output_dir / "random_forest_test.csv"
    if not pred_path.exists():
        pred_path = output_dir / "forecast_detail.csv"

    pred = _read_csv(pred_path)
    if "y_pred" in pred.columns and "forecast" not in pred.columns:
        pred = pred.rename(columns={"y_pred": "forecast"})
    required_pred = ["date", "item_id", "forecast"]
    missing = [c for c in required_pred if c not in pred.columns]
    if missing:
        raise ValueError(
            f"{pred_path.name} thiếu cột {missing}. "
            "Cần file dự báo có date, item_id, forecast."
        )
    pred["date"] = pd.to_datetime(pred["date"])
    pred["forecast"] = pd.to_numeric(pred["forecast"], errors="coerce")
    pred = pred.dropna(subset=["date", "item_id", "forecast"])

    # Nếu dùng forecast_detail gộp nhiều mô hình, chỉ giữ Random Forest.
    if "model" in pred.columns:
        rf = pred[pred["model"].astype(str).str.contains(
            "random forest", case=False, na=False
        )]
        if not rf.empty:
            pred = rf.copy()

    pred = pred.groupby(["date", "item_id"], as_index=False)["forecast"].mean()
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    train["sales"] = pd.to_numeric(train["sales"], errors="coerce").fillna(0)
    test["sales"] = pd.to_numeric(test["sales"], errors="coerce").fillna(0)

    # Chỉ tính sai số từ phần dự báo trước giai đoạn test nếu có.
    # Không dùng actual của giai đoạn đang mô phỏng để tạo safety stock.
    sigma = pd.DataFrame(columns=["item_id", "sigma_e"])
    detail_path = output_dir / "forecast_detail.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        needed = {"date", "item_id", "actual", "forecast"}
        if needed.issubset(detail.columns):
            detail["date"] = pd.to_datetime(detail["date"])
            if "model" in detail.columns:
                detail = detail[detail["model"].astype(str).str.contains(
                    "random forest", case=False, na=False
                )]
            first_test_date = pred["date"].min()
            validation = detail[detail["date"] < first_test_date].copy()
            if not validation.empty:
                validation["error"] = (
                    pd.to_numeric(validation["actual"], errors="coerce")
                    - pd.to_numeric(validation["forecast"], errors="coerce")
                )
                sigma = (
                    validation.dropna(subset=["error"])
                    .groupby("item_id")["error"]
                    .std()
                    .fillna(0)
                    .clip(lower=0)
                    .rename("sigma_e")
                    .reset_index()
                )

    # Fallback: độ lệch chuẩn doanh số lịch sử 28 ngày, ghi rõ là đại diện thay thế.
    if sigma.empty:
        recent_start = train["date"].max() - pd.Timedelta(days=27)
        recent = train[train["date"] >= recent_start]
        sigma = (
            recent.groupby("item_id")["sales"]
            .std()
            .fillna(0)
            .rename("sigma_e")
            .reset_index()
        )
        sigma["sigma_source"] = "Độ lệch chuẩn doanh số 28 ngày (thay thế)"

    if "sigma_source" not in sigma.columns:
        sigma["sigma_source"] = "Sai số dự báo Random Forest trên Validation"

    return train, test, pred, sigma


def build_recommendations(train, predictions, sigma, lead_time=6,
                          review_interval=1, stock_cover_days=3, z_value=1.65,
                          custom_stock=None, custom_on_order=None):
    """Tính lượng bổ sung theo nhu cầu trong lead time + chu kỳ xem xét."""
    if lead_time < 1 or review_interval < 1 or stock_cover_days < 0:
        raise ValueError("Thời gian giao hàng/chu kỳ xem xét phải >= 1; tồn kho phủ >= 0.")
    horizon = int(lead_time + review_interval)
    as_of = pd.to_datetime(train["date"]).max()
    future = predictions[
        (predictions["date"] > as_of)
        & (predictions["date"] <= as_of + pd.Timedelta(days=horizon))
    ].copy()
    if future.empty:
        raise ValueError(
            f"Không có dự báo trong {horizon} ngày sau {as_of.date()}."
        )

    days_per_item = future.groupby("item_id")["date"].nunique()
    valid_items = days_per_item[days_per_item == horizon].index
    future = future[future["item_id"].isin(valid_items)]
    if future.empty:
        raise ValueError(
            f"Không có sản phẩm nào có đủ {horizon} ngày dự báo."
        )

    demand = (
        future.groupby("item_id", as_index=False)
        .agg(forecast_demand=("forecast", "sum"),
             forecast_days=("date", "nunique"))
    )

    history_start = as_of - pd.Timedelta(days=27)
    recent = train[
        train["date"].between(history_start, as_of)
    ]
    avg = (
        recent.groupby("item_id", as_index=False)["sales"]
        .mean()
        .rename(columns={"sales": "avg_daily_sales"})
    )
    result = demand.merge(avg, on="item_id", how="left")
    result = result.merge(sigma[["item_id", "sigma_e"]], on="item_id", how="left")
    result["avg_daily_sales"] = result["avg_daily_sales"].fillna(0)
    result["sigma_e"] = result["sigma_e"].fillna(0).clip(lower=0)

    result["on_hand"] = np.ceil(
        result["avg_daily_sales"] * stock_cover_days
    ).astype(int)
    if custom_stock is not None and "item_id" in custom_stock.columns:
        stock_map = custom_stock.set_index("item_id")["on_hand"]
        result["on_hand"] = result["item_id"].map(stock_map).fillna(
            result["on_hand"]
        ).clip(lower=0).astype(int)

    result["on_order"] = 0
    if custom_on_order is not None and "item_id" in custom_on_order.columns:
        order_map = custom_on_order.set_index("item_id")["on_order"]
        result["on_order"] = result["item_id"].map(order_map).fillna(0).clip(lower=0).astype(int)

    result["safety_stock"] = z_value * result["sigma_e"] * np.sqrt(horizon)
    result["inventory_position"] = result["on_hand"] + result["on_order"]
    result["suggested_qty"] = np.ceil(
        (result["forecast_demand"] + result["safety_stock"]
         - result["inventory_position"]).clip(lower=0)
    ).astype(int)
    result["recommendation"] = np.where(
        result["suggested_qty"] > 0, "Cần bổ sung", "Chưa cần bổ sung"
    )
    result["planning_date"] = as_of
    result["lead_time_days"] = lead_time
    result["review_interval_days"] = review_interval
    result["coverage_days"] = horizon
    return result.sort_values(
        ["suggested_qty", "forecast_demand"], ascending=False
    ).reset_index(drop=True)


def simulate_inventory(train, test, predictions, sigma, lead_time=6,
                       review_interval=1, stock_cover_days=3, z_value=1.65):
    """Mô phỏng từng ngày; đơn hàng được nhận đúng ngày dự kiến."""
    horizon = int(lead_time + review_interval)
    actual_forecast = test[["date", "item_id", "sales"]].merge(
        predictions[["date", "item_id", "forecast"]],
        on=["date", "item_id"], how="inner"
    )
    actual_forecast = actual_forecast.merge(
        sigma[["item_id", "sigma_e"]], on="item_id", how="left"
    )
    actual_forecast["sigma_e"] = actual_forecast["sigma_e"].fillna(0).clip(lower=0)
    actual_forecast = actual_forecast.sort_values(
        ["item_id", "date"]
    ).reset_index(drop=True)
    if actual_forecast.empty:
        raise ValueError("Không ghép được dữ liệu test với dự báo Random Forest.")

    start_date = test["date"].min()
    history_start = start_date - pd.Timedelta(days=28)
    # Tồn đầu kỳ được giả lập từ doanh số 28 ngày gần nhất của tập train,
    # không dùng doanh số thực tế trong giai đoạn test.
    history = train[
        train["date"].between(history_start, start_date - pd.Timedelta(days=1))
    ]
    avg_history = history.groupby("item_id")["sales"].mean().fillna(0)
    initial_stock = np.ceil(avg_history * stock_cover_days).astype(int)

    rows = []
    for item_id, group in actual_forecast.groupby("item_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        stock = int(initial_stock.get(item_id, 0))
        sigma_value = float(group["sigma_e"].iloc[0])
        pending = {}
        for i, row in group.iterrows():
            received = int(pending.pop(i, 0))
            stock += received
            on_order = int(sum(pending.values()))
            inventory_position = stock + on_order
            order_qty = 0
            safety_stock = z_value * sigma_value * np.sqrt(horizon)
            # Chỉ đặt đơn khi còn đủ dữ liệu dự báo để tính nhu cầu phủ.
            if i + horizon <= len(group):
                demand_forecast = float(
                    group.loc[i:i + horizon - 1, "forecast"].clip(lower=0).sum()
                )
                target_stock = demand_forecast + safety_stock
                order_qty = int(np.ceil(max(0, target_stock - inventory_position)))
                arrival_index = i + lead_time
                if order_qty > 0 and arrival_index < len(group):
                    pending[arrival_index] = pending.get(arrival_index, 0) + order_qty

            demand_actual = max(0, int(round(float(row["sales"]))))
            sold = min(stock, demand_actual)
            lost = demand_actual - sold
            beginning_stock = stock
            stock -= sold
            rows.append({
                "item_id": item_id,
                "date": row["date"],
                "beginning_stock": beginning_stock,
                "received_stock": received,
                "demand": demand_actual,
                "forecast": float(row["forecast"]),
                "sold": sold,
                "lost_sales": lost,
                "ending_stock": stock,
                "safety_stock": safety_stock,
                "inventory_position": inventory_position,
                "order_qty": order_qty,
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("Không tạo được kết quả mô phỏng.")
    total_demand = int(detail["demand"].sum())
    total_sold = int(detail["sold"].sum())
    summary = pd.DataFrame([{
        "total_demand": total_demand,
        "fulfilled_demand": total_sold,
        "lost_sales": int(detail["lost_sales"].sum()),
        "service_level_pct": round(
            total_sold / total_demand * 100, 2
        ) if total_demand else 0,
        "stockout_item_days": int(
            ((detail["ending_stock"] == 0) & (detail["demand"] > 0)).sum()
        ),
        "average_ending_stock": round(float(detail["ending_stock"].mean()), 2),
        "total_order_qty": int(detail["order_qty"].sum()),
        "total_received_qty": int(detail["received_stock"].sum()),
    }])
    return detail, summary
