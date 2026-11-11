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


def _normalise_dates(df):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_project_data(project_dir):
    """Đọc dữ liệu train/test, dự báo RF và ước lượng độ lệch sai số."""
    root = Path(project_dir)
    data_dir = root / "data" / "processed"
    output_dir = root / "output" / "ml_models"

    train = _read_csv(data_dir / "train.csv", ["date", "item_id", "sales"])
    test = _read_csv(data_dir / "test.csv", ["date", "item_id", "sales"])
    train, test = _normalise_dates(train), _normalise_dates(test)

    pred_path = output_dir / "random_forest_test.csv"
    if not pred_path.exists():
        pred_path = output_dir / "forecast_detail.csv"
    pred = _read_csv(pred_path)
    if "y_pred" in pred.columns and "forecast" not in pred.columns:
        pred = pred.rename(columns={"y_pred": "forecast"})
    if not {"date", "item_id", "forecast"}.issubset(pred.columns):
        raise ValueError(
            f"{pred_path.name} cần có các cột date, item_id, forecast (hoặc y_pred)."
        )
    pred = _normalise_dates(pred)
    pred["forecast"] = pd.to_numeric(pred["forecast"], errors="coerce")
    pred = pred.dropna(subset=["date", "item_id", "forecast"])
    if "model" in pred.columns:
        rf = pred[pred["model"].astype(str).str.contains("random forest", case=False, na=False)]
        if not rf.empty:
            pred = rf.copy()
    pred = pred.groupby(["date", "item_id"], as_index=False)["forecast"].mean()

    for frame in (train, test):
        frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce").fillna(0).clip(lower=0)

    # Ưu tiên sai số RF trên validation; tuyệt đối không lấy actual test để ước lượng sigma.
    sigma = pd.DataFrame(columns=["item_id", "sigma_e", "sigma_source"])
    detail_path = output_dir / "forecast_detail.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        required = {"date", "item_id", "actual", "forecast"}
        if required.issubset(detail.columns):
            detail = _normalise_dates(detail)
            if "model" in detail.columns:
                detail = detail[detail["model"].astype(str).str.contains("random forest", case=False, na=False)]
            first_test_date = test["date"].min()
            validation = detail[detail["date"] < first_test_date].copy()
            validation["error"] = (
                pd.to_numeric(validation["actual"], errors="coerce")
                - pd.to_numeric(validation["forecast"], errors="coerce")
            )
            sigma = (
                validation.dropna(subset=["error"])
                .groupby("item_id")["error"].std().fillna(0).clip(lower=0)
                .rename("sigma_e").reset_index()
            )
            if not sigma.empty:
                sigma["sigma_source"] = "Sai số dự báo RF trên giai đoạn trước Test"

    if sigma.empty:
        recent_dates = sorted(train["date"].dropna().unique())[-28:]
        recent = train[train["date"].isin(recent_dates)]
        sigma = (
            recent.groupby("item_id")["sales"].std().fillna(0).clip(lower=0)
            .rename("sigma_e").reset_index()
        )
        sigma["sigma_source"] = "Độ lệch chuẩn doanh số 28 ngày train gần nhất (thay thế)"

    return train, test, pred, sigma


def _recent_history(train, end_date, window=28):
    eligible = train[train["date"] <= end_date]
    dates = sorted(eligible["date"].dropna().unique())[-window:]
    return eligible[eligible["date"].isin(dates)]


def build_recommendations(
    train, predictions, sigma, lead_time=6, review_interval=1,
    stock_cover_days=3, z_value=1.65, custom_stock=None, custom_on_order=None
):
    """Đề xuất theo Q=max(0, nhu cầu dự báo + tồn an toàn - vị thế tồn kho)."""
    if lead_time < 1 or review_interval < 1 or stock_cover_days < 0:
        raise ValueError("Lead time và chu kỳ xem xét phải >= 1; tồn kho phủ phải >= 0.")
    horizon = int(lead_time + review_interval)
    train = _normalise_dates(train)
    predictions = _normalise_dates(predictions)
    as_of = predictions["date"].min() - pd.Timedelta(days=1)

    future = predictions[
        (predictions["date"] > as_of)
        & (predictions["date"] <= as_of + pd.Timedelta(days=horizon))
    ].copy()
    if future.empty:
        raise ValueError(
            f"Không có dự báo trong {horizon} ngày từ {as_of.date()}. "
            "Kiểm tra khoảng ngày của random_forest_test.csv."
        )
    day_counts = future.groupby("item_id")["date"].nunique()
    valid_items = day_counts[day_counts == horizon].index
    future = future[future["item_id"].isin(valid_items)]
    if future.empty:
        raise ValueError(f"Không sản phẩm nào có đủ {horizon} ngày dự báo.")

    demand = future.groupby("item_id", as_index=False).agg(
        forecast_demand=("forecast", lambda x: x.clip(lower=0).sum()),
        forecast_days=("date", "nunique")
    )
    recent = _recent_history(train, as_of, 28)
    avg = recent.groupby("item_id", as_index=False)["sales"].mean().rename(
        columns={"sales": "avg_daily_sales"}
    )
    result = demand.merge(avg, on="item_id", how="left")
    result = result.merge(
        sigma[["item_id", "sigma_e", "sigma_source"]].drop_duplicates("item_id"),
        on="item_id", how="left"
    )
    result["avg_daily_sales"] = result["avg_daily_sales"].fillna(0)
    result["sigma_e"] = result["sigma_e"].fillna(0).clip(lower=0)
    result["sigma_source"] = result["sigma_source"].fillna("Không có dữ liệu; tạm dùng 0")

    result["on_hand"] = np.ceil(result["avg_daily_sales"] * stock_cover_days).astype(int)
    if custom_stock is not None and {"item_id", "on_hand"}.issubset(custom_stock.columns):
        stock_map = custom_stock.drop_duplicates("item_id").set_index("item_id")["on_hand"]
        result["on_hand"] = result["item_id"].map(stock_map).fillna(
            result["on_hand"]
        ).clip(lower=0).astype(int)

    result["on_order"] = 0
    if custom_on_order is not None and {"item_id", "on_order"}.issubset(custom_on_order.columns):
        order_map = custom_on_order.drop_duplicates("item_id").set_index("item_id")["on_order"]
        result["on_order"] = result["item_id"].map(order_map).fillna(0).clip(lower=0).astype(int)

    result["safety_stock"] = z_value * result["sigma_e"] * np.sqrt(horizon)
    result["inventory_position"] = result["on_hand"] + result["on_order"]
    result["suggested_qty"] = np.ceil(
        (result["forecast_demand"] + result["safety_stock"] - result["inventory_position"]).clip(lower=0)
    ).astype(int)
    result["recommendation"] = np.where(
        result["suggested_qty"] > 0, "Cần bổ sung", "Chưa cần bổ sung"
    )
    result["planning_date"] = as_of
    result["lead_time_days"] = int(lead_time)
    result["review_interval_days"] = int(review_interval)
    result["coverage_days"] = horizon
    return result.sort_values(
        ["suggested_qty", "forecast_demand"], ascending=False
    ).reset_index(drop=True)


def simulate_inventory(
    train, test, predictions, sigma, lead_time=6,
    review_interval=1, stock_cover_days=3, z_value=1.65
):
    """Mô phỏng chính sách đặt hàng theo dự báo trên giai đoạn Test."""
    if lead_time < 1 or review_interval < 1:
        raise ValueError("Lead time và chu kỳ xem xét phải >= 1.")
    horizon = int(lead_time + review_interval)
    test = _normalise_dates(test)
    predictions = _normalise_dates(predictions)
    train = _normalise_dates(train)

    actual_forecast = test[["date", "item_id", "sales"]].merge(
        predictions[["date", "item_id", "forecast"]],
        on=["date", "item_id"], how="inner"
    )
    actual_forecast = actual_forecast.merge(
        sigma[["item_id", "sigma_e"]].drop_duplicates("item_id"),
        on="item_id", how="left"
    )
    actual_forecast["sigma_e"] = actual_forecast["sigma_e"].fillna(0).clip(lower=0)
    actual_forecast = actual_forecast.sort_values(
        ["item_id", "date"]
    ).reset_index(drop=True)
    if actual_forecast.empty:
        raise ValueError("Không ghép được dữ liệu test với dự báo Random Forest.")

    start_date = actual_forecast["date"].min()
    history = _recent_history(train, start_date - pd.Timedelta(days=1), 28)
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
            safety_stock = float(z_value * sigma_value * np.sqrt(horizon))

            if i + horizon <= len(group):
                demand_forecast = float(
                    group.loc[i:i + horizon - 1, "forecast"].clip(lower=0).sum()
                )
                order_qty = int(np.ceil(max(
                    0, demand_forecast + safety_stock - inventory_position
                )))
                arrival_index = i + int(lead_time)
                if order_qty > 0 and arrival_index < len(group):
                    pending[arrival_index] = pending.get(arrival_index, 0) + order_qty

            demand_actual = max(0, int(round(float(row["sales"]))))
            beginning_stock = stock
            sold = min(stock, demand_actual)
            lost = demand_actual - sold
            stock -= sold
            rows.append({
                "item_id": item_id, "date": row["date"],
                "beginning_stock": beginning_stock, "received_stock": received,
                "demand": demand_actual, "forecast": float(row["forecast"]),
                "sold": sold, "lost_sales": lost, "ending_stock": stock,
                "safety_stock": safety_stock, "inventory_position": inventory_position,
                "order_qty": order_qty, "outstanding_order": int(sum(pending.values()))
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
        "service_level_pct": round(total_sold / total_demand * 100, 2) if total_demand else 0,
        "stockout_item_days": int(((detail["ending_stock"] == 0) & (detail["demand"] > 0)).sum()),
        "average_ending_stock": round(float(detail["ending_stock"].mean()), 2),
        "total_order_qty": int(detail["order_qty"].sum()),
        "total_received_qty": int(detail["received_stock"].sum()),
        "initial_stock": int(detail.groupby("item_id")["beginning_stock"].first().sum()),
        "simulation_start": detail["date"].min(),
        "simulation_end": detail["date"].max()
    }])
    return detail, summary
