"""
Mo phong ton kho cho de tai M5 Forecasting
Pham vi: CA_1 - FOODS, 1.437 san pham
Ngay lap ke hoach: 2016-03-27
Thoi gian mo phong: 2016-03-28 den 2016-04-03 (7 ngay)

Chay tai thu muc goc project:
    python .\script\simulate_inventory.py

Dau vao:
    data/raw/sales_train_validation.csv
    data/raw/calendar.csv
    output/ml_models/forecast_detail.csv

Dau ra:
    output/inventory_simulation/
"""

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CAU HINH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
ML_OUTPUT_DIR = PROJECT_DIR / "output" / "ml_models"
OUTPUT_DIR = PROJECT_DIR / "output" / "inventory_simulation"

SALES_FILE = RAW_DIR / "sales_train_validation.csv"
CALENDAR_FILE = RAW_DIR / "calendar.csv"
FORECAST_FILE = ML_OUTPUT_DIR / "forecast_detail.csv"

STORE_ID = "CA_1"
CATEGORY_ID = "FOODS"
MAIN_MODEL = "Random Forest"

PLANNING_DATE = pd.Timestamp("2016-03-27")
SIMULATION_START = pd.Timestamp("2016-03-28")
SIMULATION_DAYS = 7

STOCK_COVER_DAYS = 3
LEAD_TIME = 6
REVIEW_PERIOD = 1
SERVICE_FACTOR_Z = 1.65
INITIAL_ON_ORDER = 0

# Theo bao cao: T = L + R = 6 + 1 = 7 ngay
COVERAGE_PERIOD = LEAD_TIME + REVIEW_PERIOD

# Ngay nhan hang: don hang dat tai ngay lap ke hoach,
# giao sau 6 ngay, hang duoc tinh vao ton kho dau ngay thu 7.
RECEIPT_DAY_NUMBER = LEAD_TIME + 1


# ============================================================
# 2. HAM HO TRO
# ============================================================

def stop_with_error(message: str) -> None:
    print("\n[LOI]", message)
    sys.exit(1)


def check_input_files() -> None:
    required_files = [SALES_FILE, CALENDAR_FILE, FORECAST_FILE]
    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        stop_with_error(
            "Khong tim thay tep dau vao:\n- " + "\n- ".join(missing)
        )


def get_day_columns(calendar: pd.DataFrame, start_date, end_date):
    """Tra ve cac cot d_x tuong ung voi khoang ngay can lay."""
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])

    selected = cal[
        (cal["date"] >= pd.Timestamp(start_date))
        & (cal["date"] <= pd.Timestamp(end_date))
    ]

    if selected.empty:
        stop_with_error(
            f"Calendar khong co ngay trong khoang {start_date} - {end_date}."
        )

    return selected[["date", "d"]].sort_values("date")


def load_recent_sales_and_actuals():
    """
    Doc doanh so 28 ngay truoc/ngay lap ke hoach.
    Chi doc cac cot d_x can thiet, sau do loc CA_1 va FOODS.
    """
    print("1/7 - Dang doc calendar va du lieu ban hang...")

    calendar = pd.read_csv(CALENDAR_FILE, usecols=["date", "d"])
    calendar["date"] = pd.to_datetime(calendar["date"])

    recent_start = PLANNING_DATE - pd.Timedelta(days=27)
    recent_days = get_day_columns(calendar, recent_start, PLANNING_DATE)
    simulation_days = get_day_columns(
        calendar,
        SIMULATION_START,
        SIMULATION_START + pd.Timedelta(days=SIMULATION_DAYS - 1),
    )

    if len(recent_days) != 28:
        stop_with_error(
            f"Can 28 ngay doanh so gan nhat, nhung tim thay {len(recent_days)} ngay."
        )

    if len(simulation_days) != SIMULATION_DAYS:
        stop_with_error(
            f"Can {SIMULATION_DAYS} ngay mo phong, nhung calendar chi co "
            f"{len(simulation_days)} ngay."
        )

    recent_columns = recent_days["d"].tolist()
    simulation_columns = simulation_days["d"].tolist()
    required_day_columns = list(dict.fromkeys(recent_columns + simulation_columns))

    sales_header = pd.read_csv(SALES_FILE, nrows=0).columns.tolist()
    id_columns = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    missing_columns = [
        col for col in id_columns + required_day_columns
        if col not in sales_header
    ]
    if missing_columns:
        stop_with_error(
            "Tep sales_train_validation.csv thieu cac cot: "
            + ", ".join(missing_columns[:15])
        )

    sales = pd.read_csv(
        SALES_FILE,
        usecols=id_columns + required_day_columns,
    )

    sales = sales[
        (sales["store_id"] == STORE_ID)
        & (sales["cat_id"] == CATEGORY_ID)
    ].copy()

    if sales.empty:
        stop_with_error(
            f"Khong tim thay san pham {CATEGORY_ID} tai cua hang {STORE_ID}."
        )

    sales = sales.set_index("item_id")

    # Trung binh doanh so moi ngay trong 28 ngay gan nhat.
    recent_mean = sales[recent_columns].mean(axis=1)

    # Doanh so thuc te tung ngay trong 7 ngay mo phong.
    actual_long = (
        sales[simulation_columns]
        .rename_axis("item_id")
        .reset_index()
        .melt(
            id_vars="item_id",
            var_name="d",
            value_name="actual_from_raw",
        )
    )

    day_to_date = dict(
        zip(simulation_days["d"], simulation_days["date"])
    )
    actual_long["date"] = actual_long["d"].map(day_to_date)
    actual_long["actual_from_raw"] = (
        pd.to_numeric(actual_long["actual_from_raw"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )

    recent_mean.name = "average_daily_sales_28d"

    print(f"   So san pham sau khi loc: {len(sales):,}")
    print(
        f"   Doanh so gan nhat: {recent_start.date()} "
        f"den {PLANNING_DATE.date()}"
    )
    print(
        f"   Ngay mo phong: {SIMULATION_START.date()} den "
        f"{(SIMULATION_START + pd.Timedelta(days=6)).date()}"
    )

    return recent_mean, actual_long, sales.index.tolist()


def load_forecasts_and_product_rmse(product_ids):
    """
    Lay du bao Random Forest cho 7 ngay mo phong.
    RMSE tung san pham duoc tinh tren 28 ngay Validation,
    khong su dung sai so cua tap Test de tinh ton kho an toan.
    """
    print("2/7 - Dang doc du bao Random Forest va tinh RMSE tung san pham...")

    forecast = pd.read_csv(FORECAST_FILE)
    required = {"date", "item_id", "actual", "forecast", "model"}
    missing = required - set(forecast.columns)

    if missing:
        stop_with_error(
            "forecast_detail.csv thieu cot: " + ", ".join(sorted(missing))
        )

    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast = forecast[forecast["model"] == MAIN_MODEL].copy()

    simulation_end = SIMULATION_START + pd.Timedelta(days=SIMULATION_DAYS - 1)
    test_forecast = forecast[
        (forecast["date"] >= SIMULATION_START)
        & (forecast["date"] <= simulation_end)
        & (forecast["item_id"].isin(product_ids))
    ].copy()

    # Moi san pham phai co du bao cho du 7 ngay.
    counts = test_forecast.groupby("item_id")["date"].nunique()
    incomplete = [item for item in product_ids if counts.get(item, 0) != SIMULATION_DAYS]

    if incomplete:
        stop_with_error(
            f"Co {len(incomplete)} san pham khong co du bao du {SIMULATION_DAYS} ngay. "
            "Hay kiem tra forecast_detail.csv."
        )

    test_forecast["forecast"] = (
        pd.to_numeric(test_forecast["forecast"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    test_forecast["actual"] = (
        pd.to_numeric(test_forecast["actual"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )

    # RMSE theo tung san pham tren Validation 28 ngay.
    validation_start = PLANNING_DATE - pd.Timedelta(days=27)
    validation = forecast[
        (forecast["date"] >= validation_start)
        & (forecast["date"] <= PLANNING_DATE)
        & (forecast["item_id"].isin(product_ids))
    ].copy()

    validation["actual"] = pd.to_numeric(validation["actual"], errors="coerce")
    validation["forecast"] = pd.to_numeric(validation["forecast"], errors="coerce")
    validation = validation.dropna(subset=["actual", "forecast"])

    validation["squared_error"] = (
        validation["actual"] - validation["forecast"]
    ) ** 2

    rmse_by_item = (
        validation.groupby("item_id")["squared_error"]
        .mean()
        .pow(0.5)
        .rename("forecast_rmse_validation")
    )

    if rmse_by_item.empty:
        stop_with_error(
            "Khong tinh duoc RMSE tren Validation. "
            "Can co du lieu Random Forest tu 2016-02-29 den 2016-03-27."
        )

    fallback_rmse = float(rmse_by_item.median())
    rmse_by_item = rmse_by_item.reindex(product_ids).fillna(fallback_rmse)

    # Lay actual tu forecast_detail va doi chieu voi raw sales.
    test_forecast = test_forecast[
        ["date", "item_id", "actual", "forecast"]
    ].copy()

    print(f"   So dong du bao 7 ngay: {len(test_forecast):,}")
    print(f"   RMSE trung vi theo san pham: {fallback_rmse:.4f}")

    return test_forecast, rmse_by_item


def build_replenishment_recommendations(
    recent_mean,
    rmse_by_item,
    product_ids,
    test_forecast,
):
    """Tinh OnHand, safety stock va luong hang can bo sung theo muc 4.3."""
    print("3/7 - Dang tinh luong hang de xuat bo sung...")

    recommendations = pd.DataFrame(index=pd.Index(product_ids, name="item_id"))
    recommendations = recommendations.join(recent_mean)
    recommendations = recommendations.join(rmse_by_item)

    recommendations["average_daily_sales_28d"] = (
        recommendations["average_daily_sales_28d"].fillna(0).clip(lower=0)
    )
    recommendations["forecast_rmse_validation"] = (
        recommendations["forecast_rmse_validation"].fillna(0).clip(lower=0)
    )

    # OnHand_i = ceil(Sales_i,28 * StockCoverDays)
    recommendations["on_hand_initial"] = np.ceil(
        recommendations["average_daily_sales_28d"] * STOCK_COVER_DAYS
    ).astype(int)

    recommendations["on_order"] = INITIAL_ON_ORDER

    # D_i = tong du bao trong 7 ngay.
    forecast_demand = (
        test_forecast.groupby("item_id")["forecast"]
        .sum()
        .rename("forecast_demand_7d")
    )
    recommendations = recommendations.join(forecast_demand)
    recommendations["forecast_demand_7d"] = (
        recommendations["forecast_demand_7d"].fillna(0).clip(lower=0)
    )

    # S_i = z * e_i * sqrt(T)
    # e_i duoc uoc tinh bang RMSE tren Validation.
    recommendations["safety_stock"] = (
        SERVICE_FACTOR_Z
        * recommendations["forecast_rmse_validation"]
        * math.sqrt(COVERAGE_PERIOD)
    )

    # I_i = OnHand_i + OnOrder_i
    recommendations["inventory_position"] = (
        recommendations["on_hand_initial"] + recommendations["on_order"]
    )

    # Q_i = max(0, D_i + S_i - I_i); sau do lam tron len.
    recommendations["order_quantity_unrounded"] = (
        recommendations["forecast_demand_7d"]
        + recommendations["safety_stock"]
        - recommendations["inventory_position"]
    ).clip(lower=0)

    recommendations["recommended_order_quantity"] = np.ceil(
        recommendations["order_quantity_unrounded"]
    ).astype(int)

    recommendations = recommendations.reset_index()

    return recommendations


def simulate_one_scenario(
    scenario_name,
    recommendations,
    test_forecast,
    receive_replenishment,
):
    """
    Mo phong tung san pham theo tung ngay.
    Hang bo sung (neu co) duoc nhap dau ngay thu 7,
    truoc khi dap ung nhu cau cua ngay thu 7.
    """
    rows = []

    rec = recommendations.set_index("item_id")
    forecast_by_key = test_forecast.set_index(["item_id", "date"])

    simulation_dates = [
        SIMULATION_START + pd.Timedelta(days=i)
        for i in range(SIMULATION_DAYS)
    ]

    for item_id in recommendations["item_id"]:
        inventory = int(rec.loc[item_id, "on_hand_initial"])
        order_qty = int(rec.loc[item_id, "recommended_order_quantity"])

        for day_index, date in enumerate(simulation_dates, start=1):
            actual_demand = float(
                forecast_by_key.loc[(item_id, date), "actual"]
            )
            forecast_demand = float(
                forecast_by_key.loc[(item_id, date), "forecast"]
            )

            opening_inventory = inventory
            receipt = 0

            if (
                receive_replenishment
                and day_index == RECEIPT_DAY_NUMBER
            ):
                receipt = order_qty

            available_inventory = opening_inventory + receipt
            fulfilled = min(available_inventory, actual_demand)
            unmet = max(0.0, actual_demand - fulfilled)
            ending_inventory = max(0.0, available_inventory - fulfilled)

            rows.append(
                {
                    "scenario": scenario_name,
                    "item_id": item_id,
                    "date": date,
                    "day_number": day_index,
                    "forecast_demand": forecast_demand,
                    "actual_demand": actual_demand,
                    "opening_inventory": opening_inventory,
                    "replenishment_received": receipt,
                    "available_inventory": available_inventory,
                    "demand_fulfilled": fulfilled,
                    "unmet_demand": unmet,
                    "ending_inventory": ending_inventory,
                }
            )

            inventory = ending_inventory

    return pd.DataFrame(rows)


def calculate_scenario_metrics(simulation_detail):
    """Tong hop cac chi tieu danh gia cho tung kich ban."""
    results = []

    for scenario, group in simulation_detail.groupby("scenario"):
        total_demand = float(group["actual_demand"].sum())
        total_shortage = float(group["unmet_demand"].sum())
        total_fulfilled = float(group["demand_fulfilled"].sum())

        # Ton kho cuoi ky: lay ton kho cuoi ngay cuoi cung cua moi san pham.
        final_inventory = (
            group.sort_values("date")
            .groupby("item_id")
            .tail(1)["ending_inventory"]
            .sum()
        )

        fulfillment_rate = (
            (1 - total_shortage / total_demand) * 100
            if total_demand > 0
            else 100.0
        )

        stockout_item_days = int((group["unmet_demand"] > 0).sum())
        products_with_stockout = int(
            group.loc[group["unmet_demand"] > 0, "item_id"].nunique()
        )

        results.append(
            {
                "scenario": scenario,
                "total_actual_demand": total_demand,
                "total_fulfilled_demand": total_fulfilled,
                "total_shortage": total_shortage,
                "ending_inventory_total": float(final_inventory),
                "fulfillment_rate_percent": fulfillment_rate,
                "stockout_item_days": stockout_item_days,
                "products_with_stockout": products_with_stockout,
            }
        )

    return pd.DataFrame(results)


def create_charts(simulation_detail, metrics):
    """Tao bieu do tong hop de dua vao bao cao."""
    print("6/7 - Dang tao bieu do...")

    daily = (
        simulation_detail.groupby(["scenario", "date"])
        .agg(
            ending_inventory=("ending_inventory", "sum"),
            shortage=("unmet_demand", "sum"),
            actual_demand=("actual_demand", "sum"),
        )
        .reset_index()
    )

    # Bieu do ton kho cuoi ngay.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for scenario, group in daily.groupby("scenario"):
        ax.plot(
            group["date"],
            group["ending_inventory"],
            marker="o",
            linewidth=2,
            label=scenario,
        )
    ax.set_title("So sanh tong ton kho cuoi ngay theo kich ban")
    ax.set_xlabel("Ngay mo phong")
    ax.set_ylabel("Tong ton kho (don vi)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "inventory_by_day.png", dpi=180)
    plt.close(fig)

    # Bieu do luong hang thieu moi ngay.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for scenario, group in daily.groupby("scenario"):
        ax.plot(
            group["date"],
            group["shortage"],
            marker="o",
            linewidth=2,
            label=scenario,
        )
    ax.set_title("So sanh luong hang thieu theo ngay")
    ax.set_xlabel("Ngay mo phong")
    ax.set_ylabel("Luong hang thieu (don vi)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shortage_by_day.png", dpi=180)
    plt.close(fig)

    # Bieu do so sanh cac chi tieu tong hop.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(metrics))
    labels = metrics["scenario"].tolist()

    axes[0].bar(labels, metrics["total_shortage"])
    axes[0].set_title("Tong luong hang thieu")
    axes[0].set_ylabel("Don vi san pham")
    axes[0].tick_params(axis="x", rotation=12)

    axes[1].bar(labels, metrics["fulfillment_rate_percent"])
    axes[1].set_title("Ty le dap ung nhu cau")
    axes[1].set_ylabel("Phan tram (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=12)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scenario_comparison.png", dpi=180)
    plt.close(fig)

    daily.to_csv(
        OUTPUT_DIR / "daily_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 3. CHUONG TRINH CHINH
# ============================================================

def main():
    print("=" * 68)
    print("MO PHONG TON KHO - M5 FORECASTING")
    print("=" * 68)

    check_input_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recent_mean, actual_raw, product_ids = load_recent_sales_and_actuals()

    test_forecast, rmse_by_item = load_forecasts_and_product_rmse(product_ids)

    # Dung doanh so thuc te doc tu sales_train_validation.csv de danh gia.
    # Cot actual trong forecast_detail duoc giu lai de doi chieu.
    actual_check = actual_raw.rename(
        columns={"actual_from_raw": "actual_from_sales_file"}
    )
    test_forecast = test_forecast.merge(
        actual_check[["item_id", "date", "actual_from_sales_file"]],
        on=["item_id", "date"],
        how="left",
        validate="one_to_one",
    )

    if test_forecast["actual_from_sales_file"].isna().any():
        stop_with_error(
            "Thieu doanh so thuc te trong sales_train_validation.csv "
            "cho mot so san pham/ngay mo phong."
        )

    difference = (
        test_forecast["actual"] - test_forecast["actual_from_sales_file"]
    ).abs().max()
    if difference > 1e-8:
        print(
            "   CANH BAO: actual trong forecast_detail khac sales raw. "
            "Chuong trinh se uu tien doanh so thuc te tu sales raw."
        )

    test_forecast["actual"] = test_forecast["actual_from_sales_file"]
    test_forecast.drop(columns=["actual_from_sales_file"], inplace=True)

    recommendations = build_replenishment_recommendations(
        recent_mean,
        rmse_by_item,
        product_ids,
        test_forecast,
    )

    print("4/7 - Dang mo phong hai kich ban ton kho...")

    no_replenishment = simulate_one_scenario(
        "Khong nhap them hang",
        recommendations,
        test_forecast,
        receive_replenishment=False,
    )

    with_replenishment = simulate_one_scenario(
        "Nhap theo de xuat",
        recommendations,
        test_forecast,
        receive_replenishment=True,
    )

    simulation_detail = pd.concat(
        [no_replenishment, with_replenishment],
        ignore_index=True,
    )

    print("5/7 - Dang tong hop chi tieu danh gia...")

    metrics = calculate_scenario_metrics(simulation_detail)

    # Luu cac bang ket qua.
    recommendations.to_csv(
        OUTPUT_DIR / "replenishment_recommendations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    simulation_detail.to_csv(
        OUTPUT_DIR / "simulation_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        OUTPUT_DIR / "scenario_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_charts(simulation_detail, metrics)

    # Bao cao tong ket.
    total_order = int(recommendations["recommended_order_quantity"].sum())
    total_initial_stock = int(recommendations["on_hand_initial"].sum())

    print("\n7/7 - HOAN TAT")
    print("=" * 68)
    print(f"Cua hang: {STORE_ID}")
    print(f"Nhom hang: {CATEGORY_ID}")
    print(f"So san pham: {len(recommendations):,}")
    print(f"Ton kho ban dau (gia dinh): {total_initial_stock:,.0f}")
    print(f"Tong luong hang de xuat nhap: {total_order:,.0f}")
    print(
        f"Ngay nhan hang trong mo phong: ngay thu {RECEIPT_DAY_NUMBER} "
        f"({(SIMULATION_START + pd.Timedelta(days=RECEIPT_DAY_NUMBER - 1)).date()})"
    )

    print("\nKET QUA SO SANH HAI KICH BAN")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\nTep ket qua da duoc luu tai:")
    print(OUTPUT_DIR)
    print("\nCac tep chinh:")
    print("- replenishment_recommendations.csv")
    print("- simulation_detail.csv")
    print("- scenario_metrics.csv")
    print("- daily_scenario_summary.csv")
    print("- inventory_by_day.png")
    print("- shortage_by_day.png")
    print("- scenario_comparison.png")

    print("\nLuu y:")
    print(
        "- Ton kho ban dau, lead time, chu ky xem xet va hang dang ve "
        "la cac gia dinh mo phong, khong phai ton kho thuc te cua M5."
    )
    print(
        "- Nhu cau thuc te trong 7 ngay duoc dung de danh gia hai kich ban; "
        "du bao Random Forest duoc dung de tinh luong hang de xuat."
    )
    print(
        "- Hang bo sung duoc cong vao ton kho dau ngay thu 7, "
        "sau 6 ngay cho giao hang."
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
