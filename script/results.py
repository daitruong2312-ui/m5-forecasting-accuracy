
import pandas as pd
from pathlib import Path

folder = Path("data_processed")

# Hiển thị bảng kết quả sau tiền xử lý
df = pd.read_csv(folder / "model_data.csv")

print("\n" + "=" * 85)
print("KET QUA SAU TIEN XU LY DU LIEU M5")
print("=" * 85)

summary = pd.DataFrame({
    "Chi tieu": [
        "Cua hang",
        "Nhom san pham",
        "So san pham",
        "Tong so dong",
        "Tong so cot",
        "Ngay bat dau",
        "Ngay ket thuc",
        "So gia tri sales bi thieu"
    ],
    "Gia tri": [
        "CA_1",
        "FOODS",
        df["item_id"].nunique(),
        len(df),
        len(df.columns),
        df["date"].min(),
        df["date"].max(),
        df["sales"].isna().sum()
    ]
})

print(summary.to_string(index=False))

# Hiển thị một phần dữ liệu đã xử lý
columns = [
    "date", "item_id", "sales",
    "sell_price", "is_weekend", "is_event",
    "sales_lag_7", "sales_roll_mean_7"
]

print("\n" + "=" * 100)
print("DU LIEU MAU SAU TIEN XU LY")
print("=" * 100)

print(df[columns].head(8).to_string(index=False))

# Hiển thị kết quả chia train/test
print("\n" + "=" * 85)
print("KET QUA CHIA TAP DU LIEU")
print("=" * 85)

rows = []

for filename, label in [
    ("train.csv", "Train"),
    ("test.csv", "Test")
]:
    data = pd.read_csv(
        folder / filename,
        usecols=["date", "item_id"]
    )

    rows.append({
        "Tap du lieu": label,
        "So dong": len(data),
        "So san pham": data["item_id"].nunique(),
        "Ngay bat dau": data["date"].min(),
        "Ngay ket thuc": data["date"].max()
    })

print(pd.DataFrame(rows).to_string(index=False))
