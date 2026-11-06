
import pandas as pd
from pathlib import Path

folder = Path("data_processed")

for filename in ["model_data.csv", "train.csv", "test.csv"]:
    path = folder / filename

    print("\n" + "=" * 50)
    print("FILE:", filename)

    if not path.exists():
        print("Khong tim thay file:", path)
        continue

    df = pd.read_csv(path)

    print("So dong:", len(df))
    print("So cot:", len(df.columns))

    if "date" in df.columns:
        print("Tu ngay:", df["date"].min())
        print("Den ngay:", df["date"].max())

    if "item_id" in df.columns:
        print("So san pham:", df["item_id"].nunique())

    if "sales" in df.columns:
        print("So luong ban trung binh:", round(df["sales"].mean(), 3))
        print("So gia tri sales bi thieu:", df["sales"].isna().sum())

    print("Dung luong doc du lieu (MB):",
          round(df.memory_usage(deep=True).sum() / 1024**2, 2))
