from pathlib import Path
import pandas as pd

raw_file = Path("data/raw/online_retail.csv")
out_file = Path("data/processed/cleaned_retail.csv")

df = pd.read_csv(raw_file)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Invoice"] = df["Invoice"].astype(str).str.strip()
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["Customer ID"] = (df["Customer ID"].astype(str).str.strip().str.replace(".0", "", regex=False))
df.loc[df["Customer ID"].isin(["", "nan", "None"]), "Customer ID"] = None

df["Is Cancelled"] = df["Invoice"].str.startswith("C") | (df["Quantity"] < 0)

df = df.dropna(subset=["Invoice", "StockCode"])
df = df.drop_duplicates()
df = df[(df["Price"] > 0) & (df["Quantity"] != 0)].reset_index(drop=True)

df.to_csv(out_file, index=False)

print(f"Saved {len(df)} rows to {out_file}")