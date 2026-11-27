import pandas as pd

df = pd.read_csv("data/processed/cleaned_retail.csv", parse_dates=["InvoiceDate"])
df = df.sort_values("InvoiceDate").reset_index(drop=True)       

df["hour"] = df["InvoiceDate"].dt.hour
df["day_of_week"] = df["InvoiceDate"].dt.dayofweek
df["month"] = df["InvoiceDate"].dt.month

df["stock_tx_count"] = df.groupby("StockCode")["StockCode"].transform("count")
df["stock_avg_price"] = df.groupby("StockCode")["Price"].transform("mean")

# 1. Classification
clf_df = df[["hour", "day_of_week", "month", "stock_tx_count", "stock_avg_price", "Is Cancelled"]]
clf_df.to_csv("data/processed/features_classification.csv", index=False)

# 2. Regression
sales_df = df[~df["Is Cancelled"]].copy()
sales_df["line_revenue"] = sales_df["Quantity"] * sales_df["Price"]

reg_df = sales_df[["hour", "day_of_week", "month", "stock_tx_count", "stock_avg_price", "line_revenue"]]
reg_df.to_csv("data/processed/features_regression.csv", index=False)

# 3. Time series (daily revenue)
sales_df["date"] = sales_df["InvoiceDate"].dt.floor("D")
ts_df = sales_df.groupby("date", as_index=False)["line_revenue"].sum()
ts_df = ts_df.rename(columns={"line_revenue": "daily_revenue"})

ts_df["day_of_week"] = ts_df["date"].dt.dayofweek
ts_df["month"] = ts_df["date"].dt.month
ts_df["lag_1d"] = ts_df["daily_revenue"].shift(1)
ts_df["lag_7d"] = ts_df["daily_revenue"].shift(7)
ts_df = ts_df.dropna().reset_index(drop=True)
ts_df.to_csv("data/processed/features_timeseries.csv", index=False)

print("Done")