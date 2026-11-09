
import pandas as pd
from pathlib import Path

# 1. Set project paths
ROOT = Path(__file__).resolve().parents[1]

orders_path = ROOT / "data" / "clean" / "orders.csv"
output_path = ROOT / "outputs" / "rfm_table.csv"

# 2. Load cleaned orders
orders = pd.read_csv(orders_path)

# 3. Convert order timestamp into datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 4. Keep only completed orders
completed_orders = orders[
    orders["status"].str.lower() == "completed"
].copy()

# 5. Define snapshot date
snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

# 6. Group orders by customer and calculate RFM
rfm = completed_orders.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# 7. Calculate recency in whole days
rfm["recency"] = (
    snapshot_date - rfm["last_order_ts"]
).dt.days

# 8. Keep required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# 9. Create outputs folder if it does not exist
output_path.parent.mkdir(parents=True, exist_ok=True)

# 10. Save RFM table
rfm.to_csv(output_path, index=False)

# 11. Display results
print("RFM table shape:", rfm.shape)

print("\nRFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nRFM table preview:")
print(rfm.head())

print("\nRFM table saved successfully!")
