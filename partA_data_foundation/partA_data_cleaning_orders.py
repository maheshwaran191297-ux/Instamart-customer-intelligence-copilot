
import pandas as pd
from pathlib import Path

# 1. Locate the project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

# 2. Load raw orders and products
orders = pd.read_csv(RAW_DIR / "orders.csv")
products = pd.read_csv(RAW_DIR / "products.csv")

# 3. Convert order timestamps into datetime format
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 4. Remove exact duplicate rows
orders = orders.drop_duplicates()

# 5. Keep only the latest record for each order ID
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset="order_id", keep="last")

# 6. Convert order amounts to numeric values
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# 7. Convert delivery times to numeric values
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# 8. Treat negative delivery times as missing
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")

# 9. Add product price and category to each order
product_info = products[["product_id", "price_inr", "category"]]

orders = orders.merge(
    product_info,
    on="product_id",
    how="left",
    validate="many_to_one"
)

# 10. Calculate missing order amounts using the product price
missing_amount = orders["order_amount"].isna()

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders.loc[missing_amount, "order_amount"] = calculated_amount[missing_amount]

# 11. Fill missing delivery times for completed orders only
completed = orders["status"].eq("completed")

category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

completed_delivery = orders.loc[completed, "delivery_minutes"]
orders.loc[completed, "delivery_minutes"] = (
    completed_delivery.fillna(category_medians)
)

# 12. Remove the extra product columns
orders = orders.drop(columns=["price_inr", "category"])

# 13. Create the output folder and save the cleaned data
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

# 14. Display the cleaning results
print("Cleaned orders shape:", orders.shape)
print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nMissing order amounts:", orders["order_amount"].isna().sum())
print(
    "Missing delivery times among completed orders:",
    orders.loc[completed, "delivery_minutes"].isna().sum()
)
print(
    "Missing delivery times among cancelled orders:",
    orders.loc[~completed, "delivery_minutes"].isna().sum()
)

print(
    "\nCompleted order revenue (INR):",
    round(orders.loc[completed, "order_amount"].sum(), 2)
)

# 15. Validate the expected results
assert len(orders) == 12309
assert orders["order_id"].is_unique
assert "pending" not in orders["status"].values
assert orders["status"].value_counts().to_dict() == {
    "completed": 11110,
    "cancelled": 1199
}
assert orders["order_amount"].isna().sum() == 0
assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0
assert orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum() == 1199
assert round(orders.loc[completed, "order_amount"].sum(), 2) == 2998668.80

print("\nORDER CLEANING CHECKS PASSED")
