
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

# Load cleaned datasets
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Keep completed orders only
completed_orders = orders[
    orders["status"] == "completed"
].copy()

print("\n========== 1. PIVOT TABLE ==========")

# Join orders with customers and products
analysis_data = completed_orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

analysis_data = analysis_data.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Create pivot table
pivot = pd.pivot_table(
    analysis_data,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True
)

print(pivot)

print(
    "\nPivot table grand total:",
    round(pivot.loc["All", "All"], 2)
)

print(
    "Completed order revenue:",
    round(completed_orders["order_amount"].sum(), 2)
)


print("\n========== 2. CITY-WISE ANALYSIS ==========")

city_summary = analysis_data.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_amount=("order_amount", "mean"),
    completed_order_count=("order_id", "count")
).reset_index()

print(city_summary.round(2).to_string(index=False))


print("\n========== 3. ORDER AMOUNT DISTRIBUTION ==========")

amount = completed_orders["order_amount"]

mean_value = amount.mean()
median_value = amount.median()
skewness = amount.skew()

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)

iqr = q3 - q1

lower_fence = q1 - (1.5 * iqr)
upper_fence = q3 + (1.5 * iqr)

upper_outliers = (amount > upper_fence).sum()
lower_outliers = (amount < lower_fence).sum()

print("Mean:", round(mean_value, 2))
print("Median:", round(median_value, 2))
print("Skewness:", round(skewness, 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Upper outliers:", upper_outliers)
print("Lower outliers:", lower_outliers)


print("\n========== 4. CORRELATION ANALYSIS ==========")

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_matrix = completed_orders[
    correlation_columns
].corr()

print(correlation_matrix.round(3))


print("\n========== PANDAS ANALYSIS COMPLETED ==========")
