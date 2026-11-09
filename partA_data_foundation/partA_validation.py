
import pandas as pd
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"

# Load cleaned CSV files
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Checkpoint 1: Customer row count
assert len(customers) == 1200

# Checkpoint 2: Product row count
assert len(products) == 40

# Checkpoint 3: Order row count
assert len(orders) == 12309

# Checkpoint 4: Unique customer IDs
assert customers["customer_id"].is_unique

# Checkpoint 5: Unique product IDs
assert products["product_id"].is_unique

# Checkpoint 6: Unique order IDs
assert orders["order_id"].is_unique

# Checkpoint 7: Exactly six cities
assert customers["city"].nunique() == 6

# Checkpoint 8: No missing customer ages
assert customers["age"].isna().sum() == 0

# Checkpoint 9: No pending orders
assert "pending" not in orders["status"].values

# Checkpoint 10: No missing order amounts
assert orders["order_amount"].isna().sum() == 0

# Checkpoint 11: No missing delivery times for completed orders
completed = orders["status"] == "completed"
assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0

# Checkpoint 12: Cancelled orders have missing delivery times
cancelled = orders["status"] == "cancelled"
assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199

# Checkpoint 13: Completed order revenue
revenue = round(orders.loc[completed, "order_amount"].sum(), 2)
assert revenue == 2998668.80

# Checkpoint 14: SQLite database tables
with sqlite3.connect(DB_PATH) as connection:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        connection
    )["name"].tolist()

    assert set(tables) >= {"customers", "products", "orders"}

    # Check database row counts
    for table, expected in {
        "customers": 1200,
        "products": 40,
        "orders": 12309
    }.items():
        actual = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}",
            connection
        )["total"].iloc[0]

        assert actual == expected

print("ALL CHECKPOINTS PASSED")
