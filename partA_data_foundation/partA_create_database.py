
import sqlite3
import pandas as pd
from pathlib import Path

# Locate the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"

# Load the three cleaned CSV files
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")

# Keep order timestamps as text in the required format
orders = pd.read_csv(
    CLEAN_DIR / "orders.csv",
    dtype={"order_ts": "string"}
)

# Create the database and save all three tables
with sqlite3.connect(DB_PATH) as connection:
    customers.to_sql("customers", connection, if_exists="replace", index=False)
    products.to_sql("products", connection, if_exists="replace", index=False)
    orders.to_sql("orders", connection, if_exists="replace", index=False)

    # Validate table names and row counts
    expected_counts = {
        "customers": len(customers),
        "products": len(products),
        "orders": len(orders),
    }

    for table, expected in expected_counts.items():
        actual = connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {actual} rows")
        assert actual == expected, f"Row count mismatch in {table}"

    # Confirm order timestamps are stored as text
    timestamp_type = connection.execute(
        "SELECT typeof(order_ts) FROM orders LIMIT 1"
    ).fetchone()[0]

    print("order_ts SQLite type:", timestamp_type)
    assert timestamp_type == "text"

    # Confirm order IDs are unique
    duplicate_ids = connection.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT order_id
            FROM orders
            GROUP BY order_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    assert duplicate_ids == 0

print(f"\nDatabase created successfully: {DB_PATH}")
print("DATABASE VALIDATION PASSED")
