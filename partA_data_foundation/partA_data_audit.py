#import required libraries
import pandas as pd
from pathlib import Path

# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Load the raw CSV files
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# Store the tables so we can inspect them using a loop
tables = {
    "customers": customers,
    "products": products,
    "orders": orders,
}

# Display the shape, data types, and missing values of each table
for name, df in tables.items():
    print(f"\n{'=' * 50}")
    print(f"TABLE: {name.upper()}")
    print("=" * 50)

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values by column:")
    missing_report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (df.isna().mean() * 100).round(2),
    })
    print(missing_report)

# Audit the raw orders
numeric_amount = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery_minutes = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

order_audit = {
    "Raw order rows": len(orders),
    "Exact duplicate rows": int(orders.duplicated().sum()),
    "Pending status rows": int(
        orders["status"].astype("string").str.strip().str.lower().eq("pending").sum()
    ),
    "Distinct order IDs": int(orders["order_id"].nunique()),
    "Unconvertible order amounts": int(numeric_amount.isna().sum()),
    "Negative delivery minutes": int((delivery_minutes < 0).sum()),
    "Blank delivery minutes": int(
        orders["delivery_minutes"].isna().sum()
        + orders["delivery_minutes"].astype("string").str.strip().eq("").sum()
    ),
}

# Audit customer data
age_numeric = pd.to_numeric(customers["age"], errors="coerce")

customer_audit = {
    "Blank ages": int(
        customers["age"].isna().sum()
        + customers["age"].astype("string").str.strip().eq("").sum()
    ),
    "Ages outside 10–100": int(
        ((age_numeric < 10) | (age_numeric > 100)).sum()
    ),
    "Distinct raw city spellings": int(
        customers.loc[
            customers["city"].notna()
            & customers["city"].astype("string").str.strip().ne(""),
            "city",
        ].nunique()
    ),
    "Distinct non-blank raw gender spellings": int(
        customers.loc[
            customers["gender"].notna()
            & customers["gender"].astype("string").str.strip().ne(""),
            "gender",
        ].nunique()
    ),
    "Blank genders": int(
        customers["gender"].isna().sum()
        + customers["gender"].astype("string").str.strip().eq("").sum()
    ),
}

print("\n" + "=" * 50)
print("RAW ORDERS AUDIT")
print("=" * 50)
for label, value in order_audit.items():
    print(f"{label}: {value}")

print("\n" + "=" * 50)
print("RAW CUSTOMERS AUDIT")
print("=" * 50)
for label, value in customer_audit.items():
    print(f"{label}: {value}")
