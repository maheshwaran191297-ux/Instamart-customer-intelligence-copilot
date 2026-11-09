
import pandas as pd
from pathlib import Path

# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define input and output locations
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

# Load raw product data
products = pd.read_csv(RAW_DIR / "products.csv")

# Remove duplicate rows
products = products.drop_duplicates()

# Clean text columns
for column in ["product_id", "product_name", "category"]:
    products[column] = products[column].astype("string").str.strip()

# Convert price to numeric
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# Validate required values
assert products["product_id"].notna().all()
assert products["product_id"].is_unique
assert products["price_inr"].notna().all()
assert (products["price_inr"] > 0).all()

# Save the cleaned products
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
products.to_csv(CLEAN_DIR / "products.csv", index=False)

print("Cleaned products shape:", products.shape)
print("\nProduct categories:")
print(products["category"].value_counts())
print("\nPRODUCT CLEANING CHECKS PASSED")
