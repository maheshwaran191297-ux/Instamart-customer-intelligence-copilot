#importing required libraries
import pandas as pd
from pathlib import Path

# 1. Find the project folder and load raw customer data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

customers = pd.read_csv(RAW_DIR / "customers.csv")

# 2.Standardize city names
customers["city"] = customers["city"].str.strip().str.title()

# 3.Standardize gender values
customers["gender"] = (customers["gender"].astype("string").str.strip().str.lower())

gender_mapping = {"m": "male","male": "male","f": "female","female": "female",}

customers["gender"] = (customers["gender"].map(gender_mapping).fillna("unknown"))

# 4. Convert age to numeric values
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")

# 5. Treat ages below 10 or above 100 as missing
invalid_age = (customers["age"] < 10) | (customers["age"] > 100)
customers.loc[invalid_age, "age"] = float("nan")

# 6. Fill missing ages using the median age of each city
city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)

# 7. Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")

# 8. Create the output folder and save the cleaned data
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
customers.to_csv(CLEAN_DIR / "customers.csv", index=False)

# 9. Validate the results
print("City counts:")
print(customers["city"].value_counts().sort_index())

print("\nGender counts:")
print(customers["gender"].value_counts().sort_index())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())
print("Signup date type:", customers["signup_date"].dtype)
print("Cleaned customer rows:", len(customers))

assert customers["city"].nunique() == 6
assert customers["gender"].value_counts().to_dict() == {
    "female": 610,
    "male": 560,
    "unknown": 30,
}
assert customers["age"].isna().sum() == 0
assert customers["age"].between(18, 60).all()

print("\nCUSTOMER CLEANING CHECKS PASSED")
