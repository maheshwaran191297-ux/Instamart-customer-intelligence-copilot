
import pandas as pd
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the raw ticket dataset
tickets_path = PROJECT_ROOT / "data" / "raw" / "tickets.csv"

tickets = pd.read_csv(tickets_path, keep_default_na=False)

# Display basic dataset information
print("TICKET DATA AUDIT")
print("-" * 40)

print("Dataset shape:", tickets.shape)
print("\nColumn names:")
print(tickets.columns.tolist())

print("\nFirst 5 tickets:")
print(tickets.head().to_string(index=False))

# Check missing and empty values
print("\nEmpty values in each column:")
print((tickets == "").sum())

# Check duplicate ticket IDs
print("\nDuplicate ticket IDs:", tickets["ticket_id"].duplicated().sum())

# Check ticket category distribution
print("\nTicket category distribution:")
print(tickets["true_category"].value_counts())

print("\nTicket audit completed.")
