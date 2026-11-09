
import pandas as pd
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
SQL_PATH = PROJECT_ROOT / "sql" / "queries.sql"

print("Reading SQL file...")

sql_text = SQL_PATH.read_text(encoding="utf-8")

queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

print("Queries detected:", len(queries))

with sqlite3.connect(DB_PATH) as connection:

    for i, query in enumerate(queries, start=1):

        # Remove SQL comment lines
        clean_query = "\n".join(
            line for line in query.splitlines()
            if not line.strip().startswith("--")
        ).strip()

        if not clean_query:
            continue

        print(f"\n========== QUERY {i} ==========")

        result = pd.read_sql_query(clean_query, connection)

        print("Rows returned:", len(result))
        print(result.head(10).to_string(index=False))

print("\nALL SQL QUERIES EXECUTED SUCCESSFULLY")