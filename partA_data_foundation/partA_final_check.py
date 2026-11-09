
import pandas as pd
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"

with sqlite3.connect(DB_PATH) as connection:

    # Q6: Count ranking differences
    q6 = pd.read_sql_query("""
        WITH customer_spend AS (
            SELECT
                c.city,
                c.customer_id,
                SUM(o.order_amount) AS total_spend
            FROM customers c
            JOIN orders o
                ON c.customer_id = o.customer_id
            WHERE o.status = 'completed'
            GROUP BY c.city, c.customer_id
        ),
        rankings AS (
            SELECT
                city,
                customer_id,
                RANK() OVER (
                    PARTITION BY city
                    ORDER BY total_spend DESC
                ) AS spend_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY city
                    ORDER BY total_spend DESC
                ) AS row_num
            FROM customer_spend
        )
        SELECT COUNT(*) AS differences
        FROM rankings
        WHERE spend_rank != row_num;
    """, connection)

    differences = q6["differences"].iloc[0]

    print("Q6 ranking differences:", differences)
    assert differences == 8

    # Q7: Check final cumulative revenue
    q7 = pd.read_sql_query("""
        WITH daily_revenue AS (
            SELECT
                SUBSTR(order_ts, 1, 10) AS day,
                SUM(order_amount) AS revenue
            FROM orders
            WHERE status = 'completed'
            GROUP BY SUBSTR(order_ts, 1, 10)
        )
        SELECT
            day,
            SUM(revenue) OVER (
                ORDER BY day
                ROWS BETWEEN UNBOUNDED PRECEDING
                AND CURRENT ROW
            ) AS cumulative_revenue
        FROM daily_revenue
        ORDER BY day DESC
        LIMIT 1;
    """, connection)

    final_revenue = round(
        q7["cumulative_revenue"].iloc[0], 2
    )

    print("Q7 final cumulative revenue:", final_revenue)

    assert final_revenue == 2998668.80

print("\nALL FINAL CHECKS PASSED")
