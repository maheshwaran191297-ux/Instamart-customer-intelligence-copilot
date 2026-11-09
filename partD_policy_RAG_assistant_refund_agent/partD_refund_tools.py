
import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_DIR / "data" / "instamart.db"
OUTPUT_DIR = PROJECT_DIR / "outputs"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"

OUTPUT_DIR.mkdir(exist_ok=True)


# Import your existing policy retrieval function
from partD_retrieval import retrieve


# ------------------------------------------------
# TOOL 1: Get order details from SQLite database
# ------------------------------------------------

def get_order(order_id):

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
    """, (order_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": (
            None if row["delivery_minutes"] is None
            else int(row["delivery_minutes"])
        )
    }



# TOOL 2: Search relevant policy documents

def search_policy(query):

    # Retrieve top 3 relevant policy chunks
    results = retrieve(query, k=3)

    return {
        "results": [
            {
                "source": item["source"],
                "score": item["score"],
                "text": item["text"]
            }
            for item in results
        ]
    }


# ------------------------------------------------
# TOOL 3: Calculate refund amount
# ------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):

    if delivery_minutes is None or delivery_minutes <= 45:
        refund_amount = 0.0
    else:
        refund_amount = float(order_amount)

    return {
        "refund_inr": round(refund_amount, 2)
    }


# ------------------------------------------------
# TOOL 4: Issue an automatic refund
# ------------------------------------------------

def issue_refund(order_id, refund_inr):

    OUTPUT_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {
        "refund_inr": round(float(refund_inr), 2)
    }


# ------------------------------------------------
# TOOL 5: Escalate refund to a human
# ------------------------------------------------

def escalate_to_human(order_id, refund_inr):

    OUTPUT_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {
        "refund_inr": round(float(refund_inr), 2)
    }


# ------------------------------------------------
# TOOL SPECIFICATIONS
# ------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order details using an order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search relevant refund policy documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund amount using order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "integer"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


# ------------------------------------------------
# TOOL MAP
# ------------------------------------------------

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


print("Refund tools created successfully!")
print("Total tools:", len(TOOL_MAP))
