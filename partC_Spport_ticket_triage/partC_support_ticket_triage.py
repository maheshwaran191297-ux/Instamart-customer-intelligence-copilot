#importing required libraries
import json
import random
import time
import pandas as pd
import re
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import mock_llm

from mock_llm import est_tokens
# 1. Set project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "tickets.csv"

# 2. Load ticket data
tickets = pd.read_csv(INPUT_FILE, keep_default_na=False)

# 3. Set maximum allowed input tokens
MAX_INPUT_TOKENS = 400

# 4. Define privacy redaction patterns
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"

# 5. Create a redaction function
def redact_text(text):
    text = str(text)

    text, email_count = re.subn(EMAIL_PATTERN, "[EMAIL]", text)
    text, phone_count = re.subn(PHONE_PATTERN, "[PHONE]", text)
    text, order_count = re.subn(ORDER_PATTERN, "[ORDER_ID]", text)

    counts = {
        "email": email_count,
        "phone": phone_count,
        "order_id": order_count
    }

    return text, counts


# 6. Apply input guardrails and redaction
results = []

total_emails = 0
total_phones = 0
total_order_ids = 0

for _, ticket in tickets.iterrows():

    original_text = ticket["text"]

    # Check empty ticket
    if not original_text.strip():
        status = "skipped_empty"
        redacted_text = ""
        counts = {"email": 0, "phone": 0, "order_id": 0}

    # Check excessively long ticket
    elif est_tokens(original_text) > MAX_INPUT_TOKENS:
        status = "flagged_long"
        redacted_text = ""
        counts = {"email": 0, "phone": 0, "order_id": 0}

    else:
        status = "ready"
        redacted_text, counts = redact_text(original_text)

    total_emails += counts["email"]
    total_phones += counts["phone"]
    total_order_ids += counts["order_id"]

    results.append({
        "ticket_id": ticket["ticket_id"],
        "customer_id": ticket["customer_id"],
        "order_id": ticket["order_id"],
        "original_text": original_text,
        "redacted_text": redacted_text,
        "status": status
    })


# 7. Display audit results
print("INPUT GUARDRAIL AND PRIVACY AUDIT")
print("-" * 45)

print("Total tickets:", len(tickets))
print("Empty tickets:", sum(r["status"] == "skipped_empty" for r in results))
print("Long tickets:", sum(r["status"] == "flagged_long" for r in results))

print("\nRedaction counts:")
print("Emails:", total_emails)
print("Phone numbers:", total_phones)
print("Order IDs:", total_order_ids)

print("Total redaction:", total_emails + total_phones + total_order_ids)

print("\nStatus distribution:")
print(pd.Series([r["status"] for r in results]).value_counts())

print("\nInput guardrail and privacy audit completed.")
# 8. Validate that no PII remains
for item in results:
    text = item["redacted_text"]

    assert re.search(EMAIL_PATTERN, text) is None
    assert re.search(PHONE_PATTERN, text) is None
    assert re.search(ORDER_PATTERN, text) is None

print("\nPII validation passed: No residual sensitive information found.")
# 9. Define allowed ticket categories

ALLOWED_CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]


# 10. Zero-shot system prompt

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Return your answer in JSON only, using this format:
{"category": "...", "reason": "..."}
"""


# 11. Few-shot system prompt

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

EXAMPLES:

Ticket: I noticed an unauthorized transaction in my account.
Answer: {"category": "fraud", "reason": "Unauthorized transaction"}

Ticket: My app crashes whenever I try to open it.
Answer: {"category": "technical", "reason": "App malfunction"}

Ticket: I was charged twice for the same order.
Answer: {"category": "billing", "reason": "Duplicate charge"}

Return your answer in JSON only, using this format:
{"category": "...", "reason": "..."}
"""


# 12. Validate prompt definitions

assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "EXAMPLES:" not in ZERO_SHOT_PROMPT

assert "CLASSIFY" in FEW_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT

for category in ALLOWED_CATEGORIES:
    assert category in ZERO_SHOT_PROMPT
    assert category in FEW_SHOT_PROMPT

print("\nZero-shot and few-shot prompts created successfully.")
# 13. Retry configuration

MAX_RETRIES = 3
BASE_DELAY = 0.05


# 14. Function to call the model with retry handling
from mock_llm import est_tokens
def call_with_retry(messages, ticket_id):

    for retry_number in range(MAX_RETRIES + 1):

        try:
            response = mock_llm.chat(messages)
            return response

        except mock_llm.TransientLLMError:

            if retry_number == MAX_RETRIES:
                print(f"Ticket {ticket_id}: Failed after maximum retries.")
                raise

            wait_time = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Ticket {ticket_id} | "
                f"Retry {retry_number} | "
                f"Waiting {wait_time:.4f} seconds"
            )

            time.sleep(wait_time)


print("\nRetry mechanism created successfully.")
# 15. Function to validate model responses

def validate_response(response_text):

    try:
        result = json.loads(response_text)

        category = result.get("category")

        if category in ALLOWED_CATEGORIES:
            return category, "classified_json"

        return None, "flagged_invalid_output"

    except json.JSONDecodeError:

        words = re.findall(r"[a-z]+", response_text.lower())

        found_categories = [
            category
            for category in ALLOWED_CATEGORIES
            if category in words
        ]

        if len(found_categories) == 1:
            return found_categories[0], "classified_keyword"

        return None, "flagged_invalid_output"


print("\nResponse validation function created successfully.")
# 16. Run the complete ticket classification pipeline

def run_classification(system_prompt, run_name, token_budget=None):

    # Reset mock model before each complete run
    mock_llm.reset()

    classification_results = []
    total_tokens = 0

    for item in results:

        ticket_id = item["ticket_id"]

        # Skip empty and long tickets
        if item["status"] != "ready":

            classification_results.append({
    "ticket_id": ticket_id,
    "customer_id": item["customer_id"],
    "status": status,
    "category": category,
    "tokens_used": tokens_used
})

            continue

        # Check token budget before making a model call
        if token_budget is not None and total_tokens >= token_budget:

            classification_results.append({
    "ticket_id": ticket_id,
    "customer_id": item["customer_id"],
    "status": item["status"],
    "category": None,
    "tokens_used": 0
})

            continue
        #budget_exhausted
        if token_budget is not None and total_tokens + MAX_INPUT_TOKENS > token_budget:
            classification_results.append({
                "ticket_id": ticket_id,
                "customer_id": item["customer_id"],
                "status": "budget_exhausted",
                "category": None,
                "tokens_used": 0
            })
            continue
        # User message contains only redacted ticket text
        user_message = item["redacted_text"]

        # PII safety checks
        assert re.search(EMAIL_PATTERN, user_message) is None
        assert re.search(PHONE_PATTERN, user_message) is None
        assert re.search(ORDER_PATTERN, user_message) is None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Call model with retry handling
        response = call_with_retry(messages, ticket_id)

        response_text = response.choices[0].message.content

        # Extract token usage
        tokens_used = response.usage.total_tokens
        total_tokens += tokens_used

        # Validate model response
        category, status = validate_response(response_text)

        classification_results.append({
            "ticket_id": ticket_id,
            "status": status,
            "category": category,
            "tokens_used": tokens_used
        })

    print(f"\n{run_name} classification completed.")
    print("Total tickets processed:", len(classification_results))
    print("Total response tokens:", total_tokens)

    print("\nClassification status distribution:")

    print(
        pd.Series(
            [r["status"] for r in classification_results]
        ).value_counts()
    )

    return classification_results, total_tokens
# 17. Run zero-shot and few-shot experiments

zero_results, zero_tokens = run_classification(
    ZERO_SHOT_PROMPT,
    "ZERO-SHOT"
)

few_results, few_tokens = run_classification(
    FEW_SHOT_PROMPT,
    "FEW-SHOT"
)


# 18. Create evaluation DataFrames

zero_df = pd.DataFrame(zero_results)
few_df = pd.DataFrame(few_results)


# Add actual ticket categories for evaluation
zero_df = zero_df.merge(
    tickets[["ticket_id", "true_category"]],
    on="ticket_id",
    how="left"
)

few_df = few_df.merge(
    tickets[["ticket_id", "true_category"]],
    on="ticket_id",
    how="left"
)


# 19. Calculate classification metrics

def evaluate_model(df, model_name, total_tokens):

    classified = df[
        df["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    accuracy = (
        classified["category"] == classified["true_category"]
    ).mean()

    classified_count = len(classified)

    tokens_per_ticket = (
        total_tokens / classified_count
        if classified_count > 0
        else 0
    )

    print(f"\n{model_name} EVALUATION")
    print("-" * 40)

    print("Classified tickets:", classified_count)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Total response tokens:", total_tokens)
    print("Tokens per classified ticket:", round(tokens_per_ticket, 2))

    return accuracy


zero_accuracy = evaluate_model(
    zero_df,
    "ZERO-SHOT",
    zero_tokens
)

few_accuracy = evaluate_model(
    few_df,
    "FEW-SHOT",
    few_tokens
)


# 20. Display zero-shot confusion table

zero_classified = zero_df[
    zero_df["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

confusion_table = pd.crosstab(
    zero_classified["true_category"],
    zero_classified["category"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print("\nZERO-SHOT CONFUSION TABLE")
print(confusion_table)


print("\nZero-shot vs Few-shot comparison completed.")
# 21. Run zero-shot classification with a token budget

TOKEN_BUDGET = 6000

budget_results, budget_tokens = run_classification(
    ZERO_SHOT_PROMPT,
    "ZERO-SHOT BUDGET DEMO",
    token_budget=TOKEN_BUDGET
)


# 22. Save budget demonstration results

budget_df = pd.DataFrame(budget_results)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

budget_file = OUTPUT_DIR / "triage_budget_demo.csv"

budget_df.to_csv(budget_file, index=False)

print("\nBUDGET DEMONSTRATION")
print("-" * 40)

print("Token budget:", TOKEN_BUDGET)
print("Actual tokens used:", budget_tokens)

print(
    "Budget-exhausted tickets:",
    (budget_df["status"] == "budget_exhausted").sum()
)

print("Saved to:", budget_file)

assert (budget_df["status"] == "budget_exhausted").sum() > 0

print("\nToken budget validation passed.")
# 23. Final few-shot classification with a large token budget

FINAL_TOKEN_BUDGET = 100000

final_results, final_tokens = run_classification(
    FEW_SHOT_PROMPT,
    "FINAL FEW-SHOT RUN",
    token_budget=FINAL_TOKEN_BUDGET
)

print("\nFinal few-shot classification completed.")
print("Total tokens used:", final_tokens)
# 24. Load customer segmentation data

SEGMENT_FILE = OUTPUT_DIR / "customer_segments.csv"

segments = pd.read_csv(SEGMENT_FILE)

print("\nCustomer segmentation data loaded.")
print("Total customer records:", len(segments))
print("Available columns:", segments.columns.tolist())
# 25. Create final ticket routing

final_df = pd.DataFrame(final_results)

# Add actual customer segment information
final_df = final_df.merge(
    segments[["customer_id", "segment_name", "is_premium"]],
    on="customer_id",
    how="left"
)

# Missing segment means the customer is not premium
final_df["is_premium"] = (
    final_df["is_premium"].fillna(False).astype(bool)
)

# Keep segment name blank when unavailable
final_df["segment_name"] = final_df["segment_name"].fillna("")

# Assign priority and route
def assign_route(row):

    if row["status"] in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    ]:
        return "human_review", "NORMAL"

    elif row["category"] == "fraud":
        return "security_team", "HIGH"

    elif row["is_premium"]:
        return "premium_desk", "HIGH"

    else:
        return "support_queue", "NORMAL"


final_df[["route", "priority"]] = final_df.apply(
    assign_route,
    axis=1,
    result_type="expand"
)

# Prepare the required output columns
final_df = final_df[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]

# Replace unavailable categories with blank values
final_df["category"] = final_df["category"].fillna("")

# Save final results
FINAL_FILE = OUTPUT_DIR / "triage_results.csv"

final_df.to_csv(FINAL_FILE, index=False)

print("\nFINAL TICKET ROUTING COMPLETED")
print("-" * 40)

print("Total tickets:", len(final_df))

print("\nRoute distribution:")
print(final_df["route"].value_counts())

print("\nPriority distribution:")
print(final_df["priority"].value_counts())

print("\nStatus distribution:")
print(final_df["status"].value_counts())

print("\nSaved final results to:", FINAL_FILE)
# 26. Final output validation

assert len(final_df) == 164

assert list(final_df.columns) == [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

assert final_df["ticket_id"].is_unique

assert (
    final_df.loc[
        final_df["status"].isin([
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted"
        ]),
        "route"
    ] == "human_review"
).all()

print("\nAll final output validation checks passed.")
import pandas as pd

df = pd.read_csv("outputs/triage_results.csv")

print(df["status"].value_counts())
print(df[df["status"].isin([
    "skipped_empty",
    "flagged_long"
])][["ticket_id", "status", "route"]])
# Display tickets that failed validation

invalid_tickets = df[
    df["status"] == "flagged_invalid_output"
]

print(invalid_tickets[
    ["ticket_id", "status", "route"]
])

print("Invalid tickets:", len(invalid_tickets))