# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an end-to-end Customer Intelligence and Support Copilot using synthetic Instamart data. It combines data engineering, customer segmentation, machine learning, LLM-based ticket classification, Retrieval-Augmented Generation (RAG), and an agent-based refund workflow.

The project is divided into four parts:

- Part A: Data Foundation
- Part B: Customer Intelligence
- Part C: Support Ticket Triage
- Part D: Policy RAG Assistant and Refund Agent

## Reproduce

1. Create and activate a Python virtual environment.
2. Install the required dependencies from `requirements.txt`.
3. Run the project scripts from the project root directory.
4. Execute the Part A data audit, cleaning, database creation, and SQL validation scripts.
5. Execute the Part B customer intelligence and prediction scripts.
6. Execute the Part C ticket classification and routing scripts.
7. Execute the Part D policy loader, retrieval, RAG evaluation, refund agent, and crash-resume test scripts.

The project uses a supplied offline `mock_llm.py` for LLM interactions. No external API key or network connection is required for the mock-based workflows.

## Part A - Data Foundation

- Audited raw customer, product, and order datasets to identify missing values, duplicates, and data inconsistencies.
- Cleaned customer data by handling missing ages, standardizing city names, and normalizing gender values.
- Cleaned product data by removing duplicates and validating product details and prices.
- Cleaned order data by removing duplicates, handling missing amounts, and correcting invalid delivery times.
- Created and validated a SQLite database containing the cleaned customers, products, and orders tables.
- Performed SQL queries and Pandas analysis to explore revenue, customer spending, order trends, and data relationships.

## Part B - Customer Intelligence

- Performed RFM (Recency, Frequency, Monetary) analysis to understand customer purchasing behavior.
- Applied K-Means clustering with different K values and selected four clusters using inertia and silhouette scores.
- Created customer segments such as Premium, Active, and At-Risk based on purchasing patterns.
- Used PCA to visualize customer segments, with the first two components explaining 85.2% of the variance.
- Built and evaluated customer lapse prediction models using Logistic Regression and Random Forest, with oversampling to handle class imbalance.
- Saved the final lapse prediction model, customer segmentation results, and a prioritized retention list for targeted customer engagement.

## Part C - Support Ticket Triage

- Processed customer support tickets using an LLM-based classification system.
- Developed ticket classification functions to identify support ticket categories.
- Implemented JSON response validation with keyword-based fallback handling.
- Classified 164 tickets and routed them based on category and priority.
- Flagged invalid model outputs for human review to ensure safe ticket routing.
- Implemented ticket routing to support, security, premium, and human-review queues.

## Part D - Policy RAG Assistant and Refund Agent

### 1. Policy Document Processing

- Loaded five policy documents covering delivery, escalation, payment, privacy, and refunds.
- Implemented text chunking with a chunk size of 40 words and an overlap of 10 words.
- Generated 15 policy chunks for retrieval.

### 2. TF-IDF Retrieval

- Implemented TF-IDF using `TfidfVectorizer`.
- Converted policy chunks and user queries into TF-IDF vectors.
- Used cosine similarity to retrieve the top three relevant policy chunks.
- Validated that the top three results for the refund and late-delivery query came from `refund_policy.txt`.

### 3. RAG Assistant and Evaluation

- Developed an `answer(question)` function to retrieve policy context and generate grounded responses using the offline mock LLM.
- Restricted answers to the supplied policy context.
- Implemented a fallback response for questions not covered by the policy documents.
- Evaluated the system using 14 questions: 12 answerable and 2 unanswerable.

**Policy evaluation results:**

| Evaluation metric | 10-word overlap | Zero overlap |
|---|---:|---:|
| Expected document retrieved in top 3 | 12/12 | 12/12 |
| Expected fact found in answer | 12/12 | 10/12 |
| Correct unanswerable fallback | 2/2 | 2/2 |

The zero-overlap experiment failed on two questions because important information was split across chunk boundaries. The 10-word overlap configuration successfully retrieved the required facts for all 12 answerable questions.

### 4. Refund Tools and ReAct Agent

Implemented five tools:

- `get_order`: Retrieves order details from the SQLite database.
- `search_policy`: Retrieves relevant policy chunks.
- `calculate_refund`: Calculates refund eligibility and amount.
- `issue_refund`: Records approved refunds in the refund ledger.
- `escalate_to_human`: Records refunds requiring human approval.

Implemented a ReAct-style agent using the offline mock LLM to reason, request tools, process observations, and produce final responses.

### 5. Agent Scenario Evaluation

All seven agent scenarios passed validation.

| Scenario | Expected status | Actual status | Model calls | Tokens used |
|---|---|---|---:|---:|
| S1 | REFUND | REFUND | 5 | 1,147 |
| S2 | REFUND | REFUND | 5 | 1,147 |
| S3 | ESCALATE | ESCALATE | 5 | 1,134 |
| S4 | DENIED | DENIED | 4 | 805 |
| S5 | DENIED | DENIED | 3 | 487 |
| S6 | NOT_FOUND | NOT_FOUND | 2 | 172 |
| S7 | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS | 6 | 3,300 |

### 6. Safety Limits and Crash Recovery

- Implemented maximum iteration, token usage, and execution time limits.
- Added fallback handling for token-limit, timeout, and maximum-iteration conditions.
- Saved agent checkpoints after every tool observation.
- Implemented `resume_agent()` to continue execution from a saved checkpoint.
- Tested a simulated crash after the second tool execution and successfully resumed the refund workflow.
- Verified that the four required tools executed exactly once across the crash and resume sequence.

### 7. ReAct vs Plan-and-Execute

The ReAct agent selects tools dynamically based on the latest observation. In the tested scenarios, S1 required five model calls, S5 required three, and S6 required two. This allows the agent to stop early when an order does not exist or is not eligible for a refund.

A plan-and-execute approach would first create a complete execution plan and then carry out its steps. This can provide more predictable execution, but may perform unnecessary steps when an order is missing or a refund is denied. ReAct was selected because refund decisions depend on order status, delivery time, refund amount, and human approval requirements. Dynamic tool selection is useful for these conditional workflows.

## Limitations and Next Steps

- The project uses synthetic datasets and a deterministic offline mock LLM rather than a production LLM.
- TF-IDF retrieval is based on keyword relevance and does not provide semantic embeddings.
- The RAG system depends on the quality and completeness of the supplied policy documents.
- The refund workflow demonstrates tool execution and ledger recording but does not connect to a real payment gateway.
- The agent uses predefined scenarios and simulated tool responses rather than live customer interactions.

**Future improvements:**

- Integrate a production LLM with secure API key management.
- Explore embedding-based semantic retrieval and vector databases.
- Add real-time customer support and order-tracking integrations.
- Introduce authentication, authorization, and audit controls for financial operations.
- Deploy the application through a web-based interface.
