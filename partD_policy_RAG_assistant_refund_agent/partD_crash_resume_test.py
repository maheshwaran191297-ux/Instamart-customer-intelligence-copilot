import sys
from pathlib import Path

# Get the project root directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Add project root to Python's search path
sys.path.insert(0, str(PROJECT_DIR))

import csv
import mock_llm

from partD_refund_agents import (
    run_agent,
    resume_agent,
    SimulatedCrash
)

# Read official S1 scenario
scenario_path = (
    PROJECT_DIR / "data" / "raw" / "agent_scenarios.csv"
)

with open(scenario_path, "r", encoding="utf-8", newline="") as file:
    scenarios = list(csv.DictReader(file))

s1 = next(
    scenario for scenario in scenarios
    if scenario["scenario_id"] == "S1"
)

# Reset mock LLM
if hasattr(mock_llm, "reset"):
    mock_llm.reset()

print("\nSTARTING S1 CRASH SIMULATION")

try:

    # Start agent and crash after second tool
    run_agent(
        user_message=s1["message"],
        scenario_id="S1_CRASH_TEST",
        crash_after_tools=2
    )

except SimulatedCrash as error:

    run_id = error.run_id

    print("\nSIMULATED CRASH OCCURRED")
    print("Run ID:", run_id)

    print("\nRESUMING FROM CHECKPOINT")

    # Resume saved agent
    result = resume_agent(run_id)

    print("\nFINAL RESUMED RESULT")
    print("------------------------------")

    print("Status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Model calls:", result["llm_calls"])
    print("Tokens used:", result["tokens_used"])

    print("\nTOOLS EXECUTED ACROSS CRASH AND RESUME:")

    for tool in result["tools_executed"]:
        print(tool)

    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund"
    ]

    print("\nVALIDATION")

    if (
        result["tools_executed"] == expected_tools
        and result["status"] == "REFUND"
        and len(result["tools_executed"]) == 4
    ):
        print("CRASH AND RESUME: PASSED")
    else:
        print("CRASH AND RESUME: FAILED")
print("CRASH AND RESUME: FAILED")
# =========================================
# TEST 1: TOKEN LIMIT FALLBACK
# =========================================

print("\n\nTESTING TOKEN LIMIT")
print("------------------------------")

if hasattr(mock_llm, "reset"):
    mock_llm.reset()

token_result = run_agent(
    user_message=s1["message"],
    scenario_id="S1_TOKEN_LIMIT_TEST",
    token_limit=200
)

print("Status:", token_result["status"])
print("Tokens used:", token_result["tokens_used"])
print("Model calls:", token_result["llm_calls"])


# =========================================
# TEST 2: TIME LIMIT FALLBACK
# =========================================

print("\n\nTESTING TIME LIMIT")
print("------------------------------")

if hasattr(mock_llm, "reset"):
    mock_llm.reset()

timeout_result = run_agent(
    user_message=s1["message"],
    scenario_id="S1_TIMEOUT_TEST",
    time_limit_seconds=0
)

print("Status:", timeout_result["status"])
print("Tokens used:", timeout_result["tokens_used"])
print("Model calls:", timeout_result["llm_calls"])