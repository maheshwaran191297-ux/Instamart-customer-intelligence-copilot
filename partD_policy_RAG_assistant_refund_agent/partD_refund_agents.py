
import sys
import json
import time
import uuid
import csv
from pathlib import Path

# ------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import mock_llm

from partD_refund_tools import TOOL_MAP, TOOLS_SPEC

OUTPUT_DIR = PROJECT_DIR / "outputs"
STATE_DIR = PROJECT_DIR / "state"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

RUN_LOG = OUTPUT_DIR / "agent_runs.jsonl"


# ------------------------------------------------
# CHECKPOINT FUNCTIONS
# ------------------------------------------------

class SimulatedCrash(Exception):
    def __init__(self, run_id):
        self.run_id = run_id
        super().__init__(f"Simulated crash for run {run_id}")


def save_checkpoint(state):

    checkpoint_path = STATE_DIR / f"{state['run_id']}.json"

    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4, ensure_ascii=False)

    return checkpoint_path


def load_checkpoint(run_id):

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run ID: {run_id}"
        )

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_run_log(result):

    with open(RUN_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


# ------------------------------------------------
# REACT AGENT EXECUTION
# ------------------------------------------------

def continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):

    start_time = time.monotonic()

    # Continue from the saved model-call count
    while state["llm_calls"] < max_iterations:

        # Check timeout before model call
        elapsed = time.monotonic() - start_time

        if elapsed >= time_limit_seconds:

            state["status"] = "FALLBACK_TIMEOUT"
            state["final_answer"] = (
                "FALLBACK_TIMEOUT: Request sent to human support."
            )
            break

        # Call offline mock LLM
        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1

        state["tokens_used"] += response.usage.total_tokens

        message = response.choices[0].message
        tool_calls = message.tool_calls

        # Check token limit before executing requested tools
        if state["tokens_used"] > token_limit:

            state["status"] = "FALLBACK_TOKEN_LIMIT"
            state["final_answer"] = (
                "FALLBACK_TOKEN_LIMIT: Request sent to human support."
            )
            break

        # Model has provided its final answer
        if not tool_calls:

            state["final_answer"] = message.content
            state["status"] = message.content.split(":")[0].strip()
            break

        # Store assistant tool-call message
        assistant_tool_calls = []

        for tool_call in tool_calls:

            assistant_tool_calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        # Execute requested tools
        for tool_call in tool_calls:

            tool_name = tool_call.name
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            tool_function = TOOL_MAP.get(tool_name)

            if tool_function is None:
                observation = {"error": "unknown tool"}
            else:
                observation = tool_function(**arguments)

            # Record completed tool execution
            state["tools_executed"].append(tool_name)

            step = {
                "step": len(state["steps"]) + 1,
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            state["steps"].append(step)

            # Add tool observation to conversation
            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(
                    observation,
                    ensure_ascii=False
                )
            })

            # IMPORTANT: Save after every tool observation
            save_checkpoint(state)

            print(f"Checkpoint saved after tool: {tool_name}")

            # Simulate crash only AFTER checkpoint is saved
            if (
                crash_after_tools is not None
                and len(state["tools_executed"]) >= crash_after_tools
            ):
                raise SimulatedCrash(state["run_id"])

    # Maximum iteration fallback
    if state["final_answer"] is None:

        state["status"] = "FALLBACK_MAX_ITERATIONS"
        state["final_answer"] = (
            "FALLBACK_MAX_ITERATIONS: "
            "Request sent to human support for review."
        )

    # Mark run as completed
    state["completed"] = True

    # Save final state
    save_checkpoint(state)

    result = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": state["status"],
        "final_answer": state["final_answer"],
        "steps": state["steps"],
        "tools_executed": state["tools_executed"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"]
    }

    write_run_log(result)

    return result


# ------------------------------------------------
# START A NEW AGENT RUN
# ------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):

    run_id = str(uuid.uuid4())

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,

        "messages": [
            {
                "role": "system",
                "content": """
                AGENT

                You are an Instamart refund processing agent.

                Use the available tools to process refund requests.
                Always check the order and policy before deciding.
                Never issue refunds for cancelled orders.
                Follow the refund rules strictly.

                Provide a final answer with a clear status:
                REFUND, ESCALATE, DENIED or NOT_FOUND.
                """
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        "steps": [],
        "tools_executed": [],
        "llm_calls": 0,
        "tokens_used": 0,
        "status": None,
        "final_answer": None,
        "completed": False
    }

    # Save initial state too
    save_checkpoint(state)

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools
    )


# ------------------------------------------------
# RESUME AN EXISTING AGENT RUN
# ------------------------------------------------

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):

    print("\nRESUMING AGENT RUN")
    print("------------------------------")

    # Load previous checkpoint
    state = load_checkpoint(run_id)

    # Avoid executing a completed run again
    if state.get("completed", False):

        print("This run is already completed.")

        return {
            "run_id": state["run_id"],
            "scenario_id": state["scenario_id"],
            "status": state["status"],
            "final_answer": state["final_answer"],
            "steps": state["steps"],
            "tools_executed": state["tools_executed"],
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"]
        }

    print("Scenario:", state["scenario_id"])
    print("Previous model calls:", state["llm_calls"])
    print("Previous tools:", state["tools_executed"])

    # Continue using the saved messages and execution history
    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )


# ------------------------------------------------
# RUN ALL SEVEN ASSIGNMENT SCENARIOS
# ------------------------------------------------

if __name__ == "__main__":

    SCENARIO_PATH = (
        PROJECT_DIR / "data" / "raw" / "agent_scenarios.csv"
    )

    with open(
        SCENARIO_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        scenarios = list(csv.DictReader(file))

    print("\nRUNNING ALL AGENT SCENARIOS")
    print("=" * 50)

    for scenario in scenarios:

        scenario_id = scenario["scenario_id"]
        customer_message = scenario["message"]
        expected_status = scenario["expected_outcome"]

        if expected_status == "FALLBACK":
            expected_status = "FALLBACK_MAX_ITERATIONS"

        if hasattr(mock_llm, "reset"):
            mock_llm.reset()

        print(f"\nRunning scenario: {scenario_id}")

        result = run_agent(
            user_message=customer_message,
            scenario_id=scenario_id
        )

        print("Expected:", expected_status)
        print("Actual:", result["status"])
        print("Model calls:", result["llm_calls"])
        print("Tokens used:", result["tokens_used"])
        print("Tools:", result["tools_executed"])

        if result["status"] == expected_status:
            print("Validation: PASSED")
        else:
            print("Validation: FAILED")

    print("\nALL SCENARIO TESTS COMPLETED")