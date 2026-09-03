"""
Evaluation harness for the merchant onboarding auditor.

Runs every scenario in eval/dataset.jsonl through the compiled LangGraph
`app`, compares the resulting decision / escalation flag against the
expected values, and prints a pass/fail summary. For any escalated case
it also prints the answer to the demo question:

    "Which checks caused this application to be escalated, and what
    evidence supports each one?"

Usage:
    python -m eval.run_eval
    (requires GOOGLE_API_KEY in .env for the LLM-backed fraud scoring /
    summary nodes; everything else in the pipeline is deterministic.)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from src.workflow import workflow, _empty_state  # noqa: E402
from src.schemas import RiskDossier  # noqa: E402

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
EVAL_DB_PATH = Path(__file__).parent / "eval_checkpoints.sqlite"


def load_dataset():
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_scenario(app, scenario: dict) -> dict:
    thread_config = {"configurable": {"thread_id": f"eval_{scenario['id']}"}}
    initial_state = _empty_state(
        scenario["merchant_name"], scenario["registration_number"], scenario["website_url"]
    )
    list(app.stream(initial_state, thread_config))
    final_state = app.get_state(thread_config).values
    return final_state


def main():
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print(
            "WARNING: GOOGLE_API_KEY / GEMINI_API_KEY not set. The fraud-scoring "
            "and summary nodes call an LLM and will fail without it. Set one in "
            ".env before running a full evaluation.\n"
        )

    # Each run must start from a clean checkpoint DB. The state channels for
    # policy_checks/discrepancies/escalation_reasons use LangGraph's
    # operator.add reducer, so re-running against a stale DB with the same
    # thread_ids would silently *append* onto results from a previous run
    # instead of replacing them.
    if EVAL_DB_PATH.exists():
        EVAL_DB_PATH.unlink()

    scenarios = load_dataset()
    passed, failed = 0, 0

    with SqliteSaver.from_conn_string(str(EVAL_DB_PATH)) as checkpointer:
        app = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

        for scenario in scenarios:
            print(f"\n{'=' * 70}\nScenario: {scenario['id']} - {scenario['merchant_name'][:60]}")
            try:
                final_state = run_scenario(app, scenario)
            except Exception as exc:  # pragma: no cover - depends on live API access
                print(f"  ERROR running scenario: {exc}")
                failed += 1
                continue

            decision = final_state.get("final_decision")
            requires_review = final_state.get("requires_human_review")

            decision_ok = decision == scenario["expected_decision"]
            review_ok = requires_review == scenario["expected_requires_human_review"]
            ok = decision_ok and review_ok

            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] decision={decision} (expected {scenario['expected_decision']}) | "
                  f"requires_human_review={requires_review} (expected {scenario['expected_requires_human_review']})")
            print(f"  notes: {scenario['notes']}")

            if requires_review and final_state.get("dossier"):
                dossier = RiskDossier(**final_state["dossier"])
                print("\n  --- Demo question answer ---")
                print("  " + dossier.escalation_report().replace("\n", "\n  "))

            if ok:
                passed += 1
            else:
                failed += 1

    print(f"\n{'=' * 70}\nEvaluation complete: {passed} passed, {failed} failed out of {len(scenarios)}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
