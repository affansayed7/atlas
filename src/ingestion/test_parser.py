"""Regression tests for the LLM parser — each maps to a case in docs/eval_cases.md.

Run: uv run python -m src.ingestion.test_parser
NOTE: makes real LLM calls (costs quota, needs network). Fine at this scale.
"""

from src.ingestion.parser import parse_message


def test_case_001_future_intention_ignored():
    """Case 001: future intentions must NOT be logged as completed events."""
    msg = "did percentages apti today, will solve more apti later tonight"
    events = parse_message(msg)
    apti_events = [e for e in events if e["subject"] == "apti"]
    assert len(apti_events) == 1, f"Expected 1 apti event, got {len(apti_events)}: {events}"
    print("PASS: case 001 — intention correctly ignored")


def test_multi_event_split_still_works():
    """Guard: the fix must not break normal multi-event parsing."""
    msg = "solved 3 dp mediums and read dbms indexing 45 mins"
    events = parse_message(msg)
    assert len(events) == 2, f"Expected 2 events, got {len(events)}: {events}"
    print("PASS: multi-event split intact")


if __name__ == "__main__":
    test_case_001_future_intention_ignored()
    test_multi_event_split_still_works()
    print("\nAll regression tests passed ✅")