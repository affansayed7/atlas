"""Unit tests for LeetCode diff logic — pure, no network calls.

Tests compute_new_solves(): turning snapshot counts into new-solve events.
"""

from src.ingestion.leetcode import compute_new_solves


def test_first_run_no_baseline():
    """First poll (no previous snapshot) must NOT retroactively log all solves."""
    result = compute_new_solves(None, {"Easy": 10, "Medium": 5, "Hard": 1})
    assert result == {}, f"Expected empty on first run, got {result}"
    print("PASS: first run establishes baseline, no events")


def test_new_solves_detected():
    """Positive deltas per difficulty become new-solve counts."""
    prev = {"Easy": 10, "Medium": 5, "Hard": 1}
    curr = {"Easy": 12, "Medium": 5, "Hard": 2}
    result = compute_new_solves(prev, curr)
    assert result == {"Easy": 2, "Hard": 1}, f"Got {result}"
    print("PASS: detects new solves by difficulty")


def test_no_change():
    """Identical snapshots → no new events."""
    snap = {"Easy": 10, "Medium": 5, "Hard": 1}
    assert compute_new_solves(snap, snap) == {}
    print("PASS: no change → no events")


def test_ignores_negative_delta():
    """If a count drops (glitch/removed problem), don't log negative solves."""
    prev = {"Easy": 10, "Medium": 5, "Hard": 2}
    curr = {"Easy": 10, "Medium": 5, "Hard": 1}
    assert compute_new_solves(prev, curr) == {}
    print("PASS: negative delta ignored")


if __name__ == "__main__":
    test_first_run_no_baseline()
    test_new_solves_detected()
    test_no_change()
    test_ignores_negative_delta()
    print("\nAll LeetCode diff tests passed ✅")