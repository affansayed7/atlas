"""Unit tests for streak logic — fast, deterministic, no external calls."""

from datetime import date
from src.analytics.streak import calculate_streak

TODAY = date(2026, 7, 15)


def test_empty_history():
    assert calculate_streak([], TODAY) == 0
    print("PASS: empty history → 0")


def test_today_only():
    assert calculate_streak(["2026-07-15"], TODAY) == 1
    print("PASS: today only → 1")


def test_three_consecutive_ending_today():
    assert calculate_streak(["2026-07-15", "2026-07-14", "2026-07-13"], TODAY) == 3
    print("PASS: 3 consecutive → 3")


def test_gap_breaks_streak():
    # logged today, skipped yesterday, logged before → streak is just today
    assert calculate_streak(["2026-07-15", "2026-07-13", "2026-07-12"], TODAY) == 1
    print("PASS: gap breaks streak → 1")


def test_grace_today_not_logged_yet():
    # nothing today, but logged yesterday & before → streak still counts (grace)
    assert calculate_streak(["2026-07-14", "2026-07-13"], TODAY) == 2
    print("PASS: grace period (today not logged) → 2")


if __name__ == "__main__":
    test_empty_history()
    test_today_only()
    test_three_consecutive_ending_today()
    test_gap_breaks_streak()
    test_grace_today_not_logged_yet()
    print("\nAll streak tests passed ✅")