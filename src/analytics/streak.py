"""Streak logic — pure functions, no DB/network, fully unit-testable."""

from datetime import date, timedelta


def calculate_streak(active_dates: list[str], today: date) -> int:
    """Count consecutive days ending at today (or yesterday) with activity.

    Rules:
    - active_dates: 'YYYY-MM-DD' strings, any order.
    - A streak counts backwards from today. If today isn't logged yet,
      we still count from yesterday (so the streak isn't shown broken
      mid-morning before you've logged).
    - The first missing day breaks the streak.
    """
    if not active_dates:
        return 0

    active = {date.fromisoformat(d) for d in active_dates}

    # Grace: start from today if logged, else yesterday.
    cursor = today if today in active else today - timedelta(days=1)

    streak = 0
    while cursor in active:
        streak += 1
        cursor -= timedelta(days=1)
    return streak