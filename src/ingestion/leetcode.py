"""LeetCode fetcher — pulls public solve stats via the unofficial GraphQL API.

IMPORTANT: leetcode.com/graphql is undocumented and unsupported. It can
change shape, rate-limit, or block us at any time. Therefore every path
here is defensive: on ANY failure we return None and never crash the caller.
"""

import logging
import requests

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://leetcode.com/graphql"

# Query for a user's solved-problem counts by difficulty
STATS_QUERY = """
query userProblemsSolved($username: String!) {
  matchedUser(username: $username) {
    submitStatsGlobal {
      acSubmissionNum {
        difficulty
        count
      }
    }
  }
}
"""


def fetch_solved_counts(username: str) -> dict | None:
    """Return {'All': int, 'Easy': int, 'Medium': int, 'Hard': int} or None on failure."""
    try:
        response = requests.post(
            GRAPHQL_URL,
            json={"query": STATS_QUERY, "variables": {"username": username}},
            headers={"Content-Type": "application/json", "Referer": "https://leetcode.com"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("LeetCode request failed: %s", exc)
        return None
    except ValueError:  # JSON decode error
        logger.error("LeetCode returned non-JSON")
        return None

    # Defensive navigation — the response shape may not be what we expect
    try:
        matched = data["data"]["matchedUser"]
        if matched is None:
            logger.warning("LeetCode user '%s' not found", username)
            return None
        stats = matched["submitStatsGlobal"]["acSubmissionNum"]
    except (KeyError, TypeError) as exc:
        logger.error("Unexpected LeetCode response shape: %s", exc)
        return None

    counts = {item["difficulty"]: item["count"] for item in stats}
    logger.info("Fetched LeetCode stats for %s: %s", username, counts)
    return counts

def compute_new_solves(previous: dict | None, current: dict) -> dict:
    """Diff two snapshots → new solves per difficulty.

    First run (previous is None) returns {} — we establish a baseline and
    do NOT retroactively log every historical solve. Only positive deltas
    count (a dropped count from a glitch shouldn't log negative solves).
    """
    if previous is None:
        return {}
    diff = {}
    for level in ("Easy", "Medium", "Hard"):
        delta = current.get(level, 0) - previous.get(level, 0)
        if delta > 0:
            diff[level] = delta
    return diff

def poll_and_log(user_id: str, username: str) -> dict:
    """Full poll cycle: fetch → diff against last snapshot → save new snapshot → log events.

    Returns the dict of new solves logged (empty on first run or no change).
    Imported here (not top-level) to keep this module usable without the DB in tests.
    """
    from src.db.repository import get_latest_snapshot, save_snapshot, save_parsed_events

    current = fetch_solved_counts(username)
    if current is None:
        logger.warning("Poll skipped — could not fetch LeetCode stats")
        return {}

    previous = get_latest_snapshot(user_id)
    new_solves = compute_new_solves(previous, current)

    # Always save the snapshot (baseline on first run, history thereafter)
    save_snapshot(user_id, username, current)

    # Turn new solves into events
    if new_solves:
        events = []
        for difficulty, n in new_solves.items():
            events.append({
                "subject": "dsa",
                "topic": None,
                "activity": "solved",
                "count": n,
                "correct": None,
                "difficulty": difficulty.lower(),
                "outcome": None,
                "duration_min": None,
            })
        raw = f"[LeetCode auto] {new_solves}"
        save_parsed_events(user_id=user_id, raw_text=raw, events=events, source="leetcode_api")
        logger.info("Logged %d new LeetCode solve-groups", len(events))

    return new_solves


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.db.schema import init_db
    init_db()
    result = poll_and_log(user_id="8257170471", username="NEOWMEOW")
    print("New solves logged:", result)