"""Estimate per-subject mastery by running BKT over a user's logged attempts.

Modeling decisions (defensible in interview):
- Only events with a clear outcome count as 'attempts'. A log with no
  outcome (e.g. "read dbms 1hr") isn't a graded attempt — it's study, not
  assessment — so it doesn't move the mastery estimate.
- outcome mapping: 'clean' -> correct; 'struggled'/'failed' -> incorrect.
- Ordered oldest->newest so BKT sees the true learning sequence.
"""

from src.analytics.bkt import BKT
from src.db.schema import get_connection

# Which outcomes count as a graded attempt, and whether they're "correct"
OUTCOME_TO_CORRECT = {"clean": True, "struggled": False, "failed": False}


def get_subject_attempts(user_id: str) -> dict[str, list[bool]]:
    """Return {subject: [correct, correct, incorrect, ...]} oldest->newest."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT subject, outcome
            FROM events
            WHERE user_id = ? AND subject != 'unparsed' AND outcome IS NOT NULL
            ORDER BY timestamp ASC
            """,
            (user_id,),
        ).fetchall()

    attempts: dict[str, list[bool]] = {}
    for subject, outcome in rows:
        if outcome in OUTCOME_TO_CORRECT:
            attempts.setdefault(subject, []).append(OUTCOME_TO_CORRECT[outcome])
    return attempts


def estimate_mastery(user_id: str) -> list[dict]:
    """Run BKT per subject. Returns [{subject, mastery, attempts}] sorted by mastery."""
    attempts_by_subject = get_subject_attempts(user_id)
    results = []
    for subject, attempts in attempts_by_subject.items():
        bkt = BKT()
        mastery = bkt.fit_sequence(attempts)
        results.append({
            "subject": subject,
            "mastery": mastery,
            "attempts": len(attempts),
        })
    results.sort(key=lambda r: r["mastery"], reverse=True)
    return results