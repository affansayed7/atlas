"""Repository layer — the only place that talks SQL to the events table.

Bot/agent code calls these functions and never writes SQL directly
(separation of concerns: swapping the DB later touches only this file).
"""

import logging
from src.db.schema import get_connection

logger = logging.getLogger(__name__)


def get_today_events(user_id: str) -> list[dict]:
    """Events logged 'today' in IST (not UTC).

    Timestamps are stored in UTC. We shift both the stored timestamp and
    'now' by +5:30 to IST before comparing dates, so a 2 AM IST log counts
    as today, not yesterday.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT subject, topic, activity, count, raw_text,
                   datetime(timestamp, '+5 hours', '+30 minutes') AS ist_time
            FROM events
            WHERE user_id = ?
              AND subject != 'unparsed'
              AND date(timestamp, '+5 hours', '+30 minutes') = date('now', '+5 hours', '+30 minutes')
            ORDER BY timestamp DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {"subject": r[0], "topic": r[1], "activity": r[2], "count": r[3],
         "raw_text": r[4], "ist_time": r[5]}
        for r in rows
    ]



def get_summary(user_id: str) -> list[dict]:
    """Aggregate events per subject for a user. Returns list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT subject,
                   COUNT(*)              AS event_count,
                   SUM(COALESCE(count, 0))     AS total_count,
                   SUM(COALESCE(duration_min, 0)) AS total_minutes
            FROM events
            WHERE user_id = ? AND subject != 'unparsed'
            GROUP BY subject
            ORDER BY event_count DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {"subject": r[0], "events": r[1], "items": r[2], "minutes": r[3]}
        for r in rows
    ]


def save_parsed_events(user_id: str, raw_text: str, events: list[dict], source: str = "telegram") -> list[int]:
    """Persist parsed events. Returns list of new row ids."""
    ids = []
    with get_connection() as conn:
        for e in events:
            cursor = conn.execute(
                """
                INSERT INTO events
                    (user_id, subject, topic, activity, count, correct,
                     difficulty, outcome, duration_min, raw_text, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, e["subject"], e.get("topic"), e["activity"],
                 e.get("count"), e.get("correct"), e.get("difficulty"),
                 e.get("outcome"), e.get("duration_min"), raw_text, source),
            )
            ids.append(cursor.lastrowid)
    logger.info("Saved %d parsed events for user=%s", len(ids), user_id)
    return ids


def save_raw_message(user_id: str, raw_text: str, source: str = "telegram") -> int:
    """Persist an unparsed message as a raw event. Returns the new row id.

    v0.1: subject/activity are placeholders ('unparsed'/'logged') —
    the LLM parser (v0.2) will re-process rows where subject='unparsed'
    using raw_text as the source of truth.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO events (user_id, subject, activity, raw_text, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, "unparsed", "logged", raw_text, source),
        )
        row_id = cursor.lastrowid
    logger.info("Saved raw event id=%s for user=%s", row_id, user_id)
    return row_id


def count_events(user_id: str) -> int:
    """Total events logged by a user (for the bot's little confirmations)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row[0]