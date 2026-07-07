"""Repository layer — the only place that talks SQL to the events table.

Bot/agent code calls these functions and never writes SQL directly
(separation of concerns: swapping the DB later touches only this file).
"""

import logging
from src.db.schema import get_connection

logger = logging.getLogger(__name__)


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