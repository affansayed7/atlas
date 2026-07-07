"""Atlas database schema — creates the events table.

Design notes:
- user_id on every row: multi-tenancy from day one
- raw_text always stored: source of truth, allows re-parsing history
- nullable detail columns: partial logs must be accepted (friction kills logging)
- one message may produce MANY event rows (long-format logs)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "atlas.db"

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
    subject      TEXT NOT NULL,
    topic        TEXT,
    activity     TEXT NOT NULL,
    count        INTEGER,
    correct      INTEGER,
    difficulty   TEXT,
    outcome      TEXT,
    duration_min INTEGER,
    raw_text     TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'telegram'
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_user_subject ON events (user_id, subject);",
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp);",
]


def get_connection() -> sqlite3.Connection:
    """Open a connection to the Atlas database (creates file if missing)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    with get_connection() as conn:
        conn.execute(CREATE_EVENTS_TABLE)
        for stmt in CREATE_INDEXES:
            conn.execute(stmt)
    print(f"Database initialised at {DB_PATH}")


if __name__ == "__main__":
    init_db()