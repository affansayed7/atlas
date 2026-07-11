"""LLM parser — turns raw log messages into structured event dicts.

Design principles:
- One message may contain MANY events (long-format logs) → always returns a list
- The LLM is NOT trusted: output is validated before anything touches the DB
- Parsing failure is safe: raw_text is already persisted; we can always retry
"""

import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

VALID_SUBJECTS = {"dsa", "oops", "dbms", "cn", "os", "apti", "system_design", "gym", "other"}
VALID_ACTIVITIES = {"solved", "read", "revised", "watched", "tested", "trained", "other"}

SYSTEM_PROMPT = """You extract structured study/fitness events from a user's log message.

Return ONLY a JSON array. No prose, no markdown, no code fences. Each element:
{
  "subject": one of ["dsa","oops","dbms","cn","os","apti","system_design","gym","other"],
  "topic": short lowercase string or null (e.g. "dp", "bfs", "indexing", "pushups"),
  "activity": one of ["solved","read","revised","watched","tested","trained","other"],
  "count": integer or null (number of problems/questions/reps),
  "correct": integer or null (only if the message states how many were right),
  "difficulty": one of ["easy","medium","hard"] or null,
  "outcome": one of ["clean","struggled","failed"] or null,
  "duration_min": integer minutes or null
}

Rules:
- A message with multiple activities returns multiple array elements.
- Never invent values not stated in the message. Missing = null.
- Only extract COMPLETED activities. IGNORE intentions, plans, or future
  tense ("will do", "planning to", "later today", "tomorrow I'll").
- Aptitude/quant topics (percentages, time-and-distance, boats-streams) → subject "apti".
- Exercise/workout content → subject "gym", activity "trained".
- If nothing extractable, return [].

Example input: "today did 20 apti questions got 16 right and gym 1hr push day"
Example output: [{"subject":"apti","topic":null,"activity":"solved","count":20,"correct":16,"difficulty":null,"outcome":null,"duration_min":null},{"subject":"gym","topic":"push day","activity":"trained","count":null,"correct":null,"difficulty":null,"outcome":null,"duration_min":60}]
Example input: "today did 20 apti questions got 16 right and gym 1hr push day, will solve dsa later tonight"
Example output: [{"subject":"apti","topic":null,"activity":"solved","count":20,"correct":16,"difficulty":null,"outcome":null,"duration_min":null},{"subject":"gym","topic":"push day","activity":"trained","count":null,"correct":null,"difficulty":null,"outcome":null,"duration_min":60}]"""

def parse_message(raw_text: str) -> list[dict]:
    """Extract structured events from a raw message. Returns [] on any failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        events = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON for: %s", raw_text)
        return []
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return []

    if not isinstance(events, list):
        logger.warning("LLM returned non-list JSON for: %s", raw_text)
        return []

    return [e for e in events if _is_valid(e)]


def _is_valid(event: dict) -> bool:
    """Reject events that don't conform to our schema constraints."""
    if not isinstance(event, dict):
        return False
    if event.get("subject") not in VALID_SUBJECTS:
        return False
    if event.get("activity") not in VALID_ACTIVITIES:
        return False
    for int_field in ("count", "correct", "duration_min"):
        val = event.get(int_field)
        if val is not None and not isinstance(val, int):
            return False
    return True