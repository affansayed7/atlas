"""Atlas Telegram bot — entry point.

v0.1 scope: receive messages, reply, log raw text to the events table.
"""

import logging
import os
from src.db.repository import get_summary, save_raw_message, save_parsed_events
from src.ingestion.parser import parse_message

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

def _plural(n: int, word: str) -> str:
    """Return '1 session' / '2 sessions' — handles singular/plural."""
    return f"{n} {word}{'' if n == 1 else 's'}"


# Load .env so the token is available via os.getenv
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /start command — Atlas introduces itself."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hello {user.first_name}! I'm Atlas 🗺️\n"
        "Your personal learning coach.\n\n"
        "For now, send me any study/gym log and I'll remember it. "
        "Soon I'll understand and analyse them too."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parse the message into structured events; fall back to raw storage."""
    text = update.message.text
    user_id = str(update.effective_user.id)

    events = parse_message(text)
    if events:
        ids = save_parsed_events(user_id=user_id, raw_text=text, events=events)
        summary = "\n".join(
            f"• {e['subject']}" + (f"/{e['topic']}" if e.get("topic") else "") +
            f" — {e['activity']}" + (f" ×{e['count']}" if e.get("count") else "")
            for e in events
        )
        await update.message.reply_text(f"Understood — logged {len(ids)} event(s) ✅\n{summary}")
    else:
        row_id = save_raw_message(user_id=user_id, raw_text=text)
        await update.message.reply_text(
            f"Couldn't fully parse that, but saved it raw (event #{row_id}) — I'll understand it later 📥"
        )

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /summary — Atlas reports the user's totals per subject."""
    user_id = str(update.effective_user.id)
    data = get_summary(user_id)
    if not data:
        await update.message.reply_text("No logs yet — send me what you studied and I'll start tracking! 📊")
        return

    lines = ["📊 *Your Atlas summary:*\n"]
    for d in data:
        line = f"• *{d['subject']}* — {_plural(d['events'], 'session')}"
        if d["items"]:
            line += f", {_plural(d['items'], 'item')}"
        if d["minutes"]:
            line += f", {d['minutes']} min"
        lines.append(line)
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


def main() -> None:
    """Build and run the bot (long polling)."""
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not found — check your .env file")

    app = (
        Application.builder()
        .token(TOKEN)
        .connect_timeout(20)
        .read_timeout(20)
        .write_timeout(20)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("summary", summary))
    logger.info("Atlas bot starting — polling for messages...")
    app.run_polling(bootstrap_retries=3)


if __name__ == "__main__":
    main()