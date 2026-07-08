"""Atlas Telegram bot — entry point.

v0.1 scope: receive messages, reply, log raw text to the events table.
"""

import logging
import os
from src.db.repository import save_raw_message, count_events, save_parsed_events
from src.ingestion.parser import parse_message

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from src.db.repository import save_raw_message, count_events


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

    logger.info("Atlas bot starting — polling for messages...")
    app.run_polling(bootstrap_retries=3)


if __name__ == "__main__":
    main()