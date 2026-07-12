"""Atlas Telegram bot — entry point.

v0.1 scope: receive messages, reply, log raw text to the events table.
"""

import logging
import os
from src.db.repository import get_today_events, get_summary, save_raw_message, save_parsed_events
from src.ingestion.parser import parse_message
from src.viz.chart import make_subject_chart
from datetime import datetime, timedelta, timezone
from src.db.repository import get_active_dates
from src.analytics.streak import calculate_streak
from src.ingestion.leetcode import poll_and_log
from src.db.repository import get_daily_counts
from src.viz.chart import make_activity_chart
from src.analytics.mastery import estimate_mastery


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


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /today — what the user logged today (IST)."""
    user_id = str(update.effective_user.id)
    events = get_today_events(user_id)
    if not events:
        await update.message.reply_text("Nothing logged today yet — the day's young! 📅")
        return

    lines = [f"📅 *Today's log* ({len(events)} events):\n"]
    for e in events:
        line = f"• *{e['subject']}*"
        if e["topic"]:
            line += f"/{e['topic']}"
        line += f" — {e['activity']}"
        if e["count"]:
            line += f" ×{e['count']}"
        lines.append(line)
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /chart — sends a bar chart of sessions per subject."""
    user_id = str(update.effective_user.id)
    data = get_summary(user_id)
    if not data:
        await update.message.reply_text("No data to chart yet — log something first! 📊")
        return

    buf = make_subject_chart(data)
    await update.message.reply_photo(photo=buf, caption="📊 Your sessions per subject")


async def streak(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /streak — consecutive logging days (IST)."""
    user_id = str(update.effective_user.id)
    active_dates = get_active_dates(user_id)

    # 'Today' in IST = UTC + 5:30
    ist_today = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).date()
    days = calculate_streak(active_dates, ist_today)

    if days == 0:
        await update.message.reply_text("No streak yet — log something today to start one! 🔥")
    else:
        await update.message.reply_text(
            f"🔥 *{days}-day streak!* Keep it going.", parse_mode="Markdown"
        )


async def sync(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /sync — poll LeetCode on demand and report new solves."""
    user_id = str(update.effective_user.id)
    await update.message.reply_text("Syncing with LeetCode... 🔄")

    # poll_and_log uses blocking `requests` + sqlite. Running it directly would
    # freeze the bot's event loop. asyncio.to_thread offloads it to a worker
    # thread so the bot stays responsive.
    import asyncio
    LEETCODE_USERNAME = "NEOWMEOW"  # TODO: make per-user configurable later
    new_solves = await asyncio.to_thread(poll_and_log, user_id, LEETCODE_USERNAME)

    if new_solves:
        detail = ", ".join(f"{n} {d}" for d, n in new_solves.items())
        await update.message.reply_text(f"🎉 Logged new solves: {detail}")
    else:
        await update.message.reply_text("All caught up — no new solves since last sync ✅")

async def activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /activity — daily activity bar chart, last 14 days."""
    user_id = str(update.effective_user.id)
    data = get_daily_counts(user_id, days=14)
    if not any(d["count"] for d in data):
        await update.effective_message.reply_text("No activity in the last 14 days — log something! 📊")
        return
    buf = make_activity_chart(data)
    await update.effective_message.reply_photo(photo=buf, caption="📈 Your daily activity (last 14 days)")


async def mastery(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /mastery — BKT-estimated mastery per subject."""
    user_id = str(update.effective_user.id)
    data = estimate_mastery(user_id)
    if not data:
        await update.effective_message.reply_text(
            "Not enough graded attempts yet. Log solves with outcomes "
            "(e.g. 'solved 3 dp, 1 struggled') and I'll estimate mastery. 🧠"
        )
        return

    lines = ["🧠 *Estimated mastery* (BKT):\n"]
    for d in data:
        bar = "█" * round(d["mastery"] * 10) + "░" * (10 - round(d["mastery"] * 10))
        lines.append(f"*{d['subject']}*  {bar}  {d['mastery']:.0%}  _({d['attempts']} attempts)_")
    lines.append("\n_Estimates are rough — they sharpen with more logged attempts._")
    await update.effective_message.reply_text("\n".join(lines), parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /help — lists everything Atlas can do."""
    text = (
        "🗺️ *Atlas — your learning coach*\n\n"
        "*Logging:*\n"
        "Just message me naturally — e.g. _\"solved 3 dp mediums, 1 struggled\"_ "
        "or _\"gym 1hr push day\"_. I'll parse and remember it.\n\n"
        "*Commands:*\n"
        "/summary — totals per subject\n"
        "/today — what you logged today\n"
        "/streak — your current logging streak 🔥\n"
        "/chart — sessions-per-subject bar chart 📊\n"
        "/activity — daily activity, last 14 days 📈\n"
        "/mastery — estimated skill mastery (BKT) 🧠\n"
        "/sync — pull latest LeetCode solves 🔄\n"
        "/help — this message"
    )
    await update.effective_message.reply_text(text, parse_mode="Markdown")


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
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("chart", chart))
    app.add_handler(CommandHandler("streak", streak))
    app.add_handler(CommandHandler("sync", sync))
    app.add_handler(CommandHandler("activity", activity))
    app.add_handler(CommandHandler("mastery", mastery))
    app.add_handler(CommandHandler("help", help_command))
    logger.info("Atlas bot starting — polling for messages...")
    app.run_polling(bootstrap_retries=3)


if __name__ == "__main__":
    main()