"""Visualization layer — renders charts as in-memory PNG images.

Design: charts render to a BytesIO buffer (RAM), never to disk —
no temp files, no cleanup, no race conditions. seek(0) rewinds the
buffer so the reader starts from the beginning.
"""

import io
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — no GUI, safe on a server
import matplotlib.pyplot as plt


def make_activity_chart(daily_data: list[dict]) -> io.BytesIO:
    """Render a daily-activity bar chart (last N days) — dark theme."""
    dates = [d["date"][5:] for d in daily_data]
    counts = [d["count"] for d in daily_data]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#0d1117")   # GitHub-dark backdrop
    ax.set_facecolor("#0d1117")

    # Gradient-ish: zero days dim, active days bright cyan-green
    colors = ["#21262d" if c == 0 else "#2dd4bf" for c in counts]
    bars = ax.bar(dates, counts, color=colors, edgecolor="none", width=0.7)

    # Value labels on active bars
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(c), ha="center", va="bottom", color="#e6edf3",
                    fontsize=10, fontweight="bold")

    ax.set_title("Atlas — Daily Activity (last 14 days)", fontsize=15,
                 fontweight="bold", color="#e6edf3", pad=15)
    ax.set_ylabel("Events", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#21262d", linewidth=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def make_subject_chart(summary_data: list[dict]) -> io.BytesIO:
    """Render a bar chart of events-per-subject into an in-memory PNG buffer."""
    subjects = [d["subject"] for d in summary_data]
    counts = [d["events"] for d in summary_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(subjects, counts, color="#4C72B0")

    ax.set_title("Atlas — Sessions per Subject", fontsize=14, fontweight="bold")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Sessions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Label each bar with its value
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(count), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)          # free the figure's memory — critical in a long-running bot
    buf.seek(0)             # rewind so Telegram reads from byte 0
    return buf