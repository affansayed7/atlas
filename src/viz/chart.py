"""Visualization layer — renders charts as in-memory PNG images.

Design: charts render to a BytesIO buffer (RAM), never to disk —
no temp files, no cleanup, no race conditions. seek(0) rewinds the
buffer so the reader starts from the beginning.
"""

import io
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — no GUI, safe on a server
import matplotlib.pyplot as plt


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