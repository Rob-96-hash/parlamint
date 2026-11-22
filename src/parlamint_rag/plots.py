"""Figures for the README, made from the evaluation result files (no model needed).

Run with:  python -m parlamint_rag.plots
Writes docs/recall_at_k.png and docs/gate_scores.png.
"""

import matplotlib

matplotlib.use("Agg")  # render to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from parlamint_rag import config

DOCS = config.ROOT / "docs"
CSV = {"sep": ";", "encoding": "utf-8-sig"}

# Two well-separated steps of the viridis colour map (also distinguishable with colour
# vision deficiency); every series is labelled as well, so colour is never the only cue
PURPLE = "#414487"   # viridis 0.20
GREEN = "#22a884"    # viridis 0.60
INK = "#333333"      # text
MUTED = "#767676"    # secondary text, reference lines
GRID = "#e6e6e6"


def style_axes(ax) -> None:
    """Quiet axes: no top/right frame, light horizontal grid, grey ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def recall_curve(path) -> np.ndarray:
    """Recall@k for k = 1..10 from a retrieval CSV (rank of the first gold speech, empty = miss)."""
    ranks = pd.read_csv(path, **CSV)["rank"]
    # A missing rank (not in the top 10) compares as False, so it counts as a miss
    return np.array([(ranks <= k).mean() for k in range(1, 11)])


def plot_recall_at_k() -> None:
    """Recall@k curves for vector search alone and with the cross-encoder reranker."""
    ks = np.arange(1, 11)
    curves = {
        "Vector search only": (recall_curve(config.RESULTS_DIR / "retrieval.csv"), PURPLE),
        "+ cross-encoder reranker": (recall_curve(config.RESULTS_DIR / "retrieval_rerank.csv"), GREEN),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    style_axes(ax)

    # The language model only sees the top 5 chunks, so k <= 5 is what matters for the answer
    ax.axvline(5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(5.12, 0.3, "the LLM sees\nthe top 5", color=MUTED, fontsize=9)

    for label, (values, colour) in curves.items():
        ax.plot(ks, values, color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, label=label)
        # Direct label at the end of the line, in text colour (the line carries the colour)
        ax.annotate(f"{label}  {values[-1]:.2f}", (ks[-1], values[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK)

    ax.set_xticks(ks)
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("k (number of retrieved chunks)", color=INK, fontsize=10)
    ax.set_ylabel("Recall@k", color=INK, fontsize=10)
    ax.set_title("Share of the 20 answerable questions with a gold speech in the top k",
                 loc="left", color=INK, fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "recall_at_k.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gate_scores() -> None:
    """Best reranker score per question: answerable vs unanswerable, correct vs wrong, threshold."""
    scores = pd.read_csv(config.RESULTS_DIR / "top_scores_rerank.csv", **CSV)
    answers = pd.read_csv(config.RESULTS_DIR / "answers_qwen3-1.7b_rerank.csv", **CSV)
    df = scores.merge(answers[["id", "correct"]], on="id")
    df["answerable"] = df["type"] != "unanswerable"

    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=160)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)

    rows = {True: (1.0, PURPLE), False: (0.0, GREEN)}
    df["y"] = 0.0
    for answerable, (y, colour) in rows.items():
        group = df[df["answerable"] == answerable].sort_values("top_score")
        # Small alternating vertical offsets so that close scores do not hide each other
        offsets = np.tile([0.0, 0.12, -0.12], len(group))[: len(group)]
        df.loc[group.index, "y"] = y + offsets
        filled = group["correct"].astype(int) == 1
        # Filled marker = correct answer, hollow marker = wrong answer
        ax.scatter(group["top_score"], y + offsets, s=70, linewidths=1.6, edgecolors=colour,
                   facecolors=np.where(filled, colour, "white"), zorder=3)

    # The relevance gate: left of the line the LLM is not asked
    ax.axvline(0.0, color=INK, linewidth=1.2)
    ax.text(0.12, 1.58, "threshold 0.0 (left of it: no-answer sentence, LLM not asked)",
            ha="left", va="top", color=INK, fontsize=9)

    # Name the few points the text discusses: (label, text offset in points, alignment)
    notes = {"q19": ("speaker question,\ngold never retrieved", (-6, -34), "left"),
             "q21": ("World Cup 2014", (0, -22), "center"),
             "q25": ("AI rules 2023 (hard negative)", (12, -3), "left")}
    for qid, (text, offset, align) in notes.items():
        row = df[df["id"] == qid].iloc[0]
        ax.annotate(text, (row["top_score"], row["y"]), xytext=offset, textcoords="offset points",
                    fontsize=8.5, color=MUTED, ha=align, va="center")

    ax.set_yticks([0, 1], ["unanswerable\n(n = 5)", "answerable\n(n = 20)"], color=INK)
    ax.set_ylim(-0.6, 1.62)
    ax.set_xlabel("best cross-encoder score of the question", color=INK, fontsize=10)
    ax.set_title("Where the relevance gate cuts: best reranker score per question",
                 loc="left", color=INK, fontsize=11)
    legend = [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=MUTED, label="answer correct"),
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white",
               markeredgecolor=MUTED, markeredgewidth=1.6, label="answer wrong"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "gate_scores.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Write both README figures to docs/."""
    DOCS.mkdir(exist_ok=True)
    plot_recall_at_k()
    plot_gate_scores()
    print(f"[plots] {DOCS / 'recall_at_k.png'}\n[plots] {DOCS / 'gate_scores.png'}")


if __name__ == "__main__":
    main()
