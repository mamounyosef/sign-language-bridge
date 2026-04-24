#!/usr/bin/env python3
"""
08_dataset_analytics.py
========================
Computes and saves analytics plots for the training split of the final dataset.
Input: C:\\My Projects\\sign-language-bridge\\data\\2_dataset_train.tsv

Plots saved to:
  C:\\My Projects\\sign-language-bridge\\data\\analytics\\

Figures produced:
  01_clip_duration_distribution.png
  02_word_count_distribution.png
  03_source_breakdown_pie.png
  04_duration_by_source.png
  05_words_per_second_distribution.png
  06_text_length_distribution.png
  07_word_count_vs_duration_scatter.png
  08_top50_words.png
  09_cumulative_hours_by_source.png
  analytics_summary.txt
"""

import csv
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

class _Tee:
    def __init__(self, filepath):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.__stdout__
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()


# =============================================================================
DATA_DIR     = Path(r"C:\My Projects\sign-language-bridge\data")
TSV_PATH     = DATA_DIR / "6_dataset_train.tsv"
OUT_DIR      = Path(r"C:\My Projects\sign-language-bridge\images")
# =============================================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "that", "this", "i", "you", "we", "they",
    "he", "she", "so", "are", "was", "be", "have", "not", "do", "as",
    "by", "from", "if", "just", "my", "your", "it's", "that's", "i'm",
    "don't", "about", "what", "there", "its", "s", "re", "ve", "ll",
}


def load_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def savefig(name: str) -> None:
    path = OUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_histogram(values, title, xlabel, filename, color, bins=50, xlim=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of clips", fontsize=12)
    if xlim:
        ax.set_xlim(xlim)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    ax.axvline(mean_val,   color="red",    linestyle="--", linewidth=1.2, label=f"Mean   {mean_val:.2f}")
    ax.axvline(median_val, color="orange", linestyle="--", linewidth=1.2, label=f"Median {median_val:.2f}")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    savefig(filename)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _log_path = Path(__file__).resolve().parent / "analytics_report.txt"
    _tee = _Tee(_log_path)
    sys.stdout = _tee
    _run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Dataset Analytics Report — generated {_run_ts}")
    print("=" * 60)

    print(f"Loading {TSV_PATH} ...")
    rows = load_tsv(TSV_PATH)
    print(f"  {len(rows):,} rows loaded.\n")

    # ------------------------------------------------------------------
    # Parse fields
    # ------------------------------------------------------------------
    durations   = []
    word_counts = []
    text_lengths= []
    wps_values  = []   # words per second
    sources     = []
    all_words   = []

    for row in rows:
        try:
            dur = float(row["duration_sec"])
        except (ValueError, KeyError):
            continue

        try:
            wc = int(float(row.get("word_count") or 0))
        except ValueError:
            wc = 0

        text = row.get("text", "").strip()
        src  = row.get("source", "unknown").strip().lower()

        durations.append(dur)
        word_counts.append(wc)
        text_lengths.append(len(text))
        sources.append(src)
        all_words.extend(tokenize(text))

        if dur > 0 and wc > 0:
            wps_values.append(wc / dur)

    durations    = np.array(durations,    dtype=float)
    word_counts  = np.array(word_counts,  dtype=int)
    text_lengths = np.array(text_lengths, dtype=int)
    wps_values   = np.array(wps_values,   dtype=float)

    source_counter = Counter(sources)

    # ------------------------------------------------------------------
    # 01 — Clip duration distribution
    # ------------------------------------------------------------------
    plot_histogram(
        durations, "Clip Duration Distribution (train)",
        "Duration (seconds)", "01_clip_duration_distribution.png",
        color="#4C9BE8", bins=60, xlim=(0, 9),
    )

    # ------------------------------------------------------------------
    # 02 — Word count distribution
    # ------------------------------------------------------------------
    plot_histogram(
        word_counts, "Word Count Distribution (train)",
        "Words per clip", "02_word_count_distribution.png",
        color="#57B55A", bins=range(0, max(word_counts) + 2),
    )

    # ------------------------------------------------------------------
    # 03 — Source breakdown pie
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    labels  = [f"{s}\n{c:,} clips" for s, c in source_counter.items()]
    sizes   = list(source_counter.values())
    colors  = ["#4C9BE8", "#F28E2B", "#76B7B2", "#E15759"][:len(sizes)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("Clips by Source (train)", fontsize=14, fontweight="bold")
    savefig("03_source_breakdown_pie.png")

    # ------------------------------------------------------------------
    # 04 — Duration distribution by source (overlapping histograms)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_map = {"how2sign": "#4C9BE8", "openasl": "#F28E2B"}
    for src, color in colors_map.items():
        mask = [s == src for s in sources]
        vals = durations[mask]
        if len(vals):
            ax.hist(vals, bins=60, alpha=0.6, color=color,
                    edgecolor="white", linewidth=0.3,
                    label=f"{src} ({len(vals):,})")
    ax.set_title("Clip Duration by Source (train)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Duration (seconds)", fontsize=12)
    ax.set_ylabel("Number of clips", fontsize=12)
    ax.set_xlim(0, 9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    savefig("04_duration_by_source.png")

    # ------------------------------------------------------------------
    # 05 — Words-per-second distribution
    # ------------------------------------------------------------------
    plot_histogram(
        wps_values, "Words-per-Second Distribution (train)",
        "Words per second", "05_words_per_second_distribution.png",
        color="#E15759", bins=50,
    )

    # ------------------------------------------------------------------
    # 06 — Text length (characters) distribution
    # ------------------------------------------------------------------
    plot_histogram(
        text_lengths, "Text Length Distribution (train)",
        "Characters per clip", "06_text_length_distribution.png",
        color="#9467BD", bins=60,
    )

    # ------------------------------------------------------------------
    # 07 — Word count vs duration scatter (sampled for readability)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    rng = np.random.default_rng(42)
    n_sample = min(5000, len(durations))
    idx = rng.choice(len(durations), n_sample, replace=False)
    ax.scatter(durations[idx], word_counts[idx],
               alpha=0.25, s=8, color="#4C9BE8", rasterized=True)
    ax.set_title(f"Word Count vs Duration — {n_sample:,} random clips (train)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Duration (seconds)", fontsize=12)
    ax.set_ylabel("Word count", fontsize=12)
    ax.set_xlim(0, 9)
    ax.grid(alpha=0.3)
    savefig("07_word_count_vs_duration_scatter.png")

    # ------------------------------------------------------------------
    # 08 — Top 50 most frequent content words
    # ------------------------------------------------------------------
    content_words = [w for w in all_words if w not in STOPWORDS and len(w) > 1]
    top50 = Counter(content_words).most_common(50)
    words_labels, counts = zip(*top50)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(50), counts, color="#4C9BE8", edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(50))
    ax.set_xticklabels(words_labels, rotation=60, ha="right", fontsize=8)
    ax.set_title("Top 50 Most Frequent Content Words (train)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)
    savefig("08_top50_words.png")

    # ------------------------------------------------------------------
    # 09 — Cumulative hours by source (sorted by duration)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for src, color in colors_map.items():
        mask = np.array([s == src for s in sources])
        vals = np.sort(durations[mask])
        if len(vals):
            cumulative_hrs = np.cumsum(vals) / 3600
            ax.plot(np.arange(1, len(vals) + 1), cumulative_hrs,
                    color=color, linewidth=1.5,
                    label=f"{src}  total={cumulative_hrs[-1]:.1f} h")
    ax.set_title("Cumulative Video Hours by Source (train)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of clips", fontsize=12)
    ax.set_ylabel("Cumulative hours", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    savefig("09_cumulative_hours_by_source.png")

    # ------------------------------------------------------------------
    # Summary text file
    # ------------------------------------------------------------------
    total_hours = durations.sum() / 3600
    vocab_size  = len(set(all_words))

    lines = [
        "TRAINING SPLIT ANALYTICS SUMMARY",
        "=" * 50,
        f"Total clips                : {len(rows):,}",
        f"Total video hours          : {total_hours:.2f} h",
        "",
        "By source:",
    ]
    for src, cnt in source_counter.items():
        mask = np.array([s == src for s in sources])
        hrs  = durations[mask].sum() / 3600
        lines.append(f"  {src:<12} : {cnt:>6,} clips  ({hrs:.2f} h)")

    lines += [
        "",
        "Clip duration (seconds):",
        f"  Min    : {durations.min():.4f}",
        f"  Max    : {durations.max():.4f}",
        f"  Mean   : {durations.mean():.4f}",
        f"  Median : {float(np.median(durations)):.4f}",
        f"  Std    : {durations.std():.4f}",
        "",
        "Word count per clip:",
        f"  Min    : {word_counts.min()}",
        f"  Max    : {word_counts.max()}",
        f"  Mean   : {word_counts.mean():.2f}",
        f"  Median : {float(np.median(word_counts)):.2f}",
        f"  Std    : {word_counts.std():.2f}",
        "",
        "Words per second:",
        f"  Mean   : {wps_values.mean():.2f}",
        f"  Median : {float(np.median(wps_values)):.2f}",
        f"  Std    : {wps_values.std():.2f}",
        "",
        f"Total vocabulary size      : {vocab_size:,}",
        f"Total word tokens          : {len(all_words):,}",
    ]

    summary_path = OUT_DIR / "analytics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved: {summary_path.name}")

    print()
    print("\n".join(lines))

    print()
    print("=" * 60)
    print(f"Report complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Saved to: {_log_path}")
    print("=" * 60)

    sys.stdout = sys.__stdout__
    _tee.close()


if __name__ == "__main__":
    main()
