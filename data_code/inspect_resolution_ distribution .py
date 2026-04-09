"""
Resolution distribution analysis across the full combined dataset.
Reads 2_dataset_train.tsv, 2_dataset_val.tsv, 2_dataset_test.tsv
Probes each video with ffprobe (parallelized) and plots resolution distributions.
Output: images/combined_resolution_distribution.png
"""

import os
import csv
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

TSV_FILES = [
    r"data/2_dataset_train.tsv",
    r"data/2_dataset_val.tsv",
    r"data/2_dataset_test.tsv",
]
OUTPUT_IMAGE = r"images/combined_resolution_distribution.png"
NUM_WORKERS = 18


def probe_video(row):
    """Returns (vid, source, width, height) or (vid, source, None, None) on failure."""
    vid, file_path, source = row["vid"], row["file_path"], row["source"]
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            w = streams[0].get("width")
            h = streams[0].get("height")
            return vid, source, w, h
    except Exception:
        pass
    return vid, source, None, None


def load_tsv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({"vid": row["vid"], "file_path": row["file_path"], "source": row["source"]})
    return rows


def main():
    all_rows = []
    for tsv in TSV_FILES:
        rows = load_tsv(tsv)
        print(f"Loaded {len(rows):,} rows from {tsv}")
        all_rows.extend(rows)

    print(f"\nTotal clips: {len(all_rows):,}. Probing resolutions with {NUM_WORKERS} workers...")

    results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(probe_video, row): row for row in all_rows}
        done = 0
        for future in as_completed(futures):
            vid, source, w, h = future.result()
            done += 1
            if done % 2000 == 0:
                print(f"  Probed {done:,} / {len(all_rows):,}...")
            if w is not None:
                results.append((source, w, h))
            else:
                errors += 1

    print(f"\nProbed OK: {len(results):,}  |  Errors: {errors}")

    how2sign = [(w, h) for s, w, h in results if s == "how2sign"]
    openasl  = [(w, h) for s, w, h in results if s == "openasl"]

    # ── Plot ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Resolution Distribution — Full Dataset (n={len(results):,})\n"
        f"How2Sign: {len(how2sign):,}  |  OpenASL: {len(openasl):,}",
        fontsize=14, fontweight="bold"
    )

    color_h = "#2196F3"
    color_o = "#FF5722"

    # 1. Scatter — width vs height
    ax = axes[0, 0]
    if how2sign:
        ws, hs = zip(*how2sign)
        ax.scatter(ws, hs, alpha=0.15, s=4, color=color_h, label=f"How2Sign ({len(how2sign):,})")
    if openasl:
        ws, hs = zip(*openasl)
        ax.scatter(ws, hs, alpha=0.15, s=4, color=color_o, label=f"OpenASL ({len(openasl):,})")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Width vs Height (scatter)")
    ax.legend(markerscale=4)
    ax.grid(alpha=0.3)

    # 2. Width histogram
    ax = axes[0, 1]
    all_widths_h = [w for w, h in how2sign]
    all_widths_o = [w for w, h in openasl]
    bins = np.linspace(0, max([w for s,w,h in results]) + 50, 60)
    ax.hist(all_widths_h, bins=bins, color=color_h, alpha=0.6, label="How2Sign")
    ax.hist(all_widths_o, bins=bins, color=color_o, alpha=0.6, label="OpenASL")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Count")
    ax.set_title("Width distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Height histogram
    ax = axes[1, 0]
    all_heights_h = [h for w, h in how2sign]
    all_heights_o = [h for w, h in openasl]
    bins = np.linspace(0, max([h for s,w,h in results]) + 50, 60)
    ax.hist(all_heights_h, bins=bins, color=color_h, alpha=0.6, label="How2Sign")
    ax.hist(all_heights_o, bins=bins, color=color_o, alpha=0.6, label="OpenASL")
    ax.set_xlabel("Height (px)")
    ax.set_ylabel("Count")
    ax.set_title("Height distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Top-20 most common resolutions (combined)
    ax = axes[1, 1]
    res_counter = Counter([(w, h) for s, w, h in results])
    top20 = res_counter.most_common(20)
    labels = [f"{w}×{h}" for (w, h), _ in top20]
    counts = [c for _, c in top20]
    # Color bars by source majority
    bar_colors = []
    for (w, h), _ in top20:
        n_h = sum(1 for s, bw, bh in results if bw == w and bh == h and s == "how2sign")
        n_o = sum(1 for s, bw, bh in results if bw == w and bh == h and s == "openasl")
        bar_colors.append(color_h if n_h >= n_o else color_o)
    bars = ax.barh(range(len(labels)), counts, color=bar_colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Top 20 most common resolutions")
    patch_h = mpatches.Patch(color=color_h, label="How2Sign dominant")
    patch_o = mpatches.Patch(color=color_o, label="OpenASL dominant")
    ax.legend(handles=[patch_h, patch_o], fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_IMAGE}")

    # ── Summary stats ────────────────────────────────────────────────────────
    all_w = [w for s, w, h in results]
    all_h = [h for s, w, h in results]
    print(f"\nCombined Width:  min={min(all_w)}  max={max(all_w)}  median={int(np.median(all_w))}")
    print(f"Combined Height: min={min(all_h)}  max={max(all_h)}  median={int(np.median(all_h))}")
    print(f"Unique resolutions: {len(res_counter):,}")
    print(f"\nTop 10:")
    for (w, h), c in res_counter.most_common(10):
        print(f"  {w}×{h}: {c:,}")


if __name__ == "__main__":
    main()