#!/usr/bin/env python3
"""
06_fix_durations.py
===================
Reads 1_dataset_{split}.tsv, probes the actual duration of every video file
on disk via ffprobe, and writes corrected 2_dataset_{split}.tsv files.

The duration_sec column is overwritten with the ffprobe-measured value.
Rows whose file cannot be found or probed are skipped and reported.

Uses a thread pool for parallel ffprobe calls so this finishes quickly.
"""

import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# =============================================================================
DATA_DIR    = Path(r"C:\My Projects\sign-language-bridge\data")
IN_PATTERN  = "1_dataset_{split}.tsv"
OUT_PATTERN = "2_dataset_{split}.tsv"
SPLITS      = ["train", "val", "test"]
FFPROBE     = r"C:\ffmpeg\bin\ffprobe.exe"
WORKERS     = 16   # parallel ffprobe threads
OUT_COLUMNS = ["vid", "text", "duration_sec", "word_count", "file_path", "source"]
# =============================================================================


def probe_duration(path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "csv=p=0",
             str(path)],
            capture_output=True, text=True, timeout=10,
        )
        text = r.stdout.strip()
        return float(text) if text else None
    except Exception:
        return None


def probe_row(row: dict) -> tuple[dict, Optional[float], str]:
    """Returns (row, probed_duration, error_message)."""
    path = Path(row["file_path"])
    if not path.exists():
        return row, None, "file not found"
    dur = probe_duration(path)
    if dur is None:
        return row, None, "ffprobe failed"
    return row, dur, ""


def process_split(split: str) -> None:
    tsv_in  = DATA_DIR / IN_PATTERN.format(split=split)
    tsv_out = DATA_DIR / OUT_PATTERN.format(split=split)

    if not tsv_in.exists():
        print(f"  [WARN] Not found, skipping: {tsv_in}")
        return

    with open(tsv_in, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    n_total   = len(rows)
    n_skipped = 0
    results   = {}   # vid -> probed duration

    print(f"  [{split}] Probing {n_total:,} files with {WORKERS} threads ...")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(probe_row, row): row for row in rows}
        done = 0
        for fut in as_completed(futures):
            row, dur, err = fut.result()
            done += 1
            if done % 2000 == 0 or done == n_total:
                print(f"    {done:,}/{n_total:,} probed ...", end="\r")
            if err:
                n_skipped += 1
                print(f"\n  [SKIP] {row['vid']}: {err}")
            else:
                results[row["vid"]] = dur

    print()  # newline after progress

    out_rows = []
    for row in rows:
        vid = row["vid"]
        if vid not in results:
            continue
        out_rows.append({
            "vid":          vid,
            "text":         row["text"],
            "duration_sec": f"{results[vid]:.4f}",
            "word_count":   row["word_count"],
            "file_path":    row["file_path"],
            "source":       row["source"],
        })

    with open(tsv_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"  [{split}] Written: {len(out_rows):,} rows  |  Skipped: {n_skipped}  ->  {tsv_out}")


def main():
    print("=" * 66)
    print("  Fixing duration_sec from actual video files (ffprobe)")
    print(f"  Input  : {IN_PATTERN}")
    print(f"  Output : {OUT_PATTERN}")
    print("=" * 66)
    print()
    for split in SPLITS:
        process_split(split)
        print()
    print("Done.")


if __name__ == "__main__":
    main()
