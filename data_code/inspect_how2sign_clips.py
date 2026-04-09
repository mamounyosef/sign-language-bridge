#!/usr/bin/env python3
"""
inspect_how2sign_clips.py
=========================
Diagnostic script: match every How2Sign row in the TSV annotation files
against the corresponding raw clip on disk, then compare the clip's actual
duration against the TSV-annotated sentence duration (end - start).

What it checks
--------------
For each TSV row where source == 'how2sign':
  1. Does the corresponding clip file exist?
  2. What is the clip's actual duration (via ffprobe)?
  3. How does that compare to the annotated sentence duration?

     diff = clip_duration - tsv_duration
       diff ≈ 0      → clip is already trimmed to the sentence
       diff > 0      → clip has extra content (leading and/or trailing)
       diff < 0      → clip is shorter than the annotation (data issue)

Output
------
  - Summary printed to stdout
  - Detailed per-clip CSV written to:
      C:\My Projects\sign-language-bridge\data\how2sign_clip_audit.csv

Usage
-----
  python scripts/inspect_how2sign_clips.py
"""

import csv
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR        = Path(r"C:\My Projects\sign-language-bridge\data")
RAW_CLIPS_DIR   = DATA_DIR / "how2sign_raw_clips"
TSV_PATTERN     = "final_full_{split}_dataset.tsv"
SPLITS          = ["train", "val", "test"]
FFPROBE         = r"C:\ffmpeg\bin\ffprobe.exe"

# Tolerance for "already trimmed" (seconds).  Clips within ±TRIM_TOLERANCE
# of their annotated duration are considered already trimmed.
TRIM_TOLERANCE  = 2   # 1000 ms  (≈ 2 frames at 20 fps)

# Valid sentence duration range (seconds).
MIN_DURATION    = 1.3
MAX_DURATION    = 8.0

OUTPUT_CSV      = DATA_DIR / "how2sign_clip_audit.csv"

# ---------------------------------------------------------------------------
# ffprobe helper
# ---------------------------------------------------------------------------

def get_duration(path: Path) -> float | None:
    """Return the duration of a video file in seconds, or None on failure."""
    try:
        r = subprocess.run(
            [
                FFPROBE,
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Clip path resolver
# ---------------------------------------------------------------------------

def find_clip(vid: str) -> Path | None:
    """
    Search for {vid}.mp4 across all *_rgb_front_clips sub-directories.
    Returns the first match, or None if not found.
    """
    for subdir in RAW_CLIPS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / f"{vid}.mp4"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows_all: list[dict] = []

    for split in SPLITS:
        tsv_path = DATA_DIR / TSV_PATTERN.format(split=split)
        if not tsv_path.exists():
            print(f"[WARN] TSV not found, skipping: {tsv_path}")
            continue
        with open(tsv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if row.get("source", "").strip().lower() == "how2sign":
                    rows_all.append({**row, "_split": split})

    total = len(rows_all)
    print(f"\nHow2Sign rows in TSV files: {total:,}")
    print(f"Searching for clips in: {RAW_CLIPS_DIR}\n")

    # Counters
    n_missing          = 0
    n_trimmed          = 0   # diff within TRIM_TOLERANCE
    n_needs_trim       = 0   # clip is longer by > TRIM_TOLERANCE
    n_too_short        = 0   # clip is shorter than annotated duration
    n_ffprobe_err      = 0
    n_tsv_out_of_range = 0   # TSV sentence duration outside [MIN_DURATION, MAX_DURATION]
    n_clip_out_of_range= 0   # actual clip duration outside [MIN_DURATION, MAX_DURATION]

    audit_rows: list[dict] = []

    for i, row in enumerate(rows_all, 1):
        vid   = row["vid"].strip()
        split = row["_split"]

        try:
            tsv_start = float(row["start"])
            tsv_end   = float(row["end"])
            tsv_dur   = tsv_end - tsv_start
        except (ValueError, KeyError):
            tsv_dur = None

        # Check TSV-annotated sentence duration against allowed range
        tsv_in_range = (
            tsv_dur is not None
            and MIN_DURATION <= tsv_dur <= MAX_DURATION
        )
        if tsv_dur is not None and not tsv_in_range:
            n_tsv_out_of_range += 1

        clip_path = find_clip(vid)
        if clip_path is None:
            n_missing += 1
            status = "MISSING"
            clip_dur = None
            diff = None
        else:
            clip_dur = get_duration(clip_path)
            if clip_dur is None:
                n_ffprobe_err += 1
                status = "FFPROBE_ERROR"
                diff = None
            elif tsv_dur is None:
                status = "BAD_TSV_TIMING"
                diff = None
            else:
                diff = clip_dur - tsv_dur
                if abs(diff) <= TRIM_TOLERANCE:
                    n_trimmed += 1
                    status = "ALREADY_TRIMMED"
                elif diff > TRIM_TOLERANCE:
                    n_needs_trim += 1
                    status = "NEEDS_TRIM"
                else:
                    n_too_short += 1
                    status = "CLIP_TOO_SHORT"

        # Check actual clip duration against allowed range (if available)
        clip_in_range = (
            clip_dur is not None
            and MIN_DURATION <= clip_dur <= MAX_DURATION
        )
        if clip_dur is not None and not clip_in_range:
            n_clip_out_of_range += 1

        audit_rows.append({
            "split":             split,
            "vid":               vid,
            "tsv_start":         row.get("start", ""),
            "tsv_end":           row.get("end", ""),
            "tsv_duration":      f"{tsv_dur:.4f}" if tsv_dur is not None else "",
            "tsv_in_range":      ("YES" if tsv_in_range else "NO") if tsv_dur is not None else "",
            "clip_path":         str(clip_path) if clip_path else "",
            "clip_duration":     f"{clip_dur:.4f}" if clip_dur is not None else "",
            "clip_in_range":     ("YES" if clip_in_range else "NO") if clip_dur is not None else "",
            "diff_sec":          f"{diff:.4f}" if diff is not None else "",
            "status":            status,
        })

        # Progress every 500 rows
        if i % 500 == 0 or i == total:
            pct = 100 * i / total
            print(f"  [{i:>6,}/{total:,}]  {pct:5.1f}%  "
                  f"missing={n_missing}  trimmed={n_trimmed}  "
                  f"needs_trim={n_needs_trim}  too_short={n_too_short}", end="\r")

    print()  # newline after progress line

    # Write CSV
    fieldnames = [
        "split", "vid", "tsv_start", "tsv_end", "tsv_duration", "tsv_in_range",
        "clip_path", "clip_duration", "clip_in_range", "diff_sec", "status",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    # Print summary
    n_found = total - n_missing
    print(f"""
=============================================================
  HOW2SIGN CLIP AUDIT — SUMMARY
=============================================================
  Valid duration range             : {MIN_DURATION}s – {MAX_DURATION}s

  Total TSV rows (how2sign)        : {total:>8,}

  Clip file found on disk          : {n_found:>8,}
  Clip file MISSING                : {n_missing:>8,}

  TSV annotation duration checks:
    Within [{MIN_DURATION}s, {MAX_DURATION}s]          : {total - n_tsv_out_of_range:>8,}
    OUT OF RANGE                     : {n_tsv_out_of_range:>8,}

  Actual clip duration checks (found clips):
    Within [{MIN_DURATION}s, {MAX_DURATION}s]          : {n_found - n_clip_out_of_range - n_missing:>8,}
    OUT OF RANGE                     : {n_clip_out_of_range:>8,}

  Trim status (found clips):
    Already trimmed  (diff ≤ {TRIM_TOLERANCE*1000:.0f} ms)  : {n_trimmed:>8,}
    Needs trimming   (clip longer)   : {n_needs_trim:>8,}
    Clip too short   (clip shorter)  : {n_too_short:>8,}
    ffprobe error                    : {n_ffprobe_err:>8,}
=============================================================
  Full details written to:
    {OUTPUT_CSV}
=============================================================
""")

    # Show a few examples from each status / out-of-range category
    for status_label in ["ALREADY_TRIMMED", "NEEDS_TRIM", "CLIP_TOO_SHORT", "MISSING"]:
        examples = [r for r in audit_rows if r["status"] == status_label][:3]
        if not examples:
            continue
        print(f"  Examples — {status_label}:")
        for ex in examples:
            print(f"    vid={ex['vid']!r}  tsv_dur={ex['tsv_duration']}s "
                  f"(in_range={ex['tsv_in_range']})  "
                  f"clip_dur={ex['clip_duration']}s "
                  f"(in_range={ex['clip_in_range']})  diff={ex['diff_sec']}s")
        print()

    # Out-of-range examples
    oor_tsv = [r for r in audit_rows if r["tsv_in_range"] == "NO"][:5]
    if oor_tsv:
        print(f"  Examples — TSV duration OUT OF RANGE [{MIN_DURATION}s, {MAX_DURATION}s]:")
        for ex in oor_tsv:
            print(f"    vid={ex['vid']!r}  tsv_dur={ex['tsv_duration']}s")
        print()

    oor_clip = [r for r in audit_rows if r["clip_in_range"] == "NO"][:5]
    if oor_clip:
        print(f"  Examples — clip duration OUT OF RANGE [{MIN_DURATION}s, {MAX_DURATION}s]:")
        for ex in oor_clip:
            print(f"    vid={ex['vid']!r}  clip_dur={ex['clip_duration']}s")
        print()


if __name__ == "__main__":
    main()
