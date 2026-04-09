#!/usr/bin/env python3
"""
How2Sign Filtered Dataset Inspector
=====================================
Reads data\\final_full_{split}_dataset.tsv for each split, keeps only rows
where source == 'how2sign', then runs ffprobe on every clip listed in the
file_path column.

Reports per-split AND combined statistics on:

  VIDEO
  • FPS                      (min / max / mean / median / std / percentiles)
  • Width  (px)              (min / max / mean / median / std)
  • Height (px)              (min / max / mean / median / std)
  • Unique resolutions       (ranked bar chart)
  • Aspect ratios            (ranked bar chart)
  • Video codec

  TIMING
  • Actual duration (s)      (from ffprobe — what the file really contains)
  • Annotated duration (s)   (end - start from TSV)
  • Δ duration (actual - annotated)   (should be ~0; flags bad trims)
  • Total footage hours

  TRANSCRIPT
  • Word count per clip      (min / max / mean / median / std / percentiles)
  • Words-per-second         (word_count / annotated_duration)

  FILE
  • File size (MB)           (min / max / mean / total GB)
  • Bit-rate (kbps)          (min / max / mean)

  AUDIO
  • Audio codec
  • Sample rate (Hz)
  • Channels

  ERRORS
  • TSV rows where the .mp4 file does not exist on disk
  • ffprobe failures

Everything is printed to the terminal AND saved to:
  data\\inspect_how2sign_report.txt

Usage
-----
  python inspect_clips.py                   # all three splits
  python inspect_clips.py --splits train    # single split
  python inspect_clips.py --workers 8       # more parallel probing
  python inspect_clips.py --limit 200       # quick sanity-check

Requirements
------------
  Python 3.9+   (no extra packages needed)
  FFmpeg / ffprobe on PATH:   winget install Gyan.FFmpeg
"""

import csv
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  ▶  CONFIGURATION  — only these lines should ever need editing
# ══════════════════════════════════════════════════════════════════════════════

# Directory that contains the TSV files
DATA_DIR    = Path(r"C:\My Projects\sign-language-bridge\data")

# TSV filename pattern  — {split} → train / val / test
TSV_PATTERN = "2_dataset_{split}.tsv"

# Where to write the report and progress log
REPORT_PATH = DATA_DIR / "inspect_how2sign_report.txt"

# ffprobe binary (leave as "ffprobe" if it is on PATH)
FFPROBE     = r"C:\ffmpeg\bin\ffprobe.exe"

# Splits to process
ALL_SPLITS  = ["train", "val", "test"]

# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TsvRow:
    """One how2sign row as read from the TSV."""
    vid:             str
    text:            str
    start:           float
    end:             float
    annotated_dur:   float   # end - start  (from TSV duration_sec column)
    word_count:      int
    file_path:       Path
    split:           str


@dataclass
class ClipInfo:
    """ffprobe result + TSV metadata for one clip."""
    tsv:             TsvRow
    # video (from ffprobe)
    width:           Optional[int]   = None
    height:          Optional[int]   = None
    fps:             Optional[float] = None
    actual_dur:      Optional[float] = None
    video_codec:     Optional[str]   = None
    bit_rate_kbps:   Optional[float] = None
    # audio
    audio_codec:     Optional[str]   = None
    sample_rate_hz:  Optional[int]   = None
    audio_channels:  Optional[int]   = None
    # file
    file_size_mb:    Optional[float] = None
    # status
    file_missing:    bool            = False
    error:           Optional[str]   = None


# ─────────────────────────────────────────────────────────────────────────────
#  TSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_tsv(split: str) -> Tuple[List[TsvRow], int, int]:
    """
    Read the TSV for one split, return:
      (how2sign_rows, total_rows_in_file, how2sign_count)
    Skips rows where source != 'how2sign'.
    """
    path = DATA_DIR / TSV_PATTERN.format(split=split)
    if not path.exists():
        return [], 0, 0

    rows: List[TsvRow] = []
    total = 0

    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for raw in reader:
            total += 1
            if raw.get("source", "").strip().lower() != "how2sign":
                continue

            try:
                ann_dur = float(raw.get("duration_sec") or 0)
                wc      = int(float(raw.get("word_count") or 0))
            except (ValueError, KeyError):
                ann_dur, wc = 0.0, 0

            rows.append(TsvRow(
                vid           = raw.get("vid", "").strip(),
                text          = raw.get("text", "").strip(),
                start         = 0.0,
                end           = ann_dur,
                annotated_dur = ann_dur,
                word_count    = wc,
                file_path     = Path(raw.get("file_path", "").strip()),
                split         = split,
            ))

    return rows, total, len(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  ffprobe worker  (must be top-level for Windows multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def probe_clip(tsv_row: TsvRow) -> ClipInfo:
    """Run ffprobe on one file and return a ClipInfo."""
    info = ClipInfo(tsv=tsv_row)

    if not tsv_row.file_path.exists():
        info.file_missing = True
        info.error = f"file not found on disk: {tsv_row.file_path}"
        return info

    try:
        info.file_size_mb = tsv_row.file_path.stat().st_size / (1024 ** 2)

        result = subprocess.run(
            [
                FFPROBE,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(tsv_row.file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            info.error = f"ffprobe exit {result.returncode}: {result.stderr[-300:]}"
            return info

        data = json.loads(result.stdout)

        fmt = data.get("format", {})
        if "duration" in fmt:
            info.actual_dur = float(fmt["duration"])
        if "bit_rate" in fmt:
            info.bit_rate_kbps = int(fmt["bit_rate"]) / 1000

        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type", "")

            if codec_type == "video" and info.width is None:
                info.width       = stream.get("width")
                info.height      = stream.get("height")
                info.video_codec = stream.get("codec_name")

                if info.actual_dur is None and "duration" in stream:
                    info.actual_dur = float(stream["duration"])

                for key in ("avg_frame_rate", "r_frame_rate"):
                    raw = stream.get(key, "")
                    if raw and raw != "0/0":
                        try:
                            num, den = raw.split("/")
                            if int(den) != 0:
                                info.fps = round(int(num) / int(den), 4)
                                break
                        except (ValueError, ZeroDivisionError):
                            pass

            elif codec_type == "audio" and info.audio_codec is None:
                info.audio_codec    = stream.get("codec_name")
                sr = stream.get("sample_rate")
                info.sample_rate_hz = int(sr) if sr else None
                ch = stream.get("channels")
                info.audio_channels = int(ch) if ch else None

    except subprocess.TimeoutExpired:
        info.error = "ffprobe timed out (>30 s)"
    except json.JSONDecodeError as exc:
        info.error = f"JSON parse error: {exc}"
    except Exception as exc:
        info.error = f"{type(exc).__name__}: {exc}"

    return info


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_stats(values: List[float], label: str, unit: str = "",
               decimals: int = 2) -> List[str]:
    if not values:
        return [f"  {label:<30} no data"]
    mn   = min(values)
    mx   = max(values)
    mean = statistics.mean(values)
    med  = statistics.median(values)
    std  = statistics.stdev(values) if len(values) > 1 else 0.0
    fmt  = f"{{:.{decimals}f}}"
    u    = f" {unit}" if unit else ""
    return [
        f"  {label:<30}",
        f"    min    = {fmt.format(mn)}{u}",
        f"    max    = {fmt.format(mx)}{u}",
        f"    mean   = {fmt.format(mean)}{u}",
        f"    median = {fmt.format(med)}{u}",
        f"    std    = {fmt.format(std)}{u}",
        f"    count  = {len(values):,}",
    ]


def pct_lines(values: List[float], unit: str = "", decimals: int = 2) -> List[str]:
    if len(values) < 4:
        return []
    sv  = sorted(values)
    n   = len(sv)
    fmt = f"{{:.{decimals}f}}"
    u   = f" {unit}" if unit else ""

    def pct(p):
        idx = (p / 100) * (n - 1)
        lo  = int(idx)
        hi  = min(lo + 1, n - 1)
        return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])

    return [
        f"    P1     = {fmt.format(pct(1))}{u}",
        f"    P5     = {fmt.format(pct(5))}{u}",
        f"    P25    = {fmt.format(pct(25))}{u}",
        f"    P75    = {fmt.format(pct(75))}{u}",
        f"    P95    = {fmt.format(pct(95))}{u}",
        f"    P99    = {fmt.format(pct(99))}{u}",
    ]


def top_counter(counter: Counter, n: int = 15, label: str = "") -> List[str]:
    if not counter:
        return [f"  {label}  (no data)"] if label else []
    lines = []
    if label:
        lines.append(f"  {label}")
    max_v = max(counter.values())
    for val, cnt in counter.most_common(n):
        bar = "█" * max(1, int(cnt / max_v * 30))
        lines.append(f"    {str(val):<28} {cnt:>8,}  {bar}")
    if len(counter) > n:
        lines.append(f"    … and {len(counter) - n} more unique values")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
#  Report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(clips: List[ClipInfo], label: str) -> List[str]:
    SEP  = "─" * 72
    SEP2 = "═" * 72

    missing = [c for c in clips if c.file_missing]
    errors  = [c for c in clips if c.error and not c.file_missing]
    ok      = [c for c in clips if c.error is None]

    lines: List[str] = [
        SEP2,
        f"  {label}  ({len(clips):,} clips from TSV)",
        SEP2,
        f"  Probed successfully : {len(ok):,}",
        f"  File not on disk    : {len(missing):,}",
        f"  ffprobe errors      : {len(errors):,}",
        "",
    ]

    if not ok:
        lines += ["  (no valid clips to analyse)", ""]
        return lines

    # ── Collect fields ────────────────────────────────────────────────────
    fpss         = [c.fps            for c in ok if c.fps            is not None]
    widths       = [c.width          for c in ok if c.width          is not None]
    heights      = [c.height         for c in ok if c.height         is not None]
    actual_durs  = [c.actual_dur     for c in ok if c.actual_dur     is not None]
    ann_durs     = [c.tsv.annotated_dur for c in ok]
    word_counts  = [float(c.tsv.word_count) for c in ok]
    sizes        = [c.file_size_mb   for c in ok if c.file_size_mb   is not None]
    bitrates     = [c.bit_rate_kbps  for c in ok if c.bit_rate_kbps  is not None]

    # Duration delta: actual vs annotated (flags bad trims)
    dur_deltas = [
        c.actual_dur - c.tsv.annotated_dur
        for c in ok
        if c.actual_dur is not None
    ]

    # Words per second (annotated duration as the reference)
    wps = [
        c.tsv.word_count / c.tsv.annotated_dur
        for c in ok
        if c.tsv.annotated_dur > 0
    ]

    resolutions = Counter(
        f"{c.width}×{c.height}" for c in ok
        if c.width and c.height
    )
    aspect_ratios = Counter()
    for c in ok:
        if c.width and c.height:
            g = math.gcd(c.width, c.height)
            aspect_ratios[f"{c.width//g}:{c.height//g}"] += 1

    video_codecs = Counter(c.video_codec   for c in ok if c.video_codec)
    audio_codecs = Counter(c.audio_codec   for c in ok if c.audio_codec)
    sample_rates = Counter(c.sample_rate_hz for c in ok if c.sample_rate_hz)
    channels     = Counter(c.audio_channels  for c in ok if c.audio_channels)

    # ── VIDEO ─────────────────────────────────────────────────────────────
    lines += [SEP, "  FRAME RATE (FPS)", SEP]
    lines += safe_stats(fpss, "FPS", unit="fps", decimals=4)
    lines += pct_lines(fpss, unit="fps", decimals=4)
    lines.append("")
    lines += top_counter(Counter(round(f, 3) for f in fpss),
                         n=10, label="Unique FPS values (top 10):")
    lines.append("")

    lines += [SEP, "  RESOLUTION", SEP]
    lines += safe_stats([float(w) for w in widths],  "Width",  unit="px", decimals=0)
    lines.append("")
    lines += safe_stats([float(h) for h in heights], "Height", unit="px", decimals=0)
    lines.append("")
    lines += top_counter(resolutions,   n=15, label="Unique resolutions (top 15):")
    lines.append("")
    lines += top_counter(aspect_ratios, n=10, label="Aspect ratios (top 10):")
    lines.append("")

    lines += [SEP, "  VIDEO CODEC", SEP]
    lines += top_counter(video_codecs, n=5, label="Video codec:")
    lines.append("")

    # ── TIMING ────────────────────────────────────────────────────────────
    lines += [SEP, "  DURATION — ACTUAL  (from ffprobe)", SEP]
    lines += safe_stats(actual_durs, "Actual duration", unit="s", decimals=3)
    lines += pct_lines(actual_durs, unit="s", decimals=3)
    if actual_durs:
        total_h = sum(actual_durs) / 3600
        lines.append(
            f"    Total footage  = {total_h:.3f} h"
            f"  ({sum(actual_durs)/60:.1f} min  /  {sum(actual_durs):.0f} s)"
        )
    lines.append("")

    lines += [SEP, "  DURATION — ANNOTATED  (end - start from TSV)", SEP]
    lines += safe_stats(ann_durs, "Annotated duration", unit="s", decimals=3)
    lines += pct_lines(ann_durs, unit="s", decimals=3)
    if ann_durs:
        total_h_ann = sum(ann_durs) / 3600
        lines.append(
            f"    Total annotated = {total_h_ann:.3f} h"
            f"  ({sum(ann_durs)/60:.1f} min  /  {sum(ann_durs):.0f} s)"
        )
    lines.append("")

    lines += [SEP, "  DURATION DELTA  (actual − annotated)", SEP]
    lines += [
        "  A delta near 0 is correct.  Large values flag",
        "  clips that were trimmed incorrectly or not at all.",
        "",
    ]
    lines += safe_stats(dur_deltas, "Δ duration", unit="s", decimals=3)
    lines += pct_lines(dur_deltas, unit="s", decimals=3)
    if dur_deltas:
        n_large = sum(1 for d in dur_deltas if abs(d) > 1.0)
        lines.append(
            f"    Clips with |Δ| > 1 s : {n_large:,}  "
            f"({100*n_large/len(dur_deltas):.1f} %)"
        )
    lines.append("")

    # ── TRANSCRIPT ────────────────────────────────────────────────────────
    lines += [SEP, "  TRANSCRIPT — WORD COUNT", SEP]
    lines += safe_stats(word_counts, "Words per clip", decimals=1)
    lines += pct_lines(word_counts, decimals=1)
    lines.append("")

    lines += [SEP, "  TRANSCRIPT — WORDS PER SECOND", SEP]
    lines += safe_stats(wps, "Words / second", unit="w/s", decimals=3)
    lines += pct_lines(wps, unit="w/s", decimals=3)
    lines.append("")

    # ── FILE SIZE ─────────────────────────────────────────────────────────
    lines += [SEP, "  FILE SIZE", SEP]
    lines += safe_stats(sizes, "File size", unit="MB", decimals=2)
    lines += pct_lines(sizes, unit="MB", decimals=2)
    if sizes:
        total_gb = sum(sizes) / 1024
        lines.append(
            f"    Total on disk  = {total_gb:.3f} GB  ({sum(sizes):.1f} MB)"
        )
    lines.append("")

    # ── BIT-RATE ──────────────────────────────────────────────────────────
    lines += [SEP, "  BIT-RATE", SEP]
    lines += safe_stats(bitrates, "Bit-rate", unit="kbps", decimals=1)
    lines.append("")

    # ── AUDIO ─────────────────────────────────────────────────────────────
    lines += [SEP, "  AUDIO", SEP]
    lines += top_counter(audio_codecs,  n=5, label="Audio codec:")
    lines.append("")
    lines += top_counter(sample_rates,  n=5, label="Sample rate (Hz):")
    lines.append("")
    lines += top_counter(channels,       n=5, label="Channels:")
    lines.append("")

    # ── MISSING FILES ─────────────────────────────────────────────────────
    if missing:
        lines += [SEP, f"  FILES NOT FOUND ON DISK  ({len(missing):,})", SEP]
        for c in missing[:60]:
            lines.append(f"  {c.tsv.vid[:65]}")
        if len(missing) > 60:
            lines.append(f"  … and {len(missing)-60} more (see full report file)")
        lines.append("")

    # ── PROBE ERRORS ──────────────────────────────────────────────────────
    if errors:
        lines += [SEP, f"  FFPROBE ERRORS  ({len(errors):,})", SEP]
        for c in errors[:50]:
            lines.append(
                f"  {c.tsv.file_path.name[:52]:<52}  "
                f"{(c.error or '')[:80]}"
            )
        if len(errors) > 50:
            lines.append(f"  … and {len(errors)-50} more")
        lines.append("")

    lines.append("")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(report_path: Path) -> logging.Logger:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = report_path.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("inspector")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Inspect How2Sign filtered clips using TSV annotation files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS,
                   choices=ALL_SPLITS,
                   help="Splits to inspect (default: train val test).")
    p.add_argument("--workers", type=int, default=10,
                   help="Parallel ffprobe workers (default: 10).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only probe the first N clips per split "
                        "(useful for a quick sanity-check).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    log  = setup_logging(REPORT_PATH)

    # ── Verify ffprobe ────────────────────────────────────────────────────
    try:
        r = subprocess.run([FFPROBE, "-version"],
                           capture_output=True, text=True, timeout=10)
        log.info("ffprobe: " + r.stdout.splitlines()[0])
    except FileNotFoundError:
        log.error(
            f"ffprobe not found at '{FFPROBE}'.\n"
            "  Install FFmpeg: winget install Gyan.FFmpeg"
        )
        sys.exit(1)

    # ── Load TSV files ────────────────────────────────────────────────────
    log.info("")
    all_tsv_rows: List[TsvRow] = []
    tsv_summary: List[Tuple] = []   # for the header table

    for split in args.splits:
        rows, total_rows, h2s_count = load_tsv(split)
        other_count = total_rows - h2s_count

        if total_rows == 0:
            log.warning(f"  [{split:5s}]  TSV not found or empty: "
                        f"{DATA_DIR / TSV_PATTERN.format(split=split)}")
            continue

        if args.limit:
            rows = rows[:args.limit]

        log.info(
            f"  [{split:5s}]  TSV rows: {total_rows:,}  |  "
            f"how2sign: {h2s_count:,}  |  "
            f"other (skipped): {other_count:,}"
            + (f"  |  limited to: {args.limit}" if args.limit else "")
        )
        tsv_summary.append((split, total_rows, h2s_count, other_count, len(rows)))
        all_tsv_rows.extend(rows)

    if not all_tsv_rows:
        log.error(
            "No how2sign rows found.  Check:\n"
            "  1. DATA_DIR points to the correct folder\n"
            "  2. The 'source' column contains 'how2sign' (case-insensitive)\n"
            "  3. TSV filenames match TSV_PATTERN"
        )
        sys.exit(1)

    total = len(all_tsv_rows)
    log.info(f"\n  Total clips to probe: {total:,}  with {args.workers} workers …\n")

    # ── Run ffprobe in parallel ───────────────────────────────────────────
    results: Dict[str, List[ClipInfo]] = {s: [] for s in args.splits}
    all_results: List[ClipInfo] = []

    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe_clip, row): row for row in all_tsv_rows}

        for i, fut in enumerate(as_completed(futures), 1):
            info: ClipInfo = fut.result()
            results[info.tsv.split].append(info)
            all_results.append(info)

            elapsed = time.perf_counter() - t0
            rate    = i / elapsed if elapsed > 1e-6 else 0.001
            eta_s   = (total - i) / rate
            eta_str = (f"{eta_s/60:.0f}m{eta_s%60:.0f}s"
                       if eta_s >= 60 else f"{eta_s:.0f}s")

            if i % 1000 == 0 or i == total or i <= 5:
                if info.file_missing:
                    status = "✗ MISSING"
                elif info.error:
                    status = "✗ ERR    "
                else:
                    status = "✓        "
                log.info(
                    f"[{i:>7,}/{total:,}] {status}  "
                    f"{info.tsv.file_path.name[:48]:<48}  "
                    f"rate={rate:.0f}/s  ETA={eta_str}"
                )

    elapsed_total = time.perf_counter() - t0
    log.info(
        f"\n  Probing done in {elapsed_total:.1f} s  "
        f"({elapsed_total/60:.1f} min)  —  "
        f"rate: {total/elapsed_total:.0f} files/s\n"
    )

    # ── Build report ──────────────────────────────────────────────────────
    ok_count      = sum(1 for c in all_results if c.error is None)
    missing_count = sum(1 for c in all_results if c.file_missing)
    err_count     = sum(1 for c in all_results if c.error and not c.file_missing)

    header: List[str] = [
        "═" * 72,
        "  How2Sign Filtered Dataset  —  Inspection Report",
        "═" * 72,
        f"  TSV directory    : {DATA_DIR}",
        f"  Splits inspected : {args.splits}",
        f"  Probe time       : {elapsed_total:.1f} s  ({elapsed_total/60:.1f} min)",
        "",
        "  TSV BREAKDOWN",
        f"  {'Split':<8} {'All rows':>9} {'How2Sign':>10} "
        f"{'Other':>8} {'Probed':>8}",
        f"  {'-'*8} {'-'*9} {'-'*10} {'-'*8} {'-'*8}",
    ]
    for split, tot, h2s, oth, probed in tsv_summary:
        header.append(
            f"  {split:<8} {tot:>9,} {h2s:>10,} {oth:>8,} {probed:>8,}"
        )
    if len(tsv_summary) > 1:
        header.append(
            f"  {'TOTAL':<8} "
            f"{sum(r[1] for r in tsv_summary):>9,} "
            f"{sum(r[2] for r in tsv_summary):>10,} "
            f"{sum(r[3] for r in tsv_summary):>8,} "
            f"{sum(r[4] for r in tsv_summary):>8,}"
        )
    header += [
        "",
        "  PROBE RESULTS",
        f"  Succeeded        : {ok_count:,}",
        f"  File not on disk : {missing_count:,}",
        f"  ffprobe errors   : {err_count:,}",
        "",
    ]

    report_lines: List[str] = header

    # Per-split sections
    for split in args.splits:
        if results.get(split):
            report_lines += build_report(results[split], split.upper())

    # Combined section
    if len(args.splits) > 1:
        report_lines += build_report(all_results, "ALL SPLITS COMBINED")

    # Write + print
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n" + "\n".join(report_lines))

    log.info(f"Report saved to : {REPORT_PATH}")
    log.info(f"Progress log    : {REPORT_PATH.with_suffix('.log')}")
    log.info(f"Probed OK       : {ok_count:,}")
    log.info(f"Missing files   : {missing_count:,}")
    log.info(f"ffprobe errors  : {err_count:,}")


# ── Windows multiprocessing guard  —  DO NOT REMOVE ──────────────────────────
if __name__ == "__main__":
    main()