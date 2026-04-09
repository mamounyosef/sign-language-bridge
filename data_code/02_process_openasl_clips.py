#!/usr/bin/env python3
"""
OpenASL Video Clip Pipeline  (NVENC / RTX 4060 edition)
=========================================================
Reads train / val / test TSV annotation files, filters rows where
source == 'openasl', then for each clip:

  1. GPU-accelerated seek to the annotated start time (CUDA decode)
  2. Trim to the annotated end time
  3. Crop the frame using the normalised bounding box from bbox-v1.0.json
     (skips crop, trim-only, if the clip has no bbox entry)
  4. Re-encode with NVENC (h264_nvenc) at visually transparent quality
  5. Output at exactly 20 FPS
  6. [FULL RUN ONLY] Once every clip from a source .mp4 has been
     successfully processed, the source .mp4 is deleted to free disk space.
     If any clip from that video fails, the raw video is KEPT so you can retry.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QUALITY NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Because the crop filter MUST re-encode (stream copy + spatial crop is
 impossible), we use NVENC VBR + CQ 18.  CQ 18 on Ada Lovelace NVENC
 (RTX 4060) is visually indistinguishable from the source.  If you later
 want even higher quality, lower CQ_VALUE toward 10; for smaller files
 raise it toward 23.  We also preserve the original pixel format (yuv420p)
 and re-encode audio at 128 kbps AAC.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage
-----
  # ── TEST: 20 random clips → data\\test_clips\\  (NO deletion) ────────────
  python process_openasl_clips.py --test --num-samples 20

  # ── TEST: dry-run, see exact FFmpeg commands without running them ────────
  python process_openasl_clips.py --test --num-samples 5 --dry-run

  # ── FULL RUN (all splits, 6 workers, NVENC, with deletion) ──────────────
  python process_openasl_clips.py --splits train val test --workers 6

  # ── FULL RUN, single split ────────────────────────────────────────────────
  python process_openasl_clips.py --splits train --workers 8

Requirements
------------
  Python 3.9+  (no extra pip packages needed)
  FFmpeg with NVENC — use the full Gyan build (NOT the essentials build):
    Windows: winget install Gyan.FFmpeg   (then restart your terminal)
    Or download from: https://www.gyan.dev/ffmpeg/builds/
  NVIDIA drivers up-to-date (Game Ready / Studio ≥ 522)
"""

import csv
import json
import logging
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  ▶  USER CONFIGURATION  — edit these to match your system
# ══════════════════════════════════════════════════════════════════════════════

# Directory containing the raw full-length YouTube videos (.mp4)
RAW_VIDEOS_DIR  = Path(r"D:\OpenASL\raw_videos")

# Root directory containing the TSV files and bbox JSON
DATA_DIR        = Path(r"C:\My Projects\sign-language-bridge\data")


# Bounding-box JSON file
BBOX_FILE       = DATA_DIR / "bbox-v1.0.json"

# Output directory for the full pipeline run
OUTPUT_DIR      = Path(r"D:\OpenASL\raw_videos_clipped")

# Output directory used only in --test mode (inspect these before full run)
TEST_OUTPUT_DIR = DATA_DIR / "test_clips"

# Path to the ffmpeg binary.
# If ffmpeg is on PATH just leave "ffmpeg".
# Full path example:  r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# TSV filename pattern — {split} is replaced with train / val / test
TSV_PATTERN     = "final_full_{split}_dataset.tsv"

# All available splits
ALL_SPLITS      = ["train", "val", "test"]

# ── Encoding quality (NVENC / RTX 4060) ──────────────────────────────────────
# CQ controls perceptual quality:  lower = better quality / larger file.
#   18 → near-transparent (recommended default)
#   15 → overkill / lossless-ish
#   23 → visibly lossy, smaller files
CQ_VALUE        = 18

# NVENC speed/quality preset: p1 (fastest) → p7 (highest quality / slowest)
# p5 is a strong balance for RTX 4060.
NVENC_PRESET    = "slow"

# Output frame rate for all clips
OUTPUT_FPS      = 20

# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def tsv_path(split: str) -> Path:
    return DATA_DIR / TSV_PATTERN.format(split=split)


def sanitize_filename(vid: str) -> str:
    """
    Make a vid string safe as a Windows / Linux filename.
    OpenASL vids contain colons  (e.g. ZfJikvphhgY-00:01:38.099-00:01:45.120)
    which are ILLEGAL on Windows.  We replace ':' with '.' in the filename only.
    The original vid string is kept everywhere else (bbox lookup, logging, etc.).
    Result: ZfJikvphhgY-00.01.38.099-00.01.45.120
    """
    return vid.replace(":", ".")


def hms_to_seconds(t: str) -> float:
    """
    Convert a time string to seconds (float).
      'HH:MM:SS.mmm'  →  OpenASL format
      '183.12'        →  How2Sign format (safe fallback, not used for openasl)
    """
    t = t.strip()
    if ":" in t:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(t)


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClipJob:
    vid:          str                    # original vid key (bbox lookup, logs)
    text:         str                    # transcription
    start:        str                    # raw start value from TSV
    end:          str                    # raw end value from TSV
    source_video: Path                   # path to the raw full-length .mp4
    output_path:  Path                   # desired output path for the clip
    bbox:         Optional[List[float]]  # [x1,y1,x2,y2] normalised, or None
    split:        str


# ─────────────────────────────────────────────────────────────────────────────
#  FFmpeg command builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ffmpeg_command(job: ClipJob, no_audio: bool = False) -> List[str]:
    """
    Build an FFmpeg command that:
      • Decodes on the GPU (CUDA hardware acceleration for RTX 4060)
      • Seeks to start_sec (fast pre-input keyframe seek)
      • Reads exactly `duration` seconds
      • Optionally applies a normalised bounding-box crop
        (crop runs on CPU; FFmpeg handles the GPU→CPU transfer automatically)
      • Re-encodes video with h264_nvenc at quality level CQ_VALUE
      • Outputs at exactly OUTPUT_FPS frames per second
      • Re-encodes audio at 128 kbps AAC (or drops it with --no-audio)

    Bounding-box convention
    -----------------------
    [x_min, y_min, x_max, y_max] as fractions of frame width / height.
    Values outside [0,1] are clamped — the JSON legitimately contains values
    like y1 = -0.044 or y2 = 1.095; clamping is correct and intentional.

    Even-dimension enforcement
    --------------------------
    NVENC (like libx264) requires even pixel dimensions for yuv420p.
    trunc(.../2)*2  rounds each side down to the nearest even integer.
    This is done via an FFmpeg filter expression, so it works at any resolution.
    """
    start_sec = hms_to_seconds(job.start)
    end_sec   = hms_to_seconds(job.end)
    duration  = max(end_sec - start_sec, 0.1)   # guard against zero-length

    cmd: List[str] = [
        FFMPEG_PATH,
        "-y",                           # overwrite output without prompting
        "-hwaccel", "cuda",             # GPU-accelerated H.264 decode (RTX 4060)
        "-ss", f"{start_sec:.6f}",      # seek BEFORE input → fast keyframe seek
        "-i", str(job.source_video),
        "-t", f"{duration:.6f}",        # read exactly this many seconds
    ]

    # ── Video-filter chain ────────────────────────────────────────────────
    vf: List[str] = []

    if job.bbox is not None:
        x1, y1, x2, y2 = job.bbox

        # Expand width by 3% on each side, then clamp to [0, 1]
        x1 = max(0.0, min(1.0, float(x1) - 0.03))
        y1 = max(0.0, min(1.0, float(y1)))
        x2 = max(0.0, min(1.0, float(x2) + 0.03))
        y2 = max(0.0, min(1.0, float(y2)))

        if x2 > x1 and y2 > y1:
            # Resolution-independent crop via iw / ih expressions.
            # trunc(.../2)*2 enforces even dimensions (NVENC requirement).
            cw = f"trunc(({x2:.8f}-{x1:.8f})*iw/2)*2"
            ch = f"trunc(({y2:.8f}-{y1:.8f})*ih/2)*2"
            cx = f"trunc({x1:.8f}*iw)"
            cy = f"trunc({y1:.8f}*ih)"
            vf.append(f"crop=w={cw}:h={ch}:x={cx}:y={cy}")

    # Frame-rate conversion (always applied — ensures exactly OUTPUT_FPS)
    vf.append(f"fps={OUTPUT_FPS}")

    cmd += ["-vf", ",".join(vf)]

    # ── NVENC encoding settings (RTX 4060 / Ada Lovelace) ────────────────
    #
    #  -rc vbr           Variable-bitrate mode — required for -cq to work
    #  -cq CQ_VALUE      Perceptual quality target (analogous to CRF in x264).
    #                    Lower = better quality.  18 = visually near-lossless.
    #  -b:v 0            No explicit target bitrate in VBR+CQ mode
    #  -maxrate 50M      Safety ceiling to prevent runaway frame sizes
    #  -bufsize 100M     VBV buffer for the ceiling above
    #  -preset PRESET    p5 = strong quality/speed balance on Ada Lovelace
    #  -profile:v high   H.264 High profile — best compression efficiency
    #  -pix_fmt yuv420p  Matches source; maximum decoder compatibility
    #
    cmd += [
        "-c:v",      "h264_nvenc",
        "-preset",   NVENC_PRESET,
        "-rc",       "vbr",
        "-cq",       str(CQ_VALUE),
        "-b:v",      "0",
        "-maxrate",  "50M",
        "-bufsize",  "100M",
        "-profile:v","high",
        "-pix_fmt",  "yuv420p",
    ]

    # ── Audio ─────────────────────────────────────────────────────────────
    if no_audio:
        cmd += ["-an"]
    else:
        cmd += [
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar",  "44100",
        ]

    # ── Container ─────────────────────────────────────────────────────────
    cmd += [
        "-movflags", "+faststart",   # moov atom at front — good for streaming
        str(job.output_path),
    ]

    return cmd


# ─────────────────────────────────────────────────────────────────────────────
#  Worker  (top-level function — required for Windows multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def process_clip(args_tuple: Tuple) -> Tuple[str, bool, str, str]:
    """
    Execute one trim+crop job.
    Returns  (vid, success: bool, message: str, source_video_str: str).
    source_video_str is passed back so the main thread can track per-source-video
    completion for the deletion logic — without needing shared memory.
    """
    job: ClipJob
    dry_run: bool
    no_audio: bool
    job, dry_run, no_audio = args_tuple
    src_str = str(job.source_video)

    try:
        # ── Skip if already done ──────────────────────────────────────────
        if job.output_path.exists() and job.output_path.stat().st_size > 500:
            return (job.vid, True, "skipped — already exists", src_str)

        # ── Source video must exist ───────────────────────────────────────
        if not job.source_video.exists():
            return (job.vid, False,
                    f"source video not found: {job.source_video}", src_str)

        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_ffmpeg_command(job, no_audio=no_audio)

        if dry_run:
            return (job.vid, True, "DRY RUN — cmd: " + " ".join(cmd), src_str)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,      # 3 minutes per clip (very generous)
        )

        if result.returncode != 0:
            tail = (result.stderr or "(no stderr)")[-1000:]
            return (job.vid, False,
                    f"ffmpeg exit {result.returncode}: {tail}", src_str)

        # ── Sanity-check the output ───────────────────────────────────────
        if not job.output_path.exists() or job.output_path.stat().st_size < 500:
            return (job.vid, False,
                    "output file missing or suspiciously small (<500 bytes)",
                    src_str)

        return (job.vid, True, "ok", src_str)

    except subprocess.TimeoutExpired:
        return (job.vid, False, "ffmpeg timed out (>180 s)", src_str)
    except Exception as exc:
        return (job.vid, False, f"{type(exc).__name__}: {exc}", src_str)


# ─────────────────────────────────────────────────────────────────────────────
#  TSV loader & job builder
# ─────────────────────────────────────────────────────────────────────────────

def load_tsv(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def build_jobs(
    rows: List[dict],
    bbox_data: Dict[str, List[float]],
    output_dir: Path,
    split: str,
) -> Tuple[List[ClipJob], int, int]:
    """
    Filter rows where source == 'openasl' and build ClipJob objects.
    Returns  (jobs, n_missing_source_video, n_missing_bbox).
    """
    jobs: List[ClipJob] = []
    n_no_video = 0
    n_no_bbox  = 0

    for row in rows:
        if row.get("source", "").strip().lower() != "openasl":
            continue

        vid       = row["vid"].strip()
        file_path = row.get("file_path", "").strip()

        # Derive the YouTube ID from the file_path column's filename stem.
        # We do NOT use the full path from the TSV because it refers to a
        # different machine.  We always reconstruct from RAW_VIDEOS_DIR.
        ytid     = Path(file_path).stem if file_path else vid[:11]
        src_vid  = RAW_VIDEOS_DIR / f"{ytid}.mp4"
        out_path = output_dir / split / f"{sanitize_filename(vid)}.mp4"

        bbox = bbox_data.get(vid)
        if bbox is None:
            n_no_bbox += 1
        if not src_vid.exists():
            n_no_video += 1

        jobs.append(ClipJob(
            vid=vid,
            text=row.get("text", ""),
            start=row["start"],
            end=row["end"],
            source_video=src_vid,
            output_path=out_path,
            bbox=bbox,
            split=split,
        ))

    return jobs, n_no_video, n_no_bbox


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline runner  (parallel + per-source-video deletion logic)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    jobs: List[ClipJob],
    workers: int,
    log: logging.Logger,
    dry_run: bool            = False,
    no_audio: bool           = False,
    delete_source: bool      = False,
    deleted_videos_log: Optional[Path] = None,
) -> dict:
    """
    Run all jobs in parallel via ProcessPoolExecutor.

    Source-video deletion logic (active only when delete_source=True)
    -----------------------------------------------------------------
    We track, per source .mp4 file, how many clips have been submitted and
    how many have completed (ok or failed).  When the last clip for a given
    source video finishes:

      • ALL clips succeeded (ok + skipped) → delete the source .mp4
      • ANY clip failed                    → KEEP the source .mp4 (retry later)

    'Skipped' clips (output file already existed) count as success — they
    were processed correctly in a previous run.

    This tracking is done entirely in the main thread using the result tuples
    returned by each worker, so no locks or shared memory are needed.

    deleted_videos_log
    ------------------
    If provided, every successfully deleted source video is APPENDED to this
    file immediately after deletion — one entry per line in the format:
        2026-04-06 14:32:07  ZfJikvphhgY.mp4  (312.4 MB)
    The file is opened in append mode so it survives restarts and accumulates
    a complete history across multiple runs.
    """

    # ── Per-source-video tracker ──────────────────────────────────────────
    vid_tracker: Dict[str, dict] = {}
    for job in jobs:
        key = str(job.source_video)
        if key not in vid_tracker:
            vid_tracker[key] = {
                "total": 0, "done": 0, "ok": 0, "failed": 0,
                "path": job.source_video,
            }
        vid_tracker[key]["total"] += 1

    stats = {
        "total":           len(jobs),
        "ok_new":          0,
        "ok_skipped":      0,
        "failed":          0,
        "failed_vids":     [],   # list of (vid, reason)
        "no_src_video":    0,
        "deleted_src":     0,
        "kept_src_failed": 0,
        "freed_bytes":     0,
    }

    t0 = time.perf_counter()
    arg_tuples = [(job, dry_run, no_audio) for job in jobs]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_job: Dict = {
            pool.submit(process_clip, arg): arg[0]
            for arg in arg_tuples
        }

        for i, fut in enumerate(as_completed(future_to_job), 1):
            job: ClipJob = future_to_job[fut]
            vid, success, msg, src_str = fut.result()

            elapsed = time.perf_counter() - t0
            rate    = i / elapsed if elapsed > 1e-6 else 0.001
            eta_s   = (stats["total"] - i) / rate

            # ── Accumulate stats ──────────────────────────────────────────
            if success:
                if "skipped" in msg:
                    stats["ok_skipped"] += 1
                else:
                    stats["ok_new"] += 1
            else:
                stats["failed"] += 1
                stats["failed_vids"].append((vid, msg))
                if "source video not found" in msg:
                    stats["no_src_video"] += 1

            # ── Update per-source tracker ─────────────────────────────────
            vt = vid_tracker[src_str]
            vt["done"]   += 1
            vt["ok"]     += 1 if success else 0
            vt["failed"] += 0 if success else 1

            # ── Log progress ──────────────────────────────────────────────
            ok_sofar = stats["ok_new"] + stats["ok_skipped"]
            eta_str  = (f"{eta_s/3600:.1f}h" if eta_s >= 3600
                        else f"{eta_s/60:.0f}m{eta_s%60:.0f}s")

            # Always print in test/dry-run/small runs; otherwise every 500
            always_print = dry_run or stats["total"] <= 50 or i <= 5
            if always_print or i % 500 == 0 or not success:
                icon = "✓" if success else "✗"
                log.info(
                    f"[{i:>7,}/{stats['total']:,}] {icon}  "
                    f"{vid[:50]:<50}  "
                    f"rate={rate:5.1f}/s  ETA={eta_str}  {msg}"
                )

            # ── Deletion check ────────────────────────────────────────────
            if delete_source and vt["done"] == vt["total"]:
                src_path: Path = vt["path"]

                if vt["failed"] == 0:
                    # Every clip from this raw video is confirmed good → delete
                    if src_path.exists():
                        try:
                            freed = src_path.stat().st_size
                            src_path.unlink()
                            stats["deleted_src"]  += 1
                            stats["freed_bytes"]  += freed
                            freed_mb = freed / (1024 ** 2)
                            log.info(
                                f"  🗑️  DELETED  {src_path.name}  "
                                f"({freed_mb:.1f} MB freed)"
                                f"  — all {vt['total']} clip(s) OK"
                            )
                            # ── Append to the deleted-videos record file ──
                            if deleted_videos_log is not None:
                                try:
                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                    with open(deleted_videos_log, "a", encoding="utf-8") as dlf:
                                        dlf.write(
                                            f"{timestamp}  {src_path.name}"
                                            f"  ({freed_mb:.1f} MB)"
                                            f"  clips={vt['total']}\n"
                                        )
                                except Exception as log_exc:
                                    log.warning(
                                        f"  ⚠️  Could not write to deleted-videos log: {log_exc}"
                                    )
                        except Exception as exc:
                            log.warning(
                                f"  ⚠️  Could not delete {src_path}: {exc}"
                            )
                else:
                    # ≥1 clip failed → keep the source for retry
                    stats["kept_src_failed"] += 1
                    log.warning(
                        f"  ⚠️  KEPT  {src_path.name}"
                        f"  — {vt['failed']} clip(s) failed, retry later"
                    )

    stats["elapsed_sec"] = time.perf_counter() - t0
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Time estimator
# ─────────────────────────────────────────────────────────────────────────────

def time_estimate_table(n_clips: int) -> str:
    """
    Estimated wall-clock time (empirical, SSD, ~720p, NVENC fast preset, RTX 4060).
    NVENC:    ~0.40 s/clip/worker  (≈ 12–15x realtime)
    CPU x264: ~1.50 s/clip/worker  (fast preset, for reference only)
    """
    lines = [
        f"  {'Workers':<10} {'NVENC / RTX 4060':<22} {'CPU x264 (ref)':<18}",
        f"  {'-'*10} {'-'*22} {'-'*18}",
    ]
    for w in [2, 4, 6, 8]:
        gpu = n_clips * 0.40 / w
        cpu = n_clips * 1.50 / w
        def fmt(s):
            return f"{s/3600:.1f} h" if s >= 3600 else f"{s/60:.0f} min"
        lines.append(f"  {w:<10} {fmt(gpu):<22} {fmt(cpu):<18}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> "logging.Logger":
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("openasl")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Trim, crop, and (optionally) delete OpenASL source videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test", action="store_true",
        help="Test mode: process a small sample, save to TEST_OUTPUT_DIR, "
             "NEVER delete source videos regardless of other flags.")
    p.add_argument("--num-samples", type=int, default=20,
        help="Number of clips to process in test mode (default: 20).")
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS,
        choices=ALL_SPLITS,
        help="Splits to process (default: train val test).")
    p.add_argument("--workers", type=int, default=4,
        help="Parallel FFmpeg processes (default: 4). "
             "RTX 4060 NVENC is not the bottleneck — CPU/workers are.")
    p.add_argument("--seed", type=int, default=32,
        help="Random seed for test-mode sampling (default: 42).")
    p.add_argument("--dry-run", action="store_true",
        help="Print FFmpeg commands without executing.  No files written or deleted.")
    p.add_argument("--no-audio", action="store_true",
        help="Drop audio from output clips (slightly faster).")
    p.add_argument("--no-bbox", action="store_true",
        help="Ignore bounding boxes — trim only, no spatial crop.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = TEST_OUTPUT_DIR if args.test else OUTPUT_DIR
    log     = setup_logging(out_dir)

    # Deletion is ONLY active on a proper full run (not test, not dry-run)
    delete_source = (not args.test) and (not args.dry_run)

    # Append-only record of every source video that gets deleted.
    # Sits next to pipeline.log.  Created on first deletion; appended to on
    # every subsequent run so you have a permanent cumulative history.
    deleted_videos_log: Optional[Path] = (
        out_dir / "deleted_videos.txt" if delete_source else None
    )

    log.info("=" * 76)
    log.info("  OpenASL Clip Pipeline  —  NVENC / RTX 4060 edition")
    log.info(f"  Mode          : {'TEST (no deletion)' if args.test else 'FULL RUN'}"
             + ("  [DRY RUN — no files written or deleted]" if args.dry_run else ""))
    log.info(f"  Splits        : {args.splits}")
    log.info(f"  Workers       : {args.workers}")
    log.info(f"  Output FPS    : {OUTPUT_FPS}")
    log.info(f"  NVENC preset  : {NVENC_PRESET}  |  CQ : {CQ_VALUE}")
    log.info(f"  Bbox crop     : {'DISABLED (--no-bbox)' if args.no_bbox else 'enabled'}")
    log.info(f"  Audio         : {'dropped (--no-audio)' if args.no_audio else 'kept (AAC 128k)'}")
    log.info(f"  Raw videos    : {RAW_VIDEOS_DIR}")
    log.info(f"  Output dir    : {out_dir}")
    log.info(f"  Delete source : {delete_source}")
    if deleted_videos_log:
        log.info(f"  Deletion log  : {deleted_videos_log}")
    log.info("=" * 76)

    # ── Verify FFmpeg and NVENC ───────────────────────────────────────────
    log.info("Checking FFmpeg / NVENC …")
    try:
        enc_r = subprocess.run([FFMPEG_PATH, "-encoders"],
                               capture_output=True, text=True, timeout=15)
        if "h264_nvenc" not in enc_r.stdout:
            log.error(
                "h264_nvenc NOT found in this FFmpeg build.\n"
                "  Install the full Gyan build (NOT essentials):\n"
                "    winget install Gyan.FFmpeg   then restart your terminal.\n"
                "  Or: https://www.gyan.dev/ffmpeg/builds/"
            )
            sys.exit(1)
        ver_r = subprocess.run([FFMPEG_PATH, "-version"],
                               capture_output=True, text=True, timeout=10)
        log.info("  FFmpeg : " + (ver_r.stdout.splitlines()[0] if ver_r.stdout else "?"))
        log.info("  NVENC  : h264_nvenc found ✓")
    except FileNotFoundError:
        log.error(
            f"FFmpeg not found at '{FFMPEG_PATH}'.\n"
            "  Install it and add its bin\\ folder to your PATH."
        )
        sys.exit(1)

    # ── Load bounding boxes ───────────────────────────────────────────────
    log.info(f"\nLoading bounding boxes from {BBOX_FILE} …")
    if not BBOX_FILE.exists():
        log.error(f"Bbox file not found: {BBOX_FILE}")
        sys.exit(1)
    with open(BBOX_FILE, "r", encoding="utf-8") as fh:
        bbox_data: Dict[str, List[float]] = json.load(fh)
    if args.no_bbox:
        bbox_data = {}
    log.info(f"  {len(bbox_data):,} bbox entries loaded"
             + (" (ignored — --no-bbox)" if args.no_bbox else ""))

    # ── Load TSVs and build jobs ──────────────────────────────────────────
    log.info("")
    all_jobs: List[ClipJob] = []
    tsv_rows = []   # for the summary table

    for split in args.splits:
        p = tsv_path(split)
        if not p.exists():
            log.warning(f"  TSV not found, skipping: {p}")
            continue

        rows  = load_tsv(p)
        jobs, n_no_vid, n_no_bbox = build_jobs(rows, bbox_data, out_dir, split)

        n_h2s   = sum(1 for r in rows if r.get("source","").strip().lower() == "how2sign")
        n_total = len(rows)
        tsv_rows.append((split, n_total, n_h2s, len(jobs), n_no_vid, n_no_bbox))
        all_jobs.extend(jobs)

    # ── TSV summary table ─────────────────────────────────────────────────
    log.info(f"  {'Split':<8} {'All rows':>9} {'How2Sign':>9} "
             f"{'OpenASL':>9} {'No src vid':>11} {'No bbox':>8}")
    log.info(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*11} {'-'*8}")
    for split, tot, h2s, oa, nov, nob in tsv_rows:
        log.info(f"  {split:<8} {tot:>9,} {h2s:>9,} {oa:>9,} {nov:>11,} {nob:>8,}")
    if tsv_rows:
        log.info(f"  {'TOTAL':<8} "
                 f"{sum(r[1] for r in tsv_rows):>9,} "
                 f"{sum(r[2] for r in tsv_rows):>9,} "
                 f"{sum(r[3] for r in tsv_rows):>9,} "
                 f"{sum(r[4] for r in tsv_rows):>11,} "
                 f"{sum(r[5] for r in tsv_rows):>8,}")

    if not all_jobs:
        log.error(
            "\nNo OpenASL jobs found.  Check:\n"
            "  1. DATA_DIR points to the correct folder\n"
            "  2. The 'source' column contains 'openasl' (case-insensitive)\n"
            "  3. The split names match the TSV filenames"
        )
        sys.exit(1)

    unique_src = len({str(j.source_video) for j in all_jobs})
    log.info(f"\n  {len(all_jobs):,} OpenASL clips from {unique_src:,} unique source videos")

    # ── Test mode: subsample ──────────────────────────────────────────────
    if args.test:
        random.seed(args.seed)
        n = min(args.num_samples, len(all_jobs))
        all_jobs = random.sample(all_jobs, n)
        log.info(
            f"\n  TEST MODE — sampled {n} clips (seed={args.seed})\n"
            f"  Output: {out_dir}\n"
            f"  Source-video deletion: DISABLED (always off in test mode)"
        )
    else:
        log.info(f"\n  FULL RUN — {len(all_jobs):,} clips")
        log.info(f"  Estimated wall-clock time:\n{time_estimate_table(len(all_jobs))}")
        if delete_source:
            log.info(
                "\n  ⚠️  SPACE-EFFICIENT MODE ACTIVE:\n"
                "  Each raw source video will be DELETED once ALL its clips\n"
                "  are confirmed successful.  Source videos with ANY failed\n"
                "  clip are KEPT so you can retry them safely."
            )

    # ── Print example commands ────────────────────────────────────────────
    n_ex = min(3, len(all_jobs))
    log.info(f"\n  Example FFmpeg commands (first {n_ex} jobs):")
    for job in all_jobs[:n_ex]:
        cmd = build_ffmpeg_command(job, no_audio=args.no_audio)
        log.info(f"\n    vid  : {job.vid}")
        log.info(f"    text : {job.text[:80]}")
        log.info(f"    bbox : {job.bbox}")
        log.info(f"    src  : {job.source_video}  (exists={job.source_video.exists()})")
        log.info(f"    out  : {job.output_path}")
        log.info(f"    cmd  : {' '.join(cmd)}")

    # ── Run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("\n  DRY RUN — remove --dry-run to execute.")
    else:
        log.info(
            f"\n  Starting {len(all_jobs):,} clip jobs, "
            f"{args.workers} parallel workers …\n"
        )

    stats = run_pipeline(
        all_jobs,
        workers=args.workers,
        log=log,
        dry_run=args.dry_run,
        no_audio=args.no_audio,
        delete_source=delete_source,
        deleted_videos_log=deleted_videos_log,
    )

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════
    elapsed_s  = stats["elapsed_sec"]
    ok_total   = stats["ok_new"] + stats["ok_skipped"]
    throughput = ok_total / elapsed_s if elapsed_s > 0 else 0
    freed_gb   = stats["freed_bytes"] / (1024 ** 3)

    log.info("\n" + "═" * 76)
    log.info("  PIPELINE COMPLETE — FINAL REPORT")
    log.info("═" * 76)
    log.info(f"  Clips submitted              : {stats['total']:>10,}")
    log.info(f"  ─────────────────────────────────────────────────────")
    log.info(f"  Newly created (success)      : {stats['ok_new']:>10,}")
    log.info(f"  Skipped (already existed)    : {stats['ok_skipped']:>10,}  (counted as success)")
    log.info(f"  Total succeeded              : {ok_total:>10,}")
    log.info(f"  ─────────────────────────────────────────────────────")
    log.info(f"  Failed                       : {stats['failed']:>10,}")
    log.info(f"    └─ source video not found  : {stats['no_src_video']:>10,}  (not yet downloaded)")
    log.info(f"  ─────────────────────────────────────────────────────")
    log.info(f"  Success rate                 : {100*ok_total/max(stats['total'],1):>9.2f} %")
    log.info(f"  ─────────────────────────────────────────────────────")
    if delete_source:
        log.info(f"  Source videos deleted        : {stats['deleted_src']:>10,}  "
                 f"({freed_gb:.2f} GB freed)")
        log.info(f"  Source videos KEPT (retries) : {stats['kept_src_failed']:>10,}  "
                 f"(had ≥1 failed clip)")
        log.info(f"  ─────────────────────────────────────────────────────")
    log.info(f"  Elapsed time                 : "
             f"{elapsed_s/3600:.2f} h  "
             f"({elapsed_s/60:.1f} min  /  {elapsed_s:.0f} s)")
    if ok_total > 0 and not args.dry_run:
        log.info(f"  Throughput                   : {throughput:>9.1f} clips/s")
    log.info("═" * 76)

    # ── Detailed failure list ─────────────────────────────────────────────
    if stats["failed_vids"]:
        MAX_SHOW = 60
        shown = stats["failed_vids"][:MAX_SHOW]
        log.info(
            f"\n  Failed clips "
            f"({len(shown)} of {len(stats['failed_vids'])} shown"
            + (f" — see pipeline.log for all" if len(stats["failed_vids"]) > MAX_SHOW else "")
            + "):"
        )
        log.info(f"  {'vid':<52}  reason")
        log.info(f"  {'-'*52}  {'-'*50}")
        for vid, reason in shown:
            # Collapse ffmpeg stderr blocks to a single line for the summary
            reason_short = reason.replace("\n", " ")[:120]
            log.info(f"  {vid[:52]:<52}  {reason_short}")
    log.info("")

    if args.test and not args.dry_run:
        log.info(
            f"  ✅  Test clips saved to: {out_dir}\n"
            "     Inspect them manually.\n"
            "     When satisfied, re-run WITHOUT --test for the full pipeline."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Windows multiprocessing guard  —  DO NOT REMOVE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()