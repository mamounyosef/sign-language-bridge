#!/usr/bin/env python3
"""
How2Sign Video Clip Pipeline  (NVENC / RTX 4060 edition)
=========================================================
Reads train / val / test TSV annotation files, filters rows where
source == 'how2sign', then for each clip:

  1. GPU-accelerated seek to the annotated start time within the
     source file  (the file_path column in the TSV)
  2. Trims to exactly  (end - start)  seconds
  3. Re-encodes with NVENC (h264_nvenc) at visually transparent quality
  4. Outputs at exactly 20 FPS
  5. Saves to  D:\\how2sign\\{split}\\{vid}.mp4

Source-file layout
------------------
Each TSV row's  file_path  column points directly to the source clip, e.g.:
  C:\\...\\train_rgb_front_clips\\--7E2sU6zP4_13-5-rgb_front.mp4

The  start  and  end  columns are plain floats in seconds (e.g. 183.12).
We seek to  start  inside that file and read exactly  end - start  seconds.

No bounding-box cropping is applied (How2Sign has no bbox data).
No source-file deletion (disk space is not a concern).

Output TSV files
----------------
After all clips are processed, three new TSV files are written to DATA_DIR:
  final_full_train_dataset_trimmed.tsv
  final_full_val_dataset_trimmed.tsv
  final_full_test_dataset_trimmed.tsv

These are identical to the input TSVs for how2sign rows, except:
  - Only how2sign rows are included
  - start and end columns are removed
  - file_path is updated to the new output clip path
  - Only rows whose clip was successfully written to disk are included

QUALITY NOTE
------------
Re-encoding is required to normalise FPS to 20.  We use NVENC VBR + CQ 18
on the RTX 4060 (Ada Lovelace) — visually indistinguishable from the source.
Lower CQ_VALUE -> better quality / larger files.

Usage
-----
  # TEST: 20 random clips -> D:\\how2sign\\test_clips\\  (no output TSVs)
  python process_how2sign_clips.py --test --num-samples 20

  # TEST: dry-run, see exact FFmpeg commands without running them
  python process_how2sign_clips.py --test --num-samples 5 --dry-run

  # FULL RUN (all splits, 6 workers)
  python process_how2sign_clips.py --splits train val test --workers 6

  # FULL RUN, single split
  python process_how2sign_clips.py --splits train --workers 8

Requirements
------------
  Python 3.9+  (no extra pip packages needed)
  FFmpeg with NVENC — full Gyan build (NOT essentials):
    winget install Gyan.FFmpeg   (then restart terminal)
  NVIDIA drivers up-to-date (Game Ready / Studio >= 522)
"""

import csv
import logging
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==============================================================================
#  USER CONFIGURATION  — edit these to match your system
# ==============================================================================

# Directory containing the TSV annotation files
DATA_DIR        = Path(r"C:\My Projects\sign-language-bridge\data")

# TSV filename pattern  — {split} -> train / val / test
TSV_PATTERN     = "final_full_{split}_dataset.tsv"

# Root output directory.  Clips are saved to OUTPUT_DIR / {split} / {vid}.mp4
OUTPUT_DIR      = Path(r"D:\how2sign")

# Output directory used ONLY in --test mode
TEST_OUTPUT_DIR = OUTPUT_DIR / "test_clips"

# Path to ffmpeg binary.  Leave as "ffmpeg" if it is on PATH.
FFMPEG_PATH     = r"C:\ffmpeg\bin\ffmpeg.exe"

# All available splits
ALL_SPLITS      = ["train", "val", "test"]

# CQ quality: lower = better quality / larger file.
#   18 -> near-transparent (recommended)
#   15 -> overkill / lossless-ish
#   23 -> visibly lossy, smallest files
CQ_VALUE        = 18

# NVENC preset: p1 (fastest) -> p7 (best quality / slowest)
NVENC_PRESET    = "p5"

# Output frame rate
OUTPUT_FPS      = 20

# ==============================================================================


# ------------------------------------------------------------------------------
#  Data model
# ------------------------------------------------------------------------------

@dataclass
class ClipJob:
    vid:          str    # e.g. --7E2sU6zP4_13-5-rgb_front
    text:         str    # transcription
    start:        float  # start offset in seconds within the source file
    end:          float  # end offset in seconds within the source file
    duration_sec: float  # end - start
    word_count:   int
    source_file:  Path   # full path to the existing per-clip source .mp4
    output_path:  Path   # where the trimmed output clip is saved
    split:        str


# ------------------------------------------------------------------------------
#  TSV loader & job builder
# ------------------------------------------------------------------------------

def load_tsv(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def build_jobs(
    rows: List[dict],
    output_dir: Path,
    split: str,
) -> Tuple[List[ClipJob], int, int, int]:
    """
    Filter rows where source == 'how2sign' and build ClipJob objects.
    Returns  (jobs, n_total_how2sign, n_missing_source_file, n_bad_timing).

    Source file
    -----------
    We use the file_path column directly — it already contains the full
    path to the existing per-clip .mp4.  No path reconstruction needed.

    Output filename
    ---------------
    {vid}.mp4  — How2Sign vids contain no colons or reserved characters,
    so they are safe Windows filenames as-is.

    Counting note
    -------------
    n_total_how2sign = len(jobs) + n_bad_timing
    Missing-source-file rows ARE still added to jobs (the worker reports
    them as clean failures), so they are already counted inside len(jobs).
    """
    jobs: List[ClipJob] = []
    n_missing    = 0
    n_bad_timing = 0

    for row in rows:
        if row.get("source", "").strip().lower() != "how2sign":
            continue

        vid       = row["vid"].strip()
        file_path = Path(row.get("file_path", "").strip())

        # Parse timing — these are plain-float seconds for How2Sign
        try:
            start = float(row["start"])
            end   = float(row["end"])
        except (ValueError, KeyError):
            n_bad_timing += 1
            continue

        duration_sec = end - start
        if duration_sec <= 0:
            n_bad_timing += 1
            continue

        try:
            word_count = int(float(row.get("word_count") or 0))
        except ValueError:
            word_count = 0

        out_path = output_dir / split / f"{vid}.mp4"

        if not file_path.exists():
            n_missing += 1
            # Still add the job — the worker will report it as a clean failure
            # rather than silently skipping it.

        jobs.append(ClipJob(
            vid=vid,
            text=row.get("text", "").strip(),
            start=start,
            end=end,
            duration_sec=duration_sec,
            word_count=word_count,
            source_file=file_path,
            output_path=out_path,
            split=split,
        ))

    # FIX: n_missing rows are already inside jobs, so don't add them again.
    # n_total_how2sign = len(jobs) [which includes missing-source rows] + n_bad_timing
    return jobs, len(jobs) + n_bad_timing, n_missing, n_bad_timing


# ------------------------------------------------------------------------------
#  FFmpeg command builder
# ------------------------------------------------------------------------------

def build_ffmpeg_command(job: ClipJob, no_audio: bool = False) -> List[str]:
    """
    Build an FFmpeg command that:
      * Decodes on the GPU (CUDA, RTX 4060)
      * Re-encodes the full source clip — How2Sign source files in
        *_rgb_front_clips/ are already individually trimmed per sentence.
        The start/end columns in the TSV are timestamps from the original
        full YouTube video, NOT offsets within the per-clip file.
        Seeking with -ss into these short clips would seek past EOF.
      * Pads to even dimensions (NVENC yuv420p requirement)
      * Resamples to OUTPUT_FPS
      * Encodes with h264_nvenc at CQ_VALUE quality
      * Re-encodes audio at AAC 128 k / 44100 Hz (or drops it)

    Even-dimension enforcement
    --------------------------
    pad=ceil(iw/2)*2:ceil(ih/2)*2  adds at most 1 px of black on the right
    and/or bottom edge if the source has an odd dimension.  This is
    preferable to truncating (trunc) because no content is lost.
    """
    cmd: List[str] = [
        FFMPEG_PATH,
        "-y",
        "-hwaccel", "cuda",
        "-i", str(job.source_file),
        # No -ss / -t: source files are already per-sentence clips.
        # start/end in the TSV are original-video timestamps, not clip offsets.
    ]

    # Video filters: even-dimension pad then FPS resample
    vf = [
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        f"fps={OUTPUT_FPS}",
    ]
    cmd += ["-vf", ",".join(vf)]

    # NVENC encoding
    cmd += [
        "-c:v",       "h264_nvenc",
        "-preset",    NVENC_PRESET,
        "-rc",        "vbr",
        "-cq",        str(CQ_VALUE),
        "-b:v",       "0",
        "-maxrate",   "50M",
        "-bufsize",   "100M",
        "-profile:v", "high",
        "-pix_fmt",   "yuv420p",
    ]

    # Audio
    if no_audio:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "44100"]

    # Container
    cmd += ["-movflags", "+faststart", str(job.output_path)]

    return cmd


# ------------------------------------------------------------------------------
#  Worker  (top-level — required for Windows multiprocessing)
# ------------------------------------------------------------------------------

def process_clip(args_tuple: Tuple) -> Tuple[str, bool, str]:
    """
    Execute one trim job.
    Returns  (vid, success: bool, message: str).
    """
    job: ClipJob
    dry_run: bool
    no_audio: bool
    job, dry_run, no_audio = args_tuple

    try:
        # FIX: if output already exists and is valid, skip unconditionally.
        # Checking for the source file here is pointless — we don't need it
        # if the output is already good.
        if job.output_path.exists() and job.output_path.stat().st_size > 500:
            return (job.vid, True, "skipped — already exists")

        if not job.source_file.exists():
            return (job.vid, False,
                    f"source file not found: {job.source_file}")

        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_ffmpeg_command(job, no_audio=no_audio)

        if dry_run:
            return (job.vid, True, "DRY RUN — cmd: " + " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            tail = (result.stderr or "(no stderr)")[-1000:]
            return (job.vid, False,
                    f"ffmpeg exit {result.returncode}: {tail}")

        if not job.output_path.exists() or job.output_path.stat().st_size < 500:
            tail = (result.stderr or "(no stderr)")[-1000:]
            return (job.vid, False,
                    f"output file missing or suspiciously small (<500 bytes). stderr: {tail}")

        return (job.vid, True, "ok")

    except subprocess.TimeoutExpired:
        return (job.vid, False, "ffmpeg timed out (>180 s)")
    except Exception as exc:
        return (job.vid, False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------------------
#  Pipeline runner
# ------------------------------------------------------------------------------

def run_pipeline(
    jobs: List[ClipJob],
    workers: int,
    log: logging.Logger,
    dry_run:  bool = False,
    no_audio: bool = False,
) -> Tuple[dict, List[str]]:
    """
    Run all jobs in parallel.
    Returns  (stats, succeeded_vids).
    succeeded_vids is the list of vids confirmed written to disk — used
    afterwards to write the output TSV files.
    """
    stats = {
        "total":       len(jobs),
        "ok_new":      0,
        "ok_skipped":  0,
        "failed":      0,
        "failed_vids": [],   # list of (vid, reason)
        "no_src_file": 0,
    }
    succeeded_vids: List[str] = []

    t0 = time.perf_counter()
    arg_tuples = [(job, dry_run, no_audio) for job in jobs]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_job: Dict = {
            pool.submit(process_clip, arg): arg[0]
            for arg in arg_tuples
        }

        for i, fut in enumerate(as_completed(future_to_job), 1):
            job: ClipJob = future_to_job[fut]
            vid, success, msg = fut.result()

            elapsed = time.perf_counter() - t0
            rate    = i / elapsed if elapsed > 1e-6 else 0.001
            eta_s   = (stats["total"] - i) / rate
            eta_str = (f"{eta_s/3600:.1f}h" if eta_s >= 3600
                       else f"{eta_s/60:.0f}m{eta_s%60:.0f}s")

            if success:
                if "skipped" in msg:
                    stats["ok_skipped"] += 1
                else:
                    stats["ok_new"] += 1
                succeeded_vids.append(vid)
            else:
                stats["failed"] += 1
                stats["failed_vids"].append((vid, msg))
                if "source file not found" in msg:
                    stats["no_src_file"] += 1

            always_print = dry_run or stats["total"] <= 50 or i <= 5
            if always_print or i % 500 == 0 or not success:
                icon = "✓" if success else "✗"
                log.info(
                    f"[{i:>7,}/{stats['total']:,}] {icon}  "
                    f"{vid[:50]:<50}  "
                    f"rate={rate:5.1f}/s  ETA={eta_str}  {msg}"
                )

    stats["elapsed_sec"] = time.perf_counter() - t0
    return stats, succeeded_vids


# ------------------------------------------------------------------------------
#  Output TSV writer
# ------------------------------------------------------------------------------

def write_output_tsv(
    jobs: List[ClipJob],
    succeeded_vids: List[str],
    split: str,
    output_dir: Path,
    log: logging.Logger,
) -> None:
    """
    Write a new TSV for one split containing only successfully processed clips.

    Columns (start and end are deliberately omitted):
      vid  text  duration_sec  word_count  file_path  source

    file_path is updated to the NEW output clip path (in OUTPUT_DIR / split).
    duration_sec is the annotated duration (end - start) from the original TSV,
    which is the true duration of the trimmed output file.

    Output rows are sorted by vid for reproducible output across runs.
    """
    out_path = DATA_DIR / f"final_full_{split}_dataset_trimmed.tsv"
    fieldnames = ["vid", "text", "duration_sec", "word_count", "file_path", "source"]

    job_lookup: Dict[str, ClipJob] = {j.vid: j for j in jobs}

    # FIX: sort succeeded_vids so output row order is deterministic across runs.
    rows_written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for vid in sorted(succeeded_vids):
            job = job_lookup.get(vid)
            if job is None:
                continue
            writer.writerow({
                "vid":          job.vid,
                "text":         job.text,
                "duration_sec": f"{job.duration_sec:.6f}",
                "word_count":   job.word_count,
                "file_path":    str(job.output_path),
                "source":       "how2sign",
            })
            rows_written += 1

    log.info(f"  [{split:5s}]  TSV written: {out_path}  ({rows_written:,} rows)")


# ------------------------------------------------------------------------------
#  Time estimator
# ------------------------------------------------------------------------------

def time_estimate_table(n_clips: int) -> str:
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


# ------------------------------------------------------------------------------
#  Logging
# ------------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
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
    return logging.getLogger("how2sign")


# ------------------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Trim How2Sign source clips to exact TSV annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test", action="store_true",
        help="Test mode: process a small sample, save to TEST_OUTPUT_DIR, "
             "skip writing output TSV files.")
    p.add_argument("--num-samples", type=int, default=20,
        help="Number of clips to process in test mode (default: 20).")
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS,
        choices=ALL_SPLITS,
        help="Splits to process (default: train val test).")
    p.add_argument("--workers", type=int, default=4,
        help="Parallel FFmpeg processes (default: 4).")
    p.add_argument("--seed", type=int, default=42,
        help="Random seed for test-mode sampling (default: 42).")
    p.add_argument("--dry-run", action="store_true",
        help="Print FFmpeg commands without executing.  No files written.")
    p.add_argument("--no-audio", action="store_true",
        help="Drop audio from output clips.")
    return p.parse_args()


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = TEST_OUTPUT_DIR if args.test else OUTPUT_DIR
    log     = setup_logging(out_dir)

    log.info("=" * 76)
    log.info("  How2Sign Clip Pipeline  —  NVENC / RTX 4060 edition")
    log.info(f"  Mode          : {'TEST (no output TSVs)' if args.test else 'FULL RUN'}"
             + ("  [DRY RUN — no files written]" if args.dry_run else ""))
    log.info(f"  Splits        : {args.splits}")
    log.info(f"  Workers       : {args.workers}")
    log.info(f"  Output FPS    : {OUTPUT_FPS}")
    log.info(f"  NVENC preset  : {NVENC_PRESET}  |  CQ : {CQ_VALUE}")
    log.info(f"  Audio         : {'dropped (--no-audio)' if args.no_audio else 'kept (AAC 128k)'}")
    log.info(f"  Output dir    : {out_dir}")
    log.info(f"  Output TSVs   : {'SKIPPED (test mode)' if args.test else str(DATA_DIR)}")
    log.info("=" * 76)

    # Verify FFmpeg and NVENC
    log.info("Checking FFmpeg / NVENC ...")
    try:
        enc_r = subprocess.run([FFMPEG_PATH, "-encoders"],
                               capture_output=True, text=True, timeout=15)
        if "h264_nvenc" not in enc_r.stdout:
            log.error(
                "h264_nvenc NOT found in this FFmpeg build.\n"
                "  Install the full Gyan build (NOT essentials):\n"
                "    winget install Gyan.FFmpeg  then restart your terminal.\n"
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

    # Load TSVs and build jobs
    log.info("")
    all_jobs: List[ClipJob] = []
    jobs_by_split: Dict[str, List[ClipJob]] = {}
    tsv_rows = []

    for split in args.splits:
        tsv_file = DATA_DIR / TSV_PATTERN.format(split=split)
        if not tsv_file.exists():
            log.warning(f"  TSV not found, skipping: {tsv_file}")
            continue

        rows = load_tsv(tsv_file)
        jobs, n_h2s_total, n_missing, n_bad = build_jobs(rows, out_dir, split)

        n_other = len(rows) - n_h2s_total - n_bad
        tsv_rows.append((split, len(rows), n_other, n_h2s_total, n_missing, n_bad))
        jobs_by_split[split] = jobs
        all_jobs.extend(jobs)

    # TSV summary table
    log.info(f"  {'Split':<8} {'All rows':>9} {'Other src':>10} "
             f"{'How2Sign':>9} {'No file':>8} {'Bad timing':>11}")
    log.info(f"  {'-'*8} {'-'*9} {'-'*10} {'-'*9} {'-'*8} {'-'*11}")
    for split, tot, oth, h2s, miss, bad in tsv_rows:
        log.info(f"  {split:<8} {tot:>9,} {oth:>10,} "
                 f"{h2s:>9,} {miss:>8,} {bad:>11,}")
    if tsv_rows:
        log.info(f"  {'TOTAL':<8} "
                 f"{sum(r[1] for r in tsv_rows):>9,} "
                 f"{sum(r[2] for r in tsv_rows):>10,} "
                 f"{sum(r[3] for r in tsv_rows):>9,} "
                 f"{sum(r[4] for r in tsv_rows):>8,} "
                 f"{sum(r[5] for r in tsv_rows):>11,}")

    if not all_jobs:
        log.error(
            "\nNo How2Sign jobs found.  Check:\n"
            "  1. DATA_DIR points to the correct folder\n"
            "  2. The 'source' column contains 'how2sign' (case-insensitive)\n"
            "  3. The split names match the TSV filenames"
        )
        sys.exit(1)

    # Test mode: subsample
    if args.test:
        random.seed(args.seed)
        n = min(args.num_samples, len(all_jobs))
        all_jobs = random.sample(all_jobs, n)
        log.info(
            f"\n  TEST MODE — sampled {n} clips (seed={args.seed})\n"
            f"  Output: {out_dir}"
        )
    else:
        log.info(f"\n  FULL RUN — {len(all_jobs):,} clips")
        log.info(f"  Estimated wall-clock time:\n{time_estimate_table(len(all_jobs))}")

    # Print example commands for the first few jobs
    n_ex = min(3, len(all_jobs))
    log.info(f"\n  Example FFmpeg commands (first {n_ex} jobs):")
    for job in all_jobs[:n_ex]:
        cmd = build_ffmpeg_command(job, no_audio=args.no_audio)
        log.info(f"\n    vid    : {job.vid}")
        log.info(f"    text   : {job.text[:80]}")
        log.info(f"    start  : {job.start:.3f} s    end: {job.end:.3f} s    "
                 f"duration: {job.duration_sec:.3f} s")
        log.info(f"    src    : {job.source_file}  (exists={job.source_file.exists()})")
        log.info(f"    out    : {job.output_path}")
        log.info(f"    cmd    : {' '.join(cmd)}")

    if args.dry_run:
        log.info("\n  DRY RUN — remove --dry-run to execute.")
    else:
        log.info(
            f"\n  Starting {len(all_jobs):,} clip jobs, "
            f"{args.workers} parallel workers ...\n"
        )

    stats, succeeded_vids = run_pipeline(
        all_jobs,
        workers=args.workers,
        log=log,
        dry_run=args.dry_run,
        no_audio=args.no_audio,
    )

    # =========================================================================
    #  FINAL REPORT
    # =========================================================================
    elapsed_s  = stats["elapsed_sec"]
    ok_total   = stats["ok_new"] + stats["ok_skipped"]
    throughput = ok_total / elapsed_s if elapsed_s > 0 else 0

    log.info("\n" + "=" * 76)
    log.info("  PIPELINE COMPLETE — FINAL REPORT")
    log.info("=" * 76)
    log.info(f"  Clips submitted              : {stats['total']:>10,}")
    log.info(f"  ─────────────────────────────────────────────────────────")
    log.info(f"  Newly created (success)      : {stats['ok_new']:>10,}")
    log.info(f"  Skipped (already existed)    : {stats['ok_skipped']:>10,}  (counted as success)")
    log.info(f"  Total succeeded              : {ok_total:>10,}")
    log.info(f"  ─────────────────────────────────────────────────────────")
    log.info(f"  Failed                       : {stats['failed']:>10,}")
    log.info(f"    └─ source file not found   : {stats['no_src_file']:>10,}")
    log.info(f"  ─────────────────────────────────────────────────────────")
    log.info(f"  Success rate                 : "
             f"{100*ok_total/max(stats['total'],1):>9.2f} %")
    log.info(f"  ─────────────────────────────────────────────────────────")
    log.info(f"  Elapsed time                 : "
             f"{elapsed_s/3600:.2f} h  "
             f"({elapsed_s/60:.1f} min  /  {elapsed_s:.0f} s)")
    if ok_total > 0 and not args.dry_run:
        log.info(f"  Throughput                   : {throughput:>9.1f} clips/s")
    log.info("=" * 76)

    # Detailed failure list
    if stats["failed_vids"]:
        MAX_SHOW = 60
        shown = stats["failed_vids"][:MAX_SHOW]
        log.info(
            f"\n  Failed clips "
            f"({len(shown)} of {len(stats['failed_vids'])} shown"
            + (f" — see pipeline.log for full list"
               if len(stats["failed_vids"]) > MAX_SHOW else "")
            + "):"
        )
        log.info(f"  {'vid':<52}  reason")
        log.info(f"  {'-'*52}  {'-'*50}")
        for vid, reason in shown:
            reason_short = reason.replace("\n", " ")[:120]
            log.info(f"  {vid[:52]:<52}  {reason_short}")
    log.info("")

    # =========================================================================
    #  WRITE OUTPUT TSV FILES
    # =========================================================================
    if not args.test and not args.dry_run:
        log.info("Writing output TSV files ...")

        for split in args.splits:
            jobs_for_split = jobs_by_split.get(split, [])
            if not jobs_for_split:
                continue
            split_vids_set = {j.vid for j in jobs_for_split}
            # FIX: removed unused `succeeded_set`; filter inline instead.
            split_succeeded = [v for v in succeeded_vids if v in split_vids_set]
            write_output_tsv(
                jobs=jobs_for_split,
                succeeded_vids=split_succeeded,
                split=split,
                output_dir=out_dir,
                log=log,
            )
        log.info("  Output TSVs written successfully.\n")
    elif args.test:
        log.info(
            f"  Output TSVs: SKIPPED in test mode.\n"
            f"  Run without --test to generate them.\n"
        )

    if args.test and not args.dry_run:
        log.info(
            f"  Test clips saved to: {out_dir}\n"
            "  Inspect them manually.\n"
            "  When satisfied, re-run WITHOUT --test for the full pipeline."
        )


# -- Windows multiprocessing guard  —  DO NOT REMOVE --------------------------
if __name__ == "__main__":
    main()
