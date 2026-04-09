#!/usr/bin/env python3
"""
04_build_how2sign_dataset.py
============================
Builds the final How2Sign clip dataset in D:\\how2sign\\{split}\\.

For every How2Sign row in the TSV annotation files:
  1. Locate the corresponding source clip on disk.
  2. Probe its actual duration with ffprobe.
  3. If the actual duration is within [MIN_DURATION, MAX_DURATION]:
       → Re-encode with NVENC (h264_nvenc, 20 FPS) to D:\\how2sign\\{split}\\{vid}.mp4
  4. Otherwise → discard (skip).

Duration gating is based ONLY on the actual clip file duration.
start / end / duration_sec columns in the TSV are intentionally ignored.

After processing, writes output TSV files to DATA_DIR:
  final_full_train_dataset_h2s.tsv
  final_full_val_dataset_h2s.tsv
  final_full_test_dataset_h2s.tsv

Each output TSV has columns: vid, text, word_count, file_path, source, clip_duration_sec

Usage
-----
  # Test: 20 random clips, no output TSVs
  python scripts/04_build_how2sign_dataset.py --test --num-samples 20

  # Full run
  python scripts/04_build_how2sign_dataset.py --workers 6

  # Dry run (prints FFmpeg commands, no files written)
  python scripts/04_build_how2sign_dataset.py --test --num-samples 5 --dry-run
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

# =============================================================================
#  CONFIGURATION
# =============================================================================

DATA_DIR        = Path(r"C:\My Projects\sign-language-bridge\data")
RAW_CLIPS_DIR   = DATA_DIR / "how2sign_raw_clips"
TSV_PATTERN     = "final_full_{split}_dataset.tsv"
ALL_SPLITS      = ["train", "val", "test"]

OUTPUT_DIR      = Path(r"D:\how2sign")
TEST_OUTPUT_DIR = OUTPUT_DIR / "test_clips"

FFMPEG_PATH     = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE_PATH    = r"C:\ffmpeg\bin\ffprobe.exe"

# Clips outside this range (judged by actual file duration) are discarded.
MIN_DURATION    = 1.3   # seconds
MAX_DURATION    = 8.0   # seconds

OUTPUT_FPS      = 20
NVENC_PRESET    = "p5"
CQ_VALUE        = 18

# =============================================================================


# -----------------------------------------------------------------------------
#  Data model
# -----------------------------------------------------------------------------

@dataclass
class ClipJob:
    vid:          str
    text:         str
    word_count:   int
    source_file:  Path
    output_path:  Path
    split:        str
    clip_duration: float   # actual duration from ffprobe


# -----------------------------------------------------------------------------
#  ffprobe
# -----------------------------------------------------------------------------

def probe_duration(path: Path) -> Optional[float]:
    """Return the actual duration of a video file in seconds, or None on error."""
    try:
        r = subprocess.run(
            [
                FFPROBE_PATH,
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


# -----------------------------------------------------------------------------
#  Clip finder
# -----------------------------------------------------------------------------

def find_clip(vid: str) -> Optional[Path]:
    """Search for {vid}.mp4 across all *_rgb_front_clips sub-directories."""
    for subdir in RAW_CLIPS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / f"{vid}.mp4"
        if candidate.exists():
            return candidate
    return None


# -----------------------------------------------------------------------------
#  TSV loader & job builder
# -----------------------------------------------------------------------------

def load_tsv(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def build_jobs(
    rows: List[dict],
    output_dir: Path,
    split: str,
    log: logging.Logger,
) -> Tuple[List[ClipJob], int, int, int, int]:
    """
    For each how2sign row:
      - Find the source clip on disk
      - Probe its actual duration
      - Include only if duration is within [MIN_DURATION, MAX_DURATION]

    Returns (jobs, n_how2sign_total, n_missing, n_out_of_range, n_probe_error)
    """
    jobs: List[ClipJob] = []
    n_missing      = 0
    n_out_of_range = 0
    n_probe_error  = 0
    n_h2s_total    = 0

    for row in rows:
        if row.get("source", "").strip().lower() != "how2sign":
            continue
        n_h2s_total += 1

        vid = row["vid"].strip()

        try:
            word_count = int(float(row.get("word_count") or 0))
        except ValueError:
            word_count = 0

        source_file = find_clip(vid)
        if source_file is None:
            n_missing += 1
            continue

        clip_dur = probe_duration(source_file)
        if clip_dur is None:
            n_probe_error += 1
            log.warning(f"  ffprobe failed for {vid} — skipping")
            continue

        if not (MIN_DURATION <= clip_dur <= MAX_DURATION):
            n_out_of_range += 1
            continue

        jobs.append(ClipJob(
            vid=vid,
            text=row.get("text", "").strip(),
            word_count=word_count,
            source_file=source_file,
            output_path=output_dir / split / f"{vid}.mp4",
            split=split,
            clip_duration=clip_dur,
        ))

    return jobs, n_h2s_total, n_missing, n_out_of_range, n_probe_error


# -----------------------------------------------------------------------------
#  FFmpeg command builder
# -----------------------------------------------------------------------------

def build_ffmpeg_command(job: ClipJob, no_audio: bool = False) -> List[str]:
    """
    Re-encode the full source clip to normalize FPS.
    No seeking — the entire file is the sentence.
    """
    cmd: List[str] = [
        FFMPEG_PATH,
        "-y",
        "-hwaccel", "cuda",
        "-i", str(job.source_file),
    ]

    vf = [
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        f"fps={OUTPUT_FPS}",
    ]
    cmd += ["-vf", ",".join(vf)]

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

    if no_audio:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "44100"]

    cmd += ["-movflags", "+faststart", str(job.output_path)]
    return cmd


# -----------------------------------------------------------------------------
#  Worker  (top-level for Windows multiprocessing)
# -----------------------------------------------------------------------------

def process_clip(args_tuple: Tuple) -> Tuple[str, bool, str, Optional[float]]:
    """
    Returns (vid, success, message, output_duration_sec).
    output_duration_sec is probed from the encoded output file so the TSV
    reflects the exact duration of the file that was actually written to disk,
    not the source file's duration.  None on failure or dry-run.
    """
    job: ClipJob
    dry_run: bool
    no_audio: bool
    job, dry_run, no_audio = args_tuple

    try:
        if job.output_path.exists() and job.output_path.stat().st_size > 500:
            # Already exists — probe it to get its true duration.
            out_dur = probe_duration(job.output_path)
            return (job.vid, True, "skipped — already exists", out_dur)

        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_ffmpeg_command(job, no_audio=no_audio)

        if dry_run:
            return (job.vid, True, "DRY RUN — cmd: " + " ".join(cmd), None)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            tail = (result.stderr or "(no stderr)")[-800:]
            return (job.vid, False, f"ffmpeg exit {result.returncode}: {tail}", None)

        if not job.output_path.exists() or job.output_path.stat().st_size < 500:
            tail = (result.stderr or "(no stderr)")[-800:]
            return (job.vid, False, f"output missing or too small. stderr: {tail}", None)

        # Probe the freshly encoded output file for its exact duration.
        out_dur = probe_duration(job.output_path)
        return (job.vid, True, "ok", out_dur)

    except subprocess.TimeoutExpired:
        return (job.vid, False, "ffmpeg timed out (>180 s)", None)
    except Exception as exc:
        return (job.vid, False, f"{type(exc).__name__}: {exc}", None)


# -----------------------------------------------------------------------------
#  Pipeline runner
# -----------------------------------------------------------------------------

def run_pipeline(
    jobs: List[ClipJob],
    workers: int,
    log: logging.Logger,
    dry_run:  bool = False,
    no_audio: bool = False,
) -> Tuple[dict, Dict[str, float]]:
    """
    Returns (stats, output_durations).
    output_durations maps vid -> probed duration of the encoded output file.
    """
    stats = {
        "total":       len(jobs),
        "ok_new":      0,
        "ok_skipped":  0,
        "failed":      0,
        "failed_vids": [],
    }
    # vid -> actual duration of the encoded output file (from ffprobe)
    output_durations: Dict[str, float] = {}

    t0 = time.perf_counter()
    arg_tuples = [(job, dry_run, no_audio) for job in jobs]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_job: Dict = {
            pool.submit(process_clip, arg): arg[0]
            for arg in arg_tuples
        }

        for i, fut in enumerate(as_completed(future_to_job), 1):
            job: ClipJob = future_to_job[fut]
            vid, success, msg, out_dur = fut.result()

            elapsed = time.perf_counter() - t0
            rate    = i / elapsed if elapsed > 1e-6 else 0.001
            eta_s   = (stats["total"] - i) / rate
            eta_str = (f"{eta_s/3600:.1f}h" if eta_s >= 3600
                       else f"{eta_s/60:.0f}m{int(eta_s%60)}s")

            if success:
                if "skipped" in msg:
                    stats["ok_skipped"] += 1
                else:
                    stats["ok_new"] += 1
                output_durations[vid] = out_dur  # may be None for dry-run
            else:
                stats["failed"] += 1
                stats["failed_vids"].append((vid, msg))

            always_print = dry_run or stats["total"] <= 50 or i <= 5
            if always_print or i % 500 == 0 or not success:
                icon = "✓" if success else "✗"
                log.info(
                    f"[{i:>7,}/{stats['total']:,}] {icon}  "
                    f"{vid[:50]:<50}  "
                    f"rate={rate:5.1f}/s  ETA={eta_str}  {msg}"
                )

    stats["elapsed_sec"] = time.perf_counter() - t0
    return stats, output_durations


# -----------------------------------------------------------------------------
#  Output TSV writer
# -----------------------------------------------------------------------------

def write_output_tsv(
    jobs: List[ClipJob],
    output_durations: Dict[str, float],
    split: str,
    log: logging.Logger,
) -> None:
    """
    output_durations: vid -> duration probed from the encoded output file.
    This is the authoritative duration written to the TSV — it reflects the
    exact length of the file on disk after FPS normalization, not the source.
    """
    out_path   = DATA_DIR / f"final_full_{split}_dataset_h2s.tsv"
    fieldnames = ["vid", "text", "word_count", "file_path", "source", "clip_duration_sec"]
    job_lookup = {j.vid: j for j in jobs}

    rows_written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for vid in sorted(output_durations.keys()):
            job = job_lookup.get(vid)
            if job is None:
                continue
            out_dur = output_durations[vid]
            # Fall back to source duration only if ffprobe on the output failed.
            dur_str = f"{out_dur:.4f}" if out_dur is not None else f"{job.clip_duration:.4f}"
            writer.writerow({
                "vid":               job.vid,
                "text":              job.text,
                "word_count":        job.word_count,
                "file_path":         str(job.output_path),
                "source":            "how2sign",
                "clip_duration_sec": dur_str,
            })
            rows_written += 1

    log.info(f"  [{split:5s}]  TSV written: {out_path}  ({rows_written:,} rows)")


# -----------------------------------------------------------------------------
#  Logging
# -----------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "build_pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("h2s_build")


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Build final How2Sign dataset: filter by actual clip duration, encode with NVENC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test", action="store_true",
        help="Test mode: process a small sample, save to test_clips/, skip output TSVs.")
    p.add_argument("--num-samples", type=int, default=20,
        help="Number of clips to process in test mode (default: 20).")
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
        help="Print FFmpeg commands without executing.")
    p.add_argument("--no-audio", action="store_true",
        help="Drop audio from output clips.")
    return p.parse_args()


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = TEST_OUTPUT_DIR if args.test else OUTPUT_DIR
    log     = setup_logging(out_dir)

    log.info("=" * 76)
    log.info("  How2Sign Dataset Builder")
    log.info(f"  Mode          : {'TEST' if args.test else 'FULL RUN'}"
             + ("  [DRY RUN]" if args.dry_run else ""))
    log.info(f"  Duration gate : [{MIN_DURATION}s, {MAX_DURATION}s]  (actual clip duration)")
    log.info(f"  Splits        : {args.splits}")
    log.info(f"  Workers       : {args.workers}")
    log.info(f"  Output FPS    : {OUTPUT_FPS}")
    log.info(f"  NVENC preset  : {NVENC_PRESET}  |  CQ : {CQ_VALUE}")
    log.info(f"  Output dir    : {out_dir}")
    log.info("=" * 76)

    # Verify FFmpeg / NVENC
    log.info("Checking FFmpeg / NVENC ...")
    try:
        enc_r = subprocess.run([FFMPEG_PATH, "-encoders"],
                               capture_output=True, text=True, timeout=15)
        if "h264_nvenc" not in enc_r.stdout:
            log.error("h264_nvenc NOT found. Install the full Gyan FFmpeg build.")
            sys.exit(1)
        ver_r = subprocess.run([FFMPEG_PATH, "-version"],
                               capture_output=True, text=True, timeout=10)
        log.info("  FFmpeg : " + (ver_r.stdout.splitlines()[0] if ver_r.stdout else "?"))
        log.info("  NVENC  : h264_nvenc found ✓")
    except FileNotFoundError:
        log.error(f"FFmpeg not found at '{FFMPEG_PATH}'.")
        sys.exit(1)

    # Load TSVs and build jobs (probing each clip's actual duration)
    log.info("")
    log.info("Loading TSVs and probing clip durations ...")
    log.info("(This may take a few minutes for large splits)\n")

    all_jobs: List[ClipJob] = []
    jobs_by_split: Dict[str, List[ClipJob]] = {}
    summary_rows = []

    for split in args.splits:
        tsv_file = DATA_DIR / TSV_PATTERN.format(split=split)
        if not tsv_file.exists():
            log.warning(f"  TSV not found, skipping: {tsv_file}")
            continue

        rows = load_tsv(tsv_file)
        jobs, n_h2s, n_missing, n_oor, n_probe_err = build_jobs(
            rows, out_dir, split, log
        )

        log.info(
            f"  {split:<6}  how2sign={n_h2s:,}  missing={n_missing}  "
            f"out_of_range={n_oor}  probe_error={n_probe_err}  "
            f"accepted={len(jobs):,}"
        )
        summary_rows.append((split, n_h2s, n_missing, n_oor, n_probe_err, len(jobs)))
        jobs_by_split[split] = jobs
        all_jobs.extend(jobs)

    if not all_jobs:
        log.error("No eligible clips found. Check DATA_DIR and RAW_CLIPS_DIR.")
        sys.exit(1)

    log.info(f"\n  TOTAL accepted jobs: {len(all_jobs):,}")

    # Test mode: subsample
    if args.test:
        random.seed(args.seed)
        n = min(args.num_samples, len(all_jobs))
        all_jobs = random.sample(all_jobs, n)
        log.info(f"  TEST MODE — sampled {n} clips (seed={args.seed})")

    if not args.dry_run:
        log.info(f"\n  Starting {len(all_jobs):,} encode jobs, {args.workers} workers ...\n")

    stats, output_durations = run_pipeline(
        all_jobs,
        workers=args.workers,
        log=log,
        dry_run=args.dry_run,
        no_audio=args.no_audio,
    )

    # Final report
    elapsed_s = stats["elapsed_sec"]
    ok_total  = stats["ok_new"] + stats["ok_skipped"]
    log.info("\n" + "=" * 76)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 76)
    log.info(f"  Clips submitted          : {stats['total']:>10,}")
    log.info(f"  Newly encoded (success)  : {stats['ok_new']:>10,}")
    log.info(f"  Skipped (already exist)  : {stats['ok_skipped']:>10,}")
    log.info(f"  Total succeeded          : {ok_total:>10,}")
    log.info(f"  Failed                   : {stats['failed']:>10,}")
    log.info(f"  Success rate             : {100*ok_total/max(stats['total'],1):>9.2f} %")
    log.info(f"  Elapsed                  : {elapsed_s/3600:.2f} h  "
             f"({elapsed_s/60:.1f} min  /  {elapsed_s:.0f} s)")
    log.info("=" * 76)

    if stats["failed_vids"]:
        MAX_SHOW = 60
        shown = stats["failed_vids"][:MAX_SHOW]
        log.info(f"\n  Failed clips ({len(shown)} of {len(stats['failed_vids'])} shown):")
        log.info(f"  {'vid':<52}  reason")
        log.info(f"  {'-'*52}  {'-'*50}")
        for vid, reason in shown:
            log.info(f"  {vid[:52]:<52}  {reason.replace(chr(10), ' ')[:120]}")

    # Write output TSVs
    if not args.test and not args.dry_run:
        log.info("\nWriting output TSV files ...")
        for split in args.splits:
            jobs_for_split = jobs_by_split.get(split, [])
            if not jobs_for_split:
                continue
            split_vid_set = {j.vid for j in jobs_for_split}
            # Only pass durations for vids belonging to this split.
            split_durations = {
                vid: dur for vid, dur in output_durations.items()
                if vid in split_vid_set
            }
            write_output_tsv(jobs_for_split, split_durations, split, log)
        log.info("  Done.\n")
    elif args.test:
        log.info("\n  Output TSVs: SKIPPED in test mode.")
        log.info(f"  Test clips saved to: {out_dir}")
        log.info("  Inspect them, then re-run without --test for the full pipeline.")


if __name__ == "__main__":
    main()
