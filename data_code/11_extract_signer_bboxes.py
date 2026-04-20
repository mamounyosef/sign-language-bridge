"""
11_extract_signer_bboxes.py
===========================
Offline extraction of one static signer bounding box per clip, for all three
dataset splits. Writes three CSVs that the training DataLoader merges by ``vid``
to crop frames to the signing region before feeding them to Qwen3-VL.

Input:
    data/2_dataset_train.tsv
    data/2_dataset_val.tsv
    data/2_dataset_test.tsv

Output (alongside the TSVs):
    data/2_dataset_train_bboxes.csv
    data/2_dataset_val_bboxes.csv
    data/2_dataset_test_bboxes.csv

CSV columns:
    vid, file_path, source, x1, y1, x2, y2,
    frame_width, frame_height, detection_rate,
    num_sampled_frames, num_detected_frames, failed

Parallelized across processes. Each worker owns its own SignerCropper instance
(MediaPipe landmarker is not thread/process-safe; one per worker avoids
contention). The model file is downloaded once in the main process BEFORE
workers are spawned, so every worker sees a cached file.

Usage:
    conda activate base
    python data_code/11_extract_signer_bboxes.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp_proc
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from signer_cropper import SignerCropper, BBoxResult, _ensure_model_downloaded


def _silence_native_stderr() -> None:
    """Redirect fd 2 (native-layer stderr) to /dev/null.

    MediaPipe's C++ layer writes ``W0000`` / ``I0000`` glog messages directly
    to fd 2, bypassing Python's logging and the ``GLOG_minloglevel`` env var
    on some Windows builds. Redirecting the file descriptor catches everything.

    This only affects fd 2 (stderr) — fd 1 (stdout) is untouched, so Python
    ``print()`` calls and tqdm progress bars (when configured with
    ``file=sys.stdout``) remain visible. Python tracebacks that use
    ``sys.stderr`` (a file object with a cached fd) will still appear if any
    Python exception escapes a worker — though ``pool.imap_unordered`` will
    re-raise them in the main process, where stderr is clean.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        # devnull_fd is not closed — it stays open for the process lifetime.
    except Exception:
        pass

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

SPLITS = {
    "train": DATA_DIR / "2_dataset_train.tsv",
    "val":   DATA_DIR / "2_dataset_val.tsv",
    "test":  DATA_DIR / "2_dataset_test.tsv",
}

OUTPUT_CSVS = {
    "train": DATA_DIR / "2_dataset_train_bboxes.csv",
    "val":   DATA_DIR / "2_dataset_val_bboxes.csv",
    "test":  DATA_DIR / "2_dataset_test_bboxes.csv",
}

# SignerCropper settings (must stay in sync with training/inference usage).
SAMPLE_EVERY_N           = 4
PADDING_RATIO            = 0.25
MIN_DETECTION_CONFIDENCE = 0.5
MODEL_VARIANT            = "full"   # "lite" | "full" | "heavy"

DEFAULT_NUM_WORKERS      = 10
# ──────────────────────────────────────────────────────────────────────────────

# Worker-local SignerCropper (one per process, lazily initialised on first task).
_WORKER_CROPPER: Optional[SignerCropper] = None


def _worker_init() -> None:
    """Build the SignerCropper once per worker process, and silence native stderr."""
    _silence_native_stderr()
    global _WORKER_CROPPER
    _WORKER_CROPPER = SignerCropper(
        sample_every_n=SAMPLE_EVERY_N,
        padding_ratio=PADDING_RATIO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        model_variant=MODEL_VARIANT,
    )


def _process_one(task: tuple[str, str, str]) -> dict:
    """Run SignerCropper on one clip. Runs in worker process."""
    vid, file_path, source = task
    assert _WORKER_CROPPER is not None, "Worker not initialised (missing initializer?)."

    base = {
        "vid": vid,
        "file_path": file_path,
        "source": source,
        "x1": 0, "y1": 0, "x2": 0, "y2": 0,
        "frame_width": 0, "frame_height": 0,
        "detection_rate": 0.0,
        "num_sampled_frames": 0,
        "num_detected_frames": 0,
        "failed": True,
        "error": "",
    }

    if not Path(file_path).exists():
        base["error"] = "file_not_found"
        return base

    try:
        bbox: BBoxResult = _WORKER_CROPPER.compute_bbox(file_path)
    except Exception as exc:
        # Any exception counts as a failure for this clip.
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    base.update(asdict(bbox))
    return base


def _load_tasks(tsv_path: Path) -> list[tuple[str, str, str]]:
    """Read a TSV and return a list of (vid, file_path, source) tuples."""
    df = pd.read_csv(tsv_path, sep="\t")
    required = {"vid", "file_path", "source"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{tsv_path} missing columns: {missing}")
    return [
        (str(r["vid"]), str(r["file_path"]), str(r["source"]))
        for _, r in df.iterrows()
    ]


def _run_split(
    split_name: str,
    tsv_path: Path,
    output_csv: Path,
    num_workers: int,
) -> pd.DataFrame:
    tasks = _load_tasks(tsv_path)
    print(f"\n[{split_name}] {tsv_path.name}: {len(tasks):,} clips")
    print(f"[{split_name}] Using {num_workers} worker process(es)")

    results: list[dict] = []
    t0 = time.time()

    if num_workers <= 1:
        # Single-process path (useful for debugging).
        _worker_init()
        iterator = (_process_one(t) for t in tasks)
        for res in tqdm(iterator, total=len(tasks), desc=f"{split_name}", unit="clip", file=sys.stdout):
            results.append(res)
    else:
        ctx = mp_proc.get_context("spawn")  # Windows-safe
        with ctx.Pool(processes=num_workers, initializer=_worker_init) as pool:
            for res in tqdm(
                pool.imap_unordered(_process_one, tasks, chunksize=4),
                total=len(tasks),
                desc=f"{split_name}",
                unit="clip",
                file=sys.stdout,
            ):
                results.append(res)

    elapsed = time.time() - t0
    df = pd.DataFrame(results)

    # Persist in deterministic row order (sorted by vid so reruns produce stable diffs).
    df = df.sort_values("vid").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[{split_name}] Wrote {output_csv}  ({elapsed/60:.1f} min, "
          f"{len(tasks)/max(elapsed,1e-6):.1f} clips/sec)")
    return df


def _print_report(dfs_by_split: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 72)
    print("EXTRACTION REPORT")
    print("=" * 72)
    total_clips = 0
    total_failed = 0

    for split, df in dfs_by_split.items():
        n = len(df)
        n_failed = int(df["failed"].sum())
        n_ok = n - n_failed
        mean_rate = float(df.loc[~df["failed"], "detection_rate"].mean()) if n_ok else 0.0

        print(f"\n[{split}] total={n:,}  success={n_ok:,}  failed={n_failed:,} "
              f"({100 * n_failed / max(n, 1):.2f}%)")
        print(f"[{split}] mean detection_rate (successful): {mean_rate:.3f}")

        # Per-source breakdown
        for source, sub in df.groupby("source"):
            ns = len(sub)
            nsf = int(sub["failed"].sum())
            print(f"[{split}]   source={source:10s}  total={ns:,}  "
                  f"failed={nsf:,} ({100 * nsf / max(ns, 1):.2f}%)")

        # Error breakdown for failed clips
        if n_failed > 0:
            errs = df.loc[df["failed"] & (df["error"] != ""), "error"]
            if len(errs) > 0:
                top = errs.value_counts().head(5)
                print(f"[{split}] top failure reasons:")
                for err_msg, count in top.items():
                    print(f"[{split}]   {count:6d}  {err_msg}")

        total_clips += n
        total_failed += n_failed

    print("\n" + "-" * 72)
    print(f"OVERALL  total={total_clips:,}  failed={total_failed:,} "
          f"({100 * total_failed / max(total_clips, 1):.2f}%)")
    print("=" * 72)

    # Warnings
    warnings = []
    for split, df in dfs_by_split.items():
        for source, sub in df.groupby("source"):
            if len(sub) == 0:
                continue
            fail_rate = sub["failed"].mean()
            if source == "how2sign" and fail_rate > 0.05:
                warnings.append(
                    f"  [{split}] how2sign failure rate is {100*fail_rate:.1f}% "
                    "(expected < 5% — investigate)"
                )
            if source == "openasl" and fail_rate > 0.20:
                warnings.append(
                    f"  [{split}] openasl failure rate is {100*fail_rate:.1f}% "
                    "(consider tuning min_detection_confidence)"
                )
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(w)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline signer bbox extraction.")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"Number of parallel worker processes (default {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()),
                        choices=list(SPLITS.keys()),
                        help="Which splits to process (default: all three).")
    parser.add_argument("--overwrite", action="store_true",
                        help="If set, re-extract splits whose output CSV already exists.")
    args = parser.parse_args()

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"DATA_DIR     = {DATA_DIR}")
    print(f"Workers      = {args.workers}")
    print(f"Splits       = {args.splits}")

    # Pre-download the MediaPipe model ONCE in the main process so every worker
    # finds it cached on disk (avoids parallel-download races).
    print(f"\nEnsuring pose_landmarker_{MODEL_VARIANT}.task is available ...")
    _ensure_model_downloaded(MODEL_VARIANT)

    dfs_by_split: dict[str, pd.DataFrame] = {}

    for split in args.splits:
        tsv = SPLITS[split]
        out = OUTPUT_CSVS[split]
        if not tsv.exists():
            print(f"[{split}] SKIP — TSV not found: {tsv}")
            continue
        if out.exists() and not args.overwrite:
            print(f"[{split}] SKIP — {out.name} exists. Re-run with --overwrite to regenerate.")
            # Still load it so the final report includes this split.
            dfs_by_split[split] = pd.read_csv(out)
            continue
        dfs_by_split[split] = _run_split(split, tsv, out, args.workers)

    _print_report(dfs_by_split)
    print("\nDone.")


if __name__ == "__main__":
    # Required on Windows for multiprocessing.
    mp_proc.freeze_support()
    main()
