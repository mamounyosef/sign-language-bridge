"""
10_test_signer_crop.py
======================
Visual test for the SignerCropper pipeline.

Samples 10 random clips (5 how2sign + 5 openasl) from the train TSV, runs
MediaPipe pose-based bbox computation, and renders side-by-side MP4s to
``C:\\Users\\mamou\\Downloads\\``:

    Left half  : original frame with the computed bbox rectangle drawn on it.
    Right half : cropped frame (what would actually be fed to the model).

A thin top label bar shows source + vid for each clip.

After running this, visually verify each output video:
    - Does the bbox enclose the signer's upper body + arms at maximum
      extension across the whole clip?
    - Does the cropped half show the signer clearly with ~25% margin on
      all sides?
    - Are there frames where the signer is outside the crop?

If 9/10 or 10/10 are clean, proceed to 11_extract_signer_bboxes.py.

Usage:
    conda activate base  (or whichever env has cv2, mediapipe, pandas)
    python data_code/10_test_signer_crop.py
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from signer_cropper import SignerCropper, BBoxResult

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_TSV    = PROJECT_ROOT / "data" / "2_dataset_train.tsv"
OUTPUT_DIR   = Path(r"C:\Users\mamou\Downloads")

NUM_PER_SOURCE = 5             # 5 how2sign + 5 openasl = 10 total
SOURCES        = ("how2sign", "openasl")
SEED           = 42

# SignerCropper settings (keep in sync with training/inference usage).
SAMPLE_EVERY_N           = 4
PADDING_RATIO            = 0.25
MIN_DETECTION_CONFIDENCE = 0.5
MODEL_VARIANT            = "full"   # "lite" | "full" | "heavy"

# Rendering
LABEL_BAR_HEIGHT = 40
LABEL_FONT       = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE      = 0.6
LABEL_THICKNESS  = 1
BBOX_COLOR       = (0, 255, 0)   # green, BGR
BBOX_THICKNESS   = 2
# ──────────────────────────────────────────────────────────────────────────────


def pick_sample_rows(df: pd.DataFrame, n_per_source: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    parts = []
    for src in SOURCES:
        pool = df[df["source"] == src]
        if len(pool) < n_per_source:
            raise RuntimeError(
                f"Not enough rows for source={src}: have {len(pool)}, need {n_per_source}"
            )
        indices = rng.sample(range(len(pool)), n_per_source)
        parts.append(pool.iloc[indices])
    return pd.concat(parts, ignore_index=True)


def load_video_frames(path: str) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise IOError(f"Zero readable frames: {path}")
    return np.stack(frames, axis=0), fps


def render_side_by_side(
    frames: np.ndarray,
    bbox: BBoxResult,
    source: str,
    vid: str,
    fps: float,
    output_path: Path,
) -> tuple[int, int]:
    """Render left=original+bbox, right=cropped, with a label bar on top."""
    T, H, W, _ = frames.shape
    crop_h = bbox.height
    crop_w = bbox.width

    # Resize cropped half to match original height so the two halves line up.
    target_h = H
    if crop_h == 0 or crop_w == 0:
        # Failed bbox: just show original on both sides with a "FAILED" label.
        cropped_resized_w = W
    else:
        scale = target_h / crop_h
        cropped_resized_w = max(1, int(round(crop_w * scale)))

    final_w = W + cropped_resized_w
    final_h = target_h + LABEL_BAR_HEIGHT

    label_text = f"{source} | {vid} | detected {bbox.num_detected_frames}/{bbox.num_sampled_frames}"
    if bbox.failed:
        label_text += "  [FAILED]"

    # cv2 VideoWriter with mp4v fourcc for broad playback support.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (final_w, final_h))
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {output_path}")

    try:
        for t in range(T):
            frame = frames[t]

            # Left half: original + bbox rectangle (if not failed)
            left = frame.copy()
            if not bbox.failed:
                cv2.rectangle(
                    left,
                    (bbox.x1, bbox.y1),
                    (bbox.x2, bbox.y2),
                    BBOX_COLOR,
                    BBOX_THICKNESS,
                )

            # Right half: cropped, resized to match target height
            if not bbox.failed:
                cropped = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2, :]
                right = cv2.resize(cropped, (cropped_resized_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Fallback: show a dimmed copy of the original with "FAILED" text.
                right = cv2.resize(frame, (cropped_resized_w, target_h), interpolation=cv2.INTER_LINEAR)
                right = (right.astype(np.int16) // 2).astype(np.uint8)
                cv2.putText(
                    right, "FAILED", (20, target_h // 2),
                    LABEL_FONT, 1.5, (0, 0, 255), 3, cv2.LINE_AA,
                )

            # Top label bar
            label_bar = np.zeros((LABEL_BAR_HEIGHT, final_w, 3), dtype=np.uint8)
            cv2.putText(
                label_bar, label_text, (10, int(LABEL_BAR_HEIGHT * 0.7)),
                LABEL_FONT, LABEL_SCALE, (255, 255, 255), LABEL_THICKNESS, cv2.LINE_AA,
            )

            # Stitch: label_bar on top, [left | right] below
            body = np.hstack([left, right])
            full = np.vstack([label_bar, body])
            writer.write(full)
    finally:
        writer.release()

    return final_w, final_h


def main() -> None:
    if not TRAIN_TSV.exists():
        raise FileNotFoundError(f"TSV not found: {TRAIN_TSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {TRAIN_TSV}")
    df = pd.read_csv(TRAIN_TSV, sep="\t")
    if "source" not in df.columns:
        raise RuntimeError("TSV missing 'source' column.")
    print(f"Total clips: {len(df)}   sources: {df['source'].value_counts().to_dict()}")

    samples = pick_sample_rows(df, NUM_PER_SOURCE, SEED)
    print(f"\nSelected {len(samples)} clips (seed={SEED}):")
    for i, row in samples.iterrows():
        print(f"  [{i}] {row['source']:10s} | {row['vid']}")

    cropper = SignerCropper(
        sample_every_n=SAMPLE_EVERY_N,
        padding_ratio=PADDING_RATIO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        model_variant=MODEL_VARIANT,
    )

    n_success = 0
    n_failed = 0
    print()
    for i, row in samples.iterrows():
        vid = str(row["vid"])
        source = str(row["source"])
        video_path = str(row["file_path"])

        print(f"[{i + 1:2d}/{len(samples)}] {source:10s} | {vid}")
        print(f"         video : {video_path}")

        if not Path(video_path).exists():
            print(f"         SKIP (file not found)")
            n_failed += 1
            continue

        try:
            frames, fps = load_video_frames(video_path)
        except Exception as exc:
            print(f"         SKIP (load error: {exc})")
            n_failed += 1
            continue

        T, H, W, _ = frames.shape
        print(f"         frames: {T} @ {fps:.2f} fps   {W}x{H}")

        bbox = cropper.compute_bbox(frames)
        if bbox.failed:
            print(f"         BBOX : FAILED  (detected {bbox.num_detected_frames}/{bbox.num_sampled_frames} sampled)")
            n_failed += 1
        else:
            print(
                f"         BBOX : ({bbox.x1}, {bbox.y1}) -> ({bbox.x2}, {bbox.y2})  "
                f"size {bbox.width}x{bbox.height}  "
                f"detected {bbox.num_detected_frames}/{bbox.num_sampled_frames} "
                f"({100 * bbox.detection_rate:.0f}%)"
            )
            n_success += 1

        out_name = f"signer_crop_test_{i + 1:02d}_{source}_{vid}.mp4"
        # Sanitize filename — some vids contain characters illegal on Windows
        out_name = out_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        out_path = OUTPUT_DIR / out_name
        try:
            render_side_by_side(frames, bbox, source, vid, fps, out_path)
            print(f"         wrote : {out_path}")
        except Exception as exc:
            print(f"         WRITE ERROR: {exc}")

    print(f"\n{'=' * 60}")
    print(f"Summary: {n_success}/{len(samples)} successful, {n_failed} failed.")
    print(f"Videos saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    if n_failed > 1:
        print(
            "WARNING: More than 1 failure out of 10 samples.\n"
            "         Investigate before running the full extraction script.\n"
            "         Consider: lowering min_detection_confidence, "
            "increasing model_complexity, or checking the failed video files."
        )


if __name__ == "__main__":
    main()
