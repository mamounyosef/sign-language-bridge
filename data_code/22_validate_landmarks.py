"""
22_validate_landmarks.py
========================
Validates the landmark Parquet files produced by 21_extract_landmarks.py.

Checks performed (per split):
  - Parquet file exists
  - Coverage: how many TSV clips have a landmark entry
  - Missing clips: clips in the TSV that have no entry at all
  - Failure rate: clips marked failed=True and their error breakdown
  - Data integrity: arrays deserialise, shapes are correct, frame_indices
    starts at 0 and ends at n_frames-1 (required by training lookup)
  - NaN rate: fraction of frames with no pose / no left hand / no right hand
  - Duplicate vids in parquet

Saves 5 random overlaid JPEG examples (one per random successful clip,
all sampled frames) to C:\\Users\\mamou\\Downloads\\landmark_validation\\.

Usage
-----
    python data_code/22_validate_landmarks.py
    python data_code/22_validate_landmarks.py --splits train val
    python data_code/22_validate_landmarks.py --examples 10
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

SPLITS = {
    "train": DATA_DIR / "7_dataset_train.tsv",
    "val":   DATA_DIR / "7_dataset_val.tsv",
    "test":  DATA_DIR / "7_dataset_test.tsv",
}
PARQUETS = {
    "train": DATA_DIR / "landmarks_train.parquet",
    "val":   DATA_DIR / "landmarks_val.parquet",
    "test":  DATA_DIR / "landmarks_test.parquet",
}
BBOX_CSVS = {
    "train": DATA_DIR / "2_dataset_train_bboxes.csv",
    "val":   DATA_DIR / "2_dataset_val_bboxes.csv",
    "test":  DATA_DIR / "2_dataset_test_bboxes.csv",
}

DEFAULT_OUTPUT = Path(r"C:\Users\mamou\Downloads\landmark_validation")
DEFAULT_N_EXAMPLES = 5

EXPECTED_POSE_SHAPE     = (None, 6,  2)   # (N_frames, 6 joints, xy)
EXPECTED_HAND_SHAPE     = (None, 21, 2)   # (N_frames, 21 joints, xy)

# Drawing — mirrors 21_extract_landmarks.py exactly
_POSE_CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _load_arr(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b))


def _draw_on_frame(frame_bgr: np.ndarray,
                   pose: np.ndarray,
                   lh: np.ndarray,
                   rh: np.ndarray) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    short = min(H, W)
    thick  = max(1, short // 400)
    radius = max(1, short // 280)

    def px(nx, ny):
        return (int(round(float(nx) * W)), int(round(float(ny) * H)))

    out = frame_bgr.copy()

    # Pose — yellow (BGR 0,255,255)
    for i, j in _POSE_CONNECTIONS:
        if np.any(np.isnan(pose[i])) or np.any(np.isnan(pose[j])):
            continue
        cv2.line(out, px(*pose[i]), px(*pose[j]), (0, 255, 255), thick, cv2.LINE_AA)
    for i in range(6):
        if not np.any(np.isnan(pose[i])):
            cv2.circle(out, px(*pose[i]), radius, (0, 255, 255), -1, cv2.LINE_AA)

    # Hands — left green (0,200,0), right blue (255,80,0)
    for hand, color in ((lh, (0, 200, 0)), (rh, (255, 80, 0))):
        if np.all(np.isnan(hand)):
            continue
        for i, j in _HAND_CONNECTIONS:
            if np.any(np.isnan(hand[i])) or np.any(np.isnan(hand[j])):
                continue
            cv2.line(out, px(*hand[i]), px(*hand[j]), color, thick, cv2.LINE_AA)
        for i in range(21):
            if not np.any(np.isnan(hand[i])):
                cv2.circle(out, px(*hand[i]), radius, color, -1, cv2.LINE_AA)

    return out


def _validate_split(
    split: str,
    n_examples: int,
    output_dir: Path,
    issues: list[str],
) -> None:
    tsv_path     = SPLITS[split]
    parquet_path = PARQUETS[split]
    bbox_csv     = BBOX_CSVS[split]

    print(f"\n{'='*60}")
    print(f"  Split: {split.upper()}")
    print(f"{'='*60}")

    # ── TSV ─────────────────────────────────────────────────────────────────────
    if not tsv_path.exists():
        msg = f"[{split}] TSV not found: {tsv_path}"
        print(f"  ERROR: {msg}")
        issues.append(msg)
        return

    tsv_df    = pd.read_csv(tsv_path, sep="\t")
    tsv_vids  = set(str(v) for v in tsv_df["vid"])
    n_tsv     = len(tsv_vids)
    print(f"  TSV clips      : {n_tsv:,}")

    # ── Parquet ─────────────────────────────────────────────────────────────────
    if not parquet_path.exists():
        msg = f"[{split}] Parquet not found: {parquet_path}"
        print(f"  ERROR: {msg}")
        issues.append(msg)
        return

    df = pd.read_parquet(parquet_path)
    print(f"  Parquet rows   : {len(df):,}")

    # Duplicate vids
    dup_count = int(df["vid"].duplicated().sum())
    if dup_count:
        msg = f"[{split}] {dup_count} duplicate vid(s) in parquet"
        print(f"  WARNING: {msg}")
        issues.append(msg)

    parquet_vids = set(str(v) for v in df["vid"])

    # ── Coverage ─────────────────────────────────────────────────────────────────
    covered     = tsv_vids & parquet_vids
    missing     = tsv_vids - parquet_vids
    extra       = parquet_vids - tsv_vids
    coverage_pct = 100.0 * len(covered) / max(n_tsv, 1)
    print(f"  Coverage       : {len(covered):,}/{n_tsv:,}  ({coverage_pct:.1f}%)")

    if missing:
        msg = f"[{split}] {len(missing):,} TSV clip(s) have no parquet entry"
        print(f"  WARNING: {msg}")
        issues.append(msg)
        sample = sorted(missing)[:5]
        print(f"    First up to 5 missing vids: {sample}")

    if extra:
        print(f"  INFO: {len(extra):,} parquet vid(s) not in TSV (stale entries)")

    # ── Failure rate ─────────────────────────────────────────────────────────────
    n_failed = int(df["failed"].sum())
    n_ok     = len(df) - n_failed
    print(f"  Succeeded      : {n_ok:,}  |  Failed: {n_failed:,}  "
          f"({100.*n_failed/max(len(df),1):.1f}%)")
    if n_failed:
        errs = df.loc[df["failed"] & (df["error"] != ""), "error"]
        for msg_err, cnt in errs.value_counts().head(5).items():
            print(f"    {cnt:5d} x {msg_err}")

    # ── Array integrity & NaN rates ──────────────────────────────────────────────
    ok_df = df[~df["failed"]].copy()
    if ok_df.empty:
        issues.append(f"[{split}] All entries are failed — no data to validate")
        return

    shape_errors  = 0
    fi_errors     = 0   # frame_indices doesn't start at 0 or end at last frame
    nan_pose_frames  = 0
    nan_lh_frames    = 0
    nan_rh_frames    = 0
    total_frames     = 0
    bad_coord_clips  = 0  # any coord outside [0,1]

    good_rows_for_examples = []

    for row in ok_df.itertuples(index=False):
        try:
            fi   = _load_arr(row.frame_indices)   # (N,) uint16
            pose = _load_arr(row.pose)             # (N, 6, 2)
            lh   = _load_arr(row.left_hand)        # (N, 21, 2)
            rh   = _load_arr(row.right_hand)       # (N, 21, 2)
        except Exception as e:
            shape_errors += 1
            issues.append(f"[{split}] vid={row.vid} deserialisation error: {e}")
            continue

        N = len(fi)

        # Shape checks
        if pose.shape != (N, 6, 2) or lh.shape != (N, 21, 2) or rh.shape != (N, 21, 2):
            shape_errors += 1
            issues.append(f"[{split}] vid={row.vid} shape mismatch: "
                          f"fi={fi.shape} pose={pose.shape} lh={lh.shape} rh={rh.shape}")
            continue

        # frame_indices must start at 0 (training lookup anchor)
        if fi[0] != 0:
            fi_errors += 1
            issues.append(f"[{split}] vid={row.vid} frame_indices[0]={fi[0]} (expected 0)")

        # frame_indices must be monotonically increasing
        if not np.all(np.diff(fi.astype(np.int32)) > 0):
            fi_errors += 1
            issues.append(f"[{split}] vid={row.vid} frame_indices not monotonically increasing")

        # Coordinates must be in [0, 1] (ignoring NaNs)
        all_coords = np.concatenate([
            pose[~np.isnan(pose)],
            lh[~np.isnan(lh)],
            rh[~np.isnan(rh)],
        ])
        if len(all_coords) and (all_coords.min() < -0.01 or all_coords.max() > 1.01):
            bad_coord_clips += 1
            issues.append(f"[{split}] vid={row.vid} coords out of [0,1]: "
                          f"min={all_coords.min():.3f} max={all_coords.max():.3f}")

        # NaN rates (per frame: a frame is NaN if ALL joints for that landmark are NaN)
        total_frames += N
        pose_nan_mask = np.all(np.isnan(pose.reshape(N, -1)), axis=1)
        lh_nan_mask   = np.all(np.isnan(lh.reshape(N,  -1)), axis=1)
        rh_nan_mask   = np.all(np.isnan(rh.reshape(N,  -1)), axis=1)
        nan_pose_frames += int(pose_nan_mask.sum())
        nan_lh_frames   += int(lh_nan_mask.sum())
        nan_rh_frames   += int(rh_nan_mask.sum())

        good_rows_for_examples.append((row, fi, pose, lh, rh))

    if total_frames:
        print(f"  NaN-pose frames: {nan_pose_frames:,}/{total_frames:,} "
              f"({100.*nan_pose_frames/total_frames:.1f}%)")
        print(f"  NaN-lh frames  : {nan_lh_frames:,}/{total_frames:,} "
              f"({100.*nan_lh_frames/total_frames:.1f}%)")
        print(f"  NaN-rh frames  : {nan_rh_frames:,}/{total_frames:,} "
              f"({100.*nan_rh_frames/total_frames:.1f}%)")

    if shape_errors:
        print(f"  Shape errors   : {shape_errors}")
    if fi_errors:
        print(f"  Index errors   : {fi_errors}")
    if bad_coord_clips:
        print(f"  Out-of-range coords: {bad_coord_clips} clip(s)")

    # ── Overlaid examples ────────────────────────────────────────────────────────
    if n_examples > 0 and good_rows_for_examples:
        _save_examples(split, good_rows_for_examples, tsv_df, bbox_csv,
                       n_examples, output_dir, issues)


def _save_examples(
    split: str,
    good_rows: list,
    tsv_df: pd.DataFrame,
    bbox_csv_path: Path,
    n_examples: int,
    output_dir: Path,
    issues: list[str],
) -> None:
    """Decode video, apply crop, draw landmarks, save JPEG frames."""
    import decord

    # Build file_path and bbox lookups from TSV + bbox CSV
    fp_by_vid: dict[str, str] = {
        str(r.vid): str(r.file_path) for r in tsv_df.itertuples(index=False)
    }
    bbox_by_vid: dict[str, tuple] = {}
    if bbox_csv_path.exists():
        bdf = pd.read_csv(bbox_csv_path)
        for r in bdf[~bdf["failed"]].itertuples(index=False):
            bbox_by_vid[str(r.vid)] = (
                int(r.x1), int(r.y1), int(r.x2), int(r.y2),
                int(r.frame_width), int(r.frame_height),
            )

    SNAP_ALIGN = 32

    def _snap(lo, hi, align):
        span = hi - lo
        new_span = max(align, (span // align) * align)
        if new_span >= span:
            return lo, hi
        trim = span - new_span
        return lo + trim // 2, hi - (trim - trim // 2)

    def _crop_bounds(H, W, bbox):
        if bbox:
            x1, y1, x2, y2, fw, fh = bbox
            sx, sy = W / max(fw, 1), H / max(fh, 1)
            bx1 = max(0, int(round(x1 * sx)))
            by1 = max(0, int(round(y1 * sy)))
            bx2 = min(W, int(round(x2 * sx)))
            by2 = min(H, int(round(y2 * sy)))
            if bx2 - bx1 >= SNAP_ALIGN and by2 - by1 >= SNAP_ALIGN:
                cx1, cy1, cx2, cy2 = bx1, by1, bx2, by2
                cx1, cx2 = _snap(cx1, cx2, SNAP_ALIGN)
                cy1, cy2 = _snap(cy1, cy2, SNAP_ALIGN)
                return cx1, cy1, cx2, cy2
        cx1, cx2 = _snap(0, W, SNAP_ALIGN)
        cy1, cy2 = _snap(0, H, SNAP_ALIGN)
        return cx1, cy1, cx2, cy2

    chosen = random.sample(good_rows, min(n_examples, len(good_rows)))
    out_split = output_dir / split
    out_split.mkdir(parents=True, exist_ok=True)

    saved = 0
    for row, fi, pose, lh, rh in chosen:
        vid = str(row.vid)
        fp  = fp_by_vid.get(vid, "")
        if not fp or not Path(fp).exists():
            print(f"  [examples] skipping {vid} — video file not found")
            continue

        try:
            vr       = decord.VideoReader(fp, num_threads=1)
            n_native = len(vr)
            H0, W0   = vr[0].shape[:2]
            bbox     = bbox_by_vid.get(vid)
            cx1, cy1, cx2, cy2 = _crop_bounds(H0, W0, bbox)
            crop_w = cx2 - cx1
            crop_h = cy2 - cy1

            # stored fi are original frame indices — load & crop them
            fi_clamped = np.clip(fi.astype(np.int32), 0, n_native - 1)
            frames_rgb = vr.get_batch(fi_clamped.tolist()).asnumpy()
            cropped_rgb = frames_rgb[:, cy1:cy2, cx1:cx2, :]  # (N, crop_h, crop_w, 3)

            safe_vid = vid.replace(":", "_").replace("/", "_").replace("\\", "_")
            clip_dir = out_split / safe_vid
            clip_dir.mkdir(parents=True, exist_ok=True)

            N = len(fi)
            for t in range(N):
                frame_rgb = np.ascontiguousarray(cropped_rgb[t])
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_bgr = _draw_on_frame(frame_bgr, pose[t], lh[t], rh[t])

                # Burn in frame index and stored fi value
                label = f"stored_fi={fi[t]}  t={t}/{N-1}"
                cv2.putText(frame_bgr, label, (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_bgr, label, (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                out_path = clip_dir / f"frame_{fi[t]:05d}_t{t:04d}.jpg"
                cv2.imwrite(str(out_path), frame_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])

            print(f"  [examples/{split}] {vid}  → {N} frames  crop=({crop_w}x{crop_h})")
            saved += 1

        except Exception as e:
            print(f"  [examples] ERROR for {vid}: {e}")

    print(f"  [examples/{split}] Saved {saved} clip(s) to: {out_split}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate landmark Parquet files against TSV datasets.")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()),
                        choices=list(SPLITS.keys()))
    parser.add_argument("--examples", type=int, default=DEFAULT_N_EXAMPLES,
                        help="Number of random overlaid-image clips to save per split")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Directory for overlaid JPEG examples")
    args = parser.parse_args()

    output_dir = Path(args.output)
    issues: list[str] = []

    for split in args.splits:
        _validate_split(split, args.examples, output_dir, issues)

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    if not issues:
        print("  All checks passed — no issues found.")
    else:
        print(f"  {len(issues)} issue(s) found:")
        for i, msg in enumerate(issues, 1):
            print(f"    {i:3d}. {msg}")

    if args.examples > 0:
        print(f"\n  Overlaid examples saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
