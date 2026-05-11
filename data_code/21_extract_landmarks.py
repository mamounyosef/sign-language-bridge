"""
21_extract_landmarks.py
=======================
Offline extraction of per-frame pose and hand landmarks for every clip, for
all three dataset splits. Writes three Parquet files consumed by the training
collator to draw a skeleton overlay on cropped frames before the model sees them.

Backend: rtmlib + RTMPose (ONNX Runtime GPU) — ~10-20x faster than MediaPipe
on a CUDA-capable GPU. Each worker loads one Wholebody model instance that
returns body + both hands in a single forward pass with correct left/right
identification via body-context (no manual tracking heuristics needed).

Install:
    pip install rtmlib
    pip uninstall onnxruntime          # remove CPU-only version if present
    pip install onnxruntime-gpu        # CUDA-accelerated ONNX Runtime

Consistency guarantee
---------------------
Applies EXACTLY the same crop logic as ``_apply_signer_crop`` in
``qwen3vl_training.py`` (32-pixel aligned bbox). Landmarks are stored as
normalized (0.0-1.0) coordinates relative to the cropped frame so they
scale to any resolution at draw time: px = norm_x * crop_W.

Landmark layout
---------------
Pose: 6 upper-body joints from RTMPose COCO-WholeBody indices [5,6,7,8,9,10]:
    position 0 -> index 5  (left shoulder)
    position 1 -> index 6  (right shoulder)
    position 2 -> index 7  (left elbow)
    position 3 -> index 8  (right elbow)
    position 4 -> index 9  (left wrist)
    position 5 -> index 10 (right wrist)

Hands: 21 keypoints per hand (COCO-WholeBody / MediaPipe-compatible layout).
    Left hand  -> COCO-WholeBody indices 91-111
    Right hand -> COCO-WholeBody indices 112-132
    Left/right is determined by the model using body-pose context (reliable).

Parquet schema (one row per clip):
    vid, source, crop_w, crop_h, orig_fps,
    frame_indices (bytes), pose (bytes), left_hand (bytes), right_hand (bytes),
    failed, error

Arrays are serialized via np.save -> bytes for compact, fast storage.

Usage
-----
    # Test on 5 random clips, save overlaid images:
    python data_code/21_extract_landmarks.py --test --n 5 --output "C:/Users/mamou/Downloads/test"

    # Full extraction (all splits):
    python data_code/21_extract_landmarks.py

    # Re-extract even if Parquet already exists:
    python data_code/21_extract_landmarks.py --overwrite
"""

from __future__ import annotations

import argparse
import io
import math
import multiprocessing as mp_proc
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Optional


def _add_torch_cuda_dlls() -> None:
    """Point Windows DLL loader at PyTorch's bundled cuDNN/CUDA libraries.

    onnxruntime-gpu requires cudnn64_9.dll which isn't on PATH by default.
    PyTorch (already installed for training) ships with cuDNN 9, so we add
    its lib directory to the Windows DLL search path before any ONNX Runtime
    session is created.  No-op on non-Windows or if torch is not installed.
    """
    if os.name != 'nt':
        return
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / 'lib'
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass


# Must be called before any onnxruntime import so the DLL loader finds cuDNN.
_add_torch_cuda_dlls()

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # ── Data paths (local) ──
    'data_dir':    PROJECT_ROOT / 'data',
    'splits': {
        'train': PROJECT_ROOT / 'data' / '7_dataset_train.tsv',
        'val':   PROJECT_ROOT / 'data' / '7_dataset_val.tsv',
        'test':  PROJECT_ROOT / 'data' / '7_dataset_test.tsv',
    },
    'bbox_csv': PROJECT_ROOT / 'data' / 'bboxes_all.csv',
    'output_parquets': {
        'train': PROJECT_ROOT / 'data' / 'landmarks_train.parquet',
        'val':   PROJECT_ROOT / 'data' / 'landmarks_val.parquet',
        'test':  PROJECT_ROOT / 'data' / 'landmarks_test.parquet',
    },

    # ── Landmark extraction settings ──
    # Cap stored landmark density to this FPS.  For clips whose native FPS is
    # above this value, every ceil(native_fps / TARGET_SAMPLE_FPS)-th frame is
    # sampled.  20 FPS gives headroom above the training decoder's 15 FPS.
    'target_sample_fps': 20,
    'snap_align': 32,           # must match CollatorForVideoTextChat._snap_align

    # RTMPose model quality: 'performance' (best accuracy) or 'balanced' (faster).
    'rtmpose_mode': 'performance',

    # Confidence thresholds
    'pose_conf_threshold': 0.3,  # joints below this → NaN
    'hand_conf_threshold': 0.2,  # whole hand rejected if mean conf below this

    # ── Local runtime ──
    # Worker processes for the local multiprocessing pool.
    # Each loads one RTMPose model instance; keep ≤ 4 on an 8 GB VRAM GPU.
    'local_workers': 2,

    # ── Modal ──────────────────────────────────────────────────────────────────
    # Set 'is_modal': True to run on Modal's cloud GPUs instead of locally.
    # Run with:  modal run data_code/21_extract_landmarks.py
    # All data paths are remapped to Modal volume mount points automatically.
    'is_modal': True,

    # Hardware
    # GPU: A10G (24 GB VRAM) — RTMPose performance model uses ~1.5 GB/worker,
    #   giving room for 8+ parallel workers while staying well within budget.
    #   Significantly cheaper than A100 for a data-processing job.
    # CPU: 16 vCPUs — feeds 8 video-decoding (decord) workers without starving the GPU.
    # RAM: 32 GB — comfortable for video frame buffers and NumPy arrays across 8 workers.
    # GPU strings: 'T4', 'A10G', 'A100', 'A100-80GB', 'H100'
    'modal_gpu':     'A10G',
    'modal_cpu':     16,
    'modal_memory':  32,          # GB (converted to MiB when passed to Modal)
    'modal_timeout': 6 * 3600,   # Max wall-clock seconds; 6 h covers a full 3-split run
    'modal_retries': 0,

    # Modal worker count: how many parallel RTMPose instances inside the container.
    # A10G (24 GB VRAM) comfortably fits 8 workers with the performance model.
    # Increase to 10 if memory headroom allows; reduce to 4 for T4 (16 GB VRAM).
    'modal_workers': 8,

    # App
    'modal_app_name': 'signbridge-landmarks',

    # Volumes — shared with the training script (create once with `modal volume create <name>`)
    #   signbridge-data   : TSV files, bboxes_all.csv (READ only here)
    #   signbridge-videos : raw video clips (READ only)
    #   signbridge-outputs: landmark parquets are WRITTEN here (same volume as training checkpoints)
    'modal_data_volume':   'signbridge-data',
    'modal_video_volume':  'signbridge-videos',
    'modal_output_volume': 'signbridge-outputs',

    # Mount paths inside the container
    'modal_data_mount':   '/data',
    'modal_video_mount':  '/videos',
    'modal_output_mount': '/outputs',
}

# ── Modal path overrides ──────────────────────────────────────────────────────
# Applied when is_modal=True. Reads come from signbridge-data and signbridge-videos;
# output parquets are written to signbridge-outputs (separate from the old lower-quality
# parquets still sitting in signbridge-data).
#
# Volume upload commands (run once — same volumes as training, no re-upload needed):
#
#   modal volume put signbridge-data "data/7_dataset_train.tsv"        /data/7_dataset_train.tsv
#   modal volume put signbridge-data "data/7_dataset_val.tsv"          /data/7_dataset_val.tsv
#   modal volume put signbridge-data "data/7_dataset_test.tsv"         /data/7_dataset_test.tsv
#   modal volume put signbridge-data "data/bboxes_all.csv"             /bboxes_all.csv
#
#   # Video clips (mirrors D:\signbridge_dataset\)
#   modal volume put signbridge-videos "D:/signbridge_dataset/how2sign" /how2sign
#   modal volume put signbridge-videos "D:/signbridge_dataset/openasl"  /openasl
#
# Output landmark parquets land in signbridge-outputs at:
#   /outputs/landmarks_train.parquet
#   /outputs/landmarks_val.parquet
#   /outputs/landmarks_test.parquet
# After the job, download with:
#   modal volume get signbridge-outputs /outputs/landmarks_train.parquet data/
#   modal volume get signbridge-outputs /outputs/landmarks_val.parquet   data/
#   modal volume get signbridge-outputs /outputs/landmarks_test.parquet  data/
if CONFIG['is_modal']:
    _MDATA = Path(CONFIG['modal_data_mount'])
    _MOUT  = Path(CONFIG['modal_output_mount'])
    CONFIG.update({
        'splits': {
            'train': _MDATA / 'data' / '7_dataset_train.tsv',
            'val':   _MDATA / 'data' / '7_dataset_val.tsv',
            'test':  _MDATA / 'data' / '7_dataset_test.tsv',
        },
        'bbox_csv': _MDATA / 'bboxes_all.csv',
        'output_parquets': {
            'train': _MOUT / 'landmarks_train.parquet',
            'val':   _MOUT / 'landmarks_val.parquet',
            'test':  _MOUT / 'landmarks_test.parquet',
        },
    })
# ─────────────────────────────────────────────────────────────────────────────

# Derive module-level constants from CONFIG so the rest of the code is unchanged.
DATA_DIR         = CONFIG['data_dir']
SPLITS           = CONFIG['splits']
BBOX_CSV         = CONFIG['bbox_csv']
OUTPUT_PARQUETS  = CONFIG['output_parquets']
TARGET_SAMPLE_FPS = CONFIG['target_sample_fps']
SNAP_ALIGN       = CONFIG['snap_align']
DEFAULT_WORKERS  = CONFIG['local_workers'] if not CONFIG['is_modal'] else CONFIG['modal_workers']
RTMPOSE_MODE     = CONFIG['rtmpose_mode']

# RTMPose COCO-WholeBody upper-body indices (shoulders, elbows, wrists)
POSE_INDICES     = [5, 6, 7, 8, 9, 10]
LEFT_HAND_SLICE  = slice(91, 112)   # 21 keypoints
RIGHT_HAND_SLICE = slice(112, 133)  # 21 keypoints

POSE_CONF_THRESHOLD = CONFIG['pose_conf_threshold']
HAND_CONF_THRESHOLD = CONFIG['hand_conf_threshold']
# ──────────────────────────────────────────────────────────────────────────────


def _remap_path_for_modal(fp: str) -> str:
    """Convert a local Windows file_path from the TSV to its Modal volume path.

    D:\\signbridge_dataset\\how2sign\\train\\foo.mp4  →  /videos/how2sign/train/foo.mp4
    D:\\signbridge_dataset\\openasl\\train\\foo.mp4   →  /videos/openasl/train/foo.mp4
    """
    video_mount = CONFIG['modal_video_mount']
    p = Path(fp.replace('\\', '/'))
    parts = [x for x in p.parts if x not in ('', '/') and ':' not in x]
    # Expected: ['signbridge_dataset', <source>, <split>, <filename>]
    if len(parts) < 4:
        return fp
    source   = parts[1]   # 'how2sign' or 'openasl'
    split    = parts[2]   # 'train', 'val', 'test'
    filename = parts[3]
    return f'{video_mount}/{source}/{split}/{filename}'


# ── Crop logic (mirrors _apply_signer_crop exactly) ──────────────────────────

def _snap_down_to_align(lo: int, hi: int, align: int) -> tuple[int, int]:
    """Shrink [lo, hi) so (hi-lo) is a multiple of align, centred."""
    span = hi - lo
    new_span = max(align, (span // align) * align)
    if new_span >= span:
        return lo, hi
    trim  = span - new_span
    left  = trim // 2
    right = trim - left
    return lo + left, hi - right


def _compute_crop(
    frames: np.ndarray,
    bbox: Optional[tuple],
) -> tuple[np.ndarray, int, int, int, int]:
    """Apply the same bbox crop as _apply_signer_crop to a (T, H, W, C) array."""
    T, H, W = frames.shape[:3]
    cx1, cy1, cx2, cy2 = 0, 0, W, H
    bbox_used = False

    if bbox is not None:
        x1, y1, x2, y2, fw, fh = bbox
        sx  = W / max(fw, 1)
        sy  = H / max(fh, 1)
        bx1 = max(0, int(round(x1 * sx)))
        by1 = max(0, int(round(y1 * sy)))
        bx2 = min(W, int(round(x2 * sx)))
        by2 = min(H, int(round(y2 * sy)))
        if bx2 - bx1 >= SNAP_ALIGN and by2 - by1 >= SNAP_ALIGN:
            cx1, cy1, cx2, cy2 = bx1, by1, bx2, by2
            bbox_used = True

    cx1, cx2 = _snap_down_to_align(cx1, cx2, SNAP_ALIGN)
    cy1, cy2 = _snap_down_to_align(cy1, cy2, SNAP_ALIGN)

    if not bbox_used and cx1 == 0 and cy1 == 0 and cx2 == W and cy2 == H:
        return frames, cx1, cy1, cx2, cy2

    return frames[:, cy1:cy2, cx1:cx2, :], cx1, cy1, cx2, cy2


# ── Array serialization ───────────────────────────────────────────────────────

def _arr_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ── Worker state ──────────────────────────────────────────────────────────────

_WORKER_MODEL = None  # type: ignore[var-annotated]


def _worker_init() -> None:
    """Load only the RTMPose pose model (no YOLOX detector) per worker.

    Since frames are already tightly cropped around the signer, person
    detection is unnecessary.  RTMPose accepts bboxes=[] which makes it
    treat the full frame as the person region — identical to what YOLOX
    would return on a single-person crop.  Skipping YOLOX removes an
    entire GPU forward pass per frame, roughly halving inference time.
    """
    global _WORKER_MODEL
    from rtmlib import RTMPose, Wholebody
    mode_cfg = Wholebody.MODE[RTMPOSE_MODE]
    _WORKER_MODEL = RTMPose(
        mode_cfg['pose'],
        model_input_size=mode_cfg['pose_input_size'],
        to_openpose=False,
        backend='onnxruntime',
        device='cuda',
    )


# ── Per-frame inference ───────────────────────────────────────────────────────

def _run_rtmpose_on_frame(
    frame_rgb: np.ndarray,
    crop_h: int,
    crop_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run RTMPose Wholebody on one cropped RGB frame (converts RGB→BGR internally)."""
    frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    return _run_rtmpose_on_frame_bgr(frame_bgr, crop_h, crop_w)


def _run_rtmpose_on_frame_bgr(
    frame_bgr: np.ndarray,
    crop_h: int,
    crop_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run RTMPose Wholebody on one cropped BGR frame (no color conversion).

    Returns normalized (0.0-1.0) coordinates relative to the crop:
        pose_row: (6, 2)  float32 — upper-body joints, NaN where below threshold
        lh_row:   (21, 2) float32 — left hand, all-NaN if not detected
        rh_row:   (21, 2) float32 — right hand, all-NaN if not detected
    """
    nan6  = np.full((6, 2),  np.nan, dtype=np.float32)
    nan21 = np.full((21, 2), np.nan, dtype=np.float32)

    try:
        # bboxes=[] → RTMPose uses the full frame as the person bbox.
        # This skips YOLOX detection (valid because frames are already
        # tightly cropped around the signer — one person fills the frame).
        keypoints, scores = _WORKER_MODEL(frame_bgr, bboxes=[])
        # keypoints: (N_persons, 133, 2) — pixel (x, y)
        # scores:    (N_persons, 133)    — confidence [0, 1]

        if len(keypoints) == 0:
            return nan6.copy(), nan21.copy(), nan21.copy()

        kps = keypoints[0].astype(np.float32)  # (133, 2) pixel coords
        scr = scores[0].astype(np.float32)     # (133,)

        # Normalize pixel coords to [0, 1] relative to crop dimensions
        kps_norm = kps.copy()
        kps_norm[:, 0] = np.clip(kps_norm[:, 0] / max(crop_w, 1), 0.0, 1.0)
        kps_norm[:, 1] = np.clip(kps_norm[:, 1] / max(crop_h, 1), 0.0, 1.0)

        # Pose: 6 upper-body joints — mark low-confidence joints as NaN
        pose_row = nan6.copy()
        for out_i, coco_i in enumerate(POSE_INDICES):
            if scr[coco_i] >= POSE_CONF_THRESHOLD:
                pose_row[out_i] = kps_norm[coco_i]

        # Left hand: accept hand if mean confidence meets threshold,
        # then NaN-out individual joints that are below threshold.
        lh_scores = scr[LEFT_HAND_SLICE]
        if np.mean(lh_scores) >= HAND_CONF_THRESHOLD:
            lh_row = kps_norm[LEFT_HAND_SLICE].copy()
            lh_row[lh_scores < HAND_CONF_THRESHOLD] = np.nan
        else:
            lh_row = nan21.copy()

        # Right hand
        rh_scores = scr[RIGHT_HAND_SLICE]
        if np.mean(rh_scores) >= HAND_CONF_THRESHOLD:
            rh_row = kps_norm[RIGHT_HAND_SLICE].copy()
            rh_row[rh_scores < HAND_CONF_THRESHOLD] = np.nan
        else:
            rh_row = nan21.copy()

        return pose_row, lh_row, rh_row

    except Exception as e:
        print(f"  [rtmpose] exception: {e}", flush=True)
        return nan6.copy(), nan21.copy(), nan21.copy()


# ── Temporal post-processing ──────────────────────────────────────────────────

def _hand_centroid(hand: np.ndarray) -> np.ndarray | None:
    """Mean (x,y) of non-NaN landmarks in a (21,2) row, or None if all NaN."""
    valid = hand[~np.any(np.isnan(hand), axis=1)]
    return valid.mean(axis=0) if len(valid) else None


def _reject_spikes(arr: np.ndarray, threshold: float = 0.12) -> np.ndarray:
    """Mark frames whose centroid deviates from its neighbours' midpoint as NaN.

    Catches brief teleport detections that passed the jump threshold.
    threshold=0.12 = 12% of the crop dimension.
    """
    out = arr.copy()
    N   = arr.shape[0]
    for t in range(1, N - 1):
        c_prev = _hand_centroid(arr[t - 1])
        c_curr = _hand_centroid(arr[t])
        c_next = _hand_centroid(arr[t + 1])
        if c_prev is None or c_curr is None or c_next is None:
            continue
        if np.linalg.norm(c_curr - (c_prev + c_next) / 2.0) > threshold:
            out[t] = np.nan
    return out


def _fill_landmark_gaps(arr: np.ndarray, max_gap: int = 8) -> np.ndarray:
    """Forward-fill NaN runs of <= max_gap frames per joint.

    Covers brief occlusions and the re-entry detection lag (hand leaving and
    coming back into frame). Gaps longer than max_gap stay NaN.
    """
    arr = arr.copy()
    N, K, _ = arr.shape
    for k in range(K):
        last_valid = None
        gap_start  = None
        for t in range(N):
            missing = bool(np.any(np.isnan(arr[t, k])))
            if not missing:
                if gap_start is not None and last_valid is not None:
                    if (t - gap_start) <= max_gap:
                        arr[gap_start:t, k] = last_valid
                gap_start  = None
                last_valid = arr[t, k].copy()
            else:
                if gap_start is None:
                    gap_start = t
    return arr


def _smooth_landmarks(arr: np.ndarray) -> np.ndarray:
    """3-frame weighted smoothing [0.25, 0.5, 0.25] along the time axis.

    Only applied where all three frames in the window are non-NaN.
    """
    if arr.shape[0] < 3:
        return arr
    out = arr.copy()
    for t in range(1, arr.shape[0] - 1):
        window = arr[t - 1 : t + 2]
        if not np.any(np.isnan(window)):
            out[t] = 0.25 * arr[t - 1] + 0.5 * arr[t] + 0.25 * arr[t + 1]
    return out


def _postprocess(pose: np.ndarray, lh: np.ndarray, rh: np.ndarray):
    """Full temporal post-processing pipeline: spike rejection -> gap fill -> smooth."""
    lh   = _smooth_landmarks(_fill_landmark_gaps(_reject_spikes(lh),   max_gap=8))
    rh   = _smooth_landmarks(_fill_landmark_gaps(_reject_spikes(rh),   max_gap=8))
    pose = _smooth_landmarks(_fill_landmark_gaps(pose, max_gap=8))
    return pose, lh, rh


# ── Drawing (for test mode) ───────────────────────────────────────────────────

_POSE_CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _draw_landmarks_on_frame(
    frame_rgb: np.ndarray,
    pose: np.ndarray,   # (6, 2)  normalized
    lh:   np.ndarray,   # (21, 2) normalized
    rh:   np.ndarray,   # (21, 2) normalized
) -> np.ndarray:
    """Draw skeleton onto an HWC RGB frame; return modified RGB copy."""
    H, W       = frame_rgb.shape[:2]
    short_side = min(H, W)
    thickness  = max(1, short_side // 400)
    radius     = max(1, short_side // 280)
    frame_bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()

    def to_px(nx, ny):
        return (int(round(float(nx) * W)), int(round(float(ny) * H)))

    # Pose skeleton (yellow — BGR: 0,255,255)
    for i, j in _POSE_CONNECTIONS:
        if np.any(np.isnan(pose[i])) or np.any(np.isnan(pose[j])):
            continue
        cv2.line(frame_bgr, to_px(*pose[i]), to_px(*pose[j]), (0, 255, 255), thickness, cv2.LINE_AA)
    for i in range(6):
        if not np.any(np.isnan(pose[i])):
            cv2.circle(frame_bgr, to_px(*pose[i]), radius, (0, 255, 255), -1, cv2.LINE_AA)

    # Left hand (green — BGR: 0,200,0), right hand (blue — BGR: 255,80,0)
    for hand, color in ((lh, (0, 200, 0)), (rh, (255, 80, 0))):
        if np.all(np.isnan(hand)):
            continue
        for i, j in _HAND_CONNECTIONS:
            if np.any(np.isnan(hand[i])) or np.any(np.isnan(hand[j])):
                continue
            cv2.line(frame_bgr, to_px(*hand[i]), to_px(*hand[j]), color, thickness, cv2.LINE_AA)
        for i in range(21):
            if not np.any(np.isnan(hand[i])):
                cv2.circle(frame_bgr, to_px(*hand[i]), radius, color, -1, cv2.LINE_AA)

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ── Worker function ───────────────────────────────────────────────────────────

def _process_one(task: tuple) -> dict:
    """Extract landmarks for one clip. Runs in a worker process."""
    vid, file_path, source, bbox = task

    base = {
        "vid": vid, "source": source,
        "crop_w": 0, "crop_h": 0, "orig_fps": 0.0,
        "frame_indices": b"", "pose": b"", "left_hand": b"", "right_hand": b"",
        "failed": True, "error": "",
    }

    if not Path(file_path).exists():
        base["error"] = "file_not_found"
        return base

    try:
        import decord
        vr       = decord.VideoReader(file_path, num_threads=1)
        orig_fps = float(vr.get_avg_fps())
        n_frames = len(vr)
        if n_frames == 0:
            base["error"] = "empty_video"
            return base

        # Compute which original frame indices to sample.
        # step=1 for clips at or below TARGET_SAMPLE_FPS; higher for faster clips.
        step = max(1, math.ceil(orig_fps / TARGET_SAMPLE_FPS))
        sampled_indices = list(range(0, n_frames, step))
        # Always include the very last frame so stored_fi[-1] == n_frames-1,
        # which is required by the proportional lookup in _apply_landmark_overlay.
        if not sampled_indices or sampled_indices[-1] != n_frames - 1:
            sampled_indices.append(n_frames - 1)
        N = len(sampled_indices)

        # Load ONLY the sampled frames — decord decodes exactly the requested indices.
        # Decord native bridge returns RGB uint8.
        frames_rgb = vr.get_batch(sampled_indices).asnumpy()   # (N, H, W, 3)

        # Apply bbox crop. _compute_crop uses shape[:3] for T,H,W; here T==N.
        # Crop bounds depend only on H,W, so the result is (N, crop_h, crop_w, 3).
        cropped_rgb, *_ = _compute_crop(frames_rgb, bbox)
        N, crop_h, crop_w = cropped_rgb.shape[:3]

        pose_arr = np.full((N, 6,  2), np.nan, dtype=np.float32)
        lh_arr   = np.full((N, 21, 2), np.nan, dtype=np.float32)
        rh_arr   = np.full((N, 21, 2), np.nan, dtype=np.float32)

        # Convert entire clip RGB→BGR once (avoids repeated per-frame Python overhead)
        cropped_bgr = np.ascontiguousarray(cropped_rgb[:, :, :, ::-1])

        for out_t in range(N):
            pose_arr[out_t], lh_arr[out_t], rh_arr[out_t] = \
                _run_rtmpose_on_frame_bgr(cropped_bgr[out_t], crop_h, crop_w)

        pose_arr, lh_arr, rh_arr = _postprocess(pose_arr, lh_arr, rh_arr)
        fi_arr = np.array(sampled_indices, dtype=np.uint16)

        has_any = (not np.all(np.isnan(pose_arr))) or \
                  (not np.all(np.isnan(lh_arr)))   or \
                  (not np.all(np.isnan(rh_arr)))

        base.update({
            "crop_w": int(crop_w), "crop_h": int(crop_h), "orig_fps": float(orig_fps),
            "frame_indices": _arr_to_bytes(fi_arr),
            "pose":          _arr_to_bytes(pose_arr),
            "left_hand":     _arr_to_bytes(lh_arr),
            "right_hand":    _arr_to_bytes(rh_arr),
            "failed": not has_any, "error": "",
        })

    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"

    return base


# ── Test mode ─────────────────────────────────────────────────────────────────

def _run_test_mode(n: int, output_dir: Path, split: str = "train") -> None:
    """Run on n random clips and save overlaid frames as JPEGs to output_dir."""
    print(f"\n[test] Running on {n} random clips from '{split}' split ...")

    tsv_path = SPLITS[split]

    if not tsv_path.exists():
        print(f"[test] ERROR: TSV not found: {tsv_path}"); return
    if not BBOX_CSV.exists():
        print(f"[test] ERROR: Bbox CSV not found: {BBOX_CSV}"); return

    df          = pd.read_csv(tsv_path, sep="\t")
    bbox_by_vid = _load_bbox_lookup()

    valid_rows = [
        r for r in df.itertuples(index=False)
        if str(r.vid) in bbox_by_vid and Path(str(r.file_path)).exists()
    ]
    if not valid_rows:
        print("[test] No valid clips found."); return

    chosen = random.sample(valid_rows, min(n, len(valid_rows)))
    _worker_init()   # load model in main process for test mode
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in chosen:
        vid       = str(row.vid)
        file_path = str(row.file_path)
        bbox      = bbox_by_vid.get(vid)
        print(f"[test] Processing {vid} ...")

        try:
            import decord
            vr       = decord.VideoReader(file_path, num_threads=1)
            orig_fps = float(vr.get_avg_fps())
            n_frames = len(vr)

            step = max(1, math.ceil(orig_fps / TARGET_SAMPLE_FPS))
            sampled_indices = list(range(0, n_frames, step))
            if not sampled_indices or sampled_indices[-1] != n_frames - 1:
                sampled_indices.append(n_frames - 1)
            N = len(sampled_indices)

            frames_rgb = vr.get_batch(sampled_indices).asnumpy()   # (N, H, W, 3)

            cropped_rgb, *_ = _compute_crop(frames_rgb, bbox)
            N, crop_h, crop_w = cropped_rgb.shape[:3]
            print(f"[test]   native_fps={orig_fps:.1f}  step={step}  "
                  f"sampled={N} frames  crop=({crop_w}x{crop_h})")

            pose_buf = np.full((N, 6,  2), np.nan, dtype=np.float32)
            lh_buf   = np.full((N, 21, 2), np.nan, dtype=np.float32)
            rh_buf   = np.full((N, 21, 2), np.nan, dtype=np.float32)

            cropped_bgr = np.ascontiguousarray(cropped_rgb[:, :, :, ::-1])
            for t_out in range(N):
                pose_buf[t_out], lh_buf[t_out], rh_buf[t_out] = \
                    _run_rtmpose_on_frame_bgr(cropped_bgr[t_out], crop_h, crop_w)

            pose_buf, lh_buf, rh_buf = _postprocess(pose_buf, lh_buf, rh_buf)

            n_pose = 0; n_lh = 0; n_rh = 0
            safe_vid = vid.replace(":", "_").replace("/", "_")
            clip_dir = output_dir / safe_vid
            clip_dir.mkdir(parents=True, exist_ok=True)

            for t_out, frame_idx in enumerate(sampled_indices):
                frame_rgb = np.ascontiguousarray(cropped_rgb[frame_idx])
                pose_row, lh_row, rh_row = pose_buf[t_out], lh_buf[t_out], rh_buf[t_out]
                if not np.all(np.isnan(pose_row)): n_pose += 1
                if not np.all(np.isnan(lh_row)):   n_lh += 1
                if not np.all(np.isnan(rh_row)):   n_rh += 1
                overlaid = _draw_landmarks_on_frame(frame_rgb, pose_row, lh_row, rh_row)
                out_path = clip_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(
                    str(out_path),
                    cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )

            print(f"[test]   Saved {N} frames  pose:{n_pose}/{N}  lh:{n_lh}/{N}  rh:{n_rh}/{N}")
            print(f"[test]   Output: {clip_dir}")

        except Exception as exc:
            print(f"[test]   ERROR for {vid}: {exc}")
            traceback.print_exc()

    print(f"\n[test] Done. Inspect output at: {output_dir}")


# ── Full extraction ────────────────────────────────────────────────────────────

def _load_bbox_lookup() -> dict[str, tuple]:
    """Load the unified bbox CSV into a vid -> (x1,y1,x2,y2,fw,fh) dict."""
    bbox_df = pd.read_csv(BBOX_CSV)
    bbox_ok = bbox_df[~bbox_df["failed"]]
    return {
        str(r.vid): (int(r.x1), int(r.y1), int(r.x2), int(r.y2),
                     int(r.frame_width), int(r.frame_height))
        for r in bbox_ok.itertuples(index=False)
    }


def _load_tasks(tsv_path: Path, bbox_by_vid: dict[str, tuple]) -> list[tuple]:
    df = pd.read_csv(tsv_path, sep="\t")
    remap = _remap_path_for_modal if CONFIG['is_modal'] else (lambda fp: fp)
    return [
        (str(r.vid), remap(str(r.file_path)), str(r.source), bbox_by_vid[str(r.vid)])
        for r in df.itertuples(index=False)
        if str(r.vid) in bbox_by_vid
    ]


CHECKPOINT_EVERY = 5   # flush to disk every N completed clips


def _save_checkpoint(results: list[dict], path: Path) -> None:
    """Atomically overwrite the parquet with all results accumulated so far.

    Writes to a temp file first then replaces, so a crash mid-write never
    leaves a corrupted parquet.
    """
    tmp = path.with_suffix(".parquet.tmp")
    df  = pd.DataFrame(results).sort_values("vid").reset_index(drop=True)
    df.to_parquet(tmp, index=False)
    tmp.replace(path)   # atomic on Windows (same volume)


def _load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    """Load an existing parquet checkpoint. Returns (rows, set_of_done_vids).

    Corrupt or missing files return empty state so the split starts fresh.
    """
    if not path.exists():
        return [], set()
    try:
        df = pd.read_parquet(path)
        rows     = [r for r in df.to_dict("records") if not r.get("failed", False)]
        done_ids = set(str(r["vid"]) for r in rows)
        return rows, done_ids
    except Exception as e:
        print(f"  [checkpoint] Could not load {path.name}: {e} — starting fresh.")
        return [], set()


def _run_split(
    split_name: str,
    tsv_path: Path,
    output_parquet: Path,
    num_workers: int,
    overwrite: bool = False,
) -> pd.DataFrame:
    all_tasks = _load_tasks(tsv_path, _load_bbox_lookup())

    # Resume from existing parquet unless --overwrite
    if overwrite:
        results:  list[dict] = []
        done_ids: set[str]   = set()
    else:
        results, done_ids = _load_checkpoint(output_parquet)

    tasks = [t for t in all_tasks if t[0] not in done_ids]

    print(f"\n[{split_name}] {tsv_path.name}: {len(all_tasks):,} total clips, "
          f"{len(done_ids):,} already done, {len(tasks):,} remaining")
    if not tasks:
        print(f"[{split_name}] Nothing to do.")
        return pd.DataFrame(results)

    print(f"[{split_name}] Using {num_workers} worker process(es), "
          f"checkpoint every {CHECKPOINT_EVERY} clips")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    t0          = time.time()
    since_save  = 0

    def _on_result(res: dict) -> None:
        nonlocal since_save
        results.append(res)
        since_save += 1
        if since_save >= CHECKPOINT_EVERY:
            _save_checkpoint(results, output_parquet)
            since_save = 0

    if num_workers <= 1:
        _worker_init()
        for task in tqdm(tasks, desc=split_name, unit="clip", file=sys.stdout):
            _on_result(_process_one(task))
    else:
        ctx = mp_proc.get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=_worker_init) as pool:
            for res in tqdm(
                pool.imap_unordered(_process_one, tasks, chunksize=2),
                total=len(tasks), desc=split_name, unit="clip", file=sys.stdout,
            ):
                _on_result(res)

    # Final save — guarantees everything is on disk
    _save_checkpoint(results, output_parquet)

    elapsed = time.time() - t0
    df    = pd.DataFrame(results).sort_values("vid").reset_index(drop=True)
    n_ok  = int((~df["failed"]).sum())
    n_fail = int(df["failed"].sum())
    print(f"[{split_name}] Done. {output_parquet.name}: "
          f"{n_ok:,} ok, {n_fail:,} failed — "
          f"{len(tasks)/max(elapsed, 1e-6):.1f} clips/sec this run ({elapsed/60:.1f} min)")
    if n_fail > 0:
        errs = df.loc[df["failed"] & (df["error"] != ""), "error"]
        for err_msg, count in errs.value_counts().head(5).items():
            print(f"[{split_name}]   {count:5d} x {err_msg}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RTMPose landmark extraction.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Worker processes (default {DEFAULT_WORKERS}). "
                             "Each loads one model instance; keep <=4 to stay within 8 GB VRAM.")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()),
                        choices=list(SPLITS.keys()))
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if output Parquet already exists.")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: run on --n random clips, save overlaid images.")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output", type=str, default=r"C:\Users\mamou\Downloads\test")
    parser.add_argument("--split", type=str, default="train", choices=list(SPLITS.keys()))
    args = parser.parse_args()

    print(f"PROJECT_ROOT  = {PROJECT_ROOT}")
    print(f"DATA_DIR      = {DATA_DIR}")
    print(f"RTMPose mode  = {RTMPOSE_MODE}")

    # Pre-download models in the main process so workers find them cached.
    # Wholebody downloads both YOLOX and RTMPose; workers use RTMPose only.
    # Use CPU here to avoid CUDA init in the main process; workers use CUDA.
    print("\nPre-downloading RTMPose models (first run only) ...")
    from rtmlib import RTMPose, Wholebody
    mode_cfg = Wholebody.MODE[RTMPOSE_MODE]
    _tmp = RTMPose(mode_cfg['pose'], model_input_size=mode_cfg['pose_input_size'],
                   backend='onnxruntime', device='cpu')
    del _tmp
    print("Models ready (pose model only — YOLOX detector skipped at runtime).")

    if args.test:
        _run_test_mode(args.n, Path(args.output), split=args.split)
        return

    print(f"Workers = {args.workers}")
    print(f"Splits  = {args.splits}")

    if not BBOX_CSV.exists():
        print(f"ERROR: unified bbox CSV not found: {BBOX_CSV}. Run 23_merge_bbox_csvs.py first.")
        return

    for split in args.splits:
        tsv = SPLITS[split]
        out = OUTPUT_PARQUETS[split]

        if not tsv.exists():
            print(f"[{split}] SKIP - TSV not found: {tsv}"); continue

        _run_split(split, tsv, out, args.workers, args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    mp_proc.freeze_support()
    main()


# ═══════════════════════════════════════════════════════════════════════════════
#  MODAL APP  (only active when is_modal=True)
#
#  Usage:
#    1. Set  'is_modal': True  in CONFIG above.
#    2. Run: modal run data_code/21_extract_landmarks.py
#
#  The container image is built once and cached.  The RTMPose ONNX models are
#  baked into the image at build time so cold-start containers need no download.
#
#  Volume layout (shared with qwen3vl_training.py — no re-upload needed):
#    signbridge-data   → /data   (read TSVs + bboxes; landmark parquets WRITTEN here)
#    signbridge-videos → /videos (read video clips)
#
#  If the volumes are not yet populated, upload with:
#    modal volume put signbridge-data "data/7_dataset_train.tsv" /data/7_dataset_train.tsv
#    modal volume put signbridge-data "data/7_dataset_val.tsv"   /data/7_dataset_val.tsv
#    modal volume put signbridge-data "data/7_dataset_test.tsv"  /data/7_dataset_test.tsv
#    modal volume put signbridge-data "data/bboxes_all.csv"      /bboxes_all.csv
#    modal volume put signbridge-videos "D:/signbridge_dataset/how2sign" /how2sign
#    modal volume put signbridge-videos "D:/signbridge_dataset/openasl"  /openasl
#
#  Output landmark parquets land at:
#    /data/landmarks_train.parquet  (signbridge-data volume)
#    /data/landmarks_val.parquet
#    /data/landmarks_test.parquet
#  These paths match exactly what qwen3vl_training.py reads when is_modal=True.
# ═══════════════════════════════════════════════════════════════════════════════

if CONFIG['is_modal']:
    import modal

    # ── Container image ────────────────────────────────────────────────────────
    # Base: CUDA 12.4 runtime (no nvcc needed — ONNX Runtime ships its own CUDA
    # kernels; no source compilation required unlike flash-attn in training).
    _modal_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .entrypoint([])   # silence verbose CUDA entrypoint logging
        .apt_install(
            "ffmpeg",           # video decoding backend for decord
            "libgl1",           # OpenCV headless dependency
            "libglib2.0-0",
        )
        .pip_install(
            # ONNX Runtime GPU must be installed BEFORE rtmlib so rtmlib's
            # dependency resolver does not pull in the CPU-only onnxruntime.
            "onnxruntime-gpu==1.20.1",
            # Pose estimation
            "rtmlib",
            # Video decoding
            "decord",
            # Vision / data
            "opencv-python-headless",
            "numpy",
            "pandas",
            "pyarrow",    # parquet support
            "tqdm",
        )
        # Pre-download RTMPose ONNX model weights into the image at build time.
        # Workers find the weights in the on-disk cache on cold start — no HTTP
        # download delay per container.  GPU is not available during image build
        # so we use the CPU backend just to trigger the download.
        .run_commands(
            "python -c \""
            "from rtmlib import RTMPose, Wholebody; "
            "cfg = Wholebody.MODE['performance']; "
            "RTMPose(cfg['pose'], model_input_size=cfg['pose_input_size'], "
            "backend='onnxruntime', device='cpu')"
            "\""
        )
    )

    # ── Volumes ────────────────────────────────────────────────────────────────
    _modal_data_vol   = modal.Volume.from_name(CONFIG['modal_data_volume'],   create_if_missing=False)
    _modal_video_vol  = modal.Volume.from_name(CONFIG['modal_video_volume'],  create_if_missing=False)
    _modal_output_vol = modal.Volume.from_name(CONFIG['modal_output_volume'], create_if_missing=True)

    # ── App ────────────────────────────────────────────────────────────────────
    app = modal.App(CONFIG['modal_app_name'], image=_modal_image)

    @app.function(
        gpu=CONFIG['modal_gpu'],
        cpu=CONFIG['modal_cpu'],
        memory=CONFIG['modal_memory'] * 1024,   # MiB
        timeout=CONFIG['modal_timeout'],
        retries=modal.Retries(
            max_retries=CONFIG['modal_retries'],
            backoff_coefficient=1.0,
            initial_delay=5.0,
        ),
        volumes={
            CONFIG['modal_data_mount']:   _modal_data_vol,
            CONFIG['modal_video_mount']:  _modal_video_vol,
            CONFIG['modal_output_mount']: _modal_output_vol,
        },
    )
    def run_extraction_on_modal():
        """Entry point executed inside the Modal container."""
        main()
        print("\n[modal] Committing signbridge-outputs volume ...")
        _modal_output_vol.commit()
        print("[modal] Volume committed — landmark parquets are now persisted.")

    @app.local_entrypoint()
    def modal_entrypoint():
        """Called on your local machine when you run `modal run <this_file>`."""
        print(
            f"  Dispatching landmark extraction to Modal "
            f"({CONFIG['modal_gpu']} GPU, {CONFIG['modal_cpu']} CPUs, "
            f"{CONFIG['modal_memory']} GB RAM, "
            f"timeout={CONFIG['modal_timeout'] // 3600}h, "
            f"workers={CONFIG['modal_workers']}) ..."
        )
        run_extraction_on_modal.remote()
