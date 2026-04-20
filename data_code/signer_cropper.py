"""
signer_cropper.py
=================
Shared module for computing a signer-centric bounding box from a video clip.

Used by:
    - 10_test_signer_crop.py     (visual verification on 10 samples)
    - 11_extract_signer_bboxes.py (offline CSV extraction over full dataset)
    - (future) inference code    (real-time crop before feeding frames to model)

Method:
    Sample every Nth frame (default every 4th), run MediaPipe PoseLandmarker on
    each, collect upper-body + hand landmarks (indices 0–22: nose, eyes, mouth,
    ears, shoulders, elbows, wrists, hand keypoints). Compute the UNION of all
    landmark positions across all sampled frames → one static bbox per clip.
    This guarantees that a briefly-extended hand (e.g., one second of full
    reach) is captured in the crop for the entire clip.

    Final bbox = padded union, clamped to frame bounds.

Uses MediaPipe's modern Tasks API (``mediapipe.tasks.vision.PoseLandmarker``)
which is compatible with current protobuf versions. The model file
(``pose_landmarker_full.task``, ~7 MB) is auto-downloaded once into
``data_code/models/`` on first use.

Designed so ``compute_bbox`` accepts either a video path OR an in-memory frame
buffer (for real-time inference).
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

# Silence MediaPipe / glog native-layer INFO + WARNING spam BEFORE importing
# ``mediapipe`` — these env vars are read when the native library loads.
# glog levels: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL → 2 hides INFO + WARNING.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("absl_logging_verbosity", "2")

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

# Upper-body + hand landmarks (excludes legs/hips for a tight signing-space box).
# PoseLandmarker returns 33 landmarks with the same index scheme as the legacy
# solution: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
#   0      nose
#   1-6    eyes (inner/outer)
#   7-8    ears
#   9-10   mouth (left/right corner)
#   11-12  shoulders
#   13-14  elbows
#   15-16  wrists
#   17-22  hand keypoints (pinky, index, thumb — each hand)
SIGNING_LANDMARK_INDICES: tuple[int, ...] = tuple(range(0, 23))

# Minimum per-landmark visibility to include it in the bbox union.
MIN_LANDMARK_VISIBILITY = 0.5

# Model variants (speed vs accuracy trade-off).
MODEL_URLS = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class BBoxResult:
    """Result of computing a signer bbox for one clip.

    Coordinates are in PIXEL space of the original video (x is width, y is height).
    If ``failed`` is True, the coordinates are all 0 and the clip should be skipped.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    frame_width: int
    frame_height: int
    detection_rate: float      # fraction of sampled frames with any valid landmarks
    num_sampled_frames: int
    num_detected_frames: int
    failed: bool

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _ensure_model_downloaded(variant: str) -> Path:
    """Download the PoseLandmarker task model into data_code/models/ on first use."""
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant: {variant!r}. Must be one of {list(MODEL_URLS)}.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if not model_path.exists():
        url = MODEL_URLS[variant]
        logger.info("Downloading MediaPipe PoseLandmarker model (%s) from %s", variant, url)
        print(f"[signer_cropper] Downloading pose_landmarker_{variant}.task (~7 MB) to {model_path} ...")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"[signer_cropper] Download complete: {model_path}")
    return model_path


class SignerCropper:
    """Computes one static bbox per clip from union of pose landmarks across sampled frames."""

    def __init__(
        self,
        sample_every_n: int = 4,
        padding_ratio: float = 0.25,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        landmark_indices: Iterable[int] = SIGNING_LANDMARK_INDICES,
        min_landmark_visibility: float = MIN_LANDMARK_VISIBILITY,
        model_variant: str = "full",
    ) -> None:
        self.sample_every_n = int(sample_every_n)
        self.padding_ratio = float(padding_ratio)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_presence_confidence = float(min_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.landmark_indices = tuple(landmark_indices)
        self.min_landmark_visibility = float(min_landmark_visibility)
        self.model_variant = str(model_variant)

        self._model_path = _ensure_model_downloaded(self.model_variant)

        # Use the minimal options pattern from the official PoseLandmarker
        # Python sample. IMAGE mode is the default; setting ``running_mode`` or
        # ``min_tracking_confidence`` explicitly has caused serialization
        # mismatches with the native proto bindings in some MediaPipe versions.
        self._landmarker_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self._model_path)),
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def compute_bbox(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> BBoxResult:
        """Compute one static signer bbox for a clip.

        Parameters
        ----------
        source
            Either a path to a video file, OR a numpy array of frames with
            shape (T, H, W, 3) in BGR (cv2 default) for real-time inference.
        """
        if isinstance(source, (str, Path)):
            frames, _ = self._load_video(str(source))
        else:
            frames = np.asarray(source)
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(
                    f"Expected frames of shape (T, H, W, 3); got {frames.shape}"
                )
        return self._compute_from_frames(frames)

    def crop_frames(
        self, frames: np.ndarray, bbox: BBoxResult
    ) -> np.ndarray:
        """Crop a frame buffer using a BBoxResult. Microsecond cost (numpy view).

        If ``bbox.failed`` is True, the original frames are returned unchanged
        and a warning is logged.
        """
        if bbox.failed:
            logger.warning("crop_frames called with failed bbox — returning uncropped.")
            return frames
        return frames[:, bbox.y1:bbox.y2, bbox.x1:bbox.x2, :]

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_video(path: str) -> tuple[np.ndarray, float]:
        """Load all frames of a video into memory as (T, H, W, 3) BGR uint8."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise IOError(f"Video contained zero readable frames: {path}")
        return np.stack(frames, axis=0), fps

    def _compute_from_frames(self, frames: np.ndarray) -> BBoxResult:
        """Run pose on every Nth frame, union keypoints, pad, clamp.

        ``frames`` is assumed BGR (cv2 default). MediaPipe expects RGB.
        """
        T, H, W, _ = frames.shape

        # Sample every Nth frame; always include last frame for coverage.
        sampled_indices = list(range(0, T, self.sample_every_n))
        if sampled_indices and sampled_indices[-1] != T - 1:
            sampled_indices.append(T - 1)
        elif not sampled_indices:
            sampled_indices = [0]

        xs: list[float] = []
        ys: list[float] = []
        num_detected = 0

        with mp_vision.PoseLandmarker.create_from_options(self._landmarker_options) as landmarker:
            for idx in sampled_indices:
                frame_bgr = frames[idx]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # MediaPipe expects an mp.Image wrapping contiguous uint8 RGB data.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect(mp_image)
                if not result.pose_landmarks:
                    continue
                landmarks = result.pose_landmarks[0]  # num_poses=1

                got_any = False
                for lm_idx in self.landmark_indices:
                    lm = landmarks[lm_idx]
                    # The Tasks API exposes ``visibility`` and ``presence`` on each landmark.
                    visibility = getattr(lm, "visibility", 1.0)
                    if visibility is not None and visibility < self.min_landmark_visibility:
                        continue
                    # Skip clearly out-of-frame predictions (MediaPipe can predict
                    # outside 0..1 when body is partly occluded).
                    if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
                        continue
                    xs.append(lm.x * W)
                    ys.append(lm.y * H)
                    got_any = True
                if got_any:
                    num_detected += 1

        num_sampled = len(sampled_indices)
        detection_rate = num_detected / num_sampled if num_sampled else 0.0

        # Failure: no usable keypoints anywhere in the clip.
        if not xs:
            return BBoxResult(
                x1=0, y1=0, x2=0, y2=0,
                frame_width=W, frame_height=H,
                detection_rate=detection_rate,
                num_sampled_frames=num_sampled,
                num_detected_frames=num_detected,
                failed=True,
            )

        raw_x1, raw_x2 = min(xs), max(xs)
        raw_y1, raw_y2 = min(ys), max(ys)

        # 25% padding: expand each side by padding_ratio * bbox_dim.
        bw = raw_x2 - raw_x1
        bh = raw_y2 - raw_y1
        pad_x = self.padding_ratio * bw
        pad_y = self.padding_ratio * bh

        x1 = int(round(max(0.0, raw_x1 - pad_x)))
        y1 = int(round(max(0.0, raw_y1 - pad_y)))
        x2 = int(round(min(float(W), raw_x2 + pad_x)))
        y2 = int(round(min(float(H), raw_y2 + pad_y)))

        # Final sanity check: bbox must have positive area.
        if x2 <= x1 or y2 <= y1:
            return BBoxResult(
                x1=0, y1=0, x2=0, y2=0,
                frame_width=W, frame_height=H,
                detection_rate=detection_rate,
                num_sampled_frames=num_sampled,
                num_detected_frames=num_detected,
                failed=True,
            )

        return BBoxResult(
            x1=x1, y1=y1, x2=x2, y2=y2,
            frame_width=W, frame_height=H,
            detection_rate=detection_rate,
            num_sampled_frames=num_sampled,
            num_detected_frames=num_detected,
            failed=False,
        )
