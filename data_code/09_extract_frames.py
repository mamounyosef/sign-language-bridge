"""
09_extract_frames.py
====================
Pre-extracts video clips to JPEG frames for faster training data loading.

Output layout:
    D:\\extracted_images_full_dataset\\{split}\\{vid}\\0000.jpg
                                                      0001.jpg
                                                      ...

New TSV files (written after extraction):
    data\\3_dataset_train.tsv
    data\\3_dataset_val.tsv
    data\\3_dataset_test.tsv
    (identical to 2_dataset_*.tsv but file_path points to the image folder)

Error log:
    D:\\extracted_images_full_dataset\\errors.log

Usage:
    conda activate base
    python data_code/09_extract_frames.py

Resume: already-extracted clips (non-empty output folder) are skipped
automatically, so you can safely re-run after a crash.
"""

import os
import time
import multiprocessing
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_ROOT  = Path(r"D:\extracted_images_full_dataset")
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # sign-language-bridge/
DATA_DIR     = PROJECT_ROOT / "data"

SPLITS = {
    "train": DATA_DIR / "2_dataset_train.tsv",
    "val":   DATA_DIR / "2_dataset_val.tsv",
    "test":  DATA_DIR / "2_dataset_test.tsv",
}

MAX_FRAMES   = 160       # cap at 160 frames (8 s @ 20 fps)
TARGET_SIZE  = 256       # letterbox target (px)
JPEG_QUALITY = 95        # JPEG quality (95 = near-lossless, ~8 KB/frame)
USE_GPU      = True      # GPU-accelerated decode + letterbox (decord.gpu(0))
                         # Set False to use CPU only
# Workers: GPU mode → 2 (CUDA context per process, contention above 2)
#          CPU mode → 4 (no GPU contention, more parallelism)
NUM_WORKERS  = 2 if USE_GPU else 4
# ──────────────────────────────────────────────────────────────────────────────


def _letterbox_frames_batch(frames, target_size):
    """
    Letterbox a (T, C, H, W) uint8 tensor to target_size × target_size.
    One vectorized torchvision call — no Python loop over frames.
    """
    import torchvision.transforms.functional as TF
    T, C, H, W = frames.shape
    scale = min(target_size / H, target_size / W)
    new_h, new_w = int(H * scale), int(W * scale)
    resized = TF.resize(frames, [new_h, new_w],
                        interpolation=TF.InterpolationMode.BILINEAR,
                        antialias=True)
    pad_top    = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left   = (target_size - new_w) // 2
    pad_right  = target_size - new_w - pad_left
    return TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)


def _extract_clip(args):
    """
    Worker function — runs in a spawned process.
    Returns (vid, status_str, num_frames_saved).
    """
    vid, src_path, out_dir_str, max_frames, target_size, jpeg_quality, use_gpu = args
    out_dir = Path(out_dir_str)

    # ── Resume: skip if folder already has JPEG files ──
    if out_dir.exists():
        existing = list(out_dir.glob("*.jpg"))
        if len(existing) > 0:
            return vid, "skipped", len(existing)

    try:
        import torch
        import torchvision.io

        gpu_available = use_gpu and torch.cuda.is_available()

        # ── Decode video ──
        try:
            import decord
            if gpu_available:
                # GPU decode: frames land directly in GPU memory
                ctx = decord.gpu(0)
                decord.bridge.set_bridge("torch")
                vr = decord.VideoReader(src_path, ctx=ctx)
            else:
                decord.bridge.set_bridge("torch")
                vr = decord.VideoReader(src_path, num_threads=1)
            n = min(len(vr), max_frames)
            raw = vr.get_batch(list(range(n)))      # (T, H, W, C) uint8
            frames = raw.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W) uint8
        except Exception:
            # Fallback: torchvision (CPU only)
            video, _, _ = torchvision.io.read_video(src_path, pts_unit="sec",
                                                    output_format="TCHW")
            frames = video[:max_frames]             # (T, C, H, W) uint8
            gpu_available = False                   # torchvision fallback is CPU

        if frames.shape[0] == 0:
            return vid, "error: 0 frames decoded", 0

        # ── Letterbox on GPU (if available) or CPU ──
        if gpu_available and frames.device.type != "cuda":
            frames = frames.cuda()
        frames = _letterbox_frames_batch(frames, target_size)  # (T, 3, 256, 256)

        # ── Move to CPU for JPEG saving (write_jpeg requires CPU tensor) ──
        if frames.device.type != "cpu":
            frames = frames.cpu()

        # ── Save each frame as JPEG ──
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(frames.shape[0]):
            torchvision.io.write_jpeg(
                frames[i],
                str(out_dir / f"{i:04d}.jpg"),
                quality=jpeg_quality,
            )

        return vid, "ok", frames.shape[0]

    except Exception as exc:
        return vid, f"error: {exc}", 0


def _build_jobs(splits_dict, output_root, max_frames, target_size, jpeg_quality, use_gpu):
    """Build the flat list of job tuples for all splits."""
    jobs = []
    for split, tsv_path in splits_dict.items():
        df = pd.read_csv(tsv_path, sep="\t")
        for _, row in df.iterrows():
            out_dir = output_root / split / row["vid"]
            jobs.append((
                row["vid"],
                row["file_path"],
                str(out_dir),
                max_frames,
                target_size,
                jpeg_quality,
                use_gpu,
            ))
    return jobs


def _write_new_tsvs(splits_dict, output_root, data_dir):
    """
    Write 3_dataset_{split}.tsv files.
    file_path → extracted image folder if successfully extracted,
                original video path otherwise (fallback for failed clips).
    """
    for split, tsv_path in splits_dict.items():
        df = pd.read_csv(tsv_path, sep="\t")
        new_paths = []
        for _, row in df.iterrows():
            img_dir = output_root / split / row["vid"]
            if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 0:
                new_paths.append(str(img_dir))
            else:
                new_paths.append(row["file_path"])   # fallback to original mp4
        df["file_path"] = new_paths
        out_tsv = data_dir / f"3_dataset_{split}.tsv"
        df.to_csv(out_tsv, sep="\t", index=False)
        n_extracted = sum(1 for p in new_paths if not p.endswith(".mp4"))
        print(f"  ✅ {out_tsv.name}  ({n_extracted}/{len(df)} clips using extracted frames)")


def _format_time(seconds):
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if d > 0:   return f"{d}d {h}h {m}m {s}s"
    if h > 0:   return f"{h}h {m}m {s}s"
    if m > 0:   return f"{m}m {s}s"
    return f"{s}s"


def main():
    multiprocessing.set_start_method("spawn", force=True)

    print("=" * 68)
    print("  FRAME EXTRACTION  —  09_extract_frames.py")
    print("=" * 68)
    print(f"  Output root  : {OUTPUT_ROOT}")
    print(f"  Max frames   : {MAX_FRAMES}  (@ 20 FPS = {MAX_FRAMES // 20}s per clip)")
    print(f"  Target size  : {TARGET_SIZE}×{TARGET_SIZE}  (letterbox, no crop)")
    print(f"  JPEG quality : {JPEG_QUALITY}")
    print(f"  GPU mode     : {USE_GPU}")
    print(f"  Workers      : {NUM_WORKERS}")
    print()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    error_log = OUTPUT_ROOT / "errors.log"

    # ── Build job list ──
    print("📋 Building job list ...")
    jobs = _build_jobs(SPLITS, OUTPUT_ROOT, MAX_FRAMES, TARGET_SIZE, JPEG_QUALITY, USE_GPU)
    total = len(jobs)

    already_done = sum(
        1 for _, _, out_dir_str, *_ in jobs
        if Path(out_dir_str).exists()
        and len(list(Path(out_dir_str).glob("*.jpg"))) > 0
    )
    remaining = total - already_done

    print(f"  Total clips  : {total:,}")
    print(f"  Already done : {already_done:,}  (will be skipped)")
    print(f"  To extract   : {remaining:,}")
    print()

    if remaining == 0:
        print("✅ All clips already extracted. Writing TSV files ...")
        _write_new_tsvs(SPLITS, OUTPUT_ROOT, DATA_DIR)
        return

    # ── Extract ──
    t_start = time.time()
    n_ok      = already_done
    n_skipped = 0
    n_error   = 0
    errors    = []

    with open(error_log, "a", encoding="utf-8") as err_f:
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=total, desc="Extracting", unit="clip",
                      initial=already_done, dynamic_ncols=True) as pbar:
                for vid, status, nf in pool.imap_unordered(_extract_clip, jobs):
                    if status == "ok":
                        n_ok += 1
                    elif status == "skipped":
                        n_skipped += 1
                    else:
                        n_error += 1
                        msg = f"{vid}\t{status}\n"
                        err_f.write(msg)
                        errors.append(msg.strip())

                    pbar.update(1)

                    # Live ETA
                    elapsed  = time.time() - t_start
                    done_now = n_ok + n_skipped + n_error - already_done
                    if done_now > 0 and remaining > done_now:
                        eta = (elapsed / done_now) * (remaining - done_now)
                        pbar.set_postfix(ok=n_ok, err=n_error,
                                         ETA=_format_time(eta))

    # ── Summary ──
    elapsed_total = time.time() - t_start
    print()
    print("=" * 68)
    print("  EXTRACTION COMPLETE")
    print("=" * 68)
    print(f"  Total time   : {_format_time(elapsed_total)}")
    print(f"  Extracted ok : {n_ok:,}")
    print(f"  Skipped      : {n_skipped:,}  (already existed)")
    print(f"  Errors       : {n_error:,}")
    if errors:
        print(f"  Error log    : {error_log}")
        print("  First 5 errors:")
        for e in errors[:5]:
            print(f"    {e}")
    print()

    # ── Write new TSV files ──
    print("📄 Writing new TSV files ...")
    _write_new_tsvs(SPLITS, OUTPUT_ROOT, DATA_DIR)
    print()
    print("✅ Done.  Switch training to extracted frames by setting in CONFIG:")
    print("     'data_train_csv': Path('..') / 'data' / '3_dataset_train.tsv',")
    print("     'data_val_csv':   Path('..') / 'data' / '3_dataset_val.tsv',")


if __name__ == "__main__":
    main()
