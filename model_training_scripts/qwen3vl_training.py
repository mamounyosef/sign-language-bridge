
## Imports & Setup

import os
import sys
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Reduce CUDA memory fragmentation from variable-length video sequences.
# expandable_segments lets the allocator grow existing segments instead of
# carving new fixed blocks — critical when sequence length varies each step.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Prevent HuggingFace tokenizer threads from conflicting with DataLoader workers.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from pathlib import Path
from typing import Optional
import math
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.tensorboard import SummaryWriter

# Qwen3-VL video utilities
from qwen_vl_utils import process_vision_info

import torchvision.transforms.v2 as T_v2

## Additional Imports
import time
import csv
import threading
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import sacrebleu

import tempfile
import shutil
import gc
import torch.multiprocessing as mp
from PIL import Image

# Windows: prevent ERROR_COMMITMENT_LIMIT with DataLoader workers
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# benchmark=True lets cuDNN profile and select the fastest convolution algorithm for each
# unique input shape (vision encoder patch embedding / pooling layers benefit most).
# deterministic=False re-enables those faster non-deterministic kernels.
# Note: bit-exact reproducibility is already broken by stochastic augmentation and LoRA
# dropout; all RNG states are saved in checkpoints for within-run resume accuracy.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # faster bf16 reductions
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)          # disable slow fallback; flash/mem-efficient cover all cases

# %%
# ─── Training Constants ───
MAX_PPL_CAP = 20  # math.exp(20) ~ 485M — cap for display

# %%
## Configuration

CONFIG = {
    # ── Data Paths ──
    'data_train_tsv': Path('..') / 'data' / '2_dataset_train.tsv',
    'data_val_tsv': Path('..') / 'data' / '2_dataset_val.tsv',
    'data_test_tsv': Path('..') / 'data' / '2_dataset_test.tsv',
    'tsv_sep': '\t',

    # ── Model ──
    'model_name': 'Qwen/Qwen3-VL-2B-Instruct',
    'attn_implementation': 'flash_attention_2',  # 'flash_attention_2', 'sdpa', or 'eager'
    'dtype': 'bfloat16',                         # Model weights precision

    # ── Quantization (QLoRA) ──
    'use_qlora': True,                            # True = 4-bit quantized base model
    'bnb_4bit_compute_dtype': 'bfloat16',
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant': True,

    # ── Video Processing ──
    'video_fps': 14,                              # Frames per second to sample (lowered from 18 to fit bigger per-frame budget within 20M total cap)
    'video_min_pixels': 4 * 32 * 32,              # Min visual tokens per frame pair (~4 tokens)
    'video_max_pixels': 100 * 32 * 32,            # Max visual tokens per frame pair (100 = 320*320 at patch_size=16, merge=2)
    'video_total_pixels': 20480 * 32 * 32,        # Total pixel budget cap across all frames (None = no cap)

    # ── Signer Cropping (pre-computed MediaPipe bboxes) ──
    # Bboxes are one static box per clip (padded union of pose landmarks across
    # sampled frames). Crop is applied to decoded frames BEFORE Qwen3-VL's
    # internal resize, so the full pixel budget lands on the signing region.
    # CSVs are produced by data_code/11_extract_signer_bboxes.py.
    'use_signer_crop': True,
    'bbox_csv_train': Path('..') / 'data' / '2_dataset_train_bboxes.csv',
    'bbox_csv_val':   Path('..') / 'data' / '2_dataset_val_bboxes.csv',
    'bbox_csv_test':  Path('..') / 'data' / '2_dataset_test_bboxes.csv',

    # ── Chat Template / Prompts ──
    'system_prompt': 'You are a sign language translator.',
    'user_prompt': 'Translate this American Sign Language video into English.',

    # ── Sequence Lengths ──
    'max_text_tokens': 70,                        # Max tokens for the assistant response (translation)

    # ── LoRA ──
    # Alpha = 2x rank throughout, consistent with RSLoRA (alpha/sqrt(r)) scaling.
    'lora_t1_r': 16,                             
    'lora_t1_alpha': 32,
    'lora_t1_modules': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # LM attention projections
        'gate_proj', 'up_proj', 'down_proj',      # LM MLP projections
    ],
    'lora_t2_r': 32,                             
    'lora_t2_alpha': 64,
    'lora_t2_modules': [
        'qkv', 'proj',                            # Vision attention: fused QKV + output proj (incl. Conv3d patch_embed.proj)
        'linear_fc1', 'linear_fc2',               # Vision MLP + merger + deepstack_merger_list
        'pos_embed',                             # Vision positional embeddings (Embedding)

    ],
    'lora_t3_r': 2,                              
    'lora_t3_alpha': 4,
    'lora_t3_modules': [                         # Remaining trainable layers (Linear + Embedding)
        'lm_head',                               # Output head (Linear; base weight tied to embed_tokens)
        # 'embed_tokens',                          # LM input embeddings (Embedding; tied to lm_head)
    ],
    'lora_dropout': 0.08,
    'lora_bias': 'none',
    'lora_use_dora': False,                       # Weight-Decomposed LoRA
    'lora_use_rslora': True,                      # Rank-Stabilized LoRA (sqrt(r) scaling; better stability)
    'lora_init_weights': True,                    # Default init (PiSSA not supported for Conv3d layers in vision encoder)

    # ── Core Training ──
    'num_epochs': 3,
    'batch_size': 1,                              # VRAM constraint with vision LoRA — 1 sample per micro-batch

    # ── Gradient Accumulation ──
    'grad_accum_steps': 32,                       # Effective batch = 1 x 32 = 32

    # ── DataLoader Config ──
    'train_num_workers': 1,                       # 2 workers; lower count avoids Windows shared-memory exhaustion (error 1455)
    'train_prefetch_factor': 1,                   # Pre-load 2 batches ahead (safe now that pin_memory is permanently off)
    'train_pin_memory': True,                 
    'train_persistent_workers': True,             # Keep worker alive across epochs — avoids re-spawn overhead

    'val_num_workers': 1,                          # 1 worker overlaps video decoding with GPU inference during validation
    'val_prefetch_factor': 1,                      # Pre-load 2 batches ahead during validation
    'val_pin_memory': False,                        # Pinned memory for async CPU→GPU DMA during validation
    'val_persistent_workers': False,               # Keep val worker alive across validation runs

    # ── Learning Rates ──
    'lr_tier1_lm':          3e-5,                 # LM LoRA (attn + MLP) — moderate rate, already fluent English
    'min_lr_tier1':         3e-7,                 # Cosine floor for tier 1 1% of it's value

    'lr_tier2_vision':      5e-5,                 # Vision LoRA — highest rate, adapting ViT to ASL
    'min_lr_tier2':         5e-7,                 # Cosine floor for tier 2 (deeper decay) 1% of it's value

    'lr_tier3_embed_head':  2e-5,                 # Embed + LM head — moderate rate
    'min_lr_tier3':         2e-7,                 # Cosine floor for tier 3 1% of it's value

    'warmup_ratio': 0.03,                         # Fraction of total steps for warmup
    'warmup_steps': 350,                          # If set, overrides warmup_ratio (raised from 100 to let bigger vision LoRA stabilise)
    'weight_decay': 0.01,
    'adam_betas': (0.9, 0.98),
    'max_grad_norm': 1.0,

    # ── Logging ──
    'log_every_steps': 1,
    'train_log_file': Path('..') / 'saved_metrics' / 'qwen3vl_train_log.csv',
    'val_log_file': Path('..') / 'saved_metrics' / 'qwen3vl_val_log.csv',
    'gen_samples_log_file': Path('..') / 'saved_metrics' / 'qwen3vl_gen_samples_log.csv',
    'tensorboard_dir': Path('..') / 'saved_metrics' / 'tensorboard' / 'qwen3vl_training',

    # ── Checkpointing ──
    'save_every_steps': 10,
    'keep_last_n_checkpoints': 3,
    'checkpoint_dir': Path('..') / 'checkpoints' / 'qwen3vl',

    # ── Evaluation ──
    'eval_every_steps': 80,
    'eval_every_steps_warmup': 80,               # More frequent eval early on
    'eval_warmup_threshold': 1000,                # Switch to normal eval freq after this step
    'max_eval_batches': 60,
    'num_print_samples': 5,
    'val_gen_batch_size': 1,                      # Low — generation is VRAM-intensive
    'max_generate_samples': 100,
    'val_beam_size': 2,                             # 1 = greedy (faster validation); run beam=4 on final checkpoint
    'val_length_penalty': 1.0,                      # > 1.0 favors longer outputs (counters BLEU brevity penalty); < 1.0 favors shorter
    'val_no_repeat_ngram_size': 0,                  # Block any n-gram from repeating; improves BLEU precision (0 = disabled)
    'val_repetition_penalty': 1.0,
    'val_max_new_tokens': 70,

    # ── Early Stopping ──
    'early_stopping_patience': 12,                # Evals without improvement before stopping

    # ── Performance & Memory Optimizations ──
    'use_8bit_adam': True,
    'use_gradient_checkpointing': True,
    'use_torch_compile': False,
    'torch_compile_mode': 'default',              # 'default' or 'max-autotune'

    # ── Resuming ──
    # Fresh run — old checkpoints are NOT compatible with the new Tier-2 LoRA
    # rank (4 → 32) or the new signer-crop data path.
    'resume_training': True,                   # Set to True to resume from checkpoint
    'load_best_model': False,
    'resume_checkpoint_step': 10,             # None = latest, or specific step number

    # ── Mid-Training LR Override ──────────────────────────────────────────────
    # SPECIAL USE ONLY: Use this block to manually correct the learning rate when
    # resuming from a checkpoint mid-training (e.g. if LR is too high and causing
    # noisy loss, or you want to fine-tune from a specific checkpoint with lower LR).
    # Set 'enabled' to False after the resumed run starts successfully.
    #
    # How it works:
    #   Rebuilds a per-tier LambdaLR cosine schedule starting from the given
    #   peak LRs and decaying to the given minimums over the remaining steps.
    #   No warmup — straight into cosine decay from the override values.
    #   Each tier is independent; omit a tier key to keep its original LR.
    # ─────────────────────────────────────────────────────────────────────────
    'lr_override': {
        'enabled': False,                   # Set True to apply, False after resuming
        'lr_tier1_lm':          1e-5,       # New peak LR for Tier 1 (LM attn + MLP)
        'lr_tier2_vision':      5e-5,       # New peak LR for Tier 2 (Vision encoder)
        'lr_tier3_embed_head':  1e-5,       # New peak LR for Tier 3 (Embed + LM head)
        'min_lr_tier1':         1e-6,       # Cosine floor for Tier 1
        'min_lr_tier2':         5e-7,       # Cosine floor for Tier 2
        'min_lr_tier3':         1e-6,       # Cosine floor for Tier 3
    },

    # ── Bucket Batch Sampling ──
    'use_bucket_batching': True,                  # Group clips by duration to minimise padding waste

    # ── Label Smoothing ──
    # Gentle smoothing helps with OpenASL's noisy ASR-style captions without
    # overly dulling the training signal on clean How2Sign data.
    'label_smoothing': 0.04,

    # ── Seed ──
    'seed': 42,

    # ── Runtime ──
    'wait_for_manual_start': False,               # Interactive safety gate before training loop starts

    # ── Data Augmentation (video frames, training only) ──
    'aug_start_epoch': 2,                         # Epoch at which to enable augmentation (1 = from the start)
    'aug_temporal_jitter': True,
    'aug_temporal_jitter_range': 2,           # Max frames to shift (±2)
    'aug_temporal_jitter_prob': 0.4,

    'aug_color_jitter': True,
    'aug_color_jitter_prob': 0.45,             
    'aug_color_jitter_brightness': 0.1,
    'aug_color_jitter_contrast': 0.1,
    'aug_color_jitter_saturation': 0.1,       # Moderate swing
    'aug_color_jitter_hue': 0.1,             # ±43° — covers green-screen variation without being extreme

    'aug_random_grayscale': True,
    'aug_random_grayscale_prob': 0.08,        # Forces shape/motion reliance over colour

    'aug_gaussian_blur': False,
    'aug_gaussian_blur_prob': 0.1,
    'aug_gaussian_blur_kernel': (3, 3),

    'aug_solarize': True,                     # Inverts pixels above threshold — extreme colour variety
    'aug_solarize_prob': 0.07,
    'aug_solarize_threshold': 220,            # 0–255; pixels above this get inverted

    'aug_equalize': True,                     # Histogram equalisation — bridges studio vs natural lighting
    'aug_equalize_prob': 0.1,

    'aug_random_erasing': False,              # Randomly blacks out small patches — occlusion robustness
    'aug_random_erasing_prob': 0.2,
    'aug_random_erasing_scale': (0.02, 0.1), # Fraction of frame area erased
    'aug_random_erasing_ratio': (0.3, 3.3),  # Aspect ratio of erased region

    'aug_affine': True,                       # Small rotation + translation + scale — camera angle variation
    'aug_affine_prob': 0.3,
    'aug_affine_degrees': 4,                  # ±4° rotation
    'aug_affine_translate': 0.05,             # ±5% of frame width/height
    'aug_affine_scale_min': 0.95,             # 95%–105% zoom
    'aug_affine_scale_max': 1.05,

    'aug_speed_perturb': True,                # Simulate faster/slower signing by resampling frames
    'aug_speed_perturb_prob': 0.2,
    'aug_speed_perturb_min': 0.9,             # 0.9× = 10% slower (frames stretched)
    'aug_speed_perturb_max': 1.0,             # 1.0× = no change (frames compressed, tail padded)

    # ── Debug: save augmented frames to disk ──
    'aug_debug_save_images': True,
    'aug_debug_save_interval': 20,          # Save one frame every N optimiser steps
    'aug_debug_save_dir': Path('..') / 'data' / 'debugging_images',
}

torch.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed_all(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])


# ═══════════════════════════════════════════════════════════════
#  INTERACTIVE TRAINING CONTROLLER
# ═══════════════════════════════════════════════════════════════

class InteractiveController:
    """
    Allows live adjustment of hyperparameters during training without stopping.

    A background thread watches stdin. Typing 'p' + Enter sets a pause flag.
    The training loop calls .check() after each optimizer step; when the flag
    is set the loop blocks here and accepts commands before continuing.

    Commands (case-insensitive):
        lr=<t1>,<t2>,<t3>   Set per-tier LRs and rebuild per-tier cosine scheduler.
                             Provide all three values (e.g. lr=1e-5,3e-5,1e-5).
                             Alternatively lr=<value> sets all tiers to the same LR.
        clip=<value>         Set gradient clip max_norm.
        resume               Continue training.
    """

    def __init__(self):
        self._pause_flag = threading.Event()
        self._stop_flag  = threading.Event()
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name="interactive-ctrl"
        )
        self._thread.start()
        _SEP = "─" * 64
        print()
        print(f"  ┌{_SEP}┐")
        print(f"  │{'  Interactive Training Controller  active':^64}│")
        print(f"  │{'  Type  p + Enter  at any time to pause training':^64}│")
        print(f"  │{'  Commands:  lr=<t1>,<t2>,<t3>   clip=<val>   resume':^64}│")
        print(f"  └{_SEP}┘")
        print()

    # ── Background stdin listener ───────────────────────────────────────────

    def _listen(self):
        while not self._stop_flag.is_set():
            try:
                line = sys.stdin.readline()
                if not line:            # EOF / pipe closed
                    break
                if line.strip().lower() in ('p', 'pause'):
                    print(
                        "\n  [ ⏸  Pause requested — will pause after current optimizer step... ]\n"
                    )
                    sys.stdout.flush()
                    self._pause_flag.set()
            except Exception:
                break

    def stop(self):
        """Signal the listener thread to exit (called at end of training)."""
        self._stop_flag.set()

    # ── Interactive pause UI ────────────────────────────────────────────────

    def check(self, optimizer, scheduler_ref, train_config, global_step, total_optimizer_steps):
        """
        Call this after every optimizer step.
        Blocks until the user types 'resume' if a pause was requested.

        scheduler_ref is a one-element list [scheduler] so this method can
        swap the scheduler in-place when lr= is used.
        """
        if not self._pause_flag.is_set():
            return
        self._pause_flag.clear()

        lr_t1        = optimizer.param_groups[0]['lr']
        lr_t2        = optimizer.param_groups[1]['lr']
        lr_t3        = optimizer.param_groups[2]['lr']
        current_clip = train_config['max_grad_norm']
        remaining    = total_optimizer_steps - global_step

        _THICK = "═" * 64
        _THIN  = "─" * 64

        print()
        print(f"  {_THICK}")
        print(f"  ⏸  TRAINING PAUSED")
        print(f"  {_THIN}")
        print(f"  Step      : {global_step} / {total_optimizer_steps}  ({remaining} steps remaining)")
        print(f"  LR T1 (LM)         : {lr_t1:.4e}")
        print(f"  LR T2 (Vision)     : {lr_t2:.4e}")
        print(f"  LR T3 (Embed/Head) : {lr_t3:.4e}")
        print(f"  Grad clip : {current_clip}")
        print(f"  {_THIN}")
        print(f"  Commands (type one then press Enter):")
        print(f"    lr=<t1>,<t2>,<t3>  — set per-tier LRs  (e.g. lr=1e-5,3e-5,1e-5)")
        print(f"    lr=<value>         — set all tiers to the same LR")
        print(f"    clip=<value>       — set gradient clip max_norm")
        print(f"    resume             — continue training")
        print(f"  {_THICK}")

        while True:
            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  ▶  Resuming (EOF / interrupt)...\n")
                break

            cmd = raw.lower()

            if cmd in ('resume', 'r', ''):
                break

            elif cmd.startswith('lr='):
                try:
                    raw_val = cmd[3:]
                    parts = [p.strip() for p in raw_val.split(',')]
                    if len(parts) == 3:
                        new_lrs = [float(p) for p in parts]
                    elif len(parts) == 1:
                        # Single value — apply uniformly to all tiers
                        new_lrs = [float(parts[0])] * 3
                    else:
                        print(f"  ✗  Expected 1 or 3 comma-separated values.  Example: lr=1e-5,3e-5,1e-5")
                        continue
                    if any(lr <= 0 for lr in new_lrs):
                        print(f"  ✗  All LRs must be positive.  Got: {new_lrs}")
                        continue
                    old_lrs = [pg['lr'] for pg in optimizer.param_groups]
                    min_lrs = [
                        train_config.get('min_lr_tier1', 1e-6),
                        train_config.get('min_lr_tier2', 5e-7),
                        train_config.get('min_lr_tier3', 1e-6),
                    ]
                    for pg, new_lr in zip(optimizer.param_groups, new_lrs):
                        pg['lr']         = new_lr
                        pg['initial_lr'] = new_lr
                    # Rebuild per-tier cosine LambdaLR over remaining steps
                    def _pause_lambda(peak: float, floor: float):
                        ratio = floor / peak
                        def _fn(step: int) -> float:
                            progress = min(max(step / max(remaining, 1), 0.0), 1.0)
                            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
                            return ratio + (1.0 - ratio) * cos
                        return _fn
                    scheduler_ref[0] = torch.optim.lr_scheduler.LambdaLR(
                        optimizer,
                        lr_lambda=[_pause_lambda(pk, mn) for pk, mn in zip(new_lrs, min_lrs)],
                    )
                    _tier_labels = ['T1 LM', 'T2 Vision', 'T3 Embed/Head']
                    print(f"  ✓  Per-tier LR update  ({remaining} steps cosine):")
                    for label, old, new, mn in zip(_tier_labels, old_lrs, new_lrs, min_lrs):
                        print(f"     {label:16s}: {old:.4e}  →  {new:.4e}  (floor {mn:.2e})")
                except ValueError:
                    print(f"  ✗  Invalid value: '{raw}'.  Example: lr=1e-5,3e-5,1e-5")

            elif cmd.startswith('clip='):
                try:
                    new_clip = float(cmd[5:])
                    if new_clip <= 0:
                        print(f"  ✗  Clip must be positive.  Got: {new_clip}")
                        continue
                    old_clip = train_config['max_grad_norm']
                    train_config['max_grad_norm'] = new_clip
                    print(f"  ✓  Grad clip      :  {old_clip}  →  {new_clip}")
                except ValueError:
                    print(f"  ✗  Invalid value: '{raw}'.  Example: clip=0.5")

            else:
                print(f"  ✗  Unknown command: '{raw}'")
                print(f"     Available: lr=<value>  |  clip=<value>  |  resume")

        _SEP = "─" * 64
        print()
        print(f"  ┌{_SEP}┐")
        print(f"  │{'  ▶  Resuming training...  ':^64}│")
        print(f"  └{_SEP}┘")
        print()


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class SignLanguageQwen3VLDataset(Dataset):
    """
    Reads TSV manifest and returns raw sample dicts.
    Video loading and tokenization happens in the collator (via Qwen3-VL processor).
    """

    def __init__(self, tsv_path, sep='\t', bbox_csv_path=None,
                 tokenizer=None, max_text_tokens=None):
        df = pd.read_csv(tsv_path, sep=sep)
        required_cols = {'vid', 'file_path', 'text', 'duration_sec', 'source'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise RuntimeError(f"{tsv_path} missing columns: {missing_cols}")

        # Load bbox CSV and build O(1) lookup by vid. Filter failed extractions.
        bbox_by_vid = {}
        n_bbox_failed = 0
        if bbox_csv_path is not None:
            bbox_df = pd.read_csv(bbox_csv_path)
            n_bbox_failed = int(bbox_df['failed'].sum())
            bbox_ok = bbox_df[~bbox_df['failed']]
            for r in bbox_ok.itertuples(index=False):
                bbox_by_vid[str(r.vid)] = (
                    int(r.x1), int(r.y1), int(r.x2), int(r.y2),
                    int(r.frame_width), int(r.frame_height),
                )
            print(f"  ✓ Loaded {len(bbox_by_vid)} bboxes from {Path(bbox_csv_path).name} "
                  f"({n_bbox_failed} failed, filtered out)")

        do_truncate = tokenizer is not None and max_text_tokens is not None and max_text_tokens > 0

        self.samples = []
        missing = 0
        no_bbox = 0
        truncated = 0

        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f'Loading {Path(tsv_path).stem}'):
            fp = str(row.file_path)
            if not Path(fp).exists():
                missing += 1
                continue
            vid = str(row.vid)
            bbox = bbox_by_vid.get(vid) if bbox_csv_path is not None else None
            if bbox_csv_path is not None and bbox is None:
                no_bbox += 1
                continue
            text = str(row.text)
            if do_truncate:
                assert tokenizer is not None and max_text_tokens is not None
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > int(max_text_tokens):
                    text = tokenizer.decode(ids[:int(max_text_tokens)], skip_special_tokens=True)
                    truncated += 1
            self.samples.append({
                'vid': vid,
                'text': text,
                'duration_sec': float(row.duration_sec),
                'file_path': fp,
                'source': str(row.source),
                'bbox': bbox,
            })

        trunc_msg = f", {truncated} truncated to ≤{max_text_tokens} tokens" if do_truncate else ""
        print(f"  ✓ Loaded {len(self.samples)} samples ({missing} missing files, {no_bbox} without valid bbox{trunc_msg})")
        del df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    Groups samples with similar durations into the same batch
    to minimise padding waste in the Qwen3-VL processor.
    Supports saving/restoring batch order for mid-epoch resume without re-decoding.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Sort by duration to group similar-length clips
        lengths = [s['duration_sec'] for s in dataset.samples]
        sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
        del lengths  # no longer needed once sorted_indices is built
        self.batches = [
            sorted_indices[i:i + batch_size]
            for i in range(0, len(sorted_indices), batch_size)
        ]
        del sorted_indices  # no longer needed once batches are built
        self._skip_first_n = 0  # number of batches to skip on next iteration

    def set_skip(self, n):
        """Skip the first *n* batches on the next iteration (for mid-epoch resume)."""
        self._skip_first_n = n

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for i, batch in enumerate(self.batches):
            if i < self._skip_first_n:
                continue
            yield batch
        self._skip_first_n = 0  # reset after full epoch

    def __len__(self):
        return len(self.batches)


# ═══════════════════════════════════════════════════════════════
#  COLLATOR
# ═══════════════════════════════════════════════════════════════

class Qwen3VLCollator:
    """
    Collates raw sample dicts into model-ready batches.
    Handles: video loading, chat template formatting, tokenization, label masking.
    """

    def __init__(self, processor, config, is_training=True):
        self.processor = processor
        self.config = config
        self.is_training = is_training
        self.augmentation_enabled = False          # Toggled by training loop based on aug_start_epoch
        self.debug_save_images = is_training and config.get('aug_debug_save_images', False)

        # Rate-limited logging for crop-alignment snaps in _apply_signer_crop. 
        # Qwen3-VL requires H/W divisible by patch_size*merge_size (=32); bboxes
        # rarely align, so we snap and report aggregated stats every N batches.
        self._snap_align = 32
        self._snap_log_every = int(config.get('snap_log_every_batches', 2000))
        self._snap_batches = 0
        self._snap_total_clips = 0
        self._snap_snapped_clips = 0
        self._snap_total_trim_px = 0
        self._snap_max_trim_px = 0
        self._snap_degenerate_bbox_ids = set()

        # ── Build spatial augmentation transform (applied per-clip, training only) ──
        # torchvision v2 applies the SAME random parameters to every frame in a
        # (T, C, H, W) tensor — correct for video (no inter-frame spatial jitter).
        if is_training:
            aug_transforms = []
            if config.get('aug_color_jitter', False):
                aug_transforms.append(
                    T_v2.RandomApply([
                        T_v2.ColorJitter(
                            brightness=config.get('aug_color_jitter_brightness', 0.3),
                            contrast=config.get('aug_color_jitter_contrast', 0.3),
                            saturation=config.get('aug_color_jitter_saturation', 0.2),
                            hue=config.get('aug_color_jitter_hue', 0.05),
                        )
                    ], p=config.get('aug_color_jitter_prob', 0.5))
                )
            if config.get('aug_random_grayscale', False):
                aug_transforms.append(
                    T_v2.RandomGrayscale(p=config.get('aug_random_grayscale_prob', 0.1))
                )
            if config.get('aug_gaussian_blur', False):
                kernel = config.get('aug_gaussian_blur_kernel', (5, 5))
                aug_transforms.append(
                    T_v2.RandomApply([
                        T_v2.GaussianBlur(kernel_size=kernel)
                    ], p=config.get('aug_gaussian_blur_prob', 0.2))
                )
            if config.get('aug_solarize', False):
                aug_transforms.append(
                    T_v2.RandomApply([
                        T_v2.RandomSolarize(threshold=config.get('aug_solarize_threshold', 128))
                    ], p=config.get('aug_solarize_prob', 0.1))
                )
            if config.get('aug_equalize', False):
                aug_transforms.append(
                    T_v2.RandomApply([
                        T_v2.RandomEqualize()
                    ], p=config.get('aug_equalize_prob', 0.15))
                )
            if config.get('aug_affine', False):
                aug_transforms.append(
                    T_v2.RandomApply([
                        T_v2.RandomAffine(
                            degrees=config.get('aug_affine_degrees', 8),
                            translate=(
                                config.get('aug_affine_translate', 0.05),
                                config.get('aug_affine_translate', 0.05),
                            ),
                            scale=(
                                config.get('aug_affine_scale_min', 0.95),
                                config.get('aug_affine_scale_max', 1.05),
                            ),
                            fill=0,
                        )
                    ], p=config.get('aug_affine_prob', 0.3))
                )
            if config.get('aug_random_erasing', False):
                aug_transforms.append(
                    T_v2.RandomErasing(
                        p=config.get('aug_random_erasing_prob', 0.2),
                        scale=config.get('aug_random_erasing_scale', (0.02, 0.1)),
                        ratio=config.get('aug_random_erasing_ratio', (0.3, 3.3)),
                        value=0,
                    )
                )
            self.aug_transform = T_v2.Compose(aug_transforms) if aug_transforms else None
        else:
            self.aug_transform = None

        # Pre-tokenize the assistant header to find it in input_ids for label masking
        # Qwen3-VL uses: <|im_start|>assistant\n
        self._assistant_header_ids = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self._im_end_id = self.processor.tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        )[0]
        self._pad_id = self.processor.tokenizer.pad_token_id

    def _build_messages(self, sample, include_assistant=True):
        """Build chat messages for a single sample."""
        # Use plain local path — qwen_vl_utils supports local paths directly.
        # "file://D:/..." is malformed on Windows (needs "file:///D:/..."),
        # causing both decord and torchvision backends to fail.
        video_path = sample['file_path'].replace('\\', '/')

        video_content = {
            "type": "video",
            "video": video_path,
            "fps": self.config['video_fps'],
            "min_pixels": self.config['video_min_pixels'],
            "max_pixels": self.config['video_max_pixels'],
        }
        if self.config['video_total_pixels'] is not None:
            video_content["total_pixels"] = self.config['video_total_pixels']

        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": self.config['system_prompt']}
            ]},
            {"role": "user", "content": [
                video_content,
                {"type": "text", "text": self.config['user_prompt']},
            ]},
        ]

        if include_assistant:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": sample['text']}]
            })

        return messages

    def _find_subsequence(self, seq, subseq):
        """Find the LAST occurrence of subseq in seq. Returns start index or -1."""
        n, m = len(seq), len(subseq)
        for i in range(n - m, -1, -1):
            if seq[i:i + m] == subseq:
                return i
        return -1

    def _mask_labels(self, input_ids):
        """
        Create labels tensor: -100 for all tokens except the assistant's response.
        The response starts after '<|im_start|>assistant\n' and ends at '<|im_end|>'.
        """
        IGNORE_INDEX = -100
        labels = input_ids.clone()

        for i in range(labels.shape[0]):
            ids = input_ids[i].tolist()

            # Find last assistant header
            header_start = self._find_subsequence(ids, self._assistant_header_ids)
            if header_start == -1:
                # No assistant header found — mask everything (shouldn't happen)
                labels[i, :] = IGNORE_INDEX
                continue

            # Response starts right after the header
            response_start = header_start + len(self._assistant_header_ids)

            # Mask everything before the response
            labels[i, :response_start] = IGNORE_INDEX

            # Also mask padding tokens
            labels[i, input_ids[i] == self._pad_id] = IGNORE_INDEX

            del ids  # free the Python list for this row before moving to the next

        return labels

    def _snap_down_to_align(self, lo, hi, align):
        """Shrink [lo, hi) within [0, limit) so (hi-lo) is a multiple of `align`,
        keeping the window centered on its midpoint. Returns (lo, hi)."""
        span = hi - lo
        new_span = max(align, (span // align) * align)
        if new_span >= span:
            return lo, hi
        trim = span - new_span
        left = trim // 2
        right = trim - left
        return lo + left, hi - right

    def _maybe_log_snap_stats(self):
        self._snap_batches += 1
        if self._snap_batches % self._snap_log_every != 0:
            return
        if self._snap_total_clips == 0:
            return
        avg_trim = self._snap_total_trim_px / max(1, self._snap_snapped_clips)
        print(
            f"[signer_crop snap] batches={self._snap_batches} "
            f"clips={self._snap_total_clips} snapped={self._snap_snapped_clips} "
            f"avg_trim={avg_trim:.1f}px max_trim={self._snap_max_trim_px}px "
            f"degenerate_bboxes={len(self._snap_degenerate_bbox_ids)}",
            flush=True,
        )

    def _apply_signer_crop(self, videos, sample_bboxes):
        """Crop each clip to its precomputed signer bbox.

        The bbox was computed on original-video pixels (``frame_width`` /
        ``frame_height`` stored with it); ``process_vision_info`` may return
        frames at a different resolution, so we rescale the bbox to the
        current frame H/W before slicing.

        Reused by both ``__call__`` (training) and the validation generation
        loop so train/eval see the same pixel distribution.
        """
        use_crop = bool(self.config.get('use_signer_crop'))
        if not use_crop or not videos:
            return videos
        align = self._snap_align
        cropped = []
        for vid, bbox in zip(videos, sample_bboxes):
            self._snap_total_clips += 1
            if isinstance(vid, torch.Tensor):
                cur_h, cur_w = int(vid.shape[-2]), int(vid.shape[-1])
                arr = None
            else:
                arr = np.asarray(vid)
                vid = arr
                cur_h, cur_w = int(arr.shape[-2]), int(arr.shape[-1])

            cx1, cy1, cx2, cy2 = 0, 0, cur_w, cur_h
            bbox_used = False
            if use_crop and bbox is not None:
                x1, y1, x2, y2, fw, fh = bbox
                sx = cur_w / max(fw, 1)
                sy = cur_h / max(fh, 1)
                bx1 = max(0, int(round(x1 * sx)))
                by1 = max(0, int(round(y1 * sy)))
                bx2 = min(cur_w, int(round(x2 * sx)))
                by2 = min(cur_h, int(round(y2 * sy)))
                if bx2 - bx1 >= align and by2 - by1 >= align:
                    cx1, cy1, cx2, cy2 = bx1, by1, bx2, by2
                    bbox_used = True
                else:
                    # Degenerate/tiny bbox — fall back to full frame, warn once.
                    vid_id = None
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 6:
                        vid_id = (float(fw), float(fh), float(x1), float(y1), float(x2), float(y2))
                    if vid_id not in self._snap_degenerate_bbox_ids:
                        self._snap_degenerate_bbox_ids.add(vid_id)
                        print(
                            f"[signer_crop] degenerate bbox {bbox} -> using full frame "
                            f"({cur_w}x{cur_h}); will be shown once per unique bbox.",
                            flush=True,
                        )

            # Snap the chosen rectangle so (cy2-cy1) and (cx2-cx1) are multiples of `align`.
            orig_w = cx2 - cx1
            orig_h = cy2 - cy1
            cx1, cx2 = self._snap_down_to_align(cx1, cx2, align)
            cy1, cy2 = self._snap_down_to_align(cy1, cy2, align)
            trim = (orig_w - (cx2 - cx1)) + (orig_h - (cy2 - cy1))
            if trim > 0:
                self._snap_snapped_clips += 1
                self._snap_total_trim_px += trim
                if trim > self._snap_max_trim_px:
                    self._snap_max_trim_px = trim

            # If the whole frame was smaller than `align` on either axis,
            # _snap_down_to_align would have left it unchanged — only slice when safe.
            take_full_frame = (
                not bbox_used
                and cx1 == 0 and cy1 == 0
                and cx2 == cur_w and cy2 == cur_h
            )
            if take_full_frame:
                cropped.append(vid if isinstance(vid, torch.Tensor) else arr)
                continue

            if isinstance(vid, torch.Tensor):
                cropped.append(vid[..., cy1:cy2, cx1:cx2].contiguous())
            else:
                cropped.append(arr[..., cy1:cy2, cx1:cx2])

        self._maybe_log_snap_stats()
        return cropped

    def __call__(self, batch):
        """
        Process a batch of samples into model-ready inputs.
        batch: list of dicts from the Dataset
        """
        all_texts = []
        all_images = []
        all_videos = []
        all_video_metadatas = []
        all_video_kwargs = {}
        sample_bboxes = []  # parallel to all_videos: (x1,y1,x2,y2,fw,fh) or None

        for sample in batch:
            messages = self._build_messages(sample, include_assistant=True)

            # Get text template (not tokenized)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            all_texts.append(text)

            # Extract video frames using qwen_vl_utils
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16,
                return_video_kwargs=True, return_video_metadata=True,
            )
            del messages  # free the message dicts (contain video path strings & nested dicts)

            if images:
                all_images.extend(images)

            if videos is not None:
                vids, metas = zip(*videos)
                all_videos.extend(list(vids))
                all_video_metadatas.extend(list(metas))
                sample_bboxes.extend([sample.get('bbox')] * len(vids))
            del images, videos  # free intermediate references

            # Merge video_kwargs (should be same for all samples)
            if video_kwargs:
                all_video_kwargs.update(video_kwargs)

        # ── Signer-centric crop (applied BEFORE augmentation) ──
        all_videos = self._apply_signer_crop(all_videos, sample_bboxes)

        # ── Data Augmentation (training only) ──
        # Applied AFTER process_vision_info (frames decoded & resized to the Qwen3-VL
        # pixel budget) but BEFORE the processor (which applies mean/std normalization).
        # Each clip is augmented independently with freshly sampled random parameters.
        # process_vision_info returns (T, C, H, W) torch.Tensor; we ensure uint8 for
        # torchvision v2 transforms, then keep uint8 for the processor (which normalizes
        # internally regardless of the input dtype).
        if self.is_training and self.augmentation_enabled and all_videos:
            augmented = []
            for vid in all_videos:
                # Work on a local reference so we can clear the slot in all_videos
                # as soon as augmentation finishes — the original and augmented copy
                # would otherwise coexist in RAM for the full loop duration.
                # Ensure torch.Tensor uint8 [0, 255] for T_v2 compatibility.
                if not isinstance(vid, torch.Tensor):
                    vid = torch.as_tensor(np.asarray(vid))
                if vid.dtype != torch.uint8:
                    if vid.max() <= 1.0 + 1e-6:
                        vid = (vid * 255.0).clamp(0, 255).to(torch.uint8)
                    else:
                        vid = vid.clamp(0, 255).to(torch.uint8)

                T = vid.shape[0]

                # Speed perturbation: resample frames to simulate faster/slower signing.
                # s > 1 → faster (subsample, pad tail); s < 1 → slower (upsample, truncate).
                if (self.config.get('aug_speed_perturb', False)
                        and random.random() < self.config.get('aug_speed_perturb_prob', 0.4)):
                    s = random.uniform(
                        self.config.get('aug_speed_perturb_min', 0.9),
                        self.config.get('aug_speed_perturb_max', 1.1),
                    )
                    new_len = max(1, int(round(T / s)))
                    indices = torch.linspace(0, T - 1, new_len).long()
                    vid = vid[indices]
                    if new_len < T:
                        # Faster clip: fewer frames — pad tail with last frame
                        pad = vid[-1:].expand(T - new_len, -1, -1, -1).contiguous()
                        vid = torch.cat([vid, pad], dim=0)
                    else:
                        # Slower clip: more frames — truncate back to T
                        vid = vid[:T]

                # Temporal jitter: shift sequence by ±max_shift frames, padding with black.
                if (self.config.get('aug_temporal_jitter', False)
                        and random.random() < self.config.get('aug_temporal_jitter_prob', 0.5)):
                    max_shift = self.config.get('aug_temporal_jitter_range', 2)
                    shift = random.randint(-max_shift, max_shift)
                    if shift != 0:
                        if shift > 0:
                            # Shift right: black prefix, drop last `shift` frames
                            pad = torch.zeros(shift, *vid.shape[1:], dtype=torch.uint8)
                            vid = torch.cat([pad, vid[:-shift]], dim=0)
                        else:
                            # Shift left: drop first `|shift|` frames, black suffix
                            pad = torch.zeros(-shift, *vid.shape[1:], dtype=torch.uint8)
                            vid = torch.cat([vid[-shift:], pad], dim=0)

                # Spatial augmentations (color, blur, affine, erasing, …)
                # T_v2 applies the SAME random parameters to all T frames — correct for video.
                if self.aug_transform is not None:
                    vid = self.aug_transform(vid)

                augmented.append(vid)
                del vid  # drop the reference to the (possibly dtype-converted) clip
            all_videos.clear()  # release original frame tensors before assigning augmented
            all_videos = augmented
            del augmented

        # Capture one frame for debug saving AFTER augmentation.
        # Shape is (C, H, W) uint8 from the first frame of the first clip.
        _debug_frame = None
        if self.debug_save_images and all_videos:
            try:
                _debug_frame = all_videos[0][0]  # (C, H, W) uint8
            except Exception:
                pass

        # Tokenize and process through the Qwen3-VL processor
        inputs = self.processor(
            text=all_texts,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            video_metadata=all_video_metadatas if all_video_metadatas else None,
            return_tensors="pt",
            padding=True,
            do_resize=False,  # qwen_vl_utils already resized
            **all_video_kwargs,
        )

        # Free heavy intermediate lists now that the processor has consumed them
        del all_texts, all_images, all_videos, all_video_metadatas, all_video_kwargs

        # Build labels with masking
        inputs['labels'] = self._mask_labels(inputs['input_ids'])

        # Store metadata for logging
        inputs['_vids'] = [s['vid'] for s in batch]
        inputs['_ground_truths'] = [s['text'] for s in batch]
        inputs['_debug_frame'] = _debug_frame

        return inputs


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def compute_bleu(references, hypotheses, max_n=4):
    if not references or not hypotheses:
        return 0.0
    bleu = sacrebleu.BLEU(max_ngram_order=max_n, tokenize='13a')
    return bleu.corpus_score(hypotheses, [references]).score


def compute_wer(references, hypotheses):
    """Word Error Rate: (S + D + I) / N, averaged over the corpus, returned as %."""
    if not references or not hypotheses:
        return 0.0
    total_errors = 0
    total_ref_len = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        n, m = len(ref_words), len(hyp_words)
        # Levenshtein distance at word level
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                temp = dp[j]
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        total_errors += dp[m]
        total_ref_len += max(n, 1)
    return (total_errors / total_ref_len) * 100


def compute_meteor(references, hypotheses):
    """METEOR score using NLTK. Returns 0.0 if nltk is not available."""
    try:
        from nltk.translate.meteor_score import meteor_score as _meteor
    except ImportError:
        return 0.0
    if not references or not hypotheses:
        return 0.0
    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(_meteor([ref.strip().split()], hyp.strip().split()))
    return (sum(scores) / len(scores)) * 100


def compute_rouge_l(references, hypotheses):
    if not references or not hypotheses:
        return 0.0

    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()

        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue

        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]

        precision = lcs_len / n
        recall = lcs_len / m
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return (sum(scores) / len(scores)) * 100


# ═══════════════════════════════════════════════════════════════
#  CHECKPOINT MANAGER
# ═══════════════════════════════════════════════════════════════

class CheckpointManager:
    """Manages model checkpoints with a sliding window for periodic saves."""

    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_path = self.checkpoint_dir / 'best_model'

        # Reconstruct sliding window from existing checkpoints on disk
        existing = sorted(
            [d for d in self.checkpoint_dir.glob('checkpoint_step_*') if d.is_dir()],
            key=lambda p: int(p.name.split('_')[-1]),
        )
        self.periodic_checkpoints = existing
        if existing:
            print(f"  📋 CheckpointManager: found {len(existing)} existing checkpoint(s) "
                  f"(keep_last_n={keep_last_n}).")

    def save_periodic(self, model, optimizer, scheduler, epoch, global_step,
                      steps_done_in_epoch, best_val_loss, evals_without_improvement, elapsed_sec):
        """Save LoRA adapter + training state."""
        path = self.checkpoint_dir / f'checkpoint_step_{global_step}'
        path.mkdir(parents=True, exist_ok=True)

        try:
            # Save LoRA adapter weights
            model.save_pretrained(path / 'adapter')

            # Save training state
            state = {
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'steps_done_in_epoch': steps_done_in_epoch,
                'best_val_loss': best_val_loss,
                'evals_without_improvement': evals_without_improvement,
                'elapsed_sec': elapsed_sec,
                'rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
            }
            if torch.cuda.is_available():
                state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
            torch.save(state, path / 'training_state.pt')

            self.periodic_checkpoints.append(path)
            print(f"  💾 Saved periodic checkpoint: {path.name}")

        except Exception as e:
            print(f"  ⚠️  Warning: Failed to save checkpoint: {e}")
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            return

        # Evict old checkpoints
        while len(self.periodic_checkpoints) > self.keep_last_n:
            old_path = self.periodic_checkpoints.pop(0)
            if old_path.exists() and old_path != self.best_path:
                shutil.rmtree(old_path, ignore_errors=True)
                print(f"  🗑️  Evicted old checkpoint: {old_path.name}")

    def save_best(self, model, optimizer, scheduler, epoch, global_step,
                  steps_done_in_epoch, best_val_loss, evals_without_improvement, elapsed_sec):
        """Save best model checkpoint."""
        path = self.best_path
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

        try:
            model.save_pretrained(path / 'adapter')
            state = {
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'steps_done_in_epoch': steps_done_in_epoch,
                'best_val_loss': best_val_loss,
                'evals_without_improvement': evals_without_improvement,
                'elapsed_sec': elapsed_sec,
                'rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
            }
            if torch.cuda.is_available():
                state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
            torch.save(state, path / 'training_state.pt')
            print(f"  ⭐ Saved best model: {path.name}")
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to save best model: {e}")


class CSVLogger:
    def __init__(self, log_file, fieldnames):
        self.log_file = Path(log_file)
        self.fieldnames = fieldnames
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row_dict):
        row_dict['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row_dict.get(k, '') for k in self.fieldnames})


def format_time(seconds):
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if d > 0:
        return f"{d}d {h}h {m}m {s}s"
    elif h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def get_cuda_mem():
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        return current, peak
    return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════

@torch.inference_mode()
def validate(model, processor, val_loader, val_dataset, config, val_collator=None):
    """Full validation: loss + generation metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # ── Part 1: Validation Loss (teacher-forced) ──
    loss_pbar = tqdm(total=config['max_eval_batches'], desc="  Val loss", unit="batch",
                     leave=False, ncols=80, dynamic_ncols=False)

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= config['max_eval_batches']:
            break

        # Extract metadata before moving to device (discard — not needed for loss)
        batch.pop('_vids', None)
        batch.pop('_ground_truths', None)
        batch.pop('_debug_frame', None)

        batch_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**batch_gpu)

        total_loss += outputs.loss.item()
        num_batches += 1
        del outputs, batch_gpu, batch
        loss_pbar.update(1)
    loss_pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, MAX_PPL_CAP))

    # Free VRAM from loss evaluation before starting generation
    gc.collect()
    torch.cuda.empty_cache()

    # ── Part 2: Generate Text for BLEU / ROUGE-L ──
    _orig_use_cache = getattr(model.config, 'use_cache', None) if hasattr(model, 'config') else None
    if _orig_use_cache is not None:
        model.config.use_cache = True

    references = []
    hypotheses = []
    sources = []  # parallel to references/hypotheses
    sample_pairs = []

    num_samples = min(config['max_generate_samples'], len(val_dataset))
    sample_indices = torch.randperm(
        len(val_dataset), generator=torch.Generator().manual_seed(42)
    )[:num_samples].tolist()

    # Reuse the passed-in collator, or create one if not provided
    gen_collator = val_collator if val_collator is not None else Qwen3VLCollator(processor, config, is_training=False)

    gen_pbar = tqdm(total=num_samples, desc="  Val gen ", unit="sample",
                    leave=False, ncols=80, dynamic_ncols=False)

    for i in range(0, num_samples, config['val_gen_batch_size']):
        batch_end = min(i + config['val_gen_batch_size'], num_samples)
        batch_indices = sample_indices[i:batch_end]
        batch_samples = [val_dataset[idx] for idx in batch_indices]

        try:
            # Build messages WITHOUT assistant response for generation
            all_texts = []
            all_images = []
            all_videos = []
            all_video_metadatas = []
            all_video_kwargs = {}
            sample_bboxes = []  # parallel to all_videos

            for sample in batch_samples:
                messages = gen_collator._build_messages(sample, include_assistant=False)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                all_texts.append(text)

                images, videos, video_kwargs = process_vision_info(
                    messages, image_patch_size=16,
                    return_video_kwargs=True, return_video_metadata=True,
                )
                del messages
                if images:
                    all_images.extend(images)
                if videos is not None:
                    vids, metas = zip(*videos)
                    all_videos.extend(list(vids))
                    all_video_metadatas.extend(list(metas))
                    sample_bboxes.extend([sample.get('bbox')] * len(vids))
                del images, videos
                if video_kwargs:
                    all_video_kwargs.update(video_kwargs)

            # Apply the SAME signer crop as training to keep train/eval distributions matched.
            all_videos = gen_collator._apply_signer_crop(all_videos, sample_bboxes)

            inputs = processor(
                text=all_texts,
                images=all_images if all_images else None,
                videos=all_videos if all_videos else None,
                video_metadata=all_video_metadatas if all_video_metadatas else None,
                return_tensors="pt",
                padding=True,
                do_resize=False,
                **all_video_kwargs,
            )
            # Free decoded video frames before moving tensors to GPU
            del all_texts, all_images, all_videos, all_video_metadatas, all_video_kwargs
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config['val_max_new_tokens'],
                    num_beams=config['val_beam_size'],
                    length_penalty=config['val_length_penalty'],
                    no_repeat_ngram_size=config['val_no_repeat_ngram_size'],
                    repetition_penalty=config['val_repetition_penalty'],
                    do_sample=False,
                )

            # Trim input prefix from generated ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            batch_hypotheses = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for j, (sample, hyp_text) in enumerate(zip(batch_samples, batch_hypotheses)):
                ref_text = sample['text'].strip()
                hyp_text = hyp_text.strip()
                src = sample.get('source', 'unknown')

                references.append(ref_text)
                hypotheses.append(hyp_text)
                sources.append(src)

                if len(sample_pairs) < config['num_print_samples']:
                    sample_pairs.append((ref_text, hyp_text, src))

            del inputs, generated_ids, generated_ids_trimmed, batch_hypotheses, batch_samples
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠️  OOM during generation — skipping batch {i}")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ⚠️  Error during generation batch {i}: {e}")
            gc.collect()
            torch.cuda.empty_cache()

        gen_pbar.update(batch_end - i)
    gen_pbar.close()
    del sample_indices  # no longer needed after generation loop

    def _metrics_for(refs, hyps):
        if not refs:
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu4': 0.0,
                    'rouge_l': 0.0, 'wer': 0.0, 'meteor': 0.0, 'n': 0}
        return {
            'bleu1':   compute_bleu(refs, hyps, max_n=1),
            'bleu2':   compute_bleu(refs, hyps, max_n=2),
            'bleu4':   compute_bleu(refs, hyps, max_n=4),
            'rouge_l': compute_rouge_l(refs, hyps),
            'wer':     compute_wer(refs, hyps),
            'meteor':  compute_meteor(refs, hyps),
            'n':       len(refs),
        }

    overall = _metrics_for(references, hypotheses)
    per_source = {}
    for src_name in ('how2sign', 'openasl'):
        idxs = [i for i, s in enumerate(sources) if s == src_name]
        per_source[src_name] = _metrics_for(
            [references[i] for i in idxs],
            [hypotheses[i] for i in idxs],
        )

    if _orig_use_cache is not None:
        model.config.use_cache = _orig_use_cache

    model.train()

    return {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
        'bleu1': overall['bleu1'],
        'bleu2': overall['bleu2'],
        'bleu4': overall['bleu4'],
        'rouge_l': overall['rouge_l'],
        'wer': overall['wer'],
        'meteor': overall['meteor'],
        'per_source': per_source,  # {'how2sign': {...}, 'openasl': {...}}
        'sample_pairs': sample_pairs,
        'all_pairs': list(zip(references, hypotheses, sources)),
        'num_eval_batches': num_batches,
        'num_gen_samples': len(hypotheses),
    }


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(model, processor, train_loader, val_loader, val_dataset,
          optimizer, scheduler, train_config, ckpt_manager,
          train_csv_logger, val_csv_logger, gen_samples_csv_logger, tb_writer,
          _val_collator=None, _train_collator=None,
          start_epoch=1, start_global_step=0, start_steps_done_in_epoch=None,
          best_val_loss=float('inf'), start_evals_without_improvement=0, start_elapsed_sec=0.0,
          controller=None):
    """Full training loop for Qwen3-VL LoRA fine-tuning."""

    # Unpack config
    num_epochs = train_config['num_epochs']
    grad_accum_steps = train_config['grad_accum_steps']
    # max_grad_norm is read live from train_config['max_grad_norm'] each step
    # so that InteractiveController.check() can update it mid-training.
    log_every = train_config['log_every_steps']
    save_every = train_config['save_every_steps']
    eval_every = train_config['eval_every_steps']
    eval_every_warmup = train_config.get('eval_every_steps_warmup', eval_every)
    eval_warmup_threshold = train_config.get('eval_warmup_threshold', 0)
    patience = train_config['early_stopping_patience']
    label_smoothing = train_config['label_smoothing']

    # Calculate total optimizer steps
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * num_epochs

    # State tracking
    global_step = start_global_step
    evals_without_improvement = start_evals_without_improvement
    training_start = time.time() - start_elapsed_sec
    step_losses = []
    step_grad_norms = []
    log_step_start = time.time()

    # Wrap scheduler in a one-element list so InteractiveController can replace
    # it in-place (e.g. after the user types lr=<new_value>).
    _sched = [scheduler]

    # ── CUDA RT handle for sticky-error recovery ─────────────────────────────
    # Resolve the CUDA runtime DLL once and cache it.  Used by the CUDA error
    # handler to call cudaGetLastError(), which clears the sticky error flag so
    # tensor destructors (StorageImpl::~StorageImpl → CUDAEvent::record) don't
    # trigger the c10 AbortHandler while freeing memory after an OOM.
    _cudart_handle = None
    try:
        import ctypes, os, glob as _glob
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        # torch ships cudart64_12.dll (or cudart64_110.dll) inside its own lib dir
        _dll_candidates = (
            _glob.glob(os.path.join(_torch_lib, 'cudart64_*.dll'))
            + ['cudart64_12.dll', 'cudart64_120.dll', 'cudart64_110.dll', 'cudart64_11.dll']
        )
        for _dll in _dll_candidates:
            try:
                _cudart_handle = ctypes.CDLL(_dll)
                _ = _cudart_handle.cudaGetLastError   # verify symbol exists
                break
            except (OSError, AttributeError):
                _cudart_handle = None
    except Exception:
        pass

    def _clear_cuda_sticky_error():
        """Call cudaGetLastError() to reset CUDA's per-thread sticky error flag."""
        if _cudart_handle is not None:
            try:
                _cudart_handle.cudaGetLastError()
                return
            except Exception:
                pass
        # Fallback: PyTorch's internal binding (may be None in type stubs but
        # is always a valid ctypes handle at runtime when CUDA is initialized).
        try:
            _rt = torch.cuda.cudart()
            if _rt is not None:
                _rt.cudaGetLastError()  # type: ignore[union-attr]
        except Exception:
            pass

    # Proactive VRAM guard threshold: skip a batch if free VRAM falls below
    # this fraction of total to avoid RuntimeError: CUDA error: out of memory,
    # which corrupts the CUDA context and cannot be safely recovered.
    _total_vram = torch.cuda.get_device_properties(device).total_memory
    _vram_guard_bytes = max(int(_total_vram * 0.12), 1 * 1024 ** 3)  # 12% or 1 GB floor

    # Pre-collect trainable params for clip_grad_norm_
    _all_trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in _all_trainable_params)
    print(f"  Trainable parameters for grad clipping: {trainable_count:,}")

    # Clean up before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.train()

    if train_config.get('wait_for_manual_start', False):
        input("Press Enter to start training...")
    print("\n" + "=" * 60)
    print("🚀 TRAINING STARTED")
    print("=" * 60 + "\n")
    if start_global_step > 0:
        print(f"  ▶ Resuming training from step {start_global_step}\n")

    # Handle mid-epoch resume
    if start_steps_done_in_epoch is not None:
        steps_done_in_epoch = start_steps_done_in_epoch
    else:
        steps_done_in_epoch = start_global_step % optimizer_steps_per_epoch
    if steps_done_in_epoch == 0 and start_global_step > 0:
        start_epoch += 1
        print(f"  ⏭️  Checkpoint was at exact end of Epoch {start_epoch - 1}. Advancing to Epoch {start_epoch}.")

    aug_start_epoch = train_config.get('aug_start_epoch', 1)

    for epoch in range(start_epoch, num_epochs + 1):
        # Toggle data augmentation based on aug_start_epoch
        if _train_collator is not None:
            was_enabled = _train_collator.augmentation_enabled
            should_aug = epoch >= aug_start_epoch
            _train_collator.augmentation_enabled = should_aug

            if not should_aug:
                print(f"  🧊 Data augmentation: OFF — training on clean data (augmentation starts at epoch {aug_start_epoch})")
            elif should_aug and not was_enabled:
                print("")
                print("  " + "=" * 55)
                print("  🔥🎨 DATA AUGMENTATION ACTIVATED! 🎨🔥")
                print(f"  📊 Epoch {epoch}: switching from clean → augmented data")
                print("  " + "=" * 55)
                print("")
            else:
                print(f"  🎨 Data augmentation: ON")

        epoch_start = time.time()
        epoch_loss_sum = None
        epoch_microbatches = 0

        optimizer.zero_grad(set_to_none=True)
        valid_microbatches_in_window = 0
        window_raw_loss_sum = None

        # Mid-epoch resume: skip already-processed micro-batches at the sampler level
        # so the collator (video decoding) never runs for skipped batches.
        micro_steps_already_processed = 0
        # Offset added to `micro_step` to recover its absolute index within the epoch;
        # only non-zero on the sampler-level set_skip() resume path.
        _abs_micro_offset = 0
        if epoch == start_epoch and start_global_step > 0 and steps_done_in_epoch > 0:
            micro_steps_already_processed = steps_done_in_epoch * grad_accum_steps
            # Tell the bucket sampler to skip at index level (no collator overhead)
            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_skip'):
                train_loader.batch_sampler.set_skip(micro_steps_already_processed)
                _abs_micro_offset = micro_steps_already_processed
                print(f"  ⏭️  Resuming mid-epoch: skipping {micro_steps_already_processed} micro-batches (sampler-level, no video decoding)")
            else:
                print(f"  ⏭️  Resuming mid-epoch: skipping {micro_steps_already_processed} micro-batches (iterate-and-discard fallback)")

        # Number of micro-batches the loader will actually yield this epoch
        # (may be less than steps_per_epoch when resuming mid-epoch via set_skip).
        _actual_steps_this_epoch = steps_per_epoch - micro_steps_already_processed

        for micro_step, batch in enumerate(train_loader):

            # Fallback skip for non-bucket samplers (old checkpoints or shuffle DataLoader)
            if micro_step < micro_steps_already_processed and not (hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_skip')):
                del batch
                continue

            # Extract metadata (pop to keep them out of the model forward pass)
            batch.pop('_vids', None)
            batch.pop('_ground_truths', None)
            _dbg_frame = batch.pop('_debug_frame', None)

            is_accum_step = (micro_step + 1) % grad_accum_steps == 0
            # Use the number of batches actually yielded this epoch so is_last_step
            # is True on the final batch even when resuming mid-epoch via set_skip.
            is_last_step = (micro_step + 1) == _actual_steps_this_epoch

            if is_last_step and not is_accum_step:
                actual_accum_steps = (micro_step + 1) % grad_accum_steps
            else:
                actual_accum_steps = grad_accum_steps

            # ── Forward pass with mixed precision ──
            try:
                # Proactive VRAM guard — skip BEFORE touching the GPU if free memory
                # is below the safety threshold.  Prevents RuntimeError: CUDA error:
                # out of memory, which corrupts the CUDA context and cannot be safely
                # recovered (unlike torch.cuda.OutOfMemoryError).
                #
                # Use PyTorch-aware free memory, not raw CUDA free memory.
                # torch.cuda.mem_get_info() returns CUDA-level free bytes, but PyTorch's
                # caching allocator has already reserved a large pool.  That reserved-but-
                # idle memory is "used" from CUDA's view yet fully reusable by PyTorch.
                # Correct formula: (reserved − allocated) + truly_free_cuda
                _reserved  = torch.cuda.memory_reserved(device)
                _allocated = torch.cuda.memory_allocated(device)
                _cuda_free = torch.cuda.mem_get_info(device)[0]
                _free_vram = (_reserved - _allocated) + _cuda_free
                if _free_vram < _vram_guard_bytes:
                    print(f"  ⚠️  Low VRAM ({_free_vram / 1024**3:.2f} GB free) — skipping batch proactively")
                    del batch
                    continue

                batch_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                del batch  # free CPU tensors

                labels = batch_gpu['labels']
                if label_smoothing > 0.0:
                    # Avoid duplicate CE loss compute inside the model when using label smoothing.
                    forward_inputs = {k: v for k, v in batch_gpu.items() if k != 'labels'}
                else:
                    forward_inputs = batch_gpu

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(**forward_inputs)

                # Compute loss with label smoothing (model's built-in loss ignores it).
                # Shift logits/labels for causal LM: predict token t+1 from position t.
                # Early deletion strategy: free the full logits tensor BEFORE creating the
                # shifted copy so both never occupy VRAM simultaneously. At T≈2000 tokens and
                # vocab=151,936, this saves ~600 MB peak VRAM per step.
                if label_smoothing > 0.0:
                    logits = outputs.logits
                    del outputs                                      # free wrapper; logits still holds data
                    shift_logits = logits[:, :-1, :].contiguous()
                    del logits                                       # free full copy; shift is all we need
                    shift_labels = labels[:, 1:].contiguous()
                    del labels
                    raw_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        label_smoothing=label_smoothing,
                    )
                    del shift_logits, shift_labels
                else:
                    raw_loss = outputs.loss
                    del outputs, labels

                loss = raw_loss / actual_accum_steps
                loss.backward()                         # dispatch backward immediately (no .item() sync before this)
                raw_loss_detached = raw_loss.detach()

                if window_raw_loss_sum is None:
                    window_raw_loss_sum = raw_loss_detached
                else:
                    window_raw_loss_sum = window_raw_loss_sum + raw_loss_detached

                if epoch_loss_sum is None:
                    epoch_loss_sum = raw_loss_detached
                else:
                    epoch_loss_sum = epoch_loss_sum + raw_loss_detached

                del batch_gpu, forward_inputs, raw_loss, raw_loss_detached, loss  # outputs already deleted above

                valid_microbatches_in_window += 1

            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  OOM at micro_step {micro_step} — skipping batch (accumulated gradients preserved), clearing cache")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            except RuntimeError as e:
                # RuntimeError: CUDA error: out of memory  (and other CUDA runtime errors)
                # are *not* torch.cuda.OutOfMemoryError — they set CUDA's "sticky" error flag.
                # Any subsequent CUDA call (including cudaFree inside StorageImpl::~StorageImpl)
                # sees the stale error and fires c10 AbortHandler, crashing the process.
                #
                # Fix: call cudaGetLastError() FIRST to clear the sticky flag so that tensor
                # destructors (and all later CUDA ops) can proceed safely.  Only then is it
                # safe to delete locals, zero grads, synchronize, or empty the cache.
                err_str = str(e)
                if "CUDA" in err_str or "cuda" in err_str:
                    print(f"  ⚠️  CUDA RuntimeError at micro_step {micro_step}: {e} — clearing CUDA error state, skipping batch")
                    # ── Step 1: clear CUDA's sticky error flag ──────────────────────────
                    # Must happen before ANY tensor destructor runs.  The flag is "sticky":
                    # StorageImpl::~StorageImpl calls CUDAEvent::record which calls
                    # c10_cuda_check_implementation which re-throws on a non-zero flag,
                    # making std::terminate() fire (the c10 AbortHandler).
                    _clear_cuda_sticky_error()
                    # ── Step 2: release CUDA tensor locals now that destructors are safe ─
                    # Each del may internally touch CUDA (CUDAEvent::record for free-tracking).
                    # That's now safe because the flag was cleared in Step 1.
                    try: del batch_gpu
                    except NameError: pass
                    try: del forward_inputs
                    except NameError: pass
                    try: del outputs
                    except NameError: pass
                    try: del logits
                    except NameError: pass
                    try: del shift_logits
                    except NameError: pass
                    try: del shift_labels
                    except NameError: pass
                    try: del labels
                    except NameError: pass
                    try: del raw_loss
                    except NameError: pass
                    try: del loss
                    except NameError: pass
                    try: del raw_loss_detached
                    except NameError: pass
                    # ── Step 3: zero gradients and reclaim VRAM ─────────────────────────
                    # NOTE: do NOT call torch.cuda.synchronize() here — it waits for
                    # pending async CUDA work and if any of it failed it RE-ARMS the
                    # sticky error flag, causing the same abort on the next destructor.
                    try:
                        optimizer.zero_grad(set_to_none=True)
                    except Exception:
                        pass
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # ── Step 4: clear again ─────────────────────────────────────────────
                    # Steps 2–3 may have triggered new CUDA calls (CUDAEvent records,
                    # allocator bookkeeping) that re-set the flag.  Clear one final time
                    # so any remaining locals cleaned up by Python's frame teardown
                    # (autograd graph nodes, dict-comprehension temporaries) also land
                    # on a clean CUDA context when their destructors fire.
                    _clear_cuda_sticky_error()
                else:
                    print(f"  ⚠️  RuntimeError at micro_step {micro_step}: {e} — skipping batch (accumulated gradients preserved)")
                    gc.collect()
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  ⚠️  Error at micro_step {micro_step}: {e} — skipping batch (accumulated gradients preserved)")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            epoch_microbatches += 1

            if is_accum_step or is_last_step:
                if valid_microbatches_in_window == 0:
                    # All micro-batches in this window failed — skip optimizer step entirely
                    print(f"  ⚠️  All {actual_accum_steps} micro-batches failed in this accumulation window — skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    valid_microbatches_in_window = 0
                    window_raw_loss_sum = None
                    continue

                # If some micro-batches failed, re-scale gradients so the effective loss
                # magnitude stays correct (we divided each loss by actual_accum_steps, but
                # fewer micro-batches contributed).
                if valid_microbatches_in_window < actual_accum_steps:
                    scale = actual_accum_steps / valid_microbatches_in_window
                    for p in _all_trainable_params:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                    print(f"  ⚠️  Only {valid_microbatches_in_window}/{actual_accum_steps} micro-batches succeeded — "
                          f"re-scaled gradients by {scale:.2f}x")

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    _all_trainable_params, max_norm=train_config['max_grad_norm']
                ).item()

                window_valid_microbatches = valid_microbatches_in_window
                optimizer.step()
                _sched[0].step()
                optimizer.zero_grad(set_to_none=True)

                # ── Interactive controller check ──────────────────────────────────
                # Runs AFTER optimizer/scheduler step and gradient zero — the only
                # completely safe point to block the training loop for user input.
                if controller is not None:
                    controller.check(optimizer, _sched, train_config, global_step + 1, total_optimizer_steps)

                valid_microbatches_in_window = 0

                global_step += 1
                window_avg_loss = (window_raw_loss_sum / window_valid_microbatches).item()
                step_losses.append(window_avg_loss)
                step_grad_norms.append(grad_norm)
                window_raw_loss_sum = None

                # ── Debug: save one augmented frame ──
                # _dbg_frame is (C, H, W) uint8 captured after augmentation in the collator.
                if (_dbg_frame is not None
                        and global_step % train_config.get('aug_debug_save_interval', 500) == 0):
                    try:
                        _dbg_dir = Path(train_config.get('aug_debug_save_dir', '../data/debugging_images'))
                        _dbg_dir.mkdir(parents=True, exist_ok=True)
                        _dbg_np = _dbg_frame.cpu().numpy() if hasattr(_dbg_frame, 'cpu') else np.asarray(_dbg_frame)
                        # Squeeze out any stray leading dimensions
                        while _dbg_np.ndim > 3:
                            _dbg_np = _dbg_np[0]
                        # Transpose (C, H, W) → (H, W, C) for PIL
                        if _dbg_np.ndim == 3 and _dbg_np.shape[0] in (1, 3, 4):
                            _dbg_np = np.transpose(_dbg_np, (1, 2, 0))
                        # Ensure uint8 [0, 255]
                        if _dbg_np.dtype != np.uint8:
                            if _dbg_np.max() <= 1.0 + 1e-6:
                                _dbg_np = _dbg_np * 255.0
                            _dbg_np = np.clip(_dbg_np, 0, 255).astype(np.uint8)
                        # Squeeze single-channel to (H, W) for grayscale PIL
                        if _dbg_np.ndim == 3 and _dbg_np.shape[-1] == 1:
                            _dbg_np = _dbg_np[:, :, 0]
                        Image.fromarray(_dbg_np).save(
                            str(_dbg_dir / f"step_{global_step:07d}.jpg"), quality=90
                        )
                    except Exception:
                        pass  # Never let a debug save crash training
                del _dbg_frame

                # ── Logging ──
                if global_step % log_every == 0:
                    avg_loss = sum(step_losses) / len(step_losses)
                    avg_grad_norm = sum(step_grad_norms) / len(step_grad_norms)
                    _last_lrs = _sched[0].get_last_lr()
                    current_lr   = _last_lrs[0]  # tier 1 (LM) — primary display
                    lr_t1 = _last_lrs[0] if len(_last_lrs) > 0 else current_lr
                    lr_t2 = _last_lrs[1] if len(_last_lrs) > 1 else current_lr
                    lr_t3 = _last_lrs[2] if len(_last_lrs) > 2 else current_lr
                    elapsed = time.time() - training_start
                    log_elapsed = time.time() - log_step_start
                    speed = len(step_losses) / max(log_elapsed, 1e-6)
                    cuda_mem, cuda_peak = get_cuda_mem()

                    eta_seconds = (elapsed / global_step) * (total_optimizer_steps - global_step) if global_step > 0 else 0

                    train_ppl = math.exp(min(avg_loss, MAX_PPL_CAP))

                    print("=" * 160)
                    print(
                        f"[Step {global_step:>6d}/{total_optimizer_steps}] "
                        f"[Epoch {epoch:>2d}/{num_epochs}] "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"Train PPL: {train_ppl:.3f} | "
                        f"LR[T1/T2/T3]: {lr_t1:.2e}/{lr_t2:.2e}/{lr_t3:.2e} | "
                        f"Grad Norm: {avg_grad_norm:.3f} | "
                        f"Speed: {speed:.3f} steps/s | "
                        f"VRAM: {cuda_mem:.3f}/{cuda_peak:.3f} GB | "
                        f"Elapsed: {format_time(elapsed)} | "
                        f"ETA: {format_time(eta_seconds)}"
                    )

                    train_csv_logger.log({
                        'global_step': global_step,
                        'epoch': epoch,
                        'train_loss': f"{avg_loss:.6f}",
                        'train_ppl': f"{train_ppl:.4f}",
                        'lr': f"{current_lr:.3e}",
                        'lr_t1': f"{lr_t1:.3e}",
                        'lr_t2': f"{lr_t2:.3e}",
                        'lr_t3': f"{lr_t3:.3e}",
                        'grad_norm': f"{avg_grad_norm:.4f}",
                        'cuda_mem_gb': f"{cuda_mem:.2f}",
                        'cuda_peak_gb': f"{cuda_peak:.2f}",
                        'steps_per_sec': f"{speed:.2f}",
                        'elapsed_sec': f"{elapsed:.1f}",
                        'eta_sec': f"{eta_seconds:.1f}",
                    })

                    tb_writer.add_scalar('Loss/Train', avg_loss, global_step)
                    tb_writer.add_scalar('Perplexity/Train', train_ppl, global_step)
                    tb_writer.add_scalar('GradNorm/Train', avg_grad_norm, global_step)
                    tb_writer.add_scalar('LearningRate', current_lr, global_step)

                    step_losses.clear()
                    step_grad_norms.clear()
                    log_step_start = time.time()

                # ── Validation ──
                if global_step < eval_warmup_threshold:
                    _should_eval = (global_step % eval_every_warmup == 0)
                else:
                    _should_eval = ((global_step - eval_warmup_threshold) % eval_every == 0)

                if _should_eval:
                    print(f"\n{'─' * 60}")
                    print(f"  📊 Validation @ Step {global_step} / Epoch {epoch}")
                    print(f"{'─' * 60}")

                    val_results = validate(model, processor, val_loader, val_dataset, train_config, val_collator=_val_collator)

                    print(f"  Val Loss: {val_results['val_loss']:.4f} | "
                          f"Val PPL: {val_results['val_ppl']:.3f} | "
                          f"BLEU-1: {val_results['bleu1']:.4f} | "
                          f"BLEU-2: {val_results['bleu2']:.4f} | "
                          f"BLEU-4: {val_results['bleu4']:.4f} | "
                          f"ROUGE-L: {val_results['rouge_l']:.3f}% | "
                          f"WER: {val_results['wer']:.2f}% | "
                          f"METEOR: {val_results['meteor']:.2f}%")
                    print(f"  (Evaluated on {val_results['num_eval_batches']} batches, "
                          f"generated {val_results['num_gen_samples']} samples)")

                    if val_results['sample_pairs']:
                        print(f"\n  Sample Generations:")
                        for i, (ref, hyp, src) in enumerate(val_results['sample_pairs'], 1):
                            print(f"    [{i}] ({src}) REF: \"{ref}\"")
                            print(f"            HYP: \"{hyp}\"")

                    # Per-source summary line
                    ps = val_results['per_source']
                    print(f"\n  Per-source BLEU-4:  "
                          f"how2sign={ps['how2sign']['bleu4']:.3f}% (n={ps['how2sign']['n']})  |  "
                          f"openasl={ps['openasl']['bleu4']:.3f}% (n={ps['openasl']['n']})")

                    # Log to CSV
                    val_row = {
                        'global_step': global_step,
                        'epoch': epoch,
                        'val_loss': f"{val_results['val_loss']:.6f}",
                        'val_ppl': f"{val_results['val_ppl']:.4f}",
                        'bleu1': f"{val_results['bleu1']:.4f}",
                        'bleu2': f"{val_results['bleu2']:.4f}",
                        'bleu4': f"{val_results['bleu4']:.4f}",
                        'rouge_l': f"{val_results['rouge_l']:.4f}",
                        'wer': f"{val_results['wer']:.4f}",
                        'meteor': f"{val_results['meteor']:.4f}",
                        'num_eval_batches': val_results['num_eval_batches'],
                        'num_gen_samples': val_results['num_gen_samples'],
                    }
                    for _src in ('how2sign', 'openasl'):
                        _m = ps[_src]
                        val_row[f'bleu1_{_src}']   = f"{_m['bleu1']:.4f}"
                        val_row[f'bleu2_{_src}']   = f"{_m['bleu2']:.4f}"
                        val_row[f'bleu4_{_src}']   = f"{_m['bleu4']:.4f}"
                        val_row[f'rouge_l_{_src}'] = f"{_m['rouge_l']:.4f}"
                        val_row[f'wer_{_src}']     = f"{_m['wer']:.4f}"
                        val_row[f'meteor_{_src}']  = f"{_m['meteor']:.4f}"
                        val_row[f'n_{_src}']       = _m['n']
                    val_csv_logger.log(val_row)

                    # Log generated samples
                    for ref, hyp, src in val_results['all_pairs']:
                        gen_samples_csv_logger.log({
                            'global_step': global_step,
                            'epoch': epoch,
                            'source': src,
                            'reference': ref,
                            'hypothesis': hyp,
                        })

                    # TensorBoard
                    tb_writer.add_scalar('Loss/Val', val_results['val_loss'], global_step)
                    tb_writer.add_scalar('Perplexity/Val', val_results['val_ppl'], global_step)
                    tb_writer.add_scalar('BLEU/BLEU-1', val_results['bleu1'], global_step)
                    tb_writer.add_scalar('BLEU/BLEU-2', val_results['bleu2'], global_step)
                    tb_writer.add_scalar('BLEU/BLEU-4', val_results['bleu4'], global_step)
                    tb_writer.add_scalar('ROUGE-L', val_results['rouge_l'], global_step)
                    tb_writer.add_scalar('WER', val_results['wer'], global_step)
                    tb_writer.add_scalar('METEOR', val_results['meteor'], global_step)
                    for _src in ('how2sign', 'openasl'):
                        _m = val_results['per_source'][_src]
                        if _m['n'] > 0:
                            tb_writer.add_scalar(f'BLEU/BLEU-4_{_src}', _m['bleu4'], global_step)
                            tb_writer.add_scalar(f'ROUGE-L_{_src}',   _m['rouge_l'], global_step)
                            tb_writer.add_scalar(f'WER_{_src}',       _m['wer'], global_step)

                    # Early stopping / best model
                    elapsed = time.time() - training_start
                    if val_results['val_loss'] < best_val_loss:
                        best_val_loss = val_results['val_loss']
                        evals_without_improvement = 0
                        ckpt_manager.save_best(
                            model, optimizer, _sched[0], epoch, global_step,
                            (_abs_micro_offset + micro_step + 1) // grad_accum_steps,
                            best_val_loss,
                            evals_without_improvement, elapsed
                        )
                        print(f"\n  ⭐ New best val_loss: {best_val_loss:.4f}")
                    else:
                        evals_without_improvement += 1
                        print(f"\n  Val loss did not improve. ({evals_without_improvement}/{patience} evals without improvement)")

                    if evals_without_improvement >= patience:
                        print("=" * 60)
                        print(f"🛑 EARLY STOPPING triggered at step {global_step}, epoch {epoch}")
                        print(f"   Best val loss: {best_val_loss:.4f}")
                        print("=" * 60)
                        return global_step, best_val_loss

                    # Reclaim VRAM from validation before resuming training
                    del val_results
                    gc.collect()
                    torch.cuda.empty_cache()

                    print(f"{'─' * 60}\n")

                # ── Periodic Checkpoint ──
                if global_step % save_every == 0:
                    elapsed = time.time() - training_start
                    ckpt_manager.save_periodic(
                        model, optimizer, _sched[0], epoch, global_step,
                        (_abs_micro_offset + micro_step + 1) // grad_accum_steps,
                        best_val_loss,
                        evals_without_improvement, elapsed
                    )
                    # Checkpoint serialization creates temporary CPU copies — reclaim them
                    gc.collect()
                    torch.cuda.empty_cache()

        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = ((epoch_loss_sum / max(epoch_microbatches, 1)).item()
                          if epoch_loss_sum is not None else 0.0)
        epoch_loss_sum = None  # release the accumulated loss tensor for the next epoch
        print(f"\n{'=' * 60}")
        print(f"  ✅ Epoch {epoch}/{num_epochs} Complete")
        print(f"  Avg Train Loss: {epoch_avg_loss:.4f} | Time: {format_time(epoch_time)}")
        print(f"{'=' * 60}\n")

    return global_step, best_val_loss


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def _worker_init_fn(worker_id):
    try:
        import torch.multiprocessing as _mp
        _mp.set_sharing_strategy('file_system')
    except Exception:
        pass

def main():
    """Main entry point for Qwen3-VL training."""
    global device

    # ── Open timestamped log file ──
    _run_dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _log_dir = Path("../saved_metrics/train_config")
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _log_dir / f"qwen3vl_training_config_{_run_dt}.txt"

    class _Tee:
        """Write to both stdout and a file simultaneously."""
        def __init__(self, file_obj):
            self._file = file_obj
            self._stdout = sys.stdout
        def write(self, data):
            self._stdout.write(data)
            self._file.write(data)
        def flush(self):
            self._stdout.flush()
            self._file.flush()
        def __getattr__(self, name):
            return getattr(self._stdout, name)

    _log_file = open(_log_path, 'w', encoding='utf-8')
    _tee_out = _Tee(_log_file)
    _tee_err = _Tee(_log_file)
    _tee_err._stdout = sys.stderr  # _Tee mirrors to the original stream + file
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = _tee_out
    sys.stderr = _tee_err

    # ── Print Configuration Summary ──
    _W = 72
    _SEP = "=" * _W
    effective_batch = CONFIG['batch_size'] * CONFIG['grad_accum_steps']
    print("\n" + _SEP)
    print("  QWEN3-VL TRAINING CONFIGURATION")
    print(_SEP)

    print("\n  [ ENVIRONMENT ]")
    print(f"    Date                   : {_run_dt}")
    print(f"    Python                 : {sys.version.split()[0]}")
    print(f"    PyTorch                : {torch.__version__}")
    import transformers as _tf_ver; print(f"    Transformers           : {_tf_ver.__version__}")
    import peft as _peft_ver; print(f"    PEFT                   : {_peft_ver.__version__}")
    if _BNB_AVAILABLE:
        print(f"    Bitsandbytes           : {bnb.__version__}")
    print(f"    SacreBLEU              : {sacrebleu.__version__}")
    if torch.cuda.is_available():
        print(f"    CUDA                   : {torch.version.cuda}")
        print(f"    cuDNN                  : {torch.backends.cudnn.version()}")
        print(f"    GPU                    : {torch.cuda.get_device_name(0)}")
        print(f"    GPU VRAM               : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("\n  [ DATA ]")
    print(f"    Train TSV              : {CONFIG['data_train_tsv']}")
    print(f"    Val TSV                : {CONFIG['data_val_tsv']}")

    print("\n  [ MODEL ]")
    print(f"    Model                  : {CONFIG['model_name']}")
    print(f"    Attention impl         : {CONFIG['attn_implementation']}")
    print(f"    Precision              : {CONFIG['dtype']}")
    print(f"    QLoRA (4-bit)          : {CONFIG['use_qlora']}")

    print("\n  [ VIDEO PROCESSING ]")
    print(f"    FPS                    : {CONFIG['video_fps']}")
    print(f"    Min pixels             : {CONFIG['video_min_pixels']}")
    print(f"    Max pixels             : {CONFIG['video_max_pixels']}")
    print(f"    Total pixels budget    : {CONFIG['video_total_pixels']}")

    print("\n  [ PROMPTS ]")
    print(f"    System                 : {CONFIG['system_prompt']}")
    print(f"    User                   : {CONFIG['user_prompt']}")

    print("\n  [ LoRA ]")
    print(f"    Tier 1 — LM attn + MLP : r={CONFIG['lora_t1_r']}, alpha={CONFIG['lora_t1_alpha']}  modules: {', '.join(CONFIG['lora_t1_modules'])}")
    print(f"    Tier 2 — Vision encoder: r={CONFIG['lora_t2_r']}, alpha={CONFIG['lora_t2_alpha']}  modules: {', '.join(CONFIG['lora_t2_modules'])}")
    print(f"    Tier 3 — embed+head    : r={CONFIG['lora_t3_r']}, alpha={CONFIG['lora_t3_alpha']}  modules: {', '.join(CONFIG['lora_t3_modules'])}")
    print(f"    Dropout                : {CONFIG['lora_dropout']}")
    print(f"    DoRA / RSLoRA          : {CONFIG['lora_use_dora']} / {CONFIG['lora_use_rslora']}")

    print("\n  [ TRAINING ]")
    print(f"    Epochs                 : {CONFIG['num_epochs']}")
    print(f"    Batch size             : {CONFIG['batch_size']}  (effective: {effective_batch} with {CONFIG['grad_accum_steps']}x accum)")
    print(f"    Mixed precision        : bfloat16")
    print(f"    Grad clip max_norm     : {CONFIG['max_grad_norm']}")
    print(f"    Weight decay           : {CONFIG['weight_decay']}")
    print(f"    Adam betas             : {CONFIG['adam_betas']}")
    print(f"    8-bit AdamW            : {CONFIG['use_8bit_adam']}")
    print(f"    Gradient checkpointing : {CONFIG['use_gradient_checkpointing']}")
    print(f"    Bucket batching        : {CONFIG.get('use_bucket_batching', False)}")
    print(f"    Label smoothing        : {CONFIG['label_smoothing']}")
    print(f"    DataLoader workers     : {CONFIG['train_num_workers']} train / {CONFIG['val_num_workers']} val")

    print("\n  [ LEARNING RATE ]")
    warmup_info = f"{CONFIG['warmup_steps']} steps" if CONFIG['warmup_steps'] else f"{CONFIG['warmup_ratio']:.1%} of total"
    print(f"    Warmup                 : {warmup_info}")
    print(f"    Tier 1 (LM LoRA)       : {CONFIG['lr_tier1_lm']:.2e}  →  {CONFIG['min_lr_tier1']:.2e}  (cosine floor)")
    print(f"    Tier 2 (Vision LoRA)   : {CONFIG['lr_tier2_vision']:.2e}  →  {CONFIG['min_lr_tier2']:.2e}  (cosine floor)")
    print(f"    Tier 3 (Embed/Head)    : {CONFIG['lr_tier3_embed_head']:.2e}  →  {CONFIG['min_lr_tier3']:.2e}  (cosine floor)")

    print("\n  [ EVALUATION & CHECKPOINTING ]")
    print(f"    Eval every             : {CONFIG['eval_every_steps']} steps  (warmup: {CONFIG['eval_every_steps_warmup']} until step {CONFIG['eval_warmup_threshold']})")
    print(f"    Max eval batches       : {CONFIG['max_eval_batches']}")
    print(f"    Max gen samples        : {CONFIG['max_generate_samples']}")
    print(f"    Beam size              : {CONFIG['val_beam_size']}  |  Length penalty: {CONFIG['val_length_penalty']}  |  No-repeat ngram: {CONFIG['val_no_repeat_ngram_size']}  |  Rep penalty: {CONFIG['val_repetition_penalty']}")
    print(f"    Save every             : {CONFIG['save_every_steps']} steps  |  Keep last: {CONFIG['keep_last_n_checkpoints']}")
    print(f"    Early stopping pat.    : {CONFIG['early_stopping_patience']} evals")
    print(f"    Checkpoint dir         : {CONFIG['checkpoint_dir'].resolve()}")

    print("\n  [ DATA AUGMENTATION (training only) ]")
    _aug_start = CONFIG['aug_start_epoch']
    _aug_note = 'from the start' if _aug_start <= 1 else f'clean data for first {_aug_start - 1} epoch(s)'
    print(f"    Augmentation start     : epoch {_aug_start}  ({_aug_note})")
    print(f"    Horizontal flip        : DISABLED  (sign language handedness is semantically meaningful)")
    print(f"    Color jitter           : {'ON' if CONFIG['aug_color_jitter'] else 'OFF'}  (prob={CONFIG['aug_color_jitter_prob']}, b={CONFIG['aug_color_jitter_brightness']}, c={CONFIG['aug_color_jitter_contrast']}, s={CONFIG['aug_color_jitter_saturation']}, h={CONFIG['aug_color_jitter_hue']})")
    print(f"    Random grayscale       : {'ON' if CONFIG['aug_random_grayscale'] else 'OFF'}  (prob={CONFIG['aug_random_grayscale_prob']})")
    print(f"    Gaussian blur          : {'ON' if CONFIG['aug_gaussian_blur'] else 'OFF'}  (prob={CONFIG['aug_gaussian_blur_prob']}, kernel={CONFIG['aug_gaussian_blur_kernel']})")
    print(f"    Solarize               : {'ON' if CONFIG['aug_solarize'] else 'OFF'}  (prob={CONFIG['aug_solarize_prob']}, threshold={CONFIG['aug_solarize_threshold']})")
    print(f"    Equalize               : {'ON' if CONFIG['aug_equalize'] else 'OFF'}  (prob={CONFIG['aug_equalize_prob']})")
    print(f"    Random erasing         : {'ON' if CONFIG['aug_random_erasing'] else 'OFF'}  (prob={CONFIG['aug_random_erasing_prob']}, scale={CONFIG['aug_random_erasing_scale']})")
    print(f"    Affine                 : {'ON' if CONFIG['aug_affine'] else 'OFF'}  (prob={CONFIG['aug_affine_prob']}, deg=±{CONFIG['aug_affine_degrees']}, translate={CONFIG['aug_affine_translate']}, scale={CONFIG['aug_affine_scale_min']}–{CONFIG['aug_affine_scale_max']})")
    print(f"    Temporal jitter        : {'ON' if CONFIG['aug_temporal_jitter'] else 'OFF'}  (prob={CONFIG['aug_temporal_jitter_prob']}, ±{CONFIG['aug_temporal_jitter_range']} frames)")
    print(f"    Speed perturbation     : {'ON' if CONFIG['aug_speed_perturb'] else 'OFF'}  (prob={CONFIG['aug_speed_perturb_prob']}, range={CONFIG['aug_speed_perturb_min']}×–{CONFIG['aug_speed_perturb_max']}×)")
    print(f"    Debug image save       : {'ON' if CONFIG['aug_debug_save_images'] else 'OFF'}  (every {CONFIG['aug_debug_save_interval']} steps → {CONFIG['aug_debug_save_dir']})")

    print("\n  [ RESUME ]")
    print(f"    Resume training        : {CONFIG['resume_training']}")
    if CONFIG['resume_training']:
        print(f"    Resume step            : {CONFIG['resume_checkpoint_step'] or 'latest'}")

    lr_ov = CONFIG.get('lr_override', {})
    if lr_ov.get('enabled', False):
        print("\n  [ LR OVERRIDE (one-time) ]")
        print(f"    LR                     : {lr_ov['lr']:.2e} -> {lr_ov['eta_min']:.2e}")

    # Full CONFIG dump so nothing is missed
    print("\n  [ FULL CONFIG DUMP ]")
    for _k, _v in CONFIG.items():
        print(f"    {_k:30s}: {_v}")

    print("\n" + _SEP + "\n")

    # ── Tokenizer (loaded early so the Dataset can truncate reference text
    # to CONFIG['max_text_tokens']; the full processor is loaded with the model).
    print("📂 Loading tokenizer for reference-text truncation ...")
    from transformers import AutoTokenizer
    _trunc_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], local_files_only=True)

    # ── Load Data Manifests ──
    print("📂 Loading training data...")
    train_dataset = SignLanguageQwen3VLDataset(
        CONFIG['data_train_tsv'], CONFIG['tsv_sep'],
        bbox_csv_path=CONFIG['bbox_csv_train'] if CONFIG.get('use_signer_crop') else None,
        tokenizer=_trunc_tokenizer,
        max_text_tokens=CONFIG['max_text_tokens'],
    )

    print("\n📂 Loading validation data...")
    val_dataset = SignLanguageQwen3VLDataset(
        CONFIG['data_val_tsv'], CONFIG['tsv_sep'],
        bbox_csv_path=CONFIG['bbox_csv_val'] if CONFIG.get('use_signer_crop') else None,
        tokenizer=_trunc_tokenizer,
        max_text_tokens=CONFIG['max_text_tokens'],
    )
    del _trunc_tokenizer  # processor's tokenizer takes over once model is loaded

    print(f"\n📦 Train dataset: {len(train_dataset)} samples")
    print(f"📦 Val dataset:   {len(val_dataset)} samples")

    # ── Load Model ──
    print(f"\n🧠 Loading model: {CONFIG['model_name']}...")
    load_start = time.time()

    model_kwargs = {
        'dtype': torch.bfloat16,
    }

    # Attention implementation
    attn_impl = CONFIG['attn_implementation']
    try:
        import flash_attn  # noqa: F401
        model_kwargs['attn_implementation'] = 'flash_attention_2'
        print("  ✓ Using Flash Attention 2")
    except ImportError:
        if attn_impl == 'flash_attention_2':
            print("  ⚠️  Flash Attention 2 not available, falling back to SDPA")
            attn_impl = 'sdpa'
        model_kwargs['attn_implementation'] = attn_impl
        print(f"  Using {attn_impl}")

    # QLoRA quantization
    if CONFIG['use_qlora']:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=CONFIG['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=CONFIG['bnb_4bit_use_double_quant'],
        )
        model_kwargs['device_map'] = 'auto'
        print("  ✓ QLoRA enabled (4-bit quantization)")
    else:
        model_kwargs['device_map'] = 'cuda'

    model = AutoModelForImageTextToText.from_pretrained(CONFIG['model_name'], local_files_only=True, **model_kwargs)
    del model_kwargs  # no longer needed after model is loaded

    if CONFIG['use_qlora']:
        model = prepare_model_for_kbit_training(model)

    processor = AutoProcessor.from_pretrained(CONFIG['model_name'], local_files_only=True)

    load_time = time.time() - load_start
    model_vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  Model loaded in {load_time:.1f}s | VRAM: {model_vram:.2f} GB")

    # ── Apply LoRA ──
    print("\nApplying LoRA...")

    target_modules = (
        list(CONFIG['lora_t1_modules'])
        + list(CONFIG['lora_t2_modules'])
        + list(CONFIG['lora_t3_modules'])
    )

    # rank_pattern / alpha_pattern use PEFT's get_pattern_key() suffix matching:
    #   re.match(rf"(.*\.)?({key})$", module_path)
    # Tier 2 leaf names match only vision modules (e.g. 'proj' matches '.proj' not 'q_proj').
    # Tier 3 keys match only their respective modules (lm_head, embed_tokens, pos_embed).
    # Tier 1 (LM) needs no entry — it inherits the default r=lora_t1_r.
    # NOTE: lm_head and embed_tokens share their base weight (tie_word_embeddings=True).
    #   PEFT adds independent LoRA delta matrices — the tie is preserved.
    rank_pattern  = {k: CONFIG['lora_t2_r']    for k in CONFIG['lora_t2_modules']}
    alpha_pattern = {k: CONFIG['lora_t2_alpha'] for k in CONFIG['lora_t2_modules']}
    for k in CONFIG['lora_t3_modules']:
        rank_pattern[k]  = CONFIG['lora_t3_r']
        alpha_pattern[k] = CONFIG['lora_t3_alpha']

    lora_config = LoraConfig(
        r=CONFIG['lora_t1_r'],
        lora_alpha=CONFIG['lora_t1_alpha'],
        target_modules=target_modules,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_dropout=CONFIG['lora_dropout'],
        bias=CONFIG['lora_bias'],
        task_type='CAUSAL_LM',
        use_dora=CONFIG['lora_use_dora'],
        use_rslora=CONFIG['lora_use_rslora'],
        init_lora_weights=CONFIG.get('lora_init_weights', True),
    )
    del target_modules, rank_pattern, alpha_pattern  # consumed by lora_config

    # Check if resuming — load LoRA from checkpoint
    start_epoch = 1
    start_global_step = 0
    start_steps_done_in_epoch = None
    best_val_loss = float('inf')
    start_evals_without_improvement = 0
    start_elapsed_sec = 0.0
    training_state = None

    if CONFIG['resume_training']:
        ckpt_dir = Path(CONFIG['checkpoint_dir'])
        if CONFIG['load_best_model']:
            ckpt_path = ckpt_dir / 'best_model'
        elif CONFIG['resume_checkpoint_step'] is not None:
            ckpt_path = ckpt_dir / f'checkpoint_step_{CONFIG["resume_checkpoint_step"]}'
        else:
            # Find latest checkpoint
            existing = sorted(
                [d for d in ckpt_dir.glob('checkpoint_step_*') if d.is_dir()],
                key=lambda p: int(p.name.split('_')[-1]),
            )
            ckpt_path = existing[-1] if existing else None

        if ckpt_path and ckpt_path.exists():
            print(f"\n🔄 Resuming from checkpoint: {ckpt_path.name}")
            model = PeftModel.from_pretrained(model, ckpt_path / 'adapter', is_trainable=True)
            training_state = torch.load(ckpt_path / 'training_state.pt', map_location='cpu', weights_only=False)
            start_epoch = training_state['epoch']
            start_global_step = training_state['global_step']
            start_steps_done_in_epoch = training_state.get('steps_done_in_epoch')
            best_val_loss = training_state['best_val_loss']
            start_evals_without_improvement = training_state['evals_without_improvement']
            start_elapsed_sec = training_state.get('elapsed_sec', 0.0)
            print(f"  ✅ Resumed at step {start_global_step}, epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
        else:
            print("  No checkpoint found — starting fresh.")
            model = get_peft_model(model, lora_config)
    else:
        model = get_peft_model(model, lora_config)
    del lora_config  # consumed by get_peft_model / PeftModel

    # ── Parameter summary (per-tier breakdown) ──
    # Classify trainable parameters by tier by inspecting the leaf module name
    # that appears immediately before the lora_A/lora_B marker in each param path.
    _t1_set      = set(CONFIG['lora_t1_modules'])
    _t2_set      = set(CONFIG['lora_t2_modules'])
    _t3_set      = set(CONFIG['lora_t3_modules'])
    _lora_marks  = ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B')
    _tier_params = {'Tier 1': 0, 'Tier 2': 0, 'Tier 3': 0}
    _tier_layers = {'Tier 1': set(), 'Tier 2': set(), 'Tier 3': set()}

    for _pname, _param in model.named_parameters():
        if not _param.requires_grad:
            continue
        _parts = _pname.split('.')
        for _i, _p in enumerate(_parts):
            if _p in _lora_marks and _i > 0:
                _leaf = _parts[_i - 1]
                if   _leaf in _t2_set: _tier = 'Tier 2'
                elif _leaf in _t3_set: _tier = 'Tier 3'
                elif _leaf in _t1_set: _tier = 'Tier 1'
                else:                  _tier = None
                if _tier:
                    _tier_params[_tier] += _param.numel()
                    _tier_layers[_tier].add('.'.join(_parts[:_i]))
                break

    _total_params    = sum(p.numel() for p in model.parameters())
    _trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _frozen_params   = _total_params - _trainable_params
    _lora_total      = sum(_tier_params.values())

    _tier_meta = {
        'Tier 1': (f"LM attention + MLP",       CONFIG['lora_t1_r']),
        'Tier 2': (f"Vision encoder",            CONFIG['lora_t2_r']),
        'Tier 3': (f"Embeddings + output head",  CONFIG['lora_t3_r']),
    }

    _W = 80
    print("\n" + "=" * _W)
    print("  MODEL PARAMETER SUMMARY")
    print("=" * _W)
    print(f"  {'Tier':<10} {'Description':<28} {'Rank':>5} {'LoRA Layers':>12} {'Trainable Params':>18}")
    print("  " + "─" * (_W - 2))
    for _tl in ['Tier 1', 'Tier 2', 'Tier 3']:
        _desc, _r = _tier_meta[_tl]
        _n = len(_tier_layers[_tl])
        _p = _tier_params[_tl]
        print(f"  {_tl:<10} {_desc:<28} {_r:>5} {_n:>12,} {_p:>18,}")
    print("  " + "─" * (_W - 2))
    print(f"  {'Total LoRA':<10} {'(all trainable adapters)':<28} {'':>5} {'':>12} {_lora_total:>18,}")
    print(f"  {'Frozen':<10} {'Base model weights':<28} {'':>5} {'':>12} {_frozen_params:>18,}")
    print(f"  {'TOTAL':<10} {'All parameters':<28} {'':>5} {'':>12} {_total_params:>18,}")
    print("  " + "─" * (_W - 2))
    print(f"  Trainable: {_trainable_params:,} / {_total_params:,}  ({_trainable_params / _total_params * 100:.4f}% of total model)")
    print("=" * _W)
    del _t1_set, _t2_set, _t3_set, _lora_marks, _tier_params, _tier_layers
    del _lora_total, _frozen_params, _tier_meta, _W

    # ── Gradient Checkpointing ──
    if CONFIG['use_gradient_checkpointing']:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, 'config'):
            model.config.use_cache = False
        print("  ✓ Gradient checkpointing enabled")

    # ── torch.compile ──
    if CONFIG['use_torch_compile']:
        print(f"  ⚙️  Compiling model with mode='{CONFIG['torch_compile_mode']}'...")
        model = torch.compile(model, mode=CONFIG['torch_compile_mode'])

    lora_vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  VRAM after LoRA: {lora_vram:.2f} GB")

    # ── Create Data Loaders ──
    print("\n⚙️  Creating data loaders...")
    train_collator = Qwen3VLCollator(processor, CONFIG, is_training=True)
    val_collator = Qwen3VLCollator(processor, CONFIG, is_training=False)

    _nw_train = CONFIG['train_num_workers']

    if CONFIG.get('use_bucket_batching', False):
        train_batch_sampler = BucketBatchSampler(
            train_dataset, batch_size=CONFIG['batch_size'], shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=train_collator,
            num_workers=_nw_train,
            prefetch_factor=CONFIG['train_prefetch_factor'] if _nw_train > 0 else None,
            pin_memory=CONFIG['train_pin_memory'],
            persistent_workers=CONFIG['train_persistent_workers'] and _nw_train > 0,
            worker_init_fn=_worker_init_fn,
        )
        print(f"  Using bucket batch sampling ({len(train_batch_sampler)} batches)")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            collate_fn=train_collator,
            num_workers=_nw_train,
            prefetch_factor=CONFIG['train_prefetch_factor'] if _nw_train > 0 else None,
            pin_memory=CONFIG['train_pin_memory'],
            persistent_workers=CONFIG['train_persistent_workers'] and _nw_train > 0,
            worker_init_fn=_worker_init_fn,
        )

    _nw_val = CONFIG['val_num_workers']
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=val_collator,
        num_workers=_nw_val,
        prefetch_factor=CONFIG['val_prefetch_factor'] if _nw_val > 0 else None,
        pin_memory=CONFIG['val_pin_memory'],
        persistent_workers=CONFIG['val_persistent_workers'] and _nw_val > 0,
        worker_init_fn=_worker_init_fn,
    )

    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / CONFIG['grad_accum_steps'])
    total_optimizer_steps = optimizer_steps_per_epoch * CONFIG['num_epochs']

    print(f"  Train loader: {len(train_loader)} micro-batches/epoch")
    print(f"  Optimizer steps/epoch: {optimizer_steps_per_epoch}")
    print(f"  Total optimizer steps: {total_optimizer_steps}")

    # ── Optimizer: 3-tier differential LR ──
    # Partition trainable params by which LoRA tier their module name falls into.
    # Matching mirrors PEFT's rank_pattern suffix matching (key at end of module path).
    print("\n⚙️  Setting up optimizer (3-tier differential LR)...")

    def _matches_tier(name: str, module_list) -> bool:
        # LoRA param names look like `base_model.model.<...>.<tier_module>.lora_A.default.weight`
        # Match if any tier module name appears as a `.<mod>.` or `.<mod>` segment.
        return any(f".{m}." in name or name.endswith(f".{m}") or f".{m}.lora_" in name
                   for m in module_list)

    t1_params, t2_params, t3_params, unclassified = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _matches_tier(n, CONFIG['lora_t3_modules']):
            t3_params.append(p)
        elif _matches_tier(n, CONFIG['lora_t2_modules']):
            t2_params.append(p)
        elif _matches_tier(n, CONFIG['lora_t1_modules']):
            t1_params.append(p)
        else:
            unclassified.append(n)

    total_trainable = len(t1_params) + len(t2_params) + len(t3_params)
    print(f"  Tier 1 (LM):         {len(t1_params)} params  @ lr={CONFIG['lr_tier1_lm']:.2e} → {CONFIG['min_lr_tier1']:.2e}")
    print(f"  Tier 2 (Vision):     {len(t2_params)} params  @ lr={CONFIG['lr_tier2_vision']:.2e} → {CONFIG['min_lr_tier2']:.2e}")
    print(f"  Tier 3 (Embed/Head): {len(t3_params)} params  @ lr={CONFIG['lr_tier3_embed_head']:.2e} → {CONFIG['min_lr_tier3']:.2e}")
    if unclassified:
        raise RuntimeError(
            f"{len(unclassified)} trainable param(s) unclassified by LoRA tier matcher. "
            f"Examples: {unclassified[:5]}"
        )

    param_groups = [
        {"params": t1_params, "lr": CONFIG['lr_tier1_lm'],         "initial_lr": CONFIG['lr_tier1_lm'],         "tier": 1},
        {"params": t2_params, "lr": CONFIG['lr_tier2_vision'],     "initial_lr": CONFIG['lr_tier2_vision'],     "tier": 2},
        {"params": t3_params, "lr": CONFIG['lr_tier3_embed_head'], "initial_lr": CONFIG['lr_tier3_embed_head'], "tier": 3},
    ]

    if CONFIG['use_8bit_adam'] and _BNB_AVAILABLE:
        optimizer = bnb.optim.PagedAdamW8bit(
            param_groups,
            betas=CONFIG['adam_betas'],
            weight_decay=CONFIG['weight_decay'],
        )
        print("  ✓ Using Paged 8-bit AdamW (bitsandbytes) — optimizer states page to CPU under VRAM pressure")
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=CONFIG['adam_betas'],
            weight_decay=CONFIG['weight_decay'],
        )
        print("  Using standard AdamW")
    del t1_params, t2_params, t3_params, param_groups

    # ── Scheduler: per-group linear warmup + cosine to per-group eta_min ──
    if CONFIG['warmup_steps'] is not None:
        warmup_steps = CONFIG['warmup_steps']
    else:
        warmup_steps = int(total_optimizer_steps * CONFIG['warmup_ratio'])

    cosine_steps = max(total_optimizer_steps - warmup_steps, 1)

    tier_min_ratios = [
        CONFIG['min_lr_tier1'] / CONFIG['lr_tier1_lm'],
        CONFIG['min_lr_tier2'] / CONFIG['lr_tier2_vision'],
        CONFIG['min_lr_tier3'] / CONFIG['lr_tier3_embed_head'],
    ]

    def _make_lambda(min_ratio: float):
        def _fn(step: int) -> float:
            if step < warmup_steps:
                # Match prior linear warmup (from ~0 to 1.0 of base lr)
                return max(1e-8, (step + 1) / max(warmup_steps, 1))
            progress = (step - warmup_steps) / cosine_steps
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cos
        return _fn

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[_make_lambda(r) for r in tier_min_ratios]
    )

    print(f"\n📈 LR Schedule (per-group cosine with linear warmup):")
    print(f"  Warmup: {warmup_steps} steps | Cosine: {cosine_steps} steps")
    print(f"  T1 LM:         {CONFIG['lr_tier1_lm']:.2e} → {CONFIG['min_lr_tier1']:.2e}")
    print(f"  T2 Vision:     {CONFIG['lr_tier2_vision']:.2e} → {CONFIG['min_lr_tier2']:.2e}")
    print(f"  T3 Embed/Head: {CONFIG['lr_tier3_embed_head']:.2e} → {CONFIG['min_lr_tier3']:.2e}")

    # ── Resume optimizer/scheduler state ──
    if training_state is not None:
        try:
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            scheduler.load_state_dict(training_state['scheduler_state_dict'])
            print("  ✅ Optimizer and scheduler state restored")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not restore optimizer/scheduler state: {e}")

        # Restore RNG states (may be absent in older checkpoints — skip gracefully)
        if 'rng_state' in training_state:
            try:
                torch.set_rng_state(training_state['rng_state'])
                np.random.set_state(training_state['numpy_rng_state'])
                random.setstate(training_state['python_rng_state'])
                if torch.cuda.is_available() and 'cuda_rng_state_all' in training_state:
                    torch.cuda.set_rng_state_all(training_state['cuda_rng_state_all'])
                print("  ✅ RNG states restored")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not restore RNG states: {e}")
        else:
            print("  ℹ️  RNG states not in checkpoint (older save) — continuing with current RNG state")

        del training_state
        gc.collect()

    # ── Mid-Training LR Override ──
    lr_override = CONFIG.get('lr_override', {})
    if lr_override.get('enabled', False):
        remaining_steps = total_optimizer_steps - start_global_step

        # Per-tier peak and floor LRs — fall back to CONFIG defaults if not set
        override_lrs = [
            lr_override.get('lr_tier1_lm',         CONFIG['lr_tier1_lm']),
            lr_override.get('lr_tier2_vision',      CONFIG['lr_tier2_vision']),
            lr_override.get('lr_tier3_embed_head',  CONFIG['lr_tier3_embed_head']),
        ]
        override_min_lrs = [
            lr_override.get('min_lr_tier1', CONFIG['min_lr_tier1']),
            lr_override.get('min_lr_tier2', CONFIG['min_lr_tier2']),
            lr_override.get('min_lr_tier3', CONFIG['min_lr_tier3']),
        ]

        # Reset each param group's base LR independently
        for pg, peak_lr in zip(optimizer.param_groups, override_lrs):
            pg['lr']         = peak_lr
            pg['initial_lr'] = peak_lr

        # Rebuild per-tier cosine LambdaLR (no warmup — straight cosine from override peak)
        def _override_lambda(min_ratio: float):
            def _fn(step: int) -> float:
                progress = min(max(step / max(remaining_steps, 1), 0.0), 1.0)
                cos = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cos
            return _fn

        override_min_ratios = [
            mn / pk for pk, mn in zip(override_lrs, override_min_lrs)
        ]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[_override_lambda(r) for r in override_min_ratios]
        )

        _tier_names = ['T1 LM', 'T2 Vision', 'T3 Embed/Head']
        print(f"\n⚠️  LR OVERRIDE APPLIED  (per-tier, {remaining_steps} remaining steps):")
        for name, pk, mn in zip(_tier_names, override_lrs, override_min_lrs):
            print(f"    {name:16s}: {pk:.2e} → {mn:.2e}")
        print(f"    ⚠️  WARNING: Set lr_override.enabled = False after this run!\n")

    # ── Checkpoint Manager ──
    ckpt_manager = CheckpointManager(CONFIG['checkpoint_dir'], CONFIG['keep_last_n_checkpoints'])

    # ── CSV Loggers ──
    train_csv_logger = CSVLogger(CONFIG['train_log_file'], [
        'timestamp', 'global_step', 'epoch', 'train_loss', 'train_ppl',
        'lr', 'lr_t1', 'lr_t2', 'lr_t3',
        'grad_norm', 'cuda_mem_gb', 'cuda_peak_gb',
        'steps_per_sec', 'elapsed_sec', 'eta_sec',
    ])
    val_csv_logger = CSVLogger(CONFIG['val_log_file'], [
        'timestamp', 'global_step', 'epoch', 'val_loss', 'val_ppl',
        'bleu1', 'bleu2', 'bleu4', 'rouge_l', 'wer', 'meteor',
        'bleu1_how2sign', 'bleu2_how2sign', 'bleu4_how2sign',
        'rouge_l_how2sign', 'wer_how2sign', 'meteor_how2sign', 'n_how2sign',
        'bleu1_openasl', 'bleu2_openasl', 'bleu4_openasl',
        'rouge_l_openasl', 'wer_openasl', 'meteor_openasl', 'n_openasl',
        'num_eval_batches', 'num_gen_samples',
    ])
    gen_samples_csv_logger = CSVLogger(CONFIG['gen_samples_log_file'], [
        'timestamp', 'global_step', 'epoch', 'source', 'reference', 'hypothesis',
    ])

    # ── TensorBoard ──
    tb_dir = CONFIG['tensorboard_dir']
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Verify label masking (first batch debug) ──
    print("\n🔍 Label masking verification (first training batch):")
    try:
        first_sample = train_dataset[0]
        first_batch = train_collator([first_sample])
        input_ids = first_batch['input_ids'][0]
        labels = first_batch['labels'][0]

        # Decode only the non-masked label positions
        label_tokens = labels[labels != -100]
        decoded_labels = processor.tokenizer.decode(label_tokens, skip_special_tokens=True)
        print(f"  Ground truth: {first_sample['text']}")
        print(f"  Decoded labels (non-masked): {decoded_labels}")
        print(f"  Total tokens: {len(input_ids)} | Masked: {(labels == -100).sum().item()} | Unmasked: {(labels != -100).sum().item()}")

        del first_batch, first_sample, input_ids, labels, label_tokens, decoded_labels
    except Exception as e:
        print(f"  ⚠️  Warning: Label verification failed: {e}")
    gc.collect()
    torch.cuda.empty_cache()

    # ── Train ──
    print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "")
    
    controller = InteractiveController()
    final_step, final_best_loss = train(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=CONFIG,
        ckpt_manager=ckpt_manager,
        train_csv_logger=train_csv_logger,
        val_csv_logger=val_csv_logger,
        gen_samples_csv_logger=gen_samples_csv_logger,
        tb_writer=tb_writer,
        _val_collator=val_collator,
        _train_collator=train_collator,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
        start_steps_done_in_epoch=start_steps_done_in_epoch,
        best_val_loss=best_val_loss,
        start_evals_without_improvement=start_evals_without_improvement,
        start_elapsed_sec=start_elapsed_sec,
        controller=controller,
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final step:     {final_step}")
    print(f"  Best val loss:  {final_best_loss:.4f}")
    print("=" * 60)

    tb_writer.close()
    _log_file.close()
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr


def _safe_main():
    """Wrapper that guarantees log file and stdout/stderr are restored on crash."""
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout/stderr even if main() crashes partway through
        if not isinstance(sys.stdout, type(sys.__stdout__)):
            sys.stdout = sys.__stdout__
        if not isinstance(sys.stderr, type(sys.__stderr__)):
            sys.stderr = sys.__stderr__


if __name__ == '__main__':
    _safe_main()

# %%
