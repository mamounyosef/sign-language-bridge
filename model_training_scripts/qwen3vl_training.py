
## Imports & Setup

import os
import sys
import warnings
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Warning filters ───────────────────────────────────────────────────────────
# Suppress known-harmless per-step noise from internal HF/PEFT/torchvision code.
# These are informational deprecation/compatibility warnings that are not
# actionable for this training setup and flood the log at every forward pass.
warnings.filterwarnings(
    'ignore',
    message='.*use_reentrant.*',           # gradient checkpointing in HF internals
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message='.*video decoding and encoding.*',  # torchvision video API deprecation (qwen-vl-utils)
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message='.*tie_word_embeddings.*',     # tied lm_head LoRA adapter warning on load/resume
    category=UserWarning,
)
# ─────────────────────────────────────────────────────────────────────────────
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
import io
import torch.multiprocessing as mp
from PIL import Image
import cv2

# Windows: prevent ERROR_COMMITMENT_LIMIT with DataLoader workers
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _remap_path_for_kaggle(fp: str) -> str:
    """
    Converts a local Windows file_path from the TSV to its Kaggle equivalent.
    Works on paths from both sources:
      D:\\signbridge_dataset\\how2sign\\<split>\\<file>  →
        /kaggle/input/datasets/mamounhussam/sign-bridge/how2sign/how2sign/<split>/<file>
      D:\\signbridge_dataset\\openasl\\<split>\\<file>   →
        /kaggle/input/datasets/mamounhussam/sign-bridge/openasl/openasl/<split>/<file>
    The doubled subfolder (how2sign/how2sign, openasl/openasl) matches Kaggle's
    upload structure where the dataset root contains a folder named after the source.
    """
    _KAGGLE_BASE = '/kaggle/input/datasets/mamounhussam/sign-bridge'
    p = Path(fp.replace('\\', '/'))
    # parts are: ['D:', 'signbridge_dataset', '<source>', '<split>', '<file>']
    parts = [x for x in p.parts if x not in ('', '/') and ':' not in x]
    # parts[0] = 'signbridge_dataset', parts[1] = source, parts[2] = split, parts[3] = filename
    if len(parts) < 4:
        return fp  # unexpected format — return unchanged
    source   = parts[1]   # 'how2sign' or 'openasl'
    split    = parts[2]   # 'train', 'val', 'test'
    filename = parts[3]
    return f'{_KAGGLE_BASE}/{source}/{source}/{split}/{filename}'

def _remap_path_for_modal(fp: str) -> str:
    """
    Converts a local Windows file_path from the TSV to its Modal volume equivalent.
    Expects video clips mirrored under the modal_video_mount volume:
      D:\\signbridge_dataset\\how2sign\\train\\foo.mp4  →
        /videos/how2sign/train/foo.mp4
    Upload with:
      modal volume put signbridge-videos "D:/signbridge_dataset/how2sign" /how2sign
      modal volume put signbridge-videos "D:/signbridge_dataset/openasl"  /openasl
    """
    video_mount = CONFIG.get('modal_video_mount', '/videos')
    p = Path(fp.replace('\\', '/'))
    parts = [x for x in p.parts if x not in ('', '/') and ':' not in x]
    # Expected: ['signbridge_dataset', <source>, <split>, <filename>]
    if len(parts) < 4:
        return fp
    source   = parts[1]  # 'how2sign' or 'openasl'
    split    = parts[2]  # 'train', 'val', 'test'
    filename = parts[3]
    return f'{video_mount}/{source}/{split}/{filename}'

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

# Resolve the repo root whether running as a .py script or a Kaggle notebook cell.
# __file__ is undefined in notebook/REPL contexts; fall back to cwd in that case.
_SCRIPT_DIR = Path(__file__).parent if '__file__' in dir() else Path.cwd()
_REPO_ROOT  = _SCRIPT_DIR.parent if '__file__' in dir() else _SCRIPT_DIR

# %%
## Configuration

CONFIG = {
    # ── Data Paths ──
    'data_train_tsv': _REPO_ROOT / 'data' / '7_dataset_train.tsv',
    'data_val_tsv': _REPO_ROOT / 'data' / '7_dataset_val.tsv',
    'data_test_tsv': _REPO_ROOT / 'data' / '7_dataset_test.tsv',
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
    # Module-name substrings to EXCLUDE from 4-bit quantization (kept in bf16).
    # Vision tower stays in bf16 — NF4 was tuned for LM weight distributions and
    # starves the vision encoder on ASL (huge natural-image → signing shift).
    'quantize_skip_modules': ['visual'],

    # ── Video Processing ──
    'video_fps': 18,                              # Frames per second to sample
    'video_min_pixels': 4 * 32 * 32,              # Min visual tokens per frame pair (~4 tokens)
    'video_max_pixels': 116 * 32 * 32,            # Max visual tokens per frame pair (100 = 320*320 at patch_size=16, merge=2)
    'video_total_pixels': 20480 * 32 * 32,        # Total pixel budget cap across all frames (None = no cap)

    # ── Signer Cropping (pre-computed MediaPipe bboxes) ──
    # Bboxes are one static box per clip (padded union of pose landmarks across
    # sampled frames). Crop is applied to decoded frames BEFORE Qwen3-VL's
    # internal resize, so the full pixel budget lands on the signing region.
    # Unified CSV produced by data_code/23_merge_bbox_csvs.py (merges the
    # original split-specific CSVs so clips re-shuffled between splits are covered).
    'use_signer_crop': True,
    'bbox_csv': _REPO_ROOT / 'data' / 'bboxes_all.csv',

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
    
    'lora_t3_r': 8,
    'lora_t3_alpha': 16,
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
    'batch_size': 4,                              # VRAM constraint with vision LoRA — 1 sample per micro-batch
    # ── Gradient Accumulation ──
    'grad_accum_steps': 6,                       # Effective batch = 1 x 24 = 24

    # ── DataLoader Config ──
    'train_num_workers': 8,                       
    'train_prefetch_factor': 4,
    'train_pin_memory': True,                     # Pinned memory for async CPU→GPU DMA during training (disabled to reduce RAM usage and fragmentation)
    'train_persistent_workers': True,             # Keep worker alive across epochs — avoids re-spawn overhead

    'val_num_workers': 7,                          # 1 worker overlaps video decoding with GPU inference during validation
    'val_prefetch_factor': 4,                      # Pre-load 2 batches ahead during validation
    'val_pin_memory': True,                        # Pinned memory for async CPU→GPU DMA during validation
    'val_persistent_workers': False,               # Keep val worker alive across validation runs

    # ── Dataset Phases ──────────────────────────────────────────────────────────────────────
    # Phase A: OpenASL  (noisy, larger — pre-training phase)
    'openasl_source_name':             'openasl',
    'openasl_num_epochs':              2,
    # Two-phase freeze for OpenASL: Phase 1 = vision-only, Phase 2 = all tiers
    'openasl_enable_two_phase':        True,
    'openasl_phase1_fraction':         0.20,        # fraction of OpenASL optimizer steps in Phase 1
    # Curriculum for OpenASL: list of within-phase epoch numbers that use easy-first ordering
    'openasl_curriculum_epochs':       [1],
    # Augmentation for OpenASL: within-phase epoch at which augmentation turns ON
    'openasl_aug_start_epoch':         2,

    # Phase B: How2Sign  (clean, smaller — fine-tuning phase)
    'how2sign_source_name':            'how2sign',
    'how2sign_num_epochs':             2,
    # Two-phase freeze for How2Sign (recommended OFF — all tiers active from start)
    'how2sign_enable_two_phase':       False,
    'how2sign_phase1_fraction':        0.20,        # ignored when enable_two_phase=False
    # Curriculum for How2Sign
    'how2sign_curriculum_epochs':      [1],
    # Augmentation for How2Sign
    'how2sign_aug_start_epoch':        2,

    # ── Learning Rates (per-tier; each tier owns its intrinsic peak/floor) ──
    # OpenASL phase LRs
    # Tier 1 (LM LoRA): attn + MLP. Active only during Phase 2.
    'lr_openasl_tier1':             3e-5,
    'min_lr_openasl_tier1':         1.5e-6,
    # Tier 2 (Vision LoRA). Active from step 0 to end (uninterrupted through phase transition).
    'lr_openasl_tier2':             5e-5,
    'min_lr_openasl_tier2':         2.5e-6,
    # Tier 3 (Head LoRA): lm_head. Active only during Phase 2.
    'lr_openasl_tier3':             2e-5,
    'min_lr_openasl_tier3':         1e-6,
    # Tier 4 (InfoNCE projections: vision_proj + text_proj). Active from step 0 to end.
    'lr_openasl_tier4':             5e-5,
    'min_lr_openasl_tier4':         2.5e-6,

    # How2Sign fine-tuning LRs — lower than OpenASL (fresh restart at phase transition)
    # Tier 1 (LM LoRA)
    'lr_how2sign_tier1':               1e-5,
    'min_lr_how2sign_tier1':           5e-7,
    # Tier 2 (Vision LoRA)
    'lr_how2sign_tier2':               2e-5,
    'min_lr_how2sign_tier2':           1e-6,
    # Tier 3 (Head LoRA)
    'lr_how2sign_tier3':               1e-5,
    'min_lr_how2sign_tier3':           5e-7,
    # Tier 4 (InfoNCE projections)
    'lr_how2sign_tier4':               2e-5,
    'min_lr_how2sign_tier4':           1e-6,

    # easy_threshold_sec: shared across both phases
    'easy_threshold_sec':              4.0,

    # Single shared warmup ratio; each tier applies it against its own active-step count.
    # T2 / T4 active-step count = total_optimizer_steps.
    # T1 / T3 active-step count = phase2_optimizer_steps (born at phase transition).
    'warmup_ratio': 0.05,
    'weight_decay': 0.01,
    'adam_betas': (0.9, 0.98),
    # ── Per-tier gradient clipping ────────────────────────────────────────────
    # Each tier is clipped independently so that a high-norm tier (e.g. T2
    # Vision LoRA) cannot drag down the gradients of well-behaved tiers.
    # The joint norm is still computed and logged for diagnostics.
    # T2 gets a higher limit because vision LoRA gradients are naturally larger.
    # T4 (InfoNCE projections) gets a generous limit — it's a small head.
    'max_grad_norm_t1': 4.0,   # LM LoRA (attn + MLP)
    'max_grad_norm_t2': 6.0,   # Vision LoRA (encoder) — naturally larger grads
    'max_grad_norm_t3': 2.0,   # Head LoRA (lm_head)
    'max_grad_norm_t4': 2.0,   # InfoNCE projection layers
    # Legacy global fallback — used only for the spike-logging threshold and
    # the interactive-controller clip= command (which now sets all tiers).
    'max_grad_norm': 2.5,

    # ── Logging ──
    'log_every_steps': 1,
    'train_log_file': _REPO_ROOT / 'saved_metrics' / 'qwen3vl_train_log.csv',
    'val_log_file': _REPO_ROOT / 'saved_metrics' / 'qwen3vl_val_log.csv',
    'gen_samples_log_file': _REPO_ROOT / 'saved_metrics' / 'qwen3vl_gen_samples_log.csv',
    'tensorboard_dir': _REPO_ROOT / 'saved_metrics' / 'tensorboard' / 'qwen3vl_training',

    # ── Checkpointing ──
    'save_every_steps': 10,
    'keep_last_n_checkpoints': 3,
    'checkpoint_dir': _REPO_ROOT / 'checkpoints' / 'qwen3vl',

    # ── Evaluation ──
    'eval_every_steps': 80,
    'eval_every_steps_warmup': 80,               # More frequent eval early on
    'eval_warmup_threshold': 1000,                # Switch to normal eval freq after this step
    'max_eval_batches': 72,
    'num_print_samples': 5,
    'val_gen_batch_size': 12,                
    'max_generate_samples': 130,
    'val_beam_size': 1,                             # 1 = greedy (faster validation, honest diagnostic); run beam=4 on final checkpoint
    'val_length_penalty': 1.0,                      # > 1.0 favors longer outputs (counters BLEU brevity penalty); < 1.0 favors shorter
    'val_no_repeat_ngram_size': 0,                  # Block any n-gram from repeating; improves BLEU precision (0 = disabled)
    'val_repetition_penalty': 1.0,
    'val_max_new_tokens': 70,

    # ── Early Stopping ──
    'early_stopping_patience': 14,                # Evals without improvement before stopping

    # ── Performance & Memory Optimizations ──
    'use_8bit_adam': True,
    'use_gradient_checkpointing': False,
    'use_liger_kernel': True,                     # Master switch — fused Triton kernels for faster forward/backward + lower VRAM
    'liger_rms_norm': True,                       # LigerRMSNorm: 113 modules (28 LM decoder layers × 4 + 1 final norm)
    'liger_layer_norm': True,                     # LigerLayerNorm: 52 modules (24 vision blocks × 2 + 4 merger norms)
    # liger_swiglu / liger_cross_entropy: not yet supported for Qwen3VL class names;
    # will be enabled when liger adds apply_liger_kernel_to_qwen3_vl.
    'use_torch_compile': False,
    'torch_compile_mode': 'reduce-overhead',      # 'reduce-overhead' (CUDA graphs, fewer kernel launches) | 'default' | 'max-autotune'

    # ── Resuming ──
    # Fresh run: old checkpoints are incompatible with LM-only quantization (new bf16
    # vision path), new Tier-3 LoRA rank (2 → 8), and the new InfoNCE projection modules.
    'resume_training': True,
    'load_best_model': False,
    'resume_checkpoint_step': 500,            # None = latest, or specific step number

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
        'lr_openasl_tier1':             1e-5,       # peak LR for Tier 1 (LM attn + MLP)
        'lr_openasl_tier2':             5e-5,       # peak LR for Tier 2 (Vision encoder)
        'lr_openasl_tier3':             1e-5,       # peak LR for Tier 3 (Embed + LM head)
        'min_lr_openasl_tier1':         1e-6,       # Cosine floor for Tier 1
        'min_lr_openasl_tier2':         5e-7,       # Cosine floor for Tier 2
        'min_lr_openasl_tier3':         1e-6,       # Cosine floor for Tier 3
    },

    # ── Bucket Batch Sampling ──
    'use_bucket_batching': True,                  # Group clips by duration to minimise padding waste

    # ── Source-balanced sampling ──
    # WeightedRandomSampler with weight = 1 / count_of_source. Expected per-batch
    # balance ~50/50 how2sign/openasl. With replacement — some clips seen >1× per
    # epoch, others not seen. Deterministic given (base_seed, epoch).
    'sampling_strategy': 'weighted',              # 'weighted' | 'uniform'
    'balance_by_source': True,

    # ── Temporal jitter padding mode ──
    # 'edge_replicate' = repeat first/last frame into shifted slots (in-domain).
    # 'black'          = fill shifted slots with zeros (legacy; OOD for vision tower).
    'temporal_jitter_mode': 'edge_replicate',

    # ── InfoNCE auxiliary contrastive loss ──
    # Aligns pooled video embedding with pooled reference-text embedding across
    # the accumulated effective batch. Primary anti-mode-collapse signal.
    'enable_infonce': True,
    'infonce_temperature': 0.10,                  # Relaxed from 0.07 to reduce high-variance logit scaling
    'infonce_proj_dim': 256,
    'infonce_queue_size': 64,                     # MoCo-style queue depth: max number of past (v, t) embeddings retained.
    'infonce_lambda_phase1': 0.3,                 # Strong in Phase 1 (LM frozen; vision's only alignment signal).
    'infonce_lambda_phase2': 0.1,                 # Auxiliary in Phase 2.
    'infonce_min_pairs': 4,                       # Skip InfoNCE if fewer than this many micro-batches succeeded in window.

    # ── Diagnostics ──
    'grad_norm_spike_threshold': 500.0,           # Log batch shape + detailed info when global grad_norm exceeds this.
    'log_distinct_n': 2,                          # distinct-N diversity metric on val generations (1 = unigrams, 2 = bigrams, ...).

    # ── Label Smoothing ──
    # Gentle smoothing helps with OpenASL's noisy ASR-style captions without
    # overly dulling the training signal on clean How2Sign data.
    'label_smoothing': 0.04,

    # ── Seed ──
    'seed': 42,

    # ── Runtime ──
    'wait_for_manual_start': True,               # Interactive safety gate before training loop starts

    # ────── Data Augmentation ──────
    # ── CLAHE (always-on preprocessing, all splits including inference) ──
    # Applied after signer crop, before augmentation. Replaces RandomEqualize.
    # Operates on L channel in LAB space to preserve hue while enhancing local contrast.
    'use_clahe': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid': [8, 8],

    # ── Landmark overlay (pre-computed offline by 21_extract_landmarks.py) ──
    'use_landmark_overlay': True,            # Enable after running 21_extract_landmarks.py
    'landmark_overlay_prob': 1.0,
    'landmark_parquet_train': _REPO_ROOT / 'data' / 'landmarks_train.parquet',
    'landmark_parquet_val':   _REPO_ROOT / 'data' / 'landmarks_val.parquet',
    'landmark_parquet_test':  _REPO_ROOT / 'data' / 'landmarks_test.parquet',

    'aug_temporal_jitter': True,
    'aug_temporal_jitter_range': 2,           # Max frames to shift (±2)
    'aug_temporal_jitter_prob': 0.4,

    'aug_color_jitter': True,
    'aug_color_jitter_prob': 0.40,             
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
    'aug_solarize_threshold': 225,            # 0–255; pixels above this get inverted

    'aug_equalize': False,                    # Replaced by always-on CLAHE below

    'aug_random_erasing': False,              # Randomly blacks out small patches — occlusion robustness
    'aug_random_erasing_prob': 0.2,
    'aug_random_erasing_scale': (0.02, 0.1), # Fraction of frame area erased
    'aug_random_erasing_ratio': (0.3, 3.3),  # Aspect ratio of erased region

    'aug_affine': True,                       # Small rotation + translation + scale — camera angle variation
    'aug_affine_prob': 0.25,
    'aug_affine_degrees': 2,                  # ±2° rotation
    'aug_affine_translate': 0.05,             # ±5% of frame width/height
    'aug_affine_scale_min': 0.95,             # 95%–105% zoom
    'aug_affine_scale_max': 1.00,

    'aug_speed_perturb': True,                # Simulate faster/slower signing by resampling frames
    'aug_speed_perturb_prob': 0.15,
    'aug_speed_perturb_min': 0.9,             # 0.9× = 10% slower (frames stretched)
    'aug_speed_perturb_max': 1.0,             # 1.0× = no change (frames compressed, tail padded)

    # ── Debug: save augmented frames to disk ──
    'aug_debug_save_images': True,
    'aug_debug_save_interval': 2,          # Save one frame every N optimiser steps
    'aug_debug_save_dir': _REPO_ROOT / 'data' / 'debugging_images',

    # ── Kaggle ──
    # Set True when running on Kaggle. Remaps all paths to /kaggle/input|working.
    # file_path values in the TSVs are remapped on-the-fly; no new TSV files needed.
    'is_kaggle': False,

    # ── Modal ──────────────────────────────────────────────────────────────────
    # Set 'is_modal': True to run on Modal's cloud GPUs instead of locally.
    # Run with:  modal run model_training_scripts/qwen3vl_training.py
    # All paths are remapped to Modal volume mount points automatically.
    'is_modal': True,

    # Hardware
    # GPU strings: 'T4', 'A10G', 'A100', 'A100-80GB', 'H100', 'H100:2' (multi-GPU)
    # Fallback list also accepted: ['H100', 'A100-80GB']
    'modal_gpu': 'A100-80GB',
    'modal_cpu': 10,                    # Virtual CPUs allocated to the container
    'modal_memory': 81920,             # RAM in MB (80 GB)
    'modal_timeout': 24 * 3600,        # Max wall-clock seconds before Modal kills job
    'modal_retries': 0,                # Container-level retries on failure (not step-level)

    # App
    'modal_app_name': 'signbridge-training',
    'modal_sync_metrics_on_exit': False,       # Pull /saved_metrics back to local machine after run?

    # Volumes — must be created beforehand with `modal volume create <name>`
    # signbridge-data  : TSV files, bboxes_all.csv, landmarks parquets
    # signbridge-videos: raw video clips (mirrors D:\signbridge_dataset\)
    # signbridge-outputs: checkpoints, logs, tensorboard
    'modal_data_volume':   'signbridge-data',
    'modal_video_volume':  'signbridge-videos',
    'modal_output_volume': 'signbridge-outputs',

    # Mount paths inside the container
    'modal_data_mount':   '/data',
    'modal_video_mount':  '/videos',
    'modal_output_mount': '/outputs',

    # Modal Secret containing HF_TOKEN (create with `modal secret create huggingface-secret HF_TOKEN=...`)
    # Set to None to skip (model downloaded anonymously — only works for public models)
    'modal_hf_secret': 'huggingface-secret',
}

# ── Kaggle path overrides ─────────────────────────────────────────────────────
# Applied immediately after CONFIG is defined. All output paths go to
# /kaggle/working/ (the only writable location). Input datasets are read-only
# under /kaggle/input/datasets/mamounhussam/sign-bridge/.
if CONFIG.get('is_kaggle'):
    _KIN   = Path('/kaggle/input/datasets/mamounhussam/sign-bridge')
    _KOUT  = Path('/kaggle/working')
    CONFIG.update({
        'data_train_tsv':         _KIN  / '7_dataset_train.tsv',
        'data_val_tsv':           _KIN  / '7_dataset_val.tsv',
        'data_test_tsv':          _KIN  / '7_dataset_test.tsv',
        'bbox_csv_train':         _KIN  / 'bboxes_all.csv',
        'bbox_csv_val':           _KIN  / 'bboxes_all.csv',
        'bbox_csv_test':          _KIN  / 'bboxes_all.csv',
        'landmark_parquet_train': _KIN  / 'landmarks_train.parquet',
        'landmark_parquet_val':   _KIN  / 'landmarks_val.parquet',
        'landmark_parquet_test':  _KIN  / 'landmarks_test.parquet',
        'train_log_file':         _KOUT / 'saved_metrics' / 'qwen3vl_train_log.csv',
        'val_log_file':           _KOUT / 'saved_metrics' / 'qwen3vl_val_log.csv',
        'gen_samples_log_file':   _KOUT / 'saved_metrics' / 'qwen3vl_gen_samples_log.csv',
        'tensorboard_dir':        _KOUT / 'saved_metrics' / 'tensorboard' / 'qwen3vl_training',
        'checkpoint_dir':         _KOUT / 'checkpoints' / 'qwen3vl',
        'aug_debug_save_dir':     _KOUT / 'data' / 'debugging_images',
        'model_name':             '/kaggle/input/models/qwen-lm/qwen-3-vl/transformers/2b-instruct/1',
    })
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
# ─────────────────────────────────────────────────────────────────────────────

# ── Modal path overrides ──────────────────────────────────────────────────────
# Applied when is_modal=True. Input data is read from Modal volumes; outputs
# (checkpoints, logs) go to the output volume. Video file_path values in the
# TSVs are remapped on-the-fly in the dataset loader via _remap_path_for_modal().
if CONFIG.get('is_modal'):
    _MDATA = Path(CONFIG['modal_data_mount'])
    _MOUT  = Path(CONFIG['modal_output_mount'])
    CONFIG.update({
        # TSVs were uploaded into a 'data/' subdir of the volume
        # (modal volume put signbridge-data "data/..." /data/...) so they live
        # one level deeper than the volume root once mounted at /data.
        'data_train_tsv':         _MDATA / 'data' / '7_dataset_train.tsv',
        'data_val_tsv':           _MDATA / 'data' / '7_dataset_val.tsv',
        'data_test_tsv':          _MDATA / 'data' / '7_dataset_test.tsv',
        # bboxes and landmarks were uploaded to the volume root, so they sit
        # directly under the mount point.
        'bbox_csv':               _MDATA / 'bboxes_all.csv',
        'landmark_parquet_train': _MDATA / 'landmarks_train.parquet',
        'landmark_parquet_val':   _MDATA / 'landmarks_val.parquet',
        'landmark_parquet_test':  _MDATA / 'landmarks_test.parquet',
        'train_log_file':         _MOUT  / 'saved_metrics' / 'qwen3vl_train_log.csv',
        'val_log_file':           _MOUT  / 'saved_metrics' / 'qwen3vl_val_log.csv',
        'gen_samples_log_file':   _MOUT  / 'saved_metrics' / 'qwen3vl_gen_samples_log.csv',
        'tensorboard_dir':        _MOUT  / 'saved_metrics' / 'tensorboard' / 'qwen3vl_training',
        'checkpoint_dir':         _MOUT  / 'checkpoints'   / 'qwen3vl',
        'aug_debug_save_dir':     _MOUT  / 'data'          / 'debugging_images',
        'wait_for_manual_start':  False,  # no interactive terminal on Modal
    })
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed_all(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])


def _get_phase_info(global_epoch: int, openasl_epochs: int) -> tuple[str, int]:
    """Return (dataset_phase_name, epoch_within_phase) for a given global epoch (1-indexed)."""
    if global_epoch <= openasl_epochs:
        return 'openasl', global_epoch
    else:
        return 'how2sign', global_epoch - openasl_epochs


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
        print(f"  Grad clip : T1={train_config.get('max_grad_norm_t1', train_config['max_grad_norm'])}  T2={train_config.get('max_grad_norm_t2', train_config['max_grad_norm'])}  T3={train_config.get('max_grad_norm_t3', train_config['max_grad_norm'])}  T4={train_config.get('max_grad_norm_t4', train_config['max_grad_norm'])}")
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
                        train_config.get('min_lr_openasl_tier1', 1e-6),
                        train_config.get('min_lr_openasl_tier2', 5e-7),
                        train_config.get('min_lr_openasl_tier3', 1e-6),
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
                    parts = cmd[5:].split(',')
                    if len(parts) == 4:
                        new_clips = [float(p) for p in parts]
                        if any(c <= 0 for c in new_clips):
                            print(f"  ✗  All clip norms must be positive.")
                            continue
                        for key, val in zip(['max_grad_norm_t1','max_grad_norm_t2','max_grad_norm_t3','max_grad_norm_t4'], new_clips):
                            train_config[key] = val
                        print(f"  ✓  Per-tier clip norms: T1={new_clips[0]}  T2={new_clips[1]}  T3={new_clips[2]}  T4={new_clips[3]}")
                    elif len(parts) == 1:
                        new_clip = float(parts[0])
                        if new_clip <= 0:
                            print(f"  ✗  Clip must be positive.  Got: {new_clip}")
                            continue
                        old_t1 = train_config.get('max_grad_norm_t1', train_config['max_grad_norm'])
                        old_t2 = train_config.get('max_grad_norm_t2', train_config['max_grad_norm'])
                        old_t3 = train_config.get('max_grad_norm_t3', train_config['max_grad_norm'])
                        old_t4 = train_config.get('max_grad_norm_t4', train_config['max_grad_norm'])
                        train_config['max_grad_norm'] = new_clip
                        train_config['max_grad_norm_t1'] = new_clip
                        train_config['max_grad_norm_t2'] = new_clip
                        train_config['max_grad_norm_t3'] = new_clip
                        train_config['max_grad_norm_t4'] = new_clip
                        print(f"  ✓  All tiers clip: T1 {old_t1}→{new_clip}  T2 {old_t2}→{new_clip}  T3 {old_t3}→{new_clip}  T4 {old_t4}→{new_clip}")
                        print(f"     Tip: use clip=t1,t2,t3,t4 to set tiers independently  e.g. clip=1.0,5.0,1.0,5.0")
                    else:
                        print(f"  ✗  Expected 1 value (all tiers) or 4 comma-separated values (t1,t2,t3,t4).")
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
                 landmark_parquet_path=None,
                 tokenizer=None, max_text_tokens=None,
                 source_filter=None):
        df = pd.read_csv(tsv_path, sep=sep)
        required_cols = {'vid', 'file_path', 'text', 'duration_sec', 'source'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise RuntimeError(f"{tsv_path} missing columns: {missing_cols}")
        if source_filter is not None:
            df = df[df['source'] == source_filter].reset_index(drop=True)
            print(f"  ✓ source_filter='{source_filter}': {len(df)} rows retained")

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

        # Load landmark parquet and build O(1) lookup by vid.
        landmark_by_vid = {}
        if landmark_parquet_path is not None and Path(landmark_parquet_path).exists():
            lm_df = pd.read_parquet(landmark_parquet_path)
            lm_ok = lm_df[~lm_df['failed']]
            for r in lm_ok.itertuples(index=False):
                landmark_by_vid[str(r.vid)] = {
                    'frame_indices': np.load(io.BytesIO(r.frame_indices)),
                    'pose':          np.load(io.BytesIO(r.pose)),
                    'left_hand':     np.load(io.BytesIO(r.left_hand)),
                    'right_hand':    np.load(io.BytesIO(r.right_hand)),
                }
            print(f"  ✓ Loaded {len(landmark_by_vid)} landmark entries from "
                  f"{Path(landmark_parquet_path).name}")

        do_truncate = tokenizer is not None and max_text_tokens is not None and max_text_tokens > 0

        self.samples = []
        missing = 0
        no_bbox = 0
        truncated = 0

        # On Kaggle/Modal the dataset filesystem is slow for per-file stat calls.
        # Pre-scan all candidate directories once to build an O(1) existence set.
        existing_files: set[str] | None = None
        if CONFIG.get('is_kaggle') or CONFIG.get('is_modal'):
            _remap_fn = _remap_path_for_kaggle if CONFIG.get('is_kaggle') else _remap_path_for_modal
            _scan_roots: set[Path] = set()
            for row in df.itertuples(index=False):
                fp_remapped = _remap_fn(str(row.file_path))
                _scan_roots.add(Path(fp_remapped).parent)
            existing_files = set()
            for root in _scan_roots:
                if root.is_dir():
                    for p in root.iterdir():
                        existing_files.add(str(p))
            print(f"  ✓ Pre-scanned {len(existing_files)} files across {len(_scan_roots)} directories")

        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f'Loading {Path(tsv_path).stem}'):
            fp = str(row.file_path)
            if CONFIG.get('is_kaggle'):
                fp = _remap_path_for_kaggle(fp)
            elif CONFIG.get('is_modal'):
                fp = _remap_path_for_modal(fp)
            file_exists = (fp in existing_files) if existing_files is not None else Path(fp).exists()
            if not file_exists:
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
                'landmarks': landmark_by_vid.get(vid),
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
        self._dataset = dataset
        # Cache per-sample durations for O(1) rebuilds
        self._durations = [s['duration_sec'] for s in dataset.samples]
        # Initial build: use full dataset
        self._build_batches(list(range(len(dataset))), preserve_partition_order=False)
        self._skip_first_n = 0

    def _build_batches(self, indices, preserve_partition_order=False, partition_sizes=None):
        """Build batches from an index list, bucketing by duration.
        If preserve_partition_order=True and partition_sizes given, bucket within each
        partition separately and concatenate (curriculum epoch 1)."""
        if preserve_partition_order and partition_sizes:
            self.batches = []
            offset = 0
            for psize in partition_sizes:
                part = indices[offset:offset + psize]
                offset += psize
                part_sorted = sorted(part, key=lambda i: self._durations[i])
                self.batches.extend(
                    part_sorted[i:i + self.batch_size]
                    for i in range(0, len(part_sorted), self.batch_size)
                )
            self._partitioned = True
            self._partition_sizes = partition_sizes
        else:
            sorted_idx = sorted(indices, key=lambda i: self._durations[i])
            self.batches = [
                sorted_idx[i:i + self.batch_size]
                for i in range(0, len(sorted_idx), self.batch_size)
            ]
            self._partitioned = False
            self._partition_sizes = None

    def rebuild_with_indices(self, indices, partition_sizes=None):
        """Rebuild batches from a new epoch's sampled index list (weighted + optional curriculum)."""
        self._build_batches(indices,
                            preserve_partition_order=partition_sizes is not None,
                            partition_sizes=partition_sizes)

    def set_skip(self, n):
        """Skip the first *n* batches on the next iteration (for mid-epoch resume)."""
        self._skip_first_n = n

    def __iter__(self):
        if self.shuffle:
            if self._partitioned and self._partition_sizes:
                # Shuffle batches within each partition independently; preserve partition order
                shuffled = []
                cursor = 0
                # Partition boundaries in terms of #batches
                for psize in self._partition_sizes:
                    # Count batches that belong to this partition
                    nbatches = math.ceil(psize / self.batch_size)
                    chunk = self.batches[cursor:cursor + nbatches]
                    random.shuffle(chunk)
                    shuffled.extend(chunk)
                    cursor += nbatches
                shuffled.extend(self.batches[cursor:])
                iter_batches = shuffled
            else:
                iter_batches = list(self.batches)
                random.shuffle(iter_batches)
        else:
            iter_batches = self.batches
        for i, batch in enumerate(iter_batches):
            if i < self._skip_first_n:
                continue
            yield batch
        self._skip_first_n = 0

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

    # ─────────────────────────────────────────────────────────────
    #  CLAHE preprocessing (always-on, all splits)
    # ─────────────────────────────────────────────────────────────

    def _apply_clahe(self, all_videos):
        """Apply CLAHE to every frame of every clip.

        Operates on the L channel in LAB colour space so hue is preserved.
        Applied unconditionally on all splits (train, val, test) as a
        deterministic preprocessing step, not an augmentation.

        Uses Kornia for batched, vectorised processing (all T frames of a clip
        in one pass — no per-frame Python loop). Falls back to OpenCV if Kornia
        is not installed.

        Input/output: list of (T, C, H, W) uint8 torch.Tensor.
        """
        if not all_videos:
            return all_videos
        clip_limit = float(self.config.get('clahe_clip_limit', 2.0))
        tile = self.config.get('clahe_tile_grid', [8, 8])
        grid_size = (int(tile[0]), int(tile[1]))

        try:
            import kornia.color
            import kornia.enhance
        except ImportError:
            return self._apply_clahe_opencv(all_videos)

        result = []
        for vid in all_videos:
            if not isinstance(vid, torch.Tensor):
                vid = torch.as_tensor(np.asarray(vid))
            if vid.dtype != torch.uint8:
                vid = vid.clamp(0, 255).to(torch.uint8)

            # (T, 3, H, W) uint8 → float32 [0, 1] on CPU
            vid_f = vid.float() / 255.0

            # RGB → LAB: L in [0, 100], a/b in [-128, 127]
            lab = kornia.color.rgb_to_lab(vid_f)

            # Normalise L to [0, 1] for equalize_clahe
            l_norm = (lab[:, 0:1] / 100.0).clamp(0.0, 1.0)  # (T, 1, H, W)

            # Apply CLAHE over all T frames in one batched call — no per-frame loop
            l_clahe = kornia.enhance.equalize_clahe(
                l_norm, clip_limit=clip_limit, grid_size=grid_size,
                slow_and_differentiable=False,
            )

            lab[:, 0:1] = l_clahe * 100.0
            rgb_f = kornia.color.lab_to_rgb(lab).clamp(0.0, 1.0)
            result.append((rgb_f * 255.0).clamp(0, 255).to(torch.uint8))
            del vid_f, lab, l_norm, l_clahe, rgb_f

        return result

    def _apply_clahe_opencv(self, all_videos):
        """OpenCV fallback for _apply_clahe when Kornia is unavailable."""
        clip_limit = float(self.config.get('clahe_clip_limit', 2.0))
        tile = self.config.get('clahe_tile_grid', [8, 8])
        tile_grid = (int(tile[0]), int(tile[1]))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        result = []
        for vid in all_videos:
            if not isinstance(vid, torch.Tensor):
                vid = torch.as_tensor(np.asarray(vid))
            if vid.dtype != torch.uint8:
                vid = vid.clamp(0, 255).to(torch.uint8)
            T_frames = vid.shape[0]
            out = vid.clone()
            for t in range(T_frames):
                frame_rgb = vid[t].permute(1, 2, 0).contiguous().numpy()
                frame_lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
                l_ch, a_ch, b_ch = cv2.split(frame_lab)
                l_ch = clahe.apply(l_ch)
                frame_lab = cv2.merge([l_ch, a_ch, b_ch])
                frame_rgb = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2RGB)
                out[t] = torch.from_numpy(frame_rgb).permute(2, 0, 1)
            result.append(out)
        return result

    # ─────────────────────────────────────────────────────────────
    #  Landmark overlay (pre-computed offline)
    # ─────────────────────────────────────────────────────────────

    # Pose array stores indices [11,12,13,14,15,16] at positions [0,1,2,3,4,5].
    _POSE_CONNECTIONS = [
        (0, 1),  # left shoulder - right shoulder
        (0, 2),  # left shoulder - left elbow
        (2, 4),  # left elbow - left wrist
        (1, 3),  # right shoulder - right elbow
        (3, 5),  # right elbow - right wrist
    ]

    _HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),            # index
        (0, 9), (9, 10), (10, 11), (11, 12),       # middle
        (0, 13), (13, 14), (14, 15), (15, 16),     # ring
        (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
        (5, 9), (9, 13), (13, 17),                 # palm arch
    ]

    def _apply_landmark_overlay(self, all_videos, sample_landmarks):
        """Draw pre-computed pose + hand landmarks onto cropped frames.

        Landmarks are normalized (0.0-1.0) relative to the crop at extraction
        time. Multiplying by current crop dims gives pixel coords at any scale.

        Input/output: list of (T, C, H, W) uint8 torch.Tensor.
        """
        prob = float(self.config.get('landmark_overlay_prob', 1.0))
        result = []
        for vid, lm in zip(all_videos, sample_landmarks):
            if lm is None or (prob < 1.0 and random.random() > prob):
                result.append(vid)
                continue

            if not isinstance(vid, torch.Tensor):
                vid = torch.as_tensor(np.asarray(vid))
            if vid.dtype != torch.uint8:
                vid = vid.clamp(0, 255).to(torch.uint8)

            T_frames, _, cur_h, cur_w = vid.shape
            stored_fi = lm['frame_indices'].astype(np.float32)  # (N,) native frame indices
            pose_lms = lm['pose']        # (N, 6, 2)
            lh_lms   = lm['left_hand']   # (N, 21, 2)
            rh_lms   = lm['right_hand']  # (N, 21, 2)

            # Map decoded frame index → native frame space so the lookup is
            # correct regardless of what FPS the training decoder used.
            t_native_max = float(stored_fi[-1]) if len(stored_fi) else 0.0
            t_decoded_max = max(T_frames - 1, 1)

            out = vid.clone()
            for t in range(T_frames):
                t_native = t / t_decoded_max * t_native_max
                nearest = int(np.abs(stored_fi - t_native).argmin())

                frame_rgb = vid[t].permute(1, 2, 0).contiguous().numpy().copy()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                def to_px(nx, ny, _w=cur_w, _h=cur_h):
                    return (int(round(float(nx) * _w)), int(round(float(ny) * _h)))

                # Pose skeleton (yellow — BGR: 0,255,255)
                pose = pose_lms[nearest]  # (6, 2)
                for i, j in self._POSE_CONNECTIONS:
                    if np.any(np.isnan(pose[i])) or np.any(np.isnan(pose[j])):
                        continue
                    cv2.line(frame_bgr, to_px(*pose[i]), to_px(*pose[j]),
                             (0, 255, 255), 1, cv2.LINE_AA)
                for i in range(6):
                    if not np.any(np.isnan(pose[i])):
                        cv2.circle(frame_bgr, to_px(*pose[i]), 1,
                                   (0, 255, 255), -1, cv2.LINE_AA)

                # Left hand (green BGR: 0,200,0), right hand (blue BGR: 255,80,0)
                for hand_lms, color in (
                    (lh_lms, (0, 200, 0)),
                    (rh_lms, (255, 80, 0)),
                ):
                    hand = hand_lms[nearest]  # (21, 2)
                    if np.all(np.isnan(hand)):
                        continue
                    for i, j in self._HAND_CONNECTIONS:
                        if np.any(np.isnan(hand[i])) or np.any(np.isnan(hand[j])):
                            continue
                        cv2.line(frame_bgr, to_px(*hand[i]), to_px(*hand[j]),
                                 color, 1, cv2.LINE_AA)
                    for i in range(21):
                        if not np.any(np.isnan(hand[i])):
                            cv2.circle(frame_bgr, to_px(*hand[i]), 1,
                                       color, -1, cv2.LINE_AA)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                out[t] = torch.from_numpy(frame_rgb).permute(2, 0, 1)

            result.append(out)
        return result

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
        sample_bboxes = []      # parallel to all_videos: (x1,y1,x2,y2,fw,fh) or None
        sample_landmarks = []   # parallel to all_videos: landmark dict or None

        for sample in batch:
            messages = self._build_messages(sample, include_assistant=True)

            # Get text template (not tokenized)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )

            # Extract video frames using qwen_vl_utils.
            # High-resolution videos (e.g. 1080p) allocate hundreds of MB of RAM
            # for the raw frame buffer before any pixel-budget downsampling occurs.
            # Skip samples that OOM during loading rather than crashing the worker.
            try:
                images, videos, video_kwargs = process_vision_info(
                    messages, image_patch_size=16,
                    return_video_kwargs=True, return_video_metadata=True,
                )
            except (MemoryError, RuntimeError) as e:
                import warnings
                warnings.warn(
                    f"Skipping sample {sample.get('file_path', '?')} — OOM during "
                    f"video decode: {e}"
                )
                del messages
                continue
            del messages  # free the message dicts (contain video path strings & nested dicts)

            all_texts.append(text)

            if images:
                all_images.extend(images)

            if videos is not None:
                vids, metas = zip(*videos)
                all_videos.extend(list(vids))
                all_video_metadatas.extend(list(metas))
                sample_bboxes.extend([sample.get('bbox')] * len(vids))
                sample_landmarks.extend([sample.get('landmarks')] * len(vids))
            del images, videos  # free intermediate references

            # Merge video_kwargs (should be same for all samples)
            if video_kwargs:
                all_video_kwargs.update(video_kwargs)

        # ── Signer-centric crop (applied BEFORE augmentation) ──
        all_videos = self._apply_signer_crop(all_videos, sample_bboxes)

        # ── CLAHE (always-on, all splits) ──
        if self.config.get('use_clahe', False) and all_videos:
            all_videos = self._apply_clahe(all_videos)

        # ── Landmark overlay (always-on when enabled, all splits) ──
        if self.config.get('use_landmark_overlay', False) and all_videos:
            all_videos = self._apply_landmark_overlay(all_videos, sample_landmarks)

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

                # Temporal jitter: shift sequence by ±max_shift frames.
                # Padding mode:
                #   'edge_replicate' (default) — fill with repeated first/last frame (in-domain).
                #   'black'                    — fill with zeros (legacy; OOD for vision tower).
                if (self.config.get('aug_temporal_jitter', False)
                        and random.random() < self.config.get('aug_temporal_jitter_prob', 0.5)):
                    max_shift = self.config.get('aug_temporal_jitter_range', 2)
                    shift = random.randint(-max_shift, max_shift)
                    if shift != 0:
                        _jitter_mode = self.config.get('temporal_jitter_mode', 'edge_replicate')
                        if shift > 0:
                            # Shift right: prepend `shift` copies of first frame, drop last `shift` frames
                            if _jitter_mode == 'edge_replicate':
                                pad = vid[:1].expand(shift, -1, -1, -1).contiguous()
                            else:  # 'black'
                                pad = torch.zeros(shift, *vid.shape[1:], dtype=vid.dtype)
                            vid = torch.cat([pad, vid[:-shift]], dim=0)
                        else:
                            # Shift left: drop first `|shift|` frames, append `|shift|` copies of last frame
                            if _jitter_mode == 'edge_replicate':
                                pad = vid[-1:].expand(-shift, -1, -1, -1).contiguous()
                            else:  # 'black'
                                pad = torch.zeros(-shift, *vid.shape[1:], dtype=vid.dtype)
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
    """METEOR score using NLTK. Returns 0.0 if nltk is not available.

    NLTK's meteor_score requires the 'wordnet' and 'omw-1.4' corpora.
    These are pre-baked into the Modal image, but we also auto-download
    them on first use as a fallback (e.g. local runs).
    """
    try:
        from nltk.translate.meteor_score import meteor_score as _meteor
    except ImportError:
        return 0.0
    if not references or not hypotheses:
        return 0.0

    # Ensure wordnet corpus is available; download silently if missing.
    try:
        import nltk
        from nltk.corpus import wordnet as _wn
        _wn.ensure_loaded()
    except LookupError:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    scores = []
    for ref, hyp in zip(references, hypotheses):
        try:
            scores.append(_meteor([ref.strip().split()], hyp.strip().split()))
        except LookupError:
            # wordnet still unavailable — skip this sample
            scores.append(0.0)
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


def compute_distinct_n(hypotheses, n=2):
    """Distinct-N: unique N-grams / total N-grams across all hypotheses.
    Higher = more lexical diversity; counters mode-collapse."""
    if not hypotheses:
        return 0.0
    total_ngrams = 0
    unique_ngrams = set()
    for hyp in hypotheses:
        toks = hyp.strip().split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i:i + n])
            unique_ngrams.add(ng)
            total_ngrams += 1
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams


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
                      steps_done_in_epoch, best_val_loss, evals_without_improvement, elapsed_sec,
                      extra_state=None):
        """Save LoRA adapter + training state. `extra_state` merges into the state dict."""
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
            if extra_state:
                state.update(extra_state)
            torch.save(state, path / 'training_state.pt')

            self.periodic_checkpoints.append(path)

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
                  steps_done_in_epoch, best_val_loss, evals_without_improvement, elapsed_sec,
                  extra_state=None):
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
            if extra_state:
                state.update(extra_state)
            torch.save(state, path / 'training_state.pt')
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to save best model: {e}")


class CSVLogger:
    def __init__(self, log_file, fieldnames):
        self.log_file = Path(log_file)
        self.fieldnames = fieldnames
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row_dict):
        row_dict['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
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
            labels = batch_gpu.pop('labels')
            outputs = model(**batch_gpu, labels=None)  # logits only — avoids OOM from full [B,T,V] cross-entropy
            logits = outputs.logits
            del outputs

            # Chunked cross-entropy: never materialise full [B*T, V] at once
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            del logits, labels
            B, T, V = shift_logits.shape
            flat_logits = shift_logits.view(B * T, V)
            flat_labels = shift_labels.view(-1)
            del shift_logits, shift_labels
            chunk_size = 512
            loss_sum = 0.0
            token_count = 0
            for c in range(0, B * T, chunk_size):
                cl = flat_logits[c:c + chunk_size]
                ck = flat_labels[c:c + chunk_size]
                valid = ck != -100
                if valid.any():
                    loss_sum += torch.nn.functional.cross_entropy(
                        cl[valid], ck[valid], reduction='sum'
                    ).item()
                    token_count += valid.sum().item()
            del flat_logits, flat_labels
            batch_loss = loss_sum / token_count if token_count > 0 else 0.0

        total_loss += batch_loss
        num_batches += 1
        del batch_gpu, batch
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

            # Qwen3-VL uses per-frame timestamps, so _expand_inputs_for_generation
            # (called by beam search) splits video_grid_thw by video_nums which equals
            # the number of frames T. Pre-convert from [[T,H,W]] to T×[[1,H,W]] so
            # the split matches and beam search works correctly.
            if inputs.get('video_grid_thw') is not None and config['val_beam_size'] > 1:
                vgt = inputs['video_grid_thw']
                vgt = torch.repeat_interleave(vgt, vgt[:, 0], dim=0).clone()
                vgt[:, 0] = 1
                inputs['video_grid_thw'] = vgt

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config['val_max_new_tokens'],
                    num_beams=config['val_beam_size'],
                    length_penalty=config['val_length_penalty'],
                    no_repeat_ngram_size=config['val_no_repeat_ngram_size'],
                    repetition_penalty=config['val_repetition_penalty'],
                    do_sample=False,
                    # Suppress "invalid generation flags" warning: temperature / top_p /
                    # top_k are baked into the model's generation_config.json but are
                    # not used during beam search (do_sample=False). Passing None here
                    # explicitly overrides those baked-in values.
                    temperature=None,
                    top_p=None,
                    top_k=None,
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

def _make_tier_lambda(peak_lr, floor_lr, total_active_steps, warmup_ratio):
    """LambdaLR multiplier: linear warmup then cosine decay to floor_ratio.
    Measured in each tier's own active-step lifetime (§8)."""
    warmup_steps = max(int(warmup_ratio * total_active_steps), 1)
    cosine_steps = max(total_active_steps - warmup_steps, 1)
    floor_ratio = floor_lr / peak_lr

    def _fn(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(warmup_steps, 1))
        progress = (step - warmup_steps) / cosine_steps
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio + (1.0 - floor_ratio) * cos
    return _fn


def train(model, processor, train_loader, val_loader, val_dataset,
          opt_tier2, sched_tier2, opt_tier4, sched_tier4,
          opt_tier1, sched_tier1, opt_tier3, sched_tier3,
          phase_transition_step, phase2_steps,
          train_config, ckpt_manager,
          train_csv_logger, val_csv_logger, gen_samples_csv_logger, tb_writer,
          _val_collator=None, _train_collator=None,
          start_epoch=1, start_global_step=0, start_steps_done_in_epoch=None,
          best_val_loss=float('inf'), start_evals_without_improvement=0, start_elapsed_sec=0.0,
          controller=None,
          global_epoch_offset=0, dataset_phase_name='openasl',
          global_total_optimizer_steps=None, global_steps_offset=0,
          global_training_start_time=None, start_infonce_queues=None):
    """Full training loop for one dataset phase (OpenASL or How2Sign)."""

    # Unpack config
    _phase = dataset_phase_name  # short alias for per-phase key lookups
    num_epochs = train_config['num_epochs']   # per-phase epoch count (2 for each phase)
    curriculum_epochs = train_config.get('curriculum_epochs', [1])   # within-phase list
    aug_start_epoch   = train_config.get('aug_start_epoch', 2)       # within-phase start
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
    step_grad_norms_t1 = []
    step_grad_norms_t2 = []
    step_grad_norms_t3 = []
    step_grad_norms_t4 = []
    step_contrast_losses = []
    log_step_start = time.time()

    # ── InfoNCE MoCo-style queue (live anchor vs detached queue of past embeddings) ──
    infonce_enabled = bool(CONFIG.get('enable_infonce', True))
    if start_infonce_queues is not None:
        infonce_queue_v = start_infonce_queues.get('v', [])
        infonce_queue_t = start_infonce_queues.get('t', [])
        print(f"  ✅ InfoNCE queues restored (size: {len(infonce_queue_v)})")
    else:
        infonce_queue_v = []
        infonce_queue_t = []
    infonce_queue_max = int(CONFIG.get('infonce_queue_size', 64))
    infonce_tau = float(CONFIG.get('infonce_temperature', 0.10))
    infonce_skip_count = 0

    # ── Phase management ──
    _enable_two_phase_local = train_config.get('enable_two_phase', True)
    if not _enable_two_phase_local:
        # All tiers active from step 0 — no freeze/unfreeze ever needed
        current_phase = 2
    else:
        current_phase = 1 if start_global_step < phase_transition_step else 2

    # Create list of active (optimizer, scheduler) pairs per phase
    def _get_active_opt_sched():
        """Return list of (optimizer, scheduler) tuples for current phase."""
        if current_phase == 1:
            return [(opt_tier2, sched_tier2), (opt_tier4, sched_tier4)]
        else:  # Phase 2
            return [(opt_tier1, sched_tier1), (opt_tier2, sched_tier2),
                    (opt_tier3, sched_tier3), (opt_tier4, sched_tier4)]

    active_opt_sched = _get_active_opt_sched()
    _two_phase_tag = f"two-phase enabled (transition at step {phase_transition_step})" if _enable_two_phase_local else "two-phase DISABLED (all tiers active)"
    print(f"  Starting in model phase {current_phase}  |  {_two_phase_tag}")

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
    _vram_guard_bytes = max(int(_total_vram * 0.06), 512 * 1024 ** 2)  # 6% or 512 MB

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

    _W = 100
    _phase2_pct = int(phase_transition_step / total_optimizer_steps * 100)
    _phase2_steps = total_optimizer_steps - phase_transition_step
    _aug_on  = CONFIG.get('aug_color_jitter', False) or CONFIG.get('aug_random_grayscale', False) \
               or CONFIG.get('aug_gaussian_blur', False) or CONFIG.get('aug_solarize', False) \
               or CONFIG.get('aug_equalize', False) or CONFIG.get('aug_random_erasing', False) \
               or CONFIG.get('aug_affine', False) or CONFIG.get('aug_temporal_jitter', False) \
               or CONFIG.get('aug_speed_perturb', False)
    _enable_two_phase = train_config.get('enable_two_phase', True)
    _curriculum_on = bool(curriculum_epochs)
    _weighted_sampling = CONFIG.get('sampling_strategy', 'weighted') == 'weighted' and CONFIG.get('balance_by_source', True)
    _early_stop = train_config['early_stopping_patience']
    _label_smooth = train_config['label_smoothing']
    _grad_clip = train_config['max_grad_norm']
    _warmup_pct = int(CONFIG.get('warmup_ratio', 0.05) * 100)
    _easy_thr = float(CONFIG.get('easy_threshold_sec', 4.0))
    _phase_label = dataset_phase_name.upper()
    _global_epochs_str = f"{global_epoch_offset + 1}–{global_epoch_offset + num_epochs}"

    print("\n" + "━" * _W)
    _resume_tag = "  —  RESUMING" if start_global_step > 0 else ""
    print(f"  🚀  TRAINING STARTED  ·  DATASET PHASE: {_phase_label}{_resume_tag}")
    print("━" * _W)

    print(f"\n  ┌─ PHASE OVERVIEW {'─' * (_W - 20)}┐")
    print(f"  │  Dataset           : {dataset_phase_name}  (source filter applied)")
    print(f"  │  Global epochs     : {_global_epochs_str}  (within-phase: 1–{num_epochs})")
    print(f"  │  Total opt steps   : {total_optimizer_steps:,}   │   Steps/epoch : {optimizer_steps_per_epoch:,}")
    print(f"  │  Grad accum        : {grad_accum_steps} micro-batches/step   │   Warmup : {_warmup_pct}% of each tier's active life")
    print(f"  │  Log/Save/Eval     : every {log_every}/{save_every}/{eval_every} steps  (warmup eval: {eval_every_warmup} until step {eval_warmup_threshold})")
    print(f"  │  Early stopping    : patience = {_early_stop} evals   │   Label smoothing : {_label_smooth}   │   Grad clip : {_grad_clip}")
    print(f"  └{'─' * (_W - 2)}┘")

    print(f"\n  ┌─ FREEZE / UNFREEZE SCHEDULE {'─' * (_W - 32)}┐")
    if _enable_two_phase:
        _p1_pct = int(train_config.get('phase1_fraction', 0.20) * 100)
        print(f"  │  Two-phase freeze  : ENABLED  (phase1_fraction={train_config.get('phase1_fraction', 0.20)})")
        print(f"  │  Phase 1  (steps 1 → {phase_transition_step:,},  first {_p1_pct}%)")
        print(f"  │    Active : T2 (Vision LoRA)  +  T4 (InfoNCE Projections)")
        print(f"  │    Frozen : T1 (LM LoRA)  +  T3 (Head LoRA)")
        print(f"  │    Purpose: Align vision features before touching the LM")
        print(f"  │  Phase 2  (steps {phase_transition_step + 1:,} → {total_optimizer_steps:,},  {_phase2_steps:,} steps)")
        print(f"  │    Active : T1 + T2 + T3 + T4  (all tiers)")
        print(f"  │  Current  : Phase {current_phase}  (at step {start_global_step})")
    else:
        print(f"  │  Two-phase freeze  : DISABLED  →  All tiers (T1 T2 T3 T4) active from step 0")
    print(f"  └{'─' * (_W - 2)}┘")

    print(f"\n  ┌─ LEARNING RATES  ({dataset_phase_name}) {'─' * (_W - 25)}┐")
    _lr1 = train_config.get('lr_openasl_tier1', 0);  _fl1 = train_config.get('min_lr_openasl_tier1', 0)
    _lr2 = train_config.get('lr_openasl_tier2', 0);  _fl2 = train_config.get('min_lr_openasl_tier2', 0)
    _lr3 = train_config.get('lr_openasl_tier3', 0);  _fl3 = train_config.get('min_lr_openasl_tier3', 0)
    _lr4 = train_config.get('lr_openasl_tier4', 0);  _fl4 = train_config.get('min_lr_openasl_tier4', 0)
    _born1 = 'at phase transition' if _enable_two_phase else 'from step 0'
    _born3 = 'at phase transition' if _enable_two_phase else 'from step 0'
    print(f"  │  Schedule          : Linear warmup ({_warmup_pct}%) → Cosine decay (fresh for this phase)")
    print(f"  │  T1 (LM LoRA)      : peak={_lr1:.2e}  floor={_fl1:.2e}  [{_born1}]")
    print(f"  │  T2 (Vision LoRA)  : peak={_lr2:.2e}  floor={_fl2:.2e}  [from step 0]")
    print(f"  │  T3 (Head LoRA)    : peak={_lr3:.2e}  floor={_fl3:.2e}  [{_born3}]")
    print(f"  │  T4 (InfoNCE proj) : peak={_lr4:.2e}  floor={_fl4:.2e}  [from step 0]")
    print(f"  └{'─' * (_W - 2)}┘")

    print(f"\n  ┌─ CURRICULUM LEARNING {'─' * (_W - 24)}┐")
    print(f"  │  Status            : {'ENABLED' if _curriculum_on else 'DISABLED'}")
    if _curriculum_on:
        _cl_epochs_disp = [global_epoch_offset + e for e in curriculum_epochs]
        print(f"  │  Within-phase epochs : {curriculum_epochs}  →  global epochs {_cl_epochs_disp}")
        print(f"  │  Strategy          : Easy-first — clips ≤ {_easy_thr}s first, then longer clips")
        print(f"  │  Other epochs      : Standard weighted sampling")
    print(f"  │  Weighted sampling : {'ENABLED — inverse-frequency per source' if _weighted_sampling else 'DISABLED — uniform'}")
    print(f"  └{'─' * (_W - 2)}┘")

    print(f"\n  ┌─ DATA AUGMENTATION {'─' * (_W - 22)}┐")
    _aug_epochs_disp = [global_epoch_offset + e for e in range(aug_start_epoch, num_epochs + 1)]
    print(f"  │  Augmented epochs  : within-phase epoch ≥ {aug_start_epoch}  →  global epochs {_aug_epochs_disp}")
    if _aug_on:
        def _aug_flag(key, label, extra=''):
            on = CONFIG.get(key, False)
            return f"  │    {'✓' if on else '✗'}  {label:<26} {'ON' if on else 'OFF'}" + (f"  {extra}" if on and extra else '')
        print(_aug_flag('aug_color_jitter',     'Color jitter',        f"p={CONFIG.get('aug_color_jitter_prob')}"))
        print(_aug_flag('aug_random_grayscale', 'Random grayscale',    f"p={CONFIG.get('aug_random_grayscale_prob')}"))
        print(_aug_flag('aug_gaussian_blur',    'Gaussian blur',       f"p={CONFIG.get('aug_gaussian_blur_prob')}"))
        print(_aug_flag('aug_solarize',         'Solarize',            f"p={CONFIG.get('aug_solarize_prob')}"))
        print(_aug_flag('aug_equalize',         'Equalize',            ''))
        print(_aug_flag('aug_random_erasing',   'Random erasing',      f"p={CONFIG.get('aug_random_erasing_prob')}"))
        print(_aug_flag('aug_affine',           'Affine transform',    f"p={CONFIG.get('aug_affine_prob')}  deg=±{CONFIG.get('aug_affine_degrees')}"))
        print(_aug_flag('aug_temporal_jitter',  'Temporal jitter',     f"p={CONFIG.get('aug_temporal_jitter_prob')}  ±{CONFIG.get('aug_temporal_jitter_range')} frames"))
        print(_aug_flag('aug_speed_perturb',    'Speed perturbation',  f"p={CONFIG.get('aug_speed_perturb_prob')}  {CONFIG.get('aug_speed_perturb_min')}×–{CONFIG.get('aug_speed_perturb_max')}×"))
        print(f"  │    ✗  Horizontal flip         OFF  (handedness is semantically meaningful)")
    print(f"  └{'─' * (_W - 2)}┘")

    print(f"\n  ┌─ INFONCE CONTRASTIVE LOSS {'─' * (_W - 29)}┐")
    print(f"  │  Status            : {'ENABLED' if infonce_enabled else 'DISABLED'}")
    if infonce_enabled:
        print(f"  │  Temperature (τ)   : {infonce_tau}   │   Queue size : {infonce_queue_max}")
        print(f"  │  Lambda (Phase 1)  : {CONFIG.get('infonce_lambda_phase1', '?')}   (vision-only phase — primary alignment signal)")
        print(f"  │  Lambda (Phase 2)  : {CONFIG.get('infonce_lambda_phase2', '?')}   (all-tiers phase — auxiliary signal)")
        print(f"  │  Min pairs         : {CONFIG.get('infonce_min_pairs', '?')}   (skip InfoNCE if fewer valid pairs in batch)")
        print(f"  │  Proj dim          : {CONFIG.get('infonce_proj_dim', '?')}   (vision → proj_dim ← text)")
    print(f"  └{'─' * (_W - 2)}┘")

    if start_global_step > 0:
        print(f"\n  ▶  Resuming from global step {start_global_step}  (within-phase epoch {start_epoch}/{num_epochs})")
    print()

    # Handle mid-epoch resume
    if start_steps_done_in_epoch is not None:
        steps_done_in_epoch = start_steps_done_in_epoch
    else:
        steps_done_in_epoch = start_global_step % optimizer_steps_per_epoch
    if steps_done_in_epoch == 0 and start_global_step > 0:
        start_epoch += 1
        print(f"  ⏭️  Checkpoint was at exact end of Epoch {start_epoch - 1}. Advancing to Epoch {start_epoch}.")

    # ── Build per-sample source info for weighted sampling (once) ──
    base_seed = int(CONFIG.get('seed', 42))
    train_ds = getattr(train_loader, 'dataset', None)
    _sample_weights = None
    _easy_mask = None
    if train_ds is not None and CONFIG.get('sampling_strategy', 'weighted') == 'weighted' and CONFIG.get('balance_by_source', True):
        try:
            _src_counts = {}
            for s in train_ds.samples:
                _src = s.get('source', 'unknown')
                _src_counts[_src] = _src_counts.get(_src, 0) + 1
            _sample_weights = [1.0 / _src_counts.get(s.get('source', 'unknown'), 1) for s in train_ds.samples]
            _easy_thr = float(CONFIG.get('easy_threshold_sec', 4.0))
            _easy_mask = [(s['duration_sec'] <= _easy_thr) for s in train_ds.samples]
            print(f"  ✓ Weighted sampling: source counts={_src_counts}, "
                  f"easy (≤{_easy_thr}s) = {sum(_easy_mask)}/{len(_easy_mask)}")
        except Exception as _we:
            print(f"  ⚠️  Could not build weighted sampler ({_we}) — falling back to uniform")
            _sample_weights = None

    def _build_epoch_indices(epoch_num):
        """Weighted-sampled index list for this epoch. epoch_num is within-phase (1-indexed)."""
        if _sample_weights is None:
            return None, None
        gen = torch.Generator().manual_seed(base_seed + global_epoch_offset + epoch_num)
        N = len(_sample_weights)
        if epoch_num in curriculum_epochs:
            easy_idx = [i for i, e in enumerate(_easy_mask) if e]
            rest_idx = [i for i, e in enumerate(_easy_mask) if not e]
            easy_w = torch.tensor([_sample_weights[i] for i in easy_idx], dtype=torch.double)
            rest_w = torch.tensor([_sample_weights[i] for i in rest_idx], dtype=torch.double)
            gen2 = torch.Generator().manual_seed(base_seed + global_epoch_offset + epoch_num + 10_000)
            easy_draws = torch.multinomial(easy_w, len(easy_idx), replacement=True, generator=gen).tolist()
            rest_draws = torch.multinomial(rest_w, len(rest_idx), replacement=True, generator=gen2).tolist()
            picked_easy = [easy_idx[j] for j in easy_draws]
            picked_rest = [rest_idx[j] for j in rest_draws]
            return picked_easy + picked_rest, [len(picked_easy), len(picked_rest)]
        else:
            w = torch.tensor(_sample_weights, dtype=torch.double)
            draws = torch.multinomial(w, N, replacement=True, generator=gen).tolist()
            return draws, None

    _prev_epoch_curriculum = None   # track curriculum state across epochs for transition prints

    for epoch in range(start_epoch, num_epochs + 1):
        # ── Per-epoch: rebuild weighted-sampled indices ──
        if _sample_weights is not None and hasattr(train_loader, 'batch_sampler') \
                and hasattr(train_loader.batch_sampler, 'rebuild_with_indices'):
            try:
                _ep_idx, _ep_parts = _build_epoch_indices(epoch)
                if _ep_idx is not None:
                    train_loader.batch_sampler.rebuild_with_indices(_ep_idx, partition_sizes=_ep_parts)
                    _ep_hash = hash(tuple(_ep_idx[:1024]))
                    _g_ep = global_epoch_offset + epoch
                    if _ep_parts:
                        print(f"\n  ┌─ EPOCH {_g_ep} [{dataset_phase_name} {epoch}/{num_epochs}]  SAMPLER  (Curriculum Mode) {'─' * 30}┐")
                        print(f"  │  Strategy   : Easy-first curriculum  (within-phase epoch {epoch})")
                        print(f"  │  Easy draws : {_ep_parts[0]:,}  (clips ≤ {_easy_thr}s)")
                        print(f"  │  Hard draws : {_ep_parts[1]:,}  (clips > {_easy_thr}s)")
                        print(f"  │  Total      : {len(_ep_idx):,}  │  idx_hash : {_ep_hash}")
                        print(f"  └{'─' * 92}┘\n")
                    else:
                        print(f"\n  ┌─ EPOCH {_g_ep} [{dataset_phase_name} {epoch}/{num_epochs}]  SAMPLER  (Weighted Sampling) {'─' * 30}┐")
                        print(f"  │  Strategy   : Inverse-frequency weighted sampling (balanced per source)")
                        print(f"  │  Total      : {len(_ep_idx):,}  │  idx_hash : {_ep_hash}")
                        print(f"  └{'─' * 92}┘\n")
            except Exception as _se:
                print(f"  ⚠️  Per-epoch sampler rebuild failed: {_se}")

        # ── Curriculum transition print ──
        _cur_epoch_curriculum = epoch in curriculum_epochs
        if _prev_epoch_curriculum is not None and _cur_epoch_curriculum != _prev_epoch_curriculum:
            _g_ep_cl = global_epoch_offset + epoch
            if _cur_epoch_curriculum:
                print(f"\n  📚  CURRICULUM LEARNING ON  →  Global Epoch {_g_ep_cl}  [{dataset_phase_name} {epoch}/{num_epochs}]  —  easy-first ordering active")
            else:
                print(f"\n  📚  CURRICULUM LEARNING OFF →  Global Epoch {_g_ep_cl}  [{dataset_phase_name} {epoch}/{num_epochs}]  —  switching to uniform weighted sampling")
        _prev_epoch_curriculum = _cur_epoch_curriculum

        # Toggle data augmentation: epoch is within-phase, aug_start_epoch is per-phase config
        if _train_collator is not None:
            was_enabled = _train_collator.augmentation_enabled
            should_aug = epoch >= aug_start_epoch
            _train_collator.augmentation_enabled = should_aug
            _g_ep_aug = global_epoch_offset + epoch
            if not should_aug:
                print(f"  🧊 Augmentation: OFF  (global epoch {_g_ep_aug}, within-phase {epoch}/{num_epochs} — starts at within-phase epoch {aug_start_epoch})")
            elif should_aug and not was_enabled:
                print("")
                print("  " + "━" * 70)
                print(f"    🔥🎨  DATA AUGMENTATION ACTIVATED  —  Global Epoch {_g_ep_aug}  [{dataset_phase_name} {epoch}/{num_epochs}]  🎨🔥")
                print("  " + "━" * 70)
                print("")
            else:
                print(f"  🎨 Augmentation: ON  (global epoch {_g_ep_aug}, within-phase {epoch}/{num_epochs})")

        epoch_start = time.time()
        epoch_loss_sum = None
        epoch_microbatches = 0

        # Zero gradients for all active optimizers
        for opt, _ in active_opt_sched:
            opt.zero_grad(set_to_none=True)
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

        # _abs_micro_offset is 0 on the fallback path (loader yields all batches), so this is correct for both resume paths.
        _actual_steps_this_epoch = steps_per_epoch - _abs_micro_offset

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

                # ── InfoNCE: capture live (v, t) embeddings before labels/outputs freed ──
                contrast_loss_val = 0.0
                contrast_term = None
                if infonce_enabled and model._last_merger_out is not None:
                    try:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            _mo = model._last_merger_out
                            # Pool all leading dims to a single [D] vector
                            _v_pool = _mo.reshape(-1, _mo.shape[-1]).mean(dim=0)
                            _v_live = torch.nn.functional.normalize(model.vision_proj(_v_pool), dim=-1)

                            _lbl = batch_gpu['labels'][0]
                            _mask = (_lbl != -100)
                            if _mask.any():
                                _ids = _lbl[_mask].unsqueeze(0)
                                _temb = model.get_input_embeddings()(_ids).mean(dim=1).squeeze(0)
                                _t_live = torch.nn.functional.normalize(model.text_proj(_temb), dim=-1)

                                # Build contrast with queue negatives (detached, no grad)
                                if infonce_queue_v:
                                    _qv = torch.stack(infonce_queue_v, dim=0)  # [K, D]
                                    _qt = torch.stack(infonce_queue_t, dim=0)  # [K, D]
                                    # v_live vs [t_live, *queue_t]
                                    _cand_t = torch.cat([_t_live.unsqueeze(0), _qt], dim=0)
                                    _cand_v = torch.cat([_v_live.unsqueeze(0), _qv], dim=0)
                                    _logits_v = (_v_live.unsqueeze(0) @ _cand_t.T) / infonce_tau
                                    _logits_t = (_t_live.unsqueeze(0) @ _cand_v.T) / infonce_tau
                                    _tgt = torch.zeros(1, dtype=torch.long, device=_v_live.device)
                                    _lc = 0.5 * (torch.nn.functional.cross_entropy(_logits_v, _tgt)
                                                 + torch.nn.functional.cross_entropy(_logits_t, _tgt))
                                    _lam = CONFIG['infonce_lambda_phase1'] if current_phase == 1 else CONFIG['infonce_lambda_phase2']
                                    contrast_term = _lam * _lc
                                    contrast_loss_val = float(_lc.detach().item())

                                # Enqueue detached copies
                                infonce_queue_v.append(_v_live.detach())
                                infonce_queue_t.append(_t_live.detach())
                                while len(infonce_queue_v) > infonce_queue_max:
                                    infonce_queue_v.pop(0)
                                    infonce_queue_t.pop(0)
                            del _mo
                        model._last_merger_out = None
                    except Exception as _ie:
                        infonce_skip_count += 1
                        contrast_term = None
                        model._last_merger_out = None
                        # Log InfoNCE failures so device-mismatch / shape bugs are visible
                        if infonce_skip_count <= 3 or infonce_skip_count % 100 == 0:
                            print(f"  ⚠️  InfoNCE skip #{infonce_skip_count}: {type(_ie).__name__}: {_ie}")

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
                if contrast_term is not None:
                    loss = loss + (contrast_term / actual_accum_steps)
                    step_contrast_losses.append(contrast_loss_val)
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
                        for opt, _ in active_opt_sched:
                            opt.zero_grad(set_to_none=True)
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
                    for opt, _ in active_opt_sched:
                        opt.zero_grad(set_to_none=True)
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

                # ── Per-tier grad norms (pre-clip) ──
                def _tier_grad_norm(params):
                    gnorms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
                    if not gnorms:
                        return 0.0
                    return torch.norm(torch.stack(gnorms), 2).item()
                grad_norm_t1 = _tier_grad_norm(model._t1_params)
                grad_norm_t2 = _tier_grad_norm(model._t2_params)
                grad_norm_t3 = _tier_grad_norm(model._t3_params)
                grad_norm_t4 = _tier_grad_norm(model._t4_params)

                # ── Per-tier independent gradient clipping ────────────────────
                # Each tier is clipped against its own max_grad_norm so that a
                # high-norm tier (T2 Vision) cannot scale down well-behaved tiers.
                # clip_grad_norm_ returns the pre-clip norm of the passed params.
                _cn_t1 = train_config.get('max_grad_norm_t1', train_config['max_grad_norm'])
                _cn_t2 = train_config.get('max_grad_norm_t2', train_config['max_grad_norm'])
                _cn_t3 = train_config.get('max_grad_norm_t3', train_config['max_grad_norm'])
                _cn_t4 = train_config.get('max_grad_norm_t4', train_config['max_grad_norm'])
                _pre_t1 = torch.nn.utils.clip_grad_norm_(model._t1_params, max_norm=_cn_t1).item() if model._t1_params else 0.0
                _pre_t2 = torch.nn.utils.clip_grad_norm_(model._t2_params, max_norm=_cn_t2).item() if model._t2_params else 0.0
                _pre_t3 = torch.nn.utils.clip_grad_norm_(model._t3_params, max_norm=_cn_t3).item() if model._t3_params else 0.0
                _pre_t4 = torch.nn.utils.clip_grad_norm_(model._t4_params, max_norm=_cn_t4).item() if model._t4_params else 0.0

                # Joint global norm (of already-clipped gradients) for diagnostics/logging
                _clipped_gnorms = [
                    p.grad.detach().norm(2)
                    for p in _all_trainable_params if p.grad is not None
                ]
                grad_norm = torch.norm(torch.stack(_clipped_gnorms), 2).item() if _clipped_gnorms else 0.0
                del _clipped_gnorms

                # ── Grad-norm clip / spike logging ──
                # Report when any tier was clipped meaningfully (>10% over its limit).
                _clipped_tiers = []
                if _pre_t1 > _cn_t1 * 1.1: _clipped_tiers.append(f"t1={_pre_t1:.2f}→{_cn_t1}")
                if _pre_t2 > _cn_t2 * 1.1: _clipped_tiers.append(f"t2={_pre_t2:.2f}→{_cn_t2}")
                if _pre_t3 > _cn_t3 * 1.1: _clipped_tiers.append(f"t3={_pre_t3:.2f}→{_cn_t3}")
                if _pre_t4 > _cn_t4 * 1.1: _clipped_tiers.append(f"t4={_pre_t4:.2f}→{_cn_t4}")
                if _clipped_tiers:
                    _loud = any(v > 2.0 for v in [_pre_t1/_cn_t1, _pre_t2/_cn_t2, _pre_t3/_cn_t3, _pre_t4/_cn_t4])
                    if _loud:
                        print(f"  ✂️  grad clipped  step={global_step}  {' | '.join(_clipped_tiers)}  global_post={grad_norm:.2f}")
                    else:
                        print(f"  ✂️  step={global_step}  {' | '.join(_clipped_tiers)}")
                _max_norm = train_config['max_grad_norm']  # kept for spike threshold only
                if grad_norm > CONFIG.get('grad_norm_spike_threshold', 500.0):
                    try:
                        _bshape = tuple(batch_gpu['pixel_values_videos'].shape) if 'batch_gpu' in dir() and isinstance(batch_gpu, dict) and 'pixel_values_videos' in batch_gpu else None
                    except Exception:
                        _bshape = None
                    print(f"  ⚡ grad_norm spike: step={global_step+1} global={grad_norm:.2f} "
                          f"t1={grad_norm_t1:.2f} t2={grad_norm_t2:.2f} t3={grad_norm_t3:.2f} t4={grad_norm_t4:.2f} "
                          f"batch_shape={_bshape}")
                    if tb_writer is not None:
                        try:
                            _spikes = getattr(tb_writer, '_spike_count', 0) + 1
                            tb_writer._spike_count = _spikes
                            tb_writer.add_scalar('grad_norm/spike_count', _spikes, global_step + 1)
                        except Exception:
                            pass

                window_valid_microbatches = valid_microbatches_in_window

                # Step all active optimizers and schedulers
                for opt, sched in active_opt_sched:
                    opt.step()
                    sched.step()

                # Zero gradients for all active optimizers
                for opt, _ in active_opt_sched:
                    opt.zero_grad(set_to_none=True)

                valid_microbatches_in_window = 0
                global_step += 1

                # ── Phase transition check (only once, only when two-phase enabled) ──
                if _enable_two_phase_local and current_phase == 1 and global_step == phase_transition_step:
                    _pt_W = 100
                    _pt_pct = int(global_step / total_optimizer_steps * 100)
                    print(f"\n{'━' * _pt_W}")
                    print(f"  🔀  PHASE TRANSITION  —  Phase 1 → Phase 2")
                    print(f"{'━' * _pt_W}")
                    print(f"  Reached step {global_step:,} / {total_optimizer_steps:,}  ({_pt_pct}% of training complete)")
                    print(f"")
                    print(f"  What changes now:")
                    _pt_t1_peak  = train_config.get('lr_tier1',     train_config.get('lr_openasl_tier1'))
                    _pt_t1_floor = train_config.get('min_lr_tier1', train_config.get('min_lr_openasl_tier1'))
                    _pt_t3_peak  = train_config.get('lr_tier3',     train_config.get('lr_openasl_tier3'))
                    _pt_t3_floor = train_config.get('min_lr_tier3', train_config.get('min_lr_openasl_tier3'))
                    print(f"    ✓  Tier 1 — LM LoRA          : UNFROZEN  →  now training  (lr peak {_pt_t1_peak:.2e}, floor {_pt_t1_floor:.2e}  with fresh warmup)")
                    print(f"    ✓  Tier 3 — Embed / Head LoRA : UNFROZEN  →  now training  (lr peak {_pt_t3_peak:.2e}, floor {_pt_t3_floor:.2e}  with fresh warmup)")
                    print(f"    ─  Tier 2 — Vision LoRA       : continues uninterrupted  (lr {lr_t2:.2e})")
                    print(f"    ─  Tier 4 — Projections       : continues uninterrupted  (lr {lr_t4:.2e})")
                    print(f"")
                    print(f"  Remaining steps in Phase 2 : {total_optimizer_steps - global_step:,}")
                    print(f"{'━' * _pt_W}\n")

                    # Flip requires_grad for Tier 1 and Tier 3 (stashed on model at setup)
                    for p in model._t1_params:
                        p.requires_grad = True
                    for p in model._t3_params:
                        p.requires_grad = True

                    # Build Tier 1 and Tier 3 optimizers + schedulers (tier-local warmup)
                    _t1_peak  = train_config.get('lr_tier1',     train_config.get('lr_openasl_tier1'))
                    _t1_floor = train_config.get('min_lr_tier1', train_config.get('min_lr_openasl_tier1'))
                    _t3_peak  = train_config.get('lr_tier3',     train_config.get('lr_openasl_tier3'))
                    _t3_floor = train_config.get('min_lr_tier3', train_config.get('min_lr_openasl_tier3'))
                    _betas = CONFIG['adam_betas']
                    _wd = CONFIG.get('weight_decay', 0.0)
                    if CONFIG.get('use_8bit_adam', True) and _BNB_AVAILABLE:
                        opt_tier1 = bnb.optim.PagedAdamW8bit(model._t1_params, lr=_t1_peak, betas=_betas, weight_decay=_wd)
                        opt_tier3 = bnb.optim.PagedAdamW8bit(model._t3_params, lr=_t3_peak, betas=_betas, weight_decay=_wd)
                    else:
                        opt_tier1 = torch.optim.AdamW(model._t1_params, lr=_t1_peak, betas=_betas, weight_decay=_wd)
                        opt_tier3 = torch.optim.AdamW(model._t3_params, lr=_t3_peak, betas=_betas, weight_decay=_wd)
                    sched_tier1 = torch.optim.lr_scheduler.LambdaLR(
                        opt_tier1, lr_lambda=_make_tier_lambda(_t1_peak, _t1_floor, phase2_steps, CONFIG['warmup_ratio']))
                    sched_tier3 = torch.optim.lr_scheduler.LambdaLR(
                        opt_tier3, lr_lambda=_make_tier_lambda(_t3_peak, _t3_floor, phase2_steps, CONFIG['warmup_ratio']))

                    # Update phase and re-build active optimizer/scheduler list
                    current_phase = 2
                    active_opt_sched = [(opt_tier1, sched_tier1), (opt_tier2, sched_tier2),
                                        (opt_tier3, sched_tier3), (opt_tier4, sched_tier4)]

                    # Save phase transition checkpoint
                    _extra = {
                        'phase': current_phase,
                        'phase_transition_step': phase_transition_step,
                        'dataset_phase_name': dataset_phase_name,
                        'epoch_within_phase': epoch,
                        'global_epoch_offset': global_epoch_offset,
                        'opt_tier1_state_dict': opt_tier1.state_dict(),
                        'sched_tier1_state_dict': sched_tier1.state_dict(),
                        'opt_tier3_state_dict': opt_tier3.state_dict(),
                        'sched_tier3_state_dict': sched_tier3.state_dict(),
                        'opt_tier4_state_dict': opt_tier4.state_dict(),
                        'sched_tier4_state_dict': sched_tier4.state_dict(),
                        'vision_proj_state_dict': model.vision_proj.state_dict(),
                        'text_proj_state_dict':   model.text_proj.state_dict(),
                        'infonce_queue_v':        infonce_queue_v,
                        'infonce_queue_t':        infonce_queue_t,
                    }
                    ckpt_manager.save_periodic(
                        model, opt_tier2, sched_tier2, epoch, global_step,
                        0, best_val_loss, evals_without_improvement,
                        time.time() - training_start, extra_state=_extra,
                    )
                    print(f"  Phase transition checkpoint saved at step {global_step}")

                # ── Interactive controller check ──────────────────────────────────
                # Runs AFTER optimizer/scheduler step and gradient zero — the only
                # completely safe point to block the training loop for user input.
                if controller is not None:
                    # Pass the first (primary) optimizer for controller compatibility
                    controller.check(opt_tier2, [sched_tier2], train_config, global_step, total_optimizer_steps)
                window_avg_loss = (window_raw_loss_sum / window_valid_microbatches).item()
                step_losses.append(window_avg_loss)
                step_grad_norms.append(grad_norm)
                step_grad_norms_t1.append(grad_norm_t1)
                step_grad_norms_t2.append(grad_norm_t2)
                step_grad_norms_t3.append(grad_norm_t3)
                step_grad_norms_t4.append(grad_norm_t4)
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
                    avg_gn_t1 = sum(step_grad_norms_t1) / len(step_grad_norms_t1) if step_grad_norms_t1 else 0.0
                    avg_gn_t2 = sum(step_grad_norms_t2) / len(step_grad_norms_t2) if step_grad_norms_t2 else 0.0
                    avg_gn_t3 = sum(step_grad_norms_t3) / len(step_grad_norms_t3) if step_grad_norms_t3 else 0.0
                    avg_gn_t4 = sum(step_grad_norms_t4) / len(step_grad_norms_t4) if step_grad_norms_t4 else 0.0
                    avg_contrast = (sum(step_contrast_losses) / len(step_contrast_losses)) if step_contrast_losses else 0.0
                    lr_t1 = sched_tier1.get_last_lr()[0] if sched_tier1 is not None else 0.0
                    lr_t2 = sched_tier2.get_last_lr()[0]
                    lr_t3 = sched_tier3.get_last_lr()[0] if sched_tier3 is not None else 0.0
                    lr_t4 = sched_tier4.get_last_lr()[0]
                    current_lr = lr_t2  # primary display (always active)
                    elapsed = time.time() - training_start
                    log_elapsed = time.time() - log_step_start
                    speed = len(step_losses) / max(log_elapsed, 1e-6)
                    cuda_mem, cuda_peak = get_cuda_mem()

                    # ── Phase ETA: time remaining in this dataset phase only ──
                    # `global_step` carries over from the previous dataset phase.
                    # `global_steps_offset` is the total steps from all previous phases.
                    phase_step = global_step - global_steps_offset   # steps done so far in this phase
                    eta_seconds = (elapsed / phase_step) * (total_optimizer_steps - phase_step) if phase_step > 0 else 0

                    # ── Global ETA: time remaining across ALL dataset phases ──
                    # We use the average speed of the current phase to estimate remaining global time.
                    _g_total   = global_total_optimizer_steps if global_total_optimizer_steps is not None else total_optimizer_steps
                    _g_done    = global_step   # total steps completed across all phases
                    global_eta_seconds = (elapsed / phase_step) * (_g_total - _g_done) if phase_step > 0 else 0

                    # ── Decide ETA display: last phase → Phase ETA IS Global ETA ──
                    _is_last_dataset_phase = (
                        global_total_optimizer_steps is None
                        or (global_steps_offset + total_optimizer_steps) >= _g_total
                    )
                    if _is_last_dataset_phase:
                        _eta_suffix = (
                            f" │ ETA {format_time(eta_seconds)}"
                            f" [{dataset_phase_name}: {phase_step}/{total_optimizer_steps} steps done"
                            f"  ·  final phase, no further training]"
                        )
                    else:
                        _eta_suffix = (
                            f" │ Phase ETA {format_time(eta_seconds)}"
                            f" [{dataset_phase_name}: {phase_step}/{total_optimizer_steps} steps done]"
                            f" │ Global ETA {format_time(global_eta_seconds)}"
                            f" [all phases: {_g_done}/{_g_total} steps done]"
                        )

                    train_ppl = math.exp(min(avg_loss, MAX_PPL_CAP))

                    print('━' * 120)
                    _g_ep_log = global_epoch_offset + epoch
                    _total_global_epochs = global_epoch_offset + num_epochs
                    # ── Main status line ──
                    print(
                        f"  [{dataset_phase_name}] Phase: {current_phase}"
                        f" │ Global Epoch: {_g_ep_log}/{_total_global_epochs}   Phase Epoch: {epoch}/{num_epochs}"
                        f" │ Phase Step: {phase_step}/{total_optimizer_steps}   Global Step: {global_step}/{_g_total}"
                        f" │ Loss: {avg_loss:.4f} │ Contrast: {avg_contrast:.4f} │ PPL: {train_ppl:.3f}"
                    )
                    # ── Per-tier LR / GradNorm table ──
                    _tier_rows = [
                        ("T1 (LM LoRA)",     lr_t1, avg_gn_t1, sched_tier1 is not None),
                        ("T2 (Vision LoRA)", lr_t2, avg_gn_t2, True),
                        ("T3 (Head LoRA)",   lr_t3, avg_gn_t3, sched_tier3 is not None),
                        ("T4 (InfoNCE)",     lr_t4, avg_gn_t4, True),
                    ]
                    print('─'*58)
                    _th = f"  {'Tier':<18}  {'Learn Rate':>12}  {'Grad Norm':>12}  {'Status':<8}"
                    print(_th)
                    print(f"  {'─'*18}  {'─'*12}  {'─'*12}  {'─'*8}")
                    for _tn, _lr, _gn, _active in _tier_rows:
                        _lr_str = f"{_lr:.3e}"    if _active else "      —     "
                        _gn_str = f"{_gn:>12.4f}" if _active else "           —"
                        _st_str = "active" if _active else "frozen"
                        print(f"  {_tn:<18}  {_lr_str:>12}  {_gn_str}  {_st_str:<8}")
                    print(f"  {'─'*18}  {'─'*12}  {'─'*12}  {'─'*8}")
                    print(f"  {'Global (clipped)':<18}  {'':>12}  {avg_grad_norm:>12.4f}")
                    # ── Footer: throughput / hardware / time ──
                    print('─'*58)
                    print(
                        f"  Speed {speed:.3f} step/s │ VRAM {cuda_mem:.2f}/{cuda_peak:.2f} GB"
                        f" │ Elapsed {format_time(elapsed)}"
                        + _eta_suffix
                    )

                    train_csv_logger.log({
                        'global_step': global_step,
                        'epoch': epoch,
                        'phase': current_phase,
                        'train_loss': f"{avg_loss:.6f}",
                        'train_ppl': f"{train_ppl:.4f}",
                        'contrast_loss': f"{avg_contrast:.6f}",
                        'lr_t1': f"{lr_t1:.3e}",
                        'lr_t2': f"{lr_t2:.3e}",
                        'lr_t3': f"{lr_t3:.3e}",
                        'lr_t4': f"{lr_t4:.3e}",
                        'grad_norm_t1': f"{avg_gn_t1:.4f}",
                        'grad_norm_t2': f"{avg_gn_t2:.4f}",
                        'grad_norm_t3': f"{avg_gn_t3:.4f}",
                        'grad_norm_t4': f"{avg_gn_t4:.4f}",
                        'grad_norm': f"{avg_grad_norm:.4f}",
                        'cuda_mem_gb': f"{cuda_mem:.2f}",
                        'cuda_peak_gb': f"{cuda_peak:.2f}",
                        'steps_per_sec': f"{speed:.2f}",
                        'elapsed_sec': f"{elapsed:.1f}",
                        'eta_sec': f"{eta_seconds:.1f}",
                        'global_eta_sec': f"{global_eta_seconds:.1f}",
                    })

                    tb_writer.add_scalar('Loss/Train', avg_loss, global_step)
                    tb_writer.add_scalar('Perplexity/Train', train_ppl, global_step)
                    tb_writer.add_scalar('GradNorm/Train', avg_grad_norm, global_step)
                    tb_writer.add_scalar('loss/contrast', avg_contrast, global_step)
                    tb_writer.add_scalar('training/phase', current_phase, global_step)
                    tb_writer.add_scalar('grad_norm/tier1', avg_gn_t1, global_step)
                    tb_writer.add_scalar('grad_norm/tier2', avg_gn_t2, global_step)
                    tb_writer.add_scalar('grad_norm/tier3', avg_gn_t3, global_step)
                    tb_writer.add_scalar('grad_norm/tier4', avg_gn_t4, global_step)
                    tb_writer.add_scalar('lr/tier1_lm', lr_t1, global_step)
                    tb_writer.add_scalar('lr/tier2_vision', lr_t2, global_step)
                    tb_writer.add_scalar('lr/tier3_head', lr_t3, global_step)
                    tb_writer.add_scalar('lr/tier4_projections', lr_t4, global_step)

                    step_losses.clear()
                    step_grad_norms.clear()
                    step_grad_norms_t1.clear()
                    step_grad_norms_t2.clear()
                    step_grad_norms_t3.clear()
                    step_grad_norms_t4.clear()
                    step_contrast_losses.clear()
                    log_step_start = time.time()

                # ── Periodic Checkpoint ──
                if global_step % save_every == 0:
                    elapsed = time.time() - training_start
                    _ckpt_extra = {
                        'phase':                  current_phase,
                        'phase_transition_step':  phase_transition_step,
                        'dataset_phase_name':     dataset_phase_name,
                        'epoch_within_phase':     epoch,
                        'global_epoch_offset':    global_epoch_offset,
                        'opt_tier2_state_dict':   opt_tier2.state_dict(),
                        'sched_tier2_state_dict': sched_tier2.state_dict(),
                        'opt_tier4_state_dict':   opt_tier4.state_dict(),
                        'sched_tier4_state_dict': sched_tier4.state_dict(),
                        'vision_proj_state_dict': model.vision_proj.state_dict(),
                        'text_proj_state_dict':   model.text_proj.state_dict(),
                        'infonce_queue_v':        infonce_queue_v,
                        'infonce_queue_t':        infonce_queue_t,
                    }
                    if opt_tier1 is not None:
                        _ckpt_extra['opt_tier1_state_dict']  = opt_tier1.state_dict()
                        _ckpt_extra['sched_tier1_state_dict'] = sched_tier1.state_dict()
                    if opt_tier3 is not None:
                        _ckpt_extra['opt_tier3_state_dict']  = opt_tier3.state_dict()
                        _ckpt_extra['sched_tier3_state_dict'] = sched_tier3.state_dict()
                    print(f"  💾  Periodic checkpoint saved  step={global_step}  [{dataset_phase_name} model-phase={current_phase}]  →  {CONFIG['checkpoint_dir']}/checkpoint_step_{global_step}")
                    ckpt_manager.save_periodic(
                        model, opt_tier2, sched_tier2, epoch, global_step,
                        (_abs_micro_offset + micro_step + 1) // grad_accum_steps,
                        best_val_loss,
                        evals_without_improvement, elapsed,
                        extra_state=_ckpt_extra,
                    )
                    # Checkpoint serialization creates temporary CPU copies — reclaim them
                    gc.collect()
                    torch.cuda.empty_cache()

                # ── Validation ──
                if global_step < eval_warmup_threshold:
                    _should_eval = (global_step % eval_every_warmup == 0)
                else:
                    _should_eval = ((global_step - eval_warmup_threshold) % eval_every == 0)

                if _should_eval:
                    print(f"\n{'━' * 80}")
                    print(f"  📊  Validation  ·  Step {global_step}  ·  Epoch {epoch}")
                    print(f"{'━' * 80}")

                    val_results = validate(model, processor, val_loader, val_dataset, train_config, val_collator=_val_collator)
                    print(f"\n{'━' * 80}")
                    print(
                        f"  Val Loss {val_results['val_loss']:.4f} │ Val PPL {val_results['val_ppl']:.3f}"
                        f" │ BLEU-1 {val_results['bleu1']:.4f} │ BLEU-2 {val_results['bleu2']:.4f} │ BLEU-4 {val_results['bleu4']:.4f}"
                        f" │ ROUGE-L {val_results['rouge_l']:.2f}% │ WER {val_results['wer']:.2f}% │ METEOR {val_results['meteor']:.2f}%"
                        f"  ({val_results['num_eval_batches']} eval batches · {val_results['num_gen_samples']} generated samples)"
                    )

                    if val_results['sample_pairs']:
                        print(f"\n  Sample generations")
                        for i, (ref, hyp, src) in enumerate(val_results['sample_pairs'], 1):
                            print(f"    [{i}] ({src})")
                            print(f"        REF  \"{ref}\"")
                            print(f"        HYP  \"{hyp}\"")

                    # Per-source breakdown
                    ps = val_results['per_source']
                    _w = 60
                    print(f"\n  {'metric':<10}  {'combined':>10}  {'how2sign':>10}  {'openasl':>10}")
                    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
                    _overall = {k: val_results[k] for k in ('bleu1','bleu2','bleu4','rouge_l','wer','meteor')}
                    _rows = [
                        ('BLEU-1',   'bleu1',   '{:.2f}'),
                        ('BLEU-2',   'bleu2',   '{:.2f}'),
                        ('BLEU-4',   'bleu4',   '{:.2f}'),
                        ('ROUGE-L',  'rouge_l', '{:.2f}%'),
                        ('WER',      'wer',     '{:.2f}%'),
                        ('METEOR',   'meteor',  '{:.2f}%'),
                    ]
                    for _label, _key, _fmt in _rows:
                        _ov  = _fmt.format(_overall[_key])
                        _h2s = _fmt.format(ps['how2sign'][_key]) if ps['how2sign']['n'] > 0 else '   —'
                        _osl = _fmt.format(ps['openasl'][_key])  if ps['openasl']['n']  > 0 else '   —'
                        print(f"  {_label:<10}  {_ov:>10}  {_h2s:>10}  {_osl:>10}")
                    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
                    print(f"  {'samples':<10}  {val_results['num_gen_samples']:>10}  "
                          f"{ps['how2sign']['n']:>10}  {ps['openasl']['n']:>10}")

                    # Compute distinct_N on all generated hypotheses (mode-collapse detector)
                    _all_hyps = [hyp for (_r, hyp, _s) in val_results.get('all_pairs', [])]
                    _distinct_n = CONFIG.get('log_distinct_n', 2)
                    _distinct_val = compute_distinct_n(_all_hyps, n=_distinct_n)
                    print(f"  distinct-{_distinct_n}  {_distinct_val:.4f}  (diversity; low = mode collapse)")
                    tb_writer.add_scalar(f'val/distinct_{_distinct_n}', _distinct_val, global_step)

                    # Log to CSV
                    val_row = {
                        'global_step': global_step,
                        'epoch': epoch,
                        'phase': current_phase,
                        'distinct_2': f"{_distinct_val:.4f}",
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
                            tb_writer.add_scalar(f'BLEU/BLEU-1_{_src}', _m['bleu1'], global_step)
                            tb_writer.add_scalar(f'BLEU/BLEU-2_{_src}', _m['bleu2'], global_step)
                            tb_writer.add_scalar(f'BLEU/BLEU-4_{_src}', _m['bleu4'], global_step)
                            tb_writer.add_scalar(f'ROUGE-L_{_src}',     _m['rouge_l'], global_step)
                            tb_writer.add_scalar(f'WER_{_src}',         _m['wer'], global_step)
                            tb_writer.add_scalar(f'METEOR_{_src}',      _m['meteor'], global_step)

                    # Early stopping / best model
                    elapsed = time.time() - training_start
                    _prev_best = best_val_loss
                    _prev_ewi  = evals_without_improvement
                    if val_results['val_loss'] < best_val_loss:
                        best_val_loss = val_results['val_loss']
                        evals_without_improvement = 0
                        _best_extra = {
                            'phase':                  current_phase,
                            'phase_transition_step':  phase_transition_step,
                            'dataset_phase_name':     dataset_phase_name,
                            'epoch_within_phase':     epoch,
                            'global_epoch_offset':    global_epoch_offset,
                            'opt_tier4_state_dict':   opt_tier4.state_dict(),
                            'sched_tier4_state_dict': sched_tier4.state_dict(),
                            'vision_proj_state_dict': model.vision_proj.state_dict(),
                            'text_proj_state_dict':   model.text_proj.state_dict(),
                        }
                        if opt_tier1 is not None:
                            _best_extra['opt_tier1_state_dict']   = opt_tier1.state_dict()
                            _best_extra['sched_tier1_state_dict'] = sched_tier1.state_dict()
                        if opt_tier3 is not None:
                            _best_extra['opt_tier3_state_dict']   = opt_tier3.state_dict()
                            _best_extra['sched_tier3_state_dict'] = sched_tier3.state_dict()
                        ckpt_manager.save_best(
                            model, opt_tier2, sched_tier2, epoch, global_step,
                            (_abs_micro_offset + micro_step + 1) // grad_accum_steps,
                            best_val_loss,
                            evals_without_improvement, elapsed,
                            extra_state=_best_extra,
                        )
                        _delta = (_prev_best - best_val_loss) if _prev_best != float('inf') else float('nan')
                        _delta_str = f"  (Δ −{_delta:.4f} from {_prev_best:.4f})" if not math.isnan(_delta) else "  (first improvement)"
                        print(f"\n  ⭐  New best val_loss: {best_val_loss:.4f}{_delta_str}  →  saved best_model checkpoint")
                        print(f"      patience counter reset  ({_prev_ewi}  →  0 / {patience})")
                    else:
                        evals_without_improvement += 1
                        print(f"\n  ⚠️  No improvement  ({evals_without_improvement}/{patience} evals)  best={best_val_loss:.4f}  current={val_results['val_loss']:.4f}")

                    if evals_without_improvement >= patience:
                        print("\n" + "━" * 80)
                        print(f"  🛑  EARLY STOPPING  at step {global_step}, epoch {epoch}")
                        print(f"      Best val loss: {best_val_loss:.4f}")
                        print("━" * 80)
                        return global_step, best_val_loss

                    # Reclaim VRAM from validation before resuming training
                    del val_results
                    gc.collect()
                    torch.cuda.empty_cache()

                    print(f"{'━' * 80}\n")

        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = ((epoch_loss_sum / max(epoch_microbatches, 1)).item()
                          if epoch_loss_sum is not None else 0.0)
        epoch_loss_sum = None  # release the accumulated loss tensor for the next epoch
        _g_ep_end = global_epoch_offset + epoch
        _ep_lr_t1 = sched_tier1.get_last_lr()[0] if sched_tier1 is not None else 0.0
        _ep_lr_t2 = sched_tier2.get_last_lr()[0]
        _ep_lr_t3 = sched_tier3.get_last_lr()[0] if sched_tier3 is not None else 0.0
        _ep_lr_t4 = sched_tier4.get_last_lr()[0]
        _ep_active_tiers = ' '.join(
            t for t, active in [('T1', sched_tier1 is not None), ('T2', True), ('T3', sched_tier3 is not None), ('T4', True)]
            if active
        )
        _ep_aug_state = 'ON' if (_train_collator is not None and _train_collator.augmentation_enabled) else 'OFF'
        _ep_cl_state  = 'YES' if (epoch in curriculum_epochs) else 'NO'
        print(f"\n{'━' * 80}")
        print(f"  ✅  Global Epoch {_g_ep_end}  [{dataset_phase_name} {epoch}/{num_epochs}]  Complete")
        print(f"      Avg train loss : {epoch_avg_loss:.4f}  ·  Time : {format_time(epoch_time)}")
        print(f"      Model phase    : {current_phase}   Active tiers : {_ep_active_tiers}")
        print(f"      LR end-of-epoch: T1={_ep_lr_t1:.2e}  T2={_ep_lr_t2:.2e}  T3={_ep_lr_t3:.2e}  T4={_ep_lr_t4:.2e}")
        print(f"      Curriculum     : {_ep_cl_state}   Augmentation : {_ep_aug_state}   best_val_loss : {best_val_loss:.4f}")
        print(f"{'━' * 80}\n")

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
    _log_dir = Path(CONFIG['train_log_file']).parent / 'train_config'
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
    _SEP = "━" * _W
    effective_batch = CONFIG['batch_size'] * CONFIG['grad_accum_steps']
    print("\n" + _SEP)
    print("  🔧  QWEN3-VL TRAINING CONFIGURATION")
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
    print(f"    Epochs (OpenASL)       : {CONFIG['openasl_num_epochs']}")
    print(f"    Epochs (How2Sign)      : {CONFIG['how2sign_num_epochs']}")
    print(f"    Total epochs           : {CONFIG['openasl_num_epochs'] + CONFIG['how2sign_num_epochs']}")
    print(f"    Batch size             : {CONFIG['batch_size']}  (effective: {effective_batch} with {CONFIG['grad_accum_steps']}x accum)")
    print(f"    Mixed precision        : bfloat16")
    print(f"    Grad clip (per-tier)   : T1={CONFIG.get('max_grad_norm_t1', CONFIG['max_grad_norm'])}  T2={CONFIG.get('max_grad_norm_t2', CONFIG['max_grad_norm'])}  T3={CONFIG.get('max_grad_norm_t3', CONFIG['max_grad_norm'])}  T4={CONFIG.get('max_grad_norm_t4', CONFIG['max_grad_norm'])}  (global fallback={CONFIG['max_grad_norm']})")
    print(f"    Weight decay           : {CONFIG['weight_decay']}")
    print(f"    Adam betas             : {CONFIG['adam_betas']}")
    print(f"    8-bit AdamW            : {CONFIG['use_8bit_adam']}")
    print(f"    Gradient checkpointing : {CONFIG['use_gradient_checkpointing']}")
    print(f"    Liger Kernel           : {CONFIG.get('use_liger_kernel', False)}")
    print(f"    Torch compile          : {CONFIG['use_torch_compile']}  (mode={CONFIG['torch_compile_mode']})")
    print(f"    Bucket batching        : {CONFIG.get('use_bucket_batching', False)}")
    print(f"    Label smoothing        : {CONFIG['label_smoothing']}")
    print(f"    DataLoader workers     : {CONFIG['train_num_workers']} train / {CONFIG['val_num_workers']} val")

    print("\n  [ LEARNING RATES ]")
    print(f"    Warmup                 : {CONFIG['warmup_ratio']:.1%} of each tier's active life (per phase)")
    print(f"    Schedule               : Linear warmup → Cosine decay  |  Fresh restart at How2Sign phase")
    print(f"    OpenASL  T1 (LM)       : peak={CONFIG['lr_openasl_tier1']:.2e}  floor={CONFIG['min_lr_openasl_tier1']:.2e}")
    print(f"    OpenASL  T2 (Vision)   : peak={CONFIG['lr_openasl_tier2']:.2e}  floor={CONFIG['min_lr_openasl_tier2']:.2e}")
    print(f"    OpenASL  T3 (Head)     : peak={CONFIG['lr_openasl_tier3']:.2e}  floor={CONFIG['min_lr_openasl_tier3']:.2e}")
    print(f"    OpenASL  T4 (InfoNCE)  : peak={CONFIG['lr_openasl_tier4']:.2e}  floor={CONFIG['min_lr_openasl_tier4']:.2e}")
    print(f"    How2Sign T1 (LM)       : peak={CONFIG['lr_how2sign_tier1']:.2e}  floor={CONFIG['min_lr_how2sign_tier1']:.2e}")
    print(f"    How2Sign T2 (Vision)   : peak={CONFIG['lr_how2sign_tier2']:.2e}  floor={CONFIG['min_lr_how2sign_tier2']:.2e}")
    print(f"    How2Sign T3 (Head)     : peak={CONFIG['lr_how2sign_tier3']:.2e}  floor={CONFIG['min_lr_how2sign_tier3']:.2e}")
    print(f"    How2Sign T4 (InfoNCE)  : peak={CONFIG['lr_how2sign_tier4']:.2e}  floor={CONFIG['min_lr_how2sign_tier4']:.2e}")

    print("\n  [ EVALUATION & CHECKPOINTING ]")
    print(f"    Eval every             : {CONFIG['eval_every_steps']} steps  (warmup: {CONFIG['eval_every_steps_warmup']} until step {CONFIG['eval_warmup_threshold']})")
    print(f"    Max eval batches       : {CONFIG['max_eval_batches']}")
    print(f"    Max gen samples        : {CONFIG['max_generate_samples']}")
    print(f"    Beam size              : {CONFIG['val_beam_size']}  |  Length penalty: {CONFIG['val_length_penalty']}  |  No-repeat ngram: {CONFIG['val_no_repeat_ngram_size']}  |  Rep penalty: {CONFIG['val_repetition_penalty']}")
    print(f"    Save every             : {CONFIG['save_every_steps']} steps  |  Keep last: {CONFIG['keep_last_n_checkpoints']}")
    print(f"    Early stopping pat.    : {CONFIG['early_stopping_patience']} evals")
    print(f"    Checkpoint dir         : {CONFIG['checkpoint_dir'].resolve()}")

    print("\n  [ DATA AUGMENTATION (training only) ]")
    print(f"    OpenASL  aug starts    : within-phase epoch {CONFIG['openasl_aug_start_epoch']}  (global epoch {CONFIG['openasl_aug_start_epoch']})")
    print(f"    How2Sign aug starts    : within-phase epoch {CONFIG['how2sign_aug_start_epoch']}  (global epoch {CONFIG['openasl_num_epochs'] + CONFIG['how2sign_aug_start_epoch']})")
    print(f"    Horizontal flip        : DISABLED  (sign language handedness is semantically meaningful)")
    print(f"    Color jitter           : {'ON' if CONFIG['aug_color_jitter'] else 'OFF'}  (prob={CONFIG['aug_color_jitter_prob']}, b={CONFIG['aug_color_jitter_brightness']}, c={CONFIG['aug_color_jitter_contrast']}, s={CONFIG['aug_color_jitter_saturation']}, h={CONFIG['aug_color_jitter_hue']})")
    print(f"    Random grayscale       : {'ON' if CONFIG['aug_random_grayscale'] else 'OFF'}  (prob={CONFIG['aug_random_grayscale_prob']})")
    print(f"    Gaussian blur          : {'ON' if CONFIG['aug_gaussian_blur'] else 'OFF'}  (prob={CONFIG['aug_gaussian_blur_prob']}, kernel={CONFIG['aug_gaussian_blur_kernel']})")
    print(f"    Solarize               : {'ON' if CONFIG['aug_solarize'] else 'OFF'}  (prob={CONFIG['aug_solarize_prob']}, threshold={CONFIG['aug_solarize_threshold']})")
    print(f"    Equalize               : {'ON' if CONFIG['aug_equalize'] else 'OFF'}")
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
        print(f"    LR T1 (LM)             : {lr_ov.get('lr_openasl_tier1', '—')}")
        print(f"    LR T2 (Vision)         : {lr_ov.get('lr_openasl_tier2', '—')}")
        print(f"    LR T3 (Head)           : {lr_ov.get('lr_openasl_tier3', '—')}")
        print(f"    Floor T1               : {lr_ov.get('min_lr_openasl_tier1', '—')}")
        print(f"    Floor T2               : {lr_ov.get('min_lr_openasl_tier2', '—')}")
        print(f"    Floor T3               : {lr_ov.get('min_lr_openasl_tier3', '—')}")

    # Full CONFIG dump so nothing is missed
    print("\n  [ FULL CONFIG DUMP ]")
    for _k, _v in CONFIG.items():
        print(f"    {_k:30s}: {_v}")

    print("\n" + _SEP + "\n")

    # ── Tokenizer (loaded early so the Dataset can truncate reference text
    # to CONFIG['max_text_tokens']; the full processor is loaded with the model).
    print("📂 Loading tokenizer for reference-text truncation ...")
    from transformers import AutoTokenizer
    _trunc_tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'],
        local_files_only=CONFIG.get('is_kaggle', False),
    )

    # ── Load Data Manifests ──
    print("📂 Loading training data...")
    train_dataset = SignLanguageQwen3VLDataset(
        CONFIG['data_train_tsv'], CONFIG['tsv_sep'],
        bbox_csv_path=CONFIG.get('bbox_csv_train', CONFIG.get('bbox_csv')) if CONFIG.get('use_signer_crop') else None,
        landmark_parquet_path=CONFIG.get('landmark_parquet_train') if CONFIG.get('use_landmark_overlay') else None,
        tokenizer=_trunc_tokenizer,
        max_text_tokens=CONFIG['max_text_tokens'],
    )

    print("\n📂 Loading validation data...")
    val_dataset = SignLanguageQwen3VLDataset(
        CONFIG['data_val_tsv'], CONFIG['tsv_sep'],
        bbox_csv_path=CONFIG.get('bbox_csv_val', CONFIG.get('bbox_csv')) if CONFIG.get('use_signer_crop') else None,
        landmark_parquet_path=CONFIG.get('landmark_parquet_val') if CONFIG.get('use_landmark_overlay') else None,
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

    # QLoRA quantization — LM-only. llm_int8_skip_modules is an int8-era parameter that
    # silently misfires with 4-bit: skipped modules end up routed through
    # _move_missing_keys_from_meta_to_cpu → CPU quantization, allocating ~20GB of RAM for
    # the [N, 16] NF4 intermediate tensor. Fix: quantize everything during load (GPU path),
    # then cast skip-prefix modules back to bf16 immediately after.
    if CONFIG['use_qlora']:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        _skip = list(CONFIG.get('quantize_skip_modules', []) or [])
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=CONFIG['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=CONFIG['bnb_4bit_use_double_quant'],
            # bnb_4bit_quant_storage intentionally omitted (defaults to uint8).
            # Using bfloat16 storage packs 4 NF4 values per element → packed weight shape
            # (out*in//4, 1); dequantize() then returns this shape instead of (out, in),
            # causing _dequant_to_bf16 to create a misshapen nn.Linear that makes PEFT
            # build LoRA matrices of the wrong massive size (3B+ trainable for vision tier).
        )
        model_kwargs['device_map'] = 'auto'
        print(f"  ✓ QLoRA enabled (4-bit quantization); will restore bf16 post-load: {_skip}" if _skip
              else "  ✓ QLoRA enabled (4-bit quantization; all modules quantized)")
    else:
        model_kwargs['device_map'] = 'cuda'
        _skip = []

    model = AutoModelForImageTextToText.from_pretrained(CONFIG['model_name'], local_files_only=CONFIG.get('is_kaggle', False), **model_kwargs)

    # Restore vision tower to bf16 by replacing Linear4bit modules with real bf16 Linear.
    # .to(bfloat16) alone does NOT dequantize Linear4bit — the weights stay packed NF4.
    if _skip:
        try:
            from bitsandbytes.nn import Linear4bit as _Linear4bit_restore
        except Exception:
            _Linear4bit_restore = None

        try:
            import bitsandbytes.functional as _bnb_F
        except Exception:
            _bnb_F = None

        def _dequant_to_bf16(parent, child_name, child_mod):
            """Replace a Linear4bit with a plain bf16 Linear using dequantized weights."""
            if _Linear4bit_restore is None or not isinstance(child_mod, _Linear4bit_restore):
                return
            # Always use in_features/out_features from the Linear4bit attrs — never w.shape.
            # Params4bit.dequantize() is unreliable: it may return the raw packed storage
            # tensor (uint8 → out*in//2 elements, or bfloat16 → out*in//4 elements) instead
            # of the full dequantized weight.  Use bnb.functional.dequantize_4bit directly,
            # which respects quant_state.shape and always returns (out, in) shaped floats.
            in_f  = child_mod.in_features
            out_f = child_mod.out_features
            qs    = child_mod.weight.quant_state
            with torch.no_grad():
                if _bnb_F is not None and qs is not None:
                    w = _bnb_F.dequantize_4bit(
                        child_mod.weight.data, qs,
                        quant_type=child_mod.weight.quant_type,
                    ).to(torch.bfloat16)
                else:
                    # Fallback: Params4bit not yet quantized (CPU, no quant_state).
                    # The raw data is already the original float weight.
                    w = child_mod.weight.data.to(torch.bfloat16)
                if w.shape != (out_f, in_f):
                    w = w.reshape(out_f, in_f)
            bias = child_mod.bias
            new_linear = torch.nn.Linear(in_f, out_f, bias=bias is not None,
                                         dtype=torch.bfloat16, device=w.device)
            new_linear.weight = torch.nn.Parameter(w)
            if bias is not None:
                new_linear.bias = torch.nn.Parameter(bias.to(torch.bfloat16))
            setattr(parent, child_name, new_linear)

        def _is_under_skip(name):
            # Matches e.g. sp='visual' against 'model.visual', 'model.visual.blocks.0.attn', etc.
            # Uses the same component-boundary check as the audit below.
            return any(
                name == sp or name.startswith(sp + '.') or f'.{sp}.' in f'.{name}.'
                for sp in _skip
            )

        # Two-pass: collect (parent, attr_name, module) for all Linear4bit under skip prefixes,
        # then replace (avoids mutating named_modules() during iteration).
        _to_replace = []
        for _name, _mod in model.named_modules():
            if not _is_under_skip(_name):
                continue
            for _cname, _cmod in _mod.named_children():
                if _Linear4bit_restore is not None and isinstance(_cmod, _Linear4bit_restore):
                    _to_replace.append((_mod, _cname, _cmod))
        _n_replaced = len(_to_replace)
        for _parent, _cname, _cmod in _to_replace:
            _dequant_to_bf16(_parent, _cname, _cmod)
        del _to_replace
        # Cast any remaining non-Linear parameters (norms, embeddings, etc.) to bf16.
        for _name, _mod in model.named_modules():
            if _is_under_skip(_name):
                for _p in _mod.parameters(recurse=False):
                    _p.data = _p.data.to(torch.bfloat16)
        print(f"  ✓ Restored {_skip} to bf16 (dequantized {_n_replaced} Linear4bit modules)")
    del model_kwargs  # no longer needed after model is loaded

    # ── Post-load quantization audit ──
    # Verify: modules under any skip-prefix are NOT Linear4bit; modules under
    # the language_model prefix ARE Linear4bit (if QLoRA enabled).
    if CONFIG['use_qlora']:
        _skip_prefixes = list(CONFIG.get('quantize_skip_modules', []) or [])
        _bucket_bf16 = {'count': 0, 'params': 0}
        _bucket_4bit = {'count': 0, 'params': 0}
        _violations  = []
        try:
            from bitsandbytes.nn import Linear4bit as _Linear4bit
        except Exception:
            _Linear4bit = None
        for _name, _mod in model.named_modules():
            if _Linear4bit is not None and isinstance(_mod, _Linear4bit):
                _bucket_4bit['count'] += 1
                _bucket_4bit['params'] += sum(p.numel() for p in _mod.parameters())
                # Violation: a Linear4bit inside a skip-prefix module.
                if any(_name.startswith(sp) or f'.{sp}.' in f'.{_name}.' for sp in _skip_prefixes):
                    _violations.append(_name)
            elif isinstance(_mod, torch.nn.Linear):
                _bucket_bf16['count'] += 1
                _bucket_bf16['params'] += sum(p.numel() for p in _mod.parameters())
        print(f"  [quant audit] bf16 Linear: {_bucket_bf16['count']} modules, "
              f"{_bucket_bf16['params']/1e6:.1f}M params  |  "
              f"4bit Linear: {_bucket_4bit['count']} modules, "
              f"{_bucket_4bit['params']/1e6:.1f}M params")
        if _violations:
            print(f"  ⚠️  quant-skip violations ({len(_violations)}): "
                  f"{_violations[:3]}{'...' if len(_violations) > 3 else ''}")
        del _bucket_bf16, _bucket_4bit, _violations

    if CONFIG['use_qlora']:
        model = prepare_model_for_kbit_training(model)

    # ── Liger Kernel ──
    # Manual module replacement by exact class name, confirmed against the Qwen3-VL
    # architecture. apply_liger_kernel_to_qwen2_vl is NOT used because Qwen3-VL uses
    # distinct class names (Qwen3VLTextRMSNorm, Qwen3VLTextMLP, Qwen3VLForConditionalGeneration)
    # that don't match what the Qwen2-VL patcher targets.
    #
    # What gets patched:
    #   RMSNorm  → LigerRMSNorm  : 113 modules (28 decoder layers × 4 + 1 final norm)
    #   LayerNorm → LigerLayerNorm: 52 modules (24 vision blocks × 2 + 4 merger norms)
    #   SwiGLU, CrossEntropy: skipped — require native Qwen3-VL support in liger-kernel.
    #     When liger adds apply_liger_kernel_to_qwen3_vl, enable them here.
    if CONFIG.get('use_liger_kernel', False):
        try:
            from liger_kernel.transformers import LigerRMSNorm, LigerLayerNorm

            _do_rms = CONFIG.get('liger_rms_norm', True)
            _do_ln  = CONFIG.get('liger_layer_norm', True)

            _to_rms, _to_ln = [], []
            for _parent in model.modules():
                for _cname, _cmod in _parent.named_children():
                    _cls = type(_cmod).__name__
                    if _do_rms and _cls == 'Qwen3VLTextRMSNorm':
                        _to_rms.append((_parent, _cname, _cmod))
                    elif _do_ln and isinstance(_cmod, torch.nn.LayerNorm) and not isinstance(_cmod, LigerLayerNorm):
                        _to_ln.append((_parent, _cname, _cmod))

            for _parent, _name, _orig in _to_rms:
                _eps = getattr(_orig, 'variance_epsilon', getattr(_orig, 'eps', 1e-6))
                _new = LigerRMSNorm(_orig.weight.shape[0], eps=_eps)
                _new.weight = _orig.weight
                setattr(_parent, _name, _new)

            for _parent, _name, _orig in _to_ln:
                _new = LigerLayerNorm(_orig.normalized_shape, eps=_orig.eps)
                if _orig.elementwise_affine:
                    _new.weight = _orig.weight
                    _new.bias   = _orig.bias
                setattr(_parent, _name, _new)

            # Runtime audit — count actual Liger modules in the model after patching.
            # Expected: RMSNorm=113, LayerNorm=52. Any other count means a class name
            # changed in the transformers version you're running.
            _audit = {'LigerRMSNorm': 0, 'LigerLayerNorm': 0}
            for _, _m in model.named_modules():
                _t = type(_m).__name__
                if _t in _audit:
                    _audit[_t] += 1

            if _do_rms:
                _ok = '✓' if _audit['LigerRMSNorm']  == 113 else '⚠️ '
                print(f"  {_ok} Liger RMSNorm  : {_audit['LigerRMSNorm']:>4} patched  (expected 113: 28 layers × 4 + 1 final)")
            if _do_ln:
                _ok = '✓' if _audit['LigerLayerNorm'] == 52  else '⚠️ '
                print(f"  {_ok} Liger LayerNorm: {_audit['LigerLayerNorm']:>4} patched  (expected  52: 24 vision blocks × 2 + 4 merger norms)")
            del _to_rms, _to_ln, _audit, _do_rms, _do_ln, _ok

        except ImportError:
            print("  ⚠️  liger-kernel not installed — skipping (pip install liger-kernel)")
        except Exception as _liger_err:
            print(f"  ⚠️  Liger Kernel failed: {_liger_err} — skipping")

    processor = AutoProcessor.from_pretrained(CONFIG['model_name'], local_files_only=CONFIG.get('is_kaggle', False))
    processor.tokenizer.padding_side = 'left'

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
            _resume_source = 'best_model' if CONFIG['load_best_model'] else ('step ' + str(CONFIG['resume_checkpoint_step']) if CONFIG['resume_checkpoint_step'] is not None else 'latest')
            print(f"\n🔄 Resuming from checkpoint: {ckpt_path.name}  (source: {_resume_source})")
            print(f"   Full path: {ckpt_path.resolve()}")
            model = PeftModel.from_pretrained(model, ckpt_path / 'adapter', is_trainable=True)
            training_state = torch.load(ckpt_path / 'training_state.pt', map_location='cpu', weights_only=False)
            start_epoch = training_state['epoch']
            start_global_step = training_state['global_step']
            start_steps_done_in_epoch = training_state.get('steps_done_in_epoch')
            best_val_loss = training_state['best_val_loss']
            start_evals_without_improvement = training_state['evals_without_improvement']
            start_elapsed_sec = training_state.get('elapsed_sec', 0.0)
            _resume_phase_from_ckpt = training_state.get('phase', 1)
            _resume_ds_phase = training_state.get('dataset_phase_name', 'openasl')
            _resume_pt_step  = training_state.get('phase_transition_step', '?')
            print(f"  ✅ Resumed at global step {start_global_step}, within-phase epoch {start_epoch}")
            print(f"     Dataset phase   : {_resume_ds_phase}   Model phase (freeze) : {_resume_phase_from_ckpt}")
            print(f"     best_val_loss   : {best_val_loss:.4f}   evals_without_improvement : {start_evals_without_improvement}")
            print(f"     Elapsed so far  : {format_time(start_elapsed_sec)}")
            print(f"     Phase-transition step : {_resume_pt_step}")
        else:
            print("  No checkpoint found — starting fresh.")
            model = get_peft_model(model, lora_config)
    else:
        model = get_peft_model(model, lora_config)
    del lora_config  # consumed by get_peft_model / PeftModel

    # ── Explicit base-weight freeze ──
    # prepare_model_for_kbit_training sets requires_grad=True on every non-4bit parameter
    # (including all ~407 M bf16 vision-encoder weights) so gradient flow works through
    # frozen embeddings under gradient checkpointing.  PEFT's get_peft_model is documented
    # to freeze the base model, but in practice it preserves the requires_grad=True state
    # that prepare_model_for_kbit_training wrote — it does not call requires_grad_(False)
    # on modules that were explicitly enabled before it ran.
    #
    # Consequence without this block:
    #   • 407 M vision base weights land in the T2 optimizer param group
    #     (_matches_tier sees ".qkv.", ".proj.", etc. in the base-weight path).
    #   • Adam m + v states for those weights consume ~3.4 GB of VRAM on top of the
    #     model weights, immediately exhausting an 8 GB GPU and triggering the proactive
    #     VRAM skip on every single batch.
    #   • _trainable_params in the summary counts the base weights as trainable,
    #     producing the absurd "74 % trainable" figure.
    #
    # Fix: after LoRA wrapping, unconditionally freeze everything that is not a LoRA
    # adapter weight (lora_A / lora_B tensors).  Tier-4 projections are added below
    # and will naturally have requires_grad=True when they are created.
    _LORA_KEYWORDS = frozenset(['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B'])
    _n_frozen_explicit = 0
    for _pn, _pp in model.named_parameters():
        if not any(kw in _pn.split('.') for kw in _LORA_KEYWORDS):
            if _pp.requires_grad:
                _pp.requires_grad_(False)
                _n_frozen_explicit += 1
    print(f"  ✓ Explicit base-weight freeze: {_n_frozen_explicit} params → requires_grad=False")
    del _LORA_KEYWORDS, _n_frozen_explicit

    # ── Tier 4: InfoNCE projection modules ──
    # Add two trainable Linear projections for InfoNCE contrastive loss.
    # Vision merger outputs 2048 dims, language model hidden is 2048 dims.
    # Both project to CONFIG['infonce_proj_dim'] (256) for contrastive alignment.
    _vision_output_dim = 2048  # from model.visual.merger.linear_fc2 output
    _text_hidden_dim = 2048    # from model.language_model config
    _proj_dim = CONFIG['infonce_proj_dim']

    model.vision_proj = torch.nn.Linear(_vision_output_dim, _proj_dim, bias=True)
    model.text_proj = torch.nn.Linear(_text_hidden_dim, _proj_dim, bias=True)

    # Initialize with small Gaussian to break symmetry
    torch.nn.init.normal_(model.vision_proj.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(model.text_proj.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(model.vision_proj.bias)
    torch.nn.init.zeros_(model.text_proj.bias)

    # Move to GPU and cast to bf16 to match model dtype.
    # Without this, the projections stay on CPU and every InfoNCE forward call
    # raises a device-mismatch RuntimeError that is silently caught.
    model.vision_proj = model.vision_proj.to(device=device, dtype=torch.bfloat16)
    model.text_proj = model.text_proj.to(device=device, dtype=torch.bfloat16)

    print(f"  ✓ Tier 4 projections added: vision_proj ({_vision_output_dim}→{_proj_dim}), "
          f"text_proj ({_text_hidden_dim}→{_proj_dim})  [device={device}, dtype=bfloat16]")

    # ── Register forward hook on visual.merger to capture pooled vision embedding ──
    # Used by InfoNCE contrastive loss. Hook stores merger output on model._last_merger_out.
    if CONFIG.get('enable_infonce', True):
        def _merger_hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            model._last_merger_out = out
        try:
            # Try all possible paths in order
            _merger_mod = None
            for _path in [
                'base_model.model.model.visual.merger',  # QLoRA + PEFT
                'base_model.model.visual.merger',         # PEFT no QLoRA
                'model.model.visual.merger',              # no PEFT
                'model.visual.merger',                    # bare model
            ]:
                try:
                    _merger_mod = model.get_submodule(_path)
                    print(f"  ✓ Found merger at: {_path}")
                    break
                except AttributeError:
                    continue
            
            if _merger_mod is None:
                raise RuntimeError("merger not found at any expected path")
            
            _merger_mod.register_forward_hook(_merger_hook)
            model._last_merger_out = None
            print(f"  ✓ InfoNCE forward hook registered on visual.merger")
        except Exception as _e:
            print(f"  ⚠️  Could not register merger hook ({_e}) — InfoNCE disabled")
            CONFIG['enable_infonce'] = False
    del _vision_output_dim, _text_hidden_dim, _proj_dim

    # ── Parameter summary (per-tier breakdown) ──
    # Classify trainable parameters by tier by inspecting the leaf module name
    # that appears immediately before the lora_A/lora_B marker in each param path.
    # Tier 4 (InfoNCE projections) is counted separately.
    _t1_set      = set(CONFIG['lora_t1_modules'])
    _t2_set      = set(CONFIG['lora_t2_modules'])
    _t3_set      = set(CONFIG['lora_t3_modules'])
    _lora_marks  = ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B')
    _tier_params = {'Tier 1': 0, 'Tier 2': 0, 'Tier 3': 0, 'Tier 4': 0}
    _tier_layers = {'Tier 1': set(), 'Tier 2': set(), 'Tier 3': set(), 'Tier 4': set()}

    # Count Tier 4 (vision_proj + text_proj) separately
    _t4_projections = [('vision_proj', model.vision_proj), ('text_proj', model.text_proj)]
    for _t4_name, _t4_module in _t4_projections:
        for _param in _t4_module.parameters():
            if _param.requires_grad:
                _tier_params['Tier 4'] += _param.numel()
        _tier_layers['Tier 4'].add(_t4_name)

    # ── DIAGNOSTIC: dump every trainable param's name / shape / numel ──
    # Remove this block once the Tier-2 inflation is understood.
    # print("\n  [DIAG] All trainable parameters (requires_grad=True):")
    # _diag_total = 0
    # for _pname, _param in model.named_parameters():
    #     if not _param.requires_grad:
    #         continue
    #     _diag_total += _param.numel()
    #     print(f"    {_pname}  shape={tuple(_param.shape)}  numel={_param.numel():,}  dtype={_param.dtype}")
    # print(f"  [DIAG] Total trainable numel = {_diag_total:,}")
    # print()

    for _pname, _param in model.named_parameters():
        if not _param.requires_grad:
            continue
        # Skip Tier 4 projections (already counted)
        if 'vision_proj' in _pname or 'text_proj' in _pname:
            continue
        _parts = _pname.split('.')
        # Guard: only count LoRA adapter tensors, never base-model weights.
        # After the explicit freeze above all base weights are requires_grad=False,
        # but this check makes the summary robust to future regressions.
        _has_lora_mark = any(_p2 in _lora_marks for _p2 in _parts)
        if not _has_lora_mark:
            continue
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

    # Count logical parameters correctly.
    # bitsandbytes Linear4bit packs 2 NF4 values per uint8 byte, so p.numel()
    # returns half the actual logical parameter count for quantized weights.
    # LoRA adapters and non-quantized modules (e.g. vision encoder) are bf16 → numel() is exact.
    #
    # Build the module name→module dict ONCE here (O(n_modules)).
    # The previous implementation called dict(model.named_modules()) inside the closure,
    # rebuilding it for every single parameter — O(n_params × n_modules) ≈ 100 M ops.
    try:
        from bitsandbytes.nn import Linear4bit as _Linear4bit_count
        _mod_dict_for_numel = dict(model.named_modules())
    except Exception:
        _Linear4bit_count = None
        _mod_dict_for_numel = {}

    def _logical_numel(p, param_name):
        """Return the logical (unquantized) parameter count for a parameter tensor."""
        if _Linear4bit_count is None:
            return p.numel()
        # Find the parent module; if it is a Linear4bit the weight is NF4-packed at ×2.
        parts = param_name.rsplit('.', 1)
        if len(parts) == 2:
            mod = _mod_dict_for_numel.get(parts[0])
            if mod is not None and isinstance(mod, _Linear4bit_count):
                return p.numel() * 2
        return p.numel()

    _named_params = list(model.named_parameters())
    _total_params     = sum(_logical_numel(p, n) for n, p in _named_params)
    _trainable_params = sum(_logical_numel(p, n) for n, p in _named_params if p.requires_grad)
    _frozen_params    = _total_params - _trainable_params
    _lora_total       = sum(_tier_params.values())
    del _mod_dict_for_numel

    _tier_meta = {
        'Tier 1': (f"LM attention + MLP",       CONFIG['lora_t1_r']),
        'Tier 2': (f"Vision encoder",            CONFIG['lora_t2_r']),
        'Tier 3': (f"Embeddings + output head",  CONFIG['lora_t3_r']),
        'Tier 4': (f"InfoNCE projections",       '—'),
    }

    _W = 80
    print("\n" + "━" * _W)
    print("  📊  MODEL PARAMETER SUMMARY")
    print("━" * _W)
    print(f"  {'Tier':<10} {'Description':<28} {'Rank':>5} {'LoRA Layers':>12} {'Trainable Params':>18}")
    print("  " + "─" * (_W - 2))
    for _tl in ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']:
        _desc, _r = _tier_meta[_tl]
        _n = len(_tier_layers[_tl])
        _p = _tier_params[_tl]
        _rank_str = f"{_r:>5}" if isinstance(_r, int) else f"{_r:>5}"
        print(f"  {_tl:<10} {_desc:<28} {_rank_str} {_n:>12,} {_p:>18,}")
    print("  " + "─" * (_W - 2))
    print(f"  {'Total LoRA':<10} {'(all trainable adapters)':<28} {'':>5} {'':>12} {_lora_total:>18,}")
    print(f"  {'Frozen':<10} {'Base model weights':<28} {'':>5} {'':>12} {_frozen_params:>18,}")
    print(f"  {'TOTAL':<10} {'All parameters':<28} {'':>5} {'':>12} {_total_params:>18,}")
    print("  " + "─" * (_W - 2))
    print(f"  Trainable: {_trainable_params:,} / {_total_params:,}  ({_trainable_params / _total_params * 100:.4f}% of total model)")
    print("━" * _W)

    # ── Dataset-phase training plan summary ──
    _p1_active_params = _tier_params['Tier 2'] + _tier_params['Tier 4']
    _p1_frozen_lora   = _tier_params['Tier 1'] + _tier_params['Tier 3']
    _all_active_params = _trainable_params

    print(f"\n  📋  DATASET PHASE TRAINING PLAN")
    print("  " + "─" * (_W - 2))
    # OpenASL
    _osl_p1_pct = CONFIG['openasl_phase1_fraction'] * 100
    _osl_p2_pct = 100 - _osl_p1_pct
    print(f"  DATASET PHASE A  ──  OpenASL  ({CONFIG['openasl_num_epochs']} epochs  ·  global epochs 1–{CONFIG['openasl_num_epochs']})")
    print(f"    Two-phase freeze : {'ENABLED' if CONFIG['openasl_enable_two_phase'] else 'DISABLED'}", end='')
    if CONFIG['openasl_enable_two_phase']:
        print(f"  (phase1_fraction={CONFIG['openasl_phase1_fraction']}  →  first {_osl_p1_pct:.0f}% T2+T4 only, remaining {_osl_p2_pct:.0f}% all tiers)")
    else:
        print(f"  →  all tiers active from step 0")
    print(f"    Curriculum epochs: {CONFIG['openasl_curriculum_epochs']}  (easy ≤{CONFIG['easy_threshold_sec']}s first)")
    print(f"    Augmentation ON  : within-phase epoch ≥ {CONFIG['openasl_aug_start_epoch']}")
    print(f"    LR T1/T2/T3/T4   : {CONFIG['lr_openasl_tier1']:.1e}/{CONFIG['lr_openasl_tier2']:.1e}/{CONFIG['lr_openasl_tier3']:.1e}/{CONFIG['lr_openasl_tier4']:.1e}")
    _g2_start = CONFIG['openasl_num_epochs'] + 1
    _g2_end   = CONFIG['openasl_num_epochs'] + CONFIG['how2sign_num_epochs']
    print("  " + "─" * (_W - 2))
    print(f"  DATASET PHASE B  ──  How2Sign  ({CONFIG['how2sign_num_epochs']} epochs  ·  global epochs {_g2_start}–{_g2_end})")
    print(f"    Two-phase freeze : {'ENABLED' if CONFIG['how2sign_enable_two_phase'] else 'DISABLED'}", end='')
    if CONFIG['how2sign_enable_two_phase']:
        _h2_p1_pct = CONFIG['how2sign_phase1_fraction'] * 100
        print(f"  (phase1_fraction={CONFIG['how2sign_phase1_fraction']}  →  first {_h2_p1_pct:.0f}% T2+T4 only)")
    else:
        print(f"  →  all tiers active from step 0  (recommended for fine-tuning)")
    print(f"    Curriculum epochs: {CONFIG['how2sign_curriculum_epochs']}  (easy ≤{CONFIG['easy_threshold_sec']}s first)")
    print(f"    Augmentation ON  : within-phase epoch ≥ {CONFIG['how2sign_aug_start_epoch']}")
    print(f"    LR T1/T2/T3/T4   : {CONFIG['lr_how2sign_tier1']:.1e}/{CONFIG['lr_how2sign_tier2']:.1e}/{CONFIG['lr_how2sign_tier3']:.1e}/{CONFIG['lr_how2sign_tier4']:.1e}  (fresh restart)")
    print("  " + "─" * (_W - 2))
    print(f"  Tier params — T1: {_tier_params['Tier 1']:,}  T2: {_tier_params['Tier 2']:,}  T3: {_tier_params['Tier 3']:,}  T4: {_tier_params['Tier 4']:,}  |  All: {_all_active_params:,}")
    print("━" * _W)

    del _t1_set, _t2_set, _t3_set, _lora_marks, _tier_params, _tier_layers, _t4_projections
    del _lora_total, _frozen_params, _tier_meta, _W
    del _p1_active_params, _p1_frozen_lora, _all_active_params

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
        model = torch.compile(model, mode=CONFIG['torch_compile_mode'], dynamic=True)

    lora_vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  VRAM after LoRA: {lora_vram:.2f} GB")

    # ── Collators (shared across both dataset phases) ──
    print("\n⚙️  Creating collators and data loaders...")
    train_collator = Qwen3VLCollator(processor, CONFIG, is_training=True)
    val_collator   = Qwen3VLCollator(processor, CONFIG, is_training=False)

    def _make_loader(dataset, is_train):
        """Build a DataLoader for the given dataset (train or val)."""
        _nw = CONFIG['train_num_workers'] if is_train else CONFIG['val_num_workers']
        _collator = train_collator if is_train else val_collator
        if is_train and CONFIG.get('use_bucket_batching', False):
            _bs = BucketBatchSampler(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            _loader = DataLoader(
                dataset,
                batch_sampler=_bs,
                collate_fn=_collator,
                num_workers=_nw,
                prefetch_factor=CONFIG['train_prefetch_factor'] if _nw > 0 else None,
                pin_memory=CONFIG['train_pin_memory'],
                persistent_workers=CONFIG['train_persistent_workers'] and _nw > 0,
                worker_init_fn=_worker_init_fn,
            )
            print(f"  Bucket sampling: {len(_bs)} batches")
        elif is_train:
            _loader = DataLoader(
                dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                collate_fn=_collator,
                num_workers=_nw,
                prefetch_factor=CONFIG['train_prefetch_factor'] if _nw > 0 else None,
                pin_memory=CONFIG['train_pin_memory'],
                persistent_workers=CONFIG['train_persistent_workers'] and _nw > 0,
                worker_init_fn=_worker_init_fn,
            )
        else:
            _loader = DataLoader(
                dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=False,
                collate_fn=_collator,
                num_workers=_nw,
                prefetch_factor=CONFIG['val_prefetch_factor'] if _nw > 0 else None,
                pin_memory=CONFIG['val_pin_memory'],
                persistent_workers=CONFIG['val_persistent_workers'] and _nw > 0,
                worker_init_fn=_worker_init_fn,
            )
        return _loader

    # ── Build filtered datasets for each phase ──
    print("\n📂 Building per-phase filtered datasets...")
    _common_ds_kwargs = dict(
        sep=CONFIG['tsv_sep'],
        bbox_csv_path=CONFIG.get('bbox_csv_train', CONFIG.get('bbox_csv')) if CONFIG.get('use_signer_crop') else None,
        landmark_parquet_path=CONFIG.get('landmark_parquet_train') if CONFIG.get('use_landmark_overlay') else None,
        tokenizer=None,   # processor tokenizer loaded below
        max_text_tokens=CONFIG['max_text_tokens'],
    )
    _common_val_kwargs = dict(
        sep=CONFIG['tsv_sep'],
        bbox_csv_path=CONFIG.get('bbox_csv_val', CONFIG.get('bbox_csv')) if CONFIG.get('use_signer_crop') else None,
        landmark_parquet_path=CONFIG.get('landmark_parquet_val') if CONFIG.get('use_landmark_overlay') else None,
        tokenizer=None,
        max_text_tokens=CONFIG['max_text_tokens'],
    )
    print(f"  OpenASL train ({CONFIG['openasl_source_name']}):")
    openasl_train_ds = SignLanguageQwen3VLDataset(
        CONFIG['data_train_tsv'], source_filter=CONFIG['openasl_source_name'], **_common_ds_kwargs)
    print(f"  OpenASL val:")
    openasl_val_ds = SignLanguageQwen3VLDataset(
        CONFIG['data_val_tsv'], source_filter=CONFIG['openasl_source_name'], **_common_val_kwargs)
    print(f"  How2Sign train ({CONFIG['how2sign_source_name']}):")
    how2sign_train_ds = SignLanguageQwen3VLDataset(
        CONFIG['data_train_tsv'], source_filter=CONFIG['how2sign_source_name'], **_common_ds_kwargs)
    print(f"  How2Sign val:")
    how2sign_val_ds = SignLanguageQwen3VLDataset(
        CONFIG['data_val_tsv'], source_filter=CONFIG['how2sign_source_name'], **_common_val_kwargs)
    del _common_ds_kwargs, _common_val_kwargs

    # Build loaders
    openasl_train_loader  = _make_loader(openasl_train_ds,  is_train=True)
    openasl_val_loader    = _make_loader(openasl_val_ds,    is_train=False)
    how2sign_train_loader = _make_loader(how2sign_train_ds, is_train=True)
    how2sign_val_loader   = _make_loader(how2sign_val_ds,   is_train=False)

    # ── Compute per-phase step counts ──
    _osl_steps_per_epoch  = len(openasl_train_loader)
    _osl_opt_steps_epoch  = math.ceil(_osl_steps_per_epoch  / CONFIG['grad_accum_steps'])
    _osl_opt_steps_total  = _osl_opt_steps_epoch  * CONFIG['openasl_num_epochs']
    _h2s_steps_per_epoch  = len(how2sign_train_loader)
    _h2s_opt_steps_epoch  = math.ceil(_h2s_steps_per_epoch  / CONFIG['grad_accum_steps'])
    _h2s_opt_steps_total  = _h2s_opt_steps_epoch  * CONFIG['how2sign_num_epochs']

    print(f"\n  OpenASL  : {len(openasl_train_ds):,} train / {len(openasl_val_ds):,} val  |  {_osl_steps_per_epoch} batches/epoch  |  {_osl_opt_steps_epoch} opt-steps/epoch  |  {_osl_opt_steps_total} total")
    print(f"  How2Sign : {len(how2sign_train_ds):,} train / {len(how2sign_val_ds):,} val  |  {_h2s_steps_per_epoch} batches/epoch  |  {_h2s_opt_steps_epoch} opt-steps/epoch  |  {_h2s_opt_steps_total} total")
    print(f"  Grand total optimizer steps: {_osl_opt_steps_total + _h2s_opt_steps_total:,}")

    # ── Pre-training pipeline summary (printed after dataset sizes are known) ──
    _osl_n_ep   = CONFIG['openasl_num_epochs']
    _h2s_n_ep   = CONFIG['how2sign_num_epochs']
    _osl_tp_on  = CONFIG['openasl_enable_two_phase']
    _h2s_tp_on  = CONFIG['how2sign_enable_two_phase']
    _osl_pt_step_summary = int(CONFIG['openasl_phase1_fraction'] * _osl_opt_steps_total) if _osl_tp_on else None
    _h2s_pt_step_summary = int(CONFIG['how2sign_phase1_fraction'] * _h2s_opt_steps_total) if _h2s_tp_on else None
    _warmup_pct_summary  = int(CONFIG['warmup_ratio'] * 100)
    _PW = 78
    print()
    print("╔" + "═" * _PW + "╗")
    print("║" + "  TRAINING PIPELINE CONFIGURATION SUMMARY".center(_PW) + "║")
    print("╚" + "═" * _PW + "╝")
    print()
    print(f"  MODEL")
    print(f"    Name        : {CONFIG['model_name']}")
    print(f"    dtype       : {CONFIG['dtype']}  |  Attention: {CONFIG['attn_implementation']}")
    print(f"    QLoRA       : {CONFIG['use_qlora']} (nf4 quantized base, vision in bf16)")
    print()
    print(f"  DATASET PHASES")
    print(f"    Phase A  ──  OpenASL  (source='{CONFIG['openasl_source_name']}')")
    print(f"      Train clips           : {len(openasl_train_ds):,}   Val clips : {len(openasl_val_ds):,}")
    print(f"      Epochs                : {_osl_n_ep}  (global epochs 1–{_osl_n_ep})")
    print(f"      Optimizer steps/epoch : {_osl_opt_steps_epoch:,}   Total OpenASL steps : {_osl_opt_steps_total:,}")
    print()
    print(f"    Phase B  ──  How2Sign  (source='{CONFIG['how2sign_source_name']}')")
    print(f"      Train clips           : {len(how2sign_train_ds):,}   Val clips : {len(how2sign_val_ds):,}")
    print(f"      Epochs                : {_h2s_n_ep}  (global epochs {_osl_n_ep + 1}–{_osl_n_ep + _h2s_n_ep})")
    print(f"      Optimizer steps/epoch : {_h2s_opt_steps_epoch:,}   Total How2Sign steps : {_h2s_opt_steps_total:,}")
    print()
    print(f"    Grand total optimizer steps : {_osl_opt_steps_total + _h2s_opt_steps_total:,}")
    print()
    print(f"  TWO-PHASE FREEZE (per dataset phase)")
    if _osl_tp_on:
        _osl_p2_steps_summary = _osl_opt_steps_total - _osl_pt_step_summary
        print(f"    OpenASL  : ENABLED  |  Phase-1 fraction : {CONFIG['openasl_phase1_fraction'] * 100:.0f}%")
        print(f"               Phase-1 steps : {_osl_pt_step_summary:,}  →  Phase-2 starts at step {_osl_pt_step_summary + 1:,}")
        print(f"               Phase-1 active : T2 (Vision LoRA), T4 (InfoNCE)")
        print(f"               Phase-2 active : T1 + T2 + T3 + T4 (all)")
    else:
        print(f"    OpenASL  : DISABLED  →  All tiers active from step 0")
    if _h2s_tp_on:
        _h2s_p2_steps_summary = _h2s_opt_steps_total - _h2s_pt_step_summary
        print(f"    How2Sign : ENABLED  |  Phase-1 fraction : {CONFIG['how2sign_phase1_fraction'] * 100:.0f}%")
        print(f"               Phase-1 steps : {_h2s_pt_step_summary:,}  →  Phase-2 starts at step {_h2s_pt_step_summary + 1:,}")
    else:
        print(f"    How2Sign : DISABLED  →  All tiers active from step 0")
    print()
    print(f"  LEARNING RATES")
    _lr_hdr = f"  {'Tier':<10}  {'OpenASL Phase':<30}  {'How2Sign Phase':<30}"
    _lr_sep = "  " + "─" * 74
    print(_lr_hdr)
    print(_lr_sep)
    _lr_rows = [
        ("T1 (LM)",  CONFIG['lr_openasl_tier1'],          CONFIG['min_lr_openasl_tier1'],          CONFIG['lr_how2sign_tier1'],  CONFIG['min_lr_how2sign_tier1']),
        ("T2 (Vis)", CONFIG['lr_openasl_tier2'],          CONFIG['min_lr_openasl_tier2'],          CONFIG['lr_how2sign_tier2'],  CONFIG['min_lr_how2sign_tier2']),
        ("T3 (Hd)",  CONFIG['lr_openasl_tier3'],          CONFIG['min_lr_openasl_tier3'],          CONFIG['lr_how2sign_tier3'],  CONFIG['min_lr_how2sign_tier3']),
        ("T4 (NCE)", CONFIG['lr_openasl_tier4'],          CONFIG['min_lr_openasl_tier4'],          CONFIG['lr_how2sign_tier4'],  CONFIG['min_lr_how2sign_tier4']),
    ]
    for _tname, _pk_o, _fl_o, _pk_h, _fl_h in _lr_rows:
        _osl_str = f"peak={_pk_o:.1e}  floor={_fl_o:.1e}"
        _h2s_str = f"peak={_pk_h:.1e}  floor={_fl_h:.1e}"
        print(f"  {_tname:<10}  {_osl_str:<30}  {_h2s_str:<30}")
    print(f"  Schedule : Linear warmup ({_warmup_pct_summary}%) → Cosine decay  |  Fresh restart at How2Sign")
    print(f"  Warmup step counts (actual steps at warmup_ratio={CONFIG['warmup_ratio']}):")
    _osl_t2_warmup = int(CONFIG['warmup_ratio'] * _osl_opt_steps_total)
    _osl_t1_warmup = int(CONFIG['warmup_ratio'] * (_osl_opt_steps_total - (_osl_pt_step_summary or 0))) if _osl_tp_on else int(CONFIG['warmup_ratio'] * _osl_opt_steps_total)
    _h2s_t2_warmup = int(CONFIG['warmup_ratio'] * _h2s_opt_steps_total)
    print(f"    OpenASL  T2/T4 : {_osl_t2_warmup} steps   T1/T3 (born at phase-2) : {_osl_t1_warmup} steps" if _osl_tp_on else f"    OpenASL  all tiers : {_osl_t2_warmup} steps")
    print(f"    How2Sign all tiers : {_h2s_t2_warmup} steps")
    _infonce_on = CONFIG.get('enable_infonce', False)
    print()
    print(f"  INFONCE CONTRASTIVE LOSS")
    print(f"    Status   : {'ENABLED' if _infonce_on else 'DISABLED'}")
    if _infonce_on:
        print(f"    Lambda   : Phase-1 = {CONFIG.get('infonce_lambda_phase1')}  Phase-2 = {CONFIG.get('infonce_lambda_phase2')}")
        print(f"    Queue    : {CONFIG.get('infonce_queue_size', 64)}   τ = {CONFIG.get('infonce_temperature')}   min_pairs = {CONFIG.get('infonce_min_pairs')}   proj_dim = {CONFIG.get('infonce_proj_dim')}")
    print()
    print(f"  CURRICULUM LEARNING")
    _osl_cl = CONFIG['openasl_curriculum_epochs']
    _h2s_cl = CONFIG['how2sign_curriculum_epochs']
    _osl_cl_global = [e for e in _osl_cl]
    _h2s_cl_global = [_osl_n_ep + e for e in _h2s_cl]
    print(f"    OpenASL  curriculum epochs (within-phase) : {_osl_cl}  →  global epochs {_osl_cl_global}")
    print(f"    How2Sign curriculum epochs (within-phase) : {_h2s_cl}  →  global epochs {_h2s_cl_global}")
    print(f"    Easy threshold : ≤ {CONFIG['easy_threshold_sec']} s  |  Strategy: easy clips first, then rest")
    print()
    print(f"  DATA AUGMENTATION")
    _osl_aug = CONFIG['openasl_aug_start_epoch']
    _h2s_aug = CONFIG['how2sign_aug_start_epoch']
    print(f"    OpenASL  aug starts at within-phase epoch : {_osl_aug}  →  global epoch {_osl_aug}")
    print(f"    How2Sign aug starts at within-phase epoch : {_h2s_aug}  →  global epoch {_osl_n_ep + _h2s_aug}")
    print()
    print(f"  VALIDATION")
    print(f"    OpenASL epochs  →  validate on OpenASL val set  ({len(openasl_val_ds):,} clips)")
    print(f"    How2Sign epochs →  validate on How2Sign val set  ({len(how2sign_val_ds):,} clips)")
    print(f"    Eval every : {CONFIG['eval_every_steps']} optimizer steps  (warmup: {CONFIG['eval_every_steps_warmup']} until step {CONFIG['eval_warmup_threshold']})")
    print()
    print(f"  EPOCH-BY-EPOCH PLAN")
    _ep_header = f"    {'Epoch':<7}  {'Dataset':<12}  {'Two-Phase':<11}  {'Curriculum':<12}  {'Augmentation':<14}  Notes"
    print(_ep_header)
    print("    " + "─" * 74)
    _total_ep = _osl_n_ep + _h2s_n_ep
    for _gep in range(1, _total_ep + 1):
        if _gep <= _osl_n_ep:
            _ds   = 'OpenASL'
            _wep  = _gep
            _nep  = _osl_n_ep
            _tp   = 'YES' if (_osl_tp_on and _gep == 1) else 'NO '
            _cl   = 'YES' if (_wep in CONFIG['openasl_curriculum_epochs']) else 'NO '
            _aug  = 'YES' if (_wep >= _osl_aug) else 'NO '
        else:
            _ds   = 'How2Sign'
            _wep  = _gep - _osl_n_ep
            _nep  = _h2s_n_ep
            _tp   = 'YES' if (_h2s_tp_on and _wep == 1) else 'NO '
            _cl   = 'YES' if (_wep in CONFIG['how2sign_curriculum_epochs']) else 'NO '
            _aug  = 'YES' if (_wep >= _h2s_aug) else 'NO '
        _notes = ''
        if _gep <= _osl_n_ep and _osl_tp_on and _gep == 1:
            _notes = 'T2+T4 only → all at step ' + str(_osl_pt_step_summary)
        elif _gep > _osl_n_ep and _wep == 1:
            _notes = 'Fresh LR restart, all tiers'
        print(f"    Epoch {_gep:<2}  [{_ds:<8} {_wep}/{_nep}]  Two-Phase={_tp}  Curriculum={_cl}  Augmentation={_aug}  {_notes}")
    print()
    print(f"  CHECKPOINTING")
    print(f"    Save every : {CONFIG['save_every_steps']} steps   Resume from : {'step ' + str(CONFIG['resume_checkpoint_step']) if CONFIG['resume_training'] else 'scratch'}")
    print("═" * (_PW + 2))

    # ── Partition trainable params by LoRA tier ──
    print("\n⚙️  Partitioning parameters by LoRA tier...")

    def _matches_tier(name: str, module_list) -> bool:
        return any(f".{m}." in name or name.endswith(f".{m}") or f".{m}.lora_" in name
                   for m in module_list)

    t4_params = [p for p in [model.vision_proj, model.text_proj]
                 for p in p.parameters() if p.requires_grad]
    _t4_ids = {id(p) for p in t4_params}

    t1_params, t2_params, t3_params, unclassified = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in _t4_ids:
            continue
        if _matches_tier(n, CONFIG['lora_t3_modules']):
            t3_params.append(p)
        elif _matches_tier(n, CONFIG['lora_t2_modules']):
            t2_params.append(p)
        elif _matches_tier(n, CONFIG['lora_t1_modules']):
            t1_params.append(p)
        else:
            unclassified.append(n)

    if unclassified:
        raise RuntimeError(
            f"{len(unclassified)} trainable param(s) unclassified by LoRA tier matcher. "
            f"Examples: {unclassified[:5]}"
        )

    model._t1_params = t1_params
    model._t2_params = t2_params
    model._t3_params = t3_params
    model._t4_params = t4_params
    print(f"  T1: {len(t1_params)}  T2: {len(t2_params)}  T3: {len(t3_params)}  T4: {len(t4_params)}")

    def _create_optimizer(tier_params, peak_lr):
        if CONFIG['use_8bit_adam'] and _BNB_AVAILABLE:
            return bnb.optim.PagedAdamW8bit(
                tier_params, lr=peak_lr,
                betas=CONFIG['adam_betas'], weight_decay=CONFIG['weight_decay'])
        return torch.optim.AdamW(
            tier_params, lr=peak_lr,
            betas=CONFIG['adam_betas'], weight_decay=CONFIG['weight_decay'])

    # ── Determine resume state ──
    _resume_phase_name = 'openasl'  # default: start from OpenASL
    _resume_global_epoch_offset = 0
    if training_state is not None:
        _resume_phase_name = training_state.get('dataset_phase_name', 'openasl')
        _resume_global_epoch_offset = training_state.get('global_epoch_offset', 0)

    # ── Restore RNG states ──
    if training_state is not None:
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
            print("  ℹ️  RNG states not in checkpoint (older save)")

    # ── Checkpoint Manager & Loggers ──
    ckpt_manager = CheckpointManager(CONFIG['checkpoint_dir'], CONFIG['keep_last_n_checkpoints'])

    train_csv_logger = CSVLogger(CONFIG['train_log_file'], [
        'timestamp', 'global_step', 'epoch', 'phase',
        'train_loss', 'train_ppl', 'contrast_loss',
        'lr_t1', 'lr_t2', 'lr_t3', 'lr_t4',
        'grad_norm_t1', 'grad_norm_t2', 'grad_norm_t3', 'grad_norm_t4',
        'grad_norm', 'cuda_mem_gb', 'cuda_peak_gb',
        'steps_per_sec', 'elapsed_sec', 'eta_sec', 'global_eta_sec',
    ])
    val_csv_logger = CSVLogger(CONFIG['val_log_file'], [
        'timestamp', 'global_step', 'epoch', 'phase',
        'val_loss', 'val_ppl', 'distinct_2',
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

    tb_dir = CONFIG['tensorboard_dir']
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Verify label masking (first OpenASL batch) ──
    print("\n🔍 Label masking verification (first OpenASL training batch):")
    try:
        _first = openasl_train_ds[0]
        _fbatch = train_collator([_first])
        _input_ids = _fbatch['input_ids'][0]
        _labels = _fbatch['labels'][0]
        _label_tokens = _labels[_labels != -100]
        _decoded = processor.tokenizer.decode(_label_tokens, skip_special_tokens=True)
        print(f"  Ground truth: {_first['text']}")
        print(f"  Decoded labels: {_decoded}")
        print(f"  Total tokens: {len(_input_ids)} | Masked: {(_labels == -100).sum().item()} | Unmasked: {(_labels != -100).sum().item()}")
        del _fbatch, _first, _input_ids, _labels, _label_tokens, _decoded
    except Exception as e:
        print(f"  ⚠️  Warning: Label verification failed: {e}")
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # InteractiveController uses stdin — disable on Modal where there is no TTY.
    controller = InteractiveController() if not CONFIG.get('is_modal') else None

    # ════════════════════════════════════════════════════════════════
    #  DATASET PHASE A: OpenASL
    # ════════════════════════════════════════════════════════════════
    _osl_epoch_offset = 0   # global epoch offset: OpenASL epochs are 1–N
    _skip_openasl = (_resume_phase_name == 'how2sign')

    if not _skip_openasl:
        print("\n" + "═" * 80)
        print("  📦  DATASET PHASE A: OpenASL")
        print("═" * 80)

        # Shared wall-clock for global ETA — starts once before Phase A
        # (or before Phase B if resuming directly into How2Sign).
        # We record it here rather than inside train() so both phases share
        # the same reference point.
        _global_training_start_time = time.time()
        _global_total_opt_steps = _osl_opt_steps_total + _h2s_opt_steps_total

        _osl_enable_tp  = CONFIG['openasl_enable_two_phase']
        _osl_p1_frac    = CONFIG['openasl_phase1_fraction']
        if _osl_enable_tp:
            _osl_pt_step = int(_osl_p1_frac * _osl_opt_steps_total)
            _osl_p2_steps = _osl_opt_steps_total - _osl_pt_step
        else:
            _osl_pt_step  = _osl_opt_steps_total + 1   # sentinel: never fires
            _osl_p2_steps = _osl_opt_steps_total

        # Freeze T1/T3 if two-phase is enabled for OpenASL
        if _osl_enable_tp:
            for p in t1_params: p.requires_grad = False
            for p in t3_params: p.requires_grad = False
            print(f"  🧊 T1 ({len(t1_params)}) and T3 ({len(t3_params)}) frozen for OpenASL Phase 1")
        else:
            for p in t1_params: p.requires_grad = True
            for p in t3_params: p.requires_grad = True

        # Create OpenASL optimizers
        opt_tier2 = _create_optimizer(t2_params, CONFIG['lr_openasl_tier2'])
        opt_tier4 = _create_optimizer(t4_params, CONFIG['lr_openasl_tier4'])
        opt_tier1 = None
        opt_tier3 = None
        sched_tier2 = torch.optim.lr_scheduler.LambdaLR(
            opt_tier2, lr_lambda=_make_tier_lambda(
                CONFIG['lr_openasl_tier2'], CONFIG['min_lr_openasl_tier2'], _osl_opt_steps_total, CONFIG['warmup_ratio']))
        sched_tier4 = torch.optim.lr_scheduler.LambdaLR(
            opt_tier4, lr_lambda=_make_tier_lambda(
                CONFIG['lr_openasl_tier4'], CONFIG['min_lr_openasl_tier4'], _osl_opt_steps_total, CONFIG['warmup_ratio']))
        sched_tier1 = None
        sched_tier3 = None

        # Restore OpenASL optimizer/scheduler states from checkpoint
        if training_state is not None and _resume_phase_name == 'openasl':
            try:
                if 'opt_tier2_state_dict' in training_state:
                    opt_tier2.load_state_dict(training_state['opt_tier2_state_dict'])
                if 'opt_tier4_state_dict' in training_state:
                    opt_tier4.load_state_dict(training_state['opt_tier4_state_dict'])
                sched_tier2.load_state_dict(training_state['sched_tier2_state_dict'])
                sched_tier4.load_state_dict(training_state['sched_tier4_state_dict'])
                _ckpt_phase = training_state.get('phase', 1)
                if _ckpt_phase == 2 and 'opt_tier1_state_dict' in training_state:
                    opt_tier1 = _create_optimizer(t1_params, CONFIG['lr_openasl_tier1'])
                    opt_tier3 = _create_optimizer(t3_params, CONFIG['lr_openasl_tier3'])
                    opt_tier1.load_state_dict(training_state['opt_tier1_state_dict'])
                    opt_tier3.load_state_dict(training_state['opt_tier3_state_dict'])
                    sched_tier1 = torch.optim.lr_scheduler.LambdaLR(
                        opt_tier1, lr_lambda=_make_tier_lambda(
                            CONFIG['lr_openasl_tier1'], CONFIG['min_lr_openasl_tier1'], _osl_p2_steps, CONFIG['warmup_ratio']))
                    sched_tier3 = torch.optim.lr_scheduler.LambdaLR(
                        opt_tier3, lr_lambda=_make_tier_lambda(
                            CONFIG['lr_openasl_tier3'], CONFIG['min_lr_openasl_tier3'], _osl_p2_steps, CONFIG['warmup_ratio']))
                    sched_tier1.load_state_dict(training_state['sched_tier1_state_dict'])
                    sched_tier3.load_state_dict(training_state['sched_tier3_state_dict'])
                    for p in t1_params: p.requires_grad = True
                    for p in t3_params: p.requires_grad = True
                    print("  ✅ OpenASL optimizers restored (phase 2: T1 T2 T3 T4)")
                else:
                    print("  ✅ OpenASL optimizers restored (phase 1: T2 T4)")
                # Restore InfoNCE projection weights (Tier 4 modules)
                if 'vision_proj_state_dict' in training_state:
                    model.vision_proj.load_state_dict(training_state['vision_proj_state_dict'])
                    model.text_proj.load_state_dict(training_state['text_proj_state_dict'])
                    print("  ✅ InfoNCE projection weights restored from checkpoint")
                else:
                    print("  ℹ️  No projection weights in checkpoint (older save) — starting T4 from fresh init")
            except Exception as _re:
                print(f"  ⚠️  Could not restore OpenASL optimizer state: {_re}")

        _osl_train_config = {
            **CONFIG,
            'num_epochs':        CONFIG['openasl_num_epochs'],
            'enable_two_phase':  CONFIG['openasl_enable_two_phase'],
            'phase1_fraction':   CONFIG['openasl_phase1_fraction'],
            'curriculum_epochs': CONFIG['openasl_curriculum_epochs'],
            'aug_start_epoch':   CONFIG['openasl_aug_start_epoch'],
        }

        osl_final_step, best_val_loss = train(
            model=model,
            processor=processor,
            train_loader=openasl_train_loader,
            val_loader=openasl_val_loader,
            val_dataset=openasl_val_ds,
            opt_tier2=opt_tier2, sched_tier2=sched_tier2,
            opt_tier4=opt_tier4, sched_tier4=sched_tier4,
            opt_tier1=opt_tier1, sched_tier1=sched_tier1,
            opt_tier3=opt_tier3, sched_tier3=sched_tier3,
            phase_transition_step=_osl_pt_step,
            phase2_steps=_osl_p2_steps,
            train_config=_osl_train_config,
            ckpt_manager=ckpt_manager,
            train_csv_logger=train_csv_logger,
            val_csv_logger=val_csv_logger,
            gen_samples_csv_logger=gen_samples_csv_logger,
            tb_writer=tb_writer,
            _val_collator=val_collator,
            _train_collator=train_collator,
            start_epoch=start_epoch if _resume_phase_name == 'openasl' else 1,
            start_global_step=start_global_step if _resume_phase_name == 'openasl' else 0,
            start_steps_done_in_epoch=start_steps_done_in_epoch if _resume_phase_name == 'openasl' else None,
            best_val_loss=best_val_loss,
            start_evals_without_improvement=start_evals_without_improvement if _resume_phase_name == 'openasl' else 0,
            start_elapsed_sec=start_elapsed_sec if _resume_phase_name == 'openasl' else 0.0,
            controller=controller,
            global_epoch_offset=_osl_epoch_offset,
            dataset_phase_name='openasl',
            global_total_optimizer_steps=_global_total_opt_steps,
            global_steps_offset=0,          # Phase A starts at global step 0
            global_training_start_time=_global_training_start_time,
            start_infonce_queues={'v': training_state.get('infonce_queue_v', []), 't': training_state.get('infonce_queue_t', [])} if training_state else None,
        )

        # Ensure all tiers are unfrozen before How2Sign (safety guard)
        for p in t1_params: p.requires_grad = True
        for p in t3_params: p.requires_grad = True

        # Clean up OpenASL optimizer memory
        del opt_tier1, opt_tier2, opt_tier3, opt_tier4
        del sched_tier1, sched_tier2, sched_tier3, sched_tier4
        gc.collect()
        torch.cuda.empty_cache()

        # ── Dataset phase transition banner ──────────────────────────────────────
        _dpt_W = 100
        print(f"\n{'═' * _dpt_W}")
        print(f"  🔄  DATASET PHASE TRANSITION  —  Phase A (OpenASL) → Phase B (How2Sign)")
        print(f"{'═' * _dpt_W}")
        print(f"  OpenASL training complete at global step {osl_final_step:,}  (global epochs 1–{CONFIG['openasl_num_epochs']})")
        print(f"")
        print(f"  What changes now:")
        print(f"    ✓  Dataset       : OpenASL (noisy, large)  →  How2Sign (clean, smaller)")
        print(f"    ✓  All optimizers: RESET  (fresh AdamW states, no momentum carry-over)")
        _h2s_tp_preview = CONFIG['how2sign_enable_two_phase']
        if _h2s_tp_preview:
            _h2s_p1_pct = int(CONFIG['how2sign_phase1_fraction'] * 100)
            print(f"    ✓  Freeze schedule: two-phase enabled  (first {_h2s_p1_pct}% T2+T4 only, then all tiers)")
        else:
            print(f"    ✓  Freeze schedule: all tiers (T1 T2 T3 T4) active from step 0")
        print(f"    ✓  Learning rates : fresh cosine schedule  (lower How2Sign LRs)")
        print(f"       T1 peak {CONFIG['lr_how2sign_tier1']:.2e}  →  floor {CONFIG['min_lr_how2sign_tier1']:.2e}")
        print(f"       T2 peak {CONFIG['lr_how2sign_tier2']:.2e}  →  floor {CONFIG['min_lr_how2sign_tier2']:.2e}")
        print(f"       T3 peak {CONFIG['lr_how2sign_tier3']:.2e}  →  floor {CONFIG['min_lr_how2sign_tier3']:.2e}")
        print(f"       T4 peak {CONFIG['lr_how2sign_tier4']:.2e}  →  floor {CONFIG['min_lr_how2sign_tier4']:.2e}")
        _h2s_cl_preview = CONFIG['how2sign_curriculum_epochs']
        print(f"    ✓  Curriculum     : {'epochs ' + str(_h2s_cl_preview) + ' (within-phase)' if _h2s_cl_preview else 'DISABLED'}")
        print(f"    ✓  Augmentation   : starts at within-phase epoch {CONFIG['how2sign_aug_start_epoch']}")
        print(f"{'═' * _dpt_W}\n")
        # ─────────────────────────────────────────────────────────────────────────
    else:
        osl_final_step = start_global_step
        # Resuming directly into How2Sign — start the global clock now
        _global_training_start_time = time.time()
        _global_total_opt_steps = _osl_opt_steps_total + _h2s_opt_steps_total
        # Ensure all tiers unfrozen when skipping directly to How2Sign
        for p in t1_params: p.requires_grad = True
        for p in t3_params: p.requires_grad = True

    # ════════════════════════════════════════════════════════════════
    #  DATASET PHASE B: How2Sign
    # ════════════════════════════════════════════════════════════════
    _h2s_epoch_offset = CONFIG['openasl_num_epochs']

    print("\n" + "═" * 80)
    print("  📦  DATASET PHASE B: How2Sign  (fine-tuning on clean data)")
    print("═" * 80)

    _h2s_enable_tp = CONFIG['how2sign_enable_two_phase']
    _h2s_p1_frac   = CONFIG['how2sign_phase1_fraction']
    if _h2s_enable_tp:
        _h2s_pt_step  = int(_h2s_p1_frac * _h2s_opt_steps_total)
        _h2s_p2_steps = _h2s_opt_steps_total - _h2s_pt_step
        for p in t1_params: p.requires_grad = False
        for p in t3_params: p.requires_grad = False
        print(f"  🧊 T1/T3 frozen for How2Sign Phase 1")
    else:
        _h2s_pt_step  = _h2s_opt_steps_total + 1   # sentinel: never fires
        _h2s_p2_steps = _h2s_opt_steps_total

    # Create How2Sign optimizers (fresh, lower LRs) — all 4 tiers
    opt_tier1 = _create_optimizer(t1_params, CONFIG['lr_how2sign_tier1'])
    opt_tier2 = _create_optimizer(t2_params, CONFIG['lr_how2sign_tier2'])
    opt_tier3 = _create_optimizer(t3_params, CONFIG['lr_how2sign_tier3'])
    opt_tier4 = _create_optimizer(t4_params, CONFIG['lr_how2sign_tier4'])

    if _h2s_enable_tp:
        sched_tier1 = None
        sched_tier3 = None
    else:
        sched_tier1 = torch.optim.lr_scheduler.LambdaLR(
            opt_tier1, lr_lambda=_make_tier_lambda(
                CONFIG['lr_how2sign_tier1'], CONFIG['min_lr_how2sign_tier1'], _h2s_opt_steps_total, CONFIG['warmup_ratio']))
        sched_tier3 = torch.optim.lr_scheduler.LambdaLR(
            opt_tier3, lr_lambda=_make_tier_lambda(
                CONFIG['lr_how2sign_tier3'], CONFIG['min_lr_how2sign_tier3'], _h2s_opt_steps_total, CONFIG['warmup_ratio']))
    sched_tier2 = torch.optim.lr_scheduler.LambdaLR(
        opt_tier2, lr_lambda=_make_tier_lambda(
            CONFIG['lr_how2sign_tier2'], CONFIG['min_lr_how2sign_tier2'], _h2s_opt_steps_total, CONFIG['warmup_ratio']))
    sched_tier4 = torch.optim.lr_scheduler.LambdaLR(
        opt_tier4, lr_lambda=_make_tier_lambda(
            CONFIG['lr_how2sign_tier4'], CONFIG['min_lr_how2sign_tier4'], _h2s_opt_steps_total, CONFIG['warmup_ratio']))

    # Restore How2Sign optimizer states when resuming directly into this phase
    _h2s_start_epoch = 1
    _h2s_start_step  = osl_final_step
    _h2s_start_steps_in_ep = None
    _h2s_start_evals = 0
    _h2s_start_elapsed = 0.0

    if _skip_openasl and training_state is not None:
        try:
            if 'opt_tier2_state_dict' in training_state:
                opt_tier2.load_state_dict(training_state['opt_tier2_state_dict'])
            if 'opt_tier4_state_dict' in training_state:
                opt_tier4.load_state_dict(training_state['opt_tier4_state_dict'])
            sched_tier2.load_state_dict(training_state['sched_tier2_state_dict'])
            sched_tier4.load_state_dict(training_state['sched_tier4_state_dict'])
            if 'opt_tier1_state_dict' in training_state and sched_tier1 is not None:
                opt_tier1.load_state_dict(training_state['opt_tier1_state_dict'])
                opt_tier3.load_state_dict(training_state['opt_tier3_state_dict'])
                sched_tier1.load_state_dict(training_state['sched_tier1_state_dict'])
                sched_tier3.load_state_dict(training_state['sched_tier3_state_dict'])
            # Restore InfoNCE projection weights (Tier 4 modules)
            if 'vision_proj_state_dict' in training_state:
                model.vision_proj.load_state_dict(training_state['vision_proj_state_dict'])
                model.text_proj.load_state_dict(training_state['text_proj_state_dict'])
                print("  ✅ InfoNCE projection weights restored from checkpoint")
            else:
                print("  ℹ️  No projection weights in checkpoint (older save) — starting T4 from fresh init")
            _h2s_start_epoch    = start_epoch - CONFIG['openasl_num_epochs']
            _h2s_start_step     = start_global_step
            _h2s_start_steps_in_ep = start_steps_done_in_epoch
            _h2s_start_evals    = start_evals_without_improvement
            _h2s_start_elapsed  = start_elapsed_sec
            best_val_loss       = best_val_loss
            print("  ✅ How2Sign optimizers restored from checkpoint")
        except Exception as _re:
            print(f"  ⚠️  Could not restore How2Sign optimizer state: {_re}")

    if training_state is not None:
        del training_state
        gc.collect()

    _h2s_train_config = {
        **CONFIG,
        'num_epochs':        CONFIG['how2sign_num_epochs'],
        'enable_two_phase':  CONFIG['how2sign_enable_two_phase'],
        'phase1_fraction':   CONFIG['how2sign_phase1_fraction'],
        'curriculum_epochs': CONFIG['how2sign_curriculum_epochs'],
        'aug_start_epoch':   CONFIG['how2sign_aug_start_epoch'],
        'lr_tier1':          CONFIG['lr_how2sign_tier1'],
        'min_lr_tier1':      CONFIG['min_lr_how2sign_tier1'],
        'lr_tier2':          CONFIG['lr_how2sign_tier2'],
        'min_lr_tier2':      CONFIG['min_lr_how2sign_tier2'],
        'lr_tier3':          CONFIG['lr_how2sign_tier3'],
        'min_lr_tier3':      CONFIG['min_lr_how2sign_tier3'],
        'lr_tier4':          CONFIG['lr_how2sign_tier4'],
        'min_lr_tier4':      CONFIG['min_lr_how2sign_tier4'],
    }

    final_step, final_best_loss = train(
        model=model,
        processor=processor,
        train_loader=how2sign_train_loader,
        val_loader=how2sign_val_loader,
        val_dataset=how2sign_val_ds,
        opt_tier2=opt_tier2, sched_tier2=sched_tier2,
        opt_tier4=opt_tier4, sched_tier4=sched_tier4,
        opt_tier1=opt_tier1, sched_tier1=sched_tier1,
        opt_tier3=opt_tier3, sched_tier3=sched_tier3,
        phase_transition_step=_h2s_pt_step,
        phase2_steps=_h2s_p2_steps,
        train_config=_h2s_train_config,
        ckpt_manager=ckpt_manager,
        train_csv_logger=train_csv_logger,
        val_csv_logger=val_csv_logger,
        gen_samples_csv_logger=gen_samples_csv_logger,
        tb_writer=tb_writer,
        _val_collator=val_collator,
        _train_collator=train_collator,
        start_epoch=_h2s_start_epoch,
        start_global_step=_h2s_start_step,
        start_steps_done_in_epoch=_h2s_start_steps_in_ep,
        best_val_loss=best_val_loss,
        start_evals_without_improvement=_h2s_start_evals,
        start_elapsed_sec=_h2s_start_elapsed,
        controller=controller,
        global_epoch_offset=_h2s_epoch_offset,
        dataset_phase_name='how2sign',
        global_total_optimizer_steps=_global_total_opt_steps,
        global_steps_offset=_osl_opt_steps_total,   # Phase A steps already done
        global_training_start_time=_global_training_start_time,
        start_infonce_queues={'v': training_state.get('infonce_queue_v', []), 't': training_state.get('infonce_queue_t', [])} if training_state else None,
    )

    print("\n" + "━" * 80)
    print("  ✅  TRAINING COMPLETE")
    print("━" * 80)
    print(f"  Final step:      {final_step}")
    print(f"  Best val loss:   {final_best_loss:.4f}")
    print("━" * 80)

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


# ═══════════════════════════════════════════════════════════════════════════════
#  MODAL APP  (only active when is_modal=True)
#
#  Usage:
#    1. Set  'is_modal': True  in CONFIG above.
#    2. Run: modal run model_training_scripts/qwen3vl_training.py
#
#  The container image is built once and cached. Subsequent runs reuse the
#  cache unless pip packages or the Dockerfile layer changes.
#
#  Volume upload commands (run once from your local machine):
#    modal volume create signbridge-data
#    modal volume create signbridge-videos
#    modal volume create signbridge-outputs
#
#    # Metadata files
#    modal volume put signbridge-data "data/7_dataset_train.tsv"  /7_dataset_train.tsv
#    modal volume put signbridge-data "data/7_dataset_val.tsv"    /7_dataset_val.tsv
#    modal volume put signbridge-data "data/7_dataset_test.tsv"   /7_dataset_test.tsv
#    modal volume put signbridge-data "data/bboxes_all.csv"       /bboxes_all.csv
#    modal volume put signbridge-data "data/landmarks_train.parquet" /landmarks_train.parquet
#    modal volume put signbridge-data "data/landmarks_val.parquet"   /landmarks_val.parquet
#    modal volume put signbridge-data "data/landmarks_test.parquet"  /landmarks_test.parquet
#
#    # Raw video clips (large — mirrors D:\signbridge_dataset\)
#    modal volume put signbridge-videos "D:/signbridge_dataset/how2sign" /how2sign
#    modal volume put signbridge-videos "D:/signbridge_dataset/openasl"  /openasl
#
#  HuggingFace secret (for downloading Qwen3-VL-2B-Instruct):
#    modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
# ═══════════════════════════════════════════════════════════════════════════════

if CONFIG.get('is_modal'):
    import modal

    _HOURS = 3600

    # ── Container image ────────────────────────────────────────────────────────
    # Base: NVIDIA CUDA 12.4 devel image (provides nvcc for flash-attn compilation)
    _modal_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            add_python="3.11",
        )
        .entrypoint([])  # silence verbose CUDA entrypoint logging
        .apt_install(
            "git",
            "ffmpeg",          # video decoding (cv2 / torchvision backend)
            "libgl1",          # OpenCV headless dependency
            "libglib2.0-0",
            "clang",           # required by flash-attn build system (clang++)
        )
        .pip_install(
            # Core ML
            "torch==2.10.0",
            "torchvision==0.25.0",   # must match torch major.minor (2.10 → torchvision 0.25)
            "torchaudio==2.10.0",
            # Transformers stack
            "transformers==4.57.3",
            "peft==0.18.1",
            "accelerate==1.12.0",
            "bitsandbytes==0.48.2",
            # Qwen3-VL utilities
            "qwen-vl-utils==0.0.14",
            # Training utilities
            "sacrebleu",
            "tensorboard",
            "liger-kernel==0.7.0",
            "nltk",
            "kornia",
            # Data
            "pandas",
            "numpy",
            "pyarrow",          # parquet support
            "tqdm",
            # Vision
            "opencv-python-headless",
            "Pillow",
            # HuggingFace transfer (faster model downloads)
            "hf-transfer",
            "huggingface-hub",
        )
        # flash-attn is compiled from source and needs its own build-tool pre-install.
        # --no-build-isolation lets it link against the torch already in site-packages.
        # clang/clang++ is installed above via apt; packaging/wheel/ninja are the
        # remaining build-time deps that setup.py imports before compilation starts.
        .run_commands(
            "pip install wheel packaging ninja "
            "&& pip install flash-attn==2.8.3 --no-build-isolation",
            gpu="any",  # GPU must be present during compilation
        )
        .env({
            # Speed up HuggingFace model downloads
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Prevent tokenizer thread conflicts with DataLoader workers
            "TOKENIZERS_PARALLELISM": "false",
            # Reduce CUDA memory fragmentation from variable-length sequences
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
    )

    # ── Volumes ────────────────────────────────────────────────────────────────
    _modal_data_vol    = modal.Volume.from_name(CONFIG['modal_data_volume'],   create_if_missing=False)
    _modal_video_vol   = modal.Volume.from_name(CONFIG['modal_video_volume'],  create_if_missing=False)
    _modal_output_vol  = modal.Volume.from_name(CONFIG['modal_output_volume'], create_if_missing=True)

    # ── Secrets ────────────────────────────────────────────────────────────────
    _modal_secrets = []
    if CONFIG.get('modal_hf_secret'):
        _modal_secrets.append(modal.Secret.from_name(CONFIG['modal_hf_secret']))

    # ── App ────────────────────────────────────────────────────────────────────
    app = modal.App(CONFIG['modal_app_name'], image=_modal_image)

    @app.function(
        gpu=CONFIG['modal_gpu'],
        cpu=CONFIG['modal_cpu'],
        memory=CONFIG['modal_memory'],
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
        secrets=_modal_secrets,
    )
    def run_training_on_modal():
        """Entry point executed inside the Modal container."""
        _safe_main()

    @app.local_entrypoint()
    def modal_entrypoint():
        """Called on your local machine when you run `modal run <this_file>`."""
        print(f"  Dispatching training job to Modal ({CONFIG['modal_gpu']} GPU, "
              f"{CONFIG['modal_cpu']} CPUs, {CONFIG['modal_memory'] // 1024} GB RAM, "
              f"timeout={CONFIG['modal_timeout'] // 3600}h)...")
        try:
            run_training_on_modal.remote()
        finally:
            if CONFIG.get('modal_sync_metrics_on_exit'):
                print("\n  [Syncing metrics (TensorBoard, CSVs) from Modal volume to local machine...]")
                import subprocess
                # This pulls everything in /outputs/saved_metrics from the cloud and overwrites your local ./saved_metrics
                subprocess.run([
                    "modal", "volume", "get", 
                    CONFIG['modal_output_volume'], 
                    "/saved_metrics", 
                    "./"
                ])
            else:
                print("\n  [Sync skipped: 'modal_sync_metrics_on_exit' is False. Run 'modal volume get signbridge-outputs /saved_metrics ./' manually if needed.]")