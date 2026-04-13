
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

## Additional Imports
import time
import csv
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
# Deterministic mode for reproducibility (slightly slower than benchmark=True,
# but required so results are bit-exact across runs — important for a paper).
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
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
    'use_qlora': False,                           # True = 4-bit quantized base model
    'bnb_4bit_compute_dtype': 'bfloat16',
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant': True,

    # ── Video Processing ──
    'video_fps': 18,                              # Frames per second to sample (18 minimum for sign language)
    'video_min_pixels': 4 * 32 * 32,              # Min visual tokens per frame pair (~4 tokens)
    'video_max_pixels': 49 * 32 * 32,             # Max visual tokens per frame pair (49 = 224x224 at patch_size=16, merge=2)
    'video_total_pixels': 20480 * 32 * 32,        # Total pixel budget cap across all frames (None = no cap)

    # ── Chat Template / Prompts ──
    'system_prompt': 'You are a sign language translator.',
    'user_prompt': 'Translate this American Sign Language video into English.',

    # ── Sequence Lengths ──
    'max_text_tokens': 60,                        # Max tokens for the assistant response (translation)

    # ── LoRA ──
    'lora_r': 16,
    'lora_alpha': 32,                             # 2x rank
    'lora_target_modules': [                      # Language model targets: attention + MLP
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # attention projections
        'gate_proj', 'up_proj', 'down_proj',      # MLP projections
    ],
    'lora_include_vision': False,                  # Apply LoRA to vision encoder too
    'lora_vision_targets': [                      # Vision encoder targets
        'qkv', 'proj',                            # vision attention (fused qkv + output proj)
        'linear_fc1', 'linear_fc2',               # merger + deepstack_merger_list
    ],
    'lora_dropout': 0.05,
    'lora_bias': 'none',
    'lora_use_dora': False,                       # Weight-Decomposed LoRA
    'lora_use_rslora': True,                      # Rank-Stabilized LoRA (sqrt(r) scaling; better stability)

    # ── Core Training ──
    'num_epochs': 3,
    'batch_size': 1,                              # VRAM constraint — 1 sample per micro-batch

    # ── Gradient Accumulation ──
    'grad_accum_steps': 16,                       # Effective batch = 1 x 16 = 16

    # ── DataLoader Config ──
    'train_num_workers': 2,                       # 1 worker overlaps video decoding (~1s) with GPU compute (~17s)
    'train_prefetch_factor': 1,                   # Pre-load 1 batch ahead; reduced to lower pinned-memory pressure
    'train_pin_memory': False,                     # Disabled — pinned memory caused CUDA OOM in pin_memory thread
    'train_persistent_workers': True,             # Keep worker alive across epochs — avoids re-spawn overhead

    'val_num_workers': 1,                          # 1 worker overlaps video decoding with GPU inference during validation
    'val_prefetch_factor': 2,                      # Pre-load 2 batches ahead during validation
    'val_pin_memory': False,                        # Pinned memory for async CPU→GPU DMA during validation
    'val_persistent_workers': False,               # Keep val worker alive across validation runs

    # ── Learning Rate ──
    'learning_rate': 2e-5,
    'min_lr': 1e-6,                               # Cosine annealing down to it's 5% value
    'warmup_ratio': 0.03,                         # Fraction of total steps for warmup
    'warmup_steps': 100,                         # If set, overrides warmup_ratio
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
    'save_every_steps': 15,
    'keep_last_n_checkpoints': 3,
    'checkpoint_dir': Path('..') / 'checkpoints' / 'qwen3vl',

    # ── Evaluation ──
    'eval_every_steps': 200,
    'eval_every_steps_warmup': 200,               # More frequent eval early on
    'eval_warmup_threshold': 1000,                # Switch to normal eval freq after this step
    'max_eval_batches': 60,
    'num_print_samples': 5,
    'val_gen_batch_size': 1,                      # Low — generation is VRAM-intensive
    'max_generate_samples': 100,
    'val_beam_size': 1,                             # 1 = greedy (faster validation); run beam=4 on final checkpoint
    'val_length_penalty': 1.0,                      # > 1.0 favors longer outputs (counters BLEU brevity penalty); < 1.0 favors shorter
    'val_no_repeat_ngram_size': 3,                  # Block any n-gram from repeating; improves BLEU precision (0 = disabled)
    'val_repetition_penalty': 1.1,
    'val_max_new_tokens': 60,

    # ── Early Stopping ──
    'early_stopping_patience': 12,                # Evals without improvement before stopping

    # ── Performance & Memory Optimizations ──
    'use_8bit_adam': True,
    'use_gradient_checkpointing': True,
    'use_torch_compile': False,
    'torch_compile_mode': 'default',              # 'default' or 'max-autotune'

    # ── Resuming ──
    'resume_training': True,
    'load_best_model': False,
    'resume_checkpoint_step': 195,               # None = latest, or specific step number

    # ── Mid-Training LR Override ──────────────────────────────────────────────
    # SPECIAL USE ONLY: Use this block to manually correct the learning rate when
    # resuming from a checkpoint mid-training (e.g. if LR is too high and causing
    # noisy loss, or you want to fine-tune from a specific checkpoint with lower LR).
    # Set 'enabled' to False after the resumed run starts successfully.
    #
    # How it works:
    #   Resets the CosineAnnealingLR to start from 'lr' and decay to 'eta_min'
    #   over a fresh T_max cycle — no warmup, straight into cosine decay.
    # ─────────────────────────────────────────────────────────────────────────
    'lr_override': {
        'enabled': False,             # Set True to apply, False after resuming
        'lr': 2e-5,                   # New peak/starting LR for cosine
        'eta_min': 1e-6,              # New minimum LR floor for cosine
    },

    # ── Bucket Batch Sampling ──
    'use_bucket_batching': True,                  # Group clips by duration to minimise padding waste

    # ── Label Smoothing ──
    'label_smoothing': 0.1,

    # ── Seed ──
    'seed': 42,

    # ── Runtime ──
    'wait_for_manual_start': False,               # Interactive safety gate before training loop starts

    # ── Debug: save intermediate frames to disk ──
    'aug_debug_save_images': True,
    'aug_debug_save_interval': 20,          # Save one frame every N optimiser steps
    'aug_debug_save_dir': Path('..') / 'data' / 'debugging_images',
}

torch.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed_all(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class SignLanguageQwen3VLDataset(Dataset):
    """
    Reads TSV manifest and returns raw sample dicts.
    Video loading and tokenization happens in the collator (via Qwen3-VL processor).
    """

    def __init__(self, tsv_path, sep='\t'):
        df = pd.read_csv(tsv_path, sep=sep)
        self.samples = []
        missing = 0

        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f'Loading {Path(tsv_path).stem}'):
            fp = str(row.file_path)
            if Path(fp).exists():
                self.samples.append({
                    'vid': row.vid,
                    'text': str(row.text),
                    'duration_sec': float(row.duration_sec),
                    'file_path': fp,
                })
            else:
                missing += 1

        print(f"  ✓ Loaded {len(self.samples)} samples ({missing} missing files)")
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
        self.batches = [
            sorted_indices[i:i + batch_size]
            for i in range(0, len(sorted_indices), batch_size)
        ]
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
        self.debug_save_images = is_training and config.get('aug_debug_save_images', False)

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

        return labels

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

        for sample in batch:
            messages = self._build_messages(sample, include_assistant=True)

            # Get text template (not tokenized)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            all_texts.append(text)

            # Extract video frames using qwen_vl_utils
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16,
                return_video_kwargs=True, return_video_metadata=True,
            )

            if images:
                all_images.extend(images)

            if videos is not None:
                vids, metas = zip(*videos)
                all_videos.extend(list(vids))
                all_video_metadatas.extend(list(metas))

            # Merge video_kwargs (should be same for all samples)
            if video_kwargs:
                all_video_kwargs.update(video_kwargs)

        # Capture the first raw frame for debug saving (before processor converts to float tensors).
        # all_videos[0] is a numpy array of shape (T, H, W, C) uint8.
        _debug_frame = None
        if self.debug_save_images and all_videos:
            try:
                _debug_frame = all_videos[0][0]  # first frame of first video: (H, W, C) uint8
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

        # Extract metadata before moving to device
        _vids = batch.pop('_vids', [])
        _gts = batch.pop('_ground_truths', [])
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

    # ── Part 2: Generate Text for BLEU / ROUGE-L ──
    _orig_use_cache = getattr(model.config, 'use_cache', None) if hasattr(model, 'config') else None
    if _orig_use_cache is not None:
        model.config.use_cache = True

    references = []
    hypotheses = []
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

            for sample in batch_samples:
                messages = gen_collator._build_messages(sample, include_assistant=False)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_texts.append(text)

                images, videos, video_kwargs = process_vision_info(
                    messages, image_patch_size=16,
                    return_video_kwargs=True, return_video_metadata=True,
                )
                if images:
                    all_images.extend(images)
                if videos is not None:
                    vids, metas = zip(*videos)
                    all_videos.extend(list(vids))
                    all_video_metadatas.extend(list(metas))
                if video_kwargs:
                    all_video_kwargs.update(video_kwargs)

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

                references.append(ref_text)
                hypotheses.append(hyp_text)

                if len(sample_pairs) < config['num_print_samples']:
                    sample_pairs.append((ref_text, hyp_text))

            del inputs, generated_ids, generated_ids_trimmed

        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠️  OOM during generation — skipping batch {i}")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ⚠️  Error during generation batch {i}: {e}")

        gen_pbar.update(batch_end - i)
    gen_pbar.close()

    bleu1 = compute_bleu(references, hypotheses, max_n=1)
    bleu2 = compute_bleu(references, hypotheses, max_n=2)
    bleu4 = compute_bleu(references, hypotheses, max_n=4)
    rouge_l = compute_rouge_l(references, hypotheses)
    wer = compute_wer(references, hypotheses)
    meteor = compute_meteor(references, hypotheses)

    if _orig_use_cache is not None:
        model.config.use_cache = _orig_use_cache

    model.train()

    return {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu4': bleu4,
        'rouge_l': rouge_l,
        'wer': wer,
        'meteor': meteor,
        'sample_pairs': sample_pairs,
        'all_pairs': list(zip(references, hypotheses)),
        'num_eval_batches': num_batches,
        'num_gen_samples': len(hypotheses),
    }


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(model, processor, train_loader, val_loader, val_dataset,
          optimizer, scheduler, train_config, ckpt_manager,
          train_csv_logger, val_csv_logger, gen_samples_csv_logger, tb_writer,
          _val_collator=None,
          start_epoch=1, start_global_step=0, start_steps_done_in_epoch=None,
          best_val_loss=float('inf'), start_evals_without_improvement=0, start_elapsed_sec=0.0):
    """Full training loop for Qwen3-VL LoRA fine-tuning."""

    # Unpack config
    num_epochs = train_config['num_epochs']
    grad_accum_steps = train_config['grad_accum_steps']
    max_grad_norm = train_config['max_grad_norm']
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

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        epoch_loss_sum = None
        epoch_microbatches = 0

        optimizer.zero_grad(set_to_none=True)
        valid_microbatches_in_window = 0
        window_raw_loss_sum = None

        # Mid-epoch resume: skip already-processed micro-batches at the sampler level
        # so the collator (video decoding) never runs for skipped batches.
        micro_steps_already_processed = 0
        if epoch == start_epoch and start_global_step > 0 and steps_done_in_epoch > 0:
            micro_steps_already_processed = steps_done_in_epoch * grad_accum_steps
            # Tell the bucket sampler to skip at index level (no collator overhead)
            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_skip'):
                train_loader.batch_sampler.set_skip(micro_steps_already_processed)
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

            # Extract metadata
            _vids = batch.pop('_vids', [])
            _gts = batch.pop('_ground_truths', [])
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
                if label_smoothing > 0.0:
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    raw_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        label_smoothing=label_smoothing,
                    )
                    del logits, shift_logits, shift_labels
                else:
                    raw_loss = outputs.loss
                del labels

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

                del outputs, batch_gpu, forward_inputs, raw_loss, raw_loss_detached, loss

                valid_microbatches_in_window += 1

            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  OOM at micro_step {micro_step} — skipping batch (accumulated gradients preserved), clearing cache")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  ⚠️  Error at micro_step {micro_step}: {e} — skipping batch (accumulated gradients preserved)")
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
                    _all_trainable_params, max_norm=max_grad_norm
                ).item()

                window_valid_microbatches = valid_microbatches_in_window
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                valid_microbatches_in_window = 0

                global_step += 1
                window_avg_loss = (window_raw_loss_sum / window_valid_microbatches).item()
                step_losses.append(window_avg_loss)
                step_grad_norms.append(grad_norm)
                window_raw_loss_sum = None

                # ── Debug: save one frame from the processed video ──
                if (_dbg_frame is not None
                        and global_step % train_config.get('aug_debug_save_interval', 500) == 0):
                    _dbg_dir = Path(train_config.get('aug_debug_save_dir', '../data/debugging_images'))
                    _dbg_dir.mkdir(parents=True, exist_ok=True)
                    _dbg_np = _dbg_frame.cpu().numpy() if hasattr(_dbg_frame, 'cpu') else np.asarray(_dbg_frame)
                    # Squeeze out any batch/time dimensions added by the collate fn
                    while _dbg_np.ndim > 3:
                        _dbg_np = _dbg_np[0]
                    # Only save if we have a plausible image shape (H, W) or (H, W, C)
                    if _dbg_np.ndim in (2, 3) and _dbg_np.shape[-1] in (1, 3, 4):
                        if _dbg_np.dtype != np.uint8:
                            _dbg_np = np.clip(_dbg_np, 0, 255).astype(np.uint8)
                        if _dbg_np.ndim == 3 and _dbg_np.shape[-1] == 1:
                            _dbg_np = _dbg_np[:, :, 0]
                        Image.fromarray(_dbg_np).save(
                            str(_dbg_dir / f"step_{global_step:07d}.jpg"), quality=90
                        )

                # ── Logging ──
                if global_step % log_every == 0:
                    avg_loss = sum(step_losses) / len(step_losses)
                    avg_grad_norm = sum(step_grad_norms) / len(step_grad_norms)
                    current_lr = scheduler.get_last_lr()[0]
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
                        f"LR: {current_lr:.4e} | "
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
                        for i, (ref, hyp) in enumerate(val_results['sample_pairs'], 1):
                            print(f"    [{i}] REF: \"{ref}\"")
                            print(f"        HYP: \"{hyp}\"")

                    # Log to CSV
                    val_csv_logger.log({
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
                    })

                    # Log generated samples
                    for ref, hyp in val_results['all_pairs']:
                        gen_samples_csv_logger.log({
                            'global_step': global_step,
                            'epoch': epoch,
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

                    # Early stopping / best model
                    elapsed = time.time() - training_start
                    if val_results['val_loss'] < best_val_loss:
                        best_val_loss = val_results['val_loss']
                        evals_without_improvement = 0
                        ckpt_manager.save_best(
                            model, optimizer, scheduler, epoch, global_step,
                            (micro_step + 1) // grad_accum_steps, best_val_loss,
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

                    print(f"{'─' * 60}\n")

                # ── Periodic Checkpoint ──
                if global_step % save_every == 0:
                    elapsed = time.time() - training_start
                    ckpt_manager.save_periodic(
                        model, optimizer, scheduler, epoch, global_step,
                        (micro_step + 1) // grad_accum_steps, best_val_loss,
                        evals_without_improvement, elapsed
                    )

        # End of epoch
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = ((epoch_loss_sum / max(epoch_microbatches, 1)).item()
                          if epoch_loss_sum is not None else 0.0)
        print(f"\n{'=' * 60}")
        print(f"  ✅ Epoch {epoch}/{num_epochs} Complete")
        print(f"  Avg Train Loss: {epoch_avg_loss:.4f} | Time: {format_time(epoch_time)}")
        print(f"{'=' * 60}\n")

    return global_step, best_val_loss


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

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
    print(f"    Rank / Alpha           : {CONFIG['lora_r']} / {CONFIG['lora_alpha']}")
    print(f"    Target modules (LM)    : {', '.join(CONFIG['lora_target_modules'])}")
    print(f"    Include vision encoder : {CONFIG['lora_include_vision']}")
    if CONFIG['lora_include_vision']:
        print(f"    Vision targets         : {', '.join(CONFIG['lora_vision_targets'])}")
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
    print(f"    LR                     : {CONFIG['learning_rate']:.2e} -> {CONFIG['min_lr']:.2e}  (warmup: {warmup_info})")

    print("\n  [ EVALUATION & CHECKPOINTING ]")
    print(f"    Eval every             : {CONFIG['eval_every_steps']} steps  (warmup: {CONFIG['eval_every_steps_warmup']} until step {CONFIG['eval_warmup_threshold']})")
    print(f"    Max eval batches       : {CONFIG['max_eval_batches']}")
    print(f"    Max gen samples        : {CONFIG['max_generate_samples']}")
    print(f"    Beam size              : {CONFIG['val_beam_size']}  |  Length penalty: {CONFIG['val_length_penalty']}  |  No-repeat ngram: {CONFIG['val_no_repeat_ngram_size']}  |  Rep penalty: {CONFIG['val_repetition_penalty']}")
    print(f"    Save every             : {CONFIG['save_every_steps']} steps  |  Keep last: {CONFIG['keep_last_n_checkpoints']}")
    print(f"    Early stopping pat.    : {CONFIG['early_stopping_patience']} evals")
    print(f"    Checkpoint dir         : {CONFIG['checkpoint_dir'].resolve()}")
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

    # ── Load Data Manifests ──
    print("📂 Loading training data...")
    train_dataset = SignLanguageQwen3VLDataset(CONFIG['data_train_tsv'], CONFIG['tsv_sep'])

    print("\n📂 Loading validation data...")
    val_dataset = SignLanguageQwen3VLDataset(CONFIG['data_val_tsv'], CONFIG['tsv_sep'])

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

    model = AutoModelForImageTextToText.from_pretrained(CONFIG['model_name'], **model_kwargs)

    if CONFIG['use_qlora']:
        model = prepare_model_for_kbit_training(model)

    processor = AutoProcessor.from_pretrained(CONFIG['model_name'])

    load_time = time.time() - load_start
    model_vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  Model loaded in {load_time:.1f}s | VRAM: {model_vram:.2f} GB")

    # ── Apply LoRA ──
    print("\nApplying LoRA...")
    target_modules = list(CONFIG['lora_target_modules'])
    if CONFIG['lora_include_vision']:
        target_modules.extend(CONFIG['lora_vision_targets'])

    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=target_modules,
        lora_dropout=CONFIG['lora_dropout'],
        bias=CONFIG['lora_bias'],
        task_type='CAUSAL_LM',
        use_dora=CONFIG['lora_use_dora'],
        use_rslora=CONFIG['lora_use_rslora'],
    )

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

    model.print_trainable_parameters()
    _total_params = sum(p.numel() for p in model.parameters())
    _trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {_total_params:,}")
    print(f"  Trainable parameters: {_trainable_params:,} ({_trainable_params / _total_params * 100:.2f}%)")

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
    )

    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / CONFIG['grad_accum_steps'])
    total_optimizer_steps = optimizer_steps_per_epoch * CONFIG['num_epochs']

    print(f"  Train loader: {len(train_loader)} micro-batches/epoch")
    print(f"  Optimizer steps/epoch: {optimizer_steps_per_epoch}")
    print(f"  Total optimizer steps: {total_optimizer_steps}")

    # ── Optimizer ──
    print("\n⚙️  Setting up optimizer...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if CONFIG['use_8bit_adam'] and _BNB_AVAILABLE:
        optimizer = bnb.optim.PagedAdamW8bit(
            trainable_params,
            lr=CONFIG['learning_rate'],
            betas=CONFIG['adam_betas'],
            weight_decay=CONFIG['weight_decay'],
        )
        print("  ✓ Using Paged 8-bit AdamW (bitsandbytes) — optimizer states page to CPU under VRAM pressure")
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=CONFIG['learning_rate'],
            betas=CONFIG['adam_betas'],
            weight_decay=CONFIG['weight_decay'],
        )
        print("  Using standard AdamW")

    # ── Scheduler: Linear warmup + Cosine annealing ──
    if CONFIG['warmup_steps'] is not None:
        warmup_steps = CONFIG['warmup_steps']
    else:
        warmup_steps = int(total_optimizer_steps * CONFIG['warmup_ratio'])

    cosine_steps = max(total_optimizer_steps - warmup_steps, 1)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-8 / CONFIG['learning_rate'],
        end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=CONFIG['min_lr']
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    print(f"\n📈 LR Schedule:")
    print(f"  Warmup: {warmup_steps} steps | Cosine: {cosine_steps} steps")
    print(f"  LR: {CONFIG['learning_rate']:.2e} -> {CONFIG['min_lr']:.2e}")

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
        override_lr = lr_override['lr']
        override_eta_min = lr_override['eta_min']
        remaining_steps = total_optimizer_steps - start_global_step

        # Reset optimizer LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = override_lr

        # Build a fresh cosine scheduler from the override LR
        fresh_cosine = CosineAnnealingLR(
            optimizer, T_max=max(remaining_steps, 1), eta_min=override_eta_min
        )
        # Replace the current scheduler with a simple wrapper
        scheduler = fresh_cosine

        print(f"\n⚠️  LR OVERRIDE APPLIED:")
        print(f"    New LR: {override_lr:.2e} -> {override_eta_min:.2e}")
        print(f"    Cosine T_max: {remaining_steps} remaining steps")
        print(f"    ⚠️  WARNING: Set lr_override.enabled = False after this run!\n")

    # ── Checkpoint Manager ──
    ckpt_manager = CheckpointManager(CONFIG['checkpoint_dir'], CONFIG['keep_last_n_checkpoints'])

    # ── CSV Loggers ──
    train_csv_logger = CSVLogger(CONFIG['train_log_file'], [
        'timestamp', 'global_step', 'epoch', 'train_loss', 'train_ppl',
        'lr', 'grad_norm', 'cuda_mem_gb', 'cuda_peak_gb',
        'steps_per_sec', 'elapsed_sec', 'eta_sec',
    ])
    val_csv_logger = CSVLogger(CONFIG['val_log_file'], [
        'timestamp', 'global_step', 'epoch', 'val_loss', 'val_ppl',
        'bleu1', 'bleu2', 'bleu4', 'rouge_l', 'wer', 'meteor',
        'num_eval_batches', 'num_gen_samples',
    ])
    gen_samples_csv_logger = CSVLogger(CONFIG['gen_samples_log_file'], [
        'timestamp', 'global_step', 'epoch', 'reference', 'hypothesis',
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

        del first_batch
    except Exception as e:
        print(f"  ⚠️  Warning: Label verification failed: {e}")

    # ── Train ──
    print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "")

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
        start_epoch=start_epoch,
        start_global_step=start_global_step,
        start_steps_done_in_epoch=start_steps_done_in_epoch,
        best_val_loss=best_val_loss,
        start_evals_without_improvement=start_evals_without_improvement,
        start_elapsed_sec=start_elapsed_sec,
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
