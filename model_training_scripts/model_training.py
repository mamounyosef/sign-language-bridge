
## Imports & Setup

# Disable TensorFlow backend to avoid protobuf conflicts (PyTorch-only project)
import os
import sys
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

from pathlib import Path
from functools import partial, wraps
from typing import Optional, Tuple
import math
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

# CNN backbone
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.v2 as T_v2
import torchvision.transforms.functional as TF
import torchvision.io

# Video decoding
try:
    import decord
    decord.bridge.set_bridge('torch')
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False

## Additional Imports for Training
import time
import csv
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import sacrebleu

import tempfile
import shutil
import gc
import torch.multiprocessing as mp
import torch.utils.checkpoint

# Windows: must be set at module level so spawned DataLoader workers inherit it.
# Prevents ERROR_COMMITMENT_LIMIT (error 1455) when sharing large frame tensors.
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# %%
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# %%
# ─── Training Constants ───

# Maximum loss value for perplexity calculation to prevent overflow
# math.exp(20) ≈ 485 million — a reasonable max for display
MAX_PPL_CAP = 20

# ImageNet normalization constants (used for ResNet50 preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# %%
## Configuration
CONFIG = {
    # ── Data Paths ──
    'data_train_csv': Path('..') / 'data' / '2_dataset_train.tsv',
    'data_val_csv': Path('..') / 'data' / '2_dataset_val.tsv',
    'csv_sep': '\t',

    # ── CNN Backbone (ResNet50) ──
    'resnet_output_dim': 2048,                # ResNet50 output feature dimension
    'cnn_frame_size': 256,                    # Letterbox target size (both width and height)
    'cnn_chunk_size': 16,                      # Frames processed through ResNet50 per CUDA call (tune based on VRAM)
    'cnn_freeze_layers': ['conv1', 'bn1', 'layer1', 'layer2'],  # Frozen early layers

    # ── CNN Projection MLP ──
    'cnn_projection_dim': 768,                # Output dim → encoder input (== encoder_d_model)
    'cnn_projection_dropout': 0.15,

    # ── Sequence Lengths ──
    'max_video_frames': 160,                  # At 20 FPS = 8s (max clip ~8s)
    'max_text_tokens': 50,                    # Excluding BOS/EOS

    # ── Tokenizer / Decoder ──
    'decoder_model_name': 'meta-llama/Llama-3.2-1B',
    'decoder_hidden_size': 2048,
    'decoder_num_layers': 16,

    # ── Transformer Encoder ──
    'encoder_d_model': 768,
    'encoder_num_heads': 8,
    'encoder_num_layers': 6,
    'encoder_feedforward_dim': 3072,          # 4 * 768
    'encoder_dropout': 0.15,
    'encoder_max_len': 5000,

    # ── Cross-Attention ──
    'bottleneck_dim': 768,
    'cross_attn_num_heads': 8,
    'cross_attn_dropout': 0.1,
    'cross_attn_layers': [0, 2, 4, 7, 10, 13, 15],  # 7 out of 16
    'weight_sharing_pairs': [],

    # ── LoRA ──
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    'lora_modules_to_save': [],  # embed_tokens and lm_head kept frozen
    'lora_dropout': 0.1,
    'lora_bias': "none",

    # ── Data Augmentation (video frames, training only) ──
    'aug_temporal_jitter': True,
    'aug_temporal_jitter_range': 2,           # Max frames to shift (±2)
    'aug_temporal_jitter_prob': 0.5,
    'aug_color_jitter': True,
    'aug_color_jitter_prob': 0.5,
    'aug_color_jitter_brightness': 0.3,
    'aug_color_jitter_contrast': 0.3,
    'aug_color_jitter_saturation': 0.2,
    'aug_color_jitter_hue': 0.05,
    'aug_random_grayscale': True,
    'aug_random_grayscale_prob': 0.1,
    'aug_gaussian_blur': True,
    'aug_gaussian_blur_prob': 0.2,
    'aug_gaussian_blur_kernel': (5, 5),
}

## Training Configuration
TRAIN_CONFIG = {
    # ── Core Training ──
    'num_epochs': 25,
    'batch_size': 4,

    # ── Gradient Accumulation ──
    'grad_accum_steps': 16,                   # effective batch = 2 * 32 = 64

    # ── DataLoader config ──
    # Frames are now uint8 (~125MB/batch vs ~500MB float32)
    # 2 workers keeps CPU free for interactive use during long runs
    # (increase to 4 if GPU shows idle gaps and you don't need the PC)
    'train_num_workers': 2,
    'train_prefetch_factor': 1,
    'val_num_workers': 1,
    'val_prefetch_factor': 1,
    'use_bucket_batching': True,

    # ── Learning Rates ──
    # New random weights (MLP + Encoder + Cross-attention)
    'encoder_lr': 9e-5,
    'encoder_min_lr': 9e-7, # 1% of initial LR
    # CNN backbone (ResNet50 layer3 + layer4)
    'cnn_lr': 1e-5,
    'cnn_min_lr': 1e-7, # 1% of initial LR
    # LoRA adapters (fine-tuning pretrained decoder)
    'decoder_lr': 1e-4,
    'decoder_min_lr': 1e-6, # 1% of initial LR

    'warmup_steps': 250,                      # Warmup for new_weights optimizer
    'cnn_warmup_steps': 100,                  # Warmup for CNN optimizer (starts at cnn_freeze_steps)
    'decoder_warmup_steps': 100,              # Warmup for decoder optimizer (starts at decoder_freeze_steps)
    'weight_decay': 0.05,
    'adam_betas': (0.9, 0.98),
    'max_grad_norm': 0.6,

    # ── Phased Training Curriculum ──
    # Stage 1 (0 → cnn_freeze_steps): only MLP + Encoder + Cross-attn train
    # Stage 2 (cnn_freeze_steps → decoder_freeze_steps): + CNN layer3/layer4
    # Stage 3 (decoder_freeze_steps → end): + Decoder (LoRA + norms)
    'cnn_freeze_steps': 0,                 # CNN unfreezes at step 1000
    'decoder_freeze_steps': 0,             # Decoder unfreezes at step 1500

    # ── Logging ──
    'log_every_steps': 5,
    'train_log_file': Path('..') / 'saved_metrics' / 'train_log.csv',
    'val_log_file': Path('..') / 'saved_metrics' / 'val_log.csv',
    'gen_samples_log_file': Path('..') / 'saved_metrics' / 'gen_samples_log.csv',
    'tensorboard_dir': Path('..') / 'saved_metrics' / 'tensorboard' / 'signbridge_training',

    # ── Checkpointing ──
    'save_every_steps': 200,
    'keep_last_n_checkpoints': 3,
    'checkpoint_dir': Path('..') / 'checkpoints',

    # ── Evaluation ──
    'eval_every_steps': 220,
    'eval_every_steps_warmup': 900,
    'eval_warmup_threshold': 2000,
    'max_eval_batches': 60,
    'max_generate_samples': 120,
    'num_print_samples': 4,
    'val_gen_batch_size': 8,                  # Reduced from 16 (frames use more VRAM per sample)
    'val_beam_size': 4,
    'val_repetition_penalty': 1.15,
    'val_use_kv_cache': True,

    # ── Early Stopping ──
    'early_stopping_patience': 25,


    # ── Performance & Memory Optimizations ──
    'use_sdpa': True,
    'use_8bit_adam': True,
    'use_gradient_checkpointing_encoder': True,
    'use_gradient_checkpointing_cnn': True,   # Checkpoints layer3/layer4 blocks
    'use_gradient_checkpointing_decoder': True,   # Saves 300-500MB VRAM in Stage 3, ~20-30% slower

    # ── Resuming ──
    'resume_training': False,
    'load_best_model': True,
    'resume_checkpoint_step': 0,
}

# Max sequence lengths (safety caps to avoid OOM)
MAX_VIDEO_FRAMES = CONFIG['max_video_frames']
MAX_TEXT_TOKENS = CONFIG['max_text_tokens']


# %%
# ═══════════════════════════════════════════════════════════════
#  VIDEO FRAME LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════

def letterbox_frames_batch(frames, target_size):
    """
    Letterbox a batch of frames to target_size x target_size in one vectorized call.
    All frames in a video share the same H and W, so scale/padding is computed once.
    Preserves aspect ratio, pads with black (zeros).

    Args:
        frames: (T, C, H, W) uint8 tensor (values 0-255)
        target_size: int, e.g. 256

    Returns:
        (T, C, target_size, target_size) uint8 tensor
    """
    T, C, H, W = frames.shape
    scale = min(target_size / H, target_size / W)
    new_h = int(H * scale)
    new_w = int(W * scale)

    resized = TF.resize(frames, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    return TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)


def load_video_frames(video_path, max_frames, target_size):
    """
    Load video frames using decord (with torchvision fallback).
    Returns uint8 tensors — float conversion and ImageNet normalization
    happen on the GPU inside ResNetFrameEncoder.forward() to save CPU RAM.

    Args:
        video_path: str, path to .mp4 file
        max_frames: int, maximum number of frames to load
        target_size: int, letterbox target size

    Returns:
        frames: (T, 3, target_size, target_size) uint8 tensor (values 0-255)
    """
    if _DECORD_AVAILABLE:
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            num_frames = min(len(vr), max_frames)
            indices = list(range(num_frames))
            # decord with torch bridge returns (T, H, W, C) uint8 tensor
            raw_frames = vr.get_batch(indices)  # (T, H, W, C) uint8
            frames = raw_frames.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W) uint8
        except Exception as e:
            raise RuntimeError(f"Failed to load video with decord: {video_path}\n{e}")
    else:
        # Fallback to torchvision
        import torchvision.io
        video, _, info = torchvision.io.read_video(video_path, pts_unit='sec')
        # video: (T, H, W, C) uint8
        if video.shape[0] > max_frames:
            video = video[:max_frames]
        frames = video.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W) uint8

    # Letterbox all frames in one vectorized call — torchvision handles (T, C, H, W) natively
    frames = letterbox_frames_batch(frames, target_size)  # (T, 3, target_size, target_size) uint8

    return frames


def load_frames_from_folder(folder_path, max_frames):
    """
    Load pre-extracted JPEG frames from a folder produced by 09_extract_frames.py.
    Frames are already letterboxed to 256×256 — no resizing needed here.

    Args:
        folder_path: str or Path, folder containing 0000.jpg, 0001.jpg, ...
        max_frames: int, maximum number of frames to load

    Returns:
        (T, 3, H, W) uint8 tensor (values 0-255)
    """
    folder = Path(folder_path)
    frame_files = sorted(folder.glob("*.jpg"))[:max_frames]
    if len(frame_files) == 0:
        raise RuntimeError(f"No JPEG frames found in: {folder}")
    frames = torch.stack([torchvision.io.read_image(str(f)) for f in frame_files], dim=0)
    return frames  # (T, 3, H, W) uint8


class SignLanguageDataset(Dataset):
    def __init__(self, manifest, tokenizer, max_frames=MAX_VIDEO_FRAMES, max_tokens=MAX_TEXT_TOKENS,
                 train=False, augment_config=None):
        """
        Args:
            manifest: List of dicts with keys: sentence_name, text, duration, file_path
            tokenizer: Pretrained tokenizer (Llama 3.2)
            max_frames: Max video frames (truncate if longer)
            max_tokens: Max text tokens excluding BOS/EOS (truncate if longer)
            train: Whether this is training data (enables augmentation)
            augment_config: Dict with augmentation parameters (uses CONFIG if None)
        """
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.train = train
        self.aug_config = augment_config if augment_config is not None else CONFIG
        self.target_size = CONFIG['cnn_frame_size']

        # Note: ImageNet normalization is NOT done here.
        # Frames stay as uint8 (0-255) in the DataLoader to save CPU RAM (~4x reduction).
        # Conversion to float and normalization happen on GPU in ResNetFrameEncoder.forward().

        # Build training augmentation transforms (applied per-frame)
        if self.train:
            aug_transforms = []
            if self.aug_config.get('aug_color_jitter', False):
                aug_transforms.append(
                    T_v2.RandomApply([
                        T_v2.ColorJitter(
                            brightness=self.aug_config.get('aug_color_jitter_brightness', 0.3),
                            contrast=self.aug_config.get('aug_color_jitter_contrast', 0.3),
                            saturation=self.aug_config.get('aug_color_jitter_saturation', 0.2),
                            hue=self.aug_config.get('aug_color_jitter_hue', 0.05),
                        )
                    ], p=self.aug_config.get('aug_color_jitter_prob', 0.5))
                )
            if self.aug_config.get('aug_random_grayscale', False):
                aug_transforms.append(
                    T_v2.RandomGrayscale(p=self.aug_config.get('aug_random_grayscale_prob', 0.1))
                )
            if self.aug_config.get('aug_gaussian_blur', False):
                kernel = self.aug_config.get('aug_gaussian_blur_kernel', (5, 5))
                aug_transforms.append(
                    T_v2.RandomApply([
                        T_v2.GaussianBlur(kernel_size=kernel)
                    ], p=self.aug_config.get('aug_gaussian_blur_prob', 0.2))
                )
            self.aug_transform = T_v2.Compose(aug_transforms) if aug_transforms else None
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        sample = self.manifest[idx]

        # Load frames — auto-detect pre-extracted folder vs raw video file
        try:
            fp = sample['file_path']
            if Path(fp).is_dir():
                # Pre-extracted JPEGs (09_extract_frames.py) — already letterboxed
                frames = load_frames_from_folder(fp, max_frames=self.max_frames)
            else:
                # Raw video file — decode + letterbox on the fly
                frames = load_video_frames(fp, max_frames=self.max_frames,
                                           target_size=self.target_size)
            # (T, 3, H, W) uint8 (values 0-255)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load frames for {sample['sentence_name']}: {e}\n"
                f"Path: {sample['file_path']}"
            )

        # Truncate frames if too long
        if frames.shape[0] > self.max_frames:
            frames = frames[:self.max_frames]

        # ── Data Augmentation (training only) ──
        if self.train:
            # Temporal jitter: shift sequence by ±max_shift frames
            if self.aug_config.get('aug_temporal_jitter', False) and random.random() < self.aug_config.get('aug_temporal_jitter_prob', 0.5):
                max_shift = self.aug_config.get('aug_temporal_jitter_range', 2)
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    if shift > 0:
                        # Shift right: pad at start with black (uint8 zeros), truncate at end
                        pad = torch.zeros(shift, *frames.shape[1:], dtype=torch.uint8)
                        frames = torch.cat([pad, frames[:-shift]], dim=0)
                    else:
                        # Shift left: truncate at start, pad at end
                        pad = torch.zeros(-shift, *frames.shape[1:], dtype=torch.uint8)
                        frames = torch.cat([frames[-shift:], pad], dim=0)

            # Per-frame spatial augmentations (color jitter, grayscale, blur).
            # torchvision v2 transforms operate natively on uint8 tensors — no float
            # conversion needed here, saving CPU RAM.
            if self.aug_transform is not None:
                frames = self.aug_transform(frames)

        # No normalization here — frames remain uint8 (0-255).
        # float conversion + ImageNet normalization happen on GPU in ResNetFrameEncoder.forward().

        # Tokenize text
        text = sample['text']
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Truncate tokens if too long (reserve 2 for BOS/EOS)
        if len(token_ids) > self.max_tokens:
            token_ids = token_ids[:self.max_tokens]

        # Add BOS at start, EOS at end
        token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        return {
            'frames': frames,              # (T, 3, H, W) uint8  — normalized to float on GPU
            'token_ids': token_ids,        # (L,) where L = text_len + 2 (BOS + EOS)
            'sentence_name': sample['sentence_name']  # For debugging
        }


# %%
## Collate Function for Batching

def collate_fn(batch, pad_token_id):
    """
    Collate function to batch variable-length video frame samples.

    Args:
        batch: List of dicts from Dataset.__getitem__
        pad_token_id: Token ID to use for padding text

    Returns:
        Dict with batched tensors:
            frames: (B, max_T, 3, H, W) uint8 - padded frames (0=black for padding)
            frame_padding_mask: (B, max_T) - bool, True=padding, False=real frame
            token_ids: (B, max_L) - padded token IDs
            text_attention_mask: (B, max_L) - bool, True=real token, False=padding
    """
    frames_list = [item['frames'] for item in batch]           # List of (T_i, 3, H, W) uint8
    token_ids_list = [item['token_ids'] for item in batch]     # List of (L_i,)

    # Find max lengths in this batch
    max_frames = max(f.shape[0] for f in frames_list)
    max_token_length = max(tokens.shape[0] for tokens in token_ids_list)

    batch_size = len(batch)
    _, C, H, W = frames_list[0].shape

    # Initialize padded tensors — uint8 zeros = black pixels (correct padding value)
    padded_frames = torch.zeros(batch_size, max_frames, C, H, W, dtype=torch.uint8)
    frame_padding_mask = torch.ones(batch_size, max_frames, dtype=torch.bool)  # True = padding
    padded_token_ids = torch.full((batch_size, max_token_length), pad_token_id, dtype=torch.long)
    text_attention_mask = torch.zeros(batch_size, max_token_length, dtype=torch.bool)

    # Fill in actual data
    for i in range(batch_size):
        T = frames_list[i].shape[0]
        padded_frames[i, :T] = frames_list[i]
        frame_padding_mask[i, :T] = False  # Real frames are NOT padding

        L = token_ids_list[i].shape[0]
        padded_token_ids[i, :L] = token_ids_list[i]
        text_attention_mask[i, :L] = True  # Real tokens

    return {
        'frames': padded_frames,                  # (B, max_T, 3, H, W)
        'frame_padding_mask': frame_padding_mask,  # (B, max_T) bool, True=padding
        'token_ids': padded_token_ids,             # (B, max_L) long
        'text_attention_mask': text_attention_mask, # (B, max_L) bool
    }


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    Groups samples with similar sequence lengths into the same batch
    to minimise padding waste.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        lengths = [
            min(int(s.get('duration', 0) * 20), MAX_VIDEO_FRAMES)
            for s in dataset.manifest
        ]
        sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
        self.batches = [
            sorted_indices[i:i + batch_size]
            for i in range(0, len(sorted_indices), batch_size)
        ]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# %%
# ═══════════════════════════════════════════════════════════════
#  MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════

def init_from_scratch_weights(module, std=0.02):
    """
    Applies standard Transformer initialization (Normal(0, std)) to
    from-scratch layers.
    """
    for name, param in module.named_parameters():
        if 'bias' in name:
            if param.requires_grad:
                nn.init.zeros_(param)
        elif 'weight' in name:
            if 'norm' in name.lower():
                if param.requires_grad:
                    nn.init.ones_(param)
            else:
                if param.requires_grad and param.dim() >= 2:
                    nn.init.normal_(param, mean=0.0, std=std)


# ── ResNet50 Frame Encoder ──

class ResNetFrameEncoder(nn.Module):
    """
    Wraps ResNet50 for per-frame visual feature extraction.
    Frozen early layers, fine-tuned deep layers, chunked forward pass.

    Input:  (B, T, 3, H, W)
    Output: (B, T, 2048)
    """
    def __init__(self, freeze_layers=None, use_gradient_checkpointing=False, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Load pretrained ResNet50
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Extract layers (remove fc classification head)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool  # AdaptiveAvgPool2d(1, 1)
        # fc is NOT kept — we only want features

        # ImageNet normalization constants as buffers — moved to GPU with the model,
        # used to normalize uint8 frames to float in forward() instead of in the DataLoader.
        self.register_buffer(
            'norm_mean',
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'norm_std',
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

        # Freeze specified layers
        if freeze_layers is None:
            freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2']
        self.freeze_layers = freeze_layers  # store for use in train() override
        for layer_name in freeze_layers:
            layer = getattr(self, layer_name)
            for param in layer.parameters():
                param.requires_grad = False
            layer.eval()  # Keep frozen layers in eval mode (affects BatchNorm)

    def train(self, mode=True):
        """Override train to keep frozen layers in eval mode."""
        super().train(mode)
        # Frozen layers must stay in eval mode for correct BatchNorm behavior
        for layer_name in self.freeze_layers:
            layer = getattr(self, layer_name, None)
            if layer is not None:
                layer.eval()
        return self

    def _forward_features(self, x):
        """Forward through all conv layers (for a chunk of frames)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # Gradient checkpointing on trainable layers
        if self.training and self.use_gradient_checkpointing:
            for block in self.layer3:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            for block in self.layer4:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.avgpool(x)        # (N, 2048, 1, 1)
        x = torch.flatten(x, 1)    # (N, 2048)
        return x

    def _preprocess_chunk(self, x_uint8):
        """Convert a uint8 chunk to normalized bfloat16 on GPU.
        Done per-chunk to avoid materializing ALL B*T frames as float32 at once."""
        x = x_uint8.float() / 255.0
        x = (x - self.norm_mean.float()) / self.norm_std.float()
        return x.to(dtype=self.conv1.weight.dtype)

    def forward(self, frames):
        """
        Args:
            frames: (B, T, 3, H, W) uint8 tensor (values 0-255) on GPU

        Returns:
            (B, T, 2048) - per-frame feature vectors
        """
        B, T, C, H, W = frames.shape

        # Keep uint8 until chunking — only convert to float per-chunk to limit peak VRAM.
        # With B=2, T=160, chunk_size=8: peak is 8×3×256×256×4 ≈ 6MB instead of ~251MB.
        flat = frames.reshape(B * T, C, H, W)  # stays uint8
        chunks = flat.split(self.chunk_size, dim=0)
        feats = torch.cat(
            [self._forward_features(self._preprocess_chunk(chunk)) for chunk in chunks],
            dim=0,
        )  # (B*T, 2048)

        feats = feats.reshape(B, T, -1)  # (B, T, 2048)
        return feats


# ── CNN Projection MLP ──

class CNNProjection(nn.Module):
    """
    Projects CNN features (2048) down to encoder d_model (768).

    Input:  (B, T, 2048)
    Output: (B, T, 768)
    """
    def __init__(self, input_dim=2048, d_model=768, dropout=0.15):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.d_model = d_model
        init_from_scratch_weights(self)

    def forward(self, x):
        """
        Args:
            x: (B, T, 2048) CNN features
        Returns:
            (B, T, d_model) projected representations
        """
        return self.projection(x)


# ── Transformer Encoder ──

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=768, num_heads=8, num_layers=4,
                 feedforward_dim=3072, dropout=0.15, max_len=5000):
        super().__init__()

        self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        init_from_scratch_weights(self)

    def forward(self, x, src_key_padding_mask=None):
        x = self.pos_encoder(x)

        if self.training and getattr(self, 'use_gradient_checkpointing', False):
            for layer in self.transformer_encoder.layers:
                def make_layer_fn(l):
                    def fn(h):
                        return l(h, src_key_padding_mask=src_key_padding_mask)
                    return fn
                x = torch.utils.checkpoint.checkpoint(
                    make_layer_fn(layer), x, use_reentrant=False
                )
            return x

        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)


# ── Encoder Projection (normalization) ──

class EncoderProjection(nn.Module):
    def __init__(self, encoder_dim=768):
        super().__init__()
        self.layer_norm = nn.LayerNorm(encoder_dim)
        init_from_scratch_weights(self)

    def forward(self, encoder_outputs):
        return self.layer_norm(encoder_outputs)


# ── Cross-Attention ──

class CrossAttentionModule(nn.Module):
    """Bottleneck cross-attention module (can be shared across layers)."""
    def __init__(self, hidden_size, encoder_dim, bottleneck_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn_in_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.key_proj = nn.Linear(encoder_dim, bottleneck_dim)
        self.value_proj = nn.Linear(encoder_dim, bottleneck_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_out_proj = nn.Linear(bottleneck_dim, hidden_size)
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        init_from_scratch_weights(self)

        # Zero-init output projection: cross-attention starts as no-op
        nn.init.zeros_(self.cross_attn_out_proj.weight)
        nn.init.zeros_(self.cross_attn_out_proj.bias)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
        original_dtype = hidden_states.dtype

        work_dtype = torch.float32 if original_dtype == torch.float16 else original_dtype
        hidden_states = hidden_states.to(work_dtype)
        encoder_hidden_states = encoder_hidden_states.to(work_dtype)

        residual = hidden_states
        hidden_states_norm = self.cross_attn_layer_norm(hidden_states)

        query = self.cross_attn_in_proj(hidden_states_norm)
        key = self.key_proj(encoder_hidden_states)
        value = self.value_proj(encoder_hidden_states)

        cross_attn_output, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=encoder_attention_mask,
            need_weights=False,
        )
        cross_attn_output = self.cross_attn_out_proj(cross_attn_output)
        output = residual + self.dropout(cross_attn_output)

        return output.to(original_dtype)


# ── Wrapped Decoder Layer ──

class LlamaDecoderLayerWithOptionalCrossAttention(nn.Module):
    """Llama decoder layer with optional cross-attention."""
    def __init__(self, original_layer, cross_attn_module=None):
        super().__init__()
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.cross_attn_module = cross_attn_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # 1. Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # 2. Cross-Attention (if this layer has it)
        if self.cross_attn_module is not None and encoder_hidden_states is not None:
            hidden_states = self.cross_attn_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )

        # 3. MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ── Patch Llama Forward ──

def patch_llama_model_forward(model, use_gradient_checkpointing=False):
    original_forward = model.base_model.model.model.forward

    @wraps(original_forward)
    def patched_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        embed_tokens = model.base_model.model.model.embed_tokens
        layers = model.base_model.model.model.layers
        norm = model.base_model.model.model.norm
        rotary_emb = model.base_model.model.model.rotary_emb

        if inputs_embeds is None:
            inputs_embeds = embed_tokens(input_ids)

        hidden_states = inputs_embeds
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = 0
        if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = rotary_emb(hidden_states, position_ids)

        fill_val = -1e4 if dtype == torch.float16 else -1e9
        total_len = past_seen_tokens + seq_len

        if past_seen_tokens > 0 and seq_len == 1:
            causal_mask = torch.zeros(1, 1, 1, total_len, device=device, dtype=dtype)
        else:
            def create_full_causal_mask(seq_len, device, dtype, attn_mask=None):
                mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                mask = torch.where(mask,
                    torch.full([], fill_val, device=device, dtype=dtype),
                    torch.zeros([], device=device, dtype=dtype))
                mask = mask.unsqueeze(0).unsqueeze(0)
                if attn_mask is not None:
                    padding = attn_mask.unsqueeze(1).unsqueeze(2).to(dtype)
                    padding = (1.0 - padding) * fill_val
                    mask = mask + padding
                return mask

            causal_mask = create_full_causal_mask(seq_len, device, dtype, attention_mask)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        _do_checkpoint = (
            use_gradient_checkpointing
            and model.training
            and not use_cache
            and not output_attentions
        )

        for decoder_layer in layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if _do_checkpoint:
                def make_layer_fn(l):
                    def fn(h):
                        return l(
                            h,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            output_attentions=False,
                            use_cache=False,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                        )[0]
                    return fn
                hidden_states = torch.utils.checkpoint.checkpoint(
                    make_layer_fn(decoder_layer), hidden_states, use_reentrant=False
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    model.base_model.model.model.forward = patched_forward
    print("  ✓ Patched Llama with standard causal attention + encoder-decoder support")


# ── Complete Model ──

class SignLanguageTranslationModel(nn.Module):
    def __init__(self, cnn_encoder, cnn_projection, encoder, encoder_projection, decoder, tokenizer):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.cnn_projection = cnn_projection
        self.encoder = encoder
        self.encoder_projection = encoder_projection
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)

    def forward(self, frames, frame_padding_mask, token_ids, text_attention_mask, return_loss=True):
        """
        Args:
            frames: (B, T, 3, H, W) - video frames
            frame_padding_mask: (B, T) - True=padding, False=real frame
            token_ids: (B, L) - full sequence [BOS, ..., EOS]
            text_attention_mask: (B, L) - True=real, False=padding
        """
        # CNN feature extraction
        cnn_features = self.cnn_encoder(frames)  # (B, T, 2048)

        # Project to encoder dimension
        encoder_input = self.cnn_projection(cnn_features.to(next(self.cnn_projection.parameters()).dtype))  # (B, T, 768)

        # Encoder
        encoder_output = self.encoder(encoder_input, src_key_padding_mask=frame_padding_mask)
        encoder_hidden_states = self.encoder_projection(encoder_output)

        # Decoder (teacher forcing)
        decoder_input_ids = token_ids[:, :-1].contiguous()
        labels = token_ids[:, 1:].contiguous()
        decoder_attention_mask = text_attention_mask[:, :-1]

        encoder_hidden_states = encoder_hidden_states.to(next(self.decoder.parameters()).dtype)

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=frame_padding_mask,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits

        if return_loss:
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return {'loss': loss, 'logits': logits}

        return {'logits': logits}

    def _encode_frames(self, frames, frame_padding_mask):
        """Shared encoder path for both training and generation."""
        cnn_features = self.cnn_encoder(frames)
        encoder_input = self.cnn_projection(cnn_features.to(next(self.cnn_projection.parameters()).dtype))
        encoder_output = self.encoder(encoder_input, src_key_padding_mask=frame_padding_mask)
        encoder_hidden_states = self.encoder_projection(encoder_output)
        return encoder_hidden_states.to(next(self.decoder.parameters()).dtype), frame_padding_mask

    def _apply_repetition_penalty(self, logits, generated_ids, penalty):
        if penalty == 1.0:
            return logits
        score = torch.gather(logits, 1, generated_ids)
        score = torch.where(score > 0, score / penalty, score * penalty)
        logits = logits.scatter(1, generated_ids, score)
        return logits

    @torch.no_grad()
    def _beam_search_single(self, encoder_hidden_states, encoder_padding_mask,
                            device, max_new_tokens, beam_size, repetition_penalty, use_kv_cache=True):
        eos_id = self.tokenizer.eos_token_id
        bos_id = self.tokenizer.bos_token_id

        enc_hs = encoder_hidden_states.expand(beam_size, -1, -1)
        enc_mask = encoder_padding_mask.expand(beam_size, -1)

        generated = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)

        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = -1e9

        completed = []
        past_key_values = None

        for step in range(max_new_tokens):
            if use_kv_cache and past_key_values is not None:
                input_ids = generated[:, -1:]
            else:
                input_ids = generated

            outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_mask,
                use_cache=use_kv_cache,
                return_dict=True,
                past_key_values=past_key_values if use_kv_cache else None,
            )

            if use_kv_cache:
                past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)

            log_probs = torch.log_softmax(logits, dim=-1)
            vocab_size = log_probs.shape[-1]

            next_scores = beam_scores.unsqueeze(1) + log_probs
            flat_scores = next_scores.view(-1)

            num_candidates = min(2 * beam_size, flat_scores.shape[0])
            top_scores, top_flat_idx = flat_scores.topk(num_candidates)

            beam_idx = top_flat_idx // vocab_size
            token_idx = top_flat_idx % vocab_size

            new_beams = []
            new_scores_list = []
            new_beam_source_indices = []

            for i in range(len(top_scores)):
                if len(new_beams) >= beam_size:
                    break

                b = beam_idx[i].item()
                t = token_idx[i].item()
                s = top_scores[i].item()

                new_seq = torch.cat([generated[b], torch.tensor([t], device=device)])

                if t == eos_id:
                    completed.append((s / len(new_seq), new_seq))
                else:
                    new_beams.append(new_seq)
                    new_scores_list.append(s)
                    new_beam_source_indices.append(b)

            if len(new_beams) == 0:
                break

            while len(new_beams) < beam_size:
                new_beams.append(new_beams[-1].clone())
                new_scores_list.append(-1e9)
                new_beam_source_indices.append(new_beam_source_indices[-1])

            if use_kv_cache and past_key_values is not None:
                cache_reorder_idx = torch.tensor(
                    new_beam_source_indices[:beam_size], dtype=torch.long, device=device
                )
                past_key_values.reorder_cache(cache_reorder_idx)

            generated = torch.stack(new_beams[:beam_size])
            beam_scores = torch.tensor(new_scores_list[:beam_size], device=device)

            if len(completed) >= beam_size:
                break

        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]
        else:
            best = beam_scores.argmax().item()
            return generated[best]

    @torch.no_grad()
    def generate(self, frames, frame_padding_mask, max_new_tokens=50, temperature=1.0,
                 top_k=50, repetition_penalty=1.0, beam_size=1, use_kv_cache=True):
        """
        Generate text from video frames.

        Args:
            frames: (B, T, 3, H, W) - batched video frames
            frame_padding_mask: (B, T) - True=padding
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k sampling
            repetition_penalty: penalty factor for repeated tokens
            beam_size: number of beams (1 = greedy)
            use_kv_cache: whether to use KV-cache

        Returns:
            dict with 'generated_ids' (B, L) and 'generated_text' (list of B strings)
        """
        device = frames.device
        batch_size = frames.shape[0]

        # Encode frames
        encoder_hidden_states, encoder_padding_mask = self._encode_frames(frames, frame_padding_mask)

        # ── Beam Search Path ──
        if beam_size > 1:
            all_sequences = []
            for i in range(batch_size):
                best_seq = self._beam_search_single(
                    encoder_hidden_states[i:i+1],
                    encoder_padding_mask[i:i+1],
                    device, max_new_tokens, beam_size, repetition_penalty, use_kv_cache,
                )
                all_sequences.append(best_seq)

            generated_texts = [
                self.tokenizer.decode(seq, skip_special_tokens=True)
                for seq in all_sequences
            ]
            max_len = max(seq.shape[0] for seq in all_sequences)
            padded_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id,
                                   device=device, dtype=torch.long)
            for i, seq in enumerate(all_sequences):
                padded_ids[i, :seq.shape[0]] = seq

            return {'generated_ids': padded_ids, 'generated_text': generated_texts}

        # ── Greedy / Sampling Path ──
        generated_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, device=device, dtype=torch.long)
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        past_key_values = None

        for _ in range(max_new_tokens):
            if use_kv_cache and past_key_values is not None:
                input_ids = generated_ids[:, -1:]
            else:
                input_ids = generated_ids

            outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_padding_mask,
                use_cache=use_kv_cache,
                return_dict=True,
                past_key_values=past_key_values if use_kv_cache else None,
            )

            if use_kv_cache:
                past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, repetition_penalty
                )

            if temperature < 1e-5:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            active_sequences &= (next_token.squeeze(-1) != self.tokenizer.eos_token_id)

            if not active_sequences.any():
                break

        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        return {'generated_ids': generated_ids, 'generated_text': generated_texts}


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS: Metrics, Checkpointing, CSV Logging
# ═══════════════════════════════════════════════════════════════

def compute_bleu(references, hypotheses, max_n=4):
    if not references or not hypotheses:
        return 0.0

    bleu_result = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize='13a'
    )

    if max_n == 4:
        return bleu_result.score
    else:
        precisions = bleu_result.precisions[:max_n]
        bp = bleu_result.bp

        if any(p == 0 for p in precisions):
            return 0.0

        log_precision_sum = sum(math.log(p) for p in precisions)
        geo_mean = math.exp(log_precision_sum / max_n)

        return bp * geo_mean


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


class CheckpointManager:
    """Manages model checkpoints with a sliding window for periodic saves."""

    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.periodic_checkpoints = []
        self.best_path = self.checkpoint_dir / 'best_model.pt'

    def _atomic_save(self, state_dict, path):
        temp_fd, temp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix='.pt.tmp')
        try:
            os.close(temp_fd)
            torch.save(state_dict, temp_path)
            shutil.move(temp_path, path)
            return True
        except Exception as e:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise RuntimeError(f"Failed to save checkpoint to {path}: {e}")

    def save_periodic(self, state_dict, step):
        path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        try:
            self._atomic_save(state_dict, path)
            self.periodic_checkpoints.append(path)
            print(f"  💾 Saved periodic checkpoint: {path.name}")
        except RuntimeError as e:
            print(f"  ⚠️  Warning: {e}")
            return

        while len(self.periodic_checkpoints) > self.keep_last_n:
            old_path = self.periodic_checkpoints.pop(0)
            if old_path.exists() and old_path != self.best_path:
                old_path.unlink()
                print(f"  🗑️  Evicted old checkpoint: {old_path.name}")

    def save_best(self, state_dict):
        try:
            self._atomic_save(state_dict, self.best_path)
            print(f"  ⭐ Saved best model: {self.best_path.name}")
        except RuntimeError as e:
            print(f"  ⚠️  Warning: Failed to save best model: {e}")

    def _build_state_dict(self, model,
                          optimizer_new_weights, optimizer_cnn, optimizer_decoder,
                          scheduler_new_weights, scheduler_cnn, scheduler_decoder,
                          epoch, global_step, best_val_loss, evals_without_improvement, elapsed_sec):
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_new_weights_state_dict': optimizer_new_weights.state_dict(),
            'optimizer_cnn_state_dict': optimizer_cnn.state_dict(),
            'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
            'scheduler_new_weights_state_dict': scheduler_new_weights.state_dict(),
            'scheduler_cnn_state_dict': scheduler_cnn.state_dict(),
            'scheduler_decoder_state_dict': scheduler_decoder.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'evals_without_improvement': evals_without_improvement,
            'elapsed_sec': elapsed_sec,
            'rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if torch.cuda.is_available():
            state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        return state


def _remap_opt_state_for_bnb(state_dict):
    """Convert standard AdamW state keys to bitsandbytes AdamW8bit format."""
    new_state = {}
    for idx, param_state in state_dict['state'].items():
        s = dict(param_state)
        if 'exp_avg' in s:
            s['state1'] = s.pop('exp_avg')
        if 'exp_avg_sq' in s:
            s['state2'] = s.pop('exp_avg_sq')
        new_state[idx] = s
    return {'state': new_state, 'param_groups': state_dict['param_groups']}


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

@torch.no_grad()
def validate(model, val_loader, val_dataset, tokenizer, device,
             max_eval_batches, max_generate_samples, num_print_samples, val_gen_batch_size,
             val_beam_size=1, val_repetition_penalty=1.0, val_use_kv_cache=True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    # ── Part 1: Validation Loss + Token Accuracy (teacher-forced) ──
    loss_pbar = tqdm(total=max_eval_batches, desc="  Val loss", unit="batch",
                     leave=False, ncols=80, dynamic_ncols=False)
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_eval_batches:
            break

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(
                frames=batch['frames'].to(device, non_blocking=True),
                frame_padding_mask=batch['frame_padding_mask'].to(device, non_blocking=True),
                token_ids=batch['token_ids'].to(device, non_blocking=True),
                text_attention_mask=batch['text_attention_mask'].to(device, non_blocking=True),
                return_loss=True,
            )

        total_loss += output['loss'].item()
        num_batches += 1

        logits = output['logits']
        preds = logits.argmax(dim=-1)
        labels = batch['token_ids'][:, 1:].to(device, non_blocking=True)
        mask = labels != tokenizer.pad_token_id
        total_correct += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

        loss_pbar.update(1)
    loss_pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, MAX_PPL_CAP))
    token_acc = (total_correct / max(total_tokens, 1)) * 100

    # ── Part 2: Generate Text for BLEU / ROUGE-L ──
    references = []
    hypotheses = []
    sample_pairs = []

    num_samples = min(max_generate_samples, len(val_dataset))
    sample_indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(42))[:num_samples].tolist()

    gen_pbar = tqdm(total=num_samples, desc="  Val gen ", unit="sample",
                    leave=False, ncols=80, dynamic_ncols=False)
    for batch_start in range(0, num_samples, val_gen_batch_size):
        batch_end = min(batch_start + val_gen_batch_size, num_samples)
        batch_indices = sample_indices[batch_start:batch_end]
        batch_samples = [val_dataset[idx] for idx in batch_indices]

        # Extract and collate frames
        frames_list = [s['frames'] for s in batch_samples]
        token_ids_list = [s['token_ids'] for s in batch_samples]

        max_frames = max(f.shape[0] for f in frames_list)
        batch_size = len(batch_samples)
        _, C, H, W = frames_list[0].shape

        padded_frames = torch.zeros(batch_size, max_frames, C, H, W, dtype=torch.uint8)
        frame_padding_mask = torch.ones(batch_size, max_frames, dtype=torch.bool)

        for i, f in enumerate(frames_list):
            T = f.shape[0]
            padded_frames[i, :T] = f
            frame_padding_mask[i, :T] = False

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            gen_output = model.generate(
                frames=padded_frames.to(device, non_blocking=True),
                frame_padding_mask=frame_padding_mask.to(device, non_blocking=True),
                max_new_tokens=50,
                temperature=0.0,
                top_k=50,
                repetition_penalty=val_repetition_penalty,
                beam_size=val_beam_size,
                use_kv_cache=val_use_kv_cache,
            )

        batch_hypotheses = gen_output['generated_text']
        if isinstance(batch_hypotheses, str):
            batch_hypotheses = [batch_hypotheses]

        for i, (tokens, hyp_text) in enumerate(zip(token_ids_list, batch_hypotheses)):
            ref_text = tokenizer.decode(tokens[1:-1], skip_special_tokens=True).strip()
            hyp_text = hyp_text.strip()

            references.append(ref_text)
            hypotheses.append(hyp_text)

            if len(sample_pairs) < num_print_samples:
                sample_pairs.append((ref_text, hyp_text))

        gen_pbar.update(batch_end - batch_start)
    gen_pbar.close()

    bleu1 = compute_bleu(references, hypotheses, max_n=1)
    bleu2 = compute_bleu(references, hypotheses, max_n=2)
    bleu4 = compute_bleu(references, hypotheses, max_n=4)
    rouge_l = compute_rouge_l(references, hypotheses)

    model.train()

    return {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
        'token_acc': token_acc,
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu4': bleu4,
        'rouge_l': rouge_l,
        'sample_pairs': sample_pairs,
        'all_pairs': list(zip(references, hypotheses)),
        'num_eval_batches': num_batches,
        'num_gen_samples': num_samples,
    }


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader, val_dataset, tokenizer,
          optimizer_new_weights, optimizer_cnn, optimizer_decoder,
          scheduler_new_weights, scheduler_cnn, scheduler_decoder,
          device, train_config, ckpt_manager, train_csv_logger, val_csv_logger, gen_samples_csv_logger, tb_writer,
          start_epoch=1, start_global_step=0, best_val_loss=float('inf'), start_evals_without_improvement=0, start_elapsed_sec=0.0):
    """Full training loop with 3-stage phased curriculum."""

    # Unpack config
    num_epochs = train_config['num_epochs']
    grad_accum_steps = train_config['grad_accum_steps']
    max_grad_norm = train_config['max_grad_norm']
    log_every = train_config['log_every_steps']
    save_every = train_config['save_every_steps']
    eval_every = train_config['eval_every_steps']
    eval_every_warmup = train_config.get('eval_every_steps_warmup', eval_every)
    eval_warmup_threshold = train_config.get('eval_warmup_threshold', 0)
    max_eval_batches = train_config['max_eval_batches']
    max_generate_samples = train_config['max_generate_samples']
    num_print_samples = train_config['num_print_samples']
    val_gen_batch_size = train_config['val_gen_batch_size']
    val_beam_size = train_config['val_beam_size']
    val_repetition_penalty = train_config['val_repetition_penalty']
    val_use_kv_cache = train_config['val_use_kv_cache']
    patience = train_config['early_stopping_patience']

    cnn_freeze_steps = train_config.get('cnn_freeze_steps', 1000)
    decoder_freeze_steps = train_config.get('decoder_freeze_steps', 1500)

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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()

    input("Press Enter to start training...")
    print("\n" + "=" * 60)
    print("🚀 TRAINING STARTED")
    print("=" * 60 + "\n")

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)

    steps_done_in_epoch = start_global_step % optimizer_steps_per_epoch
    if steps_done_in_epoch == 0 and start_global_step > 0:
        start_epoch += 1
        print(f"  ⏭️ Checkpoint was at exact end of Epoch {start_epoch - 1}. Advancing to Epoch {start_epoch}.")

    # ── Phased Training Curriculum Setup ──
    # Collect parameter references for each group
    cnn_params_for_freeze = [p for group in optimizer_cnn.param_groups for p in group['params']]
    decoder_params_for_freeze = [p for group in optimizer_decoder.param_groups for p in group['params']]

    # Determine initial freeze state (handles resume correctly)
    cnn_is_frozen = (start_global_step < cnn_freeze_steps)
    decoder_is_frozen = (start_global_step < decoder_freeze_steps)

    # Apply initial freeze states
    if cnn_is_frozen:
        for p in cnn_params_for_freeze:
            p.requires_grad = False
        remaining = cnn_freeze_steps - start_global_step
        print(f"\n  ❄️  CNN FROZEN | Will unfreeze at step {cnn_freeze_steps} ({remaining} steps remaining)")
    else:
        print(f"\n  ✅ CNN freeze phase already completed (resumed at step {start_global_step} >= {cnn_freeze_steps})")

    if decoder_is_frozen:
        for p in decoder_params_for_freeze:
            p.requires_grad = False
        remaining = decoder_freeze_steps - start_global_step
        print(f"  ❄️  Decoder FROZEN | Will unfreeze at step {decoder_freeze_steps} ({remaining} steps remaining)")
    else:
        print(f"  ✅ Decoder freeze phase already completed (resumed at step {start_global_step} >= {decoder_freeze_steps})")

    print(f"  ℹ️  Stage 1 (0-{cnn_freeze_steps}): MLP + Encoder + Cross-attn only")
    print(f"  Stage 2 ({cnn_freeze_steps}-{decoder_freeze_steps}): + CNN layer3/layer4")
    print(f"  Stage 3 ({decoder_freeze_steps}+): + Decoder (full training)")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_microbatches = 0

        optimizer_new_weights.zero_grad()
        optimizer_cnn.zero_grad()
        optimizer_decoder.zero_grad()

        # How many micro-batches to skip if resuming mid-epoch
        micro_steps_already_processed = 0
        if epoch == start_epoch and start_global_step > 0 and steps_done_in_epoch > 0:
            micro_steps_already_processed = steps_done_in_epoch * grad_accum_steps
            if micro_steps_already_processed > 0:
                print(f"  ⏭️ Resuming mid-epoch. Fast-forwarding {micro_steps_already_processed} micro-batches...")

        for micro_step, batch in enumerate(train_loader):
            if epoch == start_epoch and micro_step < micro_steps_already_processed:
                continue

            # ── Forward pass with mixed precision ──
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(
                    frames=batch['frames'].to(device, non_blocking=True),
                    frame_padding_mask=batch['frame_padding_mask'].to(device, non_blocking=True),
                    token_ids=batch['token_ids'].to(device, non_blocking=True),
                    text_attention_mask=batch['text_attention_mask'].to(device, non_blocking=True),
                    return_loss=True,
                )

            is_accum_step = (micro_step + 1) % grad_accum_steps == 0
            is_last_step = (micro_step + 1) == len(train_loader)

            if is_last_step and not is_accum_step:
                actual_accum_steps = (micro_step + 1) % grad_accum_steps
            else:
                actual_accum_steps = grad_accum_steps

            loss = output['loss'] / actual_accum_steps
            loss.backward()

            epoch_loss += output['loss'].item()
            epoch_microbatches += 1

            if is_accum_step or is_last_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                ).item()

                # Always step new_weights optimizer (active from step 0)
                optimizer_new_weights.step()
                scheduler_new_weights.step()

                # CNN optimizer only steps when unfrozen
                if not cnn_is_frozen:
                    optimizer_cnn.step()
                    scheduler_cnn.step()

                # Decoder optimizer only steps when unfrozen
                if not decoder_is_frozen:
                    optimizer_decoder.step()
                    scheduler_decoder.step()

                optimizer_new_weights.zero_grad()
                optimizer_cnn.zero_grad()
                optimizer_decoder.zero_grad()

                global_step += 1

                # ── Unfreeze CNN ──
                if cnn_is_frozen and global_step >= cnn_freeze_steps:
                    cnn_is_frozen = False
                    for p in cnn_params_for_freeze:
                        p.requires_grad = True
                    optimizer_cnn.zero_grad()
                    print(f"\n{'=' * 60}")
                    print(f"  🔓 CNN UNFROZEN at step {global_step} — CNN fine-tuning begins")
                    print(f"  CNN LR schedule starting its warmup from this point.")
                    print(f"{'=' * 60}\n")

                # ── Unfreeze decoder ──
                if decoder_is_frozen and global_step >= decoder_freeze_steps:
                    decoder_is_frozen = False
                    for p in decoder_params_for_freeze:
                        p.requires_grad = True
                    optimizer_decoder.zero_grad()
                    print(f"\n{'=' * 60}")
                    print(f"  🔓 Decoder UNFROZEN at step {global_step} — full training begins now")
                    print(f"  Decoder LR schedule starting its warmup from this point.")
                    print(f"{'=' * 60}\n")

                step_losses.append(output['loss'].item())
                step_grad_norms.append(grad_norm)

                # ── Logging ──
                if global_step % log_every == 0:
                    avg_loss = sum(step_losses) / len(step_losses)
                    avg_grad_norm = sum(step_grad_norms) / len(step_grad_norms)
                    new_weights_lr = scheduler_new_weights.get_last_lr()[0]
                    cnn_lr = scheduler_cnn.get_last_lr()[0] if not cnn_is_frozen else 0.0
                    decoder_lr = scheduler_decoder.get_last_lr()[0] if not decoder_is_frozen else 0.0
                    elapsed = time.time() - training_start
                    log_elapsed = time.time() - log_step_start
                    speed = len(step_losses) / max(log_elapsed, 1e-6)
                    cuda_mem, cuda_peak = get_cuda_mem()

                    eta_seconds = (elapsed / global_step) * (total_optimizer_steps - global_step)

                    train_ppl = math.exp(min(avg_loss, MAX_PPL_CAP))

                    # Phase tag
                    if decoder_is_frozen and cnn_is_frozen:
                        phase_tag = f" | PHASE 1: MLP+Enc+XAttn only (CNN unfreeze in {cnn_freeze_steps - global_step})"
                    elif decoder_is_frozen:
                        phase_tag = f" | PHASE 2: +CNN (Dec unfreeze in {decoder_freeze_steps - global_step})"
                    else:
                        phase_tag = " | PHASE 3: Full training"

                    print("=" * 190)
                    print(
                        f"[Step {global_step:>6d}/{total_optimizer_steps}] "
                        f"[Epoch {epoch:>2d}/{num_epochs}] "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"Train PPL: {train_ppl:.3f} | "
                        f"LR_MLP+encoder+cross_attn: {new_weights_lr:.4e} | "
                        f"LR_cnn: {cnn_lr:.4e} | "
                        f"LR_decoder: {decoder_lr:.4e} | "
                        f"Grad Norm: {avg_grad_norm:.3f} | "
                        f"Speed: {speed:.3f} steps/s | "
                        f"VRAM: {cuda_mem:.3f}/{cuda_peak:.3f} GB | "
                        f"Elapsed: {format_time(elapsed)} | "
                        f"ETA: {format_time(eta_seconds)}"
                        f"{phase_tag}"
                    )

                    train_csv_logger.log({
                        'global_step': global_step,
                        'epoch': epoch,
                        'train_loss': f"{avg_loss:.6f}",
                        'train_ppl': f"{train_ppl:.4f}",
                        'lr_new_weights': f"{new_weights_lr:.3e}",
                        'lr_cnn': f"{cnn_lr:.3e}",
                        'lr_decoder': f"{decoder_lr:.3e}",
                        'grad_norm': f"{avg_grad_norm:.4f}",
                        'cuda_mem_gb': f"{cuda_mem:.2f}",
                        'cuda_peak_gb': f"{cuda_peak:.2f}",
                        'steps_per_sec': f"{speed:.2f}",
                        'elapsed_sec': f"{elapsed:.1f}",
                    })

                    tb_writer.add_scalar('Loss/Train', avg_loss, global_step)
                    tb_writer.add_scalar('Perplexity/Train', train_ppl, global_step)
                    tb_writer.add_scalar('GradNorm/Train', avg_grad_norm, global_step)
                    tb_writer.add_scalar('Training/CNNFrozen', 1.0 if cnn_is_frozen else 0.0, global_step)
                    tb_writer.add_scalar('Training/DecoderFrozen', 1.0 if decoder_is_frozen else 0.0, global_step)
                    tb_writer.add_scalars('LearningRates', {
                        'NewWeights': new_weights_lr,
                        'CNN': cnn_lr,
                        'Decoder': decoder_lr,
                    }, global_step)

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

                    val_metrics = validate(
                        model, val_loader, val_dataset, tokenizer, device,
                        max_eval_batches, max_generate_samples, num_print_samples, val_gen_batch_size,
                        val_beam_size, val_repetition_penalty, val_use_kv_cache,
                    )

                    cuda_mem, cuda_peak = get_cuda_mem()
                    elapsed = time.time() - training_start

                    try:
                        val_loss_str = f"{val_metrics['val_loss']:.4f}"
                        val_ppl_str = f"{val_metrics['val_ppl']:.3f}"
                        val_acc_str = f"{val_metrics['token_acc']:.4f}"
                        print(f"  Val Loss: {val_loss_str} | Val PPL: {val_ppl_str} | Token Acc: {val_acc_str}%")
                    except Exception as e:
                        print(f"  [Error formatting validation metrics: {e}]")

                    print(f"  BLEU-1: {val_metrics['bleu1']:.4f} | "
                          f"BLEU-2: {val_metrics['bleu2']:.4f} | "
                          f"BLEU-4: {val_metrics['bleu4']:.4f} | "
                          f"ROUGE-L: {val_metrics['rouge_l']:.3f}%")
                    print(f"  (Evaluated on {val_metrics['num_eval_batches']} batches, "
                          f"generated {val_metrics['num_gen_samples']} samples)")

                    if val_metrics['sample_pairs']:
                        print(f"\n  Sample Generations:")
                        for i, (ref, hyp) in enumerate(val_metrics['sample_pairs'], 1):
                            print(f"    [{i}] REF: \"{ref}\"")
                            print(f"        HYP: \"{hyp}\"")

                    val_csv_logger.log({
                        'global_step': global_step,
                        'epoch': epoch,
                        'val_loss': f"{val_metrics['val_loss']:.6f}",
                        'val_ppl': f"{val_metrics['val_ppl']:.4f}",
                        'bleu1': f"{val_metrics['bleu1']:.4f}",
                        'bleu2': f"{val_metrics['bleu2']:.4f}",
                        'bleu4': f"{val_metrics['bleu4']:.4f}",
                        'rouge_l': f"{val_metrics['rouge_l']:.4f}",
                        'token_acc': f"{val_metrics['token_acc']:.4f}",
                        'cuda_mem_gb': f"{cuda_mem:.2f}",
                        'cuda_peak_gb': f"{cuda_peak:.2f}",
                        'elapsed_sec': f"{elapsed:.1f}",
                    })

                    tb_writer.add_scalar('Loss/Validation', val_metrics['val_loss'], global_step)
                    tb_writer.add_scalar('Perplexity/Validation', val_metrics['val_ppl'], global_step)
                    tb_writer.add_scalar('Metrics/TokenAccuracy', val_metrics['token_acc'], global_step)
                    tb_writer.add_scalar('Metrics/ROUGE-L', val_metrics['rouge_l'], global_step)
                    tb_writer.add_scalars('Metrics/BLEU', {
                        'BLEU-1': val_metrics['bleu1'],
                        'BLEU-2': val_metrics['bleu2'],
                        'BLEU-4': val_metrics['bleu4']
                    }, global_step)

                    for ref_text, hyp_text in val_metrics['all_pairs']:
                        gen_samples_csv_logger.log({
                            'global_step': global_step,
                            'epoch': epoch,
                            'generated': hyp_text,
                            'reference': ref_text,
                            'elapsed_sec': f"{elapsed:.1f}",
                        })

                    val_loss = val_metrics['val_loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        evals_without_improvement = 0
                        state = ckpt_manager._build_state_dict(
                            model,
                            optimizer_new_weights, optimizer_cnn, optimizer_decoder,
                            scheduler_new_weights, scheduler_cnn, scheduler_decoder,
                            epoch, global_step, best_val_loss, evals_without_improvement, time.time() - training_start
                        )
                        ckpt_manager.save_best(state)
                        del state
                        torch.cuda.empty_cache()
                        print(f"\n  ⭐ New best val loss: {best_val_loss:.4f}")
                    else:
                        evals_without_improvement += 1
                        print(f"\n  Val loss did not improve. "
                              f"Best: {best_val_loss:.4f} | "
                              f"Early stop: {evals_without_improvement}/{patience}")

                    print(f"{'─' * 60}\n")

                    model.train()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if evals_without_improvement >= patience:
                        print("=" * 60)
                        print(f"🛑 EARLY STOPPING triggered at step {global_step}, epoch {epoch}")
                        print(f"   Best val loss: {best_val_loss:.4f}")
                        print("=" * 60)

                        total_time = time.time() - training_start
                        print(f"\n  Total training time: {format_time(total_time)}")
                        cuda_mem, cuda_peak = get_cuda_mem()
                        print(f"  Peak VRAM usage: {cuda_peak:.2f} GB")
                        tb_writer.close()
                        return best_val_loss

                # ── Periodic checkpoint ──
                if global_step % save_every == 0:
                    state = ckpt_manager._build_state_dict(
                        model,
                        optimizer_new_weights, optimizer_cnn, optimizer_decoder,
                        scheduler_new_weights, scheduler_cnn, scheduler_decoder,
                        epoch, global_step, best_val_loss, evals_without_improvement, time.time() - training_start
                    )
                    ckpt_manager.save_periodic(state, global_step)
                    del state
                    torch.cuda.empty_cache()

        # ── Epoch summary ──
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / max(epoch_microbatches, 1)
        cuda_mem, cuda_peak = get_cuda_mem()

        print(f"\n{'=' * 60}")
        print(f"  ✅ Epoch {epoch}/{num_epochs} Complete")
        print(f"  Avg Train Loss: {epoch_avg_loss:.4f} | "
              f"Epoch Time: {format_time(epoch_time)}")
        print(f"  VRAM: {cuda_mem:.1f}/{cuda_peak:.1f} GB | "
              f"Total Time: {format_time(time.time() - training_start)}")
        print(f"{'=' * 60}\n")

    # ── Training complete ──
    total_time = time.time() - training_start
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total epochs: {num_epochs}")
    print(f"  Total optimizer steps: {global_step}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total time: {format_time(total_time)}")
    cuda_mem, cuda_peak = get_cuda_mem()
    print(f"  Peak VRAM: {cuda_peak:.2f} GB")
    print("=" * 60)

    return best_val_loss


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point for training."""
    global device

    # ── Open timestamped log file (tee: terminal + file) ──
    _run_dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _log_dir = Path("saved_metrics/train_config")
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _log_dir / f"training_config_{_run_dt}.txt"

    class _Tee:
        """Write to both the real stdout and a file simultaneously."""
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
    _tee = _Tee(_log_file)
    sys.stdout = _tee

    # ── Print Configuration Summary ──
    _W = 72
    _SEP = "=" * _W
    effective_batch = TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['grad_accum_steps']
    print("\n" + _SEP)
    print("  CONFIGURATION SUMMARY")
    print(_SEP)

    print("\n  [ DATA ]")
    print(f"    Train CSV              : {CONFIG['data_train_csv']}")
    print(f"    Val CSV                : {CONFIG['data_val_csv']}")

    print("\n  [ CNN BACKBONE  –  ResNet50 ]")
    print(f"    Frame size             : {CONFIG['cnn_frame_size']} × {CONFIG['cnn_frame_size']}  (letterbox, no crop)")
    print(f"    Frozen layers          : {', '.join(CONFIG['cnn_freeze_layers'])}")
    print(f"    Trainable layers       : layer3, layer4")
    print(f"    Chunk size             : {CONFIG['cnn_chunk_size']} frames per CUDA call")
    print(f"    ResNet output dim      : {CONFIG['resnet_output_dim']}")
    print(f"    Grad checkpointing     : {TRAIN_CONFIG.get('use_gradient_checkpointing_cnn', True)}")

    print("\n  [ CNN PROJECTION MLP ]")
    print(f"    Architecture           : Linear({CONFIG['resnet_output_dim']} → {CONFIG['cnn_projection_dim']}) → GELU → Dropout → LayerNorm")
    print(f"    Dropout                : {CONFIG['cnn_projection_dropout']}")

    print("\n  [ TRANSFORMER ENCODER ]")
    print(f"    d_model / heads        : {CONFIG['encoder_d_model']} / {CONFIG['encoder_num_heads']}")
    print(f"    Layers                 : {CONFIG['encoder_num_layers']}")
    print(f"    Feedforward dim        : {CONFIG['encoder_feedforward_dim']}")
    print(f"    Dropout                : {CONFIG['encoder_dropout']}")
    print(f"    Grad checkpointing     : {TRAIN_CONFIG.get('use_gradient_checkpointing_encoder', False)}")

    print("\n  [ CROSS-ATTENTION ]")
    print(f"    Decoder layers         : {CONFIG['cross_attn_layers']}  ({len(CONFIG['cross_attn_layers'])}/{ CONFIG['decoder_num_layers']})")
    print(f"    Bottleneck dim / heads : {CONFIG['bottleneck_dim']} / {CONFIG['cross_attn_num_heads']}")
    print(f"    Weight sharing pairs   : {CONFIG['weight_sharing_pairs'] or 'none'}")
    print(f"    Dropout                : {CONFIG['cross_attn_dropout']}")

    print("\n  [ DECODER  –  Llama 3.2-1B  +  LoRA ]")
    print(f"    Model                  : {CONFIG['decoder_model_name']}")
    print(f"    Hidden size / layers   : {CONFIG['decoder_hidden_size']} / {CONFIG['decoder_num_layers']}")
    print(f"    LoRA rank / alpha      : {CONFIG['lora_r']} / {CONFIG['lora_alpha']}  (DoRA + RsLoRA)")
    print(f"    LoRA targets           : {', '.join(CONFIG['lora_target_modules'])}")
    print(f"    LoRA dropout           : {CONFIG['lora_dropout']}")
    print(f"    Grad checkpointing     : {TRAIN_CONFIG.get('use_gradient_checkpointing_decoder', False)}")

    print("\n  [ SEQUENCE LENGTHS ]")
    print(f"    Max video frames       : {CONFIG['max_video_frames']}  (@ 20 FPS ≈ {CONFIG['max_video_frames']/20:.0f}s)")
    print(f"    Max text tokens        : {CONFIG['max_text_tokens']}  (excl. BOS/EOS)")

    print("\n  [ TRAINING ]")
    print(f"    Epochs                 : {TRAIN_CONFIG['num_epochs']}")
    print(f"    Batch size             : {TRAIN_CONFIG['batch_size']}  (effective: {effective_batch} with {TRAIN_CONFIG['grad_accum_steps']}× accum)")
    print(f"    Mixed precision        : bfloat16")
    print(f"    Grad clip max_norm     : {TRAIN_CONFIG['max_grad_norm']}")
    print(f"    Weight decay           : {TRAIN_CONFIG['weight_decay']}")
    print(f"    Adam betas             : {TRAIN_CONFIG['adam_betas']}")
    print(f"    8-bit AdamW            : {TRAIN_CONFIG.get('use_8bit_adam', False)}")
    print(f"    SDPA attention         : {TRAIN_CONFIG.get('use_sdpa', True)}")
    print(f"    Bucket batching        : {TRAIN_CONFIG.get('use_bucket_batching', True)}")
    print(f"    DataLoader workers     : {TRAIN_CONFIG['train_num_workers']} train / {TRAIN_CONFIG['val_num_workers']} val")

    print("\n  [ LEARNING RATES ]")
    print(f"    New weights            : {TRAIN_CONFIG['encoder_lr']:.2e} → {TRAIN_CONFIG['encoder_min_lr']:.2e}  ({TRAIN_CONFIG['warmup_steps']} warmup steps, from step 0)")
    print(f"    CNN  (layer3/layer4)   : {TRAIN_CONFIG['cnn_lr']:.2e} → {TRAIN_CONFIG['cnn_min_lr']:.2e}  ({TRAIN_CONFIG.get('cnn_warmup_steps',100)} warmup steps, from step {TRAIN_CONFIG['cnn_freeze_steps']})")
    print(f"    Decoder (LoRA+norms)   : {TRAIN_CONFIG['decoder_lr']:.2e} → {TRAIN_CONFIG['decoder_min_lr']:.2e}  ({TRAIN_CONFIG.get('decoder_warmup_steps',100)} warmup steps, from step {TRAIN_CONFIG['decoder_freeze_steps']})")

    print("\n  [ PHASED TRAINING CURRICULUM ]")
    print(f"    Stage 1  steps 0 → {TRAIN_CONFIG['cnn_freeze_steps']}        : MLP + Encoder + Cross-Attn only")
    print(f"    Stage 2  steps {TRAIN_CONFIG['cnn_freeze_steps']} → {TRAIN_CONFIG['decoder_freeze_steps']}    : + CNN (layer3/layer4)")
    print(f"    Stage 3  steps {TRAIN_CONFIG['decoder_freeze_steps']} → end      : + Decoder (LoRA + norms)")

    print("\n  [ EVALUATION & CHECKPOINTING ]")
    print(f"    Eval every             : {TRAIN_CONFIG['eval_every_steps']} steps  (warmup: {TRAIN_CONFIG['eval_every_steps_warmup']} until step {TRAIN_CONFIG['eval_warmup_threshold']})")
    print(f"    Max eval batches       : {TRAIN_CONFIG['max_eval_batches']}")
    print(f"    Max generation samples : {TRAIN_CONFIG['max_generate_samples']}")
    print(f"    Beam size              : {TRAIN_CONFIG['val_beam_size']}  |  Repetition penalty: {TRAIN_CONFIG['val_repetition_penalty']}")
    print(f"    Save every             : {TRAIN_CONFIG['save_every_steps']} steps  |  Keep last: {TRAIN_CONFIG['keep_last_n_checkpoints']}")
    print(f"    Early stopping pat.    : {TRAIN_CONFIG['early_stopping_patience']} evals")
    print(f"    Checkpoint dir         : {TRAIN_CONFIG['checkpoint_dir'].resolve()}")

    print("\n  [ DATA AUGMENTATION (training only) ]")
    print(f"    Horizontal flip        : DISABLED  (ASL handedness is semantically meaningful)")
    print(f"    Color jitter           : {'ON' if CONFIG['aug_color_jitter'] else 'OFF'}  (prob={CONFIG['aug_color_jitter_prob']}, b={CONFIG['aug_color_jitter_brightness']}, c={CONFIG['aug_color_jitter_contrast']}, s={CONFIG['aug_color_jitter_saturation']}, h={CONFIG['aug_color_jitter_hue']})")
    print(f"    Random grayscale       : {'ON' if CONFIG['aug_random_grayscale'] else 'OFF'}  (prob={CONFIG['aug_random_grayscale_prob']})")
    print(f"    Gaussian blur          : {'ON' if CONFIG['aug_gaussian_blur'] else 'OFF'}  (prob={CONFIG['aug_gaussian_blur_prob']}, kernel={CONFIG['aug_gaussian_blur_kernel']})")
    print(f"    Temporal jitter        : {'ON' if CONFIG['aug_temporal_jitter'] else 'OFF'}  (prob={CONFIG['aug_temporal_jitter_prob']}, ±{CONFIG['aug_temporal_jitter_range']} frames)")
    print("\n" + _SEP + "\n")

    # ── Load Data Manifests ──
    TRAIN_CSV = CONFIG['data_train_csv']
    df = pd.read_csv(TRAIN_CSV, sep=CONFIG['csv_sep'])
    print(f'📂 Total training examples: {len(df)}')

    # Build manifest: list of dicts with file_path pointing to .mp4 videos
    manifest = []
    missing = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Building train manifest'):
        sentence_name = row['vid']
        text = row['text']
        duration = row['duration_sec']
        file_path = row['file_path']

        if Path(file_path).exists():
            manifest.append({
                'sentence_name': sentence_name,
                'text': text,
                'duration': duration,
                'file_path': str(file_path),
            })
        else:
            missing.append(sentence_name)

    print(f'\n📋 Train manifest:')
    print(f'  Total CSV rows: {len(df)}')
    print(f'  Found videos: {len(manifest)}')
    print(f'  Missing videos: {len(missing)}')
    print(f'  Success rate: {len(manifest)/len(df)*100:.1f}%')

    if len(missing) > 0:
        print(f'\nSample missing files (first 5):')
        for name in missing[:5]:
            print(f'  {name}')

    del df, missing  # free pandas DataFrame + missing list — no longer needed

    VAL_CSV = CONFIG['data_val_csv']
    df_val = pd.read_csv(VAL_CSV, sep=CONFIG['csv_sep'])

    val_manifest = []
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc='Building val manifest'):
        sentence_name = row['vid']
        text = row['text']
        duration = row['duration_sec']
        file_path = row['file_path']

        if Path(file_path).exists():
            val_manifest.append({
                'sentence_name': sentence_name,
                'text': text,
                'duration': duration,
                'file_path': str(file_path),
            })

    print(f'📋 Val manifest: {len(val_manifest)} samples')
    del df_val  # free pandas DataFrame — no longer needed

    # ── Load Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['decoder_model_name'])
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    print("🔤 Tokenizer loaded:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  BOS: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  EOS: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  PAD: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # ── Create Datasets ──
    train_dataset = SignLanguageDataset(manifest, tokenizer, train=True, augment_config=CONFIG)
    val_dataset = SignLanguageDataset(val_manifest, tokenizer, train=False)
    print(f'📦 Train dataset: {len(train_dataset)} samples')
    print(f'📦 Val dataset: {len(val_dataset)} samples')
    print(f'Max video frames: {MAX_VIDEO_FRAMES}')
    print(f'Max text tokens: {MAX_TEXT_TOKENS} (excluding BOS/EOS)')

    # ── Create DataLoaders ──
    collate_fn_with_tokenizer = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_workers = TRAIN_CONFIG['train_num_workers']
    val_workers = TRAIN_CONFIG['val_num_workers']
    train_prefetch_factor = TRAIN_CONFIG['train_prefetch_factor']
    val_prefetch_factor = TRAIN_CONFIG['val_prefetch_factor']

    if TRAIN_CONFIG.get('use_bucket_batching', True):
        train_sampler = BucketBatchSampler(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
        val_sampler = BucketBatchSampler(val_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn_with_tokenizer,
            num_workers=train_workers,
            pin_memory=True,
            prefetch_factor=train_prefetch_factor if train_workers > 0 else None,
            persistent_workers=train_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_fn_with_tokenizer,
            num_workers=val_workers,
            pin_memory=True,
            prefetch_factor=val_prefetch_factor if val_workers > 0 else None,
            persistent_workers=val_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=collate_fn_with_tokenizer,
            num_workers=train_workers,
            pin_memory=True,
            prefetch_factor=train_prefetch_factor if train_workers > 0 else None,
            persistent_workers=train_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            collate_fn=collate_fn_with_tokenizer,
            num_workers=val_workers,
            pin_memory=True,
            prefetch_factor=val_prefetch_factor if val_workers > 0 else None,
            persistent_workers=val_workers > 0,
        )

    print(f"\n⚙️  DataLoaders:")
    print(f"  Train workers: {train_workers}")
    print(f"  Val workers: {val_workers}")

    # ── Load Pretrained Decoder ──
    model_name = CONFIG['decoder_model_name']
    _attn_impl = "sdpa" if TRAIN_CONFIG.get('use_sdpa', True) else "eager"
    decoder = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device, attn_implementation=_attn_impl)

    gc.collect()  # reclaim intermediate objects created during model loading
    print(f"\n🧠 Loaded: {model_name}")
    if decoder.config.vocab_size != len(tokenizer):
        print(f"  📐 Resizing embeddings from {decoder.config.vocab_size} to {len(tokenizer)}")
        decoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # ── Apply LoRA ──
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['lora_target_modules'],
        modules_to_save=CONFIG['lora_modules_to_save'],
        lora_dropout=CONFIG['lora_dropout'],
        bias=CONFIG['lora_bias'],
        task_type="CAUSAL_LM",
        use_rslora=True,
        use_dora=True,
    )
    decoder = get_peft_model(decoder, lora_config)

    # Enable norm layers
    norm_params_count = 0
    for name, param in decoder.named_parameters():
        if 'norm' in name:
            param.requires_grad = True
            norm_params_count += param.numel()

    print(f"  ✓ LoRA applied (r={lora_config.r}, alpha={lora_config.lora_alpha})")
    print(f"  Norm layers trainable: {norm_params_count / 1e6:.2f}M params")
    decoder.print_trainable_parameters()

    # ── Create CNN Encoder ──
    cnn_encoder = ResNetFrameEncoder(
        freeze_layers=CONFIG['cnn_freeze_layers'],
        use_gradient_checkpointing=TRAIN_CONFIG.get('use_gradient_checkpointing_cnn', True),
        chunk_size=CONFIG['cnn_chunk_size'],
    ).to(torch.bfloat16)

    cnn_trainable = sum(p.numel() for p in cnn_encoder.parameters() if p.requires_grad)
    cnn_frozen = sum(p.numel() for p in cnn_encoder.parameters() if not p.requires_grad)
    print(f"\n🖼️  ResNet50 Frame Encoder:")
    print(f"  Frozen: {cnn_frozen / 1e6:.2f}M | Trainable: {cnn_trainable / 1e6:.2f}M")
    print(f"  Frozen layers: {CONFIG['cnn_freeze_layers']}")
    print(f"  Chunk size: {CONFIG['cnn_chunk_size']} frames")
    print(f"  Grad checkpointing: {TRAIN_CONFIG.get('use_gradient_checkpointing_cnn', True)}")

    # ── Create CNN Projection ──
    cnn_projection = CNNProjection(
        input_dim=CONFIG['resnet_output_dim'],
        d_model=CONFIG['cnn_projection_dim'],
        dropout=CONFIG['cnn_projection_dropout'],
    ).to(torch.bfloat16)

    proj_params = sum(p.numel() for p in cnn_projection.parameters())
    print(f"\n🔗 CNN Projection MLP:")
    print(f"  Architecture: Linear({CONFIG['resnet_output_dim']}→{CONFIG['cnn_projection_dim']}) → GELU → Dropout → LayerNorm")
    print(f"  Parameters: {proj_params / 1e6:.2f}M")

    # ── Create Encoder ──
    encoder = TransformerEncoder(
        d_model=CONFIG['encoder_d_model'],
        num_heads=CONFIG['encoder_num_heads'],
        num_layers=CONFIG['encoder_num_layers'],
        feedforward_dim=CONFIG['encoder_feedforward_dim'],
        dropout=CONFIG['encoder_dropout']
    ).to(torch.bfloat16)
    encoder.use_gradient_checkpointing = TRAIN_CONFIG.get('use_gradient_checkpointing_encoder', False)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n⚡ Transformer Encoder:")
    print(f"  d_model: {CONFIG['encoder_d_model']}, heads: {CONFIG['encoder_num_heads']}, layers: {CONFIG['encoder_num_layers']}")
    print(f"  feedforward_dim: {CONFIG['encoder_feedforward_dim']}")
    print(f"  Parameters: {encoder_params / 1e6:.2f}M")

    # ── Create Cross-Attention Modules ──
    ENCODER_DIM = CONFIG['encoder_d_model']
    BOTTLENECK_DIM = CONFIG['bottleneck_dim']
    NUM_HEADS = CONFIG['cross_attn_num_heads']
    TOTAL_LAYERS = CONFIG['decoder_num_layers']

    layers_with_cross_attn = CONFIG['cross_attn_layers']
    weight_sharing_pairs = CONFIG['weight_sharing_pairs']

    cross_attn_modules = {}
    shared_modules = {}

    for layer_a, layer_b in weight_sharing_pairs:
        if layer_a in layers_with_cross_attn and layer_b in layers_with_cross_attn:
            shared_module = CrossAttentionModule(
                hidden_size=CONFIG['decoder_hidden_size'],
                encoder_dim=ENCODER_DIM,
                bottleneck_dim=BOTTLENECK_DIM,
                num_heads=NUM_HEADS,
                dropout=CONFIG['cross_attn_dropout']
            )
            shared_modules[layer_a] = shared_module
            shared_modules[layer_b] = shared_module

    for layer_idx in layers_with_cross_attn:
        if layer_idx not in shared_modules:
            cross_attn_modules[layer_idx] = CrossAttentionModule(
                hidden_size=CONFIG['decoder_hidden_size'],
                encoder_dim=ENCODER_DIM,
                bottleneck_dim=BOTTLENECK_DIM,
                num_heads=NUM_HEADS,
                dropout=CONFIG['cross_attn_dropout']
            )
        else:
            cross_attn_modules[layer_idx] = shared_modules[layer_idx]

    for module in {id(m): m for m in cross_attn_modules.values()}.values():
        module.to(torch.bfloat16)

    unique_modules = len(set(id(m) for m in cross_attn_modules.values()))
    cross_attn_total_params = sum(
        p.numel() for module in set(cross_attn_modules.values())
        for p in module.parameters()
    )

    print(f"\n🔀 Cross-Attention:")
    print(f"  Layers: {layers_with_cross_attn} ({len(layers_with_cross_attn)}/{TOTAL_LAYERS})")
    print(f"  Unique modules: {unique_modules}")
    print(f"  Bottleneck dim: {BOTTLENECK_DIM}")
    print(f"  Parameters: {cross_attn_total_params / 1e6:.3f}M")

    # Create encoder projection
    encoder_projection = EncoderProjection(encoder_dim=ENCODER_DIM).to(torch.bfloat16)

    # Wrap decoder layers
    num_layers = len(decoder.base_model.model.model.layers)
    for i in range(num_layers):
        original_layer = decoder.base_model.model.model.layers[i]
        cross_attn_module = cross_attn_modules.get(i, None)
        decoder.base_model.model.model.layers[i] = LlamaDecoderLayerWithOptionalCrossAttention(
            original_layer,
            cross_attn_module=cross_attn_module
        )

    _dec_ckpt = TRAIN_CONFIG.get('use_gradient_checkpointing_decoder', False)
    patch_llama_model_forward(decoder, use_gradient_checkpointing=_dec_ckpt)

    # ── Create Complete Model ──
    model = SignLanguageTranslationModel(
        cnn_encoder=cnn_encoder,
        cnn_projection=cnn_projection,
        encoder=encoder,
        encoder_projection=encoder_projection,
        decoder=decoder,
        tokenizer=tokenizer,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Complete Model Created!")
    print(f"  Total trainable: {total_params / 1e6:.2f}M params")

    # ── Detailed Parameter Summary ──
    def _pcount(module, trainable_only=False):
        return sum(p.numel() for p in module.parameters()
                   if (p.requires_grad if trainable_only else True))

    def _row(label, total, trainable, indent=4):
        frozen = total - trainable
        pct = (trainable / max(total, 1)) * 100
        pad = " " * indent
        status = "all trainable" if frozen == 0 else ("all frozen" if trainable == 0 else f"{pct:.1f}% trainable")
        print(f"{pad}{label:<38}  {total/1e6:>8.3f} M   trainable: {trainable/1e6:>8.3f} M   [{status}]")

    _PW = 72
    print("\n" + "=" * _PW)
    print("  📊 DETAILED MODEL PARAMETER SUMMARY")
    print("=" * _PW)

    grand_total = 0
    grand_trainable = 0

    # ── 1. ResNet50 CNN Encoder ──
    cnn = model.cnn_encoder
    cnn_groups = [
        ("conv1 + bn1  (frozen)",
            sum(p.numel() for m in [cnn.conv1, cnn.bn1] for p in m.parameters()),
            0),
        ("layer1       (frozen)",
            _pcount(cnn.layer1), 0),
        ("layer2       (frozen)",
            _pcount(cnn.layer2), 0),
        ("layer3       (trainable)",
            _pcount(cnn.layer3), _pcount(cnn.layer3, trainable_only=True)),
        ("layer4       (trainable)",
            _pcount(cnn.layer4), _pcount(cnn.layer4, trainable_only=True)),
    ]
    cnn_total     = _pcount(model.cnn_encoder)
    cnn_trainable_n = _pcount(model.cnn_encoder, trainable_only=True)
    print(f"\n  ResNet50 CNN Encoder")
    for label, tot, tr in cnn_groups:
        _row(label, tot, tr)
    print(f"  {'─'*(_PW-2)}")
    _row("SUBTOTAL", cnn_total, cnn_trainable_n)
    grand_total += cnn_total; grand_trainable += cnn_trainable_n

    # ── 2. CNN Projection MLP ──
    proj_total = _pcount(model.cnn_projection)
    proj_tr    = _pcount(model.cnn_projection, trainable_only=True)
    print(f"\n  CNN Projection MLP  (Linear {CONFIG['resnet_output_dim']}→{CONFIG['cnn_projection_dim']} + GELU + Dropout + LayerNorm)")
    _row("Linear + LayerNorm", proj_total, proj_tr)
    print(f"  {'─'*(_PW-2)}")
    _row("SUBTOTAL", proj_total, proj_tr)
    grand_total += proj_total; grand_trainable += proj_tr

    # ── 3. Transformer Encoder ──
    enc_total = _pcount(model.encoder)
    enc_tr    = _pcount(model.encoder, trainable_only=True)
    print(f"\n  Transformer Encoder  ({CONFIG['encoder_num_layers']} layers, d_model={CONFIG['encoder_d_model']}, ff={CONFIG['encoder_feedforward_dim']})")
    for i, layer in enumerate(model.encoder.transformer_encoder.layers):
        lt = _pcount(layer); ltr = _pcount(layer, trainable_only=True)
        _row(f"layer {i}", lt, ltr)
    _row("positional encoding", _pcount(model.encoder.pos_encoder),
         _pcount(model.encoder.pos_encoder, trainable_only=True))
    print(f"  {'─'*(_PW-2)}")
    _row("SUBTOTAL", enc_total, enc_tr)
    grand_total += enc_total; grand_trainable += enc_tr

    # ── 4. Encoder Projection ──
    ep_total = _pcount(model.encoder_projection)
    ep_tr    = _pcount(model.encoder_projection, trainable_only=True)
    print(f"\n  Encoder Projection")
    _row("LayerNorm + Linear", ep_total, ep_tr)
    print(f"  {'─'*(_PW-2)}")
    _row("SUBTOTAL", ep_total, ep_tr)
    grand_total += ep_total; grand_trainable += ep_tr

    # ── 5. Cross-Attention modules ──
    _seen_ca = set()
    _ca_unique_params = 0
    for _layer in model.decoder.base_model.model.model.layers:
        if hasattr(_layer, 'cross_attn_module') and _layer.cross_attn_module is not None:
            _mid = id(_layer.cross_attn_module)
            if _mid not in _seen_ca:
                _seen_ca.add(_mid)
                _ca_unique_params += _pcount(_layer.cross_attn_module)
    _num_ca_layers = len([l for l in model.decoder.base_model.model.model.layers
                          if hasattr(l, 'cross_attn_module') and l.cross_attn_module is not None])
    _num_ca_unique = len(_seen_ca)
    print(f"\n  Cross-Attention  ({_num_ca_layers} decoder layers, {_num_ca_unique} unique modules, bottleneck={CONFIG['bottleneck_dim']})")
    _row(f"{_num_ca_unique} unique modules (all trainable)", _ca_unique_params, _ca_unique_params)
    if _num_ca_layers != _num_ca_unique:
        _row(f"shared weight savings ({_num_ca_layers - _num_ca_unique} shared refs)", 0, 0)
    print(f"  {'─'*(_PW-2)}")
    _row("SUBTOTAL", _ca_unique_params, _ca_unique_params)
    grand_total += _ca_unique_params; grand_trainable += _ca_unique_params

    # ── 6. Decoder (Llama + LoRA + norms, excl. cross-attn) ──
    _ca_pids = set(id(p) for l in model.decoder.base_model.model.model.layers
                   if hasattr(l, 'cross_attn_module') and l.cross_attn_module is not None
                   for p in l.cross_attn_module.parameters())
    _dec_frozen, _dec_lora, _dec_norm, _dec_other_tr = 0, 0, 0, 0
    for n, p in model.decoder.named_parameters():
        if id(p) in _ca_pids:
            continue
        if not p.requires_grad:
            _dec_frozen += p.numel()
        elif 'lora_' in n:
            _dec_lora += p.numel()
        elif 'norm' in n.lower():
            _dec_norm += p.numel()
        else:
            _dec_other_tr += p.numel()
    _dec_total_tr = _dec_lora + _dec_norm + _dec_other_tr
    _dec_total    = _dec_frozen + _dec_total_tr
    print(f"\n  Decoder  (Llama 3.2-1B + LoRA r={CONFIG['lora_r']} — excl. cross-attn counted above)")
    _row("Base weights        (frozen)",   _dec_frozen,   0)
    _row("LoRA adapters       (trainable)", _dec_lora,    _dec_lora)
    _row("Norm layers         (trainable)", _dec_norm,    _dec_norm)
    if _dec_other_tr > 0:
        _row("Other trainable",             _dec_other_tr, _dec_other_tr)
    print(f"  {'─'*(_PW-2)}")
    _row("SUBTOTAL", _dec_total, _dec_total_tr)
    grand_total += _dec_total; grand_trainable += _dec_total_tr

    # ── Grand Total ──
    grand_frozen = grand_total - grand_trainable
    print(f"\n{'=' * _PW}")
    print(f"  OVERALL MODEL")
    print(f"{'─' * _PW}")
    _row("Total parameters",      grand_total,     grand_total,     indent=4)
    _row("Trainable parameters",  grand_trainable, grand_trainable, indent=4)
    _row("Frozen parameters",     grand_frozen,    0,               indent=4)
    print(f"  Trainable ratio: {(grand_trainable/max(grand_total,1))*100:.2f}%")
    print("=" * _PW + "\n")

    # ── Close log tee, restore stdout ──
    sys.stdout = _tee._stdout
    _log_file.close()
    print(f"📄 Configuration + parameter summary saved → {_log_path}")

    # ── Create 3 Optimizers ──

    # Optimizer 1: New random weights (MLP projection + Encoder + Encoder Projection + Cross-attention)
    new_weights_params = list(model.cnn_projection.parameters()) + \
                         list(model.encoder.parameters()) + \
                         list(model.encoder_projection.parameters())

    # Collect cross-attn params (deduplicate weight-shared modules)
    seen_ids = set()
    cross_attn_param_list = []
    for layer in model.decoder.base_model.model.model.layers:
        if hasattr(layer, 'cross_attn_module') and layer.cross_attn_module is not None:
            for p in layer.cross_attn_module.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    cross_attn_param_list.append(p)

    new_weights_all_params = new_weights_params + cross_attn_param_list

    # Optimizer 2: CNN backbone (trainable layers only: layer3 + layer4)
    cnn_trainable_params = [p for p in model.cnn_encoder.parameters() if p.requires_grad]

    # Optimizer 3: Decoder (LoRA + norms, excluding cross-attn)
    cross_attn_param_ids = set(id(p) for p in cross_attn_param_list)
    decoder_params_with_names = [(n, p) for n, p in model.decoder.named_parameters()
                                  if p.requires_grad and id(p) not in cross_attn_param_ids]
    decoder_params = [p for n, p in decoder_params_with_names]

    # Select optimizer class
    if TRAIN_CONFIG.get('use_8bit_adam', False):
        if _BNB_AVAILABLE:
            _OptimizerClass = bnb.optim.AdamW8bit
            print("  ✓ Using 8-bit AdamW (bitsandbytes)")
        else:
            _OptimizerClass = AdamW
            print("  bitsandbytes not available, falling back to AdamW")
    else:
        _OptimizerClass = AdamW

    optimizer_new_weights = _OptimizerClass(
        new_weights_all_params,
        lr=TRAIN_CONFIG['encoder_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['adam_betas']
    )

    optimizer_cnn = _OptimizerClass(
        cnn_trainable_params,
        lr=TRAIN_CONFIG['cnn_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['adam_betas']
    )

    optimizer_decoder = _OptimizerClass(
        decoder_params,
        lr=TRAIN_CONFIG['decoder_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['adam_betas']
    )

    print(f"\n⚙️  3 Optimizers created:")
    print(f"  New weights: {sum(p.numel() for p in new_weights_all_params) / 1e6:.2f}M params (LR={TRAIN_CONFIG['encoder_lr']:.2e})")
    print(f"  CNN:         {sum(p.numel() for p in cnn_trainable_params) / 1e6:.2f}M params (LR={TRAIN_CONFIG['cnn_lr']:.2e})")
    print(f"  Decoder:     {sum(p.numel() for p in decoder_params) / 1e6:.2f}M params (LR={TRAIN_CONFIG['decoder_lr']:.2e})")

    # ── Create 3 LR Schedulers ──
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / TRAIN_CONFIG['grad_accum_steps'])
    total_optimizer_steps = optimizer_steps_per_epoch * TRAIN_CONFIG['num_epochs']
    warmup_steps = TRAIN_CONFIG['warmup_steps']

    # Scheduler 1: New weights (active from step 0)
    cosine_decay_steps_new = max(total_optimizer_steps - warmup_steps, 1)
    warmup_new = LinearLR(optimizer_new_weights, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine_new = CosineAnnealingLR(optimizer_new_weights, T_max=cosine_decay_steps_new, eta_min=TRAIN_CONFIG['encoder_min_lr'])
    scheduler_new_weights = SequentialLR(optimizer_new_weights, schedulers=[warmup_new, cosine_new], milestones=[warmup_steps])

    # Scheduler 2: CNN (activates at cnn_freeze_steps)
    cnn_freeze_steps = TRAIN_CONFIG['cnn_freeze_steps']
    cnn_warmup = TRAIN_CONFIG.get('cnn_warmup_steps', 100)
    cnn_effective_steps = max(total_optimizer_steps - cnn_freeze_steps, 1)
    cnn_cosine_decay_steps = max(cnn_effective_steps - cnn_warmup, 1)
    warmup_cnn = LinearLR(optimizer_cnn, start_factor=1e-8, end_factor=1.0, total_iters=cnn_warmup)
    cosine_cnn = CosineAnnealingLR(optimizer_cnn, T_max=cnn_cosine_decay_steps, eta_min=TRAIN_CONFIG['cnn_min_lr'])
    scheduler_cnn = SequentialLR(optimizer_cnn, schedulers=[warmup_cnn, cosine_cnn], milestones=[cnn_warmup])

    # Scheduler 3: Decoder (activates at decoder_freeze_steps)
    decoder_freeze_steps = TRAIN_CONFIG['decoder_freeze_steps']
    decoder_warmup = TRAIN_CONFIG.get('decoder_warmup_steps', 100)
    decoder_effective_steps = max(total_optimizer_steps - decoder_freeze_steps, 1)
    decoder_cosine_decay_steps = max(decoder_effective_steps - decoder_warmup, 1)
    warmup_dec = LinearLR(optimizer_decoder, start_factor=1e-8, end_factor=1.0, total_iters=decoder_warmup)
    cosine_dec = CosineAnnealingLR(optimizer_decoder, T_max=decoder_cosine_decay_steps, eta_min=TRAIN_CONFIG['decoder_min_lr'])
    scheduler_decoder = SequentialLR(optimizer_decoder, schedulers=[warmup_dec, cosine_dec], milestones=[decoder_warmup])

    print(f"\n📈 LR Schedules (3-stage):")
    print(f"  New weights: {warmup_steps} warmup → {cosine_decay_steps_new} cosine ({TRAIN_CONFIG['encoder_lr']:.2e} → {TRAIN_CONFIG['encoder_min_lr']:.2e})")
    print(f"  CNN:         {cnn_warmup} warmup → {cnn_cosine_decay_steps} cosine, starts at step {cnn_freeze_steps} ({TRAIN_CONFIG['cnn_lr']:.2e} → {TRAIN_CONFIG['cnn_min_lr']:.2e})")
    print(f"  Decoder:     {decoder_warmup} warmup → {decoder_cosine_decay_steps} cosine, starts at step {decoder_freeze_steps} ({TRAIN_CONFIG['decoder_lr']:.2e} → {TRAIN_CONFIG['decoder_min_lr']:.2e})")

    # ── Step calculations ──
    steps_per_epoch_micro = len(train_loader)
    steps_per_epoch_optim = math.ceil(steps_per_epoch_micro / TRAIN_CONFIG['grad_accum_steps'])
    total_optim_steps = steps_per_epoch_optim * TRAIN_CONFIG['num_epochs']
    effective_batch_size = TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['grad_accum_steps']

    print("\n" + "=" * 60)
    print("📋 TRAINING OVERVIEW")
    print("=" * 60)
    print(f"  Train samples:           {len(train_dataset)}")
    print(f"  Val samples:             {len(val_dataset)}")
    print(f"  Batch size:              {TRAIN_CONFIG['batch_size']}")
    print(f"  Gradient accumulation:   {TRAIN_CONFIG['grad_accum_steps']} steps")
    print(f"  Effective batch size:    {effective_batch_size}")
    print(f"  Micro-batch steps/epoch: {steps_per_epoch_micro}")
    print(f"  Optimizer steps/epoch:   {steps_per_epoch_optim}")
    print(f"  Total optimizer steps:   {total_optim_steps}")
    print(f"  Num epochs:              {TRAIN_CONFIG['num_epochs']}")
    print(f"  CNN frame size:          {CONFIG['cnn_frame_size']}x{CONFIG['cnn_frame_size']}")
    print(f"  CNN chunk size:          {CONFIG['cnn_chunk_size']} frames")
    print(f"  Phased curriculum:       Stage 1 (0-{cnn_freeze_steps}) → Stage 2 ({cnn_freeze_steps}-{decoder_freeze_steps}) → Stage 3 ({decoder_freeze_steps}+)")
    print(f"  Mixed precision:         bfloat16")
    print(f"  Grad clipping:           max_norm={TRAIN_CONFIG['max_grad_norm']}")
    print(f"  Early stopping:          patience={TRAIN_CONFIG['early_stopping_patience']}")
    print("=" * 60)

    # ── Initialize Managers ──
    ckpt_manager = CheckpointManager(
        checkpoint_dir=TRAIN_CONFIG['checkpoint_dir'],
        keep_last_n=TRAIN_CONFIG['keep_last_n_checkpoints'],
    )

    train_csv_logger = CSVLogger(
        log_file=TRAIN_CONFIG['train_log_file'],
        fieldnames=[
            'timestamp', 'global_step', 'epoch',
            'train_loss', 'train_ppl',
            'lr_new_weights', 'lr_cnn', 'lr_decoder',
            'grad_norm',
            'cuda_mem_gb', 'cuda_peak_gb',
            'steps_per_sec', 'elapsed_sec',
        ],
    )
    val_csv_logger = CSVLogger(
        log_file=TRAIN_CONFIG['val_log_file'],
        fieldnames=[
            'timestamp', 'global_step', 'epoch',
            'val_loss', 'val_ppl',
            'bleu1', 'bleu2', 'bleu4', 'rouge_l', 'token_acc',
            'cuda_mem_gb', 'cuda_peak_gb',
            'elapsed_sec',
        ],
    )
    gen_samples_csv_logger = CSVLogger(
        log_file=TRAIN_CONFIG['gen_samples_log_file'],
        fieldnames=[
            'timestamp', 'global_step', 'epoch',
            'generated', 'reference',
            'elapsed_sec',
        ],
    )

    tensorboard_dir = TRAIN_CONFIG.get('tensorboard_dir', Path('..') / 'runs' / 'signbridge_training')
    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    print(f"\n📁 Checkpoints dir:  {TRAIN_CONFIG['checkpoint_dir'].resolve()}")
    print(f"Train log:        {TRAIN_CONFIG['train_log_file'].resolve()}")
    print(f"Val log:          {TRAIN_CONFIG['val_log_file'].resolve()}")
    print(f"Gen samples log:  {TRAIN_CONFIG['gen_samples_log_file'].resolve()}")

    # ── Resume from checkpoint ──
    start_epoch = 1
    start_global_step = 0
    best_val_loss_val = float('inf')
    start_evals_without_improvement = 0
    start_elapsed_sec = 0.0

    if TRAIN_CONFIG.get('resume_training', False):
        if TRAIN_CONFIG.get('load_best_model', False):
            ckpt_path = ckpt_manager.best_path
        else:
            ckpt_path = ckpt_manager.checkpoint_dir / f"checkpoint_step_{TRAIN_CONFIG.get('resume_checkpoint_step', 0)}.pt"

        if ckpt_path.exists():
            print(f"\n🔄 Resuming training from {ckpt_path.name}...")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

            model.load_state_dict(checkpoint['model_state_dict'])

            _use_8bit = TRAIN_CONFIG.get('use_8bit_adam', False) and _BNB_AVAILABLE

            _new_opt_state = checkpoint.get('optimizer_new_weights_state_dict',
                                             checkpoint.get('optimizer_encoder_state_dict'))
            _cnn_opt_state = checkpoint.get('optimizer_cnn_state_dict')
            _dec_opt_state = checkpoint['optimizer_decoder_state_dict']

            if _use_8bit:
                if _new_opt_state:
                    _new_opt_state = _remap_opt_state_for_bnb(_new_opt_state)
                if _cnn_opt_state:
                    _cnn_opt_state = _remap_opt_state_for_bnb(_cnn_opt_state)
                _dec_opt_state = _remap_opt_state_for_bnb(_dec_opt_state)

            if _new_opt_state:
                optimizer_new_weights.load_state_dict(_new_opt_state)
            if _cnn_opt_state:
                optimizer_cnn.load_state_dict(_cnn_opt_state)
            optimizer_decoder.load_state_dict(_dec_opt_state)

            _new_sched_state = checkpoint.get('scheduler_new_weights_state_dict',
                                               checkpoint.get('scheduler_encoder_state_dict'))
            _cnn_sched_state = checkpoint.get('scheduler_cnn_state_dict')
            _dec_sched_state = checkpoint['scheduler_decoder_state_dict']

            if _new_sched_state:
                scheduler_new_weights.load_state_dict(_new_sched_state)
            if _cnn_sched_state:
                scheduler_cnn.load_state_dict(_cnn_sched_state)
            scheduler_decoder.load_state_dict(_dec_sched_state)

            start_epoch = checkpoint['epoch']
            start_global_step = checkpoint['global_step']
            best_val_loss_val = checkpoint.get('best_val_loss', float('inf'))
            start_evals_without_improvement = checkpoint.get('evals_without_improvement', 0)
            start_elapsed_sec = checkpoint.get('elapsed_sec', 0.0)

            if 'rng_state' in checkpoint:
                rng_state = checkpoint['rng_state']
                if rng_state.device.type != 'cpu':
                    rng_state = rng_state.cpu()
                torch.set_rng_state(rng_state)
            if 'cuda_rng_state_all' in checkpoint and torch.cuda.is_available():
                cuda_rng_states = [s.cpu() if s.device.type != 'cpu' else s for s in checkpoint['cuda_rng_state_all']]
                torch.cuda.set_rng_state_all(cuda_rng_states)
            if 'numpy_rng_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_rng_state'])
            if 'python_rng_state' in checkpoint:
                random.setstate(checkpoint['python_rng_state'])
            del checkpoint
            torch.cuda.empty_cache()
            print(f"  ✅ Loaded state: Epoch {start_epoch}, Step {start_global_step}, Best Val Loss: {best_val_loss_val:.4f}")
        else:
            print(f"\n  ⚠️  Resume requested but checkpoint not found: {ckpt_path}. Starting from scratch.")

    # ── Launch Training ──
    try:
        best_val_loss = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            optimizer_new_weights=optimizer_new_weights,
            optimizer_cnn=optimizer_cnn,
            optimizer_decoder=optimizer_decoder,
            scheduler_new_weights=scheduler_new_weights,
            scheduler_cnn=scheduler_cnn,
            scheduler_decoder=scheduler_decoder,
            device=device,
            train_config=TRAIN_CONFIG,
            ckpt_manager=ckpt_manager,
            train_csv_logger=train_csv_logger,
            val_csv_logger=val_csv_logger,
            gen_samples_csv_logger=gen_samples_csv_logger,
            tb_writer=tb_writer,
            start_epoch=start_epoch,
            start_global_step=start_global_step,
            best_val_loss=best_val_loss_val,
            start_evals_without_improvement=start_evals_without_improvement,
            start_elapsed_sec=start_elapsed_sec,
        )
    finally:
        tb_writer.close()

    print(f"\nBest validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
