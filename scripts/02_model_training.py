
## Imports & Setup

# Disable TensorFlow backend to avoid protobuf conflicts (PyTorch-only project)
import os
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
import torch._dynamo

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

## Additional Imports for Training
import time
import csv
from collections import Counter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR
import sacrebleu

import tempfile
import shutil
import torch.multiprocessing as mp
import torch.utils.checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

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
# Without this cap, loss spikes (early training, bad batches) cause PPL in billions or inf (loss > ~710)
# This clamps displayed perplexity to a sane maximum while keeping the actual loss value intact
MAX_PPL_CAP = 20

# %%
## Configuration
CONFIG = {
    # ── Data Paths ──
    'data_train_csv': Path('..') / 'data' / 'final_full_train_dataset.tsv',
    'data_val_csv': Path('..') / 'data' / 'final_full_val_dataset.tsv',
    'keypoints_train_dir': Path('..') / 'data' / 'full_dataset_keypoints_preprocessed' / 'train',
    'keypoints_val_dir': Path('..') / 'data' / 'full_dataset_keypoints_preprocessed' / 'val',
    'csv_sep': '\t',

    # ── Keypoint Dimensions ──
    'num_landmarks': 116,       # 60 face + 14 pose + 42 hands
    'coord_dim': 2,             # x, y (z dropped)

    # ── Sequence Lengths ──
    'max_keypoint_frames': 160, # At 20 FPS = 10s (max clip ~8s)
    'max_text_tokens': 50,     # Excluding BOS/EOS

    # ── Tokenizer / Decoder ──
    'decoder_model_name': 'meta-llama/Llama-3.2-1B',
    'decoder_hidden_size': 2048,
    'decoder_num_layers': 16,

    # ── MLP Projection ──
    'projection_hidden_dim': 512,
    'projection_d_model': 512,  # Output dim → encoder input
    'projection_dropout': 0.15,  # Increased from 0.1 to combat overfitting

    # ── Transformer Encoder ──
    'encoder_d_model': 512,
    'encoder_num_heads': 8,
    'encoder_num_layers': 4,
    'encoder_feedforward_dim': 2048,
    'encoder_dropout': 0.15,  # Increased from 0.1 to combat overfitting
    'encoder_max_len': 5000,

    # ── Cross-Attention ──
    'bottleneck_dim': 512,      # 512 / 8 heads = 64 dims/head
    'cross_attn_num_heads': 8,
    'cross_attn_dropout': 0.1,
    'cross_attn_layers': [0, 1, 2, 8, 13, 15],  # 6 out of 16
    'weight_sharing_pairs': [],              # 2 pairs

    # ── LoRA ──
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    'lora_modules_to_save': [],  # embed_tokens and lm_head kept frozen
    'lora_dropout': 0.1,
    'lora_bias': "none",

    # ── Data Augmentation ──
    'aug_temporal_jitter': True,       # Enable temporal jitter augmentation
    'aug_temporal_jitter_range': 2,    # Max frames to shift (±2)
    'aug_temporal_jitter_prob': 0.5,   # Probability of applying
    'aug_gaussian_noise': True,        # Enable Gaussian noise
    'aug_noise_std': 0.01,             # Noise std for normalized coords
    'aug_noise_prob': 0.5,             # Probability of applying
    'aug_spatial_shift': True,         # Enable spatial translation (X/Y shift)
    'aug_spatial_shift_range': 0.05,   # Max shift as fraction (±5% applied to all frames/landmarks)
    'aug_spatial_shift_prob': 0.5,     # Probability of applying
    'aug_spatial_scale': True,         # Enable spatial scaling (resize skeleton)
    'aug_spatial_scale_range': (0.9, 1.1),  # Scale factor range (90%-110% size)
    'aug_spatial_scale_prob': 0.5,     # Probability of applying
}

## Training Configuration
TRAIN_CONFIG = {
    # ── Core Training ──
    'num_epochs': 30,
    'batch_size': 8,

    # ── Gradient Accumulation ──
    'grad_accum_steps': 8,          # effective batch = batch_size * grad_accum_steps

    # ── Learning Rates (separate for different components) ──
    # Encoder + Cross-attention (new, untrained components)
    'encoder_lr': 4e-5,              # Higher LR for new encoder/cross-attn
    'encoder_min_lr': 4e-7, # 1% of initial LR
    # LoRA adapters (fine-tuning pretrained decoder)
    'decoder_lr': 3e-5,              # Lower LR for LoRA fine-tuning
    'decoder_min_lr': 3e-7, # 1% of initial LR

    'warmup_steps': 250,
    'weight_decay': 0.05,
    'adam_betas': (0.9, 0.98),
    'max_grad_norm': 0.6,

    # ── Logging ──
    'log_every_steps': 5,          # print training stats every N optimizer steps
    'train_log_file': Path('..') / 'saved_metrics' / 'train_log.csv',
    'val_log_file': Path('..') / 'saved_metrics' / 'val_log.csv',
    'gen_samples_log_file': Path('..') / 'saved_metrics' / 'gen_samples_log.csv',
    'tensorboard_dir': Path('..') / 'saved_metrics' / 'tensorboard' / 'signbridge_training',

    # ── Checkpointing ──
    'save_every_steps': 400,        # save checkpoint every N optimizer steps
    'keep_last_n_checkpoints': 3,   # sliding window: keep only last N periodic checkpoints
    'checkpoint_dir': Path('..') / 'checkpoints',

    # ── Evaluation ──
    'eval_every_steps': 250,        # run validation every N optimizer steps (after threshold)
    'eval_every_steps_warmup': 900, # run validation every N steps while below eval_warmup_threshold
    'eval_warmup_threshold': 2000,  # step at which to switch from warmup to normal eval frequency
                                    # counter resets at threshold: next eval at threshold + eval_every_steps
    'max_eval_batches': 60,         # max batches for validation loss computation
    'max_generate_samples': 200,     # max samples for BLEU/ROUGE generation
    'num_print_samples': 4,         # how many generated samples to print during eval
    'val_gen_batch_size': 16,       # batch size for validation generation (speeds up BLEU/ROUGE)
    'val_beam_size': 4,             # beam search width for validation generation (1 = greedy)
    'val_repetition_penalty': 1.15,  # repetition penalty for validation generation (1.0 = off)
    'val_use_kv_cache': True,       # use KV-cache during generation (faster, disable if issues)

    # ── Early Stopping ──
    'early_stopping_patience': 25,   # stop after N evaluations without improvement

    # ── DataLoader config ──
    'train_num_workers': 2,          # workers for training DataLoader (fewer saves RAM; each worker forks the tokenizer ~300MB)
    'train_prefetch_factor': 2,      # prefetch factor for training DataLoader
    'val_num_workers': 1,            # workers for validation DataLoader
    'val_prefetch_factor': 2,        # prefetch factor for validation DataLoader
    'use_bucket_batching': True,     # sort batches by sequence length to reduce padding waste (10–25% speedup)

    # ── Performance & Memory Optimizations ──
    'use_sdpa': True,                             # use PyTorch SDPA fused kernel for Llama self-attention (no extra packages; saves attention VRAM + 10–20% speedup)
    'use_8bit_adam': True,                        # 8-bit AdamW from bitsandbytes — saves ~480 MB VRAM in optimizer states with zero quality loss
                                                  # (block-wise quantized moments only; weight updates are still fp32 — convergence is identical)
    'use_gradient_checkpointing_encoder': True,   # recompute encoder activations on backward (saves ~150 MB VRAM, ~15% encoder compute overhead)
    'use_gradient_checkpointing_decoder': False,   # recompute Llama activations on backward (saves 500 MB–1 GB VRAM, ~25% decoder compute overhead)
                                                  # incompatible with KV-cache; automatically disabled during generation (use_cache=True)

    # ── Resuming ──
    'resume_training': True,       # set to True to load a checkpoint
    'load_best_model': True,       # if True, loads best_model.pt instead of periodic checkpoint
    'resume_checkpoint_step': 3200, # step number of the checkpoint to load if load_best_model is False

    # ── Decoder Freeze Phase ──
    # Freezes the entire decoder (LoRA + embeddings + norms — everything) for the first N
    # optimizer steps, forcing cross-attention to establish a real signal before Llama adapts.
    # Cross-attention + encoder continue to train normally throughout this phase.
    # The decoder LR schedule starts its own independent clock only after the freeze ends,
    # so its warmup → cosine decay runs over the remaining training window, not the full run.
    'freeze_decoder': True,          # if True, freeze decoder for the first N optimizer steps
    'decoder_freeze_steps': 2000,    # number of optimizer steps to keep the decoder frozen
}

# Max sequence lengths (safety caps to avoid OOM)
MAX_KEYPOINT_FRAMES = CONFIG['max_keypoint_frames']
MAX_TEXT_TOKENS = CONFIG['max_text_tokens']

class SignLanguageDataset(Dataset):
    def __init__(self, manifest, tokenizer, max_frames=MAX_KEYPOINT_FRAMES, max_tokens=MAX_TEXT_TOKENS,
                 train=False, augment_config=None):
        """
        Args:
            manifest: List of dicts with keys: sentence_name, text, duration, keypoint_path
            tokenizer: Pretrained tokenizer (Llama 3.2)
            max_frames: Max keypoint frames (truncate if longer)
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
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        sample = self.manifest[idx]
        
        # Load keypoints with error handling
        try:
            data = np.load(sample['keypoint_path'])
            keypoints = data['keypoints'][:, :, :2]  # (T, N, 2) - drop Z
            mask = data['mask']  # (T, N)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load keypoints for {sample['sentence_name']}: {e}\n"
                f"Path: {sample['keypoint_path']}"
            )
        
        # Truncate keypoints if too long
        if keypoints.shape[0] > self.max_frames:
            keypoints = keypoints[:self.max_frames]
            mask = mask[:self.max_frames]
        
        # Convert keypoints to torch tensors
        keypoints = torch.from_numpy(keypoints).float()  # (T, N, 2)
        mask = torch.from_numpy(mask).bool()  # (T, N) - bool for attention masking

        # ── Data Augmentation (training only) ──
        if self.train:
            # Temporal jitter: shift sequence by ±max_shift frames
            if self.aug_config.get('aug_temporal_jitter', False) and random.random() < self.aug_config.get('aug_temporal_jitter_prob', 0.5):
                max_shift = self.aug_config.get('aug_temporal_jitter_range', 2)
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    T = keypoints.shape[0]
                    if shift > 0:
                        # Shift right: pad at start, truncate at end
                        keypoints = torch.cat([torch.zeros(shift, *keypoints.shape[1:]), keypoints[:-shift]], dim=0)
                        mask = torch.cat([torch.zeros(shift, *mask.shape[1:], dtype=torch.bool), mask[:-shift]], dim=0)
                    else:
                        # Shift left: truncate at start, pad at end
                        keypoints = torch.cat([keypoints[-shift:], torch.zeros(-shift, *keypoints.shape[1:])], dim=0)
                        mask = torch.cat([mask[-shift:], torch.zeros(-shift, *mask.shape[1:], dtype=torch.bool)], dim=0)

            # Gaussian noise: add small perturbations to coordinates
            if self.aug_config.get('aug_gaussian_noise', False) and random.random() < self.aug_config.get('aug_noise_prob', 0.5):
                noise_std = self.aug_config.get('aug_noise_std', 0.01)
                # Only add noise to valid keypoints (where mask is True)
                noise = torch.randn_like(keypoints) * noise_std
                keypoints = keypoints + noise * mask.unsqueeze(-1).float()  # Apply only where mask=True

            # Spatial shift: translate all keypoints by consistent X/Y offset
            if self.aug_config.get('aug_spatial_shift', False) and random.random() < self.aug_config.get('aug_spatial_shift_prob', 0.5):
                shift_range = self.aug_config.get('aug_spatial_shift_range', 0.05)
                # Sample independent shifts for X and Y axes (applied to ALL frames/landmarks)
                shift_x = random.uniform(-shift_range, shift_range)
                shift_y = random.uniform(-shift_range, shift_range)

                # Apply consistent translation to all valid keypoints
                # keypoints: (T, N, 2) where [..., 0] is X and [..., 1] is Y
                shift_tensor = torch.tensor([shift_x, shift_y], dtype=keypoints.dtype)
                keypoints = keypoints + shift_tensor * mask.unsqueeze(-1).float()  # Apply only where mask=True

            # Spatial scaling: resize skeleton around centroid (simulates different signer sizes/distances)
            if self.aug_config.get('aug_spatial_scale', False) and random.random() < self.aug_config.get('aug_spatial_scale_prob', 0.5):
                scale_range = self.aug_config.get('aug_spatial_scale_range', (0.9, 1.1))
                scale_factor = random.uniform(scale_range[0], scale_range[1])

                # Scale around centroid to avoid position drift
                # Only use valid keypoints to compute centroid
                if mask.any():
                    centroid = keypoints[mask].mean(dim=0)  # (2,) - average X,Y of valid keypoints

                    # Scale all keypoints around centroid (keypoints - centroid) * scale + centroid
                    keypoints_scaled = centroid + (keypoints - centroid) * scale_factor

                    # Apply only to valid keypoints, preserve invalid ones as-is
                    mask_expanded = mask.unsqueeze(-1)  # (T, N, 1) -> broadcasts to (T, N, 2)
                    keypoints = torch.where(mask_expanded, keypoints_scaled, keypoints)

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
            'keypoints': keypoints,  # (T, N, 2)
            'keypoint_mask': mask,  # (T, N) bool
            'token_ids': token_ids,  # (L,) where L = text_len + 2 (BOS + EOS)
            'sentence_name': sample['sentence_name']  # For debugging
        }

        # ● The 3 dimensions explained:
        #   torch.Size([144, 116, 3]) means:
        #   - 144 = Number of frames (T) - temporal dimension
        #   - 116 = Number of landmarks (N) - 60 face + 14 pose + 42 hands
        #   - 3 = Coordinates (x, y, z) for each landmark

        #   So shape is (T, N, C) = (frames, landmarks, coordinates)

# %%
## Step 7: Create Collate Function for Batching

def collate_fn(batch, pad_token_id):
    """
    Collate function to batch variable-length samples.
    
    Args:
        batch: List of dicts from Dataset.__getitem__
        pad_token_id: Token ID to use for padding text
        
    Returns:
        Dict with batched tensors:
            keypoints: (B, max_T, N, 2) - padded keypoints
            keypoint_mask: (B, max_T, N) - bool, True=valid, False=padded
            token_ids: (B, max_L) - padded token IDs
            text_attention_mask: (B, max_L) - bool, True=real token, False=padding
    """
    # Extract individual components
    keypoints_list = [item['keypoints'] for item in batch]  # List of (T_i, N, 2)
    keypoint_mask_list = [item['keypoint_mask'] for item in batch]  # List of (T_i, N)
    token_ids_list = [item['token_ids'] for item in batch]  # List of (L_i,)
    
    # Find max lengths in this batch
    max_keypoint_frames = max(kp.shape[0] for kp in keypoints_list)
    max_token_length = max(tokens.shape[0] for tokens in token_ids_list)
    
    batch_size = len(batch)
    num_landmarks = keypoints_list[0].shape[1]  # N = 116
    
    # Initialize padded tensors
    padded_keypoints = torch.zeros(batch_size, max_keypoint_frames, num_landmarks, 2)
    padded_keypoint_mask = torch.zeros(batch_size, max_keypoint_frames, num_landmarks, dtype=torch.bool)
    padded_token_ids = torch.full((batch_size, max_token_length), pad_token_id, dtype=torch.long)
    text_attention_mask = torch.zeros(batch_size, max_token_length, dtype=torch.bool)
    
    # Fill in actual data
    for i in range(batch_size):
        # Keypoints
        T = keypoints_list[i].shape[0]
        padded_keypoints[i, :T] = keypoints_list[i]
        padded_keypoint_mask[i, :T] = keypoint_mask_list[i]
        
        # Token IDs
        L = token_ids_list[i].shape[0]
        padded_token_ids[i, :L] = token_ids_list[i]
        text_attention_mask[i, :L] = True  # Real tokens
    
    return {
        'keypoints': padded_keypoints,  # (B, max_T, N, 2)
        'keypoint_mask': padded_keypoint_mask,  # (B, max_T, N) bool
        'token_ids': padded_token_ids,  # (B, max_L) long
        'text_attention_mask': text_attention_mask,  # (B, max_L) bool
    }

# What's in the batch (what we just saw):
# keypoints: (4, 160, 116, 2) - encoder input (4 samples, up to 160 frames, 116 landmarks, x/y)
# keypoint_mask: (4, 160, 116) bool - which landmarks are valid
# token_ids: (4, 21) - FULL sequence: [BOS, token1, ..., tokenN, EOS]
# text_attention_mask: (4, 21) bool - which tokens are real vs padding

# 1. ENCODER gets:
# Input: keypoints = (4, 160, 116, 2) *****(It's the MLP output accutally, but we will add that later)
# Mask: keypoint_mask = (4, 160, 116)
# ↓
# Encoder processes this
# ↓
# Output: encoder_hidden_states = (4, 160, d_model)
# # where d_model is encoder's hidden dimension (e.g., 512)

# 2. DECODER gets (teacher forcing):
# # We SPLIT token_ids into two:

# Decoder INPUT (shifted right):
# decoder_input_ids = token_ids[:, :-1]  # (4, 20)
# # = [BOS, token1, token2, ..., tokenN]
# # Remove last token (EOS)

# Decoder CROSS-ATTENDS to:
# encoder_hidden_states = (4, 160, d_model)

# Decoder OUTPUT:
# logits = (4, 20, vocab_size)  # Predictions for each position

# 3. LOSS computed on:
# Predictions: logits (4, 20, vocab_size)
# Targets: token_ids[:, 1:]  # (4, 20)
#         # = [token1, token2, ..., tokenN, EOS]
#         # Remove first token (BOS)

class BucketBatchSampler(torch.utils.data.Sampler):
    """
    Groups samples with similar keypoint sequence lengths into the same batch
    to minimise padding waste. Within each epoch, batch order is shuffled to
    preserve randomness while keeping similar-length sequences together.

    Sequence lengths are estimated from the 'duration' field in the manifest
    at 20 FPS (capped at MAX_KEYPOINT_FRAMES), so no disk I/O is needed at
    initialisation.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        lengths = [
            min(int(s.get('duration', 0) * 20), MAX_KEYPOINT_FRAMES)
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
## Step 8: Load Pretrained Llama 3.2-1B Decoder

# %%
# %%
# %%
# %%
## Step 8 (continued): Apply Hybrid LoRA Configuration for Llama 3.2-1B

# %%
## Step 9: Build MLP Projection Layer

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

class KeypointProjection(nn.Module):
    def __init__(self, num_landmarks=116, coord_dim=2, hidden_dim=512, d_model=512):
        """
        Projects flattened keypoints to encoder dimension.
        
        Args:
            num_landmarks: Number of landmarks (116)
            coord_dim: Coordinates per landmark (2: x, y)
            hidden_dim: Hidden layer size
            d_model: Output dimension (encoder d_model = 512)
        """
        super().__init__()
        
        input_dim = num_landmarks * coord_dim  # 116 * 2 = 232
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(CONFIG['projection_dropout']),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)  # Normalize output for stable encoder input
        )
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        init_from_scratch_weights(self)
    
    def forward(self, keypoints):
        """
        Args:
            keypoints: (B, T, N, 2) where N=116, 2=x,y
            
        Returns:
            (B, T, d_model) projected representations
        """
        B, T, N, C = keypoints.shape
        
        # Flatten landmarks: (B, T, N, C) -> (B, T, N*C)
        x = keypoints.reshape(B, T, -1)  # (B, T, 232)
        
        # Project to d_model
        x = self.projection(x)  # (B, T, 512)
        
        return x

# %%
## Step 10: Build Transformer Encoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Sinusoidal positional encoding for temporal sequences.
        
        Args:
            d_model: Model dimension (512)
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, 
                 feedforward_dim=2048, dropout=0.1, max_len=5000):
        """
        Transformer encoder for keypoint sequences.
        
        Args:
            d_model: Model dimension (512)
            num_heads: Number of attention heads (8)
            num_layers: Number of encoder layers (6)
            feedforward_dim: Feedforward network dimension (2048)
            dropout: Dropout rate
            max_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Input shape: (B, T, d_model)
            norm_first=True    # Pre-norm (more stable for training from scratch)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        init_from_scratch_weights(self)
        
    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: (B, T, d_model) - projected keypoints from MLP
            src_key_padding_mask: (B, T) - bool mask, True for padding positions

        Returns:
            (B, T, d_model) - encoded representations
        """
        x = self.pos_encoder(x)

        if self.training and getattr(self, 'use_gradient_checkpointing', False):
            # Recompute activations layer-by-layer on backward instead of storing them.
            # Saves activation VRAM at ~15% extra encoder compute.
            # src_key_padding_mask is loop-invariant; captured safely via factory closure.
            for layer in self.transformer_encoder.layers:  # type: ignore[union-attr]
                def make_layer_fn(l):
                    def fn(h):
                        return l(h, src_key_padding_mask=src_key_padding_mask)
                    return fn
                x = torch.utils.checkpoint.checkpoint(
                    make_layer_fn(layer), x, use_reentrant=False
                )
            return x

        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)


# %%
## Step 11: Optimized Cross-Attention for Llama 3.2-1B (Bottleneck: 512 dims)

# Encoder projection: Just normalization (encoder already at 512)
class EncoderProjection(nn.Module):
    def __init__(self, encoder_dim=512):
        """Normalizes encoder outputs."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(encoder_dim)
        
        init_from_scratch_weights(self)
    
    def forward(self, encoder_outputs):
        return self.layer_norm(encoder_outputs)


class CrossAttentionModule(nn.Module):
    """Bottleneck cross-attention module (can be shared across layers)."""
    def __init__(self, hidden_size, encoder_dim, bottleneck_dim, num_heads, dropout=0.1):
        super().__init__()
        # Projections for query (decoder), key and value (encoder)
        self.cross_attn_in_proj = nn.Linear(hidden_size, bottleneck_dim)  # decoder: decoder_hidden_size → 512
        self.key_proj = nn.Linear(encoder_dim, bottleneck_dim)  # encoder: encoder_dim → bottleneck_dim
        self.value_proj = nn.Linear(encoder_dim, bottleneck_dim)  # encoder: encoder_dim → bottleneck_dim
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn_out_proj = nn.Linear(bottleneck_dim, hidden_size)  # bottleneck_dim → decoder_hidden_size
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Apply standard normal initialization
        init_from_scratch_weights(self)

        # OVERRIDE: Zero-init output projection
        # Cross-attention starts as no-op, learns gradually
        # This prevents the decoder from learning to suppress random noise at init
        nn.init.zeros_(self.cross_attn_out_proj.weight)
        nn.init.zeros_(self.cross_attn_out_proj.bias)
        
    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
        original_dtype = hidden_states.dtype
        
        # Only upcast fp16; bfloat16 is numerically stable enough
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
        
        # Cast back to original dtype
        return output.to(original_dtype)


# %%
## Step 12: Build Complete Encoder-Decoder Model

# ========== WRAPPED DECODER LAYER WITH CROSS-ATTENTION ==========

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
        # KV-cache is supported: self_attn updates the cache in-place,
        # cross-attention doesn't need caching (encoder output is constant during generation)

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


# ========== PATCH LLAMA FORWARD TO SUPPORT ENCODER-DECODER ==========

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

        # ── KV-Cache support ──
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

        # Single rotary embedding (Llama uses standard RoPE, no dual rotary)
        position_embeddings = rotary_emb(hidden_states, position_ids)

        # ── Build causal attention mask ──
        fill_val = -1e4 if dtype == torch.float16 else -1e9
        total_len = past_seen_tokens + seq_len

        if past_seen_tokens > 0 and seq_len == 1:
            # Decode mode: new token can attend to all cached tokens → no masking
            causal_mask = torch.zeros(1, 1, 1, total_len, device=device, dtype=dtype)
        else:
            # Prefill / training mode: standard causal mask
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

        # Gradient checkpointing: recompute activations on backward instead of storing them.
        # Disabled automatically during generation (use_cache=True) and when attention
        # weights are requested (output_attentions=True), since those require the full forward.
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
                # Factory closure correctly captures the current `decoder_layer` at each
                # iteration — avoids the Python late-binding gotcha in for-loop closures.
                # _do_checkpoint guarantees output_attentions=False, so only hidden_states
                # is returned and all_self_attns is never touched in this branch.
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
                        )[0]  # return only hidden_states
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
    print("✓ Patched Llama with standard causal attention + encoder-decoder support")

# ========== COMPLETE MODEL ==========
class SignLanguageTranslationModel(nn.Module):
    def __init__(self, keypoint_projection, encoder, encoder_projection, decoder, tokenizer):
        super().__init__()
        self.keypoint_projection = keypoint_projection
        self.encoder = encoder
        self.encoder_projection = encoder_projection
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)

    def forward(self, keypoints, keypoint_mask, token_ids, text_attention_mask, return_loss=True):
        device = keypoints.device

        # Encoder
        encoder_input = self.keypoint_projection(keypoints.to(next(self.keypoint_projection.parameters()).dtype))
        encoder_padding_mask = ~(keypoint_mask.any(dim=-1))
        encoder_output = self.encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
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
            encoder_attention_mask=encoder_padding_mask,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits

        if return_loss:
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))  # Bug #6: use self.loss_fn
            return {'loss': loss, 'logits': logits}

        return {'logits': logits}

    def _apply_repetition_penalty(self, logits, generated_ids, penalty):
        """Apply repetition penalty to logits for previously generated tokens.
        Positive logits are divided by penalty, negative logits are multiplied.
        This discourages the model from repeating tokens it has already produced.
        """
        if penalty == 1.0:
            return logits
        # Gather logits for previously generated tokens
        score = torch.gather(logits, 1, generated_ids)
        # Penalize: reduce positive scores, amplify negative scores
        score = torch.where(score > 0, score / penalty, score * penalty)
        logits = logits.scatter(1, generated_ids, score)
        return logits

    @torch.no_grad()
    def _beam_search_single(self, encoder_hidden_states, encoder_padding_mask,
                            device, max_new_tokens, beam_size, repetition_penalty, use_kv_cache=True):
        """
        Beam search for a single sample. Beams are batched for GPU efficiency.

        Args:
            encoder_hidden_states: (1, T, D) - single sample encoder output
            encoder_padding_mask: (1, T) - single sample mask
            device: torch device
            max_new_tokens: max tokens to generate
            beam_size: number of beams
            repetition_penalty: penalty for repeated tokens

        Returns:
            best_sequence: (L,) tensor of token IDs
        """
        eos_id = self.tokenizer.eos_token_id
        bos_id = self.tokenizer.bos_token_id

        # Expand encoder states for all beams: (1, T, D) → (K, T, D)
        enc_hs = encoder_hidden_states.expand(beam_size, -1, -1).contiguous()
        enc_mask = encoder_padding_mask.expand(beam_size, -1).contiguous()

        # Initialize: all beams start with BOS
        generated = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)

        # Log-probability scores for each beam
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = -1e9  # Only beam 0 is active initially

        completed = []  # List of (length_normalized_score, sequence)
        past_key_values = None  # KV-cache for self-attention

        for step in range(max_new_tokens):
            # With KV-cache: only pass the new token after the first step
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
            logits = outputs.logits[:, -1, :]  # (K, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)

            log_probs = torch.log_softmax(logits, dim=-1)  # (K, V)
            vocab_size = log_probs.shape[-1]

            # Compute next scores: beam_score + log_prob for each candidate
            next_scores = beam_scores.unsqueeze(1) + log_probs  # (K, V)
            flat_scores = next_scores.view(-1)  # (K * V)

            # Select top 2*K candidates (extra to handle EOS completions)
            num_candidates = min(2 * beam_size, flat_scores.shape[0])
            top_scores, top_flat_idx = flat_scores.topk(num_candidates)

            beam_idx = top_flat_idx // vocab_size   # Which beam each came from
            token_idx = top_flat_idx % vocab_size   # Which token was selected

            new_beams = []
            new_scores_list = []
            new_beam_source_indices = []  # Track source beam for cache reordering

            for i in range(len(top_scores)):
                if len(new_beams) >= beam_size:
                    break

                b = beam_idx[i].item()
                t = token_idx[i].item()
                s = top_scores[i].item()

                new_seq = torch.cat([generated[b], torch.tensor([t], device=device)])

                if t == eos_id:
                    # Completed beam — store with length-normalized score
                    completed.append((s / len(new_seq), new_seq))
                else:
                    new_beams.append(new_seq)
                    new_scores_list.append(s)
                    new_beam_source_indices.append(b)

            # All beams ended with EOS this step
            if len(new_beams) == 0:
                break

            # Pad to beam_size if needed (some beams completed)
            while len(new_beams) < beam_size:
                new_beams.append(new_beams[-1].clone())
                new_scores_list.append(-1e9)
                new_beam_source_indices.append(new_beam_source_indices[-1])

            # Reorder KV-cache to match the new beam assignments
            if use_kv_cache and past_key_values is not None:
                cache_reorder_idx = torch.tensor(
                    new_beam_source_indices[:beam_size], dtype=torch.long, device=device
                )
                past_key_values.reorder_cache(cache_reorder_idx)

            generated = torch.stack(new_beams[:beam_size])
            beam_scores = torch.tensor(new_scores_list[:beam_size], device=device)

            # Early stop: enough completed hypotheses
            if len(completed) >= beam_size:
                break

        if completed:
            # Return highest-scoring completed sequence
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]
        else:
            # No beam produced EOS — return best active beam
            best = beam_scores.argmax().item()
            return generated[best]

    @torch.no_grad()
    def generate(self, keypoints, keypoint_mask, max_new_tokens=50, temperature=1.0,
                 top_k=50, repetition_penalty=1.0, beam_size=1, use_kv_cache=True):
        """
        Generate text from keypoints with support for batched inputs.

        Args:
            keypoints: (B, T, N, 2) - batched keypoints
            keypoint_mask: (B, T, N) - batched masks
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k sampling
            repetition_penalty: penalty factor for repeated tokens (1.0 = no penalty)
            beam_size: number of beams for beam search (1 = greedy/sampling)
            use_kv_cache: whether to use KV-cache for faster generation

        Returns:
            dict with 'generated_ids' (B, L) and 'generated_text' (list of B strings)
        """
        device = keypoints.device
        batch_size = keypoints.shape[0]

        # Encode keypoints (shared by all decoding strategies)
        encoder_input = self.keypoint_projection(keypoints.to(next(self.keypoint_projection.parameters()).dtype))
        encoder_padding_mask = ~(keypoint_mask.any(dim=-1))
        encoder_output = self.encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
        encoder_hidden_states = self.encoder_projection(encoder_output).to(next(self.decoder.parameters()).dtype)

        # ── Beam Search Path ──
        if beam_size > 1:
            # Process each sample individually (beams are batched per sample)
            all_sequences = []
            for i in range(batch_size):
                best_seq = self._beam_search_single(
                    encoder_hidden_states[i:i+1],
                    encoder_padding_mask[i:i+1],
                    device, max_new_tokens, beam_size, repetition_penalty, use_kv_cache,
                )
                all_sequences.append(best_seq)

            # Decode and pad for return
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
        past_key_values = None  # KV-cache for self-attention

        for _ in range(max_new_tokens):
            # With KV-cache: only pass the new token after the first step
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
            next_token_logits = outputs.logits[:, -1, :]  # (B, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, repetition_penalty
                )

            # Greedy decoding if temperature is 0 or very low
            if temperature < 1e-5:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Update active sequences: mark as inactive if EOS is generated
            active_sequences &= (next_token.squeeze(-1) != self.tokenizer.eos_token_id)

            # Stop if all sequences have generated EOS
            if not active_sequences.any():
                break

        # Decode all generated sequences
        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        return {'generated_ids': generated_ids, 'generated_text': generated_texts}

## Helper Functions: Metrics, Checkpointing, CSV Logging
# ─── Metric Functions ───

def compute_bleu(references, hypotheses, max_n=4):
    """
    Compute corpus-level BLEU score using sacrebleu for standardized, reproducible metrics.

    Args:
        references: List of reference strings
        hypotheses: List of hypothesis strings
        max_n: Maximum n-gram order (1 for BLEU-1, 4 for BLEU-4)

    Returns:
        BLEU score as a percentage (0-100) with proper tokenization normalization.
    """
    if not references or not hypotheses:
        return 0.0

    # sacrebleu expects List[List[str]] where outer list is reference SETS, not sentences
    # We have ONE reference set containing all reference sentences: [references]
    bleu_result = sacrebleu.corpus_bleu(
        hypotheses,
        [references],  # Single reference set with all sentences
        tokenize='13a'  # Standard Moses tokenizer (most common in literature)
    )

    # sacrebleu computes BLEU-4 by default with all n-gram precisions
    # To get BLEU-1, BLEU-2, etc., we compute geometric mean of first N precisions
    if max_n == 4:
        # Return full BLEU-4 score
        return bleu_result.score
    else:
        # Extract first max_n precisions and compute BLEU-n manually
        precisions = bleu_result.precisions[:max_n]  # List of individual n-gram precisions
        bp = bleu_result.bp  # Brevity penalty

        # Geometric mean of precisions (in log space to avoid underflow)
        if any(p == 0 for p in precisions):
            return 0.0

        log_precision_sum = sum(math.log(p) for p in precisions)
        geo_mean = math.exp(log_precision_sum / max_n)

        # BLEU = brevity_penalty * geometric_mean_of_precisions
        return bp * geo_mean


def compute_rouge_l(references, hypotheses):
    """
    Compute average ROUGE-L F1 score across sentence pairs.
    Returns score as a percentage (0-100).
    """
    if not references or not hypotheses:
        return 0.0

    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()

        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue

        # LCS via dynamic programming
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


# ─── Checkpoint Manager ───

class CheckpointManager:
    """Manages model checkpoints with a sliding window for periodic saves."""

    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.periodic_checkpoints = []  # list of paths (oldest first)
        self.best_path = self.checkpoint_dir / 'best_model.pt'

    def _atomic_save(self, state_dict, path):
        """Atomically save checkpoint to prevent corruption on disk errors."""


        # Save to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix='.pt.tmp')
        try:
            # Close the file descriptor, torch.save will open it again
            os.close(temp_fd)

            # Save to temp file
            torch.save(state_dict, temp_path)

            # Atomic rename (replaces old file if exists)
            shutil.move(temp_path, path)
            return True
        except Exception as e:
            # Clean up temp file on error
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise RuntimeError(f"Failed to save checkpoint to {path}: {e}")

    def save_periodic(self, state_dict, step):
        """Save a periodic checkpoint with sliding window eviction."""
        path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        try:
            self._atomic_save(state_dict, path)
            self.periodic_checkpoints.append(path)
            print(f"  💾 Saved periodic checkpoint: {path.name}")
        except RuntimeError as e:
            print(f"  ⚠️  Warning: {e}")
            return

        # Evict oldest if exceeding window
        while len(self.periodic_checkpoints) > self.keep_last_n:
            old_path = self.periodic_checkpoints.pop(0)
            if old_path.exists() and old_path != self.best_path:
                old_path.unlink()
                print(f"  🗑️  Evicted old checkpoint: {old_path.name}")

    def save_best(self, state_dict):
        """Save the best model (always kept, never evicted). Uses atomic save to prevent corruption."""
        try:
            self._atomic_save(state_dict, self.best_path)
            print(f"  ⭐ Saved best model: {self.best_path.name}")
        except RuntimeError as e:
            print(f"  ⚠️  Warning: Failed to save best model: {e}")
            print(f"  💡 Check disk space - you may need to free up storage!")

    def _build_state_dict(self, model, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, epoch, global_step, best_val_loss, evals_without_improvement, elapsed_sec):
        
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
            'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
            'scheduler_encoder_state_dict': scheduler_encoder.state_dict(),
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


# ─── Optimizer State Compatibility ───

def _remap_opt_state_for_bnb(state_dict):
    """
    Convert a standard PyTorch AdamW optimizer state dict to the format expected
    by bitsandbytes AdamW8bit.

    PyTorch AdamW stores the first and second moments as:
        'exp_avg'    (first moment  / m1)
        'exp_avg_sq' (second moment / m2)

    bitsandbytes AdamW8bit stores the same values under different keys:
        'state1'     (first moment  / m1)
        'state2'     (second moment / m2)

    The values are numerically identical — only the key names differ.
    This function renames the keys so a checkpoint saved with standard AdamW
    can be loaded into AdamW8bit without losing any momentum history.

    If the state dict already uses 'state1'/'state2' (saved with AdamW8bit),
    it is returned unchanged.
    """
    new_state = {}
    for idx, param_state in state_dict['state'].items():
        s = dict(param_state)
        if 'exp_avg' in s:
            s['state1'] = s.pop('exp_avg')
        if 'exp_avg_sq' in s:
            s['state2'] = s.pop('exp_avg_sq')
        new_state[idx] = s
    return {'state': new_state, 'param_groups': state_dict['param_groups']}


# ─── CSV Logger ───

class CSVLogger:
    """Logs metrics to a CSV file with custom fieldnames."""

    def __init__(self, log_file, fieldnames):
        self.log_file = Path(log_file)
        self.fieldnames = fieldnames
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Write header if file doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row_dict):
        """Append a row to the CSV. Missing fields will be empty."""
        row_dict['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row_dict.get(k, '') for k in self.fieldnames})


# ─── Utility ───

def format_time(seconds):
    """Format seconds as Xh Ym Zs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def get_cuda_mem():
    """Returns (current_allocated_GB, peak_allocated_GB)."""
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        return current, peak
    return 0.0, 0.0

# %%
## Validation Function

@torch.no_grad()
def validate(model, val_loader, val_dataset, tokenizer, device,
             max_eval_batches, max_generate_samples, num_print_samples, val_gen_batch_size,
             val_beam_size=1, val_repetition_penalty=1.0, val_use_kv_cache=True):
    """
    Run validation: compute loss, perplexity, token accuracy,
    then generate text for BLEU & ROUGE-L using batched generation.

    Returns dict with all validation metrics.
    """
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
                keypoints=batch['keypoints'].to(device, non_blocking=True),
                keypoint_mask=batch['keypoint_mask'].to(device, non_blocking=True),
                token_ids=batch['token_ids'].to(device, non_blocking=True),
                text_attention_mask=batch['text_attention_mask'].to(device, non_blocking=True),
                return_loss=True,
            )

        total_loss += output['loss'].item()
        num_batches += 1

        # Token accuracy (excluding padding)
        logits = output['logits']                         # (B, L, vocab)
        preds = logits.argmax(dim=-1)                     # (B, L)
        labels = batch['token_ids'][:, 1:].to(device, non_blocking=True)     # (B, L) shifted
        mask = labels != tokenizer.pad_token_id
        total_correct += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

        loss_pbar.update(1)
    loss_pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, MAX_PPL_CAP))
    token_acc = (total_correct / max(total_tokens, 1)) * 100

    # ── Part 2: Generate Text for BLEU / ROUGE-L (Batched for speed) ──
    references = []
    hypotheses = []
    sample_pairs = []  # (ref, hyp) pairs for printing

    # Sample indices from validation set (use fixed seed for consistency)
    num_samples = min(max_generate_samples, len(val_dataset))
    sample_indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(42))[:num_samples].tolist()

    # Process in batches for faster generation
    gen_pbar = tqdm(total=num_samples, desc="  Val gen ", unit="sample",
                    leave=False, ncols=80, dynamic_ncols=False)
    for batch_start in range(0, num_samples, val_gen_batch_size):
        batch_end = min(batch_start + val_gen_batch_size, num_samples)
        batch_indices = sample_indices[batch_start:batch_end]
        batch_samples = [val_dataset[idx] for idx in batch_indices]

        # Extract and collate keypoints/masks (pad to max length in this batch)
        keypoints_list = [s['keypoints'] for s in batch_samples]
        keypoint_mask_list = [s['keypoint_mask'] for s in batch_samples]
        token_ids_list = [s['token_ids'] for s in batch_samples]

        max_frames = max(kp.shape[0] for kp in keypoints_list)
        batch_size = len(batch_samples)
        num_landmarks = keypoints_list[0].shape[1]

        # Pad keypoints and masks
        padded_keypoints = torch.zeros(batch_size, max_frames, num_landmarks, 2)
        padded_masks = torch.zeros(batch_size, max_frames, num_landmarks, dtype=torch.bool)

        for i, (kp, mask) in enumerate(zip(keypoints_list, keypoint_mask_list)):
            T = kp.shape[0]
            padded_keypoints[i, :T] = kp
            padded_masks[i, :T] = mask

        # Batch generate (beam search for higher-quality evaluation outputs)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            gen_output = model.generate(
                keypoints=padded_keypoints.to(device, non_blocking=True),
                keypoint_mask=padded_masks.to(device, non_blocking=True),
                max_new_tokens=50,
                temperature=0.0,
                top_k=50,
                repetition_penalty=val_repetition_penalty,
                beam_size=val_beam_size,
                use_kv_cache=val_use_kv_cache,
            )

        # Decode all outputs in batch
        batch_hypotheses = gen_output['generated_text']  # List of strings or single string
        if isinstance(batch_hypotheses, str):
            batch_hypotheses = [batch_hypotheses]

        # Extract references and store results
        for i, (tokens, hyp_text) in enumerate(zip(token_ids_list, batch_hypotheses)):
            ref_text = tokenizer.decode(tokens[1:-1], skip_special_tokens=True).strip()
            hyp_text = hyp_text.strip()

            references.append(ref_text)
            hypotheses.append(hyp_text)

            if len(sample_pairs) < num_print_samples:
                sample_pairs.append((ref_text, hyp_text))

        gen_pbar.update(batch_end - batch_start)
    gen_pbar.close()

    # Compute generation metrics
    bleu1 = compute_bleu(references, hypotheses, max_n=1)  # Early learning signal
    bleu2 = compute_bleu(references, hypotheses, max_n=2)  # Bigram precision
    bleu4 = compute_bleu(references, hypotheses, max_n=4)  # Strict metric
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
        'all_pairs': list(zip(references, hypotheses)),  # All (ref, hyp) pairs for CSV logging
        'num_eval_batches': num_batches,
        'num_gen_samples': num_samples,
    }

# %%
## Training Loop


def train(model, train_loader, val_loader, val_dataset, tokenizer,
          optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder,
          device, train_config, ckpt_manager, train_csv_logger, val_csv_logger, gen_samples_csv_logger, tb_writer,
          start_epoch=1, start_global_step=0, best_val_loss=float('inf'), start_evals_without_improvement=0, start_elapsed_sec=0.0):
    """Full training loop with all bells and whistles."""

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
    freeze_decoder = train_config.get('freeze_decoder', False)
    decoder_freeze_steps = train_config.get('decoder_freeze_steps', 2000)

    # Calculate total optimizer steps
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * num_epochs

    # State tracking
    global_step = start_global_step # optimizer steps
    evals_without_improvement = start_evals_without_improvement
    training_start = time.time() - start_elapsed_sec
    step_losses = []                # losses between log intervals
    step_grad_norms = []            # grad norms between log intervals
    log_step_start = time.time()

    # Reset peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()

    print("\n" + "=" * 60)
    print("🚀 TRAINING STARTED")
    print("=" * 60 + "\n")

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)

    # Check if we finished the epoch exactly at the checkpoint step
    steps_done_in_epoch = start_global_step % optimizer_steps_per_epoch
    if steps_done_in_epoch == 0 and start_global_step > 0:
        start_epoch += 1
        print(f"  ⏭️ Checkpoint was at exact end of Epoch {start_epoch - 1}. Advancing to Epoch {start_epoch}.")

    # ── Decoder Freeze Phase Setup ──
    # Collect the exact parameter objects that optimizer_decoder manages.
    # These are the same tensor references, so toggling requires_grad here affects
    # gradient computation immediately (no optimizer rebuild needed).
    decoder_params_for_freeze = [p for group in optimizer_decoder.param_groups for p in group['params']]

    # Determine initial freeze state.
    # Handles both fresh start (start_global_step=0) and mid-freeze resume correctly.
    decoder_is_frozen = freeze_decoder and (start_global_step < decoder_freeze_steps)

    if freeze_decoder:
        if decoder_is_frozen:
            for p in decoder_params_for_freeze:
                p.requires_grad = False
            remaining = decoder_freeze_steps - start_global_step
            print(f"\n  🔒 Decoder FROZEN | Will unfreeze at step {decoder_freeze_steps} ({remaining} steps remaining)")
            print(f"     Only Encoder + Cross-Attention will update during this phase.")
        else:
            # Resume past the freeze window — nothing to do
            print(f"\n  ✅ Decoder freeze phase already completed (resumed at step {start_global_step} ≥ {decoder_freeze_steps})")
    else:
        print(f"\n  ℹ️  Decoder freeze phase: DISABLED")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_microbatches = 0

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        # How many micro-batches to skip if resuming mid-epoch
        micro_steps_already_processed = 0
        if epoch == start_epoch and start_global_step > 0 and steps_done_in_epoch > 0:
            micro_steps_already_processed = steps_done_in_epoch * grad_accum_steps
            if micro_steps_already_processed > 0:
                print(f"  ⏭️ Resuming mid-epoch. Fast-forwarding {micro_steps_already_processed} micro-batches...")

        for micro_step, batch in enumerate(train_loader):
            # Mid-epoch skipping
            if epoch == start_epoch and micro_step < micro_steps_already_processed:
                continue

            # ── Forward pass with mixed precision ──
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(
                    keypoints=batch['keypoints'].to(device, non_blocking=True),
                    keypoint_mask=batch['keypoint_mask'].to(device, non_blocking=True),
                    token_ids=batch['token_ids'].to(device, non_blocking=True),
                    text_attention_mask=batch['text_attention_mask'].to(device, non_blocking=True),
                    return_loss=True,
                )

            # ── Optimizer step check ──
            is_accum_step = (micro_step + 1) % grad_accum_steps == 0
            is_last_step = (micro_step + 1) == len(train_loader)

            # Scale loss for gradient accumulation (handle partial last batch correctly)
            if is_last_step and not is_accum_step:
                # Last partial batch: only accumulate remaining steps
                actual_accum_steps = (micro_step + 1) % grad_accum_steps
            else:
                actual_accum_steps = grad_accum_steps

            loss = output['loss'] / actual_accum_steps
            loss.backward()

            epoch_loss += output['loss'].item()
            epoch_microbatches += 1

            if is_accum_step or is_last_step:
                # Gradient clipping (frozen params have no grad, so norm is over encoder only during freeze)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                ).item()

                # Always step encoder + cross-attention
                optimizer_encoder.step()
                scheduler_encoder.step()

                # Decoder optimizer and scheduler only step when NOT frozen.
                # The decoder scheduler runs on its own independent clock: it is never
                # stepped during the freeze phase, so its warmup → cosine decay starts
                # from step 0 the moment the decoder is unfrozen.
                if not decoder_is_frozen:
                    optimizer_decoder.step()
                    scheduler_decoder.step()

                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()

                global_step += 1

                # ── Unfreeze decoder once the freeze phase is complete ──
                # Check after incrementing: global_step now equals the number of completed steps.
                # When global_step reaches decoder_freeze_steps, the N frozen steps are done.
                if decoder_is_frozen and global_step >= decoder_freeze_steps:
                    decoder_is_frozen = False
                    for p in decoder_params_for_freeze:
                        p.requires_grad = True
                    optimizer_decoder.zero_grad()  # clean slate before first decoder update
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
                    encoder_lr = scheduler_encoder.get_last_lr()[0]
                    # Decoder scheduler has not been stepped during freeze — guard against that
                    decoder_lr = scheduler_decoder.get_last_lr()[0] if not decoder_is_frozen else 0.0
                    elapsed = time.time() - training_start
                    log_elapsed = time.time() - log_step_start
                    speed = len(step_losses) / max(log_elapsed, 1e-6)
                    cuda_mem, cuda_peak = get_cuda_mem()

                    # Calculate ETA
                    eta_seconds = (elapsed / global_step) * (total_optimizer_steps - global_step)

                    train_ppl = math.exp(min(avg_loss, MAX_PPL_CAP))

                    # Freeze-phase tag shown in the log line
                    freeze_tag = (
                        f" | DEC: FROZEN ({decoder_freeze_steps - global_step} steps left)"
                        if decoder_is_frozen else ""
                    )

                    print("=" * 180)
                    print(
                        f"[Step {global_step:>6d}/{total_optimizer_steps}] "
                        f"[Epoch {epoch:>2d}/{num_epochs}] "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"Train PPL: {train_ppl:.3f} | "
                        f"LR_enc: {encoder_lr:.4e} | "
                        f"LR_dec: {decoder_lr:.4e} | "
                        f"Train Grad Norm: {avg_grad_norm:.3f} | "
                        f"Speed: {speed:.3f} steps/s | "
                        f"VRAM: {cuda_mem:.3f}/{cuda_peak:.3f} GB (current/peak) | "
                        f"Elapsed: {format_time(elapsed)} | "
                        f"ETA: {format_time(eta_seconds)}"
                        f"{freeze_tag}"
                    )

                    train_csv_logger.log({
                        'global_step': global_step,
                        'epoch': epoch,
                        'train_loss': f"{avg_loss:.6f}",
                        'train_ppl': f"{train_ppl:.4f}",
                        'lr_encoder': f"{encoder_lr:.3e}",
                        'lr_decoder': f"{decoder_lr:.3e}",
                        'grad_norm': f"{avg_grad_norm:.4f}",
                        'cuda_mem_gb': f"{cuda_mem:.2f}",
                        'cuda_peak_gb': f"{cuda_peak:.2f}",
                        'steps_per_sec': f"{speed:.2f}",
                        'elapsed_sec': f"{elapsed:.1f}",
                    })

                    # ── TensorBoard Logging ──
                    tb_writer.add_scalar('Loss/Train', avg_loss, global_step)
                    tb_writer.add_scalar('Perplexity/Train', train_ppl, global_step)
                    tb_writer.add_scalar('GradNorm/Train', avg_grad_norm, global_step)
                    tb_writer.add_scalar('Training/DecoderFrozen', 1.0 if decoder_is_frozen else 0.0, global_step)
                    tb_writer.add_scalars('LearningRates', {
                        'Encoder': encoder_lr,
                        'Decoder': decoder_lr
                    }, global_step)

                    step_losses.clear()
                    step_grad_norms.clear()
                    log_step_start = time.time()

                # ── Validation ──
                # Before eval_warmup_threshold: evaluate every eval_every_warmup steps.
                # At and after the threshold: counter resets — evaluate every eval_every steps
                # counting from the threshold (i.e. when (global_step - eval_warmup_threshold) % eval_every == 0).
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
                        print(f"  Raw values: loss={val_metrics['val_loss']}, ppl={val_metrics['val_ppl']}, acc={val_metrics['token_acc']}")

                    print(f"  BLEU-1: {val_metrics['bleu1']:.4f} | "
                          f"BLEU-2: {val_metrics['bleu2']:.4f} | "
                          f"BLEU-4: {val_metrics['bleu4']:.4f} | "
                          f"ROUGE-L: {val_metrics['rouge_l']:.3f}%")
                    print(f"  (Evaluated on {val_metrics['num_eval_batches']} batches, "
                          f"generated {val_metrics['num_gen_samples']} samples)")

                    # Print sample generations
                    if val_metrics['sample_pairs']:
                        print(f"\n  Sample Generations:")
                        for i, (ref, hyp) in enumerate(val_metrics['sample_pairs'], 1):
                            print(f"    [{i}] REF: \"{ref}\"")
                            print(f"        HYP: \"{hyp}\"")

                    # CSV log
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

                    # ── TensorBoard Logging ──
                    tb_writer.add_scalar('Loss/Validation', val_metrics['val_loss'], global_step)
                    tb_writer.add_scalar('Perplexity/Validation', val_metrics['val_ppl'], global_step)
                    tb_writer.add_scalar('Metrics/TokenAccuracy', val_metrics['token_acc'], global_step)
                    tb_writer.add_scalar('Metrics/ROUGE-L', val_metrics['rouge_l'], global_step)
                    tb_writer.add_scalars('Metrics/BLEU', {
                        'BLEU-1': val_metrics['bleu1'],
                        'BLEU-2': val_metrics['bleu2'],
                        'BLEU-4': val_metrics['bleu4']
                    }, global_step)

                    # Log all generated samples to separate CSV
                    for ref_text, hyp_text in val_metrics['all_pairs']:
                        gen_samples_csv_logger.log({
                            'global_step': global_step,
                            'epoch': epoch,
                            'generated': hyp_text,
                            'reference': ref_text,
                            'elapsed_sec': f"{elapsed:.1f}",
                        })

                    # ── Best model check ──
                    val_loss = val_metrics['val_loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        evals_without_improvement = 0
                        state = ckpt_manager._build_state_dict(
                            model, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, epoch, global_step, best_val_loss, evals_without_improvement, time.time() - training_start
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

                    # Clear CUDA cache after validation to free temporary generation memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # ── Early stopping ──
                    if evals_without_improvement >= patience:
                        print("=" * 60)
                        print(f"🛑 EARLY STOPPING triggered at step {global_step}, "
                              f"epoch {epoch}")
                        print(f"   Best val loss: {best_val_loss:.4f}")
                        print("=" * 60)

                        total_time = time.time() - training_start
                        print(f"\n⏱️  Total training time: {format_time(total_time)}")
                        cuda_mem, cuda_peak = get_cuda_mem()
                        print(f"📊 Peak VRAM usage: {cuda_peak:.2f} GB")
                        tb_writer.close()  # Close TensorBoard logging gracefully
                        return best_val_loss

                # ── Periodic checkpoint ──
                if global_step % save_every == 0:
                    state = ckpt_manager._build_state_dict(
                        model, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, epoch, global_step, best_val_loss, evals_without_improvement, time.time() - training_start
                    )
                    ckpt_manager.save_periodic(state, global_step)
                    del state
                    torch.cuda.empty_cache()

        # ── Epoch summary ──
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / max(epoch_microbatches, 1)
        cuda_mem, cuda_peak = get_cuda_mem()

        print(f"\n{'═' * 60}")
        print(f"  📘 Epoch {epoch}/{num_epochs} Complete")
        print(f"  Avg Train Loss: {epoch_avg_loss:.4f} | "
              f"Epoch Time: {format_time(epoch_time)}")
        print(f"  VRAM: {cuda_mem:.1f}/{cuda_peak:.1f} GB | "
              f"Total Time: {format_time(time.time() - training_start)}")
        print(f"{'═' * 60}\n")

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


# %%
## Run Training


def main():
    """Main entry point for training."""
    global device

    # Fix for Windows multiprocessing with persistent workers
    try:
        mp.set_sharing_strategy('file_system')
    except:
        pass

    # Paths
    TRAIN_CSV = CONFIG['data_train_csv']
    KEYPOINTS_DIR = CONFIG['keypoints_train_dir']

    # Load CSV
    df = pd.read_csv(TRAIN_CSV, sep=CONFIG['csv_sep'])
    print(f'Total training examples: {len(df)}')
    print(f'Columns: {list(df.columns)}')

    # Pick first example
    sample_row = df.sample(1).iloc[0]
    sentence_name = sample_row['vid']
    sentence_text = sample_row['text']
    duration_sec = sample_row['duration_sec']

    print(f'\nSample: {sentence_name}')
    print(f'Text: "{sentence_text}"')
    print(f'Duration: {duration_sec:.2f}s')

    # Load keypoints (replace colons with dashes for Windows filenames)
    keypoint_filename = sentence_name.replace(':', '-')
    keypoint_path = KEYPOINTS_DIR / f'{keypoint_filename}.npz'

    if keypoint_path.exists():
        data = np.load(keypoint_path)
        keypoints = data['keypoints']  # (T, N, 3)
        mask = data['mask']  # (T, N)

        T, N, C = keypoints.shape

        print(f'\nKeypoint file: {keypoint_path.name}')
        print(f'  Shape: (T={T} frames, N={N} landmarks, C={C} coords)')
        print(f'  Valid landmarks: {mask.sum()} / {mask.size} ({mask.sum()/mask.size*100:.1f}%)')
        print(f'  Coordinate ranges:')
        valid_kp = keypoints[mask == 1]
        print(f'    X: [{valid_kp[:, 0].min():.3f}, {valid_kp[:, 0].max():.3f}]')
        print(f'    Y: [{valid_kp[:, 1].min():.3f}, {valid_kp[:, 1].max():.3f}]')
        print(f'    Z: [{valid_kp[:, 2].min():.3f}, {valid_kp[:, 2].max():.3f}]')
    else:
        print(f'\nKeypoint file not found: {keypoint_path}')

    # Paths
    TRAIN_CSV = CONFIG['data_train_csv']
    KEYPOINTS_DIR = CONFIG['keypoints_train_dir']

    # Load CSV
    df = pd.read_csv(TRAIN_CSV, sep=CONFIG['csv_sep'])

    # Build manifest: list of (sentence_name, text, duration, keypoint_path)
    manifest = []
    missing = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Building manifest'):
        sentence_name = row['vid']
        text = row['text']
        duration = row['duration_sec']
        # Replace colons with dashes for Windows-compatible filenames
        keypoint_filename = sentence_name.replace(':', '-')
        keypoint_path = KEYPOINTS_DIR / f'{keypoint_filename}.npz'

        if keypoint_path.exists():
            manifest.append({
                'sentence_name': sentence_name,
                'text': text,
                'duration': duration,
                'keypoint_path': str(keypoint_path)
            })
        else:
            missing.append(sentence_name)

    print(f'\nDataset manifest:')
    print(f'  Total CSV rows: {len(df)}')
    print(f'  Found keypoints: {len(manifest)}')
    print(f'  Missing keypoints: {len(missing)}')
    print(f'  Success rate: {len(manifest)/len(df)*100:.1f}%')

    if len(missing) > 0:
        print(f'\nSample missing files (first 5):')
        for name in missing[:5]:
            print(f'  {name}')

    VAL_CSV = CONFIG['data_val_csv']
    KEYPOINTS_VAL_DIR = CONFIG['keypoints_val_dir']

    df_val = pd.read_csv(VAL_CSV, sep=CONFIG['csv_sep'])

    val_manifest = []
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc='Building val manifest'):
        sentence_name = row['vid']
        text = row['text']
        duration = row['duration_sec']
        # Replace colons with dashes for Windows-compatible filenames
        keypoint_filename = sentence_name.replace(':', '-')
        keypoint_path = KEYPOINTS_VAL_DIR / f'{keypoint_filename}.npz'

        if keypoint_path.exists():
            val_manifest.append({
                'sentence_name': sentence_name,
                'text': text,
                'duration': duration,
                'keypoint_path': str(keypoint_path)
            })

    print(f'Val manifest: {len(val_manifest)} samples')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['decoder_model_name'])

    # Llama 3.2 has no pad token by default. We must add a dedicated one.
    # Using eos_token as pad would cause CrossEntropyLoss(ignore_index=eos_id) to
    # also suppress the real EOS token in labels — the model would never learn to stop.
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    print("Tokenizer loaded:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"\nSpecial tokens:")
    print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

    # Test tokenization on a sample sentence from dataset
    sample_text = manifest[0]['text']
    tokens = tokenizer(sample_text, return_tensors='pt')

    print(f"\nSample text: \"{sample_text}\"")
    print(f"Tokenized:")
    print(f"  Input IDs shape: {tokens['input_ids'].shape}")
    print(f"  Input IDs: {tokens['input_ids'][0].tolist()[:20]}{'...' if len(tokens['input_ids'][0]) > 20 else ''}")
    print(f"  Decoded back: \"{tokenizer.decode(tokens['input_ids'][0])}\"")

    # Show token breakdown
    print(f"\nToken breakdown (first 10):")
    for i, token_id in enumerate(tokens['input_ids'][0][:10].tolist()):
        token_str = tokenizer.decode([token_id])
        print(f"  {i}: {token_id:5d} -> '{token_str}'")

    # Llama tokenizer: BOS/EOS are built-in; <pad> was added explicitly above
    print("Special tokens verification:")
    print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id}) ✓")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id}) ✓")
    print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id}) ✓ (added explicitly)")
    print(f"  UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")
    print(f"\nVocab size: {len(tokenizer)} (128256 base + 1 <pad> token)")

    # Test tokenization with special tokens
    sample_text = manifest[0]['text']
    tokens = tokenizer(sample_text, add_special_tokens=True, return_tensors='pt')

    print(f"\nSample: \"{sample_text}\"")
    print(f"Token IDs (first 10): {tokens['input_ids'][0][:10].tolist()}")
    print(f"First token (BOS): {tokens['input_ids'][0][0].item()} = {tokenizer.bos_token}")
    print(f"Last token (EOS): {tokens['input_ids'][0][-1].item()} = {tokenizer.eos_token}")

    print(f"\nNote: BOS/EOS will be added explicitly in Dataset class")
    print(f"      PAD (id={tokenizer.pad_token_id}) will be used for batching")

    # Create datasets with augmentation enabled for training
    train_dataset = SignLanguageDataset(manifest, tokenizer, train=True, augment_config=CONFIG)
    val_dataset = SignLanguageDataset(val_manifest, tokenizer, train=False)  # No augmentation for validation
    print(f'Train dataset: {len(train_dataset)} samples')
    print(f'Val dataset: {len(val_dataset)} samples')
    print(f'Max keypoint frames: {MAX_KEYPOINT_FRAMES}')
    print(f'Max text tokens: {MAX_TEXT_TOKENS} (excluding BOS/EOS)')

    # Test: load one sample
    sample = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Keypoints shape: {sample['keypoints'].shape}")
    print(f"  Keypoint mask shape: {sample['keypoint_mask'].shape}, dtype: {sample['keypoint_mask'].dtype}")
    print(f"  Token IDs shape: {sample['token_ids'].shape}")
    print(f"  Token IDs: {sample['token_ids'][:12].tolist()}...")
    print(f"  First token (BOS): {sample['token_ids'][0].item()} (should be {tokenizer.bos_token_id})")
    print(f"  Last token (EOS): {sample['token_ids'][-1].item()} (should be {tokenizer.eos_token_id})")
    print(f"  Decoded text: \"{tokenizer.decode(sample['token_ids'])}\"")
    print(f"\nDuring training:")
    print(f"  Decoder input = token_ids[:-1] (BOS to second-to-last)")
    print(f"  Labels = token_ids[1:] (second to EOS)")

    # Create partial function with pad_token_id for DataLoader
    collate_fn_with_tokenizer = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    # Test collate function with a small batch
    test_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_with_tokenizer)
    batch = next(iter(test_loader))

    print("Batch shapes:")
    print(f"  Keypoints: {batch['keypoints'].shape}")
    print(f"  Keypoint mask: {batch['keypoint_mask'].shape}, dtype: {batch['keypoint_mask'].dtype}")
    print(f"  Token IDs: {batch['token_ids'].shape}")
    print(f"  Text attention mask: {batch['text_attention_mask'].shape}, dtype: {batch['text_attention_mask'].dtype}")

    print(f"\nBatch details:")
    print(f"  Batch size: {batch['keypoints'].shape[0]}")
    print(f"  Max keypoint frames in batch: {batch['keypoints'].shape[1]}")
    print(f"  Max text length in batch: {batch['token_ids'].shape[1]}")
    print(f"  Num landmarks: {batch['keypoints'].shape[2]}")

    print(f"\nSample token IDs from batch[0]:")
    print(f"  First 10: {batch['token_ids'][0, :10].tolist()}")
    print(f"  Attention mask first 10: {batch['text_attention_mask'][0, :10].tolist()}")

    # Load pretrained decoder in fp16
    model_name = CONFIG['decoder_model_name']
    _attn_impl = "sdpa" if TRAIN_CONFIG.get('use_sdpa', True) else "eager"
    decoder = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device, attn_implementation=_attn_impl)

    print(f"Loaded: {model_name}")
    print(f"Original vocab size: {decoder.config.vocab_size}")

    # Verify vocab matches tokenizer (Llama base has 128256; we added 1 <pad> token → 128257)
    if decoder.config.vocab_size != len(tokenizer):
        print(f"Note: Resizing embeddings from {decoder.config.vocab_size} to {len(tokenizer)}")
        decoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    else:
        print(f"Vocab size matches tokenizer: {len(tokenizer)} tokens ✓")


    # print(decoder)
    
    # for name, _ in decoder.named_parameters():
    #     print(name)

    # for name, module in decoder.named_modules():
    #     print(f"{name}: {type(module).__name__}")

    # Configure LoRA
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

    # Apply LoRA
    decoder = get_peft_model(decoder, lora_config)

    # Manually enable norm layers (not covered by LoRA or modules_to_save)
    # Llama has 2 norm layers per decoder layer (input_layernorm, post_attention_layernorm) + 1 final norm
    norm_params_count = 0
    for name, param in decoder.named_parameters():
        if 'norm' in name:
            param.requires_grad = True
            norm_params_count += param.numel()

    print("LoRA Configuration:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Target modules (LoRA): {lora_config.target_modules}")
    print(f"  Modules to save (fully trainable): {lora_config.modules_to_save}")
    print(f"  Norm layers: manually set to trainable ({norm_params_count / 1e6:.2f}M params)")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Bias: {lora_config.bias}")

    # Print trainable parameters
    print("\nTrainable parameters breakdown:")
    decoder.print_trainable_parameters()

    print(f"\nLlama 3.2-1B Decoder architecture:")
    print(f"  Hidden size (d_model): {decoder.config.hidden_size}")
    print(f"  Num layers: {decoder.config.num_hidden_layers}")
    print(f"  Num attention heads: {decoder.config.num_attention_heads}")

    print(f"\nHybrid LoRA strategy:")
    print(f"  ✓ LoRA adapters: Self-attention + MLP (r=8, α=16)")
    print(f"  ✓ Fully trainable: All norm layers (2 per layer) — embed_tokens & lm_head FROZEN")
    print(f"  ✓ Cross-attention: Will be fully trainable when added (max learning capacity)")

    # Create projection layer
    projection = KeypointProjection(
        num_landmarks=CONFIG['num_landmarks'],
        coord_dim=CONFIG['coord_dim'],
        d_model=CONFIG['projection_d_model'],
        hidden_dim=CONFIG['projection_hidden_dim'],
    ).to(torch.bfloat16)

    # Count parameters
    proj_params = sum(p.numel() for p in projection.parameters())
    print(f"MLP Projection Layer:")
    print(f"  Architecture: Linear(232→512) → ReLU → Dropout → Linear(512→512) → LayerNorm")
    print(f"  Input: (B, T, 116, 2) -> flatten to (B, T, 232)")
    print(f"  Output: (B, T, 512) - normalized")
    print(f"  Parameters: {proj_params / 1e6:.2f}M")

    # Test with sample batch
    test_keypoints = batch['keypoints'].to(torch.bfloat16)  # (4, 160, 116, 2)
    test_output = projection(test_keypoints)

    print(f"\nTest:")
    print(f"  Input shape: {test_keypoints.shape}")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Output dtype: {test_output.dtype}")
    print(f"  Output mean: {test_output.mean().item():.4f}, std: {test_output.std().item():.4f}")
    print(f"  (LayerNorm ensures stable distribution for encoder input)")


    # Create encoder
    encoder = TransformerEncoder(
        d_model=CONFIG['encoder_d_model'],
        num_heads=CONFIG['encoder_num_heads'],
        num_layers=CONFIG['encoder_num_layers'],
        feedforward_dim=CONFIG['encoder_feedforward_dim'],
        dropout=CONFIG['encoder_dropout']
    ).to(torch.bfloat16)
    encoder.use_gradient_checkpointing = TRAIN_CONFIG.get('use_gradient_checkpointing_encoder', False)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Transformer Encoder:")
    print(f"  d_model: 512")
    print(f"  num_heads: 8")
    print(f"  num_layers: 6")
    print(f"  feedforward_dim: 2048")
    print(f"  dropout: 0.1")
    print(f"  norm_first: True (pre-norm for stability)")
    print(f"  Parameters: {encoder_params / 1e6:.2f}M")

    # Test with projected keypoints
    test_projected = projection(batch['keypoints'].to(torch.bfloat16))  # (4, 160, 512)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {test_projected.shape}")

    # Create padding mask from keypoint mask
    # keypoint_mask: (B, T, N) -> (B, T) by checking if all landmarks are invalid
    # True = padding (to ignore), False = real data
    keypoint_padding_mask = ~(batch['keypoint_mask'].any(dim=-1))  # (4, 160)
    print(f"  Padding mask shape: {keypoint_padding_mask.shape}")
    print(f"  Padding positions: {keypoint_padding_mask.sum().item()} / {keypoint_padding_mask.numel()}")

    # Forward pass
    encoder_output = encoder(test_projected, src_key_padding_mask=keypoint_padding_mask)
    print(f"  Output shape: {encoder_output.shape}")
    print(f"  Output dtype: {encoder_output.dtype}")
    print(f"  Output mean: {encoder_output.mean().item():.4f}, std: {encoder_output.std().item():.4f}")

    print(f"\nEncoder ready to produce hidden states for decoder cross-attention!")


    # Configuration for Llama 3.2-1B
    ENCODER_DIM = CONFIG['encoder_d_model']
    BOTTLENECK_DIM = CONFIG['bottleneck_dim']
    NUM_HEADS = CONFIG['cross_attn_num_heads']
    TOTAL_LAYERS = CONFIG['decoder_num_layers']

    # Determine which layers get cross-attention (13/18 layers)
    # First 5: 0, 1, 2, 3, 4
    # Every-other middle: 6, 8, 10, 12, 14
    # Last 3: 15, 16, 17
    layers_with_cross_attn = CONFIG['cross_attn_layers']
    # layers_with_cross_attn = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 16, 17]

    # Weight sharing pairs: (layer_a, layer_b) share the same cross-attention module
    # 4 pairs total
    weight_sharing_pairs = CONFIG['weight_sharing_pairs']

    print(f"Optimized Cross-Attention Configuration for Llama 3.2-1B:")
    print(f"  Encoder dim: {ENCODER_DIM}, Bottleneck dim: {BOTTLENECK_DIM}")
    print(f"  Decoder hidden size: {CONFIG['decoder_hidden_size']} (Llama), Num heads: {NUM_HEADS} (64 dims/head)")
    print(f"  Total decoder layers: {TOTAL_LAYERS}")
    print(f"  Layers with cross-attention: {len(layers_with_cross_attn)}/{TOTAL_LAYERS}")
    print(f"  Cross-attn layers: {layers_with_cross_attn}")
    print(f"  Weight sharing: {len(weight_sharing_pairs)} pairs")
    print(f"  Pairs: {weight_sharing_pairs}")

    # Create cross-attention modules
    cross_attn_modules = {}
    shared_modules = {}

    # Create shared modules first
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

    # Create unique modules for non-shared layers
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

    # Calculate unique modules
    unique_modules = len(set(id(m) for m in cross_attn_modules.values()))

    print(f"\nCross-attention modules created:")
    print(f"  Layers with cross-attn: {len(layers_with_cross_attn)}")
    print(f"  Unique modules: {unique_modules}")
    print(f"  Shared instances: {len(layers_with_cross_attn) - unique_modules}")

    # Create encoder projection
    encoder_projection = EncoderProjection(encoder_dim=ENCODER_DIM).to(torch.bfloat16)

    # Count parameters
    cross_attn_params = sum(
        p.numel() for module in set(cross_attn_modules.values())
        for p in module.parameters()
    )

    encoder_proj_params = sum(p.numel() for p in encoder_projection.parameters())

    print(f"\nTrainable parameters:")
    print(f"  Cross-attention modules: {cross_attn_params / 1e6:.3f}M")
    print(f"  Encoder projection: {encoder_proj_params / 1e6:.5f}M")
    print(f"  Total new params: {(cross_attn_params + encoder_proj_params) / 1e6:.3f}M")
    print(f"  Sample/param ratio: {20000 / cross_attn_params:.6f}")

    print(f"\n✅ Final Optimized Architecture for Llama 3.2-1B:")
    print(f"  • Decoder: {CONFIG['decoder_hidden_size']} hidden, {TOTAL_LAYERS} layers")
    print(f"  • Bottleneck: {BOTTLENECK_DIM} dims (Query: {CONFIG['decoder_hidden_size']}→{BOTTLENECK_DIM}, Key/Value: {ENCODER_DIM}→{BOTTLENECK_DIM})")
    print(f"  • Cross-attention @ {BOTTLENECK_DIM} dims, Output: {BOTTLENECK_DIM}→{CONFIG['decoder_hidden_size']}")
    print(f"  • Selective: {len(layers_with_cross_attn)}/{TOTAL_LAYERS} layers, Weight sharing: {len(weight_sharing_pairs)} pairs → {unique_modules} unique")

    # Wrap decoder layers with cross-attention
    print("Wrapping decoder layers with LlamaDecoderLayerWithOptionalCrossAttention...")
    num_layers = len(decoder.base_model.model.model.layers)
    for i in range(num_layers):
        original_layer = decoder.base_model.model.model.layers[i]
        cross_attn_module = cross_attn_modules.get(i, None)
        decoder.base_model.model.model.layers[i] = LlamaDecoderLayerWithOptionalCrossAttention(
            original_layer,
            cross_attn_module=cross_attn_module
        )
    print(f"✓ Wrapped {num_layers} layers")

    # Apply the patch
    _dec_ckpt = TRAIN_CONFIG.get('use_gradient_checkpointing_decoder', False)
    patch_llama_model_forward(decoder, use_gradient_checkpointing=_dec_ckpt)
    print(f"✓ Patched Llama with standard causal attention + encoder-decoder support (decoder grad checkpointing: {'enabled' if _dec_ckpt else 'disabled'})")



    # Create model
    model = SignLanguageTranslationModel(
        keypoint_projection=projection,
        encoder=encoder,
        encoder_projection=encoder_projection,
        decoder=decoder,
        tokenizer=tokenizer,
    ).to(device)

    # Compile model (components already in bfloat16)
    # model = torch.compile(model, backend="aot_eager") disabled to avoid issues

    print("\n✅ Complete Encoder-Decoder Model Created!")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable: {total_params / 1e6:.2f}M params")
    
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters())

    def count_trainable_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("📊 DETAILED MODEL PARAMETER SUMMARY")
    print("=" * 60)

    components = {
        "Keypoint Projection": model.keypoint_projection,
        "Transformer Encoder": model.encoder,
        "Encoder Projection": model.encoder_projection,
        "Decoder (Llama + LoRA + Cross-Attn)": model.decoder,
    }

    total_all = 0
    total_trainable_all = 0

    for name, component in components.items():
        total = count_parameters(component)
        trainable = count_trainable_parameters(component)
        total_all += total
        total_trainable_all += trainable

        print(f"{name}:")
        print(f"  Total:     {total / 1e6:>8.2f}M")
        print(f"  Trainable: {trainable / 1e6:>8.2f}M  ({(trainable/max(total, 1))*100:>5.1f}%)")
        print("-" * 60)

    print("OVERALL MODEL:")
    print(f"  Total:     {total_all / 1e6:>8.2f}M")
    print(f"  Trainable: {total_trainable_all / 1e6:>8.2f}M  ({(total_trainable_all/max(total_all, 1))*100:>5.1f}%)")
    print("=" * 60 + "\n")

    print("TRAIN_CONFIG loaded ✓")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k}: {v}")

    # %%
    # DataLoaders with configurable workers (auto-capped to CPU count)
    train_workers = TRAIN_CONFIG['train_num_workers']
    val_workers = TRAIN_CONFIG['val_num_workers']
    train_prefetch_factor = TRAIN_CONFIG['train_prefetch_factor']
    val_prefetch_factor = TRAIN_CONFIG['val_prefetch_factor']   

    if TRAIN_CONFIG.get('use_bucket_batching', True):
        train_sampler = BucketBatchSampler(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
        val_sampler   = BucketBatchSampler(val_dataset,   batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
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

    print(f"\n📦 DataLoaders:")
    print(f"  Train workers: {train_workers} (config: {TRAIN_CONFIG['train_num_workers']})")
    print(f"  Val workers: {val_workers} (config: {TRAIN_CONFIG['val_num_workers']})")

    # Separate optimizers for encoder and decoder with different LRs
    # Optimizer 1: Encoder + Cross-attention (new, untrained components)
    encoder_params = list(model.keypoint_projection.parameters()) + \
                     list(model.encoder.parameters()) + \
                     list(model.encoder_projection.parameters())

    # Collect cross-attn params (deduplicate weight-shared modules)
    seen_ids = set()
    cross_attn_params = []
    for layer in model.decoder.base_model.model.model.layers:
        if hasattr(layer, 'cross_attn_module') and layer.cross_attn_module is not None:
            for p in layer.cross_attn_module.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    cross_attn_params.append(p)

    encoder_all_params = encoder_params + cross_attn_params

    # 8-bit AdamW (bitsandbytes): quantises the first/second moments to 8-bit blocks.
    # The actual weight update is still computed in fp32, so convergence and final
    # quality are identical to standard AdamW. Saves ~480 MB VRAM in optimizer states.
    if TRAIN_CONFIG.get('use_8bit_adam', False):
        if _BNB_AVAILABLE:
            _OptimizerClass = bnb.optim.AdamW8bit  # type: ignore[attr-defined]
            print("  Using 8-bit AdamW (bitsandbytes) — saves ~480 MB VRAM from optimizer states.")
        else:
            _OptimizerClass = AdamW
            print("  ⚠️  use_8bit_adam=True but bitsandbytes is not installed. Falling back to AdamW.")
            print("       Install with: pip install bitsandbytes")
    else:
        _OptimizerClass = AdamW

    optimizer_encoder = _OptimizerClass(
        encoder_all_params,
        lr=TRAIN_CONFIG['encoder_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['adam_betas']
    )

    # Optimizer 2: All decoder trainable params (LoRA adapters + norm layers)
    # Exclude cross_attn params (already in encoder optimizer)
    # Note: embed_tokens & lm_head are kept frozen (not in modules_to_save)
    cross_attn_param_ids = set(id(p) for p in cross_attn_params)
    decoder_params_with_names = [(n, p) for n, p in model.decoder.named_parameters()
                                  if p.requires_grad and id(p) not in cross_attn_param_ids]
    decoder_params = [p for n, p in decoder_params_with_names]

    # Debug: Check if embed_tokens or lm_head snuck in
    decoder_param_names = [n for n, p in decoder_params_with_names]
    has_embed = any('embed_tokens' in n for n in decoder_param_names)
    has_lm_head = any('lm_head' in n for n in decoder_param_names)

    if has_embed or has_lm_head:
        print(f"\n⚠️  WARNING: embed_tokens or lm_head found in decoder optimizer!")
        print(f"  embed_tokens: {has_embed}, lm_head: {has_lm_head}")
        print(f"  This should NOT happen with lora_modules_to_save=[]")
        print(f"  Affected params:")
        for n in decoder_param_names:
            if 'embed_tokens' in n or 'lm_head' in n:
                print(f"    - {n}")

    optimizer_decoder = _OptimizerClass(
        decoder_params,
        lr=TRAIN_CONFIG['decoder_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['adam_betas']
    )

    # ── Detailed Parameter Breakdown ──
    def numel(params):
        return sum(p.numel() for p in params)

    def numel_named(named_params, condition):
        return sum(p.numel() for n, p in named_params if condition(n))

    kp_n   = numel(model.keypoint_projection.parameters())
    enc_n  = numel(model.encoder.parameters())
    eproj_n = numel(model.encoder_projection.parameters())
    xattn_n = numel(cross_attn_params)
    enc_total_n = numel(encoder_all_params)

    lora_a_n = numel_named(decoder_params_with_names, lambda n: 'lora_A' in n)
    lora_b_n = numel_named(decoder_params_with_names, lambda n: 'lora_B' in n)
    norm_n   = numel_named(decoder_params_with_names, lambda n: 'norm' in n.lower())
    embed_n  = numel_named(decoder_params_with_names, lambda n: 'embed_tokens' in n)
    lmhead_n = numel_named(decoder_params_with_names, lambda n: 'lm_head' in n)
    other_n  = numel(decoder_params) - lora_a_n - lora_b_n - norm_n - embed_n - lmhead_n
    dec_total_n = numel(decoder_params)

    grand_total_n = enc_total_n + dec_total_n

    # Tensor counts (number of distinct parameter matrices, for reference)
    kp_t    = sum(1 for _ in model.keypoint_projection.parameters())
    enc_t   = sum(1 for _ in model.encoder.parameters())
    eproj_t = sum(1 for _ in model.encoder_projection.parameters())
    xattn_t = len(cross_attn_params)
    lora_a_t = sum(1 for n, _ in decoder_params_with_names if 'lora_A' in n)
    lora_b_t = sum(1 for n, _ in decoder_params_with_names if 'lora_B' in n)
    norm_t   = sum(1 for n, _ in decoder_params_with_names if 'norm' in n.lower())

    print(f"\n{'=' * 72}")
    print(f"📊 OPTIMIZER PARAMETER BREAKDOWN")
    print(f"{'=' * 72}")

    print(f"\nENCODER OPTIMIZER  (LR: {TRAIN_CONFIG['encoder_lr']:.2e})")
    print(f"  {'Component':<30} {'Tensors':>8}  {'Parameters':>12}")
    print(f"  {'─' * 54}")
    print(f"  {'Keypoint Projection':<30} {kp_t:>8}  {kp_n / 1e6:>10.3f}M")
    print(f"  {'Transformer Encoder':<30} {enc_t:>8}  {enc_n / 1e6:>10.3f}M")
    print(f"  {'Encoder Projection':<30} {eproj_t:>8}  {eproj_n / 1e6:>10.5f}M")
    print(f"  {'Cross-Attention Modules':<30} {xattn_t:>8}  {xattn_n / 1e6:>10.3f}M")
    print(f"  {'─' * 54}")
    print(f"  {'TOTAL':<30} {kp_t+enc_t+eproj_t+xattn_t:>8}  {enc_total_n / 1e6:>10.3f}M")

    print(f"\nDECODER OPTIMIZER  (LR: {TRAIN_CONFIG['decoder_lr']:.2e})")
    print(f"  {'Component':<30} {'Tensors':>8}  {'Parameters':>12}")
    print(f"  {'─' * 54}")
    print(f"  {'LoRA A matrices':<30} {lora_a_t:>8}  {lora_a_n / 1e6:>10.3f}M")
    print(f"  {'LoRA B matrices':<30} {lora_b_t:>8}  {lora_b_n / 1e6:>10.3f}M")
    print(f"  {'Norm layers':<30} {norm_t:>8}  {norm_n / 1e6:>10.4f}M")
    if embed_n > 0:
        print(f"  {'⚠️  embed_tokens':<30} {'—':>8}  {embed_n / 1e6:>10.3f}M  ← SHOULD BE 0!")
    if lmhead_n > 0:
        print(f"  {'⚠️  lm_head':<30} {'—':>8}  {lmhead_n / 1e6:>10.3f}M  ← SHOULD BE 0!")
    if other_n > 0:
        print(f"  {'Other':<30} {'—':>8}  {other_n / 1e6:>10.3f}M")
    print(f"  {'─' * 54}")
    print(f"  {'TOTAL':<30} {lora_a_t+lora_b_t+norm_t:>8}  {dec_total_n / 1e6:>10.3f}M")

    print(f"\n{'─' * 72}")
    print(f"  {'GRAND TOTAL (both optimizers)':<30}          {grand_total_n / 1e6:>10.3f}M")
    print(f"  {'Cross-check vs model summary':<30}          {total_trainable_all / 1e6:>10.3f}M")
    match = "✓ match" if abs(grand_total_n - total_trainable_all) < 1000 else "⚠️  MISMATCH — check for frozen params leaking into optimizers"
    print(f"  {match}")
    print(f"{'=' * 72}\n")

    # Separate schedulers for each optimizer
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / TRAIN_CONFIG['grad_accum_steps'])
    total_optimizer_steps = optimizer_steps_per_epoch * TRAIN_CONFIG['num_epochs']
    warmup_steps = TRAIN_CONFIG['warmup_steps']
    # Calculate cosine decay steps (to end of training)
    cosine_decay_steps = total_optimizer_steps - warmup_steps

    # Scheduler 1: Encoder + Cross-attention (2-stage: warmup → cosine)
    warmup_encoder = LinearLR(
        optimizer_encoder,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_encoder = CosineAnnealingLR(
        optimizer_encoder,
        T_max=cosine_decay_steps,
        eta_min=TRAIN_CONFIG['encoder_min_lr'],
    )
    scheduler_encoder = SequentialLR(
        optimizer_encoder,
        schedulers=[warmup_encoder, cosine_encoder],
        milestones=[warmup_steps],
    )

    # Scheduler 2: Decoder/LoRA (2-stage: warmup → cosine)
    # If freeze_decoder is enabled, the decoder scheduler runs on an independent clock:
    # it is not stepped during the freeze phase, so its warmup → cosine decay covers
    # only the remaining training window after the freeze ends.
    if TRAIN_CONFIG.get('freeze_decoder', False):
        _freeze_steps = TRAIN_CONFIG['decoder_freeze_steps']
        # Effective decoder training steps = total steps minus the frozen steps
        decoder_effective_steps = max(total_optimizer_steps - _freeze_steps, 1)
        decoder_cosine_decay_steps = max(decoder_effective_steps - warmup_steps, 1)
    else:
        decoder_cosine_decay_steps = cosine_decay_steps

    warmup_decoder = LinearLR(
        optimizer_decoder,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_decoder = CosineAnnealingLR(
        optimizer_decoder,
        T_max=decoder_cosine_decay_steps,
        eta_min=TRAIN_CONFIG['decoder_min_lr'],
    )
    scheduler_decoder = SequentialLR(
        optimizer_decoder,
        schedulers=[warmup_decoder, cosine_decoder],
        milestones=[warmup_steps],
    )

    print(f"\n📈 LR Schedules (2-stage):")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  Encoder  cosine decay: {cosine_decay_steps} steps (full training window)")
    if TRAIN_CONFIG.get('freeze_decoder', False):
        print(f"  Decoder  cosine decay: {decoder_cosine_decay_steps} steps (post-freeze window, starts at step {TRAIN_CONFIG['decoder_freeze_steps']})")
    else:
        print(f"  Decoder  cosine decay: {decoder_cosine_decay_steps} steps (full training window)")
    print(f"  Encoder:  {TRAIN_CONFIG['encoder_lr']:.2e} → {TRAIN_CONFIG['encoder_min_lr']:.2e}")
    print(f"  Decoder:  {TRAIN_CONFIG['decoder_lr']:.2e} → {TRAIN_CONFIG['decoder_min_lr']:.2e}")

    print("Optimizer & Scheduler created ✓")

    # %%

    steps_per_epoch_micro = len(train_loader)
    steps_per_epoch_optim = math.ceil(steps_per_epoch_micro / TRAIN_CONFIG['grad_accum_steps'])
    total_micro_steps = steps_per_epoch_micro * TRAIN_CONFIG['num_epochs']
    total_optim_steps = steps_per_epoch_optim * TRAIN_CONFIG['num_epochs']

    print("=" * 60)
    print("STEP CALCULATIONS")
    print("=" * 60)
    print(f"  Micro-batch steps per epoch:   {steps_per_epoch_micro}")
    print(f"  Optimizer steps per epoch:     {steps_per_epoch_optim}")
    print(f"  Total micro-batch steps:       {total_micro_steps}  ({TRAIN_CONFIG['num_epochs']} epochs × {steps_per_epoch_micro} steps)")
    print(f"  Total optimizer steps:         {total_optim_steps}  ({steps_per_epoch_optim} per epoch × {TRAIN_CONFIG['num_epochs']})")
    print("=" * 60)

    # %%
    ## Training Info

    effective_batch_size = TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['grad_accum_steps']

    print("=" * 60)
    print("TRAINING OVERVIEW")
    print("=" * 60)
    print(f"  Train samples:           {len(train_dataset)}")
    print(f"  Val samples:             {len(val_dataset)}")
    print(f"  Batch size:              {TRAIN_CONFIG['batch_size']}")
    print(f"  Gradient accumulation:   {TRAIN_CONFIG['grad_accum_steps']} steps")
    print(f"  Effective batch size:    {effective_batch_size}")
    print(f"  Steps per epoch:         {steps_per_epoch} micro-batches")
    print(f"  Optimizer steps/epoch:   {optimizer_steps_per_epoch}")
    print(f"  Total optimizer steps:   {total_optimizer_steps}")
    print(f"  Num epochs:              {TRAIN_CONFIG['num_epochs']}")
    print(f"  Warmup steps:            {warmup_steps}")
    print(f"  LR decay epochs:         {TRAIN_CONFIG['num_epochs']} (all the way to end)")
    print(f"  Encoder LR:              {TRAIN_CONFIG['encoder_lr']:.2e} → {TRAIN_CONFIG['encoder_min_lr']:.2e}")
    print(f"  Decoder LR:              {TRAIN_CONFIG['decoder_lr']:.2e} → {TRAIN_CONFIG['decoder_min_lr']:.2e}")
    print(f"  Mixed precision:         bfloat16")
    print(f"  Grad clipping:           max_norm={TRAIN_CONFIG['max_grad_norm']}")
    print(f"  Early stopping:          patience={TRAIN_CONFIG['early_stopping_patience']}")
    print("=" * 60)

    # %%
    # Initialize managers
    ckpt_manager = CheckpointManager(
        checkpoint_dir=TRAIN_CONFIG['checkpoint_dir'],
        keep_last_n=TRAIN_CONFIG['keep_last_n_checkpoints'],
    )

    # Separate CSV loggers for training and validation
    train_csv_logger = CSVLogger(
        log_file=TRAIN_CONFIG['train_log_file'],
        fieldnames=[
            'timestamp', 'global_step', 'epoch',
            'train_loss', 'train_ppl', 'lr_encoder', 'lr_decoder', 'grad_norm',
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

    print(f"Checkpoints dir:  {TRAIN_CONFIG['checkpoint_dir'].resolve()}")
    print(f"Train log:        {TRAIN_CONFIG['train_log_file'].resolve()}")
    print(f"Val log:          {TRAIN_CONFIG['val_log_file'].resolve()}")
    print(f"Gen samples log:  {TRAIN_CONFIG['gen_samples_log_file'].resolve()}")

    start_epoch = 1
    start_global_step = 0
    best_val_loss_val = float('inf')

    if TRAIN_CONFIG.get('resume_training', False):
        if TRAIN_CONFIG.get('load_best_model', False):
            ckpt_path = ckpt_manager.best_path
        else:
            ckpt_path = ckpt_manager.checkpoint_dir / f"checkpoint_step_{TRAIN_CONFIG.get('resume_checkpoint_step', 0)}.pt"
        
        if ckpt_path.exists():
            print(f"\n🔄 Resuming training from {ckpt_path.name}...")
            # weights_only=False needed to load optimizer and other generic python dicts
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # Load states
            model.load_state_dict(checkpoint['model_state_dict'])

            # When switching from standard AdamW to bitsandbytes AdamW8bit, the
            # checkpoint will have 'exp_avg'/'exp_avg_sq' keys instead of 'state1'/'state2'.
            # _remap_opt_state_for_bnb renames them so the momentum history is preserved
            # exactly. If the checkpoint was already saved with AdamW8bit the function
            # is a no-op (keys are unchanged).
            _use_8bit = TRAIN_CONFIG.get('use_8bit_adam', False) and _BNB_AVAILABLE
            _enc_opt_state = checkpoint['optimizer_encoder_state_dict']
            _dec_opt_state = checkpoint['optimizer_decoder_state_dict']
            if _use_8bit:
                _enc_opt_state = _remap_opt_state_for_bnb(_enc_opt_state)
                _dec_opt_state = _remap_opt_state_for_bnb(_dec_opt_state)
                print("  ↳ Remapped AdamW → AdamW8bit optimizer state keys (momentum history preserved).")
            optimizer_encoder.load_state_dict(_enc_opt_state)
            optimizer_decoder.load_state_dict(_dec_opt_state)
            scheduler_encoder.load_state_dict(checkpoint['scheduler_encoder_state_dict'])
            scheduler_decoder.load_state_dict(checkpoint['scheduler_decoder_state_dict'])
            
            # Load tracking vars
            start_epoch = checkpoint['epoch']
            start_global_step = checkpoint['global_step']
            best_val_loss_val = checkpoint.get('best_val_loss', float('inf'))
            start_evals_without_improvement = checkpoint.get('evals_without_improvement', 0)
            start_elapsed_sec = checkpoint.get('elapsed_sec', 12029.3)
            
            # Load RNG if they exist
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
            print(f"  ✅ Loaded state: Epoch {start_epoch}, Step {start_global_step}, Best Val Loss: {best_val_loss_val:.4f}, Elapsed: {start_elapsed_sec:.1f}s")
        else:
            print(f"\n⚠️  Resume requested but checkpoint not found: {ckpt_path}. Starting from scratch.")
            start_evals_without_improvement = 0
            start_elapsed_sec = 0.0

    else:
        start_evals_without_improvement = 0
        start_elapsed_sec = 0.0

    # Launch training
    try:
        best_val_loss = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            optimizer_encoder=optimizer_encoder,
            optimizer_decoder=optimizer_decoder,
            scheduler_encoder=scheduler_encoder,
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
    # Close TensorBoard writer
        tb_writer.close()

    print(f"\nBest validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
