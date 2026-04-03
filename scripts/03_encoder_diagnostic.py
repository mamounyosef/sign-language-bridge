"""
Encoder Diagnostic Script for Sign Language Translation Model

This script tests whether the encoder provides useful signal to the decoder by:
1. Running normal validation with encoder outputs
2. Running ablation validation with zeroed encoder outputs
3. Comparing BLEU scores to quantify encoder contribution

Usage: python scripts/03_encoder_diagnostic.py
"""

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

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import LoraConfig, get_peft_model
import sacrebleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# PyTorch optimization settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Maximum loss value for perplexity calculation to prevent overflow
MAX_PPL_CAP = 20

## Configuration (copied exactly from 02_model_training.py)

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
    'decoder_model_name': 'google/gemma-3-270m',
    'decoder_hidden_size': 640,
    'decoder_num_layers': 18,

    # ── MLP Projection ──
    'projection_hidden_dim': 512,
    'projection_d_model': 512,  # Output dim → encoder input
    'projection_dropout': 0.15,  # Increased from 0.1 to combat overfitting

    # ── Transformer Encoder ──
    'encoder_d_model': 512,
    'encoder_num_heads': 8,
    'encoder_num_layers': 6,
    'encoder_feedforward_dim': 2048,
    'encoder_dropout': 0.15,  # Increased from 0.1 to combat overfitting
    'encoder_max_len': 5000,

    # ── Cross-Attention ──
    'bottleneck_dim': 448,      # 448 / 8 heads = 56 dims/head
    'cross_attn_num_heads': 8,
    'cross_attn_dropout': 0.1,
    'cross_attn_layers': [0, 1, 3, 6, 9, 12, 15, 16, 17],  # 9 out of 18
    'weight_sharing_pairs': [(0, 1), (15, 16)],              # 2 pairs

    # ── LoRA ──
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    'lora_modules_to_save': ["embed_tokens", "lm_head"],
    'lora_dropout': 0.1,
    'lora_bias': "none",

    # ── Data Augmentation (disabled for validation) ──
    'aug_temporal_jitter': False,
    'aug_temporal_jitter_range': 2,
    'aug_temporal_jitter_prob': 0.5,
    'aug_gaussian_noise': False,
    'aug_noise_std': 0.01,
    'aug_noise_prob': 0.5,
    'aug_spatial_shift': False,
    'aug_spatial_shift_range': 0.05,
    'aug_spatial_shift_prob': 0.5,
    'aug_spatial_scale': False,
    'aug_spatial_scale_range': (0.9, 1.1),
    'aug_spatial_scale_prob': 0.5,
}

## Training Configuration (validation settings)
TRAIN_CONFIG = {
    'batch_size': 6,
    'max_eval_batches': 60,         # max batches for validation loss computation
    'max_generate_samples': 200,     # max samples for BLEU/ROUGE generation
    'num_print_samples': 3,         # how many generated samples to print during eval
    'val_gen_batch_size': 16,       # batch size for validation generation (speeds up BLEU/ROUGE)
    'val_beam_size': 4,             # beam search width for validation generation (1 = greedy)
    'val_repetition_penalty': 1.2,  # repetition penalty for validation generation (1.0 = off)
    'val_use_kv_cache': True,       # use KV-cache during generation (faster, disable if issues)
    'val_num_workers': 2,            # workers for validation DataLoader
    'val_prefetch_factor': 4,        # prefetch factor for validation DataLoader
}

# Max sequence lengths (safety caps to avoid OOM)
MAX_KEYPOINT_FRAMES = CONFIG['max_keypoint_frames']
MAX_TEXT_TOKENS = CONFIG['max_text_tokens']

## Dataset (copied from 02_model_training.py)

class SignLanguageDataset(Dataset):
    def __init__(self, manifest, tokenizer, max_frames=MAX_KEYPOINT_FRAMES, max_tokens=MAX_TEXT_TOKENS,
                 train=False, augment_config=None):
        """
        Args:
            manifest: List of dicts with keys: sentence_name, text, duration, keypoint_path
            tokenizer: Pretrained tokenizer (Gemma 3)
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

        # Data augmentation disabled for validation (train=False)

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


## Model Components (copied from 02_model_training.py)

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
        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        # Note: src_key_padding_mask expects True for positions to IGNORE
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        return output


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
    def __init__(self, hidden_size=640, encoder_dim=512, bottleneck_dim=448, num_heads=8, dropout=0.1):
        super().__init__()
        # Projections for query (decoder), key and value (encoder)
        self.cross_attn_in_proj = nn.Linear(hidden_size, bottleneck_dim)  # decoder: 640 → 448
        self.key_proj = nn.Linear(encoder_dim, bottleneck_dim)  # encoder: 512 → 448
        self.value_proj = nn.Linear(encoder_dim, bottleneck_dim)  # encoder: 512 → 448

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_out_proj = nn.Linear(bottleneck_dim, hidden_size)  # 448 → 640
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Apply standard normal initialization
        init_from_scratch_weights(self)

        # OVERRIDE: Zero-init output projection
        # Cross-attention starts as no-op, learns gradually
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


class Gemma3DecoderLayerWithOptionalCrossAttention(nn.Module):
    """Gemma 3 decoder layer with optional cross-attention."""
    def __init__(self, original_layer, cross_attn_module=None):
        super().__init__()
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        self.pre_feedforward_layernorm = original_layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = original_layer.post_feedforward_layernorm
        self.cross_attn_module = cross_attn_module
        # Preserve attention_type for mask routing in patched forward
        self.attention_type = getattr(original_layer, 'attention_type', 'full_attention')

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
        position_embeddings_global: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_local: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # 1. Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # 2. Cross-Attention (if this layer has it)
        if self.cross_attn_module is not None and encoder_hidden_states is not None:
            hidden_states = self.cross_attn_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )

        # 3. MLP
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


def patch_gemma_model_forward(model):
    """Patch Gemma forward pass to support encoder-decoder architecture."""
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
        rotary_emb_local = model.base_model.model.model.rotary_emb_local
        config = model.base_model.model.config

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

        # Dual rotary embeddings
        position_embeddings_global = rotary_emb(hidden_states, position_ids)
        position_embeddings_local = rotary_emb_local(hidden_states, position_ids)

        # ── Build attention masks ──
        window_size = getattr(config, 'sliding_window', 4096)
        fill_val = -1e4 if dtype == torch.float16 else -1e9
        total_len = past_seen_tokens + seq_len

        if past_seen_tokens > 0 and seq_len == 1:
            # Decode mode
            full_mask = torch.zeros(1, 1, 1, total_len, device=device, dtype=dtype)

            if total_len > window_size:
                positions = torch.arange(total_len, device=device)
                current_pos = cache_position[0]
                out_of_window = (current_pos - positions) >= window_size
                sliding_mask = torch.where(
                    out_of_window,
                    torch.full([], fill_val, device=device, dtype=dtype),
                    torch.zeros([], device=device, dtype=dtype),
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            else:
                sliding_mask = torch.zeros(1, 1, 1, total_len, device=device, dtype=dtype)

            causal_mask_mapping = {
                "full_attention": full_mask,
                "sliding_attention": sliding_mask,
            }
        else:
            # Prefill or training mode
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

            def create_sliding_window_mask(seq_len, window_size, device, dtype, attn_mask=None):
                row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
                col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
                can_attend = (col_idx <= row_idx) & ((row_idx - col_idx) < window_size)
                mask = torch.where(~can_attend,
                    torch.full([], fill_val, device=device, dtype=dtype),
                    torch.zeros([], device=device, dtype=dtype))
                mask = mask.unsqueeze(0).unsqueeze(0)
                if attn_mask is not None:
                    padding = attn_mask.unsqueeze(1).unsqueeze(2).to(dtype)
                    padding = (1.0 - padding) * fill_val
                    mask = mask + padding
                return mask

            causal_mask_mapping = {
                "full_attention": create_full_causal_mask(seq_len, device, dtype, attention_mask),
                "sliding_attention": create_sliding_window_mask(seq_len, window_size, device, dtype, attention_mask),
            }

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_attention_type = getattr(decoder_layer, 'attention_type', 'full_attention')
            layer_causal_mask = causal_mask_mapping[layer_attention_type]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
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
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return {'loss': loss, 'logits': logits}

        return {'logits': logits}

    def _apply_repetition_penalty(self, logits, generated_ids, penalty):
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits
        score = torch.gather(logits, 1, generated_ids)
        score = torch.where(score > 0, score / penalty, score * penalty)
        logits = logits.scatter(1, generated_ids, score)
        return logits

    @torch.no_grad()
    def _beam_search_single(self, encoder_hidden_states, encoder_padding_mask,
                            device, max_new_tokens, beam_size, repetition_penalty, use_kv_cache=True):
        """Beam search for a single sample."""
        eos_id = self.tokenizer.eos_token_id
        bos_id = self.tokenizer.bos_token_id

        # Expand encoder states for all beams
        enc_hs = encoder_hidden_states.expand(beam_size, -1, -1).contiguous()
        enc_mask = encoder_padding_mask.expand(beam_size, -1).contiguous()

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
    def generate(self, keypoints, keypoint_mask, max_new_tokens=50, temperature=1.0,
                 top_k=50, repetition_penalty=1.0, beam_size=1, use_kv_cache=True):
        """
        Generate text from keypoints.
        """
        device = keypoints.device
        batch_size = keypoints.shape[0]

        # Encode keypoints
        encoder_input = self.keypoint_projection(keypoints.to(next(self.keypoint_projection.parameters()).dtype))
        encoder_padding_mask = ~(keypoint_mask.any(dim=-1))
        encoder_output = self.encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
        encoder_hidden_states = self.encoder_projection(encoder_output).to(next(self.decoder.parameters()).dtype)

        # Beam search
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

        # Greedy / Sampling
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


## Ablation Wrapper (NEW)

class AblationSignLanguageTranslationModel(nn.Module):
    """
    Wrapper around SignLanguageTranslationModel that can ablate encoder outputs.

    When ablate_encoder=True, encoder_hidden_states are zeroed before passing to decoder,
    allowing us to test if the encoder provides useful signal.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.ablate_encoder = False  # Toggle flag

    def forward(self, keypoints, keypoint_mask, token_ids, text_attention_mask, return_loss=True):
        device = keypoints.device

        # Run encoder pipeline normally
        encoder_input = self.base_model.keypoint_projection(
            keypoints.to(next(self.base_model.keypoint_projection.parameters()).dtype)
        )
        encoder_padding_mask = ~(keypoint_mask.any(dim=-1))
        encoder_output = self.base_model.encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
        encoder_hidden_states = self.base_model.encoder_projection(encoder_output)

        # ABLATION: Zero encoder outputs if flag is set
        if self.ablate_encoder:
            encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        # Continue with decoder
        decoder_input_ids = token_ids[:, :-1].contiguous()
        labels = token_ids[:, 1:].contiguous()
        decoder_attention_mask = text_attention_mask[:, :-1]

        encoder_hidden_states = encoder_hidden_states.to(next(self.base_model.decoder.parameters()).dtype)

        outputs = self.base_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_padding_mask,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits

        if return_loss:
            loss = self.base_model.loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return {'loss': loss, 'logits': logits}

        return {'logits': logits}

    def generate(self, keypoints, keypoint_mask, max_new_tokens=50, temperature=1.0,
                 top_k=50, repetition_penalty=1.0, beam_size=1, use_kv_cache=True):
        """
        Generate with optional encoder ablation.
        """
        device = keypoints.device
        batch_size = keypoints.shape[0]

        # Encode keypoints
        encoder_input = self.base_model.keypoint_projection(
            keypoints.to(next(self.base_model.keypoint_projection.parameters()).dtype)
        )
        encoder_padding_mask = ~(keypoint_mask.any(dim=-1))
        encoder_output = self.base_model.encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
        encoder_hidden_states = self.base_model.encoder_projection(encoder_output)

        # ABLATION: Zero encoder outputs if flag is set
        if self.ablate_encoder:
            encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        encoder_hidden_states = encoder_hidden_states.to(next(self.base_model.decoder.parameters()).dtype)

        # Use base model's generation logic (beam search or greedy)
        if beam_size > 1:
            all_sequences = []
            for i in range(batch_size):
                best_seq = self.base_model._beam_search_single(
                    encoder_hidden_states[i:i+1],
                    encoder_padding_mask[i:i+1],
                    device, max_new_tokens, beam_size, repetition_penalty, use_kv_cache,
                )
                all_sequences.append(best_seq)

            generated_texts = [
                self.base_model.tokenizer.decode(seq, skip_special_tokens=True)
                for seq in all_sequences
            ]
            max_len = max(seq.shape[0] for seq in all_sequences)
            padded_ids = torch.full((batch_size, max_len), self.base_model.tokenizer.pad_token_id,
                                   device=device, dtype=torch.long)
            for i, seq in enumerate(all_sequences):
                padded_ids[i, :seq.shape[0]] = seq

            return {'generated_ids': padded_ids, 'generated_text': generated_texts}

        # Greedy / Sampling
        generated_ids = torch.full((batch_size, 1), self.base_model.tokenizer.bos_token_id,
                                   device=device, dtype=torch.long)
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        past_key_values = None

        for _ in range(max_new_tokens):
            if use_kv_cache and past_key_values is not None:
                input_ids = generated_ids[:, -1:]
            else:
                input_ids = generated_ids

            outputs = self.base_model.decoder(
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
                next_token_logits = self.base_model._apply_repetition_penalty(
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
            active_sequences &= (next_token.squeeze(-1) != self.base_model.tokenizer.eos_token_id)

            if not active_sequences.any():
                break

        generated_texts = [
            self.base_model.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        return {'generated_ids': generated_ids, 'generated_text': generated_texts}


## Metric Functions (copied from 02_model_training.py)

def compute_bleu(references, hypotheses, max_n=4):
    """
    Compute corpus-level BLEU score using sacrebleu.
    """
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
    """
    Compute average ROUGE-L F1 score.
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


## Validation Function (copied from 02_model_training.py)

@torch.no_grad()
def validate(model, val_loader, val_dataset, tokenizer, device,
             max_eval_batches, max_generate_samples, num_print_samples, val_gen_batch_size,
             val_beam_size=1, val_repetition_penalty=1.0, val_use_kv_cache=True):
    """
    Run validation: compute loss, perplexity, token accuracy,
    then generate text for BLEU & ROUGE-L.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    # Part 1: Validation Loss + Token Accuracy
    print("  Computing validation loss...", end=" ", flush=True)
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

        # Token accuracy
        logits = output['logits']
        preds = logits.argmax(dim=-1)
        labels = batch['token_ids'][:, 1:].to(device, non_blocking=True)
        mask = labels != tokenizer.pad_token_id
        total_correct += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, MAX_PPL_CAP))
    token_acc = (total_correct / max(total_tokens, 1)) * 100
    print("Done")

    # Part 2: Generate Text for BLEU / ROUGE-L
    print(f"  Generating {max_generate_samples} samples (beam={val_beam_size})...", end=" ", flush=True)
    references = []
    hypotheses = []
    sample_pairs = []

    num_samples = min(max_generate_samples, len(val_dataset))
    sample_indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(42))[:num_samples].tolist()

    for batch_start in tqdm(range(0, num_samples, val_gen_batch_size), desc="  Generating", leave=False):
        batch_end = min(batch_start + val_gen_batch_size, num_samples)
        batch_indices = sample_indices[batch_start:batch_end]
        batch_samples = [val_dataset[idx] for idx in batch_indices]

        keypoints_list = [s['keypoints'] for s in batch_samples]
        keypoint_mask_list = [s['keypoint_mask'] for s in batch_samples]
        token_ids_list = [s['token_ids'] for s in batch_samples]

        max_frames = max(kp.shape[0] for kp in keypoints_list)
        batch_size = len(batch_samples)
        num_landmarks = keypoints_list[0].shape[1]

        padded_keypoints = torch.zeros(batch_size, max_frames, num_landmarks, 2)
        padded_masks = torch.zeros(batch_size, max_frames, num_landmarks, dtype=torch.bool)

        for i, (kp, mask) in enumerate(zip(keypoints_list, keypoint_mask_list)):
            T = kp.shape[0]
            padded_keypoints[i, :T] = kp
            padded_masks[i, :T] = mask

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

    print("Done")

    # Compute generation metrics
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
        'num_eval_batches': num_batches,
        'num_gen_samples': num_samples,
    }


## Model Building Function

def build_model(config, tokenizer, device):
    """
    Build the complete sign language translation model.

    Returns:
        SignLanguageTranslationModel with encoder, decoder, and cross-attention
    """
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)

    # Load decoder
    print(f"\n1. Loading decoder: {config['decoder_model_name']}")
    decoder = AutoModelForCausalLM.from_pretrained(
        config['decoder_model_name'],
        dtype=torch.bfloat16,
        device_map=device
    )

    if decoder.config.vocab_size != len(tokenizer):
        decoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Apply LoRA
    print("2. Applying LoRA configuration")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['lora_target_modules'],
        modules_to_save=config['lora_modules_to_save'],
        lora_dropout=config['lora_dropout'],
        bias=config['lora_bias'],
        task_type="CAUSAL_LM",
        use_rslora=True,
        ensure_weight_tying=True
    )
    decoder = get_peft_model(decoder, lora_config)

    # Enable norm layers
    for name, param in decoder.named_parameters():
        if 'norm' in name:
            param.requires_grad = True

    print(f"   LoRA rank: {config['lora_r']}, alpha: {config['lora_alpha']}")

    # Create encoder components
    print("3. Creating encoder components")
    projection = KeypointProjection(
        num_landmarks=config['num_landmarks'],
        coord_dim=config['coord_dim'],
        hidden_dim=config['projection_hidden_dim'],
        d_model=config['projection_d_model'],
    ).to(torch.bfloat16)

    encoder = TransformerEncoder(
        d_model=config['encoder_d_model'],
        num_heads=config['encoder_num_heads'],
        num_layers=config['encoder_num_layers'],
        feedforward_dim=config['encoder_feedforward_dim'],
        dropout=config['encoder_dropout']
    ).to(torch.bfloat16)

    encoder_projection = EncoderProjection(encoder_dim=config['encoder_d_model']).to(torch.bfloat16)

    print(f"   Encoder: d_model={config['encoder_d_model']}, layers={config['encoder_num_layers']}")

    # Create cross-attention modules
    print("4. Creating cross-attention modules")
    cross_attn_modules = {}
    shared_modules = {}

    # Create shared modules first
    for layer_a, layer_b in config['weight_sharing_pairs']:
        if layer_a in config['cross_attn_layers'] and layer_b in config['cross_attn_layers']:
            shared_module = CrossAttentionModule(
                hidden_size=config['decoder_hidden_size'],
                encoder_dim=config['encoder_d_model'],
                bottleneck_dim=config['bottleneck_dim'],
                num_heads=config['cross_attn_num_heads'],
                dropout=config['cross_attn_dropout']
            )
            shared_modules[layer_a] = shared_module
            shared_modules[layer_b] = shared_module

    # Create unique modules for non-shared layers
    for layer_idx in config['cross_attn_layers']:
        if layer_idx not in shared_modules:
            cross_attn_modules[layer_idx] = CrossAttentionModule(
                hidden_size=config['decoder_hidden_size'],
                encoder_dim=config['encoder_d_model'],
                bottleneck_dim=config['bottleneck_dim'],
                num_heads=config['cross_attn_num_heads'],
                dropout=config['cross_attn_dropout']
            )
        else:
            cross_attn_modules[layer_idx] = shared_modules[layer_idx]

    for module in {id(m): m for m in cross_attn_modules.values()}.values():
        module.to(torch.bfloat16)

    unique_modules = len(set(id(m) for m in cross_attn_modules.values()))
    print(f"   Cross-attention: {len(config['cross_attn_layers'])} layers, {unique_modules} unique modules")

    # Wrap decoder layers with cross-attention
    print("5. Wrapping decoder layers with cross-attention")
    num_layers = len(decoder.base_model.model.model.layers)
    for i in range(num_layers):
        original_layer = decoder.base_model.model.model.layers[i]
        cross_attn_module = cross_attn_modules.get(i, None)
        decoder.base_model.model.model.layers[i] = Gemma3DecoderLayerWithOptionalCrossAttention(
            original_layer,
            cross_attn_module=cross_attn_module
        )

    # Patch Gemma forward
    print("6. Patching Gemma forward pass for encoder-decoder")
    patch_gemma_model_forward(decoder)

    # Create complete model
    print("7. Assembling complete model")
    model = SignLanguageTranslationModel(
        keypoint_projection=projection,
        encoder=encoder,
        encoder_projection=encoder_projection,
        decoder=decoder,
        tokenizer=tokenizer,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model built successfully")
    print(f"  Total trainable parameters: {total_params / 1e6:.2f}M")
    print("=" * 80)

    return model


## Main Diagnostic Function

def print_comparison(normal_metrics, ablation_metrics, checkpoint_info):
    """Print side-by-side comparison of normal vs ablation metrics."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC RESULTS: ENCODER SIGNAL ANALYSIS")
    print("=" * 80)

    # Metrics table
    print("\n{:<20} {:>15} {:>15} {:>15}".format("Metric", "Normal", "Ablated", "Delta"))
    print("-" * 80)

    metrics_to_compare = [
        ('val_loss', 'Loss', '{:.4f}', False),
        ('val_ppl', 'Perplexity', '{:.2f}', False),
        ('token_acc', 'Token Accuracy', '{:.2f}%', True),
        ('bleu1', 'BLEU-1', '{:.2f}', True),
        ('bleu2', 'BLEU-2', '{:.2f}', True),
        ('bleu4', 'BLEU-4', '{:.2f}', True),
        ('rouge_l', 'ROUGE-L', '{:.2f}', True),
    ]

    for key, label, fmt, higher_is_better in metrics_to_compare:
        normal_val = normal_metrics[key]
        ablation_val = ablation_metrics[key]
        delta = normal_val - ablation_val

        # Format values
        normal_str = fmt.format(normal_val)
        ablation_str = fmt.format(ablation_val)

        # Delta with sign
        if higher_is_better:
            delta_str = fmt.format(delta) if delta >= 0 else fmt.format(delta)
        else:
            delta_str = fmt.format(delta) if delta <= 0 else "+" + fmt.format(delta)

        print("{:<20} {:>15} {:>15} {:>15}".format(label, normal_str, ablation_str, delta_str))

    print("-" * 80)

    # Sample generations
    print("\nSample Generations (Normal):")
    for i, (ref, hyp) in enumerate(normal_metrics['sample_pairs'][:3], 1):
        print(f"  [{i}] REF: {ref}")
        print(f"      HYP: {hyp}")

    print("\nSample Generations (Ablated):")
    for i, (ref, hyp) in enumerate(ablation_metrics['sample_pairs'][:3], 1):
        print(f"  [{i}] REF: {ref}")
        print(f"      HYP: {hyp}")

    # Analysis
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    bleu4_delta = normal_metrics['bleu4'] - ablation_metrics['bleu4']
    loss_delta = ablation_metrics['val_loss'] - normal_metrics['val_loss']
    token_acc_delta = normal_metrics['token_acc'] - ablation_metrics['token_acc']

    print(f"\nEncoder Contribution:")
    print(f"  BLEU-4 improvement: +{bleu4_delta:.2f} points")
    print(f"  Loss reduction: -{loss_delta:.4f}")
    print(f"  Token accuracy gain: +{token_acc_delta:.2f}%")

    if bleu4_delta > 10:
        conclusion = "✓ STRONG SIGNAL - Encoder is essential for translation"
        explanation = "BLEU-4 drops significantly without encoder. Cross-attention is working correctly."
    elif bleu4_delta > 5:
        conclusion = "✓ MODERATE SIGNAL - Encoder provides useful information"
        explanation = "Encoder contributes but decoder also relies on language priors."
    elif bleu4_delta > 0:
        conclusion = "⚠ WEAK SIGNAL - Encoder contribution is minimal"
        explanation = "Decoder relies heavily on language model. Check cross-attention weights."
    else:
        conclusion = "✗ NO SIGNAL - Encoder may be providing noise"
        explanation = "Critical issue: Model performs better without encoder. Debug architecture."

    print(f"\n{conclusion}")
    print(f"  → {explanation}")
    print("=" * 80 + "\n")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ENCODER DIAGNOSTIC EVALUATION")
    print("Sign Language Translation Model - Ablation Test")
    print("=" * 80)

    # Setup
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Load validation data
    print("\n" + "-" * 80)
    print("LOADING VALIDATION DATA")
    print("-" * 80)

    val_csv = CONFIG['data_val_csv']
    keypoints_dir = CONFIG['keypoints_val_dir']

    print(f"Reading: {val_csv}")
    df = pd.read_csv(val_csv, sep=CONFIG['csv_sep'])
    print(f"Total samples: {len(df)}")

    # Build manifest
    manifest = []
    missing_count = 0
    for idx, row in df.iterrows():
        sentence_name = row['vid']
        keypoint_filename = sentence_name.replace(':', '-')
        keypoint_path = keypoints_dir / f'{keypoint_filename}.npz'

        if keypoint_path.exists():
            manifest.append({
                'sentence_name': sentence_name,
                'text': row['text'],
                'duration': row['duration_sec'],
                'keypoint_path': str(keypoint_path)
            })
        else:
            missing_count += 1

    print(f"Valid samples: {len(manifest)} ({missing_count} missing keypoint files)")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['decoder_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and loader
    val_dataset = SignLanguageDataset(
        manifest,
        tokenizer,
        max_frames=MAX_KEYPOINT_FRAMES,
        max_tokens=MAX_TEXT_TOKENS,
        train=False,  # No augmentation
        augment_config=CONFIG
    )

    collate_fn_partial = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_partial,
        num_workers=TRAIN_CONFIG['val_num_workers'],
        pin_memory=True,
        prefetch_factor=TRAIN_CONFIG['val_prefetch_factor']
    )

    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  Validation batches: {len(val_loader)}")

    # Build model
    base_model = build_model(CONFIG, tokenizer, device)

    # Load checkpoint
    print("\n" + "-" * 80)
    print("LOADING CHECKPOINT")
    print("-" * 80)

    checkpoint_path = Path('..') / 'checkpoints' / 'best_model.pt'
    print(f"Loading: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'step': checkpoint.get('global_step', 'N/A'),
        'best_val_loss': checkpoint.get('best_val_loss', 'N/A'),
    }

    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Step: {checkpoint_info['step']}")
    print(f"  Best val loss: {checkpoint_info['best_val_loss']:.4f}")

    # Wrap model with ablation capability
    model = AblationSignLanguageTranslationModel(base_model)
    model.to(device)
    model.eval()

    print("\n✓ Model loaded and ready for diagnostic evaluation")

    # Validation config
    val_config = {
        'max_eval_batches': TRAIN_CONFIG['max_eval_batches'],
        'max_generate_samples': TRAIN_CONFIG['max_generate_samples'],
        'num_print_samples': TRAIN_CONFIG['num_print_samples'],
        'val_gen_batch_size': TRAIN_CONFIG['val_gen_batch_size'],
        'val_beam_size': TRAIN_CONFIG['val_beam_size'],
        'val_repetition_penalty': TRAIN_CONFIG['val_repetition_penalty'],
        'val_use_kv_cache': TRAIN_CONFIG['val_use_kv_cache'],
    }

    print(f"\nValidation configuration:")
    print(f"  Max eval batches: {val_config['max_eval_batches']}")
    print(f"  Max generate samples: {val_config['max_generate_samples']}")
    print(f"  Beam size: {val_config['val_beam_size']}")
    print(f"  Repetition penalty: {val_config['val_repetition_penalty']}")

    # Run normal validation
    print("\n" + "=" * 80)
    print("[1/2] NORMAL VALIDATION (Encoder Active)")
    print("=" * 80)

    model.ablate_encoder = False
    normal_metrics = validate(
        model, val_loader, val_dataset, tokenizer, device,
        **val_config
    )

    print(f"\nNormal Results:")
    print(f"  Loss: {normal_metrics['val_loss']:.4f} | PPL: {normal_metrics['val_ppl']:.2f}")
    print(f"  Token Acc: {normal_metrics['token_acc']:.2f}%")
    print(f"  BLEU-1: {normal_metrics['bleu1']:.2f} | BLEU-2: {normal_metrics['bleu2']:.2f} | BLEU-4: {normal_metrics['bleu4']:.2f}")
    print(f"  ROUGE-L: {normal_metrics['rouge_l']:.2f}")

    # Run ablation validation
    print("\n" + "=" * 80)
    print("[2/2] ABLATION VALIDATION (Encoder Zeroed)")
    print("=" * 80)

    model.ablate_encoder = True
    ablation_metrics = validate(
        model, val_loader, val_dataset, tokenizer, device,
        **val_config
    )

    print(f"\nAblation Results:")
    print(f"  Loss: {ablation_metrics['val_loss']:.4f} | PPL: {ablation_metrics['val_ppl']:.2f}")
    print(f"  Token Acc: {ablation_metrics['token_acc']:.2f}%")
    print(f"  BLEU-1: {ablation_metrics['bleu1']:.2f} | BLEU-2: {ablation_metrics['bleu2']:.2f} | BLEU-4: {ablation_metrics['bleu4']:.2f}")
    print(f"  ROUGE-L: {ablation_metrics['rouge_l']:.2f}")

    # Print comparison
    print_comparison(normal_metrics, ablation_metrics, checkpoint_info)

    print("Diagnostic evaluation complete!")


if __name__ == '__main__':
    main()
