# %% [markdown]
# # SignBridge - Model Training & Inference

# %%
## Imports & Setup

from pathlib import Path
from functools import partial, wraps
from typing import Optional, Tuple
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import LoraConfig, get_peft_model

## Additional Imports for Training
import time
import csv
import os
import math
from collections import Counter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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
## Configuration

CONFIG = {
    # ── Data Paths ──
    'data_train_csv': Path('..') / 'data' / '(final)_how2sign_train_filtered.csv',
    'data_val_csv': Path('..') / 'data' / '(final)_how2sign_val_filtered.csv',
    'keypoints_train_dir': Path('..') / 'data' / 'keypoints_preprocessed' / 'train',
    'keypoints_val_dir': Path('..') / 'data' / 'keypoints_preprocessed' / 'val',
    'csv_sep': '\t',

    # ── Keypoint Dimensions ──
    'num_landmarks': 116,       # 60 face + 14 pose + 42 hands
    'coord_dim': 2,             # x, y (z dropped)

    # ── Sequence Lengths ──
    'max_keypoint_frames': 200, # At 20 FPS = 10s (max clip ~8s)
    'max_text_tokens': 100,     # Excluding BOS/EOS

    # ── Tokenizer / Decoder ──
    'decoder_model_name': 'google/gemma-3-270m',
    'decoder_hidden_size': 640,
    'decoder_num_layers': 18,

    # ── MLP Projection ──
    'projection_hidden_dim': 512,
    'projection_d_model': 512,  # Output dim → encoder input
    'projection_dropout': 0.1,

    # ── Transformer Encoder ──
    'encoder_d_model': 512,
    'encoder_num_heads': 8,
    'encoder_num_layers': 6,
    'encoder_feedforward_dim': 2048,
    'encoder_dropout': 0.1,
    'encoder_max_len': 5000,

    # ── Cross-Attention ──
    'bottleneck_dim': 448,      # 448 / 8 heads = 56 dims/head
    'cross_attn_num_heads': 8,
    'cross_attn_dropout': 0.1,
    'cross_attn_layers': [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17],   # 14 out of 18
    'weight_sharing_pairs': [(2, 3), (5, 6), (10, 12), (14, 15)],           # 4 pairs

    # ── LoRA ──
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    'lora_modules_to_save': ["embed_tokens", "lm_head"],
    'lora_dropout': 0.1,
    'lora_bias': "none",
}

# print("Configuration loaded ✓")

# %%
## Step 1: Load and Inspect One Sample

# %%
## Step 2a: Build Dataset Manifest

# %%
## Step 2b: Build Validation Manifest

# %%
## Step 4: Setup Gemma 3 270M Tokenizer

# %%
## Step 5: Verify Special Tokens (Gemma has all tokens already!)

# %%
# $$
## Step 6: Update Dataset to Tokenize Text (with BOS/EOS, max_length, error handling)

# Max sequence lengths (safety caps to avoid OOM)
MAX_KEYPOINT_FRAMES = CONFIG['max_keypoint_frames']
MAX_TEXT_TOKENS = CONFIG['max_text_tokens']

class SignLanguageDataset(Dataset):
    def __init__(self, manifest, tokenizer, max_frames=MAX_KEYPOINT_FRAMES, max_tokens=MAX_TEXT_TOKENS):
        """
        Args:
            manifest: List of dicts with keys: sentence_name, text, duration, keypoint_path
            tokenizer: Pretrained tokenizer (Gemma 3)
            max_frames: Max keypoint frames (truncate if longer)
            max_tokens: Max text tokens excluding BOS/EOS (truncate if longer)
        """
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_tokens = max_tokens
    
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

# %%
## Step 8: Load Pretrained Gemma 3 270M Decoder

# %%
# %%
# %%
# %%
## Step 8 (continued): Apply Hybrid LoRA Configuration for Gemma 3 270M

# %%
## Step 9: Build MLP Projection Layer

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


# %%
## Step 11: Optimized Cross-Attention for Gemma 3 270M (Bottleneck: 448 dims)

# Encoder projection: Just normalization (encoder already at 512)
class EncoderProjection(nn.Module):
    def __init__(self, encoder_dim=512):
        """Normalizes encoder outputs."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(encoder_dim)
    
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

        # Zero-init output projection: cross-attention starts as no-op, learns gradually
        # This prevents the decoder from learning to suppress random noise at init
        nn.init.zeros_(self.cross_attn_out_proj.weight)
        nn.init.zeros_(self.cross_attn_out_proj.bias)

        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
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
        self.cross_attn_module = cross_attn_module  # None if no cross-attention
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # 1. Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
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
        
        if output_attentions:
            return (hidden_states, present_key_value, self_attn_weights)
        return (hidden_states, present_key_value)


# %%
## Step 12: Build Complete Encoder-Decoder Model

# ========== UPDATED WRAPPED LAYER WITH POSITION EMBEDDINGS FIX ==========

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

        # FIX: Layer returns (hidden_states,) or (hidden_states, self_attn_weights) — no cache
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ========== PATCH GEMMA FORWARD TO SUPPORT ENCODER-DECODER ==========

def patch_gemma_model_forward(model):
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

        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position.unsqueeze(0)
            else:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        if cache_position is None:
            cache_position = torch.arange(seq_len, device=device)

        # Dual rotary embeddings
        position_embeddings_global = rotary_emb(hidden_states, position_ids)
        position_embeddings_local = rotary_emb_local(hidden_states, position_ids)

        # Build both attention masks
        def create_full_causal_mask(seq_len, device, dtype, attn_mask=None):
            fill_val = -1e4 if dtype == torch.float16 else -1e9
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
            fill_val = -1e4 if dtype == torch.float16 else -1e9
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

        window_size = getattr(config, 'sliding_window', 4096)
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
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    model.base_model.model.model.forward = patched_forward
    print("✓ Patched Gemma with hybrid attention + encoder-decoder support")


# ========== RE-WRAP DECODER LAYERS WITH FIXED CLASS ==========
# Note: Must re-wrap since the class definition changed

# ========== COMPLETE MODEL ==========
class SignLanguageTranslationModel(nn.Module):
    def __init__(self, keypoint_projection, encoder, encoder_projection, decoder, tokenizer):
        super().__init__()
        self.keypoint_projection = keypoint_projection
        self.encoder = encoder
        self.encoder_projection = encoder_projection
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.15)

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

    @torch.no_grad()
    def generate(self, keypoints, keypoint_mask, max_new_tokens=50, temperature=1.0, top_k=50):
        device = keypoints.device

        # Bug #4: cast keypoints to projection dtype
        encoder_input = self.keypoint_projection(keypoints.to(next(self.keypoint_projection.parameters()).dtype))
        encoder_padding_mask = ~(keypoint_mask.any(dim=-1))
        encoder_output = self.encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)
        # Bug #3: use next().dtype instead of self.decoder.dtype
        encoder_hidden_states = self.encoder_projection(encoder_output).to(next(self.decoder.parameters()).dtype)

        generated_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=device, dtype=torch.long)

        for _ in range(max_new_tokens):
            outputs = self.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_padding_mask,
                use_cache=False,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # Greedy decoding if temperature is 0 or very low
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

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return {'generated_ids': generated_ids, 'generated_text': generated_text}


# %%
## Step 13: Detailed Model Parameter Count

# %%
# ## Quick Training Test (100 samples, 3 epochs, no saving)

# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset

# # Small subset
# test_indices = list(range(100))
# test_subset = Subset(train_dataset, test_indices)
# test_train_loader = DataLoader(test_subset, batch_size=4, shuffle=True, collate_fn=collate_fn_with_tokenizer)

# # Optimizer
# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# model.train()

# print("Quick training test (100 samples, 3 epochs)...")
# for epoch in range(3):
#     total_loss = 0
#     num_batches = 0
    
#     for batch in test_train_loader:
#         optimizer.zero_grad()
        
#         output = model(
#             keypoints=batch['keypoints'].to(device),
#             keypoint_mask=batch['keypoint_mask'].to(device),
#             token_ids=batch['token_ids'].to(device),
#             text_attention_mask=batch['text_attention_mask'].to(device),
#             return_loss=True,
#         )
        
#         loss = output['loss']
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         total_loss += loss.item()
#         num_batches += 1
    
#     avg_loss = total_loss / num_batches
#     print(f"  Epoch {epoch+1}/3 — Loss: {avg_loss:.4f}")

# print("✓ Test complete. Loss should be decreasing.")

# %%
## Training Configuration

TRAIN_CONFIG = {
    # ── Core Training ──
    'num_epochs': 30,
    'batch_size': 8,

    # ── Learning Rates (separate for different components) ──
    # Encoder + Cross-attention (new, untrained components)
    'encoder_lr': 6e-5,              # Higher LR for new encoder/cross-attn
    'encoder_min_lr': 6e-7,
    # LoRA adapters (fine-tuning pretrained decoder)
    'decoder_lr': 4e-5,              # Lower LR for LoRA fine-tuning
    'decoder_min_lr': 4e-7,

    'warmup_steps': 320,
    'weight_decay': 0.05,
    'adam_betas': (0.9, 0.999),
    'max_grad_norm': 0.5,

    # ── Gradient Accumulation ──
    'grad_accum_steps': 10,          # effective batch = batch_size * grad_accum_steps

    # ── Logging ──
    'log_every_steps': 10,          # print training stats every N optimizer steps
    'train_log_file': Path('..') / 'saved_metrics' / 'train_log.csv',
    'val_log_file': Path('..') / 'saved_metrics' / 'val_log.csv',

    # ── Checkpointing ──
    'save_every_steps': 400,        # save checkpoint every N optimizer steps
    'keep_last_n_checkpoints': 3,   # sliding window: keep only last N periodic checkpoints
    'checkpoint_dir': Path('..') / 'checkpoints',

    # ── Evaluation ──
    'eval_every_steps': 160,        # run validation every N optimizer steps
    'max_eval_batches': 60,         # max batches for validation loss computation
    'max_generate_samples': 50,     # max samples for BLEU/ROUGE generation
    'num_print_samples': 3,         # how many generated samples to print during eval

    # ── Early Stopping ──
    'early_stopping_patience': 20,   # stop after N evaluations without improvement
}


## Helper Functions: Metrics, Checkpointing, CSV Logging
# ─── Metric Functions ───

def compute_bleu(references, hypotheses, max_n=4):
    """
    Compute corpus-level BLEU score (BLEU-1 through BLEU-max_n).
    Returns BLEU-4 as a percentage (0-100).
    """
    if not references or not hypotheses:
        return 0.0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()

        if not hyp_tokens:
            continue

        ref_len += len(ref_tokens)
        hyp_len += len(hyp_tokens)

        for n in range(1, max_n + 1):
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i + n])
                for i in range(len(ref_tokens) - n + 1)
            )
            hyp_ngrams = Counter(
                tuple(hyp_tokens[i:i + n])
                for i in range(len(hyp_tokens) - n + 1)
            )

            for ng in hyp_ngrams:
                clipped_counts[n - 1] += min(hyp_ngrams[ng], ref_ngrams.get(ng, 0))
            total_counts[n - 1] += sum(hyp_ngrams.values())

    if hyp_len == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - ref_len / hyp_len))

    # Geometric mean of clipped precisions
    log_avg = 0.0
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_avg += math.log(clipped_counts[n] / total_counts[n]) / max_n

    return bp * math.exp(log_avg) * 100


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

    def save_periodic(self, state_dict, step):
        """Save a periodic checkpoint with sliding window eviction."""
        path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        torch.save(state_dict, path)
        self.periodic_checkpoints.append(path)
        print(f"  💾 Saved periodic checkpoint: {path.name}")

        # Evict oldest if exceeding window
        while len(self.periodic_checkpoints) > self.keep_last_n:
            old_path = self.periodic_checkpoints.pop(0)
            if old_path.exists() and old_path != self.best_path:
                old_path.unlink()
                print(f"  🗑️  Evicted old checkpoint: {old_path.name}")

    def save_best(self, state_dict):
        """Save the best model (always kept, never evicted)."""
        torch.save(state_dict, self.best_path)
        print(f"  ⭐ Saved best model: {self.best_path.name}")

    def _build_state_dict(self, model, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, epoch, global_step, best_val_loss):
        return {
            'model_state_dict': model.state_dict(),
            'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
            'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
            'scheduler_encoder_state_dict': scheduler_encoder.state_dict(),
            'scheduler_decoder_state_dict': scheduler_decoder.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
        }


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


# print("Helper functions loaded ✓")

# %%
## Validation Function

@torch.no_grad()
def validate(model, val_loader, val_dataset, tokenizer, device,
             max_eval_batches, max_generate_samples, num_print_samples):
    """
    Run validation: compute loss, perplexity, token accuracy,
    then generate text for BLEU & ROUGE-L.
    
    Returns dict with all validation metrics.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    # ── Part 1: Validation Loss + Token Accuracy (teacher-forced) ──
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

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid inf
    token_acc = (total_correct / max(total_tokens, 1)) * 100

    # ── Part 2: Generate Text for BLEU / ROUGE-L ──
    references = []
    hypotheses = []
    sample_pairs = []  # (ref, hyp) pairs for printing

    # Sample indices from validation set (use fixed seed for consistency)
    num_samples = min(max_generate_samples, len(val_dataset))
    sample_indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(42))[:num_samples].tolist()

    for idx in sample_indices:
        sample = val_dataset[idx]

        # Reference text: decode token_ids (skip BOS/EOS)
        ref_text = tokenizer.decode(sample['token_ids'][1:-1], skip_special_tokens=True).strip()

        # Generate (greedy decoding for deterministic, reproducible evaluation)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            gen_output = model.generate(
                keypoints=sample['keypoints'].unsqueeze(0).to(device, non_blocking=True),
                keypoint_mask=sample['keypoint_mask'].unsqueeze(0).to(device, non_blocking=True),
                max_new_tokens=50,
                temperature=0.0,  # Greedy decoding for faster, deterministic validation
                top_k=50,
            )
        hyp_text = gen_output['generated_text'].strip()

        references.append(ref_text)
        hypotheses.append(hyp_text)

        if len(sample_pairs) < num_print_samples:
            sample_pairs.append((ref_text, hyp_text))

    # Compute generation metrics
    bleu1 = compute_bleu(references, hypotheses, max_n=1)  # Early learning signal
    bleu4 = compute_bleu(references, hypotheses, max_n=4)  # Strict metric
    rouge_l = compute_rouge_l(references, hypotheses)

    model.train()

    return {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
        'token_acc': token_acc,
        'bleu1': bleu1,
        'bleu4': bleu4,
        'rouge_l': rouge_l,
        'sample_pairs': sample_pairs,
        'num_eval_batches': num_batches,
        'num_gen_samples': num_samples,
    }


# print("Validation function loaded ✓")

# %%
## Training Loop

def train(model, train_loader, val_loader, val_dataset, tokenizer,
          optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder,
          device, train_config, ckpt_manager, train_csv_logger, val_csv_logger):
    """Full training loop with all bells and whistles."""

    # Unpack config
    num_epochs = train_config['num_epochs']
    grad_accum_steps = train_config['grad_accum_steps']
    max_grad_norm = train_config['max_grad_norm']
    log_every = train_config['log_every_steps']
    save_every = train_config['save_every_steps']
    eval_every = train_config['eval_every_steps']
    max_eval_batches = train_config['max_eval_batches']
    max_generate_samples = train_config['max_generate_samples']
    num_print_samples = train_config['num_print_samples']
    patience = train_config['early_stopping_patience']

    # Calculate total optimizer steps
    steps_per_epoch = len(train_loader)
    total_microbatch_steps = steps_per_epoch * num_epochs
    total_optimizer_steps = total_microbatch_steps // grad_accum_steps

    # State tracking
    global_step = 0                 # optimizer steps
    best_val_loss = float('inf')
    evals_without_improvement = 0
    training_start = time.time()
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

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_microbatches = 0

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        for micro_step, batch in enumerate(train_loader):
            # ── Forward pass with mixed precision ──
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(
                    keypoints=batch['keypoints'].to(device, non_blocking=True),
                    keypoint_mask=batch['keypoint_mask'].to(device, non_blocking=True),
                    token_ids=batch['token_ids'].to(device, non_blocking=True),
                    text_attention_mask=batch['text_attention_mask'].to(device, non_blocking=True),
                    return_loss=True,
                )

            # Scale loss for gradient accumulation
            loss = output['loss'] / grad_accum_steps
            loss.backward()

            epoch_loss += output['loss'].item()
            epoch_microbatches += 1

            # ── Optimizer step (every grad_accum_steps micro-batches) ──
            is_accum_step = (micro_step + 1) % grad_accum_steps == 0
            is_last_step = (micro_step + 1) == len(train_loader)

            if is_accum_step or is_last_step:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                ).item()

                optimizer_encoder.step()
                optimizer_decoder.step()
                scheduler_encoder.step()
                scheduler_decoder.step()
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()

                global_step += 1
                step_losses.append(output['loss'].item())
                step_grad_norms.append(grad_norm)

                # ── Logging ──
                if global_step % log_every == 0:
                    avg_loss = sum(step_losses) / len(step_losses)
                    avg_grad_norm = sum(step_grad_norms) / len(step_grad_norms)
                    encoder_lr = scheduler_encoder.get_last_lr()[0]
                    decoder_lr = scheduler_decoder.get_last_lr()[0]
                    elapsed = time.time() - training_start
                    log_elapsed = time.time() - log_step_start
                    speed = len(step_losses) / max(log_elapsed, 1e-6)
                    cuda_mem, cuda_peak = get_cuda_mem()

                    # Calculate ETA
                    eta_seconds = (elapsed / global_step) * (total_optimizer_steps - global_step)

                    train_ppl = math.exp(min(avg_loss, 20))

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

                    step_losses.clear()
                    step_grad_norms.clear()
                    log_step_start = time.time()

                # ── Validation ──
                if global_step % eval_every == 0:
                    print(f"\n{'─' * 60}")
                    print(f"  📊 Validation @ Step {global_step} / Epoch {epoch}")
                    print(f"{'─' * 60}")

                    val_metrics = validate(
                        model, val_loader, val_dataset, tokenizer, device,
                        max_eval_batches, max_generate_samples, num_print_samples,
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
                        'bleu4': f"{val_metrics['bleu4']:.4f}",
                        'rouge_l': f"{val_metrics['rouge_l']:.4f}",
                        'token_acc': f"{val_metrics['token_acc']:.4f}",
                        'cuda_mem_gb': f"{cuda_mem:.2f}",
                        'cuda_peak_gb': f"{cuda_peak:.2f}",
                        'elapsed_sec': f"{elapsed:.1f}",
                    })

                    # ── Best model check ──
                    val_loss = val_metrics['val_loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        evals_without_improvement = 0
                        state = ckpt_manager._build_state_dict(
                            model, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, epoch, global_step, best_val_loss
                        )
                        ckpt_manager.save_best(state)
                        print(f"\n  ⭐ New best val loss: {best_val_loss:.4f}")
                    else:
                        evals_without_improvement += 1
                        print(f"\n  Val loss did not improve. "
                              f"Best: {best_val_loss:.4f} | "
                              f"Early stop: {evals_without_improvement}/{patience}")

                    print(f"{'─' * 60}\n")

                    model.train()

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
                        return best_val_loss

                # ── Periodic checkpoint ──
                if global_step % save_every == 0:
                    state = ckpt_manager._build_state_dict(
                        model, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, epoch, global_step, best_val_loss
                    )
                    ckpt_manager.save_periodic(state, global_step)

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
    import torch.multiprocessing as mp
    try:
        mp.set_sharing_strategy('file_system')
    except:
        pass

    # From original line ~100
    # Paths
    TRAIN_CSV = CONFIG['data_train_csv']
    KEYPOINTS_DIR = CONFIG['keypoints_train_dir']

    # Load CSV
    df = pd.read_csv(TRAIN_CSV, sep=CONFIG['csv_sep'])
    print(f'Total training examples: {len(df)}')
    print(f'Columns: {list(df.columns)}')

    # Pick first example
    sample_row = df.sample(1).iloc[0]
    sentence_name = sample_row['SENTENCE_NAME']
    sentence_text = sample_row['SENTENCE']
    duration_sec = sample_row['duration_sec']

    print(f'\nSample: {sentence_name}')
    print(f'Text: "{sentence_text}"')
    print(f'Duration: {duration_sec:.2f}s')

    # Load keypoints
    keypoint_path = KEYPOINTS_DIR / f'{sentence_name}.npz'

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


    # From original line ~144
    # Paths
    TRAIN_CSV = CONFIG['data_train_csv']
    KEYPOINTS_DIR = CONFIG['keypoints_train_dir']

    # Load CSV
    df = pd.read_csv(TRAIN_CSV, sep=CONFIG['csv_sep'])

    # Build manifest: list of (sentence_name, text, duration, keypoint_path)
    manifest = []
    missing = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Building manifest'):
        sentence_name = row['SENTENCE_NAME']
        text = row['SENTENCE']
        duration = row['END_REALIGNED'] - row['START_REALIGNED']
        keypoint_path = KEYPOINTS_DIR / f'{sentence_name}.npz'

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



    # From original line ~187
    VAL_CSV = CONFIG['data_val_csv']
    KEYPOINTS_VAL_DIR = CONFIG['keypoints_val_dir']

    df_val = pd.read_csv(VAL_CSV, sep=CONFIG['csv_sep'])

    val_manifest = []
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc='Building val manifest'):
        sentence_name = row['SENTENCE_NAME']
        text = row['SENTENCE']
        duration = row['END_REALIGNED'] - row['START_REALIGNED']
        keypoint_path = KEYPOINTS_VAL_DIR / f'{sentence_name}.npz'

        if keypoint_path.exists():
            val_manifest.append({
                'sentence_name': sentence_name,
                'text': text,
                'duration': duration,
                'keypoint_path': str(keypoint_path)
            })

    print(f'Val manifest: {len(val_manifest)} samples')


    # From original line ~213
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['decoder_model_name'])

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


    # From original line ~244
    # Gemma tokenizer already has all special tokens we need
    print("Special tokens verification:")
    print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id}) ✓")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id}) ✓")
    print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id}) ✓")
    print(f"  UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id}) ✓")
    print(f"\nVocab size: {len(tokenizer)} (no custom tokens needed!)")

    # Test tokenization with special tokens
    sample_text = manifest[0]['text']
    tokens = tokenizer(sample_text, add_special_tokens=True, return_tensors='pt')

    print(f"\nSample: \"{sample_text}\"")
    print(f"Token IDs (first 10): {tokens['input_ids'][0][:10].tolist()}")
    print(f"First token (BOS): {tokens['input_ids'][0][0].item()} = {tokenizer.bos_token}")
    print(f"Last token (EOS): {tokens['input_ids'][0][-1].item()} = {tokenizer.eos_token}")

    print(f"\nNote: BOS/EOS will be added explicitly in Dataset class")
    print(f"      PAD (id={tokenizer.pad_token_id}) will be used for batching")


    # From original line ~332
    # Create updated dataset with tokenizer
    train_dataset = SignLanguageDataset(manifest, tokenizer)
    val_dataset = SignLanguageDataset(val_manifest, tokenizer)
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



    # From original line ~420
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


    # From original line ~482
    # Load pretrained decoder in fp16
    model_name = CONFIG['decoder_model_name']
    decoder = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device)

    print(f"Loaded: {model_name}")
    print(f"Original vocab size: {decoder.config.vocab_size}")

    # Verify vocab matches tokenizer (should already match for Gemma)
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


    # From original line ~514
    # Configure LoRA with comprehensive coverage
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['lora_target_modules'],
        modules_to_save=CONFIG['lora_modules_to_save'],
        lora_dropout=CONFIG['lora_dropout'],
        bias=CONFIG['lora_bias'],
        task_type="CAUSAL_LM",
        use_rslora=True,
        ensure_weight_tying=True
    )

    # Apply LoRA
    decoder = get_peft_model(decoder, lora_config)

    # Manually enable norm layers (not covered by LoRA or modules_to_save)
    # Gemma has 4 norm layers per decoder layer + 1 final norm
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

    print(f"\nGemma 3 270M Decoder architecture:")
    print(f"  Hidden size (d_model): {decoder.config.hidden_size}")
    print(f"  Num layers: {decoder.config.num_hidden_layers}")
    print(f"  Num attention heads: {decoder.config.num_attention_heads}")

    print(f"\nHybrid LoRA strategy:")
    print(f"  ✓ LoRA adapters: Self-attention + MLP (r=16, optimized for 270M model)")
    print(f"  ✓ Fully trainable: Embeddings, lm_head, all norm layers (4 per layer)")
    print(f"  ✓ Cross-attention: Will be fully trainable when added (max learning capacity)")
    print(f"  → Smaller model + lower rank = better fit for 20k samples!")


    # From original line ~610
    # Create projection layer
    projection = KeypointProjection(
        num_landmarks=CONFIG['num_landmarks'],
        coord_dim=CONFIG['coord_dim'],
        d_model=CONFIG['projection_d_model']
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


    # From original line ~731
    # Create encoder
    encoder = TransformerEncoder(
        d_model=CONFIG['encoder_d_model'],
        num_heads=CONFIG['encoder_num_heads'],
        num_layers=CONFIG['encoder_num_layers'],
        feedforward_dim=CONFIG['encoder_feedforward_dim'],
        dropout=CONFIG['encoder_dropout']
    ).to(torch.bfloat16)

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


    # From original line ~889
    # Configuration for Gemma 3 270M
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

    print(f"Optimized Cross-Attention Configuration for Gemma 3 270M:")
    print(f"  Encoder dim: {ENCODER_DIM}, Bottleneck dim: {BOTTLENECK_DIM}")
    print(f"  Decoder hidden size: {CONFIG['decoder_hidden_size']} (Gemma), Num heads: {NUM_HEADS} (56 dims/head)")
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

    for module in set(id(m) for m in cross_attn_modules.values()):
        pass

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

    # Wrap all decoder layers (path after LoRA: base_model.model.model.layers)
    num_layers = len(decoder.base_model.model.model.layers)
    for i in range(num_layers):
        original_layer = decoder.base_model.model.model.layers[i]
        cross_attn_module = cross_attn_modules.get(i, None)  # None if layer doesn't have cross-attn
        decoder.base_model.model.model.layers[i] = Gemma3DecoderLayerWithOptionalCrossAttention(
            original_layer,
            cross_attn_module=cross_attn_module
        )

    print(f"\n✓ Wrapped {num_layers} Gemma decoder layers")

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

    print(f"\n✅ Final Optimized Architecture for Gemma 3 270M:")
    print(f"  • Decoder: {CONFIG['decoder_hidden_size']} hidden, {TOTAL_LAYERS} layers")
    print(f"  • Bottleneck: {BOTTLENECK_DIM} dims (Query: {CONFIG['decoder_hidden_size']}→{BOTTLENECK_DIM}, Key/Value: {ENCODER_DIM}→{BOTTLENECK_DIM})")
    print(f"  • Cross-attention @ {BOTTLENECK_DIM} dims, Output: {BOTTLENECK_DIM}→{CONFIG['decoder_hidden_size']}")
    print(f"  • Selective: {len(layers_with_cross_attn)}/{TOTAL_LAYERS} layers, Weight sharing: {len(weight_sharing_pairs)} pairs → {unique_modules} unique")
    print(f"  • Much better suited for 20k samples than Qwen!")

    # Re-apply cross attention wrapping with fixed class
    print("Re-wrapping decoder layers with fixed Gemma3DecoderLayerWithOptionalCrossAttention...")
    num_layers = len(decoder.base_model.model.model.layers)
    for i in range(num_layers):
        original_layer = decoder.base_model.model.model.layers[i]
        cross_attn_module = cross_attn_modules.get(i, None)
        decoder.base_model.model.model.layers[i] = Gemma3DecoderLayerWithOptionalCrossAttention(
            original_layer,
            cross_attn_module=cross_attn_module
        )
    print(f"✓ Re-wrapped {num_layers} layers")

    # Apply the patch
    patch_gemma_model_forward(decoder)



    # Create model
    model = SignLanguageTranslationModel(
        keypoint_projection=projection,
        encoder=encoder,
        encoder_projection=encoder_projection,
        decoder=decoder,
        tokenizer=tokenizer,
    ).to(device)

    # Compile model (components already in bfloat16)
    # model = torch.compile(model, backend="eager")

    print("\n✅ Complete Encoder-Decoder Model Created!")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable: {total_params / 1e6:.2f}M params")
    
    # From original line ~1342
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
        "Decoder (Gemma + LoRA + Cross-Attn)": model.decoder,
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


    # From original line ~1463
    print("TRAIN_CONFIG loaded ✓")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k}: {v}")

    # %%
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True  # Keep workers alive between validations
    )

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

    optimizer_encoder = AdamW(
        encoder_all_params,
        lr=TRAIN_CONFIG['encoder_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['adam_betas']
    )

    # Optimizer 2: All decoder trainable params (LoRA + embeddings + lm_head + norms)
    # Exclude cross_attn params (already in encoder optimizer)
    cross_attn_param_ids = set(id(p) for p in cross_attn_params)
    decoder_params = [p for n, p in model.decoder.named_parameters()
                      if p.requires_grad and id(p) not in cross_attn_param_ids]

    optimizer_decoder = AdamW(
        decoder_params,
        lr=TRAIN_CONFIG['decoder_lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        betas=TRAIN_CONFIG['adam_betas']
    )

    print(f"\n📊 Optimizers created:")
    print(f"  Encoder + Cross-attn: {len(encoder_all_params):4d} params | LR: {TRAIN_CONFIG['encoder_lr']:.2e}")
    print(f"  Decoder (LoRA+other): {len(decoder_params):4d} params | LR: {TRAIN_CONFIG['decoder_lr']:.2e}")
    print(f"  Total trainable:      {len(encoder_all_params) + len(decoder_params):4d} params")

    # Separate schedulers for each optimizer
    steps_per_epoch = len(train_loader)
    total_microbatch_steps = steps_per_epoch * TRAIN_CONFIG['num_epochs']
    total_optimizer_steps = total_microbatch_steps // TRAIN_CONFIG['grad_accum_steps']
    warmup_steps = TRAIN_CONFIG['warmup_steps']

    # Scheduler 1: Encoder + Cross-attention (higher LR → higher min_lr)
    warmup_encoder = LinearLR(
        optimizer_encoder,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_encoder = CosineAnnealingLR(
        optimizer_encoder,
        T_max=total_optimizer_steps - warmup_steps,
        eta_min=TRAIN_CONFIG['encoder_min_lr'],
    )
    scheduler_encoder = SequentialLR(
        optimizer_encoder,
        schedulers=[warmup_encoder, cosine_encoder],
        milestones=[warmup_steps],
    )

    # Scheduler 2: LoRA adapters (lower LR → lower min_lr)
    warmup_decoder = LinearLR(
        optimizer_decoder,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_decoder = CosineAnnealingLR(
        optimizer_decoder,
        T_max=total_optimizer_steps - warmup_steps,
        eta_min=TRAIN_CONFIG['decoder_min_lr'],
    )
    scheduler_decoder = SequentialLR(
        optimizer_decoder,
        schedulers=[warmup_decoder, cosine_decoder],
        milestones=[warmup_steps],
    )

    print(f"\n📈 LR Schedules (Warmup {warmup_steps} steps → Cosine decay):")
    print(f"  Encoder:  {TRAIN_CONFIG['encoder_lr']:.2e} → {TRAIN_CONFIG['encoder_min_lr']:.2e}")
    print(f"  Decoder:  {TRAIN_CONFIG['decoder_lr']:.2e} → {TRAIN_CONFIG['decoder_min_lr']:.2e}")

    print("Optimizer & Scheduler created ✓")

    # %%

    steps_per_epoch_micro = len(train_loader)
    steps_per_epoch_optim = steps_per_epoch_micro // TRAIN_CONFIG['grad_accum_steps']
    total_micro_steps = steps_per_epoch_micro * TRAIN_CONFIG['num_epochs']
    total_optim_steps = total_micro_steps // TRAIN_CONFIG['grad_accum_steps']

    print("=" * 60)
    print("STEP CALCULATIONS")
    print("=" * 60)
    print(f"  Micro-batch steps per epoch:   {steps_per_epoch_micro}")
    print(f"  Optimizer steps per epoch:     {steps_per_epoch_optim}")
    print(f"  Total micro-batch steps:       {total_micro_steps}  ({TRAIN_CONFIG['num_epochs']} epochs × {steps_per_epoch_micro} steps)")
    print(f"  Total optimizer steps:         {total_optim_steps}  ({total_micro_steps} ÷ {TRAIN_CONFIG['grad_accum_steps']} accum)")
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
    print(f"  Optimizer steps/epoch:   {steps_per_epoch // TRAIN_CONFIG['grad_accum_steps']}")
    print(f"  Total optimizer steps:   {total_optimizer_steps}")
    print(f"  Num epochs:              {TRAIN_CONFIG['num_epochs']}")
    print(f"  Warmup steps:            {warmup_steps}")
    print(f"  Encoder LR:              {TRAIN_CONFIG['encoder_lr']:.2e} → {TRAIN_CONFIG['encoder_min_lr']:.2e}")
    print(f"  Decoder LR:              {TRAIN_CONFIG['decoder_lr']:.2e} → {TRAIN_CONFIG['decoder_min_lr']:.2e}")
    print(f"  Mixed precision:         bfloat16")
    print(f"  Grad clipping:           max_norm={TRAIN_CONFIG['max_grad_norm']}")
    print(f"  Early stopping:          patience={TRAIN_CONFIG['early_stopping_patience']}")
    print("=" * 60)

    # %%

    # From original line ~2094
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
            'bleu1', 'bleu4', 'rouge_l', 'token_acc',
            'cuda_mem_gb', 'cuda_peak_gb',
            'elapsed_sec',
        ],
    )

    print(f"Checkpoints dir:  {TRAIN_CONFIG['checkpoint_dir'].resolve()}")
    print(f"Train log:        {TRAIN_CONFIG['train_log_file'].resolve()}")
    print(f"Val log:          {TRAIN_CONFIG['val_log_file'].resolve()}")

    # Launch training
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
    )

    print(f"\nBest validation loss: {best_val_loss:.4f}")



if __name__ == '__main__':
    main()
