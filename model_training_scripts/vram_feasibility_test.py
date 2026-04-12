"""
VRAM Feasibility Test for Qwen3-VL-2B-Instruct Fine-Tuning

Tests multiple configurations (FPS, quantization) to determine the best
viable setup for training on the available GPU (RTX 4060, 8GB VRAM).

Run:  python model_training_scripts/vram_feasibility_test.py
"""

import os
import sys
import gc
import time
from pathlib import Path

import pandas as pd
import torch

# ── Configuration ──
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
TRAIN_TSV = Path('..') / 'data' / '2_dataset_train.tsv'
TSV_SEP = '\t'

# Pick a ~6s clip for worst-case VRAM testing
TARGET_DURATION = 6.0

# LoRA config (same as training)
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
LORA_DROPOUT = 0.05

# Test configurations: (name, fps, resolution, use_qlora)
TEST_CONFIGS = [
    ("A: bf16 LoRA, 20 FPS, 256px",  20, 256, False),
    ("B: QLoRA 4-bit, 20 FPS, 256px", 20, 256, True),
    ("C: bf16 LoRA, 18 FPS, 256px",  18, 256, False),
    ("D: QLoRA 4-bit, 18 FPS, 256px", 18, 256, True),
]


def get_test_sample(tsv_path, sep, target_duration):
    """Find a clip close to target_duration for worst-case VRAM testing."""
    df = pd.read_csv(tsv_path, sep=sep)
    df['dur_diff'] = (df['duration_sec'] - target_duration).abs()
    df = df.sort_values('dur_diff')

    for _, row in df.iterrows():
        fp = row['file_path']
        if Path(fp).exists():
            print(f"  Selected test clip: {row['vid']}")
            print(f"    Duration: {row['duration_sec']:.2f}s")
            print(f"    Text: {row['text'][:80]}...")
            print(f"    Path: {fp}")
            return {
                'vid': row['vid'],
                'text': row['text'],
                'duration_sec': row['duration_sec'],
                'file_path': fp,
            }

    raise RuntimeError("No valid video files found in the training TSV!")


def test_config(config_name, fps, resolution, use_qlora, sample):
    """Test a single configuration: load model, forward+backward, report VRAM."""
    print(f"\n{'=' * 70}")
    print(f"  TESTING: {config_name}")
    print(f"{'=' * 70}")

    # Force clean GPU state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, get_peft_model
        from qwen_vl_utils import process_vision_info

        # ── Load model ──
        print("  Loading model...")
        load_start = time.time()

        model_kwargs = {
            'dtype': torch.bfloat16,
            'device_map': 'cuda',
        }

        # Try Flash Attention 2, fall back to SDPA
        try:
            import flash_attn  # noqa: F401
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("  Using Flash Attention 2")
        except ImportError:
            model_kwargs['attn_implementation'] = 'sdpa'
            print("  Using SDPA (Flash Attention 2 not available)")

        if use_qlora:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
            )
            # device_map must be 'auto' for quantized models
            model_kwargs['device_map'] = 'auto'

        model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, **model_kwargs)

        if use_qlora:
            model = prepare_model_for_kbit_training(model)

        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        load_time = time.time() - load_start
        print(f"  Model loaded in {load_time:.1f}s")

        # Report model VRAM before LoRA
        model_vram = torch.cuda.memory_allocated() / 1e9
        print(f"  Model VRAM (before LoRA): {model_vram:.2f} GB")

        # ── Apply LoRA ──
        print("  Applying LoRA...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        lora_vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after LoRA: {lora_vram:.2f} GB")

        # ── Build chat messages ──
        print(f"  Building message (fps={fps}, resolution={resolution}x{resolution})...")
        max_pixels = (resolution // 16) ** 2 * 16 * 16  # = resolution^2, aligned to patch grid

        # Use plain local path — qwen_vl_utils supports local paths directly.
        # "file://D:/..." is malformed on Windows (needs "file:///D:/..."),
        # causing both decord and torchvision backends to fail.
        video_path = sample['file_path'].replace('\\', '/')

        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are a sign language translator."}
            ]},
            {"role": "user", "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "min_pixels": 4 * 32 * 32,          # Minimum 4 tokens
                    "max_pixels": max_pixels,
                    "total_pixels": 20480 * 32 * 32,     # Total pixel budget cap
                },
                {"type": "text", "text": "Translate this American Sign Language video into English."},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample['text']}
            ]},
        ]

        # ── Process with processor ──
        print("  Processing video + text...")
        process_start = time.time()

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16,
            return_video_kwargs=True, return_video_metadata=True,
        )

        # Unpack video tuples: process_vision_info returns list of (tensor, metadata)
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = processor(
            text=[text],
            images=images if images else None,
            videos=videos if videos else None,
            video_metadata=video_metadatas,
            return_tensors="pt",
            padding=True,
            do_resize=False,  # qwen_vl_utils already resized
            **video_kwargs,
        )

        process_time = time.time() - process_start

        # Move inputs to device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Report token counts
        input_ids = inputs['input_ids']
        total_tokens = input_ids.shape[1]
        # Count video tokens (token id 151656 = <|video_pad|> placeholder)
        video_token_id = 151656
        video_tokens = (input_ids == video_token_id).sum().item()
        text_tokens = total_tokens - video_tokens

        print(f"  Processing time: {process_time:.2f}s")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Video tokens: {video_tokens}")
        print(f"  Text tokens: {text_tokens}")

        if 'pixel_values_videos' in inputs:
            pvv = inputs['pixel_values_videos']
            print(f"  pixel_values_videos shape: {pvv.shape}")
            print(f"  pixel_values_videos VRAM: {pvv.element_size() * pvv.nelement() / 1e9:.3f} GB")

        pre_forward_vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM before forward: {pre_forward_vram:.2f} GB")

        # ── Build labels (mask non-assistant tokens) ──
        labels = inputs['input_ids'].clone()
        # Find assistant header: <|im_start|>assistant
        # For simplicity in the test, just use input_ids as labels (loss on everything)
        # The actual training script will do proper masking
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels

        # ── Forward pass ──
        print("  Running forward pass...")
        torch.cuda.reset_peak_memory_stats()
        forward_start = time.time()

        model.train()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**inputs)
            loss = outputs.loss

        forward_time = time.time() - forward_start
        forward_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Forward time: {forward_time:.2f}s")
        print(f"  Forward peak VRAM: {forward_peak:.2f} GB")
        print(f"  Loss: {loss.item():.4f}")

        # ── Backward pass ──
        print("  Running backward pass...")
        torch.cuda.reset_peak_memory_stats()
        backward_start = time.time()

        loss.backward()

        backward_time = time.time() - backward_start
        backward_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Backward time: {backward_time:.2f}s")
        print(f"  Backward peak VRAM: {backward_peak:.2f} GB")

        # ── Overall peak ──
        overall_peak = max(forward_peak, backward_peak)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"\n  {'=' * 50}")
        print(f"  RESULT: {config_name}")
        print(f"  {'=' * 50}")
        print(f"  Overall peak VRAM:  {overall_peak:.2f} GB / {gpu_total:.2f} GB")
        print(f"  VRAM headroom:      {gpu_total - overall_peak:.2f} GB")
        print(f"  Video tokens:       {video_tokens}")
        print(f"  Total tokens:       {total_tokens}")
        print(f"  Forward time:       {forward_time:.2f}s")
        print(f"  Backward time:      {backward_time:.2f}s")

        feasible = overall_peak < (gpu_total - 0.5)  # 0.5 GB safety margin
        status = "FEASIBLE" if feasible else "TOO TIGHT / OOM RISK"
        print(f"  Status:             {status}")

        return {
            'config': config_name,
            'peak_vram': overall_peak,
            'gpu_total': gpu_total,
            'video_tokens': video_tokens,
            'total_tokens': total_tokens,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'feasible': feasible,
        }

    except torch.cuda.OutOfMemoryError:
        print(f"  OOM! This configuration does not fit in VRAM.")
        return {
            'config': config_name,
            'peak_vram': float('inf'),
            'gpu_total': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            'video_tokens': 0,
            'total_tokens': 0,
            'forward_time': 0,
            'backward_time': 0,
            'feasible': False,
        }

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config': config_name,
            'peak_vram': 0,
            'gpu_total': 0,
            'video_tokens': 0,
            'total_tokens': 0,
            'forward_time': 0,
            'backward_time': 0,
            'feasible': False,
        }

    finally:
        # Clean up completely between tests
        try:
            del model, processor, inputs, outputs, loss, labels
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("  QWEN3-VL-2B VRAM FEASIBILITY TEST")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name}")
    print(f"  Total VRAM: {gpu_total:.2f} GB")

    # Find test sample
    print(f"\n  Finding test clip (~{TARGET_DURATION}s)...")
    sample = get_test_sample(TRAIN_TSV, TSV_SEP, TARGET_DURATION)

    # Run tests
    results = []
    for config_name, fps, resolution, use_qlora in TEST_CONFIGS:
        result = test_config(config_name, fps, resolution, use_qlora, sample)
        results.append(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<40} {'Peak VRAM':>10} {'Tokens':>8} {'Status':>15}")
    print("  " + "-" * 75)

    best = None
    for r in results:
        peak_str = f"{r['peak_vram']:.2f} GB" if r['peak_vram'] != float('inf') else "OOM"
        status = "FEASIBLE" if r['feasible'] else "NO"
        print(f"  {r['config']:<40} {peak_str:>10} {r['total_tokens']:>8} {status:>15}")
        if r['feasible'] and (best is None or
            # Prefer non-QLoRA (better quality) then higher FPS
            (not 'QLoRA' in r['config'] and 'QLoRA' in best['config']) or
            (('QLoRA' in r['config']) == ('QLoRA' in best['config']) and
             r['total_tokens'] > best['total_tokens'])):
            best = r

    if best:
        print(f"\n  RECOMMENDED: {best['config']}")
        print(f"    Peak VRAM: {best['peak_vram']:.2f} GB / {best['gpu_total']:.2f} GB")
        print(f"    Video tokens: {best['video_tokens']}")
    else:
        print("\n  WARNING: No configuration fits! Consider:")
        print("    - Reducing video resolution below 256")
        print("    - Using a smaller model")
        print("    - Using a GPU with more VRAM")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
