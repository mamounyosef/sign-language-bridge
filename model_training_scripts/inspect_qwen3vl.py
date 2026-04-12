"""
Inspect Qwen3-VL-2B-Instruct architecture.
Loads the model on CPU (no GPU needed) and prints full architecture details.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

def count_parameters(module):
    """Count total and trainable parameters for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def format_params(n):
    """Format parameter count in human-readable form."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)

def main():
    print("=" * 80)
    print(f"Loading model config: {MODEL_NAME}")
    print("=" * 80)

    # 1. Print config first (no download of weights needed)
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("\n### MODEL CONFIG ###")
    print(config)

    # 2. Load processor
    print("\n" + "=" * 80)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"Tokenizer class: {type(processor.tokenizer).__name__}")
    print(f"Vocab size: {processor.tokenizer.vocab_size}")
    print(f"Model max length: {processor.tokenizer.model_max_length}")
    print(f"Special tokens: {processor.tokenizer.special_tokens_map}")
    if hasattr(processor, 'image_processor'):
        print(f"\nImage processor class: {type(processor.image_processor).__name__}")
        print(f"Image processor config: {processor.image_processor.to_dict()}")

    # 3. Load model on CPU (no GPU required)
    print("\n" + "=" * 80)
    print("Loading model on CPU (this may take a minute)...")
    print("=" * 80)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # CPU doesn't support bf16 well
        device_map="cpu",
        trust_remote_code=True,
    )

    # 4. Print full architecture
    print("\n### FULL MODEL ARCHITECTURE ###")
    print(model)

    # 5. Parameter breakdown by top-level module
    print("\n" + "=" * 80)
    print("### PARAMETER BREAKDOWN BY MODULE ###")
    print("=" * 80)
    total_all, trainable_all = count_parameters(model)
    print(f"\nTotal parameters: {format_params(total_all)} ({total_all:,})")
    print(f"Trainable parameters: {format_params(trainable_all)} ({trainable_all:,})")

    print("\nTop-level modules:")
    print(f"{'Module':<50} {'Total':>15} {'Trainable':>15}")
    print("-" * 80)
    for name, module in model.named_children():
        total, trainable = count_parameters(module)
        print(f"{name:<50} {format_params(total):>15} {format_params(trainable):>15}")

    # 6. Detailed breakdown — second level
    print("\n" + "=" * 80)
    print("### DETAILED PARAMETER BREAKDOWN (2 levels deep) ###")
    print("=" * 80)
    for name, module in model.named_children():
        total, _ = count_parameters(module)
        print(f"\n--- {name} ({format_params(total)}) ---")
        for sub_name, sub_module in module.named_children():
            sub_total, sub_trainable = count_parameters(sub_module)
            print(f"  {sub_name:<45} {format_params(sub_total):>12} params")
            # Go one more level for important submodules
            for sub_sub_name, sub_sub_module in sub_module.named_children():
                ss_total, _ = count_parameters(sub_sub_module)
                if ss_total > 0:
                    print(f"    {sub_sub_name:<43} {format_params(ss_total):>12} params")

    # 7. List all named modules with their types (for LoRA target selection)
    print("\n" + "=" * 80)
    print("### ALL LINEAR LAYERS (potential LoRA targets) ###")
    print("=" * 80)
    print(f"{'Layer Name':<70} {'Shape':>25}")
    print("-" * 95)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            shape = f"{tuple(module.weight.shape)}"
            print(f"{name:<70} {shape:>25}")

    # 8. List all norm layers
    print("\n" + "=" * 80)
    print("### ALL NORM LAYERS ###")
    print("=" * 80)
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)) or \
           'norm' in type(module).__name__.lower():
            total, _ = count_parameters(module)
            print(f"{name:<70} {type(module).__name__:<20} {format_params(total):>10}")

    # 9. Vision encoder specifics
    print("\n" + "=" * 80)
    print("### VISION ENCODER DETAILS ###")
    print("=" * 80)
    if hasattr(model, 'visual') or hasattr(model, 'vision_model') or hasattr(model, 'model'):
        # Try to find the vision component
        vision = None
        for attr_name in ['visual', 'vision_model', 'vision_tower']:
            if hasattr(model, attr_name):
                vision = getattr(model, attr_name)
                print(f"Vision component found at: model.{attr_name}")
                break
        if vision is None and hasattr(model, 'model'):
            for attr_name in ['visual', 'vision_model', 'vision_tower']:
                if hasattr(model.model, attr_name):
                    vision = getattr(model.model, attr_name)
                    print(f"Vision component found at: model.model.{attr_name}")
                    break

        if vision is not None:
            v_total, v_trainable = count_parameters(vision)
            print(f"Vision parameters: {format_params(v_total)} ({v_total:,})")
            if hasattr(vision, 'config'):
                print(f"Vision config: {vision.config}")
        else:
            print("Could not locate vision component — check full architecture above")

    # 10. Embedding and output head
    print("\n" + "=" * 80)
    print("### EMBEDDING & OUTPUT HEAD ###")
    print("=" * 80)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            print(f"{name:<60} Embedding({module.num_embeddings}, {module.embedding_dim})")
    for name, module in model.named_modules():
        if 'lm_head' in name and isinstance(module, torch.nn.Linear):
            print(f"{name:<60} Linear({module.in_features}, {module.out_features})")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
