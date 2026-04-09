from torchvision.models import resnet50, ResNet50_Weights
import torch

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
print("*" * 65)
print(model)
print("*" * 65)
print("=" * 65)
print("  ResNet50 — Layer-by-layer parameter breakdown")
print("=" * 65)
total = 0
for name, module in model.named_children():
    p = sum(x.numel() for x in module.parameters())
    total += p
    print(f"\n[{name}]  type={type(module).__name__}  params={p:,}")
    for sub_name, sub_mod in module.named_children():
        sp = sum(x.numel() for x in sub_mod.parameters())
        print(f"    └─ [{sub_name}]  type={type(sub_mod).__name__}  params={sp:,}")
        for sub2_name, sub2_mod in sub_mod.named_children():
            sp2 = sum(x.numel() for x in sub2_mod.parameters())
            print(f"         └─ [{sub2_name}]  params={sp2:,}")
print(f"\nTotal: {total:,} params")

# Quick trainability check — recommended freeze config:
freeze_names = ['conv1', 'bn1', 'layer1', 'layer2']
trainable = sum(p.numel() for n, p in model.named_parameters()
                if not any(n.startswith(f) for f in freeze_names))
frozen   = sum(p.numel() for n, p in model.named_parameters()
               if any(n.startswith(f) for f in freeze_names))
print(f"\nWith freeze={freeze_names}:")
print(f"  Frozen    : {frozen:,}")
print(f"  Trainable : {trainable:,}")