import torch
import torch.nn as nn
import time

torch.set_float32_matmul_precision('high')

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

def benchmark(model, x, n=500, label=""):
    # Warmup
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"{label}: {elapsed:.3f}s for {n} iters ({elapsed/n*1000:.3f} ms/iter)")

device = "cuda"
x = torch.randn(32, 128, device=device)

eager_model = SimpleModel().to(device)
compiled_model = torch.compile(SimpleModel().to(device))

# Trigger compilation first
_ = compiled_model(x)
torch.cuda.synchronize()

benchmark(eager_model, x, label="Eager  ")
benchmark(compiled_model, x, label="Compiled")