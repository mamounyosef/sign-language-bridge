#!/usr/bin/env python3
"""
20_token_length_analysis.py
============================
Tokenises the text column of data/6_dataset_{train,val,test}.tsv
using the same tokenizer as the training pipeline
(Qwen/Qwen3-VL-2B-Instruct via AutoTokenizer) and reports
detailed token-length statistics.

Statistics reported:
  - Per split: min / p1 / p5 / p25 / p50 / p75 / p95 / p99 / max / mean / std
  - Total token count per split (and combined)
  - Bucket distribution (0-10, 11-20, ..., 101+)
  - Clips exceeding common max_text_tokens thresholds (32, 48, 64, 96, 128)
  - Top 10 longest examples per split

Run: python data_code/20_token_length_analysis.py
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Tee
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, filepath):
        self._file   = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.__stdout__
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CODE_DIR = Path(__file__).resolve().parent
SPLITS   = ["train", "val", "test"]

LOG_PATH = CODE_DIR / (Path(__file__).stem + ".txt")
_tee     = _Tee(LOG_PATH)
sys.stdout = _tee

RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SEP = "=" * 72
SUB = "-" * 72
def header(t):    print(f"\n{SEP}\n  {t}\n{SEP}")
def subheader(t): print(f"\n{SUB}\n  {t}\n{SUB}")

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
THRESHOLDS = [32, 48, 64, 96, 128]

print(f"Token Length Analysis — generated {RUN_TS}")
print(SEP)
print(f"  Input     : data/6_dataset_{{train,val,test}}.tsv")
print(f"  Tokenizer : {MODEL_NAME}")
print(f"  Log       : {LOG_PATH}")
print(SEP)

# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------
header("0. LOADING TOKENIZER")

try:
    from transformers import AutoTokenizer
    print(f"  Loading {MODEL_NAME} (local_files_only=True) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"  Vocab size : {tokenizer.vocab_size:,}")
    print(f"  Model max length: {getattr(tokenizer, 'model_max_length', 'N/A')}")
except Exception as e:
    print(f"  [ERROR] Could not load tokenizer: {e}")
    sys.stdout = sys.__stdout__; _tee.close(); sys.exit(1)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
header("1. LOADING DATA")

split_data = {}
for split in SPLITS:
    path = DATA_DIR / f"6_dataset_{split}.tsv"
    if not path.exists():
        print(f"  [ERROR] Not found: {path}")
        sys.stdout = sys.__stdout__; _tee.close(); sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    split_data[split] = rows
    print(f"  {split:5s}: {len(rows):>7,} rows")

# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------
header("2. TOKENIZING")

split_lengths = {}   # split -> np.array of token lengths
split_rows    = {}   # split -> list of (length, text) for long-example display

for split in SPLITS:
    rows   = split_data[split]
    texts  = [r.get("text", "").strip() for r in rows]
    print(f"  [{split}] tokenizing {len(texts):,} texts ...", end=" ", flush=True)
    encoded = tokenizer(texts, add_special_tokens=False)
    lengths = np.array([len(ids) for ids in encoded["input_ids"]], dtype=int)
    split_lengths[split] = lengths
    split_rows[split]    = list(zip(lengths.tolist(), texts))
    print(f"done  (total tokens: {lengths.sum():,})")

# ---------------------------------------------------------------------------
# 3. Per-split statistics
# ---------------------------------------------------------------------------
header("3. TOKEN LENGTH STATISTICS")

PCTS = [1, 5, 25, 50, 75, 95, 99]

print(f"\n  {'Stat':<10}" + "  ".join(f"{s:>10}" for s in SPLITS))
print(f"  {'─'*10}" + "  ".join("─"*10 for _ in SPLITS))

rows_stats = {}
for split in SPLITS:
    arr = split_lengths[split]
    pct_vals = np.percentile(arr, PCTS)
    rows_stats[split] = {
        "min":  int(arr.min()),
        "max":  int(arr.max()),
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "total": int(arr.sum()),
        **{f"p{p}": float(v) for p, v in zip(PCTS, pct_vals)},
    }

for label, key in [
    ("min",    "min"),
    ("p1",     "p1"),
    ("p5",     "p5"),
    ("p25",    "p25"),
    ("p50",    "p50"),
    ("p75",    "p75"),
    ("p95",    "p95"),
    ("p99",    "p99"),
    ("max",    "max"),
    ("mean",   "mean"),
    ("std",    "std"),
]:
    vals = "  ".join(f"{rows_stats[s][key]:>10.1f}" for s in SPLITS)
    print(f"  {label:<10}{vals}")

print(f"\n  Total tokens per split:")
grand_total = 0
for split in SPLITS:
    t = rows_stats[split]["total"]
    grand_total += t
    print(f"    {split:5s}: {t:>12,}")
print(f"    {'TOTAL':5}: {grand_total:>12,}")

# ---------------------------------------------------------------------------
# 4. Bucket distribution
# ---------------------------------------------------------------------------
header("4. BUCKET DISTRIBUTION")

BUCKETS = list(range(0, 101, 10)) + [float("inf")]
bucket_labels = [f"{BUCKETS[i]+1 if i>0 else 0}–{int(BUCKETS[i+1])}"
                 if BUCKETS[i+1] != float("inf")
                 else f"{BUCKETS[i]+1 if i>0 else 101}+"
                 for i in range(len(BUCKETS)-1)]

print(f"\n  {'Bucket':>10}  " + "  ".join(f"{s:>14}" for s in SPLITS))
print(f"  {'─'*10}  " + "  ".join("─"*14 for _ in SPLITS))

for i, label in enumerate(bucket_labels):
    lo = BUCKETS[i] + (1 if i > 0 else 0)
    hi = BUCKETS[i+1]
    parts = []
    for split in SPLITS:
        arr  = split_lengths[split]
        mask = (arr >= lo) & (arr < hi) if hi != float("inf") else (arr >= lo)
        cnt  = int(mask.sum())
        pct  = 100 * cnt / len(arr)
        parts.append(f"{cnt:>6,} ({pct:>4.1f}%)")
    print(f"  {label:>10}  " + "  ".join(f"{p:>14}" for p in parts))

# ---------------------------------------------------------------------------
# 5. Threshold exceedance
# ---------------------------------------------------------------------------
header("5. CLIPS EXCEEDING max_text_tokens THRESHOLDS")

print(f"\n  {'Threshold':>10}  " + "  ".join(f"{s:>16}" for s in SPLITS))
print(f"  {'─'*10}  " + "  ".join("─"*16 for _ in SPLITS))

for thresh in THRESHOLDS:
    parts = []
    for split in SPLITS:
        arr  = split_lengths[split]
        cnt  = int((arr > thresh).sum())
        pct  = 100 * cnt / len(arr)
        parts.append(f"{cnt:>6,} ({pct:>5.2f}%)")
    print(f"  {thresh:>10}  " + "  ".join(f"{p:>16}" for p in parts))

print(f"\n  Recommendation: choose max_text_tokens where < 1–2% of clips are truncated.")

# ---------------------------------------------------------------------------
# 6. Top 10 longest examples per split
# ---------------------------------------------------------------------------
header("6. TOP 10 LONGEST EXAMPLES PER SPLIT")

for split in SPLITS:
    subheader(f"Split: {split}")
    top10 = sorted(split_rows[split], key=lambda x: x[0], reverse=True)[:10]
    print(f"  {'Tokens':>7}  Text")
    print(f"  {'─'*7}  {'─'*60}")
    for length, text in top10:
        print(f"  {length:>7}  {text[:80]!r}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  Analysis complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Report saved to: {LOG_PATH}")
print(f"{SEP}\n")

sys.stdout = sys.__stdout__
_tee.close()
