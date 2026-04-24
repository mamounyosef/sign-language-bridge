#!/usr/bin/env python3
"""
19_restratify_splits.py
=======================
Combines data/5_dataset_{train,val,test}.tsv into one pool, then
re-splits into 90/5/5 train/val/test stratified by source so each
split mirrors the overall source ratio.

Output: data/6_dataset_{train,val,test}.tsv
Log   : data_code/19_restratify_splits.txt

Run: python data_code/19_restratify_splits.py
"""

import csv
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

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

TRAIN_RATIO = 0.90
VAL_RATIO   = 0.05
TEST_RATIO  = 0.05
SEED        = 42

OUT_COLS = ["vid", "text", "duration_sec", "word_count", "file_path", "source"]

print(f"Re-stratify Splits — generated {RUN_TS}")
print(SEP)
print(f"  Input  : data/5_dataset_{{train,val,test}}.tsv")
print(f"  Output : data/6_dataset_{{train,val,test}}.tsv")
print(f"  Ratios : train={TRAIN_RATIO:.0%}  val={VAL_RATIO:.0%}  test={TEST_RATIO:.0%}")
print(f"  Seed   : {SEED}")
print(f"  Log    : {LOG_PATH}")
print(SEP)

# ---------------------------------------------------------------------------
# 1. Load and combine all splits
# ---------------------------------------------------------------------------
header("1. LOADING & COMBINING")

all_rows = []
for split in SPLITS:
    path = DATA_DIR / f"5_dataset_{split}.tsv"
    if not path.exists():
        print(f"  [ERROR] Not found: {path}")
        sys.stdout = sys.__stdout__; _tee.close(); sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    all_rows.extend(rows)
    print(f"  {split:5s}: {len(rows):>7,} rows loaded")

print(f"\n  Combined pool: {len(all_rows):,} rows")

source_counts = Counter(r.get("source", "unknown").strip().lower() for r in all_rows)
print(f"\n  Source breakdown of full pool:")
for src, cnt in source_counts.most_common():
    pct = 100 * cnt / len(all_rows)
    print(f"    {src:<14}: {cnt:>7,} ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# 2. Stratified split by source
# ---------------------------------------------------------------------------
header("2. STRATIFIED SPLIT  (by source)")

rng = random.Random(SEED)

# Group rows by source
by_source = {}
for row in all_rows:
    src = row.get("source", "unknown").strip().lower()
    by_source.setdefault(src, []).append(row)

train_rows, val_rows, test_rows = [], [], []

for src, rows in sorted(by_source.items()):
    rng.shuffle(rows)
    n        = len(rows)
    n_val    = round(n * VAL_RATIO)
    n_test   = round(n * TEST_RATIO)
    n_train  = n - n_val - n_test

    src_train = rows[:n_train]
    src_val   = rows[n_train : n_train + n_val]
    src_test  = rows[n_train + n_val :]

    train_rows.extend(src_train)
    val_rows.extend(src_val)
    test_rows.extend(src_test)

    print(f"\n  Source: {src}  (total {n:,})")
    print(f"    train : {len(src_train):>6,}  ({100*len(src_train)/n:.1f}%)")
    print(f"    val   : {len(src_val):>6,}  ({100*len(src_val)/n:.1f}%)")
    print(f"    test  : {len(src_test):>6,}  ({100*len(src_test)/n:.1f}%)")

# Shuffle within each final split so sources are interleaved
rng.shuffle(train_rows)
rng.shuffle(val_rows)
rng.shuffle(test_rows)

# ---------------------------------------------------------------------------
# 3. Write output files
# ---------------------------------------------------------------------------
header("3. WRITING OUTPUT FILES")

out_splits = {"train": train_rows, "val": val_rows, "test": test_rows}
for split, rows in out_splits.items():
    out_path = DATA_DIR / f"6_dataset_{split}.tsv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {split:5s}: {len(rows):>7,} rows  →  {out_path.name}")

# ---------------------------------------------------------------------------
# 4. Verification summary
# ---------------------------------------------------------------------------
header("4. VERIFICATION SUMMARY")

print(f"\n  {'Split':6}  {'Clips':>8}  {'%Total':>8}  " +
      "  ".join(f"{s:>12}" for s in sorted(by_source)) +
      f"  {'ΔRatio':>8}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  " +
      "  ".join("─"*12 for _ in by_source) +
      f"  {'─'*8}")

overall_total = len(all_rows)
overall_ratios = {src: cnt / overall_total for src, cnt in source_counts.items()}

for split, rows in out_splits.items():
    n   = len(rows)
    pct = 100 * n / overall_total
    ctr = Counter(r.get("source", "unknown").strip().lower() for r in rows)
    src_parts = "  ".join(
        f"{ctr.get(s,0):>6,} ({100*ctr.get(s,0)/n:>4.1f}%)"
        for s in sorted(by_source)
    )
    delta = sum(abs(ctr.get(s, 0) / n - overall_ratios[s])
                for s in by_source) / len(by_source)
    print(f"  {split:6}  {n:>8,}  {pct:>7.1f}%  {src_parts}  {delta:>8.4f}")

print(f"\n  (ΔRatio = mean absolute source ratio difference vs overall pool)")
print(f"  (< 0.005 = excellent stratification)")

print(f"\n  Total rows in output: {sum(len(r) for r in out_splits.values()):,}")
print(f"  Total rows in input : {len(all_rows):,}")
assert sum(len(r) for r in out_splits.values()) == len(all_rows), "Row count mismatch!"
print(f"  Row count check     : PASSED")

print(f"\n{SEP}")
print(f"  Re-stratification complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Report saved to: {LOG_PATH}")
print(f"{SEP}\n")

sys.stdout = sys.__stdout__
_tee.close()
