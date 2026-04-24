#!/usr/bin/env python3
"""
18_split_anylasis.py
====================
Analyses split quality and source balance across
data/5_dataset_{train,val,test}.tsv.

Sections:
  1. Split sizes  (clips, hours, word tokens, vocab)
  2. Source balance per split and overall
  3. Cross-split representativeness  (source ratio similarity)
  4. Duration & word-count statistics per split
  5. Short-clip audit  (word_count <= 2 or duration_sec < 0.5)

Run: python data_code/18_split_anylasis.py
"""

import csv
import sys
from collections import Counter
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

print(f"Split & Source Balance Analysis — generated {RUN_TS}")
print(SEP)
print(f"  Input : data/5_dataset_{{train,val,test}}.tsv")
print(f"  Log   : {LOG_PATH}")
print(SEP)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
header("0. LOADING")

data = {}
for split in SPLITS:
    path = DATA_DIR / f"5_dataset_{split}.tsv"
    if not path.exists():
        print(f"  [ERROR] Not found: {path}")
        sys.stdout = sys.__stdout__; _tee.close(); sys.exit(1)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    data[split] = rows
    print(f"  {split:5s}: {len(rows):>7,} rows")

# ---------------------------------------------------------------------------
# Helper: parse a split's rows into arrays
# ---------------------------------------------------------------------------
def parse_split(rows):
    durations   = []
    word_counts = []
    sources     = []
    texts       = []
    for row in rows:
        try:
            dur = float(row["duration_sec"])
        except (ValueError, KeyError):
            dur = 0.0
        try:
            wc = int(float(row.get("word_count") or 0))
        except ValueError:
            wc = 0
        durations.append(dur)
        word_counts.append(wc)
        sources.append(row.get("source", "unknown").strip().lower())
        texts.append(row.get("text", "").strip())
    return (
        np.array(durations,   dtype=float),
        np.array(word_counts, dtype=int),
        sources,
        texts,
    )

parsed = {s: parse_split(data[s]) for s in SPLITS}

# ---------------------------------------------------------------------------
# Section 1: Split sizes
# ---------------------------------------------------------------------------
header("1. SPLIT SIZES")

import re as _re
def tokenize(t): return _re.findall(r"[a-zA-Z']+", t.lower())

print(f"\n  {'Split':6}  {'Clips':>8}  {'Hours':>8}  {'Tokens':>10}  {'Vocab':>8}  {'%Total':>8}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}")

total_clips = sum(len(data[s]) for s in SPLITS)
grand_vocab = set()

split_info = {}
for split in SPLITS:
    durs, wcs, srcs, txts = parsed[split]
    hours   = durs.sum() / 3600
    tokens  = sum(len(tokenize(t)) for t in txts)
    vocab   = set(w for t in txts for w in tokenize(t))
    grand_vocab |= vocab
    pct     = 100 * len(data[split]) / total_clips
    split_info[split] = dict(clips=len(data[split]), hours=hours, tokens=tokens,
                              vocab=vocab, durations=durs, wcs=wcs, srcs=srcs)
    print(f"  {split:6}  {len(data[split]):>8,}  {hours:>8.2f}  {tokens:>10,}  {len(vocab):>8,}  {pct:>7.1f}%")

print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}")
total_hours  = sum(split_info[s]['hours']  for s in SPLITS)
total_tokens = sum(split_info[s]['tokens'] for s in SPLITS)
print(f"  {'TOTAL':6}  {total_clips:>8,}  {total_hours:>8.2f}  {total_tokens:>10,}  {len(grand_vocab):>8,}")

# ---------------------------------------------------------------------------
# Section 2: Source balance per split
# ---------------------------------------------------------------------------
header("2. SOURCE BALANCE PER SPLIT")

all_sources = sorted({s for sp in SPLITS for s in split_info[sp]['srcs']})

print(f"\n  {'Split':6}  " + "  ".join(f"{s:>12}" for s in all_sources) + f"  {'Total':>8}")
print(f"  {'─'*6}  " + "  ".join("─"*12 for _ in all_sources) + f"  {'─'*8}")

source_counts = {}
for split in SPLITS:
    ctr = Counter(split_info[split]['srcs'])
    source_counts[split] = ctr
    total = sum(ctr.values())
    parts = "  ".join(f"{ctr.get(s, 0):>6,} ({100*ctr.get(s,0)/total:>4.1f}%)" for s in all_sources)
    print(f"  {split:6}  {parts}  {total:>8,}")

# Overall
print(f"\n  Overall (all splits combined):")
overall = Counter()
for split in SPLITS:
    overall += source_counts[split]
overall_total = sum(overall.values())
for src in all_sources:
    pct = 100 * overall[src] / overall_total
    print(f"    {src:<14}: {overall[src]:>7,} clips  ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# Section 3: Cross-split source ratio similarity
# ---------------------------------------------------------------------------
header("3. CROSS-SPLIT SOURCE RATIO SIMILARITY")

print(f"\n  Ideal: val and test should mirror the train source ratio.")
print()

train_total = sum(source_counts['train'].values())
train_ratios = {s: source_counts['train'].get(s, 0) / train_total for s in all_sources}

print(f"  {'Split':6}  " + "  ".join(f"{s+' ratio':>14}" for s in all_sources) + "  Δ from train")
print(f"  {'─'*6}  " + "  ".join("─"*14 for _ in all_sources) + "  ─────────────")

for split in SPLITS:
    total = sum(source_counts[split].values()) or 1
    ratios = {s: source_counts[split].get(s, 0) / total for s in all_sources}
    delta = sum(abs(ratios[s] - train_ratios[s]) for s in all_sources) / len(all_sources)
    parts = "  ".join(f"{ratios[s]:>14.3f}" for s in all_sources)
    flag = "  ← reference" if split == "train" else f"  avg Δ = {delta:.3f}"
    print(f"  {split:6}  {parts}{flag}")

print(f"\n  (avg Δ is mean absolute difference in source ratio vs train;")
print(f"   < 0.05 = well-balanced,  > 0.10 = notable skew)")

# ---------------------------------------------------------------------------
# Section 4: Duration & word-count statistics per split
# ---------------------------------------------------------------------------
header("4. DURATION & WORD-COUNT STATISTICS")

for stat_name, key in [("Duration (seconds)", "durations"), ("Word count", "wcs")]:
    print(f"\n  {stat_name}:")
    print(f"  {'Split':6}  {'Min':>6}  {'p25':>6}  {'Median':>8}  {'Mean':>8}  {'p75':>6}  {'p95':>6}  {'Max':>8}  {'Std':>8}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")
    for split in SPLITS:
        arr = split_info[split][key]
        pcts = np.percentile(arr, [25, 50, 75, 95])
        print(f"  {split:6}  {arr.min():>6.2f}  {pcts[0]:>6.2f}  {pcts[1]:>8.2f}  "
              f"{arr.mean():>8.2f}  {pcts[2]:>6.2f}  {pcts[3]:>6.2f}  {arr.max():>8.2f}  {arr.std():>8.2f}")

# ---------------------------------------------------------------------------
# Section 5: Short-clip audit
# ---------------------------------------------------------------------------
header("5. SHORT-CLIP AUDIT  (word_count ≤ 2  or  duration_sec < 0.5)")

for split in SPLITS:
    durs = split_info[split]['durations']
    wcs  = split_info[split]['wcs']
    srcs = split_info[split]['srcs']
    rows = data[split]

    short_wc   = [i for i, w in enumerate(wcs)  if w <= 2]
    short_dur  = [i for i, d in enumerate(durs) if d < 0.5]
    short_both = [i for i in range(len(rows))   if wcs[i] <= 2 or durs[i] < 0.5]

    pct = 100 * len(short_both) / len(rows) if rows else 0
    print(f"\n  [{split}]  {len(short_both):,} short clips  ({pct:.2f}% of split)")
    print(f"    word_count ≤ 2   : {len(short_wc):,}")
    print(f"    duration < 0.5 s : {len(short_dur):,}")

    if short_both:
        src_ctr = Counter(srcs[i] for i in short_both)
        for src, cnt in src_ctr.most_common():
            print(f"    source {src:<14}: {cnt:,}")
        print(f"    Examples (up to 5):")
        for i in short_both[:5]:
            print(f"      dur={durs[i]:.2f}s  wc={wcs[i]}  text={rows[i].get('text','')!r:.80}")

# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  Analysis complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Report saved to: {LOG_PATH}")
print(f"{SEP}\n")

sys.stdout = sys.__stdout__
_tee.close()
