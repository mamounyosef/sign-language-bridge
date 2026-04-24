#!/usr/bin/env python3
"""
14_wps_and_fuzzy_analysis.py
=============================
Two-part analysis on data/3_dataset_{train,val,test}.tsv:

  Part A — Words-Per-Second (WPS) Analysis
    Computes WPS = word_count / duration_sec for every clip.
    Reports full statistics, per-bucket distributions, and all outlier rows
    (WPS < 0.5  →  suspiciously slow / likely misaligned transcript
     WPS > 6.0  →  suspiciously fast / likely corrupt or split error)

  Part B — Full-Scale Fuzzy Near-Duplicate Analysis
    Finds pairs of clips whose normalised text is >= 90% similar using
    blocking by word_count (±1) to keep the comparison tractable.
    Reports within-split pairs and cross-split pairs (potential leakage).
    Uses rapidfuzz if installed (strongly recommended), falls back to difflib.

Run : python data_code/14_wps_and_fuzzy_analysis.py
Deps: pandas, (optional but recommended) rapidfuzz
"""

import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    from rapidfuzz import process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# Tee — mirror stdout to log file
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, filepath):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.__stdout__
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CODE_DIR = Path(__file__).resolve().parent
SPLITS   = ["train", "val", "test"]

LOG_PATH = CODE_DIR / "14_wps_and_fuzzy_report.txt"
_tee     = _Tee(LOG_PATH)
sys.stdout = _tee

RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------------
# WPS thresholds
# ---------------------------------------------------------------------------
WPS_LOW  = 0.5   # clips below this are suspiciously slow
WPS_HIGH = 6.0   # clips above this are suspiciously fast

# Fuzzy threshold
FUZZY_THRESHOLD = 90   # rapidfuzz ratio (0-100)

# Without rapidfuzz: cap block size to keep runtime reasonable
NO_RAPIDFUZZ_BLOCK_CAP = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEP = "=" * 72
SUB = "-" * 72

def header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subheader(title):
    print(f"\n{SUB}\n  {title}\n{SUB}")

def normalise(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def sim_ratio(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b)
    return SequenceMatcher(None, a, b).ratio() * 100


# ---------------------------------------------------------------------------
# Header banner
# ---------------------------------------------------------------------------
print(f"WPS & Fuzzy Near-Duplicate Analysis Report — generated {RUN_TS}")
print(SEP)
print(f"  Fuzzy engine : {'rapidfuzz' if HAS_RAPIDFUZZ else 'difflib  ← install rapidfuzz for full-scale speed'}")
print(f"  WPS outliers : LOW < {WPS_LOW}  |  HIGH > {WPS_HIGH}")
print(f"  Fuzzy cutoff : >= {FUZZY_THRESHOLD}% similarity  (normalised text, word_count ±1 blocking)")
print(SEP)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
header("0. LOADING DATA")

dfs = {}
for split in SPLITS:
    path = DATA_DIR / f"3_dataset_{split}.tsv"
    if not path.exists():
        print(f"  [ERROR] Not found: {path}")
        sys.stdout = sys.__stdout__; _tee.close(); sys.exit(1)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    df["word_count"]   = pd.to_numeric(df["word_count"],   errors="coerce")
    df["_wps"]         = df["word_count"] / df["duration_sec"]
    df["_norm"]        = df["text"].apply(normalise)
    df["_wc_int"]      = df["word_count"].fillna(0).astype(int)
    dfs[split] = df
    print(f"  {split:5s}: {len(df):>7,} rows  —  {path.name}")

print(f"\n  Total rows: {sum(len(df) for df in dfs.values()):,}")


# ===========================================================================
# PART A — WPS ANALYSIS
# ===========================================================================
header("PART A — WORDS-PER-SECOND (WPS) ANALYSIS")
print("  WPS = word_count / duration_sec")
print(f"  Outlier thresholds:  LOW < {WPS_LOW}  |  HIGH > {WPS_HIGH}\n")


# ---- A1. Statistics per split ----
subheader("A1. WPS Statistics Per Split")

all_wps_rows = []
for split, df in dfs.items():
    valid = df["_wps"].dropna()
    p = valid.describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99])
    print(f"  [{split}]  n={len(valid):,}  (rows with valid duration & word_count)")
    print(f"    Min        : {p['min']:.4f}")
    print(f"    1st pct    : {p['1%']:.4f}")
    print(f"    5th pct    : {p['5%']:.4f}")
    print(f"    10th pct   : {p['10%']:.4f}")
    print(f"    25th pct   : {p['25%']:.4f}")
    print(f"    Mean       : {p['mean']:.4f}")
    print(f"    Median     : {p['50%']:.4f}")
    print(f"    75th pct   : {p['75%']:.4f}")
    print(f"    90th pct   : {p['90%']:.4f}")
    print(f"    95th pct   : {p['95%']:.4f}")
    print(f"    99th pct   : {p['99%']:.4f}")
    print(f"    Max        : {p['max']:.4f}")
    print(f"    Std        : {p['std']:.4f}")
    print()
    all_wps_rows.append(df.assign(_split=split))

all_df = pd.concat(all_wps_rows, ignore_index=True)
valid_all = all_df["_wps"].dropna()
p_all = valid_all.describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99])
print(f"  [ALL SPLITS COMBINED]  n={len(valid_all):,}")
print(f"    Min / Mean / Median / Max  :  {p_all['min']:.4f} / {p_all['mean']:.4f} / {p_all['50%']:.4f} / {p_all['max']:.4f}")
print(f"    Std                        :  {p_all['std']:.4f}")


# ---- A2. Distribution buckets ----
subheader("A2. WPS Distribution Buckets (combined across all splits)")

buckets = [
    (0.0,  0.5,  "0.0 – 0.5   [VERY LOW  — suspicious]"),
    (0.5,  1.0,  "0.5 – 1.0   [Low]"),
    (1.0,  2.0,  "1.0 – 2.0   [Normal low]"),
    (2.0,  3.0,  "2.0 – 3.0   [Normal]"),
    (3.0,  4.0,  "3.0 – 4.0   [Normal high]"),
    (4.0,  5.0,  "4.0 – 5.0   [High]"),
    (5.0,  6.0,  "5.0 – 6.0   [Very high]"),
    (6.0,  999., "6.0+        [VERY HIGH — suspicious]"),
]

total_valid = len(valid_all)
print(f"  {'WPS Range':<40}  {'Count':>7}  {'%':>6}")
print(f"  {'─'*40}  {'─'*7}  {'─'*6}")
for lo, hi, label in buckets:
    cnt = ((valid_all >= lo) & (valid_all < hi)).sum()
    pct = 100 * cnt / total_valid if total_valid else 0
    print(f"  {label:<40}  {cnt:>7,}  {pct:>5.2f}%")


# ---- A3. Rows with NaN WPS (duration or word_count missing/zero) ----
subheader("A3. Rows With Invalid / Missing WPS")

for split, df in dfs.items():
    bad = df[df["_wps"].isna() | (df["duration_sec"] == 0) | (df["word_count"] == 0)]
    print(f"  [{split}] {len(bad)} invalid WPS rows (NaN duration, NaN word_count, or zero values)")
    if len(bad) > 0:
        for _, row in bad.iterrows():
            print(f"    vid={row['vid']}  dur={row['duration_sec']}  wc={row['word_count']}  text={str(row['text'])[:60]!r}")


# ---- A4. Low WPS outliers ----
subheader(f"A4. LOW WPS Outliers  (WPS < {WPS_LOW})  — full list per split")

for split, df in dfs.items():
    low = df[df["_wps"] < WPS_LOW].sort_values("_wps")
    pct = 100 * len(low) / len(df)
    print(f"\n  [{split}]  {len(low):,} rows ({pct:.2f}%) with WPS < {WPS_LOW}")
    if len(low) == 0:
        continue
    # Source breakdown
    for src, grp in low.groupby("source"):
        print(f"    {src}: {len(grp):,} rows")
    print()
    print(f"  {'#':>4}  {'WPS':>6}  {'Dur':>6}  {'WC':>4}  {'Source':10}  {'vid':40}  Text (first 70 chars)")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*4}  {'─'*10}  {'─'*40}  {'─'*40}")
    for rank, (_, row) in enumerate(low.iterrows(), 1):
        print(
            f"  {rank:>4}  {row['_wps']:>6.3f}  {row['duration_sec']:>6.2f}  "
            f"{int(row['word_count']) if pd.notna(row['word_count']) else '?':>4}  "
            f"{str(row['source'])[:10]:10}  "
            f"{str(row['vid'])[:40]:40}  "
            f"{str(row['text'])[:70]!r}"
        )


# ---- A5. High WPS outliers ----
subheader(f"A5. HIGH WPS Outliers  (WPS > {WPS_HIGH})  — full list per split")

for split, df in dfs.items():
    high = df[df["_wps"] > WPS_HIGH].sort_values("_wps", ascending=False)
    pct = 100 * len(high) / len(df)
    print(f"\n  [{split}]  {len(high):,} rows ({pct:.2f}%) with WPS > {WPS_HIGH}")
    if len(high) == 0:
        continue
    for src, grp in high.groupby("source"):
        print(f"    {src}: {len(grp):,} rows")
    print()
    print(f"  {'#':>4}  {'WPS':>6}  {'Dur':>6}  {'WC':>4}  {'Source':10}  {'vid':40}  Text (first 70 chars)")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*4}  {'─'*10}  {'─'*40}  {'─'*40}")
    for rank, (_, row) in enumerate(high.iterrows(), 1):
        print(
            f"  {rank:>4}  {row['_wps']:>6.3f}  {row['duration_sec']:>6.2f}  "
            f"{int(row['word_count']) if pd.notna(row['word_count']) else '?':>4}  "
            f"{str(row['source'])[:10]:10}  "
            f"{str(row['vid'])[:40]:40}  "
            f"{str(row['text'])[:70]!r}"
        )


# ---- A6. WPS summary ----
subheader("A6. WPS Outlier Summary Table")

print(f"  {'Split':6}  {'Total':>8}  {'Low (<{:.1f})'.format(WPS_LOW):>10}  {'High (>{:.1f})'.format(WPS_HIGH):>11}  {'% outlier':>10}")
print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*11}  {'─'*10}")
grand_total = grand_low = grand_high = 0
for split, df in dfs.items():
    total = len(df)
    n_low  = (df["_wps"] < WPS_LOW).sum()
    n_high = (df["_wps"] > WPS_HIGH).sum()
    pct = 100 * (n_low + n_high) / total
    grand_total += total; grand_low += n_low; grand_high += n_high
    print(f"  {split:6}  {total:>8,}  {n_low:>10,}  {n_high:>11,}  {pct:>9.2f}%")
grand_pct = 100 * (grand_low + grand_high) / grand_total
print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*11}  {'─'*10}")
print(f"  {'TOTAL':6}  {grand_total:>8,}  {grand_low:>10,}  {grand_high:>11,}  {grand_pct:>9.2f}%")


# ===========================================================================
# PART B — FUZZY NEAR-DUPLICATE ANALYSIS
# ===========================================================================
header("PART B — FUZZY NEAR-DUPLICATE ANALYSIS")
print(f"  Method  : {'rapidfuzz fuzz.ratio' if HAS_RAPIDFUZZ else 'difflib SequenceMatcher'}")
print(f"  Cutoff  : >= {FUZZY_THRESHOLD}% similarity")
print(f"  Blocking: word_count ±1  (only compare texts of similar length)")
if not HAS_RAPIDFUZZ:
    print(f"  WARNING : rapidfuzz not installed. Block size capped at {NO_RAPIDFUZZ_BLOCK_CAP} per word_count")
    print(f"            to keep runtime reasonable. Install rapidfuzz for full-scale analysis.")
print()


def build_wc_index(df):
    """Returns dict: word_count -> list of (row_index, norm_text, vid, original_text)."""
    idx = defaultdict(list)
    for i, row in df.iterrows():
        wc = int(row["_wc_int"])
        idx[wc].append((i, row["_norm"], row["vid"], row["text"]))
    return idx


def find_fuzzy_pairs_within(df, split_name):
    """Find all fuzzy near-duplicate pairs within a single split."""
    wc_idx = build_wc_index(df)
    pairs = []
    seen = set()

    all_wcs = sorted(wc_idx.keys())
    t0 = time.time()
    for wc in all_wcs:
        for group_a_wc, group_b_wc in [(wc, wc), (wc, wc + 1)]:
            group_a = wc_idx.get(group_a_wc, [])
            group_b = wc_idx.get(group_b_wc, [])
            if not group_a or not group_b:
                continue
            if group_a_wc == group_b_wc and len(group_a) < 2:
                continue

            if not HAS_RAPIDFUZZ:
                group_a = group_a[:NO_RAPIDFUZZ_BLOCK_CAP]
                group_b = group_b[:NO_RAPIDFUZZ_BLOCK_CAP]

            texts_a = [x[1] for x in group_a]
            texts_b = [x[1] for x in group_b]

            label = f"wc={group_a_wc}" + (f"×{group_b_wc}" if group_a_wc != group_b_wc else "(self)")
            print(f"    {label}  ({len(texts_a)}×{len(texts_b)} comparisons) ...")
            sys.stdout.flush()

            if HAS_RAPIDFUZZ:
                scores = rf_process.cdist(texts_a, texts_b, scorer=fuzz.ratio,
                                          score_cutoff=FUZZY_THRESHOLD, workers=-1)
                scores = np.asarray(scores)
            else:
                scores = np.array([[sim_ratio(texts_a[i], texts_b[j])
                                    for j in range(len(texts_b))]
                                   for i in range(len(texts_a))], dtype=np.float32)

            # Use numpy to find matching cells — avoids O(n²) Python loop
            if group_a_wc == group_b_wc:
                # Upper triangle only (j > i) to avoid self-pairs and duplicates
                match_rows, match_cols = np.nonzero(np.triu(scores, k=1))
            else:
                match_rows, match_cols = np.nonzero(scores)

            for r, c in zip(match_rows.tolist(), match_cols.tolist()):
                idx_a = group_a[r][0]
                idx_b = group_b[c][0]
                pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs.append({
                        "score":  float(scores[r, c]),
                        "vid_a":  group_a[r][2], "text_a": group_a[r][3], "norm_a": group_a[r][1],
                        "vid_b":  group_b[c][2], "text_b": group_b[c][3], "norm_b": group_b[c][1],
                        "wc_a":   group_a_wc,    "wc_b":   group_b_wc,
                    })

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s — {len(pairs)} pairs found.")
    return sorted(pairs, key=lambda x: -x["score"])


def find_fuzzy_pairs_cross(df_a, split_a, df_b, split_b):
    """Find fuzzy near-duplicate pairs between two different splits."""
    wc_idx_a = build_wc_index(df_a)
    wc_idx_b = build_wc_index(df_b)
    pairs = []
    seen = set()
    all_wcs = sorted(set(wc_idx_a.keys()) | set(wc_idx_b.keys()))

    t0 = time.time()
    for wc in all_wcs:
        for wc_offset in [0, 1]:
            group_a = wc_idx_a.get(wc, [])
            group_b = wc_idx_b.get(wc + wc_offset, [])
            if not group_a or not group_b:
                continue

            if not HAS_RAPIDFUZZ:
                group_a = group_a[:NO_RAPIDFUZZ_BLOCK_CAP]
                group_b = group_b[:NO_RAPIDFUZZ_BLOCK_CAP]

            texts_a = [x[1] for x in group_a]
            texts_b = [x[1] for x in group_b]

            if HAS_RAPIDFUZZ:
                scores = rf_process.cdist(texts_a, texts_b, scorer=fuzz.ratio,
                                          score_cutoff=FUZZY_THRESHOLD, workers=-1)
                scores = np.asarray(scores)
            else:
                scores = np.array([[sim_ratio(texts_a[i], texts_b[j])
                                    for j in range(len(texts_b))]
                                   for i in range(len(texts_a))], dtype=np.float32)

            match_rows, match_cols = np.nonzero(scores)
            for r, c in zip(match_rows.tolist(), match_cols.tolist()):
                key = (group_a[r][2], group_b[c][2])
                if key not in seen:
                    seen.add(key)
                    pairs.append({
                        "score":   float(scores[r, c]),
                        "split_a": split_a, "vid_a": group_a[r][2], "text_a": group_a[r][3],
                        "split_b": split_b, "vid_b": group_b[c][2], "text_b": group_b[c][3],
                        "wc_a":    wc,      "wc_b":  wc + wc_offset,
                    })

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s — {len(pairs)} pairs found.")
    return sorted(pairs, key=lambda x: -x["score"])


def print_pairs(pairs, cap=None):
    if not pairs:
        print("  No pairs found.")
        return
    total = len(pairs)
    shown = pairs if cap is None or total <= cap else pairs[:cap]
    print(f"  {'#':>5}  {'Sim%':>5}  {'WC_a':>4}  {'WC_b':>4}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*4}  {'─'*4}")
    for rank, p in enumerate(shown, 1):
        print(f"  {rank:>5}  {p['score']:>5.1f}  {p.get('wc_a','?'):>4}  {p.get('wc_b','?'):>4}")
        print(f"    A vid : {str(p['vid_a'])[:70]}")
        print(f"    A text: {str(p['text_a'])[:100]!r}")
        print(f"    B vid : {str(p['vid_b'])[:70]}")
        print(f"    B text: {str(p['text_b'])[:100]!r}")
    if cap is not None and total > cap:
        print(f"\n  ... {total - cap} more pairs not shown (total: {total:,})")


# ---- B1. Within-split fuzzy pairs ----
subheader("B1. Within-Split Fuzzy Near-Duplicates")

within_results = {}
for split, df in dfs.items():
    print(f"\n  [{split}]  Scanning {len(df):,} rows ...")
    sys.stdout.flush()
    pairs = find_fuzzy_pairs_within(df, split)
    within_results[split] = pairs
    src_breakdown = {}
    for p in pairs:
        row_a = df[df["vid"] == p["vid_a"]]
        row_b = df[df["vid"] == p["vid_b"]]
        src_a = row_a["source"].values[0] if len(row_a) else "?"
        src_b = row_b["source"].values[0] if len(row_b) else "?"
        key = tuple(sorted([src_a, src_b]))
        src_breakdown[key] = src_breakdown.get(key, 0) + 1

    print(f"  [{split}]  {len(pairs):,} fuzzy near-duplicate pair(s) found at >= {FUZZY_THRESHOLD}% similarity")
    for (s1, s2), cnt in sorted(src_breakdown.items()):
        print(f"    {s1} × {s2}: {cnt} pairs")
    print_pairs(pairs, cap=200)


# ---- B2. Cross-split fuzzy pairs ----
subheader("B2. Cross-Split Fuzzy Near-Duplicates (potential leakage)")

cross_pairs_map = {}
cross_split_pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val",   "test"),
]
for split_a, split_b in cross_split_pairs:
    print(f"\n  [{split_a} × {split_b}]  Scanning ...")
    sys.stdout.flush()
    pairs = find_fuzzy_pairs_cross(dfs[split_a], split_a, dfs[split_b], split_b)
    cross_pairs_map[(split_a, split_b)] = pairs
    print(f"  [{split_a} × {split_b}]  {len(pairs):,} fuzzy near-duplicate pair(s) found")
    print_pairs(pairs, cap=200)


# ---- B3. Fuzzy summary ----
subheader("B3. Fuzzy Analysis Summary")

print(f"\n  Within-split results:")
print(f"  {'Split':6}  {'Rows':>8}  {'Fuzzy pairs':>12}  {'Unique vids affected (est.)':>28}")
print(f"  {'─'*6}  {'─'*8}  {'─'*12}  {'─'*28}")
for split, pairs in within_results.items():
    affected_vids = len(set([p["vid_a"] for p in pairs] + [p["vid_b"] for p in pairs]))
    print(f"  {split:6}  {len(dfs[split]):>8,}  {len(pairs):>12,}  {affected_vids:>28,}")

print(f"\n  Cross-split results (potential leakage):")
print(f"  {'Pair':16}  {'Fuzzy pairs':>12}")
print(f"  {'─'*16}  {'─'*12}")
for (a, b), pairs in cross_pairs_map.items():
    print(f"  {a+' × '+b:16}  {len(pairs):>12,}")


# ---------------------------------------------------------------------------
# Final footer
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  Analysis complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Report saved to : {LOG_PATH}")
print(f"{SEP}\n")

sys.stdout = sys.__stdout__
_tee.close()
