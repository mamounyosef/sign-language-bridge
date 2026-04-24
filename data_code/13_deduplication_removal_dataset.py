#!/usr/bin/env python3
"""
13_remove_deduplicate_dataset.py
================================
Removes intra-split text duplicates and cross-split text leakage from
data/2_dataset_{train,val,test}.tsv and writes clean files to
data/3_dataset_{split}_deduped.tsv.

Two-step process:
  Step 1 — Per-split deduplication: keep only the first occurrence of each
            unique text string within each split. All subsequent rows with
            the same text are removed.
  Step 2 — Cross-split leakage removal: remove from val and test any row
            whose text already appears in the (post-step-1) train split.

Run: python data_code/13_remove_deduplicate_dataset.py
"""

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Tee: mirror stdout to a log file
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
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
CODE_DIR  = Path(__file__).resolve().parent
SPLITS    = ["train", "val", "test"]
IN_TMPL   = "2_dataset_{split}.tsv"
OUT_TMPL  = "3_dataset_{split}_deduped.tsv"

LOG_PATH  = CODE_DIR / "deduplication_removal_report.txt"
_tee      = _Tee(LOG_PATH)
sys.stdout = _tee

RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
SEP = "=" * 72
SUB = "-" * 72

def header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subheader(title):
    print(f"\n{SUB}\n  {title}\n{SUB}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
print(f"Deduplication Report — generated {RUN_TS}")
print(SEP)
print("  Script : 13_remove_deduplicate_dataset.py")
print(f"  Input  : data/2_dataset_{{train,val,test}}.tsv")
print(f"  Output : data/3_dataset_{{train,val,test}}_deduped.tsv")
print(f"  Log    : {LOG_PATH}")
print(SEP)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
header("0. LOADING INPUT FILES")

dfs = {}
for split in SPLITS:
    path = DATA_DIR / IN_TMPL.format(split=split)
    if not path.exists():
        print(f"  [ERROR] Missing: {path}")
        sys.stdout = sys.__stdout__
        _tee.close()
        sys.exit(1)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    dfs[split] = df
    print(f"  Loaded {split:5s}: {len(df):>7,} rows  →  {path.name}")

print(f"\n  Total input rows: {sum(len(df) for df in dfs.values()):,}")


# ---------------------------------------------------------------------------
# Step 1: Intra-split deduplication by exact text
# ---------------------------------------------------------------------------
header("STEP 1 — INTRA-SPLIT TEXT DEDUPLICATION")
print("  Rule: within each split, keep only the FIRST occurrence of each")
print("        unique 'text' value. All subsequent rows with the same text")
print("        are removed, regardless of vid, source, or duration.\n")

step1_removed = {}  # split -> DataFrame of removed rows

for split in SPLITS:
    df = dfs[split]
    subheader(f"Split: {split}  ({len(df):,} rows before)")

    # Identify all rows that are NOT the first occurrence of their text
    is_dup = df.duplicated(subset="text", keep="first")
    kept_df    = df[~is_dup].copy()
    removed_df = df[is_dup].copy()
    step1_removed[split] = removed_df

    n_removed = len(removed_df)
    n_kept    = len(kept_df)
    n_unique_texts_removed = removed_df["text"].nunique()

    print(f"  Rows before  : {len(df):,}")
    print(f"  Rows removed : {n_removed:,}  ({100*n_removed/len(df):.2f}%)")
    print(f"  Rows kept    : {n_kept:,}")
    print(f"  Distinct texts that had duplicates: {n_unique_texts_removed:,}")

    if n_removed > 0:
        # Per-source breakdown of removed rows
        print(f"\n  Removed rows by source:")
        for src, grp in removed_df.groupby("source"):
            print(f"    {src}: {len(grp):,} rows removed")

        # Full list of every duplicated text and all removed instances
        print(f"\n  Full detail — every removed text group:")
        print(f"  {'#':>4}  {'Kept':>5}  {'Rmvd':>5}  {'Source(s)':18}  Text (first 100 chars)")
        print(f"  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*18}  {'─'*50}")

        # For display: group by text, sorted by removal count descending
        removal_counts = removed_df.groupby("text").size().sort_values(ascending=False)
        all_text_groups = df.groupby("text")

        for rank, (text, n_rmv) in enumerate(removal_counts.items(), 1):
            group_rows = all_text_groups.get_group(text)
            n_total    = len(group_rows)
            n_kept_grp = 1  # always keep first
            sources    = ", ".join(sorted(group_rows["source"].unique()))
            kept_vid   = group_rows.iloc[0]["vid"]
            removed_vids = group_rows.iloc[1:]["vid"].tolist()

            print(f"  {rank:>4}  {n_kept_grp:>5}  {n_rmv:>5}  {sources:18}  {str(text)[:100]!r}")
            print(f"        Kept vid   : {kept_vid}")
            if len(removed_vids) <= 6:
                for v in removed_vids:
                    print(f"        Removed vid: {v}")
            else:
                for v in removed_vids[:3]:
                    print(f"        Removed vid: {v}")
                print(f"        ... and {len(removed_vids)-3} more removed vids")

    # Update dfs[split] to deduplicated version
    dfs[split] = kept_df.reset_index(drop=True)

print(f"\n{SUB}")
print(f"  Step 1 Summary")
print(f"{SUB}")
print(f"  {'Split':6}  {'Before':>8}  {'Removed':>8}  {'Kept':>8}  {'% removed':>10}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
for split in SPLITS:
    before  = len(dfs[split]) + len(step1_removed[split])
    removed = len(step1_removed[split])
    kept    = len(dfs[split])
    pct     = 100 * removed / before if before > 0 else 0
    print(f"  {split:6}  {before:>8,}  {removed:>8,}  {kept:>8,}  {pct:>9.2f}%")


# ---------------------------------------------------------------------------
# Step 2: Cross-split leakage — remove from val/test texts present in train
# ---------------------------------------------------------------------------
header("STEP 2 — CROSS-SPLIT TEXT LEAKAGE REMOVAL")
print("  Rule: remove from val and test any row whose exact 'text' value")
print("        already exists in the post-step-1 train split.\n")

train_texts = set(dfs["train"]["text"])
print(f"  Unique texts in (deduplicated) train: {len(train_texts):,}")

step2_removed = {}

for split in ["val", "test"]:
    df = dfs[split]
    subheader(f"Split: {split}  ({len(df):,} rows before step 2)")

    leaked_mask = df["text"].isin(train_texts)
    leaked_df   = df[leaked_mask].copy()
    clean_df    = df[~leaked_mask].copy()
    step2_removed[split] = leaked_df

    n_removed = len(leaked_df)
    n_kept    = len(clean_df)

    print(f"  Rows before  : {len(df):,}")
    print(f"  Rows removed : {n_removed:,}  ({100*n_removed/len(df):.2f}%)")
    print(f"  Rows kept    : {n_kept:,}")

    if n_removed > 0:
        print(f"\n  Removed rows by source:")
        for src, grp in leaked_df.groupby("source"):
            print(f"    {src}: {len(grp):,} rows removed")

        print(f"\n  Full detail — every leaked row removed:")
        print(f"  {'#':>4}  {'vid':40}  {'src':10}  {'dur':>6}  {'wc':>4}  Text (first 80 chars)")
        print(f"  {'─'*4}  {'─'*40}  {'─'*10}  {'─'*6}  {'─'*4}  {'─'*50}")
        for rank, (_, row) in enumerate(leaked_df.iterrows(), 1):
            print(
                f"  {rank:>4}  {str(row['vid'])[:40]:40}  "
                f"{str(row['source'])[:10]:10}  "
                f"{str(row['duration_sec']):>6}  "
                f"{str(row['word_count']):>4}  "
                f"{str(row['text'])[:80]!r}"
            )

    dfs[split] = clean_df.reset_index(drop=True)

# Also report train (no leakage removal needed, but keep consistent)
step2_removed["train"] = pd.DataFrame(columns=dfs["train"].columns)

print(f"\n{SUB}")
print(f"  Step 2 Summary")
print(f"{SUB}")
print(f"  {'Split':6}  {'Before':>8}  {'Removed':>8}  {'Kept':>8}  {'% removed':>10}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
for split in SPLITS:
    before  = len(dfs[split]) + len(step2_removed[split])
    removed = len(step2_removed[split])
    kept    = len(dfs[split])
    pct     = 100 * removed / before if before > 0 else 0
    print(f"  {split:6}  {before:>8,}  {removed:>8,}  {kept:>8,}  {pct:>9.2f}%")


# ---------------------------------------------------------------------------
# Write output files
# ---------------------------------------------------------------------------
header("WRITING OUTPUT FILES")

for split in SPLITS:
    out_path = DATA_DIR / OUT_TMPL.format(split=split)
    dfs[split].to_csv(out_path, sep="\t", index=False)
    print(f"  Wrote {split:5s}: {len(dfs[split]):>7,} rows  →  {out_path.name}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
header("FINAL SUMMARY — BEFORE vs AFTER (both steps combined)")

orig_counts = {}
for split in SPLITS:
    path = DATA_DIR / IN_TMPL.format(split=split)
    orig_counts[split] = sum(1 for _ in open(path, encoding="utf-8")) - 1  # subtract header

print(f"\n  {'Split':6}  {'Original':>9}  {'Step1 -':>8}  {'Step2 -':>8}  {'Final':>8}  {'Total lost':>11}  {'% lost':>7}")
print(f"  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*11}  {'─'*7}")

grand_orig  = 0
grand_final = 0
grand_lost  = 0

for split in SPLITS:
    orig    = orig_counts[split]
    s1_lost = len(step1_removed[split])
    s2_lost = len(step2_removed[split])
    final   = len(dfs[split])
    lost    = s1_lost + s2_lost
    pct     = 100 * lost / orig if orig > 0 else 0
    grand_orig  += orig
    grand_final += final
    grand_lost  += lost
    print(f"  {split:6}  {orig:>9,}  {s1_lost:>8,}  {s2_lost:>8,}  {final:>8,}  {lost:>11,}  {pct:>6.2f}%")

grand_pct = 100 * grand_lost / grand_orig if grand_orig > 0 else 0
print(f"  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*11}  {'─'*7}")
print(f"  {'TOTAL':6}  {grand_orig:>9,}  {sum(len(step1_removed[s]) for s in SPLITS):>8,}  "
      f"{sum(len(step2_removed[s]) for s in SPLITS):>8,}  {grand_final:>8,}  "
      f"{grand_lost:>11,}  {grand_pct:>6.2f}%")

print(f"\n  Output files written to: {DATA_DIR}")
for split in SPLITS:
    print(f"    {OUT_TMPL.format(split=split)}")

print(f"\n{SEP}")
print(f"  Deduplication complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Report saved to: {LOG_PATH}")
print(f"{SEP}\n")

sys.stdout = sys.__stdout__
_tee.close()
