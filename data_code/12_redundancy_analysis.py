#!/usr/bin/env python3
"""
12_redundancy_analysis.py
=========================
Comprehensive redundancy analysis for data/2_dataset_{train,val,test}.tsv.
Run: python data_code/12_redundancy_analysis.py
"""

import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


class _Tee:
    """Write to both stdout and a file simultaneously."""
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

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False

# ---------------------------------------------------------------------------
# Paths & output file
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS = ["train", "val", "test"]

_LOG_PATH = Path(__file__).resolve().parent / "redundancy_report.txt"
_tee = _Tee(_LOG_PATH)
sys.stdout = _tee

_RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Redundancy Analysis Report — generated {_RUN_TIMESTAMP}")
print("=" * 72)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEP = "=" * 72
SUB = "-" * 72

def header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def subheader(title: str):
    print(f"\n{SUB}")
    print(f"  {title}")
    print(SUB)

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def similarity_ratio(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
header("1. LOADING DATA")

dfs = {}
for split in SPLITS:
    path = DATA_DIR / f"2_dataset_{split}.tsv"
    if not path.exists():
        print(f"  [MISSING] {path}")
        sys.exit(1)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce")
    dfs[split] = df
    print(f"  {split:5s}: {len(df):>7,} rows | columns: {list(df.columns)}")

total_rows = sum(len(df) for df in dfs.values())
print(f"\n  Total rows across all splits: {total_rows:,}")

# Null / empty check
subheader("Null / Empty Values")
for split, df in dfs.items():
    nulls = df.isnull().sum()
    empty = (df == "").sum()
    issues = nulls + empty
    if issues.sum() > 0:
        print(f"  [{split}]")
        for col in df.columns:
            if issues[col] > 0:
                print(f"    {col}: {nulls[col]} null, {empty[col]} empty-string")
    else:
        print(f"  [{split}] No nulls or empty strings.")

# ---------------------------------------------------------------------------
# 2. Exact duplicate vid IDs
# ---------------------------------------------------------------------------
header("2. EXACT DUPLICATE vid IDs")

subheader("Within-split duplicates")
for split, df in dfs.items():
    dup_vids = df[df.duplicated("vid", keep=False)]["vid"]
    unique_dup = dup_vids.nunique()
    total_dup_rows = len(dup_vids)
    print(f"  [{split}] {unique_dup} distinct vid(s) duplicated → {total_dup_rows} rows affected")
    if unique_dup > 0:
        for vid, cnt in Counter(dup_vids).most_common(5):
            print(f"    '{vid}' appears {cnt}x")

subheader("Cross-split vid overlap (data leakage)")
all_vids = {split: set(df["vid"]) for split, df in dfs.items()}
pairs = [("train", "val"), ("train", "test"), ("val", "test")]
for a, b in pairs:
    overlap = all_vids[a] & all_vids[b]
    print(f"  {a} ∩ {b}: {len(overlap)} shared vid(s)")
    for v in list(overlap)[:5]:
        print(f"    '{v}'")

# ---------------------------------------------------------------------------
# 3. Exact duplicate file_path
# ---------------------------------------------------------------------------
header("3. EXACT DUPLICATE file_path")

subheader("Within-split duplicates")
for split, df in dfs.items():
    dup = df[df.duplicated("file_path", keep=False)]
    unique_dup = dup["file_path"].nunique()
    print(f"  [{split}] {unique_dup} distinct path(s) duplicated → {len(dup)} rows affected")
    if unique_dup > 0:
        for fp, cnt in Counter(dup["file_path"]).most_common(5):
            print(f"    (×{cnt}) {fp}")

subheader("Cross-split file_path overlap")
all_paths = {split: set(df["file_path"].dropna()) for split, df in dfs.items()}
for a, b in pairs:
    overlap = all_paths[a] & all_paths[b]
    print(f"  {a} ∩ {b}: {len(overlap)} shared path(s)")

# ---------------------------------------------------------------------------
# 4. Exact duplicate text
# ---------------------------------------------------------------------------
header("4. EXACT DUPLICATE text")

subheader("Within-split duplicates")
within_text_dup = {}
for split, df in dfs.items():
    dup = df[df.duplicated("text", keep=False)]
    unique_dup = dup["text"].nunique()
    within_text_dup[split] = unique_dup
    pct = 100 * len(dup) / len(df)
    print(f"  [{split}] {unique_dup} distinct text(s) duplicated → {len(dup)} rows ({pct:.1f}%)")

subheader("Top 20 most-repeated texts (across all splits combined)")
all_texts = pd.concat([df[["text", "source", "vid"]] for df in dfs.values()], ignore_index=True)
text_counts = Counter(all_texts["text"])
print(f"  {'Count':>6}  {'Sources':30}  Text (first 80 chars)")
for text, cnt in text_counts.most_common(20):
    if cnt < 2:
        break
    src_rows = all_texts[all_texts["text"] == text]
    sources = ", ".join(sorted(src_rows["source"].unique()))
    print(f"  {cnt:>6}  {sources:30}  {str(text)[:80]!r}")

subheader("Cross-split exact text overlap (leakage)")
all_text_sets = {split: set(df["text"].dropna()) for split, df in dfs.items()}
for a, b in pairs:
    overlap = all_text_sets[a] & all_text_sets[b]
    print(f"  {a} ∩ {b}: {len(overlap)} shared exact text(s)")
    for t in list(overlap)[:3]:
        print(f"    {str(t)[:90]!r}")

# ---------------------------------------------------------------------------
# 5. Near-duplicate text (normalised + fuzzy)
# ---------------------------------------------------------------------------
header("5. NEAR-DUPLICATE TEXT")

subheader("Normalised text duplicates (case/punctuation collapsed)")
norm_cross = {}
for split, df in dfs.items():
    df["_norm"] = df["text"].apply(normalise)
    dup = df[df.duplicated("_norm", keep=False)]
    unique_dup = dup["_norm"].nunique()
    pct = 100 * len(dup) / len(df)
    norm_cross[split] = set(df["_norm"].dropna())
    print(f"  [{split}] {unique_dup} distinct normalised text(s) duplicated → {len(dup)} rows ({pct:.1f}%)")

subheader("Cross-split normalised text overlap")
for a, b in pairs:
    overlap = norm_cross[a] & norm_cross[b]
    print(f"  {a} ∩ {b}: {len(overlap)} shared normalised text(s)")
    for t in list(overlap)[:3]:
        print(f"    {t[:90]!r}")

subheader(f"Fuzzy similarity clustering (≥90% ratio) — using {'rapidfuzz' if HAS_RAPIDFUZZ else 'difflib'}")
SAMPLE_CAP = 500
THRESHOLD = 0.90

for split, df in dfs.items():
    texts = df["_norm"].dropna().unique().tolist()
    sample = texts[:SAMPLE_CAP]
    n = len(sample)
    cluster_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_ratio(sample[i], sample[j]) >= THRESHOLD:
                cluster_pairs += 1
    note = f"(sampled {n} of {len(texts)} unique normalised texts)" if len(texts) > SAMPLE_CAP else f"({n} unique normalised texts)"
    print(f"  [{split}] {cluster_pairs} near-duplicate pairs at ≥90% similarity {note}")

# ---------------------------------------------------------------------------
# 6. Short / generic text redundancy
# ---------------------------------------------------------------------------
header("6. SHORT / GENERIC TEXT (word_count ≤ 3)")

for split, df in dfs.items():
    short = df[df["word_count"] <= 3]
    pct = 100 * len(short) / len(df)
    print(f"  [{split}] {len(short):,} rows ({pct:.1f}%) have word_count ≤ 3")
    if len(short) > 0:
        for text, cnt in Counter(short["text"]).most_common(5):
            print(f"    (×{cnt}) {str(text)[:80]!r}")

subheader("Top 50 most-repeated texts (combined, ≥2 occurrences)")
top50 = [(t, c) for t, c in text_counts.most_common(50) if c >= 2]
if top50:
    print(f"  {'Rank':>4}  {'Count':>6}  Text (first 80 chars)")
    for rank, (text, cnt) in enumerate(top50, 1):
        print(f"  {rank:>4}  {cnt:>6}  {str(text)[:80]!r}")
else:
    print("  No texts repeated ≥2 times.")

# ---------------------------------------------------------------------------
# 7. Source overlap (same text in both sources within a split)
# ---------------------------------------------------------------------------
header("7. SAME TEXT IN BOTH SOURCES (how2sign vs openasl)")

for split, df in dfs.items():
    by_source = df.groupby("source")["text"].apply(set).to_dict()
    sources = list(by_source.keys())
    if len(sources) < 2:
        print(f"  [{split}] Only one source present: {sources}")
        continue
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            s1, s2 = sources[i], sources[j]
            overlap = by_source[s1] & by_source[s2]
            print(f"  [{split}] '{s1}' ∩ '{s2}': {len(overlap)} shared text(s)")
            for t in list(overlap)[:3]:
                print(f"    {str(t)[:80]!r}")

# ---------------------------------------------------------------------------
# 8. Duration + word_count + source fingerprint duplicates
# ---------------------------------------------------------------------------
header("8. WEAK FINGERPRINT DUPLICATES (duration_sec, word_count, source)")

for split, df in dfs.items():
    fp = df[["duration_sec", "word_count", "source"]].dropna()
    dup = fp[fp.duplicated(keep=False)]
    unique_dup = dup.drop_duplicates().shape[0]
    pct = 100 * len(dup) / len(df)
    print(f"  [{split}] {unique_dup} distinct fingerprints duplicated → {len(dup)} rows ({pct:.1f}%)")
    top5 = (
        df.dropna(subset=["duration_sec", "word_count", "source"])
        .groupby(["duration_sec", "word_count", "source"])
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    for (dur, wc, src), cnt in top5.items():
        if cnt > 1:
            print(f"    dur={dur:.2f}s  wc={int(wc)}  src={src}  → {cnt} rows")

# ---------------------------------------------------------------------------
# 9. Cross-split leakage summary table
# ---------------------------------------------------------------------------
header("9. CROSS-SPLIT LEAKAGE SUMMARY TABLE")

col_w = 12
print(f"\n  {'Pair':16} {'Exact vid':>10} {'Exact text':>11} {'Norm text':>10}")
print(f"  {'-'*16} {'-'*10} {'-'*11} {'-'*10}")
for a, b in pairs:
    vid_overlap  = len(all_vids[a]      & all_vids[b])
    text_overlap = len(all_text_sets[a] & all_text_sets[b])
    norm_overlap = len(norm_cross[a]    & norm_cross[b])
    print(f"  {a+' ∩ '+b:16} {vid_overlap:>10,} {text_overlap:>11,} {norm_overlap:>10,}")

# ---------------------------------------------------------------------------
# 10. Overall redundancy score
# ---------------------------------------------------------------------------
header("10. OVERALL REDUNDANCY SCORE")

print(f"\n  {'Split':6} {'Rows':>8} {'Dup vid':>8} {'Dup text':>9} {'Norm dup':>9} {'Est. % redund.':>15}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*15}")

for split, df in dfs.items():
    total = len(df)
    dup_vid_rows  = df.duplicated("vid",   keep="first").sum()
    dup_text_rows = df.duplicated("text",  keep="first").sum()
    dup_norm_rows = df.duplicated("_norm", keep="first").sum()
    # Conservative estimate: max of the three (overlapping categories)
    est_redund = max(dup_vid_rows, dup_text_rows, dup_norm_rows)
    pct = 100 * est_redund / total
    print(f"  {split:6} {total:>8,} {dup_vid_rows:>8,} {dup_text_rows:>9,} {dup_norm_rows:>9,} {pct:>14.1f}%")

print(f"\n  Note: 'Est. % redund.' = max(dup_vid, dup_text, norm_dup) / total rows.")
print(f"  Fuzzy duplicates (Section 5) may reveal additional near-duplicates.")
print(f"  Using {'rapidfuzz' if HAS_RAPIDFUZZ else 'difflib (install rapidfuzz for speed)'}.")

print(f"\n{'=' * 72}")
print(f"  Report complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Saved to: {_LOG_PATH}")
print(f"{'=' * 72}\n")

sys.stdout = sys.__stdout__
_tee.close()
