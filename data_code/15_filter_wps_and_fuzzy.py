#!/usr/bin/env python3
"""
15_filter_wps_and_fuzzy.py
==========================
Applies three filtering passes to data/3_dataset_{train,val,test}.tsv and
writes clean files to data/4_dataset_{split}.tsv.

  Step 1 — WPS filter
    Remove every row where word_count / duration_sec > 5.0.

  Step 2 — Within-split fuzzy near-duplicate removal
    Find pairs of clips whose normalised text is >= 90% similar (blocking by
    word_count ±1). Cluster connected pairs with Union-Find and keep only the
    first row (lowest original index) per cluster.

  Step 3 — Cross-split fuzzy leakage removal
    Find fuzzy near-duplicates across splits and remove the row from the
    lower-priority split (train > val > test).

Performance notes:
  • Install rapidfuzz for a 20-50x speedup:  pip install rapidfuzz
    Without it the script falls back to difflib with capped block sizes.
  • With rapidfuzz   : each cdist block uses all CPU cores (workers=-1).
  • Without rapidfuzz: blocks are distributed across CPU cores with
    ProcessPoolExecutor so multiple cores run in parallel.

Run : python data_code/15_filter_wps_and_fuzzy.py
"""

import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Constants (module-level so worker processes can access them)
# ---------------------------------------------------------------------------
WPS_THRESHOLD   = 5.0
FUZZY_THRESHOLD = 90      # rapidfuzz ratio (0–100)
NO_RF_BLOCK_CAP = 300     # per-group cap when rapidfuzz is not available

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CODE_DIR = Path(__file__).resolve().parent


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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
SEP = "=" * 72
SUB = "-" * 72
def header(t):    print(f"\n{SEP}\n  {t}\n{SEP}")
def subheader(t): print(f"\n{SUB}\n  {t}\n{SUB}")


# ---------------------------------------------------------------------------
# Worker function — MUST be at module level for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------
def _fuzzy_block_worker(args):
    """
    Process one (wc_a, wc_b) block and return a list of matching pair dicts.
    Runs in a subprocess when HAS_RAPIDFUZZ is False.
    """
    entries_a, entries_b, wc_a, wc_b, threshold, no_rf_cap = args

    if not HAS_RAPIDFUZZ:
        entries_a = entries_a[:no_rf_cap]
        entries_b = entries_b[:no_rf_cap]

    texts_a = [e[1] for e in entries_a]
    texts_b = [e[1] for e in entries_b]

    if HAS_RAPIDFUZZ:
        # workers=1: we are already inside a parallel context
        scores = np.asarray(
            rf_process.cdist(texts_a, texts_b, scorer=fuzz.ratio,
                             score_cutoff=threshold, workers=1)
        )
    else:
        scores = np.array(
            [[SequenceMatcher(None, texts_a[i], texts_b[j]).ratio() * 100
              for j in range(len(texts_b))]
             for i in range(len(texts_a))],
            dtype=np.float32,
        )

    # Ensure entries below threshold are zero (difflib path doesn't do this automatically)
    scores[scores < threshold] = 0

    if wc_a == wc_b:
        rs, cs = np.nonzero(np.triu(scores, k=1))
    else:
        rs, cs = np.nonzero(scores)

    pairs = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        pairs.append({
            "score":  float(scores[r, c]),
            "idx_a":  entries_a[r][0], "vid_a": entries_a[r][2], "text_a": entries_a[r][3],
            "idx_b":  entries_b[c][0], "vid_b": entries_b[c][2], "text_b": entries_b[c][3],
            "wc_a":   wc_a,            "wc_b":  wc_b,
        })
    return pairs


# ---------------------------------------------------------------------------
# Fuzzy helpers
# ---------------------------------------------------------------------------
def _build_wc_index(df):
    idx = defaultdict(list)
    for i, row in df.iterrows():
        idx[int(row["_wc_int"])].append((i, row["_norm"], row["vid"], row["text"]))
    return idx


def _collect_blocks(wc_idx_a, wc_idx_b=None):
    """
    Yield (entries_a, entries_b, wc_a, wc_b) for every block to compare.
    If wc_idx_b is None, within-split mode (compare wc_idx_a against itself).
    """
    within = wc_idx_b is None
    idx_b  = wc_idx_a if within else wc_idx_b
    seen   = set()

    for wc in sorted(wc_idx_a.keys()):
        for offset in [0, 1]:
            wc_b = wc + offset
            ga = wc_idx_a.get(wc,   [])
            gb = idx_b.get(wc_b, [])
            if not ga or not gb:
                continue
            if within and wc == wc_b and len(ga) < 2:
                continue
            # For within-mode, skip the reverse duplicate (wc=5 vs wc=6 handled once)
            if within and offset == 0:
                key = (wc, wc)
                if key in seen:
                    continue
                seen.add(key)
            yield ga, gb, wc, wc_b


def _run_blocks(blocks, n_workers):
    """
    Dispatch blocks to worker(s) and collect results.
    With rapidfuzz  : run sequentially (cdist already uses all cores with workers=-1).
    Without rapidfuzz: run in parallel with ProcessPoolExecutor.
    """
    all_pairs = []
    block_list = list(blocks)

    if HAS_RAPIDFUZZ:
        # Sequential — each cdist call uses all cores internally
        for ga, gb, wc_a, wc_b in block_list:
            label = f"wc={wc_a}" + (f"×{wc_b}" if wc_a != wc_b else "(self)")
            print(f"    {label}  ({len(ga)}×{len(gb)}) ...", flush=True)
            args = (ga, gb, wc_a, wc_b, FUZZY_THRESHOLD, NO_RF_BLOCK_CAP)
            all_pairs.extend(_fuzzy_block_worker(args))
    else:
        # Parallel — distribute blocks across CPU cores
        args_list = [
            (ga, gb, wc_a, wc_b, FUZZY_THRESHOLD, NO_RF_BLOCK_CAP)
            for ga, gb, wc_a, wc_b in block_list
        ]
        print(f"    Dispatching {len(args_list)} blocks to {n_workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_fuzzy_block_worker, a): a for a in args_list}
            done = 0
            for future in as_completed(futures):
                all_pairs.extend(future.result())
                done += 1
                if done % 20 == 0 or done == len(args_list):
                    print(f"    {done}/{len(args_list)} blocks complete ...", flush=True)

    return all_pairs


def find_fuzzy_pairs_within(df, n_workers):
    wc_idx = _build_wc_index(df)
    blocks  = _collect_blocks(wc_idx)
    raw     = _run_blocks(blocks, n_workers)
    # Deduplicate pairs by (idx_a, idx_b)
    seen, pairs = set(), []
    for p in raw:
        key = (min(p["idx_a"], p["idx_b"]), max(p["idx_a"], p["idx_b"]))
        if key not in seen:
            seen.add(key)
            pairs.append(p)
    return sorted(pairs, key=lambda x: -x["score"])


def find_fuzzy_pairs_cross(df_a, df_b, n_workers):
    wc_idx_a = _build_wc_index(df_a)
    wc_idx_b = _build_wc_index(df_b)
    blocks   = _collect_blocks(wc_idx_a, wc_idx_b)
    raw      = _run_blocks(blocks, n_workers)
    # Deduplicate by (vid_a, vid_b)
    seen, pairs = set(), []
    for p in raw:
        key = (p["vid_a"], p["vid_b"])
        if key not in seen:
            seen.add(key)
            pairs.append(p)
    return sorted(pairs, key=lambda x: -x["score"])


def union_find_clusters(pairs):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        parent[find(x)] = find(y)
    for p in pairs:
        union(p["vid_a"], p["vid_b"])
    clusters = defaultdict(set)
    for v in list(parent):
        clusters[find(v)].add(v)
    return clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n_workers = max(1, (os.cpu_count() or 4) - 1)  # leave one core free
    LOG_PATH  = CODE_DIR / (Path(__file__).stem + ".txt")
    _tee      = _Tee(LOG_PATH)
    sys.stdout = _tee

    RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SPLITS = ["train", "val", "test"]

    print(f"WPS & Fuzzy Filtering Report — generated {RUN_TS}")
    print(SEP)
    print(f"  Input       : data/3_dataset_{{train,val,test}}.tsv")
    print(f"  Output      : data/4_dataset_{{train,val,test}}.tsv")
    print(f"  Log         : {LOG_PATH}")
    print(f"  WPS cutoff  : remove rows where WPS > {WPS_THRESHOLD}")
    print(f"  Fuzzy cutoff: >= {FUZZY_THRESHOLD}% similarity  (word_count ±1 blocking)")
    if HAS_RAPIDFUZZ:
        print(f"  Fuzzy engine: rapidfuzz  (cdist, workers=-1 per block, sequential blocks)")
    else:
        print(f"  Fuzzy engine: difflib  (block cap={NO_RF_BLOCK_CAP}, {n_workers} parallel workers)")
        print(f"  *** Install rapidfuzz for 20-50x speedup:  pip install rapidfuzz ***")
    print(f"  CPU cores   : {os.cpu_count()}  (using {n_workers} worker(s))")
    print(SEP)

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    header("0. LOADING INPUT FILES")

    dfs = {}
    orig_counts = {}
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
        df["_norm"]        = df["text"].apply(
            lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()
        )
        df["_wc_int"]   = df["word_count"].fillna(0).astype(int)
        df["_orig_idx"] = np.arange(len(df))
        dfs[split]         = df
        orig_counts[split] = len(df)
        print(f"  {split:5s}: {len(df):>7,} rows  —  3_dataset_{split}.tsv")

    print(f"\n  Total input rows: {sum(orig_counts.values()):,}")

    # -----------------------------------------------------------------------
    # Step 1 — WPS filter
    # -----------------------------------------------------------------------
    header(f"STEP 1 — WPS FILTER  (remove WPS > {WPS_THRESHOLD})")

    step1_removed = {}
    for split in SPLITS:
        df        = dfs[split]
        mask_high = df["_wps"] > WPS_THRESHOLD
        removed   = df[mask_high].copy()
        kept      = df[~mask_high].copy()
        step1_removed[split] = removed
        pct = 100 * len(removed) / len(df)
        print(f"\n  [{split}]  {len(df):,} → {len(kept):,}  (removed {len(removed):,} rows, {pct:.2f}%)")

        if len(removed) > 0:
            for src, grp in removed.groupby("source"):
                print(f"    {src}: {len(grp):,} rows removed")
            print()
            CAP = 10
            sorted_removed = removed.sort_values("_wps", ascending=False)
            print(f"  {'#':>4}  {'WPS':>7}  {'Dur':>6}  {'WC':>4}  {'Source':10}  vid")
            print(f"  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*4}  {'─'*10}  {'─'*60}")
            for rank, (_, row) in enumerate(sorted_removed.head(CAP).iterrows(), 1):
                print(
                    f"  {rank:>4}  {row['_wps']:>7.3f}  {row['duration_sec']:>6.2f}  "
                    f"{int(row['word_count']) if pd.notna(row['word_count']) else '?':>4}  "
                    f"{str(row['source'])[:10]:10}  "
                    f"{str(row['vid'])}"
                )
            if len(removed) > CAP:
                print(f"  ... and {len(removed) - CAP} more rows (see full list in log or re-run with higher cap)")
        dfs[split] = kept.reset_index(drop=True)

    subheader("Step 1 Summary")
    print(f"  {'Split':6}  {'Before':>8}  {'Removed':>8}  {'After':>8}  {'% removed':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
    for split in SPLITS:
        b, r, a = orig_counts[split], len(step1_removed[split]), len(dfs[split])
        print(f"  {split:6}  {b:>8,}  {r:>8,}  {a:>8,}  {100*r/b:>9.2f}%")

    # -----------------------------------------------------------------------
    # Step 2 — Within-split fuzzy near-duplicate removal
    # -----------------------------------------------------------------------
    header(f"STEP 2 — WITHIN-SPLIT FUZZY NEAR-DUPLICATE REMOVAL  (>= {FUZZY_THRESHOLD}%)")
    print("  Keeps the first row (lowest original index) per duplicate cluster.\n")

    step2_removed = {}
    for split in SPLITS:
        df = dfs[split]
        subheader(f"Split: {split}  ({len(df):,} rows)")
        t0 = time.time()
        pairs   = find_fuzzy_pairs_within(df, n_workers)
        elapsed = time.time() - t0
        print(f"\n  Scan complete in {elapsed:.1f}s  —  {len(pairs)} pair(s) found")

        if not pairs:
            print(f"  No fuzzy near-duplicates found. Nothing removed.")
            step2_removed[split] = df.iloc[0:0].copy()
            continue

        # Score distribution of pairs
        scores = [p["score"] for p in pairs]
        exact  = sum(1 for s in scores if s == 100.0)
        hi     = sum(1 for s in scores if 95.0 <= s < 100.0)
        mid    = sum(1 for s in scores if 90.0 <= s < 95.0)
        print(f"  Similarity breakdown:  100% = {exact:,}  |  95–99% = {hi:,}  |  90–94% = {mid:,}")

        # Top 10 sample only
        SAMPLE = 10
        print(f"\n  Top {min(SAMPLE, len(pairs))} highest-similarity pairs (of {len(pairs):,} total):")
        print(f"  {'#':>4}  {'Sim%':>5}  {'WC_a':>4}  {'WC_b':>4}  {'vid_a':42}  vid_b")
        print(f"  {'─'*4}  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*42}  {'─'*42}")
        for rank, p in enumerate(pairs[:SAMPLE], 1):
            print(f"  {rank:>4}  {p['score']:>5.1f}  {p['wc_a']:>4}  {p['wc_b']:>4}  "
                  f"{str(p['vid_a'])[:42]:42}  {str(p['vid_b'])[:42]}")

        clusters       = union_find_clusters(pairs)
        vids_to_remove = set()
        for members in clusters.values():
            cluster_rows = df[df["vid"].isin(members)].sort_values("_orig_idx")
            keep_vid     = cluster_rows.iloc[0]["vid"]
            remove_vids  = set(cluster_rows["vid"]) - {keep_vid}
            vids_to_remove.update(remove_vids)
        print(f"\n  Clusters formed: {len(clusters):,}  |  vids to remove: {len(vids_to_remove):,}")

        removed_df           = df[df["vid"].isin(vids_to_remove)].copy()
        kept_df              = df[~df["vid"].isin(vids_to_remove)].copy()
        step2_removed[split] = removed_df
        pct = 100 * len(removed_df) / len(df)
        print(f"\n  [{split}]  Removed {len(removed_df):,} rows ({pct:.2f}%)  →  {len(kept_df):,} remain")
        dfs[split] = kept_df.reset_index(drop=True)

    subheader("Step 2 Summary")
    print(f"  {'Split':6}  {'Before':>8}  {'Removed':>8}  {'After':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
    for split in SPLITS:
        b = len(dfs[split]) + len(step2_removed[split])
        r = len(step2_removed[split])
        print(f"  {split:6}  {b:>8,}  {r:>8,}  {len(dfs[split]):>8,}")

    # -----------------------------------------------------------------------
    # Step 3 — Cross-split fuzzy leakage removal
    # -----------------------------------------------------------------------
    header(f"STEP 3 — CROSS-SPLIT FUZZY LEAKAGE REMOVAL  (>= {FUZZY_THRESHOLD}%)")
    print("  train × val  → remove from val")
    print("  train × test → remove from test")
    print("  val   × test → remove from test\n")

    cross_removed   = {split: set() for split in SPLITS}
    cross_pairs_cfg = [
        ("train", "val",  "val"),
        ("train", "test", "test"),
        ("val",   "test", "test"),
    ]
    all_cross_pairs = {}

    for split_a, split_b, remove_from in cross_pairs_cfg:
        subheader(f"{split_a} × {split_b}  (remove from '{remove_from}')")
        t0    = time.time()
        pairs = find_fuzzy_pairs_cross(dfs[split_a], dfs[split_b], n_workers)
        elapsed = time.time() - t0
        all_cross_pairs[(split_a, split_b)] = pairs
        print(f"\n  Scan complete in {elapsed:.1f}s  —  {len(pairs)} pair(s) found")

        if not pairs:
            print(f"  No cross-split fuzzy pairs found.")
            continue

        scores = [p["score"] for p in pairs]
        exact  = sum(1 for s in scores if s == 100.0)
        hi     = sum(1 for s in scores if 95.0 <= s < 100.0)
        mid    = sum(1 for s in scores if 90.0 <= s < 95.0)
        print(f"  Similarity breakdown:  100% = {exact:,}  |  95–99% = {hi:,}  |  90–94% = {mid:,}")

        # Collect vids to remove and show top 10 sample
        SAMPLE = 10
        print(f"\n  Top {min(SAMPLE, len(pairs))} highest-similarity pairs (of {len(pairs):,} total):")
        print(f"  {'#':>4}  {'Sim%':>5}  {'WC_a':>4}  {'WC_b':>4}  {'keep vid':42}  remove vid")
        print(f"  {'─'*4}  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*42}  {'─'*42}")
        for rank, p in enumerate(pairs, 1):
            rem_vid  = p["vid_b"] if remove_from == split_b else p["vid_a"]
            keep_vid = p["vid_a"] if remove_from == split_b else p["vid_b"]
            cross_removed[remove_from].add(rem_vid)
            if rank <= SAMPLE:
                print(f"  {rank:>4}  {p['score']:>5.1f}  {p['wc_a']:>4}  {p['wc_b']:>4}  "
                      f"{str(keep_vid)[:42]:42}  {str(rem_vid)[:42]}")
        if len(pairs) > SAMPLE:
            print(f"  ... {len(pairs) - SAMPLE:,} more pairs (all processed, only top {SAMPLE} shown)")

    for split in SPLITS:
        if cross_removed[split]:
            df         = dfs[split]
            removed_df = df[df["vid"].isin(cross_removed[split])].copy()
            kept_df    = df[~df["vid"].isin(cross_removed[split])].copy()
            pct = 100 * len(removed_df) / len(df)
            print(f"\n  [{split}]  Removed {len(removed_df):,} leaking rows ({pct:.2f}%)  →  {len(kept_df):,} remain")
            dfs[split] = kept_df.reset_index(drop=True)

    subheader("Step 3 Summary")
    print(f"  {'Pair':16}  {'Pairs':>6}  {'Remove from':>12}  {'# removed':>10}")
    print(f"  {'─'*16}  {'─'*6}  {'─'*12}  {'─'*10}")
    for split_a, split_b, remove_from in cross_pairs_cfg:
        n = len(all_cross_pairs.get((split_a, split_b), []))
        print(f"  {split_a+' × '+split_b:16}  {n:>6,}  {remove_from:>12}  {n:>10,}")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    header("WRITING OUTPUT FILES")

    out_cols = ["vid", "text", "duration_sec", "word_count", "file_path", "source"]
    for split in SPLITS:
        out_path = DATA_DIR / f"4_dataset_{split}.tsv"
        dfs[split][out_cols].to_csv(out_path, sep="\t", index=False)
        print(f"  Wrote {split:5s}: {len(dfs[split]):>7,} rows  →  {out_path.name}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    header("FINAL SUMMARY — ALL STEPS COMBINED")

    print(f"\n  {'Split':6}  {'Original':>9}  {'Step1 WPS':>10}  {'Step2 Fuzz':>11}  {'Step3 Leak':>11}  {'Final':>8}  {'Total lost':>11}  {'% lost':>7}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*10}  {'─'*11}  {'─'*11}  {'─'*8}  {'─'*11}  {'─'*7}")
    grand_orig = grand_final = grand_lost = 0
    for split in SPLITS:
        orig  = orig_counts[split]
        s1    = len(step1_removed[split])
        s2    = len(step2_removed[split])
        s3    = len(cross_removed[split])
        final = len(dfs[split])
        lost  = s1 + s2 + s3
        pct   = 100 * lost / orig
        grand_orig += orig; grand_final += final; grand_lost += lost
        print(f"  {split:6}  {orig:>9,}  {s1:>10,}  {s2:>11,}  {s3:>11,}  {final:>8,}  {lost:>11,}  {pct:>6.2f}%")
    gp = 100 * grand_lost / grand_orig
    print(f"  {'─'*6}  {'─'*9}  {'─'*10}  {'─'*11}  {'─'*11}  {'─'*8}  {'─'*11}  {'─'*7}")
    print(f"  {'TOTAL':6}  {grand_orig:>9,}  "
          f"{sum(len(step1_removed[s]) for s in SPLITS):>10,}  "
          f"{sum(len(step2_removed[s]) for s in SPLITS):>11,}  "
          f"{sum(len(cross_removed[s]) for s in SPLITS):>11,}  "
          f"{grand_final:>8,}  {grand_lost:>11,}  {gp:>6.2f}%")

    print(f"\n  Output files:")
    for split in SPLITS:
        print(f"    4_dataset_{split}.tsv  ({len(dfs[split]):,} rows)")

    print(f"\n{SEP}")
    print(f"  Filtering complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Report saved to: {LOG_PATH}")
    print(f"{SEP}\n")

    sys.stdout = sys.__stdout__
    _tee.close()


# ---------------------------------------------------------------------------
# Entry point — required guard for ProcessPoolExecutor on Windows
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
