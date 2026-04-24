#!/usr/bin/env python3
"""
17_fix_text_normalization.py
============================
Applies text normalisation fixes to data/4_dataset_{train,val,test}.tsv
and writes clean files to data/5_dataset_{split}.tsv.

Fixes applied (in order):
  F1  Remove non-ASCII characters (codepoint > 127) from text; keep the row
  F2  Decode HTML entities  (&amp; &#39; etc.)
  F3  Convert curly/smart quotes to straight equivalents
  F4  Convert em dash / en dash to regular hyphen
  F5  Collapse repeated punctuation to one  (!! → ! ?? → ? .. → . -- → - etc.)
  F6  Sentence-case all-caps text: if text has > 18 uppercase letters,
      lowercase everything then capitalise after start / . / ! / ?
  F7  Remove ALL instances of any unpaired bracket or quote character
      (unmatched ( → remove all ( and )  |  odd " → remove all "  etc.)
  F8  Remove rows that contain no alphabetic content at all

  Final cleanup applied to every row regardless of other fixes:
      collapse multiple spaces → single space, strip leading/trailing whitespace

Run : python data_code/17_fix_text_normalization.py
"""

import html
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


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
# Paths & log
# ---------------------------------------------------------------------------
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

ALLCAPS_THRESHOLD = 18   # uppercase letter count above which sentence-case is applied


# ---------------------------------------------------------------------------
# Individual fix functions
# ---------------------------------------------------------------------------

def fix_non_ascii(text: str) -> str:
    """F1: Remove characters with codepoint > 127."""
    return "".join(ch for ch in text if ord(ch) <= 127)


def fix_html_entities(text: str) -> str:
    """F2: Decode HTML entities."""
    return html.unescape(text)


CURLY_MAP = str.maketrans({
    "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
    "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK
    "\u2032": "'",   # PRIME
    "\u2033": '"',   # DOUBLE PRIME
})

def fix_curly_quotes(text: str) -> str:
    """F3: Replace curly/smart quotes with straight equivalents."""
    return text.translate(CURLY_MAP)


def fix_dashes(text: str) -> str:
    """F4: Replace em dash and en dash with a regular hyphen."""
    return text.replace("\u2014", "-").replace("\u2013", "-")


REPEAT_PUNCT_RE = re.compile(r"([!?.\-,;:])\1+")

def fix_repeated_punctuation(text: str) -> str:
    """F5: Collapse runs of the same punctuation character to a single one."""
    return REPEAT_PUNCT_RE.sub(r"\1", text)


def fix_allcaps(text: str) -> str:
    """
    F6: If the text contains more than ALLCAPS_THRESHOLD uppercase letters,
    apply sentence case: lowercase everything, then capitalise the first
    alphabetic character after the start of the string and after . ! ?
    """
    upper_count = sum(1 for ch in text if ch.isupper())
    if upper_count <= ALLCAPS_THRESHOLD:
        return text

    lowered = text.lower()
    result  = []
    capitalise_next = True
    for ch in lowered:
        if capitalise_next and ch.isalpha():
            result.append(ch.upper())
            capitalise_next = False
        else:
            result.append(ch)
        if ch in ".!?":
            capitalise_next = True
    return "".join(result)


def fix_unpaired(text: str) -> str:
    """
    F7: For each bracket pair or quote character that is unmatched / unbalanced,
    remove ALL instances of that character (both opener and closer for brackets,
    all instances of the quote character for quotes).
    """
    # Bracket pairs: if opener count != closer count OR depth ever goes negative
    for open_c, close_c in [("(", ")"), ("[", "]"), ("{", "}")]:
        depth = 0
        bad   = False
        for ch in text:
            if ch == open_c:   depth += 1
            elif ch == close_c: depth -= 1
            if depth < 0:
                bad = True
                break
        if bad or depth != 0:
            text = text.replace(open_c, "").replace(close_c, "")

    # Straight double quotes: odd count → remove all
    if text.count('"') % 2 != 0:
        text = text.replace('"', "")

    return text


def fix_escaped_apostrophe(text: str) -> str:
    """
    F3b: Normalise escaped/prefixed quote characters to their plain equivalents.
    Handled variants (longer patterns replaced first to avoid partial matches):
      \\'  →  '       \\'  →  '       /'  →  '
      \\"  →  "       \\"  →  "
    """
    # Double-backslash variants first (must precede single-backslash variants)
    text = text.replace("\\\\'", "'")   # \\'  → '
    text = text.replace('\\\\"', '"')   # \\"  → "
    # Single-backslash variants
    text = text.replace("\\'", "'")     # \'   → '
    text = text.replace('\\"', '"')     # \"   → "
    # Forward-slash variant
    text = text.replace("/'", "'")      # /'   → '
    return text


def final_cleanup(text: str) -> str:
    """Collapse multiple spaces and strip ends — applied to every row."""
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


# ---------------------------------------------------------------------------
# Pipeline: apply all fixes in order and track which fired
# ---------------------------------------------------------------------------
FIXES = [
    ("F1  Non-ASCII removed",          fix_non_ascii),
    ("F2  HTML entities decoded",       fix_html_entities),
    ("F3  Curly quotes straightened",   fix_curly_quotes),
    ("F3b Escaped apostrophe \\' → '", fix_escaped_apostrophe),
    ("F4  Em/en dash → hyphen",         fix_dashes),
    ("F5  Repeated punct collapsed",    fix_repeated_punctuation),
    ("F6  All-caps → sentence case",    fix_allcaps),
    ("F7  Unpaired chars removed",      fix_unpaired),
]

def apply_fixes(text: str):
    """
    Returns (fixed_text, list_of_fix_names_that_changed_the_text).
    Does NOT include F8 (row removal) — that is handled separately.
    """
    fired = []
    for name, fn in FIXES:
        result = fn(text)
        if result != text:
            fired.append(name)
        text = result
    text = final_cleanup(text)
    return text, fired


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
print(f"Text Normalisation Fix Report — generated {RUN_TS}")
print(SEP)
print(f"  Input  : data/4_dataset_{{train,val,test}}.tsv")
print(f"  Output : data/5_dataset_{{train,val,test}}.tsv")
print(f"  Log    : {LOG_PATH}")
print(f"  All-caps threshold : > {ALLCAPS_THRESHOLD} uppercase letters")
print(SEP)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
header("0. LOADING INPUT FILES")

dfs = {}
orig_counts = {}
for split in SPLITS:
    path = DATA_DIR / f"4_dataset_{split}.tsv"
    if not path.exists():
        print(f"  [ERROR] Not found: {path}")
        sys.stdout = sys.__stdout__; _tee.close(); sys.exit(1)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    dfs[split]         = df
    orig_counts[split] = len(df)
    print(f"  {split:5s}: {len(df):>7,} rows  —  4_dataset_{split}.tsv")

print(f"\n  Total input rows: {sum(orig_counts.values()):,}")


# ---------------------------------------------------------------------------
# Apply fixes
# ---------------------------------------------------------------------------
header("APPLYING FIXES")

out_cols = ["vid", "text", "duration_sec", "word_count", "file_path", "source"]

# Accumulators
fix_stats   = {name: 0 for name, _ in FIXES}   # how many rows each fix touched
rows_removed = {}                                 # split -> count removed (F8)
rows_modified = {}                                # split -> count with any text change
SAMPLE = 5                                        # examples to show per fix per split

for split in SPLITS:
    subheader(f"Split: {split}  ({len(dfs[split]):,} rows)")
    df = dfs[split].copy()

    # Per-fix sample collector for this split
    fix_samples  = {name: [] for name, _ in FIXES}
    local_fired  = {name: 0  for name, _ in FIXES}

    new_texts = []
    remove_mask = []

    for _, row in df.iterrows():
        original_text = str(row["text"])

        # F8 check first — if no alphabetic content, flag for removal
        if not has_alpha(original_text):
            new_texts.append(original_text)
            remove_mask.append(True)
            continue

        fixed_text, fired = apply_fixes(original_text)

        # Collect samples and counts
        for name in fired:
            local_fired[name] += 1
            fix_stats[name]   += 1
            if len(fix_samples[name]) < SAMPLE:
                fix_samples[name].append({
                    "vid":    row["vid"],
                    "before": original_text[:120],
                    "after":  fixed_text[:120],
                })

        new_texts.append(fixed_text)
        remove_mask.append(False)

    df["text"]   = new_texts
    df["_remove"] = remove_mask

    # F8: remove rows with no alphabetic content
    n_removed = int(df["_remove"].sum())
    rows_removed[split] = n_removed
    df = df[~df["_remove"]].drop(columns=["_remove"]).reset_index(drop=True)

    # Count rows where text actually changed
    orig_texts = dfs[split]["text"].astype(str).tolist()
    n_modified = sum(
        1 for i, (o, n) in enumerate(zip(orig_texts, new_texts))
        if o != n and not remove_mask[i]
    )
    rows_modified[split] = n_modified

    # Print per-fix detail for this split
    print(f"\n  [{split}]  {n_modified:,} rows had text modified  |  {n_removed:,} rows removed (F8 no-alpha)")

    for fix_name, fn in FIXES:
        count = local_fired[fix_name]
        if count == 0:
            print(f"    {fix_name:35s}  {count:>5} rows affected")
            continue
        print(f"    {fix_name:35s}  {count:>5} rows affected")
        samples = fix_samples[fix_name]
        for s in samples:
            print(f"      vid   : {str(s['vid'])[:70]}")
            print(f"      before: {s['before']!r}")
            print(f"      after : {s['after']!r}")
        if count > SAMPLE:
            print(f"      ... {count - SAMPLE:,} more rows changed by this fix")

    dfs[split] = df


# ---------------------------------------------------------------------------
# Write output files
# ---------------------------------------------------------------------------
header("WRITING OUTPUT FILES")

for split in SPLITS:
    out_path = DATA_DIR / f"5_dataset_{split}.tsv"
    dfs[split][out_cols].to_csv(out_path, sep="\t", index=False)
    print(f"  Wrote {split:5s}: {len(dfs[split]):>7,} rows  →  {out_path.name}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
header("FINAL SUMMARY")

print(f"\n  Per-fix totals across all splits:")
print(f"  {'Fix':35s}  {'Rows affected':>14}")
print(f"  {'─'*35}  {'─'*14}")
for fix_name, _ in FIXES:
    print(f"  {fix_name:35s}  {fix_stats[fix_name]:>14,}")
print(f"  {'F8 No-alpha rows removed':35s}  {sum(rows_removed.values()):>14,}")

print(f"\n  Per-split results:")
print(f"  {'Split':6}  {'Original':>9}  {'Modified':>9}  {'Removed':>8}  {'Final':>8}  {'% changed':>10}")
print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*10}")
grand_orig = grand_final = 0
for split in SPLITS:
    orig    = orig_counts[split]
    mod     = rows_modified[split]
    removed = rows_removed[split]
    final   = len(dfs[split])
    pct     = 100 * (mod + removed) / orig
    grand_orig  += orig
    grand_final += final
    print(f"  {split:6}  {orig:>9,}  {mod:>9,}  {removed:>8,}  {final:>8,}  {pct:>9.2f}%")
print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*10}")
print(f"  {'TOTAL':6}  {grand_orig:>9,}  {'':>9}  {'':>8}  {grand_final:>8,}")

print(f"\n{SEP}")
print(f"  Normalisation complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Report saved to: {LOG_PATH}")
print(f"{SEP}\n")

sys.stdout = sys.__stdout__
_tee.close()
