#!/usr/bin/env python3
"""
16_text_normalization_check.py
==============================
Scans data/4_dataset_{train,val,test}.tsv for text quality and normalisation
issues. Analysis only — no files are modified.

Checks performed:
  1.  Non-ASCII characters (codepoint > 127)
  2.  HTML entities  (&amp; &#39; &nbsp; etc.)
  3.  Multiple consecutive spaces (within text)
  4.  Leading / trailing whitespace
  5.  Non-breaking space (\\xa0)
  6.  Zero-width / invisible characters (\\u200b, \\u200c, \\u200d, \\ufeff …)
  7.  Control characters (\\t \\n \\r or other non-printable inside text)
  8.  Curly / smart quotes  (' ' " ")
  9.  Em dash or en dash  (— –)
  10. Ellipsis character  (… vs three literal dots)
  11. Repeated punctuation  (!! ?? .. -- more than twice in a row)
  12. All-caps text  (entire text is uppercase, likely a transcription error)
  13. No alphabetic content  (text contains no letters at all)
  14. Suspiciously long text  (character length > 500)
  15. Unpaired / mismatched brackets and quotes

Run : python data_code/16_text_normalization_check.py
"""

import html
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Tee
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


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
print(f"Text Normalisation Check Report — generated {RUN_TS}")
print(SEP)
print(f"  Input  : data/4_dataset_{{train,val,test}}.tsv")
print(f"  Log    : {LOG_PATH}")
print(f"  Mode   : analysis only — no files modified")
print(SEP)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
header("0. LOADING DATA")

dfs = {}
for split in SPLITS:
    path = DATA_DIR / f"4_dataset_{split}.tsv"
    if not path.exists():
        print(f"  [ERROR] Not found: {path}")
        sys.stdout = sys.__stdout__; _tee.close(); sys.exit(1)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    dfs[split] = df
    print(f"  {split:5s}: {len(df):>7,} rows")

all_df = pd.concat(
    [df.assign(_split=split) for split, df in dfs.items()],
    ignore_index=True
)
total = len(all_df)
print(f"\n  Total rows: {total:,}")


# ---------------------------------------------------------------------------
# Helper: print affected rows in a consistent table
# ---------------------------------------------------------------------------
SAMPLE_CAP = 10

def print_affected(rows, detail_col="_detail", cap=SAMPLE_CAP):
    """rows: list of dicts with keys: split, vid, source, text, _detail"""
    if not rows:
        print("  No issues found.")
        return
    shown = rows if len(rows) <= cap else rows[:cap]
    print(f"  {'#':>5}  {'Split':6}  {'Source':10}  {'Detail':30}  Text (first 90 chars)")
    print(f"  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*30}  {'─'*50}")
    for i, r in enumerate(shown, 1):
        detail = str(r.get("_detail", ""))[:30]
        text   = str(r.get("text", ""))[:90]
        print(f"  {i:>5}  {str(r['_split']):6}  {str(r['source'])[:10]:10}  {detail:30}  {text!r}")
        print(f"         vid: {str(r['vid'])[:70]}")
    if len(rows) > cap:
        print(f"  ... {len(rows) - cap:,} more rows not shown (total affected: {len(rows):,})")


def split_counts(rows):
    counts = {}
    for r in rows:
        counts[r["_split"]] = counts.get(r["_split"], 0) + 1
    return "  |  ".join(f"{s}={counts.get(s,0):,}" for s in SPLITS)


def to_rows(df_subset, split, detail_fn):
    out = []
    for _, row in df_subset.iterrows():
        out.append({
            "_split":  split,
            "vid":     row.get("vid", ""),
            "source":  row.get("source", ""),
            "text":    row.get("text", ""),
            "_detail": detail_fn(row),
        })
    return out


# ---------------------------------------------------------------------------
# CHECK 1 — Non-ASCII characters
# ---------------------------------------------------------------------------
header("CHECK 1 — NON-ASCII CHARACTERS  (codepoint > 127)")

non_ascii_rows = []
for split, df in dfs.items():
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        bad_chars = [(ch, unicodedata.name(ch, "UNKNOWN"), unicodedata.category(ch))
                     for ch in text if ord(ch) > 127]
        if bad_chars:
            # Deduplicate by character
            seen_chars = {}
            for ch, name, cat in bad_chars:
                seen_chars[ch] = (name, cat)
            detail = "  ".join(
                f"U+{ord(ch):04X} {name[:20]}"
                for ch, (name, cat) in list(seen_chars.items())[:3]
            )
            non_ascii_rows.append({
                "_split": split, "vid": row["vid"], "source": row["source"],
                "text": text, "_detail": detail,
                "_chars": seen_chars,
            })

total_non_ascii = len(non_ascii_rows)
pct = 100 * total_non_ascii / total
print(f"  Total affected rows: {total_non_ascii:,}  ({pct:.2f}%)  |  {split_counts(non_ascii_rows)}\n")

if non_ascii_rows:
    # Character frequency table
    char_freq = {}
    for r in non_ascii_rows:
        for ch, (name, cat) in r["_chars"].items():
            char_freq[ch] = char_freq.get(ch, 0) + 1
    print(f"  Distinct non-ASCII characters found  ({len(char_freq)} unique):")
    print(f"  {'Char':6}  {'U+':8}  {'Count':>6}  {'Category':4}  Unicode Name")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*4}  {'─'*40}")
    for ch, cnt in sorted(char_freq.items(), key=lambda x: -x[1]):
        name = unicodedata.name(ch, "UNKNOWN")
        cat  = unicodedata.category(ch)
        print(f"  {repr(ch):6}  U+{ord(ch):04X}  {cnt:>6,}  {cat:4}  {name}")
    print()
    print_affected(non_ascii_rows, cap=100)


# ---------------------------------------------------------------------------
# CHECK 2 — HTML entities
# ---------------------------------------------------------------------------
header("CHECK 2 — HTML ENTITIES  (&amp; &#39; &nbsp; etc.)")

HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{1,6}|#x[0-9a-fA-F]{1,5});")

html_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: bool(HTML_ENTITY_RE.search(str(t))))
    for _, row in df[mask].iterrows():
        entities = HTML_ENTITY_RE.findall(str(row["text"]))
        html_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": "  ".join(dict.fromkeys(entities))[:30],
        })

pct = 100 * len(html_rows) / total
print(f"  Total affected rows: {len(html_rows):,}  ({pct:.2f}%)  |  {split_counts(html_rows)}\n")
print_affected(html_rows)


# ---------------------------------------------------------------------------
# CHECK 3 — Multiple consecutive spaces
# ---------------------------------------------------------------------------
header("CHECK 3 — MULTIPLE CONSECUTIVE SPACES")

MULTI_SPACE_RE = re.compile(r" {2,}")

multi_space_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: bool(MULTI_SPACE_RE.search(str(t))))
    for _, row in df[mask].iterrows():
        matches = MULTI_SPACE_RE.findall(str(row["text"]))
        detail  = f"{len(matches)} occurrence(s), max {max(len(m) for m in matches)} spaces"
        multi_space_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": detail,
        })

pct = 100 * len(multi_space_rows) / total
print(f"  Total affected rows: {len(multi_space_rows):,}  ({pct:.2f}%)  |  {split_counts(multi_space_rows)}\n")
print_affected(multi_space_rows)


# ---------------------------------------------------------------------------
# CHECK 4 — Leading / trailing whitespace
# ---------------------------------------------------------------------------
header("CHECK 4 — LEADING / TRAILING WHITESPACE")

strip_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: str(t) != str(t).strip())
    for _, row in df[mask].iterrows():
        t = str(row["text"])
        detail = f"leading={len(t)-len(t.lstrip())}  trailing={len(t)-len(t.rstrip())}"
        strip_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": detail,
        })

pct = 100 * len(strip_rows) / total
print(f"  Total affected rows: {len(strip_rows):,}  ({pct:.2f}%)  |  {split_counts(strip_rows)}\n")
print_affected(strip_rows)


# ---------------------------------------------------------------------------
# CHECK 5 — Non-breaking space (\xa0)
# ---------------------------------------------------------------------------
header("CHECK 5 — NON-BREAKING SPACE  (\\xa0 / &nbsp;)")

nbsp_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: "\xa0" in str(t))
    for _, row in df[mask].iterrows():
        cnt = str(row["text"]).count("\xa0")
        nbsp_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": f"{cnt} occurrence(s)",
        })

pct = 100 * len(nbsp_rows) / total
print(f"  Total affected rows: {len(nbsp_rows):,}  ({pct:.2f}%)  |  {split_counts(nbsp_rows)}\n")
print_affected(nbsp_rows)


# ---------------------------------------------------------------------------
# CHECK 6 — Zero-width / invisible characters
# ---------------------------------------------------------------------------
header("CHECK 6 — ZERO-WIDTH / INVISIBLE CHARACTERS")

ZERO_WIDTH = {
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\u200e": "LEFT-TO-RIGHT MARK",
    "\u200f": "RIGHT-TO-LEFT MARK",
    "\u2060": "WORD JOINER",
    "\ufeff": "ZERO WIDTH NO-BREAK SPACE (BOM)",
    "\u00ad": "SOFT HYPHEN",
}

zw_rows = []
for split, df in dfs.items():
    for _, row in df.iterrows():
        text = str(row["text"])
        found = {ch: name for ch, name in ZERO_WIDTH.items() if ch in text}
        if found:
            detail = "  ".join(list(found.values()))[:30]
            zw_rows.append({
                "_split": split, "vid": row["vid"], "source": row["source"],
                "text": text, "_detail": detail,
            })

pct = 100 * len(zw_rows) / total
print(f"  Total affected rows: {len(zw_rows):,}  ({pct:.2f}%)  |  {split_counts(zw_rows)}\n")
print_affected(zw_rows)


# ---------------------------------------------------------------------------
# CHECK 7 — Control characters (\t \n \r and other non-printables)
# ---------------------------------------------------------------------------
header("CHECK 7 — CONTROL CHARACTERS  (\\t \\n \\r or non-printable)")

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\t\n\r]")

ctrl_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: bool(CTRL_RE.search(str(t))))
    for _, row in df[mask].iterrows():
        chars = set(CTRL_RE.findall(str(row["text"])))
        detail = "  ".join(repr(c) for c in chars)[:30]
        ctrl_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": detail,
        })

pct = 100 * len(ctrl_rows) / total
print(f"  Total affected rows: {len(ctrl_rows):,}  ({pct:.2f}%)  |  {split_counts(ctrl_rows)}\n")
print_affected(ctrl_rows)


# ---------------------------------------------------------------------------
# CHECK 8 — Curly / smart quotes
# ---------------------------------------------------------------------------
header("CHECK 8 — CURLY / SMART QUOTES  (' ' \" \")")

CURLY = {
    "\u2018": "LEFT SINGLE QUOTATION MARK",
    "\u2019": "RIGHT SINGLE QUOTATION MARK",
    "\u201c": "LEFT DOUBLE QUOTATION MARK",
    "\u201d": "RIGHT DOUBLE QUOTATION MARK",
    "\u2032": "PRIME",
    "\u2033": "DOUBLE PRIME",
}
CURLY_RE = re.compile(f"[{''.join(CURLY)}]")

curly_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: bool(CURLY_RE.search(str(t))))
    for _, row in df[mask].iterrows():
        found = {ch: CURLY[ch] for ch in CURLY if ch in str(row["text"])}
        detail = "  ".join(f"{repr(ch)} ({name[:15]})" for ch, name in found.items())[:40]
        curly_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": detail,
        })

pct = 100 * len(curly_rows) / total
print(f"  Total affected rows: {len(curly_rows):,}  ({pct:.2f}%)  |  {split_counts(curly_rows)}\n")
print_affected(curly_rows)


# ---------------------------------------------------------------------------
# CHECK 9 — Em dash / en dash
# ---------------------------------------------------------------------------
header("CHECK 9 — EM DASH / EN DASH  (— –)")

DASH_RE = re.compile(r"[–—]")

dash_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: bool(DASH_RE.search(str(t))))
    for _, row in df[mask].iterrows():
        found = set(DASH_RE.findall(str(row["text"])))
        detail = "  ".join(
            f"{'EM DASH' if c == '—' else 'EN DASH'} ×{str(row['text']).count(c)}"
            for c in found
        )
        dash_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": detail,
        })

pct = 100 * len(dash_rows) / total
print(f"  Total affected rows: {len(dash_rows):,}  ({pct:.2f}%)  |  {split_counts(dash_rows)}\n")
print_affected(dash_rows)


# ---------------------------------------------------------------------------
# CHECK 10 — Ellipsis character (…)
# ---------------------------------------------------------------------------
header("CHECK 10 — ELLIPSIS CHARACTER  (… U+2026 vs literal ...)")

ellipsis_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: "\u2026" in str(t))
    for _, row in df[mask].iterrows():
        cnt = str(row["text"]).count("\u2026")
        ellipsis_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": f"{cnt} ellipsis char(s)",
        })

pct = 100 * len(ellipsis_rows) / total
print(f"  Total affected rows: {len(ellipsis_rows):,}  ({pct:.2f}%)  |  {split_counts(ellipsis_rows)}\n")
print_affected(ellipsis_rows)


# ---------------------------------------------------------------------------
# CHECK 11 — Repeated punctuation (!! ?? .. -- ...)
# ---------------------------------------------------------------------------
header("CHECK 11 — REPEATED PUNCTUATION  (!! ?? .. -- appearing 2+ times in a row)")

REPEAT_PUNCT_RE = re.compile(r"([!?.\-,;:])\1{1,}")

repeat_punct_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: bool(REPEAT_PUNCT_RE.search(str(t))))
    for _, row in df[mask].iterrows():
        matches = REPEAT_PUNCT_RE.findall(str(row["text"]))
        full_matches = REPEAT_PUNCT_RE.findall(str(row["text"]))
        all_full = [m.group() for m in REPEAT_PUNCT_RE.finditer(str(row["text"]))]
        detail = "  ".join(dict.fromkeys(all_full))[:30]
        repeat_punct_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": detail,
        })

pct = 100 * len(repeat_punct_rows) / total
print(f"  Total affected rows: {len(repeat_punct_rows):,}  ({pct:.2f}%)  |  {split_counts(repeat_punct_rows)}\n")
print_affected(repeat_punct_rows)


# ---------------------------------------------------------------------------
# CHECK 12 — All-caps text
# ---------------------------------------------------------------------------
header("CHECK 12 — ALL-CAPS TEXT  (entire text is uppercase, >= 4 letters)")

allcaps_rows = []
for split, df in dfs.items():
    for _, row in df.iterrows():
        text    = str(row["text"])
        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 4 and all(c.isupper() for c in letters):
            allcaps_rows.append({
                "_split": split, "vid": row["vid"], "source": row["source"],
                "text": text, "_detail": f"{len(letters)} uppercase letters",
            })

pct = 100 * len(allcaps_rows) / total
print(f"  Total affected rows: {len(allcaps_rows):,}  ({pct:.2f}%)  |  {split_counts(allcaps_rows)}\n")
print_affected(allcaps_rows)


# ---------------------------------------------------------------------------
# CHECK 13 — No alphabetic content
# ---------------------------------------------------------------------------
header("CHECK 13 — NO ALPHABETIC CONTENT  (text contains no letters)")

no_alpha_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: not any(c.isalpha() for c in str(t)))
    for _, row in df[mask].iterrows():
        no_alpha_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": f"len={len(str(row['text']))} chars",
        })

pct = 100 * len(no_alpha_rows) / total
print(f"  Total affected rows: {len(no_alpha_rows):,}  ({pct:.2f}%)  |  {split_counts(no_alpha_rows)}\n")
print_affected(no_alpha_rows)


# ---------------------------------------------------------------------------
# CHECK 14 — Suspiciously long text (> 500 characters)
# ---------------------------------------------------------------------------
header("CHECK 14 — SUSPICIOUSLY LONG TEXT  (character length > 500)")

long_rows = []
for split, df in dfs.items():
    mask = df["text"].apply(lambda t: len(str(t)) > 500)
    for _, row in df[mask].iterrows():
        long_rows.append({
            "_split": split, "vid": row["vid"], "source": row["source"],
            "text": row["text"], "_detail": f"{len(str(row['text']))} chars",
        })

pct = 100 * len(long_rows) / total
print(f"  Total affected rows: {len(long_rows):,}  ({pct:.2f}%)  |  {split_counts(long_rows)}\n")
print_affected(long_rows)


# ---------------------------------------------------------------------------
# CHECK 15 — Unpaired / mismatched brackets and quotes
# ---------------------------------------------------------------------------
header("CHECK 15 — UNPAIRED / MISMATCHED BRACKETS AND QUOTES")

def check_pairs(text):
    issues = []
    # Brackets
    for open_c, close_c in [("(", ")"), ("[", "]"), ("{", "}")]:
        depth = 0
        for ch in text:
            if ch == open_c:   depth += 1
            elif ch == close_c: depth -= 1
            if depth < 0:
                issues.append(f"extra '{close_c}'")
                break
        if depth > 0:
            issues.append(f"unclosed '{open_c}' ×{depth}")
    # Straight double quotes (should be even count)
    if text.count('"') % 2 != 0:
        issues.append(f'odd " count ({text.count(chr(34))})')
    return issues

mismatch_rows = []
for split, df in dfs.items():
    for _, row in df.iterrows():
        text   = str(row["text"])
        issues = check_pairs(text)
        if issues:
            mismatch_rows.append({
                "_split": split, "vid": row["vid"], "source": row["source"],
                "text": text, "_detail": "  ".join(issues)[:40],
            })

pct = 100 * len(mismatch_rows) / total
print(f"  Total affected rows: {len(mismatch_rows):,}  ({pct:.2f}%)  |  {split_counts(mismatch_rows)}\n")
print_affected(mismatch_rows)


# ---------------------------------------------------------------------------
# OVERALL SUMMARY
# ---------------------------------------------------------------------------
header("OVERALL SUMMARY")

checks = [
    ("1  Non-ASCII characters",          non_ascii_rows),
    ("2  HTML entities",                  html_rows),
    ("3  Multiple consecutive spaces",    multi_space_rows),
    ("4  Leading/trailing whitespace",    strip_rows),
    ("5  Non-breaking space (\\xa0)",     nbsp_rows),
    ("6  Zero-width characters",          zw_rows),
    ("7  Control characters",             ctrl_rows),
    ("8  Curly/smart quotes",             curly_rows),
    ("9  Em/en dash",                     dash_rows),
    ("10 Ellipsis character (…)",         ellipsis_rows),
    ("11 Repeated punctuation",           repeat_punct_rows),
    ("12 All-caps text",                  allcaps_rows),
    ("13 No alphabetic content",          no_alpha_rows),
    ("14 Suspiciously long text (>500c)", long_rows),
    ("15 Unpaired brackets/quotes",       mismatch_rows),
]

# Collect all affected row identifiers (split+vid) across all checks
all_affected_vids = set()
for _, rows in checks:
    for r in rows:
        all_affected_vids.add((r["_split"], r["vid"]))

print(f"\n  {'Check':<35}  {'Affected':>8}  {'%':>6}  {' | '.join(SPLITS)}")
print(f"  {'─'*35}  {'─'*8}  {'─'*6}  {'─'*30}")
for name, rows in checks:
    cnt = len(rows)
    pct = 100 * cnt / total
    by_split = "  ".join(f"{s}={sum(1 for r in rows if r['_split']==s):>4}" for s in SPLITS)
    flag = "  ← action needed" if cnt > 0 else ""
    print(f"  {name:<35}  {cnt:>8,}  {pct:>5.2f}%  {by_split}{flag}")

print(f"\n  Total rows with at least one issue : {len(all_affected_vids):,}  ({100*len(all_affected_vids)/total:.2f}%)")
print(f"  Total rows clean (no issues found) : {total - len(all_affected_vids):,}  ({100*(total-len(all_affected_vids))/total:.2f}%)")

print(f"\n{SEP}")
print(f"  Analysis complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Report saved to: {LOG_PATH}")
print(f"{SEP}\n")

sys.stdout = sys.__stdout__
_tee.close()
