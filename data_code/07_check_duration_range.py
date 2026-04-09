#!/usr/bin/env python3
"""
07_check_duration_range.py
==========================
Reads 2_dataset_{split}.tsv and reports how many rows have duration_sec
outside [1.3, 8.0]. No files are modified.
"""

import csv
from pathlib import Path

DATA_DIR    = Path(r"C:\My Projects\sign-language-bridge\data")
TSV_PATTERN = "2_dataset_{split}.tsv"
SPLITS      = ["train", "val", "test"]
MIN_DUR     = 1.3
MAX_DUR     = 8.0

def main():
    print(f"Valid range: [{MIN_DUR}s, {MAX_DUR}s]\n")

    grand_total    = 0
    grand_outliers = 0

    for split in SPLITS:
        tsv = DATA_DIR / TSV_PATTERN.format(split=split)
        if not tsv.exists():
            print(f"  [WARN] Not found: {tsv}")
            continue

        with open(tsv, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))

        n_total    = len(rows)
        n_outliers = 0
        n_too_short = 0
        n_too_long  = 0

        for row in rows:
            try:
                dur = float(row["duration_sec"])
            except (ValueError, KeyError):
                n_outliers += 1
                continue
            if dur < MIN_DUR:
                n_too_short += 1
                n_outliers  += 1
            elif dur > MAX_DUR:
                n_too_long += 1
                n_outliers += 1

        grand_total    += n_total
        grand_outliers += n_outliers

        print(f"  {split:<6}  total={n_total:,}  outliers={n_outliers:,}"
              f"  (too_short={n_too_short:,}  too_long={n_too_long:,})")

    print()
    print(f"  TOTAL  total={grand_total:,}  outliers={grand_outliers:,}")

if __name__ == "__main__":
    main()
