#!/usr/bin/env python3
"""
05_build_final_tsv.py
=====================
Produces the final combined TSV files for the How2Sign + OpenASL dataset:
    C:\\My Projects\\sign-language-bridge\\data\\1_dataset_train.tsv
    C:\\My Projects\\sign-language-bridge\\data\\1_dataset_val.tsv
    C:\\My Projects\\sign-language-bridge\\data\\1_dataset_test.tsv

What this script does for every row in the existing TSVs:
  1. Updates file_path to the new output location:
       how2sign  ->  D:\\how2sign\\{split}\\{vid}.mp4
       openasl   ->  D:\\OpenASL\\raw_videos_clipped\\{split}\\{filename}.mp4
                     (colons in vid are replaced with dots to match filenames)
  2. Checks that the output file exists on disk.
       - If it does not exist: skip the row.
  3. Uses duration_sec directly from the existing TSV — no re-probing needed.
       For OpenASL this was set correctly by the clipping pipeline.
       For How2Sign this was set correctly by 04_build_how2sign_dataset.py.
  4. Drops start and end columns entirely.
  5. Writes: vid, text, duration_sec, word_count, file_path, source

Run AFTER:
  - 04_build_how2sign_dataset.py  (populates D:\\how2sign\\)
  - 02_process_openasl_clips.py   (populates D:\\OpenASL\\raw_videos_clipped\\)
"""

import csv
from pathlib import Path

# =============================================================================
#  CONFIGURATION
# =============================================================================

DATA_DIR     = Path(r"C:\My Projects\sign-language-bridge\data")
TSV_PATTERN  = "final_full_{split}_dataset.tsv"
OUT_PATTERN  = "1_dataset_{split}.tsv"
SPLITS       = ["train", "val", "test"]

H2S_OUT_DIR  = Path(r"D:\how2sign")
ASL_OUT_DIR  = Path(r"D:\OpenASL\raw_videos_clipped")

OUT_COLUMNS  = ["vid", "text", "duration_sec", "word_count", "file_path", "source"]

# =============================================================================


def resolve_path(vid: str, source: str, split: str) -> Path:
    """
    Return the expected output file path for a given vid and source.

    How2Sign: D:\\how2sign\\{split}\\{vid}.mp4
    OpenASL:  D:\\OpenASL\\raw_videos_clipped\\{split}\\{vid_dotted}.mp4
              Colons in vid are replaced with dots to match actual filenames.
              e.g. vid  = 'RjJSxzMkC90-00:00:25.123-00:00:27.524'
                   file = 'RjJSxzMkC90-00.00.25.123-00.00.27.524.mp4'
    """
    if source == "how2sign":
        return H2S_OUT_DIR / split / f"{vid}.mp4"
    else:
        filename = vid.replace(":", ".") + ".mp4"
        return ASL_OUT_DIR / split / filename


def process_split(split: str) -> tuple[int, int, int]:
    """
    Process one split. Returns (n_total, n_missing, n_written).
    """
    tsv_in  = DATA_DIR / TSV_PATTERN.format(split=split)
    tsv_out = DATA_DIR / OUT_PATTERN.format(split=split)

    if not tsv_in.exists():
        print(f"  [WARN] Input TSV not found, skipping: {tsv_in}")
        return 0, 0, 0

    with open(tsv_in, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    n_total   = len(rows)
    n_missing = 0
    n_written = 0
    out_rows  = []

    for row in rows:
        vid    = row["vid"].strip()
        source = row.get("source", "").strip().lower()

        new_path = resolve_path(vid, source, split)

        if not new_path.exists():
            n_missing += 1
            continue

        try:
            word_count = int(float(row.get("word_count") or 0))
        except ValueError:
            word_count = 0

        out_rows.append({
            "vid":          vid,
            "text":         row.get("text", "").strip(),
            "duration_sec": row["duration_sec"].strip(),
            "word_count":   word_count,
            "file_path":    str(new_path),
            "source":       source,
        })
        n_written += 1

    with open(tsv_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    return n_total, n_missing, n_written


def main():
    print("=" * 66)
    print("  Building final combined TSV files")
    print(f"  How2Sign  output dir : {H2S_OUT_DIR}")
    print(f"  OpenASL   output dir : {ASL_OUT_DIR}")
    print(f"  Output columns       : {OUT_COLUMNS}")
    print("=" * 66)
    print()

    total_written = 0
    total_missing = 0

    for split in SPLITS:
        print(f"  Processing split: {split} ...")
        n_total, n_missing, n_written = process_split(split)

        out_path = DATA_DIR / OUT_PATTERN.format(split=split)
        print(f"    Input rows     : {n_total:,}")
        print(f"    File not found : {n_missing:,}  (skipped)")
        print(f"    Written        : {n_written:,}")
        print(f"    Output TSV     : {out_path}")
        print()

        total_written += n_written
        total_missing += n_missing

    print("=" * 66)
    print(f"  Total rows written across all splits : {total_written:,}")
    print(f"  Total skipped (file not found)       : {total_missing:,}")
    print("=" * 66)


if __name__ == "__main__":
    main()
