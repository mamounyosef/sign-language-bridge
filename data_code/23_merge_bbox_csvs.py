"""
23_merge_bbox_csvs.py
=====================
One-time utility: merges the three split-specific bbox CSVs produced by
11_extract_signer_bboxes.py into a single unified CSV.

Background
----------
The bbox CSVs were extracted before the dataset splits were re-shuffled
(scripts 13-19). As a result, many clips have a valid bbox entry but it sits
in the wrong split's CSV (e.g. a clip now in the train TSV has its bbox in
the val bbox CSV). 21_extract_landmarks.py only looks in the same-split CSV,
so those clips were silently skipped.

This script produces a single file:
    data/bboxes_all.csv

that covers all clips regardless of which split their bbox was originally
extracted under. Duplicates (same vid appearing in multiple CSVs) are resolved
by keeping the entry with the highest detection_rate, or the first ok entry if
detection_rate is equal.

After running this script, update 21_extract_landmarks.py to load
bboxes_all.csv instead of the three split-specific files.

Usage
-----
    python data_code/23_merge_bbox_csvs.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

INPUT_CSVS = [
    DATA_DIR / "2_dataset_train_bboxes.csv",
    DATA_DIR / "2_dataset_val_bboxes.csv",
    DATA_DIR / "2_dataset_test_bboxes.csv",
]
OUTPUT_CSV = DATA_DIR / "bboxes_all.csv"


def main() -> None:
    frames = []
    for p in INPUT_CSVS:
        if not p.exists():
            print(f"  SKIP (not found): {p}")
            continue
        df = pd.read_csv(p)
        df["_source_split"] = p.stem   # track origin for debugging
        frames.append(df)
        print(f"  Loaded {len(df):,} rows from {p.name}")

    if not frames:
        print("ERROR: no bbox CSVs found.")
        return

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined (before dedup): {len(combined):,} rows")

    # Separate ok and failed rows.
    ok     = combined[~combined["failed"]].copy()
    failed = combined[ combined["failed"]].copy()
    print(f"  ok: {len(ok):,}  |  failed: {len(failed):,}")

    # If a vid has at least one ok entry, keep only ok entries for it.
    # Among ok entries for the same vid, keep the one with the highest detection_rate.
    ok_sorted = ok.sort_values("detection_rate", ascending=False)
    ok_dedup  = ok_sorted.drop_duplicates(subset="vid", keep="first")

    # For vids that only appear in failed rows, keep the first failed entry.
    vids_with_ok = set(ok_dedup["vid"].astype(str))
    failed_only  = failed[~failed["vid"].astype(str).isin(vids_with_ok)]
    failed_dedup = failed_only.drop_duplicates(subset="vid", keep="first")

    result = pd.concat([ok_dedup, failed_dedup], ignore_index=True)
    result = result.sort_values("vid").reset_index(drop=True)

    # Drop the internal tracking column before saving
    result = result.drop(columns=["_source_split"])

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(result):,} unique vids to: {OUTPUT_CSV}")
    print(f"  ok entries    : {int((~result['failed']).sum()):,}")
    print(f"  failed entries: {int(result['failed'].sum()):,}")

    # Sanity check against current 7_dataset TSVs
    print("\nCoverage check against current 7_dataset TSVs:")
    for split in ["train", "val", "test"]:
        tsv_path = DATA_DIR / f"7_dataset_{split}.tsv"
        if not tsv_path.exists():
            print(f"  [{split}] TSV not found, skipping")
            continue
        tsv_vids    = set(str(v) for v in pd.read_csv(tsv_path, sep="\t")["vid"])
        result_vids = set(str(v) for v in result[~result["failed"]]["vid"])
        covered     = tsv_vids & result_vids
        missing     = tsv_vids - result_vids
        print(f"  [{split}] {len(covered):,}/{len(tsv_vids):,} covered "
              f"({100*len(covered)/max(len(tsv_vids),1):.1f}%)  |  still missing: {len(missing):,}")


if __name__ == "__main__":
    main()
