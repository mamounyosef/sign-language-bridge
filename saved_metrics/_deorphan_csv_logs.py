"""
Remove orphan rows from qwen3vl_train_log.csv (and qwen3vl_val_log.csv).

PROBLEM
───────
Training was stopped and resumed many times. Each crash leaves behind log rows
that came from steps the optimizer never persisted to disk — the model state
was rolled back to an earlier checkpoint on resume, but the rows from the
crashed run were already appended to the CSV.

The pattern looks like this in the global_step column:

    ..., 2028, 2029, 2030, 2031, 2032, 2033, 2030, 2031, 2032, 2033, 2034, ...
                          └── orphans (4 rows) ──┘   └── persisted run ──┘

The orphan block (steps 2030..2033 in their FIRST occurrence) was never saved
to the model. When the run resumed at step 2030, training continued forward
in lock-step with what the checkpoint actually contained — so the SECOND
occurrence of those steps is the authoritative one, and the FIRST should
be deleted.

ALGORITHM
─────────
Walk rows forward, maintaining a "kept" list:
  • Each row's global_step is read.
  • If current_step <= last_kept_step, this is a RESUME. Rewind the kept list
    by removing all kept rows whose global_step >= current_step. Then append
    the current row.
  • Otherwise, just append.

This is robust to multi-resume sequences (resume after resume) and to runs
where the orphan range extended past the resume point (e.g. crashed at step
2050 but resumed at 2030 — steps 2034..2050 in the first run only ever exist
in the orphan range, and the rewind correctly removes them too).

Run from the repo root, AFTER running _clean_csv_logs.py:

    python saved_metrics/_deorphan_csv_logs.py

Both files are backed up to *.bak2 before being overwritten so this can be
safely re-run.
"""
from __future__ import annotations
import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_CSV = HERE / 'qwen3vl_train_log.csv'
VAL_CSV   = HERE / 'qwen3vl_val_log.csv'

# Suffix used for backups so this doesn't collide with _clean_csv_logs.py's *.bak.
BACKUP_SUFFIX = '.bak2'


def _backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    shutil.copy2(path, bak)
    print(f"  ✓ Backup → {bak.name}")
    return bak


def _step_of(row: list[str], step_col: int) -> int | None:
    try:
        return int(float(row[step_col]))
    except (ValueError, IndexError):
        return None


def deorphan_csv(path: Path, step_field: str = 'global_step', label: str = '') -> None:
    print(f"\n[deorphan] {label or path.name}: {path}")
    if not path.exists():
        print("  (file missing — skipping)"); return

    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        print("  (no data rows)"); return

    header = rows[0]
    try:
        step_col = header.index(step_field)
    except ValueError:
        print(f"  ⚠️  '{step_field}' column not found in header — skipping")
        return

    _backup(path)

    # Single forward pass with rewind.
    kept: list[list[str]] = []
    n_dropped = 0
    n_resumes = 0
    last_kept_step = -1

    for r in rows[1:]:
        s = _step_of(r, step_col)
        if s is None:
            # Row with unparseable step — keep it (don't make decisions about junk).
            kept.append(r)
            continue

        if kept and s <= last_kept_step:
            # Resume detected. Rewind: drop all kept rows whose step >= s.
            n_resumes += 1
            cutoff = len(kept)
            while cutoff > 0:
                prev_step = _step_of(kept[cutoff - 1], step_col)
                if prev_step is None or prev_step < s:
                    break
                cutoff -= 1
            n_dropped += (len(kept) - cutoff)
            del kept[cutoff:]

        kept.append(r)
        last_kept_step = s

    # Verify monotonicity post-clean.
    bad = 0
    last = -1
    for r in kept:
        s = _step_of(r, step_col)
        if s is None:
            continue
        if s <= last:
            bad += 1
        last = s

    # Write back.
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)

    print(f"  resumes detected   : {n_resumes}")
    print(f"  orphan rows removed: {n_dropped}")
    print(f"  final row count    : {len(kept)} (was {len(rows) - 1})")
    print(f"  monotonic violations remaining: {bad}")


def main():
    print("De-orphaning saved-metrics CSV logs ...")
    deorphan_csv(TRAIN_CSV, label='train')
    deorphan_csv(VAL_CSV,   label='val')
    print("\nDone.")


if __name__ == '__main__':
    main()
