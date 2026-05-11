"""
Analysis + plots for the Qwen3-VL training run.

Reads the cleaned CSV logs in saved_metrics/ and writes:
  • PNG plots → images/log_plots/
  • metrics_summary.json + summary.txt → images/log_plots/

USAGE
─────
    python model_training_scripts/qwen3vl_log_analysis.py

Run _clean_csv_logs.py first if the CSVs haven't been cleaned yet.

All knobs live in CONFIG below.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = {
    'train_csv':  REPO_ROOT / 'saved_metrics' / 'qwen3vl_train_log.csv',
    'val_csv':    REPO_ROOT / 'saved_metrics' / 'qwen3vl_val_log.csv',
    'output_dir': REPO_ROOT / 'images' / 'log_plots',

    # Display name for the run (appears in legends / titles).
    'run_label':  'Two-phase (OpenASL → How2Sign)',

    # Smoothing window (in samples) for noisy train metrics like loss / grad_norm.
    # Set to 1 to disable smoothing; the raw curve is always plotted as a faint line behind.
    'smooth_window':          50,

    # Plateau detection: a metric is "plateaued" if its value over the last
    # `plateau_tail_frac` fraction of training varies by less than `plateau_tol_rel`
    # relative to the peak. Used in the summary stats only.
    'plateau_tail_frac':      0.20,
    'plateau_tol_rel':        0.02,   # 2% of peak

    # Dataset phase transition: step where training switches from OpenASL → How2Sign.
    # Hard-coded because the CSV 'phase' column captures the intra-OpenASL freeze
    # transition (step 489), not the dataset switch.
    'dataset_phase_step': 2449,
    # Intra-OpenASL unfreeze step (LM LoRA + Head LoRA unlocked within OpenASL phase).
    'unfreeze_step': 489,

    # Plot styling
    'dpi':            150,
    'figsize':        (10, 5.5),
    'figsize_grid':   (12, 8),
    'phase_line_color': '#9a4c95',   # vertical line at phase transition
    'phase_line_alpha': 0.45,
}


# ═════════════════════════════════════════════════════════════════════════════
#  IO HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _to_float(s: str):
    if s is None or s == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _col(rows: list[dict], key: str, x_key: str = 'global_step'):
    """Return aligned (xs, ys) lists where the value parses to float."""
    xs, ys = [], []
    for r in rows:
        x = _to_float(r.get(x_key, ''))
        y = _to_float(r.get(key, ''))
        if x is None or y is None:
            continue
        xs.append(x); ys.append(y)
    return xs, ys


def _smooth(ys: list[float], window: int) -> list[float]:
    """Simple centered moving average. Edges are shorter windows."""
    if window <= 1 or not ys:
        return ys[:]
    n = len(ys)
    out = [0.0] * n
    half = window // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = sum(ys[a:b]) / (b - a)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE TRANSITION
# ═════════════════════════════════════════════════════════════════════════════
def _detect_phase_transition_step(train_rows: list[dict]) -> float | None:
    """First global_step where the 'phase' column transitions from 1 → 2.

    In the training script's logging, 'phase' tracks the dataset phase
    (1 = OpenASL, 2 = How2Sign). The transition step is useful as a visual
    landmark on every time-series plot.
    """
    last_phase = None
    for r in train_rows:
        try:
            p = int(r.get('phase', '') or 0)
        except ValueError:
            continue
        if last_phase is not None and p != last_phase and p == 2:
            return _to_float(r.get('global_step', ''))
        last_phase = p
    return None


def _draw_phase_line(ax, step: float | None):
    if step is None:
        return
    ax.axvline(step, color=CONFIG['phase_line_color'],
               alpha=CONFIG['phase_line_alpha'], linestyle='--', linewidth=1.2,
               label=f'phase 1→2  (step {int(step)})')


def _draw_unfreeze_line(ax):
    step = CONFIG.get('unfreeze_step')
    if step is None:
        return
    ax.axvline(step, color='gray', alpha=0.6, linestyle='--', linewidth=1.2,
               label=f'unfreeze  (step {int(step)})')


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════
def _save(fig, name: str):
    out = CONFIG['output_dir'] / name
    for ax in fig.axes:
        ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out, dpi=CONFIG['dpi'])
    plt.close(fig)
    print(f"  ✓ {out.relative_to(REPO_ROOT)}")


def _line_plot(xs, ys_list, labels, title, ylabel, fname,
               phase_step=None, smooth=False, ylog=False):
    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    for ys, label in zip(ys_list, labels):
        if smooth and CONFIG['smooth_window'] > 1:
            ax.plot(xs, ys, alpha=0.18, linewidth=0.8)
            ax.plot(xs, _smooth(ys, CONFIG['smooth_window']), label=label, linewidth=1.6)
        else:
            ax.plot(xs, ys, label=label, linewidth=1.6, marker='o', markersize=2.5)
    _draw_phase_line(ax, phase_step)
    ax.set_xlabel('global_step')
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  —  {CONFIG['run_label']}")
    if ylog:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    _save(fig, fname)


# ═════════════════════════════════════════════════════════════════════════════
#  TRAIN-SIDE PLOTS
# ═════════════════════════════════════════════════════════════════════════════
def plot_train_loss(train_rows, val_rows, phase_step):
    xs_t, ys_t = _col(train_rows, 'train_loss')
    xs_v, ys_v = _col(val_rows,   'val_loss')

    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    if xs_t:
        ax.plot(xs_t, ys_t, alpha=0.18, linewidth=0.8, color='C0')
        ax.plot(xs_t, _smooth(ys_t, CONFIG['smooth_window']),
                label='train_loss (smoothed)', linewidth=1.6, color='C0')
    if xs_v:
        ax.plot(xs_v, ys_v, label='val_loss', linewidth=1.8, color='C1',
                marker='o', markersize=3)
    _draw_unfreeze_line(ax)
    if phase_step is not None:
        x_min, x_max = ax.get_xlim()
        ax.axvspan(x_min, phase_step, alpha=0.06, color='steelblue', zorder=0)
        ax.axvspan(phase_step, x_max, alpha=0.06, color='darkorange', zorder=0)
        ax.axvline(phase_step, color=CONFIG['phase_line_color'],
                   alpha=0.85, linestyle='--', linewidth=0.8,
                   label=f'phase 1→2  (step {int(phase_step)})')
        ax.text(phase_step * 0.5, 1.0, 'OpenASL',
                ha='center', va='top', fontsize=9, color='steelblue',
                fontweight='bold', transform=ax.get_xaxis_transform())
        ax.text(phase_step + (x_max - phase_step) * 0.5, 1.0, 'How2Sign',
                ha='center', va='top', fontsize=9, color='darkorange',
                fontweight='bold', transform=ax.get_xaxis_transform())
    ax.set_xlabel('global_step')
    ax.set_ylabel('cross-entropy loss')
    ax.set_title(f"Train vs Val loss  —  {CONFIG['run_label']}")
    ax.grid(True, alpha=0.25); ax.legend(loc='best', fontsize=9)
    _save(fig, '01_loss.png')


def plot_train_ppl(train_rows, val_rows, phase_step):
    xs_t, ys_t = _col(train_rows, 'train_ppl')
    xs_v, ys_v = _col(val_rows,   'val_ppl')
    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    if xs_t:
        ax.plot(xs_t, ys_t, alpha=0.18, linewidth=0.8, color='C0')
        ax.plot(xs_t, _smooth(ys_t, CONFIG['smooth_window']),
                label='train_ppl (smoothed)', linewidth=1.6, color='C0')
    if xs_v:
        ax.plot(xs_v, ys_v, label='val_ppl', linewidth=1.8, color='C1',
                marker='o', markersize=3)
    _draw_phase_line(ax, phase_step)
    ax.set_xlabel('global_step'); ax.set_ylabel('perplexity')
    ax.set_yscale('log')
    ax.set_title(f"Perplexity (log)  —  {CONFIG['run_label']}")
    ax.grid(True, alpha=0.25, which='both'); ax.legend(loc='best', fontsize=9)
    _save(fig, '02_perplexity.png')


def plot_lr_schedules(train_rows, phase_step):
    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    all_positive = []
    # T4 is nudged down by 3% on the log scale so it stays visible when it
    # overlaps exactly with T2 (they share the same peak/min LRs in Phase 1).
    tier_nudge = {'lr_t4': 0.95}
    for tier, color in [('lr_t1', 'C0'), ('lr_t2', 'C1'), ('lr_t3', 'C2'), ('lr_t4', 'C3')]:
        xs_k, ys_k = _col(train_rows, tier)
        if not xs_k:
            continue
        nudge = tier_nudge.get(tier, 1.0)
        # Replace 0.0 (inactive tier logged as zero) with NaN so log-scale gaps
        # appear instead of a misleading flat line at the axis floor.
        ys_clean = [y * nudge if y > 0 else float('nan') for y in ys_k]
        ax.plot(xs_k, ys_clean, label=tier, linewidth=1.5, color=color)
        all_positive.extend(y for y in ys_k if y > 0)
    ax.set_xlabel('global_step'); ax.set_ylabel('learning rate')
    ax.set_yscale('log')
    # Extend the lower y-limit two decades below the smallest positive LR so the
    # T3 warmup rise from near-zero is clearly visible.
    ax.set_ylim(bottom=1e-7)
    _draw_unfreeze_line(ax)
    # Phase transition annotation — with shading and dataset labels.
    if phase_step is not None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.axvspan(x_min, phase_step, alpha=0.06, color='steelblue', zorder=0)
        ax.axvspan(phase_step, x_max, alpha=0.06, color='darkorange', zorder=0)
        ax.axvline(phase_step, color=CONFIG['phase_line_color'],
                   alpha=0.85, linestyle='--', linewidth=0.8,
                   label=f'phase 1→2  (step {int(phase_step)})')
        ax.text(phase_step * 0.5, y_max, 'OpenASL',
                ha='center', va='top', fontsize=9, color='steelblue',
                fontweight='bold', transform=ax.get_xaxis_transform())
        ax.text(phase_step + (x_max - phase_step) * 0.5, y_max, 'How2Sign',
                ha='center', va='top', fontsize=9, color='darkorange',
                fontweight='bold', transform=ax.get_xaxis_transform())
    ax.set_title(f"Per-tier LR schedule  —  {CONFIG['run_label']}")
    ax.grid(True, alpha=0.25, which='both'); ax.legend(loc='best', fontsize=9)
    _save(fig, '03_lr_schedule.png')


def plot_grad_norms(train_rows, phase_step):
    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    for tier, color in [('grad_norm_t1', 'C0'), ('grad_norm_t2', 'C1'),
                        ('grad_norm_t3', 'C2'), ('grad_norm_t4', 'C3'),
                        ('grad_norm',    'k')]:
        xs_k, ys_k = _col(train_rows, tier)
        if not xs_k:
            continue
        ax.plot(xs_k, ys_k, alpha=0.18, linewidth=0.7, color=color)
        ax.plot(xs_k, _smooth(ys_k, CONFIG['smooth_window']),
                label=tier, linewidth=1.4, color=color)
    _draw_phase_line(ax, phase_step)
    ax.set_xlabel('global_step'); ax.set_ylabel('gradient norm')
    ax.set_yscale('log')
    ax.set_title(f"Per-tier gradient norms (log, smoothed)  —  {CONFIG['run_label']}")
    ax.grid(True, alpha=0.25, which='both'); ax.legend(loc='best', fontsize=9)
    _save(fig, '04_grad_norms.png')


# ═════════════════════════════════════════════════════════════════════════════
#  VAL-SIDE PLOTS
# ═════════════════════════════════════════════════════════════════════════════
def plot_bleu(val_rows, phase_step):
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['figsize_grid'])
    titles_keys = [
        ('BLEU-1', 'bleu1_how2sign', 'bleu1_openasl'),
        ('BLEU-2', 'bleu2_how2sign', 'bleu2_openasl'),
        ('BLEU-4', 'bleu4_how2sign', 'bleu4_openasl'),
        ('ROUGE-L', 'rouge_l_how2sign', 'rouge_l_openasl'),
    ]
    for ax, (title, k_h2s, k_oasl) in zip(axes.flat, titles_keys):
        # how2sign: all points where value > 0
        xs_h, ys_h = _col(val_rows, k_h2s)
        h2s_pts = [(x, y) for x, y in zip(xs_h, ys_h) if y > 0]
        h2s_steps = {x for x, _ in h2s_pts}
        if h2s_pts:
            xs2, ys2 = zip(*h2s_pts)
            ax.plot(xs2, ys2, label='how2sign', linewidth=1.5, marker='o',
                    markersize=2.8, color='C0')
        # openasl: only show at steps where how2sign has no data yet
        xs_o, ys_o = _col(val_rows, k_oasl)
        oasl_pts = [(x, y) for x, y in zip(xs_o, ys_o) if y > 0 and x not in h2s_steps]
        if oasl_pts:
            xs2, ys2 = zip(*oasl_pts)
            ax.plot(xs2, ys2, label='openasl', linewidth=1.5, marker='o',
                    markersize=2.8, color='C1')
        _draw_phase_line(ax, phase_step)
        ax.set_title(title)
        ax.set_xlabel('global_step')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=8)
    fig.suptitle(f"Generation metrics (per source)  —  {CONFIG['run_label']}")
    fig.tight_layout()
    _save(fig, '05_gen_metrics_bleu_rouge.png')


def plot_raw_train_loss(train_rows, phase_step):
    """Raw (unsmoothed) training loss only — no smoothing applied at all."""
    xs_t, ys_t = _col(train_rows, 'train_loss')
    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    if xs_t:
        ax.plot(xs_t, ys_t, label='train_loss (raw)', linewidth=0.8, color='C0', alpha=0.85)
    _draw_unfreeze_line(ax)
    if phase_step is not None:
        x_min, x_max = ax.get_xlim()
        ax.axvspan(x_min, phase_step, alpha=0.06, color='steelblue', zorder=0)
        ax.axvspan(phase_step, x_max, alpha=0.06, color='darkorange', zorder=0)
        ax.axvline(phase_step, color=CONFIG['phase_line_color'],
                   alpha=0.85, linestyle='--', linewidth=0.8,
                   label=f'phase 1→2  (step {int(phase_step)})')
        ax.text(phase_step * 0.5, 1.0, 'OpenASL',
                ha='center', va='top', fontsize=9, color='steelblue',
                fontweight='bold', transform=ax.get_xaxis_transform())
        ax.text(phase_step + (x_max - phase_step) * 0.5, 1.0, 'How2Sign',
                ha='center', va='top', fontsize=9, color='darkorange',
                fontweight='bold', transform=ax.get_xaxis_transform())
    ax.set_xlabel('global_step')
    ax.set_ylabel('cross-entropy loss')
    ax.set_title(f"Train loss (raw, unsmoothed)  —  {CONFIG['run_label']}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    _save(fig, '07_train_loss_raw.png')


def plot_val_loss_only(val_rows, phase_step):
    """Validation loss on its own plot."""
    xs_v, ys_v = _col(val_rows, 'val_loss')
    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    if xs_v:
        ax.plot(xs_v, ys_v, label='val_loss', linewidth=1.8, color='C1',
                marker='o', markersize=3)
    if phase_step is not None:
        x_min, x_max = ax.get_xlim()
        ax.axvspan(x_min, phase_step, alpha=0.06, color='steelblue', zorder=0)
        ax.axvspan(phase_step, x_max, alpha=0.06, color='darkorange', zorder=0)
        ax.axvline(phase_step, color=CONFIG['phase_line_color'],
                   alpha=0.85, linestyle='--', linewidth=0.8,
                   label=f'phase 1→2  (step {int(phase_step)})')
        ax.text(phase_step * 0.5, 1.0, 'OpenASL',
                ha='center', va='top', fontsize=9, color='steelblue',
                fontweight='bold', transform=ax.get_xaxis_transform())
        ax.text(phase_step + (x_max - phase_step) * 0.5, 1.0, 'How2Sign',
                ha='center', va='top', fontsize=9, color='darkorange',
                fontweight='bold', transform=ax.get_xaxis_transform())
    ax.set_xlabel('global_step')
    ax.set_ylabel('cross-entropy loss')
    ax.set_title(f"Validation loss  —  {CONFIG['run_label']}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    _save(fig, '08_val_loss.png')


def plot_meteor_wer_distinct(val_rows, phase_step):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # METEOR
    for k, label, color in [('meteor', 'combined', 'k'),
                            ('meteor_how2sign', 'how2sign', 'C0'),
                            ('meteor_openasl',  'openasl', 'C1')]:
        xs, ys = _col(val_rows, k)
        xs2, ys2 = zip(*[(x, y) for x, y in zip(xs, ys) if y > 0]) if any(y > 0 for y in ys) else ([], [])
        if xs2:
            axes[0].plot(xs2, ys2, label=label, linewidth=1.5, marker='o', markersize=2.8, color=color)
    _draw_phase_line(axes[0], phase_step)
    axes[0].set_title('METEOR'); axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc='best', fontsize=8); axes[0].set_xlabel('global_step')

    # WER
    for k, label, color in [('wer', 'combined', 'k'),
                            ('wer_how2sign', 'how2sign', 'C0'),
                            ('wer_openasl',  'openasl', 'C1')]:
        xs, ys = _col(val_rows, k)
        xs2, ys2 = zip(*[(x, y) for x, y in zip(xs, ys) if y > 0]) if any(y > 0 for y in ys) else ([], [])
        if xs2:
            axes[1].plot(xs2, ys2, label=label, linewidth=1.5, marker='o', markersize=2.8, color=color)
    _draw_phase_line(axes[1], phase_step)
    axes[1].set_title('WER (%)'); axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc='best', fontsize=8); axes[1].set_xlabel('global_step')

    # distinct-2 (diversity)
    xs, ys = _col(val_rows, 'distinct_2')
    if xs:
        axes[2].plot(xs, ys, label='distinct-2', linewidth=1.6, marker='o',
                     markersize=2.8, color='C2')
    _draw_phase_line(axes[2], phase_step)
    axes[2].set_title('distinct-2 (diversity ↑)'); axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc='best', fontsize=8); axes[2].set_xlabel('global_step')

    fig.suptitle(f"METEOR / WER / Diversity  —  {CONFIG['run_label']}")
    _save(fig, '06_meteor_wer_distinct.png')


# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY STATS
# ═════════════════════════════════════════════════════════════════════════════
def _argbest(xs, ys, mode='max'):
    if not xs:
        return None, None
    if mode == 'max':
        i = max(range(len(ys)), key=lambda k: ys[k])
    else:
        i = min(range(len(ys)), key=lambda k: ys[k])
    return xs[i], ys[i]


def _plateau_check(xs, ys, peak_value, tail_frac, tol_rel) -> bool:
    """True if the last tail_frac fraction of values stayed within tol_rel of peak."""
    if not xs or peak_value is None or peak_value == 0:
        return False
    n = len(xs)
    cut = max(1, int(n * (1 - tail_frac)))
    tail = ys[cut:]
    if not tail:
        return False
    rel_dev = max(abs(v - peak_value) / abs(peak_value) for v in tail)
    return rel_dev <= tol_rel


def compute_summary(train_rows, val_rows, phase_step) -> dict:
    out = {
        'run_label':    CONFIG['run_label'],
        'phase_transition_step': phase_step,
        'train': {},
        'val':   {'overall': {}, 'how2sign': {}, 'openasl': {}},
    }
    # ── total training time ─────────────────────────────────────────────
    # elapsed_sec resets at the phase transition, so sum the max of each phase.
    xs_el, ys_el = _col(train_rows, 'elapsed_sec')
    if xs_el and phase_step is not None:
        phase1_elapsed = max((y for x, y in zip(xs_el, ys_el) if x <= phase_step), default=0.0)
        phase2_elapsed = max((y for x, y in zip(xs_el, ys_el) if x > phase_step),  default=0.0)
        total_sec = phase1_elapsed + phase2_elapsed
    elif xs_el:
        total_sec = max(ys_el)
    else:
        total_sec = None
    if total_sec is not None:
        d, rem = divmod(int(total_sec), 86400)
        h, rem = divmod(rem, 3600)
        m, s   = divmod(rem, 60)
        out['total_training_time_sec'] = total_sec
        out['total_training_time_str'] = (
            f"{d}d {h:02d}h {m:02d}m {s:02d}s" if d else f"{h}h {m:02d}m {s:02d}s"
        )

    # ── train side ──────────────────────────────────────────────────────
    xs_t, ys_t = _col(train_rows, 'train_loss')
    if xs_t:
        out['train']['final_step'] = int(xs_t[-1])
        out['train']['final_loss'] = ys_t[-1]
        out['train']['min_loss']    = min(ys_t)
        out['train']['min_loss_step'] = int(xs_t[ys_t.index(min(ys_t))])
        out['train']['n_train_rows'] = len(xs_t)

    # ── val side, overall + per-source ──────────────────────────────────
    metric_groups = {
        'overall':   ['val_loss', 'val_ppl', 'distinct_2',
                      'bleu1', 'bleu2', 'bleu4', 'rouge_l', 'meteor', 'wer'],
        'how2sign':  ['bleu1_how2sign', 'bleu2_how2sign', 'bleu4_how2sign',
                      'rouge_l_how2sign', 'meteor_how2sign', 'wer_how2sign'],
        'openasl':   ['bleu1_openasl', 'bleu2_openasl', 'bleu4_openasl',
                      'rouge_l_openasl', 'meteor_openasl', 'wer_openasl'],
    }
    minimize = {'val_loss', 'val_ppl', 'wer', 'wer_how2sign', 'wer_openasl'}

    for src, keys in metric_groups.items():
        for k in keys:
            xs, ys = _col(val_rows, k)
            # Drop zero stubs for source-specific metrics where 0 means "not evaluated".
            if src != 'overall':
                paired = [(x, y) for x, y in zip(xs, ys) if y > 0]
                if paired:
                    xs, ys = zip(*paired); xs, ys = list(xs), list(ys)
                else:
                    xs, ys = [], []
            if not xs:
                continue
            mode = 'min' if k in minimize else 'max'
            best_step, best_val = _argbest(xs, ys, mode=mode)
            final_step, final_val = xs[-1], ys[-1]
            entry = {
                'best_value': best_val, 'best_step': int(best_step),
                'final_value': final_val, 'final_step': int(final_step),
                'n_eval_rows': len(xs),
                'plateaued_in_tail': _plateau_check(
                    xs, ys, best_val,
                    CONFIG['plateau_tail_frac'], CONFIG['plateau_tol_rel'],
                ),
            }
            out['val'][src][k] = entry

    # Overfitting gap: train_loss - val_loss at the latest val step.
    xs_v, ys_v = _col(val_rows, 'val_loss')
    if xs_t and xs_v:
        # Find the train_loss closest in step to the last val step.
        last_v_step = xs_v[-1]
        nearest_i = min(range(len(xs_t)), key=lambda i: abs(xs_t[i] - last_v_step))
        train_at = ys_t[nearest_i]
        val_at = ys_v[-1]
        out['overfitting_gap_at_end'] = {
            'train_loss': train_at, 'val_loss': val_at,
            'gap_val_minus_train': val_at - train_at,
            'step': int(last_v_step),
        }
    return out


def write_summary_files(summary: dict, n_total_eval_rows: int):
    out_dir = CONFIG['output_dir']
    (out_dir / 'metrics_summary.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(f"  ✓ {(out_dir / 'metrics_summary.json').relative_to(REPO_ROOT)}")

    # Human-readable
    import datetime
    lines = []
    lines.append(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"=== Training run summary — {summary['run_label']} ===\n")
    lines.append(f"Phase 1→2 transition step : {summary['phase_transition_step']}")
    if summary.get('total_training_time_str'):
        lines.append(f"Total training time       : {summary['total_training_time_str']}\n")
    else:
        lines.append("")
    if 'train' in summary and summary['train']:
        t = summary['train']
        lines.append("Train:")
        lines.append(f"  rows logged       : {t.get('n_train_rows', 0)}")
        lines.append(f"  final step        : {t.get('final_step')}")
        lines.append(f"  final train_loss  : {t.get('final_loss', float('nan')):.4f}")
        lines.append(f"  min train_loss    : {t.get('min_loss', float('nan')):.4f} "
                     f"(at step {t.get('min_loss_step')})\n")
    if 'overfitting_gap_at_end' in summary:
        g = summary['overfitting_gap_at_end']
        lines.append("Overfitting gap (at last val):")
        lines.append(f"  train_loss        : {g['train_loss']:.4f}")
        lines.append(f"  val_loss          : {g['val_loss']:.4f}")
        lines.append(f"  gap (val - train) : {g['gap_val_minus_train']:.4f}\n")
    for src in ('overall', 'how2sign', 'openasl'):
        block = summary['val'].get(src, {})
        if not block:
            continue
        lines.append(f"Val [{src}]  ({n_total_eval_rows} eval rows total):")
        for k, e in block.items():
            arrow = '↓' if k.startswith('val_') or 'wer' in k else '↑'
            lines.append(
                f"  {k:<20} best={e['best_value']:>9.4f} (step {e['best_step']:>5})  "
                f"final={e['final_value']:>9.4f} (step {e['final_step']:>5})  "
                f"{arrow}  plateau_tail={'YES' if e['plateaued_in_tail'] else 'no'}"
            )
        lines.append("")
    (out_dir / 'summary.txt').write_text("\n".join(lines), encoding='utf-8')
    print(f"  ✓ {(out_dir / 'summary.txt').relative_to(REPO_ROOT)}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("Reading CSV logs...")
    train_rows = _read_csv(CONFIG['train_csv'])
    val_rows   = _read_csv(CONFIG['val_csv'])
    print(f"  ✓ {len(train_rows)} train rows, {len(val_rows)} val rows")

    CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

    phase_step = CONFIG['dataset_phase_step']
    print(f"  Dataset phase transition (OpenASL → How2Sign) at global_step = {phase_step}")

    print("\nGenerating plots...")
    plot_train_loss(train_rows, val_rows, phase_step)
    plot_train_ppl(train_rows, val_rows, phase_step)
    plot_lr_schedules(train_rows, phase_step)
    plot_grad_norms(train_rows, phase_step)
    plot_bleu(val_rows, phase_step)
    plot_meteor_wer_distinct(val_rows, phase_step)
    plot_raw_train_loss(train_rows, phase_step)
    plot_val_loss_only(val_rows, phase_step)

    print("\nComputing summary stats...")
    summary = compute_summary(train_rows, val_rows, phase_step)
    write_summary_files(summary, n_total_eval_rows=len(val_rows))

    print(f"\nAll outputs written to {CONFIG['output_dir'].relative_to(REPO_ROOT)}\n")


if __name__ == '__main__':
    main()
