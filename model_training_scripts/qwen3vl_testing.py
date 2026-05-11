"""
Comprehensive test-set evaluation for the Qwen3-VL sign-language model.

Loads a single trained checkpoint and evaluates it on the full test set
(How2Sign + OpenASL combined), reporting a wide spread of generation
quality metrics, diversity diagnostics, and length statistics. Both the
overall numbers and a per-source breakdown are written to disk along with
per-example hypotheses for qualitative analysis.

USAGE
─────
  Local : python model_training_scripts/qwen3vl_testing.py
  Modal : modal run  model_training_scripts/qwen3vl_testing.py

All knobs live in TEST_CONFIG below. The training-script CONFIG is reused
for everything that does not change at test time (model name, LoRA shape,
data paths, etc.) — TEST_CONFIG only overrides what differs.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORT FROM TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
# All shared infrastructure (dataset, collator, validate, metric helpers, the
# CONFIG dict itself, and the Modal app skeleton's path overrides) is reused.
# This guarantees that the test pipeline matches the training pipeline byte-for-byte
# in terms of preprocessing, prompt format, and generation hyperparameters.

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import qwen3vl_training as t  # noqa: E402

CONFIG = t.CONFIG
SignLanguageQwen3VLDataset = t.SignLanguageQwen3VLDataset
Qwen3VLCollator = t.Qwen3VLCollator
validate = t.validate
compute_bleu = t.compute_bleu
compute_wer = t.compute_wer
compute_meteor = t.compute_meteor
compute_rouge_l = t.compute_rouge_l
compute_distinct_n = t.compute_distinct_n
device = t.device

# sacrebleu is already a dep of the training pipeline; chrF lives in the same package.
import sacrebleu  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
from peft import PeftModel  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
#  TEST CONFIG  (overrides + test-specific knobs)
# ═════════════════════════════════════════════════════════════════════════════
TEST_CONFIG = {
    # ── Checkpoint to evaluate ──────────────────────────────────────────────
    # 'best_model'       → load <checkpoint_dir>/best_model
    # int (e.g. 3960)    → load <checkpoint_dir>/checkpoint_step_3960
    # None               → load latest checkpoint in the dir
    'checkpoint_dir':       CONFIG['checkpoint_dir'],   # follow training paths
    'checkpoint_step':      5700,                         # set this to the step you want
    'use_best_model':       False,

    # ── Decoding ────────────────────────────────────────────────────────────
    'beam_size':                4,        # paper-grade default
    'length_penalty':           1.0,
    'no_repeat_ngram_size':     3,
    'repetition_penalty':       1.3,
    'max_new_tokens':           70,

    # ── Eval coverage ───────────────────────────────────────────────────────
    'compute_loss':             True,    # teacher-forced loss/PPL on the test set
    'gen_batch_size':           6,
    'loss_batch_size':          6,
    # Set to True to skip the loss pass and load loss/PPL from the saved
    # _loss_checkpoint.json (written at the end of a previous loss pass).
    # Useful when restarting after a generation-pass crash — saves re-running
    # the full loss pass. If True but the file is missing, a warning is printed
    # and the loss pass runs normally from scratch.
    'resume_from_loss_checkpoint': False,

    # ── What to evaluate ────────────────────────────────────────────────────
    # Sources to include — both is the recommended default.
    'eval_sources':             ('how2sign', 'openasl'),

    # ── Output ──────────────────────────────────────────────────────────────
    # Folder name follows the same suffix-on-folder pattern as training.
    # Locally: <repo>/test_results_qwen3vl/  (or under Kaggle/Modal mounts).
    'results_dir_name':         'test_results_qwen3vl',
    'results_subfolder':        None,    # auto: f"step_{step}_beam{beam}"  if None
    # Files written inside results_dir_name/results_subfolder/
    'aggregate_json':           'metrics.json',
    'per_example_csv':          'predictions.csv',
    'summary_txt':              'summary.txt',

    # ── Misc ────────────────────────────────────────────────────────────────
    'num_workers':              6,
    'prefetch_factor':          6,
    'pin_memory':               True,
    'print_first_n_samples':    20,    # echo this many ref/hyp pairs to stdout

    # ── Modal resources (override training CONFIG values) ────────────────────
    'modal_gpu':     "A100-80GB",       # e.g. 'A100-80GB', 'H100', 'T4'
    'modal_cpu':     10,
    'modal_memory':  70,    # GB
    'modal_timeout': CONFIG['modal_timeout'],   # seconds
    'modal_retries': CONFIG.get('modal_retries', 0),
}


# ═════════════════════════════════════════════════════════════════════════════
#  ADDITIONAL METRICS  (beyond what validate() already returns)
# ═════════════════════════════════════════════════════════════════════════════
def compute_chrf(references, hypotheses, beta=2):
    """chrF / chrF++ — character-n-gram F-score from sacrebleu."""
    if not references or not hypotheses:
        return 0.0
    return sacrebleu.CHRF(word_order=0, beta=beta).corpus_score(hypotheses, [references]).score


def compute_chrf_pp(references, hypotheses):
    """chrF++ uses word_order=2 — adds word-bigram contribution to chrF."""
    if not references or not hypotheses:
        return 0.0
    return sacrebleu.CHRF(word_order=2).corpus_score(hypotheses, [references]).score


def compute_length_stats(references, hypotheses):
    """Word-level length stats. Useful for spotting brevity penalty / verbosity."""
    if not references or not hypotheses:
        return {'mean_ref_len': 0.0, 'mean_hyp_len': 0.0, 'len_ratio': 0.0,
                'median_hyp_len': 0.0}
    ref_lens = [len(r.strip().split()) for r in references]
    hyp_lens = [len(h.strip().split()) for h in hypotheses]
    mean_ref = sum(ref_lens) / len(ref_lens)
    mean_hyp = sum(hyp_lens) / len(hyp_lens)
    sorted_h = sorted(hyp_lens)
    median_hyp = sorted_h[len(sorted_h) // 2]
    return {
        'mean_ref_len':   mean_ref,
        'mean_hyp_len':   mean_hyp,
        'len_ratio':      (mean_hyp / mean_ref) if mean_ref > 0 else 0.0,
        'median_hyp_len': median_hyp,
    }


def all_metrics(references, hypotheses):
    """Compute the full metric panel for a (refs, hyps) pair."""
    if not references:
        return {
            'n':               0,
            'bleu1':           0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0,
            'chrf':            0.0, 'chrf_pp': 0.0,
            'rouge_l':         0.0, 'meteor':  0.0, 'wer':    0.0,
            'distinct_1':      0.0, 'distinct_2': 0.0,
            'mean_ref_len':    0.0, 'mean_hyp_len': 0.0,
            'len_ratio':       0.0, 'median_hyp_len': 0.0,
        }
    out = {
        'n':         len(references),
        'bleu1':     compute_bleu(references, hypotheses, max_n=1),
        'bleu2':     compute_bleu(references, hypotheses, max_n=2),
        'bleu3':     compute_bleu(references, hypotheses, max_n=3),
        'bleu4':     compute_bleu(references, hypotheses, max_n=4),
        'chrf':      compute_chrf(references, hypotheses),
        'chrf_pp':   compute_chrf_pp(references, hypotheses),
        'rouge_l':   compute_rouge_l(references, hypotheses),
        'meteor':    compute_meteor(references, hypotheses),
        'wer':       compute_wer(references, hypotheses),
        'distinct_1': compute_distinct_n(hypotheses, n=1),
        'distinct_2': compute_distinct_n(hypotheses, n=2),
    }
    out.update(compute_length_stats(references, hypotheses))
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════
def _resolve_checkpoint_path() -> Path:
    ckpt_dir = Path(TEST_CONFIG['checkpoint_dir'])
    if TEST_CONFIG['use_best_model']:
        path = ckpt_dir / 'best_model'
    elif TEST_CONFIG['checkpoint_step'] is not None:
        path = ckpt_dir / f"checkpoint_step_{TEST_CONFIG['checkpoint_step']}"
    else:
        candidates = sorted(
            [d for d in ckpt_dir.glob('checkpoint_step_*') if d.is_dir()],
            key=lambda p: int(p.name.split('_')[-1]),
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
        path = candidates[-1]
    if not path.exists():
        existing = sorted(
            [int(d.name.split('_')[-1]) for d in ckpt_dir.glob('checkpoint_step_*') if d.is_dir()]
        )
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nAvailable steps in {ckpt_dir}: {existing}"
        )
    return path


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════
def _load_model_and_processor(ckpt_path: Path):
    """Load base model + LoRA adapter from a saved checkpoint, in eval mode.

    Tier-4 (InfoNCE projection) modules are NOT loaded: they are unused at
    inference time (generation goes through the LM head, not the projection
    head), so loading them would just consume VRAM for no benefit.
    """
    print(f"\n  Loading base model: {CONFIG['model_name']}")
    model_kwargs = {'dtype': torch.bfloat16}
    try:
        import flash_attn  # noqa: F401
        model_kwargs['attn_implementation'] = 'flash_attention_2'
        print("  ✓ Using Flash Attention 2")
    except ImportError:
        model_kwargs['attn_implementation'] = CONFIG.get('attn_implementation', 'sdpa')
        if model_kwargs['attn_implementation'] == 'flash_attention_2':
            model_kwargs['attn_implementation'] = 'sdpa'
        print(f"  ✓ Using {model_kwargs['attn_implementation']}")
    model_kwargs['device_map'] = 'cuda'

    model = AutoModelForImageTextToText.from_pretrained(
        CONFIG['model_name'],
        local_files_only=CONFIG.get('is_kaggle', False),
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        CONFIG['model_name'],
        local_files_only=CONFIG.get('is_kaggle', False),
    )
    processor.tokenizer.padding_side = 'left'

    # Apply Liger fused kernels (same patching as training — must run before LoRA wrap).
    # RMSNorm → LigerRMSNorm (113 modules), LayerNorm → LigerLayerNorm (52 modules).
    try:
        from liger_kernel.transformers import LigerRMSNorm, LigerLayerNorm
        _to_rms, _to_ln = [], []
        for _parent in model.modules():
            for _cname, _cmod in _parent.named_children():
                _cls = type(_cmod).__name__
                if _cls == 'Qwen3VLTextRMSNorm':
                    _to_rms.append((_parent, _cname, _cmod))
                elif isinstance(_cmod, torch.nn.LayerNorm) and not isinstance(_cmod, LigerLayerNorm):
                    _to_ln.append((_parent, _cname, _cmod))
        for _parent, _name, _orig in _to_rms:
            _eps = getattr(_orig, 'variance_epsilon', getattr(_orig, 'eps', 1e-6))
            _new = LigerRMSNorm(_orig.weight.shape[0], eps=_eps)
            _new.weight = _orig.weight
            setattr(_parent, _name, _new)
        for _parent, _name, _orig in _to_ln:
            _new = LigerLayerNorm(_orig.normalized_shape, eps=_orig.eps)
            if _orig.elementwise_affine:
                _new.weight = _orig.weight
                _new.bias   = _orig.bias
            setattr(_parent, _name, _new)
        _audit = {'LigerRMSNorm': 0, 'LigerLayerNorm': 0}
        for _, _m in model.named_modules():
            _t = type(_m).__name__
            if _t in _audit:
                _audit[_t] += 1
        _ok_rms = '✓' if _audit['LigerRMSNorm']  == 113 else '⚠️ '
        _ok_ln  = '✓' if _audit['LigerLayerNorm'] == 52  else '⚠️ '
        print(f"  {_ok_rms} Liger RMSNorm  : {_audit['LigerRMSNorm']:>4} patched  (expected 113)")
        print(f"  {_ok_ln} Liger LayerNorm: {_audit['LigerLayerNorm']:>4} patched  (expected  52)")
        del _to_rms, _to_ln, _audit, _ok_rms, _ok_ln
    except ImportError:
        print("  ⚠️  liger-kernel not installed — skipping")
    except Exception as _liger_err:
        print(f"  ⚠️  Liger Kernel failed: {_liger_err} — skipping")

    print(f"  Loading LoRA adapter from: {ckpt_path}")
    model = PeftModel.from_pretrained(model, ckpt_path / 'adapter', is_trainable=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if hasattr(model, 'config'):
        model.config.use_cache = True

    return model, processor


# ═════════════════════════════════════════════════════════════════════════════
#  TEST DATASET / LOADER
# ═════════════════════════════════════════════════════════════════════════════
def _build_test_dataset_and_loader(processor):
    """Build the combined test dataset (how2sign + openasl) and its DataLoader."""
    sources = tuple(TEST_CONFIG['eval_sources'])
    print(f"\n  Building test dataset (sources: {sources})...")

    bbox_path = (CONFIG.get('bbox_csv_test') or CONFIG.get('bbox_csv_val')
                 or CONFIG.get('bbox_csv'))
    landmark_path = CONFIG.get('landmark_parquet_test') or CONFIG.get('landmark_parquet_val')

    # If both sources are wanted, build one dataset with no source filter.
    # Otherwise filter to the single requested source.
    src_filter = None if set(sources) == {'how2sign', 'openasl'} else sources[0]

    test_dataset = SignLanguageQwen3VLDataset(
        CONFIG['data_test_tsv'],
        sep=CONFIG['tsv_sep'],
        bbox_csv_path=bbox_path if CONFIG.get('use_signer_crop') else None,
        landmark_parquet_path=landmark_path if CONFIG.get('use_landmark_overlay') else None,
        tokenizer=None,
        max_text_tokens=CONFIG['max_text_tokens'],
        source_filter=src_filter,
    )
    print(f"  ✓ Test dataset size: {len(test_dataset):,} clips")

    val_collator = Qwen3VLCollator(processor, CONFIG, is_training=False)

    loader = DataLoader(
        test_dataset,
        batch_size=TEST_CONFIG['loss_batch_size'],
        shuffle=False,
        collate_fn=val_collator,
        num_workers=TEST_CONFIG['num_workers'],
        prefetch_factor=TEST_CONFIG['prefetch_factor'] if TEST_CONFIG['num_workers'] > 0 else None,
        pin_memory=TEST_CONFIG['pin_memory'],
        persistent_workers=False,
    )
    return test_dataset, loader, val_collator


# ═════════════════════════════════════════════════════════════════════════════
#  RESULTS WRITING
# ═════════════════════════════════════════════════════════════════════════════
def _resolve_results_dir() -> Path:
    """Output dir lives inside saved_metrics/, mirroring where train/val logs go."""
    base = Path(CONFIG['train_log_file']).parent / TEST_CONFIG['results_dir_name']
    sub = TEST_CONFIG['results_subfolder']
    if sub is None:
        if TEST_CONFIG['use_best_model']:
            tag = 'best_model'
        elif TEST_CONFIG['checkpoint_step'] is not None:
            tag = f"step_{TEST_CONFIG['checkpoint_step']}"
        else:
            tag = 'latest'
        sub = f"{tag}_beam{TEST_CONFIG['beam_size']}"
    out_dir = base / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_outputs(out_dir: Path, results: dict, all_pairs: list[tuple[str, str, str]],
                   loss_value: float, ppl_value: float, ckpt_path: Path,
                   loss_elapsed_sec: float = float('nan'),
                   gen_elapsed_sec: float = float('nan'),
                   total_elapsed_sec: float = float('nan')):
    """Write all three output files. Each file is wrapped in its own try/except
    so a bug in one never prevents the others from being saved."""

    def _fmt_min(s: float) -> str:
        return f"{s/60:.1f} min" if not math.isnan(s) else "n/a"

    n_total      = sum(v.get('n', 0) for v in results['per_source'].values())
    loss_batches = math.ceil(n_total / TEST_CONFIG['loss_batch_size']) if TEST_CONFIG['compute_loss'] else 0
    gen_batches  = math.ceil(n_total / TEST_CONFIG['gen_batch_size'])

    # ── Per-example CSV ──────────────────────────────────────────────────────
    csv_path = out_dir / TEST_CONFIG['per_example_csv']
    try:
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['source', 'reference', 'hypothesis'])
            for ref, hyp, src in all_pairs:
                w.writerow([src, ref, hyp])
        print(f"  ✓ Per-example predictions → {csv_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to write predictions CSV: {e}")

    # ── Aggregate JSON ───────────────────────────────────────────────────────
    json_path = out_dir / TEST_CONFIG['aggregate_json']
    serializable = {
        'checkpoint':    str(ckpt_path),
        'beam_size':     TEST_CONFIG['beam_size'],
        'compute_loss':  TEST_CONFIG['compute_loss'],
        'test_loss':     loss_value,
        'test_ppl':      ppl_value,
        'overall':       results['overall'],
        'per_source':    results['per_source'],
        'decoding': {
            'beam_size':            TEST_CONFIG['beam_size'],
            'length_penalty':       TEST_CONFIG['length_penalty'],
            'no_repeat_ngram_size': TEST_CONFIG['no_repeat_ngram_size'],
            'repetition_penalty':   TEST_CONFIG['repetition_penalty'],
            'max_new_tokens':       TEST_CONFIG['max_new_tokens'],
        },
        'timing_minutes': {
            'loss_pass':  round(loss_elapsed_sec  / 60, 2),
            'gen_pass':   round(gen_elapsed_sec   / 60, 2),
            'total':      round(total_elapsed_sec / 60, 2),
        },
    }
    try:
        json_path.write_text(json.dumps(serializable, indent=2), encoding='utf-8')
        print(f"  ✓ Aggregate metrics → {json_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to write metrics JSON: {e}")
        # Last-resort: dump raw to stdout so nothing is lost.
        print("  ⚠️  Dumping metrics to stdout as fallback:")
        print(json.dumps(serializable, indent=2))

    # ── Human-readable summary ───────────────────────────────────────────────
    txt_path = out_dir / TEST_CONFIG['summary_txt']
    try:
        with txt_path.open('w', encoding='utf-8') as f:
            f.write("═" * 80 + "\n")
            f.write("  QWEN3-VL TEST-SET EVALUATION — SUMMARY\n")
            f.write("═" * 80 + "\n\n")

            f.write("── Checkpoint ─────────────────────────────────────────────────────────────\n")
            f.write(f"  Path         : {ckpt_path}\n")
            f.write(f"  Model        : {CONFIG['model_name']}\n\n")

            f.write("── Dataset ────────────────────────────────────────────────────────────────\n")
            f.write(f"  Eval sources : {', '.join(TEST_CONFIG['eval_sources'])}\n")
            f.write(f"  Total clips  : {n_total:,}\n")
            for src, m in results['per_source'].items():
                f.write(f"    {src:<12}: {m.get('n', 0):,} clips\n")
            f.write(f"  Data TSV     : {CONFIG['data_test_tsv']}\n\n")

            f.write("── Loss Pass ──────────────────────────────────────────────────────────────\n")
            if TEST_CONFIG['compute_loss']:
                f.write(f"  Computed     : yes\n")
                f.write(f"  Batch size   : {TEST_CONFIG['loss_batch_size']}\n")
                f.write(f"  Batches      : {loss_batches:,}\n")
                f.write(f"  Test loss    : {loss_value:.4f}\n")
                f.write(f"  Perplexity   : {ppl_value:.4f}\n")
                f.write(f"  Time         : {_fmt_min(loss_elapsed_sec)}\n")
            else:
                f.write(f"  Computed     : no  (compute_loss=False)\n")
            f.write("\n")

            f.write("── Generation Pass ────────────────────────────────────────────────────────\n")
            f.write(f"  Batch size         : {TEST_CONFIG['gen_batch_size']}\n")
            f.write(f"  Batches            : {gen_batches:,}\n")
            f.write(f"  Beam size          : {TEST_CONFIG['beam_size']}\n")
            f.write(f"  Length penalty     : {TEST_CONFIG['length_penalty']}\n")
            f.write(f"  No-repeat ngram    : {TEST_CONFIG['no_repeat_ngram_size']}\n")
            f.write(f"  Repetition penalty : {TEST_CONFIG['repetition_penalty']}\n")
            f.write(f"  Max new tokens     : {TEST_CONFIG['max_new_tokens']}\n")
            f.write(f"  Time               : {_fmt_min(gen_elapsed_sec)}\n\n")

            f.write("── Video / Vision Config ──────────────────────────────────────────────────\n")
            f.write(f"  FPS              : {CONFIG['video_fps']}\n")
            f.write(f"  Min pixels       : {CONFIG['video_min_pixels']}\n")
            f.write(f"  Max pixels       : {CONFIG['video_max_pixels']}\n")
            f.write(f"  Signer crop      : {CONFIG.get('use_signer_crop', False)}\n")
            f.write(f"  Landmark overlay : {CONFIG.get('use_landmark_overlay', False)}\n\n")

            f.write("── Timing ─────────────────────────────────────────────────────────────────\n")
            f.write(f"  Loss pass : {_fmt_min(loss_elapsed_sec)}\n")
            f.write(f"  Gen pass  : {_fmt_min(gen_elapsed_sec)}\n")
            f.write(f"  Total     : {_fmt_min(total_elapsed_sec)}\n\n")

            f.write("── Metrics ────────────────────────────────────────────────────────────────\n")
            _print_metric_table(results, file=f)
            f.write("\n")

        print(f"  ✓ Summary → {txt_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to write summary TXT: {e}")
        # Last-resort: print the metric table to stdout so nothing is lost.
        print("  ⚠️  Printing summary to stdout as fallback:")
        print(f"  Loss={loss_value:.4f}  PPL={ppl_value:.4f}  "
              f"loss_time={_fmt_min(loss_elapsed_sec)}  "
              f"gen_time={_fmt_min(gen_elapsed_sec)}  "
              f"total={_fmt_min(total_elapsed_sec)}")
        _print_metric_table(results)


# ═════════════════════════════════════════════════════════════════════════════
#  PRINTING
# ═════════════════════════════════════════════════════════════════════════════
_METRIC_ORDER = [
    ('n',              'count'),
    ('bleu1',          'BLEU-1'),
    ('bleu2',          'BLEU-2'),
    ('bleu3',          'BLEU-3'),
    ('bleu4',          'BLEU-4'),
    ('chrf',           'chrF'),
    ('chrf_pp',        'chrF++'),
    ('rouge_l',        'ROUGE-L'),
    ('meteor',         'METEOR'),
    ('wer',            'WER%'),
    ('distinct_1',     'distinct-1'),
    ('distinct_2',     'distinct-2'),
    ('mean_ref_len',   'mean ref len'),
    ('mean_hyp_len',   'mean hyp len'),
    ('len_ratio',      'len ratio'),
]


def _print_metric_table(results: dict, file=sys.stdout):
    overall = results['overall']
    per_src = results['per_source']
    sources = list(per_src.keys())
    header = f"  {'metric':<15}  {'overall':>10}" + "".join(f"  {s:>10}" for s in sources)
    file.write(header + "\n")
    file.write("  " + "─" * (len(header) - 2) + "\n")
    for key, label in _METRIC_ORDER:
        cells = [f"{overall.get(key, 0):>10.4f}" if isinstance(overall.get(key, 0), float)
                 else f"{overall.get(key, 0):>10}"]
        for s in sources:
            v = per_src[s].get(key, 0)
            cells.append(f"{v:>10.4f}" if isinstance(v, float) else f"{v:>10}")
        file.write(f"  {label:<15}  " + "  ".join(cells) + "\n")


# ═════════════════════════════════════════════════════════════════════════════
#  MODAL VOLUME COMMIT HELPER
# ═════════════════════════════════════════════════════════════════════════════
def _commit_modal_volume():
    if CONFIG.get('is_modal'):
        try:
            import modal
            modal.Volume.from_name(CONFIG['modal_output_volume']).commit()
            print(f"  ✓ Committed Modal output volume '{CONFIG['modal_output_volume']}'")
        except Exception as e:
            print(f"  ⚠️  Modal volume commit skipped: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main_test():
    t_start = time.time()
    print("\n" + "═" * 80)
    print("  QWEN3-VL TEST-SET EVALUATION")
    print("═" * 80)

    # ── Resolve checkpoint ──────────────────────────────────────────────────
    ckpt_path = _resolve_checkpoint_path()
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Beam size      : {TEST_CONFIG['beam_size']}")
    print(f"  Compute loss   : {TEST_CONFIG['compute_loss']}")
    print(f"  Eval sources   : {TEST_CONFIG['eval_sources']}")

    # ── Load model ──────────────────────────────────────────────────────────
    model, processor = _load_model_and_processor(ckpt_path)

    # ── Build test data ─────────────────────────────────────────────────────
    test_dataset, test_loader, val_collator = _build_test_dataset_and_loader(processor)
    n_total = len(test_dataset)

    full_eval_batches = math.ceil(n_total / TEST_CONFIG['loss_batch_size'])
    eval_config = dict(CONFIG)
    eval_config.update({
        'max_eval_batches':         full_eval_batches,
        'max_generate_samples':     n_total,
        'val_loss_batch_size':      TEST_CONFIG['loss_batch_size'],
        'val_gen_batch_size':       TEST_CONFIG['gen_batch_size'],
        'val_beam_size':            TEST_CONFIG['beam_size'],
        'val_length_penalty':       TEST_CONFIG['length_penalty'],
        'val_no_repeat_ngram_size': TEST_CONFIG['no_repeat_ngram_size'],
        'val_repetition_penalty':   TEST_CONFIG['repetition_penalty'],
        'val_max_new_tokens':       TEST_CONFIG['max_new_tokens'],
        'num_print_samples':        0,   # we print our own samples below
    })

    # Resolve output dir early so we can write the temp file into it.
    out_dir = _resolve_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_loss_path = out_dir / '_loss_checkpoint.json'

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Loss pass
    # ─────────────────────────────────────────────────────────────────────────
    loss_value        = float('nan')
    ppl_value         = float('nan')
    loss_elapsed_sec  = float('nan')
    gen_elapsed_sec   = float('nan')

    # Determine whether to load from checkpoint or run from scratch.
    _run_loss_pass = True
    if TEST_CONFIG['compute_loss'] and TEST_CONFIG.get('resume_from_loss_checkpoint', False):
        if tmp_loss_path.exists():
            saved = json.loads(tmp_loss_path.read_text(encoding='utf-8'))
            loss_value = float(saved.get('test_loss', float('nan')))
            ppl_value  = float(saved.get('test_ppl',  float('nan')))
            loss_elapsed_sec = float(saved.get('loss_elapsed_sec', float('nan')))
            print(f"\n  [Phase 1] Loaded loss from checkpoint: {tmp_loss_path}")
            print(f"  ✓ Loss: {loss_value:.4f}   PPL: {ppl_value:.4f}"
                  + (f"   (took {loss_elapsed_sec/60:.1f} min originally)"
                     if not math.isnan(loss_elapsed_sec) else ""))
            _run_loss_pass = False
        else:
            print(f"\n  ⚠️  resume_from_loss_checkpoint=True but no checkpoint found at:")
            print(f"       {tmp_loss_path}")
            print(f"  ⚠️  Running loss pass from scratch.")

    if TEST_CONFIG['compute_loss'] and _run_loss_pass:
        print(f"\n  [Phase 1] Loss pass: {n_total:,} clips, "
              f"{full_eval_batches} batches (batch_size={TEST_CONFIG['loss_batch_size']})")
        _t_loss_start = time.time()
        loss_out = validate(
            model=model,
            processor=processor,
            val_loader=test_loader,
            val_dataset=test_dataset,
            config={**eval_config, 'max_generate_samples': 0},  # skip generation
            val_collator=val_collator,
            compute_loss=True,
            beam_size=TEST_CONFIG['beam_size'],
        )
        loss_value = float(loss_out.get('val_loss') or float('nan'))
        ppl_value  = float(loss_out.get('val_ppl')  or float('nan'))
        loss_elapsed_sec = time.time() - _t_loss_start
        print(f"  ✓ Loss: {loss_value:.4f}   PPL: {ppl_value:.4f}")

        # Save to temp file immediately — survives a generation-pass crash.
        # Also store the elapsed time so it can be reloaded on resume.
        try:
            tmp_loss_path.write_text(
                json.dumps({
                    'test_loss':        loss_value,
                    'test_ppl':         ppl_value,
                    'loss_elapsed_sec': loss_elapsed_sec,
                }),
                encoding='utf-8',
            )
            print(f"  ✓ Loss checkpoint saved → {tmp_loss_path}")
        except Exception as e:
            print(f"  ⚠️  Could not save loss checkpoint: {e} — continuing anyway")

        # Commit to Modal volume so it's safe even mid-job.
        _commit_modal_volume()

        # Free loader memory before generation pass.
        del loss_out
        gc.collect()
        torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Generation pass
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n  [Phase 2] Generation pass: {n_total:,} clips, "
          f"beam={TEST_CONFIG['beam_size']}, batch_size={TEST_CONFIG['gen_batch_size']}")
    _t_gen_start = time.time()
    gen_out = validate(
        model=model,
        processor=processor,
        val_loader=None,           # no loss computation in this pass
        val_dataset=test_dataset,
        config=eval_config,
        val_collator=val_collator,
        compute_loss=False,
        beam_size=TEST_CONFIG['beam_size'],
    )
    gen_elapsed_sec = time.time() - _t_gen_start

    try:
        all_pairs = gen_out['all_pairs']   # list of (ref, hyp, src)
    except (KeyError, TypeError) as e:
        print(f"  ⚠️  Could not extract all_pairs from generation output: {e}")
        all_pairs = []
    refs_all  = [r for r, _, _ in all_pairs]
    hyps_all  = [h for _, h, _ in all_pairs]
    srcs_all  = [s for _, _, s in all_pairs]

    # ── Compute the full metric panel (overall + per source) ─────────────────
    print("\n  Computing comprehensive metric panel...")
    try:
        overall = all_metrics(refs_all, hyps_all)
    except Exception as e:
        print(f"  ⚠️  Overall metric computation failed: {e} — using empty metrics")
        overall = all_metrics([], [])
    per_source = {}
    for src in TEST_CONFIG['eval_sources']:
        try:
            idx = [i for i, s in enumerate(srcs_all) if s == src]
            per_source[src] = all_metrics(
                [refs_all[i] for i in idx],
                [hyps_all[i] for i in idx],
            )
        except Exception as e:
            print(f"  ⚠️  Metric computation failed for source '{src}': {e} — using empty metrics")
            per_source[src] = all_metrics([], [])
    results = {'overall': overall, 'per_source': per_source}

    # ── Print samples ─────────────────────────────────────────────────────────
    try:
        n_show = min(TEST_CONFIG['print_first_n_samples'], len(all_pairs))
        if n_show > 0:
            print("\n  Sample predictions (first {} of {}):".format(n_show, len(all_pairs)))
            for ref, hyp, src in all_pairs[:n_show]:
                print(f"    [{src:8s}]  REF  \"{ref}\"")
                print(f"               HYP  \"{hyp}\"")
    except Exception as e:
        print(f"  ⚠️  Could not print sample predictions: {e}")

    # ── Print metrics ─────────────────────────────────────────────────────────
    try:
        print("\n" + "═" * 80)
        print(f"  RESULTS  (test_loss={loss_value:.4f}  ppl={ppl_value:.4f})")
        print("═" * 80)
        _print_metric_table(results)
    except Exception as e:
        print(f"  ⚠️  Could not print metric table: {e}")

    # ── Write final output files ──────────────────────────────────────────────
    total_elapsed_sec = time.time() - t_start
    print(f"\n  Writing outputs to: {out_dir}")
    try:
        _write_outputs(out_dir, results, all_pairs, loss_value, ppl_value, ckpt_path,
                       loss_elapsed_sec=loss_elapsed_sec,
                       gen_elapsed_sec=gen_elapsed_sec,
                       total_elapsed_sec=total_elapsed_sec)
    except Exception as e:
        print(f"  ⚠️  Unexpected error in _write_outputs: {e}")
        traceback.print_exc()

    # Clean up temp loss file — wrapped so a cleanup failure never raises.
    try:
        if tmp_loss_path.exists():
            tmp_loss_path.unlink()
            print(f"  ✓ Temp loss checkpoint removed.")
    except Exception as e:
        print(f"  ⚠️  Could not remove temp loss checkpoint: {e}")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed/60:.1f} min  ({len(all_pairs)} samples processed).")
    print("═" * 80 + "\n")

    # Commit final outputs to Modal volume.
    _commit_modal_volume()

    # Free GPU before exit.
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _safe_main_test():
    try:
        main_test()
    except Exception:
        traceback.print_exc()
        raise


# ═════════════════════════════════════════════════════════════════════════════
#  MODAL ENTRYPOINT  (only active when CONFIG['is_modal'] is True)
# ═════════════════════════════════════════════════════════════════════════════
# We reuse the training script's Modal app definition (image, volumes, secrets,
# resources). We just register a separate function on it for the test job so
# `modal run model_training_scripts/qwen3vl_testing.py` dispatches the test.
if CONFIG.get('is_modal') and hasattr(t, 'app'):
    import modal  # noqa: E402

    _test_image = t._modal_image.add_local_file(
        _THIS_DIR / 'qwen3vl_training.py',
        '/root/qwen3vl_training.py',
    )

    @t.app.function(
        image=_test_image,
        gpu=TEST_CONFIG['modal_gpu'],
        cpu=TEST_CONFIG['modal_cpu'],
        memory=TEST_CONFIG['modal_memory'] * 1024,
        timeout=TEST_CONFIG['modal_timeout'],
        retries=modal.Retries(
            max_retries=TEST_CONFIG['modal_retries'],
            backoff_coefficient=1.0,
            initial_delay=5.0,
        ),
        volumes={
            CONFIG['modal_data_mount']:   t._modal_data_vol,
            CONFIG['modal_video_mount']:  t._modal_video_vol,
            CONFIG['modal_output_mount']: t._modal_output_vol,
        },
        env={'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
        secrets=t._modal_secrets,
    )
    def run_testing_on_modal():
        """Entry point executed inside the Modal container."""
        _safe_main_test()

    @t.app.local_entrypoint()
    def modal_test_entrypoint():
        """Called on your local machine when you `modal run` this file."""
        print(f"  Dispatching test job to Modal ({TEST_CONFIG['modal_gpu']} GPU, "
              f"{TEST_CONFIG['modal_cpu']} CPUs, {TEST_CONFIG['modal_memory']} GB RAM, "
              f"timeout={TEST_CONFIG['modal_timeout'] // 3600}h)...")
        run_testing_on_modal.remote()


# ═════════════════════════════════════════════════════════════════════════════
#  LOCAL ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__' and not CONFIG.get('is_modal'):
    _safe_main_test()
