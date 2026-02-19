#!/usr/bin/env python3
"""
Experiment 6: Few-Shot vs Fine-Tuning

Compare zero/few-shot prompting with base Qwen3-8B against fine-tuned LoRA-MoE
on the text expert test set.

Configurations:
  - 0-shot  (3 runs: seed 42, 43, 44 for zero-shot consistency)
  - 1-shot  (1 run)
  - 3-shot  (1 run)
  - 5-shot  (3 runs: seed 42, 43, 44)
  - LoRA-MoE fine-tuned text expert

Output: outputs/evaluations/experiments/exp6_fewshot_learning/
"""

import sys
import random
import traceback
import argparse
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config.settings import get_path_config
from src.training.data_loader import TextDatasetLoader, split_dataset_for_expert
from src.baselines.zero_shot import ZeroShotGenerator
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp6')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache' / 'few_shot'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp6_fewshot_learning'

# (n_shots, run_ids) — run_id=1 uses seed 42, run_id=2 uses seed 43, run_id=3 uses seed 44
SHOT_CONFIGS = [
    (0, [1, 2, 3]),
    (1, [1]),
    (3, [1]),
    (5, [1, 2, 3]),
]

SEED_MAP = {1: 42, 2: 43, 3: 44}


def _cache_filename(n_shots, run_id):
    return f'{n_shots}shot_text_run{run_id}_predictions.json'


def _sample_examples(train_data, n, seed):
    """Sample n diverse training examples using different source files."""
    random.seed(seed)

    # Try to pick from different source CSV files for diversity
    by_source = {}
    for item in train_data:
        src = item.get('source', 'unknown')
        by_source.setdefault(src, []).append(item)

    selected = []
    sources = list(by_source.keys())
    random.shuffle(sources)

    # Round-robin across sources
    while len(selected) < n:
        for src in sources:
            if len(selected) >= n:
                break
            pool = by_source[src]
            if pool:
                idx = random.randint(0, len(pool) - 1)
                selected.append(pool.pop(idx))

    # Fallback if not enough sources
    remaining = n - len(selected)
    if remaining > 0:
        all_items = [item for items in by_source.values() for item in items]
        selected += random.sample(all_items, min(remaining, len(all_items)))

    examples = [{'input': item['input'], 'output': item['output']} for item in selected[:n]]
    return examples


def run_few_shot(n_shots, run_id, train_data, test_data, generator, args):
    """Run or load cached few-shot inference for one (n_shots, run_id) combo."""
    filename = _cache_filename(n_shots, run_id)
    cached = load_predictions_cache(CACHE_DIR, filename)
    if cached and not args.force_regenerate:
        logger.info(f'{n_shots}-shot run{run_id}: loaded from cache')
        return cached

    seed = SEED_MAP[run_id]
    examples = _sample_examples(train_data, n_shots, seed) if n_shots > 0 else []
    logger.info(f'{n_shots}-shot run{run_id}: generating (seed={seed}, {len(examples)} examples)...')

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    predictions = generator.batch_generate(
        inputs, input_type='text', n_shots=n_shots, examples=examples
    )

    samples = [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]
    save_predictions_cache(
        samples, f'{n_shots}_shot', 'text',
        {'n_shots': n_shots, 'run_id': run_id, 'seed': seed},
        CACHE_DIR, filename
    )
    return load_predictions_cache(CACHE_DIR, filename)


def run_lora_moe(test_data, args):
    """Run or load LoRA-MoE text expert inference (reuse exp1/exp2 cache)."""
    cache_subdir = path_cfg.OUTPUTS_DIR / 'inference_cache' / 'lora_moe'
    filename = 'text_predictions.json'
    cached = load_predictions_cache(cache_subdir, filename)
    if cached and not args.force_regenerate:
        logger.info('LoRA-MoE: loaded from cache')
        return cached

    logger.info('LoRA-MoE: running text expert inference...')
    from src.experts import TextExpert
    expert = TextExpert(lora_path=None, use_4bit=True)
    if not expert.load_model():
        return None

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    try:
        predictions = expert.batch_generate_instruction(inputs, batch_size=4)
    finally:
        expert.unload_model()

    samples = [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]
    save_predictions_cache(samples, 'lora_moe', 'text', {}, cache_subdir, filename)
    return load_predictions_cache(cache_subdir, filename)


def compute_runs_stats(run_metrics):
    """Compute mean and std ROUGE-L across multiple runs."""
    rouge_vals = [
        m.get('generation_quality', {}).get('rougeL', 0)
        for m in run_metrics
    ]
    if not rouge_vals:
        return 0.0, 0.0
    return float(np.mean(rouge_vals)), float(np.std(rouge_vals))


def plot_bar_with_errorbars(shot_summary, lora_rougeL, exp_dir, test_mode=False):
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_shot_labels = []
    means = []
    stds = []

    for n_shots in sorted(shot_summary.keys()):
        mean, std = shot_summary[n_shots]
        label = f'{n_shots}-shot'
        n_shot_labels.append(label)
        means.append(mean)
        stds.append(std)

    n_shot_labels.append('LoRA-MoE')
    means.append(lora_rougeL)
    stds.append(0)

    x = np.arange(len(n_shot_labels))
    colors = ['#ff7f0e'] * (len(n_shot_labels) - 1) + ['#1f77b4']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(n_shot_labels)
    ax.set_ylabel('ROUGE-L')
    title = 'Exp6: Few-Shot vs Fine-Tuned (ROUGE-L)'
    if test_mode:
        title += ' [TEST MODE]'
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color='#ff7f0e', alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color='#1f77b4', alpha=0.85),
        ],
        labels=['Few-Shot (base model)', 'LoRA-MoE (fine-tuned)']
    )
    plt.tight_layout()
    path = plots_dir / 'fewshot_vs_finetuning.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved: {path}')


def run(args):
    logger.info('=' * 80)
    logger.info('Experiment 6: Few-Shot vs Fine-Tuning')
    logger.info('=' * 80)

    logger.info('Loading text dataset...')
    all_data = TextDatasetLoader().load_csv_files()
    train_data, _, test_data = split_dataset_for_expert(all_data, 'text')
    logger.info(f'Train={len(train_data)}, Test={len(test_data)}')

    results = {
        'experiment': 'exp6_fewshot_vs_finetuning',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'shot_configs': {},
        'lora_moe': {},
    }

    # Load zero-shot generator once for all few-shot runs
    generator = ZeroShotGenerator(use_4bit=True)
    if not generator.load_model():
        logger.error('Failed to load base model for few-shot generation')
        return

    try:
        shot_summary = {}
        all_run_metrics = {}

        for n_shots, run_ids in SHOT_CONFIGS:
            logger.info(f'\n=== {n_shots}-shot ({len(run_ids)} runs) ===')
            run_metrics = []

            for run_id in run_ids:
                try:
                    cached = run_few_shot(n_shots, run_id, train_data, test_data, generator, args)
                    if cached is None:
                        continue

                    preds = [s['prediction'] for s in cached['samples']]
                    refs = [s['reference'] for s in cached['samples']]
                    m = compute_all_metrics(preds, refs, use_bertscore=not args.no_bertscore)
                    run_metrics.append(m)

                    q = m.get('generation_quality', {})
                    logger.info(
                        f'{n_shots}-shot run{run_id}: ROUGE-L={q.get("rougeL", 0):.4f}'
                    )
                except Exception as e:
                    logger.error(f'{n_shots}-shot run{run_id} failed: {e}')
                    logger.error(traceback.format_exc())

            mean_rougeL, std_rougeL = compute_runs_stats(run_metrics)
            shot_summary[n_shots] = (mean_rougeL, std_rougeL)
            all_run_metrics[n_shots] = run_metrics

            per_run_results = []
            for i, m in enumerate(run_metrics):
                q = m.get('generation_quality', {})
                per_run_results.append({
                    'run_id': run_ids[i] if i < len(run_ids) else i + 1,
                    'generation_quality': q,
                    'binary_classification': m.get('binary_classification', {}),
                })
            results['shot_configs'][str(n_shots)] = {
                'n_shots': n_shots,
                'runs': per_run_results,
                'mean_rougeL': mean_rougeL,
                'std_rougeL': std_rougeL,
            }
            logger.info(f'{n_shots}-shot: mean ROUGE-L={mean_rougeL:.4f} (std={std_rougeL:.4f})')

    finally:
        generator.unload_model()

    # LoRA-MoE baseline
    logger.info('\n=== LoRA-MoE (fine-tuned) ===')
    try:
        cached = run_lora_moe(test_data, args)
        if cached:
            preds = [s['prediction'] for s in cached['samples']]
            refs = [s['reference'] for s in cached['samples']]
            m = compute_all_metrics(preds, refs, use_bertscore=not args.no_bertscore)
            q = m.get('generation_quality', {})
            b = m.get('binary_classification', {})
            results['lora_moe'] = {
                'n_samples': len(preds),
                'generation_quality': q,
                'binary_classification': b,
            }
            lora_rougeL = q.get('rougeL', 0)
            logger.info(f'LoRA-MoE: ROUGE-L={lora_rougeL:.4f} F1={b.get("f1_score", 0):.4f}')
        else:
            lora_rougeL = 0.0
    except Exception as e:
        logger.error(f'LoRA-MoE evaluation failed: {e}')
        lora_rougeL = 0.0

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    try:
        plot_bar_with_errorbars(shot_summary, lora_rougeL, EXP_DIR, test_mode=args.test_mode)
    except Exception as e:
        logger.warning(f'Plotting failed: {e}')

    # Summary
    logger.info('\n' + '=' * 80)
    logger.info('FEW-SHOT CONSISTENCY SUMMARY')
    logger.info('=' * 80)
    logger.info(f'{"Config":<16} {"Mean ROUGE-L":>14} {"Std":>8}')
    logger.info('-' * 40)
    for n_shots, (mean, std) in sorted(shot_summary.items()):
        logger.info(f'{n_shots}-shot{" ":>10} {mean:>14.4f} {std:>8.4f}')
    logger.info(f'LoRA-MoE{" ":>10} {lora_rougeL:>14.4f} {"0.0000":>8}')
    logger.info(f'\nResults saved to: {EXP_DIR}')


def main():
    parser = argparse.ArgumentParser(description='Exp6: Few-shot vs fine-tuning')
    parser.add_argument('--force-regenerate', action='store_true')
    parser.add_argument('--from-cache', action='store_true')
    parser.add_argument('--no-bertscore', action='store_true')
    parser.add_argument('--test-mode', action='store_true')
    args = parser.parse_args()
    if args.from_cache:
        args.force_regenerate = False
    run(args)


if __name__ == '__main__':
    main()
