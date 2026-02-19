#!/usr/bin/env python3
"""
Experiment 5: Data Efficiency Analysis

Measure how performance scales with training data fraction.

Data fractions: [0.10, 0.25, 0.50, 0.75, 1.00]
Methods: lora_moe (text expert), lora_single (general), full_finetuning (text expert)

Output: outputs/evaluations/experiments/exp5_data_efficiency/
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
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp5')

path_cfg = get_path_config()
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp5_data_efficiency'
CACHE_DIR_BASE = path_cfg.OUTPUTS_DIR / 'inference_cache'

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
METHODS = ['lora_moe', 'lora_single', 'full_finetuning']


def _fraction_tag(fraction):
    return f'{int(fraction * 100)}pct'


def _get_ckpt_path(method, fraction):
    tag = _fraction_tag(fraction)
    if fraction == 1.00:
        if method == 'lora_moe':
            return path_cfg.LORA_MOE_CKPTS['text']
        elif method == 'full_finetuning':
            return path_cfg.FULL_FINETUNING_CKPTS['text']
    return path_cfg.CHECKPOINTS_DIR / f'exp5_{method}' / f'text_{tag}'


def train_for_fraction(method, fraction, train_data, args):
    ckpt_path = _get_ckpt_path(method, fraction)

    if fraction == 1.00 and ckpt_path.exists():
        logger.info(f'{method}/{_fraction_tag(fraction)}: using existing checkpoint at {ckpt_path}')
        return

    if ckpt_path.exists() and not args.force_retrain:
        logger.info(f'{method}/{_fraction_tag(fraction)}: checkpoint exists, skipping')
        return

    logger.info(f'Training {method} at {_fraction_tag(fraction)} -> {ckpt_path}')

    n_train = max(1, int(len(train_data) * fraction))
    random.seed(42)
    subset = random.sample(train_data, n_train)
    logger.info(f'Using {len(subset)} training samples (fraction={fraction})')

    if method in ('lora_moe', 'lora_single'):
        from src.training.lora_trainer import LoRATrainer
        expert_type = 'text' if method == 'lora_moe' else 'general'
        trainer = LoRATrainer(
            expert_type=expert_type,
            output_dir=str(ckpt_path),
            debug_samples=False
        )
        trainer.setup_model()
        trainer.prepare_data()
        # Replace training data subset
        trainer.train_dataset.data = subset
        trainer.train()

    elif method == 'full_finetuning':
        from src.training.full_finetuning_trainer import FullFineTuningTrainer
        trainer = FullFineTuningTrainer(
            expert_type='text',
            output_dir=str(ckpt_path),
            debug_samples=False
        )
        trainer.setup_model()
        trainer.prepare_data()
        trainer.train_dataset.data = subset
        trainer.train()

    logger.info(f'Training complete: {ckpt_path}')


def run_inference(method, fraction, test_data, args):
    tag = _fraction_tag(fraction)
    cache_subdir = CACHE_DIR_BASE / f'{method}_exp5'
    filename = f'text_{tag}_predictions.json'

    cached = load_predictions_cache(cache_subdir, filename)
    if cached and not args.force_regenerate:
        logger.info(f'{method}/{tag}: loaded from cache')
        return cached

    ckpt_path = _get_ckpt_path(method, fraction)
    if not ckpt_path.exists():
        logger.warning(f'{method}/{tag}: checkpoint not found at {ckpt_path}')
        return None

    logger.info(f'{method}/{tag}: running inference from {ckpt_path}')

    # Determine expert class
    if method in ('lora_moe', 'full_finetuning'):
        from src.experts import TextExpert
        expert = TextExpert(lora_path=str(ckpt_path), use_4bit=True)
    else:  # lora_single uses GeneralExpert with this ckpt path
        from src.experts import GeneralExpert
        expert = GeneralExpert(lora_path=str(ckpt_path), use_4bit=True)

    if not expert.load_model():
        logger.error(f'{method}/{tag}: model load failed')
        return None

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    try:
        predictions = expert.batch_generate_instruction(inputs, batch_size=4)
    except Exception as e:
        logger.error(f'{method}/{tag}: generation failed: {e}')
        expert.unload_model()
        return None
    finally:
        expert.unload_model()

    samples = [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]
    save_predictions_cache(
        samples, method, 'text',
        {'fraction': fraction, 'n_train': int(len(train_data_global) * fraction)},
        cache_subdir, filename
    )
    return load_predictions_cache(cache_subdir, filename)


train_data_global = []   # set in run() so train_for_fraction can use it


def plot_learning_curves(fraction_results, exp_dir):
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'lora_moe': '#1f77b4', 'lora_single': '#ff7f0e', 'full_finetuning': '#2ca02c'}
    labels = {'lora_moe': 'LoRA-MoE', 'lora_single': 'LoRA-Single', 'full_finetuning': 'Full FT'}

    for method in METHODS:
        xs, ys = [], []
        for fraction in FRACTIONS:
            tag = _fraction_tag(fraction)
            key = f'{method}_{tag}'
            if key in fraction_results:
                q = fraction_results[key].get('generation_quality', {})
                xs.append(fraction * 100)
                ys.append(q.get('rougeL', 0))
        if xs:
            ax.plot(xs, ys, marker='o', label=labels[method],
                    color=colors.get(method, None), linewidth=2)

    ax.set_xlabel('Training Data (%)')
    ax.set_ylabel('ROUGE-L')
    ax.set_title('Exp5: Data Efficiency - Learning Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    path = plots_dir / 'learning_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Learning curves plot saved: {path}')


def run(args):
    global train_data_global

    logger.info('=' * 80)
    logger.info('Experiment 5: Data Efficiency Analysis')
    logger.info('=' * 80)

    logger.info('Loading text dataset...')
    all_data = TextDatasetLoader().load_csv_files()
    train_data, _, test_data = split_dataset_for_expert(all_data, 'text')
    train_data_global = train_data
    logger.info(f'Train={len(train_data)}, Test={len(test_data)}')

    results = {
        'experiment': 'exp5_data_efficiency_analysis',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'fractions': FRACTIONS,
        'methods': METHODS,
        'results': {},
    }
    fraction_results = {}

    for method in METHODS:
        logger.info(f'\n=== Method: {method} ===')
        for fraction in FRACTIONS:
            tag = _fraction_tag(fraction)
            label = f'{method}/{tag}'
            logger.info(f'\n--- {label} ---')

            try:
                train_for_fraction(method, fraction, train_data, args)
            except Exception as e:
                logger.error(f'{label}: training failed: {e}')
                logger.error(traceback.format_exc())

            try:
                cached = run_inference(method, fraction, test_data, args)
                if cached is None:
                    logger.warning(f'{label}: skipped')
                    continue

                preds = [s['prediction'] for s in cached['samples']]
                refs = [s['reference'] for s in cached['samples']]
                m = compute_all_metrics(preds, refs, use_bertscore=not args.no_bertscore)

                q = m.get('generation_quality', {})
                b = m.get('binary_classification', {})
                key = f'{method}_{tag}'
                results['results'][key] = {
                    'method': method,
                    'fraction': fraction,
                    'n_train': max(1, int(len(train_data) * fraction)),
                    'n_samples': len(preds),
                    'generation_quality': q,
                    'binary_classification': b,
                }
                fraction_results[key] = m

                logger.info(
                    f'{label}: ROUGE-L={q.get("rougeL", 0):.4f} '
                    f'F1={b.get("f1_score", 0):.4f}'
                )
            except Exception as e:
                logger.error(f'{label}: evaluation failed: {e}')
                logger.error(traceback.format_exc())

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    try:
        plot_learning_curves(fraction_results, EXP_DIR)
    except Exception as e:
        logger.warning(f'Plotting failed: {e}')

    # Summary
    logger.info('\n' + '=' * 80)
    logger.info('DATA EFFICIENCY SUMMARY')
    logger.info('=' * 80)
    header = f'{"Method":<18}'
    for f in FRACTIONS:
        header += f' {_fraction_tag(f):>8}'
    logger.info(header + '  (ROUGE-L)')
    logger.info('-' * 70)
    for method in METHODS:
        row = f'{method:<18}'
        for f in FRACTIONS:
            key = f'{method}_{_fraction_tag(f)}'
            val = fraction_results.get(key, {}).get('generation_quality', {}).get('rougeL', 0)
            row += f' {val:>8.4f}'
        logger.info(row)
    logger.info(f'\nResults saved to: {EXP_DIR}')


def main():
    parser = argparse.ArgumentParser(description='Exp5: Data efficiency analysis')
    parser.add_argument('--force-regenerate', action='store_true')
    parser.add_argument('--force-retrain', action='store_true')
    parser.add_argument('--from-cache', action='store_true')
    parser.add_argument('--no-bertscore', action='store_true')
    parser.add_argument('--test-mode', action='store_true')
    args = parser.parse_args()
    if args.from_cache:
        args.force_regenerate = False
        args.force_retrain = False
    run(args)


if __name__ == '__main__':
    main()