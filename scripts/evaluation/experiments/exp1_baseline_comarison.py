#!/usr/bin/env python3
"""
Experiment 1: Baseline Comparison

Compare retrieval/rule-based baselines against LoRA-MoE on the text expert
test set.

Methods evaluated:
  - BM25 (retrieval)
  - LSA (retrieval)
  - Template Filling (rule-based)
  - Zero-Shot (base Qwen3-8B, no LoRA)
  - LoRA-MoE (text expert fine-tuned)

Output: outputs/evaluations/experiments/exp1_baseline_comparison/
"""

import sys
import json
import argparse
import traceback
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
from src.baselines.ir_methods import BM25Retriever, LSARetriever
from src.baselines.template_filling import TemplateFiller
from src.baselines.zero_shot import ZeroShotGenerator
from src.baselines.inference_utils import (
    save_predictions_cache,
    load_predictions_cache,
    compute_all_metrics,
    save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp1')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp1_baseline_comparison'

METHODS = ['bm25', 'lsa', 'template', 'zeroshot', 'lora_moe']


def _make_samples(inputs, predictions, references):
    return [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]


def run_bm25(train_data, test_data, args):
    cache_path = CACHE_DIR / 'baselines' / 'bm25_text_predictions.json'
    cached = load_predictions_cache(cache_path.parent, cache_path.name)
    if cached and not args.force_regenerate:
        logger.info('BM25: loaded from cache')
        return cached

    logger.info('BM25: building index and retrieving...')
    retriever = BM25Retriever()
    retriever.build_index(train_data)

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    predictions = retriever.batch_retrieve(inputs)
    samples = _make_samples(inputs, predictions, references)
    save_predictions_cache(samples, 'bm25', 'text', {}, cache_path.parent, cache_path.name)
    return load_predictions_cache(cache_path.parent, cache_path.name)


def run_lsa(train_data, test_data, args):
    cache_path = CACHE_DIR / 'baselines' / 'lsa_text_predictions.json'
    cached = load_predictions_cache(cache_path.parent, cache_path.name)
    if cached and not args.force_regenerate:
        logger.info('LSA: loaded from cache')
        return cached

    logger.info('LSA: building index and retrieving...')
    retriever = LSARetriever(n_components=100)
    retriever.build_index(train_data)

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    predictions = retriever.batch_retrieve(inputs)
    samples = _make_samples(inputs, predictions, references)
    save_predictions_cache(samples, 'lsa', 'text', {}, cache_path.parent, cache_path.name)
    return load_predictions_cache(cache_path.parent, cache_path.name)


def run_template(test_data, args):
    cache_path = CACHE_DIR / 'baselines' / 'template_text_predictions.json'
    cached = load_predictions_cache(cache_path.parent, cache_path.name)
    if cached and not args.force_regenerate:
        logger.info('Template: loaded from cache')
        return cached

    logger.info('Template filling: generating...')
    filler = TemplateFiller()
    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    predictions = filler.batch_fill(inputs)
    samples = _make_samples(inputs, predictions, references)
    save_predictions_cache(samples, 'template', 'text', {}, cache_path.parent, cache_path.name)
    return load_predictions_cache(cache_path.parent, cache_path.name)


def run_zeroshot(test_data, args):
    cache_path = CACHE_DIR / 'baselines' / 'zeroshot_text_predictions.json'
    cached = load_predictions_cache(cache_path.parent, cache_path.name)
    if cached and not args.force_regenerate:
        logger.info('Zero-shot: loaded from cache')
        return cached

    logger.info('Zero-shot: loading model and generating...')
    generator = ZeroShotGenerator(use_4bit=True)
    if not generator.load_model():
        logger.error('Zero-shot model failed to load')
        return None

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    try:
        predictions = generator.batch_generate(inputs, input_type='text', n_shots=0)
    finally:
        generator.unload_model()

    samples = _make_samples(inputs, predictions, references)
    save_predictions_cache(samples, 'zero_shot', 'text', {}, cache_path.parent, cache_path.name)
    return load_predictions_cache(cache_path.parent, cache_path.name)


def run_lora_moe(test_data, args):
    cache_path = CACHE_DIR / 'lora_moe' / 'text_predictions.json'
    cached = load_predictions_cache(cache_path.parent, cache_path.name)
    if cached and not args.force_regenerate:
        logger.info('LoRA-MoE: loaded from cache')
        return cached

    logger.info('LoRA-MoE: loading text expert and generating...')
    from src.experts import TextExpert

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    expert = TextExpert(lora_path=None, use_4bit=True)
    if not expert.load_model():
        logger.error('LoRA-MoE text expert failed to load')
        return None

    try:
        predictions = expert.batch_generate_instruction(inputs, batch_size=8)
    finally:
        expert.unload_model()

    samples = _make_samples(inputs, predictions, references)
    save_predictions_cache(samples, 'lora_moe', 'text', {}, cache_path.parent, cache_path.name)
    return load_predictions_cache(cache_path.parent, cache_path.name)


def plot_comparison(metrics_by_method, exp_dir, test_mode=False):
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = ['bleu', 'rougeL', 'meteor', 'f1_score']
    metric_labels = ['BLEU', 'ROUGE-L', 'METEOR', 'F1 Score']
    method_labels = {
        'bm25': 'BM25',
        'lsa': 'LSA',
        'template': 'Template',
        'zeroshot': 'Zero-Shot',
        'lora_moe': 'LoRA-MoE',
    }

    methods = [m for m in METHODS if m in metrics_by_method]
    if not methods:
        logger.warning('No metrics to plot')
        return

    x = np.arange(len(metric_keys))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        m = metrics_by_method[method]
        quality = m.get('generation_quality', {})
        binary = m.get('binary_classification', {})
        values = [
            quality.get('bleu', 0),
            quality.get('rougeL', 0),
            quality.get('meteor', 0),
            binary.get('f1_score', 0),
        ]
        offset = (i - len(methods) / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=method_labels.get(method, method))

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    title = 'Exp1: Baseline Comparison (Text Expert)'
    if test_mode:
        title += ' [TEST MODE]'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plot_path = plots_dir / 'comparison.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved: {plot_path}')


def run(args):
    logger.info('=' * 80)
    logger.info('Experiment 1: Baseline Comparison')
    logger.info('=' * 80)

    # Load data
    logger.info('Loading text dataset...')
    loader = TextDatasetLoader()
    all_data = loader.load_csv_files()
    train_data, val_data, test_data = split_dataset_for_expert(all_data, 'text')
    logger.info(f'Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}')

    results = {
        'experiment': 'exp1_baseline_comparison',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'test_samples': len(test_data) if not args.test_mode else 10,
        'methods': {},
    }

    method_runners = {
        'bm25': lambda: run_bm25(train_data, test_data, args),
        'lsa': lambda: run_lsa(train_data, test_data, args),
        'template': lambda: run_template(test_data, args),
        'zeroshot': lambda: run_zeroshot(test_data, args),
        'lora_moe': lambda: run_lora_moe(test_data, args),
    }

    metrics_by_method = {}

    for method, runner in method_runners.items():
        logger.info(f'\n--- Running method: {method} ---')
        try:
            cached = runner()
            if cached is None:
                logger.warning(f'{method}: skipped (inference failed)')
                continue

            samples = cached.get('samples', [])
            predictions = [s['prediction'] for s in samples]
            references = [s['reference'] for s in samples]

            logger.info(f'Computing metrics for {method} ({len(predictions)} samples)...')
            m = compute_all_metrics(predictions, references, use_bertscore=not args.no_bertscore)
            metrics_by_method[method] = m

            results['methods'][method] = {
                'n_samples': len(predictions),
                'generation_quality': m.get('generation_quality', {}),
                'format_metrics': m.get('format_metrics', {}),
                'binary_classification': m.get('binary_classification', {}),
            }

            quality = m.get('generation_quality', {})
            binary = m.get('binary_classification', {})
            logger.info(
                f'{method}: BLEU={quality.get("bleu", 0):.4f} '
                f'ROUGE-L={quality.get("rougeL", 0):.4f} '
                f'F1={binary.get("f1_score", 0):.4f}'
            )

        except Exception as e:
            logger.error(f'{method} failed: {e}')
            logger.error(traceback.format_exc())

    # Save results
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    # Plot
    try:
        plot_comparison(metrics_by_method, EXP_DIR, test_mode=args.test_mode)
    except Exception as e:
        logger.warning(f'Plotting failed: {e}')

    # Summary table
    logger.info('\n' + '=' * 80)
    logger.info('RESULTS SUMMARY')
    logger.info('=' * 80)
    logger.info(f'{"Method":<16} {"BLEU":>8} {"ROUGE-L":>8} {"METEOR":>8} {"F1":>8}')
    logger.info('-' * 52)
    for method, m in results['methods'].items():
        q = m.get('generation_quality', {})
        b = m.get('binary_classification', {})
        logger.info(
            f'{method:<16} {q.get("bleu", 0):>8.4f} {q.get("rougeL", 0):>8.4f} '
            f'{q.get("meteor", 0):>8.4f} {b.get("f1_score", 0):>8.4f}'
        )
    logger.info('=' * 80)
    logger.info(f'Results saved to: {EXP_DIR}')


def main():
    parser = argparse.ArgumentParser(description='Exp1: Baseline comparison')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Re-run inference even if cache exists')
    parser.add_argument('--from-cache', action='store_true',
                        help='Skip inference, load from cache only')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='Disable BERTScore for faster evaluation')
    parser.add_argument('--test-mode', action='store_true',
                        help='Use 10 samples only (quick validation)')
    args = parser.parse_args()

    if args.from_cache:
        args.force_regenerate = False

    run(args)


if __name__ == '__main__':
    main()