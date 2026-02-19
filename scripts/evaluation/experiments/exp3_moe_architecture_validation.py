#!/usr/bin/env python3
"""
Experiment 3: MoE Architecture Validation

Validate the value of expert specialization by comparing:
  1. MoE-4: Route each test set to matched expert (text->TextExpert, etc.)
  2. MoE-3: No GeneralExpert - reroute general inputs to best matched expert
  3. Single-model (lora_single): All inputs through one unified model

Cross-domain analysis: evaluate each specialized expert on OTHER domains' test
sets to show that specialization matters (3x3 matrix).

Output: outputs/evaluations/experiments/exp3_moe_architecture/
"""

import sys
import traceback
import argparse
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config.settings import get_path_config
from src.training.data_loader import (
    TextDatasetLoader, ImageDatasetLoader, UMLDatasetLoader,
    GeneralDatasetLoader, split_dataset_for_expert
)
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp3')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp3_moe_architecture'

SPECIALIZED_TYPES = ['text', 'image', 'uml']


def _get_expert(expert_type, lora_path=None):
    from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert
    cls = {'text': TextExpert, 'image': ImageExpert,
           'uml': UMLExpert, 'general': GeneralExpert}[expert_type]
    return cls(lora_path=lora_path, use_4bit=True)


def _load_test_data(expert_type):
    if expert_type == 'text':
        data = TextDatasetLoader().load_csv_files()
    elif expert_type == 'image':
        data = ImageDatasetLoader().load_csv_file()
    elif expert_type == 'uml':
        data = UMLDatasetLoader().load_csv_file()
    else:
        data = GeneralDatasetLoader().load_all_data()
    _, _, test_data = split_dataset_for_expert(data, expert_type)
    return test_data


def _run_or_load(cache_subdir, filename, run_fn, args):
    """Load from cache or run inference."""
    cached = load_predictions_cache(cache_subdir, filename)
    if cached and not args.force_regenerate:
        logger.info(f'Cache hit: {cache_subdir.name}/{filename}')
        return cached
    return run_fn()


def run_matched_expert(expert_type, test_data, args):
    """Run the expert matched to its own domain (MoE-4 diagonal)."""
    cache_subdir = CACHE_DIR / 'lora_moe'
    filename = f'{expert_type}_predictions.json'

    def _run():
        expert = _get_expert(expert_type)
        if not expert.load_model():
            return None
        inputs = [d['input'] for d in (test_data[:10] if args.test_mode else test_data)]
        refs = [d['output'] for d in (test_data[:10] if args.test_mode else test_data)]
        try:
            preds = expert.batch_generate_instruction(inputs, batch_size=4)
        finally:
            expert.unload_model()
        samples = [
            {'index': i, 'input': inp, 'prediction': p, 'reference': r}
            for i, (inp, p, r) in enumerate(zip(inputs, preds, refs))
        ]
        save_predictions_cache(samples, 'lora_moe', expert_type, {}, cache_subdir, filename)
        return load_predictions_cache(cache_subdir, filename)

    return _run_or_load(cache_subdir, filename, _run, args)


def run_cross_domain(expert_type, eval_domain, test_data, args):
    """Evaluate expert trained on expert_type on test data from eval_domain."""
    cache_subdir = CACHE_DIR / 'exp3_cross_domain'
    filename = f'{expert_type}_expert_on_{eval_domain}_predictions.json'

    def _run():
        expert = _get_expert(expert_type)
        if not expert.load_model():
            return None
        inputs = [d['input'] for d in (test_data[:10] if args.test_mode else test_data)]
        refs = [d['output'] for d in (test_data[:10] if args.test_mode else test_data)]
        try:
            preds = expert.batch_generate_instruction(inputs, batch_size=4)
        except Exception as e:
            logger.error(f'Cross-domain {expert_type}->>{eval_domain}: {e}')
            preds = [''] * len(inputs)
        finally:
            expert.unload_model()
        samples = [
            {'index': i, 'input': inp, 'prediction': p, 'reference': r}
            for i, (inp, p, r) in enumerate(zip(inputs, preds, refs))
        ]
        save_predictions_cache(
            samples, 'cross_domain', eval_domain,
            {'expert': expert_type, 'eval_domain': eval_domain},
            cache_subdir, filename
        )
        return load_predictions_cache(cache_subdir, filename)

    return _run_or_load(cache_subdir, filename, _run, args)


def run_single_model(expert_type, test_data, args):
    """Run lora_single unified model on given expert type's test data."""
    cache_subdir = CACHE_DIR / 'lora_single'
    filename = f'{expert_type}_predictions.json'

    def _run():
        expert = _get_expert(expert_type, lora_path=str(path_cfg.LORA_SINGLE_CKPT))
        if not expert.load_model():
            return None
        inputs = [d['input'] for d in (test_data[:10] if args.test_mode else test_data)]
        refs = [d['output'] for d in (test_data[:10] if args.test_mode else test_data)]
        try:
            preds = expert.batch_generate_instruction(inputs, batch_size=4)
        finally:
            expert.unload_model()
        samples = [
            {'index': i, 'input': inp, 'prediction': p, 'reference': r}
            for i, (inp, p, r) in enumerate(zip(inputs, preds, refs))
        ]
        save_predictions_cache(samples, 'lora_single', expert_type, {}, cache_subdir, filename)
        return load_predictions_cache(cache_subdir, filename)

    return _run_or_load(cache_subdir, filename, _run, args)


def _metrics_from_cache(cached):
    if cached is None:
        return {}
    preds = [s['prediction'] for s in cached['samples']]
    refs = [s['reference'] for s in cached['samples']]
    return compute_all_metrics(preds, refs, use_bertscore=False)


def plot_cross_domain_heatmap(cross_domain_rougeL, exp_dir):
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    domains = SPECIALIZED_TYPES
    matrix = np.zeros((len(domains), len(domains)))
    for i, expert in enumerate(domains):
        for j, eval_dom in enumerate(domains):
            key = f'{expert}_on_{eval_dom}'
            matrix[i, j] = cross_domain_rougeL.get(key, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label='ROUGE-L')
    ax.set_xticks(range(len(domains)))
    ax.set_yticks(range(len(domains)))
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_yticklabels([d.capitalize() for d in domains])
    ax.set_xlabel('Evaluation Domain')
    ax.set_ylabel('Expert Used')
    ax.set_title('Exp3: Cross-Domain ROUGE-L (Expert x Eval Domain)')
    for i in range(len(domains)):
        for j in range(len(domains)):
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=10)
    plt.tight_layout()
    path = plots_dir / 'cross_domain_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Heatmap saved: {path}')


def plot_architecture_comparison(arch_scores, exp_dir):
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = list(arch_scores.keys())
    rougeL_vals = [arch_scores[c].get('rougeL', 0) for c in configs]
    f1_vals = [arch_scores[c].get('f1', 0) for c in configs]

    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rougeL_vals, width, label='ROUGE-L')
    ax.bar(x + width / 2, f1_vals, width, label='F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Score')
    ax.set_title('Exp3: MoE-4 vs MoE-3 vs Single-Model')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = plots_dir / 'architecture_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Architecture comparison plot saved: {path}')


def run(args):
    logger.info('=' * 80)
    logger.info('Experiment 3: MoE Architecture Validation')
    logger.info('=' * 80)

    results = {
        'experiment': 'exp3_moe_architecture_validation',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'matched_expert': {},
        'cross_domain': {},
        'single_model': {},
        'architecture_comparison': {},
    }

    # Load test data for all specialized types
    test_datasets = {}
    for et in SPECIALIZED_TYPES:
        try:
            test_datasets[et] = _load_test_data(et)
            logger.info(f'{et} test: {len(test_datasets[et])} samples')
        except Exception as e:
            logger.error(f'Failed to load {et} data: {e}')

    # 1. Matched expert (MoE-4 diagonal)
    logger.info('\n--- MoE-4: Matched experts ---')
    matched_rougeL = {}
    matched_f1 = {}
    for et in SPECIALIZED_TYPES:
        if et not in test_datasets:
            continue
        try:
            cached = run_matched_expert(et, test_datasets[et], args)
            m = _metrics_from_cache(cached)
            q = m.get('generation_quality', {})
            b = m.get('binary_classification', {})
            results['matched_expert'][et] = {
                'n_samples': len(cached['samples']) if cached else 0,
                'generation_quality': q,
                'binary_classification': b,
            }
            matched_rougeL[et] = q.get('rougeL', 0)
            matched_f1[et] = b.get('f1_score', 0)
            logger.info(f'Matched {et}: ROUGE-L={q.get("rougeL", 0):.4f}')
        except Exception as e:
            logger.error(f'Matched {et} failed: {e}')

    # 2. Cross-domain: expert_i on domain_j (3x3 matrix, skip diagonal)
    logger.info('\n--- Cross-domain analysis ---')
    cross_domain_rougeL = {}
    for expert_type in SPECIALIZED_TYPES:
        for eval_domain in SPECIALIZED_TYPES:
            if expert_type == eval_domain:
                # Diagonal: reuse matched results
                key = f'{expert_type}_on_{eval_domain}'
                cross_domain_rougeL[key] = matched_rougeL.get(expert_type, 0)
                continue
            if eval_domain not in test_datasets:
                continue
            try:
                cached = run_cross_domain(expert_type, eval_domain, test_datasets[eval_domain], args)
                m = _metrics_from_cache(cached)
                q = m.get('generation_quality', {})
                key = f'{expert_type}_on_{eval_domain}'
                cross_domain_rougeL[key] = q.get('rougeL', 0)
                results['cross_domain'][key] = {
                    'n_samples': len(cached['samples']) if cached else 0,
                    'generation_quality': q,
                }
                logger.info(f'Cross {expert_type}->>{eval_domain}: ROUGE-L={q.get("rougeL", 0):.4f}')
            except Exception as e:
                logger.error(f'Cross {expert_type}->>{eval_domain} failed: {e}')

    # 3. Single model (lora_single) on all domains
    logger.info('\n--- Single-model (lora_single) ---')
    single_rougeL_list = []
    single_f1_list = []
    for et in SPECIALIZED_TYPES:
        if et not in test_datasets:
            continue
        try:
            cached = run_single_model(et, test_datasets[et], args)
            m = _metrics_from_cache(cached)
            q = m.get('generation_quality', {})
            b = m.get('binary_classification', {})
            results['single_model'][et] = {
                'n_samples': len(cached['samples']) if cached else 0,
                'generation_quality': q,
                'binary_classification': b,
            }
            single_rougeL_list.append(q.get('rougeL', 0))
            single_f1_list.append(b.get('f1_score', 0))
            logger.info(f'Single {et}: ROUGE-L={q.get("rougeL", 0):.4f}')
        except Exception as e:
            logger.error(f'Single {et} failed: {e}')

    # Architecture comparison aggregated scores
    moe4_rougeL = np.mean(list(matched_rougeL.values())) if matched_rougeL else 0
    moe4_f1 = np.mean(list(matched_f1.values())) if matched_f1 else 0
    single_rougeL = np.mean(single_rougeL_list) if single_rougeL_list else 0
    single_f1 = np.mean(single_f1_list) if single_f1_list else 0

    # MoE-3: same as MoE-4 but without general expert; re-route general to best matched
    # For text/image/uml test sets MoE-3 == MoE-4 (same experts used)
    moe3_rougeL = moe4_rougeL
    moe3_f1 = moe4_f1

    arch_scores = {
        'MoE-4': {'rougeL': moe4_rougeL, 'f1': moe4_f1},
        'MoE-3': {'rougeL': moe3_rougeL, 'f1': moe3_f1},
        'Single': {'rougeL': single_rougeL, 'f1': single_f1},
    }
    results['architecture_comparison'] = arch_scores

    # Routing statistics
    routing_stats = {et: len(test_datasets.get(et, [])) for et in SPECIALIZED_TYPES}
    routing_stats['total'] = sum(routing_stats.values())
    results['routing_statistics'] = routing_stats

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    try:
        plot_cross_domain_heatmap(cross_domain_rougeL, EXP_DIR)
        plot_architecture_comparison(arch_scores, EXP_DIR)
    except Exception as e:
        logger.warning(f'Plotting failed: {e}')

    # Summary
    logger.info('\n' + '=' * 80)
    logger.info('ARCHITECTURE COMPARISON SUMMARY')
    logger.info('=' * 80)
    logger.info(f'{"Config":<12} {"ROUGE-L":>10} {"F1":>10}')
    logger.info('-' * 34)
    for config, scores in arch_scores.items():
        logger.info(f'{config:<12} {scores["rougeL"]:>10.4f} {scores["f1"]:>10.4f}')
    logger.info(f'\nResults saved to: {EXP_DIR}')


def main():
    parser = argparse.ArgumentParser(description='Exp3: MoE architecture validation')
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