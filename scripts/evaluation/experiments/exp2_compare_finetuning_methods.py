#!/usr/bin/env python3
"""
Experiment 2: Fine-Tuning Method Comparison

Compare all 5 fine-tuning methods across 4 expert types:
  Methods: lora_moe, lora_single, p_tuning, prompt_tuning, full_finetuning
  Expert types: text, image, uml, general

Output: outputs/evaluations/experiments/exp2_finetuning_methods/
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

logger = get_logger('experiments.exp2')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp2_finetuning_methods'

METHODS = ['lora_moe', 'lora_single', 'p_tuning', 'prompt_tuning', 'full_finetuning']
EXPERT_TYPES = ['text', 'image', 'uml', 'general']

METHOD_CKPT_MAP = {
    'lora_moe': lambda et: str(path_cfg.LORA_MOE_CKPTS[et]),
    'lora_single': lambda et: str(path_cfg.LORA_SINGLE_CKPT),
    'p_tuning': lambda et: str(path_cfg.PTUNING_CKPTS[et]),
    'prompt_tuning': lambda et: str(path_cfg.PROMPT_TUNING_CKPTS[et]),
    'full_finetuning': lambda et: str(path_cfg.FULL_FINETUNING_CKPTS[et]),
}

EXPERT_CLASS_MAP = None  # Imported lazily to avoid loading torch at module level


def _get_expert_class(expert_type):
    from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert
    return {
        'text': TextExpert,
        'image': ImageExpert,
        'uml': UMLExpert,
        'general': GeneralExpert,
    }[expert_type]


def _load_test_data(expert_type):
    """Load and split data for the given expert type, return test set."""
    if expert_type == 'text':
        data = TextDatasetLoader().load_csv_files()
    elif expert_type == 'image':
        data = ImageDatasetLoader().load_csv_file()
    elif expert_type == 'uml':
        data = UMLDatasetLoader().load_csv_file()
    else:  # general
        data = GeneralDatasetLoader().load_all_data()

    _, _, test_data = split_dataset_for_expert(data, expert_type)
    return test_data


def _get_checkpoint_info(method, expert_type):
    """Return (ckpt_path, adapter_size_mb, has_training_metrics)."""
    ckpt_path = Path(METHOD_CKPT_MAP[method](expert_type))
    adapter_mb = 0.0
    if ckpt_path.exists():
        total_bytes = sum(f.stat().st_size for f in ckpt_path.rglob('*.bin'))
        total_bytes += sum(f.stat().st_size for f in ckpt_path.rglob('*.safetensors'))
        adapter_mb = total_bytes / (1024 ** 2)

    training_metrics = None
    metrics_file = ckpt_path / 'training_metrics.json'
    if metrics_file.exists():
        import json
        with open(metrics_file) as f:
            training_metrics = json.load(f)

    return str(ckpt_path), adapter_mb, training_metrics


def run_inference_for_method_expert(method, expert_type, test_data, args):
    """Run or load cached inference for one method x expert type combination."""
    cache_subdir = CACHE_DIR / method
    cache_filename = f'{expert_type}_predictions.json'

    cached = load_predictions_cache(cache_subdir, cache_filename)
    if cached and not args.force_regenerate:
        logger.info(f'{method}/{expert_type}: 从缓存加载')
        return cached

    logger.info(f'{method}/{expert_type}: 执行推理...')
    ckpt_path = METHOD_CKPT_MAP[method](expert_type)

    ExpertClass = _get_expert_class(expert_type)
    expert = ExpertClass(lora_path=ckpt_path, use_4bit=True)

    if not expert.load_model():
        logger.error(f'{method}/{expert_type}: 模型加载失败')
        return None

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    try:
        predictions = expert.batch_generate_instruction(inputs, batch_size=4)
    except Exception as e:
        logger.error(f'{method}/{expert_type}: 生成失败: {e}')
        logger.error(traceback.format_exc())
        expert.unload_model()
        return None
    finally:
        expert.unload_model()

    samples = [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]
    save_predictions_cache(
        samples, method, expert_type, {'ckpt': ckpt_path},
        cache_subdir, cache_filename
    )
    return load_predictions_cache(cache_subdir, cache_filename)


def plot_grouped_bar(results_table, exp_dir):
    """Generate grouped bar charts: one per expert type showing methods x metrics."""
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = ['bleu', 'rougeL', 'meteor']
    metric_labels = ['BLEU', 'ROUGE-L', 'METEOR']
    method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for expert_type in EXPERT_TYPES:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(metric_keys))
        width = 0.15
        for i, method in enumerate(METHODS):
            key = f'{method}_{expert_type}'
            if key not in results_table:
                continue
            q = results_table[key].get('generation_quality', {})
            values = [q.get(k, 0) for k in metric_keys]
            offset = (i - len(METHODS) / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=method, color=method_colors[i % 5])

        ax.set_title(f'实验2: 微调方法对比 - {expert_type.capitalize()} 专家')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plot_path = plots_dir / f'{expert_type}_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f'图表已保存: {plot_path}')


def run(args):
    logger.info('=' * 80)
    logger.info('实验2: 微调方法对比')
    logger.info('=' * 80)

    results = {
        'experiment': 'exp2_finetuning_methods',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'methods': METHODS,
        'expert_types': EXPERT_TYPES,
        'results': {},
    }
    results_table = {}

    for expert_type in EXPERT_TYPES:
        logger.info(f'\n=== 专家类型: {expert_type} ===')
        try:
            test_data = _load_test_data(expert_type)
            logger.info(f'测试集样本数: {len(test_data)}')
        except Exception as e:
            logger.error(f'加载 {expert_type} 数据失败: {e}')
            continue

        for method in METHODS:
            label = f'{method}/{expert_type}'
            logger.info(f'\n--- {label} ---')
            try:
                cached = run_inference_for_method_expert(method, expert_type, test_data, args)
                if cached is None:
                    logger.warning(f'{label}: 已跳过')
                    continue

                preds = [s['prediction'] for s in cached['samples']]
                refs = [s['reference'] for s in cached['samples']]
                m = compute_all_metrics(preds, refs, use_bertscore=not args.no_bertscore)

                ckpt_path, adapter_mb, training_m = _get_checkpoint_info(method, expert_type)
                q = m.get('generation_quality', {})
                b = m.get('binary_classification', {})

                entry = {
                    'n_samples': len(preds),
                    'checkpoint': ckpt_path,
                    'adapter_size_mb': round(adapter_mb, 2),
                    'generation_quality': q,
                    'format_metrics': m.get('format_metrics', {}),
                    'binary_classification': b,
                }
                if training_m:
                    entry['training_metrics'] = training_m

                results['results'][f'{method}_{expert_type}'] = entry
                results_table[f'{method}_{expert_type}'] = m

                logger.info(
                    f'{label}: BLEU={q.get("bleu", 0):.4f} '
                    f'ROUGE-L={q.get("rougeL", 0):.4f} '
                    f'F1={b.get("f1_score", 0):.4f} '
                    f'适配器大小={adapter_mb:.1f}MB'
                )
            except Exception as e:
                logger.error(f'{label} 执行失败: {e}')
                logger.error(traceback.format_exc())

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    try:
        plot_grouped_bar(results_table, EXP_DIR)
    except Exception as e:
        logger.warning(f'绘图失败: {e}')

    # 汇总表
    logger.info('\n' + '=' * 80)
    logger.info('结果汇总')
    logger.info('=' * 80)
    logger.info(f'{"方法+专家":<28} {"ROUGE-L":>8} {"BLEU":>8} {"F1":>8}')
    logger.info('-' * 56)
    for key, m in results['results'].items():
        q = m.get('generation_quality', {})
        b = m.get('binary_classification', {})
        logger.info(
            f'{key:<28} {q.get("rougeL", 0):>8.4f} '
            f'{q.get("bleu", 0):>8.4f} {b.get("f1_score", 0):>8.4f}'
        )
    logger.info(f'\n结果已保存至: {EXP_DIR}')


def main():
    parser = argparse.ArgumentParser(description='Exp2: Fine-tuning method comparison')
    parser.add_argument('--force-regenerate', action='store_true')
    parser.add_argument('--from-cache', action='store_true')
    parser.add_argument('--no-bertscore', action='store_true')
    parser.add_argument('--test-mode', action='store_true',
                        help='Use 10 samples only')
    args = parser.parse_args()
    if args.from_cache:
        args.force_regenerate = False
    run(args)


if __name__ == '__main__':
    main()