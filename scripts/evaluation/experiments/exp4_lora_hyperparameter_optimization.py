#!/usr/bin/env python3
"""
Experiment 4: LoRA Hyperparameter Optimization

Find optimal LoRA rank/alpha/dropout configuration by training and evaluating
10 configurations on the text expert.

Baseline (8, 16, 0.05) reuses LORA_MOE_CKPTS['text'] without retraining.

Output: outputs/evaluations/experiments/exp4_lora_hyperparameters/
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
from src.training.data_loader import TextDatasetLoader, split_dataset_for_expert
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp4')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache' / 'lora_moe_exp4'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp4_lora_hyperparameters'

# Exactly these 10 configs: (rank, alpha, dropout)
CONFIGS = [
    (8,  16,  0.05),   # baseline - reuse LORA_MOE_CKPTS['text']
    (8,  16,  0.0),
    (8,  16,  0.1),
    (16, 32,  0.05),
    (16, 32,  0.0),
    (16, 32,  0.1),
    (32, 64,  0.05),
    (32, 64,  0.0),
    (32, 64,  0.1),
    (64, 128, 0.05),
]


def _is_full_run_cache(cache_dir, filename):
    """Return True if a non-test-mode cache file exists for this combination."""
    import json as _json
    filepath = Path(cache_dir) / filename
    if not filepath.exists():
        return False
    try:
        raw = _json.loads(filepath.read_text(encoding='utf-8'))
        return not (
            raw.get('test_mode', False)
            or raw.get('metadata', {}).get('test_mode', False)
        )
    except Exception:
        return False


def _config_name(rank, alpha, dropout):
    return f'text_r{rank}_a{alpha}_d{dropout}'


def _get_ckpt_path(rank, alpha, dropout):
    if rank == 8 and alpha == 16 and dropout == 0.05:
        return path_cfg.LORA_MOE_CKPTS['text']
    return path_cfg.CHECKPOINTS_DIR / 'lora_moe_exp4' / _config_name(rank, alpha, dropout)


def train_config(rank, alpha, dropout, args):
    """如果检查点不存在则训练该LoRA配置。"""
    ckpt_path = _get_ckpt_path(rank, alpha, dropout)

    if rank == 8 and alpha == 16 and dropout == 0.05:
        logger.info(f'基线配置 (8,16,0.05): 复用已有检查点 {ckpt_path}')
        return

    if ckpt_path.exists() and not args.force_retrain:
        logger.info(f'检查点已存在，跳过训练: {ckpt_path}')
        return

    logger.info(f'训练配置 r={rank} a={alpha} d={dropout} -> {ckpt_path}')
    from src.training.lora_trainer import LoRATrainer

    trainer = LoRATrainer(
        expert_type='text',
        output_dir=str(ckpt_path),
        debug_samples=False
    )
    trainer.lora_rank = rank
    trainer.lora_alpha = alpha
    trainer.lora_dropout = dropout

    trainer.setup_model()
    trainer.prepare_data()
    trainer.train()
    logger.info(f'训练完成: {ckpt_path}')


def run_inference(rank, alpha, dropout, test_data, args):
    """运行或从缓存加载某配置的推理结果。"""
    cfg_name = _config_name(rank, alpha, dropout)
    filename = f'{cfg_name}_predictions.json'
    cached = load_predictions_cache(CACHE_DIR, filename)
    if cached and not args.force_regenerate:
        logger.info(f'{cfg_name}: 从缓存加载')
        return cached

    ckpt_path = _get_ckpt_path(rank, alpha, dropout)
    if not ckpt_path.exists():
        logger.warning(f'{cfg_name}: 检查点不存在 {ckpt_path}')
        return None

    logger.info(f'{cfg_name}: 从 {ckpt_path} 执行推理')
    from src.experts import TextExpert

    expert = TextExpert(lora_path=str(ckpt_path), use_4bit=True)
    if not expert.load_model():
        logger.error(f'{cfg_name}: 模型加载失败')
        return None

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    try:
        predictions = expert.batch_generate_instruction(inputs, batch_size=4)
    except Exception as e:
        logger.error(f'{cfg_name}: 生成失败: {e}')
        expert.unload_model()
        return None
    finally:
        expert.unload_model()

    samples = [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]
    save_predictions_cache(
        samples, 'lora_moe_exp4', 'text',
        {'rank': rank, 'alpha': alpha, 'dropout': dropout},
        CACHE_DIR, filename
    )
    return load_predictions_cache(CACHE_DIR, filename)


def plot_rank_vs_rouge(config_results, exp_dir):
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Group by rank
    ranks = sorted(set(r for r, _, _ in CONFIGS))
    rank_rougeL = {r: [] for r in ranks}
    for (rank, alpha, dropout), m in config_results.items():
        q = m.get('generation_quality', {})
        rank_rougeL[rank].append(q.get('rougeL', 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(ranks)
    y = [np.mean(rank_rougeL[r]) for r in x]
    y_std = [np.std(rank_rougeL[r]) for r in x]
    ax.errorbar(x, y, yerr=y_std, marker='o', capsize=4, linewidth=2)
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('ROUGE-L (Mean over Dropout Settings)')
    ax.set_title('Exp4: ROUGE-L vs LoRA Rank')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = plots_dir / 'rank_vs_rougeL.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


def plot_heatmap_dropout_alpha(config_results, exp_dir, fixed_rank=16):
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(set(a for _, a, _ in CONFIGS if _ == 0.05 or True))
    dropouts = sorted(set(d for _, _, d in CONFIGS))
    # Filter to rank=fixed_rank configs
    rank_configs = {(a, d): m for (r, a, d), m in config_results.items() if r == fixed_rank}
    if not rank_configs:
        return

    unique_alphas = sorted(set(a for a, _ in rank_configs.keys()))
    unique_dropouts = sorted(set(d for _, d in rank_configs.keys()))

    matrix = np.zeros((len(unique_dropouts), len(unique_alphas)))
    for i, d in enumerate(unique_dropouts):
        for j, a in enumerate(unique_alphas):
            m = rank_configs.get((a, d), {})
            matrix[i, j] = m.get('generation_quality', {}).get('rougeL', 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='ROUGE-L')
    ax.set_xticks(range(len(unique_alphas)))
    ax.set_yticks(range(len(unique_dropouts)))
    ax.set_xticklabels([str(a) for a in unique_alphas])
    ax.set_yticklabels([str(d) for d in unique_dropouts])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Dropout')
    ax.set_title(f'Exp4: ROUGE-L Heatmap (rank={fixed_rank})')
    for i in range(len(unique_dropouts)):
        for j in range(len(unique_alphas)):
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)
    plt.tight_layout()
    path = plots_dir / f'heatmap_rank{fixed_rank}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'热图已保存: {path}')


def run(args):
    logger.info('=' * 80)
    logger.info('实验4: LoRA超参数优化')
    logger.info('=' * 80)

    # 加载文本测试数据
    logger.info('加载文本数据集...')
    all_data = TextDatasetLoader().load_csv_files()
    train_data, _, test_data = split_dataset_for_expert(all_data, 'text')
    logger.info(f'测试集样本数: {len(test_data)}')

    results = {
        'experiment': 'exp4_lora_hyperparameter_optimization',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'configs': [],
    }

    config_results = {}

    for rank, alpha, dropout in CONFIGS:
        cfg_name = _config_name(rank, alpha, dropout)
        logger.info(f'\n--- 配置: {cfg_name} ---')

        if getattr(args, 'only_missing', False) and _is_full_run_cache(
                CACHE_DIR, f'{cfg_name}_predictions.json'):
            logger.info(f'{cfg_name}: cache exists, skipping (--only-missing)')
            continue

        try:
            train_config(rank, alpha, dropout, args)
        except Exception as e:
            logger.error(f'{cfg_name}: 训练失败: {e}')
            logger.error(traceback.format_exc())

        try:
            cached = run_inference(rank, alpha, dropout, test_data, args)
            if cached is None:
                logger.warning(f'{cfg_name}: 已跳过（推理失败）')
                continue

            preds = [s['prediction'] for s in cached['samples']]
            refs = [s['reference'] for s in cached['samples']]
            m = compute_all_metrics(preds, refs, use_bertscore=not args.no_bertscore)

            q = m.get('generation_quality', {})
            b = m.get('binary_classification', {})

            config_entry = {
                'name': cfg_name,
                'rank': rank,
                'alpha': alpha,
                'dropout': dropout,
                'n_samples': len(preds),
                'generation_quality': q,
                'binary_classification': b,
            }
            results['configs'].append(config_entry)
            config_results[(rank, alpha, dropout)] = m

            logger.info(
                f'{cfg_name}: ROUGE-L={q.get("rougeL", 0):.4f} '
                f'F1={b.get("f1_score", 0):.4f}'
            )
        except Exception as e:
            logger.error(f'{cfg_name}: 评估失败: {e}')
            logger.error(traceback.format_exc())

    if results['configs']:
        best = max(results['configs'], key=lambda c: c['generation_quality'].get('rougeL', 0))
        results['best_config'] = best
        logger.info(f'\n最优配置: {best["name"]} (ROUGE-L={best["generation_quality"].get("rougeL", 0):.4f})')

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    try:
        plot_rank_vs_rouge(config_results, EXP_DIR)
        plot_heatmap_dropout_alpha(config_results, EXP_DIR, fixed_rank=16)
    except Exception as e:
        logger.warning(f'绘图失败: {e}')

    # 汇总表
    logger.info('\n' + '=' * 80)
    logger.info('配置对比汇总')
    logger.info('=' * 80)
    logger.info(f'{"配置名称":<32} {"ROUGE-L":>8} {"F1":>8}')
    logger.info('-' * 50)
    for c in results['configs']:
        q = c.get('generation_quality', {})
        b = c.get('binary_classification', {})
        logger.info(
            f'{c["name"]:<32} {q.get("rougeL", 0):>8.4f} {b.get("f1_score", 0):>8.4f}'
        )
    logger.info(f'\n结果已保存至: {EXP_DIR}')


def main():
    parser = argparse.ArgumentParser(description='Exp4: LoRA hyperparameter optimization')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Re-run inference even if cache exists')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Re-train even if checkpoint exists')
    parser.add_argument('--from-cache', action='store_true')
    parser.add_argument('--no-bertscore', action='store_true')
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--only-missing', action='store_true',
                        help='Skip configs that already have a full-run cache. '
                             'Test-mode caches are treated as missing and re-run automatically.')
    args = parser.parse_args()
    if args.from_cache:
        args.force_regenerate = False
        args.force_retrain = False
    run(args)


if __name__ == '__main__':
    main()