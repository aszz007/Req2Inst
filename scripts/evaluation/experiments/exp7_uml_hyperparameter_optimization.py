#!/usr/bin/env python3
"""
Experiment 7: UML Expert LoRA Hyperparameter Optimization

Motivation: Exp4 found text_r64_a128_d0.05 as the optimal config for Text Expert,
and this was applied globally to all experts including UML. However, UML Expert
showed no improvement or slight degradation compared to the original default.
UML data has distinct characteristics (avg 1063 tokens, highly structured
relational JSON) that may actually prefer a lower-rank configuration with
stronger regularisation.

Research question: Is rank=64 truly optimal for UML, or does UML peak at a
lower rank where generalisation matters more than raw capacity?

Baseline: Reuses the existing lora_moe UML checkpoint (trained at r64/a128/d0.05,
the current LoRATrainer default after Exp4's global parameter transfer).

CONFIGS design (12 configurations):
  - (64, 128, 0.05) baseline — reuse LORA_MOE_CKPTS['uml'], NO retrain
  - Sweep downward: r8, r16(×3 dropout), r32(×3 dropout) — test if UML prefers lower rank
  - Intermediate: r48 — does UML peak between r32 and r64?
  - Same-rank dropout ablation: r64 with d0.0 and d0.1
  - Upper bound: r96 — does further capacity help UML at all?

Output: outputs/evaluations/experiments/exp7_uml_hyperparameters/
"""

import sys
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config.settings import get_path_config
from src.training.data_loader import UMLDatasetLoader, split_dataset_for_expert
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp7')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache' / 'lora_moe_exp7'
EXP_DIR   = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp7_uml_hyperparameters'

# -------------------------------------------------------------------------
# Baseline: the existing UML checkpoint was trained with the post-Exp4
# LoRATrainer defaults: rank=64, alpha=128, dropout=0.05.
# -------------------------------------------------------------------------
BASELINE_RANK    = 64
BASELINE_ALPHA   = 128
BASELINE_DROPOUT = 0.05

# -------------------------------------------------------------------------
# 12 configurations  (rank, alpha, dropout)
# Alpha = 2 × rank throughout (same scaling convention as Exp4)
# -------------------------------------------------------------------------
CONFIGS = [
    (64, 128, 0.05),   # baseline — reuse LORA_MOE_CKPTS['uml'], NO retrain
    (8,  16,  0.05),   # floor: minimum rank to bound the performance curve
    (16, 32,  0.0),
    (16, 32,  0.05),
    (16, 32,  0.1),
    (32, 64,  0.0),
    (32, 64,  0.05),
    (32, 64,  0.1),
    (48, 96,  0.05),   # intermediate: does UML peak before r64?
    (64, 128, 0.0),    # same rank as baseline, dropout ablation
    (64, 128, 0.1),    # same rank as baseline, dropout ablation
    (96, 192, 0.05),   # upper bound: does more capacity help UML at all?
]


# =========================================================================
# Helpers
# =========================================================================

def _is_full_run_cache(cache_dir, filename):
    """Return True if a non-test-mode cache file exists for this config."""
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
    return f'uml_r{rank}_a{alpha}_d{dropout}'


def _is_baseline(rank, alpha, dropout):
    return rank == BASELINE_RANK and alpha == BASELINE_ALPHA and dropout == BASELINE_DROPOUT


def _get_ckpt_path(rank, alpha, dropout):
    """Baseline reuses the production UML ckpt; all others go to lora_moe_exp7/."""
    if _is_baseline(rank, alpha, dropout):
        return path_cfg.LORA_MOE_CKPTS['uml']
    return path_cfg.CHECKPOINTS_DIR / 'lora_moe_exp7' / _config_name(rank, alpha, dropout)


# =========================================================================
# Training
# =========================================================================

def train_config(rank, alpha, dropout, args):
    """Train the LoRA config if its checkpoint does not yet exist."""
    ckpt_path = _get_ckpt_path(rank, alpha, dropout)

    if _is_baseline(rank, alpha, dropout):
        logger.info(
            f'基线配置 ({BASELINE_RANK},{BASELINE_ALPHA},{BASELINE_DROPOUT}): '
            f'复用已有检查点 {ckpt_path}'
        )
        return

    if ckpt_path.exists() and not args.force_retrain:
        logger.info(f'检查点已存在，跳过训练: {ckpt_path}')
        return

    logger.info(f'训练 UML 配置 r={rank} a={alpha} d={dropout} -> {ckpt_path}')
    from src.training.lora_trainer import LoRATrainer

    trainer = LoRATrainer(
        expert_type='uml',
        output_dir=str(ckpt_path),
        debug_samples=False,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )
    trainer.setup_model()
    trainer.prepare_data()
    trainer.train()
    logger.info(f'训练完成: {ckpt_path}')


# =========================================================================
# Inference
# =========================================================================

def run_inference(rank, alpha, dropout, test_data, args):
    """
    Run inference for one config and cache the predictions.
    Returns the loaded cache dict, or None on failure.

    Note on batch_size:
      UML sequences average 1063 tokens (95th-percentile 1554 tokens).
      Training already uses batch=1 to avoid OOM.  For inference we use
      batch_size=2 as a safe default; reduce further if OOM occurs.
    """
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
    from src.experts import UMLExpert

    expert = UMLExpert(lora_path=str(ckpt_path), use_4bit=True)
    if not expert.load_model():
        logger.error(f'{cfg_name}: 模型加载失败')
        return None

    inputs     = [d['input']  for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    try:
        # batch_size=2: UML平均序列长度1063 tokens，比Text长3倍，需使用更保守的batch
        predictions = expert.batch_generate_instruction(inputs, batch_size=2)
    except Exception as e:
        logger.error(f'{cfg_name}: 生成失败: {e}')
        return None
    finally:
        # finally保证无论成功还是异常都释放显存，避免双重unload
        expert.unload_model()

    samples = [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]
    save_predictions_cache(
        samples, 'lora_moe_exp7', 'uml',
        {'rank': rank, 'alpha': alpha, 'dropout': dropout},
        CACHE_DIR, filename
    )
    return load_predictions_cache(CACHE_DIR, filename)


# =========================================================================
# Visualizations
# =========================================================================

def plot_rank_vs_rouge(config_results, exp_dir):
    """
    Line chart: mean ROUGE-L per rank with std error bars.
    Mirrors exp4's rank_vs_rougeL plot for direct comparison.
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    ranks = sorted(set(r for r, _, _ in CONFIGS))
    rank_rougeL = {r: [] for r in ranks}
    for (rank, alpha, dropout), m in config_results.items():
        rank_rougeL[rank].append(m.get('generation_quality', {}).get('rougeL', 0))

    x   = list(ranks)
    y   = [np.mean(rank_rougeL[r]) for r in x]
    std = [np.std(rank_rougeL[r])  for r in x]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, y, yerr=std, marker='o', capsize=5, linewidth=2,
                color='steelblue', label='Mean ROUGE-L ± std')

    # Annotate baseline
    if BASELINE_RANK in ranks:
        bi = x.index(BASELINE_RANK)
        ax.annotate(
            f'Baseline (r{BASELINE_RANK}/a{BASELINE_ALPHA}/d{BASELINE_DROPOUT})',
            (x[bi], y[bi]),
            textcoords='offset points', xytext=(12, -22),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=8, color='red'
        )

    max_std = max(std) if std else 0.01
    for xi, yi, r in zip(x, y, x):
        ax.annotate(f'n={len(rank_rougeL[r])}',
                    (xi, yi + max_std * 0.4 + 0.005),
                    ha='center', fontsize=8, color='gray')

    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('ROUGE-L (Mean over Dropout Settings)')
    ax.set_title('Exp7: UML Expert — ROUGE-L vs LoRA Rank')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = plots_dir / 'rank_vs_rougeL.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


def plot_all_configs_bar(config_results, exp_dir):
    """
    Horizontal bar chart of all 12 configs sorted by ROUGE-L.

    This provides the per-config granularity that the rank-averaged line
    chart cannot show, and is suitable for direct inclusion in a paper table
    or appendix figure.
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not config_results:
        return

    # Sort ascending so the best config appears at the top
    sorted_items = sorted(
        config_results.items(),
        key=lambda kv: kv[1].get('generation_quality', {}).get('rougeL', 0)
    )

    labels = []
    values = []
    colors = []

    baseline_rougeL = config_results.get(
        (BASELINE_RANK, BASELINE_ALPHA, BASELINE_DROPOUT), {}
    ).get('generation_quality', {}).get('rougeL', 0)

    for (rank, alpha, dropout), m in sorted_items:
        rougeL = m.get('generation_quality', {}).get('rougeL', 0)
        labels.append(_config_name(rank, alpha, dropout))
        values.append(rougeL)
        if _is_baseline(rank, alpha, dropout):
            colors.append('#aec6cf')       # baseline: light blue
        elif rougeL > baseline_rougeL:
            colors.append('#77dd77')       # better than baseline: green
        else:
            colors.append('#ff9999')       # worse than baseline: red

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.55)))
    bars = ax.barh(labels, values, color=colors, edgecolor='gray', height=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    # Baseline reference line
    if baseline_rougeL > 0:
        ax.axvline(baseline_rougeL, color='steelblue', linestyle='--',
                   linewidth=1.5, label=f'Baseline ROUGE-L = {baseline_rougeL:.4f}')
        ax.legend(fontsize=8)

    ax.set_xlabel('ROUGE-L')
    ax.set_title('Exp7: UML Expert — All Configs ROUGE-L Comparison\n'
                 '(green = better than baseline, red = worse, blue = baseline)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = plots_dir / 'all_configs_rougeL.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


def plot_heatmap_dropout_alpha(config_results, exp_dir, fixed_rank=32):
    """
    Heatmap: dropout × alpha ROUGE-L for a fixed rank.
    Mirrors exp4's heatmap.  fixed_rank=32 has the most configurations (3).
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    rank_configs = {
        (a, d): m
        for (r, a, d), m in config_results.items()
        if r == fixed_rank
    }
    if not rank_configs:
        logger.warning(f'没有 rank={fixed_rank} 的结果，跳过热图')
        return

    unique_alphas   = sorted(set(a for a, _ in rank_configs))
    unique_dropouts = sorted(set(d for _, d in rank_configs))
    matrix = np.zeros((len(unique_dropouts), len(unique_alphas)))
    for i, d in enumerate(unique_dropouts):
        for j, a in enumerate(unique_alphas):
            matrix[i, j] = rank_configs.get((a, d), {}).get(
                'generation_quality', {}).get('rougeL', 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='ROUGE-L')
    ax.set_xticks(range(len(unique_alphas)))
    ax.set_yticks(range(len(unique_dropouts)))
    ax.set_xticklabels([str(a) for a in unique_alphas])
    ax.set_yticklabels([str(d) for d in unique_dropouts])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Dropout')
    ax.set_title(f'Exp7: UML ROUGE-L Heatmap (rank={fixed_rank})')
    for i in range(len(unique_dropouts)):
        for j in range(len(unique_alphas)):
            ax.text(j, i, f'{matrix[i, j]:.3f}',
                    ha='center', va='center', fontsize=9)
    plt.tight_layout()
    path = plots_dir / f'heatmap_rank{fixed_rank}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'热图已保存: {path}')


def plot_dropout_effect(config_results, exp_dir):
    """
    Line chart: for each rank that has multiple dropout settings,
    ROUGE-L vs dropout.
    Reveals whether UML is more regularisation-sensitive than Text.
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    rank_dropout_rouge = defaultdict(dict)
    for (rank, alpha, dropout), m in config_results.items():
        rank_dropout_rouge[rank][dropout] = (
            m.get('generation_quality', {}).get('rougeL', 0)
        )

    all_dropouts = sorted(set(d for _, _, d in CONFIGS))
    ranks_with_multi = sorted(
        r for r, dd in rank_dropout_rouge.items() if len(dd) > 1
    )

    if not ranks_with_multi:
        logger.warning('没有足够的多-dropout配置来绘制 dropout 影响图')
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for rank in ranks_with_multi:
        dd = rank_dropout_rouge[rank]
        xs = [d for d in all_dropouts if d in dd]
        ys = [dd[d] for d in xs]
        ax.plot(xs, ys, marker='o', linewidth=2, label=f'rank={rank}')

    ax.set_xlabel('Dropout')
    ax.set_ylabel('ROUGE-L')
    ax.set_title('Exp7: UML Expert — Dropout Effect per Rank')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = plots_dir / 'dropout_effect_per_rank.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


def plot_uml_vs_text_transfer(config_results, exp_dir):
    """
    Bar chart comparing the text-transfer baseline (r64, applied from Exp4)
    against the best UML-specific config found in Exp7.

    Directly answers: does domain-specific hyperparameter search beat
    cross-modal parameter transfer?

    Edge cases handled:
      - If best config == baseline: show a single bar with a note.
      - If best config worse than baseline: still display both with Δ<0.
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    scores = {
        key: m.get('generation_quality', {}).get('rougeL', 0)
        for key, m in config_results.items()
    }
    if not scores:
        return

    baseline_key   = (BASELINE_RANK, BASELINE_ALPHA, BASELINE_DROPOUT)
    baseline_score = scores.get(baseline_key, 0)
    best_key       = max(scores, key=lambda k: scores[k])
    best_score     = scores[best_key]

    labels = [f'Text-Transfer Baseline\n(r{BASELINE_RANK},a{BASELINE_ALPHA},d{BASELINE_DROPOUT})']
    values = [baseline_score]
    colors = ['#aec6cf']

    if best_key != baseline_key:
        labels.append(f'Best UML-Specific\n({_config_name(*best_key)})')
        values.append(best_score)
        colors.append('#77dd77' if best_score >= baseline_score else '#ff9999')

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 3), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='gray', width=0.45)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    if len(values) == 2:
        delta = values[1] - values[0]
        sign  = '+' if delta >= 0 else ''
        base  = max(values[0], 1e-9)
        ax.annotate(
            f'Δ = {sign}{delta:.4f}  ({sign}{delta / base * 100:.1f}%)',
            xy=(0.5, 0.93), xycoords='axes fraction',
            ha='center', fontsize=10,
            color='darkgreen' if delta >= 0 else 'crimson'
        )
    else:
        # best == baseline case
        ax.annotate(
            'Baseline is already the best config found',
            xy=(0.5, 0.93), xycoords='axes fraction',
            ha='center', fontsize=10, color='steelblue'
        )

    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel('ROUGE-L')
    ax.set_title('Exp7: Cross-Modal Transfer vs UML-Specific Optimisation')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = plots_dir / 'uml_vs_text_transfer.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'对比图已保存: {path}')


# =========================================================================
# Main
# =========================================================================

def run(args):
    logger.info('=' * 80)
    logger.info('实验7: UML专家 LoRA超参数优化')
    logger.info(f'基线: rank={BASELINE_RANK}, alpha={BASELINE_ALPHA}, '
                f'dropout={BASELINE_DROPOUT}  (复用 lora_moe UML 检查点)')
    logger.info(f'配置总数: {len(CONFIGS)}')
    logger.info('=' * 80)

    # UMLDatasetLoader uses load_csv_file() (singular),
    # unlike TextDatasetLoader.load_csv_files() (plural) used in exp4
    logger.info('加载UML数据集...')
    all_data = UMLDatasetLoader().load_csv_file()
    _, _, test_data = split_dataset_for_expert(all_data, 'uml')
    logger.info(f'测试集样本数: {len(test_data)}')

    results = {
        'experiment': 'exp7_uml_hyperparameter_optimization',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'baseline': {
            'rank':    BASELINE_RANK,
            'alpha':   BASELINE_ALPHA,
            'dropout': BASELINE_DROPOUT,
        },
        'configs': [],
    }

    config_results = {}

    for rank, alpha, dropout in CONFIGS:
        cfg_name = _config_name(rank, alpha, dropout)
        logger.info(f'\n--- 配置: {cfg_name} ---')

        if getattr(args, 'only_missing', False) and _is_full_run_cache(
                CACHE_DIR, f'{cfg_name}_predictions.json'):
            logger.info(f'{cfg_name}: 缓存存在，跳过 (--only-missing)')
            continue

        # ---- Training ----
        try:
            train_config(rank, alpha, dropout, args)
        except Exception as e:
            logger.error(f'{cfg_name}: 训练失败: {e}')
            logger.error(traceback.format_exc())

        # ---- Inference + Metrics ----
        try:
            cached = run_inference(rank, alpha, dropout, test_data, args)
            if cached is None:
                logger.warning(f'{cfg_name}: 已跳过（推理失败或检查点缺失）')
                continue

            preds = [s['prediction'] for s in cached['samples']]
            refs  = [s['reference']  for s in cached['samples']]
            m = compute_all_metrics(preds, refs, use_bertscore=not args.no_bertscore)

            q = m.get('generation_quality', {})
            b = m.get('binary_classification', {})

            config_entry = {
                'name':                  cfg_name,
                'rank':                  rank,
                'alpha':                 alpha,
                'dropout':               dropout,
                'is_baseline':           _is_baseline(rank, alpha, dropout),
                'n_samples':             len(preds),
                'generation_quality':    q,
                'binary_classification': b,
            }
            results['configs'].append(config_entry)
            config_results[(rank, alpha, dropout)] = m

            baseline_marker = '  [BASELINE]' if _is_baseline(rank, alpha, dropout) else ''
            logger.info(
                f'{cfg_name}{baseline_marker}: '
                f'ROUGE-L={q.get("rougeL", 0):.4f}  '
                f'BLEU={q.get("bleu", 0):.4f}  '
                f'F1={b.get("f1_score", 0):.4f}'
            )
        except Exception as e:
            logger.error(f'{cfg_name}: 评估失败: {e}')
            logger.error(traceback.format_exc())

    # ---- Best config + delta vs baseline ----
    if results['configs']:
        best = max(results['configs'],
                   key=lambda c: c['generation_quality'].get('rougeL', 0))
        results['best_config'] = best

        baseline_entry = next(
            (c for c in results['configs'] if c.get('is_baseline')), None
        )
        best_rougeL = best['generation_quality'].get('rougeL', 0)
        logger.info(f'\n最优配置: {best["name"]}  (ROUGE-L={best_rougeL:.4f})')

        if baseline_entry:
            base_rougeL = baseline_entry['generation_quality'].get('rougeL', 0)
            delta = best_rougeL - base_rougeL
            sign  = '+' if delta >= 0 else ''
            logger.info(
                f'相对 text-transfer 基线 (r{BASELINE_RANK}) 变化: '
                f'{sign}{delta:.4f}  ({sign}{delta / max(base_rougeL, 1e-9) * 100:.1f}%)'
            )
            results['baseline_delta'] = delta

    # ---- Save results ----
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    # ---- Plots ----
    try:
        plot_rank_vs_rouge(config_results, EXP_DIR)
        plot_all_configs_bar(config_results, EXP_DIR)
        plot_heatmap_dropout_alpha(config_results, EXP_DIR, fixed_rank=32)
        plot_dropout_effect(config_results, EXP_DIR)
        plot_uml_vs_text_transfer(config_results, EXP_DIR)
    except Exception as e:
        logger.warning(f'绘图失败: {e}')
        logger.warning(traceback.format_exc())

    # ---- Summary table: sorted by ROUGE-L desc, with Δ vs baseline column ----
    baseline_rougeL = next(
        (c['generation_quality'].get('rougeL', 0)
         for c in results['configs'] if c.get('is_baseline')),
        0.0
    )

    logger.info('\n' + '=' * 90)
    logger.info('配置对比汇总（按ROUGE-L降序）')
    logger.info('=' * 90)
    logger.info(
        f'{"配置名称":<38} {"ROUGE-L":>8} {"Δ vs base":>10} {"BLEU":>8} {"F1":>8}  备注'
    )
    logger.info('-' * 90)
    for c in sorted(results['configs'],
                    key=lambda x: x['generation_quality'].get('rougeL', 0),
                    reverse=True):
        q     = c.get('generation_quality', {})
        b     = c.get('binary_classification', {})
        rl    = q.get('rougeL', 0)
        delta = rl - baseline_rougeL
        sign  = '+' if delta >= 0 else ''
        star  = ' ★' if c == results.get('best_config') else ''
        base  = ' [baseline]' if c.get('is_baseline') else ''
        logger.info(
            f'{c["name"]:<38} {rl:>8.4f} {sign}{delta:>9.4f} '
            f'{q.get("bleu", 0):>8.4f} {b.get("f1_score", 0):>8.4f}'
            f'{star}{base}'
        )

    logger.info(f'\n结果已保存至: {EXP_DIR}')


# =========================================================================
# CLI
# =========================================================================

def _delete_caches_for_rerun(rerun_configs_str):
    """Delete inference cache files for the specified config names."""
    names = [n.strip() for n in rerun_configs_str.split(',') if n.strip()]
    for name in names:
        cache_file = CACHE_DIR / f'{name}_predictions.json'
        if cache_file.exists():
            cache_file.unlink()
            logger.info(f'[rerun] 已删除缓存: {cache_file}')
        else:
            logger.info(f'[rerun] 未找到缓存，将直接执行推理: {name}')


def main():
    parser = argparse.ArgumentParser(
        description='Exp7: UML Expert LoRA hyperparameter optimization'
    )
    parser.add_argument('--force-regenerate', action='store_true',
                        help='即使缓存存在也重新推理')
    parser.add_argument('--force-retrain', action='store_true',
                        help='即使检查点存在也重新训练')
    parser.add_argument('--from-cache', action='store_true',
                        help='仅从缓存加载，不训练不推理')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='跳过BERTScore计算（加速调试）')
    parser.add_argument('--test-mode', action='store_true',
                        help='每个配置仅用10个样本（快速验证流程正确性）')
    parser.add_argument('--only-missing', action='store_true',
                        help='跳过已有完整缓存的配置（test-mode缓存视为缺失，自动重跑）')
    parser.add_argument('--rerun-configs', type=str, default='',
                        help='强制重跑指定配置（逗号分隔），如 '
                             '"uml_r32_a64_d0.05,uml_r16_a32_d0.0"；'
                             '配合 --force-retrain 可同时从头重新训练')
    args = parser.parse_args()

    if args.from_cache:
        args.force_regenerate = False
        args.force_retrain    = False
    if args.rerun_configs:
        _delete_caches_for_rerun(args.rerun_configs)

    run(args)


if __name__ == '__main__':
    main()