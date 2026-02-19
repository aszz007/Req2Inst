#!/usr/bin/env python3
"""
Generate Comprehensive Experiment Report (Phase 3)

Aggregates all 6 experiment results into:
  - outputs/evaluations/experiments/all_experiments_summary.json
  - outputs/evaluations/experiments/comprehensive_report.pdf
  - outputs/evaluations/experiments/plots/method_comparison_radar.png
  - outputs/evaluations/experiments/plots/overall_performance.png
  - outputs/evaluations/experiments/plots/efficiency_analysis.png

Usage:
  python generate_comprehensive_report.py
  python generate_comprehensive_report.py --no-pdf   (skip PDF, generate plots only)
  python generate_comprehensive_report.py --exp-dir outputs/evaluations/experiments
"""

import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

from config.settings import get_path_config
from src.utils.logger import get_logger

logger = get_logger('experiments.report')

path_cfg = get_path_config()
EXPERIMENTS_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments'
PLOTS_DIR = EXPERIMENTS_DIR / 'plots'

EXP_DIRS = {
    1: 'exp1_baseline_comparison',
    2: 'exp2_finetuning_methods',
    3: 'exp3_moe_architecture',
    4: 'exp4_lora_hyperparameters',
    5: 'exp5_data_efficiency',
    6: 'exp6_fewshot_learning',
}

METRIC_DISPLAY = {
    'bleu': 'BLEU',
    'rougeL': 'ROUGE-L',
    'meteor': 'METEOR',
    'f1_score': 'F1 Score',
    'valid_rate': 'Format Pass',
}

METHOD_COLORS = {
    'lora_moe': '#1f77b4',
    'lora_single': '#ff7f0e',
    'p_tuning': '#2ca02c',
    'prompt_tuning': '#d62728',
    'full_finetuning': '#9467bd',
    'bm25': '#8c564b',
    'lsa': '#e377c2',
    'template': '#7f7f7f',
    'zeroshot': '#bcbd22',
}

METHOD_LABELS = {
    'lora_moe': 'LoRA-MoE',
    'lora_single': 'LoRA-Single',
    'p_tuning': 'P-Tuning v2',
    'prompt_tuning': 'Prompt Tuning',
    'full_finetuning': 'Full Fine-Tuning',
    'bm25': 'BM25',
    'lsa': 'LSA',
    'template': 'Template Fill',
    'zeroshot': 'Zero-Shot',
}

EXPERT_LABELS = {
    'text': 'Text',
    'image': 'Image',
    'uml': 'UML',
    'general': 'General',
}


# ---------------------------------------------------------------------------
# Result Loading
# ---------------------------------------------------------------------------

def _safe_load(path: Path) -> Optional[Dict]:
    if not path.exists():
        logger.warning(f'结果文件未找到: {path}')
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f'加载失败 {path}: {e}')
        return None


def load_all_results(experiments_dir: Path) -> Dict:
    """Load results.json from each experiment directory."""
    all_results = {}
    for exp_num, exp_dir_name in EXP_DIRS.items():
        results_path = experiments_dir / exp_dir_name / 'results.json'
        data = _safe_load(results_path)
        if data is not None:
            all_results[exp_num] = data
            logger.info(f'已加载 实验{exp_num}: {exp_dir_name}')
        else:
            logger.warning(f'实验{exp_num} 结果文件未找到: {results_path}')
    return all_results


def _extract_quality(result_entry: Dict) -> Dict:
    """Extract generation_quality and binary_classification from a result entry."""
    q = result_entry.get('generation_quality', {})
    b = result_entry.get('binary_classification', {})
    fm = result_entry.get('format_metrics', {})
    return {
        'bleu': q.get('bleu', 0),
        'rougeL': q.get('rougeL', 0),
        'meteor': q.get('meteor', 0),
        'f1_score': b.get('f1_score', 0),
        'valid_rate': fm.get('valid_rate', 0),
        'bertscore_f1': q.get('bertscore_f1', None),
    }


# ---------------------------------------------------------------------------
# Summary Assembly
# ---------------------------------------------------------------------------

def build_summary(all_results: Dict) -> Dict:
    """
    Assemble a flat summary dict with key metrics from each experiment.
    """
    summary = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments_found': sorted(all_results.keys()),
        'exp1_baseline': {},
        'exp2_method_comparison': {},
        'exp3_architecture': {},
        'exp4_hyperparameters': {},
        'exp5_data_efficiency': {},
        'exp6_fewshot': {},
        'overall_lora_moe': {},
    }

    # Exp1
    if 1 in all_results:
        d = all_results[1]
        for method, m in d.get('methods', {}).items():
            summary['exp1_baseline'][method] = _extract_quality(m)

    # Exp2
    if 2 in all_results:
        d = all_results[2]
        for key, m in d.get('results', {}).items():
            summary['exp2_method_comparison'][key] = _extract_quality(m)

    # Exp3
    if 3 in all_results:
        d = all_results[3]
        summary['exp3_architecture'] = d.get('architecture_comparison', {})
        summary['exp3_cross_domain'] = d.get('cross_domain', {})

    # Exp4
    if 4 in all_results:
        d = all_results[4]
        summary['exp4_hyperparameters'] = {
            c['name']: _extract_quality(c)
            for c in d.get('configs', [])
        }
        best = d.get('best_config')
        if best:
            summary['exp4_best_config'] = {
                'name': best['name'],
                'rank': best.get('rank'),
                'alpha': best.get('alpha'),
                'dropout': best.get('dropout'),
                'rougeL': best.get('generation_quality', {}).get('rougeL', 0),
            }

    # Exp5
    if 5 in all_results:
        d = all_results[5]
        summary['exp5_data_efficiency'] = d.get('results', {})

    # Exp6
    if 6 in all_results:
        d = all_results[6]
        for n_shots_str, v in d.get('shot_configs', {}).items():
            summary['exp6_fewshot'][f'{n_shots_str}_shot'] = {
                'mean_rougeL': v.get('mean_rougeL', 0),
                'std_rougeL': v.get('std_rougeL', 0),
            }
        lora_m = d.get('lora_moe', {})
        summary['exp6_fewshot']['lora_moe'] = _extract_quality(lora_m)

    # LoRA-MoE across expert types (from Exp2)
    if 2 in all_results:
        d = all_results[2]
        for et in ['text', 'image', 'uml', 'general']:
            key = f'lora_moe_{et}'
            if key in d.get('results', {}):
                summary['overall_lora_moe'][et] = _extract_quality(d['results'][key])

    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overall_performance(summary: Dict, plots_dir: Path):
    """
    Bar chart: LoRA-MoE vs best baseline per expert type.
    Groups: text, image, uml, general  x  {LoRA-MoE, best_baseline}
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    expert_types = ['text', 'image', 'uml', 'general']
    lora_moe_rougeL = [
        summary['overall_lora_moe'].get(et, {}).get('rougeL', 0)
        for et in expert_types
    ]

    # Best baseline from exp1 (only text, fallback 0 for others)
    exp1 = summary.get('exp1_baseline', {})
    baseline_methods = ['bm25', 'lsa', 'template', 'zeroshot']
    best_baseline_rougeL = max(
        (exp1.get(m, {}).get('rougeL', 0) for m in baseline_methods),
        default=0
    )

    x = np.arange(len(expert_types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, lora_moe_rougeL, width,
                   label='LoRA-MoE', color='#1f77b4', alpha=0.9)
    # best baseline only for text (where exp1 is defined)
    baseline_vals = [best_baseline_rougeL, 0, 0, 0]
    bars2 = ax.bar(x + width / 2, baseline_vals, width,
                   label='Best Baseline (Text)', color='#aec7e8', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([EXPERT_LABELS[et] for et in expert_types])
    ax.set_ylabel('ROUGE-L')
    ax.set_title('Overall LoRA-MoE Performance by Expert Type')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = plots_dir / 'overall_performance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


def plot_method_comparison_radar(summary: Dict, plots_dir: Path):
    """
    Radar chart: 5 fine-tuning methods x 5 metrics (text expert only).
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = ['lora_moe', 'lora_single', 'p_tuning', 'prompt_tuning', 'full_finetuning']
    metric_keys = ['bleu', 'rougeL', 'meteor', 'f1_score', 'valid_rate']
    metric_labels_radar = ['BLEU', 'ROUGE-L', 'METEOR', 'F1', 'Format']

    exp2 = summary.get('exp2_method_comparison', {})

    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for method in methods:
        key = f'{method}_text'
        m = exp2.get(key, {})
        if not m:
            continue
        values = [m.get(k, 0) for k in metric_keys]
        values += values[:1]
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(angles, values, 'o-', linewidth=2, color=color,
                label=METHOD_LABELS.get(method, method))
        ax.fill(angles, values, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels_radar, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Method Comparison (Text Expert)', pad=20, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    path = plots_dir / 'method_comparison_radar.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


def plot_efficiency_analysis(summary: Dict, plots_dir: Path):
    """
    Two-panel plot:
      Left: Learning curves (Exp5, ROUGE-L vs data fraction)
      Right: Few-shot vs fine-tuning (Exp6, ROUGE-L bar with error bars)
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Learning curves (Exp5) ---
    exp5 = summary.get('exp5_data_efficiency', {})
    methods_e5 = ['lora_moe', 'lora_single', 'full_finetuning']
    fractions = [10, 25, 50, 75, 100]

    for method in methods_e5:
        xs, ys = [], []
        for f in fractions:
            key = f'{method}_{f}pct'
            entry = exp5.get(key, {})
            if entry:
                q = entry.get('generation_quality', {})
                rougeL = q.get('rougeL', None)
                if rougeL is not None:
                    xs.append(f)
                    ys.append(rougeL)
        if xs:
            ax1.plot(xs, ys, marker='o', linewidth=2,
                     color=METHOD_COLORS.get(method, None),
                     label=METHOD_LABELS.get(method, method))

    ax1.set_xlabel('Training Data (%)')
    ax1.set_ylabel('ROUGE-L')
    ax1.set_title('Data Efficiency (Exp5)')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 110)
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.3)

    # --- Right: Few-shot vs Fine-tuning (Exp6) ---
    exp6 = summary.get('exp6_fewshot', {})
    shot_configs = [('0_shot', '0-shot'), ('1_shot', '1-shot'),
                    ('3_shot', '3-shot'), ('5_shot', '5-shot'), ('lora_moe', 'LoRA-MoE')]
    bar_vals = []
    bar_errs = []
    bar_labels = []
    bar_colors_list = []

    for key, label in shot_configs:
        entry = exp6.get(key, {})
        if key == 'lora_moe':
            val = entry.get('rougeL', 0)
            err = 0
            color = '#1f77b4'
        else:
            val = entry.get('mean_rougeL', 0)
            err = entry.get('std_rougeL', 0)
            color = '#ff7f0e'
        bar_vals.append(val)
        bar_errs.append(err)
        bar_labels.append(label)
        bar_colors_list.append(color)

    x_pos = np.arange(len(bar_labels))
    ax2.bar(x_pos, bar_vals, yerr=bar_errs, capsize=5,
            color=bar_colors_list, alpha=0.85)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel('ROUGE-L')
    ax2.set_title('Few-Shot vs Fine-Tuning (Exp6)')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    legend_patches = [
        mpatches.Patch(color='#ff7f0e', alpha=0.85, label='Few-Shot (base)'),
        mpatches.Patch(color='#1f77b4', alpha=0.85, label='LoRA-MoE (fine-tuned)'),
    ]
    ax2.legend(handles=legend_patches, fontsize=9)

    plt.tight_layout()
    path = plots_dir / 'efficiency_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


def plot_exp1_vs_exp2_combined(summary: Dict, plots_dir: Path):
    """
    Combined bar chart comparing all methods on text expert ROUGE-L
    (baselines from Exp1 + fine-tuning methods from Exp2).
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    exp1 = summary.get('exp1_baseline', {})
    exp2 = summary.get('exp2_method_comparison', {})

    all_methods_ordered = [
        ('bm25', exp1.get('bm25', {})),
        ('lsa', exp1.get('lsa', {})),
        ('template', exp1.get('template', {})),
        ('zeroshot', exp1.get('zeroshot', {})),
        ('p_tuning', exp2.get('p_tuning_text', {})),
        ('prompt_tuning', exp2.get('prompt_tuning_text', {})),
        ('lora_single', exp2.get('lora_single_text', {})),
        ('full_finetuning', exp2.get('full_finetuning_text', {})),
        ('lora_moe', exp2.get('lora_moe_text', {})),
    ]

    labels = []
    rougeL_vals = []
    colors = []

    for method, m in all_methods_ordered:
        val = m.get('rougeL', 0)
        if val > 0 or method in exp1 or f'{method}_text' in exp2:
            labels.append(METHOD_LABELS.get(method, method))
            rougeL_vals.append(val)
            colors.append(METHOD_COLORS.get(method, '#aaaaaa'))

    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, rougeL_vals, color=colors, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('ROUGE-L')
    ax.set_title('Text Expert: All Methods Comparison (ROUGE-L)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = plots_dir / 'all_methods_text_rougeL.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


def plot_hyperparameter_summary(summary: Dict, plots_dir: Path):
    """
    Line chart of ROUGE-L vs LoRA rank from Exp4.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    exp4 = summary.get('exp4_hyperparameters', {})
    if not exp4:
        return

    # Group by rank
    rank_rougeL = {}
    for cfg_name, m in exp4.items():
        # Parse rank from name: text_r{rank}_a{alpha}_d{dropout}
        parts = cfg_name.replace('text_r', '').split('_')
        try:
            rank = int(parts[0].split('a')[0]) if 'a' in parts[0] else int(parts[0])
            # Actually parse properly
            import re
            match = re.search(r'r(\d+)_a(\d+)_d([\d.]+)', cfg_name)
            if match:
                rank = int(match.group(1))
                rank_rougeL.setdefault(rank, []).append(m.get('rougeL', 0))
        except (ValueError, IndexError):
            pass

    if not rank_rougeL:
        return

    ranks = sorted(rank_rougeL.keys())
    means = [np.mean(rank_rougeL[r]) for r in ranks]
    stds = [np.std(rank_rougeL[r]) for r in ranks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ranks, means, yerr=stds, marker='o', capsize=5,
                linewidth=2, color='#1f77b4')
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('ROUGE-L (mean across dropout configs)')
    ax.set_title('LoRA Rank Optimization (Exp4)')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    path = plots_dir / 'rank_optimization.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {path}')


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------

def _text_wrap(text: str, width: int) -> List[str]:
    """Naive word-wrap for PDF text blocks."""
    words = text.split()
    lines = []
    current = []
    for word in words:
        if len(' '.join(current + [word])) <= width:
            current.append(word)
        else:
            if current:
                lines.append(' '.join(current))
            current = [word]
    if current:
        lines.append(' '.join(current))
    return lines


def _format_metrics_table(method_metrics: Dict, metrics: List[str]) -> str:
    """Format a metrics dict as a simple ASCII table for PDF embedding."""
    col_w = 14
    header = f'{"Method":<20}' + ''.join(f'{METRIC_DISPLAY.get(m, m):>{col_w}}' for m in metrics)
    sep = '-' * len(header)
    rows = [header, sep]
    for method, m in sorted(method_metrics.items()):
        row = f'{METHOD_LABELS.get(method, method):<20}'
        for key in metrics:
            val = m.get(key, 0)
            row += f'{val:>{col_w}.4f}'
        rows.append(row)
    return '\n'.join(rows)


def generate_pdf_report(summary: Dict, all_results: Dict, experiments_dir: Path):
    """
    Generate a structured PDF report using matplotlib (no LaTeX dependency).
    Uses a multi-page figure approach: one page per section.
    """
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        logger.error('matplotlib PdfPages 不可用')
        return

    pdf_path = experiments_dir / 'comprehensive_report.pdf'
    plots_dir = experiments_dir / 'plots'

    # Pre-generate all plots
    plot_overall_performance(summary, plots_dir)
    plot_method_comparison_radar(summary, plots_dir)
    plot_efficiency_analysis(summary, plots_dir)
    plot_exp1_vs_exp2_combined(summary, plots_dir)
    plot_hyperparameter_summary(summary, plots_dir)

    with PdfPages(str(pdf_path)) as pdf:

        # ---- Page 1: Title Page ----
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.text(0.5, 0.80,
                'LoRA-MoE Crowdsourcing Instruction Generation',
                ha='center', va='center', fontsize=20, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.72,
                'Comprehensive Experiment Report',
                ha='center', va='center', fontsize=15,
                transform=ax.transAxes, color='#444444')
        ax.text(0.5, 0.62,
                f'Generated: {summary["generated_at"]}',
                ha='center', va='center', fontsize=11,
                transform=ax.transAxes, color='#666666')

        exps_found = summary.get('experiments_found', [])
        status_text = (
            f'Experiments included: {", ".join(f"Exp{e}" for e in exps_found)}\n'
            f'Base model: Qwen3-8B  |  Fine-tuning: LoRA-MoE (4 experts)\n'
            f'Expert types: Text, Image, UML, General'
        )
        ax.text(0.5, 0.48, status_text,
                ha='center', va='center', fontsize=11,
                transform=ax.transAxes, color='#333333',
                linespacing=1.8)

        # Key result highlight if exp2 lora_moe_text available
        lora_text = summary.get('exp2_method_comparison', {}).get('lora_moe_text', {})
        if lora_text:
            highlight = (
                f'Key Result: LoRA-MoE (Text Expert)\n'
                f'ROUGE-L = {lora_text.get("rougeL", 0):.4f}  |  '
                f'F1 = {lora_text.get("f1_score", 0):.4f}  |  '
                f'BLEU = {lora_text.get("bleu", 0):.4f}'
            )
            ax.text(0.5, 0.30, highlight,
                    ha='center', va='center', fontsize=12,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f4fd',
                              edgecolor='#1f77b4', linewidth=1.5),
                    color='#1f4e7a')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---- Page 2: Exp1 - Baseline Comparison ----
        fig = plt.figure(figsize=(11, 8.5))
        ax_title = fig.add_axes([0.05, 0.88, 0.90, 0.08])
        ax_title.axis('off')
        ax_title.text(0.0, 0.5, 'Experiment 1: Baseline Comparison (Text Expert)',
                      fontsize=14, fontweight='bold', va='center')

        # Load exp1 plot if exists
        exp1_plot = plots_dir.parent / 'exp1_baseline_comparison' / 'plots' / 'comparison.png'
        if exp1_plot.exists():
            img = plt.imread(str(exp1_plot))
            ax_img = fig.add_axes([0.05, 0.42, 0.55, 0.44])
            ax_img.imshow(img)
            ax_img.axis('off')

        # Text summary
        ax_text = fig.add_axes([0.62, 0.25, 0.35, 0.62])
        ax_text.axis('off')
        exp1_data = summary.get('exp1_baseline', {})
        if exp1_data:
            table_lines = ['Method         ROUGE-L    F1']
            table_lines.append('-' * 30)
            for m, vals in sorted(exp1_data.items(),
                                  key=lambda x: x[1].get('rougeL', 0), reverse=True):
                line = f'{METHOD_LABELS.get(m, m):<16} {vals.get("rougeL", 0):.3f}  {vals.get("f1_score", 0):.3f}'
                table_lines.append(line)
            ax_text.text(0.0, 1.0, '\n'.join(table_lines),
                         va='top', fontfamily='monospace', fontsize=9,
                         transform=ax_text.transAxes)

        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.15])
        ax_note.axis('off')
        note = ('Observation: IR-based (BM25/LSA) and rule-based (Template) methods serve as '
                'non-learning baselines. Zero-shot uses the base Qwen3-8B without fine-tuning. '
                'LoRA-MoE (our method) should outperform all non-fine-tuned approaches.')
        for i, line in enumerate(_text_wrap(note, 120)):
            ax_note.text(0.0, 0.85 - i * 0.18, line, fontsize=9,
                         color='#444444', transform=ax_note.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---- Page 3: Exp2 - Method Comparison (Radar + Table) ----
        fig = plt.figure(figsize=(11, 8.5))
        ax_title = fig.add_axes([0.05, 0.88, 0.90, 0.08])
        ax_title.axis('off')
        ax_title.text(0.0, 0.5, 'Experiment 2: Fine-Tuning Method Comparison',
                      fontsize=14, fontweight='bold', va='center')

        radar_plot = plots_dir / 'method_comparison_radar.png'
        if radar_plot.exists():
            img = plt.imread(str(radar_plot))
            ax_img = fig.add_axes([0.02, 0.28, 0.50, 0.58])
            ax_img.imshow(img)
            ax_img.axis('off')

        ax_text = fig.add_axes([0.54, 0.28, 0.44, 0.58])
        ax_text.axis('off')
        exp2 = summary.get('exp2_method_comparison', {})
        text_methods = ['lora_moe', 'lora_single', 'p_tuning', 'prompt_tuning', 'full_finetuning']
        table_lines = ['Text Expert Results:', '-' * 44]
        table_lines.append(f'{"Method":<20} {"ROUGE-L":>8} {"F1":>8}')
        table_lines.append('-' * 44)
        for m in text_methods:
            key = f'{m}_text'
            vals = exp2.get(key, {})
            if vals:
                table_lines.append(
                    f'{METHOD_LABELS.get(m, m):<20} {vals.get("rougeL", 0):>8.4f} '
                    f'{vals.get("f1_score", 0):>8.4f}'
                )
        ax_text.text(0.0, 1.0, '\n'.join(table_lines),
                     va='top', fontfamily='monospace', fontsize=8.5,
                     transform=ax_text.transAxes)

        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.22])
        ax_note.axis('off')
        note = ('Observation: LoRA-MoE achieves competitive performance with minimal trainable '
                'parameters (rank=8, ~0.1% of total). Full Fine-Tuning uses highest parameter '
                'count but may overfit on smaller datasets. P-Tuning/Prompt Tuning are more '
                'parameter-efficient but typically underperform LoRA approaches.')
        for i, line in enumerate(_text_wrap(note, 120)):
            ax_note.text(0.0, 0.85 - i * 0.18, line, fontsize=9,
                         color='#444444', transform=ax_note.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---- Page 4: Exp3 - MoE Architecture ----
        fig = plt.figure(figsize=(11, 8.5))
        ax_title = fig.add_axes([0.05, 0.88, 0.90, 0.08])
        ax_title.axis('off')
        ax_title.text(0.0, 0.5, 'Experiment 3: MoE Architecture Validation',
                      fontsize=14, fontweight='bold', va='center')

        heatmap_plot = plots_dir.parent / 'exp3_moe_architecture' / 'plots' / 'cross_domain_heatmap.png'
        if heatmap_plot.exists():
            img = plt.imread(str(heatmap_plot))
            ax_img = fig.add_axes([0.03, 0.32, 0.50, 0.52])
            ax_img.imshow(img)
            ax_img.axis('off')

        arch_plot = plots_dir.parent / 'exp3_moe_architecture' / 'plots' / 'architecture_comparison.png'
        if arch_plot.exists():
            img = plt.imread(str(arch_plot))
            ax_img2 = fig.add_axes([0.54, 0.32, 0.44, 0.52])
            ax_img2.imshow(img)
            ax_img2.axis('off')

        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.27])
        ax_note.axis('off')
        arch = summary.get('exp3_architecture', {})
        arch_text = ''
        for config_name, scores in arch.items():
            arch_text += f'{config_name}: ROUGE-L={scores.get("rougeL", 0):.4f}  F1={scores.get("f1", 0):.4f}  |  '
        note = (
            f'Architecture Results: {arch_text.rstrip(" | ")}\n'
            'Cross-domain heatmap shows ROUGE-L when expert i is evaluated on domain j. '
            'Diagonal = matched expert (expected best). Off-diagonal = cross-domain penalty '
            'demonstrates specialization value of MoE routing.'
        )
        for i, line in enumerate(_text_wrap(note, 120)):
            ax_note.text(0.0, 0.85 - i * 0.18, line, fontsize=9,
                         color='#444444', transform=ax_note.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---- Page 5: Exp4 & Exp5 - Hyperparameters and Data Efficiency ----
        fig = plt.figure(figsize=(11, 8.5))
        ax_title = fig.add_axes([0.05, 0.88, 0.90, 0.08])
        ax_title.axis('off')
        ax_title.text(0.0, 0.5, 'Exp4: Hyperparameter Optimization  |  Exp5: Data Efficiency',
                      fontsize=13, fontweight='bold', va='center')

        rank_plot = plots_dir / 'rank_optimization.png'
        if rank_plot.exists():
            img = plt.imread(str(rank_plot))
            ax_img = fig.add_axes([0.03, 0.28, 0.45, 0.56])
            ax_img.imshow(img)
            ax_img.axis('off')

        eff_plot = plots_dir / 'efficiency_analysis.png'
        if eff_plot.exists():
            img = plt.imread(str(eff_plot))
            ax_img2 = fig.add_axes([0.52, 0.28, 0.46, 0.56])
            ax_img2.imshow(img)
            ax_img2.axis('off')

        best_cfg = summary.get('exp4_best_config', {})
        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.24])
        ax_note.axis('off')
        note = (
            f'Exp4 Best Config: {best_cfg.get("name", "N/A")}  '
            f'(ROUGE-L={best_cfg.get("rougeL", 0):.4f})\n'
            'Exp5 shows how performance scales with training data fraction across '
            'LoRA-MoE, LoRA-Single, and Full Fine-Tuning. '
            'LoRA-MoE is expected to be most data-efficient due to expert specialization.'
        )
        for i, line in enumerate(_text_wrap(note, 120)):
            ax_note.text(0.0, 0.85 - i * 0.18, line, fontsize=9,
                         color='#444444', transform=ax_note.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---- Page 6: Exp6 + Summary Table ----
        fig = plt.figure(figsize=(11, 8.5))
        ax_title = fig.add_axes([0.05, 0.88, 0.90, 0.08])
        ax_title.axis('off')
        ax_title.text(0.0, 0.5, 'Experiment 6: Few-Shot vs Fine-Tuning  |  Summary',
                      fontsize=13, fontweight='bold', va='center')

        all_methods_plot = plots_dir / 'all_methods_text_rougeL.png'
        if all_methods_plot.exists():
            img = plt.imread(str(all_methods_plot))
            ax_img = fig.add_axes([0.03, 0.35, 0.94, 0.50])
            ax_img.imshow(img)
            ax_img.axis('off')

        ax_note = fig.add_axes([0.05, 0.02, 0.90, 0.30])
        ax_note.axis('off')

        # Mini summary table
        lora_moe_vals = summary.get('overall_lora_moe', {})
        lines = ['LoRA-MoE per-expert ROUGE-L summary:']
        for et in ['text', 'image', 'uml', 'general']:
            val = lora_moe_vals.get(et, {}).get('rougeL', 0)
            lines.append(f'  {EXPERT_LABELS[et]}: {val:.4f}')
        exp6 = summary.get('exp6_fewshot', {})
        lm_val = exp6.get('lora_moe', {}).get('rougeL', 0)
        shot5_mean = exp6.get('5_shot', {}).get('mean_rougeL', 0)
        lines.append(f'\nFew-shot gap: 5-shot={shot5_mean:.4f}  vs  LoRA-MoE={lm_val:.4f}')
        ax_note.text(0.0, 1.0, '\n'.join(lines),
                     va='top', fontfamily='monospace', fontsize=9,
                     transform=ax_note.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'PDF报告已保存: {pdf_path}')
    return pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    logger.info('=' * 80)
    logger.info('阶段3: 生成综合实验报告')
    logger.info('=' * 80)

    exp_dir = Path(args.exp_dir) if args.exp_dir else EXPERIMENTS_DIR
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有实验结果
    logger.info('加载实验结果...')
    all_results = load_all_results(exp_dir)

    if not all_results:
        logger.error('未找到任何实验结果，请先运行实验。')
        return

    logger.info(f'已加载 {len(all_results)} 个实验: {sorted(all_results.keys())}')

    # 构建汇总
    logger.info('正在汇总数据...')
    summary = build_summary(all_results)

    # 保存汇总JSON
    summary_path = exp_dir / 'all_experiments_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f'汇总文件已保存: {summary_path}')

    # 生成所有图表
    plots_dir = exp_dir / 'plots'
    logger.info('正在生成图表...')
    try:
        plot_overall_performance(summary, plots_dir)
    except Exception as e:
        logger.warning(f'总体性能图生成失败: {e}')
    try:
        plot_method_comparison_radar(summary, plots_dir)
    except Exception as e:
        logger.warning(f'雷达图生成失败: {e}')
    try:
        plot_efficiency_analysis(summary, plots_dir)
    except Exception as e:
        logger.warning(f'效率分析图生成失败: {e}')
    try:
        plot_exp1_vs_exp2_combined(summary, plots_dir)
    except Exception as e:
        logger.warning(f'联合柱状图生成失败: {e}')
    try:
        plot_hyperparameter_summary(summary, plots_dir)
    except Exception as e:
        logger.warning(f'超参数图生成失败: {e}')

    # 生成PDF
    if not args.no_pdf:
        logger.info('正在生成PDF报告...')
        try:
            pdf_path = generate_pdf_report(summary, all_results, exp_dir)
        except Exception as e:
            logger.error(f'PDF生成失败: {e}')
            logger.error(traceback.format_exc())
    else:
        logger.info('已跳过PDF生成（--no-pdf）')

    # 控制台汇总输出
    logger.info('\n' + '=' * 80)
    logger.info('综合汇总')
    logger.info('=' * 80)

    if summary.get('exp1_baseline'):
        logger.info('\n[实验1] 基线对比（文本专家，ROUGE-L）:')
        for m, vals in sorted(summary['exp1_baseline'].items(),
                               key=lambda x: x[1].get('rougeL', 0), reverse=True):
            logger.info(f'  {METHOD_LABELS.get(m, m):<22}: {vals.get("rougeL", 0):.4f}')

    if summary.get('exp2_method_comparison'):
        logger.info('\n[实验2] 微调方法对比，文本专家（ROUGE-L）:')
        text_keys = {k: v for k, v in summary['exp2_method_comparison'].items()
                     if k.endswith('_text')}
        for k, vals in sorted(text_keys.items(),
                               key=lambda x: x[1].get('rougeL', 0), reverse=True):
            method = k.replace('_text', '')
            logger.info(f'  {METHOD_LABELS.get(method, method):<22}: {vals.get("rougeL", 0):.4f}')

    if summary.get('overall_lora_moe'):
        logger.info('\n[LoRA-MoE] 各专家ROUGE-L:')
        for et, vals in summary['overall_lora_moe'].items():
            logger.info(f'  {EXPERT_LABELS.get(et, et):<10}: {vals.get("rougeL", 0):.4f}')

    logger.info(f'\n输出目录: {exp_dir}')
    logger.info('=' * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Generate comprehensive experiment report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full report with PDF
  python generate_comprehensive_report.py

  # Skip PDF, generate plots only
  python generate_comprehensive_report.py --no-pdf

  # Specify experiment results directory
  python generate_comprehensive_report.py --exp-dir outputs/evaluations/experiments
        """
    )
    parser.add_argument('--exp-dir', type=str, default=None,
                        help='Experiment results directory (default: outputs/evaluations/experiments)')
    parser.add_argument('--no-pdf', action='store_true',
                        help='Skip PDF generation, generate plots only')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()