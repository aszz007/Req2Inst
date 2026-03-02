#!/usr/bin/env python3
"""
Experiment 8: Inference Efficiency Benchmark

Measure and compare the inference efficiency of all methods:
  - Model loading time
  - Per-sample latency (batch_size=1): mean, median, P95, std
  - Batch throughput (samples/sec) at method-optimal batch size
  - Peak GPU memory during inference (MB)
  - Adapter / index size on disk (MB)

Methods benchmarked:
  CPU baselines:  bm25, lsa, template
  GPU methods:    zeroshot, lora_moe, lora_single, p_tuning, prompt_tuning, full_finetuning

All GPU methods are benchmarked on the text expert test set for fair comparison.

Protocol:
  1. Clear GPU cache
  2. Record baseline GPU memory
  3. Load model → record load time + peak memory after load
  4. Warmup: N_WARMUP samples (discarded)
  5. Latency: N_LATENCY samples one-by-one (batch_size=1), record per-sample times
  6. Throughput: N_THROUGHPUT samples at method-optimal batch size, record wall time
  7. Unload model → clear GPU cache
  8. Repeat for next method

Output: outputs/evaluations/experiments/exp8_inference_efficiency/
"""

import sys
import gc
import time
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
from src.baselines.inference_utils import save_experiment_results
from src.utils.logger import get_logger
from src.utils.group_split import group_split_by_input

logger = get_logger('experiments.exp8')

path_cfg = get_path_config()
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp8_inference_efficiency'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_WARMUP = 3          # warmup samples (discarded, stabilise GPU clocks)
N_LATENCY = 50        # samples for per-sample latency measurement (batch=1)
N_THROUGHPUT = 100     # samples for batch throughput measurement
N_LATENCY_TEST = 5    # --test-mode override
N_THROUGHPUT_TEST = 10

# Methods requiring FP16 inference (same as exp2)
METHODS_REQUIRE_FP16 = {'p_tuning', 'prompt_tuning'}

# Optimal batch sizes per method for throughput measurement
# (same convention as exp2 for text expert; CPU baselines use bulk)
THROUGHPUT_BATCH = {
    'bm25': 'bulk',
    'lsa': 'bulk',
    'template': 'bulk',
    'zeroshot': 8,
    'lora_moe': 16,
    'lora_single': 16,
    'p_tuning': 1,          # must be 1 (position-sensitive soft prompts)
    'prompt_tuning': 1,     # must be 1
    'full_finetuning': 8,   # conservative due to 7-module adapter
}

METHOD_LABELS = {
    'bm25': 'BM25',
    'lsa': 'LSA',
    'template': 'Template',
    'zeroshot': 'Zero-Shot',
    'lora_moe': 'LoRA-MoE',
    'lora_single': 'LoRA-Single',
    'p_tuning': 'P-Tuning v2',
    'prompt_tuning': 'Prompt Tuning',
    'full_finetuning': 'Full FT',
}

CPU_METHODS = {'bm25', 'lsa', 'template'}
GPU_METHODS = {'zeroshot', 'lora_moe', 'lora_single',
               'p_tuning', 'prompt_tuning', 'full_finetuning'}
ALL_METHODS = list(CPU_METHODS) + list(GPU_METHODS)

# Ordered for display: CPU first, then GPU
METHOD_ORDER = [
    'bm25', 'lsa', 'template',
    'zeroshot', 'lora_moe', 'lora_single',
    'p_tuning', 'prompt_tuning', 'full_finetuning',
]

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _clear_gpu():
    """Force-clear GPU memory and reset peak tracking."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_mem_mb():
    """Return current peak GPU memory in MB (since last reset)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _gpu_current_mb():
    """Return currently allocated GPU memory in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Adapter / index size on disk
# ---------------------------------------------------------------------------

def _disk_size_mb(method):
    """Return the on-disk size of the adapter / index for the given method (text expert)."""
    try:
        ckpt_map = {
            'lora_moe':        path_cfg.LORA_MOE_CKPTS.get('text'),
            'lora_single':     getattr(path_cfg, 'LORA_SINGLE_CKPT', None),
            'p_tuning':        path_cfg.PTUNING_CKPTS.get('text') if hasattr(path_cfg, 'PTUNING_CKPTS') else None,
            'prompt_tuning':   path_cfg.PROMPT_TUNING_CKPTS.get('text') if hasattr(path_cfg, 'PROMPT_TUNING_CKPTS') else None,
            'full_finetuning': path_cfg.FULL_FINETUNING_CKPTS.get('text') if hasattr(path_cfg, 'FULL_FINETUNING_CKPTS') else None,
        }
        ckpt = ckpt_map.get(method)
        if ckpt is None:
            return 0.0
        ckpt = Path(ckpt)
        if not ckpt.exists():
            return 0.0
        total = sum(f.stat().st_size for f in ckpt.rglob('*') if f.is_file())
        return total / (1024 ** 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Method runners — each returns (load_time_s, latencies_ms, throughput_info)
#   latencies_ms: list of per-sample times in milliseconds (batch=1)
#   throughput_info: dict {n_samples, wall_time_s, batch_size, samples_per_sec}
# ---------------------------------------------------------------------------

def _benchmark_cpu_method(method, train_data, test_inputs, n_warmup, n_latency, n_throughput):
    """Benchmark a CPU-only baseline (BM25 / LSA / Template)."""
    from src.baselines.ir_methods import BM25Retriever, LSARetriever
    from src.baselines.template_filling import TemplateFiller

    # --- Load / build index ---
    t0 = time.perf_counter()
    if method == 'bm25':
        obj = BM25Retriever()
        obj.build_index(train_data)
        predict_one = lambda inp: obj.batch_retrieve([inp])[0]
        predict_batch = lambda inps: obj.batch_retrieve(inps)
    elif method == 'lsa':
        obj = LSARetriever(n_components=100)
        obj.build_index(train_data)
        predict_one = lambda inp: obj.batch_retrieve([inp])[0]
        predict_batch = lambda inps: obj.batch_retrieve(inps)
    else:  # template
        obj = TemplateFiller()
        predict_one = lambda inp: obj.batch_fill([inp])[0]
        predict_batch = lambda inps: obj.batch_fill(inps)
    load_time = time.perf_counter() - t0

    # --- Warmup ---
    for inp in test_inputs[:n_warmup]:
        _ = predict_one(inp)

    # --- Latency (one-by-one) ---
    latencies = []
    for inp in test_inputs[n_warmup:n_warmup + n_latency]:
        t0 = time.perf_counter()
        _ = predict_one(inp)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    # --- Throughput (bulk) ---
    batch_inputs = test_inputs[:n_throughput]
    t0 = time.perf_counter()
    _ = predict_batch(batch_inputs)
    wall = time.perf_counter() - t0

    throughput_info = {
        'n_samples': len(batch_inputs),
        'wall_time_s': round(wall, 4),
        'batch_size': len(batch_inputs),
        'samples_per_sec': round(len(batch_inputs) / max(wall, 1e-9), 2),
    }
    return load_time, latencies, throughput_info


def _benchmark_gpu_method(method, test_inputs, n_warmup, n_latency, n_throughput):
    """Benchmark a GPU-based method (zero-shot / LoRA variants / P-tuning etc.)."""
    import torch

    use_4bit = method not in METHODS_REQUIRE_FP16
    batch_size = THROUGHPUT_BATCH.get(method, 8)

    # --- Determine expert class + checkpoint ---
    if method == 'zeroshot':
        from src.baselines.zero_shot import ZeroShotGenerator
        _clear_gpu()
        t0 = time.perf_counter()
        gen = ZeroShotGenerator(use_4bit=True)
        if not gen.load_model():
            logger.error(f'{method}: model load failed')
            return None, None, None
        load_time = time.perf_counter() - t0
        peak_after_load = _gpu_mem_mb()

        # warmup
        for inp in test_inputs[:n_warmup]:
            _ = gen.batch_generate([inp], input_type='text', n_shots=0)

        # latency
        latencies = []
        for inp in test_inputs[n_warmup:n_warmup + n_latency]:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = gen.batch_generate([inp], input_type='text', n_shots=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        # throughput
        batch_inputs = test_inputs[:n_throughput]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = gen.batch_generate(batch_inputs, input_type='text', n_shots=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak_inference = _gpu_mem_mb()

        gen.unload_model()
    else:
        # All other GPU methods use Expert classes
        from src.experts import TextExpert

        ckpt_map = {
            'lora_moe':        lambda: str(path_cfg.LORA_MOE_CKPTS['text']),
            'lora_single':     lambda: str(path_cfg.LORA_SINGLE_CKPT),
            'p_tuning':        lambda: str(path_cfg.PTUNING_CKPTS['text']),
            'prompt_tuning':   lambda: str(path_cfg.PROMPT_TUNING_CKPTS['text']),
            'full_finetuning': lambda: str(path_cfg.FULL_FINETUNING_CKPTS['text']),
        }
        ckpt_path = ckpt_map[method]()

        _clear_gpu()
        t0 = time.perf_counter()
        expert = TextExpert(lora_path=ckpt_path, use_4bit=use_4bit)
        if not expert.load_model():
            logger.error(f'{method}: model load failed')
            return None, None, None
        load_time = time.perf_counter() - t0
        peak_after_load = _gpu_mem_mb()

        effective_bs = 1 if method in METHODS_REQUIRE_FP16 else batch_size

        # warmup
        for inp in test_inputs[:n_warmup]:
            _ = expert.batch_generate_instruction([inp], batch_size=1)

        # latency (always batch_size=1)
        latencies = []
        for inp in test_inputs[n_warmup:n_warmup + n_latency]:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = expert.batch_generate_instruction([inp], batch_size=1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        # throughput
        batch_inputs = test_inputs[:n_throughput]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = expert.batch_generate_instruction(batch_inputs, batch_size=effective_bs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak_inference = _gpu_mem_mb()

        expert.unload_model()

    throughput_info = {
        'n_samples': len(test_inputs[:n_throughput]),
        'wall_time_s': round(wall, 4),
        'batch_size': batch_size if method != 'zeroshot' else 'all',
        'samples_per_sec': round(len(test_inputs[:n_throughput]) / max(wall, 1e-9), 2),
    }
    return load_time, latencies, throughput_info


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_latency_comparison(results_by_method, exp_dir, test_mode=False):
    """Horizontal bar chart of median latency per method."""
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    medians = []
    p95s = []
    colors = []
    for m in methods:
        r = results_by_method[m]
        medians.append(r.get('latency_median_ms', 0))
        p95s.append(r.get('latency_p95_ms', 0))
        colors.append('#4ECDC4' if m in CPU_METHODS else
                      '#1f77b4' if m == 'lora_moe' else '#ff7f0e')

    y = np.arange(len(methods))
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    bars = ax.barh(y, medians, color=colors, edgecolor='gray', height=0.55,
                   label='Median')
    # P95 markers
    ax.scatter(p95s, y, marker='|', color='red', s=120, zorder=5, label='P95')
    for i, (med, p95) in enumerate(zip(medians, p95s)):
        ax.text(max(med, p95) + max(medians) * 0.02, i,
                f'{med:.1f}ms', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Latency per Sample (ms)')
    title = 'Exp8: Inference Latency Comparison (batch_size=1)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {plots_dir / "latency_comparison.png"}')


def plot_throughput_comparison(results_by_method, exp_dir, test_mode=False):
    """Bar chart of throughput (samples/sec)."""
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    throughputs = [results_by_method[m].get('throughput_samples_per_sec', 0) for m in methods]
    colors = ['#4ECDC4' if m in CPU_METHODS else
              '#1f77b4' if m == 'lora_moe' else '#ff7f0e' for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    y = np.arange(len(methods))
    bars = ax.barh(y, throughputs, color=colors, edgecolor='gray', height=0.55)
    for bar, val in zip(bars, throughputs):
        ax.text(val + max(throughputs) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Throughput (samples/sec)')
    title = 'Exp8: Throughput Comparison (Optimal Batch Size)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / 'throughput_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {plots_dir / "throughput_comparison.png"}')


def plot_gpu_memory_comparison(results_by_method, exp_dir, test_mode=False):
    """Bar chart of peak GPU memory for GPU methods only."""
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method and m in GPU_METHODS]
    mem_vals = [results_by_method[m].get('peak_gpu_memory_mb', 0) for m in methods]
    colors = ['#1f77b4' if m == 'lora_moe' else
              '#d62728' if m in METHODS_REQUIRE_FP16 else '#ff7f0e' for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(9, max(4, len(methods) * 0.7)))
    y = np.arange(len(methods))
    bars = ax.barh(y, mem_vals, color=colors, edgecolor='gray', height=0.55)
    for bar, val in zip(bars, mem_vals):
        ax.text(val + max(mem_vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f} MB', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Peak GPU Memory (MB)')
    title = 'Exp8: GPU Memory Comparison'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color='#1f77b4'),
            plt.Rectangle((0, 0), 1, 1, color='#ff7f0e'),
            plt.Rectangle((0, 0), 1, 1, color='#d62728'),
        ],
        labels=['LoRA-MoE (4bit)', 'Other LoRA (4bit)', 'Soft-Prompt (FP16)'],
        fontsize=8
    )
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / 'gpu_memory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {plots_dir / "gpu_memory_comparison.png"}')


def plot_combined_efficiency(results_by_method, exp_dir, test_mode=False):
    """
    Scatter: latency (x) vs GPU memory (y) for GPU methods.
    Bubble size = adapter size.  LoRA-MoE highlighted.
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method and m in GPU_METHODS]
    if len(methods) < 2:
        return

    latencies = [results_by_method[m].get('latency_median_ms', 0) for m in methods]
    memories = [results_by_method[m].get('peak_gpu_memory_mb', 0) for m in methods]
    adapter_sizes = [results_by_method[m].get('adapter_size_mb', 1) for m in methods]
    # Normalise bubble sizes for visibility
    max_adapter = max(adapter_sizes) if max(adapter_sizes) > 0 else 1
    bubble_sizes = [max(40, (s / max_adapter) * 400) for s in adapter_sizes]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(methods):
        color = '#1f77b4' if m == 'lora_moe' else \
                '#d62728' if m in METHODS_REQUIRE_FP16 else '#ff7f0e'
        edge = 'black' if m == 'lora_moe' else 'gray'
        lw = 2 if m == 'lora_moe' else 0.5
        ax.scatter(latencies[i], memories[i], s=bubble_sizes[i],
                   c=color, edgecolors=edge, linewidths=lw, alpha=0.8, zorder=5)
        ax.annotate(METHOD_LABELS.get(m, m),
                    (latencies[i], memories[i]),
                    textcoords='offset points', xytext=(8, 8), fontsize=8)

    ax.set_xlabel('Median Latency (ms/sample)')
    ax.set_ylabel('Peak GPU Memory (MB)')
    title = 'Exp8: Latency vs Memory (bubble = adapter size)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'latency_vs_memory.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {plots_dir / "latency_vs_memory.png"}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    logger.info('=' * 80)
    logger.info('实验8: 推理效率基准测试')
    logger.info('=' * 80)

    n_latency = N_LATENCY_TEST if args.test_mode else N_LATENCY
    n_throughput = N_THROUGHPUT_TEST if args.test_mode else N_THROUGHPUT
    n_warmup = min(N_WARMUP, 1) if args.test_mode else N_WARMUP

    # Load text data (same set used in exp1/exp2 for text expert)
    logger.info('加载文本数据集...')
    loader = TextDatasetLoader()
    all_data = loader.load_csv_files()
    train_data, _, test_data = split_dataset_for_expert(all_data, 'text')
    test_inputs = [d['input'] for d in test_data]
    # Ensure we have enough samples
    n_needed = n_warmup + max(n_latency, n_throughput)
    if len(test_inputs) < n_needed:
        logger.warning(f'测试集仅 {len(test_inputs)} 条，需要 {n_needed} 条，将循环复用')
        while len(test_inputs) < n_needed:
            test_inputs = test_inputs + test_inputs
    logger.info(f'测试集样本: {len(test_data)} | 延迟测量: {n_latency} | 吞吐测量: {n_throughput}')

    # Select methods
    methods_to_run = list(METHOD_ORDER)
    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
    if args.skip:
        skip_set = set(m.strip() for m in args.skip.split(','))
        methods_to_run = [m for m in methods_to_run if m not in skip_set]

    results = {
        'experiment': 'exp8_inference_efficiency',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'n_warmup': n_warmup,
        'n_latency': n_latency,
        'n_throughput': n_throughput,
        'hardware': _get_hardware_info(),
        'methods': {},
    }
    results_by_method = {}

    for method in methods_to_run:
        logger.info(f'\n{"=" * 60}')
        logger.info(f'基准测试: {METHOD_LABELS.get(method, method)}')
        logger.info(f'{"=" * 60}')

        try:
            if method in CPU_METHODS:
                load_time, latencies, tp_info = _benchmark_cpu_method(
                    method, train_data, test_inputs, n_warmup, n_latency, n_throughput
                )
                peak_mem = 0.0
            elif method in GPU_METHODS:
                load_time, latencies, tp_info = _benchmark_gpu_method(
                    method, test_inputs, n_warmup, n_latency, n_throughput
                )
                if load_time is None:
                    logger.warning(f'{method}: 跳过（模型加载失败）')
                    continue
                peak_mem = _gpu_mem_mb()
                _clear_gpu()
            else:
                logger.warning(f'未知方法: {method}')
                continue

            latencies_arr = np.array(latencies) if latencies else np.array([0])
            adapter_mb = _disk_size_mb(method)

            entry = {
                'method': method,
                'label': METHOD_LABELS.get(method, method),
                'device': 'CPU' if method in CPU_METHODS else 'GPU',
                'quantisation': 'N/A' if method in CPU_METHODS else
                                ('FP16' if method in METHODS_REQUIRE_FP16 else '4bit'),
                'load_time_s': round(load_time, 3),
                'latency_mean_ms': round(float(np.mean(latencies_arr)), 2),
                'latency_median_ms': round(float(np.median(latencies_arr)), 2),
                'latency_p95_ms': round(float(np.percentile(latencies_arr, 95)), 2),
                'latency_std_ms': round(float(np.std(latencies_arr)), 2),
                'latency_n_samples': len(latencies),
                'throughput_samples_per_sec': tp_info['samples_per_sec'],
                'throughput_batch_size': tp_info['batch_size'],
                'throughput_wall_s': tp_info['wall_time_s'],
                'throughput_n_samples': tp_info['n_samples'],
                'peak_gpu_memory_mb': round(peak_mem, 1),
                'adapter_size_mb': round(adapter_mb, 2),
            }
            results['methods'][method] = entry
            results_by_method[method] = entry

            logger.info(
                f'  加载时间:    {load_time:.2f}s\n'
                f'  延迟(中位):  {entry["latency_median_ms"]:.1f}ms  '
                f'(P95={entry["latency_p95_ms"]:.1f}ms)\n'
                f'  吞吐:       {tp_info["samples_per_sec"]:.1f} samples/sec '
                f'(batch={tp_info["batch_size"]})\n'
                f'  GPU显存:    {peak_mem:.0f} MB\n'
                f'  Adapter:    {adapter_mb:.1f} MB'
            )
        except Exception as e:
            logger.error(f'{method}: 基准测试失败: {e}')
            logger.error(traceback.format_exc())
            _clear_gpu()

    # Save results
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    # Plots
    try:
        if results_by_method:
            plot_latency_comparison(results_by_method, EXP_DIR, args.test_mode)
            plot_throughput_comparison(results_by_method, EXP_DIR, args.test_mode)
            plot_gpu_memory_comparison(results_by_method, EXP_DIR, args.test_mode)
            plot_combined_efficiency(results_by_method, EXP_DIR, args.test_mode)
    except Exception as e:
        logger.warning(f'绘图失败: {e}')
        logger.warning(traceback.format_exc())

    # Summary table
    logger.info('\n' + '=' * 110)
    logger.info('推理效率汇总')
    logger.info('=' * 110)
    logger.info(
        f'{"方法":<18} {"设备":<6} {"量化":<6} '
        f'{"加载(s)":>8} {"延迟(ms)":>10} {"P95(ms)":>10} '
        f'{"吞吐(/s)":>10} {"显存(MB)":>10} {"Adapter(MB)":>12}'
    )
    logger.info('-' * 110)
    for m in METHOD_ORDER:
        if m not in results_by_method:
            continue
        e = results_by_method[m]
        logger.info(
            f'{e["label"]:<18} {e["device"]:<6} {e["quantisation"]:<6} '
            f'{e["load_time_s"]:>8.2f} {e["latency_median_ms"]:>10.1f} '
            f'{e["latency_p95_ms"]:>10.1f} {e["throughput_samples_per_sec"]:>10.1f} '
            f'{e["peak_gpu_memory_mb"]:>10.0f} {e["adapter_size_mb"]:>12.1f}'
        )
    logger.info(f'\n结果已保存至: {EXP_DIR}')


def _get_hardware_info():
    """Collect basic hardware information for reproducibility."""
    info = {}
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total_mb'] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
            )
            info['cuda_version'] = torch.version.cuda or 'N/A'
        info['torch_version'] = torch.__version__
    except Exception:
        pass
    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['ram_total_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        import os
        info['cpu_count'] = os.cpu_count()
    return info


def main():
    parser = argparse.ArgumentParser(description='Exp8: Inference efficiency benchmark')
    parser.add_argument('--test-mode', action='store_true',
                        help='Use minimal samples for quick validation')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods to benchmark '
                             '(default: all). E.g. "lora_moe,zeroshot,p_tuning"')
    parser.add_argument('--skip', type=str, default=None,
                        help='Comma-separated list of methods to skip. '
                             'E.g. "bm25,lsa,template" to skip CPU baselines')
    parser.add_argument('--n-latency', type=int, default=None,
                        help=f'Override number of samples for latency measurement '
                             f'(default: {N_LATENCY})')
    parser.add_argument('--n-throughput', type=int, default=None,
                        help=f'Override number of samples for throughput measurement '
                             f'(default: {N_THROUGHPUT})')
    args = parser.parse_args()

    # Allow CLI overrides
    global N_LATENCY, N_THROUGHPUT
    if args.n_latency is not None:
        N_LATENCY = args.n_latency
    if args.n_throughput is not None:
        N_THROUGHPUT = args.n_throughput

    run(args)


if __name__ == '__main__':
    main()