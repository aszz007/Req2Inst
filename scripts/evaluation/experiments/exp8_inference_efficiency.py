#!/usr/bin/env python3
"""
实验8: 推理效率基准测试

测量并对比所有方法的推理效率:
  - 模型加载时间 (s)
  - 单样本推理延迟 (ms): 均值/中位数/P95/最小/最大/标准差 (batch_size=1)
  - 批量吞吐量 (samples/sec): 各方法最优batch size
  - GPU峰值显存占用 (MB)
  - Adapter/索引磁盘大小 (MB)

基准测试方法 (9种):
  CPU基线:  bm25, lsa, template
  GPU方法:  zeroshot, lora_moe, lora_single, p_tuning, prompt_tuning, full_finetuning

所有GPU方法统一使用text expert测试集, 确保公平对比。

测试协议:
  1. 清空GPU缓存
  2. 记录GPU基线显存
  3. 加载模型 -> 记录加载时间 + 加载后峰值显存
  4. 预热: N_WARMUP 个样本 (结果丢弃, 稳定GPU时钟)
  5. 延迟测量: N_LATENCY 个样本逐条推理 (batch_size=1), 记录每条耗时
  6. 吞吐测量: N_THROUGHPUT 个样本按最优batch推理, 记录总耗时
  7. 卸载模型 -> 清空GPU缓存
  8. 进入下一方法

可视化 (7张图):
  - latency_comparison.png: 延迟对比柱状图 (中位数+P95标记)
  - latency_distribution.png: 延迟分布箱线图
  - throughput_comparison.png: 吞吐量对比柱状图
  - gpu_memory_comparison.png: GPU显存对比柱状图
  - load_time_comparison.png: 模型加载时间对比
  - latency_vs_memory.png: 延迟-显存权衡散点气泡图
  - summary_table.png: 论文级综合汇总表格

输出路径: outputs/evaluations/experiments/exp8_inference_efficiency/
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
PLOTS_DIR = EXP_DIR / 'plots'

# ---------------------------------------------------------------------------
# 统一配色方案
# ---------------------------------------------------------------------------

COLOR_MAP = {
    'bm25':             '#4ECDC4',   # CPU基线 - 青色
    'lsa':              '#45B7A0',   # CPU基线 - 深青色
    'template':         '#36A882',   # CPU基线 - 绿青色
    'zeroshot':         '#ff7f0e',   # GPU方法 - 橙色
    'lora_moe':         '#1f77b4',   # LoRA-MoE (主方法) - 蓝色
    'lora_single':      '#ff9933',   # GPU方法 - 深橙色
    'p_tuning':         '#d62728',   # 软提示方法 - 红色
    'prompt_tuning':    '#e74c3c',   # 软提示方法 - 浅红色
    'full_finetuning':  '#9467bd',   # 全参数基线 - 紫色
}


def _get_method_color(method):
    """根据方法名返回统一配色, 未知方法返回灰色."""
    return COLOR_MAP.get(method, '#999999')

# ---------------------------------------------------------------------------
# 基准测试配置
# ---------------------------------------------------------------------------

N_WARMUP = 3          # 预热样本数 (丢弃, 稳定GPU时钟频率)
N_LATENCY = 50        # 延迟测量样本数 (batch=1, 逐条推理)
N_THROUGHPUT = 100     # 吞吐测量样本数 (按最优batch推理)
N_LATENCY_TEST = 5    # --test-mode 覆盖值
N_THROUGHPUT_TEST = 10

# 需要FP16推理的方法 (软提示嵌入在FP16/BF16下训练, 不兼容4bit量化)
METHODS_REQUIRE_FP16 = {'p_tuning', 'prompt_tuning'}

# 各方法吞吐测量时的最优batch size
# (CPU基线使用批量处理; P-Tuning/Prompt Tuning因位置敏感必须batch=1)
THROUGHPUT_BATCH = {
    'bm25': 'bulk',
    'lsa': 'bulk',
    'template': 'bulk',
    'zeroshot': 8,
    'lora_moe': 16,
    'lora_single': 16,
    'p_tuning': 1,          # 必须为1 (位置敏感的软提示, padding导致嵌入对齐错位)
    'prompt_tuning': 1,     # 必须为1
    'full_finetuning': 8,   # 保守配置 (7类线性层adapter, 显存占用较高)
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

# 显示顺序: CPU基线在前, GPU方法在后
METHOD_ORDER = [
    'bm25', 'lsa', 'template',
    'zeroshot', 'lora_moe', 'lora_single',
    'p_tuning', 'prompt_tuning', 'full_finetuning',
]

# 有序的方法列表, 与METHOD_ORDER保持一致
ALL_METHODS = list(METHOD_ORDER)

# ---------------------------------------------------------------------------
# GPU 工具函数
# ---------------------------------------------------------------------------

def _gpu_available():
    """检查GPU是否可用."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_sync():
    """同步GPU操作, 确保所有核函数执行完毕."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _clear_gpu():
    """强制清空GPU显存并重置峰值统计."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_peak_mb():
    """返回自上次reset以来的GPU峰值显存 (MB)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _gpu_current_mb():
    """返回当前GPU已分配显存 (MB)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Adapter / 索引磁盘大小
# ---------------------------------------------------------------------------

def _disk_size_mb(method):
    """返回指定方法 (text expert) 的adapter/索引磁盘大小 (MB)."""
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
# 方法基准测试 -- 统一推理接口
#   每个方法返回: (load_time_s, latencies_ms, throughput_info)
#     latencies_ms: 每条样本的推理耗时列表 (毫秒, batch=1)
#     throughput_info: dict {n_samples, wall_time_s, batch_size, samples_per_sec}
# ---------------------------------------------------------------------------

def _benchmark_cpu_method(method, train_data, test_inputs, n_warmup, n_latency, n_throughput):
    """基准测试CPU基线方法 (BM25 / LSA / Template)."""
    from src.baselines.ir_methods import BM25Retriever, LSARetriever
    from src.baselines.template_filling import TemplateFiller

    # --- 加载 / 构建索引 ---
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

    # --- 预热 ---
    for inp in test_inputs[:n_warmup]:
        _ = predict_one(inp)

    # --- 延迟测量 (逐条推理) ---
    latencies = []
    for inp in test_inputs[n_warmup:n_warmup + n_latency]:
        t0 = time.perf_counter()
        _ = predict_one(inp)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    # --- 吞吐测量 (批量推理) ---
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
    return load_time, latencies, throughput_info, []


def _infer_one(gen_obj, inp, method):
    """统一的单样本推理接口, 消除zeroshot与expert路径的代码重复."""
    if method == 'zeroshot':
        return gen_obj.batch_generate([inp], input_type='text', n_shots=0)
    else:
        return gen_obj.batch_generate_instruction([inp], batch_size=1)


def _infer_batch(gen_obj, inputs, method, batch_size):
    """统一的批量推理接口, 消除zeroshot与expert路径的代码重复."""
    if method == 'zeroshot':
        return gen_obj.batch_generate(inputs, input_type='text', n_shots=0)
    else:
        return gen_obj.batch_generate_instruction(inputs, batch_size=batch_size)


def _benchmark_gpu_method(method, test_inputs, n_warmup, n_latency, n_throughput):
    """基准测试GPU方法 (zero-shot / LoRA变体 / P-tuning等)."""
    import torch

    use_4bit = method not in METHODS_REQUIRE_FP16
    batch_size = THROUGHPUT_BATCH.get(method, 8)

    # --- 加载模型 ---
    if method == 'zeroshot':
        from src.baselines.zero_shot import ZeroShotGenerator
        logger.info(f'  [DEBUG] 加载基础模型 (无adapter), use_4bit=True')
        _clear_gpu()
        t0 = time.perf_counter()
        gen = ZeroShotGenerator(use_4bit=True)
        if not gen.load_model():
            logger.error(f'{method}: 模型加载失败')
            return None, None, None
        load_time = time.perf_counter() - t0
    else:
        # 所有其他GPU方法使用Expert类
        from src.experts import TextExpert
        ckpt_map = {
            'lora_moe':        lambda: str(path_cfg.LORA_MOE_CKPTS['text']),
            'lora_single':     lambda: str(getattr(path_cfg, 'LORA_SINGLE_CKPT', '')),
            'p_tuning':        lambda: str(getattr(path_cfg, 'PTUNING_CKPTS', {}).get('text', '')),
            'prompt_tuning':   lambda: str(getattr(path_cfg, 'PROMPT_TUNING_CKPTS', {}).get('text', '')),
            'full_finetuning': lambda: str(getattr(path_cfg, 'FULL_FINETUNING_CKPTS', {}).get('text', '')),
        }
        ckpt_path = ckpt_map[method]()
        if not ckpt_path or not Path(ckpt_path).exists():
            logger.error(f'{method}: 检查点路径不存在或未配置: {ckpt_path}')
            return None, None, None
        logger.info(f'  [DEBUG] 检查点路径: {ckpt_path}')
        logger.info(f'  [DEBUG] use_4bit={use_4bit}, throughput_batch_size={batch_size}')
        _clear_gpu()
        t0 = time.perf_counter()
        gen = TextExpert(lora_path=ckpt_path, use_4bit=use_4bit)
        if not gen.load_model():
            logger.error(f'{method}: 模型加载失败')
            return None, None, None
        load_time = time.perf_counter() - t0

    # GPU显存快照
    logger.info(f'  [DEBUG] 模型加载后GPU显存: {_gpu_current_mb():.0f} MB (峰值: {_gpu_peak_mb():.0f} MB)')
    effective_bs = 1 if method in METHODS_REQUIRE_FP16 else batch_size
    logger.info(f'  [DEBUG] 吞吐测量batch_size: {effective_bs} (配置={batch_size}, FP16强制={"是" if method in METHODS_REQUIRE_FP16 else "否"})')

    # --- 预热 ---
    for inp in test_inputs[:n_warmup]:
        _infer_one(gen, inp, method)

    # --- 延迟测量 (逐条推理, batch_size=1) ---
    latencies = []
    output_lengths = []  # 记录每条输出字符数, 用于分析延迟差异
    for inp in test_inputs[n_warmup:n_warmup + n_latency]:
        _gpu_sync()
        t0 = time.perf_counter()
        result = _infer_one(gen, inp, method)
        _gpu_sync()
        latencies.append((time.perf_counter() - t0) * 1000)
        # 提取输出长度
        if isinstance(result, list) and len(result) > 0:
            out_text = result[0] if isinstance(result[0], str) else str(result[0])
        elif isinstance(result, str):
            out_text = result
        else:
            out_text = str(result) if result else ''
        output_lengths.append(len(out_text))
    # 输出长度统计 (用于解释延迟差异)
    if output_lengths:
        avg_len = sum(output_lengths) / len(output_lengths)
        min_len = min(output_lengths)
        max_len = max(output_lengths)
        logger.info(f'  [DEBUG] 输出长度统计: 平均={avg_len:.0f}字符, '
                     f'最短={min_len}, 最长={max_len}')

    # --- 吞吐测量 ---
    batch_inputs = test_inputs[:n_throughput]
    _gpu_sync()
    t0 = time.perf_counter()
    _infer_batch(gen, batch_inputs, method, effective_bs)
    _gpu_sync()
    wall = time.perf_counter() - t0
    # 注意: GPU峰值显存由调用方 run() 在卸载前通过 _gpu_peak_mb() 读取

    # --- 卸载模型 ---
    gen.unload_model()

    throughput_info = {
        'n_samples': len(batch_inputs),
        'wall_time_s': round(wall, 4),
        'batch_size': effective_bs,
        'samples_per_sec': round(len(batch_inputs) / max(wall, 1e-9), 2),
    }
    return load_time, latencies, throughput_info, output_lengths


# ---------------------------------------------------------------------------
# 可视化绘图
# ---------------------------------------------------------------------------

def plot_latency_comparison(results_by_method, test_mode=False):
    """延迟对比柱状图 (中位数+P95标记)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    medians = [results_by_method[m].get('latency_median_ms', 0) for m in methods]
    p95s = [results_by_method[m].get('latency_p95_ms', 0) for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    bars = ax.barh(y, medians, color=colors, edgecolor='gray', height=0.55,
                   label='Median')
    # P95 标记
    ax.scatter(p95s, y, marker='|', color='red', s=120, zorder=5, label='P95')
    for i, (med, p95) in enumerate(zip(medians, p95s)):
        offset = max(max(medians), 0.1) * 0.02
        ax.text(max(med, p95) + offset, i,
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
    plt.savefig(PLOTS_DIR / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {PLOTS_DIR / "latency_comparison.png"}')


def plot_latency_distribution(latencies_dict, test_mode=False):
    """延迟分布箱线图, 直观展示各方法延迟稳定性."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in latencies_dict and len(latencies_dict[m]) > 0]
    if not methods:
        logger.warning('无延迟数据, 跳过箱线图绘制')
        return

    data = [latencies_dict[m] for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    bp = ax.boxplot(data, vert=False, patch_artist=True, labels=labels,
                    widths=0.5, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median_line in bp['medians']:
        median_line.set(color='black', linewidth=1.5)

    ax.set_xlabel('Latency per Sample (ms)')
    title = 'Exp8: Latency Distribution (Box Plot)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'latency_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {PLOTS_DIR / "latency_distribution.png"}')


def plot_throughput_comparison(results_by_method, test_mode=False):
    """吞吐量对比柱状图 (samples/sec)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    throughputs = [results_by_method[m].get('throughput_samples_per_sec', 0) for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    y = np.arange(len(methods))
    bars = ax.barh(y, throughputs, color=colors, edgecolor='gray', height=0.55)
    for bar, val in zip(bars, throughputs):
        offset = max(max(throughputs), 0.1) * 0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
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
    plt.savefig(PLOTS_DIR / 'throughput_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {PLOTS_DIR / "throughput_comparison.png"}')


def plot_gpu_memory_comparison(results_by_method, test_mode=False):
    """GPU显存对比柱状图 (仅GPU方法)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method and m in GPU_METHODS]
    mem_vals = [results_by_method[m].get('peak_gpu_memory_mb', 0) for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(9, max(4, len(methods) * 0.7)))
    y = np.arange(len(methods))
    bars = ax.barh(y, mem_vals, color=colors, edgecolor='gray', height=0.55)
    for bar, val in zip(bars, mem_vals):
        offset = max(max(mem_vals), 0.1) * 0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f} MB', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Peak GPU Memory (MB)')
    title = 'Exp8: GPU Memory Comparison'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    # 根据实际参与测试的方法动态生成图例
    legend_handles = []
    legend_labels = []
    if 'lora_moe' in methods:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP['lora_moe']))
        legend_labels.append('LoRA-MoE (4bit)')
    other_4bit = {'zeroshot', 'lora_single', 'full_finetuning'}
    if other_4bit & set(methods):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP['lora_single']))
        legend_labels.append('Other 4bit Methods')
    soft_prompt = {'p_tuning', 'prompt_tuning'}
    if soft_prompt & set(methods):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP['p_tuning']))
        legend_labels.append('Soft-Prompt (FP16)')
    if legend_handles:
        ax.legend(handles=legend_handles, labels=legend_labels, fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'gpu_memory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {PLOTS_DIR / "gpu_memory_comparison.png"}')


def plot_load_time_comparison(results_by_method, test_mode=False):
    """模型加载时间对比图, 展示部署启动优势."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    load_times = [results_by_method[m].get('load_time_s', 0) for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    y = np.arange(len(methods))
    bars = ax.barh(y, load_times, color=colors, edgecolor='gray', height=0.55)
    for bar, val in zip(bars, load_times):
        offset = max(max(load_times), 0.1) * 0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}s', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Load Time (seconds)')
    title = 'Exp8: Model Load Time Comparison'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'load_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {PLOTS_DIR / "load_time_comparison.png"}')


def plot_combined_efficiency(results_by_method, test_mode=False):
    """延迟-显存权衡散点气泡图 (气泡大小=adapter磁盘大小, 仅GPU方法)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method and m in GPU_METHODS]
    if len(methods) < 2:
        return

    latencies = [results_by_method[m].get('latency_median_ms', 0) for m in methods]
    memories = [results_by_method[m].get('peak_gpu_memory_mb', 0) for m in methods]
    adapter_sizes = [results_by_method[m].get('adapter_size_mb', 1) for m in methods]
    # 归一化气泡大小以保证可读性
    max_adapter = max(adapter_sizes) if max(adapter_sizes) > 0 else 1
    bubble_sizes = [max(40, (s / max_adapter) * 400) for s in adapter_sizes]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(methods):
        color = _get_method_color(m)
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
    plt.savefig(PLOTS_DIR / 'latency_vs_memory.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {PLOTS_DIR / "latency_vs_memory.png"}')


def plot_summary_table(results_by_method, test_mode=False):
    """论文级综合汇总表格图片, LoRA-MoE行蓝色高亮."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    if not methods:
        return

    # 构建表格数据
    columns = ['Method', 'Device', 'Quant', 'Load(s)', 'Latency(ms)', 'P95(ms)',
               'Thru(/s)', 'Memory(MB)', 'Adapter(MB)']
    cell_data = []
    for m in methods:
        e = results_by_method[m]
        cell_data.append([
            e['label'],
            e['device'],
            e['quantisation'],
            f"{e['load_time_s']:.2f}",
            f"{e['latency_median_ms']:.1f}",
            f"{e['latency_p95_ms']:.1f}",
            f"{e['throughput_samples_per_sec']:.1f}",
            f"{e['peak_gpu_memory_mb']:.0f}",
            f"{e['adapter_size_mb']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(14, max(3, len(methods) * 0.45 + 1.5)))
    ax.axis('off')

    table = ax.table(cellText=cell_data, colLabels=columns, loc='center',
                     cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # 表头样式
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # LoRA-MoE 行蓝色高亮
    for i, m in enumerate(methods):
        row_idx = i + 1  # 跳过表头
        if m == 'lora_moe':
            for j in range(len(columns)):
                cell = table[row_idx, j]
                cell.set_facecolor('#d6eaf8')
                cell.set_text_props(fontweight='bold')
        else:
            for j in range(len(columns)):
                cell = table[row_idx, j]
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

    title = 'Exp8: Inference Efficiency Summary'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'图表已保存: {PLOTS_DIR / "summary_table.png"}')


# ---------------------------------------------------------------------------
# 实验报告自动生成
# ---------------------------------------------------------------------------

def generate_report(results, results_by_method, test_mode=False):
    """自动生成 report.md 实验报告 (与其他实验保持一致)."""
    lines = []
    lines.append('# 实验8: 推理效率基准测试报告\n')
    lines.append(f'**生成时间**: {results.get("timestamp", "N/A")}\n')
    if test_mode:
        lines.append('> **注意**: 本报告在测试模式下生成, 样本量较少, 仅供验证流程使用。\n')

    # 硬件信息
    hw = results.get('hardware', {})
    lines.append('## 1. 实验环境\n')
    lines.append(f'- GPU: {hw.get("gpu_name", "N/A")}')
    lines.append(f'- GPU显存: {hw.get("gpu_memory_total_mb", "N/A")} MB')
    lines.append(f'- CUDA: {hw.get("cuda_version", "N/A")}')
    lines.append(f'- PyTorch: {hw.get("torch_version", "N/A")}')
    lines.append(f'- CPU核心数: {hw.get("cpu_count", "N/A")}')
    lines.append(f'- 内存: {hw.get("ram_total_gb", "N/A")} GB')
    lines.append('')

    # 测试配置
    lines.append('## 2. 测试配置\n')
    lines.append(f'- 预热样本数: {results.get("n_warmup", "N/A")}')
    lines.append(f'- 延迟测量样本数: {results.get("n_latency", "N/A")}')
    lines.append(f'- 吞吐测量样本数: {results.get("n_throughput", "N/A")}')
    lines.append('')

    # 结果汇总表
    lines.append('## 3. 结果汇总\n')
    lines.append('| 方法 | 设备 | 量化 | 加载(s) | 延迟(ms) | P95(ms) | Min(ms) | Max(ms) | 吞吐(/s) | 显存(MB) | Adapter(MB) |')
    lines.append('|------|------|------|---------|----------|---------|---------|---------|----------|----------|-------------|')
    for m in METHOD_ORDER:
        if m not in results_by_method:
            continue
        e = results_by_method[m]
        highlight = '**' if m == 'lora_moe' else ''
        lines.append(
            f'| {highlight}{e["label"]}{highlight} | {e["device"]} | {e["quantisation"]} | '
            f'{e["load_time_s"]:.2f} | {e["latency_median_ms"]:.1f} | '
            f'{e["latency_p95_ms"]:.1f} | {e.get("latency_min_ms", 0):.1f} | '
            f'{e.get("latency_max_ms", 0):.1f} | {e["throughput_samples_per_sec"]:.1f} | '
            f'{e["peak_gpu_memory_mb"]:.0f} | {e["adapter_size_mb"]:.1f} |'
        )
    lines.append('')

    # 分析要点
    lines.append('## 4. 分析要点\n')

    # 找出GPU方法中延迟最低和吞吐最高的
    gpu_entries = [(m, results_by_method[m]) for m in METHOD_ORDER
                   if m in results_by_method and m in GPU_METHODS]
    if gpu_entries:
        best_latency = min(gpu_entries, key=lambda x: x[1]['latency_median_ms'])
        best_throughput = max(gpu_entries, key=lambda x: x[1]['throughput_samples_per_sec'])
        lowest_mem = min(gpu_entries, key=lambda x: x[1]['peak_gpu_memory_mb'])

        lines.append(f'- **延迟最低 (GPU)**: {METHOD_LABELS[best_latency[0]]} '
                     f'({best_latency[1]["latency_median_ms"]:.1f} ms)')
        lines.append(f'- **吞吐最高 (GPU)**: {METHOD_LABELS[best_throughput[0]]} '
                     f'({best_throughput[1]["throughput_samples_per_sec"]:.1f} samples/sec)')
        lines.append(f'- **显存最低 (GPU)**: {METHOD_LABELS[lowest_mem[0]]} '
                     f'({lowest_mem[1]["peak_gpu_memory_mb"]:.0f} MB)')

        # LoRA-MoE 对比分析
        if 'lora_moe' in results_by_method:
            moe = results_by_method['lora_moe']
            lines.append(f'\n### LoRA-MoE 效率分析\n')
            lines.append(f'- 加载时间: {moe["load_time_s"]:.2f}s')
            lines.append(f'- 中位延迟: {moe["latency_median_ms"]:.1f}ms '
                         f'(P95={moe["latency_p95_ms"]:.1f}ms, '
                         f'Std={moe["latency_std_ms"]:.1f}ms)')
            lines.append(f'- 延迟范围: {moe.get("latency_min_ms", 0):.1f}ms ~ '
                         f'{moe.get("latency_max_ms", 0):.1f}ms')
            lines.append(f'- 吞吐量: {moe["throughput_samples_per_sec"]:.1f} samples/sec '
                         f'(batch_size={moe["throughput_batch_size"]})')
            lines.append(f'- GPU显存: {moe["peak_gpu_memory_mb"]:.0f} MB')
            lines.append(f'- Adapter大小: {moe["adapter_size_mb"]:.1f} MB')
    lines.append('')

    # 图表说明
    lines.append('## 5. 可视化图表\n')
    plot_descriptions = [
        ('latency_comparison.png', '延迟对比柱状图 (中位数+P95标记)'),
        ('latency_distribution.png', '延迟分布箱线图'),
        ('throughput_comparison.png', '吞吐量对比柱状图'),
        ('gpu_memory_comparison.png', 'GPU显存对比柱状图'),
        ('load_time_comparison.png', '模型加载时间对比'),
        ('latency_vs_memory.png', '延迟-显存权衡散点气泡图'),
        ('summary_table.png', '论文级综合汇总表格'),
    ]
    for fname, desc in plot_descriptions:
        lines.append(f'- `plots/{fname}`: {desc}')
    lines.append('')

    # 写入文件
    report_path = EXP_DIR / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f'实验报告已保存: {report_path}')


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def run(args):
    logger.info('=' * 80)
    logger.info('实验8: 推理效率基准测试')
    logger.info('=' * 80)

    n_latency = N_LATENCY_TEST if args.test_mode else N_LATENCY
    n_throughput = N_THROUGHPUT_TEST if args.test_mode else N_THROUGHPUT
    n_warmup = min(N_WARMUP, 1) if args.test_mode else N_WARMUP

    # 加载文本数据集 (与exp1/exp2的text expert使用相同测试集)
    logger.info('加载文本数据集...')
    loader = TextDatasetLoader()
    all_data = loader.load_csv_files()
    train_data, _, test_data = split_dataset_for_expert(all_data, 'text')
    test_inputs = [d['input'] for d in test_data]
    # 确保样本量足够
    n_needed = n_warmup + max(n_latency, n_throughput)
    if len(test_inputs) < n_needed:
        logger.warning(f'测试集仅 {len(test_inputs)} 条, 需要 {n_needed} 条, 将循环复用')
        while len(test_inputs) < n_needed:
            test_inputs = test_inputs + test_inputs
    logger.info(f'测试集样本: {len(test_data)} | 延迟测量: {n_latency} | 吞吐测量: {n_throughput}')

    # 选择待测方法
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
    latencies_dict = {}  # 保存原始延迟数据, 用于箱线图绘制

    for method in methods_to_run:
        logger.info(f'\n{"=" * 60}')
        logger.info(f'基准测试: {METHOD_LABELS.get(method, method)}')
        logger.info(f'{"=" * 60}')

        try:
            if method in CPU_METHODS:
                load_time, latencies, tp_info, out_lens = _benchmark_cpu_method(
                    method, train_data, test_inputs, n_warmup, n_latency, n_throughput
                )
                peak_mem = 0.0
            elif method in GPU_METHODS:
                load_time, latencies, tp_info, out_lens = _benchmark_gpu_method(
                    method, test_inputs, n_warmup, n_latency, n_throughput
                )
                if load_time is None:
                    logger.warning(f'{method}: 跳过 (模型加载失败)')
                    continue
                peak_mem = _gpu_peak_mb()
                _clear_gpu()
            else:
                logger.warning(f'未知方法: {method}')
                continue

            latencies_arr = np.array(latencies) if latencies else np.array([0])
            adapter_mb = _disk_size_mb(method)

            # 保存原始延迟数据
            latencies_dict[method] = latencies if latencies else []

            # 输出长度统计 (仅GPU方法有数据)
            out_len_stats = {}
            if out_lens:
                out_arr = np.array(out_lens)
                out_len_stats = {
                    'output_length_mean': round(float(np.mean(out_arr)), 1),
                    'output_length_min': int(np.min(out_arr)),
                    'output_length_max': int(np.max(out_arr)),
                }

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
                'latency_min_ms': round(float(np.min(latencies_arr)), 2),
                'latency_max_ms': round(float(np.max(latencies_arr)), 2),
                'latency_std_ms': round(float(np.std(latencies_arr)), 2),
                'latency_n_samples': len(latencies),
                'throughput_samples_per_sec': tp_info['samples_per_sec'],
                'throughput_batch_size': tp_info['batch_size'],
                'throughput_wall_s': tp_info['wall_time_s'],
                'throughput_n_samples': tp_info['n_samples'],
                'peak_gpu_memory_mb': round(peak_mem, 1),
                'adapter_size_mb': round(adapter_mb, 2),
                **out_len_stats,
            }
            results['methods'][method] = entry
            results_by_method[method] = entry

            logger.info(
                f'  加载时间:    {load_time:.2f}s\n'
                f'  延迟(中位):  {entry["latency_median_ms"]:.1f}ms  '
                f'(P95={entry["latency_p95_ms"]:.1f}ms, '
                f'Min={entry.get("latency_min_ms", 0):.1f}ms, '
                f'Max={entry.get("latency_max_ms", 0):.1f}ms)\n'
                f'  吞吐:       {tp_info["samples_per_sec"]:.1f} samples/sec '
                f'(batch={tp_info["batch_size"]})\n'
                f'  GPU显存:    {peak_mem:.0f} MB\n'
                f'  Adapter:    {adapter_mb:.1f} MB'
            )
        except Exception as e:
            logger.error(f'{method}: 基准测试失败: {e}')
            logger.error(traceback.format_exc())
            _clear_gpu()

    # 保存结果
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    # 绘制图表 (7张)
    try:
        if results_by_method:
            plot_latency_comparison(results_by_method, args.test_mode)
            plot_latency_distribution(latencies_dict, args.test_mode)
            plot_throughput_comparison(results_by_method, args.test_mode)
            plot_gpu_memory_comparison(results_by_method, args.test_mode)
            plot_load_time_comparison(results_by_method, args.test_mode)
            plot_combined_efficiency(results_by_method, args.test_mode)
            plot_summary_table(results_by_method, args.test_mode)
    except Exception as e:
        logger.warning(f'绘图失败: {e}')
        logger.warning(traceback.format_exc())

    # 生成实验报告
    try:
        generate_report(results, results_by_method, args.test_mode)
    except Exception as e:
        logger.warning(f'报告生成失败: {e}')
        logger.warning(traceback.format_exc())

    # 控制台汇总表
    logger.info('\n' + '=' * 120)
    logger.info('推理效率汇总')
    logger.info('=' * 120)
    logger.info(
        f'{"方法":<18} {"设备":<6} {"量化":<6} '
        f'{"加载(s)":>8} {"延迟(ms)":>10} {"P95(ms)":>10} '
        f'{"Min(ms)":>10} {"Max(ms)":>10} '
        f'{"吞吐(/s)":>10} {"显存(MB)":>10} {"Adapter(MB)":>12}'
    )
    logger.info('-' * 120)
    for m in METHOD_ORDER:
        if m not in results_by_method:
            continue
        e = results_by_method[m]
        logger.info(
            f'{e["label"]:<18} {e["device"]:<6} {e["quantisation"]:<6} '
            f'{e["load_time_s"]:>8.2f} {e["latency_median_ms"]:>10.1f} '
            f'{e["latency_p95_ms"]:>10.1f} {e.get("latency_min_ms", 0):>10.1f} '
            f'{e.get("latency_max_ms", 0):>10.1f} {e["throughput_samples_per_sec"]:>10.1f} '
            f'{e["peak_gpu_memory_mb"]:>10.0f} {e["adapter_size_mb"]:>12.1f}'
        )
    # Diagnostic: output length vs latency correlation
    gpu_with_outlen = [(m, results_by_method[m]) for m in METHOD_ORDER
                       if m in results_by_method and m in GPU_METHODS
                       and 'output_length_mean' in results_by_method[m]]
    if gpu_with_outlen:
        logger.info('\n' + '=' * 80)
        logger.info('Diagnostic: Output Length vs Latency Correlation')
        logger.info('=' * 80)
        logger.info(f'{"Method":<18} {"Latency(ms)":>12} {"AvgOutput(ch)":>14} {"Min":>8} {"Max":>8} {"ms/char":>10}')
        logger.info('-' * 80)
        for m, e in gpu_with_outlen:
            avg_out = e.get('output_length_mean', 0)
            ms_per_char = e['latency_median_ms'] / max(avg_out, 1)
            logger.info(
                f'{e["label"]:<18} {e["latency_median_ms"]:>12.1f} '
                f'{avg_out:>14.0f} {e.get("output_length_min", 0):>8} '
                f'{e.get("output_length_max", 0):>8} {ms_per_char:>10.2f}'
            )

    # Save debug diagnostics JSON
    try:
        import json as _json
        diag = {
            'checkpoint_paths': {},
            'output_length_analysis': {},
            'batch_efficiency': {},
        }
        try:
            diag['checkpoint_paths'] = {
                'lora_moe_text': str(path_cfg.LORA_MOE_CKPTS.get('text', '')),
                'lora_single': str(getattr(path_cfg, 'LORA_SINGLE_CKPT', '')),
                'p_tuning_text': str(getattr(path_cfg, 'PTUNING_CKPTS', {}).get('text', '')),
                'prompt_tuning_text': str(getattr(path_cfg, 'PROMPT_TUNING_CKPTS', {}).get('text', '')),
                'full_ft_text': str(getattr(path_cfg, 'FULL_FINETUNING_CKPTS', {}).get('text', '')),
            }
        except Exception:
            pass
        for m in METHOD_ORDER:
            if m in results_by_method and 'output_length_mean' in results_by_method[m]:
                e = results_by_method[m]
                avg_out = e.get('output_length_mean', 0)
                diag['output_length_analysis'][m] = {
                    'avg_output_chars': avg_out,
                    'latency_median_ms': e['latency_median_ms'],
                    'ms_per_char': round(e['latency_median_ms'] / max(avg_out, 1), 2),
                }
        for m in METHOD_ORDER:
            if m in results_by_method and m in GPU_METHODS:
                e = results_by_method[m]
                ps_batch = e['throughput_wall_s'] / max(e['throughput_n_samples'], 1) * 1000
                diag['batch_efficiency'][m] = {
                    'latency_median_ms': e['latency_median_ms'],
                    'per_sample_in_batch_ms': round(ps_batch, 1),
                    'batch_speedup': round(e['latency_median_ms'] / max(ps_batch, 0.1), 1),
                    'batch_size': e['throughput_batch_size'],
                }
        diag_path = EXP_DIR / 'debug_diagnostics.json'
        with open(diag_path, 'w', encoding='utf-8') as df:
            _json.dump(diag, df, indent=2, ensure_ascii=False)
        logger.info(f'Debug diagnostics saved: {diag_path}')
    except Exception as diag_err:
        logger.warning(f'Failed to save diagnostics: {diag_err}')

    logger.info(f'\n结果已保存至: {EXP_DIR}')


def _get_hardware_info():
    """收集硬件信息, 用于实验结果的可复现性."""
    info = {}
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total_mb'] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
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
    # 命令行参数覆盖全局配置
    global N_LATENCY, N_THROUGHPUT

    parser = argparse.ArgumentParser(description='实验8: 推理效率基准测试')
    parser.add_argument('--test-mode', action='store_true',
                        help='使用最少样本快速验证流程')
    parser.add_argument('--methods', type=str, default=None,
                        help='逗号分隔的待测方法列表 (默认: 全部). '
                             '例如 "lora_moe,zeroshot,p_tuning"')
    parser.add_argument('--skip', type=str, default=None,
                        help='逗号分隔的跳过方法列表. '
                             '例如 "bm25,lsa,template" 跳过CPU基线')
    parser.add_argument('--n-latency', type=int, default=None,
                        help=f'覆盖延迟测量样本数 (默认: {N_LATENCY})')
    parser.add_argument('--n-throughput', type=int, default=None,
                        help=f'覆盖吞吐测量样本数 (默认: {N_THROUGHPUT})')
    args = parser.parse_args()
    if args.n_latency is not None:
        N_LATENCY = args.n_latency
    if args.n_throughput is not None:
        N_THROUGHPUT = args.n_throughput

    run(args)


if __name__ == '__main__':
    main()