#!/usr/bin/env python3
"""
Experiment 9: Routing Strategy Comparison & Router Contribution Analysis

Phase 1: Oracle上下界分析（~1.5h，必做）
  - 5种路由策略对比: Hard / Oracle / Random / Worst / General-Only
  - 新增推理仅1470条，复用exp2/exp3缓存（复用率69%）

Phase 2: Soft Routing验证（~45min，条件执行）
  - 仅General域，PEFT add_weighted_adapter 参数级融合
  - 融合比例网格搜索: alpha in {0.3, 0.5, 0.7}

Phase 3: 贡献度分析 + 可视化（~15min，必做）
  - 8张分析图表

输出: outputs/evaluations/experiments/exp9_routing_strategy/

Author: Claude
Date: 2026-03-04
"""

import sys
import gc
import argparse
import random as random_module
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

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
    GeneralDatasetLoader, split_dataset_for_expert,
)
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp9')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp9_routing_strategy'
PLOT_DIR = EXP_DIR / 'plots'

ALL_TYPES = ['text', 'image', 'uml', 'general']
SPECIALIZED_TYPES = ['text', 'image', 'uml']


# ========== 工具函数 ==========

def _get_expert(expert_type, lora_path=None):
    """获取专家实例"""
    from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert
    cls = {
        'text': TextExpert, 'image': ImageExpert,
        'uml': UMLExpert, 'general': GeneralExpert
    }[expert_type]
    return cls(lora_path=lora_path, use_4bit=True)


def _load_test_data(expert_type):
    """加载指定类型的测试集"""
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


def _is_full_run_cache(cache_dir, filename):
    """检查缓存是否为完整运行（非test-mode）"""
    cached = load_predictions_cache(cache_dir, filename)
    if cached is None:
        return False
    n = cached.get('total_samples', 0)
    return n > 15


def _metrics_from_cache(cached, use_bertscore=True):
    """从缓存计算指标"""
    if cached is None:
        return {}
    samples = cached.get('samples', [])
    preds = [s.get('prediction', '') for s in samples]
    refs = [s.get('reference', '') for s in samples]
    return compute_all_metrics(preds, refs, use_bertscore=use_bertscore)


def _get_rougeL(metrics_dict):
    """从指标字典中提取ROUGE-L"""
    return metrics_dict.get('generation_quality', {}).get('rougeL', 0.0)


def _get_format_score(metrics_dict):
    """从指标字典中提取format_score"""
    return metrics_dict.get('format_metrics', {}).get('format_score', 0.0)


def _cleanup_gpu():
    """清理GPU显存"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ========== 缓存映射 ==========

def _get_cache_location(expert_type, eval_domain):
    """
    获取 expert_type 专家在 eval_domain 测试集上的缓存位置

    复用优先级:
    1. lora_moe/ - 匹配专家推理（exp2产物）
    2. exp3_cross_domain/ - 跨域推理（exp3产物）
    3. exp3_moe3_general_via_text/ - MoE-3退化路由（exp3产物）
    4. exp9_oracle/ - 本实验新增推理

    Returns:
        (cache_dir, filename): 缓存目录和文件名
    """
    if expert_type == eval_domain:
        # 匹配专家：复用 lora_moe 缓存
        return CACHE_DIR / 'lora_moe', f'{eval_domain}_predictions.json'

    if expert_type == 'text' and eval_domain == 'general':
        # text专家处理general：复用 exp3 MoE-3缓存
        return CACHE_DIR / 'exp3_moe3_general_via_text', 'general_via_text_predictions.json'

    if expert_type in SPECIALIZED_TYPES and eval_domain in SPECIALIZED_TYPES:
        # 跨域（专项域之间）：复用 exp3 缓存
        return (CACHE_DIR / 'exp3_cross_domain',
                f'{expert_type}_expert_on_{eval_domain}_predictions.json')

    # 其他情况：需要新增推理，存入 exp9_oracle
    return (CACHE_DIR / 'exp9_oracle',
            f'{expert_type}_expert_on_{eval_domain}_predictions.json')


# ========== Phase 1: Oracle上下界分析 ==========

def _run_single_inference(expert_type, eval_domain, test_data, args):
    """
    运行单个 expert-domain 推理组合

    Args:
        expert_type: 专家类型
        eval_domain: 评估域
        test_data: 测试数据
        args: 命令行参数

    Returns:
        缓存数据字典，或None
    """
    cache_dir, filename = _get_cache_location(expert_type, eval_domain)

    # 尝试加载缓存
    cached = load_predictions_cache(cache_dir, filename)
    if cached and not args.force_regenerate:
        n = cached.get('total_samples', 0)
        if n > 15 or args.test_mode:
            logger.info(f"缓存命中: {expert_type}->{eval_domain} ({n}条)")
            return cached

    # 需要新推理
    logger.info(f"执行推理: {expert_type}->{eval_domain}")
    expert = _get_expert(expert_type)
    if not expert.load_model():
        logger.error(f"加载 {expert_type} 专家失败")
        return None

    data_subset = test_data[:10] if args.test_mode else test_data
    inputs = [d['input'] for d in data_subset]
    refs = [d['output'] for d in data_subset]

    try:
        preds = expert.batch_generate_instruction(inputs, batch_size=4)
    except Exception as e:
        logger.error(f"推理失败 {expert_type}->{eval_domain}: {e}")
        preds = [''] * len(inputs)
    finally:
        expert.unload_model()
        _cleanup_gpu()

    samples = [
        {'index': i, 'input': inp, 'prediction': p, 'reference': r}
        for i, (inp, p, r) in enumerate(zip(inputs, preds, refs))
    ]

    save_predictions_cache(
        samples, 'exp9_oracle', eval_domain,
        {'expert': expert_type, 'eval_domain': eval_domain,
         'purpose': 'oracle_label_construction'},
        cache_dir, filename
    )
    return load_predictions_cache(cache_dir, filename)


def run_phase1(args):
    """
    Phase 1: Oracle上下界分析

    构建 4专家 x 4域 = 16 组推理结果，然后计算5种路由策略的得分。

    Returns:
        Dict: Phase 1结果
    """
    logger.info("=" * 80)
    logger.info("Phase 1: Oracle上下界分析")
    logger.info("=" * 80)

    # 加载所有测试集
    logger.info("加载测试集...")
    test_datasets = {}
    for et in ALL_TYPES:
        try:
            test_datasets[et] = _load_test_data(et)
            logger.info(f"  {et}: {len(test_datasets[et])} 条")
        except Exception as e:
            logger.error(f"  加载 {et} 测试集失败: {e}")

    # --- 步骤1: 收集/补充16组推理 ---
    logger.info("\n--- 步骤1: 收集16组推理结果 ---")

    # all_caches[expert_type][eval_domain] = cached_data
    all_caches = {}
    new_inference_count = 0
    reused_count = 0

    for expert_type in ALL_TYPES:
        all_caches[expert_type] = {}
        for eval_domain in ALL_TYPES:
            if eval_domain not in test_datasets:
                continue

            cache_dir, filename = _get_cache_location(expert_type, eval_domain)
            cached = load_predictions_cache(cache_dir, filename)

            if cached and not args.force_regenerate and (
                cached.get('total_samples', 0) > 15 or args.test_mode
            ):
                all_caches[expert_type][eval_domain] = cached
                reused_count += 1
                logger.info(f"  [复用] {expert_type}->{eval_domain}: "
                          f"{cached.get('total_samples', 0)}条")
            else:
                # 需要新推理
                cached = _run_single_inference(
                    expert_type, eval_domain,
                    test_datasets[eval_domain], args
                )
                all_caches[expert_type][eval_domain] = cached
                new_inference_count += 1

    logger.info(f"\n推理统计: 复用={reused_count}, 新增={new_inference_count}, "
              f"总计={reused_count + new_inference_count}")

    # --- 步骤2: 计算每组的ROUGE-L ---
    logger.info("\n--- 步骤2: 计算指标 ---")

    # score_matrix[expert_type][eval_domain] = rougeL
    score_matrix = {}
    metrics_matrix = {}
    use_bs = not args.no_bertscore

    for expert_type in ALL_TYPES:
        score_matrix[expert_type] = {}
        metrics_matrix[expert_type] = {}
        for eval_domain in ALL_TYPES:
            cached = all_caches.get(expert_type, {}).get(eval_domain)
            if cached is None:
                score_matrix[expert_type][eval_domain] = 0.0
                continue
            m = _metrics_from_cache(cached, use_bertscore=use_bs)
            score_matrix[expert_type][eval_domain] = _get_rougeL(m)
            metrics_matrix[expert_type][eval_domain] = m
            logger.info(f"  {expert_type}->{eval_domain}: "
                      f"ROUGE-L={score_matrix[expert_type][eval_domain]:.4f}")

    # --- 步骤3: 计算5种路由策略得分 ---
    logger.info("\n--- 步骤3: 计算5种路由策略得分 ---")

    strategies = {}

    # 3a. Hard Routing（匹配专家，对角线）
    hard_scores = {}
    for domain in ALL_TYPES:
        hard_scores[domain] = score_matrix.get(domain, {}).get(domain, 0.0)
    hard_avg = np.mean(list(hard_scores.values())) if hard_scores else 0.0
    strategies['Hard Routing'] = {
        'per_domain': hard_scores,
        'average': float(hard_avg),
    }
    logger.info(f"Hard Routing: 平均ROUGE-L={hard_avg:.4f}")

    # 3b. Oracle Routing（每个样本选最优专家）
    oracle_scores = {}
    oracle_selections = {}  # domain -> {expert: count}
    for domain in ALL_TYPES:
        if domain not in test_datasets:
            continue
        n_samples = len(test_datasets[domain])
        if args.test_mode:
            n_samples = min(10, n_samples)

        # 需要逐样本比较每个专家的ROUGE-L
        domain_per_sample_best = []
        domain_selections = defaultdict(int)

        for sample_idx in range(n_samples):
            best_score = -1.0
            best_expert = domain  # 默认选匹配专家

            for expert_type in ALL_TYPES:
                cached = all_caches.get(expert_type, {}).get(domain)
                if cached is None:
                    continue
                samples = cached.get('samples', [])
                if sample_idx >= len(samples):
                    continue

                pred = samples[sample_idx].get('prediction', '')
                ref = samples[sample_idx].get('reference', '')

                if not pred or not pred.strip():
                    continue

                # 计算单样本ROUGE-L（轻量级，不用BERTScore）
                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    score = scorer.score(ref, pred)
                    rl = score['rougeL'].fmeasure
                except Exception:
                    rl = 0.0

                if rl > best_score:
                    best_score = rl
                    best_expert = expert_type

            domain_per_sample_best.append(best_score)
            domain_selections[best_expert] += 1

        oracle_scores[domain] = float(np.mean(domain_per_sample_best)) if domain_per_sample_best else 0.0
        oracle_selections[domain] = dict(domain_selections)

    oracle_avg = np.mean(list(oracle_scores.values())) if oracle_scores else 0.0
    strategies['Oracle Routing'] = {
        'per_domain': oracle_scores,
        'average': float(oracle_avg),
        'selections': oracle_selections,
    }
    logger.info(f"Oracle Routing: 平均ROUGE-L={oracle_avg:.4f}")

    # 3c. Worst Routing（每个样本选最差专家）
    worst_scores = {}
    for domain in ALL_TYPES:
        if domain not in test_datasets:
            continue
        n_samples = len(test_datasets[domain])
        if args.test_mode:
            n_samples = min(10, n_samples)

        domain_per_sample_worst = []
        for sample_idx in range(n_samples):
            worst_score = float('inf')
            has_valid = False

            for expert_type in ALL_TYPES:
                cached = all_caches.get(expert_type, {}).get(domain)
                if cached is None:
                    continue
                samples = cached.get('samples', [])
                if sample_idx >= len(samples):
                    continue

                pred = samples[sample_idx].get('prediction', '')
                ref = samples[sample_idx].get('reference', '')
                if not pred or not pred.strip():
                    continue

                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    score = scorer.score(ref, pred)
                    rl = score['rougeL'].fmeasure
                except Exception:
                    rl = 0.0

                if rl < worst_score:
                    worst_score = rl
                    has_valid = True

            domain_per_sample_worst.append(worst_score if has_valid else 0.0)

        worst_scores[domain] = float(np.mean(domain_per_sample_worst)) if domain_per_sample_worst else 0.0

    worst_avg = np.mean(list(worst_scores.values())) if worst_scores else 0.0
    strategies['Worst Routing'] = {
        'per_domain': worst_scores,
        'average': float(worst_avg),
    }
    logger.info(f"Worst Routing: 平均ROUGE-L={worst_avg:.4f}")

    # 3d. Random Routing（3次运行取均值）
    random_seeds = [42, 43, 44]
    random_runs = []

    for seed in random_seeds:
        random_module.seed(seed)
        run_scores = {}

        for domain in ALL_TYPES:
            if domain not in test_datasets:
                continue
            n_samples = len(test_datasets[domain])
            if args.test_mode:
                n_samples = min(10, n_samples)

            domain_per_sample = []
            for sample_idx in range(n_samples):
                # 随机选一个专家
                chosen_expert = random_module.choice(ALL_TYPES)
                cached = all_caches.get(chosen_expert, {}).get(domain)
                if cached is None:
                    domain_per_sample.append(0.0)
                    continue
                samples = cached.get('samples', [])
                if sample_idx >= len(samples):
                    domain_per_sample.append(0.0)
                    continue

                pred = samples[sample_idx].get('prediction', '')
                ref = samples[sample_idx].get('reference', '')
                if not pred or not pred.strip():
                    domain_per_sample.append(0.0)
                    continue

                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    score = scorer.score(ref, pred)
                    rl = score['rougeL'].fmeasure
                except Exception:
                    rl = 0.0
                domain_per_sample.append(rl)

            run_scores[domain] = float(np.mean(domain_per_sample)) if domain_per_sample else 0.0

        run_avg = float(np.mean(list(run_scores.values()))) if run_scores else 0.0
        random_runs.append({'seed': seed, 'per_domain': run_scores, 'average': run_avg})

    random_avgs = [r['average'] for r in random_runs]
    random_mean = float(np.mean(random_avgs))
    random_std = float(np.std(random_avgs))

    # 汇总各域均值
    random_per_domain = {}
    for domain in ALL_TYPES:
        domain_vals = [r['per_domain'].get(domain, 0.0) for r in random_runs]
        random_per_domain[domain] = float(np.mean(domain_vals))

    strategies['Random Routing'] = {
        'per_domain': random_per_domain,
        'average': random_mean,
        'std': random_std,
        'runs': random_runs,
    }
    logger.info(f"Random Routing: 平均ROUGE-L={random_mean:.4f} +/- {random_std:.4f}")

    # 3e. General-Only Routing（所有样本走General Expert）
    general_only_scores = {}
    for domain in ALL_TYPES:
        general_only_scores[domain] = score_matrix.get('general', {}).get(domain, 0.0)
    general_only_avg = np.mean(list(general_only_scores.values())) if general_only_scores else 0.0
    strategies['General-Only'] = {
        'per_domain': general_only_scores,
        'average': float(general_only_avg),
    }
    logger.info(f"General-Only: 平均ROUGE-L={general_only_avg:.4f}")

    # --- 步骤4: 决策点分析 ---
    gap = oracle_avg - hard_avg
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 决策点分析")
    logger.info("=" * 60)
    logger.info(f"Oracle ROUGE-L: {oracle_avg:.4f}")
    logger.info(f"Hard   ROUGE-L: {hard_avg:.4f}")
    logger.info(f"Gap (Oracle - Hard): {gap:.4f} ({gap*100:.2f}%)")

    if gap >= 0.02:
        logger.info(">> Gap >= 2%: 建议执行 Phase 2（Soft Routing）")
        phase2_recommended = True
    else:
        logger.info(">> Gap < 2%: Hard Routing已接近理论最优，Phase 2为可选项")
        phase2_recommended = False

    # General域单独分析
    general_gap = oracle_scores.get('general', 0) - hard_scores.get('general', 0)
    logger.info(f"\nGeneral域 Oracle-Hard Gap: {general_gap:.4f} ({general_gap*100:.2f}%)")

    results = {
        'phase': 'phase1',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategies': strategies,
        'score_matrix': score_matrix,
        'oracle_selections': oracle_selections,
        'gap_analysis': {
            'overall_gap': float(gap),
            'general_domain_gap': float(general_gap),
            'phase2_recommended': phase2_recommended,
            'per_domain_gaps': {
                domain: float(oracle_scores.get(domain, 0) - hard_scores.get(domain, 0))
                for domain in ALL_TYPES
            },
        },
        'test_set_sizes': {et: len(test_datasets.get(et, [])) for et in ALL_TYPES},
    }

    # 保存Phase 1结果
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'phase1_results.json')
    logger.info(f"\nPhase 1 结果已保存: {EXP_DIR / 'phase1_results.json'}")

    return results


# ========== Phase 2: Soft Routing ==========

def run_phase2(args, phase1_results=None):
    """
    Phase 2: Soft Routing验证（仅General域）

    使用PEFT add_weighted_adapter 参数级融合

    Args:
        args: 命令行参数
        phase1_results: Phase 1结果（用于对比）

    Returns:
        Dict: Phase 2结果
    """
    logger.info("=" * 80)
    logger.info("Phase 2: Soft Routing验证（General域）")
    logger.info("=" * 80)

    # 检查PEFT版本
    from src.routing.soft_router import check_peft_version, SoftRouter, \
        build_type_aware_weights, group_general_samples_by_type

    if not check_peft_version():
        logger.error("PEFT版本不支持 add_weighted_adapter，跳过Phase 2")
        return {'phase': 'phase2', 'status': 'skipped', 'reason': 'peft_version'}

    # 加载General测试集
    logger.info("加载General测试集...")
    general_data = GeneralDatasetLoader().load_all_data()
    _, _, general_test = split_dataset_for_expert(general_data, 'general')
    logger.info(f"General测试集: {len(general_test)} 条")

    if args.test_mode:
        general_test = general_test[:10]
        logger.info(f"测试模式: 截取 {len(general_test)} 条")

    # 按data_type分组
    type_groups = group_general_samples_by_type(general_test)

    # 获取adapter路径
    adapter_paths = {}
    for expert_name in ['text', 'image', 'uml', 'general']:
        adapter_path = path_cfg.get_expert_weight_path(expert_name)
        adapter_paths[f'{expert_name}_expert'] = str(adapter_path)
        logger.info(f"  {expert_name}_expert: {adapter_path}")

    # 加载基础模型和所有adapter
    logger.info("\n加载基础模型...")
    from models.language_model import LanguageModel
    lm = LanguageModel(use_4bit=True)

    soft_router = SoftRouter(
        base_model=lm.model,
        tokenizer=lm.tokenizer,
        adapter_paths=adapter_paths,
    )

    if not soft_router.load_all_adapters():
        logger.error("加载adapter失败，跳过Phase 2")
        return {'phase': 'phase2', 'status': 'failed', 'reason': 'adapter_load'}

    # 网格搜索融合比例
    alpha_values = [0.3, 0.5, 0.7]
    all_alpha_results = {}

    for alpha in alpha_values:
        logger.info(f"\n--- 融合比例 alpha={alpha} ---")

        predictions = [''] * len(general_test)

        # 按data_type分批融合推理
        for data_type, indices in type_groups.items():
            if not indices:
                continue

            weights = build_type_aware_weights(data_type, alpha=alpha)
            logger.info(f"  {data_type}类型 ({len(indices)}条): 权重={weights}")

            if not soft_router.merge_adapters(weights, merged_name=f"merged_{data_type}"):
                logger.error(f"  融合失败: {data_type}")
                continue

            # 构建prompt并推理
            batch_inputs = [general_test[i]['input'] for i in indices]

            # 使用专家的batch_generate_instruction接口格式
            # 这里需要直接使用lm的generate方法（已加载merged adapter）
            from models.prompt_templates.general_template import GeneralInstructionTemplate

            for batch_start in range(0, len(batch_inputs), 4):
                batch_end = min(batch_start + 4, len(batch_inputs))
                batch = batch_inputs[batch_start:batch_end]

                prompts = []
                for inp in batch:
                    prompt = GeneralInstructionTemplate.build_prompt(inp, force_type=data_type)
                    prompts.append(prompt)

                try:
                    # 直接使用底层模型生成
                    batch_preds = lm.generate_batch(
                        prompts=prompts,
                        max_new_tokens=2048,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        batch_size=len(prompts),
                    )
                except Exception as e:
                    logger.error(f"  生成失败: {e}")
                    batch_preds = [''] * len(prompts)

                # 写入结果
                for j, pred in enumerate(batch_preds):
                    global_idx = indices[batch_start + j]
                    predictions[global_idx] = pred

        # 计算指标
        refs = [d['output'] for d in general_test]
        metrics = compute_all_metrics(
            predictions, refs, use_bertscore=not args.no_bertscore
        )
        rougeL = _get_rougeL(metrics)

        all_alpha_results[alpha] = {
            'rougeL': rougeL,
            'metrics': metrics,
        }
        logger.info(f"  alpha={alpha}: ROUGE-L={rougeL:.4f}")

        # 保存缓存
        cache_dir = CACHE_DIR / 'exp9_soft_routing'
        samples = [
            {'index': i, 'input': general_test[i]['input'],
             'prediction': predictions[i], 'reference': refs[i]}
            for i in range(len(general_test))
        ]
        save_predictions_cache(
            samples, 'exp9_soft_routing', 'general',
            {'alpha': alpha, 'strategy': 'type_aware_soft'},
            cache_dir, f'general_soft_alpha{alpha}_predictions.json'
        )

    # 清理
    soft_router.cleanup()
    del lm
    _cleanup_gpu()

    # 选择最优alpha
    best_alpha = max(all_alpha_results, key=lambda a: all_alpha_results[a]['rougeL'])
    best_rougeL = all_alpha_results[best_alpha]['rougeL']

    # 对比Hard Routing基线
    hard_general_rougeL = 0.0
    if phase1_results:
        hard_general_rougeL = phase1_results.get('strategies', {}).get(
            'Hard Routing', {}).get('per_domain', {}).get('general', 0.0)

    improvement = best_rougeL - hard_general_rougeL

    logger.info(f"\n最优融合比例: alpha={best_alpha}")
    logger.info(f"Soft Routing ROUGE-L: {best_rougeL:.4f}")
    logger.info(f"Hard Routing ROUGE-L: {hard_general_rougeL:.4f}")
    logger.info(f"提升: {improvement:.4f} ({improvement*100:.2f}%)")

    results = {
        'phase': 'phase2',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'alpha_search': {
            str(a): {'rougeL': r['rougeL']} for a, r in all_alpha_results.items()
        },
        'best_alpha': best_alpha,
        'best_rougeL': best_rougeL,
        'hard_baseline_rougeL': hard_general_rougeL,
        'improvement': improvement,
    }

    save_experiment_results(results, EXP_DIR, 'phase2_results.json')
    logger.info(f"Phase 2 结果已保存: {EXP_DIR / 'phase2_results.json'}")

    return results


# ========== Phase 3: 可视化 ==========

def run_phase3(args, phase1_results=None, phase2_results=None):
    """
    Phase 3: 贡献度分析 + 8张可视化图表

    Args:
        phase1_results: Phase 1结果
        phase2_results: Phase 2结果（可为None）
    """
    logger.info("=" * 80)
    logger.info("Phase 3: 贡献度分析与可视化")
    logger.info("=" * 80)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 如果没有传入结果，尝试从文件加载
    if phase1_results is None:
        p1_path = EXP_DIR / 'phase1_results.json'
        if p1_path.exists():
            import json
            with open(p1_path, 'r') as f:
                phase1_results = json.load(f)
        else:
            logger.error("Phase 1结果文件不存在，无法进行可视化")
            return

    if phase2_results is None:
        p2_path = EXP_DIR / 'phase2_results.json'
        if p2_path.exists():
            import json
            with open(p2_path, 'r') as f:
                phase2_results = json.load(f)

    strategies = phase1_results.get('strategies', {})
    score_matrix = phase1_results.get('score_matrix', {})
    oracle_selections = phase1_results.get('oracle_selections', {})

    # ---- 图1: 路由贡献区间图 ----
    _plot_contribution_band(strategies, phase2_results)

    # ---- 图2: Oracle选择热力图 ----
    _plot_oracle_heatmap(oracle_selections)

    # ---- 图3: 分域策略对比柱状图 ----
    _plot_per_domain_comparison(strategies)

    # ---- 图4: General域Oracle分布饼图 ----
    _plot_general_oracle_distribution(oracle_selections)

    # ---- 图5: Oracle-Hard差距分析 ----
    _plot_gap_analysis(phase1_results.get('gap_analysis', {}))

    # ---- 图6: Random路由方差分析 ----
    _plot_random_variance(strategies.get('Random Routing', {}))

    # ---- 图7: Soft vs Hard对比图 ----
    if phase2_results and phase2_results.get('phase') == 'phase2':
        _plot_soft_vs_hard(phase2_results)

    # ---- 图8: 汇总表格 ----
    _plot_summary_table(strategies, phase2_results)

    # 生成汇总报告
    _generate_report(phase1_results, phase2_results)

    logger.info(f"\n全部图表已保存至: {PLOT_DIR}")


def _plot_contribution_band(strategies, phase2_results=None):
    """图1: 5策略ROUGE-L区间对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategy_order = ['Worst Routing', 'Random Routing', 'Hard Routing',
                      'General-Only', 'Oracle Routing']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#95a5a6', '#2ecc71']

    x_vals = []
    labels = []
    errs = []

    for name in strategy_order:
        s = strategies.get(name, {})
        avg = s.get('average', 0)
        std = s.get('std', 0)
        x_vals.append(avg)
        labels.append(name)
        errs.append(std)

    # 添加Soft Routing（如果有Phase 2结果）
    if phase2_results and 'best_rougeL' in phase2_results:
        # 放在Hard和Oracle之间
        insert_idx = 3  # Hard之后
        x_vals.insert(insert_idx, phase2_results['best_rougeL'])
        labels.insert(insert_idx, f"Soft Routing\n(alpha={phase2_results.get('best_alpha', '?')})")
        errs.insert(insert_idx, 0)
        colors.insert(insert_idx, '#9b59b6')

    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, x_vals, xerr=errs, color=colors, edgecolor='white',
                   height=0.6, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Average ROUGE-L', fontsize=12)
    ax.set_title('Routing Strategy Comparison: Contribution Band', fontsize=14)

    # 添加数值标签
    for bar, val, err in zip(bars, x_vals, errs):
        if err > 0:
            label = f'{val:.4f} +/- {err:.4f}'
        else:
            label = f'{val:.4f}'
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)

    ax.set_xlim(0, max(x_vals) * 1.15 if x_vals else 1.0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'routing_contribution_band.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [1/8] routing_contribution_band.png")


def _plot_oracle_heatmap(oracle_selections):
    """图2: 4x4 Oracle选择热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    matrix = np.zeros((4, 4))
    for d_idx, domain in enumerate(ALL_TYPES):
        selections = oracle_selections.get(domain, {})
        total = sum(selections.values()) if selections else 1
        for e_idx, expert in enumerate(ALL_TYPES):
            count = selections.get(expert, 0)
            matrix[d_idx][e_idx] = count / total * 100 if total > 0 else 0

    sns.heatmap(
        matrix, annot=True, fmt='.1f', cmap='YlOrRd',
        xticklabels=[f'{t}_expert' for t in ALL_TYPES],
        yticklabels=[f'{t}_test' for t in ALL_TYPES],
        ax=ax, vmin=0, vmax=100,
        cbar_kws={'label': 'Oracle Selection Rate (%)'}
    )
    ax.set_title('Oracle Expert Selection Heatmap', fontsize=14)
    ax.set_xlabel('Expert Used', fontsize=12)
    ax.set_ylabel('Test Domain', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'oracle_selection_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [2/8] oracle_selection_heatmap.png")


def _plot_per_domain_comparison(strategies):
    """图3: 分域柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    strategy_names = ['Worst Routing', 'Random Routing', 'Hard Routing',
                      'General-Only', 'Oracle Routing']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#95a5a6', '#2ecc71']

    x = np.arange(len(ALL_TYPES))
    width = 0.15
    offsets = np.arange(len(strategy_names)) - len(strategy_names) / 2 + 0.5

    for i, (name, color) in enumerate(zip(strategy_names, colors)):
        s = strategies.get(name, {})
        per_domain = s.get('per_domain', {})
        vals = [per_domain.get(d, 0) for d in ALL_TYPES]
        ax.bar(x + offsets[i] * width, vals, width, label=name, color=color, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t.capitalize()}' for t in ALL_TYPES], fontsize=11)
    ax.set_ylabel('ROUGE-L', fontsize=12)
    ax.set_title('Per-Domain Routing Strategy Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'per_domain_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [3/8] per_domain_strategy_comparison.png")


def _plot_general_oracle_distribution(oracle_selections):
    """图4: General域Oracle选择分布饼图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    general_sel = oracle_selections.get('general', {})
    if not general_sel:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
        ax.set_title('General Domain: Oracle Expert Selection', fontsize=14)
    else:
        labels = [f'{k}' for k in general_sel.keys()]
        sizes = list(general_sel.values())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(labels)]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, pctdistance=0.85
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax.set_title('General Domain: Oracle Expert Selection Distribution', fontsize=14)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'general_domain_oracle_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [4/8] general_domain_oracle_distribution.png")


def _plot_gap_analysis(gap_analysis):
    """图5: Oracle-Hard差距分析"""
    fig, ax = plt.subplots(figsize=(8, 5))

    per_domain_gaps = gap_analysis.get('per_domain_gaps', {})
    domains = list(per_domain_gaps.keys())
    gaps = [per_domain_gaps[d] * 100 for d in domains]

    colors = ['#e74c3c' if g < 0 else '#2ecc71' for g in gaps]
    bars = ax.bar(domains, gaps, color=colors, edgecolor='white', width=0.6)

    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('ROUGE-L Gap (%)', fontsize=12)
    ax.set_title('Oracle - Hard Routing Gap by Domain', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticklabels([d.capitalize() for d in domains], fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'routing_gap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [5/8] routing_gap_analysis.png")


def _plot_random_variance(random_data):
    """图6: Random路由方差分析"""
    fig, ax = plt.subplots(figsize=(8, 5))

    runs = random_data.get('runs', [])
    if not runs:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
    else:
        for run in runs:
            seed = run['seed']
            per_domain = run.get('per_domain', {})
            domains = list(per_domain.keys())
            vals = [per_domain[d] for d in domains]
            ax.plot(domains, vals, 'o-', label=f'seed={seed}', markersize=6)

        # 均值线
        mean_per_domain = {}
        for d in ALL_TYPES:
            d_vals = [r['per_domain'].get(d, 0) for r in runs]
            mean_per_domain[d] = np.mean(d_vals)
        ax.plot(list(mean_per_domain.keys()), list(mean_per_domain.values()),
                's--', color='black', label='Mean', markersize=8, linewidth=2)

        ax.set_ylabel('ROUGE-L', fontsize=12)
        ax.set_title('Random Routing Variance (3 Seeds)', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xticklabels([d.capitalize() for d in ALL_TYPES], fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'random_routing_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [6/8] random_routing_variance.png")


def _plot_soft_vs_hard(phase2_results):
    """图7: Soft vs Hard对比图"""
    fig, ax = plt.subplots(figsize=(8, 5))

    alpha_search = phase2_results.get('alpha_search', {})
    hard_baseline = phase2_results.get('hard_baseline_rougeL', 0)

    alphas = sorted(alpha_search.keys(), key=float)
    rougeL_vals = [alpha_search[a]['rougeL'] for a in alphas]

    ax.plot([float(a) for a in alphas], rougeL_vals, 'o-', color='#9b59b6',
            label='Soft Routing', markersize=8, linewidth=2)
    ax.axhline(y=hard_baseline, color='#3498db', linestyle='--',
               linewidth=2, label=f'Hard Routing ({hard_baseline:.4f})')

    # 标注最优点
    best_alpha = phase2_results.get('best_alpha')
    best_rougeL = phase2_results.get('best_rougeL', 0)
    if best_alpha is not None:
        ax.annotate(
            f'Best: alpha={best_alpha}\nROUGE-L={best_rougeL:.4f}',
            xy=(float(best_alpha), best_rougeL),
            xytext=(float(best_alpha) + 0.08, best_rougeL + 0.005),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8)
        )

    ax.set_xlabel('Alpha (Specialized Expert Weight)', fontsize=12)
    ax.set_ylabel('ROUGE-L (General Domain)', fontsize=12)
    ax.set_title('Soft Routing vs Hard Routing (General Domain)', fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'soft_vs_hard_general.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [7/8] soft_vs_hard_general.png")


def _plot_summary_table(strategies, phase2_results=None):
    """图8: 论文级汇总表格"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    strategy_order = ['Worst Routing', 'Random Routing', 'General-Only',
                      'Hard Routing', 'Oracle Routing']
    headers = ['Strategy', 'Text', 'Image', 'UML', 'General', 'Average']

    table_data = []
    for name in strategy_order:
        s = strategies.get(name, {})
        per_d = s.get('per_domain', {})
        row = [name]
        for d in ALL_TYPES:
            row.append(f"{per_d.get(d, 0):.4f}")
        avg = s.get('average', 0)
        std = s.get('std', 0)
        if std > 0:
            row.append(f"{avg:.4f} +/- {std:.4f}")
        else:
            row.append(f"{avg:.4f}")
        table_data.append(row)

    # 添加Soft Routing行
    if phase2_results and 'best_rougeL' in phase2_results:
        best_alpha = phase2_results.get('best_alpha', '?')
        row = [f'Soft Routing (a={best_alpha})', '-', '-', '-',
               f"{phase2_results['best_rougeL']:.4f}", '-']
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # 表头样式
    for j in range(len(headers)):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Hard Routing行高亮
    hard_row_idx = strategy_order.index('Hard Routing') + 1
    for j in range(len(headers)):
        table[hard_row_idx, j].set_facecolor('#d4e6f1')

    ax.set_title('Exp9: Routing Strategy Comparison Summary', fontsize=14,
                 fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [8/8] summary_table.png")


def _generate_report(phase1_results, phase2_results=None):
    """生成Markdown汇总报告"""
    strategies = phase1_results.get('strategies', {})
    gap_analysis = phase1_results.get('gap_analysis', {})

    lines = [
        "# Experiment 9: Routing Strategy Comparison",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Phase 1: Oracle上下界分析",
        "\n### 策略对比结果",
        "",
        "| 策略 | Text | Image | UML | General | 平均 |",
        "|------|------|-------|-----|---------|------|",
    ]

    for name in ['Worst Routing', 'Random Routing', 'General-Only',
                 'Hard Routing', 'Oracle Routing']:
        s = strategies.get(name, {})
        per_d = s.get('per_domain', {})
        avg = s.get('average', 0)
        std_str = f" +/- {s.get('std', 0):.4f}" if s.get('std', 0) > 0 else ""
        lines.append(
            f"| {name} | {per_d.get('text', 0):.4f} | {per_d.get('image', 0):.4f} | "
            f"{per_d.get('uml', 0):.4f} | {per_d.get('general', 0):.4f} | "
            f"{avg:.4f}{std_str} |"
        )

    lines.extend([
        "",
        "### 差距分析",
        f"- Oracle-Hard 总体差距: {gap_analysis.get('overall_gap', 0)*100:.2f}%",
        f"- General域差距: {gap_analysis.get('general_domain_gap', 0)*100:.2f}%",
        f"- Phase 2建议: {'建议执行' if gap_analysis.get('phase2_recommended') else '可选'}",
    ])

    per_gaps = gap_analysis.get('per_domain_gaps', {})
    for d in ALL_TYPES:
        lines.append(f"  - {d}: {per_gaps.get(d, 0)*100:.2f}%")

    if phase2_results and phase2_results.get('phase') == 'phase2':
        lines.extend([
            "",
            "## Phase 2: Soft Routing结果",
            f"- 最优alpha: {phase2_results.get('best_alpha')}",
            f"- Soft Routing ROUGE-L: {phase2_results.get('best_rougeL', 0):.4f}",
            f"- Hard Routing ROUGE-L: {phase2_results.get('hard_baseline_rougeL', 0):.4f}",
            f"- 提升: {phase2_results.get('improvement', 0)*100:.2f}%",
        ])

    report_path = EXP_DIR / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"报告已保存: {report_path}")


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description='Exp9: Routing Strategy Comparison')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='只运行指定阶段（1/2/3）')
    parser.add_argument('--all', action='store_true',
                        help='运行全部阶段（Phase 1 + Phase 2(条件) + Phase 3）')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='强制重新推理，忽略缓存')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='跳过BERTScore计算（加速）')
    parser.add_argument('--test-mode', action='store_true',
                        help='测试模式（每域仅10条）')
    parser.add_argument('--skip-phase2-check', action='store_true',
                        help='跳过Phase 2的Gap检查，强制执行')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("实验9：路由策略对比与路由器贡献度分析")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"参数: phase={args.phase}, all={args.all}, "
              f"test_mode={args.test_mode}, no_bertscore={args.no_bertscore}")
    logger.info("=" * 80)

    phase1_results = None
    phase2_results = None

    if args.phase == 1 or args.all:
        phase1_results = run_phase1(args)

    if args.phase == 2 or args.all:
        # 加载Phase 1结果（如果不是刚运行的）
        if phase1_results is None:
            import json
            p1_path = EXP_DIR / 'phase1_results.json'
            if p1_path.exists():
                with open(p1_path, 'r') as f:
                    phase1_results = json.load(f)

        # 检查是否需要执行Phase 2
        if args.skip_phase2_check:
            logger.info("跳过Gap检查，强制执行Phase 2")
            phase2_results = run_phase2(args, phase1_results)
        elif phase1_results:
            gap = phase1_results.get('gap_analysis', {}).get('overall_gap', 0)
            general_gap = phase1_results.get('gap_analysis', {}).get('general_domain_gap', 0)
            if gap >= 0.02 or general_gap >= 0.02:
                logger.info(f"Gap={gap:.4f} >= 0.02，执行Phase 2")
                phase2_results = run_phase2(args, phase1_results)
            else:
                logger.info(f"Gap={gap:.4f} < 0.02，跳过Phase 2（Hard Routing已接近理论最优）")
                logger.info("如需强制执行，请使用 --skip-phase2-check")
        else:
            logger.warning("Phase 1结果不可用，跳过Phase 2")

    if args.phase == 3 or args.all:
        run_phase3(args, phase1_results, phase2_results)

    # 合并最终结果
    if args.all or args.phase is None:
        final_results = {
            'experiment': 'exp9_routing_strategy',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        if phase1_results:
            final_results['phase1'] = phase1_results
        if phase2_results:
            final_results['phase2'] = phase2_results
        save_experiment_results(final_results, EXP_DIR, 'results.json')

    logger.info("\n" + "=" * 80)
    logger.info(f"实验9完成 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结果目录: {EXP_DIR}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()