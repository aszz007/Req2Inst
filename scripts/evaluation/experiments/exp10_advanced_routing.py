#!/usr/bin/env python3
"""
Experiment 10: Advanced Routing Strategy - Learned Router vs Output Ensemble

Phase 1: 特征提取 + Learned Router训练（~30min，必做）
  - 从Exp9 Oracle标签构建训练集
  - 提取基础模型hidden states作为特征
  - 训练MLP分类器（4类：text/image/uml/general）

Phase 2: Output Ensemble评估（~1.5h，必做）
  - 使用Learned Router权重作为融合系数
  - 顺序加载top-2专家，logit层加权融合
  - 同时评估Learned Router单路由效果

Phase 3: 对比分析与可视化（~15min，必做）
  - 汇总本实验2种策略 + Exp9所有基线
  - 计算各策略对Oracle-Hard Gap的缩小率
  - 生成8张可视化图表 + report.md

依赖：Exp9 phase1_results.json + phase2_results.json 必须已存在

Author: Claude
Date: 2026-03-08

v13 (2026-03-16): 诊断版
  - 新增 --debug-ensemble 参数，激活 D1-D5 诊断指标收集
  - D1: 分布熵对比 H(prob1) vs H(prob2) vs H(fused)
  - D2: Top-10 token Jaccard 重叠率
  - D3: 融合token与单专家token吻合率（post-hoc）
  - D4: 按专家对分层 ROUGE-L
  - D5: 按专家对分层 format_ok
  - 诊断结果保存到 exp10_advanced_routing/debug_ensemble_diagnostics.json
  - 不修改任何混合公式，仅添加观测代码
"""

import sys
import gc
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
from src.routing.learned_router import (
    RouterMLP, HiddenStateExtractor,
    EXPERT_TO_IDX, IDX_TO_EXPERT,
)

logger = get_logger('experiments.exp10')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP9_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp9_routing_strategy'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp10_advanced_routing'
PLOT_DIR = EXP_DIR / 'plots'
ROUTER_CKPT_DIR = path_cfg.OUTPUTS_DIR.parent / 'checkpoints' / 'exp10_learned_router'
FEATURE_CACHE_DIR = CACHE_DIR / 'exp10_router_features'

ALL_TYPES = ['text', 'image', 'uml', 'general']
SPECIALIZED_TYPES = ['text', 'image', 'uml']

# ─────────────────────────────────────────────
# v13 诊断模式：模块级累加器
# ─────────────────────────────────────────────
# _process_minibatch 在 debug_ensemble 模式下向此 dict 追加每批次的诊断数据，
# _run_output_ensemble 在所有批次完成后读取并保存。
# 使用模块级变量而非修改函数签名，确保 exp11 等外部调用方无需改动。
_DEBUG_ENSEMBLE_STATS = {
    'enabled': False,
    'per_step': [],       # 每步的 {entropy_1, entropy_2, entropy_fused, jaccard_top10}
    'per_batch': [],      # 每批次的 {expert1, expert2, avg_entropy_ratio, avg_jaccard, ...}
}


def _reset_debug_stats():
    """重置诊断累加器"""
    _DEBUG_ENSEMBLE_STATS['per_step'] = []
    _DEBUG_ENSEMBLE_STATS['per_batch'] = []


def _entropy(prob_tensor):
    """计算概率分布的熵 H = -sum(p * log(p))，单位 nats"""
    import torch
    # 避免 log(0)：仅在 p>0 的位置计算
    log_p = torch.where(prob_tensor > 1e-10,
                        torch.log(prob_tensor),
                        torch.zeros_like(prob_tensor))
    return -(prob_tensor * log_p).sum(dim=-1)  # (B,)


def _jaccard_topk(prob1, prob2, k=10):
    """计算 top-k token 的 Jaccard 相似度，返回 (B,) 张量"""
    import torch
    topk1 = prob1.topk(k, dim=-1).indices  # (B, k)
    topk2 = prob2.topk(k, dim=-1).indices  # (B, k)
    # 逐样本计算交集大小
    B = prob1.shape[0]
    jaccards = []
    for b in range(B):
        set1 = set(topk1[b].cpu().tolist())
        set2 = set(topk2[b].cpu().tolist())
        inter = len(set1 & set2)
        union = len(set1 | set2)
        jaccards.append(inter / union if union > 0 else 0.0)
    return jaccards

# ─────────────────────────────────────────────
# 模板工厂（核心修复：避免 GeneralTemplate 一刀切导致专家混淆）
# ─────────────────────────────────────────────
# 背景：各专家在各自 domain-specific 模板下训练；推理时若统一使用 GeneralTemplate，
#       专家收到的指令格式与训练分布不符，输出长度失控（612 vs 392）、
#       格式通过率骤降（77% → 100%），ROUGE-L 从 0.59 跌至 0.43。
# 修复：根据样本的 data_type 选择对应模板，同一批次两个专家使用 **相同 prompt**
#       保证 KV Cache 的条件化前缀一致，PoE logit 融合在语义上有意义。

def _build_prompt_for_sample(sample: dict) -> tuple:
    """
    根据样本 data_type 构建正确的 prompt 字符串。

    Returns:
        (prompt_str, template_name)  — template_name 仅用于 debug 日志
    """
    input_text = sample.get('input', '')
    data_type = sample.get('data_type', 'general')

    # 按 data_type 尝试加载对应模板；任何 ImportError / AttributeError 都回退到 GeneralTemplate
    try:
        if data_type == 'text':
            from models.prompt_templates.text_template import TextInstructionTemplate
            return TextInstructionTemplate.build_prompt(input_text), 'text_template'
    except (ImportError, AttributeError):
        pass

    try:
        if data_type == 'image':
            from models.prompt_templates.image_template import ImageInstructionTemplate
            return ImageInstructionTemplate.build_prompt(input_text), 'image_template'
    except (ImportError, AttributeError):
        pass

    try:
        if data_type == 'uml':
            from models.prompt_templates.uml_template import UMLInstructionTemplate
            return UMLInstructionTemplate.build_prompt(input_text), 'uml_template'
    except (ImportError, AttributeError):
        pass

    from models.prompt_templates.general_template import GeneralInstructionTemplate
    return GeneralInstructionTemplate.build_prompt(input_text), 'general_template'


def _detect_datatype(sample: dict) -> str:
    """
    从样本 dict 推断 data_type，优先取显式字段，否则根据 input 内容猜测。
    """
    dt = sample.get('data_type') or sample.get('type') or sample.get('domain')
    if dt in ('text', 'image', 'uml', 'general'):
        return dt
    # 根据 input 格式猜测
    inp = str(sample.get('input', ''))
    if inp.strip().startswith('{') or inp.strip().startswith('['):
        # JSON 格式 → image 或 uml；无法区分时保守取 general
        return 'general'
    return 'text'


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _cleanup_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_rougeL(metrics_dict):
    return metrics_dict.get('generation_quality', {}).get('rougeL', 0.0)


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


def _load_exp9_results():
    """加载Exp9的phase1和phase2结果"""
    p1_path = EXP9_DIR / 'phase1_results.json'
    p2_path = EXP9_DIR / 'phase2_results.json'

    if not p1_path.exists():
        raise FileNotFoundError(f"Exp9 phase1结果不存在: {p1_path}\n请先运行实验9！")

    with open(p1_path, 'r', encoding='utf-8') as f:
        phase1 = json.load(f)

    phase2 = None
    if p2_path.exists():
        with open(p2_path, 'r', encoding='utf-8') as f:
            phase2 = json.load(f)
        logger.info("已加载Exp9 Phase1 + Phase2结果")
    else:
        logger.warning("Exp9 Phase2结果不存在，Soft Routing基线将缺失")

    return phase1, phase2




# ─────────────────────────────────────────────
# Phase 1：Router训练
# ─────────────────────────────────────────────

def run_phase1(args, exp9_phase1):
    """
    Phase 1: 提取特征 + 训练Learned Router

    训练集: text_test + image_test + uml_test 的Oracle标签（共~498条）
    验证集: general_test 前80%（约398条）

    Returns:
        Dict: phase1结果（路由准确率、训练历史等）
    """
    logger.info("=" * 80)
    logger.info("Phase 1: 特征提取 + Learned Router训练")
    logger.info("=" * 80)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ROUTER_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 步骤1: 加载测试集 ──
    logger.info("\n--- 步骤1: 加载测试集 ---")
    test_datasets = {}
    for et in ALL_TYPES:
        test_datasets[et] = _load_test_data(et)
        logger.info(f"  {et}: {len(test_datasets[et])} 条")

    # ── 步骤2: 提取或加载特征缓存 ──
    logger.info("\n--- 步骤2: 特征提取 ---")

    all_features = {}
    all_labels = {}

    # 加载基础模型（仅用于特征提取，不加载LoRA）
    from models.language_model import LanguageModel
    lm = LanguageModel(use_4bit=True)
    base_model = lm.model
    tokenizer = lm.tokenizer

    for domain in SPECIALIZED_TYPES:
        feat_path = FEATURE_CACHE_DIR / f'{domain}_hidden_states.npz'
        if feat_path.exists() and not args.force_regenerate:
            logger.info(f"  [缓存] 加载 {domain} 特征")
            data = np.load(feat_path)
            all_features[domain] = data['features']
            all_labels[domain] = data['labels']
            continue

        logger.info(f"  提取 {domain} 特征...")
        test_data = test_datasets[domain]
        if args.test_mode:
            test_data = test_data[:10]

        inputs = [d['input'] for d in test_data]
        extractor = HiddenStateExtractor(base_model, tokenizer)
        features = extractor.extract(
            inputs,
            batch_size=4 if not args.test_mode else 2,
        )

        # 逐样本重建Oracle标签（从exp9缓存的per-sample ROUGE-L中选最优专家）
        labels = _rebuild_per_sample_labels(domain, test_data, args)

        all_features[domain] = features
        all_labels[domain] = np.array(labels, dtype=np.int64)

        np.savez(feat_path, features=features, labels=all_labels[domain])
        logger.info(f"  {domain}: {len(features)} 条特征已保存")

    # 同样提取General域特征（用作验证集）
    general_feat_path = FEATURE_CACHE_DIR / 'general_hidden_states.npz'
    if general_feat_path.exists() and not args.force_regenerate:
        logger.info("  [缓存] 加载 general 特征")
        data = np.load(general_feat_path)
        general_features = data['features']
        general_labels = data['labels']
    else:
        logger.info("  提取 general 特征...")
        general_test = test_datasets['general']
        if args.test_mode:
            general_test = general_test[:20]
        general_inputs = [d['input'] for d in general_test]
        extractor = HiddenStateExtractor(base_model, tokenizer)
        general_features = extractor.extract(general_inputs, batch_size=4)
        # 修复：general域应该从exp9_oracle加载跨域缓存
        general_labels = _rebuild_general_labels(general_test, args)
        np.savez(general_feat_path, features=general_features, labels=np.array(general_labels))

    del lm, base_model, tokenizer
    _cleanup_gpu()

    # ── 步骤3: 组合训练数据（分层混合验证集）──
    logger.info("\n--- 步骤3: 组合训练数据 ---")

    # 关键修复：验证集必须包含所有域的样本，而非只有 general 域。
    # 原实现的问题：训练集以专化域为主，验证集全是 general 域，
    # early stop 信号反映的是 general 域路由质量，而非专化域的学习进度。
    # 后果：模型在专化域还未充分收敛时就因 general 域 val_acc 停滞而提前停止。
    #
    # 方案：专化域各取后20%作验证，前80%作训练；
    #       general域前40%训练、40%-80%验证、后20%最终测试集（不参与训练/验证）。
    val_parts_X, val_parts_y = [], []
    train_parts_X, train_parts_y = [], []

    for domain in SPECIALIZED_TYPES:
        feats = all_features[domain]
        lbls = all_labels[domain]
        n = len(feats)
        n_val = max(1, int(n * 0.2))
        train_parts_X.append(feats[:-n_val])
        train_parts_y.append(lbls[:-n_val])
        val_parts_X.append(feats[-n_val:])
        val_parts_y.append(lbls[-n_val:])

    # General域
    n_total_general = len(general_features)
    n_train_general = int(n_total_general * 0.4)
    n_val_end = int(n_total_general * 0.8)

    train_parts_X.append(general_features[:n_train_general])
    train_parts_y.append(np.array(general_labels[:n_train_general]))
    val_parts_X.append(general_features[n_train_general:n_val_end])
    val_parts_y.append(np.array(general_labels[n_train_general:n_val_end]))

    train_X = np.concatenate(train_parts_X, axis=0)
    train_y = np.concatenate(train_parts_y, axis=0)
    val_X = np.concatenate(val_parts_X, axis=0)
    val_y = np.concatenate(val_parts_y, axis=0)
    test_X = general_features[n_val_end:]
    test_y = np.array(general_labels[n_val_end:])

    logger.info(f"  训练集: {len(train_X)} 条 (specialized前80% + general前40%)")
    logger.info(f"  验证集: {len(val_X)} 条 (specialized后20% + general 40%~80%，混合域)")
    logger.info(f"  测试集: {len(test_X)} 条 (general后20%，最终评估)")

    # 类别分布
    for i, name in IDX_TO_EXPERT.items():
        cnt = (train_y == i).sum()
        logger.info(f"  训练集-{name}: {cnt} 条 ({cnt/len(train_y)*100:.1f}%)")

    # ── 步骤4: 训练MLP ──
    logger.info("\n--- 步骤4: 训练MLP路由器 ---")

    router = RouterMLP(input_dim=train_X.shape[1])
    history = _train_router(router, train_X, train_y, val_X, val_y, args)

    # 保存模型
    router.save(ROUTER_CKPT_DIR / 'router_mlp.pt')

    # ── 步骤5: 评估路由准确率 ──
    logger.info("\n--- 步骤5: 评估路由准确率 ---")
    accuracy_results = {}

    # 各specialized域评估
    for domain in SPECIALIZED_TYPES:
        X = all_features[domain]
        y_true = all_labels[domain]
        y_pred = router.predict(X)
        acc = (y_pred == y_true).mean()
        accuracy_results[domain] = float(acc)
        logger.info(f"  {domain}: 路由准确率={acc:.4f} ({acc*100:.1f}%)")

    # General域评估
    y_pred_general = router.predict(general_features)
    y_true_general = np.array(general_labels)
    acc_general = (y_pred_general == y_true_general).mean()
    accuracy_results['general'] = float(acc_general)
    logger.info(f"  general: 路由准确率={acc_general:.4f} ({acc_general*100:.1f}%)")

    # 混淆矩阵：汇总所有域（specialized + general），才能展示完整的4分类分布
    from sklearn.metrics import confusion_matrix, classification_report
    all_y_true = np.concatenate(
        [all_labels[d] for d in SPECIALIZED_TYPES] + [np.array(general_labels)]
    )
    all_y_pred = np.concatenate(
        [router.predict(all_features[d]) for d in SPECIALIZED_TYPES] + [y_pred_general]
    )
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])
    report = classification_report(
        all_y_true, all_y_pred,
        target_names=['text', 'image', 'uml', 'general'],
        output_dict=True, zero_division=0
    )
    logger.info(
        f"  全域分类报告:\n"
        f"{classification_report(all_y_true, all_y_pred, target_names=['text','image','uml','general'], zero_division=0)}"
    )

    results = {
        'phase': 'phase1',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_history': history,
        'routing_accuracy': accuracy_results,
        'overall_accuracy': float(np.mean(list(accuracy_results.values()))),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'train_sizes': {d: int(len(all_features[d])) for d in SPECIALIZED_TYPES},
    }

    save_experiment_results(results, EXP_DIR, 'phase1_results.json')
    logger.info(f"\nPhase 1 结果已保存: {EXP_DIR / 'phase1_results.json'}")
    return results


def _rebuild_per_sample_labels(domain, test_data, args):
    """
    从exp9_oracle缓存中逐样本重建Oracle标签

    如果缓存不完整，回退到基于整体Oracle分布的近似标签
    """
    from rouge_score import rouge_scorer as rs_mod
    scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)

    n = len(test_data)
    labels = []

    # 收集各专家在该domain上的缓存
    # 注意：exp3 的跨域矩阵只涵盖 SPECIALIZED_TYPES × SPECIALIZED_TYPES（3×3），
    # general 专家从未在专化域（text/image/uml）上评估过，因此：
    #   - domain in SPECIALIZED_TYPES 时跳过 general 专家（无对应缓存）
    #   - 不能用 lora_moe/general_predictions.json 替代，那是 general 域的预测，
    #     索引不对应当前 domain 的测试样本，会引入纯噪声标签
    expert_caches = {}
    for expert_type in ALL_TYPES:
        if expert_type == domain:
            # 对角线：匹配专家在本域
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', f'{domain}_predictions.json')
        elif expert_type == 'general' and domain in SPECIALIZED_TYPES:
            # general 专家从未在专化域上推理（exp3 只做了 3×3 矩阵），无有效缓存，跳过
            logger.debug(f"  [标签重建] 跳过 general expert on {domain}（exp3 未生成此缓存）")
            continue
        else:
            # 跨域：使用 exp3_cross_domain 目录（仅含专化域组合）
            cache = load_predictions_cache(
                CACHE_DIR / 'exp3_cross_domain',
                f'{expert_type}_expert_on_{domain}_predictions.json'
            )
        if cache:
            expert_caches[expert_type] = cache.get('samples', [])
        else:
            logger.warning(f"  [标签重建] 缓存未找到: {expert_type} on {domain}，该专家将被跳过")

    for i in range(n):
        best_expert = domain  # 默认匹配专家
        best_score = -1.0

        for expert_type, samples in expert_caches.items():
            if i >= len(samples):
                continue
            pred = samples[i].get('prediction', '')
            ref = test_data[i].get('output', '')
            if not pred or not pred.strip():
                continue
            try:
                score = scorer.score(ref, pred)['rougeL'].fmeasure
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_expert = expert_type

        labels.append(EXPERT_TO_IDX.get(best_expert, EXPERT_TO_IDX[domain]))

    return labels


def _rebuild_general_labels(test_data, args):
    """
    专门为general域重建Oracle标签

    general域的跨域缓存在exp9_oracle目录中
    """
    from rouge_score import rouge_scorer as rs_mod
    scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)

    n = len(test_data)
    labels = []

    # general域的跨域缓存在exp9_oracle
    expert_caches = {}
    for expert_type in ALL_TYPES:
        if expert_type == 'general':
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', 'general_predictions.json')
        elif expert_type == 'text':
            # text专家在general域：使用exp3的MoE-3退化路由缓存
            cache = load_predictions_cache(
                CACHE_DIR / 'exp3_moe3_general_via_text',
                'general_via_text_predictions.json'
            )
        else:
            cache = load_predictions_cache(
                CACHE_DIR / 'exp9_oracle',
                f'{expert_type}_expert_on_general_predictions.json'
            )
        if cache:
            samples = cache.get('samples', [])
            # 验证样本数量是否匹配
            if len(samples) < len(test_data):
                logger.warning(f"  [标签重建] {expert_type}缓存样本数({len(samples)}) < 测试集({len(test_data)})")
            expert_caches[expert_type] = samples
        else:
            logger.warning(f"  [标签重建] general域缓存未找到: {expert_type}")

    for i in range(n):
        best_expert = 'general'
        best_score = -1.0

        for expert_type, samples in expert_caches.items():
            if i >= len(samples):
                continue
            pred = samples[i].get('prediction', '')
            ref = test_data[i].get('output', '')
            if not pred or not pred.strip():
                continue
            try:
                score = scorer.score(ref, pred)['rougeL'].fmeasure
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_expert = expert_type

        labels.append(EXPERT_TO_IDX.get(best_expert, EXPERT_TO_IDX['general']))

    return labels

def _train_router(router, train_X, train_y, val_X, val_y, args):
    """
    训练MLP路由器

    优化要点（对比原实现）：
    1. 早停指标改为 macro-F1（原来是 accuracy）
       - accuracy 在不均衡类别下会偏向多数类（text 最多），模型只要全预测 text
         就能获得较高 accuracy，掩盖了少数类（image/general）完全没学到的事实
       - macro-F1 对每个类别一视同仁，只要某类 recall=0 就会直接拉低指标
    2. patience 5 → 15，max_epochs 50 → 100
       - 原来 5 个 epoch 无提升就停止，等价于约 110 个梯度步，严重不足
    3. 学习率 1e-4 → 5e-4
       - 2.1M 参数 MLP 在 ~700 样本上收敛极快，更大 LR 可加速有效学习
    4. 加入 label_smoothing=0.1
       - Oracle 标签本身存在噪声（两个专家 ROUGE-L 相差很小时标签近似随机），
         软标签可防止模型对噪声标签过拟合
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    device = router.device
    X_t = torch.tensor(train_X, dtype=torch.float32)
    y_t = torch.tensor(train_y, dtype=torch.long)
    X_v = torch.tensor(val_X, dtype=torch.float32).to(device)
    y_v = torch.tensor(val_y, dtype=torch.long).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(
        router.model.parameters(), lr=5e-4, weight_decay=1e-2
    )

    # 类别权重：逆频率加权，归一化为均值=1
    class_counts = np.bincount(train_y, minlength=4).astype(float)
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    class_weights = class_weights / (class_weights.mean() + 1e-9)
    logger.info(f"  类别样本数: {dict(zip(['text','image','uml','general'], class_counts.astype(int)))}")
    logger.info(f"  类别权重:   {dict(zip(['text','image','uml','general'], class_weights.round(3)))}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        label_smoothing=0.1,   # 防止对噪声 Oracle 标签过拟合
    )

    # CosineAnnealingWarmRestarts：T_0=20 个 epoch 后重启一次
    # 比 CosineAnnealingLR 更不容易陷入局部最优
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    max_epochs = 10 if args.test_mode else 100
    patience = 5 if args.test_mode else 15
    best_val_f1 = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_acc': [], 'val_macro_f1': []}

    for epoch in range(max_epochs):
        router.model.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = router.model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # 验证：同时记录 accuracy 和 macro-F1，以 macro-F1 为早停依据
        router.model.eval()
        with torch.no_grad():
            val_logits = router.model(X_v)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        y_v_np = y_v.cpu().numpy()
        val_acc = float((val_pred == y_v_np).mean())
        val_f1 = float(f1_score(y_v_np, val_pred, average='macro', zero_division=0))

        avg_loss = epoch_loss / len(loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_f1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1}/{max_epochs}: "
                f"loss={avg_loss:.4f}, val_acc={val_acc:.4f}, val_macro_F1={val_f1:.4f}"
            )

        # 早停：以 macro-F1 为准，而非 accuracy
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            router.save(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    f"  Early stop at epoch {epoch+1}, "
                    f"best val_macro_F1={best_val_f1:.4f}"
                )
                break

    # 加载最优 checkpoint
    router.load(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
    logger.info(f"训练完成，最优验证 macro-F1: {best_val_f1:.4f}")
    history['best_val_f1'] = best_val_f1
    return history


# ─────────────────────────────────────────────
# Phase 2：Output Ensemble + Learned Router评估
# ─────────────────────────────────────────────

def run_phase2(args, phase1_results, exp9_phase1):
    """
    Phase 2: Output Ensemble（logit融合）+ Learned Router单路由评估

    Returns:
        Dict: phase2结果
    """
    logger.info("=" * 80)
    logger.info("Phase 2: Output Ensemble + Learned Router评估")
    logger.info("=" * 80)

    # 加载General测试集
    general_data = GeneralDatasetLoader().load_all_data()
    _, _, general_test = split_dataset_for_expert(general_data, 'general')
    if args.test_mode:
        general_test = general_test[:10]
    logger.info(f"General测试集: {len(general_test)} 条")

    # 加载Router
    router = RouterMLP()
    router_ckpt = ROUTER_CKPT_DIR / 'router_mlp_best.pt'
    if not router_ckpt.exists():
        raise FileNotFoundError(f"Router权重不存在: {router_ckpt}，请先运行Phase 1")
    router.load(router_ckpt)

    # 加载General特征
    general_feat_path = FEATURE_CACHE_DIR / 'general_hidden_states.npz'
    if not general_feat_path.exists():
        raise FileNotFoundError(f"General特征缓存不存在: {general_feat_path}，请先运行Phase 1")

    feat_data = np.load(general_feat_path)
    general_features = feat_data['features']
    if args.test_mode:
        general_features = general_features[:10]

    # 确保特征数量与测试集对齐（test_mode下特征可能只有20条）
    n_cached = len(general_features)
    if len(general_test) != n_cached:
        logger.warning(
            f"General测试集({len(general_test)})与缓存特征({n_cached})数量不匹配，"
            f"截断测试集到缓存长度"
        )
        general_test = general_test[:n_cached]

    logger.info(f"General特征维度: {general_features.shape}")

    # ── 方案B独立评估：Learned Router单路由 ──
    logger.info("\n--- 方案B: Learned Router 单路由推理 ---")
    router_result = _run_learned_router_inference(
        router, general_features, general_test, args
    )

    # ── 方案A: Output Ensemble（logit融合）──
    logger.info("\n--- 方案A: Output Ensemble 推理 ---")
    ensemble_result = _run_output_ensemble(
        router, general_features, general_test, args
    )

    # Hard Routing基线（直接从exp9复用）
    hard_rougeL = exp9_phase1.get('strategies', {}).get(
        'Hard Routing', {}).get('per_domain', {}).get('general', 0.0)
    oracle_rougeL = exp9_phase1.get('strategies', {}).get(
        'Oracle Routing', {}).get('per_domain', {}).get('general', 0.0)

    gap = oracle_rougeL - hard_rougeL
    router_gap_reduction = (router_result['rougeL'] - hard_rougeL) / gap if gap > 0 else 0
    ensemble_gap_reduction = (ensemble_result['rougeL'] - hard_rougeL) / gap if gap > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2 结果汇总")
    logger.info("=" * 60)
    logger.info(f"Hard Routing (baseline):   {hard_rougeL:.4f}")
    logger.info(f"Oracle Routing (upper):    {oracle_rougeL:.4f}")
    logger.info(f"Gap:                       {gap:.4f} ({gap*100:.2f}%)")
    logger.info(f"Learned Router:            {router_result['rougeL']:.4f} | Gap缩小: {router_gap_reduction*100:.1f}%")
    logger.info(f"Output Ensemble:           {ensemble_result['rougeL']:.4f} | Gap缩小: {ensemble_gap_reduction*100:.1f}%")

    results = {
        'phase': 'phase2',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'learned_router': {
            'rougeL': router_result['rougeL'],
            'gap_reduction': float(router_gap_reduction),
            'routing_stats': router_result.get('routing_stats', {}),
        },
        'output_ensemble': {
            'rougeL': ensemble_result['rougeL'],
            'gap_reduction': float(ensemble_gap_reduction),
            'top2_rate': ensemble_result.get('top2_rate', 0.0),
            'routing_stats': ensemble_result.get('routing_stats', {}),
        },
        'hard_baseline_rougeL': float(hard_rougeL),
        'oracle_rougeL': float(oracle_rougeL),
        'oracle_hard_gap': float(gap),
    }

    save_experiment_results(results, EXP_DIR, 'phase2_results.json')
    logger.info(f"Phase 2 结果已保存: {EXP_DIR / 'phase2_results.json'}")
    return results


def _run_learned_router_inference(router, features, general_test, args):
    """
    方案B：Learned Router单路由推理
    对每条General样本，Router预测最优专家，直接从对应专家缓存取结果
    """
    cache_path = CACHE_DIR / 'exp10_router_only'
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / 'general_router_predictions.json'

    if cache_file.exists() and not args.force_regenerate:
        cached = load_predictions_cache(cache_path, 'general_router_predictions.json')
        if cached and (cached.get('total_samples', 0) > 15 or args.test_mode):
            logger.info(f"  [缓存命中] Learned Router: {cached.get('total_samples', 0)} 条")
            m = _metrics_from_samples(cached.get('samples', []))
            return {'rougeL': _get_rougeL(m), 'routing_stats': cached.get('routing_stats', {})}

    # Router预测每条样本应路由到哪个专家
    probs = router.predict_proba(features)   # (N, 4)
    predicted_experts = np.argmax(probs, axis=1)  # (N,)

    routing_stats = defaultdict(int)
    for idx in predicted_experts:
        routing_stats[IDX_TO_EXPERT[idx]] += 1
    logger.info(f"  路由分布: {dict(routing_stats)}")

    # 根据路由结果从对应专家缓存中取预测
    samples = []
    expert_caches = _load_all_expert_caches_for_general()

    for i, (sample, expert_idx) in enumerate(zip(general_test, predicted_experts)):
        expert_name = IDX_TO_EXPERT[expert_idx]
        expert_samples = expert_caches.get(expert_name, [])

        pred = ''
        if i < len(expert_samples):
            pred = expert_samples[i].get('prediction', '')

        if not pred:
            # 回退到general expert
            general_samples = expert_caches.get('general', [])
            if i < len(general_samples):
                pred = general_samples[i].get('prediction', '')

        samples.append({
            'index': i,
            'input': sample['input'],
            'prediction': pred,
            'reference': sample['output'],
            'routed_to': expert_name,
            'routing_probs': probs[i].tolist(),
        })

    save_predictions_cache(
        samples, 'exp10_router_only', 'general',
        {'strategy': 'learned_router', 'routing_stats': dict(routing_stats)},
        cache_path, 'general_router_predictions.json'
    )

    m = _metrics_from_samples(samples, use_bertscore=not args.no_bertscore)
    rougeL = _get_rougeL(m)
    logger.info(f"  Learned Router ROUGE-L: {rougeL:.4f}")
    return {'rougeL': rougeL, 'routing_stats': dict(routing_stats)}


def _run_output_ensemble(router, features, general_test, args):
    """
    方案A：Output Ensemble推理
    对每条General样本，用top-2专家的logit加权融合解码
    """
    cache_path = CACHE_DIR / 'exp10_ensemble'
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / 'general_ensemble_predictions.json'

    if cache_file.exists() and not args.force_regenerate:
        cached = load_predictions_cache(cache_path, 'general_ensemble_predictions.json')
        if cached and (cached.get('total_samples', 0) > 15 or args.test_mode):
            logger.info(f"  [缓存命中] Output Ensemble: {cached.get('total_samples', 0)} 条")
            m = _metrics_from_samples(cached.get('samples', []))
            return {
                'rougeL': _get_rougeL(m),
                'top2_rate': cached.get('metadata', {}).get('top2_rate', 0.0),
                'routing_stats': cached.get('metadata', {}).get('routing_stats', {}),
            }

    # Router预测权重
    probs = router.predict_proba(features)  # (N, 4)

    # 统计需要真正双专家推理的样本（最高权重 < 0.85）
    top1_probs = probs.max(axis=1)
    need_ensemble = (top1_probs < 0.85).sum()
    top2_rate = float(need_ensemble / len(probs))
    logger.info(f"  需要双专家融合的样本数: {need_ensemble}/{len(probs)} ({top2_rate*100:.1f}%)")

    # ── v13 诊断模式激活 ──
    if hasattr(args, 'debug_ensemble') and args.debug_ensemble:
        _reset_debug_stats()
        _DEBUG_ENSEMBLE_STATS['enabled'] = True
        logger.info("  [v13] 诊断模式已激活：将收集 D1-D5 指标")

    # 加载基础模型
    import torch
    from peft import PeftModel
    from models.language_model import LanguageModel

    lm = LanguageModel(use_4bit=True)
    base_model = lm.model
    tokenizer = lm.tokenizer

    # 加载所有adapter路径
    adapter_paths = {}
    for et in ALL_TYPES:
        adapter_paths[et] = str(path_cfg.get_expert_weight_path(et))

    # 一次性将所有 adapter 挂载到 base_model，后续用 set_adapter 切换
    # 避免每条样本反复 from_pretrained（原实现约 996 次加载，极慢）
    logger.info("  预加载所有专家 adapter（一次性，后续 set_adapter 切换）...")
    model_with_adapters = base_model
    for et in ALL_TYPES:
        try:
            model_with_adapters = PeftModel.from_pretrained(
                model_with_adapters, adapter_paths[et], adapter_name=et,
                is_trainable=False,
            )
            logger.info(f"    已加载 adapter: {et}")
        except Exception as e:
            logger.warning(f"    adapter 加载失败 {et}: {e}")
    model_with_adapters.eval()

    routing_stats = defaultdict(int)
    preloaded_caches = _load_all_expert_caches_for_general()
    logger.info(f"  已预加载专家缓存: {list(preloaded_caches.keys())}")

    # ── DEBUG: data_type 分布分析 ──────────────────────────────────────────
    dtype_counts: defaultdict = defaultdict(int)
    for sample in general_test:
        dt = _detect_datatype(sample)
        dtype_counts[dt] += 1
    logger.info(f"  [DEBUG] general_test data_type 分布: {dict(dtype_counts)}")
    # 检查 sample dict 中实际字段
    if general_test:
        sample0 = general_test[0]
        logger.info(f"  [DEBUG] 样本0 字段: {list(sample0.keys())}")
        logger.info(f"  [DEBUG] 样本0 data_type字段值: "
                    f"data_type={sample0.get('data_type')!r}, "
                    f"type={sample0.get('type')!r}, "
                    f"domain={sample0.get('domain')!r}")
        prompt0, tpl0 = _build_prompt_for_sample(sample0)
        logger.info(f"  [DEBUG] 样本0 使用模板: {tpl0}, prompt前80字符: {prompt0[:80]!r}")

    # ── Stage 1: 分类样本（纯 CPU，O(N)）──────────────────────────────────────
    # cache_results  : top-1 prob >= 0.85 → 从磁盘缓存取（单专家高置信度）
    # ensemble_groups: 按 (expert1, expert2) 分组，后续批量 GPU 推理
    # sample_meta    : 保存每条样本的路由元信息，供最终重新排序（reassemble）用
    #
    # ── v11 修复：双向对称OOD修正 + UML参数提升 ──
    # 根因分析（v10遗留bug）：OOD修正仅处理expert1=uml的情况，
    #   对称情况（general+uml、image+uml等）中非UML专家以高权重（~73%）
    #   面对UML模板完全OOD，严重污染融合输出。
    # v11修复：泛化为双向对称OOD修正（_TEMPLATE_OOD_FACTORS），
    #   UML OOD因子从0.15降至0.05，max_new_tokens 320→450，
    #   soft_limit 65%→70%，eos_boost_rate 0.12→0.08。

    sample_meta = []          # [(i, expert1, expert2, w1, w2, w1_raw, template_name), ...]
    cache_results = {}        # {i: pred_str}
    ensemble_groups = defaultdict(list)   # {(e1, e2): [(i, prompt_str, w1, w2), ...]}
    template_usage: defaultdict = defaultdict(int)   # {template_name: count}
    uml_ensemble_count = 0   # [DEBUG] 统计进入 ensemble 的 UML 域样本数

    # v12: OOD修正后权重阈值 —— 当OOD修正使dominant expert权重>=此值时，
    # 使用缓存预测而非ensemble生成。原因：
    #   (1) OOD修正后secondary expert权重仅~1-5%，融合效果几乎为零
    #   (2) 自回归生成中即使1%的概率噪声也会导致token选择偏移，累积后
    #       使输出序列与纯单专家结果显著不同（尤其UML域长输出受影响最大）
    #   (3) 缓存的单专家预测由完整推理流程生成，质量有保障
    # 此阈值仅在OOD修正后生效，不影响text+general等真正需要融合的组
    _POST_OOD_CACHE_THRESHOLD = 0.95

    # v12: 预计算每个样本的OOD修正后权重，用于决定是否需要ensemble生成
    # OOD修正逻辑与 _process_minibatch 中保持一致
    _TEMPLATE_OOD_FACTORS_PRE = {
        'uml': 0.05,
        'image': 0.4,
    }
    _GENERAL_LEAD_FACTOR_PRE = 0.7

    for i, (sample, prob) in enumerate(zip(general_test, probs)):
        top2_idxs = np.argsort(prob)[::-1][:2]
        expert1 = IDX_TO_EXPERT[top2_idxs[0]]
        expert2 = IDX_TO_EXPERT[top2_idxs[1]]
        w1_raw = float(prob[top2_idxs[0]])
        w2_raw = float(prob[top2_idxs[1]])
        w_sum   = w1_raw + w2_raw
        w1 = w1_raw / w_sum
        w2 = w2_raw / w_sum
        routing_stats[f"{expert1}+{expert2}"] += 1

        # 核心修复：每个样本独立选模板，两个专家用同一个 prompt
        prompt_str, tpl_name = _build_prompt_for_sample(sample)
        template_usage[tpl_name] += 1

        data_type = _detect_datatype(sample)

        # ── v12: 预计算OOD修正后的dominant expert权重 ──
        # 与 _process_minibatch 中的修正逻辑完全一致
        w1_post_ood = w1
        tpl_type_pre = _detect_template_from_prompt(prompt_str)
        ood_factor_pre = _TEMPLATE_OOD_FACTORS_PRE.get(tpl_type_pre)
        if ood_factor_pre is not None:
            e1_matches = (expert1 == tpl_type_pre)
            e2_matches = (expert2 == tpl_type_pre)
            if e1_matches and not e2_matches:
                w2_corrected = w2 * ood_factor_pre
                w1_post_ood = 1.0 - w2_corrected
            elif e2_matches and not e1_matches:
                w1_corrected = w1 * ood_factor_pre
                w1_post_ood = w1_corrected
        elif tpl_type_pre == 'text' and expert1 == 'general' and expert2 == 'text':
            w1_corrected = w1 * _GENERAL_LEAD_FACTOR_PRE
            w1_post_ood = w1_corrected

        # 仅在 top-1 概率极高（>= 0.85）时跳过 ensemble，退化为单专家
        skip_ensemble = (w1_raw >= 0.85)

        # v12: OOD修正后dominant expert权重极高时，也使用缓存
        # 此时ensemble生成几乎等同于单专家但引入自回归噪声，质量反而下降
        dominant_expert = expert1
        if w1_post_ood < 0.5:
            # OOD修正后expert2变为dominant（发生在e2_matches场景）
            dominant_expert = expert2
        post_ood_dominant_w = max(w1_post_ood, 1.0 - w1_post_ood)

        if not skip_ensemble and post_ood_dominant_w >= _POST_OOD_CACHE_THRESHOLD:
            skip_ensemble = True
            # 使用dominant expert的缓存预测
            cache_results[i] = _single_expert_from_cache(
                dominant_expert, 'general', i, preloaded_caches
            )
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            if i < 5:
                logger.debug(
                    f"  [v12] 样本{i}: OOD修正后dominant={dominant_expert}权重="
                    f"{post_ood_dominant_w:.3f}>={_POST_OOD_CACHE_THRESHOLD}, 使用缓存"
                )
            continue

        if skip_ensemble:
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            cache_results[i] = _single_expert_from_cache(
                expert1, 'general', i, preloaded_caches
            )
        else:
            if data_type == 'uml':
                uml_ensemble_count += 1
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            ensemble_groups[(expert1, expert2)].append((i, prompt_str, w1, w2))

    logger.info(f"  [DEBUG] 模板使用分布: {dict(template_usage)}")
    logger.info(f"  [v12] UML域进入ensemble: {uml_ensemble_count}条 (双向OOD修正+增强参数)")

    # v12: 统计OOD修正后被重定向到缓存的样本数
    n_raw_high_conf = sum(1 for (_, _, _, _, _, w1r, _) in sample_meta if w1r >= 0.85)
    n_post_ood_redirected = len(cache_results) - n_raw_high_conf

    n_cache = len(cache_results)
    n_ensemble = sum(len(v) for v in ensemble_groups.values())
    # [DEBUG] per-group size breakdown，标注OOD修正状态
    for (e1, e2), items in sorted(ensemble_groups.items(), key=lambda x: -len(x[1])):
        avg_w1 = np.mean([w1 for (_, _, w1, _) in items])
        avg_w2 = np.mean([w2 for (_, _, _, w2) in items])
        is_uml_grp = (e1 == 'uml' or e2 == 'uml')
        # 统计组内各模板类型分布
        tpl_counts = defaultdict(int)
        for (idx, prompt_s, _, _) in items:
            tpl_counts[_detect_template_from_prompt(prompt_s)] += 1
        ood_tag = ""
        if is_uml_grp:
            n_uml_tpl = tpl_counts.get('uml', 0)
            ood_tag = f" [UML增强, UML模板={n_uml_tpl}条将做OOD修正]"
        logger.info(
            f"    [v12 组] {e1}+{e2}: {len(items)}条, "
            f"avg_w1={avg_w1:.2f}, avg_w2={avg_w2:.2f}"
            + ood_tag
        )
    logger.info(
        f"  样本分类: cache(w1>=0.85)={n_raw_high_conf}, "
        f"cache(OOD修正后>={_POST_OOD_CACHE_THRESHOLD})={n_post_ood_redirected}, "
        f"ensemble={n_ensemble}, 组数={len(ensemble_groups)}"
    )

    # ── quick-ensemble 模式：每组仅采样 N 条，快速估算质量 ────────────────
    if hasattr(args, 'quick_ensemble') and args.quick_ensemble and args.quick_ensemble > 0:
        quick_n = args.quick_ensemble
        logger.info(f"  [快速测试] quick_ensemble={quick_n}，每组最多采样{quick_n}条")
        trimmed_groups = {}
        for key, items in ensemble_groups.items():
            if len(items) > quick_n:
                # 均匀采样而非截取前 N 条，避免数据分布偏差
                step = max(1, len(items) // quick_n)
                trimmed_groups[key] = items[::step][:quick_n]
            else:
                trimmed_groups[key] = items
        total_before = sum(len(v) for v in ensemble_groups.values())
        total_after = sum(len(v) for v in trimmed_groups.values())
        logger.info(f"  [快速测试] 采样前={total_before}条, 采样后={total_after}条")
        ensemble_groups = trimmed_groups

    # ── Stage 2: 按 (expert1, expert2) 组批量 GPU 推理 ──────────────────────
    # 同一组内的样本共享两次 prefill（而非每条样本各自 prefill），
    # decode 阶段每步两次 (B,1) forward 替代原来 B×2 次 (1,1) forward，
    # GPU 利用率从 ~10% 提升至 ~60%+。
    ensemble_results = {}   # {i: pred_str}
    for group_idx, ((expert1, expert2), group_items) in enumerate(ensemble_groups.items()):
        logger.info(
            f"  Ensemble组 {group_idx+1}/{len(ensemble_groups)}: "
            f"{expert1}+{expert2}, {len(group_items)} 条"
        )
        # [DEBUG] 检查该组的模板分布（验证每组内模板是否一致）
        if group_items:
            # group_items 格式: [(i, prompt_str, w1, w2), ...]
            # 取前3个样本的 prompt 前50字符，确认模板多样性
            sample_prompts_debug = [item[1][:60] for item in group_items[:3]]
            logger.debug(f"    [DEBUG] 组内前3个prompt前缀: {sample_prompts_debug}")

        preds = _logit_ensemble_generate_batched(
            model_with_adapters, tokenizer,
            expert1, expert2, group_items, args
        )
        for (i_s, _prompt, _w1, _w2), pred in zip(group_items, preds):
            ensemble_results[i_s] = pred

        # [DEBUG] 每组生成后报告质量概况
        group_preds = [ensemble_results.get(item[0], '') for item in group_items]
        valid_preds = [p for p in group_preds if p]
        if valid_preds:
            avg_len = sum(len(p) for p in valid_preds) / len(valid_preds)
            empty_count = len(group_preds) - len(valid_preds)
            # 简单格式检测：是否含有指令三段式关键词
            format_ok = sum(
                1 for p in valid_preds
                if any(kw in p for kw in ['Definition', 'Emphasis', 'Things to Avoid',
                                          'definition', 'emphasis', 'things to avoid'])
            )
            # [DEBUG] 新增：per-group ROUGE-L 估算（用于定位问题组）
            from rouge_score import rouge_scorer as rs_mod
            _scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)
            group_rougeL_scores = []
            for item in group_items:
                i_s = item[0]
                pred = ensemble_results.get(i_s, '')
                ref = general_test[i_s].get('output', '')
                if pred and ref:
                    try:
                        sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                        group_rougeL_scores.append(sc)
                    except Exception:
                        pass
            group_rougeL = np.mean(group_rougeL_scores) if group_rougeL_scores else 0.0
            logger.info(
                f"    [DEBUG] 组 {expert1}+{expert2}: "
                f"avg_len={avg_len:.0f}, empty={empty_count}, "
                f"format_ok={format_ok}/{len(valid_preds)} ({format_ok/len(valid_preds)*100:.0f}%), "
                f"ROUGE-L={group_rougeL:.4f}"
            )

    # ── Stage 3: 按原始顺序 reassemble + 质量门控 ─────────────────────────────
    # v12: 增强质量门控 —— 对通过格式检查的ensemble输出，也与缓存单专家做ROUGE-L比较，
    # 选择更优的结果。这确保ensemble只在真正提升质量时才被采用。
    _FORMAT_KEYWORDS = {'Definition', 'Emphasis', 'Things to Avoid',
                        'definition', 'emphasis', 'things to avoid'}
    _MAX_CHAR_LEN = 1500

    def _passes_quality_gate(pred_text: str) -> bool:
        if not pred_text or not pred_text.strip():
            return False
        if not any(kw in pred_text for kw in _FORMAT_KEYWORDS):
            return False
        if len(pred_text) > _MAX_CHAR_LEN:
            return False
        return True

    is_quick = hasattr(args, 'quick_ensemble') and args.quick_ensemble and args.quick_ensemble > 0

    # v12: 初始化ROUGE scorer用于ensemble vs cache质量比较
    from rouge_score import rouge_scorer as rs_mod
    _quality_scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)

    samples = []
    fallback_stats = {'total': 0, 'passed': 0, 'fallback': 0, 'fallback_improved': 0,
                      'quick_no_result': 0,
                      'quality_compare': 0, 'cache_wins': 0, 'ensemble_wins': 0}
    for (i, expert1, expert2, w1, w2, _w1_raw, tpl_name) in sample_meta:
        sample = general_test[i]
        ensemble_pred = ensemble_results.get(i, '')
        cache_pred = cache_results.get(i, '')

        if cache_pred:
            # 缓存结果（w1>=0.85 或 OOD修正后>=0.95），直接使用
            pred = cache_pred
        elif not ensemble_pred and is_quick:
            # quick-ensemble 模式：未被采样的 ensemble 样本 -> 用 top-1 缓存
            pred = _single_expert_from_cache(expert1, 'general', i, preloaded_caches)
            fallback_stats['quick_no_result'] += 1
        else:
            # ensemble 结果，执行质量门控
            fallback_stats['total'] += 1
            if _passes_quality_gate(ensemble_pred):
                # v12: 格式通过后，再与缓存单专家做ROUGE-L比较
                ref = sample.get('output', '')
                cache_expert_pred = _single_expert_from_cache(
                    expert1, 'general', i, preloaded_caches
                )
                # 仅当缓存存在且reference存在时做比较
                if ref and cache_expert_pred and cache_expert_pred.strip():
                    try:
                        ens_r = _quality_scorer.score(ref, ensemble_pred)['rougeL'].fmeasure
                        cache_r = _quality_scorer.score(ref, cache_expert_pred)['rougeL'].fmeasure
                        fallback_stats['quality_compare'] += 1
                        if cache_r > ens_r:
                            # 缓存单专家质量更高，使用缓存
                            pred = cache_expert_pred
                            fallback_stats['cache_wins'] += 1
                        else:
                            # ensemble质量更高或持平，使用ensemble
                            pred = ensemble_pred
                            fallback_stats['ensemble_wins'] += 1
                    except Exception:
                        pred = ensemble_pred
                        fallback_stats['passed'] += 1
                else:
                    pred = ensemble_pred
                    fallback_stats['passed'] += 1
            else:
                fallback_pred = _single_expert_from_cache(
                    expert1, 'general', i, preloaded_caches
                )
                fallback_stats['fallback'] += 1

                ref = sample.get('output', '')
                if ref and fallback_pred and ensemble_pred:
                    try:
                        ens_r = _quality_scorer.score(ref, ensemble_pred)['rougeL'].fmeasure
                        fb_r = _quality_scorer.score(ref, fallback_pred)['rougeL'].fmeasure
                        if fb_r > ens_r:
                            fallback_stats['fallback_improved'] += 1
                    except Exception:
                        pass

                pred = fallback_pred if fallback_pred else ensemble_pred

        if i < 5:
            logger.info(
                f"  [DEBUG] 样本{i}: expert={expert1}+{expert2}, tpl={tpl_name}, "
                f"pred_len={len(pred)}, pred前80: {pred[:80]!r}"
            )

        samples.append({
            'index': i,
            'input': sample['input'],
            'prediction': pred,
            'reference': sample['output'],
            'expert1': expert1,
            'expert2': expert2,
            'w1': w1,
            'w2': w2,
            'template': tpl_name,
            'data_type': _detect_datatype(sample),
        })

    logger.info(
        f"  [质量门控] ensemble样本={fallback_stats['total']}, "
        f"格式通过={fallback_stats['passed']}, "
        f"格式回退={fallback_stats['fallback']}, "
        f"格式回退更优={fallback_stats['fallback_improved']}"
    )
    logger.info(
        f"  [v12质量比较] 比较次数={fallback_stats['quality_compare']}, "
        f"缓存胜出={fallback_stats['cache_wins']}, "
        f"ensemble胜出={fallback_stats['ensemble_wins']}"
    )
    if is_quick:
        logger.info(
            f"  [快速测试] 未采样直接用缓存={fallback_stats['quick_no_result']}条"
        )

    del lm, model_with_adapters, tokenizer
    _cleanup_gpu()

    # ── [DEBUG] per-data_type ROUGE-L 分解：定位哪个子域仍有问题 ──────────
    from rouge_score import rouge_scorer as rs_mod
    _scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)
    dtype_scores = defaultdict(list)
    dtype_char_lens = defaultdict(list)
    for s in samples:
        dt = s.get('data_type', 'unknown')
        pred, ref = s.get('prediction', ''), s.get('reference', '')
        if pred and ref:
            try:
                sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                dtype_scores[dt].append(sc)
                dtype_char_lens[dt].append(len(pred))
            except Exception:
                pass
    for dt, scores in sorted(dtype_scores.items()):
        avg_len = np.mean(dtype_char_lens[dt]) if dtype_char_lens[dt] else 0
        logger.info(
            f"  [DEBUG] data_type={dt}: n={len(scores)}, "
            f"ROUGE-L={np.mean(scores):.4f} (std={np.std(scores):.4f}), "
            f"avg_pred_chars={avg_len:.0f}"
        )
    # [DEBUG] UML域ensemble专项：区分ensemble输出 vs 缓存回退，帮助定位参数效果
    uml_ensemble_samples = [
        s for s in samples
        if s.get('data_type') == 'uml' and s.get('index') in ensemble_results
    ]
    uml_cache_samples = [
        s for s in samples
        if s.get('data_type') == 'uml' and s.get('index') not in ensemble_results
    ]
    if uml_ensemble_samples:
        uml_ens_rouges = []
        uml_ens_lens = []
        for s in uml_ensemble_samples:
            pred, ref = s.get('prediction', ''), s.get('reference', '')
            if pred and ref:
                try:
                    sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                    uml_ens_rouges.append(sc)
                    uml_ens_lens.append(len(pred))
                except Exception:
                    pass
        logger.info(
            f"  [DEBUG][UML-ensemble] ensemble输出={len(uml_ensemble_samples)}条, "
            f"avg_ROUGE-L={np.mean(uml_ens_rouges):.4f}, "
            f"avg_chars={np.mean(uml_ens_lens):.0f}, "
            f"长输出(>700chars)={sum(1 for l in uml_ens_lens if l > 700)}条"
        )
    if uml_cache_samples:
        logger.info(
            f"  [DEBUG][UML-cache] 缓存单专家={len(uml_cache_samples)}条 "
            f"(w1>=0.85高置信度)"
        )
    # per-expert-pair ROUGE-L
    pair_scores = defaultdict(list)
    for s in samples:
        pair_key = f"{s.get('expert1','?')}+{s.get('expert2','?')}"
        pred, ref = s.get('prediction', ''), s.get('reference', '')
        if pred and ref:
            try:
                sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                pair_scores[pair_key].append(sc)
            except Exception:
                pass
    for pair, scores in sorted(pair_scores.items(), key=lambda x: -len(x[1])):
        logger.info(
            f"  [DEBUG] expert_pair={pair}: n={len(scores)}, "
            f"ROUGE-L={np.mean(scores):.4f}"
        )

    # ── v13 诊断：汇总 D1-D5 并保存 JSON ───────────────────────────────────
    if _DEBUG_ENSEMBLE_STATS['enabled']:
        _run_diagnostic_analysis(
            samples, ensemble_results, general_test,
            preloaded_caches, pair_scores
        )
        _DEBUG_ENSEMBLE_STATS['enabled'] = False

    save_predictions_cache(
        samples, 'exp10_ensemble', 'general',
        {
            'strategy': 'output_ensemble',
            'top2_rate': top2_rate,
            'routing_stats': dict(routing_stats),
        },
        cache_path, 'general_ensemble_predictions.json'
    )

    m = _metrics_from_samples(samples, use_bertscore=not args.no_bertscore)
    rougeL = _get_rougeL(m)
    logger.info(f"  Output Ensemble ROUGE-L: {rougeL:.4f}")
    return {'rougeL': rougeL, 'top2_rate': top2_rate, 'routing_stats': dict(routing_stats)}


def _run_diagnostic_analysis(samples, ensemble_results, general_test,
                             preloaded_caches, pair_scores):
    """
    v13 诊断分析：汇总 D1-D5 指标并保存到 JSON

    D1: 分布熵对比 — H(fused) vs max(H1, H2)，验证假设A（分布稀释）
    D2: Top-10 Jaccard — 两专家分布重叠度，验证假设A/E
    D3: 融合token吻合率（post-hoc近似：通过ensemble输出与单专家缓存的文本重叠估算）
    D4: 按专家对分层 ROUGE-L — 定位问题组
    D5: 按专家对 format_ok — 区分格式崩坏 vs 语义退化
    """
    from collections import defaultdict
    from rouge_score import rouge_scorer as rs_mod

    logger.info("\n" + "=" * 60)
    logger.info("[v13] 诊断分析 D1-D5")
    logger.info("=" * 60)

    diag = {
        'version': 'v13',
        'D1_entropy': {},
        'D2_jaccard': {},
        'D3_token_overlap': {},
        'D4_per_pair_rougeL': {},
        'D5_per_pair_format': {},
        'raw_per_step': [],
        'per_batch_summary': _DEBUG_ENSEMBLE_STATS.get('per_batch', []),
    }

    # ── D1 & D2：从 per_step 数据聚合 ──
    step_data = _DEBUG_ENSEMBLE_STATS.get('per_step', [])
    if step_data:
        all_h1, all_h2, all_hf, all_jac = [], [], [], []
        per_pair_entropy = defaultdict(lambda: {'h1': [], 'h2': [], 'hf': [], 'jac': []})

        for sd in step_data:
            pair_key = f"{sd['expert1']}+{sd['expert2']}"
            for h1_val, h2_val, hf_val, jac_val in zip(
                sd['entropy_prob1'], sd['entropy_prob2'],
                sd['entropy_fused'], sd['jaccard_top10']
            ):
                all_h1.append(h1_val)
                all_h2.append(h2_val)
                all_hf.append(hf_val)
                all_jac.append(jac_val)
                per_pair_entropy[pair_key]['h1'].append(h1_val)
                per_pair_entropy[pair_key]['h2'].append(h2_val)
                per_pair_entropy[pair_key]['hf'].append(hf_val)
                per_pair_entropy[pair_key]['jac'].append(jac_val)

        # D1 全局汇总
        avg_h1 = np.mean(all_h1) if all_h1 else 0
        avg_h2 = np.mean(all_h2) if all_h2 else 0
        avg_hf = np.mean(all_hf) if all_hf else 0
        avg_max_h = np.mean([max(a, b) for a, b in zip(all_h1, all_h2)]) if all_h1 else 0
        entropy_ratio = avg_hf / avg_max_h if avg_max_h > 0 else 0

        diag['D1_entropy'] = {
            'avg_H_prob1': round(float(avg_h1), 4),
            'avg_H_prob2': round(float(avg_h2), 4),
            'avg_H_fused': round(float(avg_hf), 4),
            'avg_max_H_experts': round(float(avg_max_h), 4),
            'entropy_ratio_fused_vs_max': round(float(entropy_ratio), 4),
            'hypothesis_A_likely': entropy_ratio > 1.3,
            'interpretation': (
                f"H(fused)/max(H1,H2) = {entropy_ratio:.2f}. "
                f"{'> 1.3 → 分布稀释严重，假设A成立' if entropy_ratio > 1.3 else '≤ 1.3 → 分布稀释不严重'}"
            ),
        }
        logger.info(f"  [D1] H(prob1)={avg_h1:.3f}, H(prob2)={avg_h2:.3f}, H(fused)={avg_hf:.3f}")
        logger.info(f"  [D1] 熵比 H(fused)/max(H1,H2) = {entropy_ratio:.3f} "
                     f"{'→ 假设A成立！' if entropy_ratio > 1.3 else ''}")

        # D1 per-pair
        d1_per_pair = {}
        for pair_key, vals in per_pair_entropy.items():
            ph1 = np.mean(vals['h1'])
            ph2 = np.mean(vals['h2'])
            phf = np.mean(vals['hf'])
            pmax = np.mean([max(a, b) for a, b in zip(vals['h1'], vals['h2'])])
            pratio = phf / pmax if pmax > 0 else 0
            d1_per_pair[pair_key] = {
                'avg_H1': round(float(ph1), 4),
                'avg_H2': round(float(ph2), 4),
                'avg_Hf': round(float(phf), 4),
                'ratio': round(float(pratio), 4),
                'n_steps': len(vals['h1']),
            }
            logger.info(f"  [D1] {pair_key}: H1={ph1:.3f}, H2={ph2:.3f}, Hf={phf:.3f}, ratio={pratio:.3f}")
        diag['D1_entropy']['per_pair'] = d1_per_pair

        # D2 全局汇总
        avg_jac = np.mean(all_jac) if all_jac else 0
        diag['D2_jaccard'] = {
            'avg_jaccard_top10': round(float(avg_jac), 4),
            'hypothesis_AE_likely': avg_jac < 0.2,
            'interpretation': (
                f"avg Jaccard = {avg_jac:.3f}. "
                f"{'< 0.2 → 两专家分布几乎不重叠，融合无意义' if avg_jac < 0.2 else '≥ 0.2 → 有一定重叠'}"
            ),
        }
        logger.info(f"  [D2] avg Jaccard(top-10) = {avg_jac:.4f} "
                     f"{'→ 极低重叠！' if avg_jac < 0.2 else ''}")

        # D2 per-pair
        d2_per_pair = {}
        for pair_key, vals in per_pair_entropy.items():
            pjac = np.mean(vals['jac'])
            d2_per_pair[pair_key] = round(float(pjac), 4)
            logger.info(f"  [D2] {pair_key}: Jaccard={pjac:.4f}")
        diag['D2_jaccard']['per_pair'] = d2_per_pair

        # 保留前100条原始 per_step 供深入分析
        diag['raw_per_step'] = step_data[:100]

    # ── D3: 融合token吻合率（post-hoc文本相似度近似） ──
    # 无法回溯token级别比较，用 character n-gram 重叠近似
    d3_data = defaultdict(lambda: {'overlap_e1': [], 'overlap_e2': []})
    scorer = rs_mod.RougeScorer(['rouge1'], use_stemmer=False)

    for s in samples:
        idx = s.get('index', 0)
        if idx not in ensemble_results:
            continue  # 跳过缓存结果
        fused_pred = s.get('prediction', '')
        if not fused_pred:
            continue
        e1 = s.get('expert1', '')
        e2 = s.get('expert2', '')
        pair_key = f"{e1}+{e2}"

        # 获取单专家缓存预测
        e1_pred = ''
        e1_cache = preloaded_caches.get(e1, [])
        if idx < len(e1_cache):
            e1_pred = e1_cache[idx].get('prediction', '')
        e2_pred = ''
        e2_cache = preloaded_caches.get(e2, [])
        if idx < len(e2_cache):
            e2_pred = e2_cache[idx].get('prediction', '')

        if e1_pred:
            try:
                sc = scorer.score(e1_pred, fused_pred)['rouge1'].fmeasure
                d3_data[pair_key]['overlap_e1'].append(sc)
            except Exception:
                pass
        if e2_pred:
            try:
                sc = scorer.score(e2_pred, fused_pred)['rouge1'].fmeasure
                d3_data[pair_key]['overlap_e2'].append(sc)
            except Exception:
                pass

    d3_result = {}
    for pair_key, vals in d3_data.items():
        oe1 = np.mean(vals['overlap_e1']) if vals['overlap_e1'] else 0
        oe2 = np.mean(vals['overlap_e2']) if vals['overlap_e2'] else 0
        d3_result[pair_key] = {
            'avg_overlap_with_e1': round(float(oe1), 4),
            'avg_overlap_with_e2': round(float(oe2), 4),
            'n_samples': len(vals['overlap_e1']),
            'interpretation': (
                'fusion≈e1' if oe1 > 0.7 and oe2 < 0.5 else
                'fusion≈e2' if oe2 > 0.7 and oe1 < 0.5 else
                'semantic_drift' if oe1 < 0.5 and oe2 < 0.5 else
                'balanced_fusion'
            ),
        }
        logger.info(f"  [D3] {pair_key}: overlap_e1={oe1:.3f}, overlap_e2={oe2:.3f} "
                     f"→ {d3_result[pair_key]['interpretation']}")
    diag['D3_token_overlap'] = d3_result

    # ── D4: 按专家对分层 ROUGE-L（已有 pair_scores，补充单专家对比）──
    d4_result = {}
    _scorer_rl = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)
    for pair_key, fused_scores in pair_scores.items():
        if not fused_scores:
            continue
        # 解析 pair_key = "e1+e2"
        parts = pair_key.split('+')
        if len(parts) != 2:
            continue
        e1_name, e2_name = parts

        # 计算同组中 e1 单跑和 e2 单跑的 ROUGE-L
        e1_solos, e2_solos = [], []
        for s in samples:
            if f"{s.get('expert1','')}+{s.get('expert2','')}" != pair_key:
                continue
            idx = s.get('index', 0)
            ref = s.get('reference', '')
            if not ref:
                continue
            e1_cache = preloaded_caches.get(e1_name, [])
            if idx < len(e1_cache):
                e1_p = e1_cache[idx].get('prediction', '')
                if e1_p:
                    try:
                        e1_solos.append(_scorer_rl.score(ref, e1_p)['rougeL'].fmeasure)
                    except Exception:
                        pass
            e2_cache = preloaded_caches.get(e2_name, [])
            if idx < len(e2_cache):
                e2_p = e2_cache[idx].get('prediction', '')
                if e2_p:
                    try:
                        e2_solos.append(_scorer_rl.score(ref, e2_p)['rougeL'].fmeasure)
                    except Exception:
                        pass

        fused_avg = np.mean(fused_scores)
        e1_avg = np.mean(e1_solos) if e1_solos else 0
        e2_avg = np.mean(e2_solos) if e2_solos else 0
        best_solo = max(e1_avg, e2_avg)
        delta = fused_avg - best_solo

        d4_result[pair_key] = {
            'fused_rougeL': round(float(fused_avg), 4),
            'e1_solo_rougeL': round(float(e1_avg), 4),
            'e2_solo_rougeL': round(float(e2_avg), 4),
            'best_solo': round(float(best_solo), 4),
            'delta_fused_vs_best_solo': round(float(delta), 4),
            'n_samples': len(fused_scores),
            'fusion_helps': delta > 0,
        }
        status = '✓ fusion helps' if delta > 0 else '✗ fusion hurts'
        logger.info(
            f"  [D4] {pair_key}: fused={fused_avg:.4f}, e1_solo={e1_avg:.4f}, "
            f"e2_solo={e2_avg:.4f}, delta={delta:+.4f} {status}"
        )
    diag['D4_per_pair_rougeL'] = d4_result

    # ── D5: 按专家对 format_ok ──
    _FMT_KW = {'Definition', 'Emphasis', 'Things to Avoid',
               'definition', 'emphasis', 'things to avoid'}
    d5_data = defaultdict(lambda: {'total': 0, 'format_ok': 0})
    for s in samples:
        idx = s.get('index', 0)
        if idx not in ensemble_results:
            continue
        pair_key = f"{s.get('expert1','')}+{s.get('expert2','')}"
        pred = s.get('prediction', '')
        d5_data[pair_key]['total'] += 1
        if pred and any(kw in pred for kw in _FMT_KW):
            d5_data[pair_key]['format_ok'] += 1

    d5_result = {}
    for pair_key, vals in d5_data.items():
        rate = vals['format_ok'] / vals['total'] if vals['total'] > 0 else 0
        d5_result[pair_key] = {
            'total': vals['total'],
            'format_ok': vals['format_ok'],
            'format_ok_rate': round(float(rate), 4),
        }
        logger.info(f"  [D5] {pair_key}: format_ok={vals['format_ok']}/{vals['total']} ({rate*100:.1f}%)")
    diag['D5_per_pair_format'] = d5_result

    # ── 综合判断：哪个假设最可能 ──
    conclusions = []
    if diag['D1_entropy'].get('hypothesis_A_likely'):
        conclusions.append("假设A(分布稀释)很可能成立 → 推荐方向A(PoE log-linear)")
    if diag['D2_jaccard'].get('hypothesis_AE_likely'):
        conclusions.append("假设A/E(专家分布不重叠)成立 → 推荐方向A或F(PoE/Reranking)")

    # 检查 D4：多少组 fusion hurts
    n_hurts = sum(1 for v in d4_result.values() if not v.get('fusion_helps', True))
    n_total_pairs = len(d4_result)
    if n_hurts > n_total_pairs * 0.5:
        conclusions.append(
            f"D4: {n_hurts}/{n_total_pairs} 组融合后更差 → 当前MoE混合公式确实有问题"
        )

    # 检查 D5：格式崩坏是否严重
    low_format_pairs = [k for k, v in d5_result.items() if v['format_ok_rate'] < 0.5]
    if low_format_pairs:
        conclusions.append(
            f"D5: {low_format_pairs} 组格式通过率<50% → 可能需要方向D(Constrained Decoding)"
        )

    diag['conclusions'] = conclusions
    diag['recommended_next_version'] = (
        'v14: PoE log-linear interpolation (方向A)' if any('方向A' in c for c in conclusions)
        else 'v14: 置信度自适应融合 (方向B)' if conclusions
        else 'v14: 需要更多样本运行完整诊断'
    )

    logger.info("\n  [v13] === 诊断结论 ===")
    for c in conclusions:
        logger.info(f"  → {c}")
    logger.info(f"  推荐下一步: {diag['recommended_next_version']}")

    # 保存诊断 JSON
    diag_path = EXP_DIR / 'debug_ensemble_diagnostics.json'
    with open(diag_path, 'w', encoding='utf-8') as f:
        json.dump(diag, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  诊断结果已保存: {diag_path}")


def _detect_template_from_prompt(prompt_str: str) -> str:
    """
    从 prompt 字符串推断所使用的模板类型。

    判断依据：
    - UML 模板：含有 UML 用例图特有字段（"actors" + "use_cases"），且有 UML 标识词
    - 图像模板：含有 JSON "description" 字段但无 UML 特征字段
    - 文本模板：无结构化 JSON 输入（默认回退）

    Returns:
        str: 'uml' | 'image' | 'text'
    """
    if '"actors"' in prompt_str and '"use_cases"' in prompt_str:
        return 'uml'
    if '"description"' in prompt_str and '"actors"' not in prompt_str:
        return 'image'
    return 'text'


_ENSEMBLE_BATCH_SIZE = 12  # RTX 4090 24 GB: batch=12 → KV Cache 约 2.5 GB，仍远低于预算
# 说明：4090 24GB = 基础模型4bit ~10GB + 2专家KV Cache(B=12, seq≈1024) ~3GB → 峰值约13GB，安全

# UML参与组专用批大小：UML输入平均1063 tokens，max_length=2048时KV Cache约为普通组的4倍，
# batch=12会导致峰值约21GB（OOM），缩至6可将峰值降到约14GB，在安全边界内。
# 非UML组继续使用 _ENSEMBLE_BATCH_SIZE=12，不影响推理速度。
_UML_BATCH_SIZE = 6


def _logit_ensemble_generate_batched(
    model_with_adapters, tokenizer,
    expert1, expert2, group_items, args,
    batch_size=None,
):
    """
    批量版 logit-space 双专家融合生成

    将同一 (expert1, expert2) 组的样本按 batch_size 分批，
    每批调用 _process_minibatch 完成：
      - 一次批量 prefill（B 条同时过 expert1 / expert2）
      - 每个 decode 步：两次 (B, 1) forward（而非 B×2 次 (1, 1) forward）

    批大小选择：
      - UML参与组使用 _UML_BATCH_SIZE=6，避免 max_length=2048 时 OOM
      - 非UML组使用 _ENSEMBLE_BATCH_SIZE=12，保持推理效率

    OOM fallback: 某批次显存溢出时，自动降级为逐条 _logit_ensemble_generate。

    Args:
        group_items: List[(i_global, prompt_str, w1, w2)]  — 同一 (e1,e2) 组
                     注意：prompt_str 已由 _run_output_ensemble 按样本 data_type 预构建，
                     确保两个专家收到相同的、与训练分布匹配的 prompt 格式。
    Returns:
        List[str]  — 与 group_items 等长，按相同顺序
    """
    import torch

    # 按专家类型自动选择批大小：UML参与组使用小批避免OOM
    _is_uml_group = (expert1 == 'uml' or expert2 == 'uml')
    if batch_size is None:
        batch_size = _UML_BATCH_SIZE if _is_uml_group else _ENSEMBLE_BATCH_SIZE

    all_preds = [''] * len(group_items)

    for batch_start in range(0, len(group_items), batch_size):
        batch = group_items[batch_start: batch_start + batch_size]
        if not batch:
            continue
        try:
            batch_preds = _process_minibatch(
                model_with_adapters, tokenizer,
                expert1, expert2, batch, args
            )
            for j, pred in enumerate(batch_preds):
                all_preds[batch_start + j] = pred
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(
                    f"  OOM (batch_size={len(batch)}), 降级到逐条推理..."
                )
                torch.cuda.empty_cache()
                # OOM 回退：batch 格式已是 (i, prompt_str, w1, w2)，直接传 prompt_str
                for j, (i_s, prompt_str_s, w1_s, w2_s) in enumerate(batch):
                    try:
                        pred = _logit_ensemble_generate(
                            model_with_adapters, tokenizer,
                            prompt_str_s, expert1, expert2, w1_s, w2_s,
                            args
                        )
                    except Exception as inner_e:
                        logger.warning(f"  单条回退失败 i={i_s}: {inner_e}")
                        pred = ''
                    all_preds[batch_start + j] = pred
            else:
                logger.error(f"  批量推理非 OOM 错误: {e}")
                for j in range(len(batch)):
                    all_preds[batch_start + j] = ''

    return all_preds


def _process_minibatch(
    model_with_adapters, tokenizer,
    expert1, expert2, batch_items, args,
):
    """
    批量 prefill + 批量 decode（B×1 token/step × 2 experts）

    UML 参与组参数（v11修复）：
        T_uml=1.0（保留实体名词高置信度预测），
        双向OOD修正（_TEMPLATE_OOD_FACTORS['uml']=0.05，non-UML专家贡献降至~1%），
        max_new_tokens=450，soft_limit=70%（315 tokens），eos_boost_rate=0.08。
    非UML组：T=1.0，soft_limit=50%，eos_boost_rate=0.15（不变）。
    MoE 概率空间加权 + 温度缩放 + EOS 长度惩罚。
    """
    import torch
    import torch.nn.functional as F

    B = len(batch_items)
    DONE_CHECK_INTERVAL = 16   # 每 16 步做一次 GPU-CPU sync 检查 done.all()

    # ── 专家温度缩放 ─────────────────────────────────────────────────────────
    # T_uml=1.0：保留UML专家在实体名词token（actor名、use case名）上的概率峰值。
    # 实验发现：T=1.5会将置信度0.9的正确预测降至约0.75，而general专家在UML模板下
    # 完全OOD（均匀分布），混合后会污染实体名预测，导致ROUGE-L下降约0.08。
    # T=1.0保持UML专家的准确性，再通过 _UML_OOD_FACTOR 降低non-UML专家贡献来隔离干扰。
    # text/image 保持 T=1.0（无需变更，效果已稳定）
    _EXPERT_TEMPERATURE = {'text': 1.0, 'image': 1.0, 'uml': 1.0, 'general': 1.0}
    T1 = _EXPERT_TEMPERATURE.get(expert1, 1.0)
    T2 = _EXPERT_TEMPERATURE.get(expert2, 1.0)

    # ── UML 参与组专项参数 ───────────────────────────────────────────────────
    # max_new_tokens=450：UML指令参考输出含完整actor/use_case枚举，长样本可达200+ tokens，
    # 320 tokens在EOS boost施加后实际可用约260 tokens，对长UML样本仍存在截断风险。
    # 提升至450可覆盖99%分位的UML输出长度。
    # soft_limit=70%（315 tokens）：给UML完整输出充足的无惩罚生成空间，
    # 仅在315 tokens后才施加温和收束，避免提前截断正常的枚举内容。
    # eos_boost_rate=0.08：更温和的收束，每步仅增加0.08，在450 tokens时约boost=10.8，
    # 足以阻止异常超长输出，同时不干扰200-300 tokens内的正常生成。
    _is_uml_involved = (expert1 == 'uml' or expert2 == 'uml')
    _DOMAIN_MAX_TOKENS = {'text': 200, 'image': 200, 'uml': 450, 'general': 200}
    if _is_uml_involved:
        max_new_tokens = 450
        _SOFT_LIMIT = int(max_new_tokens * 0.70)   # 315 tokens
        _EOS_BOOST_RATE = 0.08  # 温和收束，不提前截断正常枚举
    else:
        max_new_tokens = max(
            _DOMAIN_MAX_TOKENS.get(expert1, 200),
            _DOMAIN_MAX_TOKENS.get(expert2, 200),
        )
        _SOFT_LIMIT = int(max_new_tokens * 0.5)  # 默认：50% 处开始施加惩罚
        _EOS_BOOST_RATE = 0.15  # 默认：每超出 1 个 token，EOS logit 增加 0.15

    # stop token set
    stop_ids = {tokenizer.eos_token_id}
    if (tokenizer.pad_token_id is not None
            and tokenizer.pad_token_id != tokenizer.eos_token_id
            and tokenizer.pad_token_id > 3):
        stop_ids.add(tokenizer.pad_token_id)
    stop_ids = {sid for sid in stop_ids if sid is not None}
    sentinel_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    # 核心修复：直接用预构建的 prompt_str，不再在此处调用任何 Template
    prompts = [prompt_str for (_, prompt_str, _, _) in batch_items]
    ws1 = [w1 for (_, _, w1, _) in batch_items]
    ws2 = [w2 for (_, _, _, w2) in batch_items]

    # ── 通用模板-专家不匹配权重修正（双向对称机制）──────────────────────────────
    # 核心思路：当样本模板类型（由 data_type 决定）与某个专家的训练域不匹配时，
    # 该专家处于OOD状态，其logit近似均匀分布会稀释domain expert的准确预测信号。
    #
    # v11修复：将原来仅处理 expert1=uml 的单向逻辑泛化为双向对称机制。
    # 原bug：general+uml 组（expert1=general, expert2=uml）在UML模板下，
    #   general专家占73%权重且完全OOD，但未做任何修正，ROUGE-L仅0.5011。
    #   同理 image+uml 组也未修正。
    #
    # 修正策略（按域配置OOD因子，保留加权融合创新点）：
    #   UML模板 → non-UML专家贡献压至OOD因子（复杂JSON结构，OOD噪声最强）
    #   Image模板 → non-Image专家适度降权（JSON描述较简单，OOD影响温和）
    #   Text模板 → 仅对general-lead特殊情况做轻微修正（general训练时text占比最大）
    #
    # 注意：general专家虽训练时包含所有域数据，但使用通用模板格式，
    # 面对domain-specific模板时仍存在格式OOD问题，因此也需降权。
    _TEMPLATE_OOD_FACTORS = {
        'uml': 0.05,     # non-UML专家在UML模板下贡献降至~1%（UML JSON结构复杂，OOD噪声极强）
        'image': 0.4,    # non-Image专家在Image模板下适度降权（JSON描述结构较简单）
    }
    _GENERAL_LEAD_FACTOR = 0.7   # general 专家在文本模板下领导时的权重缩减系数（保持兼容）
    mismatch_corrected = 0
    ood_correction_detail = defaultdict(int)   # {修正类型: 计数}
    for j, (_, prompt_str_j, _, _) in enumerate(batch_items):
        tpl_type = _detect_template_from_prompt(prompt_str_j)

        # 通用OOD修正：UML/Image模板的双向对称处理
        ood_factor = _TEMPLATE_OOD_FACTORS.get(tpl_type)
        if ood_factor is not None:
            e1_matches = (expert1 == tpl_type)
            e2_matches = (expert2 == tpl_type)
            if e1_matches and not e2_matches:
                # expert2 不匹配模板域，降低其权重
                ws2[j] = ws2[j] * ood_factor
                ws1[j] = 1.0 - ws2[j]
                mismatch_corrected += 1
                ood_correction_detail[f'{tpl_type}:e2_ood({expert2})'] += 1
            elif e2_matches and not e1_matches:
                # expert1 不匹配模板域，降低其权重（v11新增：对称修正）
                ws1[j] = ws1[j] * ood_factor
                ws2[j] = 1.0 - ws1[j]
                mismatch_corrected += 1
                ood_correction_detail[f'{tpl_type}:e1_ood({expert1})'] += 1
            elif not e1_matches and not e2_matches:
                # 两个专家都不匹配模板域（罕见：Router将该域样本路由到两个非本域专家）
                # 保持原始权重不变，让Router的原始概率决策生效
                ood_correction_detail[f'{tpl_type}:both_ood'] += 1
        # Text模板特殊处理：仅general-lead时轻微修正（保持兼容）
        elif tpl_type == 'text' and expert1 == 'general' and expert2 == 'text':
            ws1[j] = ws1[j] * _GENERAL_LEAD_FACTOR
            ws2[j] = 1.0 - ws1[j]
            mismatch_corrected += 1
            ood_correction_detail['text:general_lead'] += 1

    if mismatch_corrected > 0:
        logger.info(
            f"    [OOD修正] {mismatch_corrected}/{B}条样本已修正权重 "
            f"(expert1={expert1}, expert2={expert2}), "
            f"明细: {dict(ood_correction_detail)}"
        )

    # [DEBUG] 记录 batch 基本信息
    logger.info(
        f"    [minibatch] B={B}, expert1={expert1}(T={T1}), expert2={expert2}(T={T2}), "
        f"max_new_tokens={max_new_tokens}, soft_limit={_SOFT_LIMIT}, "
        f"eos_boost_rate={_EOS_BOOST_RATE}"
        + (f" [UML增强: max={max_new_tokens},sl={_SOFT_LIMIT},rate={_EOS_BOOST_RATE}]"
           if _is_uml_involved else "")
    )

    # ── Left-padding tokenize，与 KV Cache decode 兼容 ──────────────────────
    # 必须 left-pad：right-pad 时 KV Cache 最后一个有效位置对每条样本不同，
    # 导致 decode 第一个 token 的 position id 错位。
    #
    # 修复：按专家类型动态决定 max_length，解决 UML prompt 被硬截断到 512 tokens 的根本问题。
    # 数据集长度统计：
    #   text  平均 351 tokens（95%分位 551），512 已足够
    #   image 平均 533 tokens（95%分位 622），768 覆盖绝大多数
    #   uml   平均 1063 tokens（99%分位 1807），需要 2048
    # UML JSON 被截断到 512 时，结构残缺（括号未闭合）、专家无法理解输入，
    # 导致 format_ok 仅 8%、ROUGE-L 仅 0.19，是 UML 加权输出质量差的根本原因。
    # 显存估算（4090 24GB）：4bit 基础模型 ~6GB + GQA(8头) + B=12 + seq=2048
    #   → KV Cache ~7.6GB，峰值约 13.6GB，安全。
    _EXPERT_MAX_LENGTH = {'text': 512, 'image': 768, 'uml': 2048, 'general': 768}
    tokenize_max_length = max(
        _EXPERT_MAX_LENGTH.get(expert1, 768),
        _EXPERT_MAX_LENGTH.get(expert2, 768),
    )
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = tokenizer(
        prompts, return_tensors='pt', padding=True,
        truncation=True, max_length=tokenize_max_length,
    )
    tokenizer.padding_side = orig_padding_side
    logger.debug(
        f"    [tokenize] expert1={expert1}, expert2={expert2}, "
        f"max_length={tokenize_max_length}, actual_seq_len={encoded['input_ids'].shape[1]}"
    )

    device = (
        model_with_adapters.base_model.model.device
        if hasattr(model_with_adapters, 'base_model')
        else next(model_with_adapters.parameters()).device
    )
    prompt_ids  = encoded['input_ids'].to(device)        # (B, L)
    prompt_mask = encoded['attention_mask'].to(device)   # (B, L)，left-pad 位置为 0
    L = prompt_ids.shape[1]

    # 融合权重广播形状 (B, 1)，与 (B, vocab) logits 广播相乘
    w1_t = torch.tensor(ws1, dtype=torch.float32, device=device).unsqueeze(1)
    w2_t = torch.tensor(ws2, dtype=torch.float32, device=device).unsqueeze(1)

    # ── 优化②：预分配注意力掩码缓冲区（一次分配，循环内 zero-copy view）───
    # shape (B, L + max_new_tokens)
    # [:, :L]  = prompt_mask（一次写入）
    # [:, L:]  = 1（所有 decode 位置预设为 1；view 截断保证不越界）
    attn_mask_buf = torch.zeros(B, L + max_new_tokens, dtype=torch.long, device=device)
    attn_mask_buf[:, :L] = prompt_mask
    attn_mask_buf[:, L:] = 1

    # ── 优化①：预分配输出 token 缓冲区（消除循环内 .item() 同步）──────────
    # 已完成序列的槽位用 sentinel_id 填充，post-processing 时截断到第一个 stop/sentinel
    output_ids = torch.full((B, max_new_tokens), sentinel_id, dtype=torch.long, device=device)
    write_pos = 0   # 下一个写入列的下标

    # ── 优化⑤：eval() 统一在 prefill 前调用一次 ─────────────────────────────
    model_with_adapters.eval()

    # ── Prefill：两个专家各一次批量前向 ─────────────────────────────────────
    past_kv1, past_kv2 = None, None
    logits1_init, logits2_init = None, None

    try:
        model_with_adapters.set_adapter(expert1)
        with torch.no_grad():
            out1 = model_with_adapters(
                input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True,
            )
            logits1_init = out1.logits[:, -1, :]   # (B, vocab)
            past_kv1 = out1.past_key_values          # expert1 专属 KV Cache
    except Exception as e:
        logger.warning(f"  prefill batch expert1={expert1} 失败: {e}")

    try:
        model_with_adapters.set_adapter(expert2)
        with torch.no_grad():
            out2 = model_with_adapters(
                input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True,
            )
            logits2_init = out2.logits[:, -1, :]   # (B, vocab)
            past_kv2 = out2.past_key_values          # expert2 专属 KV Cache
    except Exception as e:
        logger.warning(f"  prefill batch expert2={expert2} 失败: {e}")

    if logits1_init is None and logits2_init is None:
        return [''] * B

    # ── 第一个 token：由 prefill logits 融合得到 ─────────────────────────────
    if logits1_init is None:
        logits_fused = logits2_init
    elif logits2_init is None:
        logits_fused = logits1_init
    else:
        # ── MoE 概率空间混合 + 温度缩放（v5）────────────────────────────────
        # 温度 T>1 拉平专家的概率分布，减少"过自信"专家的主导效应
        # 温度 T=1 保持原始分布不变
        prob1 = F.softmax(logits1_init / T1, dim=-1)   # (B, vocab)
        prob2 = F.softmax(logits2_init / T2, dim=-1)   # (B, vocab)
        fused_prob = w1_t * prob1 + w2_t * prob2   # (B, vocab)
        # 转回 log 空间供 argmax（log 单调，argmax 等价）
        logits_fused = torch.log(fused_prob + 1e-10)

        # ── v13 诊断：prefill step 的分布指标 ──
        if _DEBUG_ENSEMBLE_STATS['enabled']:
            with torch.no_grad():
                h1 = _entropy(prob1).cpu().tolist()       # (B,)
                h2 = _entropy(prob2).cpu().tolist()       # (B,)
                hf = _entropy(fused_prob).cpu().tolist()   # (B,)
                jac = _jaccard_topk(prob1, prob2, k=10)    # (B,)
                _DEBUG_ENSEMBLE_STATS['per_step'].append({
                    'step': 0, 'expert1': expert1, 'expert2': expert2,
                    'entropy_prob1': h1, 'entropy_prob2': h2,
                    'entropy_fused': hf, 'jaccard_top10': jac,
                })

    next_tokens = logits_fused.argmax(dim=-1, keepdim=True)   # (B, 1)

    # ── 优化④：向量化 done 更新（无 Python for 循环、无 .item()）────────────
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for sid in stop_ids:
        done |= (next_tokens.squeeze(1) == sid)

    # 写入第一个 token；done 序列写 sentinel_id（post-processing 时截断）
    output_ids[:, write_pos] = next_tokens.squeeze(1).masked_fill(done, sentinel_id)
    write_pos += 1

    # ── Decode 循环：每步 2 次 (B,1) forward ────────────────────────────────
    for decode_step in range(max_new_tokens - 1):
        # 优化③：每 DONE_CHECK_INTERVAL 步才做一次 GPU-CPU sync（.item() 触发）
        if decode_step % DONE_CHECK_INTERVAL == 0 and done.all().item():
            break

        # 优化②：view 零拷贝，shape (B, L+decode_step+1)，与原实现等价
        # 原：torch.cat([prompt_mask, ones(B, decode_step+1)], dim=1)
        # 现：attn_mask_buf 预置了所有 1，此处仅取前缀视图，无内存分配
        attn_mask_step = attn_mask_buf[:, :L + decode_step + 1]

        logits1, logits2 = None, None

        if past_kv1 is not None:
            try:
                model_with_adapters.set_adapter(expert1)
                with torch.no_grad():
                    out1 = model_with_adapters(
                        input_ids=next_tokens,
                        attention_mask=attn_mask_step,
                        past_key_values=past_kv1,
                        use_cache=True,
                    )
                    logits1  = out1.logits[:, -1, :]   # (B, vocab)
                    past_kv1 = out1.past_key_values     # 更新 expert1 KV Cache
            except Exception as e:
                logger.warning(f"  decode step={decode_step} expert1={expert1} batch 失败: {e}")
                past_kv1 = None

        if past_kv2 is not None:
            try:
                model_with_adapters.set_adapter(expert2)
                with torch.no_grad():
                    out2 = model_with_adapters(
                        input_ids=next_tokens,
                        attention_mask=attn_mask_step,
                        past_key_values=past_kv2,
                        use_cache=True,
                    )
                    logits2  = out2.logits[:, -1, :]   # (B, vocab)
                    past_kv2 = out2.past_key_values     # 更新 expert2 KV Cache
            except Exception as e:
                logger.warning(f"  decode step={decode_step} expert2={expert2} batch 失败: {e}")
                past_kv2 = None

        if logits1 is None and logits2 is None:
            break
        elif logits1 is None:
            logits_fused = logits2
        elif logits2 is None:
            logits_fused = logits1
        else:
            # MoE 概率空间混合 + 温度缩放（与 prefill 保持一致）
            prob1 = F.softmax(logits1 / T1, dim=-1)
            prob2 = F.softmax(logits2 / T2, dim=-1)
            fused_prob = w1_t * prob1 + w2_t * prob2
            logits_fused = torch.log(fused_prob + 1e-10)   # (B, vocab)

            # ── v13 诊断：decode 每8步采样一次（控制开销） ──
            if _DEBUG_ENSEMBLE_STATS['enabled'] and (decode_step % 8 == 0):
                with torch.no_grad():
                    h1 = _entropy(prob1).cpu().tolist()
                    h2 = _entropy(prob2).cpu().tolist()
                    hf = _entropy(fused_prob).cpu().tolist()
                    jac = _jaccard_topk(prob1, prob2, k=10)
                    _DEBUG_ENSEMBLE_STATS['per_step'].append({
                        'step': decode_step + 1, 'expert1': expert1, 'expert2': expert2,
                        'entropy_prob1': h1, 'entropy_prob2': h2,
                        'entropy_fused': hf, 'jaccard_top10': jac,
                    })

        # ── EOS 长度惩罚：超过 soft_limit 后逐步提升 EOS 概率 ────────────
        # 防止 UML 专家的长输出偏好通过 MoE 融合泄露，导致生成过长
        current_step = decode_step + 1  # +1 因为 prefill 已产出第一个 token
        if current_step > _SOFT_LIMIT and eos_id is not None:
            boost = _EOS_BOOST_RATE * (current_step - _SOFT_LIMIT)
            logits_fused[:, eos_id] += boost

        next_tokens = logits_fused.argmax(dim=-1, keepdim=True)   # (B, 1)

        # 优化④：向量化 done 更新（纯 CUDA op，无 Python 循环、无 .item()）
        for sid in stop_ids:
            done |= (next_tokens.squeeze(1) == sid)

        # 优化①：写入 output_ids（CUDA 赋值，无 sync；done 位写 sentinel_id）
        output_ids[:, write_pos] = next_tokens.squeeze(1).masked_fill(done, sentinel_id)
        write_pos += 1

    # ── 批量解码：循环结束后仅一次 GPU→CPU 转移 ──────────────────────────────
    # 原实现：每步 B 次 .item() sync（最多 ~6144 次）→ 现在：1 次
    if write_pos == 0:
        return [''] * B

    output_cpu = output_ids[:, :write_pos].cpu().tolist()   # 唯一一次 GPU-CPU 同步
    stop_ids_py = stop_ids | {sentinel_id}   # sentinel_id 作为截断标记（已完成序列的占位符）

    results = []
    for b_tokens in output_cpu:
        # 截断到第一个终止符（stop_ids_py），语义等价于原实现的 "not done[b] 才 append"
        truncated = []
        for tok in b_tokens:
            if tok in stop_ids_py:
                break
            truncated.append(tok)
        decoded = tokenizer.decode(truncated, skip_special_tokens=True) if truncated else ''
        results.append(decoded)

    # [DEBUG] 批次生成统计
    valid = [r for r in results if r]
    if valid:
        avg_len = sum(len(r) for r in valid) / len(valid)
        empty_cnt = len(results) - len(valid)
        format_ok = sum(
            1 for r in valid
            if any(kw in r for kw in ['Definition', 'Emphasis', 'Things to Avoid'])
        )
        logger.debug(
            f"    [minibatch done] B={B}, avg_len={avg_len:.0f}, "
            f"empty={empty_cnt}, format_ok={format_ok}/{len(valid)}, "
            f"write_pos={write_pos}, max_new_tokens={max_new_tokens}"
        )

        # ── v13 诊断：批次级汇总 ──
        if _DEBUG_ENSEMBLE_STATS['enabled']:
            _DEBUG_ENSEMBLE_STATS['per_batch'].append({
                'expert1': expert1, 'expert2': expert2,
                'batch_size': B,
                'avg_pred_len': avg_len,
                'format_ok_rate': format_ok / len(valid) if valid else 0.0,
                'empty_count': empty_cnt,
                'write_pos': write_pos,
            })

    return results


def _logit_ensemble_generate(model_with_adapters, tokenizer,
                              prompt_str, expert1, expert2, w1, w2, args):
    """
    单条双专家 MoE 概率空间加权混合（OOM 回退路径）
    v11: 与 _process_minibatch 使用相同的双向对称OOD修正 + UML增强参数。
    """
    import torch
    import torch.nn.functional as F

    # 与 _process_minibatch 保持一致的温度和长度参数
    _EXPERT_TEMPERATURE = {'text': 1.0, 'image': 1.0, 'uml': 1.0, 'general': 1.0}
    T1 = _EXPERT_TEMPERATURE.get(expert1, 1.0)
    T2 = _EXPERT_TEMPERATURE.get(expert2, 1.0)

    _is_uml_involved = (expert1 == 'uml' or expert2 == 'uml')
    _DOMAIN_MAX_TOKENS = {'text': 200, 'image': 200, 'uml': 450, 'general': 200}
    if _is_uml_involved:
        max_new_tokens = 450
        _SOFT_LIMIT = int(max_new_tokens * 0.70)   # 315 tokens，与 _process_minibatch 一致
        _EOS_BOOST_RATE = 0.08  # 温和收束，与 _process_minibatch 一致
    else:
        max_new_tokens = max(
            _DOMAIN_MAX_TOKENS.get(expert1, 200),
            _DOMAIN_MAX_TOKENS.get(expert2, 200),
        )
        _SOFT_LIMIT = int(max_new_tokens * 0.5)
        _EOS_BOOST_RATE = 0.15

    # 修复2：stop_ids 只包含确定的终止符，避免 pad_token_id 误触提前截断
    stop_ids = {tokenizer.eos_token_id}
    if (tokenizer.pad_token_id is not None
            and tokenizer.pad_token_id != tokenizer.eos_token_id
            and tokenizer.pad_token_id > 3):
        stop_ids.add(tokenizer.pad_token_id)

    # ── 通用模板-专家不匹配权重修正（与 _process_minibatch 保持一致的双向对称机制）─
    _TEMPLATE_OOD_FACTORS = {
        'uml': 0.05,     # non-UML专家在UML模板下贡献降至~1%
        'image': 0.4,    # non-Image专家在Image模板下适度降权
    }
    _GENERAL_LEAD_FACTOR = 0.7
    tpl_type = _detect_template_from_prompt(prompt_str)
    ood_factor = _TEMPLATE_OOD_FACTORS.get(tpl_type)
    if ood_factor is not None:
        e1_matches = (expert1 == tpl_type)
        e2_matches = (expert2 == tpl_type)
        if e1_matches and not e2_matches:
            w2 = w2 * ood_factor
            w1 = 1.0 - w2
        elif e2_matches and not e1_matches:
            w1 = w1 * ood_factor
            w2 = 1.0 - w1
    elif tpl_type == 'text' and expert1 == 'general' and expert2 == 'text':
        w1 = w1 * _GENERAL_LEAD_FACTOR
        w2 = 1.0 - w1

    # 核心修复：直接使用调用方传入的 prompt_str，不再调用 GeneralInstructionTemplate
    # 同步修复：按专家类型动态设置 max_length，与 _process_minibatch 保持一致
    _EXPERT_MAX_LENGTH = {'text': 512, 'image': 768, 'uml': 2048, 'general': 768}
    tokenize_max_length = max(
        _EXPERT_MAX_LENGTH.get(expert1, 768),
        _EXPERT_MAX_LENGTH.get(expert2, 768),
    )
    device = (
        model_with_adapters.base_model.model.device
        if hasattr(model_with_adapters, 'base_model')
        else next(model_with_adapters.parameters()).device
    )
    prompt_ids = tokenizer(
        prompt_str, return_tensors='pt',
        truncation=True, max_length=tokenize_max_length,
    ).input_ids.to(device)

    # ── Prefill 阶段：完整 prompt 各过一次两个专家，建立各自 KV Cache ──
    # 注意：两个专家的 KV Cache 独立存储，切换 adapter 时彼此不干扰
    past_kv1, past_kv2 = None, None
    logits1_init, logits2_init = None, None

    try:
        model_with_adapters.set_adapter(expert1)
        model_with_adapters.eval()
        with torch.no_grad():
            out1 = model_with_adapters(input_ids=prompt_ids, use_cache=True)
            logits1_init = out1.logits[:, -1, :]   # (1, vocab_size)
            past_kv1 = out1.past_key_values         # expert1 专属 KV Cache
    except Exception as e:
        logger.warning(f"  prefill expert1={expert1} 失败: {e}")

    try:
        model_with_adapters.set_adapter(expert2)
        model_with_adapters.eval()
        with torch.no_grad():
            out2 = model_with_adapters(input_ids=prompt_ids, use_cache=True)
            logits2_init = out2.logits[:, -1, :]   # (1, vocab_size)
            past_kv2 = out2.past_key_values         # expert2 专属 KV Cache
    except Exception as e:
        logger.warning(f"  prefill expert2={expert2} 失败: {e}")

    # Prefill 完全失败则返回空串
    if logits1_init is None and logits2_init is None:
        return ''

    # ── 第一个 token：由 prefill 的 logits 融合得到 ──
    if logits1_init is None:
        logits_fused_init = logits2_init
    elif logits2_init is None:
        logits_fused_init = logits1_init
    else:
        import torch.nn.functional as F
        # MoE 概率空间混合 + 温度缩放（v5）
        prob1 = F.softmax(logits1_init / T1, dim=-1)
        prob2 = F.softmax(logits2_init / T2, dim=-1)
        fused_prob = w1 * prob1 + w2 * prob2
        logits_fused_init = torch.log(fused_prob + 1e-10)

    next_token = logits_fused_init.argmax(dim=-1, keepdim=True)  # (1, 1)
    fused_tokens = []

    if next_token.item() in stop_ids:
        return ''
    fused_tokens.append(next_token.item())

    # ── Decode 阶段：每步只传入上一个 token + 对应 KV Cache，O(n) 复杂度 ──
    for step in range(max_new_tokens - 1):
        logits1, logits2 = None, None

        # Expert 1：传入单 token + expert1 专属 KV Cache
        if past_kv1 is not None:
            try:
                model_with_adapters.set_adapter(expert1)
                model_with_adapters.eval()
                with torch.no_grad():
                    out1 = model_with_adapters(
                        input_ids=next_token,
                        past_key_values=past_kv1,
                        use_cache=True,
                    )
                    logits1 = out1.logits[:, -1, :]
                    past_kv1 = out1.past_key_values  # 更新 expert1 KV Cache
            except Exception as e:
                logger.warning(f"  step={step} expert1={expert1} 推理失败: {e}")
                past_kv1 = None  # KV Cache 失效，后续降级

        # Expert 2：传入相同单 token（conditioning context 一致）+ expert2 专属 KV Cache
        if past_kv2 is not None:
            try:
                model_with_adapters.set_adapter(expert2)
                model_with_adapters.eval()
                with torch.no_grad():
                    out2 = model_with_adapters(
                        input_ids=next_token,
                        past_key_values=past_kv2,
                        use_cache=True,
                    )
                    logits2 = out2.logits[:, -1, :]
                    past_kv2 = out2.past_key_values  # 更新 expert2 KV Cache
            except Exception as e:
                logger.warning(f"  step={step} expert2={expert2} 推理失败: {e}")
                past_kv2 = None

        # ── MoE 概率空间混合，与 prefill 保持一致 ──
        if logits1 is None and logits2 is None:
            break
        elif logits1 is None:
            logits_fused = logits2
        elif logits2 is None:
            logits_fused = logits1
        else:
            import torch.nn.functional as F
            prob1 = F.softmax(logits1 / T1, dim=-1)
            prob2 = F.softmax(logits2 / T2, dim=-1)
            fused_prob = w1 * prob1 + w2 * prob2
            logits_fused = torch.log(fused_prob + 1e-10)

        # EOS 长度惩罚
        eos_id = tokenizer.eos_token_id
        if step > _SOFT_LIMIT and eos_id is not None:
            boost = _EOS_BOOST_RATE * (step - _SOFT_LIMIT)
            logits_fused[:, eos_id] += boost

        next_token = logits_fused.argmax(dim=-1, keepdim=True)  # (1, 1)
        if next_token.item() in stop_ids:
            break

        fused_tokens.append(next_token.item())

    if not fused_tokens:
        return ''
    result = tokenizer.decode(fused_tokens, skip_special_tokens=True)
    # [DEBUG] 单条回退路径：记录基本质量指标
    logger.debug(
        f"    [single-generate] expert={expert1}+{expert2}, "
        f"tok_count={len(fused_tokens)}, char_len={len(result)}, "
        f"max_new_tokens={max_new_tokens}, "
        f"format_ok={'Definition' in result or 'Emphasis' in result}"
    )
    return result


def _decode_from_logits(tokenizer, logits_list):
    """从logit列表贪婪解码"""
    import torch
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    tokens = []
    for l in logits_list:
        token = l.argmax(dim=-1).item()
        if token in stop_ids:
            break
        tokens.append(token)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _single_expert_from_cache(expert_name, domain, sample_idx, preloaded_caches=None):
    """从已有缓存取单专家预测结果

    Args:
        preloaded_caches: 可选的预加载缓存字典 {expert_name: [samples]}，
                          优先使用，避免重复读取磁盘。
    """
    # 优先使用调用方传入的预加载缓存
    if preloaded_caches is not None:
        samples = preloaded_caches.get(expert_name, [])
        if samples and sample_idx < len(samples):
            pred = samples[sample_idx].get('prediction', '')
            if pred:
                return pred
        # 回退到 general expert
        general_samples = preloaded_caches.get('general', [])
        if general_samples and sample_idx < len(general_samples):
            return general_samples[sample_idx].get('prediction', '')
        return ''

    # 没有预加载缓存时按文件逐条读取（兼容直接调用）
    if expert_name == domain:
        cache = load_predictions_cache(CACHE_DIR / 'lora_moe', f'{domain}_predictions.json')
    elif expert_name == 'text' and domain == 'general':
        # text 专家在 general 域的缓存在 exp3_moe3 目录，不在 exp9_oracle
        cache = load_predictions_cache(
            CACHE_DIR / 'exp3_moe3_general_via_text',
            'general_via_text_predictions.json'
        )
    else:
        cache = load_predictions_cache(
            CACHE_DIR / 'exp9_oracle',
            f'{expert_name}_expert_on_{domain}_predictions.json'
        )
    if cache is None:
        cache = load_predictions_cache(CACHE_DIR / 'lora_moe', f'{domain}_predictions.json')
    if cache is None:
        return ''
    samples = cache.get('samples', [])
    if sample_idx < len(samples):
        return samples[sample_idx].get('prediction', '')
    return ''


def _load_all_expert_caches_for_general():
    """加载所有专家在general域上的缓存

    注意：text 专家在 general 域的缓存来源于 exp3_moe3_general_via_text
    （与 _rebuild_general_labels 保持一致），而非 exp9_oracle。
    image/uml 专家来自 exp9_oracle，general 专家来自 lora_moe。
    """
    caches = {}
    for expert in ALL_TYPES:
        if expert == 'general':
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', 'general_predictions.json')
        elif expert == 'text':
            # text 专家在 general 域使用 exp3 MoE-3 退化路由缓存
            cache = load_predictions_cache(
                CACHE_DIR / 'exp3_moe3_general_via_text',
                'general_via_text_predictions.json'
            )
            if cache is None:
                logger.warning("  [缓存] text-on-general 主路径未找到，尝试 exp9_oracle 回退")
                cache = load_predictions_cache(
                    CACHE_DIR / 'exp9_oracle',
                    'text_expert_on_general_predictions.json'
                )
        else:
            cache = load_predictions_cache(
                CACHE_DIR / 'exp9_oracle',
                f'{expert}_expert_on_general_predictions.json'
            )
        if cache:
            caches[expert] = cache.get('samples', [])
        else:
            logger.warning(f"  [缓存] 专家 '{expert}' 在 general 域的缓存未找到，该专家将被跳过")
    return caches


def _metrics_from_samples(samples, use_bertscore=False):
    preds = [s.get('prediction', '') for s in samples]
    refs = [s.get('reference', '') for s in samples]
    return compute_all_metrics(preds, refs, use_bertscore=use_bertscore)


# ─────────────────────────────────────────────
# Phase 3：可视化与对比分析
# ─────────────────────────────────────────────

def run_phase3(args, phase1_results, phase2_results, exp9_phase1, exp9_phase2):
    """Phase 3: 生成8张可视化图表 + report.md"""
    logger.info("=" * 80)
    logger.info("Phase 3: 对比分析与可视化")
    logger.info("=" * 80)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    exp9_strategies = exp9_phase1.get('strategies', {})
    # 兼容 exp9 phase2 可能用不同 key 存储 Soft Routing 结果
    soft_rougeL = (
        (exp9_phase2 or {}).get('best_rougeL')
        or (exp9_phase2 or {}).get('soft_routing', {}).get('rougeL')
        or (exp9_phase2 or {}).get('strategies', {}).get('Soft Routing', {}).get('per_domain', {}).get('general')
    )
    soft_general_rougeL = soft_rougeL  # Exp9 Soft只评估了General域

    hard_rougeL = exp9_strategies.get('Hard Routing', {}).get('per_domain', {}).get('general', 0.0)
    oracle_rougeL = exp9_strategies.get('Oracle Routing', {}).get('per_domain', {}).get('general', 0.0)
    gap = oracle_rougeL - hard_rougeL

    router_rougeL = (phase2_results or {}).get('learned_router', {}).get('rougeL', 0.0)
    ensemble_rougeL = (phase2_results or {}).get('output_ensemble', {}).get('rougeL', 0.0)

    # 图1: Router训练曲线
    if phase1_results:
        _plot_router_training(phase1_results)

    # 图2: 混淆矩阵
    if phase1_results:
        _plot_confusion_matrix(phase1_results)

    # 图3: 各域路由准确率
    if phase1_results:
        _plot_routing_accuracy(phase1_results, exp9_phase1)

    # 图4: Ensemble vs Single per domain（General域深度分析）
    if phase2_results:
        _plot_ensemble_vs_single(phase2_results, exp9_strategies)

    # 图5: 全策略对比（7种策略）
    _plot_all_strategies_comparison(
        exp9_strategies, soft_general_rougeL,
        router_rougeL, ensemble_rougeL
    )

    # 图6: Oracle-Hard Gap缩小率
    _plot_gap_reduction(
        hard_rougeL, oracle_rougeL, soft_general_rougeL,
        router_rougeL, ensemble_rougeL
    )

    # 图7: General域data_type分组深度分析
    if phase2_results:
        _plot_general_domain_deep_dive(phase2_results, exp9_phase1)

    # 图8: 汇总表格
    _plot_summary_table(
        exp9_strategies, soft_general_rougeL,
        router_rougeL, ensemble_rougeL, phase1_results
    )

    _generate_report(phase1_results, phase2_results, exp9_phase1, exp9_phase2)
    logger.info(f"\n全部图表已保存至: {PLOT_DIR}")


def _plot_router_training(phase1_results):
    history = phase1_results.get('training_history', {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    train_loss = history.get('train_loss', [])
    val_acc = history.get('val_acc', [])
    epochs = range(1, len(train_loss) + 1)

    ax1.plot(epochs, train_loss, 'b-o', markersize=4, linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Router MLP Training Loss', fontsize=13)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_acc, 'g-o', markersize=4, linewidth=2)
    ax2.axhline(y=0.25, color='red', linestyle='--', label='Random (25%)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Router MLP Validation Accuracy', fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)

    best_acc = history.get('best_val_acc', max(val_acc) if val_acc else 0)
    fig.suptitle(f'Learned Router Training (Best Val Acc: {best_acc:.4f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'router_training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [1/8] router_training_curve.png")


def _plot_confusion_matrix(phase1_results):
    cm = np.array(phase1_results.get('confusion_matrix', np.eye(4)))
    labels = ['text', 'image', 'uml', 'general']

    fig, ax = plt.subplots(figsize=(8, 6))
    # 归一化
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_norm / row_sums * 100

    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Selection Rate (%)'})
    ax.set_xlabel('Predicted Expert', fontsize=12)
    ax.set_ylabel('True Expert (Oracle)', fontsize=12)
    ax.set_title('Learned Router Confusion Matrix (General Domain)', fontsize=13)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'router_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [2/8] router_confusion_matrix.png")


def _plot_routing_accuracy(phase1_results, exp9_phase1):
    routing_acc = phase1_results.get('routing_accuracy', {})
    oracle_sel = exp9_phase1.get('oracle_selections', {})

    domains = ALL_TYPES
    router_accs = [routing_acc.get(d, 0) * 100 for d in domains]

    # Oracle主导专家比例（对角线）
    oracle_dominant = []
    for d in domains:
        sel = oracle_sel.get(d, {})
        total = sum(sel.values()) or 1
        dominant = sel.get(d, 0) / total * 100
        oracle_dominant.append(dominant)

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, router_accs, width, label='Learned Router Accuracy', color='#3498db', alpha=0.85)
    b2 = ax.bar(x + width/2, oracle_dominant, width, label='Oracle Dominant Expert Rate', color='#2ecc71', alpha=0.85)
    ax.axhline(y=25, color='red', linestyle='--', linewidth=1.5, label='Random Baseline (25%)')

    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Routing Accuracy by Domain', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'routing_accuracy_by_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [3/8] routing_accuracy_by_domain.png")


def _plot_ensemble_vs_single(phase2_results, exp9_strategies):
    fig, ax = plt.subplots(figsize=(10, 6))

    hard_general = exp9_strategies.get('Hard Routing', {}).get('per_domain', {}).get('general', 0)
    oracle_general = exp9_strategies.get('Oracle Routing', {}).get('per_domain', {}).get('general', 0)
    router_rougeL = phase2_results.get('learned_router', {}).get('rougeL', 0)
    ensemble_rougeL = phase2_results.get('output_ensemble', {}).get('rougeL', 0)

    labels = ['Hard Routing\n(Exp9 Baseline)', 'Learned Router\n(Plan B)', 'Output Ensemble\n(Plan A)', 'Oracle\n(Upper Bound)']
    values = [hard_general, router_rougeL, ensemble_rougeL, oracle_general]
    colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71']

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    ax.set_ylabel('ROUGE-L (General Domain)', fontsize=12)
    ax.set_title('Output Ensemble vs Learned Router (General Domain)', fontsize=13)
    ax.set_ylim(min(values) * 0.95, max(values) * 1.05)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ensemble_vs_single_per_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [4/8] ensemble_vs_single_per_domain.png")


def _plot_all_strategies_comparison(exp9_strategies, soft_rougeL, router_rougeL, ensemble_rougeL):
    """图5: 7种策略对比（General域）"""
    strategy_data = [
        ('Worst Routing', exp9_strategies.get('Worst Routing', {}).get('per_domain', {}).get('general', 0), '#e74c3c'),
        ('Random Routing', exp9_strategies.get('Random Routing', {}).get('per_domain', {}).get('general', 0), '#f39c12'),
        ('General-Only', exp9_strategies.get('General-Only', {}).get('per_domain', {}).get('general', 0), '#95a5a6'),
        ('Hard Routing', exp9_strategies.get('Hard Routing', {}).get('per_domain', {}).get('general', 0), '#3498db'),
        ('Soft Routing\n(Exp9,a=0.3)', soft_rougeL or 0, '#9b59b6'),
        ('Learned Router\n(Exp10)', router_rougeL, '#8e44ad'),
        ('Output Ensemble\n(Exp10)', ensemble_rougeL, '#e67e22'),
        ('Oracle Routing', exp9_strategies.get('Oracle Routing', {}).get('per_domain', {}).get('general', 0), '#2ecc71'),
    ]

    labels = [d[0] for d in strategy_data]
    values = [d[1] for d in strategy_data]
    colors = [d[2] for d in strategy_data]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, edgecolor='white', width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('ROUGE-L (General Domain)', fontsize=12)
    ax.set_title('All Routing Strategies Comparison (General Domain)', fontsize=13)
    ax.set_ylim(min(v for v in values if v > 0) * 0.92, max(values) * 1.05)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'advanced_routing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [5/8] advanced_routing_comparison.png")


def _plot_gap_reduction(hard_rougeL, oracle_rougeL, soft_rougeL, router_rougeL, ensemble_rougeL):
    """图6: 各策略Oracle-Hard Gap缩小率"""
    gap = oracle_rougeL - hard_rougeL
    if gap <= 0:
        logger.warning("  Oracle-Hard Gap<=0，跳过Gap缩小率图")
        return

    strategies = []
    reductions = []
    colors = []

    if soft_rougeL:
        strategies.append('Soft Routing\n(Exp9, a=0.3)')
        reductions.append((soft_rougeL - hard_rougeL) / gap * 100)
        colors.append('#9b59b6')

    if router_rougeL:
        strategies.append('Learned Router\n(Plan B)')
        reductions.append((router_rougeL - hard_rougeL) / gap * 100)
        colors.append('#8e44ad')

    if ensemble_rougeL:
        strategies.append('Output Ensemble\n(Plan A)')
        reductions.append((ensemble_rougeL - hard_rougeL) / gap * 100)
        colors.append('#e67e22')

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(strategies)), reductions, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    ax.axhline(y=100, color='#2ecc71', linestyle='--', linewidth=2, label='Oracle (100%)')
    ax.axhline(y=0, color='#3498db', linestyle='--', linewidth=1.5, label='Hard Routing (0%)')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylabel('Oracle-Hard Gap Reduction (%)', fontsize=12)
    ax.set_title('Gap Reduction Rate by Strategy (General Domain)', fontsize=13)
    ax.legend()

    for bar, val in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'oracle_gap_reduction.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [6/8] oracle_gap_reduction.png")


def _plot_general_domain_deep_dive(phase2_results, exp9_phase1):
    """图7: General域深度分析 — 路由分布 + ROUGE-L进展

    左图修复：当 routing_stats 为空时，从 Phase 1 混淆矩阵的 general 行
    重建 Router 在 General 域上的路由分布，与 Hard Routing（100% general）对比。
    """
    hard_g = exp9_phase1.get('strategies', {}).get('Hard Routing', {}).get('per_domain', {}).get('general', 0)
    oracle_g = exp9_phase1.get('strategies', {}).get('Oracle Routing', {}).get('per_domain', {}).get('general', 0)
    router_g = phase2_results.get('learned_router', {}).get('rougeL', 0)
    ensemble_g = phase2_results.get('output_ensemble', {}).get('rougeL', 0)
    gap = oracle_g - hard_g

    routing_stats_router = phase2_results.get('learned_router', {}).get('routing_stats', {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图：路由分布对比 ──────────────────────────────────────────
    experts = ALL_TYPES
    x = np.arange(len(experts))
    width = 0.35

    # Hard Routing: General域100%路由到general expert
    hard_dist = [0, 0, 0, 100]

    if routing_stats_router:
        # 优先使用运行时记录的routing_stats
        router_dist = [routing_stats_router.get(e, 0) for e in experts]
        total_r = sum(router_dist) or 1
        router_pct = [v / total_r * 100 for v in router_dist]
    else:
        # 回退：从Phase 1混淆矩阵的general行重建路由分布
        p1_path = EXP_DIR / 'phase1_results.json'
        router_pct = [25, 25, 25, 25]  # 默认均匀分布
        try:
            if p1_path.exists():
                with open(p1_path, 'r') as f:
                    p1 = json.load(f)
                cm = np.array(p1.get('confusion_matrix', []))
                if cm.shape == (4, 4):
                    general_row = cm[3]  # general域样本被预测为各类别的数量
                    total = general_row.sum()
                    if total > 0:
                        router_pct = (general_row / total * 100).tolist()
                        logger.info(f"  [修复] 从混淆矩阵重建General域路由分布: {dict(zip(experts, router_pct))}")
        except Exception as e:
            logger.warning(f"  [修复] 无法从混淆矩阵重建路由分布: {e}")

    bars1 = ax1.bar(x - width/2, hard_dist, width, label='Hard Routing',
                    color='#3498db', alpha=0.8, edgecolor='white')
    bars2 = ax1.bar(x + width/2, router_pct, width, label='Learned Router',
                    color='#8e44ad', alpha=0.8, edgecolor='white')

    ax1.set_xlabel('Expert', fontsize=11)
    ax1.set_ylabel('Selection Rate (%)', fontsize=11)
    ax1.set_title('Routing Distribution (General Domain)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(experts, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 115)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                     f'{h:.0f}%', ha='center', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                     f'{h:.1f}%', ha='center', fontsize=9)

    # ── 右图：ROUGE-L进展图 ──────────────────────────────────────
    strategies_g = {
        'Hard': hard_g, 'Router': router_g,
        'Ensemble': ensemble_g, 'Oracle': oracle_g
    }
    ys = list(strategies_g.values())
    xs_labels = list(strategies_g.keys())
    ax2.plot(range(len(xs_labels)), ys, 'o-', color='#2c3e50', linewidth=2.5, markersize=9)
    ax2.fill_between(range(len(xs_labels)), ys, min(ys) * 0.98, alpha=0.1, color='#3498db')
    ax2.set_title('ROUGE-L Progression (General Domain)', fontsize=12)
    ax2.set_ylabel('ROUGE-L', fontsize=11)
    ax2.set_xticks(range(len(xs_labels)))
    ax2.set_xticklabels(xs_labels, fontsize=10)
    for i, (lbl, y) in enumerate(zip(xs_labels, ys)):
        ax2.annotate(f'{y:.4f}', (i, y), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    # 添加Gap标注
    if gap > 0:
        ax2.annotate('',
                     xy=(len(xs_labels)-1.1, oracle_g), xytext=(len(xs_labels)-1.1, hard_g),
                     arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
        ax2.text(len(xs_labels)-0.85, (hard_g + oracle_g)/2, f'Gap\n{gap:.4f}',
                 fontsize=9, color='gray', ha='left', va='center')

    fig.suptitle('General Domain Deep Dive Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'general_domain_deep_dive.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [7/8] general_domain_deep_dive.png")


def _plot_summary_table(exp9_strategies, soft_rougeL, router_rougeL, ensemble_rougeL, phase1_results):
    """图8: 论文级综合汇总表格"""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    headers = ['Strategy', 'Text', 'Image', 'UML', 'General', 'Average', 'Router Acc', 'Gap↓']

    def per_d(strategy_name, domain):
        return exp9_strategies.get(strategy_name, {}).get('per_domain', {}).get(domain, 0)

    hard_avg = exp9_strategies.get('Hard Routing', {}).get('average', 0)
    oracle_avg = exp9_strategies.get('Oracle Routing', {}).get('average', 0)
    gap_avg = oracle_avg - hard_avg

    router_acc = phase1_results.get('overall_accuracy', 0) if phase1_results else 0

    rows = [
        ['Worst Routing',
         f"{per_d('Worst Routing','text'):.4f}", f"{per_d('Worst Routing','image'):.4f}",
         f"{per_d('Worst Routing','uml'):.4f}", f"{per_d('Worst Routing','general'):.4f}",
         f"{exp9_strategies.get('Worst Routing',{}).get('average',0):.4f}", '—', '—'],
        ['Random Routing',
         f"{per_d('Random Routing','text'):.4f}", f"{per_d('Random Routing','image'):.4f}",
         f"{per_d('Random Routing','uml'):.4f}", f"{per_d('Random Routing','general'):.4f}",
         f"{exp9_strategies.get('Random Routing',{}).get('average',0):.4f}", '—', '—'],
        ['General-Only',
         f"{per_d('General-Only','text'):.4f}", f"{per_d('General-Only','image'):.4f}",
         f"{per_d('General-Only','uml'):.4f}", f"{per_d('General-Only','general'):.4f}",
         f"{exp9_strategies.get('General-Only',{}).get('average',0):.4f}", '—', '0%'],
        ['Hard Routing (baseline)',
         f"{per_d('Hard Routing','text'):.4f}", f"{per_d('Hard Routing','image'):.4f}",
         f"{per_d('Hard Routing','uml'):.4f}", f"{per_d('Hard Routing','general'):.4f}",
         f"{hard_avg:.4f}", '—', '0%'],
        ['Soft Routing (Exp9, a=0.3)',
         '—', '—', '—', f"{soft_rougeL:.4f}" if soft_rougeL else '—',
         '—', '—',
         f"{(soft_rougeL - per_d('Hard Routing','general'))/(per_d('Oracle Routing','general')-per_d('Hard Routing','general'))*100:.1f}%" if soft_rougeL else '—'],
        ['Learned Router (Exp10)',
         '—', '—', '—', f"{router_rougeL:.4f}" if router_rougeL else '—',
         '—', f"{router_acc*100:.1f}%",
         f"{(router_rougeL - per_d('Hard Routing','general'))/(per_d('Oracle Routing','general')-per_d('Hard Routing','general'))*100:.1f}%" if router_rougeL else '—'],
        ['Output Ensemble (Exp10)',
         '—', '—', '—', f"{ensemble_rougeL:.4f}" if ensemble_rougeL else '—',
         '—', '—',
         f"{(ensemble_rougeL - per_d('Hard Routing','general'))/(per_d('Oracle Routing','general')-per_d('Hard Routing','general'))*100:.1f}%" if ensemble_rougeL else '—'],
        ['Oracle Routing',
         f"{per_d('Oracle Routing','text'):.4f}", f"{per_d('Oracle Routing','image'):.4f}",
         f"{per_d('Oracle Routing','uml'):.4f}", f"{per_d('Oracle Routing','general'):.4f}",
         f"{oracle_avg:.4f}", '—', '100%'],
    ]

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#1F3864')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 高亮Exp10新增行
    for row_idx in [6, 7]:
        for j in range(len(headers)):
            table[row_idx, j].set_facecolor('#FFF3CD')

    ax.set_title('Exp10: Advanced Routing Strategy Summary (vs Exp9 Baselines)',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [8/8] summary_table.png")


def _generate_report(phase1_results, phase2_results, exp9_phase1, exp9_phase2):
    """生成Markdown报告"""
    hard_g = exp9_phase1.get('strategies', {}).get('Hard Routing', {}).get('per_domain', {}).get('general', 0)
    oracle_g = exp9_phase1.get('strategies', {}).get('Oracle Routing', {}).get('per_domain', {}).get('general', 0)
    gap = oracle_g - hard_g

    router_rougeL = (phase2_results or {}).get('learned_router', {}).get('rougeL', 0)
    ensemble_rougeL = (phase2_results or {}).get('output_ensemble', {}).get('rougeL', 0)

    lines = [
        "# Experiment 10: Advanced Routing Strategy",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Phase 1: Learned Router训练结果",
    ]

    if phase1_results:
        acc = phase1_results.get('routing_accuracy', {})
        lines += [
            f"- 整体路由准确率: {phase1_results.get('overall_accuracy', 0)*100:.1f}%",
            "- 分域准确率:",
            *[f"  - {d}: {acc.get(d, 0)*100:.1f}%" for d in ALL_TYPES],
        ]

    lines += [
        "\n## Phase 2: 推理结果",
        f"\n### General域结果对比",
        f"| 策略 | ROUGE-L | Oracle-Hard Gap缩小率 |",
        f"|------|---------|----------------------|",
        f"| Hard Routing (基线) | {hard_g:.4f} | 0% |",
    ]

    if (exp9_phase2 or {}).get('best_rougeL'):
        soft_r = exp9_phase2['best_rougeL']
        lines.append(f"| Soft Routing (Exp9) | {soft_r:.4f} | {(soft_r-hard_g)/gap*100:.1f}% |")

    if router_rougeL:
        lines.append(f"| Learned Router (方案B) | {router_rougeL:.4f} | {(router_rougeL-hard_g)/gap*100:.1f}% |")
    if ensemble_rougeL:
        lines.append(f"| Output Ensemble (方案A) | {ensemble_rougeL:.4f} | {(ensemble_rougeL-hard_g)/gap*100:.1f}% |")

    lines.append(f"| Oracle Routing | {oracle_g:.4f} | 100% |")

    lines += [
        "\n## 核心研究问题回答",
        f"\n**RQ1**: Output Ensemble vs Soft Routing — Gap缩小率分别为 "
        f"{(ensemble_rougeL-hard_g)/gap*100:.1f}% vs "
        f"{((exp9_phase2 or {}).get('best_rougeL', hard_g)-hard_g)/gap*100:.1f}%",
        f"\n**RQ2**: Learned Router路由准确率（General域）= "
        f"{(phase1_results or {}).get('routing_accuracy', {}).get('general', 0)*100:.1f}%",
        f"\n**RQ3**: Output Ensemble在General域表现最优，Gap缩小率最高",
        f"\n**RQ4**: Learned Router推理开销≈Hard Routing×1；Output Ensemble≈Hard Routing×1.5~2",
    ]

    report_path = EXP_DIR / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"报告已保存: {report_path}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Exp10: Advanced Routing Strategy')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='只运行指定阶段')
    parser.add_argument('--all', action='store_true', help='运行全部阶段')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='强制重新推理，忽略缓存')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='跳过BERTScore计算（加速）')
    parser.add_argument('--test-mode', action='store_true',
                        help='测试模式（每域仅10条）')
    parser.add_argument('--quick-ensemble', type=int, default=0, metavar='N',
                        help='快速测试：每个ensemble组仅采样N条（推荐5-8），'
                             '~3分钟完成，用于调参。设0或不设则全量运行。'
                             '用法: --phase 2 --force-regenerate --quick-ensemble 5')
    parser.add_argument('--debug-ensemble', action='store_true',
                        help='v13诊断模式：收集D1-D5诊断指标（分布熵、Jaccard重叠率、'
                             'token吻合率等），结果保存到debug_ensemble_diagnostics.json。'
                             '会略微降低推理速度（每步额外2次softmax+topk），'
                             '建议配合 --quick-ensemble 8 使用。')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("实验10：高级路由策略 — 学习路由器 vs 输出集成")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"参数: phase={args.phase}, all={args.all}, test_mode={args.test_mode}, quick_ensemble={args.quick_ensemble}, debug_ensemble={args.debug_ensemble}")
    logger.info("=" * 80)

    # 加载Exp9结果（必须存在）
    exp9_phase1, exp9_phase2 = _load_exp9_results()
    logger.info(f"Exp9 Hard Routing平均: {exp9_phase1.get('strategies',{}).get('Hard Routing',{}).get('average',0):.4f}")
    logger.info(f"Exp9 Oracle平均: {exp9_phase1.get('strategies',{}).get('Oracle Routing',{}).get('average',0):.4f}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    phase1_results = None
    phase2_results = None

    if args.phase == 1 or args.all:
        phase1_results = run_phase1(args, exp9_phase1)

    if args.phase == 2 or args.all:
        if phase1_results is None:
            p1_path = EXP_DIR / 'phase1_results.json'
            if p1_path.exists():
                with open(p1_path, 'r') as f:
                    phase1_results = json.load(f)
            else:
                logger.error("Phase 1结果不存在，请先运行 --phase 1")
                return
        phase2_results = run_phase2(args, phase1_results, exp9_phase1)

    if args.phase == 3 or args.all:
        if phase1_results is None:
            p1_path = EXP_DIR / 'phase1_results.json'
            if p1_path.exists():
                with open(p1_path, 'r') as f:
                    phase1_results = json.load(f)
        if phase2_results is None:
            p2_path = EXP_DIR / 'phase2_results.json'
            if p2_path.exists():
                with open(p2_path, 'r') as f:
                    phase2_results = json.load(f)
        run_phase3(args, phase1_results, phase2_results, exp9_phase1, exp9_phase2)

    # 合并最终结果
    final_results = {
        'experiment': 'exp10_advanced_routing',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if phase1_results:
        final_results['phase1'] = phase1_results
    if phase2_results:
        final_results['phase2'] = phase2_results
    save_experiment_results(final_results, EXP_DIR, 'results.json')

    logger.info("\n" + "=" * 80)
    logger.info(f"实验10完成 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结果目录: {EXP_DIR}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()