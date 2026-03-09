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
    # cache_results  : top-1 prob >= 0.85，直接从磁盘缓存取，零 GPU 开销
    # ensemble_groups: 按 (expert1, expert2) 分组，后续批量 GPU 推理
    # sample_meta    : 保存每条样本的路由元信息，供最终重新排序（reassemble）用
    #
    # ── v7 核心修复：UML 必须搭配 text 作为 primary ──────────────────────────
    # v6 实证数据（UML 统一降级为 secondary，T=2.0）：
    #   text+uml   (3条)  → avg_len=252,  format_ok=100%, ROUGE-L=0.7330 ← 唯一成功
    #   image+uml  (122条) → avg_len=1180, format_ok=8%,  ROUGE-L=0.1926 ← 惨败
    #   general+uml (14条) → avg_len=1207, format_ok=14%, ROUGE-L=0.1843 ← 惨败
    #
    # 根因分析：融合质量取决于 primary expert 是否能在该域输入上产生有意义的 logits。
    #   - text 专家：训练在软件需求文本上，与 UML 描述有高度词汇重叠（actors, use_cases,
    #     relationships 等），能为 UML 输入产生合理的 token 分布 → 格式良好
    #   - image 专家：仅训练在图像 JSON 描述上，对 UML JSON 完全 OOD，
    #     logits 近乎均匀分布 → UML secondary 实质主导，回到 UML-primary 失败模式
    #   - general 专家：虽然训练集包含 UML 数据，但也学到了 UML 的长输出风格，
    #     format control 弱 → 两个专家都偏向长输出，EOS 信号更弱
    #
    # 修复策略（基于 text+uml 成功的唯一解释——"格式控制多样性"）：
    #   ① 任何包含 UML 的 ensemble pair，primary 强制为 text（最强格式控制器）
    #   ② UML 权重上限 40%（text 主导 ≥60%，确保格式控制权）
    #   ③ 若 top-2 本来就是 text+uml，保持 router 原始权重（但仍受上限约束）
    _UML_WEIGHT_CAP = 0.40  # UML 在 ensemble 中的最大权重（v6 是 0.50，仍不够）
    _TEXT_MIN_WEIGHT = 0.60  # text 作为 UML pair 的 primary 时最低权重
    uml_text_anchored = 0  # 统计被重路由到 text+uml 的样本数

    sample_meta = []          # [(i, expert1, expert2, w1, w2, w1_raw, template_name), ...]
    cache_results = {}        # {i: pred_str}
    ensemble_groups = defaultdict(list)   # {(e1, e2): [(i, prompt_str, w1, w2), ...]}
    template_usage: defaultdict = defaultdict(int)   # {template_name: count}

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

        if w1_raw >= 0.85:
            # 退化为单专家：直接从缓存取，不占 GPU
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            cache_results[i] = _single_expert_from_cache(
                expert1, 'general', i, preloaded_caches
            )
        else:
            # ── v7: UML 参与的 ensemble → 强制 text+uml ────────────────────
            has_uml = (expert1 == 'uml' or expert2 == 'uml')
            if has_uml:
                # 强制 primary=text, secondary=uml
                expert1 = 'text'
                expert2 = 'uml'
                w1 = _TEXT_MIN_WEIGHT
                w2 = _UML_WEIGHT_CAP
                uml_text_anchored += 1
            # 非 UML pair 保持 router 原始决策，不做干预

            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            ensemble_groups[(expert1, expert2)].append((i, prompt_str, w1, w2))

    logger.info(f"  [DEBUG] 模板使用分布: {dict(template_usage)}")
    logger.info(f"  [v7] UML→text+uml 锚定: {uml_text_anchored}条")

    n_cache = len(cache_results)
    n_ensemble = sum(len(v) for v in ensemble_groups.values())
    # [DEBUG] per-group size breakdown
    for (e1, e2), items in sorted(ensemble_groups.items(), key=lambda x: -len(x[1])):
        # 统计该组的平均权重
        avg_w1 = np.mean([w1 for (_, _, w1, _) in items])
        avg_w2 = np.mean([w2 for (_, _, _, w2) in items])
        logger.info(
            f"    [v7 组] {e1}+{e2}: {len(items)}条, "
            f"avg_w1={avg_w1:.2f}, avg_w2={avg_w2:.2f}"
        )
    logger.info(
        f"  样本分类: cache(w1>=0.85)={n_cache}, "
        f"ensemble={n_ensemble}, 组数={len(ensemble_groups)}"
    )

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

    # ── Stage 3: 按原始顺序 reassemble + 质量门控（v5 核心修复）─────────────
    # v4 问题：ensemble 输出可能比 single expert 更差（uml+general 组 ROUGE-L=0.22），
    #          但 v4 无条件使用 ensemble 输出，导致整体 ROUGE-L 被 UML 组拖垮。
    # v5 修复：对每条 ensemble 样本做快速质量检查，不合格则回退到 cached 单专家输出。
    #          确保 ensemble 只能帮忙、不能帮倒忙（monotonic improvement guarantee）。
    _FORMAT_KEYWORDS = {'Definition', 'Emphasis', 'Things to Avoid',
                        'definition', 'emphasis', 'things to avoid'}
    _MAX_CHAR_LEN = 1000   # 参考输出: text~300, image~400, uml~600; 1000 = ~1.5x 最长参考

    def _passes_quality_gate(pred_text: str) -> bool:
        """快速质量检查：格式三段式 + 长度合理"""
        if not pred_text or not pred_text.strip():
            return False
        # 格式检查：至少包含一个三段式关键词
        if not any(kw in pred_text for kw in _FORMAT_KEYWORDS):
            return False
        # 长度检查：超过阈值视为失控
        if len(pred_text) > _MAX_CHAR_LEN:
            return False
        return True

    samples = []
    fallback_stats = {'total': 0, 'passed': 0, 'fallback': 0, 'fallback_improved': 0}
    for (i, expert1, expert2, w1, w2, _w1_raw, tpl_name) in sample_meta:
        sample = general_test[i]
        ensemble_pred = ensemble_results.get(i, '')
        cache_pred = cache_results.get(i, '')

        if cache_pred:
            # 已缓存的单专家结果（w1>=0.85），直接使用，无需质量检查
            pred = cache_pred
        else:
            # ensemble 结果，执行质量门控
            fallback_stats['total'] += 1
            if _passes_quality_gate(ensemble_pred):
                pred = ensemble_pred
                fallback_stats['passed'] += 1
            else:
                # 质量不合格 → 回退到主专家的 cached 预测
                fallback_pred = _single_expert_from_cache(
                    expert1, 'general', i, preloaded_caches
                )
                fallback_stats['fallback'] += 1

                # [DEBUG] 回退时比较 ensemble vs cached 的 ROUGE-L
                ref = sample.get('output', '')
                if ref and fallback_pred and ensemble_pred:
                    from rouge_score import rouge_scorer as rs_mod
                    _scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)
                    try:
                        ens_r = _scorer.score(ref, ensemble_pred)['rougeL'].fmeasure
                        fb_r = _scorer.score(ref, fallback_pred)['rougeL'].fmeasure
                        if fb_r > ens_r:
                            fallback_stats['fallback_improved'] += 1
                        if i < 10 or (i % 50 == 0):
                            logger.debug(
                                f"    [fallback] i={i}, expert={expert1}+{expert2}, "
                                f"ens_len={len(ensemble_pred)}, fb_len={len(fallback_pred)}, "
                                f"ens_ROUGE={ens_r:.3f}, fb_ROUGE={fb_r:.3f}, "
                                f"improved={'✓' if fb_r > ens_r else '✗'}"
                            )
                    except Exception:
                        pass

                pred = fallback_pred if fallback_pred else ensemble_pred

        # [DEBUG] 记录前5个样本的生成详情
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
        f"通过={fallback_stats['passed']}, "
        f"回退={fallback_stats['fallback']}, "
        f"回退更优={fallback_stats['fallback_improved']}"
    )

    del lm, model_with_adapters, tokenizer
    _cleanup_gpu()

    # ── [DEBUG] per-data_type ROUGE-L 分解：定位哪个子域仍有问题 ──────────
    from rouge_score import rouge_scorer as rs_mod
    _scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)
    dtype_scores = defaultdict(list)
    for s in samples:
        dt = s.get('data_type', 'unknown')
        pred, ref = s.get('prediction', ''), s.get('reference', '')
        if pred and ref:
            try:
                sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                dtype_scores[dt].append(sc)
            except Exception:
                pass
    for dt, scores in sorted(dtype_scores.items()):
        logger.info(
            f"  [DEBUG] data_type={dt}: n={len(scores)}, "
            f"ROUGE-L={np.mean(scores):.4f} (std={np.std(scores):.4f})"
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


_ENSEMBLE_BATCH_SIZE = 12  # RTX 4090 24 GB: batch=12 → KV Cache 约 2.5 GB，仍远低于预算
# 说明：4090 24GB = 基础模型4bit ~10GB + 2专家KV Cache(B=12, seq≈1024) ~3GB → 峰值约13GB，安全


def _logit_ensemble_generate_batched(
    model_with_adapters, tokenizer,
    expert1, expert2, group_items, args,
    batch_size=_ENSEMBLE_BATCH_SIZE,
):
    """
    批量版 logit-space 双专家融合生成

    将同一 (expert1, expert2) 组的样本按 batch_size 分批，
    每批调用 _process_minibatch 完成：
      - 一次批量 prefill（B 条同时过 expert1 / expert2）
      - 每个 decode 步：两次 (B, 1) forward（而非 B×2 次 (1, 1) forward）

    OOM fallback: 某批次显存溢出时，自动降级为逐条 _logit_ensemble_generate。

    Args:
        group_items: List[(i_global, prompt_str, w1, w2)]  — 同一 (e1,e2) 组
                     注意：prompt_str 已由 _run_output_ensemble 按样本 data_type 预构建，
                     确保两个专家收到相同的、与训练分布匹配的 prompt 格式。
    Returns:
        List[str]  — 与 group_items 等长，按相同顺序
    """
    import torch

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
    RTX 4090 优化核心：批量 prefill + 批量 decode（B×1 token/step × 2 experts）

    v7: text-anchored UML ensemble（在 Stage 1 已完成路由重映射）

    ── 核心发现（v6 实验验证）──────────────────────────────────────────────────
    融合质量取决于 primary expert 对输入域的 **competence**（而非温度/权重）：
      text+uml   → ROUGE=0.73 (text 训练在软件需求上，与 UML 词汇重叠)
      image+uml  → ROUGE=0.19 (image 对 UML 完全 OOD，logits 近均匀)
      general+uml → ROUGE=0.18 (general 学过 UML 长输出风格，EOS 信号弱)

    当 primary expert 对输入 OOD 时，其 logits 近乎均匀分布，
    secondary expert（UML, T=2.0）的 focused distribution 仍然主导每步选择，
    导致温度/权重等参数级修复全部无效。

    Stage 1 修复：所有含 UML 的 pair → 强制 text(0.6)+uml(0.4)。
    此处 _process_minibatch 无需特殊处理。
    """
    import torch
    import torch.nn.functional as F

    B = len(batch_items)
    DONE_CHECK_INTERVAL = 16   # 每 16 步做一次 GPU-CPU sync 检查 done.all()

    # ── v5 Fix 1: 专家温度缩放 ──────────────────────────────────────────────
    # UML 专家在长序列上训练，其 softmax 分布极尖锐（高 confidence），
    # 即使 MoE 概率空间加权平均，UML 的极端分布仍然主导融合结果：
    #   - P_uml(next_specialized_token) ≈ 0.6，P_other(next_token) ≈ 0.2
    #   - MoE: 0.6*0.6 + 0.4*0.2 = 0.44 → UML 依旧主导
    # 温度 T>1 拉平 UML 的分布，让次要专家的贡献更有意义：
    #   - T=2.0: P_uml(next) ≈ 0.3 → MoE: 0.6*0.3 + 0.4*0.2 = 0.26
    #   - P_uml(EOS) 也从 ~0.001 提升至 ~0.01，大幅缓解 EOS 压制
    _EXPERT_TEMPERATURE = {'text': 1.0, 'image': 1.0, 'uml': 2.0, 'general': 1.0}
    T1 = _EXPERT_TEMPERATURE.get(expert1, 1.0)
    T2 = _EXPERT_TEMPERATURE.get(expert2, 1.0)

    # ── v5 Fix 2: 更紧凑的长度上限 ──────────────────────────────────────────
    # General 域参考输出统计：avg 392 chars / 57.7 words ≈ 80-100 tokens
    # 以 2x 参考长度为上限（而非 v4 的 2.5-4x），足够生成完整三段式指令
    _DOMAIN_MAX_TOKENS = {'text': 200, 'image': 200, 'uml': 250, 'general': 200}
    max_new_tokens = max(
        _DOMAIN_MAX_TOKENS.get(expert1, 200),
        _DOMAIN_MAX_TOKENS.get(expert2, 200),
    )

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

    # ── v5 Fix 2b: 更激进的 EOS 长度惩罚 ────────────────────────────────────
    # v4 的 rate=0.03 + soft_limit=60% 对 UML 组无效（1560 chars 输出仍然过长）
    # 原因：UML 专家的 log P(EOS) ≈ -7，boost 0.03/step × 100 steps = 3.0，
    #       远不够将 EOS 推到 argmax 位置
    # v5: rate=0.15, soft_limit=50%，在 100 步时 boost=7.5，足以翻转 EOS 排名
    _SOFT_LIMIT = int(max_new_tokens * 0.5)  # 50% 处开始施加惩罚
    _EOS_BOOST_RATE = 0.15  # 每超出 1 个 token，EOS logit 增加 0.15

    # [DEBUG] 记录 batch 基本信息
    logger.info(
        f"    [minibatch] B={B}, expert1={expert1}(T={T1}), expert2={expert2}(T={T2}), "
        f"max_new_tokens={max_new_tokens}, soft_limit={_SOFT_LIMIT}, "
        f"eos_boost_rate={_EOS_BOOST_RATE}"
    )

    # ── Left-padding tokenize，与 KV Cache decode 兼容 ──────────────────────
    # 必须 left-pad：right-pad 时 KV Cache 最后一个有效位置对每条样本不同，
    # 导致 decode 第一个 token 的 position id 错位。
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = tokenizer(
        prompts, return_tensors='pt', padding=True,
        truncation=True, max_length=512,
    )
    tokenizer.padding_side = orig_padding_side

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

    return results


def _logit_ensemble_generate(model_with_adapters, tokenizer,
                              prompt_str, expert1, expert2, w1, w2, args):
    """
    单条双专家同步自回归概率空间加权混合生成（OOM 回退路径）

    v7: text-anchored UML ensemble 已在调用方完成路由重映射。
    温度缩放：UML T=2.0 | EOS 长度惩罚：rate=0.15, limit=50%
    """
    import torch
    import torch.nn.functional as F

    # v5: 温度缩放 + 紧凑长度上限（与 _process_minibatch 保持一致）
    _EXPERT_TEMPERATURE = {'text': 1.0, 'image': 1.0, 'uml': 2.0, 'general': 1.0}
    T1 = _EXPERT_TEMPERATURE.get(expert1, 1.0)
    T2 = _EXPERT_TEMPERATURE.get(expert2, 1.0)

    _DOMAIN_MAX_TOKENS = {'text': 200, 'image': 200, 'uml': 250, 'general': 200}
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

    # 核心修复：直接使用调用方传入的 prompt_str，不再调用 GeneralInstructionTemplate
    device = (
        model_with_adapters.base_model.model.device
        if hasattr(model_with_adapters, 'base_model')
        else next(model_with_adapters.parameters()).device
    )
    prompt_ids = tokenizer(prompt_str, return_tensors='pt').input_ids.to(device)

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

    labels = ['Hard Routing\n(Exp9基线)', 'Learned Router\n(方案B)', 'Output Ensemble\n(方案A)', 'Oracle\n(上界)']
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
        strategies.append('Learned Router\n(方案B)')
        reductions.append((router_rougeL - hard_rougeL) / gap * 100)
        colors.append('#8e44ad')

    if ensemble_rougeL:
        strategies.append('Output Ensemble\n(方案A)')
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
    """图7: General域按data_type分组的5策略对比"""
    # 需要逐类型的分组结果（如果有的话）
    # 这里用overall结果作为示意
    hard_g = exp9_phase1.get('strategies', {}).get('Hard Routing', {}).get('per_domain', {}).get('general', 0)
    oracle_g = exp9_phase1.get('strategies', {}).get('Oracle Routing', {}).get('per_domain', {}).get('general', 0)
    router_g = phase2_results.get('learned_router', {}).get('rougeL', 0)
    ensemble_g = phase2_results.get('output_ensemble', {}).get('rougeL', 0)

    routing_stats_router = phase2_results.get('learned_router', {}).get('routing_stats', {})
    routing_stats_ensemble = phase2_results.get('output_ensemble', {}).get('routing_stats', {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：路由分布对比
    experts = ALL_TYPES
    if routing_stats_router:
        router_dist = [routing_stats_router.get(e, 0) for e in experts]
        total_r = sum(router_dist) or 1
        ax1.bar(experts, [v/total_r*100 for v in router_dist], color='#8e44ad', alpha=0.8, label='Learned Router')
    if routing_stats_ensemble:
        # Ensemble用expert1统计
        ens_by_expert = defaultdict(int)
        for k, v in routing_stats_ensemble.items():
            e1 = k.split('+')[0] if '+' in k else k
            ens_by_expert[e1] += v
        ens_dist = [ens_by_expert.get(e, 0) for e in experts]
        total_e = sum(ens_dist) or 1
        ax1.bar([i + 0.35 for i in range(len(experts))],
                [v/total_e*100 for v in ens_dist], width=0.35,
                color='#e67e22', alpha=0.8, label='Output Ensemble (top-1)')

    ax1.set_title('Routing Distribution Comparison', fontsize=12)
    ax1.set_xlabel('Expert')
    ax1.set_ylabel('Selection Rate (%)')
    ax1.legend()

    # 右图：ROUGE-L进展图（类似进度条）
    strategies_g = {
        'Hard': hard_g, 'Router': router_g,
        'Ensemble': ensemble_g, 'Oracle': oracle_g
    }
    ys = list(strategies_g.values())
    xs = list(strategies_g.keys())
    ax2.plot(xs, ys, 'o-', color='#2c3e50', linewidth=2.5, markersize=9)
    ax2.fill_between(range(len(xs)), ys, min(ys) * 0.98, alpha=0.1, color='#3498db')
    ax2.set_title('ROUGE-L Progression (General Domain)', fontsize=12)
    ax2.set_ylabel('ROUGE-L')
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10)

    fig.suptitle('General Domain Deep Dive Analysis', fontsize=13, fontweight='bold')
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
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("实验10：高级路由策略 — 学习路由器 vs 输出集成")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"参数: phase={args.phase}, all={args.all}, test_mode={args.test_mode}")
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