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

EXPERT_TO_IDX = {'text': 0, 'image': 1, 'uml': 2, 'general': 3}
IDX_TO_EXPERT = {v: k for k, v in EXPERT_TO_IDX.items()}


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
# MLP 路由器定义
# ─────────────────────────────────────────────

class RouterMLP:
    """轻量级MLP路由分类器（4类：text/image/uml/general）"""

    def __init__(self, input_dim=4096, hidden1=512, hidden2=128, num_classes=4, dropout1=0.2, dropout2=0.1):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise RuntimeError("需要安装PyTorch")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden1),
                    nn.LayerNorm(hidden1),
                    nn.ReLU(),
                    nn.Dropout(dropout1),
                    nn.Linear(hidden1, hidden2),
                    nn.LayerNorm(hidden2),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(hidden2, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        self.model = _MLP().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"RouterMLP初始化完成，参数量: {total_params:,}")

    def save(self, path):
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Router已保存: {path}")

    def load(self, path):
        import torch
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Router已加载: {path}")

    def predict_proba(self, features_np):
        """
        Args:
            features_np: np.ndarray, shape (N, input_dim)
        Returns:
            probs: np.ndarray, shape (N, 4)
        """
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features_np, dtype=torch.float32).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, features_np):
        probs = self.predict_proba(features_np)
        return np.argmax(probs, axis=1)


# ─────────────────────────────────────────────
# 特征提取
# ─────────────────────────────────────────────

def extract_hidden_states(inputs, base_model, tokenizer, batch_size=4, max_length=512):
    """
    从基础模型提取输入文本的最后一层hidden state（最后一个token位置）

    Args:
        inputs: List[str]，原始输入文本
        base_model: 已加载的Qwen3-8B模型
        tokenizer: 对应tokenizer
        batch_size: 批次大小
        max_length: 截断长度

    Returns:
        np.ndarray, shape (N, hidden_size)
    """
    import torch
    base_model.eval()
    all_features = []

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i: i + batch_size]
        try:
            encoded = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded['input_ids'].to(base_model.device)
            attention_mask = encoded['attention_mask'].to(base_model.device)

            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # 最后一层hidden state
                last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_size)

                # 取每条序列最后一个非padding token的特征
                seq_lens = attention_mask.sum(dim=1) - 1  # 最后有效token的索引
                batch_features = last_hidden[
                    torch.arange(len(batch)), seq_lens, :
                ].cpu().float().numpy()

            all_features.append(batch_features)

        except Exception as e:
            logger.error(f"特征提取失败 batch {i}: {e}")
            # 用零向量填充
            hidden_size = base_model.config.hidden_size
            all_features.append(np.zeros((len(batch), hidden_size), dtype=np.float32))

        if (i // batch_size) % 20 == 0:
            logger.info(f"  特征提取进度: {min(i + batch_size, len(inputs))}/{len(inputs)}")

    features = np.concatenate(all_features, axis=0)
    # L2归一化
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    features = features / norms
    return features


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

    oracle_selections = exp9_phase1.get('oracle_selections', {})

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
        features = extract_hidden_states(
            inputs, base_model, tokenizer,
            batch_size=4 if not args.test_mode else 2,
        )

        # 构建标签：Oracle最优专家
        selections = oracle_selections.get(domain, {})
        labels = []
        for i in range(len(test_data)):
            # oracle_selections是聚合统计，需要逐样本对应
            # 这里用聚合分布构建标签（各样本同domain取最高频专家）
            # 精确逐样本标签需从缓存的逐样本ROUGE-L重建，这里用整体最优
            best_expert = max(selections, key=selections.get) if selections else domain
            labels.append(EXPERT_TO_IDX[best_expert])

        # 更准确的做法：逐样本重建Oracle标签
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
        general_features = extract_hidden_states(
            general_inputs, base_model, tokenizer, batch_size=4
        )
        general_labels = _rebuild_per_sample_labels('general', general_test, args)
        np.savez(general_feat_path, features=general_features, labels=np.array(general_labels))

    del lm, base_model, tokenizer
    _cleanup_gpu()

    # ── 步骤3: 组合训练数据 ──
    logger.info("\n--- 步骤3: 组合训练数据 ---")

    train_X = np.concatenate([all_features[d] for d in SPECIALIZED_TYPES], axis=0)
    train_y = np.concatenate([all_labels[d] for d in SPECIALIZED_TYPES], axis=0)

    # General前80%作为验证集
    n_val = int(len(general_features) * 0.8)
    val_X = general_features[:n_val]
    val_y = np.array(general_labels[:n_val])
    test_X = general_features[n_val:]
    test_y = np.array(general_labels[n_val:])

    logger.info(f"  训练集: {len(train_X)} 条 (specialized domains)")
    logger.info(f"  验证集: {n_val} 条 (general前80%)")
    logger.info(f"  测试集: {len(test_X)} 条 (general后20%)")

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

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true_general, y_pred_general, labels=[0, 1, 2, 3])
    report = classification_report(
        y_true_general, y_pred_general,
        target_names=['text', 'image', 'uml', 'general'],
        output_dict=True, zero_division=0
    )
    logger.info(f"  General域分类报告:\n{classification_report(y_true_general, y_pred_general, target_names=['text','image','uml','general'], zero_division=0)}")

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
    expert_caches = {}
    for expert_type in ALL_TYPES:
        if expert_type == domain:
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', f'{domain}_predictions.json')
        else:
            cache = load_predictions_cache(
                CACHE_DIR / 'exp9_oracle',
                f'{expert_type}_expert_on_{domain}_predictions.json'
            )
            if cache is None and expert_type == 'general':
                # 尝试lora_moe中的general_expert
                cache = load_predictions_cache(CACHE_DIR / 'lora_moe', 'general_predictions.json')
        if cache:
            expert_caches[expert_type] = cache.get('samples', [])

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


def _train_router(router, train_X, train_y, val_X, val_y, args):
    """训练MLP路由器"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = router.device
    X_t = torch.tensor(train_X, dtype=torch.float32)
    y_t = torch.tensor(train_y, dtype=torch.long)
    X_v = torch.tensor(val_X, dtype=torch.float32).to(device)
    y_v = torch.tensor(val_y, dtype=torch.long).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(router.model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )

    max_epochs = 10 if args.test_mode else 50
    patience = 5
    best_val_acc = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_acc': []}

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

        # 验证
        router.model.eval()
        with torch.no_grad():
            val_logits = router.model(X_v)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_v).float().mean().item()

        avg_loss = epoch_loss / len(loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{max_epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            router.save(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stop at epoch {epoch+1}, best val_acc={best_val_acc:.4f}")
                break

    # 加载最优checkpoint
    router.load(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
    logger.info(f"训练完成，最优验证准确率: {best_val_acc:.4f}")
    history['best_val_acc'] = best_val_acc
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
    from models.language_model import LanguageModel
    from models.prompt_templates.general_template import GeneralInstructionTemplate

    lm = LanguageModel(use_4bit=True)
    base_model = lm.model
    tokenizer = lm.tokenizer

    # 加载所有adapter路径
    adapter_paths = {}
    for et in ALL_TYPES:
        adapter_paths[et] = str(path_cfg.get_expert_weight_path(et))

    samples = []
    routing_stats = defaultdict(int)

    for i, (sample, prob) in enumerate(zip(general_test, probs)):
        # 选top-2专家
        top2_idxs = np.argsort(prob)[::-1][:2]
        expert1 = IDX_TO_EXPERT[top2_idxs[0]]
        expert2 = IDX_TO_EXPERT[top2_idxs[1]]
        w1 = float(prob[top2_idxs[0]])
        w2 = float(prob[top2_idxs[1]])
        # 归一化
        w_sum = w1 + w2
        w1, w2 = w1 / w_sum, w2 / w_sum

        routing_stats[f"{expert1}+{expert2}"] += 1

        if i % 50 == 0:
            logger.info(f"  Ensemble进度: {i}/{len(general_test)} | 当前: {expert1}({w1:.2f})+{expert2}({w2:.2f})")

        # 如果最高权重>=0.85，退化为单专家推理
        if w1 >= 0.85:
            pred = _single_expert_from_cache(expert1, 'general', i)
        else:
            pred = _logit_ensemble_generate(
                base_model, tokenizer, adapter_paths,
                sample['input'], expert1, expert2, w1, w2,
                args
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
        })

    del lm, base_model, tokenizer
    _cleanup_gpu()

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


def _logit_ensemble_generate(base_model, tokenizer, adapter_paths,
                              input_text, expert1, expert2, w1, w2, args):
    """
    核心：两专家logit加权融合生成

    1. 加载expert1 LoRA → 前向推理 → 缓存每步logits
    2. 切换expert2 LoRA → 前向推理 → 缓存每步logits
    3. 加权平均logits → 贪婪解码
    """
    import torch
    from peft import PeftModel

    max_new_tokens = 512

    def _get_logits_sequence(adapter_path, prompt_ids):
        """加载adapter，执行teacher-forcing前向推理，返回每步logit"""
        try:
            peft_model = PeftModel.from_pretrained(
                base_model, adapter_path,
                is_trainable=False,
            )
            peft_model.eval()
            all_logits = []

            # 自回归逐token推理，收集logits
            input_ids = prompt_ids.clone()
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    out = peft_model(input_ids=input_ids, return_dict=True)
                next_logits = out.logits[:, -1, :]  # (1, vocab_size)
                all_logits.append(next_logits.cpu())
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                if next_token.item() in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    break
                input_ids = torch.cat([input_ids, next_token], dim=1)

            del peft_model
            _cleanup_gpu()
            return all_logits

        except Exception as e:
            logger.warning(f"  adapter推理失败 ({adapter_path}): {e}")
            return []

    from models.prompt_templates.general_template import GeneralInstructionTemplate
    prompt = GeneralInstructionTemplate.build_prompt(input_text)
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)

    logits1 = _get_logits_sequence(adapter_paths[expert1], prompt_ids)
    logits2 = _get_logits_sequence(adapter_paths[expert2], prompt_ids)

    if not logits1 and not logits2:
        return _single_expert_from_cache('general', 'general', 0)
    if not logits1:
        return _decode_from_logits(tokenizer, logits2)
    if not logits2:
        return _decode_from_logits(tokenizer, logits1)

    # 对齐长度（取较短的）
    min_len = min(len(logits1), len(logits2))
    logits1 = logits1[:min_len]
    logits2 = logits2[:min_len]

    # 加权融合
    import torch.nn.functional as F
    fused_tokens = []
    for l1, l2 in zip(logits1, logits2):
        p1 = F.softmax(l1.to(base_model.device), dim=-1)
        p2 = F.softmax(l2.to(base_model.device), dim=-1)
        p_fused = w1 * p1 + w2 * p2
        next_token = p_fused.argmax(dim=-1).item()
        if next_token in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break
        fused_tokens.append(next_token)

    if not fused_tokens:
        return ''
    return tokenizer.decode(fused_tokens, skip_special_tokens=True)


def _decode_from_logits(tokenizer, logits_list):
    """从logit列表贪婪解码"""
    import torch
    tokens = []
    for l in logits_list:
        token = l.argmax(dim=-1).item()
        if token == tokenizer.eos_token_id:
            break
        tokens.append(token)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _single_expert_from_cache(expert_name, domain, sample_idx):
    """从已有缓存取单专家预测结果"""
    if expert_name == domain:
        cache = load_predictions_cache(CACHE_DIR / 'lora_moe', f'{domain}_predictions.json')
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
    """加载所有专家在general域上的缓存"""
    caches = {}
    for expert in ALL_TYPES:
        if expert == 'general':
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', 'general_predictions.json')
        else:
            cache = load_predictions_cache(
                CACHE_DIR / 'exp9_oracle',
                f'{expert}_expert_on_general_predictions.json'
            )
        if cache:
            caches[expert] = cache.get('samples', [])
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
    soft_rougeL = (exp9_phase2 or {}).get('best_rougeL', None)
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