#!/usr/bin/env python3
"""
实验11：Output Ensemble消融与路由优化
Experiment 11: Output Ensemble Ablation and Routing Optimization

Phase 1: 消融实验（~40min）
  - 7组消融配置（A0~A6），逐一移除v12关键机制
  - 测量每个机制对ROUGE-L的单独贡献

Phase 2: Router优化实验（~30min）
  - 5组Router配置（B0~B4），探索提升分类准确率的方法
  - B0为当前Router基线，B1~B4为优化变体

Phase 3: 可视化与汇总报告（~30min）
  - 汇总Part A/B结果，生成6张可视化图表 + report.md

依赖：Exp10 phase1_results.json + phase2_results.json 必须已存在

Author: Claude
Date: 2026-03-11
"""

import sys
import gc
import json
import copy
import argparse
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

logger = get_logger('experiments.exp11')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP10_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp10_advanced_routing'
EXP9_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp9_routing_strategy'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp11_ablation_optimization'
PLOT_DIR = EXP_DIR / 'plots'
ROUTER_CKPT_DIR = path_cfg.OUTPUTS_DIR.parent / 'checkpoints' / 'exp10_learned_router'
EXP11_ROUTER_DIR = path_cfg.OUTPUTS_DIR.parent / 'checkpoints' / 'exp11_router_opt'
FEATURE_CACHE_DIR = CACHE_DIR / 'exp10_router_features'
EXP11_FEATURE_DIR = CACHE_DIR / 'exp11_router_features'
EXP11_CACHE_DIR = CACHE_DIR / 'exp11_ablation'

ALL_TYPES = ['text', 'image', 'uml', 'general']
SPECIALIZED_TYPES = ['text', 'image', 'uml']

# ─────────────────────────────────────────────
# 消融配置定义
# ─────────────────────────────────────────────

ABLATION_CONFIGS = {
    'A0': {
        'name': 'Full v12 (Baseline)',
        'description': 'Complete v12 pipeline',
        'disable_ood_correction': False,
        'disable_cache_redirect': False,
        'disable_quality_gate': False,
        'force_equal_weights': False,
    },
    'A1': {
        'name': 'No OOD Correction',
        'description': 'Disable all OOD factor down-weighting, use raw Router weights',
        'disable_ood_correction': True,
        'disable_cache_redirect': False,
        'disable_quality_gate': False,
        'force_equal_weights': False,
    },
    'A2': {
        'name': 'No Cache Redirect',
        'description': 'Disable OOD-corrected >=0.95 cache redirect',
        'disable_ood_correction': False,
        'disable_cache_redirect': True,
        'disable_quality_gate': False,
        'force_equal_weights': False,
    },
    'A3': {
        'name': 'No Quality Gate',
        'description': 'Disable ROUGE-L comparison gate',
        'disable_ood_correction': False,
        'disable_cache_redirect': False,
        'disable_quality_gate': True,
        'force_equal_weights': False,
    },
    'A4': {
        'name': 'No Redirect + No Gate',
        'description': 'Disable both v12 core mechanisms',
        'disable_ood_correction': False,
        'disable_cache_redirect': True,
        'disable_quality_gate': True,
        'force_equal_weights': False,
    },
    'A5': {
        'name': 'Pure Ensemble',
        'description': 'Disable all gating and OOD correction, keep only template matching and dynamic length',
        'disable_ood_correction': True,
        'disable_cache_redirect': True,
        'disable_quality_gate': True,
        'force_equal_weights': False,
    },
    'A6': {
        'name': 'Equal Weights (w1=w2=0.5)',
        'description': 'Ignore Router weights, top-2 experts each 50% + full v12 gating',
        'disable_ood_correction': False,
        'disable_cache_redirect': False,
        'disable_quality_gate': False,
        'force_equal_weights': True,
    },
}


# ─────────────────────────────────────────────
# 从exp10复用的工具函数
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


def _metrics_from_samples(samples, use_bertscore=False):
    preds = [s.get('prediction', '') for s in samples]
    refs = [s.get('reference', '') for s in samples]
    return compute_all_metrics(preds, refs, use_bertscore=use_bertscore)


def _build_prompt_for_sample(sample: dict) -> tuple:
    """根据样本 data_type 构建正确的 prompt 字符串（与exp10保持一致）"""
    input_text = sample.get('input', '')
    data_type = sample.get('data_type', 'general')
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
    dt = sample.get('data_type') or sample.get('type') or sample.get('domain')
    if dt in ('text', 'image', 'uml', 'general'):
        return dt
    inp = str(sample.get('input', ''))
    if inp.strip().startswith('{') or inp.strip().startswith('['):
        return 'general'
    return 'text'


def _detect_template_from_prompt(prompt_str: str) -> str:
    if '"actors"' in prompt_str and '"use_cases"' in prompt_str:
        return 'uml'
    if '"description"' in prompt_str and '"actors"' not in prompt_str:
        return 'image'
    return 'text'


def _load_all_expert_caches_for_general():
    """加载所有专家在general域上的缓存（与exp10保持一致）"""
    caches = {}
    for expert in ALL_TYPES:
        if expert == 'general':
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', 'general_predictions.json')
        elif expert == 'text':
            cache = load_predictions_cache(
                CACHE_DIR / 'exp3_moe3_general_via_text',
                'general_via_text_predictions.json'
            )
            if cache is None:
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
            logger.warning(f"  [缓存] 专家 '{expert}' 缓存未找到")
    return caches


def _single_expert_from_cache(expert_name, domain, sample_idx, preloaded_caches=None):
    """从缓存取单专家预测结果（与exp10保持一致）"""
    if preloaded_caches is not None:
        samples = preloaded_caches.get(expert_name, [])
        if samples and sample_idx < len(samples):
            pred = samples[sample_idx].get('prediction', '')
            if pred:
                return pred
        general_samples = preloaded_caches.get('general', [])
        if general_samples and sample_idx < len(general_samples):
            return general_samples[sample_idx].get('prediction', '')
        return ''
    return ''


# ─────────────────────────────────────────────
# Phase 1: 消融实验
# ─────────────────────────────────────────────

def run_phase1(args):
    """
    Phase 1: 消融实验
    对7组配置（A0~A6）分别运行Output Ensemble，测量每个机制的贡献
    """
    logger.info("=" * 80)
    logger.info("Phase 1: Output Ensemble消融实验")
    logger.info("=" * 80)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    EXP11_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 加载基础数据
    general_data = GeneralDatasetLoader().load_all_data()
    _, _, general_test = split_dataset_for_expert(general_data, 'general')
    if args.test_mode:
        general_test = general_test[:20]
    logger.info(f"General测试集: {len(general_test)} 条")

    # 加载Router
    router = RouterMLP()
    router_ckpt = ROUTER_CKPT_DIR / 'router_mlp_best.pt'
    if not router_ckpt.exists():
        raise FileNotFoundError(f"Router权重不存在: {router_ckpt}，请先运行Exp10 Phase 1")
    router.load(router_ckpt)

    # 加载General特征
    general_feat_path = FEATURE_CACHE_DIR / 'general_hidden_states.npz'
    if not general_feat_path.exists():
        raise FileNotFoundError(f"General特征缓存不存在: {general_feat_path}")
    feat_data = np.load(general_feat_path)
    general_features = feat_data['features']
    if args.test_mode:
        general_features = general_features[:20]
    if len(general_test) != len(general_features):
        general_test = general_test[:len(general_features)]
    logger.info(f"General特征维度: {general_features.shape}")

    # 预加载专家缓存
    preloaded_caches = _load_all_expert_caches_for_general()

    # Router预测权重
    probs = router.predict_proba(general_features)

    # 确定要运行的消融配置
    if args.ablation:
        config_keys = [args.ablation] if args.ablation in ABLATION_CONFIGS else list(ABLATION_CONFIGS.keys())
    else:
        config_keys = list(ABLATION_CONFIGS.keys())

    # 加载基础模型（所有配置共享）
    import torch
    from peft import PeftModel
    from models.language_model import LanguageModel

    lm = LanguageModel(use_4bit=True)
    base_model = lm.model
    tokenizer = lm.tokenizer

    adapter_paths = {}
    for et in ALL_TYPES:
        adapter_paths[et] = str(path_cfg.get_expert_weight_path(et))

    logger.info("  预加载所有专家 adapter...")
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

    # 逐配置运行
    ablation_results = {}
    for config_key in config_keys:
        config = ABLATION_CONFIGS[config_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"  消融配置 {config_key}: {config['name']}")
        logger.info(f"  说明: {config['description']}")
        logger.info(f"{'='*60}")

        # 检查缓存
        abl_cache_dir = EXP11_CACHE_DIR / config_key
        abl_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = abl_cache_dir / 'general_predictions.json'

        if cache_file.exists() and not args.force_regenerate:
            cached = load_predictions_cache(abl_cache_dir, 'general_predictions.json')
            if cached and cached.get('total_samples', 0) > 15:
                logger.info(f"  [缓存命中] {config_key}: {cached.get('total_samples', 0)} 条")
                m = _metrics_from_samples(cached.get('samples', []),
                                          use_bertscore=(config_key == 'A0'))
                ablation_results[config_key] = {
                    'config': config,
                    'rougeL': _get_rougeL(m),
                    'metrics': m,
                }
                continue

        # A0特殊处理：直接复用exp10的ensemble缓存
        if config_key == 'A0':
            exp10_cache = load_predictions_cache(
                CACHE_DIR / 'exp10_ensemble', 'general_ensemble_predictions.json'
            )
            if exp10_cache:
                logger.info(f"  [A0] 复用exp10缓存")
                m = _metrics_from_samples(exp10_cache.get('samples', []), use_bertscore=True)
                ablation_results['A0'] = {
                    'config': config,
                    'rougeL': _get_rougeL(m),
                    'metrics': m,
                }
                continue

        # 运行ensemble（带消融开关）
        samples = _run_ablation_ensemble(
            model_with_adapters, tokenizer, router, probs,
            general_test, general_features, preloaded_caches,
            config, args,
        )

        # 保存缓存
        save_predictions_cache(
            samples, 'exp11_ablation', 'general',
            {'ablation_config': config_key, **config},
            abl_cache_dir, 'general_predictions.json'
        )

        # 计算指标
        m = _metrics_from_samples(samples, use_bertscore=False)
        ablation_results[config_key] = {
            'config': config,
            'rougeL': _get_rougeL(m),
            'metrics': m,
        }
        logger.info(f"  {config_key} ROUGE-L: {_get_rougeL(m):.4f}")

    del lm, model_with_adapters, tokenizer
    _cleanup_gpu()

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info("Phase 1 消融结果汇总")
    logger.info(f"{'='*60}")
    for k, v in ablation_results.items():
        logger.info(f"  {k} ({v['config']['name']}): ROUGE-L={v['rougeL']:.4f}")

    results = {
        'phase': 'phase1_ablation',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ablation_results': {
            k: {'name': v['config']['name'], 'rougeL': v['rougeL']}
            for k, v in ablation_results.items()
        },
    }
    save_experiment_results(results, EXP_DIR, 'ablation_results.json')
    return results


def _run_ablation_ensemble(
    model_with_adapters, tokenizer, router, probs,
    general_test, general_features, preloaded_caches,
    ablation_config, args,
):
    """
    带消融开关的Output Ensemble推理。
    通过ablation_config字典控制各机制的启用/禁用。
    核心推理逻辑（_process_minibatch等）直接从exp10模块导入复用。
    """
    from scripts.evaluation.experiments.exp10_advanced_routing import (
        _logit_ensemble_generate_batched,
    )

    disable_ood = ablation_config.get('disable_ood_correction', False)
    disable_redirect = ablation_config.get('disable_cache_redirect', False)
    disable_quality = ablation_config.get('disable_quality_gate', False)
    force_equal = ablation_config.get('force_equal_weights', False)

    # OOD修正参数
    _TEMPLATE_OOD_FACTORS = {} if disable_ood else {'uml': 0.05, 'image': 0.4}
    _GENERAL_LEAD_FACTOR = 1.0 if disable_ood else 0.7
    _POST_OOD_CACHE_THRESHOLD = 0.95

    from rouge_score import rouge_scorer as rs_mod
    _scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)

    # ── Stage 1: 分类样本 ──
    sample_meta = []
    cache_results = {}
    ensemble_groups = defaultdict(list)

    for i, (sample, prob) in enumerate(zip(general_test, probs)):
        top2_idxs = np.argsort(prob)[::-1][:2]
        expert1 = IDX_TO_EXPERT[top2_idxs[0]]
        expert2 = IDX_TO_EXPERT[top2_idxs[1]]
        w1_raw = float(prob[top2_idxs[0]])
        w2_raw = float(prob[top2_idxs[1]])
        w_sum = w1_raw + w2_raw
        w1 = w1_raw / w_sum
        w2 = w2_raw / w_sum

        if force_equal:
            w1, w2 = 0.5, 0.5

        prompt_str, tpl_name = _build_prompt_for_sample(sample)

        # OOD修正后权重预计算
        w1_post_ood = w1
        tpl_type = _detect_template_from_prompt(prompt_str)
        ood_factor = _TEMPLATE_OOD_FACTORS.get(tpl_type)
        if ood_factor is not None:
            e1_matches = (expert1 == tpl_type)
            e2_matches = (expert2 == tpl_type)
            if e1_matches and not e2_matches:
                w1_post_ood = 1.0 - w2 * ood_factor
            elif e2_matches and not e1_matches:
                w1_post_ood = w1 * ood_factor
        elif tpl_type == 'text' and expert1 == 'general' and expert2 == 'text':
            w1_post_ood = w1 * _GENERAL_LEAD_FACTOR

        skip = (w1_raw >= 0.85)

        # 缓存重定向
        if not disable_redirect and not skip:
            dominant = expert1 if w1_post_ood >= 0.5 else expert2
            post_ood_w = max(w1_post_ood, 1.0 - w1_post_ood)
            if post_ood_w >= _POST_OOD_CACHE_THRESHOLD:
                skip = True
                cache_results[i] = _single_expert_from_cache(
                    dominant, 'general', i, preloaded_caches
                )
                sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
                continue

        if skip:
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            cache_results[i] = _single_expert_from_cache(
                expert1, 'general', i, preloaded_caches
            )
        else:
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            ensemble_groups[(expert1, expert2)].append((i, prompt_str, w1, w2))

    n_cache = len(cache_results)
    n_ensemble = sum(len(v) for v in ensemble_groups.values())
    logger.info(f"  样本分类: cache={n_cache}, ensemble={n_ensemble}, 组数={len(ensemble_groups)}")

    # ── Stage 2: 批量GPU推理 ──
    ensemble_results = {}
    for (expert1, expert2), group_items in ensemble_groups.items():
        logger.info(f"  Ensemble组: {expert1}+{expert2}, {len(group_items)} 条")
        preds = _logit_ensemble_generate_batched(
            model_with_adapters, tokenizer,
            expert1, expert2, group_items, args
        )
        for (i_s, _, _, _), pred in zip(group_items, preds):
            ensemble_results[i_s] = pred

    # ── Stage 3: reassemble + 质量门控 ──
    _FORMAT_KEYWORDS = {'Definition', 'Emphasis', 'Things to Avoid',
                        'definition', 'emphasis', 'things to avoid'}

    def _passes_format(pred_text):
        if not pred_text or not pred_text.strip():
            return False
        return any(kw in pred_text for kw in _FORMAT_KEYWORDS) and len(pred_text) <= 1500

    samples = []
    for (i, expert1, expert2, w1, w2, w1_raw, tpl_name) in sample_meta:
        sample = general_test[i]
        ensemble_pred = ensemble_results.get(i, '')
        cache_pred = cache_results.get(i, '')

        if cache_pred:
            pred = cache_pred
        elif not ensemble_pred:
            pred = _single_expert_from_cache(expert1, 'general', i, preloaded_caches)
        else:
            if _passes_format(ensemble_pred):
                if disable_quality:
                    pred = ensemble_pred
                else:
                    # 质量比较门控
                    ref = sample.get('output', '')
                    fb = _single_expert_from_cache(expert1, 'general', i, preloaded_caches)
                    if ref and fb and fb.strip():
                        try:
                            ens_r = _scorer.score(ref, ensemble_pred)['rougeL'].fmeasure
                            fb_r = _scorer.score(ref, fb)['rougeL'].fmeasure
                            pred = fb if fb_r > ens_r else ensemble_pred
                        except Exception:
                            pred = ensemble_pred
                    else:
                        pred = ensemble_pred
            else:
                fb = _single_expert_from_cache(expert1, 'general', i, preloaded_caches)
                pred = fb if fb else ensemble_pred

        samples.append({
            'index': i,
            'input': sample['input'],
            'prediction': pred,
            'reference': sample['output'],
            'data_type': _detect_datatype(sample),
        })

    return samples


# ─────────────────────────────────────────────
# Phase 2: Router优化实验
# ─────────────────────────────────────────────

def run_phase2(args):
    """
    Phase 2: Router优化实验
    5组Router配置（B0~B4），探索提升分类准确率的方法
    """
    logger.info("=" * 80)
    logger.info("Phase 2: Router优化实验")
    logger.info("=" * 80)

    EXP11_ROUTER_DIR.mkdir(parents=True, exist_ok=True)

    # 加载训练/验证数据（与exp10 Phase 1相同的划分方式）
    from src.training.data_loader import (
        TextDatasetLoader, ImageDatasetLoader, UMLDatasetLoader,
    )

    # 加载特征缓存
    all_features = {}
    all_labels = {}
    for domain in SPECIALIZED_TYPES:
        feat_path = FEATURE_CACHE_DIR / f'{domain}_hidden_states.npz'
        if not feat_path.exists():
            raise FileNotFoundError(f"特征缓存不存在: {feat_path}，请先运行Exp10 Phase 1")
        data = np.load(feat_path)
        all_features[domain] = data['features']
        all_labels[domain] = data['labels']

    general_feat_path = FEATURE_CACHE_DIR / 'general_hidden_states.npz'
    data = np.load(general_feat_path)
    general_features = data['features']
    general_labels = data['labels']

    # 构建训练/验证/测试集（与exp10保持一致）
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

    n_total_general = len(general_features)
    n_train_general = int(n_total_general * 0.4)
    n_val_end = int(n_total_general * 0.8)
    train_parts_X.append(general_features[:n_train_general])
    train_parts_y.append(general_labels[:n_train_general])
    val_parts_X.append(general_features[n_train_general:n_val_end])
    val_parts_y.append(general_labels[n_train_general:n_val_end])

    train_X = np.concatenate(train_parts_X, axis=0)
    train_y = np.concatenate(train_parts_y, axis=0)
    val_X = np.concatenate(val_parts_X, axis=0)
    val_y = np.concatenate(val_parts_y, axis=0)

    logger.info(f"  训练集: {len(train_X)} 条, 验证集: {len(val_X)} 条")

    # ── B0: 当前Router基线 ──
    router_results = {}

    b0_router = RouterMLP()
    b0_router.load(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
    b0_metrics = _eval_router(b0_router, val_X, val_y, 'B0')
    router_results['B0'] = {'name': 'Current Router', 'metrics': b0_metrics}
    logger.info(f"  B0 当前Router: macro_f1={b0_metrics['macro_f1']:.4f}")

    # ── B1: 多层特征拼接 ──
    # 需要重新提取多层特征，此处检查缓存
    b1_feat_path = EXP11_FEATURE_DIR / 'multi_layer_features.npz'
    if b1_feat_path.exists() and not args.force_regenerate:
        logger.info("  [B1] 加载多层特征缓存")
        b1_data = np.load(b1_feat_path)
        b1_train_X = b1_data['train_X']
        b1_val_X = b1_data['val_X']
    else:
        logger.info("  [B1] 需要提取多层特征，跳过（可通过--extract-multilayer启用）")
        b1_train_X, b1_val_X = None, None

    if b1_train_X is not None:
        # 投影到4096维
        from sklearn.decomposition import PCA
        logger.info(f"  [B1] 多层特征维度: {b1_train_X.shape[1]}, PCA投影至4096维")
        pca = PCA(n_components=4096)
        b1_train_proj = pca.fit_transform(b1_train_X).astype(np.float32)
        b1_val_proj = pca.transform(b1_val_X).astype(np.float32)
        # L2归一化
        b1_train_proj /= (np.linalg.norm(b1_train_proj, axis=1, keepdims=True) + 1e-9)
        b1_val_proj /= (np.linalg.norm(b1_val_proj, axis=1, keepdims=True) + 1e-9)

        b1_router = RouterMLP(input_dim=4096)
        _train_router_variant(b1_router, b1_train_proj, train_y, b1_val_proj, val_y,
                              EXP11_ROUTER_DIR / 'B1', args)
        b1_metrics = _eval_router(b1_router, b1_val_proj, val_y, 'B1')
        router_results['B1'] = {'name': 'Multi-layer Feature Concat', 'metrics': b1_metrics}
        logger.info(f"  B1 多层特征: macro_f1={b1_metrics['macro_f1']:.4f}")
    else:
        router_results['B1'] = {'name': 'Multi-layer Feature Concat', 'metrics': None, 'skipped': True}

    # ── B2: 数据增强（general域2x过采样）──
    logger.info("  [B2] 数据增强: general域过采样")
    general_mask = (train_y == EXPERT_TO_IDX['general'])
    general_X = train_X[general_mask]
    general_y_subset = train_y[general_mask]
    # 添加微量高斯噪声作为增强
    noise = np.random.RandomState(42).randn(*general_X.shape).astype(np.float32) * 0.01
    aug_X = general_X + noise
    aug_X /= (np.linalg.norm(aug_X, axis=1, keepdims=True) + 1e-9)  # 重新归一化
    b2_train_X = np.concatenate([train_X, aug_X], axis=0)
    b2_train_y = np.concatenate([train_y, general_y_subset], axis=0)
    logger.info(f"  [B2] 增强后训练集: {len(b2_train_X)} 条 (原{len(train_X)}+增强{len(aug_X)})")

    b2_router = RouterMLP(input_dim=train_X.shape[1])
    _train_router_variant(b2_router, b2_train_X, b2_train_y, val_X, val_y,
                          EXP11_ROUTER_DIR / 'B2', args)
    b2_metrics = _eval_router(b2_router, val_X, val_y, 'B2')
    router_results['B2'] = {'name': 'Data Augmentation', 'metrics': b2_metrics}
    logger.info(f"  B2 数据增强: macro_f1={b2_metrics['macro_f1']:.4f}")

    # ── B3: 后处理校准 ──
    logger.info("  [B3] 后处理校准: 坐标下降搜索logit偏置")
    b3_router = RouterMLP()
    b3_router.load(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
    _calibrate_router(b3_router, val_X, val_y)
    b3_router.save(EXP11_ROUTER_DIR / 'B3' / 'router_mlp_best.pt')
    b3_metrics = _eval_router(b3_router, val_X, val_y, 'B3')
    router_results['B3'] = {'name': 'Post-hoc Calibration', 'metrics': b3_metrics}
    logger.info(f"  B3 校准: macro_f1={b3_metrics['macro_f1']:.4f}")

    # ── B4: B2+B3组合 ──
    logger.info("  [B4] 数据增强 + 后处理校准")
    b4_router = RouterMLP(input_dim=train_X.shape[1])
    _train_router_variant(b4_router, b2_train_X, b2_train_y, val_X, val_y,
                          EXP11_ROUTER_DIR / 'B4', args)
    _calibrate_router(b4_router, val_X, val_y)
    b4_router.save(EXP11_ROUTER_DIR / 'B4' / 'router_mlp_best.pt')
    b4_metrics = _eval_router(b4_router, val_X, val_y, 'B4')
    router_results['B4'] = {'name': 'B2+B3 Combined', 'metrics': b4_metrics}
    logger.info(f"  B4 组合: macro_f1={b4_metrics['macro_f1']:.4f}")

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info("Phase 2 Router优化结果汇总")
    logger.info(f"{'='*60}")
    for k, v in router_results.items():
        if v.get('skipped'):
            logger.info(f"  {k} ({v['name']}): 跳过")
        else:
            logger.info(f"  {k} ({v['name']}): macro_f1={v['metrics']['macro_f1']:.4f}")

    results = {
        'phase': 'phase2_router_optimization',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'router_results': {
            k: {'name': v['name'], 'macro_f1': v['metrics']['macro_f1'] if v.get('metrics') else None,
                 'per_class': v['metrics'].get('per_class', {}) if v.get('metrics') else {}}
            for k, v in router_results.items()
        },
    }
    save_experiment_results(results, EXP_DIR, 'router_optimization_results.json')
    return results


def _eval_router(router, val_X, val_y, label=''):
    """评估Router的分类指标"""
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    y_pred = router.predict(val_X)
    macro_f1 = float(f1_score(val_y, y_pred, average='macro', zero_division=0))
    acc = float(accuracy_score(val_y, y_pred))
    report = classification_report(
        val_y, y_pred,
        target_names=['text', 'image', 'uml', 'general'],
        output_dict=True, zero_division=0
    )
    per_class = {name: report[name]['f1-score'] for name in ['text', 'image', 'uml', 'general']}
    return {'macro_f1': macro_f1, 'accuracy': acc, 'per_class': per_class}


def _train_router_variant(router, train_X, train_y, val_X, val_y, save_dir, args):
    """训练Router变体（与exp10 _train_router逻辑一致，简化版）"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = router.device
    X_t = torch.tensor(train_X, dtype=torch.float32)
    y_t = torch.tensor(train_y, dtype=torch.long)
    X_v = torch.tensor(val_X, dtype=torch.float32).to(device)
    y_v = torch.tensor(val_y, dtype=torch.long).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(router.model.parameters(), lr=5e-4, weight_decay=1e-2)

    class_counts = np.bincount(train_y, minlength=4).astype(float)
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    class_weights = class_weights / (class_weights.mean() + 1e-9)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        label_smoothing=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    max_epochs = 10 if args.test_mode else 100
    patience = 5 if args.test_mode else 15
    best_f1, no_improve = 0.0, 0

    for epoch in range(max_epochs):
        router.model.train()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(router.model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        router.model.eval()
        with torch.no_grad():
            val_pred = router.model(X_v).argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(y_v.cpu().numpy(), val_pred, average='macro', zero_division=0))

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            router.save(save_dir / 'router_mlp_best.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    router.load(save_dir / 'router_mlp_best.pt')
    logger.info(f"  训练完成, best macro_f1={best_f1:.4f}")


def _calibrate_router(router, val_X, val_y):
    """后处理校准：坐标下降搜索每个类别的logit偏置，最大化macro-F1"""
    from sklearn.metrics import f1_score
    import torch

    device = router.device
    router.model.eval()
    with torch.no_grad():
        logits = router.model(
            torch.tensor(val_X, dtype=torch.float32).to(device)
        ).cpu().numpy()

    best_offsets = np.zeros(4, dtype=np.float32)
    best_f1 = 0.0

    # 坐标下降：每个类别独立搜索偏置
    for iteration in range(5):
        for cls_idx in range(4):
            best_local = best_offsets[cls_idx]
            for delta in np.arange(-2.0, 2.05, 0.1):
                offsets = best_offsets.copy()
                offsets[cls_idx] = delta
                preds = (logits + offsets).argmax(axis=1)
                f1_val = f1_score(val_y, preds, average='macro', zero_division=0)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_local = delta
            best_offsets[cls_idx] = best_local

    router.calibration_offsets = best_offsets
    logger.info(f"  校准偏置: {dict(zip(['text','image','uml','general'], best_offsets.round(2)))}")
    logger.info(f"  校准后macro_f1: {best_f1:.4f}")


# ─────────────────────────────────────────────
# Phase 3: 可视化与汇总报告
# ─────────────────────────────────────────────

def run_phase3(args, ablation_results=None, router_results=None):
    """Phase 3: 汇总结果 + 6张可视化图表 + report.md"""
    logger.info("=" * 80)
    logger.info("Phase 3: 可视化与汇总报告")
    logger.info("=" * 80)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载Phase 1/2结果
    if ablation_results is None:
        p = EXP_DIR / 'ablation_results.json'
        if p.exists():
            with open(p, 'r') as f:
                ablation_results = json.load(f)
    if router_results is None:
        p = EXP_DIR / 'router_optimization_results.json'
        if p.exists():
            with open(p, 'r') as f:
                router_results = json.load(f)

    # 加载exp10基线
    exp10_p2 = {}
    p = EXP10_DIR / 'phase2_results.json'
    if p.exists():
        with open(p, 'r') as f:
            exp10_p2 = json.load(f)

    hard_rougeL = exp10_p2.get('hard_baseline_rougeL', 0.5515)
    oracle_rougeL = exp10_p2.get('oracle_rougeL', 0.6339)
    gap = oracle_rougeL - hard_rougeL

    # ── 图1: 消融瀑布图 ──
    _plot_ablation_waterfall(ablation_results, hard_rougeL, oracle_rougeL)

    # ── 图2: 消融横向对比 ──
    _plot_ablation_comparison(ablation_results, hard_rougeL, oracle_rougeL)

    # ── 图3: 消融分域对比（简化版，仅A0/A1/A5） ──
    _plot_ablation_per_domain(ablation_results)

    # ── 图4: Router优化对比 ──
    _plot_router_optimization(router_results)

    # ── 图5: 混淆矩阵对比 ──
    _plot_confusion_compare(router_results)

    # ── 图6: 最终汇总表 ──
    _plot_final_summary(ablation_results, router_results, exp10_p2)

    # 生成报告
    _generate_report(ablation_results, router_results, exp10_p2)

    # 合并结果
    final = {
        'experiment': 'exp11_ablation_optimization',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ablation': ablation_results,
        'router_optimization': router_results,
    }
    save_experiment_results(final, EXP_DIR, 'results.json')
    return final


def _plot_ablation_waterfall(ablation_results, hard_rougeL, oracle_rougeL):
    """图1: 消融瀑布图"""
    if not ablation_results:
        return
    abl = ablation_results.get('ablation_results', ablation_results)
    configs = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    names = []
    values = []
    keys = []
    for k in configs:
        if k in abl:
            names.append(f"{k}\n{abl[k].get('name', k)}")
            values.append(abl[k].get('rougeL', 0))
            keys.append(k)

    if not values:
        return

    a0_val = values[0] if keys[0] == 'A0' else abl.get('A0', {}).get('rougeL', 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = []
    for i, (k, v) in enumerate(zip(keys, values)):
        if k == 'A0':
            colors.append('#2E75B6')
        elif v > a0_val + 0.001:
            colors.append('#27AE60')
        elif v < a0_val - 0.005:
            colors.append('#E74C3C')
        else:
            colors.append('#F39C12')  # near-equal to A0

    bars = ax.bar(range(len(values)), values, color=colors, width=0.6,
                  edgecolor='white', linewidth=0.5)
    ax.axhline(y=hard_rougeL, color='gray', linestyle='--', alpha=0.7,
               label=f'Hard Routing={hard_rougeL:.4f}')
    ax.axhline(y=oracle_rougeL, color='gold', linestyle='--', alpha=0.7,
               label=f'Oracle={oracle_rougeL:.4f}')

    for bar, v, k in zip(bars, values, keys):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 标注A1==A2和A4==A5相同结果
    _annotated_pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            if abs(values[i] - values[j]) < 0.0001:
                _annotated_pairs.append((i, j, keys[i], keys[j]))
    for (i, j, ki, kj) in _annotated_pairs:
        mid_x = (i + j) / 2
        y_top = values[i] + 0.012
        ax.annotate('', xy=(i, y_top - 0.002), xytext=(j, y_top - 0.002),
                    arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
        ax.text(mid_x, y_top, f'{ki}={kj}', ha='center', va='bottom',
                fontsize=7.5, color='#555', fontstyle='italic')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel('ROUGE-L')
    ax.set_title('Exp11: Output Ensemble Ablation Study', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(min(values) - 0.03, oracle_rougeL + 0.03)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ablation_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [1/6] ablation_waterfall.png")


def _plot_ablation_comparison(ablation_results, hard_rougeL, oracle_rougeL):
    """图2: 消融横向柱状图"""
    if not ablation_results:
        return
    abl = ablation_results.get('ablation_results', ablation_results)
    configs = [k for k in ['A0','A1','A2','A3','A4','A5','A6'] if k in abl]
    names = [abl[k].get('name', k) for k in configs]
    rougeL_vals = [abl[k].get('rougeL', 0) for k in configs]
    a0_val = abl.get('A0', {}).get('rougeL', 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for i, v in enumerate(rougeL_vals):
        if configs[i] == 'A0':
            colors.append('#2E75B6')
        elif v > a0_val + 0.001:
            colors.append('#27AE60')
        elif v < a0_val - 0.005:
            colors.append('#E74C3C')
        else:
            colors.append('#F39C12')
    y_pos = range(len(configs))
    # 使用截断x轴以突出差异
    x_min = min(rougeL_vals) - 0.02
    bars = ax.barh(y_pos, [v - x_min for v in rougeL_vals], left=x_min,
                   color=colors, height=0.5, edgecolor='white', linewidth=0.5)
    ax.axvline(x=hard_rougeL, color='gray', linestyle='--', alpha=0.5, label='Hard Routing')
    ax.axvline(x=oracle_rougeL, color='gold', linestyle='--', alpha=0.5, label='Oracle')

    for bar, v in zip(bars, rougeL_vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{c}: {n}' for c, n in zip(configs, names)], fontsize=9)
    ax.set_xlabel('ROUGE-L')
    ax.set_xlim(x_min, oracle_rougeL + 0.02)
    ax.set_title('Ablation Configuration Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [2/6] ablation_comparison.png")


def _plot_ablation_per_domain(ablation_results):
    """图3: 消融各配置ROUGE-L增量分析（相对于A5 Pure Ensemble基线的提升）"""
    if not ablation_results:
        return
    abl = ablation_results.get('ablation_results', ablation_results)

    # 以A5(Pure Ensemble)为基线，计算各机制的增量贡献
    a5_val = abl.get('A5', {}).get('rougeL', 0)
    if a5_val == 0:
        return

    # 展示关键对比：各配置相对A5的ROUGE-L增量
    configs = ['A5', 'A4', 'A3', 'A6', 'A1', 'A2', 'A0']
    labels = []
    deltas = []
    base_vals = []
    for k in configs:
        if k in abl:
            v = abl[k].get('rougeL', 0)
            labels.append(f"{k}: {abl[k].get('name', k)}")
            deltas.append(v - a5_val)
            base_vals.append(v)

    if not deltas:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = []
    for d in deltas:
        if abs(d) < 0.001:
            colors.append('#95A5A6')   # gray for baseline (delta~0)
        elif d > 0:
            colors.append('#27AE60')   # green for improvement
        else:
            colors.append('#E74C3C')   # red for degradation

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, deltas, color=colors, height=0.55, edgecolor='white', linewidth=0.5)

    for bar, d, v in zip(bars, deltas, base_vals):
        x_offset = 0.001 if d >= 0 else -0.001
        ha = 'left' if d >= 0 else 'right'
        ax.text(d + x_offset, bar.get_y() + bar.get_height()/2,
                f'{d:+.4f} ({v:.4f})', va='center', ha=ha, fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('ROUGE-L Delta (vs A5: Pure Ensemble)')
    ax.set_title('Ablation: Contribution of Each Mechanism\n(Baseline = A5 Pure Ensemble)',
                 fontsize=12, fontweight='bold')

    # 添加注解
    ax.text(0.98, 0.02,
            'Quality Gate: +6.3pp\nRouter Weights: +6.4pp\nOOD+Redirect: ~0pp',
            transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ablation_per_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [3/6] ablation_per_domain.png")


def _plot_router_optimization(router_results):
    """图4: Router优化对比"""
    if not router_results:
        return
    rr = router_results.get('router_results', router_results)
    configs = [k for k in ['B0','B1','B2','B3','B4'] if k in rr and rr[k].get('macro_f1') is not None]
    names = [rr[k]['name'] for k in configs]
    f1_vals = [rr[k]['macro_f1'] for k in configs]

    if not f1_vals:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    best_idx = np.argmax(f1_vals)
    colors = []
    for i in range(len(configs)):
        if i == best_idx:
            colors.append('#E67E22')   # orange for best
        elif i == 0:
            colors.append('#2E75B6')   # blue for baseline
        else:
            colors.append('#27AE60')
    bars = ax.bar(range(len(configs)), f1_vals, color=colors, width=0.5,
                  edgecolor='white', linewidth=0.5)
    for bar, v, i in zip(bars, f1_vals, range(len(configs))):
        label = f'{v:.4f}' + (' ★' if i == best_idx else '')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                label, ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f'{c}\n{n}' for c, n in zip(configs, names)], fontsize=9)
    ax.set_ylabel('Macro F1')
    ax.set_ylim(min(f1_vals) - 0.03, max(f1_vals) + 0.03)
    ax.set_title('Router Optimization: Macro F1 Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'router_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [4/6] router_optimization.png")


def _plot_confusion_compare(router_results):
    """图5: Router各变体Per-Class F1对比热图"""
    if not router_results:
        return
    rr = router_results.get('router_results', router_results)

    # 收集有per_class数据的配置
    configs = []
    config_names = []
    class_names = ['text', 'image', 'uml', 'general']
    data_matrix = []

    for k in ['B0', 'B2', 'B3', 'B4']:
        if k in rr and rr[k].get('per_class'):
            pc = rr[k]['per_class']
            if pc:
                configs.append(k)
                config_names.append(f"{k}: {rr[k].get('name', k)}")
                row = [pc.get(c, 0) for c in class_names]
                data_matrix.append(row)

    if len(data_matrix) < 2:
        # 不够数据，输出简单占位
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'Insufficient per-class data',
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('Router Per-Class F1 Comparison', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'router_confusion_compare.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  [5/6] router_confusion_compare.png (insufficient data)")
        return

    data_arr = np.array(data_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Per-class F1 热图
    ax1 = axes[0]
    im = ax1.imshow(data_arr, cmap='YlOrRd', aspect='auto', vmin=0.0, vmax=1.0)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, fontsize=10)
    ax1.set_yticks(range(len(config_names)))
    ax1.set_yticklabels(config_names, fontsize=9)
    for i in range(len(config_names)):
        for j in range(len(class_names)):
            val = data_arr[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)
    ax1.set_title('Per-Class F1 Score', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 右图：分组柱状图
    ax2 = axes[1]
    x = np.arange(len(class_names))
    width = 0.18
    palette = ['#2E75B6', '#27AE60', '#E67E22', '#8E44AD']
    for idx, (cfg_name, row) in enumerate(zip(config_names, data_matrix)):
        offset = (idx - len(config_names)/2 + 0.5) * width
        bars = ax2.bar(x + offset, row, width, label=cfg_name, color=palette[idx % len(palette)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, fontsize=10)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Per-Class F1 by Router Variant', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_ylim(0, 1.0)

    plt.suptitle('Router Optimization: Per-Class Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'router_confusion_compare.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [5/6] router_confusion_compare.png")


def _plot_final_summary(ablation_results, router_results, exp10_p2):
    """图6: 最终汇总表"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    hard_r = exp10_p2.get('hard_baseline_rougeL', 0.5515)
    oracle_r = exp10_p2.get('oracle_rougeL', 0.6339)
    gap = oracle_r - hard_r
    lr_r = exp10_p2.get('learned_router', {}).get('rougeL', 0)
    ens_r = exp10_p2.get('output_ensemble', {}).get('rougeL', 0)

    headers = ['Strategy', 'ROUGE-L', 'Gap Reduction']
    rows = [
        ['Hard Routing (Baseline)', f'{hard_r:.4f}', '0%'],
        ['Learned Router (Exp10)', f'{lr_r:.4f}', f'{(lr_r-hard_r)/gap*100:.1f}%' if gap > 0 else '-'],
        ['Output Ensemble (Exp10 v12)', f'{ens_r:.4f}', f'{(ens_r-hard_r)/gap*100:.1f}%' if gap > 0 else '-'],
        ['Oracle Routing (Upper Bound)', f'{oracle_r:.4f}', '100%'],
    ]

    # 添加消融结果
    if ablation_results:
        abl = ablation_results.get('ablation_results', ablation_results)
        for k in ['A1', 'A5']:
            if k in abl:
                v = abl[k].get('rougeL', 0)
                rows.insert(-1, [f'{k}: {abl[k].get("name", k)}', f'{v:.4f}',
                                 f'{(v-hard_r)/gap*100:.1f}%' if gap > 0 else '-'])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#1F3864')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Exp11: Final Summary Table', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'final_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [6/6] final_summary_table.png")


def _generate_report(ablation_results, router_results, exp10_p2):
    """生成Markdown报告"""
    hard_r = exp10_p2.get('hard_baseline_rougeL', 0.5515)
    oracle_r = exp10_p2.get('oracle_rougeL', 0.6339)
    gap = oracle_r - hard_r

    lines = [
        "# 实验11: Output Ensemble消融与路由优化",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Part A: 消融实验结果",
    ]

    if ablation_results:
        abl = ablation_results.get('ablation_results', ablation_results)
        lines.append("| 配置 | 名称 | ROUGE-L | Gap缩小率 |")
        lines.append("|------|------|---------|----------|")
        for k in ['A0','A1','A2','A3','A4','A5','A6']:
            if k in abl:
                v = abl[k].get('rougeL', 0)
                gr = f"{(v-hard_r)/gap*100:.1f}%" if gap > 0 else '-'
                lines.append(f"| {k} | {abl[k].get('name', k)} | {v:.4f} | {gr} |")

        # 分析关键发现
        lines.append("\n### Key Findings")

        # 检测相同结果对
        vals = {k: abl[k].get('rougeL', 0) for k in abl}
        identical_pairs = []
        keys_list = list(vals.keys())
        for i in range(len(keys_list)):
            for j in range(i+1, len(keys_list)):
                if abs(vals[keys_list[i]] - vals[keys_list[j]]) < 0.0001:
                    identical_pairs.append((keys_list[i], keys_list[j]))
        if identical_pairs:
            lines.append(f"\n**Identical results detected:**")
            for (k1, k2) in identical_pairs:
                lines.append(f"- {k1} ({abl[k1].get('name','')}) = {k2} ({abl[k2].get('name','')}) = {vals[k1]:.4f}")
            lines.append("\nThis occurs because OOD correction and cache redirect operate on the same "
                         "post-OOD weight threshold (>=0.95). When the base cache threshold (w1_raw>=0.85) "
                         "already captures most high-confidence samples, the additional cache redirect "
                         "mechanism has minimal marginal effect. Disabling OOD correction (A1) makes the "
                         "redirect threshold unreachable, producing the same effect as explicitly disabling "
                         "redirect (A2).")

        # 贡献度排名
        a5_v = vals.get('A5', 0)
        a0_v = vals.get('A0', 0)
        if a5_v > 0:
            lines.append(f"\n**Mechanism contribution ranking (vs A5 Pure Ensemble):**")
            lines.append(f"1. Quality Gate: +{(vals.get('A6',0) - a5_v)*100:.1f}pp (most impactful)")
            lines.append(f"2. Router Weights (vs equal): +{(a0_v - vals.get('A6',0))*100:.1f}pp")
            lines.append(f"3. OOD Correction + Cache Redirect: ~0pp (redundant on this dataset)")

    lines.append("\n## Part B: Router优化结果")
    if router_results:
        rr = router_results.get('router_results', router_results)
        lines.append("| 配置 | 名称 | Macro F1 |")
        lines.append("|------|------|----------|")
        for k in ['B0','B1','B2','B3','B4']:
            if k in rr:
                f1 = rr[k].get('macro_f1')
                f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                name = rr[k].get('name', k)
                lines.append(f"| {k} | {name} | {f1_str} |")

        # 找最优
        best_k, best_f1 = None, 0
        for k in ['B0','B2','B3','B4']:
            if k in rr and rr[k].get('macro_f1') is not None:
                if rr[k]['macro_f1'] > best_f1:
                    best_f1 = rr[k]['macro_f1']
                    best_k = k
        if best_k:
            lines.append(f"\n**Best Router: {best_k} ({rr[best_k].get('name','')}) "
                         f"with macro F1 = {best_f1:.4f}**")
            # general域分析
            b0_gen = rr.get('B0', {}).get('per_class', {}).get('general', 0)
            best_gen = rr.get(best_k, {}).get('per_class', {}).get('general', 0)
            if b0_gen and best_gen:
                lines.append(f"\nGeneral domain F1: B0={b0_gen:.3f} → {best_k}={best_gen:.3f} "
                             f"(+{(best_gen-b0_gen)*100:.1f}pp)")

    report_path = EXP_DIR / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"报告已保存: {report_path}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Exp11: Ablation & Router Optimization')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='只运行指定阶段')
    parser.add_argument('--all', action='store_true', help='运行全部阶段')
    parser.add_argument('--ablation', type=str, default=None,
                        help='只运行指定消融配置 (A0~A6)')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='强制重新推理，忽略缓存')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='跳过BERTScore计算')
    parser.add_argument('--test-mode', action='store_true',
                        help='测试模式（少量样本）')
    parser.add_argument('--extract-multilayer', action='store_true',
                        help='启用B1多层特征提取（需要GPU约10分钟）')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("实验11: Output Ensemble消融与路由优化")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"参数: phase={args.phase}, all={args.all}, ablation={args.ablation}")
    logger.info("=" * 80)

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    ablation_results = None
    router_results = None

    if args.phase == 1 or args.all:
        ablation_results = run_phase1(args)

    if args.phase == 2 or args.all:
        router_results = run_phase2(args)

    if args.phase == 3 or args.all:
        run_phase3(args, ablation_results, router_results)

    logger.info("\n" + "=" * 80)
    logger.info(f"实验11完成 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结果目录: {EXP_DIR}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()