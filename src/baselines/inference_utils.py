"""
Inference Utilities - Shared cache save/load and metric utilities for Phase 2 experiments.

Provides thin wrappers around EnhancedMetrics and JSON cache I/O so that all
experiment scripts share a consistent cache format compatible with
calculate_metrics_from_json.py.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced_metrics import EnhancedMetrics
from src.utils.logger import get_logger

logger = get_logger('baselines.inference_utils')


def save_predictions_cache(
    samples: List[Dict],
    method: str,
    expert_type: str,
    config: Dict,
    cache_dir: Path,
    filename: str = None
) -> Path:
    """
    Save predictions to a cache JSON file.

    The format matches what calculate_metrics_from_json.py expects:
      {
        "method": ...,
        "expert_type": ...,
        "expert_name": ...,
        "config": {...},
        "timestamp": ...,
        "total_samples": N,
        "samples": [{"index": 0, "input": ..., "prediction": ..., "reference": ...}]
      }

    Args:
        samples: List of dicts with keys: index, input, prediction, reference
        method: Method name, e.g. 'lora_moe', 'bm25', 'zero_shot'
        expert_type: Expert type, e.g. 'text', 'image', 'uml', 'general'
        config: Arbitrary config dict to store alongside predictions
        cache_dir: Directory to write cache file into (created if needed)
        filename: Override filename. Defaults to '{expert_type}_predictions.json'

    Returns:
        Path to the saved cache file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f'{expert_type}_predictions.json'

    filepath = cache_dir / filename

    expert_name = f'{expert_type}_expert'

    payload = {
        'method': method,
        'expert_type': expert_type,
        'expert_name': expert_name,
        'config': config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(samples),
        'samples': samples,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f'缓存已保存: {filepath}（{len(samples)}个样本）')
    return filepath


def load_predictions_cache(cache_dir: Path, filename: str) -> Optional[Dict]:
    """
    加载预测缓存JSON文件。

    Args:
        cache_dir: 缓存文件所在目录
        filename: 缓存文件名

    Returns:
        解析后的字典，若文件不存在则返回None
    """
    filepath = Path(cache_dir) / filename
    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f'缓存已加载: {filepath}（{data.get("total_samples", "?")}个样本）')
    return data


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    use_bertscore: bool = True
) -> Dict:
    """
    Run the full EnhancedMetrics suite and return a combined result dict.

    Filters empty predictions before evaluation and reports how many were
    skipped.

    Args:
        predictions: Model-generated strings
        references: Ground-truth strings
        use_bertscore: Whether to compute BERTScore (slower but richer)

    Returns:
        Combined dict with keys from generation_quality, format_metrics,
        binary_classification, and statistical_metrics
    """
    valid_pairs = [
        (p, r) for p, r in zip(predictions, references) if p and p.strip()
    ]

    skipped = len(predictions) - len(valid_pairs)
    if skipped:
        logger.warning(f'已跳过 {skipped} 条空预测（共{len(predictions)}条）')

    if not valid_pairs:
        logger.error('没有有效预测，无法评估')
        return {}

    valid_preds = [pair[0] for pair in valid_pairs]
    valid_refs = [pair[1] for pair in valid_pairs]

    metrics = EnhancedMetrics(use_bertscore=use_bertscore)

    quality = metrics.calculate_generation_quality(
        predictions=valid_preds,
        references=valid_refs
    )
    format_m = metrics.calculate_format_metrics(instructions=valid_preds)
    # 复用 generate_quality 已计算的 per-sample BERTScore F1，避免 calculate_binary 重复推理
    precomputed_bs = quality.get('bertscore_f1_scores', None)
    binary = metrics.calculate_binary_classification_metrics(
        predictions=valid_preds,
        references=valid_refs,
        precomputed_bertscore_f1=precomputed_bs
    )
    stats = metrics.calculate_statistical_metrics(instructions=valid_preds)

    result = {
        'total_samples': len(predictions),
        'valid_samples': len(valid_preds),
        'skipped_samples': skipped,
        'generation_quality': quality,
        'format_metrics': format_m,
        'binary_classification': binary,
        'statistical_metrics': stats,
    }

    logger.info(
        f'指标计算完成 | ROUGE-L={quality.get("rougeL", 0):.4f} '
        f'F1={binary.get("f1_score", 0):.4f}'
    )

    try:
        metrics.cleanup()
    except Exception:
        pass
    del metrics
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return result


def save_experiment_results(
    results: Dict,
    exp_dir: Path,
    filename: str = 'results.json'
) -> Path:
    """
    Save experiment results to a JSON file.

    Args:
        results: Results dict to serialize
        exp_dir: Output directory (created if needed)
        filename: Output filename

    Returns:
        Path to the saved file
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    filepath = exp_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f'实验结果已保存: {filepath}')
    return filepath