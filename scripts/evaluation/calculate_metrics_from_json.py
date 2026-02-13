"""
Calculate Metrics from JSON - Fast Metric Recalculation
从JSON快速计算指标 - 无需重新生成预测

功能:
  - 从保存的predictions JSON文件读取数据
  - 快速重新计算评估指标
  - 支持调整评估阈值
  - 避免重复生成指令，节省时间

环境要求: qwen_text
运行方式: python scripts/evaluation/calculate_metrics_from_json.py --input path/to/predictions.json

使用场景:
  - 调整评估阈值后重新计算指标
  - 对比不同阈值配置的效果
  - 快速验证评估逻辑修改

作者: Evaluation System
日期: 2025-02-12
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced_metrics import EnhancedMetrics, EvaluationThresholds
from src.utils.logger import get_logger

logger = get_logger('evaluation.calculate_metrics_from_json')


def load_predictions_json(filepath: str) -> Dict:
    """
    加载预测数据JSON文件

    Args:
        filepath: JSON文件路径

    Returns:
        dict: 包含inputs, predictions, references的字典
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    logger.info(f"加载预测数据: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取samples数据
    samples = data.get('samples', [])

    if not samples:
        raise ValueError("JSON文件中没有samples数据")

    inputs = [s['input'] for s in samples]
    predictions = [s['prediction'] for s in samples]
    references = [s['reference'] for s in samples]

    logger.info(f"成功加载 {len(samples)} 个样本")
    logger.info(f"专家: {data.get('expert_name', 'unknown')}")
    logger.info(f"时间戳: {data.get('timestamp', 'unknown')}")

    return {
        'expert_name': data.get('expert_name', 'unknown'),
        'original_timestamp': data.get('timestamp', 'unknown'),
        'inputs': inputs,
        'predictions': predictions,
        'references': references
    }


def calculate_metrics(
        predictions: List[str],
        references: List[str],
        use_bertscore: bool = True,
        rouge_threshold: float = None,
        bertscore_threshold: float = None,
        use_and_logic: bool = None,
        format_threshold: float = None
) -> Dict:
    """
    计算评估指标

    Args:
        predictions: 预测列表
        references: 参考列表
        use_bertscore: 是否使用BERTScore
        rouge_threshold: ROUGE-L阈值（None使用配置默认值）
        bertscore_threshold: BERTScore阈值（None使用配置默认值）
        use_and_logic: 是否使用AND逻辑（None使用配置默认值）
        format_threshold: 格式阈值（None使用配置默认值）

    Returns:
        dict: 评估结果
    """
    logger.info("=" * 80)
    logger.info("开始计算评估指标")
    logger.info("=" * 80)

    # 创建评估器
    metrics = EnhancedMetrics(use_bertscore=use_bertscore)

    # 过滤空预测
    valid_pairs = [
        (pred, ref) for pred, ref in zip(predictions, references)
        if pred.strip()
    ]

    if not valid_pairs:
        logger.error("没有有效的预测结果")
        return {}

    valid_predictions = [pair[0] for pair in valid_pairs]
    valid_references = [pair[1] for pair in valid_pairs]

    logger.info(f"有效样本数: {len(valid_predictions)}/{len(predictions)}")

    # 生成质量指标
    logger.info("\n[1/4] 计算生成质量指标...")
    quality_metrics = metrics.calculate_generation_quality(
        predictions=valid_predictions,
        references=valid_references
    )

    # 格式指标
    logger.info("\n[2/4] 计算格式指标...")
    format_metrics = metrics.calculate_format_metrics(
        instructions=valid_predictions
    )

    # 二分类指标
    logger.info("\n[3/4] 计算二分类指标...")
    binary_metrics = metrics.calculate_binary_classification_metrics(
        predictions=valid_predictions,
        references=valid_references,
        format_threshold=format_threshold,
        rouge_threshold=rouge_threshold,
        bertscore_threshold=bertscore_threshold,
        use_and_logic=use_and_logic
    )

    # 统计指标
    logger.info("\n[4/4] 计算统计指标...")
    statistical_metrics = metrics.calculate_statistical_metrics(
        instructions=valid_predictions
    )

    # 组合结果
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(predictions),
        'valid_samples': len(valid_predictions),
        'generation_quality': quality_metrics,
        'format_metrics': format_metrics,
        'binary_classification': binary_metrics,
        'statistical_metrics': statistical_metrics,
        'threshold_config': EvaluationThresholds.get_config()
    }

    logger.info("=" * 80)
    logger.info("指标计算完成")
    logger.info("=" * 80)

    return results


def print_metrics_summary(results: Dict, expert_name: str):
    """
    打印指标摘要

    Args:
        results: 评估结果
        expert_name: 专家名称
    """
    print("\n" + "=" * 80)
    print(f"评估结果摘要 - {expert_name}")
    print("=" * 80)

    # 生成质量
    print("\n[生成质量指标]")
    quality = results['generation_quality']
    print(f"  BLEU:        {quality['bleu']:.4f}")
    print(f"  ROUGE-L:     {quality['rougeL']:.4f}")
    print(f"  METEOR:      {quality['meteor']:.4f}")
    if 'bertscore_f1' in quality:
        print(f"  BERTScore F1: {quality['bertscore_f1']:.4f}")

    # 格式指标
    print("\n[格式指标]")
    format_m = results['format_metrics']
    print(f"  格式分数:    {format_m['avg_format_score']:.4f}")
    print(f"  通过率:      {format_m['valid_rate']:.2%}")

    # 二分类指标
    print("\n[二分类指标]")
    binary = results['binary_classification']
    print(f"  Precision:   {binary['precision']:.4f}")
    print(f"  Recall:      {binary['recall']:.4f}")
    print(f"  F1 Score:    {binary['f1_score']:.4f}")
    print(f"  TP: {binary['TP']:<6d}  FP: {binary['FP']:<6d}  FN: {binary['FN']:<6d}")

    # 阈值配置
    print("\n[阈值配置]")
    print(f"  ROUGE-L阈值:      {binary['rouge_threshold']:.2f}")
    print(f"  BERTScore阈值:    {binary['bertscore_threshold']:.2f}")
    print(f"  组合逻辑:         {'AND (两者都需满足)' if binary['use_and_logic'] else 'OR (满足一个即可)'}")
    print(f"  格式分数阈值:     {binary['format_threshold']:.2f}")

    print("=" * 80 + "\n")


def save_results(results: Dict, expert_name: str, save_dir: str):
    """
    保存评估结果

    Args:
        results: 评估结果
        expert_name: 专家名称
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{expert_name}_metrics_{timestamp}.json'
    filepath = save_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"评估结果已保存至: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='从预测JSON快速重新计算评估指标',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认阈值
  python calculate_metrics_from_json.py --input predictions.json

  # 调整ROUGE阈值
  python calculate_metrics_from_json.py --input predictions.json --rouge-threshold 0.6

  # 使用OR逻辑
  python calculate_metrics_from_json.py --input predictions.json --use-or

  # 禁用BERTScore加快速度
  python calculate_metrics_from_json.py --input predictions.json --no-bertscore
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='预测数据JSON文件路径')
    parser.add_argument('--save-dir', '-o', type=str, default='outputs/evaluations/metrics',
                        help='结果保存目录')

    # 阈值参数
    parser.add_argument('--rouge-threshold', type=float, default=None,
                        help=f'ROUGE-L阈值（默认: {EvaluationThresholds.ROUGE_L_THRESHOLD}）')
    parser.add_argument('--bertscore-threshold', type=float, default=None,
                        help=f'BERTScore F1阈值（默认: {EvaluationThresholds.BERTSCORE_F1_THRESHOLD}）')
    parser.add_argument('--format-threshold', type=float, default=None,
                        help=f'格式分数阈值（默认: {EvaluationThresholds.FORMAT_SCORE_THRESHOLD}）')

    # 逻辑选择
    logic_group = parser.add_mutually_exclusive_group()
    logic_group.add_argument('--use-and', dest='use_and_logic', action='store_true',
                             help='使用AND逻辑组合ROUGE和BERTScore（默认）')
    logic_group.add_argument('--use-or', dest='use_and_logic', action='store_false',
                             help='使用OR逻辑组合ROUGE和BERTScore')
    parser.set_defaults(use_and_logic=None)

    # BERTScore开关
    parser.add_argument('--use-bertscore', action='store_true', default=True,
                        help='使用BERTScore（默认启用）')
    parser.add_argument('--no-bertscore', dest='use_bertscore', action='store_false',
                        help='禁用BERTScore（加快计算速度）')

    args = parser.parse_args()

    # 加载预测数据
    try:
        data = load_predictions_json(args.input)
    except Exception as e:
        logger.error(f"加载预测数据失败: {e}")
        sys.exit(1)

    # 计算指标
    try:
        results = calculate_metrics(
            predictions=data['predictions'],
            references=data['references'],
            use_bertscore=args.use_bertscore,
            rouge_threshold=args.rouge_threshold,
            bertscore_threshold=args.bertscore_threshold,
            use_and_logic=args.use_and_logic,
            format_threshold=args.format_threshold
        )
    except Exception as e:
        logger.error(f"计算指标失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # 添加原始数据信息
    results['expert_name'] = data['expert_name']
    results['original_timestamp'] = data['original_timestamp']
    results['input_file'] = args.input

    # 打印摘要
    print_metrics_summary(results, data['expert_name'])

    # 保存结果
    save_results(results, data['expert_name'], args.save_dir)

    logger.info("完成!")


if __name__ == "__main__":
    main()