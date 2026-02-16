"""
从已保存的training_history.json重新绘制训练曲线

功能：
  - 读取已保存的training_history.json文件
  - 重新生成训练曲线可视化
  - 支持所有专家类型和微调方法（lora_moe, lora_single, p_tuning, prompt_tuning, full_finetuning）
  - 自动从路径推断方法名（如果training_history.json中未记录）
  - 无需重新训练

输出组织结构：
  outputs/training_curves/{timestamp}/
    ├── lora_moe/
    │   ├── text_expert.png
    │   ├── image_expert.png
    │   ├── uml_expert.png
    │   └── general_expert.png
    ├── lora_single/
    │   └── unified_expert.png
    ├── p_tuning/
    │   ├── text_expert.png
    │   ├── image_expert.png
    │   ├── uml_expert.png
    │   └── general_expert.png
    ├── prompt_tuning/
    │   └── ...
    └── full_finetuning/
        └── ...

用法：
  单文件模式：
    python scripts/utils/replot_training_curves.py checkpoints/prompt_tuning/text_expert/training_history.json
    创建新的时间戳目录，包含单个图片

  批量处理模式：
    python scripts/utils/replot_training_curves.py --all
    创建单个时间戳目录，包含所有方法和专家的图片，便于对比

作者：Training System
日期：2025-02-16
"""

import json
import math
import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger('utils.replot_training_curves')


def plot_training_curves(training_history, expert_type, method_name, output_path):
    """
    生成训练曲线可视化图表

    Args:
        training_history: 训练历史记录列表
        expert_type: 专家类型
        method_name: 微调方法名称
        output_path: 输出文件路径
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib未安装，无法生成可视化")
        return False

    # 提取数据 - 为每个指标单独记录对应的steps，并过滤掉None/NaN值
    loss_steps = []
    losses = []
    eval_steps = []
    eval_losses = []
    grad_norm_steps = []
    grad_norms = []
    lr_steps = []
    learning_rates = []

    for entry in training_history:
        step = entry.get('step', 0)

        # 处理loss（过滤None和NaN）
        if 'loss' in entry:
            loss_val = entry['loss']
            if loss_val is not None and not (isinstance(loss_val, float) and math.isnan(loss_val)):
                loss_steps.append(step)
                losses.append(loss_val)

        # 处理eval_loss（过滤None和NaN）
        if 'eval_loss' in entry:
            eval_val = entry['eval_loss']
            if eval_val is not None and not (isinstance(eval_val, float) and math.isnan(eval_val)):
                eval_steps.append(step)
                eval_losses.append(eval_val)

        # 处理grad_norm（过滤None和NaN）
        if 'grad_norm' in entry:
            grad_val = entry['grad_norm']
            if grad_val is not None and not (isinstance(grad_val, float) and math.isnan(grad_val)):
                grad_norm_steps.append(step)
                grad_norms.append(grad_val)

        # 处理learning_rate（过滤None和NaN）
        if 'learning_rate' in entry:
            lr_val = entry['learning_rate']
            if lr_val is not None and not (isinstance(lr_val, float) and math.isnan(lr_val)):
                lr_steps.append(step)
                learning_rates.append(lr_val)

    # 数据质量检查和警告
    total_entries = len(training_history)
    if total_entries < 10:
        logger.warning(f"训练历史记录很少（{total_entries}条），可能导致曲线不完整")

    if len(losses) < 3:
        logger.warning(f"训练损失数据点很少（{len(losses)}个）")
    if len(eval_losses) == 0:
        logger.warning("没有验证损失数据")
    elif len(eval_losses) < 3:
        logger.warning(f"验证损失数据点很少（{len(eval_losses)}个）")

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Curves - {expert_type.upper()} Expert ({method_name})',
                 fontsize=16, fontweight='bold')

    # 1. Training Loss
    if losses:
        axes[0, 0].plot(loss_steps, losses, 'b-', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No training loss data',
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')

    # 2. Eval Loss
    if eval_losses:
        axes[0, 1].plot(eval_steps, eval_losses, 'r-', linewidth=2, marker='o', markersize=4)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Eval Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No validation loss data',
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Eval Loss')
        axes[0, 1].set_title('Validation Loss')

    # 3. Gradient Norm
    if grad_norms:
        axes[1, 0].plot(grad_norm_steps, grad_norms, 'g-', linewidth=1, alpha=0.6)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No gradient norm data',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm')

    # 4. Learning Rate
    if learning_rates:
        axes[1, 1].plot(lr_steps, learning_rates, 'm-', linewidth=1.5)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        axes[1, 1].text(0.5, 0.5, 'No learning rate data',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')

    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"训练曲线已保存至: {output_path}")
    logger.info(f"数据统计: Loss={len(losses)}点, EvalLoss={len(eval_losses)}点, GradNorm={len(grad_norms)}点, LR={len(learning_rates)}点")
    return True


def infer_method_from_path(history_path):
    """
    从文件路径推断微调方法名称

    Args:
        history_path: Path对象，training_history.json的路径

    Returns:
        str: 推断出的方法名
    """
    path_str = str(history_path)

    # 从路径中查找方法目录
    if 'lora_moe' in path_str or 'lora-moe' in path_str:
        return 'lora_moe'
    elif 'lora_single' in path_str or 'lora-single' in path_str:
        return 'lora_single'
    elif 'p_tuning' in path_str or 'p-tuning' in path_str:
        return 'p_tuning'
    elif 'prompt_tuning' in path_str or 'prompt-tuning' in path_str:
        return 'prompt_tuning'
    elif 'full_finetuning' in path_str or 'full-finetuning' in path_str:
        return 'full_finetuning'
    else:
        return 'unknown'


def replot_single_history(history_file, output_timestamp_dir=None):
    """
    为单个训练历史文件重新绘制曲线

    Args:
        history_file: training_history.json文件路径
        output_timestamp_dir: 可选的时间戳目录，用于批量处理时统一输出位置

    Returns:
        bool: 是否成功
    """
    history_path = Path(history_file)

    if not history_path.exists():
        logger.error(f"文件不存在: {history_path}")
        return False

    try:
        # 读取训练历史，处理可能的NaN值
        with open(history_path, 'r') as f:
            content = f.read()
            # 替换NaN为null，以便JSON正确解析
            content = content.replace(': NaN', ': null')
            history_data = json.loads(content)

        expert_type = history_data.get('expert_type', 'unknown')
        method_name = history_data.get('method_name', None)

        # 如果method_name为空或unknown，从路径推断
        if not method_name or method_name == 'unknown':
            method_name = infer_method_from_path(history_path)
            logger.info(f"从路径推断方法名: {method_name}")

        training_history = history_data.get('history', [])

        if not training_history:
            logger.error(f"训练历史为空: {history_path}")
            return False

        logger.info(f"正在重新绘制曲线: {expert_type} expert, {method_name} method")
        logger.info(f"训练步数: {len(training_history)}")

        # 确定输出目录结构
        if output_timestamp_dir:
            # 批量模式：使用统一的时间戳目录
            method_dir = output_timestamp_dir / method_name
        else:
            # 单文件模式：创建新的时间戳目录
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = PROJECT_ROOT / 'outputs' / 'training_curves'
            timestamp_dir = base_dir / timestamp
            method_dir = timestamp_dir / method_name

        # 创建方法目录
        method_dir.mkdir(parents=True, exist_ok=True)

        # 输出文件路径：简化命名，只包含专家类型
        output_path = method_dir / f'{expert_type}_expert.png'

        # 绘制曲线
        success = plot_training_curves(training_history, expert_type, method_name, output_path)

        if success:
            logger.info(f"成功: {output_path}")
            return True
        else:
            logger.error(f"绘制失败: {history_path}")
            return False

    except Exception as e:
        logger.error(f"处理失败 {history_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def replot_all_histories():
    """
    批量处理所有训练历史文件

    Returns:
        tuple: (成功数量, 失败数量)
    """
    checkpoints_dir = PROJECT_ROOT / 'checkpoints'

    if not checkpoints_dir.exists():
        logger.error(f"checkpoints目录不存在: {checkpoints_dir}")
        return 0, 0

    # 查找所有training_history.json文件
    history_files = list(checkpoints_dir.glob('**/training_history.json'))

    if not history_files:
        logger.warning("未找到任何training_history.json文件")
        return 0, 0

    logger.info(f"找到 {len(history_files)} 个训练历史文件")
    logger.info("=" * 80)

    # 创建统一的时间戳目录（批量处理时所有图片放在同一个时间戳目录下）
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = PROJECT_ROOT / 'outputs' / 'training_curves'
    output_timestamp_dir = base_dir / timestamp
    output_timestamp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"批量输出目录: {output_timestamp_dir}")
    logger.info("=" * 80)

    success_count = 0
    fail_count = 0

    for i, history_file in enumerate(history_files, 1):
        logger.info(f"[{i}/{len(history_files)}] 处理: {history_file.relative_to(PROJECT_ROOT)}")

        if replot_single_history(history_file, output_timestamp_dir):
            success_count += 1
        else:
            fail_count += 1

        logger.info("-" * 80)

    logger.info("=" * 80)
    logger.info(f"批量处理完成: 成功 {success_count}, 失败 {fail_count}")
    logger.info(f"所有图片已保存至: {output_timestamp_dir}")
    logger.info("=" * 80)

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description='从已保存的training_history.json重新绘制训练曲线'
    )
    parser.add_argument(
        'history_file',
        nargs='?',
        help='training_history.json文件路径'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='批量处理checkpoints目录下的所有训练历史'
    )

    args = parser.parse_args()

    if args.all:
        # 批量处理模式
        logger.info("批量处理模式：处理所有训练历史文件")
        success, fail = replot_all_histories()
        sys.exit(0 if fail == 0 else 1)
    elif args.history_file:
        # 单文件处理模式
        success = replot_single_history(args.history_file)
        sys.exit(0 if success else 1)
    else:
        # 未指定参数，显示帮助
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()