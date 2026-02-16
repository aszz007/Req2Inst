"""
从已保存的training_history.json重新绘制训练曲线

功能：
  - 读取已保存的training_history.json文件
  - 重新生成训练曲线可视化
  - 支持所有专家类型和微调方法
  - 无需重新训练

用法：
  python scripts/utils/replot_training_curves.py checkpoints/prompt_tuning/text_expert/training_history.json
  python scripts/utils/replot_training_curves.py checkpoints/p_tuning/uml_expert/training_history.json

  或批量处理：
  python scripts/utils/replot_training_curves.py --all

作者：Training System
日期：2025-02-16
"""

import json
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

    # 提取数据 - 为每个指标单独记录对应的steps
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

        if 'loss' in entry:
            loss_steps.append(step)
            losses.append(entry['loss'])

        if 'eval_loss' in entry:
            eval_steps.append(step)
            eval_losses.append(entry['eval_loss'])

        if 'grad_norm' in entry:
            grad_norm_steps.append(step)
            grad_norms.append(entry['grad_norm'])

        if 'learning_rate' in entry:
            lr_steps.append(step)
            learning_rates.append(entry['learning_rate'])

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

    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"训练曲线已保存至: {output_path}")
    return True


def replot_single_history(history_file):
    """
    为单个训练历史文件重新绘制曲线

    Args:
        history_file: training_history.json文件路径

    Returns:
        bool: 是否成功
    """
    history_path = Path(history_file)

    if not history_path.exists():
        logger.error(f"文件不存在: {history_path}")
        return False

    try:
        # 读取训练历史
        with open(history_path, 'r') as f:
            history_data = json.load(f)

        expert_type = history_data.get('expert_type', 'unknown')
        method_name = history_data.get('method_name', 'unknown')
        training_history = history_data.get('history', [])

        if not training_history:
            logger.error(f"训练历史为空: {history_path}")
            return False

        logger.info(f"正在重新绘制曲线: {expert_type} expert, {method_name} method")
        logger.info(f"训练步数: {len(training_history)}")

        # 创建输出目录
        output_dir = PROJECT_ROOT / 'outputs' / 'training_curves'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 输出文件路径
        output_path = output_dir / f'{expert_type}_expert_{method_name}_training_curves_{timestamp}.png'

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

    success_count = 0
    fail_count = 0

    for i, history_file in enumerate(history_files, 1):
        logger.info(f"[{i}/{len(history_files)}] 处理: {history_file.relative_to(PROJECT_ROOT)}")

        if replot_single_history(history_file):
            success_count += 1
        else:
            fail_count += 1

        logger.info("-" * 80)

    logger.info("=" * 80)
    logger.info(f"批量处理完成: 成功 {success_count}, 失败 {fail_count}")
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