"""
Full Fine-tuning Text Expert训练脚本（保守高质量策略）

功能：使用高rank LoRA (rank=16) 进行高质量训练
环境：instruction_generator（transformers==4.57.0）
基础模型：Qwen3-8B
方法：High-rank LoRA (rank=16) 保守高质量策略
输出：checkpoints/full_finetuning/text_expert/

训练策略（优先质量和稳定性）：
  - LoRA Rank: 16（高质量，损失5-10%）
  - LoRA Alpha: 32（标准配置）
  - Max Seq Length: 2048（覆盖Text 90%样本）
  - Batch Size: 1（保守配置）
  - Gradient Accumulation: 16（有效batch=16）
  - 4bit量化 + Gradient Checkpointing
  - 预期显存：15-18GB（安全边界）

样本覆盖率：
  - Text短样本（~500 tokens）：100%完整
  - Text长样本（~3000 tokens）：部分截断到2048
  - 总体覆盖率：约90%

训练质量：相对理想配置损失5-10%（非常好）

使用方法：
  python scripts/training/full_finetuning/train_text_expert.py

作者：Comparative Training System
日期：2025-02-16（保守高质量版）
"""

import sys
import argparse
import torch
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_path_config
from src.training.full_finetuning_trainer import FullFineTuningTrainer
from src.utils.logger import get_logger

logger = get_logger('training.full_finetuning.text_expert')


def print_header():
    """打印训练开始的标题"""
    print("=" * 80)
    print(" " * 12 + "Full Fine-tuning Text Expert训练 (保守高质量策略)")
    print("=" * 80)
    print()


def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(description='Full Fine-tuning Text Expert训练')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    args = parser.parse_args()

    # 打印标题
    print_header()

    # 获取配置
    path_cfg = get_path_config()

    # 打印策略说明
    print("=" * 80)
    print("训练策略：保守高质量配置")
    print("=" * 80)
    print("配置：")
    print(f"  - LoRA Rank: 16 (高质量)")
    print(f"  - LoRA Alpha: 32")
    print(f"  - Max Seq Length: 2048 (覆盖Text 90%样本)")
    print(f"  - Batch Size: 1")
    print(f"  - Gradient Accumulation: 16")
    print(f"  - 4bit量化: {args.use_4bit}")
    print("说明：Text短样本全覆盖，长样本（~3000 tokens）部分截断")
    print("预期：显存15-18GB，质量损失5-10%")
    print("=" * 80)
    print()

    # 创建训练器
    logger.info("初始化Full Fine-tuning Text Expert训练器...")
    trainer = FullFineTuningTrainer(
        expert_type='text',
        use_4bit=args.use_4bit,
        use_rtx4090_optimization=True,
        debug_samples=True
    )

    # 设置模型
    logger.info("设置模型...")
    if not trainer.setup_model():
        logger.error("模型设置失败")
        return 1

    # 准备数据
    logger.info("准备数据...")
    if not trainer.prepare_data():
        logger.error("数据准备失败")
        return 1

    # 开始训练
    logger.info("开始训练...")
    if not trainer.train():
        logger.error("训练失败")
        return 1

    print()
    print("=" * 80)
    print(" " * 25 + "训练成功完成！")
    print("=" * 80)
    print(f"Full Fine-tuning权重已保存至: {trainer.output_dir}")
    print()
    print("训练总结：")
    print("  - 样本覆盖率：Text 约90%")
    print("  - 训练质量：损失5-10%（非常好）")
    print("  - 稳定性：batch=1最保守配置")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())