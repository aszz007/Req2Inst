"""
通用专家训练脚本
功能：训练General Expert，作为兜底专家处理各类需求
环境：qwen_text（transformers==4.32.0）
基础模型：Qwen-7B-Chat
输出：lora_weights/experts/general_expert/

使用方法：
  # 方法1: 通过环境管理脚本运行（推荐）
  python scripts/run_with_env.py --env text --script scripts/training/train_general_expert.py

  # 方法2: 直接在qwen_text环境中运行
  conda activate qwen_text
  python scripts/training/train_general_expert.py

作者：Training System
日期：2025-01-30
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.expert_trainer import ExpertTrainer
from config.settings import get_path_config, get_training_config, get_lora_config
from src.utils.logger import get_logger

logger = get_logger('training.train_general_expert')


def print_header():
    """打印训练开始的标题"""
    print("=" * 80)
    print(" " * 18 + "通用专家训练 (General Expert Training)")
    print("=" * 80)
    print()


def print_config():
    """打印训练配置"""
    path_cfg = get_path_config()
    train_cfg = get_training_config()
    lora_cfg = get_lora_config('conservative')

    print("训练配置信息:")
    print("-" * 80)
    print(f"专家类型: General Expert (兜底专家)")
    print(f"基础模型: {path_cfg.QWEN_7B_CHAT_PATH}")
    print(f"输出目录: {path_cfg.GENERAL_EXPERT_WEIGHTS}")
    print()
    print(f"数据来源: 文本 + 图像 + UML 混合数据")
    print()
    print(f"LoRA配置:")
    print(f"  - Rank: {lora_cfg.rank}")
    print(f"  - Alpha: {lora_cfg.alpha}")
    print(f"  - Dropout: {lora_cfg.dropout}")
    print(f"  - Target Modules: {lora_cfg.target_modules}")
    print()
    print(f"训练参数:")
    print(f"  - Batch Size: {train_cfg.batch_size}")
    print(f"  - Gradient Accumulation: {train_cfg.gradient_accumulation_steps}")
    print(f"  - 有效Batch Size: {train_cfg.batch_size * train_cfg.gradient_accumulation_steps}")
    print(f"  - Epochs: {train_cfg.num_epochs}")
    print(f"  - Learning Rate: {train_cfg.learning_rate}")
    print(f"  - Max Seq Length: {train_cfg.max_seq_length}")
    print("-" * 80)
    print()


def validate_environment():
    """验证运行环境"""
    import os

    print("验证运行环境...")
    print("-" * 80)

    # 检查transformers版本
    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers版本: {version}")

        # General Expert应该使用transformers 4.32.0
        if not version.startswith('4.32'):
            logger.warning(f"警告：当前transformers版本为{version}，推荐使用4.32.x")
            logger.warning("请确认是否在qwen_text环境中运行")
    except ImportError:
        logger.error("未安装transformers库")
        return False

    # 检查PEFT
    try:
        import peft
        print(f"PEFT版本: {peft.__version__}")
    except ImportError:
        logger.error("未安装PEFT库，请运行: pip install peft --break-system-packages")
        return False

    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")
        else:
            logger.warning("CUDA不可用，将使用CPU训练（速度极慢）")
    except ImportError:
        logger.error("未安装PyTorch库")
        return False

    print("-" * 80)
    print()
    return True


def main():
    """主训练流程"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练通用专家')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    args = parser.parse_args()

    # 打印标题
    print_header()

    # 验证环境
    if not validate_environment():
        logger.error("环境验证失败，请检查依赖库")
        return 1

    # 打印配置
    print_config()

    # 创建训练器
    logger.info("创建通用专家训练器...")
    try:
        trainer = ExpertTrainer(
            expert_type='general',
            use_4bit=args.use_4bit
        )
    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        return 1

    # 准备数据
    logger.info("准备训练数据...")
    if not trainer.prepare_data():
        logger.error("数据准备失败")
        return 1

    # 打印数据统计
    status = trainer.get_training_status()
    print(f"数据统计:")
    print(f"  - 训练样本: {status['train_samples']}")
    print(f"  - 验证样本: {status['val_samples']}")
    print()
    print("注意：通用专家使用所有类型的数据（文本、图像、UML）")
    print()

    # 设置模型
    logger.info("设置模型和LoRA配置...")
    if not trainer.setup_model():
        logger.error("模型设置失败")
        return 1

    # 开始训练
    logger.info("开始训练...")
    print("=" * 80)
    print("训练开始 - 这可能需要较长时间，请耐心等待...")
    print("=" * 80)
    print()

    success = trainer.train()

    if success:
        print()
        print("=" * 80)
        print(" " * 25 + "训练成功完成！")
        print("=" * 80)
        print()

        path_cfg = get_path_config()
        print(f"LoRA权重已保存至: {path_cfg.GENERAL_EXPERT_WEIGHTS}")
        print(f"检查点目录: {path_cfg.get_checkpoint_path('general_expert')}")
        print()
        print("下一步:")
        print("  1. 可以使用该权重进行推理测试")
        print("  2. 所有专家训练已完成，可以进行MoE系统集成")
        print()

        return 0
    else:
        print()
        print("=" * 80)
        print(" " * 28 + "训练失败")
        print("=" * 80)
        print()
        logger.error("训练过程中出现错误，请查看日志")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# 使用示例：
# 方法1：通过环境管理脚本运行（推荐）
# python scripts/run_with_env.py --env text --script scripts/training/train_general_expert.py
#
# 方法2：直接在qwen_text环境中运行
# conda activate qwen_text
# python scripts/training/train_general_expert.py
#
# 注意：通用专家使用Qwen-7B-Chat模型，无需指定版本