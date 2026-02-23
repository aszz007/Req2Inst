"""
UML专家训练脚本
功能：训练UML Expert，将UML用例图描述转换为业务逻辑实现指令
环境：instruction_generator（transformers==4.57.0）
基础模型：Qwen3-8B（默认）
数据集：uml_dataset.csv（1500条数据）
输出：checkpoints/lora_moe/uml_expert/

使用方法：
  # 方法1: 通过环境管理脚本运行（推荐）
  python scripts/run_with_env.py --env text --script scripts/training/train_uml_expert.py

  # 方法2: 直接在instruction_generator环境中运行
  conda activate instruction_generator
  python scripts/training/train_uml_expert.py

作者：Training System
日期：2025-02-15
"""

import sys
import argparse
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.lora_trainer import LoRATrainer
from config.settings import get_path_config, get_training_config, get_lora_config
from src.utils.logger import get_logger

logger = get_logger('training.train_uml_expert')


def print_header():
    """打印训练开始的标题"""
    print("=" * 80)
    print(" " * 30 + "UML专家训练 (UML Expert Training)")
    print("=" * 80)
    print()


def print_config(use_4bit: bool, use_rtx4090: bool):
    """打印训练配置"""
    path_cfg = get_path_config()
    train_cfg = get_training_config()
    lora_cfg = get_lora_config('conservative')

    print("训练配置信息:")
    print("-" * 80)
    print(f"专家类型: UML Expert")
    print(f"数据集: uml_dataset.csv（1500条数据）")
    print(f"基础模型: {path_cfg.QWEN_7B_CHAT_PATH}")
    print(f"输出目录: checkpoints/lora_moe/uml_expert/")
    print()

    print(f"LoRA配置:")
    print(f"  - Rank: {lora_cfg.rank}")
    print(f"  - Alpha: {lora_cfg.alpha}")
    print(f"  - Dropout: {lora_cfg.dropout}")
    print(f"  - Target Modules: {lora_cfg.target_modules}")
    print()

    if use_rtx4090:
        print(f"训练参数 (RTX 4090优化):")
        print(f"  - Batch Size: 8 (优化后)")
        print(f"  - Gradient Accumulation: 2 (优化后)")
        print(f"  - 有效Batch Size: 16")
        print(f"  - Epochs: {train_cfg.num_epochs}")
        print(f"  - Learning Rate: {train_cfg.learning_rate}")
        print(f"  - Max Seq Length: {train_cfg.max_seq_length}")
        print(f"  - 4bit量化: {use_4bit}")
        print(f"  - BF16混合精度: True")
        print(f"  - TF32加速: True")
        print(f"  - Fused优化器: True")
        print(f"  - 数据加载器工作进程: 8")
    else:
        print(f"训练参数:")
        print(f"  - Batch Size: {train_cfg.batch_size}")
        print(f"  - Gradient Accumulation: {train_cfg.gradient_accumulation_steps}")
        print(f"  - 有效Batch Size: {train_cfg.batch_size * train_cfg.gradient_accumulation_steps}")
        print(f"  - Epochs: {train_cfg.num_epochs}")
        print(f"  - Learning Rate: {train_cfg.learning_rate}")
        print(f"  - Max Seq Length: {train_cfg.max_seq_length}")
        print(f"  - 4bit量化: {use_4bit}")

    print("-" * 80)
    print()


def validate_environment():
    """验证运行环境"""
    print("验证运行环境...")
    print("-" * 80)

    # 检查transformers版本
    try:
        import transformers
        tf_version = transformers.__version__

        print(f"Transformers版本: {tf_version}")

        # UML Expert应该使用transformers 4.57.0（统一环境）
        try:
            v_parts = tf_version.split('.')
            major, minor = int(v_parts[0]), int(v_parts[1])
            if not (major > 4 or (major == 4 and minor >= 51)):
                logger.warning(f"警告：当前transformers版本为{tf_version}，推荐使用>=4.51.0")
                logger.warning("请确认是否在instruction_generator环境中运行")
        except (ValueError, IndexError):
            logger.warning(f"无法解析transformers版本: {tf_version}")

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
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"CUDA可用: {gpu_name}")
            print(f"显存: {gpu_memory:.2f}GB")

            # 检测是否为4090
            is_rtx4090 = 'RTX 4090' in gpu_name or 'RTX 4090D' in gpu_name
            if is_rtx4090:
                print(f"检测到RTX 4090，将启用优化配置")

        else:
            logger.warning("CUDA不可用，将使用CPU训练（速度极慢）")
    except ImportError:
        logger.error("未安装PyTorch库")
        return False

    print("-" * 80)
    print()
    return True


def detect_rtx4090() -> bool:
    """检测是否为RTX 4090显卡"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return 'RTX 4090' in gpu_name or 'RTX 4090D' in gpu_name
    except:
        pass
    return False


def main():
    """主训练流程"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练UML专家')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    parser.add_argument('--no_rtx4090_opt', action='store_true',
                        help='禁用RTX 4090优化（默认：自动检测）')
    args = parser.parse_args()

    # 检测是否为RTX 4090
    is_rtx4090 = detect_rtx4090()
    use_rtx4090_opt = is_rtx4090 and not args.no_rtx4090_opt

    if is_rtx4090:
        if use_rtx4090_opt:
            logger.info("检测到RTX 4090，启用优化配置")
        else:
            logger.info("检测到RTX 4090，但优化已禁用")

    # 打印标题
    print_header()

    # 验证环境
    if not validate_environment():
        logger.error("环境验证失败，请检查依赖库")
        return 1

    # 创建训练器（会自动打印实际配置）
    logger.info(f"创建UML专家训练器...")
    try:
        trainer = LoRATrainer(
            expert_type='uml',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt
        )
    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    # 设置模型
    logger.info("设置模型和LoRA配置...")
    if not trainer.setup_model():
        logger.error("模型设置失败")
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
    print(f"  - 数据集: uml_dataset.csv（1500条）")
    print()
    print("注意：1500条数据使用标准80:10:10划分策略")
    print()

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
        output_path = path_cfg.PROJECT_ROOT / 'checkpoints' / 'lora_moe' / 'uml_expert'
        print(f"LoRA权重已保存至: {output_path}")
        print(f"检查点目录: {output_path / 'training_checkpoints'}")
        print()
        print("下一步:")
        print("  1. 可以使用该权重进行推理测试")
        print("  2. 继续训练General Expert")
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