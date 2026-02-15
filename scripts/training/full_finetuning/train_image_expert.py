"""
图像专家训练脚本
功能：训练Image Expert，将图像描述转换为标注指令
环境：instruction_generator（transformers==4.57.0）
基础模型：Qwen3-8B（默认）
输出：checkpoints/lora_moe/image_expert/

使用方法：
  # 方法1: 通过环境管理脚本运行（推荐）
  python scripts/run_with_env.py --env text --script scripts/training/train_image_expert.py

  # 方法2: 直接在instruction_generator环境中运行
  conda activate instruction_generator
  python scripts/training/train_image_expert.py

作者：Training System
日期：2025-02-15
"""

import sys
import argparse
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.lora_trainer import LoRATrainer
from config.settings import get_path_config, get_training_config, get_lora_config
from src.utils.logger import get_logger

logger = get_logger('training.train_image_expert')

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

def print_header():
    """打印训练开始的标题"""
    print("=" * 80)
    print(" " * 22 + "图像专家训练 (Image Expert Training)")
    print("=" * 80)
    print()


def print_config(use_4bit: bool, use_rtx4090_opt: bool):
    """打印训练配置"""
    path_cfg = get_path_config()
    train_cfg = get_training_config()
    lora_cfg = get_lora_config('conservative')

    print("训练配置信息:")
    print("-" * 80)
    print(f"专家类型: Image Expert")
    print(f"基础模型: {path_cfg.QWEN_7B_CHAT_PATH}")
    print(f"输出目录: checkpoints/lora_moe/image_expert/")
    print()
    print(f"LoRA配置:")
    print(f"  - Rank: {lora_cfg.rank}")
    print(f"  - Alpha: {lora_cfg.alpha}")
    print(f"  - Dropout: {lora_cfg.dropout}")
    print(f"  - Target Modules: {lora_cfg.target_modules}")
    print()

    if use_rtx4090_opt:
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
        version = transformers.__version__
        print(f"Transformers版本: {version}")

        # Image Expert应该使用transformers 4.57.0（统一环境）
        try:
            v_parts = version.split('.')
            major, minor = int(v_parts[0]), int(v_parts[1])
            if not (major > 4 or (major == 4 and minor >= 51)):
                logger.warning(f"警告：当前transformers版本为{version}，推荐使用>=4.51.0")
                logger.warning("请确认是否在instruction_generator环境中运行")
        except (ValueError, IndexError):
            logger.warning(f"无法解析transformers版本: {version}")
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
    parser = argparse.ArgumentParser(description='训练图像专家')
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

    # 检测是否为RTX 4090
    is_rtx4090 = detect_rtx4090()
    use_rtx4090_opt = is_rtx4090

    if is_rtx4090:
        logger.info("检测到RTX 4090，启用优化配置")

    # 创建训练器（会自动打印实际配置）
    logger.info("创建图像专家训练器...")
    try:
        trainer = LoRATrainer(
            expert_type='image',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt
        )
    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        return 1

    # 设置模型（必须先于prepare_data，因为需要初始化tokenizer）
    logger.info("设置模型和LoRA配置...")
    if not trainer.setup_model():
        logger.error("模型设置失败")
        return 1

    # 准备数据（此时tokenizer已初始化）
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
        output_path = path_cfg.PROJECT_ROOT / 'checkpoints' / 'lora_moe' / 'image_expert'
        print(f"LoRA权重已保存至: {output_path}")
        print(f"检查点目录: {output_path / 'training_checkpoints'}")
        print()
        print("下一步:")
        print("  1. 可以使用该权重进行推理测试")
        print("  2. 继续训练其他专家（Text, UML, General）")
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