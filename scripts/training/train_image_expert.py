"""
图像专家训练脚本
功能：训练Image Expert，将图像描述转换为标注指令
环境：qwen_vision25（transformers==4.37.0）或 qwen_vision3（transformers==4.45.0）
基础模型：Qwen2.5-VL-7B 或 Qwen3-VL-8B
输出：lora_weights/experts/image_expert_qwen2.5/ 或 image_expert_qwen3/

使用方法：
  # 方法1: 通过环境管理脚本运行（推荐）
  # 使用Qwen2.5-VL
  python scripts/run_with_env.py --env image_qwen2.5 --script scripts/training/train_image_expert.py

  # 使用Qwen3-VL
  python scripts/run_with_env.py --env image_qwen3 --script scripts/training/train_image_expert.py

  # 方法2: 直接在对应环境中运行（需手动指定版本）
  conda activate qwen_vision25
  python scripts/training/train_image_expert.py --version qwen2.5

  conda activate qwen_vision3
  python scripts/training/train_image_expert.py --version qwen3

作者：Training System
日期：2025-01-30
"""

import sys
import argparse
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.expert_trainer import ExpertTrainer
from config.settings import (
    get_path_config,
    get_training_config,
    get_lora_config,
    get_vision_model_config,
    set_vision_model_version
)
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

def print_header(version: str):
    """打印训练开始的标题"""
    print("=" * 80)
    print(" " * 18 + f"图像专家训练 (Image Expert Training - {version.upper()})")
    print("=" * 80)
    print()


def print_config(version: str):
    """打印训练配置"""
    path_cfg = get_path_config()
    train_cfg = get_training_config()
    lora_cfg = get_lora_config('conservative')

    # 根据版本获取模型路径
    base_model_path = path_cfg.get_vision_model_path(version)
    output_dir = path_cfg.get_expert_weight_path('image', vision_version=version)

    print("训练配置信息:")
    print("-" * 80)
    print(f"专家类型: Image Expert")
    print(f"视觉模型版本: {version}")
    print(f"基础模型: {base_model_path}")
    print(f"输出目录: {output_dir}")
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


def validate_environment(version: str):
    """验证运行环境"""
    print("验证运行环境...")
    print("-" * 80)

    # 检查transformers版本
    try:
        import transformers
        tf_version = transformers.__version__

        print(f"Transformers版本: {tf_version}")

        # 检查版本是否匹配
        if version == 'qwen2.5' and not tf_version.startswith('4.37'):
            logger.warning(f"警告：Qwen2.5-VL推荐使用transformers 4.37.x，当前版本：{tf_version}")
            logger.warning("请确认是否在qwen_vision25环境中运行")
        elif version == 'qwen3' and not tf_version.startswith('4.45'):
            logger.warning(f"警告：Qwen3-VL推荐使用transformers 4.45.x，当前版本：{tf_version}")
            logger.warning("请确认是否在qwen_vision3环境中运行")

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
    parser = argparse.ArgumentParser(description='训练图像专家（支持多版本）')
    parser.add_argument(
        '--version',
        type=str,
        choices=['qwen2.5', 'qwen3'],
        help='视觉模型版本（qwen2.5 或 qwen3）'
    )
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    args = parser.parse_args()

    # 获取版本（优先级：命令行参数 > 环境变量 > 默认值）
    if args.version:
        version = args.version
        logger.info(f"使用命令行参数指定的版本: {version}")
    else:
        # 从环境变量或默认值获取
        vision_cfg = get_vision_model_config()
        version = vision_cfg.version
        logger.info(f"使用配置的版本: {version}")

    # 设置视觉模型版本
    set_vision_model_version(version)

    # 打印标题
    print_header(version)

    # 验证环境
    if not validate_environment(version):
        logger.error("环境验证失败，请检查依赖库")
        return 1

    # 打印配置
    print_config(version)

    # 检测是否为RTX 4090
    is_rtx4090 = detect_rtx4090()
    use_rtx4090_opt = is_rtx4090  # 自动启用优化

    if is_rtx4090:
        logger.info("检测到RTX 4090，启用优化配置")

    # 创建训练器
    logger.info(f"创建图像专家训练器（{version}）...")
    try:
        trainer = ExpertTrainer(
            expert_type='image',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt
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
        output_path = path_cfg.get_expert_weight_path('image', vision_version=version)
        print(f"LoRA权重已保存至: {output_path}")
        print(f"检查点目录: {path_cfg.get_checkpoint_path('image_expert')}")
        print()
        print("下一步:")
        print("  1. 可以使用该权重进行推理测试")
        print(f"  2. 如需对比实验，可训练另一版本：")
        if version == 'qwen2.5':
            print("     python scripts/run_with_env.py --env image_qwen3 --script scripts/training/train_image_expert.py")
        else:
            print("     python scripts/run_with_env.py --env image_qwen2.5 --script scripts/training/train_image_expert.py")
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
# # 使用Qwen2.5-VL（默认）
# python scripts/run_with_env.py --env image_qwen2.5 --script scripts/training/train_image_expert.py
#
# # 使用Qwen3-VL（对比实验）
# python scripts/run_with_env.py --env image_qwen3 --script scripts/training/train_image_expert.py
#
# 方法2：直接在对应环境中运行
# conda activate qwen_vision25
# python scripts/training/train_image_expert.py --version qwen2.5
#
# conda activate qwen_vision3
# python scripts/training/train_image_expert.py --version qwen3
#
# 注意事项：
# 1. 不同版本的训练需要在不同的Conda环境中进行
# 2. 权重会自动保存到对应版本的目录：
#    - Qwen2.5: lora_weights/experts/image_expert_qwen2.5/
#    - Qwen3: lora_weights/experts/image_expert_qwen3/
# 3. run_with_env.py会自动设置QWEN_VISION_VERSION环境变量