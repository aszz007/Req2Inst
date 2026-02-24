"""
LoRA-Single统一模型训练脚本

功能：训练单一LoRA模型处理所有输入类型（不使用MoE）
环境：instruction_generator（transformers==4.51.0）
基础模型：Qwen3-8B
数据集：text + image + uml混合数据集
输出：checkpoints/lora_single/unified_expert/

对比实验说明：
  - LoRA-Single vs LoRA-MoE：验证MoE架构的有效性
  - 使用相同的LoRA配置（rank=8），仅架构不同
  - 数据集与General Expert相同，但没有路由机制

使用方法：
  python scripts/training/lora_single/train_unified_expert.py

作者：Comparative Training System
日期：2025-02-15
"""

import sys
import argparse
from pathlib import Path

# 获取当前脚本的绝对路径并回溯到项目根目录
# 路径层级：scripts(1) / training(2) / lora_single(3) / 脚本(4)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 确保路径被正确插入
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.lora_trainer import LoRATrainer
from config.settings import (
    get_path_config,
    get_training_config,
    get_lora_config
)
from src.utils.logger import get_logger

logger = get_logger('training.lora_single.unified_expert')


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
    print(" " * 18 + "LoRA-Single统一模型训练 (Unified Expert Training)")
    print("=" * 80)
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
    parser = argparse.ArgumentParser(description='训练LoRA-Single统一模型')
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

    # 打印实验说明
    print("=" * 80)
    print("对比实验：LoRA-Single vs LoRA-MoE")
    print("=" * 80)
    print("目标：验证MoE架构相比单一模型的优势")
    print("配置：")
    print("  - 使用相同的LoRA超参数（rank=8, alpha=16）")
    print("  - 使用相同的训练数据集（text + image + uml）")
    print("  - 唯一区别：无MoE路由机制")
    print("预期：LoRA-MoE通过专家专业化达到更好的性能")
    print("=" * 80)
    print()

    # 创建训练器（复用ExpertTrainer，使用general类型）
    logger.info("创建LoRA-Single统一模型训练器...")
    try:
        # 使用general类型训练器（包含所有数据）
        # expert_type='general'确保加载text+image+uml混合数据集
        path_cfg = get_path_config()
        trainer = LoRATrainer(
            expert_type='general',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt,
            use_domain_templates=True
        )

        # 将输出路径和中间checkpoint路径统一指向lora_single目录
        # output_dir和checkpoint_dir必须同时更新，否则中间checkpoint会写入lora_moe路径
        trainer.output_dir = path_cfg.LORA_SINGLE_CKPT
        trainer.checkpoint_dir = path_cfg.LORA_SINGLE_CKPT / 'training_checkpoints'
        # 确保method_name正确，训练曲线文件名和日志标签会使用该值
        trainer.method_name = 'lora_single'

    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
    print(f"  - 数据来源: text + image + uml（混合数据集）")
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
        print(f"LoRA权重已保存至: {path_cfg.LORA_SINGLE_CKPT}")
        print()
        print("下一步:")
        print("  1. 使用该权重进行推理测试")
        print("  2. 与LoRA-MoE进行性能对比实验")
        print("  3. 分析单一模型 vs MoE架构的差异")
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