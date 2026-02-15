"""
Prompt Tuning文本专家训练脚本

功能：使用Prompt Tuning方法训练Text Expert，将文本需求转换为众包指令
环境：instruction_generator（transformers==4.57.0）
基础模型：Qwen3-8B（默认）
方法：Prompt Tuning（soft prompts / virtual tokens）
输出：checkpoints/prompt_tuning/text_expert/

对比实验说明：
  Prompt Tuning vs LoRA：验证不同参数高效微调方法的效果

使用方法：
  python scripts/run_with_env.py --env text --script scripts/training/prompt_tuning/train_text_expert.py

作者：Training System
日期：2025-02-15
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.prompt_tuning_trainer import PromptTuningTrainer
from src.utils.logger import get_logger

logger = get_logger('training.prompt_tuning.text_expert')


def detect_rtx4090() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return 'RTX 4090' in gpu_name or 'RTX 4090D' in gpu_name
    except:
        pass
    return False


def print_header():
    print("=" * 80)
    print(" " * 15 + "Prompt Tuning文本专家训练 (Text Expert Training)")
    print("=" * 80)
    print()


def validate_environment() -> bool:
    print("验证运行环境...")
    print("-" * 80)

    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
        v = transformers.__version__.split('.')
        major, minor = int(v[0]), int(v[1])
        if not (major > 4 or (major == 4 and minor >= 51)):
            logger.warning(f"推荐使用transformers>=4.51.0，当前: {transformers.__version__}")
    except ImportError:
        logger.error("未安装transformers库")
        return False

    try:
        import peft
        print(f"PEFT版本: {peft.__version__}")
    except ImportError:
        logger.error("未安装PEFT库，请运行: pip install peft")
        return False

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
    parser = argparse.ArgumentParser(description='使用Prompt Tuning训练文本专家')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    parser.add_argument('--no_rtx4090_opt', action='store_true',
                        help='禁用RTX 4090优化（默认：自动检测）')
    args = parser.parse_args()

    is_rtx4090 = detect_rtx4090()
    use_rtx4090_opt = is_rtx4090 and not args.no_rtx4090_opt

    if is_rtx4090:
        if use_rtx4090_opt:
            logger.info("检测到RTX 4090，启用优化配置")
        else:
            logger.info("检测到RTX 4090，但优化已禁用")

    print_header()

    if not validate_environment():
        logger.error("环境验证失败，请检查依赖库")
        return 1

    logger.info("创建Prompt Tuning文本专家训练器...")
    try:
        trainer = PromptTuningTrainer(
            expert_type='text',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt
        )
    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    # setup_model()必须在prepare_data()之前调用
    # InstructionDataset初始化时需要tokenizer已完成加载
    logger.info("设置模型和Prompt Tuning配置...")
    if not trainer.setup_model():
        logger.error("模型设置失败")
        return 1

    logger.info("准备训练数据...")
    if not trainer.prepare_data():
        logger.error("数据准备失败")
        return 1

    status = trainer.get_training_status()
    print(f"数据统计:")
    print(f"  - 训练样本: {status['train_samples']}")
    print(f"  - 验证样本: {status['val_samples']}")
    print()

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
        print(f"Prompt Tuning权重已保存至: {trainer.output_dir}")
        print(f"检查点目录: {trainer.output_dir / 'training_checkpoints'}")
        print()
        print("下一步:")
        print("  1. 可以使用该权重进行推理测试")
        print("  2. 继续训练其他专家（Image, UML, General）")
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