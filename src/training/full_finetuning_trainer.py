"""
Full Fine-tuning训练器 - 高rank LoRA配置（Full Fine-tuning对比基线）

功能：
  - 支持四种专家类型（text, image, uml, general）
  - 使用rank=16的高质量LoRA作为Full Fine-tuning对比基线
  - 可选4bit量化训练
  - 仅覆盖attention层（节省显存）
  - 继承BaseTrainer的全部训练优化策略：
      - RTX 4090自动检测与优化
      - 自适应早停机制
      - Cosine学习率调度 + Warmup
      - 步级验证策略
      - 训练曲线可视化
      - 梯度检查点
      - 权重衰减

显存策略（RTX 4090 24GB）：
  - LoRA Rank: 16（高质量配置）
  - LoRA Alpha: 32（标准2倍rank配置）
  - Target Modules: 仅attention层（节省显存）
  - Max Seq Length: 768（由_get_max_seq_length统一管理）
  - Batch Size: 按expert类型自适应
  - Gradient Checkpointing: 由TrainingArguments统一启用
  - 4bit量化: 启用（模型占用约4-5GB）
  - 内存碎片优化: 启用expandable_segments

作者：Training System
日期：2025-02-15
"""

import torch
from typing import Optional
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.full_finetuning_trainer')


class FullFineTuningTrainer(BaseTrainer):
    """
    Full Fine-tuning训练器 - 使用标准rank LoRA（显存优化版）

    继承BaseTrainer，添加Full Fine-tuning特有的：
    - 可选4bit量化配置
    - 标准rank LoRA配置（rank=8）
    - 覆盖attention层（移除FFN节省显存）
    - 激进的显存优化策略
    - Full Fine-tuning权重保存

    注意：调用顺序必须为 setup_model() -> prepare_data() -> train()
    prepare_data()创建InstructionDataset时需要tokenizer已完成初始化
    """

    def __init__(self,
                 expert_type: str,
                 base_model_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 use_4bit: bool = True,
                 use_rtx4090_optimization: bool = True,
                 debug_samples: bool = False):
        """
        初始化Full Fine-tuning训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则使用checkpoints/full_finetuning/{expert_type}_expert/）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前3个训练样本（默认关闭）
        """
        super().__init__(
            expert_type=expert_type,
            method_name='full_finetuning',
            base_model_path=base_model_path,
            output_dir=output_dir,
            use_rtx4090_optimization=use_rtx4090_optimization,
            debug_samples=debug_samples
        )

        self.use_4bit = use_4bit

        # Full Fine-tuning LoRA超参数（直接定义，便于实验调整）
        # 使用较小rank以适配24GB显存，与LoRA-MoE的rank=8区分作为对比基线
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        # 仅attention层，移除FFN以节省显存
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Full Fine-tuning显存优化：减少dataloader workers
        self.reduced_workers = True

        # 使用统一的序列长度管理（由base_trainer的_get_max_seq_length()决定）
        self.train_cfg.max_seq_length = self._get_max_seq_length()

        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"Full Fine-tuning配置: rank={self.lora_rank}, alpha={self.lora_alpha}")
        logger.info(f"Max seq length: {self.train_cfg.max_seq_length}")
        logger.info(f"Target modules: {self.target_modules}")
        logger.info("训练稳定性配置:")
        logger.info("  - 梯度裁剪: max_grad_norm=0.8 (较严格设置)")
        logger.info("  - Warmup比例: 10% (标准设置)")
        logger.info("  - NaN-aware早停: 自动忽略NaN验证损失")

        self._print_training_config()

    def _get_batch_config(self):
        """
        获取Full Fine-tuning专用的batch配置

        根据实际显存使用情况优化：
        - General (16GB使用): batch=2, grad_accum=64
        - Text (类似General): batch=2, grad_accum=64
        - UML (中等长度): batch=2, grad_accum=64
        - Image (13GB使用，数据最短): batch=4, grad_accum=32

        保持有效batch=128以保证训练稳定性

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if self.use_rtx4090_optimization:
            # 根据专家类型优化batch配置
            if self.expert_type == 'image':
                # Image数据最短（~500 tokens），显存占用最少，可用更大batch
                return 4, 32
            elif self.expert_type in ['text', 'uml', 'general']:
                # Text/UML/General使用中等batch配置
                return 2, 64
            else:
                # 默认保守配置
                return 1, 128
        else:
            # 非优化配置：使用保守设置
            return 1, 128

    def setup_model(self) -> bool:
        """
        设置模型和Full Fine-tuning LoRA配置

        必须在prepare_data()之前调用，以确保tokenizer在
        InstructionDataset初始化时已完成加载

        Returns:
            bool: 是否成功
        """
        try:
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            logger.info("已设置PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

            if torch.cuda.is_available():
                for i in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info("已三重清空GPU缓存")

                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"[初始状态] GPU显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 总计={total:.2f}GB")

            if not self._load_base_model(self.use_4bit):
                return False

            if torch.cuda.is_available() and self.use_4bit:
                allocated = torch.cuda.memory_allocated() / 1024**3
                if allocated > 8.0:
                    logger.error(f"警告: 4bit量化可能未生效，模型占用{allocated:.2f}GB，预期应<6GB")
                else:
                    logger.info(f"4bit量化正常: 模型占用{allocated:.2f}GB")

            # 配置LoRA（高rank版本，作为Full Fine-tuning对比基线）
            logger.info(f"配置Full Fine-tuning LoRA（rank={self.lora_rank}）...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias="none",
            )

            self.model = get_peft_model(self.model, peft_config)

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("=" * 80)
            logger.info("Full Fine-tuning配置完成")
            logger.info("=" * 80)
            logger.info(f"可训练参数: {trainable_params:,} ({trainable_ratio:.2f}%)")
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"LoRA Rank: {self.lora_rank}")
            logger.info(f"LoRA Alpha: {self.lora_alpha}")
            logger.info(f"LoRA Dropout: {self.lora_dropout}")
            logger.info(f"Target Modules: {self.target_modules}")
            logger.info(f"Max Seq Length: {self.train_cfg.max_seq_length}")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False



# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("Full Fine-tuning训练器测试")
    print("=" * 80)

    print("\n注意：这是一个完整的训练流程示例")
    print("实际训练请使用 scripts/training/full_finetuning/train_*_expert.py 脚本")

    print("\n训练流程：")
    print("1. 创建FullFineTuningTrainer实例")
    print("2. 调用setup_model()设置模型")
    print("3. 调用prepare_data()准备数据")
    print("4. 调用train()执行训练")
    print("5. 权重自动保存到指定目录")

    print("\n示例代码：")
    print("trainer = FullFineTuningTrainer(expert_type='text')")
    print("trainer.setup_model()")
    print("trainer.prepare_data()")
    print("trainer.train()")

    print("\n测试完成！")