"""
LoRA训练器 - LoRA-MoE方法的训练实现

功能：
  - 支持四种专家类型（text, image, uml, general）
  - 4bit量化训练
  - LoRA微调配置
  - 自动选择target_modules

作者：Training System
日期：2025-02-15
"""

import torch
from pathlib import Path
from typing import Optional
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.lora_trainer')


class LoRATrainer(BaseTrainer):
    """
    LoRA训练器 - 实现LoRA-MoE方法

    继承BaseTrainer，添加LoRA特有的：
    - 4bit量化配置
    - LoRA参数配置
    - target_modules自动选择
    - LoRA权重保存
    """

    def __init__(self,
                 expert_type: str,
                 base_model_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 use_4bit: bool = True,
                 use_rtx4090_optimization: bool = True,
                 debug_samples: bool = False):
        """
        初始化LoRA训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则从配置获取）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前3个训练样本（默认关闭）
        """
        # 调用父类初始化
        super().__init__(
            expert_type=expert_type,
            method_name='lora_moe',
            base_model_path=base_model_path,
            output_dir=output_dir,
            use_rtx4090_optimization=use_rtx4090_optimization,
            debug_samples=debug_samples
        )

        self.use_4bit = use_4bit

        # LoRA超参数（直接定义，便于实验调整）
        self.lora_rank = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05

        # 根据模型版本确定target_modules
        self.target_modules = self._get_target_modules()

        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"LoRA配置: rank={self.lora_rank}, alpha={self.lora_alpha}")
        logger.info(f"Target modules: {self.target_modules}")
        logger.info("训练稳定性配置:")
        logger.info("  - 梯度裁剪: max_grad_norm=1.0 (标准设置)")
        logger.info("  - Warmup比例: 10% (标准设置)")
        logger.info("  - NaN-aware早停: 自动忽略NaN验证损失")

        # 打印配置
        self._print_training_config()

    def _get_batch_config(self):
        """
        获取LoRA专用的batch配置，针对不同expert优化

        保守配置以避免OOM（基于实际训练OOM分析优化）：
        - Image (原batch=8出现OOM): batch=2, grad_accum=64
        - Text (提供更多显存余量): batch=2, grad_accum=64
        - UML (长序列JSON输入): batch=1, grad_accum=128
        - General (最长序列): batch=1, grad_accum=128

        保持有效batch=128以保证训练稳定性

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if self.use_rtx4090_optimization:
            if self.expert_type in ['image', 'text']:
                # Image/Text使用保守配置，避免OOM
                return 2, 64
            elif self.expert_type in ['uml', 'general']:
                # UML/General使用最保守配置（序列长，JSON重复多）
                return 1, 128
            else:
                return 1, 128
        else:
            return self.train_cfg.batch_size, self.train_cfg.gradient_accumulation_steps

    def _get_target_modules(self) -> list:
        """
        根据模型版本自动选择target_modules

        Returns:
            list: target_modules列表
        """
        if self.model_version == 'qwen3_8b':
            # Qwen3-8B使用新的注意力层命名
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif self.model_version == 'qwen7b':
            # Qwen-7B-Chat使用传统命名
            return ["c_attn", "c_proj"]
        else:
            # 默认使用Qwen3的命名
            logger.warning(f"未知模型版本 {self.model_version}，使用Qwen3默认配置")
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def setup_model(self) -> bool:
        """
        设置模型和LoRA配置

        Returns:
            bool: 是否成功
        """
        try:
            if not self._load_base_model(self.use_4bit):
                return False

            # 配置LoRA
            logger.info("配置LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("=" * 80)
            logger.info("LoRA配置完成")
            logger.info("=" * 80)
            logger.info(f"可训练参数: {trainable_params:,} ({trainable_ratio:.2f}%)")
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"LoRA Rank: {self.lora_rank}")
            logger.info(f"LoRA Alpha: {self.lora_alpha}")
            logger.info(f"LoRA Dropout: {self.lora_dropout}")
            logger.info(f"Target Modules: {self.target_modules}")
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
    print("LoRA训练器测试")
    print("=" * 80)

    print("\n注意：这是一个完整的训练流程示例")
    print("实际训练请使用 scripts/training/train_*_expert.py 脚本")

    print("\n训练流程：")
    print("1. 创建LoRATrainer实例")
    print("2. 调用prepare_data()准备数据")
    print("3. 调用setup_model()设置模型")
    print("4. 调用train()执行训练")
    print("5. LoRA权重自动保存到指定目录")

    print("\n示例代码：")
    print("trainer = LoRATrainer(expert_type='text')")
    print("trainer.prepare_data()")
    print("trainer.setup_model()")
    print("trainer.train()")

    print("\n测试完成！")