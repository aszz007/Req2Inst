"""
Prompt Tuning训练器 - Prompt Tuning方法的训练实现（对比实验）

功能：
  - 支持四种专家类型（text, image, uml, general）
  - 可选4bit量化训练
  - Prompt Tuning配置（soft prompts / virtual tokens）
  - 继承BaseTrainer的全部训练优化策略：
      - RTX 4090自动检测与优化
      - 自适应早停机制（patience根据专家类型和数据量）
      - Cosine学习率调度 + Warmup
      - 步级验证策略（UML每30步，其他每50步）
      - 训练曲线可视化
      - 梯度检查点
      - 权重衰减
  - 长序列专家优化（UML和General）：
      - 更小的batch size（2 vs 4）以避免OOM
      - 更多的梯度累积步数（8 vs 4）保持有效batch size
      - 较少的dataloader workers（4 vs 8）节省内存
      - 多阶段GPU缓存清理

作者：Training System
日期：2025-02-15
"""

import torch
from typing import Optional
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    get_peft_model,
    TaskType,
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.prompt_tuning_trainer')


class PromptTuningTrainer(BaseTrainer):
    """
    Prompt Tuning训练器 - 实现Prompt Tuning对比实验方法

    继承BaseTrainer，添加Prompt Tuning特有的：
    - 可选4bit量化配置
    - Soft Prompt（Virtual Token）参数配置
    - Prompt Tuning权重保存

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
        初始化Prompt Tuning训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则使用checkpoints/prompt_tuning/{expert_type}_expert/）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前3个训练样本（默认关闭）
        """
        super().__init__(
            expert_type=expert_type,
            method_name='prompt_tuning',
            base_model_path=base_model_path,
            output_dir=output_dir,
            use_rtx4090_optimization=use_rtx4090_optimization,
            debug_samples=debug_samples
        )

        self.use_4bit = use_4bit

        # Prompt Tuning超参数（直接定义，便于实验调整）
        self.num_virtual_tokens = 10

        # ===== NaN防护配置 =====
        # Prompt Tuning也容易产生NaN验证损失（只训练virtual token embeddings）

        # 降低学习率以防止NaN
        original_lr = self.train_cfg.learning_rate
        self.train_cfg.learning_rate = 5e-5
        logger.warning("=" * 80)
        logger.warning("Prompt Tuning NaN防护配置已启用")
        logger.warning("=" * 80)
        logger.warning(f"学习率调整: {original_lr} → {self.train_cfg.learning_rate} (降低75%)")
        logger.warning("原因: Prompt Tuning只训练virtual tokens，学习率过大会导致NaN")
        logger.warning("其他防护: 严格梯度裁剪(0.5) + NaN-aware早停 + 20% warmup")
        logger.warning("=" * 80)

        # Prompt Tuning不支持load_best_model_at_end（会导致embedding shape mismatch）
        self.disable_load_best_model = True

        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"Prompt Tuning配置: virtual_tokens={self.num_virtual_tokens}, "
                    f"init=RANDOM")

        self._print_training_config()

    def _get_batch_config(self):
        """
        获取Prompt Tuning专用的batch配置

        保守配置以避免OOM（Prompt Tuning需要存储virtual tokens梯度）：
        - Image/Text: batch=2, grad_accum=64
        - UML: batch=1, grad_accum=128
        - General: batch=1, grad_accum=128

        保持有效batch=128以保证训练稳定性

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if self.use_rtx4090_optimization:
            if self.expert_type in ['image', 'text']:
                return 2, 64
            elif self.expert_type in ['uml', 'general']:
                return 1, 128
            else:
                return 1, 128
        else:
            return self.train_cfg.batch_size, self.train_cfg.gradient_accumulation_steps

    def setup_model(self) -> bool:
        """
        设置模型和Prompt Tuning配置

        必须在prepare_data()之前调用，以确保tokenizer在
        InstructionDataset初始化时已完成加载

        Returns:
            bool: 是否成功
        """
        try:
            if not self._load_base_model(self.use_4bit):
                return False

            # 配置Prompt Tuning
            logger.info("配置Prompt Tuning...")
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.num_virtual_tokens,
                prompt_tuning_init=PromptTuningInit.RANDOM,
                tokenizer_name_or_path=str(self.base_model_path)
            )

            self.model = get_peft_model(self.model, peft_config)

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("=" * 80)
            logger.info("Prompt Tuning配置完成")
            logger.info("=" * 80)
            logger.info(f"可训练参数: {trainable_params:,} ({trainable_ratio:.4f}%)")
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"Virtual Tokens: {self.num_virtual_tokens}")
            logger.info(f"初始化方式: RANDOM")
            logger.info("=" * 80)

            if self.expert_type in ['uml', 'general']:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"已清理GPU缓存（{self.expert_type}专家长序列优化）")

            return True

        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False