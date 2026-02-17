"""
P-Tuning v2训练器 - P-Tuning v2方法的训练实现（对比实验）

功能：
  - 支持四种专家类型（text, image, uml, general）
  - 可选4bit量化训练
  - P-Tuning v2配置（Prefix Tuning - 通过MLP编码器学习前缀表示）
  - 继承BaseTrainer的全部训练优化策略：
      - RTX 4090自动检测与优化
      - 自适应早停机制（patience根据专家类型和数据量）
      - Cosine学习率调度 + Warmup
      - 步级验证策略（UML每30步，其他每50步）
      - 训练曲线可视化
      - 梯度检查点
      - 权重衰减

作者：Training System
日期：2025-02-15
"""

import torch
from typing import Optional
from peft import (
    PrefixTuningConfig,
    get_peft_model,
    TaskType,
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.p_tuning_trainer')


class PTuningTrainer(BaseTrainer):
    """
    P-Tuning v2训练器 - 实现P-Tuning v2对比实验方法

    继承BaseTrainer，添加P-Tuning v2特有的：
    - 可选4bit量化配置
    - Prefix Tuning（Virtual Token + MLP编码器）参数配置
    - P-Tuning v2权重保存

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
        初始化P-Tuning v2训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则使用checkpoints/p_tuning/{expert_type}_expert/）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前3个训练样本（默认关闭）
        """
        super().__init__(
            expert_type=expert_type,
            method_name='p_tuning',
            base_model_path=base_model_path,
            output_dir=output_dir,
            use_rtx4090_optimization=use_rtx4090_optimization,
            debug_samples=debug_samples
        )

        self.use_4bit = use_4bit

        # P-Tuning v2超参数（直接定义，便于实验调整）
        self.num_virtual_tokens = 20
        self.encoder_hidden_size = 64
        self.prefix_projection = True

        # ===== NaN防护配置 =====
        # P-Tuning v2容易产生NaN验证损失，需要更保守的配置

        # 1. 降低学习率（从0.0002降到0.00005，降低75%）
        #    原因：P-Tuning只训练~500万参数，对学习率极其敏感
        original_lr = self.train_cfg.learning_rate
        self.train_cfg.learning_rate = 5e-5
        logger.warning("=" * 80)
        logger.warning("P-Tuning v2 NaN防护配置已启用")
        logger.warning("=" * 80)
        logger.warning(f"学习率调整: {original_lr} → {self.train_cfg.learning_rate} (降低75%)")
        logger.warning("原因: P-Tuning v2参数少，学习率过大会导致NaN验证损失")

        # 2. 检查encoder_hidden_size（如果太小会警告）
        if self.encoder_hidden_size < 128:
            logger.warning(f"Encoder Hidden Size={self.encoder_hidden_size}偏小")
            logger.warning("如果训练中出现NaN，建议增加到128或256")

        logger.warning("其他NaN防护措施:")
        logger.warning("  - 严格梯度裁剪: max_grad_norm=0.5 (已在base_trainer中启用)")
        logger.warning("  - NaN-aware早停: 忽略NaN值，只基于有效loss判断")
        logger.warning("  - 增加warmup: 20%步数用于模型稳定")
        logger.warning("  - 数据质量检查: 过滤无效样本（有效labels<5）")
        logger.warning("=" * 80)

        # P-Tuning v2不支持gradient checkpointing
        self.disable_gradient_checkpointing = True

        # P-Tuning v2不支持load_best_model_at_end（会导致embedding shape mismatch）
        self.disable_load_best_model = True

        # 使用统一的序列长度管理（由base_trainer的_get_max_seq_length()决定）
        self.train_cfg.max_seq_length = self._get_max_seq_length()
        logger.info(f"Max序列长度: {self.train_cfg.max_seq_length} (由base_trainer统一管理)")

        # P-Tuning v2专用：减少dataloader workers以节省系统内存
        self.reduced_workers = True

        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"P-Tuning v2配置: virtual_tokens={self.num_virtual_tokens}, "
                    f"encoder_hidden_size={self.encoder_hidden_size}, "
                    f"prefix_projection={self.prefix_projection}")
        logger.info(f"Max序列长度: {self.train_cfg.max_seq_length} (激进显存优化)")
        logger.info("显存优化策略:")
        logger.info("  1. encoder_hidden_size=64 (50%内存减少)")
        logger.info("  2. 激进序列长度 (UML/General=1024)")
        logger.info("  3. 启用expandable_segments (减少碎片)")
        logger.info("  4. 极小batch_size=1 + 大梯度累积")
        logger.info("注意: P-Tuning v2不支持gradient checkpointing，已禁用")
        if self.expert_type in ['uml', 'general']:
            logger.warning("如果仍OOM，可考虑: encoder_hidden_size→32 或 num_virtual_tokens→15")

        self._print_training_config()

    def _get_batch_config(self):
        """
        获取P-Tuning v2专用的batch配置

        P-Tuning v2不能使用gradient checkpointing，显存占用更大
        根据实际显存监控调整：
        - Text/Image: 可适当增大batch
        - UML/General: 保持保守配置

        针对不同expert优化：
        - Text (预计14-16GB): batch=2, grad_accum=64
        - Image (预计12-14GB): batch=4, grad_accum=32
        - UML (预计16-18GB): batch=1, grad_accum=128
        - General (预计18-20GB): batch=1, grad_accum=128

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if self.expert_type == 'image':
            return 4, 32
        elif self.expert_type == 'text':
            return 2, 64
        elif self.expert_type in ['uml', 'general']:
            return 1, 128
        else:
            return 1, 128

    def setup_model(self) -> bool:
        """
        设置模型和P-Tuning v2配置

        必须在prepare_data()之前调用，以确保tokenizer在
        InstructionDataset初始化时已完成加载

        Returns:
            bool: 是否成功
        """
        try:
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            logger.info("已设置PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（减少内存碎片）")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已清空GPU缓存")

            if not self._load_base_model(self.use_4bit):
                return False

            # 配置P-Tuning v2（Prefix Tuning）
            logger.info("配置P-Tuning v2...")

            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
                logger.info("已禁用gradient checkpointing（P-Tuning v2要求）")

            if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                logger.info("启用use_cache（P-Tuning v2优化）")

            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.encoder_hidden_size,
                prefix_projection=self.prefix_projection
            )

            self.model = get_peft_model(self.model, peft_config)

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("=" * 80)
            logger.info("P-Tuning v2配置完成")
            logger.info("=" * 80)
            logger.info(f"可训练参数: {trainable_params:,} ({trainable_ratio:.4f}%)")
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"Virtual Tokens: {self.num_virtual_tokens}")
            logger.info(f"Encoder Hidden Size: {self.encoder_hidden_size}")
            logger.info(f"Prefix Projection: {self.prefix_projection}")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False