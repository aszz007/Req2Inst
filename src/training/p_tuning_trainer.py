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
import torch.utils.checkpoint
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
        self.encoder_hidden_size = 128  # 从64增加到128，提升稳定性
        self.prefix_projection = True

        # 学习率：1e-3为prefix encoder的标准配置
        # 根本原因修复（gradient checkpointing导致past_key_values被清空）后
        # 恢复为有效学习率，原2e-5为临时workaround
        original_lr = self.train_cfg.learning_rate
        self.train_cfg.learning_rate = 1e-3
        logger.info(f"学习率设置: {original_lr} -> {self.train_cfg.learning_rate} (prefix encoder标准配置)")

        # P-Tuning v2不支持gradient checkpointing
        # 原因: Qwen3的gradient checkpointing实现会强制past_key_values=None，
        # 导致prefix表示被丢弃，梯度无法回传至prefix encoder（grad_norm=0）
        # 显存优化通过MLP-level activation checkpointing（setup_model中实现）解决，
        # 该方法只触及FFN路径，不影响attention的past_key_values注入机制
        # disable_gradient_checkpointing=True确保TrainingArguments也不会
        # 通过Trainer二次调用gradient_checkpointing_enable()
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
        logger.info(f"Max序列长度: {self.train_cfg.max_seq_length}")
        logger.info("显存优化策略:")
        logger.info("  1. encoder_hidden_size=128 (平衡性能和稳定性)")
        logger.info("  2. 序列长度: 统一2048 (base_trainer管理)")
        logger.info("  3. 启用expandable_segments (减少碎片)")
        logger.info("  4. batch_size=1 + 梯度累积=128")
        logger.info("  5. 学习率=1e-3 (prefix encoder标准配置)")
        logger.info("  6. SDPA内存高效注意力 (base_trainer加载模型时启用)")
        logger.info("  7. MLP-level activation checkpointing (setup_model中启用)")
        logger.info("注意: P-Tuning v2不支持layer-level gradient checkpointing，已禁用")

        self._print_training_config()

    def _get_batch_config(self):
        """
        获取P-Tuning v2专用的batch配置

        P-Tuning v2不能使用gradient checkpointing，显存占用更大
        统一使用保守配置避免OOM：
        - 所有专家：batch=1, grad_accum=128（有效batch=128）

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        # 统一配置，避免text和image专家OOM
        return 1, 128

    def _enable_mlp_activation_checkpointing(self):
        """
        对每个decoder层的MLP子模块启用activation checkpointing

        原理：
        - Layer-level gradient checkpointing无法用于PrefixTuning（Qwen3会强制
          past_key_values=None，丢弃prefix注入，导致grad_norm=0）
        - MLP-level checkpointing仅触及FFN路径，attention的past_key_values完全不受影响
        - 效果：不保存gate/up/activated/down中间激活（约4GB @ 2048 tokens），
          backward时重新计算，以训练时间换显存

        对text/image专家（序列短）同样有效但收益较小；对uml/general专家（序列长）
        是解决OOM的关键。
        """
        try:
            base_model = self.model.get_base_model()
            if not (hasattr(base_model, 'model') and hasattr(base_model.model, 'layers')):
                logger.warning("无法访问decoder层列表，MLP activation checkpointing未启用")
                return

            num_layers = len(base_model.model.layers)
            patched = 0
            for layer in base_model.model.layers:
                if not hasattr(layer, 'mlp'):
                    continue

                original_forward = layer.mlp.forward

                def make_ckpt_forward(fwd):
                    def checkpointed_mlp_forward(hidden_states):
                        return torch.utils.checkpoint.checkpoint(
                            fwd, hidden_states, use_reentrant=False
                        )
                    return checkpointed_mlp_forward

                layer.mlp.forward = make_ckpt_forward(original_forward)
                patched += 1

            logger.info(f"MLP activation checkpointing已启用: {patched}/{num_layers}个decoder层")
            logger.info("效果: 不保存MLP中间激活（约节省4GB显存），backward时重新计算")

        except Exception as e:
            logger.warning(f"MLP activation checkpointing启用失败，将在不使用的情况下继续: {e}")
            import traceback
            logger.warning(traceback.format_exc())

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

            # use_cache must be False during training to avoid storing KV cache for every
            # layer, which wastes significant GPU memory. (base_trainer already guarantees
            # prepare_model_for_kbit_training is called without GC when
            # disable_gradient_checkpointing=True, so no explicit disable needed here.)
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
                logger.info("已禁用use_cache（训练时禁止以节省显存）")

            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.encoder_hidden_size,
                prefix_projection=self.prefix_projection
            )

            self.model = get_peft_model(self.model, peft_config)

            # Cast prefix encoder to match the base model dtype (bfloat16 or float16).
            # The PrefixEncoder MLP is initialized in float32 by default. Without this
            # cast, there is a dtype mismatch between the float32 prefix key/values and
            # the bfloat16 attention layers, which causes NaN in eval (where autocast
            # is not active) while training loss stays valid (autocast bridges the gap).
            model_dtype = torch.bfloat16 if self.use_rtx4090_optimization else torch.float16
            if hasattr(self.model, 'prompt_encoder'):
                self.model.prompt_encoder.to(model_dtype)
                logger.info(f"Prefix encoder已转换为{model_dtype}，与基础模型dtype保持一致")

            # NOTE: Gradient checkpointing is intentionally NOT enabled for PrefixTuning.
            # Qwen3's gradient checkpointing implementation forces `past_key_values=None`
            # in every decoder layer during the forward pass. PrefixTuning injects the
            # learned prefix representations via `past_key_values`, so enabling gradient
            # checkpointing silently discards all prefix key-values, making the prefix
            # encoder unreachable by gradients (grad_norm stays 0.0) and producing a
            # frozen eval_loss that never improves.
            # disable_gradient_checkpointing=True is set so TrainingArguments also does
            # not call gradient_checkpointing_enable() via the Trainer.
            #
            # Instead, activation memory is reduced via MLP-level checkpointing below,
            # which only touches the FFN path and leaves past_key_values untouched.
            self._enable_mlp_activation_checkpointing()

            # ===== 诊断性日志：检查模型各部分dtype =====
            logger.info("=" * 80)
            logger.info("模型Dtype诊断")
            logger.info("=" * 80)

            # 检查prompt_encoder的dtype
            if hasattr(self.model, 'prompt_encoder'):
                for name, param in self.model.prompt_encoder.named_parameters():
                    logger.info(f"  prompt_encoder.{name}: dtype={param.dtype}, shape={param.shape}")

            # 检查base model的部分层dtype
            base_model = self.model.get_base_model()
            if hasattr(base_model, 'model'):
                # Qwen3-8B结构
                if hasattr(base_model.model, 'layers') and len(base_model.model.layers) > 0:
                    first_layer = base_model.model.layers[0]
                    if hasattr(first_layer, 'self_attn'):
                        attn = first_layer.self_attn
                        if hasattr(attn, 'q_proj'):
                            logger.info(f"  base_model.layers[0].self_attn.q_proj.weight: dtype={attn.q_proj.weight.dtype}")

            logger.info(f"  目标dtype: {model_dtype} ({'bfloat16' if self.use_rtx4090_optimization else 'float16'})")
            logger.info("=" * 80)

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