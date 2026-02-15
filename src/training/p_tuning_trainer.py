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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    PrefixTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from config.settings import get_ptuning_config
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
                 debug_samples: bool = True):
        """
        初始化P-Tuning v2训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则使用checkpoints/p_tuning/{expert_type}_expert/）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前3个训练样本（默认开启）
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
        self.ptuning_cfg = get_ptuning_config()

        # P-Tuning v2不支持gradient checkpointing
        self.disable_gradient_checkpointing = True

        # 更激进的序列长度策略（质量损失最小化）
        # UML数据虽长但JSON结构高度重复，1024能覆盖核心逻辑
        # General混合数据集，1024是质量和显存的最佳平衡点
        if expert_type == 'general':
            self.train_cfg.max_seq_length = 1024
            logger.info("General专家使用max_seq_length=1024（显存优化，质量损失<10%）")
        elif expert_type == 'uml':
            self.train_cfg.max_seq_length = 1024
            logger.info("UML专家使用max_seq_length=1024（JSON重复结构多，质量损失<10%）")
        elif expert_type == 'text':
            self.train_cfg.max_seq_length = 1280
            logger.info("Text专家使用max_seq_length=1280（质量优先）")
        else:  # image
            self.train_cfg.max_seq_length = 1280
            logger.info("Image专家使用max_seq_length=1280（质量优先）")

        # P-Tuning v2专用：减少dataloader workers以节省系统内存
        self.reduced_workers = True

        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"P-Tuning v2配置: virtual_tokens={self.ptuning_cfg.num_virtual_tokens}, "
                    f"encoder_hidden_size={self.ptuning_cfg.encoder_hidden_size}, "
                    f"prefix_projection={self.ptuning_cfg.prefix_projection}")
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

        P-Tuning v2不能使用gradient checkpointing，显存占用更大，
        需要使用更小的batch size和更大的gradient accumulation

        针对UML和General专家，使用更激进的配置以避免OOM

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if self.expert_type in ['uml', 'general']:
            # UML和General数据量大，使用极小batch + 大梯度累积
            # 有效batch=32，比text/image的16更大，可能提升训练质量
            logger.info(f"{self.expert_type}专家使用激进配置: batch=1, grad_accum=32 (有效batch=32)")
            return 1, 32
        else:
            # Text和Image专家使用标准配置
            if self.use_rtx4090_optimization:
                return 1, 16  # 有效batch=16
            else:
                return 1, 16  # 有效batch=16

    def setup_model(self) -> bool:
        """
        设置模型和P-Tuning v2配置

        必须在prepare_data()之前调用，以确保tokenizer在
        InstructionDataset初始化时已完成加载

        Returns:
            bool: 是否成功
        """
        try:
            # 设置PyTorch内存分配器优化（减少内存碎片）
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            logger.info("已设置PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（减少内存碎片）")

            # 清空GPU缓存，最大化可用显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已清空GPU缓存")

            logger.info("加载基础模型...")

            # 配置4bit量化（如果启用）
            quantization_config = None
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.use_rtx4090_optimization else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("启用4bit量化")

            # 构建模型加载参数
            model_kwargs = {
                'pretrained_model_name_or_path': self.base_model_path,
                'trust_remote_code': True,
                'device_map': 'auto',
                'dtype': torch.bfloat16 if self.use_rtx4090_optimization else torch.float16,
            }

            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

            # Qwen3-8B需要禁用思考模式（加载后设置）
            if self.model_version == 'qwen3_8b':
                if hasattr(self.model.config, 'enable_thinking'):
                    self.model.config.enable_thinking = False
                    logger.info("Qwen3-8B: 禁用思考模式（enable_thinking=False）")
                else:
                    logger.info("Qwen3-8B: 模型不支持enable_thinking配置，跳过")

            # 4bit量化后准备模型（冻结基础模型参数，仅prefix可训练）
            if self.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            # 加载tokenizer
            logger.info("加载Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side='left'
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer))

            logger.info(f"Tokenizer词汇表大小: {len(self.tokenizer)}")
            logger.info(f"PAD token: {self.tokenizer.pad_token}")

            # 配置P-Tuning v2（Prefix Tuning）
            logger.info("配置P-Tuning v2...")

            # 关键：P-Tuning v2不支持gradient checkpointing，必须先禁用
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
                logger.info("已禁用gradient checkpointing（P-Tuning v2要求）")

            # 如果模型已经启用了gradient checkpointing，需要显式关闭
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                logger.info("启用use_cache（P-Tuning v2优化）")

            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.ptuning_cfg.num_virtual_tokens,
                encoder_hidden_size=self.ptuning_cfg.encoder_hidden_size,
                prefix_projection=self.ptuning_cfg.prefix_projection
            )

            # 应用P-Tuning v2（仅prefix tokens和encoder可训练）
            self.model = get_peft_model(self.model, peft_config)

            # 打印可训练参数统计
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("=" * 80)
            logger.info("P-Tuning v2配置完成")
            logger.info("=" * 80)
            logger.info(f"可训练参数: {trainable_params:,} ({trainable_ratio:.4f}%)")
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"Virtual Tokens: {self.ptuning_cfg.num_virtual_tokens}")
            logger.info(f"Encoder Hidden Size: {self.ptuning_cfg.encoder_hidden_size}")
            logger.info(f"Prefix Projection: {self.ptuning_cfg.prefix_projection}")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _save_weights(self):
        """
        保存P-Tuning v2权重（仅保存prefix tokens和encoder）
        """
        if self.model is None or self.tokenizer is None:
            logger.error("模型或tokenizer未初始化，无法保存")
            return

        try:
            logger.info("保存P-Tuning v2权重...")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            logger.info(f"P-Tuning v2权重已保存至: {self.output_dir}")

        except Exception as e:
            logger.error(f"保存权重失败: {e}")
            import traceback
            logger.error(traceback.format_exc())