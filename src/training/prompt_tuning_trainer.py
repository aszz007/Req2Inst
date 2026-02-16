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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from config.settings import get_prompt_tuning_config
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
        self.prompt_cfg = get_prompt_tuning_config()

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
        logger.info(f"Prompt Tuning配置: virtual_tokens={self.prompt_cfg.num_virtual_tokens}, "
                    f"init={self.prompt_cfg.prompt_tuning_init}")

        self._print_training_config()

    def _print_training_config(self):
        """打印Prompt Tuning训练配置（包括专家特定的batch配置）"""
        batch_size, gradient_accumulation_steps = self._get_batch_config()

        logger.info("=" * 80)
        logger.info("Prompt Tuning训练配置")
        logger.info("=" * 80)
        logger.info(f"专家类型: {self.expert_type}")
        logger.info(f"微调方法: {self.method_name}")
        logger.info(f"基础模型: {self.base_model_path}")
        logger.info(f"模型版本: {self.model_version}")
        logger.info(f"4bit量化: {self.use_4bit}")

        if self.use_rtx4090_optimization:
            if self.expert_type in ['uml', 'general']:
                logger.info(f"批次大小: {batch_size} (RTX 4090优化 - {self.expert_type}专家长序列配置)")
                logger.info(f"梯度累积: {gradient_accumulation_steps} (RTX 4090优化 - {self.expert_type}专家长序列配置)")
            else:
                logger.info(f"批次大小: {batch_size} (RTX 4090优化)")
                logger.info(f"梯度累积: {gradient_accumulation_steps} (RTX 4090优化)")
        else:
            logger.info(f"批次大小: {batch_size}")
            logger.info(f"梯度累积: {gradient_accumulation_steps}")

        logger.info(f"有效批次: {batch_size * gradient_accumulation_steps}")
        logger.info(f"训练轮数: {self.train_cfg.num_epochs}")
        logger.info(f"学习率: {self.train_cfg.learning_rate}")
        logger.info(f"最大序列长度: {self.train_cfg.max_seq_length}")
        logger.info("=" * 80)

    def _get_batch_config(self):
        """
        获取Prompt Tuning专用的batch配置

        Prompt Tuning由于需要存储virtual tokens的梯度，显存占用较高
        根据实际显存监控优化：
        - Text (预计14-16GB): batch=2, grad_accum=64
        - Image (预计12-14GB): batch=4, grad_accum=32
        - UML (预计16-18GB): batch=2, grad_accum=64
        - General (预计18-20GB): batch=1, grad_accum=128

        保持有效batch=128以保证训练稳定性

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if self.use_rtx4090_optimization:
            if self.expert_type == 'image':
                return 4, 32
            elif self.expert_type in ['text', 'uml']:
                return 2, 64
            elif self.expert_type == 'general':
                return 1, 128
            else:
                return 2, 64
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

            # 4bit量化后准备模型（冻结基础模型参数，仅virtual tokens可训练）
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

            # 配置Prompt Tuning
            logger.info("配置Prompt Tuning...")
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.prompt_cfg.num_virtual_tokens,
                prompt_tuning_init=PromptTuningInit.RANDOM,
                tokenizer_name_or_path=str(self.base_model_path)
            )

            # 应用Prompt Tuning（仅virtual token embeddings可训练）
            self.model = get_peft_model(self.model, peft_config)

            # 打印可训练参数统计
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("=" * 80)
            logger.info("Prompt Tuning配置完成")
            logger.info("=" * 80)
            logger.info(f"可训练参数: {trainable_params:,} ({trainable_ratio:.4f}%)")
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"Virtual Tokens: {self.prompt_cfg.num_virtual_tokens}")
            logger.info(f"初始化方式: {self.prompt_cfg.prompt_tuning_init}")
            logger.info("=" * 80)

            # 针对长序列专家清理GPU缓存
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

    def _save_weights(self):
        """
        保存Prompt Tuning权重（仅保存virtual token embeddings）
        """
        if self.model is None or self.tokenizer is None:
            logger.error("模型或tokenizer未初始化，无法保存")
            return

        try:
            logger.info("保存Prompt Tuning权重...")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            logger.info(f"Prompt Tuning权重已保存至: {self.output_dir}")

        except Exception as e:
            logger.error(f"保存权重失败: {e}")
            import traceback
            logger.error(traceback.format_exc())