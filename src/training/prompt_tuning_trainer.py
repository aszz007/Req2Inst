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
                 debug_samples: bool = True):
        """
        初始化Prompt Tuning训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则使用checkpoints/prompt_tuning/{expert_type}_expert/）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前5个训练样本（默认开启）
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

        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"Prompt Tuning配置: virtual_tokens={self.prompt_cfg.num_virtual_tokens}, "
                    f"init={self.prompt_cfg.prompt_tuning_init}")

        self._print_training_config()

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
                'torch_dtype': torch.bfloat16 if self.use_rtx4090_optimization else torch.float16,
            }

            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

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