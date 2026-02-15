"""
Full Fine-tuning训练器 - 使用高rank LoRA模拟全参数微调

功能：
  - 支持四种专家类型（text, image, uml, general）
  - 使用rank=64的LoRA模拟全参数微调
  - 可选4bit量化训练
  - 覆盖attention + FFN层
  - 继承BaseTrainer的全部训练优化策略：
      - RTX 4090自动检测与优化
      - 自适应早停机制
      - Cosine学习率调度 + Warmup
      - 步级验证策略
      - 训练曲线可视化
      - 梯度检查点
      - 权重衰减

说明：
  由于RTX 4090的24GB显存对Qwen3-8B真正的全参数微调不够
  （需要约32GB = 8GB模型 + 24GB训练状态），
  我们使用高rank LoRA (rank=64) 模拟全参数微调，优势：
  - 更稳定（避免OOM）
  - 训练更快
  - 效果接近全参数微调
  - 推理时可灵活merge/unmerge

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
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from config.settings import get_full_finetuning_config
from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.full_finetuning_trainer')


class FullFineTuningTrainer(BaseTrainer):
    """
    Full Fine-tuning训练器 - 使用高rank LoRA模拟全参数微调

    继承BaseTrainer，添加Full Fine-tuning特有的：
    - 可选4bit量化配置
    - 高rank LoRA配置（rank=64）
    - 覆盖更多层（attention + FFN）
    - 更保守的训练参数
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
                 debug_samples: bool = True):
        """
        初始化Full Fine-tuning训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则使用checkpoints/full_finetuning/{expert_type}_expert/）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前3个训练样本（默认开启）
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
        self.full_ft_cfg = get_full_finetuning_config()

        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"Full Fine-tuning配置: rank={self.full_ft_cfg.lora_rank}, "
                    f"alpha={self.full_ft_cfg.lora_alpha}")
        logger.info(f"Target modules: {self.full_ft_cfg.target_modules}")

        self._print_training_config()

    def _get_batch_config(self):
        """
        获取Full Fine-tuning专用的batch配置

        Full Fine-tuning由于参数量更大，显存占用比标准LoRA更高，
        因此使用更小的batch size

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if self.use_rtx4090_optimization:
            # RTX 4090优化配置：batch_size=4, gradient_accumulation=4, 有效batch=16
            return 4, 4
        else:
            # 非优化配置：使用Full Fine-tuning配置
            return self.full_ft_cfg.batch_size, self.full_ft_cfg.gradient_accumulation_steps

    def setup_model(self) -> bool:
        """
        设置模型和高rank LoRA配置

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

            # Qwen3-8B需要禁用思考模式
            if self.model_version == 'qwen3_8b':
                model_kwargs['enable_thinking'] = False
                logger.info("Qwen3-8B: 禁用思考模式（enable_thinking=False）")

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

            # 4bit量化后准备模型（冻结基础模型参数，仅LoRA可训练）
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

            # 配置高rank LoRA（模拟全参数微调）
            logger.info("配置高rank LoRA（模拟全参数微调）...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.full_ft_cfg.lora_rank,  # rank=64
                lora_alpha=self.full_ft_cfg.lora_alpha,  # alpha=128
                lora_dropout=self.full_ft_cfg.lora_dropout,
                target_modules=self.full_ft_cfg.target_modules,  # attention + FFN层
                bias="none",
            )

            # 应用高rank LoRA
            self.model = get_peft_model(self.model, peft_config)

            # 打印可训练参数统计
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("=" * 80)
            logger.info("Full Fine-tuning配置完成（高rank LoRA模拟）")
            logger.info("=" * 80)
            logger.info(f"可训练参数: {trainable_params:,} ({trainable_ratio:.2f}%)")
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"LoRA Rank: {self.full_ft_cfg.lora_rank} (约为标准LoRA的8倍)")
            logger.info(f"LoRA Alpha: {self.full_ft_cfg.lora_alpha}")
            logger.info(f"LoRA Dropout: {self.full_ft_cfg.lora_dropout}")
            logger.info(f"Target Modules: {self.full_ft_cfg.target_modules}")
            logger.info("说明：使用高rank LoRA覆盖attention+FFN层，接近全参数微调效果")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _save_weights(self):
        """
        保存Full Fine-tuning权重（高rank LoRA权重）
        """
        if self.model is None or self.tokenizer is None:
            logger.error("模型或tokenizer未初始化，无法保存")
            return

        try:
            logger.info("保存Full Fine-tuning权重...")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            logger.info(f"Full Fine-tuning权重已保存至: {self.output_dir}")

        except Exception as e:
            logger.error(f"保存权重失败: {e}")
            import traceback
            logger.error(traceback.format_exc())


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