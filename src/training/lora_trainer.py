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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from config.settings import get_lora_config
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
                 debug_samples: bool = True):
        """
        初始化LoRA训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则从配置获取）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前5个训练样本（默认开启）
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

        # 获取LoRA配置
        self.lora_cfg = get_lora_config('conservative')

        # 根据模型版本确定target_modules
        self.target_modules = self._get_target_modules()

        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"LoRA配置: rank={self.lora_cfg.rank}, alpha={self.lora_cfg.alpha}")
        logger.info(f"Target modules: {self.target_modules}")

        # 打印配置
        self._print_training_config()

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

            # 加载模型
            model_kwargs = {
                'pretrained_model_name_or_path': self.base_model_path,
                'trust_remote_code': True,
                'device_map': 'auto',
                'torch_dtype': torch.bfloat16 if self.use_rtx4090_optimization else torch.float16,
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

            # 如果使用4bit量化，准备模型
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

            # 配置LoRA
            logger.info("配置LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_cfg.rank,
                lora_alpha=self.lora_cfg.alpha,
                lora_dropout=self.lora_cfg.dropout,
                target_modules=self.target_modules,
                bias="none",
            )

            # 应用LoRA
            self.model = get_peft_model(self.model, lora_config)

            # 打印可训练参数统计
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("=" * 80)
            logger.info("LoRA配置完成")
            logger.info("=" * 80)
            logger.info(f"可训练参数: {trainable_params:,} ({trainable_ratio:.2f}%)")
            logger.info(f"总参数: {total_params:,}")
            logger.info(f"LoRA Rank: {self.lora_cfg.rank}")
            logger.info(f"LoRA Alpha: {self.lora_cfg.alpha}")
            logger.info(f"LoRA Dropout: {self.lora_cfg.dropout}")
            logger.info(f"Target Modules: {self.target_modules}")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _save_weights(self):
        """
        保存LoRA权重
        """
        if self.model is None or self.tokenizer is None:
            logger.error("模型或tokenizer未初始化，无法保存")
            return

        try:
            logger.info("保存LoRA权重...")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # 保存LoRA权重
            self.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            logger.info(f"LoRA权重已保存至: {self.output_dir}")

        except Exception as e:
            logger.error(f"保存权重失败: {e}")
            import traceback
            logger.error(traceback.format_exc())


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