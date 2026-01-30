"""
专家训练器 - 使用PEFT + Trainer实现LoRA微调
功能：
  - 支持四种专家类型（text, image, uml, general）
  - 集成Hugging Face Trainer
  - 支持LoRA微调
  - 支持梯度累积
  - 自动保存检查点和最终权重

作者：Training System
日期：2025-01-30
"""

import os
import torch
from pathlib import Path
from typing import Optional, List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from config.settings import (
    get_path_config,
    get_lora_config,
    get_training_config,
    get_device_config
)
from src.training.data_loader import (
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,
    InstructionDataset,
    split_dataset_for_expert
)
from src.utils.logger import get_logger
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate

logger = get_logger('training.expert_trainer')


class ExpertTrainer:
    """
    专家训练器 - 统一的LoRA微调接口

    支持四种专家类型：
    - text: 文本需求 → 众包指令
    - image: 图像描述 → 标注指令
    - uml: UML JSON → 业务逻辑指令
    - general: 通用兜底专家
    """

    def __init__(self,
                 expert_type: str,
                 base_model_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 use_4bit: bool = True):
        """
        初始化训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则从配置获取）
            use_4bit: 是否使用4bit量化训练
        """
        # 验证专家类型
        valid_types = ['text', 'image', 'uml', 'general']
        if expert_type not in valid_types:
            raise ValueError(f"不支持的专家类型: {expert_type}，支持: {valid_types}")

        self.expert_type = expert_type
        self.use_4bit = use_4bit

        # 获取配置
        self.path_cfg = get_path_config()
        self.lora_cfg = get_lora_config('conservative')  # 使用保守配置
        self.train_cfg = get_training_config()
        self.device_cfg = get_device_config()

        # 设置基础模型路径
        if base_model_path:
            self.base_model_path = base_model_path
        else:
            # 根据专家类型选择模型
            if expert_type in ['text', 'general']:
                self.base_model_path = str(self.path_cfg.QWEN_7B_CHAT_PATH)
            else:  # image, uml
                # 视觉专家使用当前配置的视觉模型版本
                self.base_model_path = str(self.path_cfg.get_vision_model_path())

        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.path_cfg.get_expert_weight_path(f"{expert_type}_expert")

        # 设置检查点目录
        self.checkpoint_dir = self.path_cfg.get_checkpoint_path(f"{expert_type}_expert")

        # 初始化模型和数据相关属性
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        logger.info(f"初始化{expert_type}专家训练器")
        logger.info(f"基础模型: {self.base_model_path}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"4bit量化: {use_4bit}")

    def prepare_data(self) -> bool:
        """
        准备训练数据

        Returns:
            bool: 是否准备成功
        """
        try:
            logger.info("准备训练数据...")

            # 根据专家类型加载数据
            if self.expert_type == 'text':
                loader = TextDatasetLoader()
                raw_data = loader.load_csv_files()
            elif self.expert_type == 'image':
                loader = ImageDatasetLoader()
                raw_data = loader.load_csv_file()
            elif self.expert_type == 'uml':
                loader = UMLDatasetLoader()
                raw_data = loader.load_csv_file()
            elif self.expert_type == 'general':
                # 通用专家使用所有数据
                text_loader = TextDatasetLoader()
                image_loader = ImageDatasetLoader()
                uml_loader = UMLDatasetLoader()
                raw_data = (
                        text_loader.load_csv_files() +
                        image_loader.load_csv_file() +
                        uml_loader.load_csv_file()
                )
            else:
                raise ValueError(f"未知的专家类型: {self.expert_type}")

            if not raw_data:
                logger.error("没有加载到数据")
                return False

            logger.info(f"原始数据量: {len(raw_data)}条")

            # 划分数据集
            train_data, val_data, test_data = split_dataset_for_expert(
                raw_data,
                self.expert_type
            )

            logger.info(f"数据集划分完成:")
            logger.info(f"  训练集: {len(train_data)}条")
            logger.info(f"  验证集: {len(val_data)}条")
            logger.info(f"  测试集: {len(test_data)}条")

            # 加载tokenizer（需要在创建Dataset之前）
            logger.info("加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side='left'
            )

            # 设置特殊tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = '<|endoftext|>'
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = '<|im_end|>'

            # 创建PyTorch Dataset
            self.train_dataset = InstructionDataset(
                train_data,
                self.tokenizer,
                max_length=self.train_cfg.max_seq_length
            )
            self.val_dataset = InstructionDataset(
                val_data,
                self.tokenizer,
                max_length=self.train_cfg.max_seq_length
            )
            self.test_dataset = InstructionDataset(
                test_data,
                self.tokenizer,
                max_length=self.train_cfg.max_seq_length
            )

            logger.info("数据准备完成")
            return True

        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            return False

    def setup_model(self) -> bool:
        """
        设置模型和LoRA

        Returns:
            bool: 是否设置成功
        """
        try:
            logger.info("设置模型和LoRA配置...")

            # 4bit量化配置
            if self.use_4bit and self.device_cfg.device == "cuda":
                logger.info("使用4bit量化训练")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None

            # 加载基础模型
            logger.info(f"加载基础模型: {self.base_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.use_4bit else None,
                low_cpu_mem_usage=True
            )

            # 为4bit训练准备模型
            if self.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            # 配置LoRA
            logger.info("配置LoRA参数...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_cfg.rank,
                lora_alpha=self.lora_cfg.alpha,
                lora_dropout=self.lora_cfg.dropout,
                target_modules=self.lora_cfg.target_modules,
                bias="none"
            )

            logger.info(f"LoRA配置: rank={self.lora_cfg.rank}, "
                        f"alpha={self.lora_cfg.alpha}, "
                        f"dropout={self.lora_cfg.dropout}")
            logger.info(f"目标模块: {self.lora_cfg.target_modules}")

            # 应用LoRA
            self.model = get_peft_model(self.model, peft_config)

            # 打印可训练参数
            self.model.print_trainable_parameters()

            logger.info("模型设置完成")
            return True

        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            return False

    def train(self) -> bool:
        """
        执行训练

        Returns:
            bool: 是否训练成功
        """
        try:
            logger.info("开始训练...")

            # 创建输出目录
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # 配置训练参数
            training_args = TrainingArguments(
                output_dir=str(self.checkpoint_dir),
                num_train_epochs=self.train_cfg.num_epochs,
                per_device_train_batch_size=self.train_cfg.batch_size,
                per_device_eval_batch_size=self.train_cfg.batch_size,
                gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
                learning_rate=self.train_cfg.learning_rate,
                weight_decay=self.train_cfg.weight_decay,
                warmup_steps=self.train_cfg.warmup_steps,
                logging_dir=str(self.path_cfg.TRAINING_LOGS_DIR / f"{self.expert_type}_expert"),
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=True if self.device_cfg.device == "cuda" else False,
                report_to="none",  # 不使用wandb等
                remove_unused_columns=False,
            )

            # 数据收集器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # 因果语言建模
            )

            # 创建Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                data_collator=data_collator,
            )

            # 执行训练
            logger.info("开始训练循环...")
            train_result = trainer.train()

            # 保存最终模型
            logger.info("保存最终LoRA权重...")
            self.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            # 保存训练指标
            metrics = train_result.metrics
            logger.info(f"训练完成！最终损失: {metrics.get('train_loss', 'N/A')}")

            # 保存训练指标到文件
            import json
            metrics_file = self.output_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"LoRA权重已保存至: {self.output_dir}")
            logger.info(f"训练指标已保存至: {metrics_file}")

            return True

        except Exception as e:
            logger.error(f"训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def save_checkpoint(self, epoch: int) -> bool:
        """
        保存检查点（由Trainer自动管理，此方法保留作为接口）

        Args:
            epoch: 当前epoch

        Returns:
            bool: 是否保存成功
        """
        # Trainer会自动保存检查点
        logger.info(f"Epoch {epoch} 检查点由Trainer自动保存")
        return True

    def save_final_weights(self) -> bool:
        """
        保存最终权重（由train()方法自动调用）

        Returns:
            bool: 是否保存成功
        """
        if self.model is None:
            logger.error("模型未初始化，无法保存")
            return False

        try:
            logger.info("保存最终LoRA权重...")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            logger.info(f"权重已保存至: {self.output_dir}")
            return True

        except Exception as e:
            logger.error(f"保存权重失败: {e}")
            return False

    def get_training_status(self) -> Dict:
        """
        获取训练状态

        Returns:
            dict: 训练状态信息
        """
        return {
            'expert_type': self.expert_type,
            'base_model': self.base_model_path,
            'output_dir': str(self.output_dir),
            'model_loaded': self.model is not None,
            'data_prepared': self.train_dataset is not None,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'use_4bit': self.use_4bit
        }


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("专家训练器测试")
    print("=" * 80)

    print("\n注意：这是一个完整的训练流程示例")
    print("实际训练请使用 scripts/training/train_*_expert.py 脚本")

    print("\n训练流程：")
    print("1. 创建ExpertTrainer实例")
    print("2. 调用prepare_data()准备数据")
    print("3. 调用setup_model()设置模型")
    print("4. 调用train()执行训练")
    print("5. LoRA权重自动保存到指定目录")

    print("\n示例代码：")
    print("trainer = ExpertTrainer(expert_type='text')")
    print("trainer.prepare_data()")
    print("trainer.setup_model()")
    print("trainer.train()")

    print("\n测试完成！")