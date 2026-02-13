"""
专家训练器 - 使用PEFT + Trainer实现LoRA微调
功能：
  - 支持四种专家类型（text, image, uml, general）
  - 集成Hugging Face Trainer
  - 支持LoRA微调
  - 支持梯度累积
  - 自动保存检查点和最终权重
  - 支持Qwen3-8B（默认）和Qwen-7B-Chat（遗留）
  - 根据模型版本自动选择target_modules

作者：Training System
日期：2025-02-13
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

# 条件导入视觉模型相关类（仅在需要时导入，避免环境兼容性问题）
# 这些类在qwen_text环境（transformers 4.32.0）中可能不存在
AutoModelForVision2Seq = None
AutoProcessor = None

def _import_vision_dependencies():
    """延迟导入视觉模型依赖（仅在需要时调用）"""
    global AutoModelForVision2Seq, AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as _AutoModelForVision2Seq
        from transformers import AutoProcessor as _AutoProcessor
        AutoModelForVision2Seq = _AutoModelForVision2Seq
        AutoProcessor = _AutoProcessor
    except ImportError as e:
        raise ImportError(
            f"无法导入视觉模型依赖: {e}\n"
            "请确保在正确的环境中运行：\n"
            "- Image/UML Expert需要qwen_vision25或qwen_vision3环境\n"
            "- Text/General Expert需要qwen_text环境"
        )

from config.settings import (
    get_path_config,
    get_lora_config,
    get_training_config,
    get_device_config,
    get_vision_model_config,
    get_model_config
)
from src.training.data_loader import (
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,
    GeneralDatasetLoader,
    InstructionDataset,
    split_dataset_for_expert
)
from src.utils.logger import get_logger
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate

logger = get_logger('training.expert_trainer')


def _get_transformers_version():
    """获取transformers版本号"""
    import transformers
    version_str = transformers.__version__
    major, minor = version_str.split('.')[:2]
    return int(major), int(minor)


def _should_use_eval_strategy():
    """检查是否应该使用eval_strategy参数（transformers >= 4.46.0）"""
    try:
        major, minor = _get_transformers_version()
        return (major > 4) or (major == 4 and minor >= 46)
    except:
        return False


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
                 use_4bit: bool = True,
                 use_rtx4090_optimization: bool = True):
        """
        初始化训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则从配置获取）
            use_4bit: 是否使用4bit量化训练
            use_rtx4090_optimization: 是否启用RTX 4090优化
        """
        # 验证专家类型
        valid_types = ['text', 'image', 'uml', 'general']
        if expert_type not in valid_types:
            raise ValueError(f"不支持的专家类型: {expert_type}，支持: {valid_types}")

        self.expert_type = expert_type
        self.use_4bit = use_4bit
        self.use_rtx4090_optimization = use_rtx4090_optimization

        # 获取配置
        self.path_cfg = get_path_config()
        self.lora_cfg = get_lora_config('conservative')
        self.train_cfg = get_training_config()
        self.device_cfg = get_device_config()
        self.model_cfg = get_model_config()  # 获取文本模型配置

        # 从环境变量读取训练参数（用于批量训练脚本的测试模式）
        import os
        if 'TRAIN_EPOCHS' in os.environ:
            try:
                epochs = int(os.environ['TRAIN_EPOCHS'])
                self.train_cfg.num_epochs = epochs
                logger.info(f"从环境变量读取训练轮数: {epochs}")
            except ValueError:
                logger.warning(f"无效的TRAIN_EPOCHS环境变量: {os.environ['TRAIN_EPOCHS']}")

        # batch_size应根据量化情况自动设置，不从环境变量读取

        # 设置基础模型路径
        if base_model_path:
            self.base_model_path = base_model_path
        else:
            # ⚠️ 重要：所有Expert都使用Qwen3-8B（默认文本模型）
            # Image/UML Expert的输入是JSON文本描述，不是图像/UML图
            # 视觉模型仅用于数据准备阶段（raw → interim）
            self.base_model_path = str(self.path_cfg.get_text_model_path())

        # 根据模型路径确定模型版本
        if 'Qwen3-8B' in self.base_model_path or 'qwen3-8B' in self.base_model_path:
            self.model_version = 'qwen3_8b'
        elif 'Qwen-7B-Chat' in self.base_model_path or 'qwen-7B-Chat' in self.base_model_path:
            self.model_version = 'qwen7b'
        else:
            self.model_version = self.model_cfg.version
            logger.warning(f"无法从路径推断模型版本，使用配置中的版本: {self.model_version}")

        # 根据模型版本确定target_modules
        self.target_modules = self._get_target_modules()

        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.path_cfg.get_expert_weight_path(f"{expert_type}_expert")

        # 设置检查点目录
        checkpoint_name = f"{expert_type}_expert"
        self.checkpoint_dir = self.path_cfg.get_checkpoint_path(checkpoint_name)

        # 初始化模型和数据相关属性
        self.model = None
        self.tokenizer = None
        self.processor = None  # 用于视觉模型
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        logger.info(f"初始化{expert_type}专家训练器")
        logger.info(f"基础模型: {self.base_model_path}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"4bit量化: {use_4bit}")
        logger.info(f"RTX 4090优化: {use_rtx4090_optimization}")

        # 打印实际训练配置
        self._print_training_config()

    def _get_target_modules(self) -> list:
        """
        根据模型版本返回适当的LoRA target_modules

        Returns:
            list: target_modules列表
        """
        if self.model_version == 'qwen7b':
            # Qwen-7B-Chat使用concatenated attention
            return ["c_attn"]
        elif self.model_version == 'qwen3_8b':
            # Qwen3-8B使用标准Transformers架构
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            # 默认使用Qwen3-8B的配置
            logger.warning(f"未知模型版本 {self.model_version}，使用Qwen3-8B的target_modules")
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def _print_training_config(self):
        """打印实际训练配置（包含从环境变量读取的参数）"""
        print()
        print("训练配置信息:")
        print("-" * 80)
        print(f"专家类型: {self.expert_type.upper()} Expert")
        print(f"基础模型: {self.base_model_path}")
        print(f"模型版本: {self.model_version}")
        print(f"输出目录: {self.output_dir}")
        print()
        print(f"LoRA配置:")
        print(f"  - Rank: {self.lora_cfg.rank}")
        print(f"  - Alpha: {self.lora_cfg.alpha}")
        print(f"  - Dropout: {self.lora_cfg.dropout}")
        print(f"  - Target Modules: {self.target_modules}")
        print()

        if self.use_rtx4090_optimization:
            print(f"训练参数 (RTX 4090优化):")
            if self.use_4bit:
                print(f"  - Batch Size: 2 (4bit量化)")
                print(f"  - Gradient Accumulation: 8")
            else:
                print(f"  - Batch Size: 2 (无量化)")
                print(f"  - Gradient Accumulation: 8")
            print(f"  - 有效Batch Size: 16")
            print(f"  - Epochs: {self.train_cfg.num_epochs}")
            print(f"  - Learning Rate: {self.train_cfg.learning_rate}")
            print(f"  - Max Seq Length: {self.train_cfg.max_seq_length}")
            print(f"  - 4bit量化: {'是' if self.use_4bit else '否'}")
            print(f"  - BF16混合精度: True")
            print(f"  - TF32加速: True")
            print(f"  - Fused优化器: True")
            print(f"  - 数据加载器工作进程: 8")
        else:
            print(f"训练参数:")
            print(f"  - Batch Size: {self.train_cfg.batch_size}")
            print(f"  - Gradient Accumulation: {self.train_cfg.gradient_accumulation_steps}")
            print(f"  - 有效Batch Size: {self.train_cfg.batch_size * self.train_cfg.gradient_accumulation_steps}")
            print(f"  - Epochs: {self.train_cfg.num_epochs}")
            print(f"  - Learning Rate: {self.train_cfg.learning_rate}")
            print(f"  - Max Seq Length: {self.train_cfg.max_seq_length}")
            print(f"  - 4bit量化: {'是' if self.use_4bit else '否'}")

        print("-" * 80)
        print()

    def prepare_data(self) -> bool:
        """准备训练数据"""
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
                # UML使用单一数据集
                loader = UMLDatasetLoader()
                raw_data = loader.load_csv_file()
            elif self.expert_type == 'general':
                # General专家使用统一的GeneralDatasetLoader
                # 确保训练推理一致：都使用GeneralInstructionTemplate
                loader = GeneralDatasetLoader()
                raw_data = loader.load_all_data()
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

            # 保存原始数据（用于调试和验证）
            self.train_data = train_data
            self.val_data = val_data
            self.test_data = test_data

            # 加载tokenizer
            # ⚠️ 重要：所有Expert（包括Image/UML）都直接加载tokenizer
            # 因为它们处理的都是文本输入（Image/UML Expert的输入是JSON文本描述）
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

            # ⚠️ 重要：所有Expert都使用AutoModelForCausalLM（Qwen3-8B默认）
            # Image/UML Expert处理的是JSON文本，不是图像/UML图
            logger.info("使用AutoModelForCausalLM加载文本模型")
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

            # ⚠️ 重要：根据模型版本选择正确的target_modules
            # Qwen-7B-Chat: ["c_attn"] (concatenated attention)
            # Qwen3-8B: ["q_proj", "k_proj", "v_proj", "o_proj"]
            target_modules = self.target_modules
            logger.info(f"使用模型: {self.model_version}, target_modules: {target_modules}")

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_cfg.rank,
                lora_alpha=self.lora_cfg.alpha,
                lora_dropout=self.lora_cfg.dropout,
                target_modules=target_modules,
                bias="none"
            )

            logger.info(f"LoRA配置: rank={self.lora_cfg.rank}, "
                        f"alpha={self.lora_cfg.alpha}, "
                        f"dropout={self.lora_cfg.dropout}")
            logger.info(f"最终目标模块: {target_modules}")

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

            # 设置CUDA内存分配器配置，避免显存碎片
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            logger.info("已设置PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（避免显存碎片）")

            # 创建输出目录
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # ===== 4090优化：动态调整训练参数 =====
            if self.use_rtx4090_optimization:
                logger.info("启用RTX 4090优化配置")

                # 根据是否使用4bit量化选择不同的配置
                if self.use_4bit:
                    # 4bit量化：保守配置以确保稳定性
                    batch_size = 2  # 降低到2以彻底避免OOM
                    gradient_accumulation_steps = 8  # 保持有效batch size=16
                    logger.info("使用4bit量化，batch size=2, gradient_accumulation=8")
                else:
                    # 无量化：显存占用大，使用最保守的配置避免OOM
                    batch_size = 2  # 从4降到2以避免OOM
                    gradient_accumulation_steps = 8  # 从4增加到8以保持有效batch size=16
                    logger.info("无量化模式，使用最保守的batch size=2以避免OOM")

                dataloader_num_workers = 8
                logging_steps = 5
                optimizer_type = "adamw_torch_fused"
            else:
                batch_size = self.train_cfg.batch_size
                gradient_accumulation_steps = self.train_cfg.gradient_accumulation_steps
                dataloader_num_workers = 2
                logging_steps = 10
                optimizer_type = "adamw_torch"

            logger.info(f"批量大小: {batch_size}")
            logger.info(f"梯度累积: {gradient_accumulation_steps}")
            logger.info(f"有效批量大小: {batch_size * gradient_accumulation_steps}")
            logger.info(f"数据加载器工作进程: {dataloader_num_workers}")

            # 检测transformers版本以使用正确的参数名
            use_eval_strategy = _should_use_eval_strategy()

            # 配置训练参数 - 兼容不同transformers版本
            training_args_dict = {
                "output_dir": str(self.checkpoint_dir),
                "num_train_epochs": self.train_cfg.num_epochs,
                "per_device_train_batch_size": batch_size,
                "per_device_eval_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": self.train_cfg.learning_rate,
                "weight_decay": self.train_cfg.weight_decay,
                "warmup_ratio": self.train_cfg.warmup_ratio,
                "logging_dir": str(self.path_cfg.TRAINING_LOGS_DIR / f"{self.expert_type}_expert"),
                "logging_steps": logging_steps,
                "save_strategy": "epoch",
                "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,

                # 显存优化选项 - 禁用gradient checkpointing以避免与LoRA的兼容性问题
                "gradient_checkpointing": False,

                # 4090优化选项
                "bf16": True if self.use_rtx4090_optimization and self.device_cfg.device == "cuda" else False,
                "tf32": True if self.use_rtx4090_optimization else False,
                "optim": optimizer_type,
                "dataloader_num_workers": dataloader_num_workers,
                "dataloader_pin_memory": True,
                "dataloader_prefetch_factor": 2,

                "fp16": False if self.use_rtx4090_optimization else (True if self.device_cfg.device == "cuda" else False),
                "report_to": "none",
                "remove_unused_columns": False,
            }

            # 根据transformers版本选择正确的参数名
            if use_eval_strategy:
                training_args_dict["eval_strategy"] = "epoch"
            else:
                training_args_dict["evaluation_strategy"] = "epoch"

            training_args = TrainingArguments(**training_args_dict)

            # 如果启用4090优化，打印优化信息
            if self.use_rtx4090_optimization:
                logger.info("=" * 60)
                logger.info("RTX 4090 优化已启用:")
                logger.info("  ✓ BF16混合精度训练")
                logger.info("  ✓ TF32加速")
                logger.info("  ✓ Fused AdamW优化器")
                logger.info("  ✓ Gradient checkpointing: False (已禁用以避免LoRA兼容性问题)")

                if self.use_4bit:
                    logger.info("  ✓ 4bit量化配置 (QLoRA标准方案):")
                    logger.info("    - Batch size: 2")
                    logger.info("    - Gradient accumulation: 8")
                    logger.info("    - 有效Batch Size: 16")
                else:
                    logger.info("  ✓ 无量化配置 (最保守策略避免OOM):")
                    logger.info("    - Batch size: 2")
                    logger.info("    - Gradient accumulation: 8")
                    logger.info("    - 有效Batch Size: 16")

                logger.info("  ✓ 数据加载器优化 (8 workers)")
                logger.info("  ✓ 预取因子: 2")
                logger.info("=" * 60)

            # 数据收集器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # 因果语言建模
            )

            # ===== 调试输出：打印前5个训练样本的完整prompt =====
            logger.info("=" * 80)
            logger.info("[训练数据调试] 前5个训练样本的完整内容:")
            logger.info("=" * 80)

            num_samples_to_show = min(5, len(self.train_data))
            for i in range(num_samples_to_show):
                sample = self.train_data[i]
                logger.info(f"\n[样本 {i+1}/{num_samples_to_show}]")
                logger.info("-" * 80)

                # 打印完整的input
                if 'input' in sample:
                    input_data = sample['input']
                    logger.info(f"Input (完整):\n{input_data}")

                # 打印完整的input_with_prompt
                if 'input_with_prompt' in sample:
                    prompt = sample['input_with_prompt']
                    logger.info(f"\nPrompt (完整, {len(prompt)}字符):\n{prompt}")

                # 打印完整的output
                if 'output' in sample:
                    output_data = sample['output']
                    logger.info(f"\nOutput (完整):\n{output_data}")

                logger.info("-" * 80)

            logger.info("=" * 80)
            logger.info("[调试输出结束] 请检查上述样本的prompt是否包含完整JSON结构")
            logger.info("=" * 80)
            # ===== 调试输出结束 =====

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