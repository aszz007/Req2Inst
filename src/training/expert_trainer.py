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
    get_vision_model_config
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
                 dataset_version: Optional[str] = None,  # 新增参数
                 use_rtx4090_optimization: bool = True):  # 新增参数
        """
        初始化训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则从配置获取）
            use_4bit: 是否使用4bit量化训练
            dataset_version: 数据集版本（仅用于UML，如'qwen2.5', 'qwen3', 'qwen235B'）
            use_rtx4090_optimization: 是否启用RTX 4090优化
        """
        # 验证专家类型
        valid_types = ['text', 'image', 'uml', 'general']
        if expert_type not in valid_types:
            raise ValueError(f"不支持的专家类型: {expert_type}，支持: {valid_types}")

        self.expert_type = expert_type
        self.use_4bit = use_4bit
        self.dataset_version = dataset_version
        self.use_rtx4090_optimization = use_rtx4090_optimization

        # 如果是UML或General且未指定数据集版本，默认使用qwen2.5
        if expert_type in ['uml', 'general'] and dataset_version is None:
            self.dataset_version = 'qwen2.5'
            logger.warning(f"{expert_type}专家未指定数据集版本，默认使用: qwen2.5")

        # 获取配置
        self.path_cfg = get_path_config()
        self.lora_cfg = get_lora_config('conservative')
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

        # 设置输出目录（考虑数据集版本）
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            if expert_type == 'uml' and dataset_version:
                # UML专家根据模型版本和数据集版本生成输出路径
                vision_cfg = get_vision_model_config()
                vision_version = vision_cfg.version
                output_name = f"uml_expert_{vision_version}_dataset_{dataset_version}"
                self.output_dir = self.path_cfg.LORA_WEIGHTS_DIR / 'experts' / output_name
            elif expert_type == 'general' and dataset_version:
                # General专家根据数据集版本生成输出路径
                output_name = f"general_expert_dataset_{dataset_version}"
                self.output_dir = self.path_cfg.LORA_WEIGHTS_DIR / 'experts' / output_name
            else:
                self.output_dir = self.path_cfg.get_expert_weight_path(f"{expert_type}_expert")

        # 设置检查点目录
        checkpoint_name = f"{expert_type}_expert"
        if expert_type == 'uml' and dataset_version:
            vision_cfg = get_vision_model_config()
            vision_version = vision_cfg.version
            checkpoint_name = f"{expert_type}_expert_{vision_version}_dataset_{dataset_version}"
        elif expert_type == 'general' and dataset_version:
            checkpoint_name = f"{expert_type}_expert_dataset_{dataset_version}"
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
        if expert_type == 'uml':
            logger.info(f"数据集版本: {self.dataset_version}")
        logger.info(f"RTX 4090优化: {use_rtx4090_optimization}")

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
                # UML使用dataset_version参数
                loader = UMLDatasetLoader(dataset_version=self.dataset_version)
                raw_data = loader.load_csv_file()
            elif self.expert_type == 'general':
                text_loader = TextDatasetLoader()
                image_loader = ImageDatasetLoader()
                # General专家使用指定版本的UML数据集
                uml_loader = UMLDatasetLoader(dataset_version=self.dataset_version)
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

            # 加载tokenizer/processor（需要在创建Dataset之前）
            if self.expert_type in ['text', 'general']:
                logger.info("加载tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_path,
                    trust_remote_code=True,
                    padding_side='left'
                )
            else:  # image, uml - 视觉模型
                # 导入视觉模型依赖
                _import_vision_dependencies()

                logger.info("加载processor（视觉模型）...")
                self.processor = AutoProcessor.from_pretrained(
                    self.base_model_path,
                    trust_remote_code=True
                )
                # 训练时只需要tokenizer部分
                self.tokenizer = self.processor.tokenizer
                self.tokenizer.padding_side = 'left'
                logger.info("从processor提取tokenizer用于训练")

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

            # 根据专家类型选择正确的模型类
            if self.expert_type in ['text', 'general']:
                # 文本模型：Qwen-7B-Chat
                logger.info("使用AutoModelForCausalLM加载文本模型")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if not self.use_4bit else None,
                    low_cpu_mem_usage=True
                )
            else:  # image, uml
                # 导入视觉模型依赖
                _import_vision_dependencies()

                # 视觉模型：Qwen2.5-VL-7B 或 Qwen3-VL-8B
                logger.info("使用AutoModelForVision2Seq加载视觉模型")
                self.model = AutoModelForVision2Seq.from_pretrained(
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

            # 根据模型类型动态选择target_modules
            # Qwen-7B-Chat使用c_attn, Qwen2.5-VL/Qwen3-VL使用标准命名
            if self.expert_type in ['text', 'general']:
                # Qwen-7B-Chat: 使用concatenated attention
                target_modules = ["c_attn"]
                logger.info("检测到文本模型(Qwen-7B-Chat), 使用target_modules: ['c_attn']")
            else:  # image, uml
                # Qwen2.5-VL / Qwen3-VL: 使用标准Transformers命名
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                logger.info("检测到视觉模型(Qwen2.5-VL/Qwen3-VL), 使用target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']")

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

                # 针对视觉模型和文本模型使用不同的策略
                if self.expert_type in ['image', 'uml']:
                    # 视觉模型：显存消耗大，使用小batch size + 大gradient accumulation
                    batch_size = 1  # 降低到1以避免OOM
                    gradient_accumulation_steps = 16  # 增加到16保持有效batch size=16
                    dataloader_num_workers = 4  # 降低worker数量
                    logger.info("检测到视觉模型，使用显存优化策略")
                else:  # text, general
                    # 文本模型：根据是否使用4bit量化选择策略
                    if self.use_4bit:
                        # 4bit量化：为避免OOM，使用保守的batch size策略
                        # 虽然量化节省显存，但lm_head层在长序列下仍需大量显存
                        batch_size = 2
                        gradient_accumulation_steps = 8
                        logger.info("文本模型 + 4bit量化，使用保守batch size策略（避免lm_head OOM）")
                    else:
                        # 无量化：显存占用大，降低batch size
                        batch_size = 4
                        gradient_accumulation_steps = 4
                        logger.info("文本模型无量化，降低batch size以避免OOM")
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

                # 显存优化选项 - 为所有专家启用gradient checkpointing以节省显存
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},

                # 4090优化选项
                "bf16": True if self.use_rtx4090_optimization and self.device_cfg.device == "cuda" else False,
                "tf32": True if self.use_rtx4090_optimization else False,
                "optim": optimizer_type,
                "dataloader_num_workers": dataloader_num_workers,
                "dataloader_pin_memory": True,
                "dataloader_prefetch_factor": 2 if self.expert_type in ['image', 'uml'] else (4 if self.use_rtx4090_optimization else 2),

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
                logger.info("  ✓ Gradient checkpointing: True (所有模型，节省显存)")

                if self.expert_type in ['image', 'uml']:
                    logger.info("  ✓ 视觉模型显存优化:")
                    logger.info("    - Batch size: 1 (降低显存)")
                    logger.info("    - Gradient accumulation: 16 (保持有效batch=16)")
                    logger.info("    - 数据加载器工作进程: 4")
                    logger.info("    - 预取因子: 2")
                else:
                    if self.use_4bit:
                        logger.info("  ✓ 文本模型 + 4bit量化优化:")
                        logger.info("    - Batch size: 2 (保守策略，避免lm_head OOM)")
                        logger.info("    - Gradient accumulation: 8 (有效batch=16)")
                        logger.info("    - Max seq length: 1536 (优化显存占用)")
                    else:
                        logger.info("  ✓ 文本模型无量化配置:")
                        logger.info("    - Batch size: 4 (避免OOM)")
                        logger.info("    - Gradient accumulation: 4 (有效batch=16)")
                    logger.info("  ✓ 数据加载器优化 (8 workers)")
                    logger.info("  ✓ 预取因子: 4")

                logger.info("=" * 60)

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