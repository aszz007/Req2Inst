"""
基础训练器 - 所有微调方法的共同逻辑

功能：
  - 数据准备和加载
  - 训练循环管理
  - 检查点保存
  - 训练曲线可视化
  - RTX 4090优化配置
  - 早停策略

作者：Training System
日期：2025-02-15
"""

import os
import json
import math
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

from config.settings import (
    get_path_config,
    get_training_config,
    get_device_config,
    get_model_config
)
from src.training.data_loader import (
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,
    GeneralDatasetLoader,
    InstructionDataset,
    InstructionDataCollator,
    split_dataset_for_expert
)
from src.utils.logger import get_logger

logger = get_logger('training.base_trainer')


def _remove_qwen_special_tokens(text: str) -> str:
    """
    Remove Qwen model special tokens from text

    Removes tokens such as:
    - <|im_start|>
    - <|im_end|>
    - <think>
    - </think>

    Args:
        text: Input text containing special tokens

    Returns:
        str: Text with special tokens removed
    """
    if not text:
        return text

    # Remove Qwen chat format tokens
    text = text.replace('<|im_start|>', '')
    text = text.replace('<|im_end|>', '')

    # Remove thinking tags
    text = text.replace('<think>', '')
    text = text.replace('</think>', '')

    return text


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


class NaNAwareEarlyStoppingCallback(TrainerCallback):
    """
    NaN-aware早停回调

    标准的EarlyStoppingCallback在遇到NaN时可能错误触发早停。
    此回调忽略NaN值，只基于有效的eval_loss值判断是否早停。
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.patience_counter = 0
        self.nan_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """在每次验证后调用"""
        if metrics is None:
            return

        eval_loss = metrics.get('eval_loss')

        # 检查NaN
        if eval_loss is None or (isinstance(eval_loss, float) and math.isnan(eval_loss)):
            self.nan_count += 1
            logger.warning(f"检测到NaN验证损失（第{self.nan_count}次）- 跳过本次早停检查")
            logger.warning("NaN可能原因: 1) 模型训练不稳定 2) 学习率过大 3) P-Tuning配置问题")
            # 不更新patience计数器，跳过本次早停检查
            return control

        # 有效的eval_loss
        metric = eval_loss

        # 初始化best_metric
        if self.best_metric is None:
            self.best_metric = metric
            logger.info(f"初始化最佳验证损失: {metric:.4f}")
            return control

        # 检查是否改进
        if metric < (self.best_metric - self.early_stopping_threshold):
            # 有改进
            self.best_metric = metric
            self.patience_counter = 0
            logger.info(f"验证损失改进: {metric:.4f} (最佳: {self.best_metric:.4f})")
        else:
            # 无改进
            self.patience_counter += 1
            logger.info(f"验证损失无改进: {metric:.4f} vs 最佳{self.best_metric:.4f} "
                       f"(patience: {self.patience_counter}/{self.early_stopping_patience})")

            if self.patience_counter >= self.early_stopping_patience:
                logger.info("=" * 80)
                logger.info(f"早停触发: 连续{self.early_stopping_patience}次验证无改进")
                logger.info(f"最佳验证损失: {self.best_metric:.4f}")
                logger.info(f"当前验证损失: {metric:.4f}")
                if self.nan_count > 0:
                    logger.info(f"训练期间遇到{self.nan_count}次NaN验证损失（已忽略）")
                logger.info("=" * 80)
                control.should_training_stop = True

        return control


class TrainingHistoryCallback(TrainerCallback):
    """
    自定义回调函数，用于记录训练过程中的所有日志

    记录内容包括：
    - loss: 训练损失
    - grad_norm: 梯度范数
    - learning_rate: 学习率
    - epoch: 当前epoch
    - eval_loss: 验证损失（如果有）
    """

    def __init__(self):
        super().__init__()
        self.training_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        在每次日志记录时调用

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            logs: 日志字典
        """
        if logs is not None:
            log_entry = {
                'step': state.global_step,
                'epoch': logs.get('epoch', state.epoch),
            }

            for key, value in logs.items():
                if key not in ['step', 'epoch']:
                    log_entry[key] = value

            self.training_history.append(log_entry)

    def get_history(self):
        """
        获取完整的训练历史

        Returns:
            list: 训练历史记录列表
        """
        return self.training_history


class BaseTrainer(ABC):
    """
    基础训练器 - 所有微调方法的抽象基类

    子类需要实现：
    - setup_model(): 设置模型（包括微调方法特定的配置）
    - _save_weights(): 保存权重的具体实现
    """

    def __init__(self,
                 expert_type: str,
                 method_name: str,
                 base_model_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 use_rtx4090_optimization: bool = True,
                 debug_samples: bool = False):
        """
        初始化基础训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            method_name: 微调方法名称（'lora_moe', 'p_tuning', 'prompt_tuning', 'full_finetuning'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则从配置获取）
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前3个训练样本（默认关闭）
        """
        valid_types = ['text', 'image', 'uml', 'general']
        if expert_type not in valid_types:
            raise ValueError(f"不支持的专家类型: {expert_type}，支持: {valid_types}")

        self.expert_type = expert_type
        self.method_name = method_name
        self.use_rtx4090_optimization = use_rtx4090_optimization
        self.debug_samples = debug_samples

        # 获取配置
        self.path_cfg = get_path_config()
        self.train_cfg = get_training_config()
        self.device_cfg = get_device_config()
        self.model_cfg = get_model_config()

        # 从环境变量读取训练参数
        if 'TRAIN_EPOCHS' in os.environ:
            try:
                epochs = int(os.environ['TRAIN_EPOCHS'])
                self.train_cfg.num_epochs = epochs
                logger.info(f"从环境变量读取训练轮数: {epochs}")
            except ValueError:
                logger.warning(f"无效的TRAIN_EPOCHS环境变量: {os.environ['TRAIN_EPOCHS']}")

        # 设置基础模型路径
        if base_model_path:
            self.base_model_path = base_model_path
        else:
            self.base_model_path = str(self.path_cfg.get_text_model_path())

        # 根据模型路径确定模型版本
        if 'Qwen3-8B' in self.base_model_path or 'qwen3-8B' in self.base_model_path:
            self.model_version = 'qwen3_8b'
        elif 'Qwen-7B-Chat' in self.base_model_path or 'qwen-7B-Chat' in self.base_model_path:
            self.model_version = 'qwen7b'
        else:
            self.model_version = self.model_cfg.version
            logger.warning(f"无法从路径推断模型版本，使用配置中的版本: {self.model_version}")

        # 设置输出目录（使用新的checkpoints路径）
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # 使用新的路径结构：checkpoints/{method_name}/{expert_type}_expert/
            self.output_dir = self.path_cfg.PROJECT_ROOT / 'checkpoints' / method_name / f"{expert_type}_expert"

        # 设置检查点目录
        checkpoint_name = f"{method_name}_{expert_type}_expert"
        self.checkpoint_dir = self.output_dir / 'training_checkpoints'

        # 初始化模型和数据相关属性
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # 初始化训练历史回调
        self.history_callback = TrainingHistoryCallback()

        logger.info(f"初始化{expert_type}专家训练器（方法：{method_name}）")
        logger.info(f"基础模型: {self.base_model_path}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"RTX 4090优化: {use_rtx4090_optimization}")

    def _print_training_config(self):
        """打印实际训练配置"""
        batch_size, gradient_accumulation_steps = self._get_batch_config()

        logger.info("=" * 80)
        logger.info("训练配置")
        logger.info("=" * 80)
        logger.info(f"专家类型: {self.expert_type}")
        logger.info(f"微调方法: {self.method_name}")
        logger.info(f"基础模型: {self.base_model_path}")
        logger.info(f"模型版本: {self.model_version}")

        if self.use_rtx4090_optimization:
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

    def _get_batch_config(self) -> Tuple[int, int]:
        """
        获取batch size和gradient accumulation配置

        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        if self.use_rtx4090_optimization:
            return 8, 2
        else:
            return self.train_cfg.batch_size, self.train_cfg.gradient_accumulation_steps

    def prepare_data(self) -> bool:
        """
        准备训练数据

        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"准备{self.expert_type}专家的训练数据...")

            # 根据专家类型加载数据
            if self.expert_type == 'text':
                loader = TextDatasetLoader()
                all_data = loader.load_csv_files()
            elif self.expert_type == 'image':
                loader = ImageDatasetLoader()
                all_data = loader.load_csv_file()
            elif self.expert_type == 'uml':
                loader = UMLDatasetLoader()
                all_data = loader.load_csv_file()
            elif self.expert_type == 'general':
                loader = GeneralDatasetLoader()
                all_data = loader.load_all_data()
            else:
                raise ValueError(f"不支持的专家类型: {self.expert_type}")

            if not all_data:
                logger.error("没有加载到任何数据")
                return False

            logger.info(f"成功加载{len(all_data)}条数据")

            # 划分数据集
            train_data, val_data, test_data = split_dataset_for_expert(
                all_data, self.expert_type
            )

            # 创建Dataset对象
            self.train_dataset = InstructionDataset(
                train_data, self.tokenizer, self.train_cfg.max_seq_length
            )
            self.val_dataset = InstructionDataset(
                val_data, self.tokenizer, self.train_cfg.max_seq_length
            )
            self.test_dataset = InstructionDataset(
                test_data, self.tokenizer, self.train_cfg.max_seq_length
            )

            logger.info(f"数据集划分完成:")
            logger.info(f"  训练集: {len(self.train_dataset)}条")
            logger.info(f"  验证集: {len(self.val_dataset)}条")
            logger.info(f"  测试集: {len(self.test_dataset)}条")

            return True

        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    @abstractmethod
    def setup_model(self) -> bool:
        """
        设置模型（子类必须实现）

        Returns:
            bool: 是否成功
        """
        pass

    def _get_early_stopping_patience(self) -> int:
        """
        根据专家类型和数据量确定早停patience

        Returns:
            int: patience值
        """
        data_size = len(self.train_dataset) if self.train_dataset else 0

        if self.expert_type == 'text' or self.expert_type == 'general':
            return 3
        elif self.expert_type == 'image':
            return 4
        elif self.expert_type == 'uml':
            return 4
        else:
            return 3

    def _get_eval_steps(self) -> int:
        """
        根据专家类型确定验证步数

        Returns:
            int: eval_steps
        """
        if self.expert_type == 'uml':
            return 30
        else:
            return 50

    def train(self) -> bool:
        """
        执行训练

        Returns:
            bool: 是否成功
        """
        if self.model is None or self.tokenizer is None:
            logger.error("模型未初始化，请先调用setup_model()")
            return False

        if self.train_dataset is None or self.val_dataset is None:
            logger.error("数据未准备，请先调用prepare_data()")
            return False

        try:
            # 训练前强制清空GPU缓存
            if torch.cuda.is_available():
                # Full Fine-tuning需要更激进的显存清理
                if self.method_name == 'full_finetuning':
                    for i in range(3):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    logger.info("Full Fine-tuning: 已三重清空GPU缓存")
                else:
                    torch.cuda.empty_cache()
                    if self.expert_type in ['uml', 'general']:
                        logger.info(f"{self.expert_type}专家训练前已清空GPU缓存，最大化可用显存")

                # 打印训练前显存状态
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - allocated
                logger.info(f"[训练前] GPU显存: 已用={allocated:.2f}GB, 可用≈{free:.2f}GB, 总计={total:.2f}GB")

                # Full Fine-tuning的显存预警
                if self.method_name == 'full_finetuning' and allocated > 10.0:
                    logger.error(f"警告: 训练前显存已占用{allocated:.2f}GB，可能导致OOM！")
                    logger.error("建议:")
                    logger.error("  1. 检查是否有其他进程占用GPU (nvidia-smi)")
                    logger.error("  2. 重启Python进程清理显存")
                    logger.error("  3. 关闭不必要的程序")

            # 创建输出目录
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # 获取batch配置
            batch_size, gradient_accumulation_steps = self._get_batch_config()

            # 获取早停和验证配置
            early_stopping_patience = self._get_early_stopping_patience()
            eval_steps = self._get_eval_steps()

            # 计算warmup步数
            num_train_samples = len(self.train_dataset)
            steps_per_epoch = num_train_samples // (batch_size * gradient_accumulation_steps)
            total_steps = steps_per_epoch * self.train_cfg.num_epochs

            # P-Tuning v2和Prompt Tuning需要更长的warmup以防止早期NaN
            if self.method_name in ['p_tuning', 'prompt_tuning']:
                warmup_ratio = 0.2  # 增加到20%以提供更好的训练稳定性
                logger.info(f"{self.method_name}使用增加的warmup比例: {warmup_ratio} (防止早期NaN)")
            else:
                warmup_ratio = 0.1

            warmup_steps = int(total_steps * warmup_ratio)

            logger.info(f"训练步数配置:")
            logger.info(f"  每epoch步数: {steps_per_epoch}")
            logger.info(f"  总训练步数: {total_steps}")
            logger.info(f"  Warmup步数: {warmup_steps}")
            logger.info(f"  验证频率: 每{eval_steps}步")
            logger.info(f"  早停patience: {early_stopping_patience}")

            # 配置TrainingArguments
            training_args_dict = {
                'output_dir': str(self.checkpoint_dir),
                'num_train_epochs': self.train_cfg.num_epochs,
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'learning_rate': self.train_cfg.learning_rate,
                'weight_decay': 0.01,
                'lr_scheduler_type': 'cosine',
                'warmup_steps': warmup_steps,
                'logging_steps': 10,
                'eval_steps': eval_steps,
                'save_steps': eval_steps,
                'save_total_limit': 3,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'report_to': 'none',
                'remove_unused_columns': False,
            }

            # 梯度裁剪配置（防止梯度爆炸导致NaN）
            # P-Tuning v2和Prompt Tuning：最严格裁剪（参数少，对梯度更敏感）
            # Full Finetuning：严格裁剪（rank小，需要稳定训练）
            # LoRA：标准裁剪（参数适中，相对稳定）
            if self.method_name in ['p_tuning', 'prompt_tuning']:
                training_args_dict['max_grad_norm'] = 0.5
                logger.warning(f"{self.method_name}使用严格梯度裁剪(0.5)以防止NaN验证损失")
            elif self.method_name == 'full_finetuning':
                training_args_dict['max_grad_norm'] = 0.8
                logger.info(f"{self.method_name}使用较严格梯度裁剪(0.8)以保证训练稳定性")
            else:
                training_args_dict['max_grad_norm'] = 1.0
                logger.info(f"{self.method_name}使用标准梯度裁剪(1.0)")

            # load_best_model_at_end（某些方法如P-Tuning v2和Prompt Tuning不支持）
            if not getattr(self, 'disable_load_best_model', False):
                training_args_dict['load_best_model_at_end'] = True
            else:
                training_args_dict['load_best_model_at_end'] = False
                logger.info("load_best_model_at_end已禁用（当前训练方法不支持）")

            # Gradient checkpointing（某些方法如P-Tuning v2不支持）
            if not getattr(self, 'disable_gradient_checkpointing', False):
                training_args_dict['gradient_checkpointing'] = True
            else:
                logger.info("Gradient checkpointing已禁用（当前训练方法不支持）")

            # 使用eval_strategy或evaluation_strategy（根据transformers版本）
            if _should_use_eval_strategy():
                training_args_dict['eval_strategy'] = 'steps'
            else:
                training_args_dict['evaluation_strategy'] = 'steps'

            # RTX 4090优化配置
            if self.use_rtx4090_optimization:
                # 检查是否需要减少workers（如P-Tuning v2或Full Fine-tuning）
                # Full Fine-tuning使用最小worker配置
                if self.method_name == 'full_finetuning':
                    num_workers = 2
                    prefetch_factor = 1
                    logger.info(f"Full Fine-tuning使用最小dataloader配置: workers=2, prefetch=1")
                elif self.expert_type in ['uml', 'general'] and getattr(self, 'reduced_workers', False):
                    num_workers = 2
                    prefetch_factor = 1
                    logger.info(f"{self.expert_type}专家使用最小dataloader配置: workers=2, prefetch=1")
                elif getattr(self, 'reduced_workers', False):
                    num_workers = 4
                    prefetch_factor = 2
                else:
                    num_workers = 8
                    prefetch_factor = 4

                training_args_dict.update({
                    'bf16': True,
                    'tf32': True,
                    'optim': 'adamw_torch_fused',
                    'dataloader_num_workers': num_workers,
                    'dataloader_prefetch_factor': prefetch_factor,
                })

                if getattr(self, 'reduced_workers', False) or self.method_name == 'full_finetuning':
                    logger.info(f"使用减少的dataloader workers: {num_workers} (显存优化)")
            else:
                training_args_dict['fp16'] = torch.cuda.is_available()

            training_args = TrainingArguments(**training_args_dict)

            # 创建数据收集器
            data_collator = InstructionDataCollator(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=8
            )

            # 调试：打印前3个训练样本
            if self.debug_samples and len(self.train_dataset) > 0:
                logger.info("=" * 80)
                logger.info("[调试输出] 打印前3个训练样本的prompt")
                logger.info("=" * 80)

                for i in range(min(3, len(self.train_dataset))):
                    sample = self.train_dataset.data[i]
                    logger.info(f"\n样本 {i+1}:")
                    logger.info("-" * 80)
                    # Remove Qwen special tokens before printing
                    prompt_text = sample.get('input_with_prompt', 'N/A')
                    clean_prompt = _remove_qwen_special_tokens(prompt_text)
                    logger.info(f"完整Prompt:\n{clean_prompt}")
                    logger.info("-" * 80)
                    logger.info(f"期望输出:\n{sample['output']}")
                    logger.info("-" * 80)

                logger.info("=" * 80)
                logger.info("[调试输出结束] 请检查上述样本的prompt是否包含完整JSON结构")
                logger.info("=" * 80)

            # 创建Trainer（使用NaN-aware早停callback）
            early_stopping_callback = NaNAwareEarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.0001
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                data_collator=data_collator,
                callbacks=[self.history_callback, early_stopping_callback],
            )

            # 执行训练
            logger.info("开始训练循环...")
            train_result = trainer.train()

            # 保存最终模型
            logger.info("保存最终权重...")
            self._save_weights()

            # 保存训练指标
            metrics = train_result.metrics
            logger.info(f"训练完成！最终损失: {metrics.get('train_loss', 'N/A')}")

            # 报告NaN统计（如果有）
            if early_stopping_callback.nan_count > 0:
                logger.warning("=" * 80)
                logger.warning(f"训练期间检测到{early_stopping_callback.nan_count}次NaN验证损失")
                logger.warning("NaN验证损失可能表明:")
                logger.warning("  1. 学习率过大导致训练不稳定")
                logger.warning("  2. P-Tuning v2的配置需要调整（如encoder_hidden_size）")
                logger.warning("  3. 数据集中存在异常样本")
                logger.warning("  4. Virtual tokens初始化不当")
                logger.warning("建议: 检查training_history.json中的eval_loss值")
                logger.warning("=" * 80)

            # 保存训练指标到文件
            metrics_file = self.output_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            # 保存训练历史记录
            training_history = self.history_callback.get_history()
            history_file = self.output_dir / "training_history.json"

            batch_size, gradient_accumulation_steps = self._get_batch_config()

            # 清理训练历史中的NaN值（NaN不是有效的JSON）
            def clean_nan_values(obj):
                """递归清理字典/列表中的NaN值，替换为None"""
                if isinstance(obj, dict):
                    return {k: clean_nan_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan_values(item) for item in obj]
                elif isinstance(obj, float) and math.isnan(obj):
                    return None
                else:
                    return obj

            cleaned_history = clean_nan_values(training_history)

            history_data = {
                'expert_type': self.expert_type,
                'method_name': self.method_name,
                'total_steps': len(training_history),
                'num_epochs': self.train_cfg.num_epochs,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': batch_size * gradient_accumulation_steps,
                'learning_rate': self.train_cfg.learning_rate,
                'use_rtx4090_optimization': self.use_rtx4090_optimization,
                'history': cleaned_history
            }

            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)

            logger.info(f"训练历史已保存至: {history_file}")
            logger.info(f"共记录 {len(training_history)} 个训练步骤的数据")

            # 生成训练曲线可视化
            logger.info("生成训练曲线可视化...")
            try:
                self._plot_training_curves(training_history, self.expert_type)
                logger.info("训练曲线可视化已生成")
            except Exception as e:
                logger.warning(f"训练曲线可视化失败: {e}")
                logger.warning("继续执行，但未生成可视化图表")

            logger.info(f"权重已保存至: {self.output_dir}")
            logger.info(f"训练指标已保存至: {metrics_file}")

            return True

        except Exception as e:
            logger.error(f"训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    @abstractmethod
    def _save_weights(self):
        """
        保存权重（子类必须实现）
        """
        pass

    def get_training_status(self) -> Dict:
        """
        获取训练状态

        Returns:
            dict: 训练状态信息
        """
        return {
            'expert_type': self.expert_type,
            'method_name': self.method_name,
            'base_model': self.base_model_path,
            'output_dir': str(self.output_dir),
            'model_loaded': self.model is not None,
            'data_prepared': self.train_dataset is not None,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
        }

    def _plot_training_curves(self, training_history: List[Dict], expert_type: str):
        """
        生成训练曲线可视化图表

        Args:
            training_history: 训练历史记录
            expert_type: 专家类型
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装，跳过可视化")
            return

        # 创建输出目录
        curves_dir = self.path_cfg.PROJECT_ROOT / 'outputs' / 'training_curves'
        curves_dir.mkdir(parents=True, exist_ok=True)

        # 提取数据 - 为每个指标单独记录对应的steps，并过滤掉None/NaN值
        loss_steps = []
        losses = []
        eval_steps = []
        eval_losses = []
        grad_norm_steps = []
        grad_norms = []
        lr_steps = []
        learning_rates = []

        for entry in training_history:
            step = entry.get('step', 0)

            # 处理loss（过滤None和NaN）
            if 'loss' in entry:
                loss_val = entry['loss']
                if loss_val is not None and not (isinstance(loss_val, float) and math.isnan(loss_val)):
                    loss_steps.append(step)
                    losses.append(loss_val)

            # 处理eval_loss（过滤None和NaN）
            if 'eval_loss' in entry:
                eval_val = entry['eval_loss']
                if eval_val is not None and not (isinstance(eval_val, float) and math.isnan(eval_val)):
                    eval_steps.append(step)
                    eval_losses.append(eval_val)

            # 处理grad_norm（过滤None和NaN）
            if 'grad_norm' in entry:
                grad_val = entry['grad_norm']
                if grad_val is not None and not (isinstance(grad_val, float) and math.isnan(grad_val)):
                    grad_norm_steps.append(step)
                    grad_norms.append(grad_val)

            # 处理learning_rate（过滤None和NaN）
            if 'learning_rate' in entry:
                lr_val = entry['learning_rate']
                if lr_val is not None and not (isinstance(lr_val, float) and math.isnan(lr_val)):
                    lr_steps.append(step)
                    learning_rates.append(lr_val)

        # 数据质量检查和警告
        total_entries = len(training_history)
        nan_eval_count = sum(1 for e in training_history if 'eval_loss' in e and
                            (e['eval_loss'] is None or (isinstance(e['eval_loss'], float) and math.isnan(e['eval_loss']))))

        if total_entries < 10:
            logger.warning(f"训练历史记录很少（{total_entries}条），可能因早停提前结束")

        if len(losses) < 3:
            logger.warning(f"训练损失数据点很少（{len(losses)}个）")
        if len(eval_losses) == 0:
            if nan_eval_count > 0:
                logger.warning(f"所有{nan_eval_count}个验证损失都是NaN（已过滤），无法绘制验证曲线")
                logger.warning("这表明训练过程不稳定，建议检查:")
                logger.warning("  1. 降低学习率")
                logger.warning("  2. 调整P-Tuning/Prompt Tuning配置")
                logger.warning("  3. 检查数据集质量")
            else:
                logger.warning("没有验证损失数据")
        elif len(eval_losses) < 3:
            if nan_eval_count > 0:
                logger.warning(f"验证损失数据点很少（{len(eval_losses)}个有效值，{nan_eval_count}个NaN已过滤）")
            else:
                logger.warning(f"验证损失数据点很少（{len(eval_losses)}个）")
        elif nan_eval_count > 0:
            logger.info(f"过滤了{nan_eval_count}个NaN验证损失，保留{len(eval_losses)}个有效值")

        logger.info(f"曲线数据统计: Loss={len(losses)}点, EvalLoss={len(eval_losses)}点, GradNorm={len(grad_norms)}点, LR={len(learning_rates)}点")

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {expert_type.upper()} Expert ({self.method_name})',
                     fontsize=16, fontweight='bold')

        # 1. Training Loss
        if losses:
            axes[0, 0].plot(loss_steps, losses, 'b-', linewidth=1.5, alpha=0.7)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No training loss data',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')

        # 2. Eval Loss
        if eval_losses:
            axes[0, 1].plot(eval_steps, eval_losses, 'r-', linewidth=2, marker='o', markersize=4)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Eval Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No validation loss data',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Eval Loss')
            axes[0, 1].set_title('Validation Loss')

        # 3. Gradient Norm
        if grad_norms:
            axes[1, 0].plot(grad_norm_steps, grad_norms, 'g-', linewidth=1, alpha=0.6)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No gradient norm data',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norm')

        # 4. Learning Rate
        if learning_rates:
            axes[1, 1].plot(lr_steps, learning_rates, 'm-', linewidth=1.5)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            axes[1, 1].text(0.5, 0.5, 'No learning rate data',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')

        plt.tight_layout()

        # 保存图表
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = curves_dir / f'{expert_type}_expert_{self.method_name}_training_curves_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"训练曲线已保存至: {plot_path}")