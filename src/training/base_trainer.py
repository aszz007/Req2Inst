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
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
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
                 debug_samples: bool = True):
        """
        初始化基础训练器

        Args:
            expert_type: 专家类型（'text', 'image', 'uml', 'general'）
            method_name: 微调方法名称（'lora_moe', 'p_tuning', 'prompt_tuning', 'full_finetuning'）
            base_model_path: 基础模型路径（None则从配置获取）
            output_dir: 输出目录（None则从配置获取）
            use_rtx4090_optimization: 是否启用RTX 4090优化
            debug_samples: 是否在训练开始前打印前3个训练样本（默认开启）
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
            # 训练前清空GPU缓存（对UML和General专家尤其重要）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.expert_type in ['uml', 'general']:
                    logger.info(f"{self.expert_type}专家训练前已清空GPU缓存，最大化可用显存")

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
            warmup_steps = int(total_steps * 0.1)

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
                'max_grad_norm': 1.0,
                'lr_scheduler_type': 'cosine',
                'warmup_steps': warmup_steps,
                'logging_steps': 10,
                'eval_steps': eval_steps,
                'save_steps': eval_steps,
                'save_total_limit': 3,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'report_to': 'none',
                'remove_unused_columns': False,
            }

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
                # 检查是否需要减少workers（如P-Tuning v2）
                num_workers = 4 if getattr(self, 'reduced_workers', False) else 8
                prefetch_factor = 2 if getattr(self, 'reduced_workers', False) else 4

                training_args_dict.update({
                    'bf16': True,
                    'tf32': True,
                    'optim': 'adamw_torch_fused',
                    'dataloader_num_workers': num_workers,
                    'dataloader_prefetch_factor': prefetch_factor,
                })

                if getattr(self, 'reduced_workers', False):
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
                    logger.info(f"完整Prompt:\n{sample.get('input_with_prompt', 'N/A')}")
                    logger.info("-" * 80)
                    logger.info(f"期望输出:\n{sample['output'][:200]}...")
                    logger.info("-" * 80)

                logger.info("=" * 80)
                logger.info("[调试输出结束] 请检查上述样本的prompt是否包含完整JSON结构")
                logger.info("=" * 80)

            # 创建Trainer
            early_stopping_callback = EarlyStoppingCallback(
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

            # 保存训练指标到文件
            metrics_file = self.output_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            # 保存训练历史记录
            training_history = self.history_callback.get_history()
            history_file = self.output_dir / "training_history.json"

            batch_size, gradient_accumulation_steps = self._get_batch_config()

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
                'history': training_history
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

        # 提取数据
        steps = []
        losses = []
        eval_losses = []
        grad_norms = []
        learning_rates = []

        for entry in training_history:
            step = entry.get('step', 0)
            steps.append(step)

            if 'loss' in entry:
                losses.append(entry['loss'])

            if 'eval_loss' in entry:
                eval_losses.append(entry['eval_loss'])

            if 'grad_norm' in entry:
                grad_norms.append(entry['grad_norm'])

            if 'learning_rate' in entry:
                learning_rates.append(entry['learning_rate'])

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {expert_type.upper()} Expert ({self.method_name})',
                     fontsize=16, fontweight='bold')

        # 1. Training Loss
        if losses:
            loss_steps = steps[:len(losses)]
            axes[0, 0].plot(loss_steps, losses, 'b-', linewidth=1.5, alpha=0.7)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Eval Loss
        if eval_losses:
            eval_steps_list = [entry['step'] for entry in training_history if 'eval_loss' in entry]
            axes[0, 1].plot(eval_steps_list, eval_losses, 'r-', linewidth=2, marker='o', markersize=4)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Eval Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Gradient Norm
        if grad_norms:
            grad_steps = steps[:len(grad_norms)]
            axes[1, 0].plot(grad_steps, grad_norms, 'g-', linewidth=1, alpha=0.6)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Learning Rate
        if learning_rates:
            lr_steps = steps[:len(learning_rates)]
            axes[1, 1].plot(lr_steps, learning_rates, 'm-', linewidth=1.5)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        plt.tight_layout()

        # 保存图表
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = curves_dir / f'{expert_type}_expert_{self.method_name}_training_curves_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"训练曲线已保存至: {plot_path}")