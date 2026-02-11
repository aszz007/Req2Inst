"""
项目配置中心
功能：统一管理所有路径、超参数、训练配置
作者：System Configuration
日期：2025-01-26（修复版）
"""

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class VisionModelConfig:
    """视觉模型版本配置"""

    # 当前使用的视觉模型版本（'qwen2.5' 或 'qwen3'）
    version: str = "qwen2.5"  # 默认使用qwen2.5作为baseline

    # 支持的版本列表
    SUPPORTED_VERSIONS: List[str] = None

    def __post_init__(self):
        """初始化支持的版本列表"""
        if self.SUPPORTED_VERSIONS is None:
            self.SUPPORTED_VERSIONS = ["qwen2.5", "qwen3"]

        # 验证版本
        if self.version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"不支持的视觉模型版本: {self.version}，"
                f"支持的版本: {self.SUPPORTED_VERSIONS}"
            )

    def get_model_name(self) -> str:
        """获取完整模型名称"""
        return {
            "qwen2.5": "Qwen2.5-VL-7B-Instruct",
            "qwen3": "Qwen3-VL-8B-Instruct"
        }[self.version]

    def get_model_size(self) -> str:
        """获取模型大小描述"""
        return {
            "qwen2.5": "7B",
            "qwen3": "8B"
        }[self.version]

class PathConfig:
    """路径配置类 - 管理所有项目路径"""

    def __init__(self):
        """初始化路径配置"""
        # ==================== 项目根目录 ====================
        self.PROJECT_ROOT = Path(__file__).parent.parent.resolve()

        # ==================== 基础模型路径 ====================
        self.BASE_MODELS_DIR = self.PROJECT_ROOT / "base_models"

        # Qwen-7B-Chat模型路径
        self.QWEN_7B_CHAT_PATH = (
            self.BASE_MODELS_DIR / "qwen-7B-Chat" / "Qwen" / "Qwen-7B-Chat"
        )

        # Qwen2.5-VL-7B模型路径
        self.QWEN_VL_7B_PATH = (
            self.BASE_MODELS_DIR / "qwen2.5-VL-7B" / "qwen" / "Qwen2.5-VL-7B-Instruct"
        )

        # Qwen3-VL-8B模型路径
        self.QWEN_VL_3_PATH = (
                self.BASE_MODELS_DIR / "qwen3-VL-8B" / "qwen" / "Qwen3-VL-8B-Instruct"
        )

        # 视觉模型路径映射（集中配置）
        self.VISION_MODEL_PATHS = {
            'qwen2.5': self.QWEN_VL_7B_PATH,
            'qwen3': self.QWEN_VL_3_PATH,
        }

        # ==================== 数据相关路径 ====================
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.INTERIM_DATA_DIR = self.DATA_DIR / "interim"

        # 原始数据子目录
        self.RAW_IMAGE_DIR = self.RAW_DATA_DIR / "image"
        self.RAW_TEXT_DIR = self.RAW_DATA_DIR / "text"
        self.RAW_UML_DIR = self.RAW_DATA_DIR / "uml"

        # 原始数据默认测试目录（用于批量识别脚本）
        self.COCO_500_DIR = self.RAW_IMAGE_DIR / "coco_500"
        self.ROBOFLOW_UML_DIR = self.RAW_UML_DIR / "roboflow_uml"
        self.MDPI_UML_DIR = self.RAW_UML_DIR / "mdpi_uml"
        self.PLANT_UML_DIR = self.RAW_UML_DIR / "plant_uml"

        # 中间处理结果子目录
        self.INTERIM_IMAGE_DIR = self.INTERIM_DATA_DIR / "image"
        self.INTERIM_TEXT_DIR = self.INTERIM_DATA_DIR / "text"
        self.INTERIM_UML_DIR = self.INTERIM_DATA_DIR / "uml"

        # ==================== 数据集路径 ====================
        self.DATASET_DIR = self.PROJECT_ROOT / "dataset"
        self.TEXT_DATASET_DIR = self.DATASET_DIR / "text"
        self.IMAGE_DATASET_DIR = self.DATASET_DIR / "image"
        self.UML_DATASET_DIR = self.DATASET_DIR / "uml"
        self.GENERAL_DATASET_DIR = self.DATASET_DIR / "general"

        # 具体数据集文件
        self.IMAGE_DATASET_CSV = self.IMAGE_DATASET_DIR / "image_dataset.csv"

        # UML数据集（单一版本 - 1500条数据）
        self.UML_DATASET_CSV = self.UML_DATASET_DIR / "uml_dataset_qwen3_v3.csv"

        # General数据集（不需要单独文件，动态加载text+image+uml）
        # self.GENERAL_DATASET_CSV - 已移除，General专家直接从三个数据源加载

        # 文本数据集（多个文件）
        self.TEXT_DATASET_FILES = {
            'CCHIT': self.TEXT_DATASET_DIR / "CCHIT_dataset.csv",
            'CM1': self.TEXT_DATASET_DIR / "CM1_dataset.csv",
            'GANNT': self.TEXT_DATASET_DIR / "GANNT_dataset.csv",
            'InfusionPump': self.TEXT_DATASET_DIR / "InfusionPump_dataset.csv",
            'Modis': self.TEXT_DATASET_DIR / "Modis_dataset.csv",
            'WARC': self.TEXT_DATASET_DIR / "WARC_dataset.csv"
        }

        # ==================== 推理输入路径 ====================
        self.INPUTS_DIR = self.PROJECT_ROOT / "inputs"
        self.INPUT_TEXT_DIR = self.INPUTS_DIR / "text"
        self.INPUT_IMAGE_DIR = self.INPUTS_DIR / "image"
        self.INPUT_UML_DIR = self.INPUTS_DIR / "uml"

        # ==================== LoRA权重路径 ====================
        self.LORA_WEIGHTS_DIR = self.PROJECT_ROOT / "lora_weights"
        self.EXPERTS_DIR = self.LORA_WEIGHTS_DIR / "experts"

        # 各专家权重路径
        self.TEXT_EXPERT_WEIGHTS = self.EXPERTS_DIR / "text_expert"

        # Image Expert只有1个版本（数据集只有1个版本）
        self.IMAGE_EXPERT_WEIGHTS = self.EXPERTS_DIR / "image_expert"

        # UML Expert（单一版本，使用qwen3_v3数据集）
        self.UML_EXPERT_WEIGHTS = self.EXPERTS_DIR / "uml_expert"

        # General Expert（单一版本）
        self.GENERAL_EXPERT_WEIGHTS = self.EXPERTS_DIR / "general_expert"

        # 专家LoRA权重映射（集中配置）
        self.EXPERT_LORA_PATHS = {
            # Text Expert（1个版本）
            'text': self.TEXT_EXPERT_WEIGHTS,
            'text_expert': self.TEXT_EXPERT_WEIGHTS,

            # Image Expert（1个版本）
            'image': self.IMAGE_EXPERT_WEIGHTS,
            'image_expert': self.IMAGE_EXPERT_WEIGHTS,

            # UML Expert（1个版本）
            'uml': self.UML_EXPERT_WEIGHTS,
            'uml_expert': self.UML_EXPERT_WEIGHTS,

            # General Expert（1个版本）
            'general': self.GENERAL_EXPERT_WEIGHTS,
            'general_expert': self.GENERAL_EXPERT_WEIGHTS,
        }

        # ==================== Checkpoint路径 ====================
        self.CHECKPOINTS_DIR = self.PROJECT_ROOT / "checkpoints"
        self.TEXT_EXPERT_CKPT = self.CHECKPOINTS_DIR / "text_expert_training"
        self.IMAGE_EXPERT_CKPT = self.CHECKPOINTS_DIR / "image_expert_training"
        self.UML_EXPERT_CKPT = self.CHECKPOINTS_DIR / "uml_expert_training"

        # ==================== 输出路径 ====================
        self.OUTPUTS_DIR = self.PROJECT_ROOT / "outputs"
        self.GENERATED_INSTRUCTIONS_DIR = self.OUTPUTS_DIR / "generated_instructions"
        self.RECOGNITION_RESULTS_DIR = self.OUTPUTS_DIR / "recognition_results"  # 识别结果输出目录
        self.IMAGE_RECOGNITION_DIR = self.RECOGNITION_RESULTS_DIR / "image"
        self.UML_RECOGNITION_DIR = self.RECOGNITION_RESULTS_DIR / "uml"
        self.EVALUATIONS_DIR = self.OUTPUTS_DIR / "evaluations"
        self.METRICS_DIR = self.EVALUATIONS_DIR / "metrics"
        self.COMPARISONS_DIR = self.EVALUATIONS_DIR / "comparisons"
        self.REPORTS_DIR = self.OUTPUTS_DIR / "reports"

        # ==================== 日志路径 ====================
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        self.TRAINING_LOGS_DIR = self.LOGS_DIR / "training"
        self.INFERENCE_LOGS_DIR = self.LOGS_DIR / "inference"
        self.PREPROCESSING_LOGS_DIR = self.LOGS_DIR / "preprocessing"

    def get_vision_model_path(self, version: str = None) -> Path:
        """
        获取视觉模型路径

        Args:
            version: 模型版本（'qwen2.5' 或 'qwen3'），None则使用配置中的版本

        Returns:
            Path: 模型路径
        """
        if version is None:
            vision_cfg = get_vision_model_config()
            version = vision_cfg.version

        if version not in self.VISION_MODEL_PATHS:
            raise ValueError(
                f"不支持的视觉模型版本: {version}，"
                f"支持的版本: {list(self.VISION_MODEL_PATHS.keys())}"
            )

        return self.VISION_MODEL_PATHS[version]

    def get_expert_weight_path(self, expert_name: str) -> Path:
        """
        获取专家LoRA权重路径

        Args:
            expert_name: 专家名称（'text', 'image', 'uml', 'general'）

        Returns:
            Path: LoRA权重路径
        """
        # 移除可能的_expert后缀，统一处理
        base_name = expert_name.replace('_expert', '')
        expert_key = base_name

        if expert_key in self.EXPERT_LORA_PATHS:
            return self.EXPERT_LORA_PATHS[expert_key]
        else:
            # 如果未定义，使用约定的命名规则
            return self.EXPERTS_DIR / expert_key

    def get_checkpoint_path(self, expert_name: str) -> Path:
        """获取专家训练检查点路径"""
        return self.CHECKPOINTS_DIR / f"{expert_name}_training"

    def create_directories(self):
        """创建所有必要的目录"""
        dirs = [
            # LoRA和检查点
            self.LORA_WEIGHTS_DIR,
            self.EXPERTS_DIR,
            self.CHECKPOINTS_DIR,
            # 数据集目录
            self.TEXT_DATASET_DIR,
            self.IMAGE_DATASET_DIR,
            self.UML_DATASET_DIR,
            self.GENERAL_DATASET_DIR,
            # 中间数据
            self.INTERIM_IMAGE_DIR,
            self.INTERIM_TEXT_DIR,
            self.INTERIM_UML_DIR,
            # 推理输入
            self.INPUT_TEXT_DIR,
            self.INPUT_IMAGE_DIR,
            self.INPUT_UML_DIR,
            # 输出
            self.GENERATED_INSTRUCTIONS_DIR,
            self.IMAGE_RECOGNITION_DIR,
            self.UML_RECOGNITION_DIR,
            self.METRICS_DIR,
            self.COMPARISONS_DIR,
            self.REPORTS_DIR,
            # 日志
            self.TRAINING_LOGS_DIR,
            self.INFERENCE_LOGS_DIR,
            self.PREPROCESSING_LOGS_DIR,
        ]

        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"✓ 已创建 {len(dirs)} 个必要目录")


@dataclass
class LoRAConfig:
    """LoRA超参数配置"""

    # LoRA rank (秩)
    rank: int = 8

    # LoRA alpha (缩放因子)
    alpha: int = 16

    # Dropout概率
    dropout: float = 0.05

    # 目标模块（应用LoRA的层）
    target_modules: List[str] = None

    # 任务类型
    task_type: str = "CAUSAL_LM"

    # 是否训练偏置
    bias: str = "none"

    def __post_init__(self):
        """
        设置默认目标模块

        注意：不同模型使用不同的注意力层命名：
        - Qwen-7B-Chat: 使用 ["c_attn"] (concatenated attention)
        - Qwen2.5-VL/Qwen3-VL: 使用 ["q_proj", "k_proj", "v_proj", "o_proj"]

        ExpertTrainer会根据模型类型自动选择正确的target_modules，
        此处默认值主要用于视觉模型的兼容性。
        """
        if self.target_modules is None:
            # 默认使用标准Transformers命名（适用于Qwen2.5-VL/Qwen3-VL）
            self.target_modules = [
                "q_proj",  # Query投影层
                "k_proj",  # Key投影层
                "v_proj",  # Value投影层
                "o_proj",  # Output投影层
            ]

    @classmethod
    def get_conservative_config(cls):
        """保守配置（较小的rank，适合数据量小的场景）"""
        return cls(rank=8, alpha=16, dropout=0.05)

    @classmethod
    def get_aggressive_config(cls):
        """激进配置（较大的rank，适合数据量充足的场景）"""
        return cls(rank=16, alpha=32, dropout=0.1)


@dataclass
class TrainingConfig:
    """训练配置"""

    # ==================== 基础配置 ====================
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_epochs: int = 3
    learning_rate: float = 2e-4

    # ==================== 优化器配置 ====================
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # ==================== 学习率调度 ====================
    lr_scheduler_type: str = "cosine"

    # ==================== 日志与保存 ====================
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    # ==================== 评估配置 ====================
    evaluation_strategy: str = "steps"
    eval_steps: int = 100

    # ==================== 其他配置 ====================
    fp16: bool = True  # 混合精度训练
    max_grad_norm: float = 1.0
    seed: int = 42
    max_seq_length: int = 1536  # 最大序列长度（优化显存占用，足够覆盖长文本需求）

    # ==================== 数据集划分比例 ====================
    # 文本数据集（2400条）
    text_train_ratio: float = 0.8
    text_val_ratio: float = 0.1
    text_test_ratio: float = 0.1

    # 图像数据集（500条）
    image_train_ratio: float = 0.8
    image_val_ratio: float = 0.1
    image_test_ratio: float = 0.1

    # UML数据集（1500条）- 使用标准80:10:10划分
    uml_train_ratio: float = 0.8
    uml_val_ratio: float = 0.1
    uml_test_ratio: float = 0.1

@dataclass
class TrainingConfig4090:
    """针对RTX 4090优化的训练配置"""

    # ===== 基础训练参数（4090优化）=====
    batch_size = 8  # 从4提升到8（4090显存充足）
    gradient_accumulation_steps = 2  # 从4降到2（保持有效batch=16）
    num_epochs = 3
    learning_rate = 2e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_seq_length = 1536  # 优化显存占用（从2048降低），足够覆盖长文本需求

    # ===== 4090专属优化 =====
    use_flash_attention = True  # 启用Flash Attention 2（提速30%）
    bf16 = True  # 使用BF16（4090支持，比FP16更稳定）
    tf32 = True  # 启用TF32（4090特有，免费提速）

    # ===== 数据加载优化 =====
    dataloader_num_workers = 8  # 从2提升到8（充分利用CPU）
    dataloader_pin_memory = True
    dataloader_prefetch_factor = 4  # 预加载4个batch

    # ===== 梯度优化 =====
    gradient_checkpointing = False  # 4090显存足够，关闭以提速
    max_grad_norm = 1.0

    # ===== 保存策略 =====
    save_strategy = "epoch"
    save_total_limit = 2  # 只保留最好的2个检查点（节省空间）
    evaluation_strategy = "epoch"
    logging_steps = 5  # 从10降到5（更频繁的日志）

    # ===== 优化器配置 =====
    optimizer_type = "adamw_torch_fused"  # 融合优化器（4090提速15%）
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8

@dataclass
class DeviceConfig:
    """设备配置"""

    device: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    is_high_end_gpu: bool = False
    enable_streaming: bool = False  # 是否启用流式输出（默认关闭）

    def __post_init__(self):
        """自动检测设备和GPU型号"""
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

                # 检测是否为高端GPU（支持高效fp16推理）
                self.is_high_end_gpu = self._detect_high_end_gpu()

                print(f"[设备] 使用GPU: {self.gpu_name}")
                print(f"[设备] 显存: {self.gpu_memory_gb:.2f}GB")
                print(f"[设备] 高端GPU模式: {'是' if self.is_high_end_gpu else '否'}")
            else:
                self.device = "cpu"
                print("[设备] 使用CPU")

    def _detect_high_end_gpu(self) -> bool:
        """
        检测是否为高端GPU

        高端GPU定义（支持高效fp16，显存>=20GB）：
        - RTX 4090 (24GB)
        - RTX 4080 (16GB)
        - A100 (40GB/80GB)
        - H100 (80GB)
        - V100 (16GB/32GB)
        - A6000 (48GB)

        Returns:
            bool: 是否为高端GPU
        """
        if not self.gpu_name:
            return False

        gpu_lower = self.gpu_name.lower()

        # 高端GPU列表
        high_end_keywords = [
            '4090', '4080',  # RTX 40系高端
            'a100', 'h100', 'a6000',  # 数据中心GPU
            'v100',  # 上一代数据中心GPU
        ]

        # 检查关键词
        for keyword in high_end_keywords:
            if keyword in gpu_lower:
                return True

        # 备用判断：显存>=20GB也视为高端GPU
        if self.gpu_memory_gb and self.gpu_memory_gb >= 20.0:
            return True

        return False

    def get_device(self) -> str:
        """获取设备名称"""
        return self.device

    def get_gpu_info(self) -> dict:
        """
        获取GPU详细信息

        Returns:
            dict: GPU信息
        """
        return {
            'device': self.device,
            'gpu_name': self.gpu_name,
            'gpu_memory_gb': self.gpu_memory_gb,
            'is_high_end_gpu': self.is_high_end_gpu
        }

    def should_use_quantization(self) -> bool:
        """
        判断是否应该使用量化

        Returns:
            bool: True表示使用4bit量化，False表示使用fp16
        """
        # 高端GPU不使用量化（fp16即可）
        # 其他GPU使用4bit量化（节省显存）
        return not self.is_high_end_gpu

    def get_gpu_tier(self) -> str:
        """
        获取GPU性能分级

        Returns:
            str: GPU性能级别 ('low' / 'mid' / 'high')
        """
        if not torch.cuda.is_available():
            return 'low'

        # 高端GPU (>16GB 或在高端列表中)
        if self.is_high_end_gpu:
            return 'high'

        # 中端GPU (7.5-16GB, 包括8GB显卡如RTX 4060)
        if self.gpu_memory_gb and 7.5 <= self.gpu_memory_gb <= 16.0:
            return 'mid'

        # 低端GPU (<7.5GB)
        return 'low'

    def get_generation_config(self, task_type: str = 'uml') -> dict:
        """
        根据GPU性能获取生成配置参数

        Args:
            task_type: 任务类型 ('uml' 或 'image')

        Returns:
            dict: 生成配置参数
        """
        tier = self.get_gpu_tier()

        # UML识别生成参数配置
        uml_configs = {
            'high': {
                'max_new_tokens': 4096,
                'batch_size': 4,
                'temperature': 0.3,
                'top_p': 0.85,
                'use_cache': True,
            },
            'mid': {
                'max_new_tokens': 2048,
                'batch_size': 2,
                'temperature': 0.5,
                'top_p': 0.9,
                'use_cache': True,
            },
            'low': {
                'max_new_tokens': 1024,
                'batch_size': 1,
                'temperature': 0.6,
                'top_p': 0.95,
                'use_cache': True,
            }
        }

        # 图像识别生成参数配置
        image_configs = {
            'high': {
                'max_new_tokens': 512,
                'batch_size': 4,
                'temperature': 0.3,
                'top_p': 0.85,
                'use_cache': True,
            },
            'mid': {
                'max_new_tokens': 200,
                'batch_size': 2,
                'temperature': 0.5,
                'top_p': 0.9,
                'use_cache': True,
            },
            'low': {
                'max_new_tokens': 150,
                'batch_size': 1,
                'temperature': 0.6,
                'top_p': 0.95,
                'use_cache': True,
            }
        }

        if task_type == 'uml':
            return uml_configs.get(tier, uml_configs['mid'])
        elif task_type == 'image':
            return image_configs.get(tier, image_configs['mid'])
        else:
            return uml_configs.get(tier, uml_configs['mid'])

# ==================== 全局配置实例 ====================
_path_config = None
_lora_config = None
_training_config = None
_device_config = None
_vision_model_config = None  # 视觉模型选择


def get_path_config() -> PathConfig:
    """获取路径配置单例"""
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    return _path_config


def get_lora_config(config_type: str = "conservative") -> LoRAConfig:
    """
    获取LoRA配置

    Args:
        config_type: 'conservative' 或 'aggressive'
    """
    global _lora_config
    if _lora_config is None:
        if config_type == "aggressive":
            _lora_config = LoRAConfig.get_aggressive_config()
        else:
            _lora_config = LoRAConfig.get_conservative_config()
    return _lora_config


def get_training_config() -> TrainingConfig:
    """获取训练配置单例"""
    global _training_config
    if _training_config is None:
        _training_config = TrainingConfig()
    return _training_config


def get_device_config() -> DeviceConfig:
    """获取设备配置单例"""
    global _device_config
    if _device_config is None:
        _device_config = DeviceConfig()
    return _device_config


def set_streaming_mode(enable: bool):
    """
    设置流式输出模式

    Args:
        enable: True启用流式输出，False禁用
    """
    global _device_config
    if _device_config is None:
        _device_config = DeviceConfig()
    _device_config.enable_streaming = enable
    print(f"流式输出模式: {'启用' if enable else '禁用'}")


def get_vision_model_config(version: str = None) -> VisionModelConfig:
    """
    获取视觉模型配置（支持多级优先级）

    优先级：命令行参数 > 环境变量 > 配置默认值

    Args:
        version: 强制指定版本（可选），None则按优先级自动选择
    """
    global _vision_model_config

    # 优先级1：命令行参数（最高优先级）
    if version is not None:
        _vision_model_config = VisionModelConfig(version=version)
        return _vision_model_config

    # 优先级2：环境变量
    env_version = os.environ.get('QWEN_VISION_VERSION')
    if env_version:
        _vision_model_config = VisionModelConfig(version=env_version)
        return _vision_model_config

    # 优先级3：配置默认值（如果未初始化）
    if _vision_model_config is None:
        _vision_model_config = VisionModelConfig()  # 使用dataclass默认值qwen2.5

    return _vision_model_config


def set_vision_model_version(version: str):
    """
    切换视觉模型版本（用于实验对比）

    Args:
        version: 'qwen2.5' 或 'qwen3'
    """
    global _vision_model_config
    _vision_model_config = VisionModelConfig(version=version)
    print(f"✓ 已切换视觉模型版本: {version}")
    print(f"  模型: {_vision_model_config.get_model_name()}")


def validate_config() -> tuple:
    """
    验证所有配置

    Returns:
        tuple: (是否通过验证, 错误/警告信息列表)
    """
    messages = []
    is_valid = True

    print("\n" + "=" * 60)
    print("配置验证中...")
    print("=" * 60)

    # 1. 验证路径
    path_cfg = get_path_config()

    print("\n[1/5] 检查基础模型路径...")
    if not path_cfg.QWEN_7B_CHAT_PATH.exists():
        messages.append(f"❌ Qwen-7B-Chat模型未找到: {path_cfg.QWEN_7B_CHAT_PATH}")
        is_valid = False
    else:
        print(f"✓ Qwen-7B-Chat模型路径正确")

    # 检查两个版本的视觉模型
    for version in ['qwen2.5', 'qwen3']:
        model_path = path_cfg.get_vision_model_path(version)
        if not model_path.exists():
            messages.append(f"⚠ {version.upper()} 模型未找到: {model_path}")
            print(f"⚠ {version.upper()} 模型未找到（如暂未下载可忽略）")
        else:
            print(f"✓ {version.upper()} 视觉模型路径正确")

    # 2. 验证数据集
    print("\n[2/5] 检查数据集...")
    if path_cfg.IMAGE_DATASET_CSV.exists():
        print(f"✓ 图像数据集存在")
    else:
        messages.append(f"⚠ 图像数据集未找到: {path_cfg.IMAGE_DATASET_CSV}")

    text_dataset_count = sum(1 for f in path_cfg.TEXT_DATASET_FILES.values() if f.exists())
    print(f"✓ 找到 {text_dataset_count}/{len(path_cfg.TEXT_DATASET_FILES)} 个文本数据集文件")

    if not path_cfg.UML_DATASET_CSV.exists():
        messages.append(f"⚠ UML数据集未找到（可能尚未创建）: {path_cfg.UML_DATASET_CSV}")

    # 3. 验证CUDA环境
    print("\n[3/5] 检查CUDA环境...")
    device_cfg = get_device_config()

    if device_cfg.device != "cuda":
        messages.append("⚠ CUDA不可用，将使用CPU模式（速度极慢）")
    else:
        print(f"✓ CUDA可用")

    # 4. 验证必要依赖
    print("\n[4/5] 检查依赖库...")
    required_packages = {
        'transformers': '模型加载',
        'torch': 'PyTorch框架',
        'peft': 'LoRA训练',
        'bitsandbytes': '4bit量化'
    }

    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} ({description})")
        except ImportError:
            messages.append(f"❌ 缺少依赖: {package} - {description}")
            is_valid = False

    # 5. 创建必要目录
    print("\n[5/5] 创建必要目录...")
    try:
        path_cfg.create_directories()
    except Exception as e:
        messages.append(f"❌ 创建目录失败: {str(e)}")
        is_valid = False

    # 输出验证结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    if is_valid:
        print("✓ 配置验证通过")
    else:
        print("✗ 配置验证失败，请修复以下问题：")

    for msg in messages:
        print(f"  {msg}")

    print("=" * 60 + "\n")

    return is_valid, messages


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("配置系统测试")
    print("=" * 60)

    # 测试路径配置
    path_cfg = get_path_config()
    print("\n[路径配置]")
    print(f"项目根目录: {path_cfg.PROJECT_ROOT}")
    print(f"Qwen-7B-Chat: {path_cfg.QWEN_7B_CHAT_PATH}")
    print(f"Qwen2.5-VL: {path_cfg.QWEN_VL_7B_PATH}")
    print(f"\n专家LoRA路径:")
    for expert in ['text', 'image', 'uml', 'general']:
        print(f"  {expert}: {path_cfg.get_expert_weight_path(expert)}")

    # 测试LoRA配置
    lora_cfg = get_lora_config("conservative")
    print(f"\n[LoRA配置]")
    print(f"Rank: {lora_cfg.rank}")
    print(f"Alpha: {lora_cfg.alpha}")
    print(f"Target Modules: {lora_cfg.target_modules}")

    # 测试训练配置
    train_cfg = get_training_config()
    print(f"\n[训练配置]")
    print(f"Batch Size: {train_cfg.batch_size}")
    print(f"有效Batch Size: {train_cfg.batch_size * train_cfg.gradient_accumulation_steps}")
    print(f"Epochs: {train_cfg.num_epochs}")
    print(f"Learning Rate: {train_cfg.learning_rate}")

    # 测试设备配置
    device_cfg = get_device_config()
    print(f"\n[设备配置]")
    print(f"Device: {device_cfg.get_device()}")

    # 完整验证
    print("\n" + "=" * 60)
    validate_config()