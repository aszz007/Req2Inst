"""
配置模块初始化
功能：简化配置导入，提供统一的配置访问接口
环境：instruction_generator（单一Conda环境，transformers==4.57.0）
"""

from .settings import (
    # 配置类
    PathConfig,
    LoRAConfig,
    TrainingConfig,
    DeviceConfig,
    VisionModelConfig,

    # 配置获取函数（推荐使用）
    get_path_config,
    get_lora_config,
    get_training_config,
    get_device_config,
    get_vision_model_config,
    set_vision_model_version,

    # 验证函数
    validate_config,
)

# 定义公开的API
__all__ = [
    # ===== 配置类（直接导入） =====
    'PathConfig',
    'LoRAConfig',
    'TrainingConfig',
    'DeviceConfig',
    'VisionModelConfig',

    # ===== 配置获取函数（推荐使用，确保单例） =====
    'get_path_config',
    'get_lora_config',
    'get_training_config',
    'get_device_config',
    'get_vision_model_config',
    'set_vision_model_version',

    # ===== 验证函数 =====
    'validate_config',
]