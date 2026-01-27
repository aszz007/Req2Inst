"""
配置模块初始化
功能：简化配置导入，提供统一的配置访问接口
作者：Crowdsourcing Instruction Generator Team
日期：2025-01-26（优化版）
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



# 版本信息
__version__ = '1.1.1'
__author__ = 'Crowdsourcing Instruction Generator Team'
__description__ = 'LoRA-MoE多模态众包指令生成系统 - 配置模块'


# ==================== 使用示例 ====================
"""
推荐的配置使用方式：

# 方式1: 获取配置实例（推荐，确保单例）
from config import get_path_config, get_lora_config

path_cfg = get_path_config()
lora_cfg = get_lora_config('conservative')  # 或 'aggressive'

# 使用配置
model_path = path_cfg.QWEN_7B_CHAT_PATH
lora_rank = lora_cfg.rank

# 方式2: 直接导入配置类（高级用法）
from config import PathConfig, LoRAConfig

path_cfg = PathConfig()  # 注意：这会创建新实例，不推荐

# 方式3: 验证所有配置
from config import validate_config

is_valid, messages = validate_config()
if not is_valid:
    print("配置验证失败！")
    for msg in messages:
        print(msg)
"""