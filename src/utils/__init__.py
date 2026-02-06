"""
工具模块初始化
功能：简化工具函数导入
"""

# 日志系统
from .logger import (
    setup_logger,
    get_logger,
    log_model_info,
    log_training_metrics,
    log_gpu_memory,
    log_data_info,
    log_config,
    log_recognition_failure
)

# 文件工具
from .file_utils import (
    # 路径操作
    ensure_dir,
    safe_path_join,
    get_relative_path,
    validate_path_exists,

    # JSON操作
    load_json,
    save_json,
    update_json,

    # CSV操作
    load_csv,
    load_csv_chunks,
    save_csv,

    # 模型权重操作
    load_lora_weights,
    save_lora_weights,
    list_checkpoints,

    # 批量操作
    scan_files,
    batch_process_files,

    # 其他工具
    get_file_size,
    copy_file_safe,
    create_backup
)

# 增强评估指标
from .enhanced_metrics import EnhancedMetrics

# 定义公开的API
__all__ = [
    # 日志系统
    'setup_logger',
    'get_logger',
    'log_model_info',
    'log_training_metrics',
    'log_gpu_memory',
    'log_data_info',
    'log_config',
    'log_recognition_failure',

    # 路径操作
    'ensure_dir',
    'safe_path_join',
    'get_relative_path',
    'validate_path_exists',

    # JSON操作
    'load_json',
    'save_json',
    'update_json',

    # CSV操作
    'load_csv',
    'load_csv_chunks',
    'save_csv',

    # 模型权重操作
    'load_lora_weights',
    'save_lora_weights',
    'list_checkpoints',

    # 批量操作
    'scan_files',
    'batch_process_files',

    # 其他工具
    'get_file_size',
    'copy_file_safe',
    'create_backup',

    # 评估指标
    'EnhancedMetrics'
]

# 版本信息
__version__ = '1.1.0'