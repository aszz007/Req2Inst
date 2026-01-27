"""
统一日志系统
功能：提供项目统一的日志记录功能
特性：
  - 按模块分类日志
  - 文件日志 + 控制台输出
  - 自动日志轮转（单文件最大10MB，保留5个）
  - 支持多级别日志（DEBUG/INFO/WARNING/ERROR）
作者：Logger System
日期：2025-01-23
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any
import torch


class LoggerManager:
    """日志管理器 - 统一管理所有模块的logger"""

    # 单例模式
    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化日志管理器"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_log_dirs()

    def _setup_log_dirs(self):
        """设置日志目录"""
        try:
            from config import get_path_config
            path_cfg = get_path_config()
            self.logs_dir = path_cfg.LOGS_DIR
            self.training_logs_dir = path_cfg.TRAINING_LOGS_DIR
            self.inference_logs_dir = path_cfg.INFERENCE_LOGS_DIR
            self.preprocessing_logs_dir = path_cfg.PREPROCESSING_LOGS_DIR
        except ImportError:
            # 如果配置未加载，使用默认路径
            project_root = Path(__file__).parent.parent.parent
            self.logs_dir = project_root / "logs"
            self.training_logs_dir = self.logs_dir / "training"
            self.inference_logs_dir = self.logs_dir / "inference"
            self.preprocessing_logs_dir = self.logs_dir / "preprocessing"

        # 确保目录存在
        for log_dir in [self.training_logs_dir, self.inference_logs_dir,
                        self.preprocessing_logs_dir]:
            log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_dir(self, module_name: str) -> Path:
        """
        根据模块名确定日志目录

        Args:
            module_name: 模块名称（如 'training.text_expert'）

        Returns:
            Path: 日志文件应该存放的目录
        """
        if 'training' in module_name:
            return self.training_logs_dir
        elif 'inference' in module_name or 'generation' in module_name:
            return self.inference_logs_dir
        elif 'preprocessing' in module_name or 'data' in module_name:
            return self.preprocessing_logs_dir
        else:
            return self.logs_dir

    def _create_formatter(self, detailed: bool = True) -> logging.Formatter:
        """
        创建日志格式化器

        Args:
            detailed: 是否使用详细格式

        Returns:
            logging.Formatter: 格式化器
        """
        if detailed:
            # 详细格式：时间戳 | 级别 | 模块名 | 函数名 | 行号 | 消息
            fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        else:
            # 简洁格式：时间戳 | 级别 | 模块名 | 消息
            fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

        return logging.Formatter(
            fmt=fmt,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def setup_logger(
            self,
            module_name: str,
            level: int = logging.INFO,
            console_output: bool = True,
            file_output: bool = True
    ) -> logging.Logger:
        """
        为指定模块创建或获取logger

        Args:
            module_name: 模块名称（如 'training.text_expert'）
            level: 日志级别（DEBUG/INFO/WARNING/ERROR）
            console_output: 是否输出到控制台
            file_output: 是否输出到文件

        Returns:
            logging.Logger: 配置好的logger实例
        """
        # 如果logger已存在，直接返回
        if module_name in self._loggers:
            return self._loggers[module_name]

        # 创建新logger
        logger = logging.getLogger(module_name)
        logger.setLevel(level)
        logger.propagate = False  # 不向父logger传播

        # 清除已有的handlers（避免重复）
        logger.handlers.clear()

        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(self._create_formatter(detailed=False))
            logger.addHandler(console_handler)

        # 文件输出
        if file_output:
            log_dir = self._get_log_dir(module_name)
            date_str = datetime.now().strftime('%Y-%m-%d')
            log_file = log_dir / f"{module_name.replace('.', '_')}_{date_str}.log"

            # 使用RotatingFileHandler实现日志轮转
            # maxBytes=10MB, backupCount=5（保留5个备份文件）
            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(self._create_formatter(detailed=True))
            logger.addHandler(file_handler)

        # 保存到字典
        self._loggers[module_name] = logger

        return logger

    def get_logger(self, module_name: str) -> logging.Logger:
        """
        获取已存在的logger，如果不存在则创建

        Args:
            module_name: 模块名称

        Returns:
            logging.Logger: logger实例
        """
        if module_name not in self._loggers:
            return self.setup_logger(module_name)
        return self._loggers[module_name]


# ===== 全局日志管理器实例 =====
_logger_manager = LoggerManager()


def setup_logger(
        module_name: str,
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
) -> logging.Logger:
    """
    便捷函数：为模块设置logger

    Args:
        module_name: 模块名称
        level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件

    Returns:
        logging.Logger: 配置好的logger

    Example:
        >>> from src.utils.logger import setup_logger
        >>> logger = setup_logger('training.text_expert')
        >>> logger.info('开始训练文本专家')
    """
    return _logger_manager.setup_logger(module_name, level, console_output, file_output)


def get_logger(module_name: str) -> logging.Logger:
    """
    便捷函数：获取logger

    Args:
        module_name: 模块名称

    Returns:
        logging.Logger: logger实例
    """
    return _logger_manager.get_logger(module_name)


# ===== 专用日志记录函数 =====

def log_model_info(logger: logging.Logger, model: Any, model_name: str = "模型"):
    """
    记录模型信息

    Args:
        logger: logger实例
        model: 模型对象
        model_name: 模型名称
    """
    try:
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info(f"{model_name}信息")
        logger.info("=" * 60)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"可训练比例: {100 * trainable_params / total_params:.2f}%")

        # 如果是CUDA模型，记录显存使用
        if next(model.parameters()).is_cuda:
            device_id = next(model.parameters()).get_device()
            logger.info(f"GPU显存使用: {torch.cuda.memory_allocated(device_id) / 1024 ** 3:.2f} GB")
            logger.info(f"GPU显存缓存: {torch.cuda.memory_reserved(device_id) / 1024 ** 3:.2f} GB")

        logger.info("=" * 60)

    except Exception as e:
        logger.warning(f"无法记录模型信息: {str(e)}")


def log_training_metrics(
        logger: logging.Logger,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        prefix: str = ""
):
    """
    记录训练指标

    Args:
        logger: logger实例
        epoch: 当前epoch
        step: 当前步数
        metrics: 指标字典 {'loss': 0.5, 'accuracy': 0.9}
        prefix: 前缀（如 'train' 或 'eval'）
    """
    prefix_str = f"[{prefix}] " if prefix else ""
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{prefix_str}Epoch {epoch} Step {step} | {metric_str}")


def log_gpu_memory(logger: logging.Logger, device_id: int = 0):
    """
    记录GPU显存使用情况

    Args:
        logger: logger实例
        device_id: GPU设备ID
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，无法记录GPU显存")
        return

    try:
        allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024 ** 3

        logger.info(f"GPU显存: 已分配={allocated:.2f}GB, 已缓存={reserved:.2f}GB, 峰值={max_allocated:.2f}GB")
    except Exception as e:
        logger.warning(f"无法记录GPU显存: {str(e)}")


def log_data_info(
        logger: logging.Logger,
        dataset_name: str,
        train_size: int,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None
):
    """
    记录数据集信息

    Args:
        logger: logger实例
        dataset_name: 数据集名称
        train_size: 训练集大小
        val_size: 验证集大小（可选）
        test_size: 测试集大小（可选）
    """
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_name}")
    logger.info("=" * 60)
    logger.info(f"训练集样本数: {train_size}")

    if val_size is not None:
        logger.info(f"验证集样本数: {val_size}")

    if test_size is not None:
        logger.info(f"测试集样本数: {test_size}")

    total = train_size + (val_size or 0) + (test_size or 0)
    logger.info(f"总样本数: {total}")
    logger.info("=" * 60)


def log_config(logger: logging.Logger, config: Dict[str, Any], config_name: str = "配置"):
    """
    记录配置信息

    Args:
        logger: logger实例
        config: 配置字典
        config_name: 配置名称
    """
    logger.info("=" * 60)
    logger.info(f"{config_name}")
    logger.info("=" * 60)

    for key, value in config.items():
        logger.info(f"{key}: {value}")

    logger.info("=" * 60)

def log_recognition_failure(
        logger: logging.Logger,
        file_path: str,
        error: str,
        retry_count: int = 0
):
    """
    记录识别失败信息

    Args:
        logger: logger实例
        file_path: 失败的文件路径
        error: 错误信息
        retry_count: 重试次数
    """
    retry_info = f"(重试{retry_count}次后)" if retry_count > 0 else ""
    logger.error(f"识别失败{retry_info}: {file_path}")
    logger.error(f"  错误详情: {error}")

# ===== 测试代码 =====
if __name__ == "__main__":
    print("=" * 60)
    print("日志系统测试")
    print("=" * 60)

    # 测试1：创建不同模块的logger
    print("\n【测试1】创建多个logger")
    print("-" * 60)

    logger_train = setup_logger('training.text_expert', level=logging.DEBUG)
    logger_inference = setup_logger('inference.generation', level=logging.INFO)
    logger_data = setup_logger('preprocessing.data_loader', level=logging.INFO)

    # 测试2：不同级别的日志
    print("\n【测试2】不同级别的日志输出")
    print("-" * 60)

    logger_train.debug("这是DEBUG级别的消息（详细调试信息）")
    logger_train.info("这是INFO级别的消息（关键流程信息）")
    logger_train.warning("这是WARNING级别的消息（警告信息）")
    logger_train.error("这是ERROR级别的消息（错误信息）")

    # 测试3：记录模型信息（模拟）
    print("\n【测试3】记录模型信息")
    print("-" * 60)


    class MockModel:
        def __init__(self):
            self.param1 = torch.nn.Parameter(torch.randn(1000, 1000))
            self.param2 = torch.nn.Parameter(torch.randn(500, 500))

        def parameters(self):
            return [self.param1, self.param2]


    mock_model = MockModel()
    log_model_info(logger_train, mock_model, "测试模型")

    # 测试4：记录训练指标
    print("\n【测试4】记录训练指标")
    print("-" * 60)

    metrics = {
        'loss': 0.5234,
        'accuracy': 0.8765,
        'learning_rate': 2e-4
    }
    log_training_metrics(logger_train, epoch=1, step=100, metrics=metrics, prefix="train")

    # 测试5：记录数据集信息
    print("\n【测试5】记录数据集信息")
    print("-" * 60)

    log_data_info(logger_data, "CCHIT数据集", train_size=800, val_size=100, test_size=100)

    # 测试6：记录配置信息
    print("\n【测试6】记录配置信息")
    print("-" * 60)

    config = {
        'batch_size': 4,
        'learning_rate': 2e-4,
        'epochs': 3,
        'lora_rank': 8
    }
    log_config(logger_train, config, "训练配置")

    # 测试7：GPU显存记录
    print("\n【测试7】GPU显存记录")
    print("-" * 60)

    log_gpu_memory(logger_train)

    # 测试8：记录识别失败
    print("\n【测试8】记录识别失败")
    print("-" * 60)
    log_recognition_failure(logger_data, "/path/to/image.jpg", "JSON解析错误", retry_count=2)

    print("\n日志系统测试完成！")
    print("请检查 logs/ 目录下的日志文件")