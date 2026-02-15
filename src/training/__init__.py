"""
训练模块
包含数据加载和专家训练功能
"""

from .data_loader import (
    # 数据加载器
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,
    GeneralDatasetLoader,

    # 数据集类
    InstructionDataset,

    # 数据收集器
    InstructionDataCollator,

    # 工具函数
    split_dataset,
    split_dataset_for_expert,
    create_dataloader,
)

from .lora_trainer import ExpertTrainer

__all__ = [
    # 数据加载器
    'TextDatasetLoader',
    'ImageDatasetLoader',
    'UMLDatasetLoader',
    'GeneralDatasetLoader',

    # 数据集
    'InstructionDataset',

    # 数据收集器
    'InstructionDataCollator',

    # 工具函数
    'split_dataset',
    'split_dataset_for_expert',
    'create_dataloader',

    # 训练器
    'ExpertTrainer',
]

# 版本信息
__version__ = '0.1.0'