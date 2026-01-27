"""
训练模块
包含数据加载和专家训练功能
"""

from .data_loader import (
    # 数据加载器
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,

    # 数据集类
    InstructionDataset,

    # 工具函数
    split_dataset,
    split_dataset_for_expert,
    create_dataloader,
)

# 待实现的训练器导入（阶段3会用到）
# from .expert_trainer import ExpertTrainer

__all__ = [
    # 数据加载器
    'TextDatasetLoader',
    'ImageDatasetLoader',
    'UMLDatasetLoader',

    # 数据集
    'InstructionDataset',

    # 工具函数
    'split_dataset',
    'split_dataset_for_expert',
    'create_dataloader',

    # 训练器
    # 'ExpertTrainer',
]

# 版本信息
__version__ = '0.1.0'