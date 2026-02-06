"""
指令生成模块
负责整合MoE系统生成最终指令
"""

from .generator import InstructionGenerator
from .quality_validator import QualityValidator, ValidationResult

__all__ = [
    'InstructionGenerator',
    'QualityValidator',
    'ValidationResult',
]

# 版本信息
__version__ = '1.0.0'