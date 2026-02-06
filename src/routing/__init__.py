"""
路由模块
负责MoE专家路由和选择
"""

from .expert_router import ExpertRouter, RoutingResult, ExpertConfig
from .moe_model import MoEModel

__all__ = [
    'ExpertRouter',
    'RoutingResult',
    'ExpertConfig',
    'MoEModel',
]

# 版本信息
__version__ = '1.0.0'