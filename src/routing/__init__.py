"""
路由模块
负责MoE专家路由和选择

包含：
  - ExpertRouter: 基于规则的Hard Routing（按输入类型选择匹配专家）
  - MoEModel: MoE模型主类（整合路由器和专家的统一生成接口）
  - SoftRouter: 基于PEFT加权融合的Soft Routing（exp9新增）
"""

from .expert_router import ExpertRouter, RoutingResult, ExpertConfig
from .moe_model import MoEModel
from .soft_router import SoftRouter, SoftRoutingConfig, build_type_aware_weights

__all__ = [
    'ExpertRouter',
    'RoutingResult',
    'ExpertConfig',
    'MoEModel',
    'SoftRouter',
    'SoftRoutingConfig',
    'build_type_aware_weights',
]

# 版本信息
__version__ = '1.1.0'