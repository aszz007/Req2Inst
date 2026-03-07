"""
路由模块
负责MoE专家路由和选择

包含：
  - ExpertRouter: 基于规则的Hard Routing（按输入类型选择匹配专家）
  - MoEModel: MoE模型主类（整合路由器和专家的统一生成接口）
  - SoftRouter: 基于PEFT加权融合的Soft Routing（exp9新增）
  - RouterMLP: 基于MLP分类器的Learned Routing（exp10新增）
  - HiddenStateExtractor: Qwen3-8B hidden state特征提取器（exp10新增）
  - LearnedRouterInference: 学习路由完整推理封装（exp10新增）
"""

from .expert_router import ExpertRouter, RoutingResult, ExpertConfig
from .moe_model import MoEModel
from .soft_router import SoftRouter, SoftRoutingConfig, build_type_aware_weights
from .learned_router import (
    RouterMLP,
    HiddenStateExtractor,
    LearnedRouterInference,
    load_router_from_checkpoint,
    EXPERT_TO_IDX,
    IDX_TO_EXPERT,
    ALL_EXPERTS,
)

__all__ = [
    # Hard Routing
    'ExpertRouter',
    'RoutingResult',
    'ExpertConfig',
    # MoE主类
    'MoEModel',
    # Soft Routing (exp9)
    'SoftRouter',
    'SoftRoutingConfig',
    'build_type_aware_weights',
    # Learned Routing (exp10)
    'RouterMLP',
    'HiddenStateExtractor',
    'LearnedRouterInference',
    'load_router_from_checkpoint',
    # 常量
    'EXPERT_TO_IDX',
    'IDX_TO_EXPERT',
    'ALL_EXPERTS',
]

# 版本信息
__version__ = '1.2.0'