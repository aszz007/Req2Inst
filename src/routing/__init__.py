"""
Routing Module - MoE Expert Router
专家路由模块

提供MoE系统的智能路由功能
"""

from .expert_router import ExpertRouter, RoutingResult, ExpertConfig

__all__ = [
    'ExpertRouter',
    'RoutingResult',
    'ExpertConfig',
]

__version__ = '1.0.0'