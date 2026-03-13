"""
核心源代码模块
包含数据处理、路由、专家、训练等核心功能
"""

# 子模块懒加载（避免循环导入和启动时间过长）
__all__ = [
    'preprocessing',
    'routing',
    'experts',
    'instruction_generation',
    'training',
    'utils',
]