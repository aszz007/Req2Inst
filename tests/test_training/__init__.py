"""
训练模块测试
测试数据加载器、训练器等训练相关功能

测试模式：
- 快速模式（默认）：使用mock数据，速度快
- 真实模式：使用项目dataset中的真实数据，较慢但更准确

运行方式：
    # 快速模式
    pytest tests/test_training/ -v

    # 真实模式
    pytest tests/test_training/ -v --real-data

    # 只运行快速测试
    pytest tests/test_training/ -v -m "not real_data"
"""

__all__ = [
    'test_data_loader',
    'test_expert_trainer',  # 待实现
]

__version__ = '0.2.0'
