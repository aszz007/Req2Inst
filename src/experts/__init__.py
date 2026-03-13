"""
专家模块 - 众包指令生成专家系统

包含四类专家:
  - TextExpert: 文本需求 -> 众包指令
  - ImageExpert: 图像描述 -> 图像标注指令
  - UMLExpert: UML用例图 -> 业务逻辑指令
  - GeneralExpert: 混合多模态 -> 通用指令(兜底)
"""

from src.experts.base_expert import BaseExpert
from src.experts.text_expert import TextExpert
from src.experts.image_expert import ImageExpert
from src.experts.uml_expert import UMLExpert
from src.experts.general_expert import GeneralExpert

__all__ = [
    'BaseExpert',
    'TextExpert',
    'ImageExpert',
    'UMLExpert',
    'GeneralExpert',
]

__version__ = '1.0.0'