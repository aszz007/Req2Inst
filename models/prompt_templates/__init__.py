"""
Prompt模板模块
功能：为不同类型的专家提供标准化的prompt构建接口
"""

from .text_template import TextInstructionTemplate
from .image_template import ImageInstructionTemplate
from .uml_template import UMLInstructionTemplate
from .general_template import GeneralInstructionTemplate

__all__ = [
    'TextInstructionTemplate',
    'ImageInstructionTemplate',
    'UMLInstructionTemplate',
    'GeneralInstructionTemplate',
]