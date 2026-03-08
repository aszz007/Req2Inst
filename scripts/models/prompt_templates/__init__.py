"""
Prompt模板模块
功能：为不同类型的专家提供标准化的prompt构建接口
"""

from .text_template import TextInstructionTemplate
from .image_template import ImageInstructionTemplate
from .uml_template import UMLInstructionTemplate

__all__ = [
    'TextInstructionTemplate',
    'ImageInstructionTemplate',
    'UMLInstructionTemplate',
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'Crowdsourcing Instruction Generator Team'

# 使用示例（注释）
"""
使用示例：

1. 文本需求生成指令：
    from models.prompt_templates import TextInstructionTemplate
    
    requirement = "测试系统的登录功能"
    prompt = TextInstructionTemplate.build_prompt(requirement)
    
    # 传递给语言模型
    from models import InstructionGenerator
    generator = InstructionGenerator()
    generator.load_expert('text_expert')
    instruction = generator.generate(prompt)

2. 图像描述生成标注指令：
    from models.prompt_templates import ImageInstructionTemplate
    import json
    
    # 从图像识别结果提取description字段
    image_json = {
        "description": "A busy urban street with cars and traffic signs",
        "details": {...}
    }
    
    # 提取description字符串或完整JSON
    description = image_json['description']  # 仅描述文本
    # 或
    description = json.dumps(image_json)  # 完整JSON
    
    prompt = ImageInstructionTemplate.build_prompt(description)
    
    # 传递给语言模型
    generator.load_expert('image_expert_qwen25')
    instruction = generator.generate(prompt)

3. UML JSON生成业务逻辑指令：
    from models.prompt_templates import UMLInstructionTemplate
    import json
    
    uml_data = {
        "actors": [...],
        "use_cases": [...],
        "relationships": [...]
    }
    
    uml_json_str = json.dumps(uml_data, ensure_ascii=False)
    prompt = UMLInstructionTemplate.build_prompt(uml_json_str)
    
    # 传递给语言模型
    generator.load_expert('uml_expert_qwen2.5')
    instruction = generator.generate(prompt)

4. 通用工作流：
    from models import InstructionGenerator
    from models.prompt_templates import (
        TextInstructionTemplate,
        ImageInstructionTemplate,
        UMLInstructionTemplate
    )
    
    generator = InstructionGenerator(use_4bit=True)
    
    # 根据输入类型选择模板
    if input_type == 'text':
        generator.load_expert('text_expert')
        prompt = TextInstructionTemplate.build_prompt(input_data)
    elif input_type == 'image':
        generator.load_expert('image_expert_qwen25')
        prompt = ImageInstructionTemplate.build_prompt(input_data)
    elif input_type == 'uml':
        generator.load_expert('uml_expert_qwen2.5')
        prompt = UMLInstructionTemplate.build_prompt(input_data)
    
    instruction = generator.generate(prompt)
    generator.unload_expert()
"""