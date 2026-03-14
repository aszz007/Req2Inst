"""
通用专家指令生成Prompt模板
功能:将多种类型的输入(文本需求/图像描述/UML描述)转换为众包指令
输入:自动检测输入类型,可以是文本、图像JSON或UML JSON
输出:三段式众包指令(Definition / Emphasis & Caution / Things to Avoid)

通用专家作为兜底专家,需要处理所有类型的输入
"""

import json
from typing import Union

from ._base import (
    build_qwen_prompt, validate_three_part_format,
    build_batch_prompts, process_json_input,
)


class GeneralInstructionTemplate:
    """通用专家Prompt模板 - 自动检测输入类型并调用对应子模板"""

    # 系统提示词(通用版本)
    SYSTEM_PROMPT = """You are an expert crowdsourcing task designer. Based on the input (which may be a text requirement, image description, or UML diagram description), write an English task instruction for crowdsourcing workers.

Core Principles:
1. Adapt to Input Type: Recognize whether the input is a text requirement, image annotation task, or UML diagram task, and generate appropriate instructions.
2. Extreme Conciseness: Keep all sections brief and focused.
3. Structured Format: Strictly follow the three-part format.
4. English Output: Always output in English."""

    # 格式要求说明(通用版本)
    FORMAT_INSTRUCTIONS = """Output Format Requirements:

Definition: Use a clear imperative sentence to describe the main task. Must start with "In this task,".
Emphasis & Caution: Highlight key requirements or common errors. Use "-" if nothing specific to emphasize.
Things to Avoid: List prohibited operations or confusing elements. Use "-" if nothing specific to avoid.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- For image tasks, explicitly mention "draw bounding boxes" in Definition
- For UML tasks, specify the diagram type (class/sequence/use case) in Definition
- Keep all sections concise
- Output ONLY these three lines, nothing else"""

    @staticmethod
    def detect_input_type(input_data: Union[str, dict]) -> str:
        """
        检测输入数据的类型

        Args:
            input_data: 输入数据(str或dict)

        Returns:
            str: 'text', 'image', 或 'uml'
        """
        # 如果是字典,先转为字符串再检测
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data)
        else:
            input_str = str(input_data)

        # 尝试解析JSON
        try:
            parsed = json.loads(input_str)
            if isinstance(parsed, dict):
                # 检查是否为图像描述JSON
                if 'description' in parsed and 'details' in parsed:
                    details = parsed.get('details', {})
                    # 图像特征:有objects和scene
                    if 'objects' in details and 'scene' in details:
                        return 'image'
                    # UML特征:有diagram_type
                    if 'diagram_type' in details:
                        return 'uml'

                # 检查是否为UML描述JSON
                if 'diagram_type' in parsed:
                    return 'uml'

                # 有description但不是image/uml,当作文本
                if 'description' in parsed:
                    return 'text'

        except (json.JSONDecodeError, TypeError):
            pass

        # 默认当作文本需求
        return 'text'

    @staticmethod
    def build_prompt(input_data: Union[str, dict], force_type: str = None) -> str:
        """
        构建通用专家的完整prompt

        自动检测输入类型并生成相应的prompt
        支持强制指定类型以提高可控性

        Args:
            input_data: 输入数据(文本/图像JSON/UML JSON)
            force_type: 强制指定类型('text'/'image'/'uml'),None则自动检测

        Returns:
            str: 完整的prompt(Qwen对话格式)

        Example:
            >>> prompt = GeneralInstructionTemplate.build_prompt("测试登录功能")
            >>> prompt = GeneralInstructionTemplate.build_prompt(image_json, force_type='image')
        """
        # 确定输入类型
        input_type = force_type or GeneralInstructionTemplate.detect_input_type(input_data)

        # 统一处理JSON输入：过滤元数据，UML类型额外过滤actor position
        is_uml = (input_type == 'uml')
        input_str = process_json_input(input_data, filter_meta=True, filter_positions=is_uml)

        # 根据类型构建不同的用户消息
        if input_type == 'image':
            user_message = f"""Image description (JSON format):
```json
{input_str}
```

Task: Generate an image annotation instruction for crowdsourcing workers to draw bounding boxes around objects.

{GeneralInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        elif input_type == 'uml':
            user_message = f"""UML diagram description (JSON format):
```json
{input_str}
```

Task: Generate a UML diagram analysis instruction for crowdsourcing workers.

{GeneralInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        else:  # text
            user_message = f"""Requirement text:
{input_str}

{GeneralInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        return build_qwen_prompt(GeneralInstructionTemplate.SYSTEM_PROMPT, user_message)

    @staticmethod
    def build_batch_prompt(input_data_list: list) -> list:
        """
        批量构建prompt

        Args:
            input_data_list: 输入数据列表

        Returns:
            list: prompt列表
        """
        return build_batch_prompts(input_data_list, GeneralInstructionTemplate.build_prompt)

    @staticmethod
    def validate_instruction(instruction: str) -> dict:
        """
        验证生成的指令是否符合三段式格式

        Args:
            instruction: 生成的指令文本

        Returns:
            dict: 验证结果
        """
        return validate_three_part_format(instruction)