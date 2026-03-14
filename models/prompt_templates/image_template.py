"""
图像标注指令生成Prompt模板
功能：将图像描述JSON转换为图像标注众包指令
输入：图像描述JSON（包含description和details字段）
输出：三段式图像标注指令（Definition / Emphasis & Caution / Things to Avoid）
"""

import json
from typing import Union

from ._base import (
    build_qwen_prompt, validate_three_part_format,
    build_batch_prompts, process_json_input, compress_json,
)


class ImageInstructionTemplate:
    """图像描述 → 图像标注指令 的Prompt模板"""

    # ==================== 识别阶段Prompt（预处理） ====================
    IMAGE_RECOGNITION_PROMPT = """Please describe this image in detail and output in JSON format with the following fields:
1. description: Overall description of the image (summarize in one sentence)
2. details: Contains the following sub-fields
   - objects: List of main objects in the image
   - scene: Scene type (e.g., "urban street", "indoor scene", etc.)
   - spatial_info: Spatial position information of objects

Please output strictly in JSON format with no other content. Use ONLY English in all fields."""

    @staticmethod
    def get_recognition_prompt() -> str:
        """
        获取图像识别Prompt（用于预处理阶段）

        Returns:
            str: 图像识别的Prompt文本
        """
        return ImageInstructionTemplate.IMAGE_RECOGNITION_PROMPT

    # ==================== 指令生成阶段Prompt ====================
    # 系统提示词（定义角色和核心原则）
    SYSTEM_PROMPT = """You are a computer vision data expert and crowdsourcing task designer. Based on the input image analysis structured data, write an English image annotation instruction for crowdsourcing workers.

Core Principles:
1. Annotation Focus: The instruction must explicitly require workers to draw bounding boxes.
2. Foreground Extraction: Extract main foreground objects (e.g., people, vehicles) from the objects list as annotation targets. Ignore background elements.
3. Direct Reference: Use English terms directly from the JSON data. Do not replace with synonyms.
4. Extreme Conciseness: Keep Emphasis and Avoid sections brief. Use "-" if no significant visual features or distractors exist."""

    # 格式要求说明
    FORMAT_INSTRUCTIONS = """Output Format Requirements:

Definition: Use a clear imperative sentence to describe the annotation targets. Must start with "In this task," and explicitly mention "draw bounding boxes around".
Emphasis & Caution: Only list highly distinctive visual features (e.g., specific colors, positions). Use "-" if nothing specific to emphasize.
Things to Avoid: Only list confusing background distractors. Use "-" if nothing specific to avoid.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Definition must include "draw bounding boxes around" and list specific objects from JSON data
- Keep all sections concise
- Output ONLY these three lines, nothing else"""

    @staticmethod
    def build_prompt(image_description: Union[str, dict]) -> str:
        """
        构建图像描述生成标注指令的完整prompt

        Args:
            image_description: 图像描述，支持两种格式：
                1. JSON字符串：完整的图像识别结果
                2. dict对象：图像识别结果字典
                3. 纯文本：仅description字段

        Returns:
            str: 完整的prompt（Qwen对话格式）

        Example:
            >>> json_str = '{"description": "A street scene", "details": {...}}'
            >>> prompt = ImageInstructionTemplate.build_prompt(json_str)

            >>> data = {"description": "A street scene", "details": {...}}
            >>> prompt = ImageInstructionTemplate.build_prompt(data)
        """
        # 处理输入格式（过滤元数据，压缩JSON）
        json_str = process_json_input(image_description, filter_meta=True, filter_positions=False)

        # 如果process_json_input返回的是非JSON纯文本，包装为标准结构
        if not json_str.startswith('{'):
            json_str = compress_json({
                "description": json_str,
                "details": {
                    "objects": [],
                    "scene": "unknown",
                    "spatial_info": ""
                }
            })

        # 构建用户消息
        user_message = f"""Image analysis structured data (JSON format):
```json
{json_str}
```

{ImageInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        return build_qwen_prompt(ImageInstructionTemplate.SYSTEM_PROMPT, user_message)

    @staticmethod
    def build_batch_prompt(image_descriptions: list) -> list:
        """
        批量构建prompt

        Args:
            image_descriptions: 图像描述列表（支持str或dict）

        Returns:
            list: prompt列表
        """
        return build_batch_prompts(image_descriptions, ImageInstructionTemplate.build_prompt)

    @staticmethod
    def extract_description_from_json(json_data: Union[str, dict]) -> str:
        """
        从完整的图像识别JSON中提取description字段

        Args:
            json_data: 完整的JSON数据（包含description, confidence等）

        Returns:
            str: description字段内容
        """
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError:
                return json_data
        else:
            data = json_data

        return data.get('description', str(data))

    @staticmethod
    def validate_instruction(instruction: str) -> dict:
        """
        验证生成的指令是否符合图像标注三段式格式

        Args:
            instruction: 生成的指令文本

        Returns:
            dict: 验证结果
        """
        extra_checks = [
            {
                'key': 'has_bounding_boxes',
                'check_fn': lambda line, ll: 'bounding box' in ll or 'draw box' in ll,
                'section': 'definition',
                'error_msg': 'Definition未明确要求画边框（draw bounding boxes）',
                'required': True,
            }
        ]
        return validate_three_part_format(instruction, extra_checks=extra_checks)