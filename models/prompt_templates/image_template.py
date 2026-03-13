"""
图像标注指令生成Prompt模板
功能：将图像描述JSON转换为图像标注众包指令
输入：图像描述JSON（包含description和details字段）
输出：三段式图像标注指令（Definition / Emphasis & Caution / Things to Avoid）
"""

import json
from typing import Union


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
            >>> # 方式1: 传入完整JSON字符串
            >>> json_str = '{"description": "A street scene", "details": {...}}'
            >>> prompt = ImageInstructionTemplate.build_prompt(json_str)

            >>> # 方式2: 传入dict对象
            >>> data = {"description": "A street scene", "details": {...}}
            >>> prompt = ImageInstructionTemplate.build_prompt(data)

            >>> # 方式3: 传入纯文本description
            >>> desc = "A street scene with cars and pedestrians"
            >>> prompt = ImageInstructionTemplate.build_prompt(desc)
        """
        # 处理输入格式
        if isinstance(image_description, dict):
            # 过滤元数据字段（confidence、recognition_status、processing_time等）
            filtered_data = {
                k: v for k, v in image_description.items()
                if k not in ['confidence', 'recognition_status', 'processing_time']
            }
            # 转为压缩JSON字符串（无空格、无换行）
            json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
        elif isinstance(image_description, str):
            # 尝试解析为JSON
            try:
                parsed = json.loads(image_description)
                # 过滤元数据字段
                filtered_data = {
                    k: v for k, v in parsed.items()
                    if k not in ['confidence', 'recognition_status', 'processing_time']
                }
                # 转为压缩JSON字符串（无空格、无换行）
                json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
            except json.JSONDecodeError:
                # 如果不是JSON，当作纯文本description处理
                # 构建一个简单的JSON结构（压缩格式）
                json_str = json.dumps({
                    "description": image_description,
                    "details": {
                        "objects": [],
                        "scene": "unknown",
                        "spatial_info": ""
                    }
                }, ensure_ascii=False, separators=(',', ':'))
        else:
            raise TypeError("image_description必须是str或dict类型")

        # 构建用户消息
        user_message = f"""Image analysis structured data (JSON format):
```json
{json_str}
```

{ImageInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        # 构建完整的Qwen格式prompt（assistant部分使用空think块禁用Qwen3思考模式）
        prompt = f"""<|im_start|>system
{ImageInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

        return prompt

    @staticmethod
    def build_batch_prompt(image_descriptions: list) -> list:
        """
        批量构建prompt

        Args:
            image_descriptions: 图像描述列表（支持str或dict）

        Returns:
            list: prompt列表
        """
        return [
            ImageInstructionTemplate.build_prompt(desc)
            for desc in image_descriptions
        ]

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

        修复：检查结构而非仅关键词存在性

        Args:
            instruction: 生成的指令文本

        Returns:
            dict: 验证结果
        """
        result = {
            'is_valid': True,
            'has_definition': False,
            'has_bounding_boxes': False,
            'has_emphasis': False,
            'has_avoid': False,
            'errors': []
        }

        # 按行分割
        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        # 至少要有3行
        if len(lines) < 3:
            result['errors'].append(f'指令行数不足，期望至少3行，实际{len(lines)}行')
            result['is_valid'] = False
            return result

        # 检查每一行的格式
        for line in lines:
            line_lower = line.lower()

            # 检查Definition行
            if line.startswith('Definition:'):
                content = line[len('Definition:'):].strip()
                if content:
                    result['has_definition'] = True
                    # 检查是否包含bounding box要求（图像标注的核心任务）
                    if 'bounding box' in line_lower or 'draw box' in line_lower:
                        result['has_bounding_boxes'] = True
                else:
                    result['errors'].append('Definition部分内容为空')

            # 检查Emphasis & Caution行
            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                result['has_emphasis'] = True

            # 检查Things to Avoid行
            elif line.startswith('Things to Avoid:'):
                result['has_avoid'] = True

        # 检查缺失的部分
        if not result['has_definition']:
            result['errors'].append('缺少"Definition:"部分或格式错误')

        if not result['has_bounding_boxes']:
            result['errors'].append('Definition未明确要求画边框（draw bounding boxes）')

        if not result['has_emphasis']:
            result['errors'].append('缺少"Emphasis & Caution:"部分或格式错误')

        if not result['has_avoid']:
            result['errors'].append('缺少"Things to Avoid:"部分或格式错误')

        # 综合判断
        result['is_valid'] = all([
            result['has_definition'],
            result['has_bounding_boxes'],
            result['has_emphasis'],
            result['has_avoid']
        ])

        return result