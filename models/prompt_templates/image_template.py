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
    SYSTEM_PROMPT = """你是一个计算机视觉数据专家与众包任务设计者。请根据以下输入的图像分析结构化数据，编写一个适合众包工人使用的英文图像标注任务指令。

核心原则：
1. 标注导向：指令必须明确要求工人进行 "Draw bounding boxes" (画边框)。
2. 前景提取：从 objects 中提取主要的前景实体（如人、车）作为目标，忽略背景元素。
3. 直接引用：直接使用 JSON 中的英文术语，不要进行同义词替换。
4. 极致精简：Emphasis 和 Avoid 部分必须言简意赅。如果 JSON 中缺乏显著的视觉特征或干扰项，直接填 "-"。"""

    # 格式要求说明 - 修复：添加明确的输出格式示例
    FORMAT_INSTRUCTIONS = """输出格式要求（严格按照此格式）：

Definition: In this task, draw bounding boxes around [主要标注对象]
Emphasis & Caution: [关键视觉特征或标注重点，无则填"-"]
Things to Avoid: [易混淆的背景元素，无则填"-"]

格式规范：
1. 每个部分必须独立成行
2. 每行必须以对应标签开头（"Definition:", "Emphasis & Caution:", "Things to Avoid:"）
3. Definition部分必须以"In this task, draw bounding boxes"开头
4. Definition部分必须明确列出要标注的对象
5. 各部分之间不需要空行

示例输出：
Definition: In this task, draw bounding boxes around all cars and traffic signs in the street scene.
Emphasis & Caution: Focus on red traffic signs and vehicles in the foreground.
Things to Avoid: Do not annotate buildings or background pedestrians."""

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
            # 如果是字典，转为JSON字符串
            json_str = json.dumps(image_description, ensure_ascii=False, indent=2)
        elif isinstance(image_description, str):
            # 尝试解析为JSON
            try:
                parsed = json.loads(image_description)
                # 如果能解析，格式化输出
                json_str = json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # 如果不是JSON，当作纯文本description处理
                # 构建一个简单的JSON结构
                json_str = json.dumps({
                    "description": image_description,
                    "details": {
                        "objects": [],
                        "scene": "unknown",
                        "spatial_info": ""
                    }
                }, ensure_ascii=False, indent=2)
        else:
            raise TypeError("image_description必须是str或dict类型")

        # 构建用户消息
        user_message = f"""图像分析结构化数据（JSON格式）：
```json
{json_str}
```

{ImageInstructionTemplate.FORMAT_INSTRUCTIONS}

请开始生成图像标注指令："""

        # 构建完整的Qwen格式prompt
        prompt = f"""<|im_start|>system
{ImageInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
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

            # 检查Definition行（必须有标签前缀和"In this task"以及"bounding box"）
            if line.startswith('Definition:'):
                if 'in this task' in line_lower:
                    result['has_definition'] = True
                    # 检查是否包含bounding box要求
                    if 'bounding box' in line_lower or 'draw box' in line_lower:
                        result['has_bounding_boxes'] = True
                else:
                    result['errors'].append('Definition部分未以"In this task"开头')

            # 检查Emphasis & Caution行（必须有标签前缀）
            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                result['has_emphasis'] = True

            # 检查Things to Avoid行（必须有标签前缀）
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


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("图像Prompt模板测试")
    print("=" * 60)

    # 测试1: 完整JSON对象
    print("\n【测试1】完整JSON对象输入")
    print("-" * 60)
    image_data = {
        "description": "A busy urban street with multiple cars and traffic signs",
        "details": {
            "objects": ["car", "traffic sign", "building"],
            "scene": "urban street",
            "spatial_info": "Cars are in the foreground, buildings in the background"
        },
        "confidence": 0.95,
        "recognition_status": "success"
    }
    prompt = ImageInstructionTemplate.build_prompt(image_data)
    print("生成的Prompt（前400字符）：")
    print(prompt[:400])
    print("...")

    # 测试2: JSON字符串
    print("\n【测试2】JSON字符串输入")
    print("-" * 60)
    json_str = json.dumps(image_data)
    prompt = ImageInstructionTemplate.build_prompt(json_str)
    print(f"输入类型: JSON字符串")
    print(f"Prompt长度: {len(prompt)} 字符")

    # 测试3: 纯文本description
    print("\n【测试3】纯文本description输入")
    print("-" * 60)
    description_only = "A colorful bento box meal featuring rice, vegetables, and meat"
    prompt = ImageInstructionTemplate.build_prompt(description_only)
    print("生成的Prompt（前400字符）：")
    print(prompt[:400])
    print("...")

    # 测试4: 提取description字段
    print("\n【测试4】提取description字段")
    print("-" * 60)
    extracted = ImageInstructionTemplate.extract_description_from_json(image_data)
    print(f"提取的description: {extracted}")

    # 测试5: 批量生成
    print("\n【测试5】批量生成")
    print("-" * 60)
    descriptions = [
        {"description": "Street scene with cars", "details": {"objects": ["car"]}},
        {"description": "Indoor office space", "details": {"objects": ["desk", "chair"]}},
        "A park with people and trees"
    ]
    prompts = ImageInstructionTemplate.build_batch_prompt(descriptions)
    print(f"成功生成 {len(prompts)} 个prompts")

    # 测试6: 指令验证 - 正确格式
    print("\n【测试6】指令格式验证 - 正确格式")
    print("-" * 60)

    valid_instruction = """Definition: In this task, draw bounding boxes around all cars and traffic signs.
Emphasis & Caution: Focus on red traffic signs in the foreground.
Things to Avoid: Do not annotate buildings or background objects."""

    result = ImageInstructionTemplate.validate_instruction(valid_instruction)
    print(f"正确指令验证: {result['is_valid']}")
    print(f"包含bounding boxes: {result['has_bounding_boxes']}")
    if not result['is_valid']:
        print(f"错误信息: {result['errors']}")

    # 测试7: 指令验证 - 错误格式（单行）
    print("\n【测试7】指令格式验证 - 错误格式（单行）")
    print("-" * 60)

    invalid_instruction1 = """In this task, draw bounding boxes around cars (Definition) - (Emphasis & Caution) - (Things to Avoid)"""

    result = ImageInstructionTemplate.validate_instruction(invalid_instruction1)
    print(f"错误指令验证: {result['is_valid']}")
    print(f"错误信息: {result['errors']}")

    # 测试8: 指令验证 - 错误格式（缺少bounding boxes）
    print("\n【测试8】指令格式验证 - 错误格式（缺少bounding boxes）")
    print("-" * 60)

    invalid_instruction2 = """Definition: In this task, identify all objects in the image.
Emphasis & Caution: Focus on accuracy.
Things to Avoid: Do not skip any objects."""

    result = ImageInstructionTemplate.validate_instruction(invalid_instruction2)
    print(f"错误指令验证: {result['is_valid']}")
    print(f"错误信息: {result['errors']}")

    print("\n测试完成！")