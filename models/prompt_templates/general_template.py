"""
通用专家指令生成Prompt模板
功能:将多种类型的输入(文本需求/图像描述/UML描述)转换为众包指令
输入:自动检测输入类型,可以是文本、图像JSON或UML JSON
输出:三段式众包指令(Definition / Emphasis & Caution / Things to Avoid)

通用专家作为兜底专家,需要处理所有类型的输入
"""

import json
from typing import Union


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
            # 不是JSON,当作纯文本
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
            >>> # 自动检测
            >>> prompt = GeneralInstructionTemplate.build_prompt("测试登录功能")
            >>> # 强制指定类型
            >>> prompt = GeneralInstructionTemplate.build_prompt(image_json, force_type='image')
        """
        # 确定输入类型
        if force_type:
            input_type = force_type
        else:
            input_type = GeneralInstructionTemplate.detect_input_type(input_data)

        # 处理输入数据格式
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data, ensure_ascii=False, indent=2)
        elif isinstance(input_data, str):
            try:
                # 尝试解析并格式化JSON
                parsed = json.loads(input_data)
                input_str = json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # 纯文本
                input_str = input_data
        else:
            input_str = str(input_data)

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

        # 构建完整的Qwen格式prompt
        prompt = f"""<|im_start|>system
{GeneralInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

        return prompt

    @staticmethod
    def build_batch_prompt(input_data_list: list) -> list:
        """
        批量构建prompt

        Args:
            input_data_list: 输入数据列表

        Returns:
            list: prompt列表
        """
        return [
            GeneralInstructionTemplate.build_prompt(data)
            for data in input_data_list
        ]

    @staticmethod
    def validate_instruction(instruction: str) -> dict:
        """
        验证生成的指令是否符合三段式格式

        Args:
            instruction: 生成的指令文本

        Returns:
            dict: 验证结果
        """
        result = {
            'is_valid': True,
            'has_definition': False,
            'has_emphasis': False,
            'has_avoid': False,
            'errors': []
        }

        # 按行分割
        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        # 至少要有3行
        if len(lines) < 3:
            result['errors'].append(f'指令行数不足,期望至少3行,实际{len(lines)}行')
            result['is_valid'] = False
            return result

        # 检查每一行的格式
        for line in lines:
            # 检查Definition行
            if line.startswith('Definition:'):
                content = line[len('Definition:'):].strip()
                if content:
                    result['has_definition'] = True
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

        if not result['has_emphasis']:
            result['errors'].append('缺少"Emphasis & Caution:"部分或格式错误')

        if not result['has_avoid']:
            result['errors'].append('缺少"Things to Avoid:"部分或格式错误')

        # 综合判断
        result['is_valid'] = all([
            result['has_definition'],
            result['has_emphasis'],
            result['has_avoid']
        ])

        return result


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("通用专家Prompt模板测试")
    print("=" * 60)

    # 测试1: 文本需求
    print("\n【测试1】文本需求自动检测")
    print("-" * 60)
    text_req = "测试系统的登录功能"
    detected_type = GeneralInstructionTemplate.detect_input_type(text_req)
    print(f"检测类型: {detected_type}")
    prompt = GeneralInstructionTemplate.build_prompt(text_req)
    print(f"Prompt长度: {len(prompt)} 字符")

    # 测试2: 图像JSON
    print("\n【测试2】图像JSON自动检测")
    print("-" * 60)
    image_json = {
        "description": "A street with cars",
        "details": {
            "objects": ["car", "traffic sign"],
            "scene": "urban street",
            "spatial_info": "cars in foreground"
        }
    }
    detected_type = GeneralInstructionTemplate.detect_input_type(image_json)
    print(f"检测类型: {detected_type}")
    prompt = GeneralInstructionTemplate.build_prompt(image_json)
    print(f"Prompt长度: {len(prompt)} 字符")

    # 测试3: UML JSON
    print("\n【测试3】UML JSON自动检测")
    print("-" * 60)
    uml_json = {
        "description": "User authentication class diagram",
        "details": {
            "diagram_type": "class diagram",
            "classes": ["User", "AuthService"]
        }
    }
    detected_type = GeneralInstructionTemplate.detect_input_type(uml_json)
    print(f"检测类型: {detected_type}")
    prompt = GeneralInstructionTemplate.build_prompt(uml_json)
    print(f"Prompt长度: {len(prompt)} 字符")

    # 测试4: 强制类型
    print("\n【测试4】强制指定类型")
    print("-" * 60)
    prompt1 = GeneralInstructionTemplate.build_prompt("任意文本", force_type='text')
    prompt2 = GeneralInstructionTemplate.build_prompt("任意文本", force_type='image')
    print(f"强制text类型: {len(prompt1)} 字符")
    print(f"强制image类型: {len(prompt2)} 字符")

    # 测试5: 批量生成
    print("\n【测试5】批量生成")
    print("-" * 60)
    inputs = [
        "测试功能",
        {"description": "image", "details": {"objects": [], "scene": "test"}},
        {"description": "uml", "details": {"diagram_type": "class"}}
    ]
    prompts = GeneralInstructionTemplate.build_batch_prompt(inputs)
    print(f"成功生成 {len(prompts)} 个prompts")

    # 测试6: 指令验证
    print("\n【测试6】指令格式验证")
    print("-" * 60)
    valid_instruction = """Definition: In this task, test the login functionality.
Emphasis & Caution: Ensure correct validation.
Things to Avoid: Do not skip error cases."""

    result = GeneralInstructionTemplate.validate_instruction(valid_instruction)
    print(f"验证结果: {result['is_valid']}")
    if not result['is_valid']:
        print(f"错误: {result['errors']}")

    print("\n测试完成!")