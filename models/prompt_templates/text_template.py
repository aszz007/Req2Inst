"""
文本指令生成Prompt模板
功能：将文本需求转换为众包任务指令
输入：Low_Requirements（文本需求描述）
输出：三段式众包指令（Definition / Emphasis & Caution / Things to Avoid）
"""


class TextInstructionTemplate:
    """文本需求 → 众包指令 的Prompt模板"""

    # 系统提示词（定义角色和核心原则）
    SYSTEM_PROMPT = """你是一个众包任务设计专家。请根据以下输入的需求文本，编写一个适合众包工人使用的英文任务指令。

核心原则：
1. 极致精简：众包工人时间宝贵，请使用最简练的语言。
2. 结构规范：严格按照下方定义的格式输出。
3. 英语输出：无论输入是何种语言，输出必须是英文。"""

    # 格式要求说明 - 修复：添加明确的输出格式示例
    FORMAT_INSTRUCTIONS = """输出格式要求（严格按照此格式）：

Definition: In this task, [主要任务目标的祈使句描述]
Emphasis & Caution: [关键注意事项或必须满足的条件，无则填"-"]
Things to Avoid: [明确禁止的操作，无则填"-"]

格式规范：
1. 每个部分必须独立成行
2. 每行必须以对应标签开头（"Definition:", "Emphasis & Caution:", "Things to Avoid:"）
3. Definition部分的内容必须以"In this task,"开头
4. 各部分之间不需要空行

示例输出：
Definition: In this task, verify the user login functionality with valid credentials.
Emphasis & Caution: Ensure both username and password validation are tested.
Things to Avoid: Do not skip error message verification."""

    @staticmethod
    def build_prompt(low_requirement: str) -> str:
        """
        构建文本需求生成指令的完整prompt

        Args:
            low_requirement: 低级需求描述文本

        Returns:
            str: 完整的prompt（Qwen对话格式）

        Example:
            >>> requirement = "测试系统的登录功能"
            >>> prompt = TextInstructionTemplate.build_prompt(requirement)
            >>> # 传递给InstructionGenerator
        """
        # 构建用户消息
        user_message = f"""需求文本：
{low_requirement}

{TextInstructionTemplate.FORMAT_INSTRUCTIONS}

请开始生成指令："""

        # 构建完整的Qwen格式prompt
        prompt = f"""<|im_start|>system
{TextInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

        return prompt

    @staticmethod
    def build_batch_prompt(low_requirements: list) -> list:
        """
        批量构建prompt

        Args:
            low_requirements: 低级需求列表

        Returns:
            list: prompt列表
        """
        return [
            TextInstructionTemplate.build_prompt(req)
            for req in low_requirements
        ]

    @staticmethod
    def validate_instruction(instruction: str) -> dict:
        """
        验证生成的指令是否符合三段式格式

        修复：检查结构而非仅关键词存在性

        Args:
            instruction: 生成的指令文本

        Returns:
            dict: 验证结果
                {
                    'is_valid': bool,
                    'has_definition': bool,
                    'has_emphasis': bool,
                    'has_avoid': bool,
                    'errors': list
                }
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
            result['errors'].append(f'指令行数不足，期望至少3行，实际{len(lines)}行')
            result['is_valid'] = False
            return result

        # 检查每一行的格式
        for i, line in enumerate(lines):
            line_lower = line.lower()

            # 检查Definition行
            if line.startswith('Definition:'):
                if 'in this task' in line_lower:
                    result['has_definition'] = True
                else:
                    result['errors'].append('Definition部分未以"In this task"开头')

            # 检查Emphasis & Caution行
            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                result['has_emphasis'] = True

            # 检查Things to Avoid行
            elif line.startswith('Things to Avoid:'):
                result['has_avoid'] = True

        # 检查是否所有部分都存在
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
    print("文本Prompt模板测试")
    print("=" * 60)

    # 测试1: 中文需求
    print("\n【测试1】中文需求 → 英文指令")
    print("-" * 60)
    requirement_cn = "测试系统的登录功能，确保用户名和密码验证正确"
    prompt = TextInstructionTemplate.build_prompt(requirement_cn)
    print("生成的Prompt（前300字符）：")
    print(prompt[:300])
    print("...")

    # 测试2: 英文需求
    print("\n【测试2】英文需求")
    print("-" * 60)
    requirement_en = "Test the user registration process and verify email validation"
    prompt = TextInstructionTemplate.build_prompt(requirement_en)
    print("生成的Prompt（前300字符）：")
    print(prompt[:300])
    print("...")

    # 测试3: 批量生成
    print("\n【测试3】批量生成")
    print("-" * 60)
    requirements = [
        "验证购物车添加功能",
        "测试支付流程",
        "检查订单查询接口"
    ]
    prompts = TextInstructionTemplate.build_batch_prompt(requirements)
    print(f"成功生成 {len(prompts)} 个prompts")

    # 测试4: 指令验证 - 正确格式
    print("\n【测试4】指令格式验证 - 正确格式")
    print("-" * 60)

    valid_instruction = """Definition: In this task, test the login functionality.
Emphasis & Caution: Ensure correct username and password validation.
Things to Avoid: Do not skip error handling tests."""

    result = TextInstructionTemplate.validate_instruction(valid_instruction)
    print(f"正确指令验证: {result['is_valid']}")
    if not result['is_valid']:
        print(f"错误信息: {result['errors']}")

    # 测试5: 指令验证 - 错误格式（单行）
    print("\n【测试5】指令格式验证 - 错误格式（单行）")
    print("-" * 60)

    invalid_instruction = """In this task, verify the login function using correct username and password. (Definition) - (Emphasis & Caution) - (Things to Avoid)"""

    result = TextInstructionTemplate.validate_instruction(invalid_instruction)
    print(f"错误指令验证: {result['is_valid']}")
    print(f"错误信息: {result['errors']}")

    # 测试6: 指令验证 - 错误格式（缺少标签）
    print("\n【测试6】指令格式验证 - 错误格式（缺少标签）")
    print("-" * 60)

    invalid_instruction2 = """Test the login functionality.
Make sure it works properly.
Avoid skipping tests."""

    result = TextInstructionTemplate.validate_instruction(invalid_instruction2)
    print(f"错误指令验证: {result['is_valid']}")
    print(f"错误信息: {result['errors']}")

    print("\n测试完成！")