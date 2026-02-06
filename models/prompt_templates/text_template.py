"""
文本指令生成Prompt模板
功能：将文本需求转换为众包任务指令
输入：Low_Requirements（文本需求描述）
输出：三段式众包指令（Definition / Emphasis & Caution / Things to Avoid）
"""


class TextInstructionTemplate:
    """文本需求 → 众包指令 的Prompt模板"""

    # 系统提示词（定义角色和核心原则）
    SYSTEM_PROMPT = """You are a crowdsourcing task design expert. Based on the input requirement text, write an English task instruction for crowdsourcing workers.

Core Principles:
1. Extreme Conciseness: Crowdsourcing workers value time. Use the most concise language possible.
2. Structured Format: Strictly follow the three-part format defined below.
3. English Output: Output must be in English regardless of input language."""

    # 格式要求说明
    FORMAT_INSTRUCTIONS = """Output Format Requirements:

Definition: Use a clear imperative sentence to describe the main objective. Must start with "In this task,".
Emphasis & Caution: Only highlight conditions most prone to error or that must be met. Use "-" if nothing specific to emphasize.
Things to Avoid: Only list prohibited operations. Use "-" if nothing specific to avoid.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Keep all sections concise
- Output ONLY these three lines, nothing else"""

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
        user_message = f"""Requirement text:
{low_requirement}

{TextInstructionTemplate.FORMAT_INSTRUCTIONS}"""

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