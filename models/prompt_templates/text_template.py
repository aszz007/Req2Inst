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

    # 格式要求说明
    FORMAT_INSTRUCTIONS = """格式要求：
- Definition：使用简明扼要的祈使句描述主要目标。必须以 "In this task," 开头。
- Emphasis & Caution：仅指出极易出错或必须满足的特定条件。如无特别强调，填入 "-"。
- Things to Avoid：仅列出禁止的操作。如无特别避免事项，填入 "-"。"""

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

        # 转小写方便检测
        instruction_lower = instruction.lower()

        # 检查Definition（必须以"In this task"开头）
        if 'in this task' in instruction_lower:
            result['has_definition'] = True
        else:
            result['errors'].append('缺少Definition部分或未以"In this task"开头')

        # 检查Emphasis & Caution
        if 'emphasis' in instruction_lower or 'caution' in instruction_lower:
            result['has_emphasis'] = True
        elif '-' in instruction and 'emphasis' not in instruction_lower:
            # 可能用"-"表示无强调
            result['has_emphasis'] = True
        else:
            result['errors'].append('缺少Emphasis & Caution部分')

        # 检查Things to Avoid
        if 'avoid' in instruction_lower or 'things to avoid' in instruction_lower:
            result['has_avoid'] = True
        elif '-' in instruction and 'avoid' not in instruction_lower:
            # 可能用"-"表示无避免事项
            result['has_avoid'] = True
        else:
            result['errors'].append('缺少Things to Avoid部分')

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

    # 测试4: 指令验证
    print("\n【测试4】指令格式验证")
    print("-" * 60)

    # 正确的指令
    valid_instruction = """Definition: In this task, test the login functionality.
Emphasis & Caution: Ensure correct username and password validation.
Things to Avoid: Do not skip error handling tests."""

    result = TextInstructionTemplate.validate_instruction(valid_instruction)
    print(f"正确指令验证: {result['is_valid']}")

    # 错误的指令
    invalid_instruction = """Test the login functionality.
Make sure it works properly."""

    result = TextInstructionTemplate.validate_instruction(invalid_instruction)
    print(f"错误指令验证: {result['is_valid']}")
    print(f"错误信息: {result['errors']}")

    print("\n测试完成！")