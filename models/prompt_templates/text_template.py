"""
文本指令生成Prompt模板
功能：将文本需求转换为众包任务指令
输入：Low_Requirements（文本需求描述）
输出：三段式众包指令（Definition / Emphasis & Caution / Things to Avoid）
"""

from ._base import build_qwen_prompt, validate_three_part_format, build_batch_prompts


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
        """
        user_message = f"""Requirement text:
{low_requirement}

{TextInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        return build_qwen_prompt(TextInstructionTemplate.SYSTEM_PROMPT, user_message)

    @staticmethod
    def build_batch_prompt(low_requirements: list) -> list:
        """
        批量构建prompt

        Args:
            low_requirements: 低级需求列表

        Returns:
            list: prompt列表
        """
        return build_batch_prompts(low_requirements, TextInstructionTemplate.build_prompt)

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