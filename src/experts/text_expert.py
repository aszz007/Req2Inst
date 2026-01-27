"""
文本指令生成专家
处理：文本需求 -> 测试用例指令
"""

from typing import Dict, Any
from .base_expert import BaseExpert


class TextExpert(BaseExpert):
    """文本需求 -> 测试用例指令的专家"""

    def __init__(self, lora_path: str = None, config: Dict = None):
        super().__init__(
            expert_name="text_expert",
            lora_path=lora_path,
            config=config
        )

    def get_prompt_template(self) -> str:
        """
        获取文本指令prompt模板
        对应你的数据集文档中的"文本-指令"模板
        """
        template = """你是一个众包任务设计专家。请根据以下输入的需求文本，编写一个适合众包工人使用的英文任务指令。

核心原则：
1. 极致精简：众包工人时间宝贵，请使用最简练的语言。
2. 结构规范：严格按照下方定义的格式输出。
3. 英语输出：无论输入是何种语言，输出必须是英文。

格式要求：
- Definition: 使用简明扼要的祈使句描述主要目标。必须以 "In this task," 开头。
- Emphasis & Caution: 仅指出极易出错或必须满足的特定条件。如无特别强调，填入 "-"。
- Things to Avoid: 仅列出禁止的操作。如无特别避免事项，填入 "-"。

Input Requirement: {input_text}

Output:"""

        return template

    def preprocess_input(self, input_data: Any) -> Dict:
        """
        预处理文本输入

        Args:
            input_data: 原始文本或字典

        Returns:
            dict: {'input_text': str, 'metadata': dict}
        """
        # TODO: 实现预处理逻辑
        # - 如果是字符串，直接使用
        # - 如果是字典，提取text字段
        # - 清理文本（去除多余空格、换行等）
        # - 提取关键词（可选）

        if isinstance(input_data, str):
            input_text = input_data
            metadata = {}
        elif isinstance(input_data, dict):
            input_text = input_data.get('text', '')
            metadata = {k: v for k, v in input_data.items() if k != 'text'}
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        return {
            'input_text': input_text.strip(),
            'metadata': metadata
        }

    def build_prompt(self, preprocessed_data: Dict) -> str:
        """
        构建完整prompt

        Args:
            preprocessed_data: 预处理后的数据

        Returns:
            str: 完整的Qwen格式prompt
        """
        # TODO: 实现prompt构建
        # 1. 获取模板
        # 2. 填充输入文本
        # 3. 添加Qwen对话格式包装

        template = self.get_prompt_template()
        input_text = preprocessed_data['input_text']

        # 填充模板
        user_content = template.format(input_text=input_text)

        # Qwen对话格式
        system_prompt = "你是一个专业的众包任务指令生成专家。"

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def postprocess_output(self, raw_output: str) -> Dict:
        """
        后处理输出

        Args:
            raw_output: 模型原始输出

        Returns:
            dict: 结构化输出
        """
        # TODO: 实现后处理逻辑
        # 1. 清理输出（去除特殊标记）
        # 2. 提取三个字段：Definition, Emphasis, Avoid
        # 3. 验证格式
        # 4. 返回结构化结果

        result = {
            'definition': '',
            'emphasis_caution': '',
            'things_to_avoid': '',
            'raw_output': raw_output
        }

        return result

    def validate_output(self, output: Dict) -> bool:
        """
        验证文本指令输出

        检查：
        - Definition必须以"In this task,"开头
        - 三个字段都不能为空
        - 格式符合要求
        """
        # TODO: 实现验证逻辑

        return True
