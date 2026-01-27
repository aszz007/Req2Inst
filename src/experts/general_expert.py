"""
通用专家
处理未知或简单任务，作为兜底
"""

from typing import Dict, Any
from .base_expert import BaseExpert


class GeneralExpert(BaseExpert):
    """通用兜底专家"""

    def __init__(self, lora_path: str = None, config: Dict = None):
        super().__init__(
            expert_name="general_expert",
            lora_path=lora_path,
            config=config
        )

    def get_prompt_template(self) -> str:
        """
        通用prompt模板
        """
        template = """你是一个众包任务设计专家。请根据以下输入，编写一个适合众包工人使用的英文任务指令。

请严格按照以下格式输出：
- Definition: 使用简明扼要的祈使句描述任务目标。必须以 "In this task," 开头。
- Emphasis & Caution: 指出重要的注意事项。如无特别强调，填入 "-"。
- Things to Avoid: 列出禁止的操作。如无特别避免事项，填入 "-"。

Input: {input_content}

Output:"""

        return template

    def preprocess_input(self, input_data: Any) -> Dict:
        """
        通用预处理 - 尽可能转为字符串
        """
        # TODO: 实现通用预处理
        # 1. 尝试转为字符串
        # 2. 如果是字典或JSON，转为格式化字符串
        # 3. 截断过长内容

        if isinstance(input_data, str):
            content = input_data
        elif isinstance(input_data, dict):
            import json
            content = json.dumps(input_data, ensure_ascii=False, indent=2)
        else:
            content = str(input_data)

        # 限制长度
        max_length = 1000
        if len(content) > max_length:
            content = content[:max_length] + "..."

        return {
            'input_content': content,
            'input_type': type(input_data).__name__,
            'metadata': {}
        }

    def build_prompt(self, preprocessed_data: Dict) -> str:
        """
        构建通用prompt
        """
        template = self.get_prompt_template()
        content = preprocessed_data['input_content']

        user_content = template.format(input_content=content)

        system_prompt = "你是一个专业的众包任务指令生成专家。"

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def postprocess_output(self, raw_output: str) -> Dict:
        """
        通用后处理
        """
        # TODO: 实现通用后处理
        # 使用最基础的提取逻辑

        result = {
            'definition': '',
            'emphasis_caution': '',
            'things_to_avoid': '',
            'raw_output': raw_output
        }

        return result

    def validate_output(self, output: Dict) -> bool:
        """
        通用验证 - 最宽松的检查
        """
        # TODO: 基础验证
        # 只检查是否有输出

        return bool(output.get('raw_output', '').strip())