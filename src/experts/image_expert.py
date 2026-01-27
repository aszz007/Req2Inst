"""
图像标注指令专家
处理：图像JSON -> 边框标注指令
"""

from typing import Dict, Any
import json
from .base_expert import BaseExpert


class ImageExpert(BaseExpert):
    """图像JSON -> 标注指令的专家"""

    def __init__(self, lora_path: str = None, config: Dict = None):
        super().__init__(
            expert_name="image_expert_qwen2.5",
            lora_path=lora_path,
            config=config
        )

    def get_prompt_template(self) -> str:
        """
        获取图像标注prompt模板
        对应你的数据集文档中的"图像-指令"模板
        """
        template = """你是一个计算机视觉数据专家与众包任务设计者。请根据以下输入的图像分析结构化数据，编写一个适合众包工人使用的英文图像标注任务指令。

核心原则：
1. 标注导向：指令必须明确要求工人进行 "Draw bounding boxes" (画边框)。
2. 前景提取：从 objects 中提取主要的前景实体（如人、车）作为目标，忽略背景元素。
3. 直接引用：直接使用 JSON 中的英文术语，不要进行同义词替换。
4. 极致精简：Emphasis 和 Avoid 部分必须言简意赅。如果 JSON 中缺乏显著的视觉特征或干扰项，直接填 "-"。

格式要求：
- Definition: 使用简明扼要的祈使句描述标注目标。必须以 "In this task," 开头。
- Emphasis & Caution: 仅列出极具识别性的视觉特征（如特定颜色、位置）。如无特别强调，填 "-"。
- Things to Avoid: 仅列出容易混淆的背景干扰项。如无特别避免事项，填 "-"。

Input JSON: {input_json}

Output:"""

        return template

    def preprocess_input(self, input_data: Any) -> Dict:
        """
        预处理图像JSON输入

        Args:
            input_data: JSON字符串或字典

        Returns:
            dict: {'image_id': str, 'parsed_json': dict, 'metadata': dict}
        """
        # TODO: 实现预处理逻辑
        # 1. 解析JSON
        # 2. 验证必需字段（file, result, objects等）
        # 3. 提取前景对象（过滤背景）
        # 4. 提取视觉特征（颜色、位置等）

        if isinstance(input_data, str):
            parsed_json = json.loads(input_data)
        elif isinstance(input_data, dict):
            parsed_json = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # 提取关键信息
        image_id = parsed_json.get('file', 'unknown')
        result = parsed_json.get('result', {})
        objects = result.get('details', {}).get('objects', [])

        return {
            'image_id': image_id,
            'parsed_json': parsed_json,
            'objects': objects,
            'metadata': {}
        }

    def build_prompt(self, preprocessed_data: Dict) -> str:
        """
        构建图像标注prompt
        """
        # TODO: 实现prompt构建
        # 1. 获取模板
        # 2. 将JSON转为字符串
        # 3. 填充模板
        # 4. 添加Qwen格式

        template = self.get_prompt_template()
        json_str = json.dumps(preprocessed_data['parsed_json'], ensure_ascii=False)

        user_content = template.format(input_json=json_str)

        system_prompt = "你是一个计算机视觉数据专家与众包任务设计者。"

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def postprocess_output(self, raw_output: str) -> Dict:
        """
        后处理图像标注指令输出
        """
        # TODO: 实现后处理
        # 1. 提取三个字段
        # 2. 验证"Draw bounding boxes"关键词
        # 3. 提取标注对象列表

        result = {
            'definition': '',
            'emphasis_caution': '',
            'things_to_avoid': '',
            'annotation_targets': [],  # 提取的标注目标
            'raw_output': raw_output
        }

        return result

    def validate_output(self, output: Dict) -> bool:
        """
        验证图像标注指令

        检查：
        - Definition包含"draw bounding boxes"
        - 明确指定了标注目标
        """
        # TODO: 实现验证

        return True

