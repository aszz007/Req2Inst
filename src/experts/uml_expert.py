"""
UML逻辑指令专家
处理：UML用例图JSON -> 逻辑实现指令
"""

from typing import Dict, Any, List
import json
from src.experts.base_expert import BaseExpert


class UMLExpert(BaseExpert):
    """UML用例图JSON -> 逻辑指令的专家"""

    def __init__(self, lora_path: str = None, config: Dict = None):
        super().__init__(
            expert_name="uml_expert_qwen2.5",
            lora_path=lora_path,
            config=config
        )

    def get_prompt_template(self) -> str:
        """
        获取UML逻辑prompt模板
        对应你的数据集文档中的"uml序列图-指令"模板
        """
        template = """你是一个软件架构与众包任务设计专家。请根据以下输入的UML用例图结构化数据(JSON格式),编写一个适合众包工人使用的英文任务指令。

核心原则：
1. 数据驱动：指令中的角色名 (Actors) 和用例名 (Use Cases) 必须严格引用 JSON 源数据，不得遗漏或随意缩写。
2. 逻辑优先，视觉为辅：输入数据中包含 position (如 top_left) 和 image_path 等视觉信息，请在生成逻辑指令时**忽略**它们。重点解析 relationships 中的业务逻辑。
3. 关系语义转译：
   - include -> 必须转化为 "Mandatory step" (必须步骤) 或 "Pre-requisite"。
   - extend -> 必须转化为 "Conditional flow" (条件流程) 或 "Optional"。
   - association -> 必须转化为 "Interaction" (交互) 或 "Access"。
4. 结构规范：严格按照下方定义的格式输出。

格式要求：
- Definition: 使用简明扼要的祈使句描述核心系统目标。必须以 "In this task," 开头。
- Emphasis & Caution: 重点指出必须包含的流程 (include) 和条件扩展流程 (extend)。如无，填 "-"。
- Things to Avoid: 列出禁止的操作（如关注节点位置、实现UI样式）。如无特殊，填 "-"。

Input JSON: {input_json}

Output:"""

        return template

    def preprocess_input(self, input_data: Any) -> Dict:
        """
        预处理UML JSON输入

        Args:
            input_data: UML用例图JSON

        Returns:
            dict: 预处理后的数据
        """
        # TODO: 实现预处理
        # 1. 解析JSON
        # 2. 提取actors, use_cases, relationships
        # 3. 分析关系类型（include/extend/association）
        # 4. 忽略position等视觉信息
        # 5. 构建逻辑依赖图（可选）

        if isinstance(input_data, str):
            parsed_json = json.loads(input_data)
        elif isinstance(input_data, dict):
            parsed_json = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        actors = parsed_json.get('actors', [])
        use_cases = parsed_json.get('use_cases', [])
        relationships = parsed_json.get('relationships', [])

        # 分类关系
        includes = [r for r in relationships if r.get('type') == 'include']
        extends = [r for r in relationships if r.get('type') == 'extend']
        associations = [r for r in relationships if r.get('type') == 'association']

        return {
            'diagram_id': parsed_json.get('id', 'unknown'),
            'parsed_json': parsed_json,
            'actors': actors,
            'use_cases': use_cases,
            'relationships': {
                'all': relationships,
                'includes': includes,
                'extends': extends,
                'associations': associations
            },
            'metadata': {}
        }

    def build_prompt(self, preprocessed_data: Dict) -> str:
        """
        构建UML逻辑prompt
        """
        # TODO: 实现prompt构建
        # 1. 过滤掉position等视觉字段
        # 2. 只保留actors, use_cases, relationships
        # 3. 填充模板

        template = self.get_prompt_template()

        # 构建干净的JSON（去除视觉信息）
        clean_json = {
            'actors': [{'name': a.get('name', a)} for a in preprocessed_data['actors']],
            'use_cases': [{'name': uc.get('name', uc)} for uc in preprocessed_data['use_cases']],
            'relationships': preprocessed_data['relationships']['all']
        }

        json_str = json.dumps(clean_json, ensure_ascii=False, indent=2)
        user_content = template.format(input_json=json_str)

        system_prompt = "你是一个软件架构与众包任务设计专家。"

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def postprocess_output(self, raw_output: str) -> Dict:
        """
        后处理UML逻辑指令输出
        """
        # TODO: 实现后处理
        # 1. 提取三个字段
        # 2. 验证关系语义转译是否正确
        # 3. 检查是否包含所有actors和use_cases

        result = {
            'definition': '',
            'emphasis_caution': '',
            'things_to_avoid': '',
            'included_flows': [],  # include关系
            'extended_flows': [],  # extend关系
            'raw_output': raw_output
        }

        return result

    def validate_output(self, output: Dict) -> bool:
        """
        验证UML逻辑指令

        检查：
        - Definition描述了核心用例
        - Emphasis提到了include关系（如果有）
        - Avoid提到了不关注position
        """
        # TODO: 实现验证

        return True

    def _translate_relationship(self, rel_type: str) -> str:
        """
        关系类型语义转译

        Args:
            rel_type: include/extend/association

        Returns:
            str: 转译后的描述
        """
        translations = {
            'include': 'Mandatory step',
            'extend': 'Conditional flow',
            'association': 'Interaction'
        }
        return translations.get(rel_type, rel_type)

