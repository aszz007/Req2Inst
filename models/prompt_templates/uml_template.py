"""
UML业务逻辑指令生成Prompt模板
功能：将UML用例图JSON转换为业务逻辑实现指令
输入：UML JSON（包含actors, use_cases, relationships等）
输出：三段式业务逻辑实现指令（Definition / Emphasis & Caution / Things to Avoid）
"""

import json
from typing import Union

from ._base import (
    build_qwen_prompt, validate_three_part_format,
    build_batch_prompts, process_json_input,
)


class UMLInstructionTemplate:
    """UML JSON → 业务逻辑实现指令 的Prompt模板"""

    # ==================== 识别阶段Prompt（预处理） ====================
    UML_RECOGNITION_PROMPT = """Please carefully analyze this Use Case Diagram and output the recognition results in JSON format.

A Use Case Diagram is a type of UML diagram used to describe system functions and user interactions. Please identify:

1. **actors**: List of actors (typically stick figures or text labels)
   - Each actor includes: name, position (e.g., "left", "right")

2. **use_cases**: List of use cases (typically ovals)
   - Each use case includes: name, description (brief description)

3. **system_boundary**: System boundary
   - Includes: name (system name), is_present (whether boundary box exists)

4. **relationships**: List of relationships
   - Each relationship includes:
     - type ("association", "include", "extend", "generalization")
     - from (starting element)
     - to (ending element)
     - description (relationship description)

5. **overall_description**: Overall description (summarize the system functionality in one paragraph)

Please output strictly in JSON format. Example:
{
  "actors": [{"name": "User", "position": "left"}],
  "use_cases": [{"name": "Login System", "description": "User login functionality"}],
  "system_boundary": {"name": "System Name", "is_present": true},
  "relationships": [{"type": "association", "from": "User", "to": "Login System", "description": "User can login"}],
  "overall_description": "This is a use case diagram..."
}

If the image is not a use case diagram or cannot be recognized, please explain in overall_description.
Important: Ensure complete JSON output with all brackets properly closed. Use English for all content."""

    @staticmethod
    def get_recognition_prompt() -> str:
        """
        获取UML识别Prompt（用于预处理阶段）

        Returns:
            str: UML识别的Prompt文本
        """
        return UMLInstructionTemplate.UML_RECOGNITION_PROMPT

    # ==================== 指令生成阶段Prompt ====================
    # 系统提示词（定义角色和核心原则）
    SYSTEM_PROMPT = """You are a software architecture and crowdsourcing task design expert. Based on the input UML Use Case Diagram structured data (JSON format), write an English task instruction for crowdsourcing workers.

Core Principles:
1. Data-Driven: Actor names and Use Case names in the instruction must strictly reference the original names from JSON source data. Do not omit, abbreviate, or rewrite.
2. Logic Priority, Visuals Secondary: Completely ignore visual layout information like position (e.g., top_left) in input data. Focus on parsing business logic in relationships.
3. Relationship Semantics Translation:
   - include -> Translate to "Mandatory step" or "Required prerequisite"
   - extend -> Translate to "Conditional flow" or "Optional"
   - association -> Translate to "Interaction" or "Access"
4. Structured Format: Strictly follow the three-part format defined below."""

    # 格式要求说明
    FORMAT_INSTRUCTIONS = """Output Format Requirements:

Definition: Use a clear imperative sentence to describe the core system objective. Must start with "In this task,".
Emphasis & Caution: Highlight mandatory flows (include) and conditional extension flows (extend). Use "-" if none.
Things to Avoid: List prohibited operations (e.g., focusing on node positions, implementing UI styles). Use "-" if nothing specific.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Definition must start with "In this task," and explicitly list actors and use cases from JSON data
- Translate relationship types (include/extend/association) to business logic terms
- Keep all sections concise
- Output ONLY these three lines, nothing else"""

    @staticmethod
    def build_prompt(uml_json: Union[str, dict]) -> str:
        """
        构建UML JSON生成业务逻辑指令的完整prompt

        Args:
            uml_json: UML用例图数据，支持两种格式：
                1. JSON字符串：完整的UML识别结果
                2. dict对象：UML识别结果字典

        Returns:
            str: 完整的prompt（Qwen对话格式）

        Example:
            >>> json_str = '{"actors": [...], "use_cases": [...], "relationships": [...]}'
            >>> prompt = UMLInstructionTemplate.build_prompt(json_str)

            >>> uml_data = {
            ...     "actors": [{"name": "User", "position": "left"}],
            ...     "use_cases": [{"name": "Login", "description": "User login"}],
            ...     "relationships": [{"type": "association", "from": "User", "to": "Login"}]
            ... }
            >>> prompt = UMLInstructionTemplate.build_prompt(uml_data)
        """
        # 统一处理：过滤元数据 + 过滤actor position + 压缩JSON
        json_str = process_json_input(uml_json, filter_meta=True, filter_positions=True)

        # 非JSON字符串直接使用
        if not isinstance(uml_json, (str, dict)):
            raise TypeError("uml_json必须是str或dict类型")

        # 构建用户消息
        user_message = f"""UML Use Case Diagram structured data (JSON format):
```json
{json_str}
```

{UMLInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        return build_qwen_prompt(UMLInstructionTemplate.SYSTEM_PROMPT, user_message)

    @staticmethod
    def build_batch_prompt(uml_jsons: list) -> list:
        """
        批量构建prompt

        Args:
            uml_jsons: UML JSON列表（支持str或dict）

        Returns:
            list: prompt列表
        """
        return build_batch_prompts(uml_jsons, UMLInstructionTemplate.build_prompt)

    @staticmethod
    def extract_key_elements(uml_data: Union[str, dict]) -> dict:
        """
        从UML JSON中提取关键业务元素（忽略视觉信息）

        Args:
            uml_data: UML JSON数据

        Returns:
            dict: 提取的关键元素
        """
        if isinstance(uml_data, str):
            try:
                data = json.loads(uml_data)
            except json.JSONDecodeError:
                return {
                    'actors': [],
                    'use_cases': [],
                    'include_relations': [],
                    'extend_relations': [],
                    'associations': []
                }
        else:
            data = uml_data

        # 提取actors（仅名称，忽略position）
        actors = [
            actor.get('name', actor) if isinstance(actor, dict) else actor
            for actor in data.get('actors', [])
        ]

        # 提取use_cases（仅名称和描述，忽略position）
        use_cases = []
        for uc in data.get('use_cases', []):
            if isinstance(uc, dict):
                use_cases.append({
                    'name': uc.get('name', ''),
                    'description': uc.get('description', '')
                })
            else:
                use_cases.append({'name': str(uc), 'description': ''})

        # 按类型分类relationships
        include_relations = []
        extend_relations = []
        associations = []

        for rel in data.get('relationships', []):
            if isinstance(rel, dict):
                rel_type = rel.get('type', '').lower()
                relation_info = {
                    'from': rel.get('from', ''),
                    'to': rel.get('to', ''),
                    'description': rel.get('description', '')
                }
                if 'include' in rel_type:
                    include_relations.append(relation_info)
                elif 'extend' in rel_type:
                    extend_relations.append(relation_info)
                elif 'association' in rel_type:
                    associations.append(relation_info)

        return {
            'actors': actors,
            'use_cases': use_cases,
            'include_relations': include_relations,
            'extend_relations': extend_relations,
            'associations': associations
        }

    @staticmethod
    def validate_instruction(instruction: str) -> dict:
        """
        验证生成的指令是否符合UML业务逻辑三段式格式

        Args:
            instruction: 生成的指令文本

        Returns:
            dict: 验证结果
        """
        business_keywords = [
            'implement', 'functionality', 'workflow', 'process',
            'interaction', 'interact', 'trigger', 'system',
            'analyze', 'manage', 'execute', 'perform'
        ]
        extra_checks = [
            {
                'key': 'has_business_logic',
                'check_fn': lambda line, ll: any(kw in ll for kw in business_keywords),
                'section': 'definition',
                'error_msg': 'Definition未体现业务逻辑实现要求',
                'required': True,
            }
        ]
        return validate_three_part_format(instruction, extra_checks=extra_checks)