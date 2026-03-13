"""
UML业务逻辑指令生成Prompt模板
功能：将UML用例图JSON转换为业务逻辑实现指令
输入：UML JSON（包含actors, use_cases, relationships等）
输出：三段式业务逻辑实现指令（Definition / Emphasis & Caution / Things to Avoid）
"""

import json
from typing import Union


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
            >>> # 方式1: 传入JSON字符串
            >>> json_str = '{"actors": [...], "use_cases": [...], "relationships": [...]}'
            >>> prompt = UMLInstructionTemplate.build_prompt(json_str)

            >>> # 方式2: 传入dict对象
            >>> uml_data = {
            ...     "actors": [{"name": "User", "position": "left"}],
            ...     "use_cases": [{"name": "Login", "description": "User login"}],
            ...     "relationships": [{"type": "association", "from": "User", "to": "Login"}]
            ... }
            >>> prompt = UMLInstructionTemplate.build_prompt(uml_data)
        """
        # 处理输入格式
        if isinstance(uml_json, dict):
            # 过滤元数据字段和actor中的position字段
            filtered_data = {
                k: v for k, v in uml_json.items()
                if k not in ['confidence', 'recognition_status', 'processing_time']
            }
            # 过滤actor中的position字段
            if 'actors' in filtered_data and isinstance(filtered_data['actors'], list):
                filtered_actors = []
                for actor in filtered_data['actors']:
                    if isinstance(actor, dict):
                        # 移除position字段
                        filtered_actor = {k: v for k, v in actor.items() if k != 'position'}
                        filtered_actors.append(filtered_actor)
                    else:
                        filtered_actors.append(actor)
                filtered_data['actors'] = filtered_actors

            # 转为压缩JSON字符串（无空格、无换行）
            json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
        elif isinstance(uml_json, str):
            # 尝试解析为JSON以验证格式
            try:
                parsed = json.loads(uml_json)
                # 过滤元数据字段和actor中的position字段
                filtered_data = {
                    k: v for k, v in parsed.items()
                    if k not in ['confidence', 'recognition_status', 'processing_time']
                }
                # 过滤actor中的position字段
                if 'actors' in filtered_data and isinstance(filtered_data['actors'], list):
                    filtered_actors = []
                    for actor in filtered_data['actors']:
                        if isinstance(actor, dict):
                            # 移除position字段
                            filtered_actor = {k: v for k, v in actor.items() if k != 'position'}
                            filtered_actors.append(filtered_actor)
                        else:
                            filtered_actors.append(actor)
                    filtered_data['actors'] = filtered_actors

                # 转为压缩JSON字符串（无空格、无换行）
                json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
            except json.JSONDecodeError:
                # 如果不是有效JSON，直接使用
                json_str = uml_json
        else:
            raise TypeError("uml_json必须是str或dict类型")

        # 构建用户消息
        user_message = f"""UML Use Case Diagram structured data (JSON format):
```json
{json_str}
```

{UMLInstructionTemplate.FORMAT_INSTRUCTIONS}"""

        # 构建完整的Qwen格式prompt（assistant部分使用空think块禁用Qwen3思考模式）
        prompt = f"""<|im_start|>system
{UMLInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

        return prompt

    @staticmethod
    def build_batch_prompt(uml_jsons: list) -> list:
        """
        批量构建prompt

        Args:
            uml_jsons: UML JSON列表（支持str或dict）

        Returns:
            list: prompt列表
        """
        return [
            UMLInstructionTemplate.build_prompt(uml)
            for uml in uml_jsons
        ]

    @staticmethod
    def extract_key_elements(uml_data: Union[str, dict]) -> dict:
        """
        从UML JSON中提取关键业务元素（忽略视觉信息）

        Args:
            uml_data: UML JSON数据

        Returns:
            dict: 提取的关键元素
                {
                    'actors': list,
                    'use_cases': list,
                    'include_relations': list,
                    'extend_relations': list,
                    'associations': list
                }
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
        relationships = data.get('relationships', [])
        include_relations = []
        extend_relations = []
        associations = []

        for rel in relationships:
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

        修复：检查结构而非仅关键词存在性

        Args:
            instruction: 生成的指令文本

        Returns:
            dict: 验证结果
        """
        result = {
            'is_valid': True,
            'has_definition': False,
            'has_business_logic': False,
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
        for line in lines:
            line_lower = line.lower()

            # 检查Definition行
            if line.startswith('Definition:'):
                content = line[len('Definition:'):].strip()
                if content:
                    result['has_definition'] = True
                    # 检查是否包含业务逻辑关键词（UML任务的核心）
                    # 注意：使用子串匹配，需包含模型实际生成的动词变体
                    # interact/interacts/interacting 均不含 interaction，需单独列出
                    business_keywords = [
                        'implement', 'functionality', 'workflow', 'process',
                        'interaction', 'interact', 'trigger', 'system',
                        'analyze', 'manage', 'execute', 'perform'
                    ]
                    if any(keyword in line_lower for keyword in business_keywords):
                        result['has_business_logic'] = True
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

        if not result['has_business_logic']:
            result['errors'].append('Definition未体现业务逻辑实现要求')

        if not result['has_emphasis']:
            result['errors'].append('缺少"Emphasis & Caution:"部分或格式错误')

        if not result['has_avoid']:
            result['errors'].append('缺少"Things to Avoid:"部分或格式错误')

        # 综合判断
        result['is_valid'] = all([
            result['has_definition'],
            result['has_business_logic'],
            result['has_emphasis'],
            result['has_avoid']
        ])

        return result