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
    SYSTEM_PROMPT = """你是一个软件架构与众包任务设计专家。请根据以下输入的UML用例图结构化数据（JSON格式），编写一个适合众包工人使用的英文业务逻辑实现指令。

核心原则：
1. 数据驱动：指令中的角色名（Actors）和用例名（Use Cases）必须严格引用JSON源数据中的英文原名，不得遗漏、缩写或改写。
2. 逻辑优先，视觉为辅：输入数据中包含position（如top_left）等视觉布局信息，请在生成业务逻辑指令时完全忽略它们。重点解析relationships中的业务逻辑关系。
3. 关系语义转译：
   - include → 必须转化为"Mandatory step"（必须步骤）或"Required prerequisite"（必需前置条件）
   - extend → 必须转化为"Conditional extension"（条件扩展）或"Optional flow"（可选流程）
   - association → 必须转化为"Actor interaction"（角色交互）或"Triggers"（触发关系）
4. 众包任务导向：明确这是给开发人员的实现指令，重点说明要实现什么功能、如何处理不同的业务流程分支。
5. 结构规范：严格按照下方定义的格式输出。"""

    # 格式要求说明
    FORMAT_INSTRUCTIONS = """格式要求：
- Definition：使用简明扼要的祈使句描述核心系统目标和主要参与角色。必须以"In this task,"开头。
- Emphasis & Caution：重点指出必须包含的流程（include关系）和条件扩展流程（extend关系），说明触发条件。如无特别强调，填"-"。
- Things to Avoid：列出禁止的操作（如关注position、实现UI样式等）。如无特殊禁止事项，填"-"。"""

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
            # 如果是字典，转为JSON字符串
            json_str = json.dumps(uml_json, ensure_ascii=False, indent=2)
        elif isinstance(uml_json, str):
            # 尝试解析为JSON以验证格式
            try:
                parsed = json.loads(uml_json)
                # 如果能解析，格式化输出
                json_str = json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # 如果不是有效JSON，直接使用
                json_str = uml_json
        else:
            raise TypeError("uml_json必须是str或dict类型")

        # 构建用户消息
        user_message = f"""UML用例图结构化数据（JSON格式）：
```json
{json_str}
```

{UMLInstructionTemplate.FORMAT_INSTRUCTIONS}

请开始生成业务逻辑实现指令："""

        # 构建完整的Qwen格式prompt
        prompt = f"""<|im_start|>system
{UMLInstructionTemplate.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
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

        instruction_lower = instruction.lower()

        # 检查Definition（必须以"In this task"开头）
        if 'in this task' in instruction_lower:
            result['has_definition'] = True
        else:
            result['errors'].append('缺少Definition部分或未以"In this task"开头')

        # 检查是否包含业务逻辑关键词
        business_keywords = [
            'implement', 'functionality', 'workflow', 'process',
            'mandatory', 'required', 'conditional', 'optional',
            'interaction', 'trigger'
        ]
        if any(keyword in instruction_lower for keyword in business_keywords):
            result['has_business_logic'] = True
        else:
            result['errors'].append('未体现业务逻辑实现要求')

        # 检查Emphasis & Caution
        if 'emphasis' in instruction_lower or 'caution' in instruction_lower or '-' in instruction:
            result['has_emphasis'] = True
        else:
            result['errors'].append('缺少Emphasis & Caution部分')

        # 检查Things to Avoid（应该提到不关注UI/position）
        if 'avoid' in instruction_lower or '-' in instruction:
            result['has_avoid'] = True
        else:
            result['errors'].append('缺少Things to Avoid部分')

        # 综合判断
        result['is_valid'] = all([
            result['has_definition'],
            result['has_business_logic'],
            result['has_emphasis'],
            result['has_avoid']
        ])

        return result


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("UML Prompt模板测试")
    print("=" * 60)

    # 测试1: 完整UML JSON对象
    print("\n【测试1】完整UML JSON对象输入")
    print("-" * 60)
    uml_data = {
        "actors": [
            {"name": "User", "position": "left"},
            {"name": "Admin", "position": "right"}
        ],
        "use_cases": [
            {"name": "Login System", "description": "User authentication"},
            {"name": "Validate Credentials", "description": "Check username and password"},
            {"name": "Send Email", "description": "Email notification"}
        ],
        "system_boundary": {
            "name": "Authentication System",
            "is_present": True
        },
        "relationships": [
            {
                "type": "association",
                "from": "User",
                "to": "Login System",
                "description": "User initiates login"
            },
            {
                "type": "include",
                "from": "Login System",
                "to": "Validate Credentials",
                "description": "Must validate before login"
            },
            {
                "type": "extend",
                "from": "Send Email",
                "to": "Login System",
                "description": "Optional email notification on success"
            }
        ],
        "overall_description": "This system handles user authentication with mandatory validation and optional email notification."
    }

    prompt = UMLInstructionTemplate.build_prompt(uml_data)
    print("生成的Prompt（前500字符）：")
    print(prompt[:500])
    print("...")

    # 测试2: JSON字符串
    print("\n【测试2】JSON字符串输入")
    print("-" * 60)
    json_str = json.dumps(uml_data)
    prompt = UMLInstructionTemplate.build_prompt(json_str)
    print(f"输入类型: JSON字符串")
    print(f"Prompt长度: {len(prompt)} 字符")

    # 测试3: 提取关键业务元素
    print("\n【测试3】提取关键业务元素（忽略position）")
    print("-" * 60)
    elements = UMLInstructionTemplate.extract_key_elements(uml_data)
    print(f"Actors: {elements['actors']}")
    print(f"Use Cases: {[uc['name'] for uc in elements['use_cases']]}")
    print(f"Include Relations: {len(elements['include_relations'])} 个")
    print(f"Extend Relations: {len(elements['extend_relations'])} 个")
    print(f"Associations: {len(elements['associations'])} 个")

    # 测试4: 批量生成
    print("\n【测试4】批量生成")
    print("-" * 60)
    uml_list = [
        {
            "actors": [{"name": "Customer"}],
            "use_cases": [{"name": "Place Order"}],
            "relationships": []
        },
        {
            "actors": [{"name": "Manager"}],
            "use_cases": [{"name": "Generate Report"}],
            "relationships": []
        }
    ]
    prompts = UMLInstructionTemplate.build_batch_prompt(uml_list)
    print(f"成功生成 {len(prompts)} 个prompts")

    # 测试5: 指令验证
    print("\n【测试5】指令格式验证")
    print("-" * 60)

    # 正确的指令
    valid_instruction = """Definition: In this task, implement the authentication system workflow with User and Admin actors interacting with Login System and Validate Credentials use cases.
Emphasis & Caution: Ensure Validate Credentials is a mandatory prerequisite (include relation) before completing login. Send Email is a conditional extension triggered on successful login.
Things to Avoid: Do not focus on UI positioning or visual layout. Avoid implementing frontend elements."""

    result = UMLInstructionTemplate.validate_instruction(valid_instruction)
    print(f"正确指令验证: {result['is_valid']}")
    print(f"包含业务逻辑: {result['has_business_logic']}")

    # 错误的指令（缺少业务逻辑）
    invalid_instruction = """Definition: In this task, draw the UML diagram.
Emphasis & Caution: Make it look nice.
Things to Avoid: -"""

    result = UMLInstructionTemplate.validate_instruction(invalid_instruction)
    print(f"\n错误指令验证: {result['is_valid']}")
    print(f"错误信息: {result['errors']}")

    print("\n测试完成！")