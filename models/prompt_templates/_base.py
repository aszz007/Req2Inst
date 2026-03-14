"""
Prompt模板公共工具
功能：提供所有Prompt模板共用的工具方法，消除模板间的重复代码
包含：JSON预处理、Qwen格式构建、三段式验证等
"""

import json
from typing import Union, Optional, List, Dict, Callable

# 元数据字段（不参与指令生成，仅用于预处理阶段的识别结果管理）
_METADATA_FIELDS = {'confidence', 'recognition_status', 'processing_time'}


def filter_metadata(data: dict) -> dict:
    """
    过滤元数据字段（confidence、recognition_status、processing_time等）

    Args:
        data: 原始字典

    Returns:
        dict: 过滤后的字典
    """
    return {k: v for k, v in data.items() if k not in _METADATA_FIELDS}


def filter_actor_positions(data: dict) -> dict:
    """
    过滤UML数据中actor的position字段（视觉布局信息不参与指令生成）

    Args:
        data: 包含actors字段的字典

    Returns:
        dict: 过滤后的字典（原字典不被修改）
    """
    if 'actors' not in data or not isinstance(data.get('actors'), list):
        return data

    import copy
    result = copy.deepcopy(data)
    result['actors'] = [
        {k: v for k, v in actor.items() if k != 'position'}
        if isinstance(actor, dict) else actor
        for actor in result['actors']
    ]
    return result


def compress_json(data) -> str:
    """
    将数据转为压缩JSON字符串（无空格、无换行）

    Args:
        data: 可序列化的数据

    Returns:
        str: 压缩后的JSON字符串
    """
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


def process_json_input(input_data: Union[str, dict],
                       filter_meta: bool = True,
                       filter_positions: bool = False) -> str:
    """
    统一处理JSON输入：解析 → 过滤 → 压缩

    适用于image_template、uml_template、general_template中的JSON输入预处理，
    将dict或JSON字符串统一转换为压缩JSON字符串。

    Args:
        input_data: 输入数据（str或dict）
        filter_meta: 是否过滤元数据字段
        filter_positions: 是否过滤actor的position字段

    Returns:
        str: 处理后的压缩JSON字符串；如果输入非JSON字符串则原样返回
    """
    # dict → 过滤 → 压缩
    if isinstance(input_data, dict):
        data = input_data
        if filter_meta:
            data = filter_metadata(data)
        if filter_positions:
            data = filter_actor_positions(data)
        return compress_json(data)

    # str → 尝试解析 → 过滤 → 压缩；解析失败则原样返回
    if isinstance(input_data, str):
        try:
            parsed = json.loads(input_data)
            if isinstance(parsed, dict):
                if filter_meta:
                    parsed = filter_metadata(parsed)
                if filter_positions:
                    parsed = filter_actor_positions(parsed)
                return compress_json(parsed)
            return compress_json(parsed)
        except json.JSONDecodeError:
            return input_data

    return str(input_data)


def build_qwen_prompt(system_prompt: str, user_message: str) -> str:
    """
    构建Qwen对话格式prompt（含空think块禁用Qwen3思考模式）

    所有模板共用此格式，避免各模板重复拼接。

    Args:
        system_prompt: 系统提示词
        user_message: 用户消息

    Returns:
        str: 完整的Qwen格式prompt
    """
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


def validate_three_part_format(instruction: str,
                                extra_checks: Optional[List[Dict]] = None) -> dict:
    """
    验证生成的指令是否符合三段式格式（所有模板共用基础验证逻辑）

    基础检查：Definition / Emphasis & Caution / Things to Avoid 三部分是否存在
    额外检查：由各模板传入的特定验证规则（如image要求bounding boxes、uml要求业务逻辑）

    Args:
        instruction: 生成的指令文本
        extra_checks: 额外的验证规则列表，每个规则为字典：
            {
                'key': str,            # 结果字段名（如 'has_bounding_boxes'）
                'check_fn': callable,  # 检查函数，接收(line: str, line_lower: str)，返回bool
                'section': str,        # 在哪个section检查（'definition'/'emphasis'/'avoid'/'any'）
                'error_msg': str,      # 检查失败时的错误消息
                'required': bool,      # 是否为is_valid必须条件（默认True）
            }

    Returns:
        dict: 验证结果，包含 is_valid, has_definition, has_emphasis, has_avoid, errors 等字段
    """
    result = {
        'is_valid': True,
        'has_definition': False,
        'has_emphasis': False,
        'has_avoid': False,
        'errors': []
    }

    # 初始化额外检查字段
    if extra_checks:
        for check in extra_checks:
            result[check['key']] = False

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
                # 对Definition行执行额外检查
                if extra_checks:
                    for check in extra_checks:
                        if check['section'] in ('definition', 'any'):
                            if check['check_fn'](line, line_lower):
                                result[check['key']] = True
            else:
                result['errors'].append('Definition部分内容为空')

        # 检查Emphasis & Caution行
        elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
            result['has_emphasis'] = True
            if extra_checks:
                for check in extra_checks:
                    if check['section'] in ('emphasis', 'any'):
                        if check['check_fn'](line, line_lower):
                            result[check['key']] = True

        # 检查Things to Avoid行
        elif line.startswith('Things to Avoid:'):
            result['has_avoid'] = True
            if extra_checks:
                for check in extra_checks:
                    if check['section'] in ('avoid', 'any'):
                        if check['check_fn'](line, line_lower):
                            result[check['key']] = True

    # 检查缺失的基础部分
    if not result['has_definition']:
        result['errors'].append('缺少"Definition:"部分或格式错误')
    if not result['has_emphasis']:
        result['errors'].append('缺少"Emphasis & Caution:"部分或格式错误')
    if not result['has_avoid']:
        result['errors'].append('缺少"Things to Avoid:"部分或格式错误')

    # 检查额外规则的错误
    required_keys = ['has_definition', 'has_emphasis', 'has_avoid']
    if extra_checks:
        for check in extra_checks:
            if not result[check['key']]:
                result['errors'].append(check['error_msg'])
            if check.get('required', True):
                required_keys.append(check['key'])

    # 综合判断
    result['is_valid'] = all(result[k] for k in required_keys)

    return result


def build_batch_prompts(input_list: list, build_fn: Callable) -> list:
    """
    批量构建prompt（所有模板共用）

    Args:
        input_list: 输入数据列表
        build_fn: 单条prompt构建函数

    Returns:
        list: prompt列表
    """
    return [build_fn(item) for item in input_list]