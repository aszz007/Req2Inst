"""
Fix JSON Output - 修复视觉模型识别结果的 JSON 格式

问题背景:
    Qwen3-VL 视觉模型（recognize_uml.py / recognize_image.py）批量识别后输出的 JSON 文件中，
    每条记录的 description 字段是一个被序列化为字符串的 JSON（即 JSON-in-string），
    没有换行和缩进，可读性差，且无法直接作为结构化数据使用。

解决方案:
    读取识别结果 JSON 文件，将每条记录中 description 字段的字符串值
    反序列化（json.loads）还原为嵌套 JSON 对象，再整体以格式化方式（indent=4）
    重新写出，使文件具备正常的层级缩进和换行。

使用方式:
    1. 修改脚本顶部的 input_path / output_path 指向目标文件
    2. 直接运行: python scripts/utils/fix_json_output.py

适用场景:
    - UML 识别结果: outputs/recognition_results/uml/*.json
    - 图像识别结果: outputs/recognition_results/image/*.json
"""

import json
import os

# 输入和输出路径（相对于项目根目录）
input_path = "outputs/recognition_results/uml/uml_recognition_qwen3_20260210_052354.json"
output_path = "outputs/recognition_results/uml/uml_recognition_qwen3_20260210_052354_fixed.json"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 读取原始 JSON 文件
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 如果顶层是列表（通常 batch 输出是 list of results）
if isinstance(data, list):
    for item in data:
        if 'description' in item and isinstance(item['description'], str):
            try:
                # 尝试将 description 字符串解析为 JSON 对象
                item['description'] = json.loads(item['description'])
            except json.JSONDecodeError as e:
                print(f"无法解析 description 字段（保留原字符串）: {item.get('image_name', 'unknown')} - 错误: {e}")
elif isinstance(data, dict):
    # 如果顶层是单个对象（不太可能，但兼容）
    if 'description' in data and isinstance(data['description'], str):
        try:
            data['description'] = json.loads(data['description'])
        except json.JSONDecodeError as e:
            print(f"无法解析顶层 description: {e}")
else:
    raise ValueError("未知的 JSON 结构：既不是列表也不是字典")

# 写入修复后的 JSON 文件（格式化、带缩进）
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"修复完成，已保存到: {output_path}")