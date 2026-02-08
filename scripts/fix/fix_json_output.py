import json
import os

# 输入和输出路径（相对于项目根目录）
input_path = "outputs/recognition_results/uml/uml_recognition_qwen3_20260209_045521.json"
output_path = "outputs/recognition_results/uml/uml_recognition_qwen3_20260209_045521_fixed.json"

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
                print(f"⚠️ 无法解析 description 字段（保留原字符串）: {item.get('image_name', 'unknown')} - 错误: {e}")
elif isinstance(data, dict):
    # 如果顶层是单个对象（不太可能，但兼容）
    if 'description' in data and isinstance(data['description'], str):
        try:
            data['description'] = json.loads(data['description'])
        except json.JSONDecodeError as e:
            print(f"⚠️ 无法解析顶层 description: {e}")
else:
    raise ValueError("未知的 JSON 结构：既不是列表也不是字典")

# 写入修复后的 JSON 文件（格式化、带缩进）
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ 修复完成！已保存到: {output_path}")