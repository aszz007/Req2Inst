#!/usr/bin/env python3
"""
批量清理所有LoRA专家的配置文件

使用方法：
python clean_all_lora_configs.py
"""

import json
import shutil
from pathlib import Path


def clean_config(config_file: Path) -> bool:
    """清理单个配置文件"""
    if not config_file.exists():
        return False

    # 备份
    backup_file = config_file.parent / "adapter_config.json.backup"
    if not backup_file.exists():
        shutil.copy2(config_file, backup_file)

    # 读取
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # peft 0.12.0 标准参数
    standard_params = {
        'peft_type', 'auto_mapping', 'base_model_name_or_path', 'revision',
        'task_type', 'inference_mode', 'r', 'target_modules', 'lora_alpha',
        'lora_dropout', 'fan_in_fan_out', 'bias', 'modules_to_save',
        'init_lora_weights', 'layers_to_transform', 'layers_pattern',
    }

    # 清理
    cleaned_config = {k: v for k, v in config.items() if k in standard_params}
    removed = set(config.keys()) - standard_params

    # 保存
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_config, f, indent=2, ensure_ascii=False)

    return len(removed) > 0


def main():
    lora_base = Path("lora_weights/experts")

    if not lora_base.exists():
        print(f"错误: 目录不存在: {lora_base}")
        return

    configs = list(lora_base.glob("*/adapter_config.json"))

    if not configs:
        print(f"未找到配置文件在: {lora_base}")
        return

    print(f"找到 {len(configs)} 个配置文件\n")

    cleaned_count = 0
    for config_file in configs:
        expert_name = config_file.parent.name
        print(f"处理: {expert_name}")

        if clean_config(config_file):
            print(f"  ✓ 已清理并备份")
            cleaned_count += 1
        else:
            print(f"  - 跳过（已是兼容格式）")

    print(f"\n完成! 清理了 {cleaned_count}/{len(configs)} 个配置文件")
    print("原始配置已备份为 adapter_config.json.backup")


if __name__ == "__main__":
    main()