#!/usr/bin/env python3
"""
清理LoRA配置文件，移除peft 0.18.1新增的不兼容参数
使这些配置可以被peft 0.12.0读取

使用方法：
python clean_lora_config.py lora_weights/experts/image_expert_qwen3
"""

import json
import sys
import shutil
from pathlib import Path


def clean_lora_config(lora_dir: str):
    """清理LoRA配置文件"""
    lora_path = Path(lora_dir)
    config_file = lora_path / "adapter_config.json"

    if not config_file.exists():
        print(f"错误: 配置文件不存在: {config_file}")
        return False

    # 备份原始配置
    backup_file = lora_path / "adapter_config.json.backup"
    shutil.copy2(config_file, backup_file)
    print(f"已备份原始配置到: {backup_file}")

    # 读取配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # peft 0.12.0 支持的标准参数（白名单）
    standard_params = {
        'peft_type',
        'auto_mapping',
        'base_model_name_or_path',
        'revision',
        'task_type',
        'inference_mode',
        'r',
        'target_modules',
        'lora_alpha',
        'lora_dropout',
        'fan_in_fan_out',
        'bias',
        'modules_to_save',
        'init_lora_weights',
        'layers_to_transform',
        'layers_pattern',
    }

    # 过滤配置
    original_keys = set(config.keys())
    cleaned_config = {k: v for k, v in config.items() if k in standard_params}
    removed_keys = original_keys - standard_params

    if removed_keys:
        print(f"\n移除的参数 ({len(removed_keys)}个):")
        for key in sorted(removed_keys):
            print(f"  - {key}: {config[key]}")

    # 保存清理后的配置
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_config, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 配置已清理: {config_file}")
    print(f"  保留参数: {len(cleaned_config)}个")
    print(f"  移除参数: {len(removed_keys)}个")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python clean_lora_config.py <lora_directory>")
        print("示例: python clean_lora_config.py lora_weights/experts/image_expert_qwen3")
        sys.exit(1)

    lora_dir = sys.argv[1]
    success = clean_lora_config(lora_dir)
    sys.exit(0 if success else 1)