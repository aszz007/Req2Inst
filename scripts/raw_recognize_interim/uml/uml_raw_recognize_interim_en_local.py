"""
UML用例图识别脚本（简化版）
功能：
  - 批量识别文件夹中的UML用例图
  - 支持Qwen2.5和Qwen3两个视觉模型版本
  - 输出识别结果到outputs/recognition_results/uml/目录
  - 直接调用VisionModel，无冗余代码

用法：
  python scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py --version qwen2.5
  python scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py --version qwen3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from models.vision_model import VisionModel
from config.settings import get_path_config


def extract_metadata_from_path(image_path: Path) -> dict:
    """
    从图片路径中提取领域和复杂度元数据

    路径格式: .../domain/domain_complexity_index.png
    例如: .../ecommerce/ecommerce_simple_001.png

    Args:
        image_path: 图片路径

    Returns:
        dict: 包含domain和complexity的字典
    """
    metadata = {
        'domain': 'unknown',
        'complexity': 'unknown'
    }

    try:
        # 从父目录名获取领域
        parent_dir = image_path.parent.name
        metadata['domain'] = parent_dir

        # 从文件名获取复杂度
        filename = image_path.stem  # 不含扩展名的文件名
        parts = filename.split('_')

        # 假设格式为: domain_complexity_index
        if len(parts) >= 2:
            complexity = parts[1]
            if complexity in ['simple', 'medium', 'complex']:
                metadata['complexity'] = complexity
    except Exception:
        # 如果提取失败，保持默认值
        pass

    return metadata


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量识别UML用例图')
    parser.add_argument(
        '--version',
        type=str,
        default='qwen2.5',
        choices=['qwen2.5', 'qwen3'],
        help='选择视觉模型版本（默认: qwen2.5）'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='输入图片文件夹路径（默认使用配置中的测试目录）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出JSON文件路径（默认输出到outputs/recognition_results/uml/）'
    )
    parser.add_argument(
        '--single',
        type=str,
        default=None,
        help='单张图片路径（用于快速测试）'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='启用流式输出模式（实时显示生成内容）'
    )
    return parser.parse_args()


def recognize_single_uml(image_path: str, version: str = 'qwen2.5', streaming: bool = False) -> Dict:
    """
    识别单张UML用例图

    Args:
        image_path: 图片路径
        version: 模型版本

    Returns:
        dict: 识别结果
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    print(f"\n{'='*80}")
    print(f"单图识别 - UML用例图")
    print(f"{'='*80}")
    print(f"模型版本: {version.upper()}")
    print(f"图片路径: {image_path}")
    print(f"{'='*80}\n")

    # 初始化模型
    print(f"[模型加载] 正在加载 {version.upper()} 视觉模型...")
    model = VisionModel(version=version)
    model_info = model.get_model_info()
    print(f"[模型信息] {model_info['model_name']}")
    print(f"[设备] {model_info['device']}\n")

    # 识别
    print(f"[识别中] 正在处理UML图...")
    result = model.recognize_uml(str(image_path), streaming=streaming)

    # 添加元数据
    result['image_path'] = str(image_path)
    result['image_name'] = image_path.name
    result['model_version'] = version

    # 提取并添加领域和复杂度信息
    metadata = extract_metadata_from_path(image_path)
    result['domain'] = metadata['domain']
    result['complexity'] = metadata['complexity']

    # 显示结果
    print(f"\n{'='*80}")
    print(f"识别结果")
    print(f"{'='*80}")
    if result.get('success', False):
        print(f"识别成功")

        # 尝试解析并显示简要信息
        try:
            desc = json.loads(result['description'])
            print(f"\n参与者数量: {len(desc.get('actors', []))}")
            print(f"用例数量: {len(desc.get('use_cases', []))}")
            print(f"关系数量: {len(desc.get('relationships', []))}")

            if desc.get('actors'):
                print(f"\n参与者: {', '.join(desc.get('actors', []))}")
            if desc.get('use_cases'):
                use_cases = [uc.get('name', '') for uc in desc.get('use_cases', [])]
                print(f"用例: {', '.join(use_cases[:3])}{'...' if len(use_cases) > 3 else ''}")
        except:
            print(f"\n描述: {result.get('description', '')[:200]}...")
    else:
        print(f"识别失败: {result.get('error', '未知错误')}")

    print(f"{'='*80}\n")

    return result


def batch_recognize_uml(
    image_folder: str,
    version: str = 'qwen2.5',
    output_file: str = None,
    streaming: bool = False
) -> List[Dict]:
    """
    批量识别文件夹中的所有UML用例图

    Args:
        image_folder: 图片文件夹路径
        version: 模型版本（'qwen2.5' 或 'qwen3'）
        output_file: 输出JSON文件路径（None则自动生成）

    Returns:
        list: 所有识别结果的列表
    """
    image_folder = Path(image_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {image_folder}")

    # 获取所有图片文件（去重）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = set()
    for ext in image_extensions:
        image_files.update(image_folder.glob(f"*{ext}"))
        image_files.update(image_folder.glob(f"*{ext.upper()}"))

    image_files = sorted(list(image_files))
    total_images = len(image_files)

    print(f"\n{'='*80}")
    print(f"批量识别UML用例图")
    print(f"{'='*80}")
    print(f"模型版本: {version.upper()}")
    print(f"图片文件夹: {image_folder}")
    print(f"找到图片数量: {total_images}")
    print(f"{'='*80}\n")

    if total_images == 0:
        print("[警告] 未找到任何图片文件")
        return []

    # 初始化模型
    print(f"[模型加载] 正在加载 {version.upper()} 视觉模型...")
    model = VisionModel(version=version)
    model_info = model.get_model_info()
    print(f"[模型信息] {model_info['model_name']}")
    print(f"[设备] {model_info['device']}\n")

    # 批量识别
    results = []
    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{total_images}] 处理: {image_path.name}")
        print("-" * 70)

        try:
            # 直接调用VisionModel的recognize_uml方法
            result = model.recognize_uml(str(image_path), streaming=streaming)

            # 添加元数据
            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            result['model_version'] = version

            # 提取并添加领域和复杂度信息
            metadata = extract_metadata_from_path(image_path)
            result['domain'] = metadata['domain']
            result['complexity'] = metadata['complexity']

            results.append(result)

            if result.get('success', False):
                success_count += 1
                print(f"✓ 识别成功")

                # 尝试解析并显示简要信息
                try:
                    desc = json.loads(result['description'])
                    print(f"  参与者数量: {len(desc.get('actors', []))}")
                    print(f"  用例数量: {len(desc.get('use_cases', []))}")
                    print(f"  关系数量: {len(desc.get('relationships', []))}")
                except:
                    pass
            else:
                fail_count += 1
                print(f"✗ 识别失败: {result.get('error', '未知错误')}")

        except Exception as e:
            fail_count += 1
            print(f"✗ 处理失败: {str(e)}")

            # 提取元数据
            metadata = extract_metadata_from_path(image_path)

            results.append({
                'image_path': str(image_path),
                'image_name': image_path.name,
                'model_version': version,
                'domain': metadata['domain'],
                'complexity': metadata['complexity'],
                'success': False,
                'error': str(e)
            })

    # 确定输出路径
    if output_file is None:
        # 使用配置中的输出目录
        path_cfg = get_path_config()
        output_dir = path_cfg.UML_RECOGNITION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"uml_recognition_{version}_{timestamp}.json"
    else:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print(f"\n{'='*80}")
    print(f"批量识别完成")
    print(f"{'='*80}")
    print(f"总图片数: {total_images}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"成功率: {success_count/total_images*100:.1f}%")
    print(f"结果已保存至: {output_file}")
    print(f"{'='*80}\n")

    return results


def main():
    """主函数"""
    args = parse_args()

    print("=" * 80)
    print(" " * 25 + f"UML用例图识别系统")
    print("=" * 80)
    print(f"模型版本: {args.version.upper()}")
    print(f"功能: 识别用例图中的元素和逻辑关系")
    print(f"输出: 英文JSON格式结果")
    print("=" * 80 + "\n")

    try:
        # 单图识别模式
        if args.single:
            result = recognize_single_uml(args.single, args.version, args.streaming)

            # 保存结果
            path_cfg = get_path_config()
            output_dir = path_cfg.UML_RECOGNITION_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"single_uml_{timestamp}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"结果已保存至: {output_file}")

        # 批量识别模式
        else:
            # 确定输入路径
            if args.input:
                image_folder = args.input
            else:
                # 使用配置中的默认测试目录
                path_cfg = get_path_config()
                image_folder = path_cfg.MDPI_UML_DIR
                print(f"[提示] 使用默认输入目录: {image_folder}")
                print(f"[提示] 可使用 --input 参数指定其他目录\n")

            # 批量识别
            results = batch_recognize_uml(
                image_folder=image_folder,
                version=args.version,
                output_file=args.output,
                streaming=args.streaming
            )

            # 展示部分结果示例
            if results and results[0].get('success', False):
                print("\n" + "="*80)
                print("结果示例（第一张图片）")
                print("="*80)
                first_result = results[0]
                # 只显示关键字段
                sample = {
                    'image_name': first_result.get('image_name'),
                    'model_version': first_result.get('model_version'),
                    'success': first_result.get('success'),
                    'description_preview': first_result.get('description', '')[:200] + "..."
                }
                print(json.dumps(sample, ensure_ascii=False, indent=2))
                print("="*80)

        print("\n所有识别任务完成")

    except Exception as e:
        print(f"\n程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# 用法示例:
# 使用 Qwen2.5-VL 模型批量识别
# python scripts/run_with_env.py --env uml_qwen2.5 --script scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py

# 使用 Qwen3-VL 模型批量识别
# python scripts/run_with_env.py --env uml_qwen3 --script scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py

# 使用 Qwen2.5 识别自定义文件夹
# python scripts/run_with_env.py --env uml_qwen2.5 --script scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py --input /path/to/your/images

# 使用 Qwen3 识别自定义文件夹
# python scripts/run_with_env.py --env uml_qwen3 --script scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py --input /path/to/your/images

# 使用 Qwen2.5 识别单张UML图
# python scripts/run_with_env.py --env uml_qwen2.5 --script scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py --single /path/to/single/uml.png

# 使用 Qwen3 识别单张UML图
# python scripts/run_with_env.py --env uml_qwen3 --script scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py --single /path/to/single/uml.png

# 自定义输出路径
# python scripts/run_with_env.py --env uml_qwen2.5 --script scripts/raw_recognize_interim/uml/uml_raw_recognize_interim_en_local.py --output /path/to/output.json