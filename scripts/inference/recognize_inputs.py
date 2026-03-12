"""
Unified Recognition Script for Inference
推理阶段的统一识别脚本

功能:
  - 识别inputs/目录中的原始图像和UML图
  - 支持单文件或批量处理
  - 自动判断图像类型（普通图像/UML图）
  - 输出JSON格式的识别结果到outputs/recognition_results/

环境要求: instruction_generator (统一Conda环境，transformers==4.57.0)
  运行前请先激活环境: conda activate instruction_generator

用法:
  # 识别单个文件
  python scripts/inference/recognize_inputs.py --input path/to/image.jpg --type image

  # 批量识别目录
  python scripts/inference/recognize_inputs.py --input path/to/dir --type uml

  # 指定视觉模型版本
  python scripts/inference/recognize_inputs.py --input path/to/image.jpg --type image --version qwen2.5

作者: Inference System
日期: 2025-02-06
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.vision_model import VisionModel
from config.settings import get_path_config
from src.utils.logger import get_logger

logger = get_logger('inference.recognize_inputs')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified recognition script for inference stage'
    )

    parser.add_argument(
        '--version',
        type=str,
        default='qwen3',
        choices=['qwen2.5', 'qwen3'],
        help='Vision model version (default: qwen3, uses Qwen3-VL-8B)'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file or directory path'
    )

    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['image', 'uml'],
        help='Recognition type: image or uml'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (optional, auto-generated if not specified)'
    )

    return parser.parse_args()


def recognize_single_file(
        file_path: str,
        rec_type: str,
        version: str
) -> Dict:
    """
    Recognize a single image file

    Args:
        file_path: Path to image file
        rec_type: Recognition type ('image' or 'uml')
        version: Model version

    Returns:
        dict: Recognition result
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Recognizing single file: {file_path.name}")
    logger.info(f"Type: {rec_type}, Model: {version}")

    # Initialize model
    model = VisionModel(version=version)

    # Recognize based on type
    try:
        if rec_type == 'image':
            result = model.recognize_image(str(file_path))
        else:  # uml
            result = model.recognize_uml(str(file_path))

        # Add metadata
        result['file_path'] = str(file_path)
        result['file_name'] = file_path.name
        result['recognition_type'] = rec_type
        result['model_version'] = version
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Recognition completed successfully")
        return result

    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        return {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'recognition_type': rec_type,
            'model_version': version,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def recognize_directory(
        dir_path: str,
        rec_type: str,
        version: str
) -> List[Dict]:
    """
    Recognize all images in a directory

    Args:
        dir_path: Directory path
        rec_type: Recognition type
        version: Model version

    Returns:
        list: List of recognition results
    """
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))

    image_files = sorted(list(set(image_files)))
    total = len(image_files)

    logger.info(f"Found {total} image files in {dir_path}")

    if total == 0:
        logger.warning("No image files found")
        return []

    # Initialize model once
    model = VisionModel(version=version)

    # Batch recognition
    results = []
    success_count = 0

    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"Processing [{idx}/{total}]: {image_path.name}")

        try:
            if rec_type == 'image':
                result = model.recognize_image(str(image_path))
            else:  # uml
                result = model.recognize_uml(str(image_path))

            # Add metadata
            result['file_path'] = str(image_path)
            result['file_name'] = image_path.name
            result['recognition_type'] = rec_type
            result['model_version'] = version
            result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            results.append(result)

            # Check success
            if rec_type == 'image':
                if result.get('recognition_status') == 'success':
                    success_count += 1
            else:  # uml
                if result.get('success', False):
                    success_count += 1

            logger.info(f"Recognition completed")

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            results.append({
                'file_path': str(image_path),
                'file_name': image_path.name,
                'recognition_type': rec_type,
                'model_version': version,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    logger.info(f"Batch recognition completed: {success_count}/{total} successful")
    return results


def main():
    """Main function"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Unified Recognition Script for Inference")
    logger.info("=" * 80)
    logger.info(f"Model Version: {args.version.upper()}")
    logger.info(f"Recognition Type: {args.type.upper()}")
    logger.info(f"Input: {args.input}")
    logger.info("=" * 80)

    try:
        input_path = Path(args.input)

        # Determine if input is file or directory
        if input_path.is_file():
            logger.info("Processing single file...")
            result = recognize_single_file(
                file_path=str(input_path),
                rec_type=args.type,
                version=args.version
            )
            results = [result]

        elif input_path.is_dir():
            logger.info("Processing directory...")
            results = recognize_directory(
                dir_path=str(input_path),
                rec_type=args.type,
                version=args.version
            )
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

        # Determine output path
        if args.output:
            output_file = Path(args.output)
        else:
            # Auto-generate output path: outputs/recognition_results/{type}/
            path_cfg = get_path_config()
            output_dir = path_cfg.PROJECT_ROOT / 'outputs' / 'recognition_results' / args.type

            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{args.type}_recognition_{args.version}_{timestamp}.json"

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("=" * 80)
        logger.info(f"Recognition completed")
        logger.info(f"Total processed: {len(results)}")
        logger.info(f"Results saved to: {output_file}")
        logger.info("=" * 80)

        # Print output path to stdout for parent script to capture
        print(f"OUTPUT_FILE:{output_file}")

    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()