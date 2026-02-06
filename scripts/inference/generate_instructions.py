"""
Automated Instruction Generation Script
自动化指令生成脚本

功能:
  - 扫描inputs/目录下的所有输入文件
  - 自动识别文件类型（文本/图像/UML）
  - 对于原始图像/UML：自动调用识别脚本（切换到vision环境）
  - 对于文本/JSON：直接读取
  - 使用MoE系统生成众包指令
  - 保存结果到outputs/generated_instructions/

环境要求: qwen_text (主脚本在此环境运行，会自动切换环境调用识别脚本)

用法:
  # 处理inputs/目录下的所有文件
  python scripts/run_with_env.py --env text --script scripts/inference/generate_instructions.py

  # 处理指定目录
  python scripts/run_with_env.py --env text --script scripts/inference/generate_instructions.py --input-dir path/to/inputs

  # 指定输出格式
  python scripts/run_with_env.py --env text --script scripts/inference/generate_instructions.py --output-format json

  # 指定vision模型版本
  python scripts/run_with_env.py --env text --script scripts/inference/generate_instructions.py --vision-version qwen3

作者: Inference System
日期: 2025-02-06
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.instruction_generation import InstructionGenerator
from config.settings import get_path_config
from src.utils.logger import get_logger

logger = get_logger('inference.generate_instructions')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Automated instruction generation script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in inputs/ directory
  python scripts/run_with_env.py --env text --script scripts/inference/generate_instructions.py

  # Process specific directory
  python scripts/run_with_env.py --env text --script scripts/inference/generate_instructions.py --input-dir /path/to/inputs

  # Specify output format
  python scripts/run_with_env.py --env text --script scripts/inference/generate_instructions.py --output-format json

  # Use Qwen3-VL for recognition
  python scripts/run_with_env.py --env text --script scripts/inference/generate_instructions.py --vision-version qwen3
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory path (default: inputs/)'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        default='json',
        choices=['text', 'json', 'markdown'],
        help='Output format (default: json)'
    )

    parser.add_argument(
        '--vision-version',
        type=str,
        default='qwen2.5',
        choices=['qwen2.5', 'qwen3'],
        help='Vision model version for image/UML recognition (default: qwen2.5)'
    )

    parser.add_argument(
        '--expert-variant',
        type=str,
        default=None,
        help='Specify expert variant for comparison experiments (optional)'
    )

    parser.add_argument(
        '--no-recognition',
        action='store_true',
        help='Skip automatic recognition, only process existing JSON files'
    )

    return parser.parse_args()


def scan_input_directory(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan input directory and categorize files by type

    Args:
        input_dir: Input directory path

    Returns:
        dict: Categorized file lists
            - 'text': .txt files
            - 'json': .json files (already recognized)
            - 'image': image files (.jpg, .png, etc.)
            - 'uml': UML diagram files (in uml/ subdirectory)
    """
    categorized = {
        'text': [],
        'json': [],
        'image': [],
        'uml': []
    }

    # Text files
    text_dir = input_dir / 'text'
    if text_dir.exists():
        categorized['text'] = list(text_dir.glob('*.txt'))

    # JSON files (already recognized)
    for subdir in ['text', 'image', 'uml']:
        json_dir = input_dir / subdir
        if json_dir.exists():
            categorized['json'].extend(json_dir.glob('*.json'))

    # Image files
    image_dir = input_dir / 'image'
    if image_dir.exists():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        for ext in image_extensions:
            categorized['image'].extend(image_dir.glob(f'*{ext}'))
            categorized['image'].extend(image_dir.glob(f'*{ext.upper()}'))
        categorized['image'] = list(set(categorized['image']))  # Remove duplicates

    # UML files
    uml_dir = input_dir / 'uml'
    if uml_dir.exists():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        for ext in image_extensions:
            categorized['uml'].extend(uml_dir.glob(f'*{ext}'))
            categorized['uml'].extend(uml_dir.glob(f'*{ext.upper()}'))
        categorized['uml'] = list(set(categorized['uml']))

    return categorized


def recognize_images(
        image_files: List[Path],
        rec_type: str,
        vision_version: str
) -> Optional[Path]:
    """
    Recognize images using vision model (automatically switches environment)

    Args:
        image_files: List of image file paths
        rec_type: Recognition type ('image' or 'uml')
        vision_version: Vision model version

    Returns:
        Path: Path to recognition results JSON file, or None if failed
    """
    if not image_files:
        return None

    logger.info(f"Recognizing {len(image_files)} {rec_type} files using {vision_version}...")

    # Determine environment
    if vision_version == 'qwen3':
        env_name = f'{rec_type}_qwen3'
    else:
        env_name = f'{rec_type}_qwen2.5'

    # Create temporary directory for recognition
    temp_dir = Path('/tmp/inference_recognition')
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save file list
    file_list = temp_dir / f'{rec_type}_files.txt'
    with open(file_list, 'w') as f:
        for img_file in image_files:
            f.write(f"{img_file}\n")

    # Prepare recognition script path
    recognize_script = project_root / 'scripts' / 'inference' / 'recognize_inputs.py'
    run_env_script = project_root / 'scripts' / 'run_with_env.py'

    # Determine input: if single file, pass file path; if multiple, pass parent directory
    if len(image_files) == 1:
        input_path = str(image_files[0])
    else:
        # All files should be in the same directory
        input_path = str(image_files[0].parent)

    # Build command
    cmd = [
        sys.executable,
        str(run_env_script),
        '--env', env_name,
        '--script', str(recognize_script),
        '--input', input_path,
        '--type', rec_type
    ]

    logger.info(f"Executing recognition command: {' '.join(cmd)}")

    try:
        # Execute recognition script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )

        # Parse output to get JSON file path
        output = result.stdout
        match = re.search(r'OUTPUT_FILE:(.+)', output)
        if match:
            json_file = Path(match.group(1).strip())
            logger.info(f"Recognition completed, results saved to: {json_file}")
            return json_file
        else:
            logger.error("Failed to parse recognition output")
            logger.error(f"stdout: {output}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Recognition failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Recognition execution error: {e}")
        return None


def load_text_input(file_path: Path) -> Dict:
    """
    Load text input file

    Args:
        file_path: Path to text file

    Returns:
        dict: Formatted input data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    return {
        'type': 'text',
        'content': content,
        'source_file': str(file_path),
        'source_type': 'text_file'
    }


def load_json_input(file_path: Path) -> Dict:
    """
    Load JSON input file (recognized results)

    Args:
        file_path: Path to JSON file

    Returns:
        dict: Formatted input data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Determine type based on file location or content
    parent_dir = file_path.parent.name
    if parent_dir == 'image':
        input_type = 'image'
        content = data.get('description', '')
    elif parent_dir == 'uml':
        input_type = 'uml'
        content = data.get('description', '')
    else:
        # Try to infer from content
        if 'actors' in str(data) or 'use_cases' in str(data):
            input_type = 'uml'
            content = data.get('description', '')
        elif 'description' in data:
            input_type = 'image'
            content = data.get('description', '')
        else:
            input_type = 'general'
            content = json.dumps(data)

    return {
        'type': input_type,
        'content': content,
        'source_file': str(file_path),
        'source_type': 'json_file'
    }


def load_recognition_results(json_file: Path, rec_type: str) -> List[Dict]:
    """
    Load recognition results and format as input data

    Args:
        json_file: Path to recognition results JSON
        rec_type: Recognition type ('image' or 'uml')

    Returns:
        list: List of formatted input data
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Handle single result vs list
    if isinstance(results, dict):
        results = [results]

    inputs = []
    for result in results:
        # Check if recognition was successful
        if rec_type == 'image':
            success = result.get('recognition_status') == 'success'
        else:  # uml
            success = result.get('success', False)

        if not success:
            logger.warning(f"Skipping failed recognition: {result.get('file_name', 'unknown')}")
            continue

        # Extract description
        description = result.get('description', '')
        if not description:
            logger.warning(f"No description found in: {result.get('file_name', 'unknown')}")
            continue

        inputs.append({
            'type': rec_type,
            'content': description,
            'source_file': result.get('file_path', ''),
            'source_type': f'{rec_type}_recognition'
        })

    return inputs


def main():
    """Main function"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Automated Instruction Generation System")
    logger.info("=" * 80)
    logger.info(f"Output Format: {args.output_format}")
    logger.info(f"Vision Model Version: {args.vision_version}")
    if args.expert_variant:
        logger.info(f"Expert Variant: {args.expert_variant}")
    logger.info("=" * 80)

    # Determine input directory
    path_cfg = get_path_config()
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = path_cfg.INPUTS_DIR

    logger.info(f"Input Directory: {input_dir}")

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        logger.info("Creating directory structure...")
        (input_dir / 'text').mkdir(parents=True, exist_ok=True)
        (input_dir / 'image').mkdir(parents=True, exist_ok=True)
        (input_dir / 'uml').mkdir(parents=True, exist_ok=True)
        logger.info(f"Please place your input files in: {input_dir}")
        return

    # Scan input directory
    logger.info("\nScanning input directory...")
    categorized_files = scan_input_directory(input_dir)

    total_files = sum(len(files) for files in categorized_files.values())
    logger.info(f"Found {total_files} files:")
    for category, files in categorized_files.items():
        if files:
            logger.info(f"  - {category}: {len(files)} files")

    if total_files == 0:
        logger.warning("No input files found!")
        logger.info(f"Please place files in subdirectories under: {input_dir}")
        logger.info("  - text/: .txt files (text requirements)")
        logger.info("  - image/: .jpg/.png files (images to annotate)")
        logger.info("  - uml/: .jpg/.png files (UML diagrams)")
        return

    # Prepare inputs
    all_inputs = []

    # 1. Load text files
    for txt_file in categorized_files['text']:
        logger.info(f"Loading text file: {txt_file.name}")
        all_inputs.append(load_text_input(txt_file))

    # 2. Load JSON files (already recognized)
    for json_file in categorized_files['json']:
        logger.info(f"Loading JSON file: {json_file.name}")
        all_inputs.append(load_json_input(json_file))

    # 3. Recognize and load image files
    if categorized_files['image'] and not args.no_recognition:
        logger.info(f"\nRecognizing {len(categorized_files['image'])} image files...")
        json_file = recognize_images(
            categorized_files['image'],
            'image',
            args.vision_version
        )
        if json_file and json_file.exists():
            recognized_inputs = load_recognition_results(json_file, 'image')
            all_inputs.extend(recognized_inputs)
            logger.info(f"Loaded {len(recognized_inputs)} recognized images")
        else:
            logger.error("Image recognition failed")

    # 4. Recognize and load UML files
    if categorized_files['uml'] and not args.no_recognition:
        logger.info(f"\nRecognizing {len(categorized_files['uml'])} UML files...")
        json_file = recognize_images(
            categorized_files['uml'],
            'uml',
            args.vision_version
        )
        if json_file and json_file.exists():
            recognized_inputs = load_recognition_results(json_file, 'uml')
            all_inputs.extend(recognized_inputs)
            logger.info(f"Loaded {len(recognized_inputs)} recognized UML diagrams")
        else:
            logger.error("UML recognition failed")

    if not all_inputs:
        logger.error("No valid inputs to process!")
        return

    logger.info(f"\nTotal inputs to process: {len(all_inputs)}")

    # Initialize instruction generator
    logger.info("\nInitializing instruction generator...")
    try:
        generator = InstructionGenerator()
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate instructions
    logger.info("\n" + "=" * 80)
    logger.info("Generating Instructions")
    logger.info("=" * 80)

    results = []
    success_count = 0

    for idx, input_data in enumerate(all_inputs, 1):
        logger.info(f"\n[{idx}/{len(all_inputs)}] Processing: {Path(input_data['source_file']).name}")
        logger.info(f"  Type: {input_data['type']}")
        logger.info(f"  Source: {input_data['source_type']}")

        try:
            result = generator.generate(
                input_data=input_data,
                output_format='json',
                expert_variant=args.expert_variant
            )

            # Add source information
            result['source_file'] = input_data['source_file']
            result['source_type'] = input_data['source_type']
            result['input_type'] = input_data['type']

            results.append(result)

            if result.get('instruction'):
                success_count += 1
                logger.info(f"  Status: Success")
                logger.info(f"  Expert: {result['expert_used']}")
            else:
                logger.warning(f"  Status: Failed - No instruction generated")

        except Exception as e:
            logger.error(f"  Status: Failed - {e}")
            results.append({
                'source_file': input_data['source_file'],
                'source_type': input_data['source_type'],
                'input_type': input_data['type'],
                'instruction': '',
                'expert_used': 'none',
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("Saving Results")
    logger.info("=" * 80)

    output_dir = path_cfg.GENERATED_INSTRUCTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_format == 'json':
        output_file = output_dir / f'instructions_{timestamp}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    elif args.output_format == 'markdown':
        output_file = output_dir / f'instructions_{timestamp}.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Generated Instructions\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total: {len(results)}\n\n")
            f.write("---\n\n")

            for idx, result in enumerate(results, 1):
                f.write(f"## Instruction {idx}\n\n")
                f.write(f"**Source**: {Path(result['source_file']).name}\n\n")
                f.write(f"**Type**: {result['input_type']}\n\n")
                f.write(f"**Expert**: {result['expert_used']}\n\n")
                f.write(f"**Instruction**:\n\n{result['instruction']}\n\n")
                f.write("---\n\n")

    else:  # text
        output_file = output_dir / f'instructions_{timestamp}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, result in enumerate(results, 1):
                f.write(f"{'=' * 80}\n")
                f.write(f"Instruction {idx}\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"Source: {Path(result['source_file']).name}\n")
                f.write(f"Type: {result['input_type']}\n")
                f.write(f"Expert: {result['expert_used']}\n")
                f.write(f"\n{result['instruction']}\n\n")

    # Save detailed JSON anyway for reference
    json_file = output_dir / f'instructions_{timestamp}_detailed.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    logger.info(f"\nResults saved to:")
    logger.info(f"  - Main: {output_file}")
    logger.info(f"  - Detailed JSON: {json_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Generation Summary")
    logger.info("=" * 80)
    logger.info(f"Total Processed: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(results) - success_count}")
    logger.info(f"Success Rate: {success_count / len(results) * 100:.1f}%")

    # Expert usage statistics
    stats = generator.get_statistics()
    if stats.get('total_routings', 0) > 0:
        logger.info("\nExpert Usage:")
        for expert, count in stats.get('expert_usage_count', {}).items():
            percentage = stats['expert_usage_percentage'][expert]
            logger.info(f"  - {expert}: {count} ({percentage:.1f}%)")

    logger.info("=" * 80)
    logger.info("Instruction generation completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()