"""
UML用例图数据集构建脚本 - 从图片和JSON识别结果构建CSV数据集
功能：
1. 遍历图片目录获取所有图片名（去除扩展名）
2. 读取JSON文件构建 图片名 -> 识别信息 映射
3. 验证映射完整性（记录缺失项）
4. 生成CSV：Header | Description | Example
5. 处理图片和JSON不匹配的情况，不终止程序
6. 单独记录错误日志和处理统计
7. Description列只保留UML描述内容（actors, use_cases, system_boundary, relationships, overall_description）
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime


class DatasetBuilder:
    """数据集构建器 - 将图片 + JSON识别结果转换为CSV数据集"""

    def __init__(self):
        """初始化构建器"""
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        self.errors = []
        self.stats = {
            'total_images': 0,
            'total_json_entries': 0,
            'matched': 0,
            'images_without_json': 0,
            'json_without_images': 0,
            'failed_recognitions': 0
        }

    def get_all_images(self, image_folder: Path) -> Set[str]:
        """
        获取文件夹中所有图片名（不带扩展名）

        Args:
            image_folder: 图片文件夹路径

        Returns:
            set: 图片名集合（不带扩展名）
        """
        if not image_folder.exists():
            raise FileNotFoundError(f"图片文件夹不存在: {image_folder}")

        image_names = set()

        for ext in self.image_extensions:
            # 处理小写和大写扩展名
            for file_path in image_folder.glob(f"*{ext}"):
                image_names.add(file_path.stem)
            for file_path in image_folder.glob(f"*{ext.upper()}"):
                image_names.add(file_path.stem)

        self.stats['total_images'] = len(image_names)
        return image_names

    def load_json_results(self, json_file: Path) -> Dict[str, Dict]:
        """
        加载JSON识别结果并构建映射

        Args:
            json_file: JSON文件路径

        Returns:
            dict: 图片名到识别信息的映射
        """
        if not json_file.exists():
            raise FileNotFoundError(f"JSON文件不存在: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 构建映射：图片名（不带扩展名）-> 识别信息
        json_mapping = {}

        for entry in results:
            image_name = entry.get('image_name', '')
            if not image_name:
                self.errors.append({
                    'type': 'INVALID_JSON_ENTRY',
                    'message': 'JSON条目缺少image_name',
                    'entry': entry
                })
                continue

            # 从图片名中移除扩展名
            name_without_ext = Path(image_name).stem

            # 检查识别是否成功
            if not entry.get('success', False):
                self.stats['failed_recognitions'] += 1
                self.errors.append({
                    'type': 'FAILED_RECOGNITION',
                    'image_name': name_without_ext,
                    'error': entry.get('error', '未知错误')
                })

            # 存入映射（即使识别失败也保留，但标记）
            json_mapping[name_without_ext] = entry

        self.stats['total_json_entries'] = len(json_mapping)
        return json_mapping

    def validate_mapping(self, image_names: Set[str], json_mapping: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
        """
        验证图片-JSON映射的完整性

        Args:
            image_names: 图片名集合
            json_mapping: JSON映射

        Returns:
            tuple: (图片缺失JSON列表, JSON缺失图片列表)
        """
        # 存在图片但没有JSON的
        images_without_json = list(image_names - json_mapping.keys())

        # 存在JSON但没有对应图片的
        json_without_images = list(json_mapping.keys() - image_names)

        self.stats['images_without_json'] = len(images_without_json)
        self.stats['json_without_images'] = len(json_without_images)
        self.stats['matched'] = len(image_names & json_mapping.keys())

        # 记录错误
        for img_name in images_without_json:
            self.errors.append({
                'type': 'IMAGE_WITHOUT_JSON',
                'image_name': img_name,
                'message': '图片存在但没有JSON识别结果'
            })

        for json_name in json_without_images:
            self.errors.append({
                'type': 'JSON_WITHOUT_IMAGE',
                'image_name': json_name,
                'message': 'JSON结果存在但没有对应图片'
            })

        return images_without_json, json_without_images

    def prepare_json_string(self, recognition_info: Dict) -> str:
        """
        准备用于CSV的JSON字符串（移除元数据，保留UML识别内容）
        只保留: actors, use_cases, system_boundary, relationships, overall_description
        移除: image_path, image_name, success, timestamp

        Args:
            recognition_info: 识别信息字典

        Returns:
            str: 格式化的JSON字符串
        """
        # 检查识别是否成功
        if not recognition_info.get('success', False):
            # 识别失败，返回错误信息
            return json.dumps({
                'recognition_status': 'failed',
                'error': recognition_info.get('error', 'Unknown error')
            }, ensure_ascii=False)

        # 识别成功，获取description字段内容
        try:
            description_data = recognition_info.get('description', {})

            # 判断description是字符串还是字典
            if isinstance(description_data, str):
                # 如果是字符串，需要解析为字典
                uml_content = json.loads(description_data)
            elif isinstance(description_data, dict):
                # 如果已经是字典，直接使用
                uml_content = description_data
            else:
                # 其他类型，使用空字典
                uml_content = {}

            # 创建只包含UML识别内容的干净副本
            clean_info = {
                'actors': uml_content.get('actors', []),
                'use_cases': uml_content.get('use_cases', []),
                'system_boundary': uml_content.get('system_boundary', {}),
                'relationships': uml_content.get('relationships', []),
                'overall_description': uml_content.get('overall_description', '')
            }

            # 转换为紧凑的JSON字符串
            return json.dumps(clean_info, ensure_ascii=False)

        except json.JSONDecodeError as e:
            # JSON解析失败，返回错误信息
            return json.dumps({
                'recognition_status': 'parse_error',
                'error': f'Failed to parse description JSON: {str(e)}',
                'raw_description': str(recognition_info.get('description', ''))[:200]
            }, ensure_ascii=False)
        except Exception as e:
            # 其他异常处理
            return json.dumps({
                'recognition_status': 'error',
                'error': f'Unexpected error: {str(e)}',
                'raw_description': str(recognition_info.get('description', ''))[:200]
            }, ensure_ascii=False)

    def build_csv_dataset(
        self,
        image_folder: str,
        json_file: str,
        output_csv: str = None,
        error_log: str = None,
        include_failed: bool = True
    ) -> str:
        """
        从图片和JSON结果构建CSV数据集

        Args:
            image_folder: 图片文件夹路径
            json_file: JSON识别结果文件
            output_csv: 输出CSV文件路径（可选）
            error_log: 错误日志文件路径（可选）
            include_failed: 是否包含识别失败的记录

        Returns:
            str: 输出CSV文件路径
        """
        print("="*80)
        print(" "*25 + "UML数据集构建器")
        print("="*80)

        # 转换为Path对象
        image_folder = Path(image_folder)
        json_file = Path(json_file)

        # 步骤1：获取所有图片
        print("\n[步骤 1/5] 扫描图片目录...")
        print(f"图片文件夹: {image_folder}")
        image_names = self.get_all_images(image_folder)
        print(f"找到 {len(image_names)} 张不重复的图片")

        # 步骤2：加载JSON结果
        print("\n[步骤 2/5] 加载JSON识别结果...")
        print(f"JSON文件: {json_file}")
        json_mapping = self.load_json_results(json_file)
        print(f"加载了 {len(json_mapping)} 条JSON记录")
        if self.stats['failed_recognitions'] > 0:
            print(f"  ⚠ 其中 {self.stats['failed_recognitions']} 条识别失败")

        # 步骤3：验证映射
        print("\n[步骤 3/5] 验证图片-JSON映射...")
        images_without_json, json_without_images = self.validate_mapping(image_names, json_mapping)

        print(f"匹配的配对: {self.stats['matched']}")
        if self.stats['images_without_json'] > 0:
            print(f"⚠ 图片缺少JSON: {self.stats['images_without_json']}")
        if self.stats['json_without_images'] > 0:
            print(f"⚠ JSON缺少图片: {self.stats['json_without_images']}")

        # 步骤4：构建CSV
        print("\n[步骤 4/5] 构建CSV数据集...")

        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = f"uml_dataset_{timestamp}.csv"

        output_csv = Path(output_csv)

        # 准备数据行
        rows = []
        for image_name in sorted(image_names):
            if image_name in json_mapping:
                recognition_info = json_mapping[image_name]

                # 如果不包含失败的记录，则跳过
                if not include_failed and not recognition_info.get('success', False):
                    continue

                # 准备行数据
                header = image_name  # 图片名（去掉扩展名）
                description = self.prepare_json_string(recognition_info)  # 只保留UML内容的JSON
                example = ""  # 空白，待后续填充

                rows.append({
                    'Header': header,
                    'Description': description,
                    'Example': example
                })
            else:
                # 图片存在但没有JSON - 添加占位符
                rows.append({
                    'Header': image_name,
                    'Description': json.dumps({
                        'recognition_status': 'missing',
                        'error': 'No JSON recognition result found'
                    }, ensure_ascii=False),
                    'Example': ""
                })

        # 写入CSV（使用utf-8-sig编码，添加BOM以避免Windows Excel乱码）
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            fieldnames = ['Header', 'Description', 'Example']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV数据集已创建: {output_csv}")
        print(f"总行数: {len(rows)}")

        # 步骤5：保存错误日志
        print("\n[步骤 5/5] 保存错误日志...")

        if error_log is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_log = f"dataset_errors_{timestamp}.json"

        error_log = Path(error_log)

        with open(error_log, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'statistics': self.stats,
                'errors': self.errors
            }, f, ensure_ascii=False, indent=2)

        print(f"错误日志已保存: {error_log}")
        print(f"总错误数: {len(self.errors)}")

        # 打印汇总信息
        print("\n" + "="*80)
        print(" "*30 + "汇总")
        print("="*80)
        print(f"总图片数: {self.stats['total_images']}")
        print(f"总JSON条目: {self.stats['total_json_entries']}")
        print(f"匹配的配对: {self.stats['matched']}")
        print(f"创建的CSV行数: {len(rows)}")
        print("-"*80)
        print(f"图片缺少JSON: {self.stats['images_without_json']}")
        print(f"JSON缺少图片: {self.stats['json_without_images']}")
        print(f"识别失败: {self.stats['failed_recognitions']}")
        print(f"记录的错误数: {len(self.errors)}")
        print("="*80 + "\n")

        return str(output_csv)


def main():
    """主函数 - 构建UML数据集"""

    # ==================== 配置区域 ====================
    # 请根据实际情况修改以下路径

    # 图片文件夹路径（与uml_web_recognizer.py中的IMAGE_FOLDER相同）
    IMAGE_FOLDER = r"D:\MyPyProject\crowdsourcing_instruction_generator\data\raw\uml\roboflow_uml"

    # JSON识别结果文件（由 uml_web_recognizer.py 生成）
    JSON_FILE = r"D:\MyPyProject\crowdsourcing_instruction_generator\outputs\recognition_results\uml\uml_recognition_qwen3_20260129_174404_fixed.json"

    # 输出CSV文件路径
    OUTPUT_CSV = r"D:\MyPyProject\crowdsourcing_instruction_generator\outputs\recognition_results\uml\uml_interim.csv"

    # 错误日志文件路径（可选）
    ERROR_LOG = "uml_dataset_errors.json"

    # 是否在CSV中包含识别失败的记录（True/False）
    INCLUDE_FAILED = True

    # ==================== 配置区域结束 ====================

    print("="*80)
    print(" "*20 + "UML用例图数据集构建工具")
    print("="*80)
    print(f"用途: 将图片 + JSON结果转换为CSV数据集")
    print(f"输出格式: Header | Description | Example")
    print(f"Description内容: 仅包含UML识别信息（actors, use_cases, etc.）")
    print(f"已过滤字段: image_path, image_name, success, timestamp")
    print("="*80 + "\n")

    try:
        # 创建构建器
        builder = DatasetBuilder()

        # 构建数据集
        output_path = builder.build_csv_dataset(
            image_folder=IMAGE_FOLDER,
            json_file=JSON_FILE,
            output_csv=OUTPUT_CSV,
            error_log=ERROR_LOG,
            include_failed=INCLUDE_FAILED
        )

        print(f"✓ 数据集构建完成！")
        print(f"✓ 输出文件: {output_path}")
        print(f"\n后续步骤:")
        print(f"1. 查看错误日志以检查任何问题")
        print(f"2. 根据需要填充 'Example' 列")
        print(f"3. 使用数据集进行模型训练或其他应用")

    except Exception as e:
        print(f"\n✗ 数据集构建失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 运行方式: python scripts/json_build_interim/uml/json_build_uml_interim_qwen3.py