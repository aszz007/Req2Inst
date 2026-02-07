"""
数据加载器 - 修复版
负责加载和处理三类数据集(文本、图像、UML)
支持合理的数据集划分

修复内容：
1. 优化编码检测顺序（UTF-8系列优先）
2. 强化列名规范化（大小写不敏感、去除BOM）
3. 完全静默异常处理（避免乱码输出）
4. 智能列名映射机制
5. Windows换行符兼容

作者：Data Loader System
日期：2025-01-29（修复版）
"""

import os
import json
import pandas as pd
import random
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch

# 禁用所有警告，避免乱码输出
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

from config.settings import get_path_config, get_training_config
from src.utils.logger import get_logger
from src.utils.file_utils import load_json

# 导入Prompt模板
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from models.prompt_templates.general_template import GeneralInstructionTemplate

logger = get_logger('training.data_loader')


def normalize_column_name(col_name: str) -> str:
    """
    强力规范化列名

    处理：
    - 去除BOM标记（UTF-8/UTF-16）
    - 去除所有空白字符
    - 转换为小写
    - 去除不可见字符

    Args:
        col_name: 原始列名

    Returns:
        规范化后的列名
    """
    # 去除BOM标记
    col_name = col_name.replace('\ufeff', '').replace('\ufffe', '')
    # 去除空白字符
    col_name = col_name.strip()
    # 转换为小写（大小写不敏感）
    col_name = col_name.lower()
    # 去除其他不可见字符
    col_name = ''.join(c for c in col_name if c.isprintable() or c.isspace())
    col_name = col_name.strip()

    return col_name


def detect_csv_encoding(filepath: Path) -> str:
    """
    智能检测CSV文件编码 - 优化版

    优先级调整：
    1. UTF-8系列（因为生成脚本使用utf-8-sig）
    2. GBK系列（Windows中文）
    3. UTF-16系列（Excel导出）
    4. Latin-1（兜底）

    Args:
        filepath: CSV文件路径

    Returns:
        str: 检测到的编码名称
    """
    # 调整编码优先级
    encodings = [
        'utf-8-sig',       # 带BOM的UTF-8（你的生成脚本用的）
        'utf-8',           # 标准UTF-8
        'gbk',             # Windows简体中文
        'gb2312',          # 简体中文
        'gb18030',         # 扩展GBK
        'utf-16',          # UTF-16（自动检测LE/BE）
        'utf-16-le',       # UTF-16 Little Endian
        'utf-16-be',       # UTF-16 Big Endian
        'latin-1'          # 兜底编码
    ]

    for encoding in encodings:
        try:
            # 尝试读取前100行
            pd.read_csv(filepath, encoding=encoding, nrows=100)
            return encoding
        except:
            # 静默跳过，不打印任何信息（避免乱码）
            continue

    # 如果所有编码都失败，返回兜底编码
    return 'latin-1'


class InstructionDataset(Dataset):
    """指令生成数据集基类"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        """
        初始化数据集

        Args:
            data: 数据列表,每项包含input和output
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 使用带prompt的完整输入
        # input_with_prompt包含完整的Qwen对话格式prompt
        input_text = item.get('input_with_prompt', item['input'])
        output_text = item['output']

        # 组合输入输出
        # 注意:input_text已经包含了完整的prompt格式,直接拼接output即可
        full_text = f"{input_text}{output_text}"

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()

        # Labels与input_ids相同(用于因果语言建模)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    强力清理DataFrame列名

    功能：
    - 去除BOM标记
    - 去除空白字符
    - 规范化列名
    - 创建列名映射

    Args:
        df: 原始DataFrame

    Returns:
        清理后的DataFrame
    """
    # 创建列名映射（原始 -> 规范化）
    column_mapping = {}
    for col in df.columns:
        normalized = normalize_column_name(col)
        column_mapping[col] = normalized

    # 重命名列
    df = df.rename(columns=column_mapping)

    return df


def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    智能查找列名（大小写不敏感）

    Args:
        df: DataFrame
        possible_names: 可能的列名列表（小写）

    Returns:
        找到的列名，如果没找到返回None
    """
    df_columns_lower = {col.lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]

    return None


class TextDatasetLoader:
    """文本数据集加载器 - 优化版"""

    def __init__(self):
        """初始化加载器"""
        path_cfg = get_path_config()
        self.dataset_dir = path_cfg.TEXT_DATASET_DIR
        logger.info(f"初始化TextDatasetLoader, 路径: {self.dataset_dir}")

    def load_csv_files(self) -> List[Dict]:
        """
        加载所有CSV文件

        Returns:
            数据列表,每项包含input(Low_Requirements)和output(Instruction)
        """
        all_data = []
        csv_files = list(Path(self.dataset_dir).glob("*.csv"))

        logger.info(f"找到{len(csv_files)}个CSV文件")

        for csv_file in csv_files:
            try:
                # 智能检测编码
                encoding = detect_csv_encoding(csv_file)
                df = pd.read_csv(csv_file, encoding=encoding)

                # 强力清理列名
                df = clean_dataframe_columns(df)

                logger.info(f"使用编码 '{encoding}' 读取: {csv_file.name}")

            except Exception:
                # 完全静默失败，不打印任何信息（避免乱码）
                logger.error(f"加载失败: {csv_file.name}")
                continue

            # 智能查找必要列（支持多种变体）
            low_req_col = find_column(df, ['low_requirements', 'lowrequirements', 'low requirements'])
            instruction_col = find_column(df, ['instruction', 'instructions'])

            if not low_req_col or not instruction_col:
                logger.warning(f"跳过文件(缺少必要列): {csv_file.name}")
                logger.warning(f"  实际列名: {list(df.columns)}")
                continue

            # 提取数据
            for _, row in df.iterrows():
                try:
                    low_req = str(row[low_req_col]).strip()
                    instruction = str(row[instruction_col]).strip()

                    # 跳过空值或nan
                    if low_req and low_req != 'nan' and instruction and instruction != 'nan':
                        # 构建带prompt的输入
                        prompt = TextInstructionTemplate.build_prompt(low_req)

                        all_data.append({
                            'input': low_req,
                            'input_with_prompt': prompt,
                            'output': instruction,
                            'source': csv_file.stem
                        })
                except:
                    # 静默跳过问题行
                    continue

            logger.info(f"加载完成: {csv_file.name}, 数据量: {len(df)}")

        logger.info(f"文本数据集总计: {len(all_data)}条")
        return all_data


class ImageDatasetLoader:
    """图像数据集加载器 - 优化版"""

    def __init__(self):
        """初始化加载器"""
        path_cfg = get_path_config()
        self.dataset_csv = path_cfg.IMAGE_DATASET_CSV
        logger.info(f"初始化ImageDatasetLoader, 路径: {self.dataset_csv}")

    def load_csv_file(self) -> List[Dict]:
        """
        加载图像数据集CSV

        Returns:
            数据列表,每项包含input(description字段)和output(Instruction)
        """
        all_data = []

        if not self.dataset_csv.exists():
            logger.warning(f"图像数据集文件不存在: {self.dataset_csv}")
            return all_data

        try:
            # 智能检测编码
            encoding = detect_csv_encoding(self.dataset_csv)
            df = pd.read_csv(self.dataset_csv, encoding=encoding)

            # 强力清理列名
            df = clean_dataframe_columns(df)

            logger.info(f"使用编码 '{encoding}' 读取图像数据集")

        except Exception:
            # 完全静默失败
            logger.error("加载图像数据集失败")
            return all_data

        # 智能查找必要列
        desc_col = find_column(df, ['description', 'desc', 'descriptions'])
        instruction_col = find_column(df, ['instruction', 'instructions'])

        if not desc_col or not instruction_col:
            logger.error("CSV缺少必要列: Description或Instruction")
            logger.error(f"实际列名: {list(df.columns)}")
            return all_data

        # 提取数据
        for idx, row in df.iterrows():
            try:
                # 解析Description（可能是JSON字符串）
                desc_str = str(row[desc_col])

                # 尝试解析JSON验证格式
                try:
                    desc_json = json.loads(desc_str)
                    # 验证是否包含description字段
                    if 'description' in desc_json:
                        # 过滤掉无用的元数据字段
                        # 只保留description和details，移除confidence, recognition_status, processing_time
                        filtered_json = {
                            'description': desc_json.get('description', ''),
                            'details': desc_json.get('details', {})
                        }
                        # 转换回JSON字符串
                        description = json.dumps(filtered_json, ensure_ascii=False)
                    else:
                        # 如果JSON不包含description字段，可能格式错误，跳过
                        logger.warning(f"行{idx}: JSON不包含description字段，跳过")
                        continue
                except (json.JSONDecodeError, TypeError, ValueError):
                    # 如果不是JSON，当作纯文本description处理
                    description = desc_str.strip()

                instruction = str(row[instruction_col]).strip()

                # 跳过空值或nan
                if description and description != 'nan' and instruction and instruction != 'nan':
                    # 构建带prompt的输入
                    prompt = ImageInstructionTemplate.build_prompt(description)

                    all_data.append({
                        'input': description,
                        'input_with_prompt': prompt,
                        'output': instruction,
                        'source': 'image_dataset'
                    })

            except:
                # 完全静默跳过问题行
                continue

        logger.info(f"图像数据集加载完成, 数据量: {len(all_data)}")

        return all_data


class UMLDatasetLoader:
    """
    UML数据集加载器 - 支持多数据集版本

    支持三种数据集：
    - 'qwen2.5': 本地Qwen2.5-VL识别的数据集
    - 'qwen3': 本地Qwen3-VL识别的数据集
    - 'qwen235B': 云端Qwen235B识别的数据集
    """

    def __init__(self, dataset_version: str = 'qwen2.5'):
        """
        初始化UML数据加载器

        Args:
            dataset_version: 数据集版本 ('qwen2.5', 'qwen3', 'qwen235B')
        """
        self.path_cfg = get_path_config()
        self.dataset_version = dataset_version

        # 数据集文件映射
        self.dataset_files = {
            'qwen2.5': 'uml_dataset_qwen25_local.csv',
            'qwen3': 'uml_dataset_qwen3_local.csv',
            'qwen235B': 'uml_dataset_qwen235B_cloud.csv'
        }

        # 验证版本有效性
        if dataset_version not in self.dataset_files:
            raise ValueError(
                f"不支持的数据集版本: {dataset_version}, "
                f"支持的版本: {list(self.dataset_files.keys())}"
            )

        logger.info(f"初始化UML数据加载器 - 数据集版本: {dataset_version}")

    def load_csv_file(self) -> List[Dict]:
        """
        加载UML CSV文件

        Returns:
            数据列表，每项包含input和output
        """
        # 获取对应版本的文件名
        csv_filename = self.dataset_files[self.dataset_version]
        csv_path = self.path_cfg.UML_DATASET_DIR / csv_filename

        logger.info(f"加载UML数据集: {csv_path}")

        if not csv_path.exists():
            logger.error(f"UML数据集文件不存在: {csv_path}")
            logger.error(f"请确保数据集文件名为: {csv_filename}")
            return []

        try:
            # 检测编码
            encoding = detect_csv_encoding(csv_path)
            logger.info(f"检测到编码: {encoding}")

            # 读取CSV
            df = pd.read_csv(csv_path, encoding=encoding)

            # 规范化列名
            df.columns = [normalize_column_name(col) for col in df.columns]
            logger.info(f"规范化后的列名: {list(df.columns)}")

            # 列名映射（灵活处理不同的命名）
            column_map = {
                'description': ['description', 'desc', 'uml_description', 'Description'],
                'instruction': ['instruction', 'Instruction', 'output', 'Output']
            }

            # 查找实际的列名
            desc_col = None
            inst_col = None

            for standard_name, possible_names in column_map.items():
                possible_names_lower = [normalize_column_name(n) for n in possible_names]
                for col in df.columns:
                    if col in possible_names_lower:
                        if standard_name == 'description':
                            desc_col = col
                        elif standard_name == 'instruction':
                            inst_col = col
                        break

            # 验证必需列
            if desc_col is None or inst_col is None:
                logger.error(f"未找到必需的列。实际列名: {list(df.columns)}")
                logger.error(f"需要包含: description 和 instruction 列")
                return []

            logger.info(f"使用列: description='{desc_col}', instruction='{inst_col}'")

            # 处理数据
            data_list = []
            for idx, row in df.iterrows():
                try:
                    description = row[desc_col]
                    instruction = row[inst_col]

                    # 跳过空值
                    if pd.isna(description) or pd.isna(instruction):
                        continue

                    # 如果description是JSON字符串，验证并保留完整JSON
                    if isinstance(description, str) and description.strip().startswith('{'):
                        try:
                            desc_json = json.loads(description)
                            # 验证是否包含必要字段（actors或use_cases）
                            if 'actors' in desc_json or 'use_cases' in desc_json:
                                # 保留完整JSON字符串（包含actors、use_cases、relationships等）
                                description = description  # 使用完整JSON字符串
                            elif 'description' in desc_json:
                                # 如果只有description字段，也保留完整JSON
                                description = description
                            else:
                                # JSON格式不符合预期，记录警告
                                logger.warning(f"行{idx}: UML JSON格式不符合预期，跳过")
                                continue
                        except json.JSONDecodeError:
                            # 不是有效JSON，当作纯文本处理
                            pass

                    # 构建带prompt的输入
                    prompt = UMLInstructionTemplate.build_prompt(str(description))

                    data_list.append({
                        'input': str(description),
                        'input_with_prompt': prompt,
                        'output': str(instruction),
                        'source': f'uml_dataset_{self.dataset_version}'
                    })

                except Exception as e:
                    logger.warning(f"处理第{idx}行时出错: {e}")
                    continue

            logger.info(f"成功加载UML数据 ({self.dataset_version}): {len(data_list)}条")
            return data_list

        except Exception as e:
            logger.error(f"加载UML数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


class GeneralDatasetLoader:
    """
    通用专家数据集加载器

    功能：
    - 加载text + image + uml三种数据
    - 统一使用GeneralInstructionTemplate处理所有输入
    - 确保训练推理一致性
    """

    def __init__(self, dataset_version: str = 'qwen2.5'):
        """
        初始化通用数据加载器

        Args:
            dataset_version: UML数据集版本 ('qwen2.5', 'qwen3', 'qwen235B')
        """
        self.dataset_version = dataset_version
        logger.info(f"初始化GeneralDatasetLoader - UML数据集版本: {dataset_version}")

    def load_all_data(self) -> List[Dict]:
        """
        加载所有类型的数据并统一使用GeneralInstructionTemplate

        Returns:
            数据列表，每项包含input、input_with_prompt和output
        """
        all_data = []

        # 1. 加载文本数据
        logger.info("加载文本数据...")
        text_loader = TextDatasetLoader()
        text_raw = text_loader.load_csv_files()

        # 重新构建prompt - 使用GeneralInstructionTemplate
        for item in text_raw:
            prompt = GeneralInstructionTemplate.build_prompt(
                item['input'],
                force_type='text'
            )
            all_data.append({
                'input': item['input'],
                'input_with_prompt': prompt,
                'output': item['output'],
                'source': item['source'],
                'data_type': 'text'
            })

        logger.info(f"文本数据: {len(text_raw)}条")

        # 2. 加载图像数据
        logger.info("加载图像数据...")
        image_loader = ImageDatasetLoader()
        image_raw = image_loader.load_csv_file()

        # 重新构建prompt - 使用GeneralInstructionTemplate
        for item in image_raw:
            # item['input']可能是完整JSON字符串或纯文本description
            # 直接传给GeneralInstructionTemplate，它会自动处理
            prompt = GeneralInstructionTemplate.build_prompt(
                item['input'],
                force_type='image'
            )
            all_data.append({
                'input': item['input'],
                'input_with_prompt': prompt,
                'output': item['output'],
                'source': item['source'],
                'data_type': 'image'
            })

        logger.info(f"图像数据: {len(image_raw)}条")

        # 3. 加载UML数据
        logger.info(f"加载UML数据 (版本: {self.dataset_version})...")
        uml_loader = UMLDatasetLoader(dataset_version=self.dataset_version)
        uml_raw = uml_loader.load_csv_file()

        # 重新构建prompt - 使用GeneralInstructionTemplate
        for item in uml_raw:
            # 尝试解析为JSON，如果不是JSON则构建简单格式
            try:
                uml_json = json.loads(item['input']) if isinstance(item['input'], str) else item['input']
            except (json.JSONDecodeError, TypeError):
                uml_json = {
                    "description": item['input'],
                    "details": {
                        "diagram_type": "use case diagram"
                    }
                }

            prompt = GeneralInstructionTemplate.build_prompt(
                uml_json,
                force_type='uml'
            )
            all_data.append({
                'input': item['input'],
                'input_with_prompt': prompt,
                'output': item['output'],
                'source': item['source'],
                'data_type': 'uml'
            })

        logger.info(f"UML数据: {len(uml_raw)}条")

        # 统计
        logger.info(f"通用数据集总计: {len(all_data)}条")
        logger.info(f"  - 文本: {len(text_raw)}条")
        logger.info(f"  - 图像: {len(image_raw)}条")
        logger.info(f"  - UML: {len(uml_raw)}条")

        return all_data


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    划分训练集、验证集、测试集

    Args:
        data: 原始数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_data, val_data, test_data)
    """
    random.seed(seed)

    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]

    logger.info(f"数据集划分 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}")

    return train_data, val_data, test_data


def split_dataset_for_expert(
    data: List[Dict],
    expert_type: str,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    根据专家类型智能划分数据集

    Args:
        data: 原始数据
        expert_type: 'text', 'image', 'uml'
        seed: 随机种子

    Returns:
        (train_data, val_data, test_data)
    """
    data_size = len(data)

    # 根据数据量选择划分策略
    if expert_type == 'uml' and data_size < 100:
        # UML数据较少(90条)，使用85:10:5划分
        train_ratio, val_ratio, test_ratio = 0.85, 0.10, 0.05
        logger.info(f"UML数据集较小({data_size}条)，使用85:10:5划分策略")
    elif data_size < 500:
        # 中等数据量，使用80:15:5划分
        train_ratio, val_ratio, test_ratio = 0.80, 0.15, 0.05
        logger.info(f"中等数据集({data_size}条)，使用80:15:5划分策略")
    else:
        # 大数据量，使用标准80:10:10划分
        train_ratio, val_ratio, test_ratio = 0.80, 0.10, 0.10
        logger.info(f"大数据集({data_size}条)，使用80:10:10划分策略")

    return split_dataset(data, train_ratio, val_ratio, test_ratio, seed)


def create_dataloader(
    dataset: InstructionDataset,
    batch_size: int = None,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    创建DataLoader

    Args:
        dataset: 数据集
        batch_size: 批次大小（如果为None则使用配置）
        shuffle: 是否打乱
        num_workers: 工作进程数

    Returns:
        DataLoader对象
    """
    if batch_size is None:
        train_cfg = get_training_config()
        batch_size = train_cfg.batch_size

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


# 测试代码
if __name__ == "__main__":
    print("="*80)
    print("数据加载器测试（修复版）")
    print("="*80)

    # 测试文本数据加载
    print("\n【测试1】文本数据加载")
    print("-"*80)
    text_loader = TextDatasetLoader()
    text_data = text_loader.load_csv_files()
    if text_data:
        print(f"✓ 数据加载成功")
        print(f"  数据量: {len(text_data)}条")
        print(f"  示例来源: {text_data[0]['source']}")
        train, val, test = split_dataset_for_expert(text_data, 'text')
        print(f"  划分结果: 训练{len(train)}, 验证{len(val)}, 测试{len(test)}")
    else:
        print("✗ 数据加载失败")

    # 测试图像数据加载
    print("\n【测试2】图像数据加载")
    print("-"*80)
    image_loader = ImageDatasetLoader()
    image_data = image_loader.load_csv_file()
    if image_data:
        print(f"✓ 数据加载成功")
        print(f"  数据量: {len(image_data)}条")
        print(f"  示例来源: {image_data[0]['source']}")
        train, val, test = split_dataset_for_expert(image_data, 'image')
        print(f"  划分结果: 训练{len(train)}, 验证{len(val)}, 测试{len(test)}")
    else:
        print("✗ 数据加载失败")

    # 测试UML数据加载
    print("\n【测试3】UML数据加载")
    print("-"*80)
    uml_loader = UMLDatasetLoader()
    uml_data = uml_loader.load_csv_file()
    if uml_data:
        print(f"✓ 数据加载成功")
        print(f"  数据量: {len(uml_data)}条")
        print(f"  示例来源: {uml_data[0]['source']}")
        train, val, test = split_dataset_for_expert(uml_data, 'uml')
        print(f"  划分结果: 训练{len(train)}, 验证{len(val)}, 测试{len(test)}")
    else:
        print("✗ 数据加载失败")

    print("\n数据加载器测试完成！")