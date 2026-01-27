"""
数据加载器
负责加载和处理三类数据集(文本、图像、UML)
支持合理的数据集划分
作者：Data Loader System
日期：2025-01-26
"""

import os
import json
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch

from config.settings import get_path_config, get_training_config
from src.utils.logger import get_logger
from src.utils.file_utils import load_json

logger = get_logger('training.data_loader')


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

        # 构建输入prompt
        input_text = item['input']
        output_text = item['output']

        # 组合输入输出
        full_text = f"### Input:\n{input_text}\n\n### Output:\n{output_text}"

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


class TextDatasetLoader:
    """文本数据集加载器"""

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
                df = pd.read_csv(csv_file)

                # 验证必要列
                if 'Low_Requirements' not in df.columns or 'Instruction' not in df.columns:
                    logger.warning(f"跳过文件(缺少必要列): {csv_file.name}")
                    continue

                # 提取数据
                for _, row in df.iterrows():
                    all_data.append({
                        'input': str(row['Low_Requirements']).strip(),
                        'output': str(row['Instruction']).strip(),
                        'source': csv_file.stem
                    })

                logger.info(f"加载完成: {csv_file.name}, 数据量: {len(df)}")

            except Exception as e:
                logger.error(f"加载失败: {csv_file.name}, 错误: {e}")

        logger.info(f"文本数据集总计: {len(all_data)}条")
        return all_data


class ImageDatasetLoader:
    """图像数据集加载器"""

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
            df = pd.read_csv(self.dataset_csv)

            # 验证必要列
            if 'Description' not in df.columns or 'Instruction' not in df.columns:
                logger.error("CSV缺少必要列: Description或Instruction")
                return all_data

            # 提取数据
            for _, row in df.iterrows():
                try:
                    # 解析Description JSON
                    desc_json = json.loads(row['Description'])

                    # 只提取description字段（忽略confidence等元数据）
                    description = desc_json.get('description', '').strip()

                    if description:
                        all_data.append({
                            'input': description,
                            'output': str(row['Instruction']).strip(),
                            'source': 'image_dataset'
                        })

                except json.JSONDecodeError:
                    logger.warning(f"JSON解析失败,跳过该行")
                    continue

            logger.info(f"图像数据集加载完成, 数据量: {len(all_data)}")

        except Exception as e:
            logger.error(f"加载图像数据集失败: {e}")

        return all_data


class UMLDatasetLoader:
    """UML数据集加载器"""

    def __init__(self):
        """初始化加载器"""
        path_cfg = get_path_config()
        self.dataset_csv = path_cfg.UML_DATASET_CSV
        logger.info(f"初始化UMLDatasetLoader, 路径: {self.dataset_csv}")

    def load_csv_file(self) -> List[Dict]:
        """
        加载UML数据集CSV

        Returns:
            数据列表,每项包含input(description)和output(Instruction)
        """
        all_data = []

        if not self.dataset_csv.exists():
            logger.warning(f"UML数据集文件不存在: {self.dataset_csv}")
            return all_data

        try:
            df = pd.read_csv(self.dataset_csv)

            # 验证必要列
            if 'Description' not in df.columns or 'Instruction' not in df.columns:
                logger.error("CSV缺少必要列: Description或Instruction")
                return all_data

            # 提取数据
            for _, row in df.iterrows():
                try:
                    # UML数据集的Description直接是JSON字符串描述
                    # 不需要解析，直接使用
                    description = str(row['Description']).strip()

                    if description:
                        all_data.append({
                            'input': description,
                            'output': str(row['Instruction']).strip(),
                            'source': 'uml_dataset'
                        })

                except Exception as e:
                    logger.warning(f"数据处理失败: {e}, 跳过该行")
                    continue

            logger.info(f"UML数据集加载完成, 数据量: {len(all_data)}")

        except Exception as e:
            logger.error(f"加载UML数据集失败: {e}")

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
    print("数据加载器测试")
    print("="*80)

    # 测试文本数据加载
    print("\n【测试1】文本数据加载")
    print("-"*80)
    text_loader = TextDatasetLoader()
    text_data = text_loader.load_csv_files()
    if text_data:
        print(f"数据示例:\n{text_data[0]}")
        train, val, test = split_dataset_for_expert(text_data, 'text')
        print(f"划分结果: 训练{len(train)}, 验证{len(val)}, 测试{len(test)}")

    # 测试图像数据加载
    print("\n【测试2】图像数据加载")
    print("-"*80)
    image_loader = ImageDatasetLoader()
    image_data = image_loader.load_csv_file()
    if image_data:
        print(f"数据示例:\n{image_data[0]}")
        train, val, test = split_dataset_for_expert(image_data, 'image')
        print(f"划分结果: 训练{len(train)}, 验证{len(val)}, 测试{len(test)}")

    # 测试UML数据加载
    print("\n【测试3】UML数据加载")
    print("-"*80)
    uml_loader = UMLDatasetLoader()
    uml_data = uml_loader.load_csv_file()
    if uml_data:
        print(f"数据示例:\n{uml_data[0]}")
        train, val, test = split_dataset_for_expert(uml_data, 'uml')
        print(f"划分结果: 训练{len(train)}, 验证{len(val)}, 测试{len(test)}")

    print("\n数据加载器测试完成！")