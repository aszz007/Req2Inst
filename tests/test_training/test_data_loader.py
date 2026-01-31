"""
数据加载器测试（双模式版）
测试 src/training/data_loader.py 中的数据加载功能

运行方式：
1. 快速模式（使用mock，默认）：
   pytest tests/test_training/test_data_loader.py -v

2. 真实模式（使用真实数据集）：
   pytest tests/test_training/test_data_loader.py -v --real-data

3. 只运行快速测试：
   pytest tests/test_training/test_data_loader.py -v -m "not real_data"
"""

import pytest
import pandas as pd
import json
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.training.data_loader import (
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,
    InstructionDataset,
    split_dataset,
    split_dataset_for_expert,
    create_dataloader
)


# ==================== pytest配置 ====================
def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--real-data",
        action="store_true",
        default=False,
        help="使用真实数据集进行测试（较慢）"
    )


def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "real_data: 使用真实数据的测试（较慢）"
    )


@pytest.fixture
def use_real_data(request):
    """判断是否使用真实数据"""
    return request.config.getoption("--real-data")


# ==================== Mock数据Fixtures ====================
@pytest.fixture
def mock_text_csv(tmp_path):
    """创建模拟的文本CSV文件"""
    data = {
        'High_Requirements': ['High req 1', 'High req 2', 'High req 3'],
        'Low_Requirements': ['Low req 1', 'Low req 2', 'Low req 3'],
        'Instruction': ['Inst 1', 'Inst 2', 'Inst 3']
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_image_csv(tmp_path):
    """创建模拟的图像CSV"""
    descriptions = []
    for i in range(5):
        desc_json = {
            "description": f"Image description {i}",
            "details": {"objects": []},
            "confidence": 0.95,
            "recognition_status": "success",
            "processing_time": 1.5
        }
        descriptions.append(json.dumps(desc_json))

    data = {
        'Header': [f'img_{i}' for i in range(5)],
        'Description': descriptions,
        'Instruction': [f'Inst {i}' for i in range(5)]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "image_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_uml_csv(tmp_path):
    """创建模拟的UML CSV"""
    descriptions = []
    for i in range(5):
        uml_json = {
            "actors": [{"name": f"Actor{i}", "position": "left"}],
            "use_cases": [{"name": f"UseCase{i}", "description": "test"}],
            "system_boundary": {"name": "System", "is_present": True},
            "relationships": [],
            "overall_description": f"UML diagram {i}"
        }
        descriptions.append(json.dumps(uml_json))

    data = {
        'Description': descriptions,
        'Instruction': [f'Inst {i}' for i in range(5)]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "uml_dataset_qwen235B_cloud.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_tokenizer():
    """创建模拟的tokenizer"""
    tokenizer = Mock()

    def mock_encode(*args, **kwargs):
        # 模拟tokenizer的返回值
        return {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }

    tokenizer.side_effect = mock_encode
    return tokenizer


# ==================== 快速模式测试（使用Mock） ====================
class TestTextDatasetLoaderMock:
    """文本数据集加载器测试（Mock模式）"""

    @patch('src.training.data_loader.get_path_config')
    def test_init(self, mock_path_cfg, tmp_path):
        """测试初始化"""
        mock_cfg = Mock()
        mock_cfg.TEXT_DATASET_DIR = tmp_path
        mock_path_cfg.return_value = mock_cfg

        loader = TextDatasetLoader()

        assert loader.dataset_dir == tmp_path

    @patch('src.training.data_loader.get_path_config')
    def test_load_csv_files_success(self, mock_path_cfg, tmp_path, mock_text_csv):
        """测试成功加载CSV文件"""
        # 设置mock配置
        mock_cfg = Mock()
        mock_cfg.TEXT_DATASET_DIR = tmp_path
        mock_path_cfg.return_value = mock_cfg

        loader = TextDatasetLoader()
        data = loader.load_csv_files()

        # 验证数据结构
        assert len(data) == 3
        assert 'input' in data[0]
        assert 'output' in data[0]
        assert 'source' in data[0]
        assert data[0]['input'] == 'Low req 1'
        assert data[0]['output'] == 'Inst 1'

    @patch('src.training.data_loader.get_path_config')
    def test_load_csv_files_missing_columns(self, mock_path_cfg, tmp_path):
        """测试缺少必要列的CSV"""
        # 创建缺少列的CSV
        data = {'OnlyOneColumn': ['value1', 'value2']}
        df = pd.DataFrame(data)
        csv_path = tmp_path / "bad_dataset.csv"
        df.to_csv(csv_path, index=False)

        mock_cfg = Mock()
        mock_cfg.TEXT_DATASET_DIR = tmp_path
        mock_path_cfg.return_value = mock_cfg

        loader = TextDatasetLoader()
        data = loader.load_csv_files()

        # 应该跳过这个文件，返回空列表
        assert len(data) == 0


class TestImageDatasetLoaderMock:
    """图像数据集加载器测试（Mock模式）"""

    @patch('src.training.data_loader.get_path_config')
    def test_load_csv_file_success(self, mock_path_cfg, mock_image_csv):
        """测试成功加载图像CSV"""
        mock_cfg = Mock()
        mock_cfg.IMAGE_DATASET_CSV = mock_image_csv
        mock_path_cfg.return_value = mock_cfg

        loader = ImageDatasetLoader()
        data = loader.load_csv_file()

        assert len(data) == 5
        # 验证只提取了description字段
        assert 'input' in data[0]
        assert 'Image description' in data[0]['input']
        # 不应该包含confidence等元数据
        assert 'confidence' not in data[0]['input']

    @patch('src.training.data_loader.get_path_config')
    def test_load_csv_file_not_exist(self, mock_path_cfg, tmp_path):
        """测试文件不存在的情况"""
        mock_cfg = Mock()
        mock_cfg.IMAGE_DATASET_CSV = tmp_path / "nonexistent.csv"
        mock_path_cfg.return_value = mock_cfg

        loader = ImageDatasetLoader()
        data = loader.load_csv_file()

        assert len(data) == 0


class TestUMLDatasetLoaderMock:
    """UML数据集加载器测试（Mock模式）"""

    @patch('src.training.data_loader.get_path_config')
    def test_load_csv_file_success(self, mock_path_cfg, mock_uml_csv):
        """测试成功加载UML CSV"""
        mock_cfg = Mock()
        mock_cfg.UML_DATASET_CSV = mock_uml_csv
        mock_path_cfg.return_value = mock_cfg

        loader = UMLDatasetLoader()
        data = loader.load_csv_file()

        assert len(data) == 5
        # UML的description应该是完整的JSON字符串
        assert 'input' in data[0]
        assert 'actors' in data[0]['input']


class TestInstructionDataset:
    """InstructionDataset测试"""

    def test_init(self, mock_tokenizer):
        """测试初始化"""
        data = [
            {'input': 'Test input 1', 'output': 'Test output 1'},
            {'input': 'Test input 2', 'output': 'Test output 2'},
        ]

        dataset = InstructionDataset(data, mock_tokenizer, max_length=512)

        assert len(dataset) == 2
        assert dataset.tokenizer == mock_tokenizer
        assert dataset.max_length == 512

    def test_len(self, mock_tokenizer):
        """测试__len__方法"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(10)]
        dataset = InstructionDataset(data, mock_tokenizer)

        assert len(dataset) == 10

    def test_getitem(self, mock_tokenizer):
        """测试__getitem__方法"""
        data = [{'input': 'Test input', 'output': 'Test output'}]
        dataset = InstructionDataset(data, mock_tokenizer)

        item = dataset[0]

        # 验证返回的字段
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item

        # 验证是tensor
        assert torch.is_tensor(item['input_ids'])
        assert torch.is_tensor(item['attention_mask'])
        assert torch.is_tensor(item['labels'])


class TestSplitDataset:
    """数据集划分测试"""

    def test_split_dataset_basic(self):
        """测试基本的数据集划分"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(100)]

        train, val, test = split_dataset(data, 0.8, 0.1, 0.1, seed=42)

        # 验证数量
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

        # 验证总和
        assert len(train) + len(val) + len(test) == 100

    def test_split_dataset_no_overlap(self):
        """测试划分后的数据集没有重叠"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(100)]

        train, val, test = split_dataset(data, 0.8, 0.1, 0.1, seed=42)

        # 提取input用于比较
        train_inputs = {item['input'] for item in train}
        val_inputs = {item['input'] for item in val}
        test_inputs = {item['input'] for item in test}

        # 验证没有重叠
        assert len(train_inputs & val_inputs) == 0
        assert len(train_inputs & test_inputs) == 0
        assert len(val_inputs & test_inputs) == 0

    def test_split_dataset_reproducible(self):
        """测试相同种子产生相同结果"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(100)]

        train1, val1, test1 = split_dataset(data, 0.8, 0.1, 0.1, seed=42)
        train2, val2, test2 = split_dataset(data, 0.8, 0.1, 0.1, seed=42)

        # 验证结果相同
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2


class TestSplitDatasetForExpert:
    """专家数据集划分测试"""

    def test_text_expert_split(self):
        """测试文本专家的划分（大数据集，80:10:10）"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(2400)]

        train, val, test = split_dataset_for_expert(data, 'text', seed=42)

        # 大数据集应使用80:10:10
        total = len(data)
        assert abs(len(train) - total * 0.8) <= 1
        assert abs(len(val) - total * 0.1) <= 1
        assert abs(len(test) - total * 0.1) <= 1

    def test_image_expert_split(self):
        """测试图像专家的划分（中等数据集，80:10:10）"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(500)]

        train, val, test = split_dataset_for_expert(data, 'image', seed=42)

        # 中等数据集应使用80:10:10
        total = len(data)
        assert abs(len(train) - total * 0.8) <= 1
        assert abs(len(val) - total * 0.1) <= 1
        assert abs(len(test) - total * 0.1) <= 1

    def test_uml_expert_split(self):
        """测试UML专家的划分（小数据集，85:10:5）"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(90)]

        train, val, test = split_dataset_for_expert(data, 'uml', seed=42)

        # 小数据集应使用85:10:5
        total = len(data)
        assert abs(len(train) - total * 0.85) <= 1
        assert abs(len(val) - total * 0.10) <= 1
        assert abs(len(test) - total * 0.05) <= 1


class TestCreateDataLoader:
    """DataLoader创建测试"""

    @patch('src.training.data_loader.get_training_config')
    def test_create_dataloader_default_batch_size(self, mock_train_cfg, mock_tokenizer):
        """测试使用默认batch_size"""
        # Mock配置
        mock_cfg = Mock()
        mock_cfg.batch_size = 4
        mock_train_cfg.return_value = mock_cfg

        # 创建数据集
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(20)]
        dataset = InstructionDataset(data, mock_tokenizer)

        # 创建DataLoader
        dataloader = create_dataloader(dataset, batch_size=None, shuffle=True)

        assert dataloader is not None
        assert dataloader.batch_size == 4

    def test_create_dataloader_custom_batch_size(self, mock_tokenizer):
        """测试自定义batch_size"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(20)]
        dataset = InstructionDataset(data, mock_tokenizer)

        dataloader = create_dataloader(dataset, batch_size=8, shuffle=False)

        assert dataloader.batch_size == 8

    def test_create_dataloader_iterable(self, mock_tokenizer):
        """测试DataLoader可以迭代"""
        data = [{'input': f'input {i}', 'output': f'output {i}'} for i in range(20)]
        dataset = InstructionDataset(data, mock_tokenizer)

        dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

        # 验证可以迭代
        batches = list(dataloader)
        assert len(batches) == 5  # 20 / 4 = 5


# ==================== 真实数据测试 ====================
@pytest.mark.real_data
class TestRealDataIntegration:
    """使用真实数据的集成测试（较慢）"""

    def test_load_real_text_data(self, use_real_data):
        """测试加载真实文本数据集"""
        if not use_real_data:
            pytest.skip("需要 --real-data 标志才能运行真实数据测试")

        loader = TextDatasetLoader()
        data = loader.load_csv_files()

        # 验证数据加载成功
        assert len(data) > 0, "文本数据集为空"

        # 验证数据结构
        sample = data[0]
        assert 'input' in sample
        assert 'output' in sample
        assert 'source' in sample

        # 验证数据内容合理
        assert len(sample['input']) > 0
        assert len(sample['output']) > 0

        print(f"\n✓ 成功加载 {len(data)} 条文本数据")
        print(f"数据示例:\n  输入: {sample['input'][:50]}...\n  输出: {sample['output'][:50]}...")

    def test_load_real_image_data(self, use_real_data):
        """测试加载真实图像数据集"""
        if not use_real_data:
            pytest.skip("需要 --real-data 标志")

        loader = ImageDatasetLoader()
        data = loader.load_csv_file()

        if len(data) == 0:
            pytest.skip("图像数据集文件不存在或为空")

        # 验证数据结构
        sample = data[0]
        assert 'input' in sample
        assert 'output' in sample

        # 验证只提取了description字段
        assert isinstance(sample['input'], str)

        print(f"\n✓ 成功加载 {len(data)} 条图像数据")
        print(f"数据示例:\n  描述: {sample['input'][:50]}...")

    def test_load_real_uml_data(self, use_real_data):
        """测试加载真实UML数据集"""
        if not use_real_data:
            pytest.skip("需要 --real-data 标志")

        loader = UMLDatasetLoader()
        data = loader.load_csv_file()

        if len(data) == 0:
            pytest.skip("UML数据集文件不存在或为空")

        # 验证数据结构
        sample = data[0]
        assert 'input' in sample
        assert 'output' in sample

        print(f"\n✓ 成功加载 {len(data)} 条UML数据")

    def test_full_pipeline_with_real_data(self, use_real_data, mock_tokenizer):
        """测试完整的数据处理流程（真实数据）"""
        if not use_real_data:
            pytest.skip("需要 --real-data 标志")

        # 1. 加载数据
        text_loader = TextDatasetLoader()
        data = text_loader.load_csv_files()

        if len(data) == 0:
            pytest.skip("没有可用的数据")

        # 限制数据量（避免测试太慢）
        data = data[:100]

        # 2. 划分数据集
        train, val, test = split_dataset_for_expert(data, 'text', seed=42)

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        # 3. 创建Dataset
        train_dataset = InstructionDataset(train, mock_tokenizer, max_length=512)

        assert len(train_dataset) == len(train)

        # 4. 创建DataLoader
        dataloader = create_dataloader(train_dataset, batch_size=4, shuffle=True)

        assert dataloader is not None

        # 5. 验证可以迭代
        first_batch = next(iter(dataloader))
        assert 'input_ids' in first_batch
        assert 'attention_mask' in first_batch
        assert 'labels' in first_batch

        print(f"\n✓ 完整流程测试通过")
        print(f"  训练集: {len(train)} 条")
        print(f"  验证集: {len(val)} 条")
        print(f"  测试集: {len(test)} 条")


# ==================== 运行测试 ====================
if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("数据加载器测试")
    print("=" * 80)
    print("\n请选择测试模式:")
    print("1. 快速模式（使用Mock，推荐）")
    print("2. 真实模式（使用真实数据集，较慢）")

    choice = input("\n请输入选择 (1/2，默认1): ").strip() or "1"

    if choice == "1":
        print("\n使用快速模式运行测试...")
        pytest.main([__file__, "-v", "-m", "not real_data"])
    elif choice == "2":
        print("\n使用真实模式运行测试...")
        pytest.main([__file__, "-v", "--real-data"])
    else:
        print("无效的选择，使用快速模式...")
        pytest.main([__file__, "-v", "-m", "not real_data"])