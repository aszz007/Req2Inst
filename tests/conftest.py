"""
pytest全局配置和fixtures
提供所有测试共享的fixtures和配置
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# 将项目根目录添加到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==================== 路径Fixtures ====================
@pytest.fixture(scope="session")
def project_root():
    """返回项目根目录"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """返回测试数据目录"""
    test_dir = project_root / "tests" / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def test_output_dir(project_root):
    """返回测试输出目录"""
    output_dir = project_root / "tests" / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_dir(tmp_path):
    """返回临时目录（每个测试独立）"""
    return tmp_path


# ==================== 配置Fixtures ====================
@pytest.fixture
def mock_path_config(tmp_path):
    """创建模拟的PathConfig对象"""
    config = Mock()

    # 基础路径
    config.PROJECT_ROOT = tmp_path
    config.BASE_MODELS_DIR = tmp_path / "base_models"

    # 模型路径
    config.QWEN_7B_CHAT_PATH = config.BASE_MODELS_DIR / "qwen-7B-Chat"
    config.QWEN_VL_7B_PATH = config.BASE_MODELS_DIR / "qwen2.5-VL-7B"

    # 数据路径
    config.DATA_DIR = tmp_path / "data"
    config.RAW_DATA_DIR = config.DATA_DIR / "raw"
    config.INTERIM_DATA_DIR = config.DATA_DIR / "interim"
    config.DATASET_DIR = tmp_path / "dataset"

    # 数据集路径
    config.TEXT_DATASET_DIR = config.DATASET_DIR / "text"
    config.IMAGE_DATASET_DIR = config.DATASET_DIR / "image"
    config.UML_DATASET_DIR = config.DATASET_DIR / "uml"

    config.IMAGE_DATASET_CSV = config.IMAGE_DATASET_DIR / "image_dataset.csv"
    config.UML_DATASET_CSV = config.UML_DATASET_DIR / "uml_dataset.csv"

    config.TEXT_DATASET_FILES = {
        'TEST': config.TEXT_DATASET_DIR / "test_dataset.csv"
    }

    # LoRA路径
    config.LORA_WEIGHTS_DIR = tmp_path / "lora_weights"
    config.EXPERTS_DIR = config.LORA_WEIGHTS_DIR / "experts"

    config.TEXT_EXPERT_WEIGHTS = config.EXPERTS_DIR / "text_expert"
    config.IMAGE_EXPERT_WEIGHTS = config.EXPERTS_DIR / "image_expert_qwen2.5"
    config.UML_EXPERT_WEIGHTS = config.EXPERTS_DIR / "uml_expert_qwen2.5"
    config.GENERAL_EXPERT_WEIGHTS = config.EXPERTS_DIR / "general_expert"

    # 专家路径映射
    config.EXPERT_LORA_PATHS = {
        'text': config.TEXT_EXPERT_WEIGHTS,
        'image': config.IMAGE_EXPERT_WEIGHTS,
        'uml': config.UML_EXPERT_WEIGHTS,
        'general': config.GENERAL_EXPERT_WEIGHTS,
    }

    # 输出路径
    config.OUTPUTS_DIR = tmp_path / "outputs"
    config.GENERATED_INSTRUCTIONS_DIR = config.OUTPUTS_DIR / "generated_instructions"
    config.EVALUATIONS_DIR = config.OUTPUTS_DIR / "evaluations"

    # 日志路径
    config.LOGS_DIR = tmp_path / "logs"

    # 方法
    def get_expert_weight_path(expert_name):
        return config.EXPERT_LORA_PATHS.get(expert_name, config.EXPERTS_DIR / expert_name)

    config.get_expert_weight_path = get_expert_weight_path

    return config


@pytest.fixture
def mock_lora_config():
    """创建模拟的LoRAConfig对象"""
    config = Mock()
    config.rank = 8
    config.alpha = 16
    config.dropout = 0.05
    config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config.task_type = "CAUSAL_LM"
    config.bias = "none"
    return config


@pytest.fixture
def mock_training_config():
    """创建模拟的TrainingConfig对象"""
    config = Mock()
    config.batch_size = 4
    config.gradient_accumulation_steps = 4
    config.num_epochs = 3
    config.learning_rate = 2e-4
    config.optimizer = "adamw_torch"
    config.weight_decay = 0.01
    config.warmup_ratio = 0.1
    config.lr_scheduler_type = "cosine"
    config.fp16 = True
    config.seed = 42
    return config


@pytest.fixture
def mock_device_config():
    """创建模拟的DeviceConfig对象"""
    config = Mock()
    config.device = "cpu"
    config.get_device = Mock(return_value="cpu")
    return config


# ==================== 测试数据Fixtures ====================
@pytest.fixture
def sample_text_requirement():
    """示例文本需求"""
    return "The system shall allow users to login with username and password."


@pytest.fixture
def sample_image_description():
    """示例图像描述JSON"""
    return {
        "description": "A colorful bento box meal featuring rice, vegetables, and protein",
        "details": {
            "objects": ["bento box", "rice", "vegetables", "protein"],
            "scene": "food photography",
            "spatial_info": "centered composition"
        },
        "confidence": 0.95,
        "recognition_status": "success"
    }


@pytest.fixture
def sample_uml_description():
    """示例UML描述JSON"""
    return {
        "actors": [
            {"name": "User", "position": "left"},
            {"name": "Admin", "position": "right"}
        ],
        "use_cases": [
            {"name": "Login System", "description": "User authentication"},
            {"name": "Manage Users", "description": "Admin user management"}
        ],
        "system_boundary": {
            "name": "Authentication System",
            "is_present": True
        },
        "relationships": [
            {"type": "association", "from": "User", "to": "Login System", "description": "uses"},
            {"type": "association", "from": "Admin", "to": "Manage Users", "description": "performs"}
        ],
        "overall_description": "A simple authentication and user management use case diagram"
    }


@pytest.fixture
def sample_instruction():
    """示例众包指令（三段式）"""
    return """Definition: In this task, you need to annotate images with bounding boxes and labels for all visible objects.

Emphasis & Caution: Please ensure that bounding boxes are accurate and labels are correct. Pay special attention to small objects that might be overlooked.

Things to Avoid: Do not create overlapping bounding boxes. Avoid labeling partially visible objects unless more than 50% of the object is visible."""


# ==================== Mock模型Fixtures ====================
@pytest.fixture
def mock_tokenizer():
    """模拟Tokenizer对象"""
    tokenizer = Mock()
    tokenizer.pad_token = '<|endoftext|>'
    tokenizer.eos_token = '<|im_end|>'
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 100

    # Mock encode/decode
    tokenizer.return_value = {
        'input_ids': [[1, 2, 3]],
        'attention_mask': [[1, 1, 1]]
    }
    tokenizer.decode = Mock(return_value="Generated text")
    tokenizer.convert_tokens_to_ids = Mock(return_value=3)

    return tokenizer


@pytest.fixture
def mock_model():
    """模拟语言模型对象"""
    model = Mock()
    model.device = "cpu"

    # Mock generate
    def mock_generate(*args, **kwargs):
        import torch
        return torch.tensor([[1, 2, 3, 4, 5]])

    model.generate = Mock(side_effect=mock_generate)
    model.eval = Mock()

    return model


@pytest.fixture
def mock_vision_model():
    """模拟视觉模型对象"""
    model = Mock()

    # Mock recognize_image
    def mock_recognize_image(image_path):
        return {
            "description": "A test image",
            "details": {"objects": [], "scene": "test", "spatial_info": "center"},
            "confidence": 0.95,
            "recognition_status": "success"
        }

    # Mock recognize_uml
    def mock_recognize_uml(uml_path, max_retries=2):
        import json
        uml_data = {
            "actors": [],
            "use_cases": [],
            "system_boundary": {"name": "Test", "is_present": True},
            "relationships": [],
            "overall_description": "Test UML"
        }
        return {
            "description": json.dumps(uml_data),
            "success": True
        }

    model.recognize_image = Mock(side_effect=mock_recognize_image)
    model.recognize_uml = Mock(side_effect=mock_recognize_uml)

    return model


# ==================== 数据集Fixtures ====================
@pytest.fixture
def sample_text_dataset():
    """示例文本数据集"""
    return [
        {
            'Low_Requirements': 'System shall support user login',
            'Instruction': 'Test the login functionality...'
        },
        {
            'Low_Requirements': 'System shall validate password',
            'Instruction': 'Verify password validation...'
        },
        {
            'Low_Requirements': 'System shall lock account after 3 failed attempts',
            'Instruction': 'Test account locking mechanism...'
        }
    ]


@pytest.fixture
def sample_image_dataset():
    """示例图像数据集"""
    import json

    dataset = []
    for i in range(5):
        desc = {
            "description": f"Image {i} description",
            "confidence": 0.95,
            "recognition_status": "success"
        }
        dataset.append({
            'Description': json.dumps(desc),
            'Instruction': f'Annotate image {i}...'
        })

    return dataset


# ==================== Pytest配置Hook ====================
def pytest_configure(config):
    """pytest配置钩子"""
    # 注册自定义标记
    config.addinivalue_line(
        "markers", "unit: 单元测试"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集结果"""
    # 可以在这里添加自动标记逻辑
    pass


# ==================== 测试辅助函数 ====================
@pytest.fixture
def assert_json_equal():
    """比较两个JSON是否相等的辅助函数"""

    def _assert_json_equal(json1, json2):
        import json
        if isinstance(json1, str):
            json1 = json.loads(json1)
        if isinstance(json2, str):
            json2 = json.loads(json2)
        assert json1 == json2

    return _assert_json_equal


@pytest.fixture
def create_mock_csv(tmp_path):
    """创建模拟CSV文件的辅助函数"""

    def _create_csv(data, filename="test.csv"):
        import pandas as pd
        df = pd.DataFrame(data)
        csv_path = tmp_path / filename
        df.to_csv(csv_path, index=False)
        return csv_path

    return _create_csv


@pytest.fixture
def create_mock_image(tmp_path):
    """创建模拟图像文件的辅助函数"""

    def _create_image(filename="test.jpg"):
        image_path = tmp_path / filename
        image_path.write_bytes(b'fake image data')
        return image_path

    return _create_image