"""
测试模块
提供单元测试和集成测试，确保代码质量
测试框架: pytest
覆盖率工具: pytest-cov

运行所有测试:
    pytest tests/ -v

运行快速测试（跳过真实数据）:
    pytest tests/ -v -m "not real_data"

运行包含真实数据的测试:
    pytest tests/ -v --real-data

运行特定模块测试:
    pytest tests/test_models/ -v

查看覆盖率:
    pytest --cov=src --cov=models --cov=config tests/

生成HTML覆盖率报告:
    pytest --cov=src --cov=models --cov=config --cov-report=html tests/
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

__version__ = '0.2.0'
__all__ = [
    'test_preprocessing',
    'test_models',
    'test_training',
]

# 测试配置
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
TEST_OUTPUT_DIR = PROJECT_ROOT / "tests" / "test_output"

# 确保测试目录存在
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""
tests/
├── __init__.py                          # 主测试模块初始化
├── conftest.py                          # pytest全局配置
├── test_data/                           # 测试数据目录
│   ├── sample_image.jpg                 # 示例图像
│   ├── sample_uml.png                   # 示例UML图
│   └── sample_text.csv                  # 示例文本数据
├── test_output/                         # 测试输出目录
│   └── .gitkeep
│
├── test_preprocessing/                  # 预处理测试
│   ├── __init__.py
│   ├── test_image_to_json.py           # 图像转JSON测试（9个用例）
│   └── test_uml_to_json.py             # UML转JSON测试（9个用例）
│
├── test_models/                         # 模型测试
│   ├── __init__.py
│   ├── test_language_model.py          # 语言模型测试（15个用例）
│   └── test_vision_model.py            # 视觉模型测试（14个用例）
│
└── test_training/                       # 训练测试
    ├── __init__.py
    ├── test_data_loader.py             # 数据加载器测试（20+个用例）
    │                                   # 支持快速模式和真实数据模式
    └── test_expert_trainer.py          # 训练器测试（待实现）

测试命令速查：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【基础运行】
  运行所有测试:
    pytest tests/ -v

  运行特定目录:
    pytest tests/test_models/ -v

  运行特定文件:
    pytest tests/test_models/test_language_model.py -v

  运行特定测试:
    pytest tests/test_models/test_language_model.py::TestLanguageModel::test_init -v

【快速模式 vs 真实数据模式】
  快速模式（默认，使用mock）:
    pytest tests/ -v -m "not real_data"

  真实数据模式（使用dataset/中的数据）:
    pytest tests/ -v --real-data

  数据加载器快速测试:
    pytest tests/test_training/test_data_loader.py -v

  数据加载器真实测试:
    pytest tests/test_training/test_data_loader.py -v --real-data

【覆盖率测试】
  查看覆盖率:
    pytest --cov=src --cov=models --cov=config tests/

  生成详细覆盖率报告:
    pytest --cov=src --cov-report=term-missing tests/

  生成HTML报告:
    pytest --cov=src --cov-report=html tests/
    # 报告位置: htmlcov/index.html

【标记过滤】
  只运行单元测试:
    pytest -m "unit" tests/

  只运行集成测试:
    pytest -m "integration" tests/

  跳过慢速测试:
    pytest -m "not slow" tests/

  跳过真实数据测试:
    pytest -m "not real_data" tests/

【调试选项】
  失败时进入调试器:
    pytest tests/ --pdb

  在第一个失败后停止:
    pytest tests/ -x

  显示print输出:
    pytest tests/ -s

  显示最慢的10个测试:
    pytest tests/ --durations=10

  更详细的输出:
    pytest tests/ -vv --tb=long

【并行运行（需要pytest-xdist）】
  4个进程并行:
    pytest tests/ -n 4

  自动检测CPU核心数:
    pytest tests/ -n auto

【交互式运行】
  直接运行Python文件（带菜单）:
    python tests/test_training/test_data_loader.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

测试标记定义：
  @pytest.mark.unit          # 单元测试（快速，测试单个函数）
  @pytest.mark.integration   # 集成测试（测试多个模块交互）
  @pytest.mark.slow          # 慢速测试（可能需要加载模型）
  @pytest.mark.real_data     # 使用真实数据的测试
  @pytest.mark.preprocessing # 预处理相关测试
  @pytest.mark.models        # 模型相关测试
  @pytest.mark.training      # 训练相关测试

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""