"""
语言模型测试
测试 models/language_model.py 中的 LanguageModel 和 InstructionGenerator 类
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from models.language_model import LanguageModel, InstructionGenerator


class TestLanguageModel:
    """测试 LanguageModel 类"""

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """创建临时模型路径"""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        return str(model_dir)

    @pytest.fixture
    def mock_lora_path(self, tmp_path):
        """创建临时LoRA路径"""
        lora_dir = tmp_path / "test_lora"
        lora_dir.mkdir()
        # 创建必要的配置文件（模拟LoRA权重）
        (lora_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        return str(lora_dir)

    @patch('models.language_model.AutoModelForCausalLM')
    @patch('models.language_model.AutoTokenizer')
    @patch('models.language_model.get_device_config')
    @patch('models.language_model.get_path_config')
    def test_init_with_default_path(self, mock_path_cfg, mock_device_cfg,
                                    mock_tokenizer, mock_model, mock_model_path):
        """测试使用默认路径初始化"""
        # 配置mock
        mock_path_cfg.return_value.QWEN_7B_CHAT_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"

        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        # 初始化模型
        model = LanguageModel(use_4bit=False)

        # 验证
        assert model.model_path == mock_model_path
        assert model.device == "cpu"
        assert model.use_4bit is False
        assert model.model is not None
        assert model.tokenizer is not None

    @patch('models.language_model.AutoModelForCausalLM')
    @patch('models.language_model.AutoTokenizer')
    @patch('models.language_model.get_device_config')
    def test_init_with_custom_path(self, mock_device_cfg, mock_tokenizer,
                                   mock_model, mock_model_path):
        """测试使用自定义路径初始化"""
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = LanguageModel(model_path=mock_model_path, use_4bit=False)

        assert model.model_path == mock_model_path

    @patch('models.language_model.AutoModelForCausalLM')
    @patch('models.language_model.AutoTokenizer')
    @patch('models.language_model.get_device_config')
    @patch('models.language_model.get_path_config')
    def test_load_lora_success(self, mock_path_cfg, mock_device_cfg,
                               mock_tokenizer, mock_model,
                               mock_model_path, mock_lora_path):
        """测试成功加载LoRA"""
        mock_path_cfg.return_value.QWEN_7B_CHAT_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_tokenizer.from_pretrained.return_value = Mock()

        base_model_mock = Mock()
        mock_model.from_pretrained.return_value = base_model_mock

        model = LanguageModel(use_4bit=False)

        # Mock PeftModel
        with patch('models.language_model.PeftModel') as mock_peft:
            mock_peft.from_pretrained.return_value = Mock()

            result = model.load_lora_from_path(mock_lora_path)

            assert result is True
            assert model.is_lora_loaded is True
            assert model.current_lora_path == mock_lora_path

    @patch('models.language_model.AutoModelForCausalLM')
    @patch('models.language_model.AutoTokenizer')
    @patch('models.language_model.get_device_config')
    @patch('models.language_model.get_path_config')
    def test_load_lora_nonexistent_path(self, mock_path_cfg, mock_device_cfg,
                                        mock_tokenizer, mock_model, mock_model_path):
        """测试加载不存在的LoRA路径"""
        mock_path_cfg.return_value.QWEN_7B_CHAT_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = LanguageModel(use_4bit=False)

        result = model.load_lora_from_path("/nonexistent/path")

        assert result is False
        assert model.is_lora_loaded is False

    @patch('models.language_model.AutoModelForCausalLM')
    @patch('models.language_model.AutoTokenizer')
    @patch('models.language_model.get_device_config')
    @patch('models.language_model.get_path_config')
    def test_unload_lora(self, mock_path_cfg, mock_device_cfg,
                         mock_tokenizer, mock_model,
                         mock_model_path, mock_lora_path):
        """测试卸载LoRA"""
        mock_path_cfg.return_value.QWEN_7B_CHAT_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = LanguageModel(use_4bit=False)

        # 先加载LoRA
        with patch('models.language_model.PeftModel') as mock_peft:
            peft_model_mock = Mock()
            peft_model_mock.unload.return_value = Mock()
            mock_peft.from_pretrained.return_value = peft_model_mock

            model.load_lora_from_path(mock_lora_path)
            assert model.is_lora_loaded is True

            # 卸载LoRA
            result = model.unload_lora()

            assert result is True
            assert model.is_lora_loaded is False
            assert model.current_lora_path is None

    @patch('models.language_model.AutoModelForCausalLM')
    @patch('models.language_model.AutoTokenizer')
    @patch('models.language_model.get_device_config')
    @patch('models.language_model.get_path_config')
    def test_generate(self, mock_path_cfg, mock_device_cfg,
                      mock_tokenizer, mock_model, mock_model_path):
        """测试文本生成"""
        mock_path_cfg.return_value.QWEN_7B_CHAT_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"

        # Mock tokenizer
        tokenizer_mock = Mock()
        tokenizer_mock.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        tokenizer_mock.pad_token_id = 0
        tokenizer_mock.eos_token_id = 2
        tokenizer_mock.convert_tokens_to_ids.return_value = 3
        tokenizer_mock.unk_token_id = 100
        tokenizer_mock.decode.return_value = "Generated text"

        mock_tokenizer.from_pretrained.return_value = tokenizer_mock

        # Mock model
        model_mock = Mock()
        model_mock.device = "cpu"
        model_mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.from_pretrained.return_value = model_mock

        model = LanguageModel(use_4bit=False)

        result = model.generate("Test prompt", max_new_tokens=50)

        assert isinstance(result, str)
        assert len(result) > 0

    @patch('models.language_model.AutoModelForCausalLM')
    @patch('models.language_model.AutoTokenizer')
    @patch('models.language_model.get_device_config')
    @patch('models.language_model.get_path_config')
    def test_get_lora_status(self, mock_path_cfg, mock_device_cfg,
                             mock_tokenizer, mock_model, mock_model_path):
        """测试获取LoRA状态"""
        mock_path_cfg.return_value.QWEN_7B_CHAT_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = LanguageModel(use_4bit=False)

        status = model.get_lora_status()

        assert isinstance(status, dict)
        assert 'is_loaded' in status
        assert 'current_path' in status
        assert 'base_model' in status
        assert status['is_loaded'] is False


class TestInstructionGenerator:
    """测试 InstructionGenerator 类"""

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """创建临时模型路径"""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        return str(model_dir)

    @patch('models.language_model.LanguageModel')
    def test_init(self, mock_language_model):
        """测试初始化"""
        mock_language_model.return_value = Mock()

        generator = InstructionGenerator(use_4bit=False)

        assert generator.language_model is not None

    @patch('models.language_model.LanguageModel')
    @patch('models.language_model.get_path_config')
    def test_load_expert_by_name(self, mock_path_cfg, mock_language_model, tmp_path):
        """测试通过专家名称加载LoRA"""
        # 创建临时LoRA路径
        lora_dir = tmp_path / "text_expert"
        lora_dir.mkdir()

        mock_path_cfg.return_value.get_expert_weight_path.return_value = lora_dir

        lang_model_mock = Mock()
        lang_model_mock.load_lora_from_path.return_value = True
        mock_language_model.return_value = lang_model_mock

        generator = InstructionGenerator(use_4bit=False)
        result = generator.load_expert('text_expert')

        assert result is True

    @patch('models.language_model.LanguageModel')
    def test_unload_expert(self, mock_language_model):
        """测试卸载专家"""
        lang_model_mock = Mock()
        lang_model_mock.unload_lora.return_value = True
        mock_language_model.return_value = lang_model_mock

        generator = InstructionGenerator(use_4bit=False)
        result = generator.unload_expert()

        assert result is True

    @patch('models.language_model.LanguageModel')
    def test_generate(self, mock_language_model):
        """测试生成指令"""
        lang_model_mock = Mock()
        lang_model_mock.generate.return_value = "Generated instruction"
        mock_language_model.return_value = lang_model_mock

        generator = InstructionGenerator(use_4bit=False)
        result = generator.generate("Test prompt")

        assert result == "Generated instruction"

    @patch('models.language_model.LanguageModel')
    def test_get_expert_status(self, mock_language_model):
        """测试获取专家状态"""
        lang_model_mock = Mock()
        lang_model_mock.get_lora_status.return_value = {
            'is_loaded': False,
            'current_path': None,
            'base_model': '/path/to/model'
        }
        mock_language_model.return_value = lang_model_mock

        generator = InstructionGenerator(use_4bit=False)
        status = generator.get_expert_status()

        assert isinstance(status, dict)
        assert 'is_loaded' in status


# ==================== 运行测试 ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])