"""
视觉模型测试
测试 models/vision_model.py 中的 VisionModel 类
"""

import pytest
import torch
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from models.vision_model import VisionModel


class TestVisionModel:
    """测试 VisionModel 类"""

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """创建临时模型路径"""
        model_dir = tmp_path / "test_vision_model"
        model_dir.mkdir()
        return str(model_dir)

    @pytest.fixture
    def mock_image_path(self, tmp_path):
        """创建临时图像文件"""
        image_file = tmp_path / "test_image.jpg"
        image_file.write_bytes(b'fake image data')
        return str(image_file)

    @pytest.fixture
    def mock_lora_path(self, tmp_path):
        """创建临时LoRA路径"""
        lora_dir = tmp_path / "test_lora"
        lora_dir.mkdir()
        (lora_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        return str(lora_dir)

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    def test_init(self, mock_path_cfg, mock_device_cfg,
                  mock_processor, mock_model, mock_model_path):
        """测试模型初始化"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"

        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = VisionModel()

        assert model.model_path == mock_model_path
        assert model.device == "cpu"
        assert model.model is not None
        assert model.processor is not None

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    def test_load_lora_success(self, mock_path_cfg, mock_device_cfg,
                               mock_processor, mock_model,
                               mock_model_path, mock_lora_path):
        """测试成功加载LoRA"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = VisionModel()

        with patch('models.vision_model.PeftModel') as mock_peft:
            mock_peft.from_pretrained.return_value = Mock()

            result = model.load_lora_from_path(mock_lora_path)

            assert result is True
            assert model.is_lora_loaded is True
            assert model.current_lora_path == mock_lora_path

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    def test_load_lora_nonexistent_path(self, mock_path_cfg, mock_device_cfg,
                                        mock_processor, mock_model, mock_model_path):
        """测试加载不存在的LoRA路径"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = VisionModel()

        result = model.load_lora_from_path("/nonexistent/path")

        assert result is False
        assert model.is_lora_loaded is False

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    def test_unload_lora(self, mock_path_cfg, mock_device_cfg,
                         mock_processor, mock_model,
                         mock_model_path, mock_lora_path):
        """测试卸载LoRA"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = VisionModel()

        # 先加载LoRA
        with patch('models.vision_model.PeftModel') as mock_peft:
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

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    @patch('models.vision_model.process_vision_info')
    def test_generate_text_only(self, mock_vision_info, mock_path_cfg,
                                mock_device_cfg, mock_processor, mock_model,
                                mock_model_path):
        """测试纯文本生成"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"

        # Mock processor
        processor_mock = Mock()
        processor_mock.apply_chat_template.return_value = "formatted text"
        processor_mock.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        processor_mock.tokenizer.pad_token_id = 0
        processor_mock.tokenizer.eos_token_id = 2
        processor_mock.batch_decode.return_value = ["Generated response"]
        mock_processor.from_pretrained.return_value = processor_mock

        # Mock model
        model_mock = Mock()
        model_mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.from_pretrained.return_value = model_mock

        # Mock vision info
        mock_vision_info.return_value = (None, None)

        model = VisionModel()
        result = model.generate("Test prompt")

        assert isinstance(result, str)
        assert len(result) > 0

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    @patch('models.vision_model.process_vision_info')
    def test_recognize_image(self, mock_vision_info, mock_path_cfg,
                             mock_device_cfg, mock_processor, mock_model,
                             mock_model_path, mock_image_path):
        """测试图像识别"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"

        # Mock processor
        processor_mock = Mock()
        processor_mock.apply_chat_template.return_value = "formatted text"
        processor_mock.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        processor_mock.tokenizer.pad_token_id = 0
        processor_mock.tokenizer.eos_token_id = 2

        # 模拟JSON响应
        json_response = {
            "description": "A test image",
            "details": {
                "objects": ["object1", "object2"],
                "scene": "test scene",
                "spatial_info": "center"
            }
        }
        processor_mock.batch_decode.return_value = [json.dumps(json_response)]
        mock_processor.from_pretrained.return_value = processor_mock

        # Mock model
        model_mock = Mock()
        output_mock = Mock()
        output_mock.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        output_mock.scores = [torch.randn(1, 32000)]  # Mock logits
        model_mock.generate.return_value = output_mock
        mock_model.from_pretrained.return_value = model_mock

        mock_vision_info.return_value = (None, None)

        model = VisionModel()
        result = model.recognize_image(mock_image_path)

        assert isinstance(result, dict)
        assert 'description' in result
        assert 'confidence' in result
        assert 'recognition_status' in result

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    @patch('models.vision_model.process_vision_info')
    def test_recognize_uml(self, mock_vision_info, mock_path_cfg,
                           mock_device_cfg, mock_processor, mock_model,
                           mock_model_path, mock_image_path):
        """测试UML识别"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"

        # Mock processor
        processor_mock = Mock()
        processor_mock.apply_chat_template.return_value = "formatted text"
        processor_mock.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        processor_mock.tokenizer.pad_token_id = 0
        processor_mock.tokenizer.eos_token_id = 2

        # 模拟UML JSON响应
        uml_response = {
            "actors": [{"name": "User", "position": "left"}],
            "use_cases": [{"name": "Login", "description": "User login"}],
            "system_boundary": {"name": "System", "is_present": True},
            "relationships": [],
            "overall_description": "A UML diagram"
        }
        processor_mock.batch_decode.return_value = [json.dumps(uml_response)]
        mock_processor.from_pretrained.return_value = processor_mock

        # Mock model
        model_mock = Mock()
        model_mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.from_pretrained.return_value = model_mock

        mock_vision_info.return_value = (None, None)

        model = VisionModel()
        result = model.recognize_uml(mock_image_path)

        assert isinstance(result, dict)
        assert 'description' in result
        assert 'success' in result

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    def test_get_lora_status(self, mock_path_cfg, mock_device_cfg,
                             mock_processor, mock_model, mock_model_path):
        """测试获取LoRA状态"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = VisionModel()
        status = model.get_lora_status()

        assert isinstance(status, dict)
        assert 'is_loaded' in status
        assert 'current_path' in status
        assert 'base_model' in status
        assert status['is_loaded'] is False

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    def test_parse_image_response_valid_json(self, mock_path_cfg, mock_device_cfg,
                                             mock_processor, mock_model, mock_model_path):
        """测试解析有效的图像JSON响应"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = VisionModel()

        json_str = '{"description": "test", "details": {"objects": []}}'
        result = model._parse_image_response(json_str, "test.jpg")

        assert 'description' in result
        assert 'details' in result

    @patch('models.vision_model.AutoModelForVision2Seq')
    @patch('models.vision_model.AutoProcessor')
    @patch('models.vision_model.get_device_config')
    @patch('models.vision_model.get_path_config')
    def test_fix_truncated_json(self, mock_path_cfg, mock_device_cfg,
                                mock_processor, mock_model, mock_model_path):
        """测试修复截断的JSON"""
        mock_path_cfg.return_value.QWEN_VL_7B_PATH = mock_model_path
        mock_device_cfg.return_value.get_device.return_value = "cpu"
        mock_processor.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = VisionModel()

        # 测试缺少闭合括号的JSON
        truncated = '{"actors": [{"name": "User"'
        fixed = model._fix_truncated_json(truncated)

        assert fixed.count('{') == fixed.count('}')
        assert fixed.count('[') == fixed.count(']')


# ==================== 运行测试 ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])