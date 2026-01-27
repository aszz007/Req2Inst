"""
图像转JSON测试
测试 src/preprocessing/image_to_json.py 中的转换函数
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.preprocessing.image_to_json import (
    convert_image_to_json,
    batch_convert_images,
    get_vision_model
)


class TestImageToJson:
    """测试图像转JSON功能"""

    @pytest.fixture
    def mock_image_path(self, tmp_path):
        """创建临时图像文件"""
        image_file = tmp_path / "test_image.jpg"
        image_file.write_bytes(b'fake image data')
        return str(image_file)

    @pytest.fixture
    def mock_output_dir(self, tmp_path):
        """创建临时输出目录"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return str(output_dir)

    @pytest.fixture
    def mock_recognition_result(self):
        """模拟识别结果"""
        return {
            "description": "A colorful bento box meal",
            "details": {
                "objects": ["bento box", "rice", "vegetables"],
                "scene": "food photography",
                "spatial_info": "centered composition"
            },
            "confidence": 0.95,
            "recognition_status": "success"
        }

    @patch('src.preprocessing.image_to_json.VisionModel')
    def test_get_vision_model_singleton(self, mock_vision_model_class):
        """测试视觉模型单例模式"""
        # 清除全局实例
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        mock_vision_model_class.return_value = Mock()

        # 第一次调用
        model1 = get_vision_model()
        # 第二次调用
        model2 = get_vision_model()

        # 应该返回同一个实例
        assert model1 is model2
        # 只应该初始化一次
        assert mock_vision_model_class.call_count == 1

    @patch('src.preprocessing.image_to_json.VisionModel')
    def test_convert_image_to_json_success(self, mock_vision_model_class,
                                           mock_image_path, mock_recognition_result):
        """测试成功转换图像"""
        # 清除全局实例
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        # Mock VisionModel
        model_mock = Mock()
        model_mock.recognize_image.return_value = mock_recognition_result
        mock_vision_model_class.return_value = model_mock

        result = convert_image_to_json(mock_image_path)

        assert result['description'] == "A colorful bento box meal"
        assert result['confidence'] == 0.95
        assert result['recognition_status'] == "success"
        assert 'processing_time' in result

    @patch('src.preprocessing.image_to_json.VisionModel')
    def test_convert_image_to_json_without_processing_time(
            self, mock_vision_model_class, mock_image_path, mock_recognition_result
    ):
        """测试不返回处理时间"""
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        model_mock = Mock()
        model_mock.recognize_image.return_value = mock_recognition_result
        mock_vision_model_class.return_value = model_mock

        result = convert_image_to_json(mock_image_path, return_processing_time=False)

        assert 'processing_time' not in result

    @patch('src.preprocessing.image_to_json.VisionModel')
    @patch('src.preprocessing.image_to_json.save_json')
    def test_convert_image_to_json_with_save(
            self, mock_save_json, mock_vision_model_class,
            mock_image_path, mock_recognition_result, tmp_path
    ):
        """测试保存结果到文件"""
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        model_mock = Mock()
        model_mock.recognize_image.return_value = mock_recognition_result
        mock_vision_model_class.return_value = model_mock

        save_path = str(tmp_path / "output.json")
        result = convert_image_to_json(mock_image_path, save_path=save_path)

        # 验证save_json被调用
        mock_save_json.assert_called_once()
        call_args = mock_save_json.call_args
        assert call_args[0][1] == save_path

    @patch('src.preprocessing.image_to_json.VisionModel')
    def test_convert_image_to_json_failure(self, mock_vision_model_class, mock_image_path):
        """测试识别失败的情况"""
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        # Mock识别失败
        model_mock = Mock()
        model_mock.recognize_image.side_effect = Exception("Recognition failed")
        mock_vision_model_class.return_value = model_mock

        result = convert_image_to_json(mock_image_path)

        assert result['recognition_status'] == "failed"
        assert 'error' in result
        assert result['confidence'] == 0.0

    @patch('src.preprocessing.image_to_json.VisionModel')
    def test_batch_convert_images_success(
            self, mock_vision_model_class, tmp_path, mock_recognition_result
    ):
        """测试批量转换图像"""
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        # 创建测试图像
        image_paths = []
        for i in range(3):
            img_path = tmp_path / f"image_{i}.jpg"
            img_path.write_bytes(b'fake image data')
            image_paths.append(str(img_path))

        # Mock模型
        model_mock = Mock()
        model_mock.recognize_image.return_value = mock_recognition_result
        mock_vision_model_class.return_value = model_mock

        # 批量转换
        results = batch_convert_images(image_paths)

        assert results['total'] == 3
        assert results['success'] == 3
        assert results['failed'] == 0
        assert len(results['results']) == 3

    @patch('src.preprocessing.image_to_json.VisionModel')
    def test_batch_convert_images_with_failures(self, mock_vision_model_class, tmp_path):
        """测试批量转换包含失败的情况"""
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        # 创建测试图像
        image_paths = []
        for i in range(3):
            img_path = tmp_path / f"image_{i}.jpg"
            img_path.write_bytes(b'fake image data')
            image_paths.append(str(img_path))

        # Mock模型 - 第二张图像识别失败
        model_mock = Mock()

        def side_effect(path):
            if "image_1" in path:
                return {
                    "recognition_status": "failed",
                    "confidence": 0.0,
                    "error": "Failed"
                }
            else:
                return {
                    "recognition_status": "success",
                    "confidence": 0.95,
                    "description": "Test"
                }

        model_mock.recognize_image.side_effect = side_effect
        mock_vision_model_class.return_value = model_mock

        results = batch_convert_images(image_paths)

        assert results['total'] == 3
        assert results['success'] == 2
        assert results['failed'] == 1

    @patch('src.preprocessing.image_to_json.VisionModel')
    def test_batch_convert_with_progress_callback(
            self, mock_vision_model_class, tmp_path, mock_recognition_result
    ):
        """测试带进度回调的批量转换"""
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        # 创建测试图像
        image_paths = []
        for i in range(3):
            img_path = tmp_path / f"image_{i}.jpg"
            img_path.write_bytes(b'fake image data')
            image_paths.append(str(img_path))

        model_mock = Mock()
        model_mock.recognize_image.return_value = mock_recognition_result
        mock_vision_model_class.return_value = model_mock

        # 创建进度回调
        callback_calls = []

        def progress_callback(current, total, result):
            callback_calls.append((current, total))

        batch_convert_images(image_paths, progress_callback=progress_callback)

        # 验证回调被调用
        assert len(callback_calls) == 3
        assert callback_calls[0] == (1, 3)
        assert callback_calls[1] == (2, 3)
        assert callback_calls[2] == (3, 3)

    @patch('src.preprocessing.image_to_json.VisionModel')
    @patch('src.preprocessing.image_to_json.save_json')
    def test_batch_convert_with_output_dir(
            self, mock_save_json, mock_vision_model_class,
            tmp_path, mock_recognition_result
    ):
        """测试批量转换并保存到输出目录"""
        import src.preprocessing.image_to_json as module
        module._vision_model = None

        # 创建测试图像
        image_paths = []
        for i in range(2):
            img_path = tmp_path / f"image_{i}.jpg"
            img_path.write_bytes(b'fake image data')
            image_paths.append(str(img_path))

        model_mock = Mock()
        model_mock.recognize_image.return_value = mock_recognition_result
        mock_vision_model_class.return_value = model_mock

        output_dir = str(tmp_path / "output")
        batch_convert_images(image_paths, output_dir=output_dir)

        # 验证save_json被调用了2次
        assert mock_save_json.call_count == 2


# ==================== 运行测试 ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])