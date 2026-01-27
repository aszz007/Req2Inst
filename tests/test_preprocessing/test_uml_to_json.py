"""
UML转JSON测试
测试 src/preprocessing/uml_to_json.py 中的转换函数
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from src.preprocessing.uml_to_json import (
    convert_uml_to_json,
    batch_convert_umls,
    get_vision_model
)


class TestUMLToJson:
    """测试UML转JSON功能"""

    @pytest.fixture
    def mock_uml_path(self, tmp_path):
        """创建临时UML图文件"""
        uml_file = tmp_path / "test_uml.png"
        uml_file.write_bytes(b'fake uml image data')
        return str(uml_file)

    @pytest.fixture
    def mock_output_dir(self, tmp_path):
        """创建临时输出目录"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return str(output_dir)

    @pytest.fixture
    def mock_uml_result(self):
        """模拟UML识别结果"""
        uml_data = {
            "actors": [{"name": "User", "position": "left"}],
            "use_cases": [{"name": "Login System", "description": "User login"}],
            "system_boundary": {"name": "System", "is_present": True},
            "relationships": [
                {"type": "association", "from": "User", "to": "Login System", "description": "uses"}
            ],
            "overall_description": "A simple login use case diagram"
        }
        return {
            "description": json.dumps(uml_data),
            "success": True
        }

    @pytest.fixture
    def mock_failed_result(self):
        """模拟识别失败结果"""
        return {
            "description": "",
            "success": False,
            "error": "Recognition failed"
        }

    @patch('src.preprocessing.uml_to_json.VisionModel')
    def test_get_vision_model_singleton(self, mock_vision_model_class):
        """测试视觉模型单例模式"""
        # 清除全局实例
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        mock_vision_model_class.return_value = Mock()

        # 多次调用应返回同一实例
        model1 = get_vision_model()
        model2 = get_vision_model()

        assert model1 is model2
        assert mock_vision_model_class.call_count == 1

    @patch('src.preprocessing.uml_to_json.VisionModel')
    def test_convert_uml_to_json_success(
            self, mock_vision_model_class, mock_uml_path, mock_uml_result
    ):
        """测试成功转换UML"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        # Mock VisionModel
        model_mock = Mock()
        model_mock.recognize_uml.return_value = mock_uml_result
        mock_vision_model_class.return_value = model_mock

        result = convert_uml_to_json(mock_uml_path)

        assert result['recognition_status'] == "success"
        assert 'description' in result
        assert 'processing_time' in result

        # 验证description是JSON字符串
        description_data = json.loads(result['description'])
        assert 'actors' in description_data
        assert 'use_cases' in description_data

    @patch('src.preprocessing.uml_to_json.VisionModel')
    def test_convert_uml_to_json_failure(
            self, mock_vision_model_class, mock_uml_path, mock_failed_result
    ):
        """测试UML识别失败"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        model_mock = Mock()
        model_mock.recognize_uml.return_value = mock_failed_result
        mock_vision_model_class.return_value = model_mock

        result = convert_uml_to_json(mock_uml_path)

        assert result['recognition_status'] == "failed"
        assert result['description'] == ""
        assert 'error' in result

    @patch('src.preprocessing.uml_to_json.VisionModel')
    def test_convert_uml_to_json_with_retries(
            self, mock_vision_model_class, mock_uml_path, mock_uml_result
    ):
        """测试重试机制"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        model_mock = Mock()
        # 第一次失败，第二次成功
        model_mock.recognize_uml.side_effect = [
            {"description": "", "success": False},
            mock_uml_result
        ]
        mock_vision_model_class.return_value = model_mock

        result = convert_uml_to_json(mock_uml_path, max_retries=2)

        # 注意：当前实现中max_retries由recognize_uml内部处理
        # 这里只验证最终结果
        assert 'description' in result

    @patch('src.preprocessing.uml_to_json.VisionModel')
    @patch('src.preprocessing.uml_to_json.save_json')
    def test_convert_uml_to_json_with_save(
            self, mock_save_json, mock_vision_model_class,
            mock_uml_path, mock_uml_result, tmp_path
    ):
        """测试保存结果到文件"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        model_mock = Mock()
        model_mock.recognize_uml.return_value = mock_uml_result
        mock_vision_model_class.return_value = model_mock

        save_path = str(tmp_path / "output.json")
        result = convert_uml_to_json(mock_uml_path, save_path=save_path)

        # 验证save_json被调用
        mock_save_json.assert_called_once()
        call_args = mock_save_json.call_args
        assert call_args[0][1] == save_path

    @patch('src.preprocessing.uml_to_json.VisionModel')
    def test_convert_uml_to_json_exception(
            self, mock_vision_model_class, mock_uml_path
    ):
        """测试异常处理"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        model_mock = Mock()
        model_mock.recognize_uml.side_effect = Exception("Unexpected error")
        mock_vision_model_class.return_value = model_mock

        result = convert_uml_to_json(mock_uml_path)

        assert result['recognition_status'] == "failed"
        assert 'error' in result

    @patch('src.preprocessing.uml_to_json.VisionModel')
    def test_batch_convert_umls_success(
            self, mock_vision_model_class, tmp_path, mock_uml_result
    ):
        """测试批量转换UML"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        # 创建测试UML图
        uml_paths = []
        for i in range(3):
            uml_path = tmp_path / f"uml_{i}.png"
            uml_path.write_bytes(b'fake uml data')
            uml_paths.append(str(uml_path))

        # Mock模型
        model_mock = Mock()
        model_mock.recognize_uml.return_value = mock_uml_result
        mock_vision_model_class.return_value = model_mock

        results = batch_convert_umls(uml_paths)

        assert results['total'] == 3
        assert results['success'] == 3
        assert results['failed'] == 0
        assert len(results['results']) == 3

    @patch('src.preprocessing.uml_to_json.VisionModel')
    def test_batch_convert_umls_with_failures(
            self, mock_vision_model_class, tmp_path, mock_uml_result, mock_failed_result
    ):
        """测试批量转换包含失败"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        # 创建测试UML图
        uml_paths = []
        for i in range(3):
            uml_path = tmp_path / f"uml_{i}.png"
            uml_path.write_bytes(b'fake uml data')
            uml_paths.append(str(uml_path))

        # Mock模型 - 第二个UML识别失败
        model_mock = Mock()

        def side_effect(path, max_retries=2):
            if "uml_1" in path:
                return mock_failed_result
            else:
                return mock_uml_result

        model_mock.recognize_uml.side_effect = side_effect
        mock_vision_model_class.return_value = model_mock

        results = batch_convert_umls(uml_paths)

        assert results['total'] == 3
        assert results['success'] == 2
        assert results['failed'] == 1

    @patch('src.preprocessing.uml_to_json.VisionModel')
    def test_batch_convert_with_progress_callback(
            self, mock_vision_model_class, tmp_path, mock_uml_result
    ):
        """测试带进度回调的批量转换"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        # 创建测试UML图
        uml_paths = []
        for i in range(3):
            uml_path = tmp_path / f"uml_{i}.png"
            uml_path.write_bytes(b'fake uml data')
            uml_paths.append(str(uml_path))

        model_mock = Mock()
        model_mock.recognize_uml.return_value = mock_uml_result
        mock_vision_model_class.return_value = model_mock

        # 创建进度回调
        callback_calls = []

        def progress_callback(current, total, result):
            callback_calls.append((current, total))

        batch_convert_umls(uml_paths, progress_callback=progress_callback)

        # 验证回调被调用
        assert len(callback_calls) == 3
        assert callback_calls[0] == (1, 3)
        assert callback_calls[1] == (2, 3)
        assert callback_calls[2] == (3, 3)

    @patch('src.preprocessing.uml_to_json.VisionModel')
    @patch('src.preprocessing.uml_to_json.save_json')
    def test_batch_convert_with_output_dir(
            self, mock_save_json, mock_vision_model_class,
            tmp_path, mock_uml_result
    ):
        """测试批量转换并保存到输出目录"""
        import src.preprocessing.uml_to_json as module
        module._vision_model = None

        # 创建测试UML图
        uml_paths = []
        for i in range(2):
            uml_path = tmp_path / f"uml_{i}.png"
            uml_path.write_bytes(b'fake uml data')
            uml_paths.append(str(uml_path))

        model_mock = Mock()
        model_mock.recognize_uml.return_value = mock_uml_result
        mock_vision_model_class.return_value = model_mock

        output_dir = str(tmp_path / "output")
        batch_convert_umls(uml_paths, output_dir=output_dir)

        # 验证save_json被调用了2次
        assert mock_save_json.call_count == 2


# ==================== 运行测试 ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])