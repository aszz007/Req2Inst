"""
UML转JSON处理函数
提供可复用的UML处理接口
作者：Preprocessing System
日期：2025-01-26
"""

from pathlib import Path
from typing import Dict, Optional
import time

from models.vision_model import VisionModel
from src.utils.logger import get_logger

logger = get_logger('preprocessing.uml_to_json')

# 全局模型实例
_vision_model = None


def get_vision_model() -> VisionModel:
    """获取视觉模型单例（UML识别使用Qwen3-VL-8B）"""
    global _vision_model
    if _vision_model is None:
        logger.info("初始化视觉模型（Qwen3-VL-8B for UML recognition）...")
        _vision_model = VisionModel(version='qwen3')
    return _vision_model


def convert_uml_to_json(
    uml_path: str,
    save_path: Optional[str] = None,
    max_retries: int = 2
) -> Dict:
    """
    将UML图转换为JSON描述

    Args:
        uml_path: UML图路径
        save_path: 保存路径（可选）
        max_retries: 最大重试次数

    Returns:
        dict: 包含description字段的JSON（UML数据集格式）

    Example:
        >>> result = convert_uml_to_json("path/to/uml.jpg")
        >>> print(result['description'])  # JSON字符串
    """
    logger.info(f"处理UML图: {Path(uml_path).name}")

    try:
        start_time = time.time()

        # 获取模型并识别
        model = get_vision_model()
        result = model.recognize_uml(uml_path, max_retries=max_retries)

        processing_time = round(time.time() - start_time, 2)

        # 构建返回结果（匹配UML数据集格式）
        if result.get('success', False):
            output_data = {
                "description": result['description'],  # 纯JSON字符串
                "processing_time": processing_time,
                "recognition_status": "success"
            }
        else:
            output_data = {
                "description": "",
                "processing_time": processing_time,
                "recognition_status": "failed",
                "error": result.get('error', '未知错误')
            }

        # 保存文件（如果指定）
        if save_path:
            from src.utils.file_utils import save_json
            save_json({"description": output_data["description"]}, save_path)
            logger.info(f"结果已保存至: {save_path}")

        return output_data

    except Exception as e:
        logger.error(f"UML处理失败: {e}")
        return {
            "description": "",
            "recognition_status": "failed",
            "error": str(e)
        }


def batch_convert_umls(
    uml_paths: list,
    output_dir: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    批量转换UML图（简化版，供脚本调用）

    Args:
        uml_paths: UML图路径列表
        output_dir: 输出目录（可选）
        progress_callback: 进度回调函数

    Returns:
        dict: 统计信息
    """
    results = []
    success = 0
    failed = 0

    for idx, uml_path in enumerate(uml_paths, 1):
        result = convert_uml_to_json(
            uml_path,
            save_path=f"{output_dir}/{Path(uml_path).stem}.json" if output_dir else None
        )

        if result['recognition_status'] == 'success':
            success += 1
        else:
            failed += 1

        results.append(result)

        if progress_callback:
            progress_callback(idx, len(uml_paths), result)

    return {
        'success': success,
        'failed': failed,
        'total': len(uml_paths),
        'results': results
    }


# 测试代码
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python uml_to_json.py <UML图路径>")
        sys.exit(1)

    result = convert_uml_to_json(sys.argv[1])
    print("\n识别结果:")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))