"""
图像转JSON处理函数
提供可复用的图像处理接口
作者：Preprocessing System
日期：2025-01-26
"""

from pathlib import Path
from typing import Dict, Optional
import time

from models.vision_model import VisionModel
from src.utils.logger import get_logger

logger = get_logger('preprocessing.image_to_json')

# 全局模型实例（避免重复加载）
_vision_model = None


def get_vision_model() -> VisionModel:
    """获取视觉模型单例（图像识别使用Qwen3-VL-8B）"""
    global _vision_model
    if _vision_model is None:
        logger.info("初始化视觉模型（Qwen3-VL-8B for image recognition）...")
        _vision_model = VisionModel(version='qwen3')
    return _vision_model


def convert_image_to_json(
    image_path: str,
    save_path: Optional[str] = None,
    return_processing_time: bool = True
) -> Dict:
    """
    将单张图像转换为JSON描述

    Args:
        image_path: 图像路径
        save_path: 保存路径（可选，如果提供则保存JSON文件）
        return_processing_time: 是否返回处理时间

    Returns:
        dict: 包含description, details, confidence等字段的JSON

    Example:
        >>> result = convert_image_to_json("path/to/image.jpg")
        >>> print(result['description'])
    """
    logger.info(f"处理图像: {Path(image_path).name}")

    try:
        start_time = time.time()

        # 获取模型并识别
        model = get_vision_model()
        result = model.recognize_image(image_path)

        # 添加处理时间
        if return_processing_time:
            result['processing_time'] = round(time.time() - start_time, 2)

        # 保存文件（如果指定）
        if save_path:
            from src.utils.file_utils import save_json
            save_json(result, save_path)
            logger.info(f"结果已保存至: {save_path}")

        return result

    except Exception as e:
        logger.error(f"图像处理失败: {e}")
        return {
            "description": "",
            "details": {"objects": [], "scene": "unknown", "spatial_info": ""},
            "confidence": 0.0,
            "recognition_status": "failed",
            "error": str(e)
        }


def batch_convert_images(
    image_paths: list,
    output_dir: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    批量转换图像（简化版，供脚本调用）

    Args:
        image_paths: 图像路径列表
        output_dir: 输出目录（可选）
        progress_callback: 进度回调函数 callback(current, total, result)

    Returns:
        dict: 统计信息 {success: int, failed: int, results: list}
    """
    results = []
    success = 0
    failed = 0

    for idx, img_path in enumerate(image_paths, 1):
        result = convert_image_to_json(
            img_path,
            save_path=f"{output_dir}/{Path(img_path).stem}.json" if output_dir else None
        )

        if result['recognition_status'] == 'success':
            success += 1
        else:
            failed += 1

        results.append(result)

        if progress_callback:
            progress_callback(idx, len(image_paths), result)

    return {
        'success': success,
        'failed': failed,
        'total': len(image_paths),
        'results': results
    }