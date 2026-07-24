"""
Soft Router - 基于PEFT加权融合的软路由实现

功能：
  - 加载多个LoRA adapter到同一基础模型
  - 使用 add_weighted_adapter() 进行参数级融合
  - 支持类型感知的soft权重构造（General域）
  - 融合后单次前向推理，显存与单专家相同

依赖：peft >= 0.6.0（add_weighted_adapter 支持）


Date: 2026-03-04
"""

import torch
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from peft import PeftModel

from src.utils.logger import get_logger

logger = get_logger('routing.soft_router')


@dataclass
class SoftRoutingConfig:
    """软路由配置"""
    # 融合比例：specialized专家的权重（general专家权重 = 1 - alpha）
    alpha: float = 0.5
    # 融合模式
    combination_type: str = "linear"


def check_peft_version():
    """
    检查PEFT版本是否支持 add_weighted_adapter

    Returns:
        bool: 是否满足版本要求
    """
    try:
        import peft
        version = peft.__version__
        major, minor = [int(x) for x in version.split('.')[:2]]
        supported = (major > 0) or (major == 0 and minor >= 6)
        if not supported:
            logger.error(f"PEFT版本过低: {version}，需要 >= 0.6.0")
        else:
            logger.info(f"PEFT版本检查通过: {version}")
        return supported
    except Exception as e:
        logger.error(f"PEFT版本检查失败: {e}")
        return False


def _clean_adapter_config(adapter_path: Path) -> Path:
    """
    清理adapter配置中不兼容的参数（与language_model.py逻辑一致）

    Args:
        adapter_path: 原始adapter路径

    Returns:
        Path: 清理后的路径（可能是临时目录）
    """
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        return adapter_path

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    incompatible_params = [
        'alora_invocation_tokens', 'alora_prefix',
        'alora_suffix', 'arrow_config',
    ]

    needs_cleaning = any(p in config for p in incompatible_params)
    if not needs_cleaning:
        return adapter_path

    temp_dir = Path(tempfile.mkdtemp(prefix="lora_soft_router_"))
    for item in adapter_path.iterdir():
        if item.is_file():
            shutil.copy2(item, temp_dir / item.name)

    for param in incompatible_params:
        if param in config:
            del config[param]

    with open(temp_dir / "adapter_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"已清理adapter配置: {adapter_path.name}")
    return temp_dir


class SoftRouter:
    """
    软路由器 - 使用PEFT加权融合多个LoRA adapter

    核心机制：
    1. 将多个专家的LoRA adapter加载到同一基础模型
    2. 使用 add_weighted_adapter() 按门控权重线性融合为merged adapter
    3. 融合后执行单次前向推理，显存开销与单专家完全相同

    使用方式：
        router = SoftRouter(base_model, tokenizer, adapter_paths)
        router.load_all_adapters()
        router.merge_adapters(weights={"text_expert": 0.5, "general_expert": 0.5})
        output = router.generate(prompt)
        router.cleanup()
    """

    # 专家名称列表（固定顺序）
    EXPERT_NAMES = ['text_expert', 'image_expert', 'uml_expert', 'general_expert']

    def __init__(
        self,
        base_model,
        tokenizer,
        adapter_paths: Dict[str, str],
    ):
        """
        初始化软路由器

        Args:
            base_model: 已加载的基础模型（AutoModelForCausalLM）
            tokenizer: 分词器
            adapter_paths: 专家名称到adapter路径的映射
                例如: {"text_expert": "/path/to/text_expert", ...}
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.adapter_paths = adapter_paths
        self.peft_model = None
        self._temp_dirs = []  # 追踪临时目录以便清理
        self._adapters_loaded = False
        self._current_merged_name = None

    def load_all_adapters(self) -> bool:
        """
        加载所有专家adapter到同一模型

        Returns:
            bool: 是否全部加载成功
        """
        try:
            adapter_names = list(self.adapter_paths.keys())
            if not adapter_names:
                logger.error("没有可用的adapter路径")
                return False

            # 加载第一个adapter（创建PeftModel）
            first_name = adapter_names[0]
            first_path = Path(self.adapter_paths[first_name])
            cleaned_path = _clean_adapter_config(first_path)
            if cleaned_path != first_path:
                self._temp_dirs.append(cleaned_path)

            logger.info(f"加载第一个adapter: {first_name} <- {first_path}")
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                str(cleaned_path),
                adapter_name=first_name,
                is_trainable=False,
            )

            # 加载后续adapter
            for name in adapter_names[1:]:
                adapter_path = Path(self.adapter_paths[name])
                cleaned_path = _clean_adapter_config(adapter_path)
                if cleaned_path != adapter_path:
                    self._temp_dirs.append(cleaned_path)

                logger.info(f"加载adapter: {name} <- {adapter_path}")
                self.peft_model.load_adapter(
                    str(cleaned_path),
                    adapter_name=name,
                )

            self._adapters_loaded = True
            logger.info(f"全部 {len(adapter_names)} 个adapter加载完成")
            return True

        except Exception as e:
            logger.error(f"加载adapter失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def merge_adapters(self, weights: Dict[str, float], merged_name: str = "merged") -> bool:
        """
        按指定权重融合adapter

        Args:
            weights: 专家名称到权重的映射
                例如: {"text_expert": 0.5, "general_expert": 0.5}
            merged_name: 融合后adapter的名称

        Returns:
            bool: 是否融合成功
        """
        if not self._adapters_loaded:
            logger.error("adapter未加载，请先调用 load_all_adapters()")
            return False

        try:
            # 如果已有之前的merged adapter，先删除
            if self._current_merged_name is not None:
                try:
                    self.peft_model.delete_adapter(self._current_merged_name)
                except Exception:
                    pass

            adapter_names = list(weights.keys())
            adapter_weights = [weights[name] for name in adapter_names]

            logger.info(f"融合adapter: {dict(zip(adapter_names, adapter_weights))}")

            self.peft_model.add_weighted_adapter(
                adapters=adapter_names,
                weights=adapter_weights,
                adapter_name=merged_name,
                combination_type="linear",
            )

            self.peft_model.set_adapter(merged_name)
            self._current_merged_name = merged_name
            logger.info(f"adapter融合完成，已切换到: {merged_name}")
            return True

        except Exception as e:
            logger.error(f"adapter融合失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def set_single_adapter(self, adapter_name: str) -> bool:
        """
        切换到单个adapter（不融合）

        Args:
            adapter_name: adapter名称

        Returns:
            bool: 是否切换成功
        """
        if not self._adapters_loaded:
            logger.error("adapter未加载")
            return False

        try:
            self.peft_model.set_adapter(adapter_name)
            logger.info(f"已切换到单adapter: {adapter_name}")
            return True
        except Exception as e:
            logger.error(f"切换adapter失败: {e}")
            return False

    def cleanup(self):
        """清理临时目录和资源"""
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")
        self._temp_dirs.clear()

        # 注意：不清理peft_model，由调用方负责模型生命周期
        self._current_merged_name = None
        self._adapters_loaded = False

    def __del__(self):
        """析构时清理临时目录"""
        try:
            self.cleanup()
        except Exception:
            pass


def build_type_aware_weights(
    data_type: str,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    根据General样本的数据来源类型构造soft权重

    策略：
    - general(text源) -> text_expert * alpha + general_expert * (1-alpha)
    - general(image源) -> image_expert * alpha + general_expert * (1-alpha)
    - general(uml源) -> uml_expert * alpha + general_expert * (1-alpha)

    Args:
        data_type: 样本来源类型（"text" / "image" / "uml"）
        alpha: specialized专家的权重

    Returns:
        Dict[str, float]: 专家名称到权重的映射
    """
    type_to_expert = {
        'text': 'text_expert',
        'image': 'image_expert',
        'uml': 'uml_expert',
    }

    specialized_expert = type_to_expert.get(data_type)
    if specialized_expert is None:
        # 未知类型，仅使用general expert
        logger.warning(f"未知数据类型: {data_type}，使用纯general权重")
        return {'general_expert': 1.0}

    return {
        specialized_expert: alpha,
        'general_expert': 1.0 - alpha,
    }


def group_general_samples_by_type(
    test_data: List[Dict],
) -> Dict[str, List[int]]:
    """
    将General测试集样本按data_type分组

    Args:
        test_data: General测试集样本列表

    Returns:
        Dict[str, List[int]]: data_type -> 样本索引列表
    """
    groups = {}
    for idx, sample in enumerate(test_data):
        dt = sample.get('data_type', 'unknown')
        if dt not in groups:
            groups[dt] = []
        groups[dt].append(idx)

    for dt, indices in groups.items():
        logger.info(f"  General样本分组: {dt} -> {len(indices)} 条")

    return groups
