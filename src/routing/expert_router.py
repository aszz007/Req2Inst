"""
MoE Expert Router - Simple Rule-based Routing with Parameterized Expert Selection
基于规则的简单路由 + 参数化专家选择

路由策略:
1. 根据输入类型(text/image/uml/general)进行简单规则路由
2. 支持通过参数指定专家变体(用于对比实验)
3. 使用最优默认变体(对比实验后确定)

Author: Claude
Date: 2026-02-03
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RoutingResult:
    """路由结果"""
    expert_name: str
    expert_path: str
    expert_type: str
    reasoning: str


@dataclass
class ExpertConfig:
    """专家配置"""
    name: str
    expert_type: str
    model_version: str
    dataset_variant: Optional[str]
    path: str
    is_default: bool


class ExpertRouter:
    """
    MoE专家路由器

    简单的基于规则的路由:
    - 根据输入类型路由到对应的专家类型
    - 支持通过参数选择专家变体(对比实验用)
    - 使用经过实验验证的最优默认变体
    """

    DEFAULT_EXPERTS = {
        'text': 'text_expert',
        'image': 'image_expert',
        'uml': 'uml_expert_dataset_qwen235B',
        'general': 'general_expert_dataset_qwen235B'
    }

    def __init__(self, lora_weights_dir: str = "lora_weights/experts"):
        """
        初始化路由器

        Args:
            lora_weights_dir: LoRA权重根目录
        """
        self.lora_weights_dir = Path(lora_weights_dir)
        self.expert_registry = self._build_expert_registry()
        self.routing_stats = defaultdict(int)

    def _build_expert_registry(self) -> Dict[str, List[ExpertConfig]]:
        """
        构建专家注册表

        包含8个专家变体（符合框架文档）:
        - text_expert: 1个
        - image_expert: 1个（数据集只有1个版本）
        - uml_expert: 3个变体（3个数据集版本：qwen25/qwen3/qwen235B）
        - general_expert: 3个变体（3个数据集版本：qwen25/qwen3/qwen235B）

        重要：所有Expert都基于Qwen-7B-Chat训练，在qwen_text环境执行
        """
        registry = {
            'text': [],
            'image': [],
            'uml': [],
            'general': []
        }

        # Text Expert: 1个
        registry['text'].append(ExpertConfig(
            name='text_expert',
            expert_type='text',
            model_version='qwen-7b',
            dataset_variant=None,
            path=str(self.lora_weights_dir / 'text_expert'),
            is_default=True
        ))

        # Image Expert: 1个（数据集只有1个版本）
        registry['image'].append(ExpertConfig(
            name='image_expert',
            expert_type='image',
            model_version='qwen-7b',
            dataset_variant=None,
            path=str(self.lora_weights_dir / 'image_expert'),
            is_default=True
        ))

        # UML Expert: 3个变体（基于不同视觉模型识别的数据集）
        for dataset in ['qwen25', 'qwen3', 'qwen235B']:
            expert_name = f'uml_expert_dataset_{dataset}'
            is_default = (expert_name == self.DEFAULT_EXPERTS['uml'])
            registry['uml'].append(ExpertConfig(
                name=expert_name,
                expert_type='uml',
                model_version='qwen-7b',
                dataset_variant=dataset,
                path=str(self.lora_weights_dir / expert_name),
                is_default=is_default
            ))

        # General Expert: 3个变体（基于不同视觉模型识别的数据集）
        for dataset in ['qwen25', 'qwen3', 'qwen235B']:
            expert_name = f'general_expert_dataset_{dataset}'
            is_default = (expert_name == self.DEFAULT_EXPERTS['general'])
            registry['general'].append(ExpertConfig(
                name=expert_name,
                expert_type='general',
                model_version='qwen-7b',
                dataset_variant=dataset,
                path=str(self.lora_weights_dir / expert_name),
                is_default=is_default
            ))

        return registry

    def route(
            self,
            input_data: dict,
            expert_variant: Optional[str] = None
    ) -> RoutingResult:
        """
        主路由函数

        Args:
            input_data: 输入数据字典，包含:
                - type: 'text', 'image', 'uml', 'general'
                - content: 实际内容
            expert_variant: 指定专家变体(对比实验用)，例如:
                - 'image_expert_qwen25'
                - 'uml_expert_dataset_qwen3'
                - None表示使用默认专家

        Returns:
            RoutingResult: 路由结果
        """
        input_type = input_data.get('type', '').lower()

        if input_type not in ['text', 'image', 'uml', 'general']:
            input_type = 'general'

        if expert_variant:
            expert = self._get_expert_by_name(expert_variant)
            if expert:
                self.routing_stats[expert.name] += 1
                return RoutingResult(
                    expert_name=expert.name,
                    expert_path=expert.path,
                    expert_type=expert.expert_type,
                    reasoning=f"Explicitly specified expert variant: {expert_variant}"
                )
            else:
                raise ValueError(f"Expert variant '{expert_variant}' not found in registry")

        default_expert_name = self.DEFAULT_EXPERTS[input_type]
        expert = self._get_expert_by_name(default_expert_name)

        self.routing_stats[expert.name] += 1

        return RoutingResult(
            expert_name=expert.name,
            expert_path=expert.path,
            expert_type=expert.expert_type,
            reasoning=f"Default expert for {input_type} input"
        )

    def _get_expert_by_name(self, expert_name: str) -> Optional[ExpertConfig]:
        """根据名称获取专家配置"""
        for expert_list in self.expert_registry.values():
            for expert in expert_list:
                if expert.name == expert_name:
                    return expert
        return None

    def list_experts(
            self,
            expert_type: Optional[str] = None,
            only_defaults: bool = False
    ) -> List[ExpertConfig]:
        """
        列出专家

        Args:
            expert_type: 专家类型过滤('text'/'image'/'uml'/'general')
            only_defaults: 只列出默认专家

        Returns:
            专家配置列表
        """
        if expert_type:
            experts = self.expert_registry.get(expert_type, [])
        else:
            experts = []
            for expert_list in self.expert_registry.values():
                experts.extend(expert_list)

        if only_defaults:
            experts = [e for e in experts if e.is_default]

        return experts

    def get_available_variants(self, expert_type: str) -> List[str]:
        """
        获取指定类型的所有可用专家变体名称

        Args:
            expert_type: 'text', 'image', 'uml', 'general'

        Returns:
            专家变体名称列表
        """
        experts = self.expert_registry.get(expert_type, [])
        return [e.name for e in experts]

    def get_routing_statistics(self) -> Dict:
        """
        获取路由统计信息

        Returns:
            统计字典
        """
        total_routings = sum(self.routing_stats.values())

        stats = {
            'total_routings': total_routings,
            'expert_usage_count': dict(self.routing_stats),
            'expert_usage_percentage': {
                expert: (count / total_routings * 100) if total_routings > 0 else 0
                for expert, count in self.routing_stats.items()
            },
            'total_experts': sum(len(experts) for experts in self.expert_registry.values()),
            'experts_by_type': {
                expert_type: len(expert_list)
                for expert_type, expert_list in self.expert_registry.items()
            }
        }

        return stats

    def reset_statistics(self):
        """重置路由统计"""
        self.routing_stats.clear()