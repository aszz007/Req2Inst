"""
MoE Model - Mixture of Experts for Instruction Generation
MoE模型主类

整合路由器和专家，提供统一的指令生成接口

Date: 2026-02-03
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import json

from .expert_router import ExpertRouter, RoutingResult


class MoEModel:
    """
    MoE指令生成模型

    整合路由系统和专家模型，提供统一的生成接口
    """

    def __init__(
            self,
            lora_weights_dir: str = "lora_weights/experts",
            base_models_dir: str = "base_models"
    ):
        """
        初始化MoE模型

        Args:
            lora_weights_dir: LoRA权重目录
            base_models_dir: 基础模型目录
        """
        self.lora_weights_dir = Path(lora_weights_dir)
        self.base_models_dir = Path(base_models_dir)

        self.router = ExpertRouter(lora_weights_dir)

        self._loaded_experts = {}

    def generate_instruction(
            self,
            input_data: Union[str, dict],
            expert_variant: Optional[str] = None,
            **generation_kwargs
    ) -> Dict:
        """
        生成众包指令

        Args:
            input_data: 输入数据
                - 如果是str: 文本需求
                - 如果是dict: 必须包含'type'和'content'字段
            expert_variant: 指定专家变体(对比实验用)
            **generation_kwargs: 生成参数(max_length, temperature等)

        Returns:
            结果字典，包含:
                - instruction: 生成的指令
                - expert_used: 使用的专家名称
                - expert_type: 专家类型
                - reasoning: 路由理由
        """
        formatted_input = self._format_input(input_data)

        routing_result = self.router.route(
            formatted_input,
            expert_variant=expert_variant
        )

        expert = self._get_or_load_expert(routing_result)

        instruction = expert.generate_instruction(
            formatted_input['content'],
            **generation_kwargs
        )

        return {
            'instruction': instruction,
            'expert_used': routing_result.expert_name,
            'expert_type': routing_result.expert_type,
            'reasoning': routing_result.reasoning
        }

    def _format_input(self, input_data: Union[str, dict]) -> dict:
        """
        格式化输入数据

        Args:
            input_data: 原始输入

        Returns:
            标准化的输入字典
        """
        if isinstance(input_data, str):
            return {
                'type': 'text',
                'content': input_data
            }
        elif isinstance(input_data, dict):
            if 'type' not in input_data or 'content' not in input_data:
                raise ValueError("Input dict must contain 'type' and 'content' fields")
            return input_data
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _get_or_load_expert(self, routing_result: RoutingResult):
        """
        获取或加载专家模型

        Args:
            routing_result: 路由结果

        Returns:
            加载的专家实例
        """
        expert_name = routing_result.expert_name

        if expert_name not in self._loaded_experts:
            self._loaded_experts[expert_name] = self._load_expert(routing_result)

        return self._loaded_experts[expert_name]

    def _load_expert(self, routing_result: RoutingResult):
        """
        加载专家模型

        Args:
            routing_result: 路由结果

        Returns:
            专家实例

        Note:
            根据简化的框架：
            - Text Expert: 1个，无需额外参数
            - Image Expert: 1个，无需额外参数
            - UML Expert: 1个，无需额外参数
            - General Expert: 1个，无需额外参数
        """
        from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert

        expert_type = routing_result.expert_type
        expert_path = routing_result.expert_path

        if expert_type == 'text':
            return TextExpert(lora_path=expert_path)

        elif expert_type == 'image':
            return ImageExpert(lora_path=expert_path)

        elif expert_type == 'uml':
            return UMLExpert(lora_path=expert_path)

        elif expert_type == 'general':
            return GeneralExpert(lora_path=expert_path)

        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

    def batch_generate(
            self,
            input_list: List[Union[str, dict]],
            expert_variant: Optional[str] = None,
            **generation_kwargs
    ) -> List[Dict]:
        """
        批量生成指令

        Args:
            input_list: 输入列表
            expert_variant: 指定专家变体
            **generation_kwargs: 生成参数

        Returns:
            结果列表
        """
        results = []
        for input_data in input_list:
            result = self.generate_instruction(
                input_data,
                expert_variant=expert_variant,
                **generation_kwargs
            )
            results.append(result)
        return results

    def get_router_statistics(self) -> Dict:
        """获取路由统计信息"""
        return self.router.get_routing_statistics()

    def reset_router_statistics(self):
        """重置路由统计"""
        self.router.reset_statistics()

    def list_available_experts(
            self,
            expert_type: Optional[str] = None
    ) -> List[str]:
        """
        列出可用的专家

        Args:
            expert_type: 专家类型筛选

        Returns:
            专家名称列表
        """
        if expert_type:
            return self.router.get_available_variants(expert_type)
        else:
            all_experts = []
            for etype in ['text', 'image', 'uml', 'general']:
                all_experts.extend(self.router.get_available_variants(etype))
            return all_experts
