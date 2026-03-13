"""
Instruction Generator - Unified Interface for MoE-based Instruction Generation
指令生成器 - 基于MoE的统一指令生成接口

功能:
  - 统一的指令生成接口
  - 整合MoE路由和专家系统
  - 支持批量生成
  - 输出格式化(text/json/markdown)

环境要求: instruction_generator
依赖: MoE系统, 专家模型

作者: Instruction Generation System
日期: 2025-02-06
"""

import json
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from src.routing.moe_model import MoEModel
from src.utils.logger import get_logger
from config.settings import get_path_config

logger = get_logger('instruction_generation.generator')


class InstructionGenerator:
    """
    指令生成器

    提供统一的指令生成接口,封装MoE系统的复杂性
    """

    def __init__(
            self,
            lora_weights_dir: Optional[str] = None,
            base_models_dir: Optional[str] = None
    ):
        """
        初始化指令生成器

        Args:
            lora_weights_dir: LoRA权重目录(None则使用配置)
            base_models_dir: 基础模型目录(None则使用配置)
        """
        path_cfg = get_path_config()

        if lora_weights_dir is None:
            # Use checkpoints/lora_moe/ as the standard weights directory per framework
            lora_weights_dir = str(path_cfg.PROJECT_ROOT / 'checkpoints' / 'lora_moe')
        if base_models_dir is None:
            base_models_dir = str(path_cfg.BASE_MODELS_DIR)

        # 初始化MoE模型
        self.moe_model = MoEModel(
            lora_weights_dir=lora_weights_dir,
            base_models_dir=base_models_dir
        )

        logger.info("指令生成器初始化完成")
        logger.info(f"LoRA权重目录: {lora_weights_dir}")
        logger.info(f"基础模型目录: {base_models_dir}")

    def generate(
            self,
            input_data: Union[str, dict],
            output_format: str = 'text',
            expert_variant: Optional[str] = None,
            **generation_kwargs
    ) -> Union[str, dict]:
        """
        生成众包指令(统一接口)

        Args:
            input_data: 输入数据
                - str: 文本需求
                - dict: 必须包含'type'和'content'字段
                  - type: 'text', 'image', 'uml', 'general'
                  - content: 实际内容
            output_format: 输出格式
                - 'text': 纯文本指令
                - 'json': JSON格式(包含元数据)
                - 'markdown': Markdown格式
            expert_variant: 指定专家变体(对比实验用)
            **generation_kwargs: 生成参数(temperature, max_new_tokens等)

        Returns:
            str or dict: 根据output_format返回不同格式
        """
        logger.info("=" * 80)
        logger.info("开始生成指令")
        logger.info("=" * 80)

        # 调用MoE系统生成
        result = self.moe_model.generate_instruction(
            input_data=input_data,
            expert_variant=expert_variant,
            **generation_kwargs
        )

        # 添加时间戳
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 根据格式返回
        if output_format == 'json':
            logger.info("返回JSON格式")
            return result
        elif output_format == 'markdown':
            logger.info("返回Markdown格式")
            return self._format_markdown(result)
        else:  # text
            logger.info("返回文本格式")
            return result['instruction']

    def batch_generate(
            self,
            input_list: List[Union[str, dict]],
            output_format: str = 'text',
            expert_variant: Optional[str] = None,
            save_path: Optional[str] = None,
            **generation_kwargs
    ) -> List[Union[str, dict]]:
        """
        批量生成指令

        Args:
            input_list: 输入数据列表
            output_format: 输出格式
            expert_variant: 指定专家变体
            save_path: 保存路径(可选)
            **generation_kwargs: 生成参数

        Returns:
            list: 生成结果列表
        """
        logger.info("=" * 80)
        logger.info(f"批量生成指令 - 共{len(input_list)}个样本")
        logger.info("=" * 80)

        results = []

        for i, input_data in enumerate(input_list, 1):
            logger.info(f"\n处理样本 {i}/{len(input_list)}")

            try:
                result = self.generate(
                    input_data=input_data,
                    output_format=output_format,
                    expert_variant=expert_variant,
                    **generation_kwargs
                )
                results.append(result)
                logger.info(f"样本 {i} 生成成功")

            except Exception as e:
                logger.error(f"样本 {i} 生成失败: {e}")
                # 添加失败标记
                if output_format == 'json':
                    results.append({
                        'instruction': '',
                        'expert_used': 'none',
                        'error': str(e),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    results.append('')

        # 保存结果
        if save_path:
            self._save_results(results, save_path, output_format)

        logger.info("=" * 80)
        logger.info(f"批量生成完成 - 成功: {len([r for r in results if r])}/{len(input_list)}")
        logger.info("=" * 80)

        return results

    def generate_from_file(
            self,
            input_file: str,
            output_file: Optional[str] = None,
            output_format: str = 'json',
            **generation_kwargs
    ) -> List[dict]:
        """
        从文件读取输入并生成指令

        Args:
            input_file: 输入文件路径
                - .txt: 每行一个文本需求
                - .json: JSON数组或JSON Lines格式
            output_file: 输出文件路径(可选)
            output_format: 输出格式
            **generation_kwargs: 生成参数

        Returns:
            list: 生成结果列表
        """
        logger.info(f"从文件生成指令: {input_file}")

        # 读取输入
        input_list = self._load_input_file(input_file)
        logger.info(f"加载了 {len(input_list)} 个输入")

        # 批量生成
        results = self.batch_generate(
            input_list=input_list,
            output_format=output_format,
            save_path=output_file,
            **generation_kwargs
        )

        return results

    def _load_input_file(self, file_path: str) -> List[Union[str, dict]]:
        """
        加载输入文件

        Args:
            file_path: 文件路径

        Returns:
            list: 输入数据列表
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {file_path}")

        if file_path.suffix == '.txt':
            # 文本文件,每行一个需求
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines

        elif file_path.suffix == '.json':
            # JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 如果是数组,直接返回
            if isinstance(data, list):
                return data
            # 如果是单个对象,包装为列表
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError(f"不支持的JSON格式: {type(data)}")

        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _save_results(
            self,
            results: List[Union[str, dict]],
            save_path: str,
            output_format: str
    ):
        """
        保存生成结果

        Args:
            results: 结果列表
            save_path: 保存路径
            output_format: 输出格式
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == 'json':
            # JSON格式
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        elif output_format == 'markdown':
            # Markdown格式
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("# Generated Instructions\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"## Instruction {i}\n\n")
                    f.write(result + "\n\n")
                    f.write("---\n\n")

        else:  # text
            # 纯文本格式
            with open(save_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results, 1):
                    f.write(f"=== Instruction {i} ===\n")
                    f.write(result + "\n\n")

        logger.info(f"结果已保存至: {save_path}")

    def _format_markdown(self, result: dict) -> str:
        """
        格式化为Markdown

        Args:
            result: 生成结果字典

        Returns:
            str: Markdown格式文本
        """
        md = "# Generated Instruction\n\n"

        # 指令内容
        md += "## Instruction\n\n"
        md += result['instruction'] + "\n\n"

        # 元数据
        md += "## Metadata\n\n"
        md += f"- **Expert Used**: {result['expert_used']}\n"
        md += f"- **Expert Type**: {result['expert_type']}\n"
        md += f"- **Timestamp**: {result.get('timestamp', 'N/A')}\n"
        md += f"- **Reasoning**: {result['reasoning']}\n"

        return md

    def get_statistics(self) -> Dict:
        """
        获取生成统计信息

        Returns:
            dict: 统计信息字典
        """
        return self.moe_model.get_router_statistics()

    def reset_statistics(self):
        """重置统计信息"""
        self.moe_model.reset_router_statistics()
        logger.info("统计信息已重置")

    def list_available_experts(self, expert_type: Optional[str] = None) -> List[str]:
        """
        列出可用的专家

        Args:
            expert_type: 专家类型筛选(可选)

        Returns:
            list: 专家名称列表
        """
        return self.moe_model.list_available_experts(expert_type)