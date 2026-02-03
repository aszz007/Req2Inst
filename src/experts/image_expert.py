"""
图像专家 - 将图像描述JSON转换为图像标注指令
功能:
  - 处理图像描述JSON
  - 生成三段式图像标注众包指令
  - 支持Qwen2.5-VL和Qwen3-VL两个版本

环境要求: qwen_vision25 或 qwen_vision3
模型: Qwen-7B-Chat(用于指令生成)
训练数据: dataset/image/

专家变体:
  - image_expert_qwen25: 使用Qwen2.5-VL数据集训练的LoRA
  - image_expert_qwen3: 使用Qwen3-VL数据集训练的LoRA(默认)

作者: Expert System
日期: 2025-02-03
"""

import json
from pathlib import Path
from typing import Optional, Union

from src.experts.base_expert import BaseExpert
from models.prompt_templates.image_template import ImageInstructionTemplate
from config.settings import get_path_config
from src.utils.logger import get_logger

logger = get_logger('experts.image')


class ImageExpert(BaseExpert):
    """图像专家 - 图像描述转图像标注指令"""

    def __init__(self,
                 version: str = 'qwen3',
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True):
        """
        初始化图像专家

        Args:
            version: 模型版本('qwen2.5' 或 'qwen3'),默认'qwen3'
            lora_path: LoRA权重路径(None则使用默认配置)
            use_4bit: 是否使用4bit量化
        """
        if version not in ['qwen2.5', 'qwen3']:
            raise ValueError(f"不支持的版本: {version},请使用'qwen2.5'或'qwen3'")

        path_cfg = get_path_config()

        # 构建专家名称
        expert_name = f'image_expert_{version.replace(".", "")}'

        # 如果没有提供lora_path,使用配置中的路径
        if lora_path is None:
            lora_path = str(path_cfg.EXPERT_LORA_PATHS.get(expert_name, ''))
            if not lora_path or not Path(lora_path).exists():
                logger.warning(f"未找到{expert_name}的LoRA权重,将使用基础模型")
                lora_path = None

        super().__init__(
            expert_name=expert_name,
            base_model_path=str(path_cfg.QWEN_7B_CHAT_PATH),
            lora_path=lora_path,
            use_4bit=use_4bit,
            version=version
        )

        logger.info(f"图像专家初始化完成 - 版本: {version}")

    def generate_instruction(self, input_data: Union[str, dict]) -> str:
        """
        生成图像标注指令

        Args:
            input_data: 图像描述数据,支持:
                - dict: 包含description字段的字典
                - str: JSON字符串或纯文本description

        Returns:
            str: 生成的三段式图像标注指令
        """
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return ""

        try:
            # 提取description字段(如果是完整JSON)
            if isinstance(input_data, dict):
                description = ImageInstructionTemplate.extract_description_from_json(input_data)
            elif isinstance(input_data, str):
                try:
                    parsed = json.loads(input_data)
                    description = ImageInstructionTemplate.extract_description_from_json(parsed)
                except json.JSONDecodeError:
                    description = input_data
            else:
                logger.error(f"不支持的输入类型: {type(input_data)}")
                return ""

            # 使用ImageInstructionTemplate构建prompt
            prompt = ImageInstructionTemplate.build_prompt(description)

            logger.debug(f"生成指令 - 图像描述: {description[:100]}...")

            # 调用模型生成
            instruction = self._generate_with_model(
                prompt=prompt,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1
            )

            # 验证输出格式
            if self.validate_output(instruction):
                logger.info("指令生成成功,格式验证通过")
                return instruction
            else:
                logger.warning("指令格式验证失败,尝试回退方案")
                return self._fallback_generation(description)

        except Exception as e:
            logger.error(f"指令生成失败: {e}")
            return ""

    def validate_output(self, instruction: str) -> bool:
        """
        验证输出格式是否符合图像标注三段式要求

        Args:
            instruction: 生成的指令

        Returns:
            bool: 是否符合格式
        """
        if not instruction or len(instruction.strip()) < 50:
            logger.debug("指令内容过短")
            return False

        result = ImageInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"格式验证失败: {result['errors']}")
            return False

        # 额外检查是否包含bounding boxes要求
        if not result['has_bounding_boxes']:
            logger.debug("缺少bounding boxes要求")
            return False

        return True

    def _fallback_generation(self, description: str) -> str:
        """
        回退方案: 生成基础格式的图像标注指令

        Args:
            description: 图像描述

        Returns:
            str: 基础格式的指令
        """
        logger.info("使用回退方案生成指令")

        fallback_instruction = f"""Definition: In this task, draw bounding boxes around all objects described in the image: {description[:150]}

Emphasis & Caution: Focus on accurately identifying and labeling all visible objects.

Things to Avoid: Do not annotate background elements or partial objects."""

        return fallback_instruction


if __name__ == "__main__":
    print("=" * 60)
    print("图像专家测试")
    print("=" * 60)

    print("\n测试1: 使用Qwen3版本")
    print("-" * 60)
    expert_qwen3 = ImageExpert(version='qwen3')
    info = expert_qwen3.get_expert_info()
    print(f"专家名称: {info['expert_name']}")
    print(f"版本: {info['version']}")

    print("\n测试2: 使用Qwen2.5版本")
    print("-" * 60)
    expert_qwen25 = ImageExpert(version='qwen2.5')
    info = expert_qwen25.get_expert_info()
    print(f"专家名称: {info['expert_name']}")
    print(f"版本: {info['version']}")

    print("\n测试3: 生成指令")
    print("-" * 60)
    if expert_qwen3.load_model():
        test_data = {
            "description": "A busy urban street with cars and traffic signs",
            "details": {
                "objects": ["car", "traffic sign"],
                "scene": "urban street"
            },
            "confidence": 0.95
        }

        instruction = expert_qwen3.generate_instruction(test_data)
        print("\n生成的指令:")
        print("-" * 60)
        print(instruction)
        print("-" * 60)

        is_valid = expert_qwen3.validate_output(instruction)
        print(f"\n格式验证: {'通过' if is_valid else '失败'}")

        expert_qwen3.unload_model()
    else:
        print("模型加载失败")

    print("\n测试完成!")