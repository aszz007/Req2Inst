"""
图像专家 - 将图像描述JSON转换为图像标注指令
功能:
  - 处理图像描述JSON
  - 生成三段式图像标注众包指令

环境要求: instruction_generator
模型: Qwen3-8B（用于指令生成）
训练数据: dataset/image/image_dataset.csv (只有1个版本)

说明:
  - Image Expert只有1个,因为图像数据集只有1个版本
  - 基础模型是Qwen3-8B,不是视觉模型
  - 输入是JSON文本描述,不是图像

作者: Expert System
日期: 2025-02-13
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
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True):
        """
        初始化图像专家

        Args:
            lora_path: LoRA权重路径(None则使用默认配置)
            use_4bit: 是否使用4bit量化
        """
        path_cfg = get_path_config()

        # 图像专家固定名称
        expert_name = 'image_expert'

        # 如果没有提供lora_path,使用配置中的路径
        if lora_path is None:
            lora_weight_path = path_cfg.EXPERT_LORA_PATHS.get(expert_name)
            if lora_weight_path is None:
                logger.warning(f"配置中未找到{expert_name}的LoRA权重路径,将使用基础模型")
                lora_path = None
            else:
                lora_path_obj = Path(lora_weight_path)
                if not lora_path_obj.exists():
                    logger.warning(f"LoRA权重路径不存在: {lora_path_obj},将使用基础模型")
                    lora_path = None
                elif not lora_path_obj.is_dir():
                    logger.warning(f"LoRA权重路径不是目录: {lora_path_obj},将使用基础模型")
                    lora_path = None
                else:
                    lora_path = str(lora_path_obj)
                    logger.info(f"找到LoRA权重路径: {lora_path}")

        super().__init__(
            expert_name=expert_name,
            base_model_path=str(path_cfg.QWEN_7B_CHAT_PATH),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("图像专家初始化完成")

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
            # === 调试输出：显示接收到的原始数据 ===
            logger.info("=" * 80)
            logger.info("[Image Expert 调试] 接收到的原始输入数据:")
            logger.info("-" * 80)
            logger.info(f"数据类型: {type(input_data).__name__}")

            if isinstance(input_data, dict):
                logger.info("数据内容（dict格式）:")
                logger.info(json.dumps(input_data, indent=2, ensure_ascii=False))
            elif isinstance(input_data, str):
                logger.info(f"数据内容（str格式，前500字符）:")
                logger.info(input_data[:500])
                # 尝试解析JSON
                try:
                    parsed = json.loads(input_data)
                    logger.info("\n可以解析为JSON:")
                    logger.info(json.dumps(parsed, indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    logger.info("\n无法解析为JSON，是纯文本description")
            else:
                logger.info(f"未知数据类型: {input_data}")
            logger.info("=" * 80)
            # === 调试输出结束 ===

            # 直接使用完整的input_data构建prompt
            # ImageInstructionTemplate.build_prompt会自动处理dict、JSON字符串和纯文本
            prompt = ImageInstructionTemplate.build_prompt(input_data)

            logger.debug(f"生成指令 - 输入数据类型: {type(input_data).__name__}")

            # 调用模型生成
            instruction = self._generate_with_model(
                prompt=prompt,
                max_new_tokens=2048,
                temperature=0.5,  # 中等稳定性,图像标注格式相对固定
                top_p=0.85,       # 适中的采样范围
                top_k=40,         # 适中的候选词数量
                repetition_penalty=1.15  # 中等惩罚
            )

            # 输出模型原始输出用于调试
            logger.info("=" * 80)
            logger.info("模型原始输出:")
            logger.info("-" * 80)
            logger.info(instruction)
            logger.info("=" * 80)

            # 验证输出格式
            if self.validate_output(instruction):
                logger.info("指令生成成功,格式验证通过")
                return instruction
            else:
                logger.warning("指令格式验证失败,尝试回退方案")
                logger.warning(f"失败的指令内容：\n{instruction}")
                return self._fallback_generation(input_data)

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

    def _fallback_generation(self, input_data: Union[str, dict]) -> str:
        """
        回退方案: 生成基础格式的图像标注指令

        Args:
            input_data: 输入数据（完整JSON或纯文本）

        Returns:
            str: 基础格式的指令
        """
        logger.info("使用回退方案生成指令")

        fallback_instruction = """Definition: In this task, draw bounding boxes around all visible objects in the image.
Emphasis & Caution: Focus on accurately identifying and labeling all foreground objects.
Things to Avoid: Do not annotate background elements or partial objects."""

        return fallback_instruction


if __name__ == "__main__":
    print("=" * 60)
    print("图像专家测试")
    print("=" * 60)

    print("\n初始化图像专家...")
    expert = ImageExpert()
    info = expert.get_expert_info()
    print(f"专家名称: {info['expert_name']}")

    print("\n加载模型...")
    if expert.load_model():
        print("模型加载成功")

        test_data = {
            "description": "A busy urban street with cars and traffic signs",
            "details": {
                "objects": ["car", "traffic sign"],
                "scene": "urban street"
            },
            "confidence": 0.95
        }

        print("\n测试生成指令:")
        instruction = expert.generate_instruction(test_data)
        print("\n生成的指令:")
        print("-" * 60)
        print(instruction)
        print("-" * 60)

        is_valid = expert.validate_output(instruction)
        print(f"\n格式验证: {'通过' if is_valid else '失败'}")

        expert.unload_model()
    else:
        print("模型加载失败")

    print("\n测试完成!")