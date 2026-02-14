"""
文本专家 - 将文本需求转换为众包指令
功能:
  - 处理Low_Requirements文本需求
  - 生成三段式众包指令
  - 使用Qwen3-8B + LoRA微调权重

环境要求: instruction_generator
模型: Qwen3-8B（默认）
训练数据: dataset/text/

作者: Expert System
日期: 2025-02-13
"""

from pathlib import Path
from typing import Optional

from src.experts.base_expert import BaseExpert
from models.prompt_templates.text_template import TextInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.text')


class TextExpert(BaseExpert):
    """文本专家 - 文本需求转众包指令"""

    def __init__(self, lora_path: Optional[str] = None, use_4bit: bool = True):
        """
        初始化文本专家

        Args:
            lora_path: LoRA权重路径(None则使用默认配置)
            use_4bit: 是否使用4bit量化
        """
        path_cfg = get_path_config()

        # 如果没有提供lora_path,使用配置中的路径
        if lora_path is None:
            lora_weight_path = path_cfg.EXPERT_LORA_PATHS.get('text_expert')
            if lora_weight_path is None:
                logger.warning("配置中未找到text_expert的LoRA权重路径,将使用基础模型")
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
            expert_name='text_expert',
            base_model_path=str(path_cfg.get_text_model_path()),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("文本专家初始化完成")

    def generate_instruction(self, input_data: str) -> str:
        """
        生成文本众包指令

        Args:
            input_data: Low_Requirements文本需求

        Returns:
            str: 生成的三段式指令
        """
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return ""

        try:
            # 使用TextInstructionTemplate构建prompt
            prompt = TextInstructionTemplate.build_prompt(input_data)

            logger.debug(f"生成指令 - 输入需求: {input_data[:100]}...")

            # 调用模型生成
            infer_cfg = get_inference_config()
            instruction = self._generate_with_model(
                prompt=prompt,
                max_new_tokens=infer_cfg.max_new_tokens,
                temperature=infer_cfg.temperature,
                top_p=infer_cfg.top_p,
                top_k=infer_cfg.top_k,
                repetition_penalty=infer_cfg.repetition_penalty
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
            return self._fallback_generation(input_data)

    def validate_output(self, instruction: str) -> bool:
        """
        验证输出格式是否符合三段式要求

        Args:
            instruction: 生成的指令

        Returns:
            bool: 是否符合格式
        """
        if not instruction or len(instruction.strip()) < 50:
            logger.debug("指令内容过短")
            return False

        result = TextInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"格式验证失败: {result['errors']}")
            return False

        return True

    def _fallback_generation(self, input_data: str) -> str:
        """
        回退方案: 生成基础格式的指令

        Args:
            input_data: 输入需求

        Returns:
            str: 基础格式的指令
        """
        logger.info("使用回退方案生成指令")

        fallback_instruction = """Definition: In this task, implement and test the specified requirement.
Emphasis & Caution: Ensure thorough testing and validation of all functionality.
Things to Avoid: Do not skip error handling or edge case testing."""

        return fallback_instruction


if __name__ == "__main__":
    print("=" * 60)
    print("文本专家测试")
    print("=" * 60)

    print("\n初始化文本专家...")
    expert = TextExpert()

    print("\n查看专家信息:")
    info = expert.get_expert_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n加载模型...")
    if expert.load_model():
        print("模型加载成功")

        print("\n测试生成指令:")
        test_requirement = "测试系统的登录功能,确保用户名和密码验证正确"
        instruction = expert.generate_instruction(test_requirement)

        print("\n生成的指令:")
        print("-" * 60)
        print(instruction)
        print("-" * 60)

        print("\n验证指令格式:")
        is_valid = expert.validate_output(instruction)
        print(f"格式验证: {'通过' if is_valid else '失败'}")

        expert.unload_model()
    else:
        print("模型加载失败")

    print("\n测试完成!")