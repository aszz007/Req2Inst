"""
通用专家 - 混合多模态数据训练的兜底专家
功能:
  - 处理文本/图像/UML混合输入
  - 生成三段式众包指令
  - 作为其他专家的兜底方案

环境要求: instruction_generator
模型: Qwen3-8B（默认）
训练数据: dataset/ (text + image + uml混合)

说明:
  - 基于Qwen3-8B训练
  - 混合数据: text(全部) + image(全部) + UML(全部)

作者: Expert System
日期: 2025-02-13
"""

import json
from pathlib import Path
from typing import Optional, Union

from src.experts.base_expert import BaseExpert
from models.prompt_templates.general_template import GeneralInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.general')


class GeneralExpert(BaseExpert):
    """通用专家 - 混合多模态数据训练的兜底专家"""

    def __init__(self,
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True):
        """
        初始化通用专家

        Args:
            lora_path: LoRA权重路径(None则使用默认配置)
            use_4bit: 是否使用4bit量化
        """
        path_cfg = get_path_config()

        # 通用专家固定名称
        expert_name = 'general_expert'

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
            base_model_path=str(path_cfg.get_text_model_path()),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("通用专家初始化完成")

    def generate_instruction(self, input_data: Union[str, dict], sample_index: int = None) -> str:
        """
        生成众包指令(自动识别输入类型)

        Args:
            input_data: 输入数据,支持:
                - str: 文本需求或JSON字符串
                - dict: 图像描述字典或UML结构字典
            sample_index: 样本索引（用于控制日志输出）

        Returns:
            str: 生成的三段式指令
        """
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return ""

        try:
            # 只在前3个样本输出调试信息
            show_debug = sample_index is None or sample_index < 3

            if show_debug:
                logger.info("=" * 80)
                logger.info("[调试] 原始输入数据:")
                logger.info("-" * 80)
                if isinstance(input_data, dict):
                    logger.info(f"输入类型: dict")
                    logger.info(f"输入内容（前500字符）: {str(input_data)[:500]}")
                else:
                    logger.info(f"输入类型: {type(input_data).__name__}")
                    logger.info(f"输入内容（前500字符）: {str(input_data)[:500]}")
                logger.info("=" * 80)

            # 自动识别输入类型（仅用于日志）
            if show_debug:
                input_type = self._detect_input_type(input_data)
                logger.info(f"[调试] 识别输入类型: {input_type}")

            # 统一使用GeneralInstructionTemplate，与训练时保持一致
            prompt = GeneralInstructionTemplate.build_prompt(input_data)

            # 调用模型生成
            infer_cfg = get_inference_config()
            instruction = self._generate_with_model(
                prompt=prompt,
                max_new_tokens=infer_cfg.max_new_tokens,
                temperature=infer_cfg.temperature,
                top_p=infer_cfg.top_p,
                top_k=infer_cfg.top_k,
                repetition_penalty=infer_cfg.repetition_penalty,
                sample_index=sample_index,
                verbose=show_debug
            )

            # 只在前3个样本输出模型原始输出
            if show_debug:
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
                if show_debug:
                    logger.warning("指令格式验证失败,尝试回退方案")
                    logger.warning(f"失败的指令内容：\n{instruction}")
                return self._fallback_generation(input_data)

        except Exception as e:
            logger.error(f"指令生成失败: {e}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return ""

    def batch_generate_instruction(self, input_data_list: list, batch_size: int = 4) -> list:
        """
        批量生成众包指令（提高GPU利用率）

        Args:
            input_data_list: 输入数据列表（支持混合类型）
            batch_size: 批处理大小（默认8，适合RTX 4090 24GB）

        Returns:
            list: 生成的指令列表
        """
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return [""] * len(input_data_list)

        try:
            logger.info(f"批量生成指令 - 共{len(input_data_list)}个样本，batch_size={batch_size}")

            # 构建所有prompts（使用统一的GeneralInstructionTemplate）
            prompts = [GeneralInstructionTemplate.build_prompt(data) for data in input_data_list]

            # 批量生成
            infer_cfg = get_inference_config()
            instructions = self._generate_batch_with_model(
                prompts=prompts,
                max_new_tokens=infer_cfg.max_new_tokens,
                temperature=infer_cfg.temperature,
                top_p=infer_cfg.top_p,
                top_k=infer_cfg.top_k,
                repetition_penalty=infer_cfg.repetition_penalty,
                batch_size=batch_size,
                start_index=0,
                verbose=True
            )

            # 输出前3个样本的生成结果
            for i in range(min(3, len(instructions))):
                logger.info("=" * 80)
                logger.info(f"[样本 {i+1}/{len(input_data_list)}] 生成的指令:")
                logger.info("-" * 80)
                logger.info(instructions[i])
                logger.info("=" * 80)

            # 验证每个输出
            validated_instructions = []
            for i, instruction in enumerate(instructions):
                if self.validate_output(instruction):
                    validated_instructions.append(instruction)
                else:
                    if i < 3:
                        logger.warning(f"样本{i+1}格式验证失败,使用回退方案")
                    validated_instructions.append(self._fallback_generation(input_data_list[i]))

            return validated_instructions

        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return [""] * len(input_data_list)

    def _detect_input_type(self, input_data: Union[str, dict]) -> str:
        """
        自动检测输入数据类型

        Args:
            input_data: 输入数据

        Returns:
            str: 'text', 'image', 'uml' 或 'unknown'
        """
        if isinstance(input_data, dict):
            # 检查是否包含UML特征
            if 'actors' in input_data and 'use_cases' in input_data:
                return 'uml'
            # 检查是否包含图像描述特征
            elif 'description' in input_data or 'details' in input_data:
                return 'image'
            else:
                return 'unknown'

        elif isinstance(input_data, str):
            # 尝试解析为JSON
            try:
                parsed = json.loads(input_data)
                return self._detect_input_type(parsed)
            except json.JSONDecodeError:
                # 纯文本,判定为文本需求
                return 'text'

        return 'unknown'

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

        # 使用GeneralInstructionTemplate的验证逻辑(三段式基础验证)
        result = GeneralInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"格式验证失败: {result['errors']}")
            return False

        return True

    def _fallback_generation(self, input_data: Union[str, dict]) -> str:
        """
        回退方案: 生成基础格式的指令

        Args:
            input_data: 输入数据

        Returns:
            str: 基础格式的指令
        """
        logger.info("使用回退方案生成指令")

        # 使用通用描述，不包含原始输入内容
        fallback_instruction = """Definition: In this task, implement or test the specified requirement.
Emphasis & Caution: Ensure comprehensive testing and validation of all functionality.
Things to Avoid: Do not skip error handling or edge case validation."""

        return fallback_instruction



if __name__ == "__main__":
    print("=" * 60)
    print("通用专家测试")
    print("=" * 60)

    print("\n测试1: 默认配置")
    print("-" * 60)
    expert = GeneralExpert()
    info = expert.get_expert_info()
    print(f"专家名称: {info['expert_name']}")

    print("\n测试2: 输入类型检测")
    print("-" * 60)

    # 文本输入
    text_input = "测试系统的登录功能"
    text_type = expert._detect_input_type(text_input)
    print(f"文本输入识别: {text_type}")

    # 图像输入
    image_input = {"description": "A street scene", "details": {}}
    image_type = expert._detect_input_type(image_input)
    print(f"图像输入识别: {image_type}")

    # UML输入
    uml_input = {"actors": ["User"], "use_cases": [{"name": "Login"}]}
    uml_type = expert._detect_input_type(uml_input)
    print(f"UML输入识别: {uml_type}")

    print("\n测试3: 生成指令(文本输入)")
    print("-" * 60)
    if expert.load_model():
        instruction = expert.generate_instruction(text_input)
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