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
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.general')

def _build_prompt_for_domain(input_data):
    """按数据类型路由到对应模板，与训练时 use_domain_templates=True 的行为保持一致。"""
    if isinstance(input_data, dict):
        data = input_data
    elif isinstance(input_data, str):
        try:
            data = json.loads(input_data)
        except (json.JSONDecodeError, ValueError):
            return TextInstructionTemplate.build_prompt(input_data), 'text'
    else:
        return TextInstructionTemplate.build_prompt(str(input_data)), 'text'

    if isinstance(data, dict):
        if 'actors' in data and 'use_cases' in data:
            return UMLInstructionTemplate.build_prompt(input_data), 'uml'
        details = data.get('details', {})
        if isinstance(details, dict) and ('objects' in details or 'scene' in details):
            return ImageInstructionTemplate.build_prompt(input_data), 'image'

    return TextInstructionTemplate.build_prompt(input_data), 'text'

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

            prompt, _ = _build_prompt_for_domain(input_data)

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

            # 规范化输出（补全 Definition: 标签、去除行尾分隔符等）
            instruction = self._normalize_instruction(instruction)

            # 只在前3个样本输出模型原始输出
            if show_debug:
                logger.info("=" * 80)
                logger.info("模型原始输出:")
                logger.info("-" * 80)
                logger.info(instruction)
                logger.info("=" * 80)

            # 验证输出格式
            # 格式验证未通过时直接返回normalize后的输出，不调用fallback。
            # 仅空输出时保留fallback兜底。
            if self.validate_output(instruction):
                logger.info("指令生成成功,格式验证通过")
                return instruction
            else:
                if show_debug:
                    logger.warning(f"指令格式验证未通过，直接使用normalize后的输出")
                if not instruction or not instruction.strip():
                    logger.warning("输出为空，使用fallback兜底")
                    return self._fallback_generation(input_data)
                return instruction

        except Exception as e:
            logger.error(f"指令生成失败: {e}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return ""

    def batch_generate_instruction(self, input_data_list: list, batch_size: int = 8) -> list:
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

            prompts = [_build_prompt_for_domain(data)[0] for data in input_data_list]

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

            # 验证每个输出（先规范化再验证）
            # 注意：格式验证未通过时直接使用normalize后的原始输出，不调用fallback。
            # fallback生成的固定模板与输入语义完全无关，会显著拉低BLEU/ROUGE/BERTScore；
            # 即使格式有小瑕疵，模型输出的内容仍围绕输入生成，指标远优于fallback。
            # 仅在模型输出为空字符串时保留fallback，避免下游metrics崩溃。
            validated_instructions = []
            for i, instruction in enumerate(instructions):
                instruction = self._normalize_instruction(instruction)
                if not self.validate_output(instruction):
                    if i < 3:
                        logger.warning(
                            f"样本{i+1}格式验证未通过，直接使用normalize后的输出"
                        )
                    if not instruction or not instruction.strip():
                        # 仅空输出才使用fallback，避免下游metrics收到空字符串
                        logger.warning(f"样本{i+1}输出为空，使用fallback兜底")
                        instruction = self._fallback_generation(input_data_list[i])
                validated_instructions.append(instruction)

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