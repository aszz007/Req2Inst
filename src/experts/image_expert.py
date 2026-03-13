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
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.image')



def _build_prompt_for_domain(input_data):
    """
    根据输入数据的实际领域类型构建对应的prompt（跨域评估场景使用）。

    检测规则（按优先级）：
      - json解析失败的字符串 → text → TextInstructionTemplate
      - JSON含 actors + use_cases → uml → UMLInstructionTemplate
      - JSON含 description + details(objects或scene) → image → ImageInstructionTemplate
      - 其他JSON → text → TextInstructionTemplate

    正常领域推理场景（image专家收到图像JSON）同样会路由到ImageInstructionTemplate，
    与原有逻辑行为一致。

    Returns:
        tuple(str, str): (构建的prompt, 检测到的领域类型 'text'/'image'/'uml')
    """
    if isinstance(input_data, dict):
        data = input_data
        text_fallback = str(input_data)
    elif isinstance(input_data, str):
        try:
            data = json.loads(input_data)
            text_fallback = input_data
        except json.JSONDecodeError:
            return TextInstructionTemplate.build_prompt(input_data), 'text'
    else:
        return TextInstructionTemplate.build_prompt(str(input_data)), 'text'

    if isinstance(data, dict):
        if 'actors' in data and 'use_cases' in data:
            return UMLInstructionTemplate.build_prompt(input_data), 'uml'
        details = data.get('details', {})
        if 'description' in data and ('objects' in details or 'scene' in details):
            return ImageInstructionTemplate.build_prompt(input_data), 'image'

    return TextInstructionTemplate.build_prompt(text_fallback), 'text'


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
            base_model_path=str(path_cfg.get_text_model_path()),
            lora_path=lora_path,
            use_4bit=use_4bit
        )

        logger.info("图像专家初始化完成")

    def generate_instruction(self, input_data: Union[str, dict], sample_index: int = None) -> str:
        """
        生成图像标注指令

        Args:
            input_data: 图像描述数据,支持:
                - dict: 包含description字段的字典
                - str: JSON字符串或纯文本description
            sample_index: 样本索引（用于控制日志输出）

        Returns:
            str: 生成的三段式图像标注指令
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
                logger.info("[Image Expert 调试] 接收到的原始输入数据:")
                logger.info("-" * 80)
                logger.info(f"数据类型: {type(input_data).__name__}")

                if isinstance(input_data, dict):
                    logger.info("数据内容（dict格式）:")
                    logger.info(json.dumps(input_data, indent=2, ensure_ascii=False))
                elif isinstance(input_data, str):
                    logger.info(f"数据内容（str格式，前500字符）:")
                    logger.info(input_data[:500])
                    try:
                        parsed = json.loads(input_data)
                        logger.info("\n可以解析为JSON:")
                        logger.info(json.dumps(parsed, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError:
                        logger.info("\n无法解析为JSON，是纯文本description")
                else:
                    logger.info(f"未知数据类型: {input_data}")
                logger.info("=" * 80)

            # 构建prompt（跨域评估时自动检测输入类型并使用对应模板）
            prompt, detected_domain = _build_prompt_for_domain(input_data)
            if detected_domain != 'image' and show_debug:
                logger.warning(
                    f"输入数据检测为{detected_domain}类型，使用对应模板（跨域评估场景）"
                )

            if show_debug:
                logger.debug(f"生成指令 - 输入数据类型: {type(input_data).__name__}")

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

            # 验证输出格式（仅记录日志，不覆盖模型输出）
            if self.validate_output(instruction):
                logger.info("指令生成成功,格式验证通过")
            else:
                if show_debug:
                    logger.warning("指令格式验证失败，直接返回模型输出")
                    logger.warning(f"验证未通过的指令内容：\n{instruction}")
            return instruction

        except Exception as e:
            logger.error(f"指令生成失败: {e}")
            return ""

    def batch_generate_instruction(self, input_data_list: list, batch_size: int = 16) -> list:
        """
        批量生成图像标注指令（提高GPU利用率）

        Args:
            input_data_list: 图像描述数据列表
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

            # 构建所有prompts（跨域评估时自动检测输入类型并使用对应模板）
            prompts = []
            for _idx, _data in enumerate(input_data_list):
                _prompt, _domain = _build_prompt_for_domain(_data)
                if _domain != 'image' and _idx < 3:
                    logger.warning(
                        f"样本{_idx}输入检测为{_domain}类型，使用对应模板（跨域评估场景）"
                    )
                prompts.append(_prompt)

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

            # 验证每个输出（先规范化再验证，仅记录日志，不覆盖模型输出）
            validated_instructions = []
            for i, instruction in enumerate(instructions):
                instruction = self._normalize_instruction(instruction)
                if not self.validate_output(instruction):
                    if i < 3:
                        logger.warning(f"样本{i+1}格式验证失败，直接使用模型输出")
                validated_instructions.append(instruction)

            return validated_instructions

        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            return [""] * len(input_data_list)

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