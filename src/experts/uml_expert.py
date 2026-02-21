"""
UML专家 - 将UML用例图JSON转换为业务逻辑实现指令
功能:
  - 处理UML用例图JSON数据
  - 生成三段式业务逻辑实现众包指令

环境要求: instruction_generator
模型: Qwen3-8B（默认）
训练数据: dataset/uml/uml_dataset.csv

说明:
  - 基于Qwen3-8B训练
  - 使用Qwen3-VL识别的UML数据集

作者: Expert System
日期: 2025-02-13
"""

import json
from pathlib import Path
from typing import Optional, Union

from src.experts.base_expert import BaseExpert
from models.prompt_templates.uml_template import UMLInstructionTemplate
from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('experts.uml')


class UMLExpert(BaseExpert):
    """UML专家 - UML用例图JSON转业务逻辑实现指令"""

    def __init__(self,
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True):
        """
        初始化UML专家

        Args:
            lora_path: LoRA权重路径(None则使用默认配置)
            use_4bit: 是否使用4bit量化
        """
        path_cfg = get_path_config()

        # UML专家固定名称
        expert_name = 'uml_expert'

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

        logger.info("UML专家初始化完成")

    def generate_instruction(self, input_data: Union[str, dict], sample_index: int = None) -> str:
        """
        生成UML业务逻辑实现指令

        Args:
            input_data: UML用例图数据,支持:
                - dict: 包含actors, use_cases, relationships字段的字典
                - str: JSON字符串
            sample_index: 样本索引（用于控制日志输出）

        Returns:
            str: 生成的三段式业务逻辑实现指令
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
                logger.info("[UML Expert 调试] 接收到的原始输入数据:")
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
                        logger.info("\n无法解析为JSON")
                else:
                    logger.info(f"未知数据类型: {input_data}")
                logger.info("=" * 80)

            # 解析输入数据并构建prompt
            if isinstance(input_data, str):
                try:
                    uml_data = json.loads(input_data)
                    # 提取关键元素（用于日志）
                    if show_debug:
                        elements = UMLInstructionTemplate.extract_key_elements(uml_data)
                        logger.debug(f"生成指令 - Actors: {elements['actors']}, Use Cases: {len(elements['use_cases'])}个")
                    prompt = UMLInstructionTemplate.build_prompt(uml_data)
                except json.JSONDecodeError:
                    # 跨域评估场景：输入为纯文本而非UML JSON，仍尝试生成
                    logger.warning("输入非JSON格式，以纯文本方式处理（跨域评估场景）")
                    uml_data = {}
                    prompt = (
                        "Based on the following requirement, generate a crowdsourcing task instruction "
                        "with three sections: Definition, Emphasis & Caution, and Things to Avoid.\n\n"
                        f"Requirement: {input_data}\n\nInstruction:"
                    )
            elif isinstance(input_data, dict):
                uml_data = input_data
                # 提取关键元素（用于日志）
                if show_debug:
                    elements = UMLInstructionTemplate.extract_key_elements(uml_data)
                    logger.debug(f"生成指令 - Actors: {elements['actors']}, Use Cases: {len(elements['use_cases'])}个")
                prompt = UMLInstructionTemplate.build_prompt(uml_data)
            else:
                logger.error(f"不支持的输入类型: {type(input_data)}")
                return ""

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

            # 规范化输出（补全 Definition: 标签、去除分隔符等）
            instruction = self._normalize_instruction(instruction)

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
                return self._fallback_generation(uml_data)

        except Exception as e:
            logger.error(f"指令生成失败: {e}")
            return ""

    def batch_generate_instruction(self, input_data_list: list, batch_size: int = 4) -> list:
        """
        批量生成UML业务逻辑实现指令（提高GPU利用率）

        Args:
            input_data_list: UML用例图数据列表
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

            # 解析所有输入数据并构建prompts
            # 只收集有效prompt及其原始索引，避免空字符串送入模型导致张量越界
            parsed_data_list = [{}] * len(input_data_list)
            valid_indices = []   # 原始索引
            valid_prompts = []   # 对应有效prompt

            for idx, data in enumerate(input_data_list):
                if isinstance(data, str):
                    try:
                        uml_data = json.loads(data)
                        parsed_data_list[idx] = uml_data
                        valid_indices.append(idx)
                        valid_prompts.append(UMLInstructionTemplate.build_prompt(uml_data))
                    except json.JSONDecodeError:
                        # 跨域评估场景：输入为纯文本而非UML JSON，仍尝试生成
                        logger.warning(f"样本{idx}输入非JSON格式，以纯文本方式处理（跨域评估场景）")
                        plain_prompt = (
                            "Based on the following requirement, generate a crowdsourcing task "
                            "instruction with three sections: Definition, Emphasis & Caution, "
                            "and Things to Avoid.\n\n"
                            "Requirement: " + str(data) + "\n\nInstruction:"
                        )
                        parsed_data_list[idx] = {}
                        valid_indices.append(idx)
                        valid_prompts.append(plain_prompt)
                elif isinstance(data, dict):
                    uml_data = data
                    parsed_data_list[idx] = uml_data
                    valid_indices.append(idx)
                    valid_prompts.append(UMLInstructionTemplate.build_prompt(uml_data))
                else:
                    logger.error(f"不支持的输入类型: {type(data)}，跳过")
                    continue

            # 批量生成（仅对有效prompt）
            raw_instructions = [""] * len(input_data_list)
            if valid_prompts:
                infer_cfg = get_inference_config()
                generated = self._generate_batch_with_model(
                    prompts=valid_prompts,
                    max_new_tokens=infer_cfg.max_new_tokens,
                    temperature=infer_cfg.temperature,
                    top_p=infer_cfg.top_p,
                    top_k=infer_cfg.top_k,
                    repetition_penalty=infer_cfg.repetition_penalty,
                    batch_size=batch_size,
                    start_index=0,
                    verbose=True
                )
                for orig_idx, instruction in zip(valid_indices, generated):
                    raw_instructions[orig_idx] = instruction

            instructions = raw_instructions

            # 输出前3个有效样本的生成结果
            shown = 0
            for i in range(len(instructions)):
                if shown >= 3:
                    break
                if i in valid_indices:
                    logger.info("=" * 80)
                    logger.info(f"[样本 {i+1}/{len(input_data_list)}] 生成的指令:")
                    logger.info("-" * 80)
                    logger.info(instructions[i])
                    logger.info("=" * 80)
                    shown += 1

            # 验证每个输出（先规范化再验证）
            validated_instructions = []
            for i, instruction in enumerate(instructions):
                instruction = self._normalize_instruction(instruction)
                if self.validate_output(instruction):
                    validated_instructions.append(instruction)
                else:
                    if i < 3:
                        logger.warning(f"样本{i+1}格式验证失败,使用回退方案")
                    validated_instructions.append(self._fallback_generation(parsed_data_list[i]))

            return validated_instructions

        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            return [""] * len(input_data_list)

    def validate_output(self, instruction: str) -> bool:
        """
        验证输出格式是否符合UML业务逻辑三段式要求

        Args:
            instruction: 生成的指令

        Returns:
            bool: 是否符合格式
        """
        if not instruction or len(instruction.strip()) < 50:
            logger.debug("指令内容过短")
            return False

        result = UMLInstructionTemplate.validate_instruction(instruction)

        if not result['is_valid']:
            logger.debug(f"格式验证失败: {result['errors']}")
            return False

        # 额外检查是否包含业务逻辑要求
        if not result['has_business_logic']:
            logger.debug("缺少业务逻辑实现要求")
            return False

        return True

    def _fallback_generation(self, uml_data: dict) -> str:
        """
        回退方案: 生成基础格式的UML业务逻辑指令

        Args:
            uml_data: UML用例图数据

        Returns:
            str: 基础格式的指令
        """
        logger.info("使用回退方案生成指令")

        fallback_instruction = """Definition: In this task, implement the system workflow with specified actors interacting with defined use cases.
Emphasis & Caution: Ensure all mandatory steps and conditional extensions are properly implemented.
Things to Avoid: Do not focus on UI positioning or visual layout. Avoid implementing frontend styling."""

        return fallback_instruction



if __name__ == "__main__":
    print("=" * 60)
    print("UML专家测试")
    print("=" * 60)

    print("\n测试1: 初始化UML专家")
    print("-" * 60)
    expert = UMLExpert()
    info = expert.get_expert_info()
    print(f"专家名称: {info['expert_name']}")

    print("\n测试2: 生成指令")
    print("-" * 60)
    if expert.load_model():
        test_data = {
            "actors": [
                {"name": "User", "position": "left"},
                {"name": "Admin", "position": "right"}
            ],
            "use_cases": [
                {"name": "Login System", "description": "User authentication"},
                {"name": "Validate Credentials", "description": "Check credentials"},
                {"name": "Send Email", "description": "Email notification"}
            ],
            "relationships": [
                {
                    "type": "association",
                    "from": "User",
                    "to": "Login System",
                    "description": "User initiates login"
                },
                {
                    "type": "include",
                    "from": "Login System",
                    "to": "Validate Credentials",
                    "description": "Must validate before login"
                },
                {
                    "type": "extend",
                    "from": "Send Email",
                    "to": "Login System",
                    "description": "Optional email on success"
                }
            ]
        }

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