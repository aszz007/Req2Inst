"""
通用专家 - 混合多模态数据训练的兜底专家
功能:
  - 处理文本/图像/UML混合输入
  - 生成三段式众包指令
  - 作为其他专家的兜底方案

环境要求: qwen_text
模型: Qwen-7B-Chat
训练数据: dataset/ (text + image + uml混合)

专家变体(3个):
  - general_expert_dataset_qwen25: 使用Qwen2.5数据集混合训练
  - general_expert_dataset_qwen3: 使用Qwen3数据集混合训练
  - general_expert_dataset_qwen235B: 使用Qwen235B数据集混合训练(默认)

说明:
  - 所有变体都基于Qwen-7B-Chat训练
  - dataset_version指的是混合的UML数据集版本
  - 混合数据: text(全部) + image(全部) + UML(不同版本)

作者: Expert System
日期: 2025-02-03
"""

import json
from pathlib import Path
from typing import Optional, Union

from src.experts.base_expert import BaseExpert
from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from config.settings import get_path_config
from src.utils.logger import get_logger

logger = get_logger('experts.general')


class GeneralExpert(BaseExpert):
    """通用专家 - 混合多模态数据训练的兜底专家"""

    def __init__(self,
                 dataset_version: str = 'qwen235B',
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True):
        """
        初始化通用专家

        Args:
            dataset_version: 数据集构建版本('qwen2.5', 'qwen3', 'qwen235B'),默认'qwen235B'
            lora_path: LoRA权重路径(None则使用默认配置)
            use_4bit: 是否使用4bit量化
        """
        if dataset_version not in ['qwen2.5', 'qwen3', 'qwen235B']:
            raise ValueError(f"不支持的数据集版本: {dataset_version}")

        path_cfg = get_path_config()

        # 构建专家名称: general_expert_dataset_{version}
        dataset_suffix = dataset_version.replace(".", "")
        expert_name = f'general_expert_dataset_{dataset_suffix}'

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
            use_4bit=use_4bit,
            version=dataset_version
        )

        self.dataset_version = dataset_version

        logger.info(f"通用专家初始化完成 - 数据集版本: {dataset_version}")

    def generate_instruction(self, input_data: Union[str, dict]) -> str:
        """
        生成众包指令(自动识别输入类型)

        Args:
            input_data: 输入数据,支持:
                - str: 文本需求或JSON字符串
                - dict: 图像描述字典或UML结构字典

        Returns:
            str: 生成的三段式指令
        """
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return ""

        try:
            # 自动识别输入类型
            input_type = self._detect_input_type(input_data)
            logger.debug(f"识别输入类型: {input_type}")

            # 根据类型选择合适的模板
            if input_type == 'text':
                prompt = self._build_text_prompt(input_data)
            elif input_type == 'image':
                prompt = self._build_image_prompt(input_data)
            elif input_type == 'uml':
                prompt = self._build_uml_prompt(input_data)
            else:
                logger.warning("无法识别输入类型,使用文本模板")
                prompt = TextInstructionTemplate.build_prompt(str(input_data))

            # 调用模型生成
            # 修复：降低temperature以获得更稳定的格式输出
            instruction = self._generate_with_model(
                prompt=prompt,
                max_new_tokens=2048,
                temperature=0.3,  # 降低温度以获得更稳定的格式化输出
                top_p=0.85,       # 稍微降低top_p以减少随机性
                top_k=40,         # 降低top_k以提高确定性
                repetition_penalty=1.1
            )

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

    def _build_text_prompt(self, input_data: Union[str, dict]) -> str:
        """构建文本需求的prompt"""
        if isinstance(input_data, dict):
            text = str(input_data)
        else:
            text = input_data
        return TextInstructionTemplate.build_prompt(text)

    def _build_image_prompt(self, input_data: Union[str, dict]) -> str:
        """构建图像描述的prompt"""
        return ImageInstructionTemplate.build_prompt(input_data)

    def _build_uml_prompt(self, input_data: Union[str, dict]) -> str:
        """构建UML数据的prompt"""
        return UMLInstructionTemplate.build_prompt(input_data)

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

        # 使用文本模板的验证逻辑(三段式基础验证)
        result = TextInstructionTemplate.validate_instruction(instruction)

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

    def get_expert_info(self) -> dict:
        """
        获取专家信息(重写以包含dataset_version)

        Returns:
            dict: 专家信息
        """
        info = super().get_expert_info()
        info['dataset_version'] = self.dataset_version
        return info


if __name__ == "__main__":
    print("=" * 60)
    print("通用专家测试")
    print("=" * 60)

    print("\n测试1: 默认配置(Qwen235B数据集)")
    print("-" * 60)
    expert = GeneralExpert()
    info = expert.get_expert_info()
    print(f"专家名称: {info['expert_name']}")
    print(f"数据集版本: {info['dataset_version']}")

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