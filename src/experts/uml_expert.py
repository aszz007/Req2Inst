"""
UML专家 - 将UML用例图JSON转换为业务逻辑实现指令
功能:
  - 处理UML用例图JSON数据
  - 生成三段式业务逻辑实现众包指令
  - 支持3个数据集版本(不同视觉模型识别生成)

环境要求: qwen_text
模型: Qwen-7B-Chat
训练数据: dataset/uml/uml_dataset_{version}.csv

专家变体(3个):
  - uml_expert_dataset_qwen25: 使用Qwen2.5-VL识别的数据集训练
  - uml_expert_dataset_qwen3: 使用Qwen3-VL识别的数据集训练
  - uml_expert_dataset_qwen235B: 使用Qwen235B云端API识别的数据集训练(默认)

说明:
  - 所有变体都基于Qwen-7B-Chat训练
  - dataset_version指的是用哪个视觉模型识别的数据集
  - 不同版本用于对比不同识别源的效果

作者: Expert System
日期: 2025-02-03
"""

import json
from pathlib import Path
from typing import Optional, Union

from src.experts.base_expert import BaseExpert
from models.prompt_templates.uml_template import UMLInstructionTemplate
from config.settings import get_path_config
from src.utils.logger import get_logger

logger = get_logger('experts.uml')


class UMLExpert(BaseExpert):
    """UML专家 - UML用例图JSON转业务逻辑实现指令"""

    def __init__(self,
                 dataset_version: str = 'qwen235B',
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True):
        """
        初始化UML专家

        Args:
            dataset_version: 数据集版本('qwen2.5', 'qwen3', 'qwen235B'),默认'qwen235B'
            lora_path: LoRA权重路径(None则使用默认配置)
            use_4bit: 是否使用4bit量化
        """
        if dataset_version not in ['qwen2.5', 'qwen3', 'qwen235B']:
            raise ValueError(f"不支持的数据集版本: {dataset_version},请使用'qwen2.5', 'qwen3'或'qwen235B'")

        path_cfg = get_path_config()

        # 构建专家名称: uml_expert_dataset_{version}
        dataset_suffix = dataset_version.replace(".", "")
        expert_name = f'uml_expert_dataset_{dataset_suffix}'

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

        logger.info(f"UML专家初始化完成 - 数据集版本: {dataset_version}")

    def generate_instruction(self, input_data: Union[str, dict]) -> str:
        """
        生成UML业务逻辑实现指令

        Args:
            input_data: UML用例图数据,支持:
                - dict: 包含actors, use_cases, relationships字段的字典
                - str: JSON字符串

        Returns:
            str: 生成的三段式业务逻辑实现指令
        """
        if not self.is_model_loaded:
            logger.warning("模型未加载,尝试加载模型...")
            if not self.load_model():
                logger.error("模型加载失败")
                return ""

        try:
            # 解析输入数据
            if isinstance(input_data, str):
                try:
                    uml_data = json.loads(input_data)
                except json.JSONDecodeError:
                    logger.error("输入不是有效的JSON格式")
                    return ""
            elif isinstance(input_data, dict):
                uml_data = input_data
            else:
                logger.error(f"不支持的输入类型: {type(input_data)}")
                return ""

            # 提取关键元素（用于日志）
            elements = UMLInstructionTemplate.extract_key_elements(uml_data)
            logger.debug(f"生成指令 - Actors: {elements['actors']}, Use Cases: {len(elements['use_cases'])}个")

            # 使用UMLInstructionTemplate构建prompt
            prompt = UMLInstructionTemplate.build_prompt(uml_data)

            # 调用模型生成
            instruction = self._generate_with_model(
                prompt=prompt,
                max_new_tokens=2048,
                temperature=0.4,  # 平衡稳定性和多样性,适合长指令生成
                top_p=0.85,       # 稍微降低top_p以减少随机性
                top_k=40,         # 降低top_k以提高确定性
                repetition_penalty=1.1
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
                return self._fallback_generation(uml_data)

        except Exception as e:
            logger.error(f"指令生成失败: {e}")
            return ""

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
    print("UML专家测试")
    print("=" * 60)

    print("\n测试1: 使用Qwen235B数据集(默认)")
    print("-" * 60)
    expert_qwen235B = UMLExpert(dataset_version='qwen235B')
    info = expert_qwen235B.get_expert_info()
    print(f"专家名称: {info['expert_name']}")
    print(f"数据集版本: {info['dataset_version']}")

    print("\n测试2: 使用Qwen3数据集")
    print("-" * 60)
    expert_qwen3 = UMLExpert(dataset_version='qwen3')
    info = expert_qwen3.get_expert_info()
    print(f"专家名称: {info['expert_name']}")
    print(f"数据集版本: {info['dataset_version']}")

    print("\n测试3: 使用Qwen2.5数据集")
    print("-" * 60)
    expert_qwen25 = UMLExpert(dataset_version='qwen2.5')
    info = expert_qwen25.get_expert_info()
    print(f"专家名称: {info['expert_name']}")
    print(f"数据集版本: {info['dataset_version']}")

    print("\n测试4: 生成指令")
    print("-" * 60)
    if expert_qwen235B.load_model():
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

        instruction = expert_qwen235B.generate_instruction(test_data)
        print("\n生成的指令:")
        print("-" * 60)
        print(instruction)
        print("-" * 60)

        is_valid = expert_qwen235B.validate_output(instruction)
        print(f"\n格式验证: {'通过' if is_valid else '失败'}")

        expert_qwen235B.unload_model()
    else:
        print("模型加载失败")

    print("\n测试完成!")