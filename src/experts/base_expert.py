"""
专家基类
定义所有专家的统一接口和共享功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseExpert(ABC):
    """专家基类 - 定义所有专家的统一接口"""

    def __init__(self,
                 expert_name: str,
                 lora_path: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        初始化专家

        Args:
            expert_name: 专家名称
            lora_path: LoRA权重路径（可选）
            config: 专家配置（可选）
        """
        self.expert_name = expert_name
        self.lora_path = lora_path
        self.config = config or {}

        # 从配置中加载参数
        self.specialization = self.config.get('specialization', '')
        self.domains = self.config.get('domains', [])

    @abstractmethod
    def get_prompt_template(self) -> str:
        """
        获取该专家的prompt模板

        Returns:
            str: prompt模板字符串
        """
        pass

    @abstractmethod
    def preprocess_input(self, input_data: Any) -> Dict:
        """
        预处理输入数据

        Args:
            input_data: 原始输入数据

        Returns:
            dict: 预处理后的数据字典
        """
        pass

    @abstractmethod
    def build_prompt(self, preprocessed_data: Dict) -> str:
        """
        构建完整的prompt

        Args:
            preprocessed_data: 预处理后的数据

        Returns:
            str: 完整的prompt字符串
        """
        pass

    @abstractmethod
    def postprocess_output(self, raw_output: str) -> Dict:
        """
        后处理模型输出

        Args:
            raw_output: 模型原始输出

        Returns:
            dict: 处理后的结构化输出
        """
        pass

    def validate_output(self, output: Dict) -> bool:
        """
        验证输出格式是否正确

        Args:
            output: 输出字典

        Returns:
            bool: 是否有效
        """
        # 基础验证：检查必需字段
        required_fields = ['Definition', 'Emphasis & Caution', 'Things to Avoid']

        # TODO: 实现验证逻辑
        return True

    def get_generation_config(self) -> Dict:
        """
        获取生成配置

        Returns:
            dict: 生成参数配置
        """
        return {
            'max_new_tokens': self.config.get('max_new_tokens', 2048),
            'temperature': self.config.get('temperature', 0.7),
            'top_p': self.config.get('top_p', 0.9),
            'top_k': self.config.get('top_k', 50),
            'repetition_penalty': self.config.get('repetition_penalty', 1.1)
        }

    def generate(self,
                 input_data: Any,
                 language_model,
                 **kwargs) -> Dict:
        """
        完整的生成流程（模板方法）

        Args:
            input_data: 输入数据
            language_model: 语言模型实例
            **kwargs: 额外参数

        Returns:
            dict: 生成结果
        """
        # 1. 预处理
        preprocessed = self.preprocess_input(input_data)

        # 2. 构建prompt
        prompt = self.build_prompt(preprocessed)

        # 3. 生成
        generation_config = self.get_generation_config()
        generation_config.update(kwargs)  # 允许覆盖配置

        raw_output = language_model.generate(prompt, **generation_config)

        # 4. 后处理
        result = self.postprocess_output(raw_output)

        # 5. 验证
        is_valid = self.validate_output(result)
        result['is_valid'] = is_valid
        result['expert_name'] = self.expert_name

        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.expert_name}')"
