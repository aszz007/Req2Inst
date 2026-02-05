"""
专家基类 - 定义统一的专家接口
功能:
  - 统一的专家接口定义
  - LoRA权重管理
  - 指令生成和验证
  - 支持多模型版本

作者: Expert System
日期: 2025-01-30
更新: 2025-02-03 - 支持版本参数和路径配置
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from models.language_model import LanguageModel
from src.utils.logger import get_logger

logger = get_logger('experts.base')


class BaseExpert(ABC):
    """
    专家基类 - 定义所有专家的统一接口

    所有专家(Text, Image, UML, General)都应继承此类
    """

    def __init__(self,
                 expert_name: str,
                 base_model_path: str,
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True,
                 version: Optional[str] = None):
        """
        初始化专家

        Args:
            expert_name: 专家名称(如'text_expert', 'image_expert_qwen25')
            base_model_path: 基础模型路径
            lora_path: LoRA权重路径(None则不加载)
            use_4bit: 是否使用4bit量化
            version: 模型版本(如'qwen2.5', 'qwen3'),用于Image/UML专家
        """
        self.expert_name = expert_name
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.use_4bit = use_4bit
        self.version = version

        self.model = None
        self.is_model_loaded = False

        logger.info(f"初始化专家: {expert_name}")
        logger.info(f"基础模型: {base_model_path}")
        if version:
            logger.info(f"模型版本: {version}")
        if lora_path:
            logger.info(f"LoRA路径: {lora_path}")

    def load_model(self) -> bool:
        """
        加载模型(基础模型 + LoRA权重)

        Returns:
            bool: 是否加载成功
        """
        try:
            logger.info(f"加载{self.expert_name}的模型...")

            # 加载基础语言模型
            self.model = LanguageModel(
                model_path=self.base_model_path,
                use_4bit=self.use_4bit
            )

            # 如果提供了LoRA路径,加载LoRA权重
            if self.lora_path:
                lora_path = Path(self.lora_path)
                if lora_path.exists():
                    logger.info(f"加载LoRA权重: {self.lora_path}")
                    success = self.model.load_lora_from_path(str(self.lora_path))
                    if not success:
                        logger.warning("LoRA加载失败,使用基础模型")
                else:
                    logger.warning(f"LoRA路径不存在: {self.lora_path}")
                    logger.warning("使用基础模型(未微调)")

            self.is_model_loaded = True
            logger.info("模型加载完成")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.is_model_loaded = False
            return False

    def unload_model(self) -> bool:
        """
        卸载模型(释放显存)

        Returns:
            bool: 是否卸载成功
        """
        try:
            if self.model:
                # 卸载LoRA
                if self.model.is_lora_loaded:
                    self.model.unload_lora()

                # 清理模型
                del self.model
                self.model = None

                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.is_model_loaded = False
                logger.info("模型已卸载")

            return True

        except Exception as e:
            logger.error(f"模型卸载失败: {e}")
            return False

    @abstractmethod
    def generate_instruction(self, input_data: Any) -> str:
        """
        生成指令的核心方法(子类必须实现)

        Args:
            input_data: 输入数据(文本/图像描述/UML JSON等)

        Returns:
            str: 生成的众包指令
        """
        pass

    @abstractmethod
    def validate_output(self, instruction: str) -> bool:
        """
        验证输出格式(子类必须实现)

        Args:
            instruction: 生成的指令

        Returns:
            bool: 是否符合格式要求
        """
        pass

    def _generate_with_model(self,
                            prompt: str,
                            max_new_tokens: int = 2048,
                            temperature: float = 0.7,
                            top_p: float = 0.9,
                            top_k: int = 50,
                            repetition_penalty: float = 1.1) -> str:
        """
        使用模型生成文本(通用方法)

        Args:
            prompt: 完整的prompt
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚

        Returns:
            str: 生成的文本
        """
        if not self.is_model_loaded:
            logger.error("模型未加载,无法生成")
            return ""

        try:
            generated_text = self.model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )

            # 调试日志：显示原始生成内容
            logger.debug(f"原始生成内容（前500字符）：\n{generated_text[:500]}")

            # 提取三段式指令（移除多余内容）
            extracted_text = self._extract_three_part_instruction(generated_text)

            # 调试日志：显示提取后内容
            if extracted_text != generated_text:
                logger.debug(f"提取后内容：\n{extracted_text}")
            else:
                logger.debug("未找到标准三段式格式，返回原始内容")

            return extracted_text

        except Exception as e:
            logger.error(f"生成失败: {e}")
            return ""

    def _extract_three_part_instruction(self, text: str) -> str:
        """
        提取三段式指令，移除多余内容

        增强功能：
        1. 优先查找标准标签格式
        2. 如果没有标签，尝试按句子智能分割
        3. 自动添加缺失的标签

        Args:
            text: 原始生成文本

        Returns:
            str: 提取的三段式指令
        """
        if not text:
            return ""

        # 按行分割
        lines = text.split('\n')

        # 查找三段式指令的三个部分
        definition_line = None
        emphasis_line = None
        avoid_line = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 检查每一行是否是三段式的开头
            if line_stripped.startswith('Definition:'):
                definition_line = i
            elif line_stripped.startswith('Emphasis & Caution:') or line_stripped.startswith('Emphasis and Caution:'):
                emphasis_line = i
            elif line_stripped.startswith('Things to Avoid:'):
                avoid_line = i

        # 如果找到完整的三段式，只保留这三行
        if definition_line is not None and emphasis_line is not None and avoid_line is not None:
            # 确保行号顺序正确
            if definition_line < emphasis_line < avoid_line:
                # 提取三行，并清理每行的尾部多余内容
                extracted_lines = []
                for line_idx in [definition_line, emphasis_line, avoid_line]:
                    line = lines[line_idx].strip()
                    # 如果行中包含中文或其他垃圾内容，截断
                    cleaned_line = self._clean_instruction_line(line)
                    extracted_lines.append(cleaned_line)
                return '\n'.join(extracted_lines)

        # 如果没有找到标准标签，尝试智能分割
        logger.debug("未找到标准三段式标签，尝试智能分割")
        return self._smart_split_to_three_parts(text)

    def _smart_split_to_three_parts(self, text: str) -> str:
        """
        智能分割无标签的内容为三段式格式

        策略：
        1. 查找"In this task"开头的句子作为Definition
        2. 查找"Focus", "Ensure", "Pay attention"等关键词的句子作为Emphasis
        3. 查找"Do not", "Avoid", "Never"等关键词的句子作为Things to Avoid

        Args:
            text: 原始文本

        Returns:
            str: 格式化的三段式指令
        """
        # 移除"Task Instructions:"等标题行
        text = text.replace('Task Instructions:', '').strip()

        # 按句子分割（支持. ! ?结尾）
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        definition = None
        emphasis = None
        avoid = None

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # 识别Definition（包含"In this task"或"draw bounding box"）
            if not definition and ('in this task' in sentence_lower or
                                  ('draw' in sentence_lower and 'box' in sentence_lower) or
                                  ('annotate' in sentence_lower and 'this task' not in sentence_lower)):
                # 确保以"In this task"开头
                if not sentence.startswith('In this task'):
                    sentence = 'In this task, ' + sentence[0].lower() + sentence[1:]
                definition = sentence

            # 识别Emphasis（包含"Focus", "Ensure", "Pay attention"等）
            elif not emphasis and any(kw in sentence_lower for kw in
                                     ['focus', 'ensure', 'pay attention', 'must', 'should',
                                      'important', 'critical', 'key']):
                emphasis = sentence

            # 识别Things to Avoid（包含"Do not", "Avoid", "Never"等）
            elif not avoid and any(kw in sentence_lower for kw in
                                   ['do not', "don't", 'avoid', 'never', 'not', 'skip']):
                avoid = sentence

        # 如果有未分配的句子，智能分配
        if len(sentences) >= 3:
            if not definition:
                definition = sentences[0]
                if not definition.startswith('In this task'):
                    definition = 'In this task, ' + definition[0].lower() + definition[1:]
            if not emphasis:
                emphasis = sentences[1] if len(sentences) > 1 else '-'
            if not avoid:
                avoid = sentences[2] if len(sentences) > 2 else '-'
        elif len(sentences) == 2:
            if not definition:
                definition = sentences[0]
            if not emphasis:
                emphasis = sentences[1]
            if not avoid:
                avoid = '-'
        elif len(sentences) == 1:
            if not definition:
                definition = sentences[0]
            emphasis = '-'
            avoid = '-'

        # 确保都有值
        definition = definition or 'In this task, complete the required task.'
        emphasis = emphasis or '-'
        avoid = avoid or '-'

        # 组装三段式格式
        formatted_instruction = f"""Definition: {definition}
Emphasis & Caution: {emphasis}
Things to Avoid: {avoid}"""

        logger.debug(f"智能分割完成：\n{formatted_instruction}")

        return formatted_instruction

    def _clean_instruction_line(self, line: str) -> str:
        """
        清理单行指令，移除尾部的多余内容和重复标签

        Args:
            line: 单行指令

        Returns:
            str: 清理后的行
        """
        # 首先移除重复的标签前缀
        prefixes = [
            'Definition:',
            'Emphasis & Caution:',
            'Emphasis and Caution:',
            'Things to Avoid:'
        ]

        for prefix in prefixes:
            if line.startswith(prefix):
                content = line[len(prefix):].strip()
                # 检查内容是否又以同样的前缀开头（重复标签）
                if content.startswith(prefix):
                    content = content[len(prefix):].strip()
                line = f"{prefix} {content}"
                break

        # 常见的垃圾模式（中文、问题等）
        unwanted_patterns = [
            '在不失准确性',
            '请对以下',
            '这句话的',
            '摘要',
            '反义词',
            '总结',
            '翻译',
            '目的 观察',
            '方法 ',
            '结果 ',
            '结论 ',
        ]

        for pattern in unwanted_patterns:
            if pattern in line:
                idx = line.find(pattern)
                if idx > 0:
                    line = line[:idx].strip()
                    break

        return line

    def get_expert_info(self) -> Dict[str, Any]:
        """
        获取专家信息

        Returns:
            dict: 专家信息
        """
        info = {
            'expert_name': self.expert_name,
            'base_model': self.base_model_path,
            'lora_path': self.lora_path,
            'is_model_loaded': self.is_model_loaded,
            'use_4bit': self.use_4bit,
            'version': self.version
        }

        if self.model and self.is_model_loaded:
            info['lora_status'] = self.model.get_lora_status()

        return info

    def __repr__(self) -> str:
        """字符串表示"""
        version_str = f", version={self.version}" if self.version else ""
        return f"<{self.__class__.__name__}: {self.expert_name}{version_str}, loaded={self.is_model_loaded}>"

    def __enter__(self):
        """上下文管理器: 进入时加载模型"""
        if not self.is_model_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器: 退出时卸载模型"""
        self.unload_model()


if __name__ == "__main__":
    print("=" * 60)
    print("专家基类测试")
    print("=" * 60)

    print("\n注意: BaseExpert是抽象类,不能直接实例化")
    print("需要实现具体的专家类(TextExpert, ImageExpert等)")
    print("\n预期的专家类结构:")
    print("  - TextExpert: 继承BaseExpert,实现文本指令生成")
    print("  - ImageExpert: 继承BaseExpert,实现图像指令生成")
    print("  - UMLExpert: 继承BaseExpert,实现UML指令生成")
    print("  - GeneralExpert: 继承BaseExpert,通用兜底专家")

    print("\n测试完成!")