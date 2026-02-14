"""
专家基类 - 定义统一的专家接口
功能:
  - 统一的专家接口定义
  - LoRA权重管理
  - 指令生成和验证
  - 支持Qwen3-8B（默认）和Qwen-7B-Chat（遗留）

作者: Expert System
日期: 2025-02-13
更新: 支持Qwen3-8B作为默认模型
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

    # 类变量：共享的基础模型
    _shared_base_model = None
    _shared_base_model_path = None

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
                            repetition_penalty: float = 1.1,
                            sample_index: int = None,
                            verbose: bool = True) -> str:
        """
        使用模型生成文本(通用方法)

        Args:
            prompt: 完整的prompt
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚
            sample_index: 样本索引（用于控制日志输出，仅前3个样本输出详细日志）
            verbose: 是否输出详细日志

        Returns:
            str: 生成的文本

        Note:
            停止token由LanguageModel内部处理(eos_token_id, <|im_end|>等)
        """
        if not self.is_model_loaded:
            logger.error("模型未加载,无法生成")
            return ""

        try:
            # 只在前3个样本输出详细调试信息
            show_debug = verbose and (sample_index is None or sample_index < 3)

            if show_debug:
                logger.info("=" * 80)
                logger.info(f"[调试] 样本 {sample_index + 1 if sample_index is not None else 'N/A'} - 完整Prompt内容:")
                logger.info("-" * 80)
                logger.info(prompt)
                logger.info("=" * 80)
                logger.info(f"[调试] 生成参数: temp={temperature}, top_p={top_p}, top_k={top_k}, rep_penalty={repetition_penalty}")

            generated_text = self.model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )

            if show_debug:
                logger.info(f"[调试] 原始生成内容长度: {len(generated_text)} 字符")
                logger.info(f"[调试] 原始生成内容（前500字符）：\n{generated_text[:500]}")
                if len(generated_text) > 500:
                    logger.info(f"[调试] 原始生成内容（后200字符）：\n{generated_text[-200:]}")

            return generated_text

        except Exception as e:
            logger.error(f"生成失败: {e}")
            return ""

    def _generate_batch_with_model(self,
                                   prompts: list,
                                   max_new_tokens: int = 2048,
                                   temperature: float = 0.7,
                                   top_p: float = 0.9,
                                   top_k: int = 50,
                                   repetition_penalty: float = 1.1,
                                   batch_size: int = None,
                                   start_index: int = 0,
                                   verbose: bool = True) -> list:
        """
        批量生成文本（通用方法，提高GPU利用率）

        Args:
            prompts: prompt列表
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚
            batch_size: 批处理大小（None则自动选择）
            start_index: 起始索引（用于日志输出）
            verbose: 是否输出详细日志

        Returns:
            list: 生成的文本列表
        """
        if not self.is_model_loaded:
            logger.error("模型未加载,无法生成")
            return [""] * len(prompts)

        try:
            # 只在前3个样本输出详细日志
            show_debug = verbose and start_index < 3

            if show_debug:
                logger.info(f"批量生成 - 共{len(prompts)}个样本，起始索引{start_index}")

            # 使用LanguageModel的批量生成方法
            if hasattr(self.model, 'generate_batch'):
                results = self.model.generate_batch(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    batch_size=batch_size
                )

                return results
            else:
                # 降级到逐个生成（兼容旧版本）
                logger.warning("模型不支持批量生成，降级到逐个生成")
                results = []
                for i, prompt in enumerate(prompts):
                    result = self._generate_with_model(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        sample_index=start_index + i,
                        verbose=verbose
                    )
                    results.append(result)
                return results

        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            return [""] * len(prompts)

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
        # logger.info("=" * 80)
        # logger.info("[提取开始] 原始生成文本:")
        # logger.info("-" * 80)
        # logger.info(text)
        # logger.info("=" * 80)

        if not text:
            logger.warning("[提取结束] 输入文本为空")
            return ""

        # 按行分割
        lines = text.split('\n')
        # logger.info(f"[提取] 分割为 {len(lines)} 行")

        # 显示所有行（用于调试）
        # for i, line in enumerate(lines):
        #     logger.info(f"[提取] 行{i}: {line[:100]}")

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
                # logger.info(f"[提取] 找到Definition在行{i}")
            elif line_stripped.startswith('Emphasis & Caution:') or line_stripped.startswith('Emphasis and Caution:'):
                emphasis_line = i
                # logger.info(f"[提取] 找到Emphasis在行{i}")
            elif line_stripped.startswith('Things to Avoid:'):
                avoid_line = i
                # logger.info(f"[提取] 找到Things to Avoid在行{i}")

        # logger.info(f"[提取] 找到的行号 - Definition: {definition_line}, Emphasis: {emphasis_line}, Avoid: {avoid_line}")

        # 如果找到完整的三段式，只保留这三行
        if definition_line is not None and emphasis_line is not None and avoid_line is not None:
            # logger.info("[提取] 找到完整的三段式标签")
            # 确保行号顺序正确
            if definition_line < emphasis_line < avoid_line:
                # logger.info("[提取] 行号顺序正确，开始清理每一行")
                # 提取三行，并清理每行的尾部多余内容
                extracted_lines = []
                for idx, line_idx in enumerate([definition_line, emphasis_line, avoid_line]):
                    line = lines[line_idx].strip()
                    # logger.info(f"[提取] 清理第{idx+1}行: {line}")
                    # 如果行中包含中文或其他垃圾内容，截断
                    cleaned_line = self._clean_instruction_line(line)
                    # logger.info(f"[提取] 清理后第{idx+1}行: {cleaned_line}")
                    extracted_lines.append(cleaned_line)

                # 检查提取的内容是否有效（不全是"-"）
                extracted_text = '\n'.join(extracted_lines)
                # 检查Definition行是否有实际内容
                def_content = extracted_lines[0].split(':', 1)[1].strip() if ':' in extracted_lines[0] else ''

                # 检查Definition内容是否包含重复的标签关键词
                invalid_keywords = ['Definition:', 'Emphasis', 'Things to Avoid', 'Caution']
                has_invalid_keyword = any(keyword in def_content for keyword in invalid_keywords)

                if def_content and def_content != '-' and not has_invalid_keyword:
                    # 强制确保Definition以"In this task,"开头
                    extracted_lines = self._ensure_definition_format(extracted_lines)
                    extracted_text = '\n'.join(extracted_lines)

                    # logger.info("[提取结束] 成功提取三段式指令")
                    # logger.info("=" * 80)
                    # logger.info("[提取结果]")
                    # logger.info("-" * 80)
                    # logger.info(extracted_text)
                    # logger.info("=" * 80)
                    return extracted_text
                else:
                    if has_invalid_keyword:
                        logger.warning(f"[提取] 检测到重复标签，Definition内容无效: {def_content}")
                    else:
                        logger.warning("[提取] 提取的Definition内容为空，尝试智能分割")
                    # 继续执行智能分割逻辑
            else:
                logger.warning(f"[提取] 行号顺序不正确: {definition_line}, {emphasis_line}, {avoid_line}")
        else:
            logger.warning("[提取] 未找到完整的三段式标签")

        # 如果没有找到标准标签，尝试智能分割
        # logger.info("[提取] 尝试智能分割")
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
        清理单行指令,移除尾部的多余内容和重复标签

        增强版本: 更强力地检测和清理各类问题

        Args:
            line: 单行指令

        Returns:
            str: 清理后的行
        """
        import re

        # logger.info(f"[清理前] {line}")

        # 定义所有可能的标签前缀
        prefixes = [
            'Definition:',
            'Emphasis & Caution:',
            'Emphasis and Caution:',
            'Things to Avoid:'
        ]

        # 查找行的标签
        current_prefix = None
        for prefix in prefixes:
            if line.startswith(prefix):
                current_prefix = prefix
                break

        if current_prefix is None:
            # 没有找到标签,返回原始行
            # logger.info("[清理] 未找到标签前缀,返回原始行")
            return line

        # 提取内容部分
        content = line[len(current_prefix):].strip()
        # logger.info(f"[步骤0] 提取内容: {content[:min(100, len(content))]}...")

        # === 步骤1: 强力移除所有重复标签 ===
        # 递归移除直到没有任何标签前缀
        original_content = content
        max_iterations = 10
        for iteration in range(max_iterations):
            found_duplicate = False
            for check_prefix in prefixes:
                if content.startswith(check_prefix):
                    content = content[len(check_prefix):].strip()
                    found_duplicate = True
                    # logger.info(f"[步骤1.{iteration}] 移除重复标签: {check_prefix}")
                    break
            if not found_duplicate:
                break

        # if content != original_content:
        #     logger.info(f"[步骤1完成] 移除重复标签后: {content[:min(100, len(content))]}...")

        # 如果内容为空或只是占位符,直接返回
        if not content or content == '-':
            # logger.info("[步骤1] 内容为空,返回占位符")
            return f"{current_prefix} -"

        # === 步骤2: 检测并截断句号后的垃圾模式 ===
        # 这些模式通常表示训练数据泄露
        garbage_patterns = [
            # 最常见的垃圾模式 - 扩展版本
            r'\.is a (list|type|kind|form|way|computer program|software|document|method|system|tool)',
            r'\.is (often used|used|typically|commonly|generally|usually|one of|the)',
            r'\.is (a type|an|the|one of|part of)',
            # 文档管理相关的泄露模式
            r'\.(document|software|system|program|application|tool|platform|service)',
            # 其他垃圾模式
            r'\.(it|this|that|these|those) (is|are|was|were|can|could|will|would|may|might)',
            r'\.the (purpose|goal|aim|objective|main|primary|key|first)',
            r'\.(in order|to ensure|for|with|by|through|via)',
            # 新增: 捕获 ".xxxx" 模式 (句号后跟小写单词)
            r'\.[a-z]{2,}',
        ]

        original_content = content
        for pattern in garbage_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # 找到垃圾模式,截断到句号位置(保留句号)
                idx = match.start()
                content = content[:idx + 1].strip()
                # logger.info(f"[步骤2] 检测到垃圾模式: {pattern}, 截断到位置{idx}")
                break

        # if content != original_content:
        #     logger.info(f"[步骤2完成] 截断垃圾内容后: {content}")

        # === 步骤3: 检测中文字符并截断 ===
        chinese_match = re.search(r'[\u4e00-\u9fff]', content)
        if chinese_match:
            idx = chinese_match.start()
            # 回溯到最近的句点
            truncate_pos = idx
            for i in range(idx - 1, max(0, idx - 50), -1):
                if content[i] in '.!?':
                    truncate_pos = i + 1
                    break
            content = content[:truncate_pos].strip()
            # logger.info(f"[步骤3] 检测到中文字符,截断到位置{truncate_pos}")

        # === 步骤4: 移除以小写字母开头的句子片段 ===
        # 通常这些是训练数据泄露
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) > 1:
            # 检查最后一个句子
            last_sentence = sentences[-1].strip()
            if last_sentence and len(last_sentence) > 0 and last_sentence[0].islower():
                # 最后一个句子以小写开头,很可能是垃圾,移除它
                content = ' '.join(sentences[:-1]).strip()
                # logger.info(f"[步骤4] 移除小写开头的尾部句子: {last_sentence[:min(50, len(last_sentence))]}")

        # === 步骤5: 确保以句号结尾 ===
        if content and not content.endswith(('.', '!', '?', '-')):
            content += '.'
            # logger.info("[步骤5] 添加结尾句号")

        # 重新组装清理后的行
        cleaned_line = f"{current_prefix} {content}"
        # logger.info(f"[清理后] {cleaned_line}")

        return cleaned_line

    def _ensure_definition_format(self, lines: list) -> list:
        """
        确保Definition行以"In this task,"开头

        规则：
        - 如果Definition行已经以"In this task,"开头，保持不变
        - 如果Definition行是其他格式（如"Draw bounding boxes..."），自动添加"In this task,"前缀

        Args:
            lines: 三行指令列表

        Returns:
            list: 修正后的三行指令列表
        """
        if not lines or len(lines) < 1:
            return lines

        # 提取Definition行
        definition_line = lines[0]

        if not definition_line.startswith('Definition:'):
            return lines

        # 提取内容部分
        content = definition_line[len('Definition:'):].strip()

        # 检查是否已经以"In this task,"开头
        if content.lower().startswith('in this task,'):
            # logger.info("[格式检查] Definition已包含'In this task,'前缀")
            return lines

        # 如果不是，添加前缀
        # 将首字母小写（因为会跟在"In this task,"后面）
        if content:
            content = content[0].lower() + content[1:] if len(content) > 1 else content.lower()

        # 重新组装
        new_definition = f"Definition: In this task, {content}"
        # logger.info(f"[格式修正] Definition添加'In this task,'前缀")
        # logger.info(f"[格式修正] 修正前: {definition_line}")
        # logger.info(f"[格式修正] 修正后: {new_definition}")

        # 替换第一行
        lines[0] = new_definition

        return lines


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

    @classmethod
    def load_shared_base_model(cls, base_model_path: str, use_4bit: bool = True) -> bool:
        """
        加载共享的基础模型（所有专家共用）

        这个方法用于统一测试场景，避免重复加载基础模型。
        加载后，各个专家只需要切换LoRA权重即可。

        Args:
            base_model_path: 基础模型路径
            use_4bit: 是否使用4bit量化

        Returns:
            bool: 是否加载成功
        """
        try:
            if cls._shared_base_model is not None and cls._shared_base_model_path == base_model_path:
                logger.info(f"共享基础模型已加载: {base_model_path}")
                return True

            logger.info(f"加载共享基础模型: {base_model_path}")
            cls._shared_base_model = LanguageModel(
                model_path=base_model_path,
                use_4bit=use_4bit
            )
            cls._shared_base_model_path = base_model_path
            logger.info("共享基础模型加载成功")
            return True

        except Exception as e:
            logger.error(f"共享基础模型加载失败: {e}")
            cls._shared_base_model = None
            cls._shared_base_model_path = None
            return False

    @classmethod
    def unload_shared_base_model(cls) -> bool:
        """
        卸载共享的基础模型

        Returns:
            bool: 是否卸载成功
        """
        try:
            if cls._shared_base_model:
                # 卸载LoRA（如果有）
                if cls._shared_base_model.is_lora_loaded:
                    cls._shared_base_model.unload_lora()

                # 清理模型
                del cls._shared_base_model
                cls._shared_base_model = None
                cls._shared_base_model_path = None

                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("共享基础模型已卸载")

            return True

        except Exception as e:
            logger.error(f"共享基础模型卸载失败: {e}")
            return False

    def load_model_with_shared_base(self) -> bool:
        """
        使用共享的基础模型加载专家（仅加载LoRA权重）

        这个方法假设共享基础模型已经通过load_shared_base_model加载。
        专家只需要加载自己的LoRA权重即可。

        Returns:
            bool: 是否加载成功
        """
        try:
            if self.__class__._shared_base_model is None:
                logger.error("共享基础模型未加载，请先调用load_shared_base_model")
                return False

            if self.base_model_path != self.__class__._shared_base_model_path:
                logger.warning(f"基础模型路径不匹配：专家期望{self.base_model_path}，共享模型是{self.__class__._shared_base_model_path}")
                logger.warning("将使用共享模型")

            logger.info(f"使用共享基础模型加载{self.expert_name}...")

            # 直接引用共享模型
            self.model = self.__class__._shared_base_model

            # 如果提供了LoRA路径，加载LoRA权重
            if self.lora_path:
                lora_path = Path(self.lora_path)
                if lora_path.exists():
                    logger.info(f"加载LoRA权重: {self.lora_path}")
                    success = self.model.load_lora_from_path(str(self.lora_path))
                    if not success:
                        logger.warning("LoRA加载失败，使用基础模型")
                else:
                    logger.warning(f"LoRA路径不存在: {self.lora_path}")
                    logger.warning("使用基础模型（未微调）")

            self.is_model_loaded = True
            logger.info(f"{self.expert_name}加载完成（使用共享基础模型）")
            return True

        except Exception as e:
            logger.error(f"使用共享基础模型加载失败: {e}")
            self.is_model_loaded = False
            return False

    def unload_model_keep_shared_base(self) -> bool:
        """
        卸载专家模型但保留共享的基础模型（仅卸载LoRA权重）

        Returns:
            bool: 是否卸载成功
        """
        try:
            if self.model:
                # 只卸载LoRA，不清理基础模型
                if self.model.is_lora_loaded:
                    self.model.unload_lora()

                # 不删除model引用，因为它指向共享模型
                self.model = None
                self.is_model_loaded = False
                logger.info(f"{self.expert_name}已卸载（保留共享基础模型）")

            return True

        except Exception as e:
            logger.error(f"卸载失败: {e}")
            return False


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