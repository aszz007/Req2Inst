"""
语言模型接口（重构版）
功能：
  - 支持LoRA权重动态加载/卸载
  - 4bit量化优化
  - 统一的generate接口
支持模型：Qwen-7B-Chat
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 尝试导入流式生成支持（仅qwen_text环境需要）
try:
    from transformers_stream_generator import init_stream_support
    init_stream_support()
    STREAM_SUPPORT = True
except ImportError:
    STREAM_SUPPORT = False
    # 在vision环境中这是正常的，不影响使用
    pass

from peft import PeftModel, PeftConfig
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from config.settings import get_path_config, get_device_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LanguageModel:
    """大语言模型类 - 支持4bit量化和LoRA动态加载"""

    def __init__(self, model_path: Optional[str] = None, use_4bit: bool = True):
        """
        初始化语言模型

        Args:
            model_path: 模型本地路径（None则使用配置）
            use_4bit: 是否使用4bit量化
        """
        # 获取配置
        path_cfg = get_path_config()
        device_cfg = get_device_config()

        self.model_path = model_path or str(path_cfg.QWEN_7B_CHAT_PATH)
        self.device = device_cfg.get_device()
        self.use_4bit = use_4bit

        self.model = None
        self.tokenizer = None
        self.current_lora_path = None  # 当前加载的LoRA路径
        self.is_lora_loaded = False    # LoRA加载状态

        logger.info(f"初始化语言模型")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"设备: {self.device}")
        logger.info(f"4bit量化: {self.use_4bit}")

        self._load_base_model()

    def _load_base_model(self):
        """加载基础模型"""
        try:
            logger.info("加载基础模型...")

            # 4bit量化配置
            if self.use_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='left',
            )

            # 设置特殊tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = '<|endoftext|>'
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = '<|im_end|>'

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.use_4bit else None,
                low_cpu_mem_usage=True
            )

            self.model.eval()
            logger.info("基础模型加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def load_lora_from_path(self, lora_path: str) -> bool:
        """
        从路径动态加载LoRA权重

        Args:
            lora_path: LoRA权重路径（绝对路径或相对路径）

        Returns:
            bool: 是否加载成功
        """
        try:
            lora_path = Path(lora_path)

            # 检查路径是否存在
            if not lora_path.exists():
                logger.error(f"LoRA路径不存在: {lora_path}")
                return False

            # 如果已加载相同的LoRA，跳过
            if self.current_lora_path == str(lora_path):
                logger.info(f"LoRA已加载: {lora_path}")
                return True

            # 如果已加载其他LoRA，先卸载
            if self.is_lora_loaded:
                logger.info("卸载旧的LoRA权重...")
                self.unload_lora()

            logger.info(f"加载LoRA权重: {lora_path}")

            # 使用PEFT加载LoRA
            self.model = PeftModel.from_pretrained(
                self.model,
                str(lora_path),
                is_trainable=False
            )

            self.current_lora_path = str(lora_path)
            self.is_lora_loaded = True

            logger.info("LoRA加载成功")
            return True

        except Exception as e:
            logger.error(f"LoRA加载失败: {e}")
            return False

    def unload_lora(self) -> bool:
        """
        卸载当前的LoRA权重

        Returns:
            bool: 是否卸载成功
        """
        try:
            if not self.is_lora_loaded:
                logger.info("没有已加载的LoRA")
                return True

            logger.info("卸载LoRA权重...")

            # 获取基础模型
            self.model = self.model.unload()

            self.current_lora_path = None
            self.is_lora_loaded = False

            logger.info("LoRA卸载成功")
            return True

        except Exception as e:
            logger.error(f"LoRA卸载失败: {e}")
            return False

    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                 repetition_penalty: float = 1.1) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚

        Returns:
            str: 生成的文本
        """
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_length = inputs['input_ids'].shape[1]
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # 准备停止tokens
            stop_tokens = []
            if self.tokenizer.eos_token_id is not None:
                stop_tokens.append(self.tokenizer.eos_token_id)

            im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
                stop_tokens.append(im_end_id)

            stop_tokens = list(set(stop_tokens))

            # 生成配置 - 修复：正确应用所有停止标记
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": True if temperature > 0 else False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": stop_tokens
            }

            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)

            # 解码输出
            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 清理特殊标记和多余内容
            generated_text = self._clean_generated_text(generated_text)

            return generated_text

        except Exception as e:
            logger.error(f"生成失败: {e}")
            return ""

    def _clean_generated_text(self, text: str) -> str:
        """
        清理生成的文本，移除特殊标记和多余内容

        Args:
            text: 原始生成文本

        Returns:
            str: 清理后的文本
        """
        # 移除特殊标记
        text = text.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()

        # 如果文本中包含多个训练样本（通常以中文或问题开头），只保留第一个完整指令
        # 检测常见的训练数据模式
        unwanted_patterns = [
            '在不失准确性',
            '请对以下',
            '这句话的',
            '摘要',
            '反义词',
            '总结',
            '翻译',
            '目的 观察',
        ]

        for pattern in unwanted_patterns:
            if pattern in text:
                # 找到模式出现的位置，截断到该位置之前
                idx = text.find(pattern)
                if idx > 0:
                    text = text[:idx].strip()
                    break

        return text

    def get_lora_status(self) -> dict:
        """
        获取LoRA状态信息

        Returns:
            dict: LoRA状态
        """
        return {
            'is_loaded': self.is_lora_loaded,
            'current_path': self.current_lora_path,
            'base_model': self.model_path
        }


class InstructionGenerator:
    """
    指令生成器（重构版）
    - 移除硬编码的expert映射
    - 支持外部传入prompt
    - 支持从路径加载LoRA
    """

    def __init__(self, model_path: Optional[str] = None, use_4bit: bool = True):
        """
        初始化指令生成器

        Args:
            model_path: 模型路径（None则使用配置）
            use_4bit: 是否使用4bit量化
        """
        self.language_model = LanguageModel(
            model_path=model_path,
            use_4bit=use_4bit
        )
        logger.info("指令生成器初始化完成")

    def load_expert(self, expert_name_or_path: str) -> bool:
        """
        加载专家LoRA权重

        Args:
            expert_name_or_path: 专家名称（如'text_expert'）或完整路径

        Returns:
            bool: 是否加载成功
        """
        # 尝试作为专家名称处理
        path_cfg = get_path_config()
        lora_path = path_cfg.get_expert_weight_path(expert_name_or_path)

        # 如果路径不存在，尝试作为完整路径处理
        if not lora_path.exists():
            lora_path = Path(expert_name_or_path)

        return self.language_model.load_lora_from_path(str(lora_path))

    def unload_expert(self) -> bool:
        """卸载当前专家"""
        return self.language_model.unload_lora()

    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, repetition_penalty: float = 1.1) -> str:
        """
        生成众包指令（接受外部构建的prompt）

        Args:
            prompt: 已构建好的完整prompt（由外部Prompt模板生成）
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚

        Returns:
            str: 生成的指令
        """
        return self.language_model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )

    def get_expert_status(self) -> dict:
        """获取当前专家状态"""
        return self.language_model.get_lora_status()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("语言模型测试（重构版）")
    print("=" * 60)

    # 测试1: 基础模型加载
    print("\n【测试1】基础模型加载")
    print("-" * 60)
    generator = InstructionGenerator(use_4bit=True)

    # 测试2: 查看状态
    print("\n【测试2】查看LoRA状态")
    print("-" * 60)
    status = generator.get_expert_status()
    print(f"LoRA已加载: {status['is_loaded']}")
    print(f"当前路径: {status['current_path']}")
    print(f"基础模型: {status['base_model']}")

    # 测试3: 生成指令（不加载LoRA）
    print("\n【测试3】基础模型生成（无LoRA）")
    print("-" * 60)
    test_prompt = """<|im_start|>system
你是一个专业的众包任务指令生成专家。<|im_end|>
<|im_start|>user
请生成一份图像标注任务的众包指令。<|im_end|>
<|im_start|>assistant
"""

    instruction = generator.generate(test_prompt, max_new_tokens=512, temperature=0.7)
    print(f"生成的指令:\n{instruction}")

    # 测试4: 加载LoRA（如果存在）
    print("\n【测试4】加载专家LoRA")
    print("-" * 60)
    success = generator.load_expert('text_expert')
    if success:
        print("LoRA加载成功")
        status = generator.get_expert_status()
        print(f"当前LoRA: {status['current_path']}")
    else:
        print("LoRA未找到（可能还未训练）")

    print("\n测试完成！")

# 使用示例：
# 方法1：通过环境管理脚本运行（推荐）
# python scripts/run_with_env.py --env text --script models/language_model.py inputs/text/requirement.txt
#
# 方法2：直接在qwen_text环境中运行
# conda activate qwen_text
# python models/language_model.py inputs/text/requirement.txt
#
# 注意：Text Expert只使用qwen-7B-Chat模型，无需指定版本