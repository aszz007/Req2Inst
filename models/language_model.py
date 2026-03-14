"""
语言模型接口（重构版）
功能：
  - 支持LoRA权重动态加载/卸载
  - 4bit量化优化
  - 统一的generate接口
  - 基于Qwen3-8B（禁用思考模式）
环境：instruction_generator（单一Conda环境，transformers==4.57.0）
支持模型：Qwen3-8B（默认）
"""

import time
import threading
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from tqdm import tqdm

from peft import PeftModel
from pathlib import Path
from typing import Optional
import warnings
from concurrent.futures import ThreadPoolExecutor
import os
import json
import tempfile
import shutil
warnings.filterwarnings('ignore')

from config.settings import get_path_config, get_device_config, get_model_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SanitizeLogitsProcessor(LogitsProcessor):
    """
    拦截 NaN / +inf 的 logits，替换为 -inf，防止采样时触发 CUDA device-side assert。
    常见于 p_tuning / prompt_tuning 的 soft prompt embeddings 与量化基座模型交互时产生
    数值不稳定的情况。
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Replace NaN and +inf with -inf so these tokens are never sampled
        bad_mask = torch.isnan(scores) | (scores == float('inf'))
        if bad_mask.any():
            scores = scores.masked_fill(bad_mask, float('-inf'))
        # If all logits in a row are -inf, multinomial sampling will fail.
        # Fall back to uniform distribution over the full vocabulary for that row.
        all_invalid = (scores == float('-inf')).all(dim=-1, keepdim=True)
        if all_invalid.any():
            scores = scores.masked_fill(all_invalid.expand_as(scores), 0.0)
        return scores


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
        model_cfg = get_model_config()

        # 如果未指定路径，使用配置中的默认模型
        if model_path is None:
            self.model_path = str(path_cfg.get_text_model_path())
            self.model_version = model_cfg.version
        else:
            self.model_path = model_path
            # 统一视为Qwen3-8B（当前唯一支持的文本模型）
            self.model_version = 'qwen3_8b'
            if 'Qwen3-8B' not in model_path and 'qwen3-8B' not in model_path:
                logger.warning(f"路径不含Qwen3-8B标识，仍以qwen3_8b版本处理: {model_path}")

        self.device = device_cfg.get_device()
        self.device_cfg = device_cfg

        # GPU性能检测和量化策略
        # 调用方显式传入 use_4bit=True 时无条件启用4bit，避免 FP16 下 soft prompt 产生 NaN/inf
        if self.device != "cuda":
            self.use_4bit = False
        elif use_4bit:
            self.use_4bit = True
        else:
            self.use_4bit = device_cfg.should_use_quantization()
        self.gpu_tier = device_cfg.get_gpu_tier()
        self.is_high_end_gpu = device_cfg.is_high_end_gpu

        self.model = None
        self.tokenizer = None
        self.current_lora_path = None  # 当前加载的LoRA路径
        self.is_lora_loaded = False    # LoRA加载状态

        logger.info(f"初始化语言模型")
        logger.info(f"模型版本: {self.model_version}")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"设备: {self.device}")
        logger.info(f"GPU信息: {device_cfg.get_gpu_info()}")
        logger.info(f"量化策略: {'4bit量化' if self.use_4bit else 'FP16（无量化）'}")
        logger.info(f"GPU配置: {self.gpu_tier.upper()}端GPU模式")

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
                use_fast=True,
            )

            # 设置特殊tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = '<|endoftext|>'
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = '<|im_end|>'

            # 使用固定GPU设备避免accelerate将层offload到CPU
            device_map = {"": 0} if self.device == "cuda" else "auto"

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.use_4bit else None,
                low_cpu_mem_usage=True
            )

            self.model.eval()
            logger.info("基础模型加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def get_target_modules(self) -> list:
        """
        返回Qwen3-8B的LoRA target_modules

        Returns:
            list: target_modules列表
        """
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def _clean_lora_config(self, lora_path: Path) -> Optional[Path]:
        """
        清理 LoRA 配置文件，移除不兼容的参数

        Args:
            lora_path: 原始 LoRA 路径

        Returns:
            Optional[Path]: 清理后的临时路径，如果失败返回 None
        """
        try:
            config_file = lora_path / "adapter_config.json"
            if not config_file.exists():
                logger.warning(f"配置文件不存在: {config_file}")
                return None

            # 读取原始配置
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 不兼容参数列表
            incompatible_params = [
                'alora_invocation_tokens',
                'alora_prefix',
                'alora_suffix',
                'arrow_config',
            ]

            # 检查是否需要清理
            needs_cleaning = any(param in config for param in incompatible_params)

            if not needs_cleaning:
                # 配置文件无需清理，直接返回原路径
                return lora_path

            # 创建临时目录
            temp_dir = Path(tempfile.mkdtemp(prefix="lora_cleaned_"))
            logger.info(f"创建临时目录: {temp_dir}")

            # 复制所有文件到临时目录
            for item in lora_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, temp_dir / item.name)

            # 清理配置
            for param in incompatible_params:
                if param in config:
                    logger.info(f"移除不兼容参数: {param}")
                    del config[param]

            # 写入清理后的配置
            cleaned_config_file = temp_dir / "adapter_config.json"
            with open(cleaned_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info("LoRA 配置已清理")
            return temp_dir

        except Exception as e:
            logger.error(f"清理 LoRA 配置失败: {e}")
            return None

    def load_lora_from_path(self, lora_path: str) -> bool:
        """
        从路径动态加载LoRA权重

        Args:
            lora_path: LoRA权重路径（绝对路径或相对路径）

        Returns:
            bool: 是否加载成功
        """
        temp_dir = None
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

            # 清理配置文件（如果需要）
            cleaned_path = self._clean_lora_config(lora_path)
            if cleaned_path is None:
                logger.warning("配置清理失败，尝试直接加载")
                cleaned_path = lora_path
            elif cleaned_path != lora_path:
                temp_dir = cleaned_path
                logger.info(f"使用清理后的配置: {cleaned_path}")

            # 使用PEFT加载LoRA
            self.model = PeftModel.from_pretrained(
                self.model,
                str(cleaned_path),
                is_trainable=False
            )

            self.current_lora_path = str(lora_path)
            self.is_lora_loaded = True

            logger.info("LoRA加载成功")
            return True

        except Exception as e:
            logger.error(f"LoRA加载失败: {e}")
            return False

        finally:
            # 清理临时目录
            if temp_dir and temp_dir != lora_path:
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"清理临时目录: {temp_dir}")
                except Exception as e:
                    logger.warning(f"清理临时目录失败: {e}")

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

            # 兼容 LoRA / P-tuning / Prompt Tuning 等不同PEFT类型
            if hasattr(self.model, 'unload'):
                self.model = self.model.unload()
            elif hasattr(self.model, 'get_base_model'):
                self.model = self.model.get_base_model()
            else:
                logger.warning("模型不支持unload()，仅重置状态标志")

            self.current_lora_path = None
            self.is_lora_loaded = False

            logger.info("LoRA卸载成功")
            return True

        except Exception as e:
            logger.error(f"LoRA卸载失败: {e}")
            # 无论是否异常，强制重置状态，避免影响下一次加载
            self.current_lora_path = None
            self.is_lora_loaded = False
            return False

    def _get_stop_tokens(self) -> list:
        """
        获取停止tokens列表（复用方法，避免generate/generate_batch中重复构建）

        Returns:
            list: 去重后的停止token ID列表
        """
        stop_tokens = []
        if self.tokenizer.eos_token_id is not None:
            stop_tokens.append(self.tokenizer.eos_token_id)

        im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
            stop_tokens.append(im_end_id)

        endoftext_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        if endoftext_id is not None and endoftext_id != self.tokenizer.unk_token_id:
            stop_tokens.append(endoftext_id)

        return list(set(stop_tokens))

    def _build_generation_config(self, max_new_tokens: int, temperature: float,
                                  top_p: float, top_k: int,
                                  repetition_penalty: float) -> dict:
        """
        构建生成配置字典（复用方法，避免generate/generate_batch中重复构建）

        Args:
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚

        Returns:
            dict: 生成配置
        """
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": True if temperature > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self._get_stop_tokens(),
            "use_cache": True,
            "logits_processor": LogitsProcessorList([SanitizeLogitsProcessor()]),
        }

    def _model_generate(self, inputs: dict, generation_config: dict) -> torch.Tensor:
        """
        执行模型生成（统一处理4bit/FP16的精度分支）

        Args:
            inputs: 编码后的输入
            generation_config: 生成配置

        Returns:
            torch.Tensor: 生成的输出
        """
        with torch.no_grad():
            if self.use_4bit:
                return self.model.generate(**inputs, **generation_config)
            else:
                with torch.cuda.amp.autocast():
                    return self.model.generate(**inputs, **generation_config)

    def _suppress_thinking(self, prompt: str) -> str:
        """
        For Qwen3-8B, append an empty think block to the prompt to disable
        chain-of-thought generation. This mirrors what
        tokenizer.apply_chat_template(..., enable_thinking=False) produces:
        the pre-filled empty <think></think> block tells the model to skip
        reasoning and output the answer directly.
        Has no effect on other model versions or if already suppressed.
        """
        if self.model_version != 'qwen3_8b':
            return prompt
        if '<think>' in prompt:
            return prompt
        return prompt + '<think>\n\n</think>\n'

    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                 repetition_penalty: float = 1.1) -> str:
        """
        生成文本（支持GPU性能优化）

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
            # For Qwen3-8B: pre-fill empty think block to disable thinking mode
            _gen_start = time.perf_counter()
            logger.info(
                f"[TIMING][generate] called | torch_threads={torch.get_num_threads()} | interop={torch.get_num_interop_threads()} | thread={threading.get_ident()}")
            prompt = self._suppress_thinking(prompt)

            # 编码输入
            _tok_start = time.perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_length = inputs['input_ids'].shape[1]
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            _tok_end = time.perf_counter()
            logger.info(f"[TIMING][generate] tokenize 1 sample: {_tok_end - _tok_start:.3f}s | input_len={input_length}")

            # 构建生成配置并执行生成
            generation_config = self._build_generation_config(
                max_new_tokens, temperature, top_p, top_k, repetition_penalty
            )
            outputs = self._model_generate(inputs, generation_config)

            # 解码输出
            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 清理特殊标记和多余内容
            _gpu_done = time.perf_counter()
            logger.info(f"[TIMING][generate] GPU done: {_gpu_done - _gen_start:.3f}s | new_tokens={len(generated_ids)}")
            generated_text = self._post_process_text(generated_text)

            _post_done = time.perf_counter()
            logger.info(
                f"[TIMING][generate] post-process: {_post_done - _gpu_done:.3f}s | total: {_post_done - _gen_start:.3f}s")
            return generated_text

        except Exception as e:
            logger.error(f"生成失败: {e}")
            return ""

    def generate_batch(self, prompts: list, max_new_tokens: int = 2048,
                      temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                      repetition_penalty: float = 1.1, batch_size: int = None) -> list:
        """
        批量生成文本（GPU性能优化）

        Args:
            prompts: prompt列表
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚
            batch_size: 批处理大小（None则根据GPU性能自动选择）

        Returns:
            list: 生成的文本列表
        """
        if not prompts:
            return []

        # 自动选择batch_size
        if batch_size is None:
            if self.gpu_tier == 'high':
                batch_size = 16
            elif self.gpu_tier == 'mid':
                batch_size = 2
            else:
                batch_size = 1

        logger.info(f"批量推理: {len(prompts)}条样本, batch_size={batch_size}")

        results = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        # 预处理所有batch列表（suppress_thinking在CPU侧预先完成）
        all_batches = []
        for i in range(0, len(prompts), batch_size):
            bp = [self._suppress_thinking(p) for p in prompts[i:i + batch_size]]
            all_batches.append(bp)

        # 构建生成配置（所有batch共用，避免重复构建）
        generation_config = self._build_generation_config(
            max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

        # tokenization函数，用于异步预取
        def _tokenize(batch):
            return self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )

        # 使用tqdm显示进度条
        pbar = tqdm(total=len(prompts), desc="批量生成", unit="样本", ncols=100)

        # 启动第一个batch的tokenization
        tok_executor = ThreadPoolExecutor(max_workers=2)
        next_tok_future = tok_executor.submit(_tokenize, all_batches[0]) if all_batches else None

        for batch_idx, batch_prompts in enumerate(all_batches):
            try:
                # 获取当前batch的tokenization结果（已在GPU推理上一batch时完成）
                _tok_fetch_start = time.perf_counter()
                inputs_raw = next_tok_future.result()

                # 立即提交下一个batch的tokenization（与GPU推理并行）
                if batch_idx + 1 < len(all_batches):
                    next_tok_future = tok_executor.submit(_tokenize, all_batches[batch_idx + 1])

                input_lengths = inputs_raw['input_ids'].shape[1]
                inputs = {k: v.to(self.model.device) for k, v in inputs_raw.items()}
                _tok_end = time.perf_counter()
                logger.info(
                    f"[TIMING][batch] tok_fetch+H2D {len(batch_prompts)} samples: {_tok_end - _tok_fetch_start:.3f}s | seq_len={input_lengths}")

                # 执行生成、解码、后处理
                _infer_start = time.perf_counter()
                batch_results = self._generate_and_decode_batch(
                    inputs, input_lengths, generation_config
                )
                _infer_end = time.perf_counter()
                logger.info(
                    f"[TIMING][batch] model.generate {len(batch_prompts)} samples: {_infer_end - _infer_start:.3f}s")

                results.extend(batch_results)
                pbar.update(len(batch_prompts))

            except Exception as e:
                error_str = str(e)
                i = batch_idx * batch_size
                if 'out of memory' in error_str.lower() and len(batch_prompts) > 1:
                    # CUDA OOM：对该batch逐步降低batch_size重试
                    logger.warning(
                        f"批量生成OOM (batch {i//batch_size + 1}/{num_batches})，"
                        f"当前batch_size={len(batch_prompts)}，尝试降级重试..."
                    )
                    torch.cuda.empty_cache()
                    retry_results = self._retry_batch_with_smaller_size(
                        batch_prompts, generation_config
                    )
                    results.extend(retry_results)
                    if all(r != "" for r in retry_results):
                        logger.info(
                            f"batch {i//batch_size + 1}/{num_batches} 降级重试成功"
                        )
                else:
                    logger.error(f"批量生成失败 (batch {i//batch_size + 1}/{num_batches}): {e}")
                    results.extend([""] * len(batch_prompts))
                pbar.update(len(batch_prompts))

        pbar.close()
        tok_executor.shutdown(wait=False)
        return results

    def _generate_and_decode_batch(self, inputs: dict, input_lengths: int,
                                    generation_config: dict) -> list:
        """
        对单个batch执行生成、解码和后处理（复用方法，避免主流程和OOM重试中重复代码）

        Args:
            inputs: 已移至设备的编码输入
            input_lengths: 输入序列长度
            generation_config: 生成配置

        Returns:
            list: 后处理后的文本列表
        """
        outputs = self._model_generate(inputs, generation_config)

        # 批量解码（Rust层并行，替代串行decode循环）
        generated_ids_list = [output[input_lengths:] for output in outputs]
        decoded_texts = self.tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True)

        # 并行后处理
        with ThreadPoolExecutor(max_workers=min(len(decoded_texts), os.cpu_count() or 16)) as _exec:
            return list(_exec.map(self._post_process_text, decoded_texts))

    def _retry_batch_with_smaller_size(self, batch_prompts: list,
                                        generation_config: dict) -> list:
        """
        OOM降级重试：将batch拆分为更小的子batch重新生成

        Args:
            batch_prompts: 需要重试的prompt列表
            generation_config: 生成配置

        Returns:
            list: 生成结果列表（失败的位置为空字符串）
        """
        retry_size = max(1, len(batch_prompts) // 2)
        retry_results = []

        for r in range(0, len(batch_prompts), retry_size):
            retry_batch = batch_prompts[r:r + retry_size]
            try:
                retry_inputs = self.tokenizer(
                    retry_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=8192
                )
                retry_input_lengths = retry_inputs['input_ids'].shape[1]
                retry_inputs = {k: v.to(self.model.device) for k, v in retry_inputs.items()}

                chunk_results = self._generate_and_decode_batch(
                    retry_inputs, retry_input_lengths, generation_config
                )
                retry_results.extend(chunk_results)
                torch.cuda.empty_cache()

            except Exception as retry_e:
                logger.error(f"降级重试失败 (retry_size={retry_size}): {retry_e}")
                retry_results.extend([""] * len(retry_batch))

        return retry_results

    def _post_process_text(self, text: str) -> str:
        """
        后处理生成文本：清理特殊标记 + 截断三段式后多余内容

        Args:
            text: 原始生成文本

        Returns:
            str: 后处理后的文本
        """
        text = self._clean_generated_text(text)
        return self._truncate_after_three_parts(text)

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

        # 检测中文字符并截断（使用Unicode范围）
        import re

        # 查找第一个中文字符的位置
        chinese_match = re.search(r'[\u4e00-\u9fff]', text)
        if chinese_match:
            idx = chinese_match.start()
            # 回溯到最近的句点或换行符，确保不破坏完整句子
            truncate_pos = idx
            for i in range(idx - 1, max(0, idx - 50), -1):
                if text[i] in '.!?\n':
                    truncate_pos = i + 1
                    break
            text = text[:truncate_pos].strip()

        # 如果文本中包含多个训练样本（通常以特定模式开头），只保留第一个完整指令
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
            '方法 ',
            '结果 ',
            '结论 ',
        ]

        for pattern in unwanted_patterns:
            if pattern in text:
                # 找到模式出现的位置，截断到该位置之前
                idx = text.find(pattern)
                if idx > 0:
                    text = text[:idx].strip()
                    break

        return text

    def _truncate_after_three_parts(self, text: str) -> str:
        """
        检测三段式格式并截断后续多余内容，同时限制 Things to Avoid 段落长度。

        策略：
        1. 支持多行格式（每段占独立行）和单行格式（三段拼在同一行）
        2. 提取 Things to Avoid 内容，最多保留 MAX_DO_NOT 条 "Do not" 语句
        3. 找不到完整三段式时原样返回

        Args:
            text: 原始生成文本

        Returns:
            str: 截断后的文本
        """
        import re

        MAX_DO_NOT = 6  # Things to Avoid 最多保留的 "Do not" 条数

        SECTION_PAT = (
            r'(?P<def>Definition:.*?)'
            r'(?P<emph>Emphasis\s*(?:&|and)\s*Caution:.*?)'
            r'(?P<avoid>Things to Avoid:\s*(?P<avoid_body>.*))'
            r'$'
        )

        def _limit_do_not(content: str) -> str:
            """将 Things to Avoid 内容截断到最多 MAX_DO_NOT 条 Do not 语句"""
            sentences = re.split(r'(?<=[.!?])\s+', content.strip())
            do_not_count = 0
            kept = []
            for s in sentences:
                if re.match(r'Do not\b', s.strip(), re.IGNORECASE):
                    do_not_count += 1
                    if do_not_count > MAX_DO_NOT:
                        break
                kept.append(s)
            return ' '.join(kept).strip()

        lines = text.split('\n')

        # ── 路径 A：多行格式 ─────────────────────────────────────────
        # 每个段头各占独立行时，按行号定位
        definition_idx = emphasis_idx = avoid_idx = None
        for i, line in enumerate(lines):
            ls = line.strip()
            if ls.startswith('Definition:') and definition_idx is None:
                definition_idx = i
            elif (ls.startswith('Emphasis & Caution:') or ls.startswith('Emphasis and Caution:')) \
                    and emphasis_idx is None:
                emphasis_idx = i
            elif ls.startswith('Things to Avoid:') and avoid_idx is None:
                avoid_idx = i

        if definition_idx is not None and emphasis_idx is not None and avoid_idx is not None:
            # 多行格式：收集 Things to Avoid 段内容（含跨行内容）
            avoid_header_line = lines[avoid_idx]
            inline = avoid_header_line[avoid_header_line.index('Things to Avoid:') + len('Things to Avoid:'):].strip()

            extra = []
            for line in lines[avoid_idx + 1:]:
                s = line.strip()
                if s == '' or re.match(
                    r'^(Definition:|Emphasis\s*(?:&|and)\s*Caution:|Things to Avoid:)',
                    s, re.IGNORECASE
                ):
                    break
                extra.append(s)

            full_avoid = (inline + ' ' + ' '.join(extra)).strip()
            limited_avoid = _limit_do_not(full_avoid)

            result_lines = lines[definition_idx:avoid_idx]
            result_lines.append(
                f"Things to Avoid: {limited_avoid}" if limited_avoid else avoid_header_line.rstrip()
            )
            return '\n'.join(result_lines)

        # ── 路径 B：单行格式 ─────────────────────────────────────────
        # 三个段头都嵌在同一行（或文本中不含换行），用正则提取
        flat = ' '.join(lines)  # 展平为单行便于匹配
        m = re.search(
            r'(Definition:.*?)'
            r'(Emphasis\s*(?:&|and)\s*Caution:.*?)'
            r'(Things to Avoid:\s*)(.*)',
            flat,
            re.DOTALL | re.IGNORECASE
        )
        if m:
            def_part    = m.group(1).strip()
            emph_part   = m.group(2).strip()
            avoid_body  = m.group(4).strip()
            limited_avoid = _limit_do_not(avoid_body)
            parts = [def_part, emph_part, f"Things to Avoid: {limited_avoid}"]
            return '\n'.join(parts)

        # ── 路径 C：未找到完整三段式，原样返回 ─────────────────────
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
    - 默认参数从 InferenceConfig 统一读取
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
        # 从配置中读取推理参数默认值
        self._inference_cfg = get_inference_config()
        logger.info("指令生成器初始化完成")

    def load_expert(self, expert_name_or_path: str) -> bool:
        """
        加载专家LoRA权重

        Args:
            expert_name_or_path: 专家名称（如'text_expert'）或完整路径

        Returns:
            bool: 是否加载成功
        """
        path_cfg = get_path_config()

        # First try checkpoints/lora_moe/{expert_name}/ (framework standard path)
        lora_path = path_cfg.PROJECT_ROOT / 'checkpoints' / 'lora_moe' / expert_name_or_path
        if not lora_path.exists():
            # Fallback: try via get_expert_weight_path (legacy compatibility)
            lora_path = path_cfg.get_expert_weight_path(expert_name_or_path)

        # 如果路径不存在，尝试作为完整路径处理
        if not lora_path.exists():
            lora_path = Path(expert_name_or_path)

        return self.language_model.load_lora_from_path(str(lora_path))

    def unload_expert(self) -> bool:
        """卸载当前专家"""
        return self.language_model.unload_lora()

    def generate(self, prompt: str, max_new_tokens: int = None,
                 temperature: float = None, top_p: float = None,
                 top_k: int = None, repetition_penalty: float = None) -> str:
        """
        生成众包指令（接受外部构建的prompt）

        参数默认值从 config.settings.InferenceConfig 读取，
        调用方可按需覆盖单个参数而不影响其他参数的灵活性。

        Args:
            prompt: 已构建好的完整prompt（由外部Prompt模板生成）
            max_new_tokens: 最大生成token数（默认从InferenceConfig读取）
            temperature: 温度参数（默认从InferenceConfig读取）
            top_p: nucleus sampling（默认从InferenceConfig读取）
            top_k: top-k sampling（默认从InferenceConfig读取）
            repetition_penalty: 重复惩罚（默认从InferenceConfig读取）

        Returns:
            str: 生成的指令
        """
        cfg = self._inference_cfg
        return self.language_model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens,
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            top_k=top_k if top_k is not None else cfg.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
        )

    def get_expert_status(self) -> dict:
        """获取当前专家状态"""
        return self.language_model.get_lora_status()

    def generate_batch(self, prompts: list, max_new_tokens: int = None,
                      temperature: float = None, top_p: float = None,
                      top_k: int = None, repetition_penalty: float = None,
                      batch_size: int = None) -> list:
        """
        批量生成众包指令

        参数默认值从 config.settings.InferenceConfig 读取。

        Args:
            prompts: prompt列表
            max_new_tokens: 最大生成token数（默认从InferenceConfig读取）
            temperature: 温度参数（默认从InferenceConfig读取）
            top_p: nucleus sampling（默认从InferenceConfig读取）
            top_k: top-k sampling（默认从InferenceConfig读取）
            repetition_penalty: 重复惩罚（默认从InferenceConfig读取）
            batch_size: 批处理大小（None则自动选择）

        Returns:
            list: 生成的指令列表
        """
        cfg = self._inference_cfg
        return self.language_model.generate_batch(
            prompts=prompts,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens,
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            top_k=top_k if top_k is not None else cfg.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
            batch_size=batch_size
        )