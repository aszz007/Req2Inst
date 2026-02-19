"""
Zero-Shot and Few-Shot Generation Baseline.

Loads the base Qwen3-8B model WITHOUT any LoRA adapter and uses it for:
  - Zero-shot instruction generation (n_shots=0)
  - Few-shot instruction generation (n_shots=1, 3, 5, ...)

The few-shot prompt prepends n example (input, output) pairs before the query,
using the same three-part structure as the fine-tuned experts.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('baselines.zero_shot')


def _build_few_shot_prompt(
    input_text: str,
    input_type: str,
    n_shots: int,
    examples: List[Dict]
) -> str:
    """
    Build a few-shot prompt by prepending example (input, output) pairs.

    The format is:
      [Example 1]
      Input: <example input>
      Output: <example output>

      [Example N]
      ...

      [Query]
      Input: <actual query>
      Output:

    For zero-shot (n_shots=0) only the query section is included.

    Args:
        input_text: Actual query input
        input_type: One of 'text', 'image', 'uml'
        n_shots: Number of examples to prepend
        examples: List of dicts with keys 'input' and 'output', sampled from
                  training set. Only the first n_shots entries are used.

    Returns:
        Complete prompt string
    """
    # Build type-specific task description header
    type_desc = {
        'text': 'software requirements to crowdsourcing instruction',
        'image': 'image description to crowdsourcing annotation instruction',
        'uml': 'UML use-case description to crowdsourcing instruction',
    }.get(input_type, 'requirements to crowdsourcing instruction')

    header = (
        f'You are an expert at converting {type_desc}.\n'
        'Each instruction must follow this exact three-part format:\n'
        '  Definition: <what the worker should do>\n'
        '  Emphasis & Caution: <what to pay attention to>\n'
        '  Things to Avoid: <common mistakes to avoid>\n\n'
    )

    shots_text = ''
    used_examples = examples[:n_shots] if examples else []
    for i, ex in enumerate(used_examples):
        shots_text += f'[Example {i + 1}]\n'
        shots_text += f'Input: {ex["input"].strip()}\n'
        shots_text += f'Output:\n{ex["output"].strip()}\n\n'

    query_text = f'[Query]\nInput: {input_text.strip()}\nOutput:\n'

    return header + shots_text + query_text


class ZeroShotGenerator:
    """
    Zero-shot and few-shot generator using the base Qwen3-8B model (no LoRA).

    load_model() / unload_model() follow the same pattern as expert classes so
    experiment scripts can treat this class uniformly.
    """

    def __init__(self, base_model_path: str = None, use_4bit: bool = True):
        """
        Args:
            base_model_path: Path to Qwen3-8B weights directory.
                             If None, resolved from get_path_config().
            use_4bit: Whether to load in 4-bit quantization
        """
        path_cfg = get_path_config()
        if base_model_path is None:
            base_model_path = str(path_cfg.get_text_model_path())
        self.base_model_path = base_model_path
        self.use_4bit = use_4bit
        self._lm = None
        self.is_model_loaded = False

    def load_model(self) -> bool:
        """
        加载不带LoRA适配器的基础语言模型。

        Returns:
            加载成功返回True，否则返回False
        """
        try:
            from models.language_model import LanguageModel

            logger.info(f'正在加载基础模型: {self.base_model_path}')
            self._lm = LanguageModel(
                model_path=self.base_model_path,
                use_4bit=self.use_4bit
            )
            self.is_model_loaded = True
            logger.info('基础模型加载成功')
            return True
        except Exception as e:
            logger.error(f'基础模型加载失败: {e}')
            import traceback
            logger.error(traceback.format_exc())
            return False

    def unload_model(self) -> bool:
        """
        从GPU显存中释放模型。

        Returns:
            卸载成功返回True
        """
        try:
            if self._lm is not None:
                del self._lm
                self._lm = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.is_model_loaded = False
            logger.info('基础模型已卸载')
            return True
        except Exception as e:
            logger.error(f'模型卸载失败: {e}')
            return False

    def generate(
        self,
        input_text: str,
        input_type: str = 'text',
        n_shots: int = 0,
        examples: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate an instruction for a single input.

        Args:
            input_text: Raw requirement / description string
            input_type: 'text', 'image', or 'uml'
            n_shots: Number of few-shot examples to prepend (0 = zero-shot)
            examples: Example dicts with keys 'input' and 'output'.
                      Required when n_shots > 0.

        Returns:
            Generated instruction string (empty string on failure)
        """
        if not self.is_model_loaded:
            logger.warning('模型未加载，尝试加载...')
            if not self.load_model():
                return ''

        if n_shots > 0 and not examples:
            logger.warning('n_shots > 0 但未提供示例，退回到零样本模式')
            n_shots = 0

        prompt = _build_few_shot_prompt(input_text, input_type, n_shots, examples or [])

        infer_cfg = get_inference_config()
        try:
            result = self._lm.generate(
                prompt,
                max_new_tokens=infer_cfg.max_new_tokens,
                temperature=infer_cfg.temperature,
                top_p=infer_cfg.top_p,
                top_k=infer_cfg.top_k,
                repetition_penalty=infer_cfg.repetition_penalty,
            )
            return result
        except Exception as e:
            logger.error(f'生成失败: {e}')
            return ''

    def batch_generate(
        self,
        inputs: List[str],
        input_type: str = 'text',
        n_shots: int = 0,
        examples: Optional[List[Dict]] = None,
        batch_size: int = 8
    ) -> List[str]:
        """
        Generate instructions for a list of inputs.

        NOTE: The underlying LanguageModel.generate() is called sequentially
        here because few-shot prompts can be very long. Set batch_size=1 if
        you encounter OOM errors.

        Args:
            inputs: List of input strings
            input_type: 'text', 'image', or 'uml'
            n_shots: Number of few-shot examples to prepend
            examples: Example dicts with keys 'input' and 'output'
            batch_size: Kept for API compatibility; sequential generation is
                        always used to stay within GPU memory limits

        Returns:
            List of generated instruction strings
        """
        if not self.is_model_loaded:
            logger.warning('模型未加载，尝试加载...')
            if not self.load_model():
                return [''] * len(inputs)

        results = []
        for i, inp in enumerate(inputs):
            out = self.generate(inp, input_type=input_type, n_shots=n_shots, examples=examples)
            results.append(out)
            if (i + 1) % 20 == 0:
                logger.info(f'零样本已生成 {i + 1}/{len(inputs)}')

        logger.info(f'批量生成完成: {len(results)}个样本')
        return results