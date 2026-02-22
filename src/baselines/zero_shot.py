"""
Zero-Shot and Few-Shot Generation Baseline.

Loads the base Qwen3-8B model WITHOUT any LoRA adapter and uses it for:
  - Zero-shot instruction generation (n_shots=0)
  - Few-shot instruction generation (n_shots=1, 3, 5, ...)

The few-shot prompt prepends n example (input, output) pairs before the query,
using the same three-part structure as the fine-tuned experts.

GPU batching note: batch_generate() delegates to LanguageModel.generate_batch()
which performs left-padded batched inference on GPU, significantly reducing
CPU overhead and improving GPU utilization compared to sequential single-sample
calls.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_path_config, get_inference_config
from src.utils.logger import get_logger

logger = get_logger('baselines.zero_shot')

# Maximum new tokens for baseline generation.  The expected output is a
# three-part instruction of ~35 words (~47 tokens).  200 tokens is generous
# enough to cover edge cases while preventing the base model from generating
# run-on explanations that collapse ROUGE-L scores.
_BASELINE_MAX_NEW_TOKENS = 200


def _build_few_shot_prompt(
    input_text: str,
    input_type: str,
    n_shots: int,
    examples: List[Dict]
) -> str:
    """
    Build a few-shot prompt in Qwen3 chat template format.

    Uses the standard <|im_start|>/<|im_end|> structure so the base model
    receives input in the same format as the fine-tuned experts, ensuring
    consistent generation quality.

    For zero-shot (n_shots=0) only the query is included in the user turn.
    For few-shot, example (input, output) pairs are prepended in the user turn.

    Args:
        input_text: Actual query input
        input_type: One of 'text', 'image', 'uml'
        n_shots: Number of examples to prepend
        examples: List of dicts with keys 'input' and 'output'. Only the
                  first n_shots entries are used.

    Returns:
        Complete prompt string in Qwen3 chat format (without thinking block;
        LanguageModel._suppress_thinking handles that at generation time)
    """
    type_desc = {
        'text': 'software requirements to crowdsourcing instruction',
        'image': 'image description to crowdsourcing annotation instruction',
        'uml': 'UML use-case description to crowdsourcing instruction',
    }.get(input_type, 'requirements to crowdsourcing instruction')

    system_content = (
        f'You are an expert at converting {type_desc}.\n'
        'Each instruction must follow this exact three-part format:\n'
        '  Definition: <what the worker should do>\n'
        '  Emphasis & Caution: <what to pay attention to>\n'
        '  Things to Avoid: <common mistakes to avoid>\n'
        'Keep each section to ONE concise sentence.'
    )

    user_parts = []
    used_examples = examples[:n_shots] if examples else []
    for i, ex in enumerate(used_examples):
        user_parts.append(
            f'[Example {i + 1}]\nInput: {ex["input"].strip()}\nOutput:\n{ex["output"].strip()}'
        )
    user_parts.append(f'[Query]\nInput: {input_text.strip()}\nOutput:')
    user_content = '\n\n'.join(user_parts)

    return (
        f'<|im_start|>system\n{system_content}<|im_end|>\n'
        f'<|im_start|>user\n{user_content}<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )


class ZeroShotGenerator:
    """
    Zero-shot and few-shot generator using the base Qwen3-8B model (no LoRA).

    load_model() / unload_model() follow the same pattern as expert classes so
    experiment scripts can treat this class uniformly.

    batch_generate() delegates to LanguageModel.generate_batch() for true GPU
    batching, eliminating the CPU bottleneck caused by sequential single-sample
    inference.
    """

    def __init__(
        self,
        base_model_path: str = None,
        use_4bit: bool = True,
        max_new_tokens: Optional[int] = None
    ):
        """
        Args:
            base_model_path: Path to Qwen3-8B weights directory.
                             If None, resolved from get_path_config().
            use_4bit: Whether to load in 4-bit quantization.
            max_new_tokens: Override max_new_tokens for generation.  Defaults
                            to _BASELINE_MAX_NEW_TOKENS (200) so the base model
                            does not produce run-on explanations that degrade
                            ROUGE-L scores.  Pass a larger value explicitly if
                            needed (e.g. for few-shot with long examples).
        """
        path_cfg = get_path_config()
        if base_model_path is None:
            base_model_path = str(path_cfg.get_text_model_path())
        self.base_model_path = base_model_path
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else _BASELINE_MAX_NEW_TOKENS
        self._lm = None
        self.is_model_loaded = False

    def load_model(self) -> bool:
        """
        Load the base language model without any LoRA adapter.

        Returns:
            True on success, False otherwise.
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
        Release the model from GPU memory.

        Returns:
            True on success.
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
            input_text: Raw requirement / description string.
            input_type: 'text', 'image', or 'uml'.
            n_shots: Number of few-shot examples to prepend (0 = zero-shot).
            examples: Example dicts with keys 'input' and 'output'.
                      Required when n_shots > 0.

        Returns:
            Generated instruction string (empty string on failure).
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
                max_new_tokens=self.max_new_tokens,
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
        Generate instructions for a list of inputs using GPU-batched inference.

        Builds all prompts first, then delegates to
        LanguageModel.generate_batch() which performs left-padded batched
        tokenization, OOM-safe retry, and tqdm progress tracking entirely on
        the GPU.  This eliminates the CPU bottleneck of sequential single-sample
        calls and achieves full GPU utilization.

        Args:
            inputs: List of input strings.
            input_type: 'text', 'image', or 'uml'.
            n_shots: Number of few-shot examples to prepend.
            examples: Example dicts with keys 'input' and 'output'.
            batch_size: Number of prompts per GPU forward pass.  Defaults to 8
                        which is well-suited for RTX 4090 with 4-bit
                        quantization and short zero/few-shot prompts.

        Returns:
            List of generated instruction strings.
        """
        if not self.is_model_loaded:
            logger.warning('模型未加载，尝试加载...')
            if not self.load_model():
                return [''] * len(inputs)

        if n_shots > 0 and not examples:
            logger.warning('n_shots > 0 但未提供示例，退回到零样本模式')
            n_shots = 0

        prompts = [
            _build_few_shot_prompt(inp, input_type, n_shots, examples or [])
            for inp in inputs
        ]

        infer_cfg = get_inference_config()
        results = self._lm.generate_batch(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=infer_cfg.temperature,
            top_p=infer_cfg.top_p,
            top_k=infer_cfg.top_k,
            repetition_penalty=infer_cfg.repetition_penalty,
            batch_size=batch_size,
        )

        logger.info(f'批量生成完成: {len(results)}个样本')
        return results