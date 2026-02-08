"""
Qwen视觉模型封装（多版本支持 + GPU性能优化）
功能：
  - 支持 Qwen2.5-VL-7B 和 Qwen3-VL-8B 两个版本
  - 图像识别和UML图识别（预处理阶段）
  - 支持LoRA动态加载（推理阶段）
  - 通用generate接口
  - 动态精度选择（高端GPU使用FP16，其他GPU使用4bit量化）
  - 置信度计算，优化的生成参数
版本: 4.0（GPU性能优化版）
更新: 2025-02 - 增加动态精度选择，优化GPU利用率
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import json
import re
import gc
from typing import Dict, Optional, Tuple
from pathlib import Path

from config.settings import get_path_config, get_device_config, get_vision_model_config
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VisionModel:
    """Qwen2.5-VL模型封装类 - 支持LoRA和通用生成"""

    def __init__(self, model_path: Optional[str] = None, version: str = None):
        """
        初始化模型

        Args:
        model_path: 模型路径（None则使用配置，优先级高于version）
        version: 模型版本（'qwen2.5' 或 'qwen3'，None则使用配置）
        """
        # 获取配置
        path_cfg = get_path_config()
        device_cfg = get_device_config()

        # 确定使用的模型版本
        if model_path is None:
            if version is None:
                vision_cfg = get_vision_model_config()
                self.version = vision_cfg.version
            else:
                self.version = version

            self.model_path = str(path_cfg.get_vision_model_path(self.version))
            self.model_name = get_vision_model_config(self.version).get_model_name()
        else:
            self.model_path = model_path
            # 从路径推断版本
            if "Qwen3" in model_path or "qwen3" in model_path:
                self.version = "qwen3"
                self.model_name = "Qwen3-VL-8B-Instruct"
            else:
                self.version = "qwen2.5"
                self.model_name = "Qwen2.5-VL-7B-Instruct"

        self.device = device_cfg.get_device()
        self.device_cfg = device_cfg
        self.model = None
        self.processor = None
        self.current_lora_path = None  # 当前加载的LoRA路径
        self.is_lora_loaded = False    # LoRA加载状态

        # 根据GPU型号决定量化策略
        self.use_quantization = device_cfg.should_use_quantization()

        logger.info(f"初始化视觉模型: {self.model_name}")
        logger.info(f"模型版本: {self.version}")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"设备: {self.device}")
        logger.info(f"GPU信息: {device_cfg.get_gpu_info()}")
        logger.info(f"量化策略: {'4bit量化' if self.use_quantization else 'FP16（无量化）'}")

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

        self._load_base_model()

    def get_model_info(self) -> dict:
        """
        获取模型信息

        Returns:
            dict: 模型详细信息
        """
        return {
            'version': self.version,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'lora_loaded': self.is_lora_loaded,
            'lora_path': self.current_lora_path,
            'device': self.device
        }

    def _load_base_model(self):
        """加载基础模型（支持动态精度选择）"""
        try:
            # 加载processor
            logger.info("加载processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 根据GPU型号选择量化配置
            if self.use_quantization:
                # 低端GPU：使用4bit量化节省显存
                logger.info("使用4bit量化配置（节省显存）...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                logger.info("加载模型（4bit量化）...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                # 高端GPU（4090等）：使用FP16，无量化，充分利用显存和计算能力
                logger.info("使用FP16配置（高端GPU优化）...")
                logger.info("加载模型（FP16，无量化）...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            logger.info("模型加载成功")

            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"已分配显存: {memory_allocated:.2f}GB")
                logger.info(f"已预留显存: {memory_reserved:.2f}GB")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def load_lora_from_path(self, lora_path: str) -> bool:
        """
        从路径动态加载LoRA权重

        Args:
            lora_path: LoRA权重路径

        Returns:
            bool: 是否加载成功
        """
        try:
            lora_path = Path(lora_path)

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

    def generate(self, prompt: str, image_path: Optional[str] = None,
                 max_new_tokens: int = 1024, temperature: float = 0.3,
                 top_p: float = 0.8, do_sample: bool = True) -> str:
        """
        通用生成接口（支持自定义prompt）

        Args:
            prompt: 文本提示词
            image_path: 图像路径（可选）
            max_new_tokens: 最大生成token数
            temperature: 温度参数（降低以提高稳定性）
            top_p: nucleus sampling（降低以提高稳定性）
            do_sample: 是否采样

        Returns:
            str: 生成的文本
        """
        try:
            # 使用 Transformers 原生接口构建输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 如果有图像，添加到content开头
            if image_path:
                messages[0]["content"].insert(0, {
                    "type": "image",
                    "image": image_path
                })

            # 使用 processor 的 apply_chat_template（无需 process_vision_info）
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # 生成（优化参数）
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p if do_sample else 1.0,
                        do_sample=do_sample,
                        use_cache=True,
                        num_beams=1,  # 明确设置beam search为1（贪心解码）
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )

            # 解码
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # 清理
            del inputs, generated_ids, generated_ids_trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return response

        except Exception as e:
            logger.error(f"生成失败: {e}")
            return ""

    def recognize_image(self, image_path: str, prompt: Optional[str] = None) -> Dict:
        """
        识别图像内容（用于一般图像预处理）

        Args:
            image_path: 图像路径
            prompt: 自定义提示词（默认使用统一模板）

        Returns:
            dict: 包含description、details、confidence等字段
        """
        if prompt is None:
            prompt = ImageInstructionTemplate.get_recognition_prompt()

        logger.info(f"识别图像: {Path(image_path).name}")

        try:
            # 使用 Transformers 原生接口
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # 使用 processor 的 apply_chat_template
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # 生成并计算置信度
            response, confidence = self._generate_with_confidence(inputs)

            # 清理显存
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # 解析响应
            result = self._parse_image_response(response, image_path)
            result["confidence"] = confidence
            result["recognition_status"] = "success"

            logger.info(f"识别成功, 置信度: {confidence:.3f}")
            return result

        except Exception as e:
            logger.error(f"识别失败: {e}")
            return {
                "description": "",
                "details": {
                    "objects": [],
                    "scene": "unknown",
                    "spatial_info": ""
                },
                "confidence": 0.0,
                "recognition_status": "failed",
                "error": str(e)
            }

    def recognize_uml(self, uml_path: str, max_retries: int = 2, prompt: Optional[str] = None) -> Dict:
        """
        识别UML图（专用于预处理）

        Args:
            uml_path: UML图路径
            max_retries: 最大重试次数
            prompt: 自定义提示词（默认使用统一模板）

        Returns:
            dict: UML识别结果
        """
        if prompt is None:
            prompt = UMLInstructionTemplate.get_recognition_prompt()

        logger.info(f"识别UML图: {Path(uml_path).name}")

        for attempt in range(max_retries):
            try:
                # 使用 Transformers 原生接口
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": uml_path},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # 使用 processor 的 apply_chat_template
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )

                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")

                # UML需要更多tokens（优化生成参数）
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=1500,
                            temperature=0.1,  # 更低的温度提高稳定性
                            do_sample=True,   # 启用采样但温度很低
                            top_p=0.85,       # 稍微收紧采样范围
                            use_cache=True,
                            num_beams=1,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                response = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                # 清理
                del inputs, generated_ids, generated_ids_trimmed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # 解析
                result = self._parse_uml_response(response, uml_path)

                if result['success'] or attempt == max_retries - 1:
                    if result['success']:
                        logger.info(f"UML识别成功")
                    return result
                else:
                    logger.warning(f"第{attempt + 1}次尝试失败，重试中...")
                    continue

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"UML识别失败: {e}")
                    return {
                        'description': f"Recognition failed: {str(e)}",
                        'success': False,
                        'error': str(e)
                    }
                else:
                    logger.warning(f"第{attempt + 1}次尝试出错: {e}，重试中...")
                    continue

    def _generate_with_confidence(self, inputs) -> Tuple[str, float]:
        """生成文本并计算置信度（基于熵，优化参数）"""
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,  # 降低温度提高稳定性
                    do_sample=True,   # 启用采样但温度很低
                    top_p=0.85,       # 收紧采样范围
                    use_cache=True,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

        # 计算置信度
        scores = outputs.scores
        entropies = []

        for score in scores:
            probs = F.softmax(score[0], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            normalized_entropy = min(entropy / 10.0, 1.0)
            entropies.append(normalized_entropy)

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.5
        confidence = 1.0 - avg_entropy

        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response, float(confidence)

    def _parse_image_response(self, response: str, image_path: str) -> Dict:
        """解析图像识别响应"""
        try:
            # 提取JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            result = json.loads(json_str)

            # 确保必需字段
            if 'description' not in result:
                result['description'] = response[:200]

            if 'details' not in result:
                result['details'] = {
                    "objects": [],
                    "scene": "unknown scene",
                    "spatial_info": ""
                }

            return result

        except json.JSONDecodeError:
            logger.warning("JSON解析失败，使用备用方案")
            return {
                "description": response[:200] if response else "",
                "details": {
                    "objects": [],
                    "scene": "unknown",
                    "spatial_info": ""
                }
            }

    def _parse_uml_response(self, response: str, uml_path: str) -> Dict:
        """解析UML识别响应"""
        try:
            # 提取JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            # 修复截断的JSON
            json_str = self._fix_truncated_json(json_str)

            result = json.loads(json_str)

            # 确保必需字段
            result.setdefault('actors', [])
            result.setdefault('use_cases', [])
            result.setdefault('system_boundary', {"name": "Not Recognized", "is_present": False})
            result.setdefault('relationships', [])
            result.setdefault('overall_description', "")

            result['success'] = True
            return {"description": json.dumps(result, ensure_ascii=False), "success": True}

        except json.JSONDecodeError as e:
            logger.error(f"UML JSON解析失败: {e}")
            return {
                'description': response[:500] if response else "",
                'success': False,
                'error': str(e)
            }

    def _fix_truncated_json(self, json_str: str) -> str:
        """修复截断的JSON"""
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')

        if open_braces > close_braces or open_brackets > close_brackets:
            last_complete = max(
                json_str.rfind('},'),
                json_str.rfind('],'),
                json_str.rfind('}')
            )

            if last_complete > 0:
                json_str = json_str[:last_complete + 1]

            json_str += ']' * (open_brackets - json_str.count(']'))
            json_str += '}' * (open_braces - json_str.count('}'))

        return json_str

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


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Qwen视觉模型测试')
    parser.add_argument('image_path', help='图像路径')
    parser.add_argument('--version', choices=['qwen2.5', 'qwen3'],
                       help='强制指定模型版本（覆盖环境变量）')

    args = parser.parse_args()

    # 初始化模型（支持命令行参数覆盖）
    model = VisionModel(version=args.version)

    # 显示模型信息
    info = model.get_model_info()
    print(f"\n使用模型: {info['model_name']}")
    print(f"模型版本: {info['version']}")
    print(f"模型路径: {info['model_path']}")
    print("-" * 60)

    # 测试图像识别
    result = model.recognize_image(args.image_path)
    print("\n识别结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 测试LoRA状态
    print("\nLoRA状态:")
    print(model.get_lora_status())

# 使用示例：
# 方法1：通过环境名自动推断版本（推荐）
# python scripts/run_with_env.py --env image_qwen2.5 --script models/vision_model.py inputs/image/000000580505.jpg
# python scripts/run_with_env.py --env image_qwen3 --script models/vision_model.py inputs/image/000000580505.jpg
#
# 方法2：通过--version参数强制指定版本（最高优先级，会覆盖环境推断）
# python scripts/run_with_env.py --env image_qwen3 --script models/vision_model.py inputs/image/000000580505.jpg --version qwen2.5
#
# 方法3：直接运行（使用环境变量或默认配置）
# conda activate qwen_vision3
# python models/vision_model.py inputs/image/000000580505.jpg --version qwen3