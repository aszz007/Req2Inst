"""
Enhanced Metrics for MoE Instruction Generation System
MoE指令生成系统的增强评估指标

评估指标体系:
1. 生成质量指标(Generation Quality Metrics)
   - BLEU, ROUGE, METEOR: 词汇匹配指标
   - BERTScore: 语义相似度指标

2. 指令格式指标(Instruction Format Metrics)
   - 三段式结构完整性
   - 各段落有效性检查

3. 统计指标(Statistical Metrics)
   - 生成长度统计
   - 专家使用统计

Author: Claude
Date: 2026-02-03
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import warnings

try:
    from evaluate import load

    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False
    warnings.warn("evaluate library not installed. BLEU/ROUGE/METEOR will be unavailable.")

try:
    from bert_score import score as bert_score_fn

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    warnings.warn("bert_score library not installed. BERTScore will be unavailable.")


@dataclass
class GenerationMetrics:
    """生成质量指标"""
    bleu: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    meteor: float = 0.0
    bert_score_f1: float = 0.0
    bert_score_precision: float = 0.0
    bert_score_recall: float = 0.0


@dataclass
class InstructionFormatMetrics:
    """指令格式指标"""
    has_definition: bool = False
    has_emphasis: bool = False
    has_things_to_avoid: bool = False
    is_complete: bool = False
    definition_valid: bool = False
    emphasis_valid: bool = False
    avoid_valid: bool = False
    format_score: float = 0.0


@dataclass
class StatisticalMetrics:
    """统计指标"""
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    std_length: float = 0.0
    avg_word_count: float = 0.0
    length_in_range: bool = True


@dataclass
class ComprehensiveMetrics:
    """综合评估结果"""
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    format: InstructionFormatMetrics = field(default_factory=InstructionFormatMetrics)
    statistical: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    expert_usage: Dict[str, int] = field(default_factory=dict)


class InstructionMetricsCalculator:
    """
    指令评估指标计算器

    提供完整的评估指标计算功能
    """

    def __init__(self):
        """初始化评估器"""
        self.bleu = None
        self.rouge = None
        self.meteor = None

        if HAS_EVALUATE:
            try:
                self.bleu = load('bleu')
                self.rouge = load('rouge')
                self.meteor = load('meteor')
            except Exception as e:
                warnings.warn(f"Failed to load evaluate metrics: {e}")

    def calculate_generation_metrics(
            self,
            predictions: List[str],
            references: List[str]
    ) -> GenerationMetrics:
        """
        计算生成质量指标

        Args:
            predictions: 生成的指令列表
            references: 参考指令列表

        Returns:
            GenerationMetrics
        """
        metrics = GenerationMetrics()

        if not predictions or not references:
            return metrics

        if len(predictions) != len(references):
            warnings.warn(f"Predictions ({len(predictions)}) and references ({len(references)}) length mismatch")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]

        if HAS_EVALUATE and self.bleu:
            try:
                bleu_result = self.bleu.compute(
                    predictions=predictions,
                    references=references
                )
                metrics.bleu = bleu_result.get('bleu', 0.0)
            except Exception as e:
                warnings.warn(f"BLEU calculation failed: {e}")

        if HAS_EVALUATE and self.rouge:
            try:
                rouge_result = self.rouge.compute(
                    predictions=predictions,
                    references=references
                )
                metrics.rouge1 = rouge_result.get('rouge1', 0.0)
                metrics.rouge2 = rouge_result.get('rouge2', 0.0)
                metrics.rougeL = rouge_result.get('rougeL', 0.0)
            except Exception as e:
                warnings.warn(f"ROUGE calculation failed: {e}")

        if HAS_EVALUATE and self.meteor:
            try:
                meteor_result = self.meteor.compute(
                    predictions=predictions,
                    references=references
                )
                metrics.meteor = meteor_result.get('meteor', 0.0)
            except Exception as e:
                warnings.warn(f"METEOR calculation failed: {e}")

        if HAS_BERTSCORE:
            try:
                P, R, F1 = bert_score_fn(
                    predictions,
                    references,
                    lang='en',
                    verbose=False
                )
                metrics.bert_score_precision = P.mean().item()
                metrics.bert_score_recall = R.mean().item()
                metrics.bert_score_f1 = F1.mean().item()
            except Exception as e:
                warnings.warn(f"BERTScore calculation failed: {e}")

        return metrics

    def check_instruction_format(self, instruction: str) -> InstructionFormatMetrics:
        """
        检查单条指令的格式完整性

        三段式格式要求:
        1. Definition: 必须有实际内容
        2. Emphasis/Caution: 可以有内容或显式的"-"
        3. Things to Avoid: 可以有内容或显式的"-"

        Args:
            instruction: 指令文本

        Returns:
            InstructionFormatMetrics
        """
        metrics = InstructionFormatMetrics()

        instruction_lower = instruction.lower()

        metrics.has_definition = self._check_section_exists(
            instruction_lower,
            ['definition', 'in this task']
        )
        metrics.has_emphasis = self._check_section_exists(
            instruction_lower,
            ['emphasis', 'caution']
        )
        metrics.has_things_to_avoid = self._check_section_exists(
            instruction_lower,
            ['things to avoid', 'avoid']
        )

        metrics.definition_valid = self._validate_definition(instruction)
        metrics.emphasis_valid = self._validate_optional_section(
            instruction,
            ['emphasis', 'caution']
        )
        metrics.avoid_valid = self._validate_optional_section(
            instruction,
            ['things to avoid', 'avoid']
        )

        metrics.is_complete = (
                metrics.has_definition and
                metrics.has_emphasis and
                metrics.has_things_to_avoid and
                metrics.definition_valid
        )

        valid_sections = sum([
            metrics.definition_valid,
            metrics.emphasis_valid,
            metrics.avoid_valid
        ])
        metrics.format_score = valid_sections / 3.0

        return metrics

    def _check_section_exists(self, text: str, keywords: List[str]) -> bool:
        """检查段落是否存在"""
        return any(keyword in text for keyword in keywords)

    def _validate_definition(self, instruction: str) -> bool:
        """
        验证Definition段落

        Definition必须有实际内容，不能只是"-"
        """
        patterns = [
            r'definition[:\s]+(.+?)(?:emphasis|caution|things to avoid|$)',
            r'in this task[,\s]+(.+?)(?:emphasis|caution|things to avoid|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, instruction, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and content != '-' and len(content) > 10:
                    return True

        return False

    def _validate_optional_section(
            self,
            instruction: str,
            keywords: List[str]
    ) -> bool:
        """
        验证可选段落(Emphasis或Things to Avoid)

        有内容或显式的"-"都算有效
        """
        for keyword in keywords:
            pattern = rf'{keyword}[:\s]+(.+?)(?:definition|emphasis|caution|things to avoid|$)'
            match = re.search(pattern, instruction, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content:
                    return True

        return False

    def calculate_format_metrics_batch(
            self,
            instructions: List[str]
    ) -> Tuple[float, int, int]:
        """
        批量计算格式指标

        Args:
            instructions: 指令列表

        Returns:
            (平均格式分数, 完整指令数量, 总指令数量)
        """
        if not instructions:
            return 0.0, 0, 0

        total_score = 0.0
        complete_count = 0

        for instruction in instructions:
            format_metrics = self.check_instruction_format(instruction)
            total_score += format_metrics.format_score
            if format_metrics.is_complete:
                complete_count += 1

        avg_score = total_score / len(instructions)
        return avg_score, complete_count, len(instructions)

    def calculate_statistical_metrics(
            self,
            instructions: List[str],
            min_length: int = 50,
            max_length: int = 1000
    ) -> StatisticalMetrics:
        """
        计算统计指标

        Args:
            instructions: 指令列表
            min_length: 最小合理长度
            max_length: 最大合理长度

        Returns:
            StatisticalMetrics
        """
        metrics = StatisticalMetrics()

        if not instructions:
            return metrics

        lengths = [len(inst) for inst in instructions]
        word_counts = [len(inst.split()) for inst in instructions]

        metrics.avg_length = np.mean(lengths)
        metrics.min_length = np.min(lengths)
        metrics.max_length = np.max(lengths)
        metrics.std_length = np.std(lengths)
        metrics.avg_word_count = np.mean(word_counts)

        in_range_count = sum(
            1 for length in lengths
            if min_length <= length <= max_length
        )
        metrics.length_in_range = (in_range_count / len(instructions)) > 0.8

        return metrics

    def evaluate_comprehensive(
            self,
            predictions: List[str],
            references: List[str],
            expert_usage: Optional[Dict[str, int]] = None
    ) -> ComprehensiveMetrics:
        """
        综合评估

        Args:
            predictions: 生成的指令列表
            references: 参考指令列表
            expert_usage: 专家使用统计

        Returns:
            ComprehensiveMetrics
        """
        result = ComprehensiveMetrics()

        result.generation = self.calculate_generation_metrics(predictions, references)

        avg_format_score, complete_count, total_count = self.calculate_format_metrics_batch(predictions)
        if predictions:
            format_metrics = self.check_instruction_format(predictions[0])
            format_metrics.format_score = avg_format_score
            result.format = format_metrics

        result.statistical = self.calculate_statistical_metrics(predictions)

        if expert_usage:
            result.expert_usage = expert_usage

        return result

    def print_metrics_report(self, metrics: ComprehensiveMetrics):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("MoE Instruction Generation - Evaluation Report")
        print("=" * 60)

        print("\n[Generation Quality Metrics]")
        print(f"  BLEU:              {metrics.generation.bleu:.4f}")
        print(f"  ROUGE-1:           {metrics.generation.rouge1:.4f}")
        print(f"  ROUGE-2:           {metrics.generation.rouge2:.4f}")
        print(f"  ROUGE-L:           {metrics.generation.rougeL:.4f}")
        print(f"  METEOR:            {metrics.generation.meteor:.4f}")
        if metrics.generation.bert_score_f1 > 0:
            print(f"  BERTScore (F1):    {metrics.generation.bert_score_f1:.4f}")
            print(f"  BERTScore (P):     {metrics.generation.bert_score_precision:.4f}")
            print(f"  BERTScore (R):     {metrics.generation.bert_score_recall:.4f}")

        print("\n[Instruction Format Metrics]")
        print(f"  Format Score:      {metrics.format.format_score:.2%}")
        print(f"  Has Definition:    {metrics.format.has_definition}")
        print(f"  Has Emphasis:      {metrics.format.has_emphasis}")
        print(f"  Has Avoid:         {metrics.format.has_things_to_avoid}")
        print(f"  Is Complete:       {metrics.format.is_complete}")

        print("\n[Statistical Metrics]")
        print(f"  Avg Length:        {metrics.statistical.avg_length:.1f} chars")
        print(f"  Length Range:      [{metrics.statistical.min_length}, {metrics.statistical.max_length}]")
        print(f"  Std Dev:           {metrics.statistical.std_length:.1f}")
        print(f"  Avg Word Count:    {metrics.statistical.avg_word_count:.1f}")
        print(f"  Length In Range:   {metrics.statistical.length_in_range}")

        if metrics.expert_usage:
            print("\n[Expert Usage Statistics]")
            total = sum(metrics.expert_usage.values())
            for expert, count in sorted(metrics.expert_usage.items()):
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  {expert:30s}: {count:4d} ({percentage:5.1f}%)")

        print("\n" + "=" * 60)