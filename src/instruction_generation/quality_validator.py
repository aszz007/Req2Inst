"""
Quality Validator - Instruction Quality Validation
质量验证器 - 指令质量验证

功能:
  - 验证指令是否符合三段式格式
  - 检查格式完整性和内容有效性
  - 提供详细的验证报告
  - 支持批量验证

环境要求: instruction_generator
依赖: 无特殊依赖

作者: Quality Validation System
日期: 2025-02-06
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger('instruction_generation.quality_validator')


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool
    has_definition: bool
    has_emphasis: bool
    has_things_to_avoid: bool
    definition_has_content: bool
    definition_starts_with_in_this_task: bool
    emphasis_is_valid: bool
    avoid_is_valid: bool
    format_score: float
    errors: List[str]
    warnings: List[str]


class QualityValidator:
    """
    质量验证器

    验证指令是否符合三段式格式要求
    """

    def __init__(self, strict_mode: bool = False):
        """
        初始化验证器

        Args:
            strict_mode: 严格模式
                - False: Definition必须有内容,Emphasis/Avoid可以是"-"
                - True: 三个部分都必须有实际内容
        """
        self.strict_mode = strict_mode
        logger.info(f"质量验证器初始化完成 - 严格模式: {strict_mode}")

    def validate_instruction(self, instruction: str) -> ValidationResult:
        """
        验证单条指令

        Args:
            instruction: 指令文本

        Returns:
            ValidationResult: 验证结果
        """
        errors = []
        warnings = []

        # 初始化结果
        result = {
            'is_valid': False,
            'has_definition': False,
            'has_emphasis': False,
            'has_things_to_avoid': False,
            'definition_has_content': False,
            'definition_starts_with_in_this_task': False,
            'emphasis_is_valid': False,
            'avoid_is_valid': False,
            'format_score': 0.0
        }

        # 基本检查
        if not instruction or len(instruction.strip()) < 20:
            errors.append("指令内容过短或为空")
            return ValidationResult(**result, errors=errors, warnings=warnings)

        # 按行分割
        lines = instruction.split('\n')

        # 查找三段式的三个部分
        definition_line = None
        emphasis_line = None
        avoid_line = None

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            if line_stripped.startswith('Definition:'):
                definition_line = line_stripped
                result['has_definition'] = True
            elif line_stripped.startswith('Emphasis & Caution:') or line_stripped.startswith('Emphasis and Caution:'):
                emphasis_line = line_stripped
                result['has_emphasis'] = True
            elif line_stripped.startswith('Things to Avoid:'):
                avoid_line = line_stripped
                result['has_things_to_avoid'] = True

        # 检查Definition
        if definition_line:
            content = definition_line.split('Definition:', 1)[1].strip()

            # Definition不能只是"-"或为空
            if content and content != '-':
                result['definition_has_content'] = True
            else:
                errors.append("Definition没有实际内容(不能只是'-')")

            # 检查是否以"In this task"开头
            if content.lower().startswith('in this task'):
                result['definition_starts_with_in_this_task'] = True
            else:
                warnings.append("Definition建议以'In this task'开头")

            # 检查Definition长度
            if len(content) < 10:
                warnings.append("Definition内容过短")
        else:
            errors.append("缺少Definition部分")

        # 检查Emphasis & Caution
        if emphasis_line:
            content = emphasis_line.split(':', 1)[1].strip()

            if self.strict_mode:
                # 严格模式:必须有实际内容
                if content and content != '-':
                    result['emphasis_is_valid'] = True
                else:
                    errors.append("Emphasis & Caution必须有实际内容(严格模式)")
            else:
                # 非严格模式:有内容或显式"-"都可以
                if content:
                    result['emphasis_is_valid'] = True
                    if content == '-':
                        warnings.append("Emphasis & Caution为'-',建议提供具体内容")
        else:
            errors.append("缺少Emphasis & Caution部分")

        # 检查Things to Avoid
        if avoid_line:
            content = avoid_line.split(':', 1)[1].strip()

            if self.strict_mode:
                # 严格模式:必须有实际内容
                if content and content != '-':
                    result['avoid_is_valid'] = True
                else:
                    errors.append("Things to Avoid必须有实际内容(严格模式)")
            else:
                # 非严格模式:有内容或显式"-"都可以
                if content:
                    result['avoid_is_valid'] = True
                    if content == '-':
                        warnings.append("Things to Avoid为'-',建议提供具体内容")
        else:
            errors.append("缺少Things to Avoid部分")

        # 计算格式分数(0-1)
        score_components = [
            result['has_definition'],
            result['definition_has_content'],
            result['definition_starts_with_in_this_task'],
            result['has_emphasis'],
            result['emphasis_is_valid'],
            result['has_things_to_avoid'],
            result['avoid_is_valid']
        ]
        result['format_score'] = sum(score_components) / len(score_components)

        # 综合判断是否有效
        if self.strict_mode:
            # 严格模式:所有部分都必须有实际内容
            result['is_valid'] = (
                    result['definition_has_content'] and
                    result['has_emphasis'] and
                    result['emphasis_is_valid'] and
                    result['has_things_to_avoid'] and
                    result['avoid_is_valid']
            )
        else:
            # 非严格模式:Definition必须有内容,Emphasis/Avoid存在即可
            result['is_valid'] = (
                    result['definition_has_content'] and
                    result['has_emphasis'] and
                    result['has_things_to_avoid']
            )

        return ValidationResult(**result, errors=errors, warnings=warnings)

    def batch_validate(
            self,
            instructions: List[str]
    ) -> Tuple[List[ValidationResult], Dict]:
        """
        批量验证指令

        Args:
            instructions: 指令列表

        Returns:
            tuple: (验证结果列表, 统计摘要字典)
        """
        logger.info(f"批量验证 - 共{len(instructions)}条指令")

        results = []
        for i, instruction in enumerate(instructions, 1):
            result = self.validate_instruction(instruction)
            results.append(result)

            if not result.is_valid:
                logger.debug(f"指令{i}验证失败: {result.errors}")

        # 生成统计摘要
        summary = self._generate_summary(results)

        logger.info(f"验证完成 - 通过率: {summary['pass_rate']:.2%}")

        return results, summary

    def _generate_summary(self, results: List[ValidationResult]) -> Dict:
        """
        生成验证统计摘要

        Args:
            results: 验证结果列表

        Returns:
            dict: 统计摘要
        """
        total = len(results)

        if total == 0:
            return {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0
            }

        passed = sum(1 for r in results if r.is_valid)
        failed = total - passed

        summary = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total,

            # 分项统计
            'definition_present_rate': sum(1 for r in results if r.has_definition) / total,
            'definition_has_content_rate': sum(1 for r in results if r.definition_has_content) / total,
            'definition_starts_with_in_this_task_rate': sum(
                1 for r in results if r.definition_starts_with_in_this_task) / total,

            'emphasis_present_rate': sum(1 for r in results if r.has_emphasis) / total,
            'emphasis_valid_rate': sum(1 for r in results if r.emphasis_is_valid) / total,

            'avoid_present_rate': sum(1 for r in results if r.has_things_to_avoid) / total,
            'avoid_valid_rate': sum(1 for r in results if r.avoid_is_valid) / total,

            # 格式分数
            'avg_format_score': sum(r.format_score for r in results) / total,
            'min_format_score': min(r.format_score for r in results),
            'max_format_score': max(r.format_score for r in results),

            # 错误统计
            'total_errors': sum(len(r.errors) for r in results),
            'total_warnings': sum(len(r.warnings) for r in results),

            # 常见错误
            'common_errors': self._count_common_errors(results)
        }

        return summary

    def _count_common_errors(self, results: List[ValidationResult]) -> Dict[str, int]:
        """
        统计常见错误

        Args:
            results: 验证结果列表

        Returns:
            dict: 错误类型及其出现次数
        """
        error_counts = {}

        for result in results:
            for error in result.errors:
                error_counts[error] = error_counts.get(error, 0) + 1

        # 按出现次数排序
        sorted_errors = dict(
            sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_errors

    def print_validation_report(
            self,
            results: List[ValidationResult],
            summary: Dict,
            show_details: bool = False
    ):
        """
        打印验证报告

        Args:
            results: 验证结果列表
            summary: 统计摘要
            show_details: 是否显示详细信息
        """
        print("\n" + "=" * 80)
        print("指令质量验证报告")
        print("=" * 80)

        # 总体统计
        print(f"\n[总体统计]")
        print(f"  总计:     {summary['total']} 条")
        print(f"  通过:     {summary['passed']} 条")
        print(f"  失败:     {summary['failed']} 条")
        print(f"  通过率:   {summary['pass_rate']:.2%}")

        # 格式分数
        print(f"\n[格式分数]")
        print(f"  平均分数: {summary['avg_format_score']:.4f}")
        print(f"  最高分数: {summary['max_format_score']:.4f}")
        print(f"  最低分数: {summary['min_format_score']:.4f}")

        # 分项统计
        print(f"\n[分项统计]")
        print(f"  Definition存在率:   {summary['definition_present_rate']:.2%}")
        print(f"  Definition有效率:   {summary['definition_has_content_rate']:.2%}")
        print(f"  Emphasis存在率:     {summary['emphasis_present_rate']:.2%}")
        print(f"  Emphasis有效率:     {summary['emphasis_valid_rate']:.2%}")
        print(f"  Avoid存在率:        {summary['avoid_present_rate']:.2%}")
        print(f"  Avoid有效率:        {summary['avoid_valid_rate']:.2%}")

        # 错误统计
        print(f"\n[错误统计]")
        print(f"  总错误数:   {summary['total_errors']}")
        print(f"  总警告数:   {summary['total_warnings']}")

        if summary['common_errors']:
            print(f"\n[常见错误Top 5]")
            for i, (error, count) in enumerate(list(summary['common_errors'].items())[:5], 1):
                print(f"  {i}. {error}: {count}次")

        # 详细信息
        if show_details and results:
            print(f"\n[详细信息]")
            for i, result in enumerate(results[:10], 1):  # 只显示前10条
                print(f"\n指令 {i}:")
                print(f"  有效: {result.is_valid}")
                print(f"  分数: {result.format_score:.4f}")
                if result.errors:
                    print(f"  错误: {', '.join(result.errors)}")
                if result.warnings:
                    print(f"  警告: {', '.join(result.warnings)}")

            if len(results) > 10:
                print(f"\n  ... (还有 {len(results) - 10} 条)")

        print("=" * 80 + "\n")

    def filter_valid_instructions(
            self,
            instructions: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        过滤出有效的指令

        Args:
            instructions: 指令列表

        Returns:
            tuple: (有效指令列表, 无效指令列表)
        """
        valid = []
        invalid = []

        for instruction in instructions:
            result = self.validate_instruction(instruction)
            if result.is_valid:
                valid.append(instruction)
            else:
                invalid.append(instruction)

        logger.info(f"过滤完成 - 有效: {len(valid)}, 无效: {len(invalid)}")

        return valid, invalid


if __name__ == "__main__":
    print("=" * 60)
    print("质量验证器测试")
    print("=" * 60)

    # 创建验证器
    validator = QualityValidator(strict_mode=False)

    # 测试数据
    test_instructions = [
        # 有效指令
        """Definition: In this task, draw bounding boxes around all visible objects in the image.
Emphasis & Caution: Focus on accurately identifying and labeling all foreground objects.
Things to Avoid: Do not annotate background elements or partial objects.""",

        # Definition为空
        """Definition: -
Emphasis & Caution: Test thoroughly.
Things to Avoid: Do not skip validation.""",

        # Emphasis和Avoid为"-"(非严格模式下有效)
        """Definition: In this task, implement the login functionality.
Emphasis & Caution: -
Things to Avoid: -""",

        # 缺少部分
        """Definition: In this task, test the system.
Emphasis & Caution: Be careful.""",

        # 格式完全正确
        """Definition: In this task, validate user credentials during authentication.
Emphasis & Caution: Ensure secure handling of passwords and implement rate limiting.
Things to Avoid: Do not store passwords in plain text or log sensitive information."""
    ]

    print("\n测试1: 单条验证")
    print("-" * 60)
    result = validator.validate_instruction(test_instructions[0])
    print(f"有效: {result.is_valid}")
    print(f"格式分数: {result.format_score:.4f}")
    print(f"错误: {result.errors}")
    print(f"警告: {result.warnings}")

    print("\n测试2: 批量验证")
    print("-" * 60)
    results, summary = validator.batch_validate(test_instructions)
    validator.print_validation_report(results, summary, show_details=True)

    print("\n测试3: 过滤有效指令")
    print("-" * 60)
    valid, invalid = validator.filter_valid_instructions(test_instructions)
    print(f"有效指令数: {len(valid)}")
    print(f"无效指令数: {len(invalid)}")

    print("\n测试完成!")