"""
统一测试所有专家
功能:
  - 一次性测试Text/Image/UML/General四个专家
  - 支持两种模式：独立加载和共享基础模型
  - 自动生成测试报告

运行环境: qwen_text
使用方法:
  python scripts/run_with_env.py --env text --script tests/test_experts/test_all_experts.py
  或
  conda activate qwen_text && python tests/test_experts/test_all_experts.py

作者: Expert System
日期: 2026-02-06
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experts.text_expert import TextExpert
from src.experts.image_expert import ImageExpert
from src.experts.uml_expert import UMLExpert
from src.experts.general_expert import GeneralExpert
from src.experts.base_expert import BaseExpert
from src.utils.logger import get_logger

logger = get_logger('tests.experts')


class ExpertTester:
    """专家测试器"""

    def __init__(self, mode: str = 'independent'):
        """
        初始化测试器

        Args:
            mode: 测试模式
                - 'independent': 独立加载模式（每个专家独立加载和卸载模型）
                - 'shared': 共享基础模型模式（只加载一次基础模型，切换LoRA）
        """
        self.mode = mode
        self.test_results = []

        logger.info(f"初始化专家测试器 - 模式: {mode}")

    def run_all_tests(self) -> Dict:
        """
        运行所有专家测试

        Returns:
            dict: 测试结果汇总
        """
        print("=" * 80)
        print("专家系统统一测试")
        print("=" * 80)
        print(f"测试模式: {self.mode}")
        print("=" * 80)

        start_time = time.time()

        if self.mode == 'independent':
            self._run_independent_tests()
        elif self.mode == 'shared':
            self._run_shared_tests()
        else:
            logger.error(f"不支持的测试模式: {self.mode}")
            return {}

        end_time = time.time()
        total_time = end_time - start_time

        # 生成测试报告
        summary = self._generate_summary(total_time)

        return summary

    def _run_independent_tests(self):
        """运行独立加载模式的测试"""
        print("\n" + "=" * 80)
        print("模式: 独立加载（每个专家独立加载和卸载模型）")
        print("=" * 80)

        # 测试Text Expert
        self._test_text_expert_independent()

        # 测试Image Expert
        self._test_image_expert_independent()

        # 测试UML Expert
        self._test_uml_expert_independent()

        # 测试General Expert
        self._test_general_expert_independent()

    def _run_shared_tests(self):
        """运行共享基础模型模式的测试"""
        print("\n" + "=" * 80)
        print("模式: 共享基础模型（只加载一次基础模型，切换LoRA权重）")
        print("=" * 80)

        from config.settings import get_path_config
        path_cfg = get_path_config()

        # 加载共享基础模型
        print("\n" + "-" * 80)
        print("加载共享基础模型...")
        print("-" * 80)
        if not BaseExpert.load_shared_base_model(str(path_cfg.QWEN_7B_CHAT_PATH), use_4bit=True):
            logger.error("共享基础模型加载失败，终止测试")
            return

        try:
            # 测试Text Expert
            self._test_text_expert_shared()

            # 测试Image Expert
            self._test_image_expert_shared()

            # 测试UML Expert
            self._test_uml_expert_shared()

            # 测试General Expert
            self._test_general_expert_shared()

        finally:
            # 卸载共享基础模型
            print("\n" + "-" * 80)
            print("卸载共享基础模型...")
            print("-" * 80)
            BaseExpert.unload_shared_base_model()

    def _test_text_expert_independent(self):
        """测试Text Expert（独立加载）"""
        print("\n" + "=" * 80)
        print("测试1: Text Expert")
        print("=" * 80)

        start_time = time.time()
        expert = TextExpert()

        try:
            # 加载模型
            print("\n加载模型...")
            if not expert.load_model():
                self._record_result('text', False, "模型加载失败", 0)
                return

            # 生成指令
            test_input = "测试系统的登录功能,确保用户名和密码验证正确"
            print(f"\n测试输入: {test_input}")
            print("\n生成指令...")

            instruction = expert.generate_instruction(test_input)

            # 验证格式
            is_valid = expert.validate_output(instruction)

            # 显示结果
            print("\n生成的指令:")
            print("-" * 80)
            print(instruction)
            print("-" * 80)
            print(f"\n格式验证: {'通过' if is_valid else '失败'}")

            # 记录结果
            elapsed = time.time() - start_time
            self._record_result('text', is_valid, instruction, elapsed)

        finally:
            # 卸载模型
            expert.unload_model()

    def _test_text_expert_shared(self):
        """测试Text Expert（共享基础模型）"""
        print("\n" + "=" * 80)
        print("测试1: Text Expert")
        print("=" * 80)

        start_time = time.time()
        expert = TextExpert()

        try:
            # 使用共享基础模型加载
            print("\n加载LoRA权重...")
            if not expert.load_model_with_shared_base():
                self._record_result('text', False, "LoRA加载失败", 0)
                return

            # 生成指令
            test_input = "测试系统的登录功能,确保用户名和密码验证正确"
            print(f"\n测试输入: {test_input}")
            print("\n生成指令...")

            instruction = expert.generate_instruction(test_input)

            # 验证格式
            is_valid = expert.validate_output(instruction)

            # 显示结果
            print("\n生成的指令:")
            print("-" * 80)
            print(instruction)
            print("-" * 80)
            print(f"\n格式验证: {'通过' if is_valid else '失败'}")

            # 记录结果
            elapsed = time.time() - start_time
            self._record_result('text', is_valid, instruction, elapsed)

        finally:
            # 仅卸载LoRA
            expert.unload_model_keep_shared_base()

    def _test_image_expert_independent(self):
        """测试Image Expert（独立加载）"""
        print("\n" + "=" * 80)
        print("测试2: Image Expert")
        print("=" * 80)

        start_time = time.time()
        expert = ImageExpert()

        try:
            # 加载模型
            print("\n加载模型...")
            if not expert.load_model():
                self._record_result('image', False, "模型加载失败", 0)
                return

            # 生成指令
            test_input = {
                "description": "A busy urban street with cars and traffic signs",
                "details": {
                    "objects": ["car", "traffic sign"],
                    "scene": "urban street"
                }
            }
            print(f"\n测试输入: {test_input['description']}")
            print("\n生成指令...")

            instruction = expert.generate_instruction(test_input)

            # 验证格式
            is_valid = expert.validate_output(instruction)

            # 显示结果
            print("\n生成的指令:")
            print("-" * 80)
            print(instruction)
            print("-" * 80)
            print(f"\n格式验证: {'通过' if is_valid else '失败'}")

            # 记录结果
            elapsed = time.time() - start_time
            self._record_result('image', is_valid, instruction, elapsed)

        finally:
            # 卸载模型
            expert.unload_model()

    def _test_image_expert_shared(self):
        """测试Image Expert（共享基础模型）"""
        print("\n" + "=" * 80)
        print("测试2: Image Expert")
        print("=" * 80)

        start_time = time.time()
        expert = ImageExpert()

        try:
            # 使用共享基础模型加载
            print("\n加载LoRA权重...")
            if not expert.load_model_with_shared_base():
                self._record_result('image', False, "LoRA加载失败", 0)
                return

            # 生成指令
            test_input = {
                "description": "A busy urban street with cars and traffic signs",
                "details": {
                    "objects": ["car", "traffic sign"],
                    "scene": "urban street"
                }
            }
            print(f"\n测试输入: {test_input['description']}")
            print("\n生成指令...")

            instruction = expert.generate_instruction(test_input)

            # 验证格式
            is_valid = expert.validate_output(instruction)

            # 显示结果
            print("\n生成的指令:")
            print("-" * 80)
            print(instruction)
            print("-" * 80)
            print(f"\n格式验证: {'通过' if is_valid else '失败'}")

            # 记录结果
            elapsed = time.time() - start_time
            self._record_result('image', is_valid, instruction, elapsed)

        finally:
            # 仅卸载LoRA
            expert.unload_model_keep_shared_base()

    def _test_uml_expert_independent(self):
        """测试UML Expert（独立加载）"""
        print("\n" + "=" * 80)
        print("测试3: UML Expert (Qwen235B)")
        print("=" * 80)

        start_time = time.time()
        expert = UMLExpert(dataset_version='qwen235B')

        try:
            # 加载模型
            print("\n加载模型...")
            if not expert.load_model():
                self._record_result('uml', False, "模型加载失败", 0)
                return

            # 生成指令
            test_input = {
                "actors": [
                    {"name": "User", "position": "left"},
                    {"name": "Admin", "position": "right"}
                ],
                "use_cases": [
                    {"name": "Login System", "description": "User authentication"},
                    {"name": "Validate Credentials", "description": "Check credentials"}
                ],
                "relationships": [
                    {
                        "type": "association",
                        "from": "User",
                        "to": "Login System"
                    },
                    {
                        "type": "include",
                        "from": "Login System",
                        "to": "Validate Credentials"
                    }
                ]
            }
            print(f"\n测试输入: UML用例图（{len(test_input['actors'])}个角色，{len(test_input['use_cases'])}个用例）")
            print("\n生成指令...")

            instruction = expert.generate_instruction(test_input)

            # 验证格式
            is_valid = expert.validate_output(instruction)

            # 显示结果
            print("\n生成的指令:")
            print("-" * 80)
            print(instruction)
            print("-" * 80)
            print(f"\n格式验证: {'通过' if is_valid else '失败'}")

            # 记录结果
            elapsed = time.time() - start_time
            self._record_result('uml', is_valid, instruction, elapsed)

        finally:
            # 卸载模型
            expert.unload_model()

    def _test_uml_expert_shared(self):
        """测试UML Expert（共享基础模型）"""
        print("\n" + "=" * 80)
        print("测试3: UML Expert (Qwen235B)")
        print("=" * 80)

        start_time = time.time()
        expert = UMLExpert(dataset_version='qwen235B')

        try:
            # 使用共享基础模型加载
            print("\n加载LoRA权重...")
            if not expert.load_model_with_shared_base():
                self._record_result('uml', False, "LoRA加载失败", 0)
                return

            # 生成指令
            test_input = {
                "actors": [
                    {"name": "User", "position": "left"},
                    {"name": "Admin", "position": "right"}
                ],
                "use_cases": [
                    {"name": "Login System", "description": "User authentication"},
                    {"name": "Validate Credentials", "description": "Check credentials"}
                ],
                "relationships": [
                    {
                        "type": "association",
                        "from": "User",
                        "to": "Login System"
                    },
                    {
                        "type": "include",
                        "from": "Login System",
                        "to": "Validate Credentials"
                    }
                ]
            }
            print(f"\n测试输入: UML用例图（{len(test_input['actors'])}个角色，{len(test_input['use_cases'])}个用例）")
            print("\n生成指令...")

            instruction = expert.generate_instruction(test_input)

            # 验证格式
            is_valid = expert.validate_output(instruction)

            # 显示结果
            print("\n生成的指令:")
            print("-" * 80)
            print(instruction)
            print("-" * 80)
            print(f"\n格式验证: {'通过' if is_valid else '失败'}")

            # 记录结果
            elapsed = time.time() - start_time
            self._record_result('uml', is_valid, instruction, elapsed)

        finally:
            # 仅卸载LoRA
            expert.unload_model_keep_shared_base()

    def _test_general_expert_independent(self):
        """测试General Expert（独立加载）"""
        print("\n" + "=" * 80)
        print("测试4: General Expert (Qwen235B)")
        print("=" * 80)

        start_time = time.time()
        expert = GeneralExpert(dataset_version='qwen235B')

        try:
            # 加载模型
            print("\n加载模型...")
            if not expert.load_model():
                self._record_result('general', False, "模型加载失败", 0)
                return

            # 生成指令（文本输入）
            test_input = "测试系统的登录功能"
            print(f"\n测试输入（文本）: {test_input}")
            print("\n生成指令...")

            instruction = expert.generate_instruction(test_input)

            # 验证格式
            is_valid = expert.validate_output(instruction)

            # 显示结果
            print("\n生成的指令:")
            print("-" * 80)
            print(instruction)
            print("-" * 80)
            print(f"\n格式验证: {'通过' if is_valid else '失败'}")

            # 记录结果
            elapsed = time.time() - start_time
            self._record_result('general', is_valid, instruction, elapsed)

        finally:
            # 卸载模型
            expert.unload_model()

    def _test_general_expert_shared(self):
        """测试General Expert（共享基础模型）"""
        print("\n" + "=" * 80)
        print("测试4: General Expert (Qwen235B)")
        print("=" * 80)

        start_time = time.time()
        expert = GeneralExpert(dataset_version='qwen235B')

        try:
            # 使用共享基础模型加载
            print("\n加载LoRA权重...")
            if not expert.load_model_with_shared_base():
                self._record_result('general', False, "LoRA加载失败", 0)
                return

            # 生成指令（文本输入）
            test_input = "测试系统的登录功能"
            print(f"\n测试输入（文本）: {test_input}")
            print("\n生成指令...")

            instruction = expert.generate_instruction(test_input)

            # 验证格式
            is_valid = expert.validate_output(instruction)

            # 显示结果
            print("\n生成的指令:")
            print("-" * 80)
            print(instruction)
            print("-" * 80)
            print(f"\n格式验证: {'通过' if is_valid else '失败'}")

            # 记录结果
            elapsed = time.time() - start_time
            self._record_result('general', is_valid, instruction, elapsed)

        finally:
            # 仅卸载LoRA
            expert.unload_model_keep_shared_base()

    def _record_result(self, expert_type: str, is_valid: bool, instruction: str, elapsed_time: float):
        """记录测试结果"""
        result = {
            'expert': expert_type,
            'valid': is_valid,
            'instruction': instruction,
            'time': elapsed_time
        }
        self.test_results.append(result)

    def _generate_summary(self, total_time: float) -> Dict:
        """生成测试摘要"""
        print("\n" + "=" * 80)
        print("测试摘要")
        print("=" * 80)

        summary = {
            'mode': self.mode,
            'total_time': total_time,
            'experts_tested': len(self.test_results),
            'all_passed': all(r['valid'] for r in self.test_results),
            'results': self.test_results
        }

        print(f"\n测试模式: {self.mode}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"测试专家数: {len(self.test_results)}")
        print(f"全部通过: {'是' if summary['all_passed'] else '否'}")

        print("\n详细结果:")
        for result in self.test_results:
            status = "通过" if result['valid'] else "失败"
            print(f"  - {result['expert']:10s}: {status:4s} ({result['time']:.2f}秒)")

        return summary


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='统一测试所有专家')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['independent', 'shared'],
        default='independent',
        help='测试模式: independent(独立加载) 或 shared(共享基础模型)'
    )

    args = parser.parse_args()

    # 运行测试
    tester = ExpertTester(mode=args.mode)
    summary = tester.run_all_tests()

    # 保存测试报告
    output_dir = Path("outputs/tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"expert_test_report_{args.mode}_{timestamp}.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n测试报告已保存: {report_file}")

    # 退出码
    exit_code = 0 if summary.get('all_passed', False) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

# python scripts/run_with_env.py --env text --script tests/test_experts/test_all_experts.py -- --mode shared
