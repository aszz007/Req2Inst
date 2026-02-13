"""
流水线管理器：自动化执行整个项目工作流
功能：
  - 自动切换环境执行各个阶段的脚本
  - 支持完整流水线和单阶段执行
  - 提供进度跟踪和错误恢复
  - 生成执行报告

用法：
  # 执行完整流水线
  python scripts/pipeline_manager.py --full --version qwen2.5

  # 仅执行预处理阶段
  python scripts/pipeline_manager.py --stage preprocess --version qwen2.5

  # 仅执行训练阶段
  python scripts/pipeline_manager.py --stage training --version qwen2.5

  # 从某个阶段恢复执行
  python scripts/pipeline_manager.py --full --version qwen2.5 --resume-from training
"""

import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import sys

# ==================== 配置 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 环境映射
ENV_MAP = {
    'text': 'qwen_text',
    'vision25': 'qwen_vision25',
    'vision3': 'qwen_vision3',
}


# ==================== 颜色输出 ====================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_stage_header(stage_name: str, stage_num: int, total_stages: int):
    """打印阶段标题"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}阶段 [{stage_num}/{total_stages}]: {stage_name}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}\n")


def print_success(message: str):
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message: str):
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def print_info(message: str):
    print(f"{Colors.CYAN}ℹ {message}{Colors.END}")


def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


# ==================== 流水线配置 ====================

class PipelineStage:
    """流水线阶段配置"""

    def __init__(self,
                 name: str,
                 script: str,
                 env: str,
                 args: List[str] = None,
                 description: str = ""):
        self.name = name
        self.script = script
        self.env = env
        self.args = args or []
        self.description = description

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'script': self.script,
            'env': self.env,
            'args': self.args,
            'description': self.description
        }


class Pipeline:
    """流水线定义"""

    def __init__(self, version: str = 'qwen2.5'):
        self.version = version
        self.stages = self._build_stages()

    def _build_stages(self) -> List[PipelineStage]:
        """构建流水线阶段"""

        vision_env = 'vision25' if self.version == 'qwen2.5' else 'vision3'

        stages = []

        # ==================== 阶段1: 数据预处理 ====================
        stages.append(PipelineStage(
            name='preprocess_image',
            script='scripts/recognize_image.py',
            env=vision_env,
            args=['--version', self.version],
            description='批量识别图像内容'
        ))

        stages.append(PipelineStage(
            name='preprocess_uml',
            script='scripts/recognize_uml.py',
            env=vision_env,
            args=['--version', self.version],
            description='批量识别UML用例图'
        ))

        # ==================== 阶段2: 数据集构建 ====================
        stages.append(PipelineStage(
            name='build_dataset_image',
            script='scripts/build_final_dataset/image/generate_instructions.py',
            env='text',  # 不需要特殊模型
            description='构建图像数据集'
        ))

        stages.append(PipelineStage(
            name='build_dataset_uml',
            script='scripts/build_final_dataset/uml/generate_instructions.py',
            env='text',
            description='构建UML数据集'
        ))

        # ==================== 阶段3: LoRA训练 ====================
        # 重要：所有Expert训练都必须在qwen_text环境执行
        # 原因：所有Expert都基于Qwen-7B-Chat训练，处理的是文本输入

        stages.append(PipelineStage(
            name='train_text_expert',
            script='scripts/training/train_text_expert.py',
            env='text',
            description='训练文本专家'
        ))

        # Image Expert: 只有1个版本（数据集只有1个版本）
        stages.append(PipelineStage(
            name='train_image_expert',
            script='scripts/training/train_image_expert.py',
            env='text',  # 必须在text环境训练
            description='训练图像专家'
        ))

        # UML Expert: 有3个版本（根据不同视觉模型识别的数据集）
        stages.append(PipelineStage(
            name=f'train_uml_expert_dataset_{self.version}',
            script='scripts/training/train_uml_expert.py',
            env='text',  # 必须在text环境训练
            args=['--dataset', self.version],  # 指定数据集版本
            description=f'训练UML专家 (dataset: {self.version})'
        ))

        # General Expert: 有3个版本（根据不同视觉模型识别的数据集）
        stages.append(PipelineStage(
            name=f'train_general_expert_dataset_{self.version}',
            script='scripts/training/train_general_expert.py',
            env='text',  # 必须在text环境训练
            args=['--dataset', self.version],  # 指定数据集版本
            description=f'训练通用专家 (dataset: {self.version})'
        ))

        # ==================== 阶段4: 评估 ====================
        stages.append(PipelineStage(
            name='evaluate_experts',
            script='scripts/evaluation/evaluate_experts.py',
            env='text',
            description='评估所有专家性能'
        ))

        return stages

    def get_stage_groups(self) -> Dict[str, List[str]]:
        """获取阶段分组"""
        return {
            'preprocess': ['preprocess_image', 'preprocess_uml'],
            'dataset': ['build_dataset_image', 'build_dataset_uml'],
            'training': [
                'train_text_expert',
                'train_image_expert',
                f'train_uml_expert_dataset_{self.version}',
                f'train_general_expert_dataset_{self.version}'
            ],
            'evaluation': ['evaluate_experts']
        }

    def get_stages_by_group(self, group: str) -> List[PipelineStage]:
        """根据分组获取阶段"""
        groups = self.get_stage_groups()
        if group not in groups:
            raise ValueError(f"未知的阶段组: {group}")

        stage_names = groups[group]
        return [s for s in self.stages if s.name in stage_names]


# ==================== 流水线执行器 ====================

class PipelineExecutor:
    """流水线执行器"""

    def __init__(self, pipeline: Pipeline, log_file: Path = None):
        self.pipeline = pipeline
        self.log_file = log_file or Path("pipeline_execution.log")
        self.results = []

    def run_stage(self, stage: PipelineStage, stage_num: int, total_stages: int) -> bool:
        """
        执行单个阶段

        Returns:
            bool: 是否成功
        """
        print_stage_header(stage.name, stage_num, total_stages)

        if stage.description:
            print_info(f"描述: {stage.description}")

        print_info(f"环境: {ENV_MAP[stage.env]}")
        print_info(f"脚本: {stage.script}")

        if stage.args:
            print_info(f"参数: {' '.join(stage.args)}")

        print()

        # 检查脚本是否存在
        script_path = PROJECT_ROOT / stage.script
        if not script_path.exists():
            print_warning(f"脚本不存在，跳过: {script_path}")
            return True  # 暂时返回True，允许继续

        # 构建命令
        cmd = [
                  'conda', 'run',
                  '-n', ENV_MAP[stage.env],
                  '--no-capture-output',
                  'python', str(script_path)
              ] + stage.args

        # 记录开始时间
        start_time = time.time()

        # 执行命令
        try:
            result = subprocess.run(cmd)
            exit_code = result.returncode
            success = (exit_code == 0)
        except KeyboardInterrupt:
            print_warning("\n用户中断执行")
            success = False
            exit_code = 130
        except Exception as e:
            print_error(f"执行失败: {e}")
            success = False
            exit_code = 1

        # 记录结束时间
        elapsed_time = time.time() - start_time

        # 记录结果
        stage_result = {
            'stage': stage.name,
            'success': success,
            'exit_code': exit_code,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(stage_result)

        # 打印结果
        print(f"\n{'-' * 80}")
        if success:
            print_success(f"阶段 {stage.name} 完成！耗时: {elapsed_time:.2f}秒")
        else:
            print_error(f"阶段 {stage.name} 失败！退出码: {exit_code}")
        print(f"{'-' * 80}\n")

        return success

    def run_pipeline(self,
                     stages: List[PipelineStage] = None,
                     resume_from: str = None) -> bool:
        """
        执行流水线

        Args:
            stages: 要执行的阶段列表（None则执行所有）
            resume_from: 从某个阶段恢复执行

        Returns:
            bool: 是否全部成功
        """
        if stages is None:
            stages = self.pipeline.stages

        # 处理resume_from
        if resume_from:
            stage_names = [s.name for s in stages]
            if resume_from not in stage_names:
                print_error(f"未找到阶段: {resume_from}")
                return False

            resume_idx = stage_names.index(resume_from)
            stages = stages[resume_idx:]
            print_info(f"从阶段 {resume_from} 恢复执行")

        total_stages = len(stages)
        success_count = 0
        fail_count = 0

        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'流水线执行开始'.center(80)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

        print_info(f"总阶段数: {total_stages}")
        print_info(f"模型版本: {self.pipeline.version}")
        print_info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 逐个执行阶段
        for i, stage in enumerate(stages, 1):
            success = self.run_stage(stage, i, total_stages)

            if success:
                success_count += 1
            else:
                fail_count += 1

                # 询问是否继续
                print_warning("阶段执行失败！")
                response = input("是否继续执行？(y/n): ").strip().lower()
                if response != 'y':
                    print_info("流水线执行中止")
                    break

        # 打印总结
        self._print_summary(success_count, fail_count, total_stages)

        # 保存执行报告
        self._save_report()

        return fail_count == 0

    def _print_summary(self, success_count: int, fail_count: int, total_stages: int):
        """打印执行总结"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'流水线执行总结'.center(80)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

        print(f"总阶段数: {total_stages}")
        print_success(f"成功: {success_count}")

        if fail_count > 0:
            print_error(f"失败: {fail_count}")
        else:
            print_info(f"失败: {fail_count}")

        success_rate = (success_count / total_stages * 100) if total_stages > 0 else 0
        print(f"成功率: {success_rate:.1f}%")

        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

    def _save_report(self):
        """保存执行报告"""
        report = {
            'version': self.pipeline.version,
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }

        report_file = PROJECT_ROOT / 'outputs' / 'pipeline_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print_info(f"执行报告已保存: {report_file}")


# ==================== 命令行接口 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='流水线管理器：自动化执行整个项目工作流',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--version',
        type=str,
        default='qwen2.5',
        choices=['qwen2.5', 'qwen3'],
        help='视觉模型版本（默认: qwen2.5）'
    )

    # 执行模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--full',
        action='store_true',
        help='执行完整流水线'
    )
    mode_group.add_argument(
        '--stage',
        type=str,
        choices=['preprocess', 'dataset', 'training', 'evaluation'],
        help='执行特定阶段组'
    )

    parser.add_argument(
        '--resume-from',
        type=str,
        help='从某个阶段恢复执行（阶段名称）'
    )

    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='列出所有阶段'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建流水线
    pipeline = Pipeline(version=args.version)

    # 列出阶段
    if args.list_stages:
        print(f"\n{Colors.BOLD}流水线阶段列表 (版本: {args.version}){Colors.END}\n")
        groups = pipeline.get_stage_groups()
        for group_name, stage_names in groups.items():
            print(f"{Colors.CYAN}{group_name}:{Colors.END}")
            for stage_name in stage_names:
                stage = next(s for s in pipeline.stages if s.name == stage_name)
                print(f"  - {stage_name}: {stage.description}")
            print()
        return

    # 创建执行器
    executor = PipelineExecutor(pipeline)

    # 确定要执行的阶段
    if args.full:
        stages = pipeline.stages
    else:
        stages = pipeline.get_stages_by_group(args.stage)

    # 执行流水线
    success = executor.run_pipeline(
        stages=stages,
        resume_from=args.resume_from
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()