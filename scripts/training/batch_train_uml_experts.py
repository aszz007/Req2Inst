#!/usr/bin/env python
"""
UML专家批量训练脚本 - 6个对比实验自动化

功能：自动执行2个模型 × 3个数据集 = 6个UML Expert训练实验
适用环境：RTX 4090 (24GB显存)

实验矩阵：
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ 模型 \ 数据集     │ QW2.5识别         │ QW3识别           │ QW235B识别        │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Qwen2.5-VL-7B   │ 实验1 ✓           │ 实验2 ✓           │ 实验3 ✓           │
│ Qwen3-VL-8B     │ 实验4 ✓           │ 实验5 ✓           │ 实验6 ✓           │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘

使用方法：
  # 执行全部6个实验
  python scripts/training/batch_train_uml_experts.py --all

  # 仅训练Qwen2.5模型（实验1-3）
  python scripts/training/batch_train_uml_experts.py --model qwen2.5

  # 仅训练Qwen3模型（实验4-6）
  python scripts/training/batch_train_uml_experts.py --model qwen3

  # 单个实验（从1到6）
  python scripts/training/batch_train_uml_experts.py --experiment 1

  # 从某个实验继续（例如从实验3开始）
  python scripts/training/batch_train_uml_experts.py --all --resume-from 3

作者：Training System
日期：2025-02-01
"""

import subprocess
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


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


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}\n")


def print_success(message: str):
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message: str):
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def print_info(message: str):
    print(f"{Colors.CYAN}ℹ {message}{Colors.END}")


def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


# ==================== 实验配置 ====================

class Experiment:
    """单个实验配置"""

    def __init__(self, exp_id: int, model_version: str, dataset_version: str, description: str):
        self.exp_id = exp_id
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.description = description
        self.env = f"uml_{model_version}"  # qwen2.5 -> uml_qwen2.5

    def __str__(self):
        return f"实验{self.exp_id}: {self.model_version} + {self.dataset_version}"


# 定义所有6个实验
ALL_EXPERIMENTS = [
    Experiment(1, "qwen2.5", "qwen2.5", "Qwen2.5-VL + Qwen2.5数据集"),
    Experiment(2, "qwen2.5", "qwen3", "Qwen2.5-VL + Qwen3数据集"),
    Experiment(3, "qwen2.5", "qwen235B", "Qwen2.5-VL + Qwen235B数据集"),
    Experiment(4, "qwen3", "qwen2.5", "Qwen3-VL + Qwen2.5数据集"),
    Experiment(5, "qwen3", "qwen3", "Qwen3-VL + Qwen3数据集"),
    Experiment(6, "qwen3", "qwen235B", "Qwen3-VL + Qwen235B数据集"),
]


# ==================== 训练执行器 ====================

class TrainingExecutor:
    """训练执行器"""

    def __init__(self, log_file: Path = None):
        self.log_file = log_file or PROJECT_ROOT / "outputs" / "batch_training_log.json"
        self.results = []

    def run_experiment(self, exp: Experiment) -> bool:
        """
        执行单个实验

        Returns:
            bool: 是否成功
        """
        print_header(f"实验 {exp.exp_id}/6: {exp.description}")

        print_info(f"模型版本: {exp.model_version}")
        print_info(f"数据集版本: {exp.dataset_version}")
        print_info(f"运行环境: {exp.env}")
        print()

        # 构建训练命令
        cmd = [
            'python',
            str(PROJECT_ROOT / 'scripts' / 'run_with_env.py'),
            '--env', exp.env,
            '--script', str(PROJECT_ROOT / 'scripts' / 'training' / 'train_uml_expert.py'),
            '--',  # 分隔符：告诉run_with_env.py后续参数传给目标脚本
            '--dataset', exp.dataset_version
        ]

        print(f"执行命令: {' '.join(cmd)}")
        print("-" * 80)
        print()

        # 记录开始时间
        start_time = time.time()

        # 执行训练
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
        exp_result = {
            'experiment_id': exp.exp_id,
            'model_version': exp.model_version,
            'dataset_version': exp.dataset_version,
            'description': exp.description,
            'success': success,
            'exit_code': exit_code,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(exp_result)

        # 打印结果
        print("\n" + "=" * 80)
        if success:
            print_success(f"实验{exp.exp_id}完成！耗时: {elapsed_time / 60:.1f}分钟")
        else:
            print_error(f"实验{exp.exp_id}失败！退出码: {exit_code}")
        print("=" * 80 + "\n")

        return success

    def run_batch(self, experiments: List[Experiment], resume_from: int = None) -> bool:
        """
        批量执行实验

        Args:
            experiments: 实验列表
            resume_from: 从哪个实验ID继续

        Returns:
            bool: 是否全部成功
        """
        # 处理resume_from
        if resume_from:
            experiments = [exp for exp in experiments if exp.exp_id >= resume_from]
            print_info(f"从实验{resume_from}继续执行")

        total = len(experiments)
        success_count = 0
        fail_count = 0

        print_header("UML专家批量训练开始")
        print_info(f"总实验数: {total}")
        print_info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 打印实验列表
        print("实验列表:")
        for exp in experiments:
            print(f"  {exp}")
        print()

        # 逐个执行实验
        for i, exp in enumerate(experiments, 1):
            print(f"\n{'#' * 80}")
            print(f"# 进度: [{i}/{total}]")
            print(f"{'#' * 80}\n")

            success = self.run_experiment(exp)

            if success:
                success_count += 1
            else:
                fail_count += 1

                # 询问是否继续
                print_warning("实验执行失败！")
                response = input("是否继续执行下一个实验？(y/n): ").strip().lower()
                if response != 'y':
                    print_info("批量训练中止")
                    break

            # 短暂休息（避免显存未完全释放）
            if i < total:
                print_info("等待5秒后开始下一个实验...")
                time.sleep(5)

        # 打印总结
        self._print_summary(success_count, fail_count, total)

        # 保存结果
        self._save_results()

        return fail_count == 0

    def _print_summary(self, success_count: int, fail_count: int, total: int):
        """打印执行总结"""
        print_header("批量训练总结")

        print(f"总实验数: {total}")
        print_success(f"成功: {success_count}")

        if fail_count > 0:
            print_error(f"失败: {fail_count}")
        else:
            print_info(f"失败: {fail_count}")

        success_rate = (success_count / total * 100) if total > 0 else 0
        print(f"成功率: {success_rate:.1f}%")
        print()

        # 打印各实验结果
        print("各实验结果:")
        print("-" * 80)
        for result in self.results:
            status = "✓" if result['success'] else "✗"
            time_str = f"{result['elapsed_time'] / 60:.1f}分钟"
            print(f"  {status} 实验{result['experiment_id']}: {result['description']} - {time_str}")
        print("-" * 80)
        print()

        # 计算总耗时
        total_time = sum(r['elapsed_time'] for r in self.results)
        print(f"总耗时: {total_time / 60:.1f}分钟 ({total_time / 3600:.2f}小时)")
        print()

    def _save_results(self):
        """保存训练结果"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'success_count': sum(1 for r in self.results if r['success']),
            'fail_count': sum(1 for r in self.results if not r['success']),
            'total_time_seconds': sum(r['elapsed_time'] for r in self.results),
            'results': self.results
        }

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print_info(f"训练报告已保存: {self.log_file}")


# ==================== 命令行接口 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='UML专家批量训练脚本 - 6个对比实验自动化',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 执行模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='执行全部6个实验'
    )
    mode_group.add_argument(
        '--model',
        type=str,
        choices=['qwen2.5', 'qwen3'],
        help='仅训练指定模型的实验（3个实验）'
    )
    mode_group.add_argument(
        '--experiment',
        type=int,
        choices=range(1, 7),
        help='执行单个实验（1-6）'
    )

    parser.add_argument(
        '--resume-from',
        type=int,
        choices=range(1, 7),
        help='从某个实验继续执行（实验ID: 1-6）'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有实验配置'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 列出实验
    if args.list:
        print_header("UML专家训练实验矩阵")
        print("总计6个实验（2个模型 × 3个数据集）:\n")
        for exp in ALL_EXPERIMENTS:
            print(f"{exp}")
        print()
        return

    # 确定要执行的实验
    if args.all:
        experiments = ALL_EXPERIMENTS
    elif args.model:
        experiments = [exp for exp in ALL_EXPERIMENTS if exp.model_version == args.model]
        print_info(f"仅执行{args.model}模型的实验")
    else:  # args.experiment
        experiments = [exp for exp in ALL_EXPERIMENTS if exp.exp_id == args.experiment]
        print_info(f"仅执行实验{args.experiment}")

    # 创建执行器
    executor = TrainingExecutor()

    # 执行批量训练
    success = executor.run_batch(
        experiments=experiments,
        resume_from=args.resume_from
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()