#!/usr/bin/env python
"""
统一批量训练脚本 - General Expert + UML Expert

功能：自动执行3个General Expert + 6个UML Expert训练任务
适用环境：RTX 4090 (24GB显存)

训练矩阵：
┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ 专家 \ 数据集        │ qwen2.5识别       │ qwen3识别         │ qwen235B识别      │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ General Expert      │ 任务1             │ 任务2             │ 任务3             │
│ (qwen_text环境)     │                  │                  │                  │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ UML Expert          │ 任务4 (Qwen2.5)  │ 任务5 (Qwen2.5)  │ 任务6 (Qwen2.5)  │
│ Qwen2.5-VL          │                  │                  │                  │
│ (qwen_vision25)     │                  │                  │                  │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ UML Expert          │ 任务7 (Qwen3)    │ 任务8 (Qwen3)    │ 任务9 (Qwen3)    │
│ Qwen3-VL            │                  │                  │                  │
│ (qwen_vision3)      │                  │                  │                  │
└─────────────────────┴──────────────────┴──────────────────┴──────────────────┘

使用方法：
  # 执行全部9个任务
  python scripts/training/batch_train_all_experts.py --all

  # 仅训练General Expert（任务1-3）
  python scripts/training/batch_train_all_experts.py --expert general

  # 仅训练UML Expert（任务4-9）
  python scripts/training/batch_train_all_experts.py --expert uml

  # 单个任务（从1到9）
  python scripts/training/batch_train_all_experts.py --task 1

  # 从某个任务继续（例如从任务5开始）
  python scripts/training/batch_train_all_experts.py --all --resume-from 5

作者：Training System
日期：2025-02-02
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
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.END}")


def print_error(message: str):
    print(f"{Colors.RED}[ERROR] {message}{Colors.END}")


def print_info(message: str):
    print(f"{Colors.CYAN}[INFO] {message}{Colors.END}")


def print_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING] {message}{Colors.END}")


# ==================== 训练任务配置 ====================

class TrainingTask:
    """单个训练任务配置"""

    def __init__(self, task_id: int, expert_type: str, env_name: str,
                 dataset_version: str, model_version: str, description: str):
        self.task_id = task_id
        self.expert_type = expert_type  # 'general' or 'uml'
        self.env_name = env_name  # 'text', 'uml_qwen2.5', 'uml_qwen3'
        self.dataset_version = dataset_version  # 'qwen2.5', 'qwen3', 'qwen235B'
        self.model_version = model_version  # 用于UML Expert（'qwen2.5' or 'qwen3'）
        self.description = description

    def __str__(self):
        return f"任务{self.task_id}: {self.description}"


# 定义所有9个训练任务
ALL_TASKS = [
    # General Expert - 3个任务（在qwen_text环境中训练）
    TrainingTask(1, "general", "text", "qwen2.5", None,
                 "General Expert + qwen2.5数据集"),
    TrainingTask(2, "general", "text", "qwen3", None,
                 "General Expert + qwen3数据集"),
    TrainingTask(3, "general", "text", "qwen235B", None,
                 "General Expert + qwen235B数据集"),

    # UML Expert (Qwen2.5-VL) - 3个任务（在qwen_vision25环境中训练）
    TrainingTask(4, "uml", "uml_qwen2.5", "qwen2.5", "qwen2.5",
                 "UML Expert (Qwen2.5-VL) + qwen2.5数据集"),
    TrainingTask(5, "uml", "uml_qwen2.5", "qwen3", "qwen2.5",
                 "UML Expert (Qwen2.5-VL) + qwen3数据集"),
    TrainingTask(6, "uml", "uml_qwen2.5", "qwen235B", "qwen2.5",
                 "UML Expert (Qwen2.5-VL) + qwen235B数据集"),

    # UML Expert (Qwen3-VL) - 3个任务（在qwen_vision3环境中训练）
    TrainingTask(7, "uml", "uml_qwen3", "qwen2.5", "qwen3",
                 "UML Expert (Qwen3-VL) + qwen2.5数据集"),
    TrainingTask(8, "uml", "uml_qwen3", "qwen3", "qwen3",
                 "UML Expert (Qwen3-VL) + qwen3数据集"),
    TrainingTask(9, "uml", "uml_qwen3", "qwen235B", "qwen3",
                 "UML Expert (Qwen3-VL) + qwen235B数据集"),
]


# ==================== 训练执行器 ====================

class TrainingExecutor:
    """训练执行器"""

    def __init__(self, log_file: Path = None):
        self.log_file = log_file or PROJECT_ROOT / "outputs" / "batch_training_all_experts_log.json"
        self.results = []

    def run_task(self, task: TrainingTask) -> bool:
        """
        执行单个训练任务

        Returns:
            bool: 是否成功
        """
        print_header(f"任务 {task.task_id}/9: {task.description}")

        print_info(f"专家类型: {task.expert_type}")
        print_info(f"数据集版本: {task.dataset_version}")
        print_info(f"运行环境: {task.env_name}")
        if task.model_version:
            print_info(f"模型版本: {task.model_version}")
        print()

        # 根据专家类型选择训练脚本
        if task.expert_type == "general":
            script_path = PROJECT_ROOT / 'scripts' / 'training' / 'train_general_expert.py'
        else:  # uml
            script_path = PROJECT_ROOT / 'scripts' / 'training' / 'train_uml_expert.py'

        # 构建训练命令
        cmd = [
            'python',
            str(PROJECT_ROOT / 'scripts' / 'run_with_env.py'),
            '--env', task.env_name,
            '--script', str(script_path),
            '--',  # 分隔符：告诉run_with_env.py后续参数传给目标脚本
            '--dataset', task.dataset_version
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
        task_result = {
            'task_id': task.task_id,
            'expert_type': task.expert_type,
            'dataset_version': task.dataset_version,
            'model_version': task.model_version,
            'description': task.description,
            'success': success,
            'exit_code': exit_code,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(task_result)

        # 打印结果
        print("\n" + "=" * 80)
        if success:
            print_success(f"任务{task.task_id}完成！耗时: {elapsed_time / 60:.1f}分钟")
        else:
            print_error(f"任务{task.task_id}失败！退出码: {exit_code}")
        print("=" * 80 + "\n")

        return success

    def run_batch(self, tasks: List[TrainingTask], resume_from: int = None,
                  auto_continue: bool = True) -> bool:
        """
        批量执行训练任务

        Args:
            tasks: 任务列表
            resume_from: 从哪个任务ID继续
            auto_continue: 失败后是否自动继续

        Returns:
            bool: 是否全部成功
        """
        # 处理resume_from
        if resume_from:
            tasks = [t for t in tasks if t.task_id >= resume_from]
            print_info(f"从任务{resume_from}继续执行")

        total = len(tasks)
        success_count = 0
        fail_count = 0

        print_header("批量训练开始")
        print_info(f"总任务数: {total}")
        print_info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 打印任务列表
        print("任务列表:")
        for task in tasks:
            print(f"  {task}")
        print()

        # 逐个执行任务
        for i, task in enumerate(tasks, 1):
            print(f"\n{'#' * 80}")
            print(f"# 进度: [{i}/{total}]")
            print(f"{'#' * 80}\n")

            success = self.run_task(task)

            if success:
                success_count += 1
            else:
                fail_count += 1

                # 根据auto_continue决定是否继续
                if not auto_continue:
                    print_warning("训练失败！")
                    response = input("是否继续执行下一个任务？(y/n): ").strip().lower()
                    if response != 'y':
                        print_info("批量训练中止")
                        break
                else:
                    print_warning("训练失败，但将自动继续下一个任务")

            # 短暂休息（避免显存未完全释放）
            if i < total:
                print_info("等待5秒后开始下一个任务...")
                time.sleep(5)

        # 打印总结
        self._print_summary(success_count, fail_count, total)

        # 保存结果
        self._save_results()

        return fail_count == 0

    def _print_summary(self, success_count: int, fail_count: int, total: int):
        """打印执行总结"""
        print_header("批量训练总结")

        print(f"总任务数: {total}")
        print_success(f"成功: {success_count}")

        if fail_count > 0:
            print_error(f"失败: {fail_count}")
        else:
            print_info(f"失败: {fail_count}")

        success_rate = (success_count / total * 100) if total > 0 else 0
        print(f"成功率: {success_rate:.1f}%")
        print()

        # 打印各任务结果
        print("各任务结果:")
        print("-" * 80)
        for result in self.results:
            status = "[SUCCESS]" if result['success'] else "[FAILED]"
            time_str = f"{result['elapsed_time'] / 60:.1f}分钟"
            print(f"  {status} 任务{result['task_id']}: {result['description']} - {time_str}")
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
            'total_tasks': len(self.results),
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
        description='统一批量训练脚本 - General Expert + UML Expert',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 执行模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='执行全部9个任务'
    )
    mode_group.add_argument(
        '--expert',
        type=str,
        choices=['general', 'uml'],
        help='仅训练指定类型的专家（general=任务1-3, uml=任务4-9）'
    )
    mode_group.add_argument(
        '--task',
        type=int,
        choices=range(1, 10),
        help='执行单个任务（1-9）'
    )

    parser.add_argument(
        '--resume-from',
        type=int,
        choices=range(1, 10),
        help='从某个任务继续执行（任务ID: 1-9）'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有训练任务'
    )

    parser.add_argument(
        '--no-auto-continue',
        action='store_true',
        help='失败后需要手动确认是否继续（默认自动继续）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 列出任务
    if args.list:
        print_header("统一批量训练任务矩阵")
        print("总计9个任务（3个General Expert + 6个UML Expert）:\n")
        for task in ALL_TASKS:
            print(f"{task}")
        print()
        return

    # 确定要执行的任务
    if args.all:
        tasks = ALL_TASKS
    elif args.expert:
        tasks = [t for t in ALL_TASKS if t.expert_type == args.expert]
        print_info(f"仅执行{args.expert}类型的任务")
    else:  # args.task
        tasks = [t for t in ALL_TASKS if t.task_id == args.task]
        print_info(f"仅执行任务{args.task}")

    # 创建执行器
    executor = TrainingExecutor()

    # 执行批量训练
    auto_continue = not args.no_auto_continue
    success = executor.run_batch(
        tasks=tasks,
        resume_from=args.resume_from,
        auto_continue=auto_continue
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()