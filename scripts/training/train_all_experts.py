"""
一键训练所有专家脚本

按顺序执行所有训练任务：
  - 第1轮：Prompt Tuning (4个专家，约4小时)
  - 第2轮：P-Tuning v2 (4个专家，约5小时)
  - 第3轮：准全参数微调 (4个专家，约7小时)

总计：12个模型，预计约16小时

使用方法：
  python scripts/training/train_all_experts.py

  可选参数：
    --method {prompt_tuning,p_tuning,full_finetuning,all} [...]
             训练指定方法，可指定多个（默认：all）
    --expert {text,image,uml,general,all}
             仅训练指定专家（默认：all）

示例：
  # 训练所有方法和专家
  python scripts/training/train_all_experts.py

  # 仅训练Prompt Tuning
  python scripts/training/train_all_experts.py --method prompt_tuning

  # 训练P-Tuning v2和准全参数微调（跳过Prompt Tuning）
  python scripts/training/train_all_experts.py --method p_tuning full_finetuning --skip-failed

  # 仅训练文本专家（所有方法）
  python scripts/training/train_all_experts.py --expert text

环境：instruction_generator (transformers==4.57.0)
作者：Training Pipeline System
日期：2025-02-15
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import time


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


TRAINING_TASKS = {
    'lora_moe': {
        # 'text': 'scripts/training/lora_moe/train_text_expert.py',  # 已完成，跳过
        'image': 'scripts/training/lora_moe/train_image_expert.py',
        'uml': 'scripts/training/lora_moe/train_uml_expert.py',
        'general': 'scripts/training/lora_moe/train_general_expert.py',
    },
    'prompt_tuning': {
        'text': 'scripts/training/prompt_tuning/train_text_expert.py',
        'image': 'scripts/training/prompt_tuning/train_image_expert.py',
        'uml': 'scripts/training/prompt_tuning/train_uml_expert.py',
        'general': 'scripts/training/prompt_tuning/train_general_expert.py',
    },
    'p_tuning': {
        'text': 'scripts/training/p_tuning/train_text_expert.py',
        'image': 'scripts/training/p_tuning/train_image_expert.py',
        'uml': 'scripts/training/p_tuning/train_uml_expert.py',
        'general': 'scripts/training/p_tuning/train_general_expert.py',
    },
    'full_finetuning': {
        'text': 'scripts/training/full_finetuning/train_text_expert.py',
        'image': 'scripts/training/full_finetuning/train_image_expert.py',
        'uml': 'scripts/training/full_finetuning/train_uml_expert.py',
        'general': 'scripts/training/full_finetuning/train_general_expert.py',
    }
}


ESTIMATED_TIME = {
    'lora_moe': {'text': 0.8, 'image': 0.2, 'uml': 0.5, 'general': 1.5},
    'prompt_tuning': {'text': 1.0, 'image': 0.3, 'uml': 0.75, 'general': 1.9},
    'p_tuning': {'text': 1.3, 'image': 0.4, 'uml': 0.9, 'general': 2.3},
    'full_finetuning': {'text': 1.8, 'image': 0.6, 'uml': 1.25, 'general': 3.0},
}


def print_header():
    """打印训练标题"""
    print("\n" + "=" * 80)
    print(" " * 22 + "一键训练所有专家")
    print("=" * 80)
    print("\n本脚本将按顺序训练16个模型：")
    print("  - 第1轮：LoRA-MoE (4个专家，约3小时)")
    print("  - 第2轮：Prompt Tuning (4个专家，约4小时)")
    print("  - 第3轮：P-Tuning v2 (4个专家，约5小时)")
    print("  - 第4轮：准全参数微调 (4个专家，约7小时)")
    print("\n总预计时间：约19小时")
    print("=" * 80 + "\n")


def print_session_header(session_num, method_name, total_time):
    """打印训练轮次标题"""
    print("\n" + "=" * 80)
    print(f"第{session_num}轮：{method_name}")
    print(f"预计耗时：{total_time:.1f}小时")
    print("=" * 80 + "\n")


def format_time(seconds):
    """将秒数格式化为可读的时间字符串"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}小时{minutes}分{secs}秒"
    elif minutes > 0:
        return f"{minutes}分{secs}秒"
    else:
        return f"{secs}秒"


def run_training_task(method, expert, script_path):
    """执行单个训练任务"""
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        print(f"错误：脚本未找到: {full_path}")
        return False

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始训练: {method}/{expert}")
    print(f"脚本: {script_path}")
    print("-" * 80)

    start_time = time.time()

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True,
            capture_output=False
        )

        elapsed = time.time() - start_time
        print("-" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成: {method}/{expert}")
        print(f"耗时: {format_time(elapsed)}")
        print(f"状态: 成功")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print("-" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 失败: {method}/{expert}")
        print(f"耗时: {format_time(elapsed)}")
        print(f"状态: 失败")
        print(f"错误码: {e.returncode}")

        return False

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n" + "-" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 中断: {method}/{expert}")
        print(f"耗时: {format_time(elapsed)}")
        print(f"状态: 用户中断")

        raise


def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(
        description='一键训练所有专家',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 训练所有方法和专家
  python scripts/training/train_all_experts.py
  
  # 仅训练Prompt Tuning
  python scripts/training/train_all_experts.py --method prompt_tuning
  
  # 训练P-Tuning v2和准全参数微调（跳过Prompt Tuning）
  python scripts/training/train_all_experts.py --method p_tuning full_finetuning --skip-failed
  
  # 仅训练文本专家（所有方法）
  python scripts/training/train_all_experts.py --expert text
  
  # 自动跳过失败的任务继续训练
  python scripts/training/train_all_experts.py --skip-failed
        """
    )

    parser.add_argument(
        '--method',
        nargs='+',
        choices=['lora_moe', 'prompt_tuning', 'p_tuning', 'full_finetuning', 'all'],
        default=['all'],
        help='训练指定方法，可指定多个（默认：all）'
    )

    parser.add_argument(
        '--expert',
        choices=['text', 'image', 'uml', 'general', 'all'],
        default='all',
        help='仅训练指定专家（默认：all）'
    )

    parser.add_argument(
        '--skip-failed',
        action='store_true',
        default=False,
        help='自动跳过失败的任务继续训练（默认：失败后停止）'
    )

    args = parser.parse_args()

    print_header()

    methods_to_train = (
        list(TRAINING_TASKS.keys()) if 'all' in args.method
        else args.method
    )

    experts_to_train = (
        ['text', 'image', 'uml', 'general'] if args.expert == 'all'
        else [args.expert]
    )

    overall_start = time.time()
    results = []
    failed_tasks = []  # 记录失败的任务

    try:
        session_num = 1
        for method in methods_to_train:
            method_display = {
                'lora_moe': 'LoRA-MoE（混合专家微调）',
                'prompt_tuning': 'Prompt Tuning（软提示）',
                'p_tuning': 'P-Tuning v2（前缀微调）',
                'full_finetuning': '准全参数微调'
            }[method]

            total_method_time = sum(
                ESTIMATED_TIME[method].get(expert, 0)
                for expert in experts_to_train
                if expert in TRAINING_TASKS[method]
            )

            print_session_header(session_num, method_display, total_method_time)
            session_num += 1

            for expert in experts_to_train:
                # 检查任务是否存在（支持手动注释跳过）
                if expert not in TRAINING_TASKS[method]:
                    print(f"\n跳过: {method}/{expert} (已注释)")
                    continue

                script_path = TRAINING_TASKS[method][expert]
                success = run_training_task(method, expert, script_path)

                results.append({
                    'method': method,
                    'expert': expert,
                    'success': success
                })

                if not success:
                    failed_tasks.append((method, expert))

                    if args.skip_failed:
                        print("\n" + "=" * 80)
                        print("任务失败，跳过并继续训练下一个任务")
                        print("=" * 80)
                        print(f"失败任务: {method}/{expert}")
                        print("继续执行...")
                        print("=" * 80 + "\n")
                        continue
                    else:
                        print("\n" + "=" * 80)
                        print("训练失败！")
                        print("=" * 80)
                        print(f"失败任务: {method}/{expert}")
                        print("停止执行。")
                        print("提示: 使用 --skip-failed 参数可自动跳过失败任务")
                        print("=" * 80 + "\n")
                        return 1

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("训练被用户中断")
        print("=" * 80 + "\n")
        return 1

    overall_elapsed = time.time() - overall_start

    print("\n\n" + "=" * 80)
    if failed_tasks:
        print(" " * 25 + "训练完成（有失败任务）")
    else:
        print(" " * 28 + "训练完成！")
    print("=" * 80)
    print(f"\n总耗时: {format_time(overall_elapsed)}")

    success_count = sum(1 for r in results if r['success'])
    print(f"成功任务: {success_count}/{len(results)}")

    if failed_tasks:
        print(f"失败任务: {len(failed_tasks)}/{len(results)}")

    print("\n结果：")
    print("-" * 80)

    for result in results:
        status = "✓ 成功" if result['success'] else "✗ 失败"
        print(f"  {result['method']:20s} / {result['expert']:10s} : {status}")

    if failed_tasks:
        print("\n" + "=" * 80)
        print("失败任务详情:")
        print("=" * 80)
        for method, expert in failed_tasks:
            print(f"  - {method}/{expert}")
        print("\n建议:")
        print("  1. 检查对应专家的训练日志: logs/training/")
        print("  2. 如果是OOM错误，考虑进一步降低序列长度或batch size")
        print("  3. 如果是配置错误，检查 config/settings.py")
        print("  4. 可以单独重新训练失败的专家")

    print("=" * 80 + "\n")

    return 0 if not failed_tasks or args.skip_failed else 1


if __name__ == "__main__":
    sys.exit(main())