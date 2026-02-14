"""
One-click Training Script for All Experts

Sequentially executes all training tasks:
  - Session 1: Prompt Tuning (4 experts, ~4 hours)
  - Session 2: P-Tuning v2 (4 experts, ~5 hours)
  - Session 3: Full Fine-tuning (4 experts, ~7 hours)

Total: 12 models, estimated ~16 hours

Usage:
  python scripts/training/train_all_experts.py

  Optional arguments:
    --method {prompt_tuning,p_tuning,full_finetuning,all}
             Train specific method only (default: all)
    --expert {text,image,uml,general,all}
             Train specific expert only (default: all)

Examples:
  # Train all methods and experts
  python scripts/training/train_all_experts.py

  # Train only Prompt Tuning
  python scripts/training/train_all_experts.py --method prompt_tuning

  # Train only Text Expert across all methods
  python scripts/training/train_all_experts.py --expert text

Environment: instruction_generator (transformers==4.57.0)
Author: Training Pipeline System
Date: 2025-02-15
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
    'prompt_tuning': {'text': 1.0, 'image': 0.3, 'uml': 0.75, 'general': 1.9},
    'p_tuning': {'text': 1.3, 'image': 0.4, 'uml': 0.9, 'general': 2.3},
    'full_finetuning': {'text': 1.8, 'image': 0.6, 'uml': 1.25, 'general': 3.0},
}


def print_header():
    """Print training header"""
    print("\n" + "=" * 80)
    print(" " * 20 + "ONE-CLICK TRAINING FOR ALL EXPERTS")
    print("=" * 80)
    print("\nThis script will train 12 models sequentially:")
    print("  - Session 1: Prompt Tuning (4 experts, ~4 hours)")
    print("  - Session 2: P-Tuning v2 (4 experts, ~5 hours)")
    print("  - Session 3: Full Fine-tuning (4 experts, ~7 hours)")
    print("\nTotal estimated time: ~16 hours")
    print("=" * 80 + "\n")


def print_session_header(session_num, method_name, total_time):
    """Print session header"""
    print("\n" + "=" * 80)
    print(f"SESSION {session_num}: {method_name}")
    print(f"Estimated time: {total_time:.1f} hours")
    print("=" * 80 + "\n")


def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def run_training_task(method, expert, script_path):
    """Run a single training task"""
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        print(f"ERROR: Script not found: {full_path}")
        return False

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting: {method}/{expert}")
    print(f"Script: {script_path}")
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {method}/{expert}")
        print(f"Time taken: {format_time(elapsed)}")
        print(f"Status: SUCCESS")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print("-" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed: {method}/{expert}")
        print(f"Time taken: {format_time(elapsed)}")
        print(f"Status: FAILED")
        print(f"Error code: {e.returncode}")

        return False

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n" + "-" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Interrupted: {method}/{expert}")
        print(f"Time taken: {format_time(elapsed)}")
        print(f"Status: INTERRUPTED BY USER")

        raise


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description='One-click training for all experts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/training/train_all_experts.py
  python scripts/training/train_all_experts.py --method prompt_tuning
  python scripts/training/train_all_experts.py --expert text
        """
    )

    parser.add_argument(
        '--method',
        choices=['prompt_tuning', 'p_tuning', 'full_finetuning', 'all'],
        default='all',
        help='Train specific method only (default: all)'
    )

    parser.add_argument(
        '--expert',
        choices=['text', 'image', 'uml', 'general', 'all'],
        default='all',
        help='Train specific expert only (default: all)'
    )

    args = parser.parse_args()

    print_header()

    methods_to_train = (
        list(TRAINING_TASKS.keys()) if args.method == 'all'
        else [args.method]
    )

    experts_to_train = (
        ['text', 'image', 'uml', 'general'] if args.expert == 'all'
        else [args.expert]
    )

    overall_start = time.time()
    results = []

    try:
        session_num = 1
        for method in methods_to_train:
            method_display = {
                'prompt_tuning': 'Prompt Tuning',
                'p_tuning': 'P-Tuning v2',
                'full_finetuning': 'Full Fine-tuning'
            }[method]

            total_method_time = sum(
                ESTIMATED_TIME[method][expert]
                for expert in experts_to_train
            )

            print_session_header(session_num, method_display, total_method_time)
            session_num += 1

            for expert in experts_to_train:
                script_path = TRAINING_TASKS[method][expert]
                success = run_training_task(method, expert, script_path)

                results.append({
                    'method': method,
                    'expert': expert,
                    'success': success
                })

                if not success:
                    print("\n" + "=" * 80)
                    print("TRAINING FAILED!")
                    print("=" * 80)
                    print(f"Failed task: {method}/{expert}")
                    print("Stopping execution.")
                    print("=" * 80 + "\n")
                    return 1

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 80 + "\n")
        return 1

    overall_elapsed = time.time() - overall_start

    print("\n\n" + "=" * 80)
    print(" " * 25 + "TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nTotal time: {format_time(overall_elapsed)}")
    print(f"Tasks completed: {len(results)}/{len(results)}")
    print("\nResults:")
    print("-" * 80)

    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"  {result['method']:20s} / {result['expert']:10s} : {status}")

    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())