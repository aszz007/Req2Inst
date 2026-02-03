#!/usr/bin/env python
"""
修正版批量训练脚本 - 在Qwen-7B-Chat上重新训练所有Image和UML Expert

问题：之前Image和UML Expert错误地在视觉模型(Qwen-VL)上训练
解决：全部重新在Qwen-7B-Chat（文本模型）上训练

需要重新训练的Expert：
1. Image Expert - image_expert (唯一的图像数据集版本)
2. UML Expert - uml_expert_dataset_qwen25 (使用Qwen2.5识别的数据集)
3. UML Expert - uml_expert_dataset_qwen3 (使用Qwen3识别的数据集)
4. UML Expert - uml_expert_dataset_qwen235B (使用Qwen235B识别的数据集)

总计：4个Expert（1个Image + 3个UML）
说明：Image数据集只有1个版本，UML数据集有3个版本（不同视觉模型识别生成）
环境：qwen_text（统一环境，transformers==4.32.0）
基础模型：Qwen-7B-Chat（统一模型，target_modules=["c_attn"]）
预计时间：5-6小时（适合挂一晚上）

使用方法：
  # 训练所有Expert
  conda activate qwen_text
  python scripts/training/batch_retrain_all_experts_correct.py --all

  # 仅训练Image Expert (1个)
  python scripts/training/batch_retrain_all_experts_correct.py --type image

  # 仅训练UML Expert (3个)
  python scripts/training/batch_retrain_all_experts_correct.py --type uml

  # 从某个实验继续（例如从实验2开始）
  python scripts/training/batch_retrain_all_experts_correct.py --all --resume-from 2

作者: Training System
日期: 2025-02-04
"""

import subprocess
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

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

class ExpertConfig:
    """单个Expert训练配置"""

    def __init__(self, exp_id: int, expert_type: str, dataset_version: str,
                 output_name: str, description: str):
        """
        Args:
            exp_id: 实验ID
            expert_type: 专家类型 ('image' or 'uml')
            dataset_version: 数据集版本
            output_name: 输出目录名称
            description: 描述
        """
        self.exp_id = exp_id
        self.expert_type = expert_type
        self.dataset_version = dataset_version
        self.output_name = output_name
        self.description = description

    def __str__(self):
        return f"实验{self.exp_id}: {self.description}"


# 定义所有需要重新训练的Expert（1个Image + 3个UML）
ALL_EXPERTS = [
    # Image Expert (1个 - 只有1个图像数据集版本)
    ExpertConfig(
        exp_id=1,
        expert_type='image',
        dataset_version=None,  # Image Expert不需要指定数据集版本
        output_name='image_expert',
        description='Image Expert'
    ),

    # UML Expert (3个 - 3个数据集版本)
    ExpertConfig(
        exp_id=2,
        expert_type='uml',
        dataset_version='qwen2.5',
        output_name='uml_expert_dataset_qwen25',
        description='UML Expert (Qwen2.5数据集)'
    ),
    ExpertConfig(
        exp_id=3,
        expert_type='uml',
        dataset_version='qwen3',
        output_name='uml_expert_dataset_qwen3',
        description='UML Expert (Qwen3数据集)'
    ),
    ExpertConfig(
        exp_id=4,
        expert_type='uml',
        dataset_version='qwen235B',
        output_name='uml_expert_dataset_qwen235B',
        description='UML Expert (Qwen235B数据集)'
    ),
]


# ==================== 训练执行器 ====================

class TrainingExecutor:
    """训练执行器"""

    def __init__(self, log_file: Path = None):
        self.log_file = log_file or PROJECT_ROOT / "outputs" / "retrain_experts_log.json"
        self.results = []

    def _create_training_script(self, config: ExpertConfig, temp_dir: Path) -> Path:
        """
        创建临时训练脚本（修正版，强制使用Qwen-7B-Chat）

        Returns:
            Path: 临时脚本路径
        """
        script_content = f'''#!/usr/bin/env python
"""
临时训练脚本 - 实验{config.exp_id}: {config.description}
自动生成，用于修正训练
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.expert_trainer import ExpertTrainer
from config.settings import get_path_config
from src.utils.logger import get_logger

logger = get_logger('training.retrain_{config.expert_type}')

def main():
    path_cfg = get_path_config()

    # 关键修改：强制使用Qwen-7B-Chat模型路径
    base_model_path = str(path_cfg.QWEN_7B_CHAT_PATH)

    # 设置正确的输出目录
    output_dir = path_cfg.LORA_WEIGHTS_DIR / 'experts' / '{config.output_name}'

    logger.info("=" * 80)
    logger.info(f"实验{config.exp_id}: {config.description}")
    logger.info("=" * 80)
    logger.info(f"专家类型: {config.expert_type}")
    logger.info(f"数据集版本: {config.dataset_version}")
    logger.info(f"基础模型: {{base_model_path}}")
    logger.info(f"输出目录: {{output_dir}}")
    logger.info("⚠️ 使用Qwen-7B-Chat（文本模型），不是视觉模型")
    logger.info("=" * 80)

    # 创建训练器（强制指定base_model_path）
    trainer = ExpertTrainer(
        expert_type='{config.expert_type}',
        base_model_path=base_model_path,  # 强制使用Qwen-7B-Chat
        output_dir=str(output_dir),
        use_4bit=True,
        dataset_version='{config.dataset_version}' if '{config.expert_type}' == 'uml' else None,
        use_rtx4090_optimization=True
    )

    # 准备数据
    if not trainer.prepare_data():
        logger.error("数据准备失败")
        return 1

    # 设置模型
    if not trainer.setup_model():
        logger.error("模型设置失败")
        return 1

    # 开始训练
    logger.info("开始训练...")
    success = trainer.train()

    if success:
        logger.info("训练成功完成！")
        logger.info(f"LoRA权重已保存至: {{output_dir}}")
        return 0
    else:
        logger.error("训练失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

        # 创建临时脚本文件
        temp_dir.mkdir(parents=True, exist_ok=True)
        script_path = temp_dir / f"train_exp{config.exp_id}_{config.expert_type}.py"

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        return script_path

    def run_experiment(self, config: ExpertConfig) -> bool:
        """
        执行单个实验

        Returns:
            bool: 是否成功
        """
        print_header(f"实验 {config.exp_id}/4: {config.description}")

        print_info(f"专家类型: {config.expert_type}")
        print_info(f"数据集版本: {config.dataset_version}")
        print_info(f"输出名称: {config.output_name}")
        print_warning("使用Qwen-7B-Chat（文本模型），不是视觉模型")
        print()

        # 创建临时训练脚本
        temp_dir = PROJECT_ROOT / "temp_training_scripts"
        script_path = self._create_training_script(config, temp_dir)

        print_info(f"生成临时脚本: {script_path}")

        # 构建执行命令（直接在qwen_text环境中运行）
        cmd = ['python', str(script_path)]

        print(f"执行命令: {' '.join(cmd)}")
        print("-" * 80)
        print()

        # 记录开始时间
        start_time = time.time()

        # 执行训练
        try:
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
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
            'experiment_id': config.exp_id,
            'expert_type': config.expert_type,
            'dataset_version': config.dataset_version,
            'output_name': config.output_name,
            'description': config.description,
            'success': success,
            'exit_code': exit_code,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(exp_result)

        # 打印结果
        print("\n" + "=" * 80)
        if success:
            print_success(f"实验{config.exp_id}完成！耗时: {elapsed_time / 60:.1f}分钟")
        else:
            print_error(f"实验{config.exp_id}失败！退出码: {exit_code}")
        print("=" * 80 + "\n")

        return success

    def run_batch(self, experts: List[ExpertConfig], resume_from: int = None) -> bool:
        """
        批量执行实验

        Args:
            experts: 实验列表
            resume_from: 从哪个实验ID继续

        Returns:
            bool: 是否全部成功
        """
        # 处理resume_from
        if resume_from:
            experts = [exp for exp in experts if exp.exp_id >= resume_from]
            print_info(f"从实验{resume_from}继续执行")

        total = len(experts)
        success_count = 0
        fail_count = 0

        print_header("修正版Expert批量重训练开始")
        print_warning("⚠️ 重要：所有Expert都将在Qwen-7B-Chat（文本模型）上训练")
        print_warning("⚠️ 不再使用视觉模型（Qwen-VL）训练Expert")
        print_info(f"总实验数: {total}")
        print_info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"运行环境: qwen_text")
        print()

        # 打印实验列表
        print("实验列表:")
        for exp in experts:
            print(f"  {exp}")
        print()

        # 确认继续
        print_warning("请确认已激活qwen_text环境：conda activate qwen_text")
        response = input("是否继续？(y/n): ").strip().lower()
        if response != 'y':
            print_info("已取消")
            return False
        print()

        # 逐个执行实验
        for i, exp in enumerate(experts, 1):
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
        print_header("批量重训练总结")

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

        # 给出后续建议
        if success_count > 0:
            print_header("后续步骤")
            print("✓ Expert重新训练完成，现在可以：")
            print("  1. 验证adapter_config.json中target_modules是['c_attn']")
            print("  2. 测试Expert生成指令的质量")
            print("  3. 对比新旧Expert的性能差异")
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
            'note': 'Retrained Image Expert (1) and UML Experts (3) on Qwen-7B-Chat (text model)',
            'architecture_fix': 'All Experts now use text model instead of vision models',
            'results': self.results
        }

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print_info(f"训练报告已保存: {self.log_file}")


# ==================== 命令行接口 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='修正版Expert批量重训练 - 在Qwen-7B-Chat上训练所有Expert',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 执行模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='训练所有Expert（4个）'
    )
    mode_group.add_argument(
        '--type',
        type=str,
        choices=['image', 'uml'],
        help='仅训练指定类型的Expert'
    )
    mode_group.add_argument(
        '--experiment',
        type=int,
        choices=range(1, 5),
        help='执行单个实验（1-4）'
    )

    parser.add_argument(
        '--resume-from',
        type=int,
        choices=range(1, 5),
        help='从某个实验继续执行（实验ID: 1-4）'
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
        print_header("修正版Expert重训练实验列表")
        print("总计4个实验（1个Image + 3个UML）:\n")
        print("⚠️ 重要：所有Expert都将在Qwen-7B-Chat（文本模型）上训练\n")
        for exp in ALL_EXPERTS:
            print(f"{exp}")
        print()
        return

    # 确定要执行的实验
    if args.all:
        experts = ALL_EXPERTS
    elif args.type:
        experts = [exp for exp in ALL_EXPERTS if exp.expert_type == args.type]
        print_info(f"仅执行{args.type}类型的实验")
    else:  # args.experiment
        experts = [exp for exp in ALL_EXPERTS if exp.exp_id == args.experiment]
        print_info(f"仅执行实验{args.experiment}")

    # 创建执行器
    executor = TrainingExecutor()

    # 执行批量训练
    success = executor.run_batch(
        experts=experts,
        resume_from=args.resume_from
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()