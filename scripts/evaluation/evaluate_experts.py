"""
Evaluate Experts - Comprehensive Expert Performance Evaluation
专家评估脚本 - 全面的专家性能评估

功能:
  - 评估各个专家的性能(BLEU, ROUGE, METEOR, BERTScore)
  - 评估指令格式质量
  - 生成详细的评估报告
  - 支持单个专家或批量评估

环境要求: qwen_text
运行方式: python scripts/run_with_env.py --env text --script scripts/evaluation/evaluate_experts.py

作者: Evaluation System
日期: 2025-02-06
"""

import sys
import json
import argparse
import gc
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_path_config
from src.utils.enhanced_metrics import EnhancedMetrics
from src.instruction_generation.quality_validator import QualityValidator
from src.training.data_loader import (
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,
    split_dataset_for_expert
)
from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert
from src.utils.logger import get_logger

logger = get_logger('evaluation.evaluate_experts')


class ExpertEvaluator:
    """专家评估器"""

    def __init__(
            self,
            use_bertscore: bool = False,
            strict_validation: bool = False
    ):
        """
        初始化评估器

        Args:
            use_bertscore: 是否使用BERTScore(计算较慢)
            strict_validation: 是否使用严格的格式验证
        """
        self.metrics = EnhancedMetrics(use_bertscore=use_bertscore)
        self.validator = QualityValidator(strict_mode=strict_validation)
        self.path_cfg = get_path_config()
        self.show_samples = False  # 是否显示样本数据

        logger.info("专家评估器初始化完成")
        logger.info(f"使用BERTScore: {use_bertscore}")
        logger.info(f"严格验证模式: {strict_validation}")

    def _force_cleanup_gpu(self):
        """
        强制清理GPU显存

        在每个专家评估完成后调用，确保显存被完全释放
        """
        import torch

        logger.info("强制清理GPU显存...")

        # 多次调用gc和cuda清理
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # 短暂延迟，让GPU有时间释放资源
        time.sleep(2)

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU显存状态 - 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")


    def _display_samples(self, test_data: List[Dict], expert_type: str, num_display: int = 5):
        """
        显示测试数据样本

        Args:
            test_data: 测试数据列表
            expert_type: 专家类型(用于日志)
            num_display: 显示的样本数量
        """
        if not self.show_samples:
            return

        logger.info("=" * 80)
        logger.info(f"[{expert_type}] 测试数据样本预览 (前{num_display}条)")
        logger.info("=" * 80)

        for i in range(min(num_display, len(test_data))):
            input_text = test_data[i]['input']
            # 截断过长的输入
            if len(input_text) > 100:
                input_text = input_text[:100] + "..."
            logger.info(f"样本 {i+1}: {input_text}")

        logger.info("=" * 80)

    def evaluate_text_expert(
            self,
            num_samples: Optional[int] = None
    ) -> Dict:
        """
        评估文本专家

        Args:
            num_samples: 使用的样本数(None表示全部)

        Returns:
            dict: 评估结果
        """
        logger.info("=" * 80)
        logger.info("评估文本专家")
        logger.info("=" * 80)

        # 加载数据集
        loader = TextDatasetLoader()
        data = loader.load_csv_files()
        _, _, test_data = split_dataset_for_expert(data, 'text')

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"测试样本数: {len(test_data)}")

        # 显示样本数据
        self._display_samples(test_data, "Text Expert")

        # 加载专家
        expert = TextExpert()
        if not expert.load_model():
            logger.error("文本专家加载失败")
            return {}

        # 生成预测
        predictions = []
        references = []

        for i, item in enumerate(test_data, 1):
            logger.info(f"生成 {i}/{len(test_data)}")

            try:
                pred = expert.generate_instruction(item['input'])
                predictions.append(pred)
                references.append(item['output'])
            except Exception as e:
                logger.error(f"生成失败: {e}")
                predictions.append("")
                references.append(item['output'])

        # 卸载模型
        expert.unload_model()

        # 强制清理GPU显存
        del expert
        self._force_cleanup_gpu()

        # 评估
        results = self._evaluate_predictions(
            predictions=predictions,
            references=references,
            expert_name='text_expert'
        )

        return results

    def evaluate_image_expert(
            self,
            num_samples: Optional[int] = None
    ) -> Dict:
        """
        评估图像专家

        Args:
            num_samples: 使用的样本数

        Returns:
            dict: 评估结果
        """
        logger.info("=" * 80)
        logger.info("评估图像专家")
        logger.info("=" * 80)

        # 加载数据集
        loader = ImageDatasetLoader()
        data = loader.load_csv_file()
        _, _, test_data = split_dataset_for_expert(data, 'image')

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"测试样本数: {len(test_data)}")

        # 显示样本数据
        self._display_samples(test_data, "Image Expert")

        # 加载专家
        expert = ImageExpert()
        if not expert.load_model():
            logger.error("图像专家加载失败")
            return {}

        # 生成预测
        predictions = []
        references = []

        for i, item in enumerate(test_data, 1):
            logger.info(f"生成 {i}/{len(test_data)}")

            try:
                pred = expert.generate_instruction(item['input'])
                predictions.append(pred)
                references.append(item['output'])
            except Exception as e:
                logger.error(f"生成失败: {e}")
                predictions.append("")
                references.append(item['output'])

        # 卸载模型
        expert.unload_model()

        # 强制清理GPU显存
        del expert
        self._force_cleanup_gpu()

        # 评估
        results = self._evaluate_predictions(
            predictions=predictions,
            references=references,
            expert_name='image_expert'
        )

        return results

    def evaluate_uml_expert(
            self,
            dataset_version: str = 'qwen235B',
            num_samples: Optional[int] = None
    ) -> Dict:
        """
        评估UML专家

        Args:
            dataset_version: 数据集版本
            num_samples: 使用的样本数

        Returns:
            dict: 评估结果
        """
        logger.info("=" * 80)
        logger.info(f"评估UML专家 - 数据集版本: {dataset_version}")
        logger.info("=" * 80)

        # 加载数据集
        loader = UMLDatasetLoader(dataset_version=dataset_version)
        data = loader.load_csv_file()
        _, _, test_data = split_dataset_for_expert(data, 'uml')

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"测试样本数: {len(test_data)}")

        # 显示样本数据
        self._display_samples(test_data, f"UML Expert ({dataset_version})")

        # 加载专家
        expert = UMLExpert(dataset_version=dataset_version)
        if not expert.load_model():
            logger.error("UML专家加载失败")
            return {}

        # 生成预测
        predictions = []
        references = []

        for i, item in enumerate(test_data, 1):
            logger.info(f"生成 {i}/{len(test_data)}")

            try:
                pred = expert.generate_instruction(item['input'])
                predictions.append(pred)
                references.append(item['output'])
            except Exception as e:
                logger.error(f"生成失败: {e}")
                predictions.append("")
                references.append(item['output'])

        # 卸载模型
        expert.unload_model()

        # 强制清理GPU显存
        del expert
        self._force_cleanup_gpu()

        # 评估
        results = self._evaluate_predictions(
            predictions=predictions,
            references=references,
            expert_name=f'uml_expert_dataset_{dataset_version}'
        )

        return results

    def evaluate_general_expert(
            self,
            dataset_version: str = 'qwen235B',
            num_samples: Optional[int] = None
    ) -> Dict:
        """
        评估通用专家

        Args:
            dataset_version: 数据集版本
            num_samples: 使用的样本数

        Returns:
            dict: 评估结果
        """
        logger.info("=" * 80)
        logger.info(f"评估通用专家 - 数据集版本: {dataset_version}")
        logger.info("=" * 80)

        # 加载三种数据集: text + image + uml
        # 1. 加载文本数据集
        logger.info("加载文本数据集...")
        text_loader = TextDatasetLoader()
        text_data = text_loader.load_csv_files()
        _, _, text_test = split_dataset_for_expert(text_data, 'text')
        logger.info(f"文本数据集测试集: {len(text_test)}条")

        # 2. 加载图像数据集
        logger.info("加载图像数据集...")
        image_loader = ImageDatasetLoader()
        image_data = image_loader.load_csv_file()
        _, _, image_test = split_dataset_for_expert(image_data, 'image')
        logger.info(f"图像数据集测试集: {len(image_test)}条")

        # 3. 加载对应版本的UML数据集
        logger.info(f"加载UML数据集 (版本: {dataset_version})...")
        uml_loader = UMLDatasetLoader(dataset_version=dataset_version)
        uml_data = uml_loader.load_csv_file()
        _, _, uml_test = split_dataset_for_expert(uml_data, 'uml')
        logger.info(f"UML数据集测试集: {len(uml_test)}条")

        # 4. 合并三种数据集的测试集
        test_data = text_test + image_test + uml_test
        logger.info(f"合并后总测试集: {len(test_data)}条 (Text: {len(text_test)} + Image: {len(image_test)} + UML: {len(uml_test)})")

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"实际使用测试样本数: {len(test_data)}")

        # 显示样本数据
        self._display_samples(test_data, f"General Expert ({dataset_version})")

        # 加载专家
        expert = GeneralExpert(dataset_version=dataset_version)
        if not expert.load_model():
            logger.error("通用专家加载失败")
            return {}

        # 生成预测
        predictions = []
        references = []

        for i, item in enumerate(test_data, 1):
            logger.info(f"生成 {i}/{len(test_data)}")

            try:
                pred = expert.generate_instruction(item['input'])
                predictions.append(pred)
                references.append(item['output'])
            except Exception as e:
                logger.error(f"生成失败: {e}")
                predictions.append("")
                references.append(item['output'])

        # 卸载模型
        expert.unload_model()

        # 强制清理GPU显存
        del expert
        self._force_cleanup_gpu()

        # 评估
        results = self._evaluate_predictions(
            predictions=predictions,
            references=references,
            expert_name=f'general_expert_dataset_{dataset_version}'
        )

        return results

    def _evaluate_predictions(
            self,
            predictions: List[str],
            references: List[str],
            expert_name: str
    ) -> Dict:
        """
        评估预测结果

        Args:
            predictions: 预测列表
            references: 参考列表
            expert_name: 专家名称

        Returns:
            dict: 评估结果
        """
        logger.info("开始评估指标计算...")

        # 过滤空预测
        valid_pairs = [
            (pred, ref) for pred, ref in zip(predictions, references)
            if pred.strip()
        ]

        if not valid_pairs:
            logger.error("没有有效的预测结果")
            return {}

        valid_predictions = [pair[0] for pair in valid_pairs]
        valid_references = [pair[1] for pair in valid_pairs]

        logger.info(f"有效样本数: {len(valid_predictions)}/{len(predictions)}")

        # 生成质量指标
        quality_metrics = self.metrics.calculate_generation_quality(
            predictions=valid_predictions,
            references=valid_references
        )

        # 格式指标
        format_metrics = self.metrics.calculate_format_metrics(
            instructions=valid_predictions
        )

        # 统计指标
        statistical_metrics = self.metrics.calculate_statistical_metrics(
            instructions=valid_predictions
        )

        # 质量验证
        validation_results, validation_summary = self.validator.batch_validate(
            instructions=valid_predictions
        )

        # 组合结果
        results = {
            'expert_name': expert_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(predictions),
            'valid_samples': len(valid_predictions),
            'generation_quality': quality_metrics,
            'format_metrics': format_metrics,
            'statistical_metrics': statistical_metrics,
            'validation_summary': validation_summary
        }

        return results

    def evaluate_all_experts(
            self,
            num_samples: Optional[int] = None,
            save_dir: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        评估所有专家

        Args:
            num_samples: 每个专家使用的样本数
            save_dir: 保存目录

        Returns:
            dict: 所有专家的评估结果
        """
        logger.info("=" * 80)
        logger.info("评估所有专家")
        logger.info("=" * 80)

        all_results = {}

        # 文本专家
        try:
            all_results['text_expert'] = self.evaluate_text_expert(num_samples)
        except Exception as e:
            logger.error(f"文本专家评估失败: {e}")
            # 即使失败也要清理GPU显存
            self._force_cleanup_gpu()

        # 图像专家
        try:
            all_results['image_expert'] = self.evaluate_image_expert(num_samples)
        except Exception as e:
            logger.error(f"图像专家评估失败: {e}")
            # 即使失败也要清理GPU显存
            self._force_cleanup_gpu()

        # UML专家(默认版本)
        try:
            all_results['uml_expert'] = self.evaluate_uml_expert('qwen235B', num_samples)
        except Exception as e:
            logger.error(f"UML专家评估失败: {e}")
            # 即使失败也要清理GPU显存
            self._force_cleanup_gpu()

        # 通用专家(默认版本)
        try:
            all_results['general_expert'] = self.evaluate_general_expert('qwen235B', num_samples)
        except Exception as e:
            logger.error(f"通用专家评估失败: {e}")
            # 即使失败也要清理GPU显存
            self._force_cleanup_gpu()

        # 保存结果
        if save_dir:
            self._save_all_results(all_results, save_dir)

        # 打印摘要
        self._print_comparison_summary(all_results)

        return all_results

    def _save_all_results(self, results: Dict, save_dir: str):
        """保存所有评估结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存完整结果
        full_path = save_dir / f'evaluation_results_{timestamp}.json'
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"评估结果已保存至: {full_path}")

        # 保存摘要
        summary = self._create_summary(results)
        summary_path = save_dir / f'evaluation_summary_{timestamp}.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"评估摘要已保存至: {summary_path}")

    def _create_summary(self, results: Dict) -> Dict:
        """创建评估摘要"""
        summary = {}

        for expert_name, result in results.items():
            if not result:
                continue

            summary[expert_name] = {
                'bleu': result['generation_quality'].get('bleu', 0),
                'rouge_l': result['generation_quality'].get('rougeL', 0),
                'meteor': result['generation_quality'].get('meteor', 0),
                'format_score': result['format_metrics'].get('avg_format_score', 0),
                'valid_rate': result['format_metrics'].get('valid_rate', 0),
                'avg_length': result['statistical_metrics']['char_length'].get('mean', 0)
            }

        return summary

    def _print_comparison_summary(self, results: Dict):
        """打印对比摘要"""
        print("\n" + "=" * 80)
        print("专家评估对比摘要")
        print("=" * 80)

        print(f"\n{'专家':<30} {'BLEU':<10} {'ROUGE-L':<10} {'METEOR':<10} {'格式分数':<10} {'通过率':<10}")
        print("-" * 80)

        for expert_name, result in results.items():
            if not result:
                continue

            bleu = result['generation_quality'].get('bleu', 0)
            rouge_l = result['generation_quality'].get('rougeL', 0)
            meteor = result['generation_quality'].get('meteor', 0)
            format_score = result['format_metrics'].get('avg_format_score', 0)
            valid_rate = result['format_metrics'].get('valid_rate', 0)

            print(
                f"{expert_name:<30} {bleu:<10.4f} {rouge_l:<10.4f} {meteor:<10.4f} {format_score:<10.4f} {valid_rate:<10.2%}")

        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='评估专家性能')
    parser.add_argument('--expert', type=str, choices=['text', 'image', 'uml', 'general', 'all'],
                        default='all', help='要评估的专家')
    parser.add_argument('--dataset-version', type=str, default='qwen235B',
                        choices=['qwen2.5', 'qwen3', 'qwen235B'],
                        help='数据集版本(用于UML/General专家)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='使用的样本数(None表示全部)')
    parser.add_argument('--test-mode', action='store_true',
                        help='测试模式:每个数据集只使用10条数据快速验证流程')
    parser.add_argument('--show-samples', action='store_true',
                        help='显示测试数据样本(前5条)')
    parser.add_argument('--use-bertscore', action='store_true',
                        help='使用BERTScore(计算较慢)')
    parser.add_argument('--strict-validation', action='store_true',
                        help='使用严格的格式验证')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='保存目录')

    args = parser.parse_args()

    # 测试模式：自动设置num_samples=10
    if args.test_mode:
        if args.num_samples is None:
            args.num_samples = 10
            logger.info("=" * 80)
            logger.info("测试模式已启用 - 每个数据集使用10条数据")
            logger.info("=" * 80)
        else:
            logger.warning(f"测试模式已启用，但--num-samples已设置为{args.num_samples}，将使用该值")

    # 创建评估器
    evaluator = ExpertEvaluator(
        use_bertscore=args.use_bertscore,
        strict_validation=args.strict_validation
    )

    # 设置是否显示样本
    evaluator.show_samples = args.show_samples

    # 设置保存目录
    if args.save_dir is None:
        path_cfg = get_path_config()
        args.save_dir = str(path_cfg.METRICS_DIR)

    # 执行评估
    if args.expert == 'all':
        results = evaluator.evaluate_all_experts(
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )
    elif args.expert == 'text':
        results = evaluator.evaluate_text_expert(num_samples=args.num_samples)
        if args.save_dir:
            evaluator._save_all_results({'text_expert': results}, args.save_dir)
    elif args.expert == 'image':
        results = evaluator.evaluate_image_expert(num_samples=args.num_samples)
        if args.save_dir:
            evaluator._save_all_results({'image_expert': results}, args.save_dir)
    elif args.expert == 'uml':
        results = evaluator.evaluate_uml_expert(
            dataset_version=args.dataset_version,
            num_samples=args.num_samples
        )
        if args.save_dir:
            evaluator._save_all_results({f'uml_expert_{args.dataset_version}': results}, args.save_dir)
    elif args.expert == 'general':
        results = evaluator.evaluate_general_expert(
            dataset_version=args.dataset_version,
            num_samples=args.num_samples
        )
        if args.save_dir:
            evaluator._save_all_results({f'general_expert_{args.dataset_version}': results}, args.save_dir)

    logger.info("评估完成!")


if __name__ == "__main__":
    main()

# 快速测试所有专家（推荐）
# python scripts/evaluation/evaluate_experts.py --test-mode --show-samples --expert all

# 测试单个专家
# python scripts/evaluation/evaluate_experts.py --test-mode --expert text

# 查看General专家的样本数据
# python scripts/evaluation/evaluate_experts.py --show-samples --expert general --dataset-version qwen235B --num-samples 20