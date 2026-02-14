"""
Enhanced Metrics Module - Comprehensive Evaluation Metrics
增强的评估指标模块

功能:
  - 生成质量指标: BLEU, ROUGE, METEOR, BERTScore
  - 指令格式指标: Definition/Emphasis/Avoid完整性和格式分数
  - 统计指标: 长度统计、专家使用统计
  - 综合评估报告生成

环境要求: qwen_text (评估通常在此环境进行)

作者: Evaluation System
日期: 2025-02-06
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings

# 忽略BERTScore的警告
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger

logger = get_logger('metrics.enhanced')


class EvaluationThresholds:
    """评估阈值配置类"""

    # 语义相似度阈值
    ROUGE_L_THRESHOLD = 0.35  # ROUGE-L阈值，从0.5降低到0.35（适配开放式指令生成任务）
    BERTSCORE_F1_THRESHOLD = 0.85  # BERTScore F1阈值，从0.6提高到0.85

    # 组合逻辑
    USE_AND_LOGIC = True  # True=AND逻辑(两个都需满足), False=OR逻辑(满足一个即可)

    # 格式分数阈值
    FORMAT_SCORE_THRESHOLD = 1.0  # 格式分数阈值(0-1)，1.0表示完全正确

    @classmethod
    def get_config(cls) -> dict:
        """获取当前配置"""
        return {
            'rouge_l_threshold': cls.ROUGE_L_THRESHOLD,
            'bertscore_f1_threshold': cls.BERTSCORE_F1_THRESHOLD,
            'use_and_logic': cls.USE_AND_LOGIC,
            'format_score_threshold': cls.FORMAT_SCORE_THRESHOLD
        }

    @classmethod
    def update_config(cls, rouge_l: float = None, bertscore_f1: float = None,
                     use_and: bool = None, format_score: float = None):
        """更新配置"""
        if rouge_l is not None:
            cls.ROUGE_L_THRESHOLD = rouge_l
        if bertscore_f1 is not None:
            cls.BERTSCORE_F1_THRESHOLD = bertscore_f1
        if use_and is not None:
            cls.USE_AND_LOGIC = use_and
        if format_score is not None:
            cls.FORMAT_SCORE_THRESHOLD = format_score


class EnhancedMetrics:
    """
    增强的评估指标系统

    包含生成质量、格式检查、统计分析三大类指标
    """

    def __init__(self, use_bertscore: bool = True):
        """
        初始化评估指标

        Args:
            use_bertscore: 是否使用BERTScore(默认开启，评估语义相似度)
        """
        self.use_bertscore = use_bertscore

        # 延迟导入evaluate库,避免环境兼容问题
        self.bleu_metric = None
        self.rouge_metric = None
        self.meteor_metric = None
        self.bertscore_metric = None

        logger.info("初始化增强评估指标模块")
        if use_bertscore:
            logger.info("BERTScore已启用（默认）- 用于评估生成指令的语义相似度")

    def _lazy_load_metrics(self):
        """延迟加载评估指标(避免import错误)"""
        if self.bleu_metric is None:
            try:
                from evaluate import load

                # 预先下载NLTK数据(METEOR依赖)
                self._ensure_nltk_data()

                logger.info("加载BLEU指标...")
                self.bleu_metric = load('bleu')

                logger.info("加载ROUGE指标...")
                self.rouge_metric = load('rouge')

                logger.info("加载METEOR指标...")
                self.meteor_metric = load('meteor')

                if self.use_bertscore:
                    try:
                        logger.info("加载BERTScore指标...")
                        self.bertscore_metric = load('bertscore')
                        logger.info("BERTScore加载成功")
                    except Exception as e:
                        logger.warning(f"BERTScore加载失败: {e}")
                        logger.warning("将跳过BERTScore计算")
                        self.use_bertscore = False

                logger.info("评估指标加载完成")
            except Exception as e:
                logger.error(f"评估指标加载失败: {e}")
                raise

    def _ensure_nltk_data(self):
        """
        确保NLTK数据已下载(METEOR依赖)

        METEOR需要的NLTK数据包:
        - wordnet: 词汇数据库
        - punkt: 句子分词器
        - omw-1.4: 开放多语言词网
        """
        try:
            import nltk
            from nltk.data import find

            required_data = [
                ('corpora/wordnet', 'wordnet'),
                ('corpora/omw-1.4', 'omw-1.4'),
                ('tokenizers/punkt', 'punkt'),
                ('tokenizers/punkt_tab', 'punkt_tab')
            ]

            logger.info("检查NLTK数据包...")

            for data_path, data_name in required_data:
                try:
                    find(data_path)
                    logger.debug(f"NLTK数据包已存在: {data_name}")
                except LookupError:
                    logger.warning(f"NLTK数据包缺失: {data_name}, 尝试下载...")
                    try:
                        nltk.download(data_name, quiet=True)
                        logger.info(f"NLTK数据包下载成功: {data_name}")
                    except Exception as e:
                        logger.warning(f"NLTK数据包下载失败: {data_name} - {e}")
                        logger.warning(f"METEOR计算可能会失败或变慢")

            logger.info("NLTK数据检查完成")

        except ImportError:
            logger.warning("NLTK未安装, METEOR计算可能会失败")
        except Exception as e:
            logger.warning(f"NLTK数据检查失败: {e}")
            logger.warning("继续执行, 但METEOR计算可能会失败")

    def calculate_generation_quality(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        计算生成质量指标

        Args:
            predictions: 生成的指令列表
            references: 参考指令列表

        Returns:
            dict: 包含BLEU, ROUGE, METEOR, BERTScore的指标字典
        """
        self._lazy_load_metrics()

        if len(predictions) != len(references):
            raise ValueError(
                f"预测和参考数量不匹配: {len(predictions)} vs {len(references)}"
            )

        logger.info(f"计算生成质量指标 - 样本数: {len(predictions)}")

        results = {}

        # BLEU
        try:
            logger.info("开始计算BLEU指标...")
            bleu_result = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            results['bleu'] = bleu_result['bleu']
            logger.info(f"BLEU计算完成: {results['bleu']:.4f}")
        except Exception as e:
            logger.error(f"BLEU计算失败: {e}")
            results['bleu'] = 0.0

        # ROUGE
        try:
            logger.info("开始计算ROUGE指标...")
            rouge_result = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            results['rouge1'] = rouge_result['rouge1']
            results['rouge2'] = rouge_result['rouge2']
            results['rougeL'] = rouge_result['rougeL']
            logger.info(f"ROUGE计算完成 - ROUGE-L: {results['rougeL']:.4f}")
        except Exception as e:
            logger.error(f"ROUGE计算失败: {e}")
            results['rouge1'] = results['rouge2'] = results['rougeL'] = 0.0

        # METEOR
        try:
            logger.info("开始计算METEOR指标...")
            logger.info(f"METEOR计算中 - 样本数: {len(predictions)}, 请耐心等待...")

            # METEOR计算可能较慢,添加详细日志
            meteor_result = self.meteor_metric.compute(
                predictions=predictions,
                references=references
            )
            results['meteor'] = meteor_result['meteor']
            logger.info(f"METEOR计算完成: {results['meteor']:.4f}")
        except Exception as e:
            logger.error(f"METEOR计算失败: {e}")
            logger.error(f"可能原因: NLTK数据缺失或网络问题")
            logger.error(f"建议: 手动下载NLTK数据或禁用METEOR")
            results['meteor'] = 0.0

        # BERTScore
        if self.use_bertscore and self.bertscore_metric is not None:
            try:
                logger.info("开始计算BERTScore指标...")
                logger.info(f"BERTScore计算中 - 这可能需要几分钟...")

                bertscore_result = self.bertscore_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang='en'
                )
                # 取平均值
                results['bertscore_precision'] = sum(bertscore_result['precision']) / len(predictions)
                results['bertscore_recall'] = sum(bertscore_result['recall']) / len(predictions)
                results['bertscore_f1'] = sum(bertscore_result['f1']) / len(predictions)
                logger.info(f"BERTScore计算完成 - F1: {results['bertscore_f1']:.4f}")
            except Exception as e:
                logger.error(f"BERTScore计算失败: {e}")
                results['bertscore_precision'] = 0.0
                results['bertscore_recall'] = 0.0
                results['bertscore_f1'] = 0.0

        logger.info("所有生成质量指标计算完成")
        return results

    def calculate_format_metrics(
        self,
        instructions: List[str]
    ) -> Dict[str, Any]:
        """
        计算指令格式指标(重新设计)

        新的格式要求:
        - Definition必须有实际内容(不能只是"-")
        - Emphasis/Avoid可以是内容或显式"-"
        - 三段式完整性检查
        - 格式分数(0-1)

        Args:
            instructions: 指令列表

        Returns:
            dict: 格式指标字典
        """
        logger.info(f"计算格式指标 - 样本数: {len(instructions)}")

        format_results = []

        for instruction in instructions:
            result = self._check_single_instruction_format(instruction)
            format_results.append(result)

        # 统计汇总
        total = len(format_results)

        summary = {
            'total_samples': total,
            'valid_count': sum(1 for r in format_results if r['is_valid']),
            'valid_rate': sum(1 for r in format_results if r['is_valid']) / total if total > 0 else 0,

            # Definition指标
            'definition_present': sum(1 for r in format_results if r['has_definition']) / total if total > 0 else 0,
            'definition_has_content': sum(1 for r in format_results if r['definition_has_content']) / total if total > 0 else 0,
            'definition_in_this_task': sum(1 for r in format_results if r['definition_starts_with_in_this_task']) / total if total > 0 else 0,

            # Emphasis指标
            'emphasis_present': sum(1 for r in format_results if r['has_emphasis']) / total if total > 0 else 0,
            'emphasis_valid': sum(1 for r in format_results if r['emphasis_is_valid']) / total if total > 0 else 0,

            # Avoid指标
            'avoid_present': sum(1 for r in format_results if r['has_avoid']) / total if total > 0 else 0,
            'avoid_valid': sum(1 for r in format_results if r['avoid_is_valid']) / total if total > 0 else 0,

            # 格式分数(0-1)
            'avg_format_score': sum(r['format_score'] for r in format_results) / total if total > 0 else 0,

            # 详细结果
            'detailed_results': format_results
        }

        logger.info(f"格式验证通过率: {summary['valid_rate']:.2%}")
        logger.info(f"平均格式分数: {summary['avg_format_score']:.4f}")

        return summary

    def _check_single_instruction_format(self, instruction: str) -> Dict[str, Any]:
        """
        检查单条指令的格式

        Args:
            instruction: 指令文本

        Returns:
            dict: 格式检查结果
        """
        result = {
            'is_valid': False,
            'has_definition': False,
            'definition_has_content': False,
            'definition_starts_with_in_this_task': False,
            'has_emphasis': False,
            'emphasis_is_valid': False,
            'has_avoid': False,
            'avoid_is_valid': False,
            'format_score': 0.0,
            'errors': []
        }

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
                result['has_avoid'] = True

        # 检查Definition
        if definition_line:
            content = definition_line.split('Definition:', 1)[1].strip()

            # Definition不能只是"-"
            if content and content != '-':
                result['definition_has_content'] = True
            else:
                result['errors'].append('Definition没有实际内容')

            # 检查是否以"In this task"开头
            if content.lower().startswith('in this task'):
                result['definition_starts_with_in_this_task'] = True
        else:
            result['errors'].append('缺少Definition')

        # 检查Emphasis(可以是内容或显式"-")
        if emphasis_line:
            content = emphasis_line.split(':', 1)[1].strip()
            # 有内容或者是显式的"-"都算有效
            if content:
                result['emphasis_is_valid'] = True
        else:
            result['errors'].append('缺少Emphasis & Caution')

        # 检查Avoid(可以是内容或显式"-")
        if avoid_line:
            content = avoid_line.split(':', 1)[1].strip()
            # 有内容或者是显式的"-"都算有效
            if content:
                result['avoid_is_valid'] = True
        else:
            result['errors'].append('缺少Things to Avoid')

        # 计算格式分数(0-1)
        score_components = [
            result['has_definition'],
            result['definition_has_content'],
            result['definition_starts_with_in_this_task'],
            result['has_emphasis'],
            result['emphasis_is_valid'],
            result['has_avoid'],
            result['avoid_is_valid']
        ]
        result['format_score'] = sum(score_components) / len(score_components)

        # 综合判断是否有效
        # 新的有效标准:
        # 1. Definition必须存在且有内容
        # 2. Emphasis必须存在
        # 3. Avoid必须存在
        result['is_valid'] = (
            result['definition_has_content'] and
            result['has_emphasis'] and
            result['has_avoid']
        )

        return result

    def calculate_statistical_metrics(
        self,
        instructions: List[str],
        expert_usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        计算统计指标

        Args:
            instructions: 指令列表
            expert_usage: 专家使用统计字典(可选)

        Returns:
            dict: 统计指标字典
        """
        logger.info(f"计算统计指标 - 样本数: {len(instructions)}")

        # 长度统计
        lengths = [len(inst) for inst in instructions]
        word_counts = [len(inst.split()) for inst in instructions]
        line_counts = [len(inst.split('\n')) for inst in instructions]

        stats = {
            # 字符长度统计
            'char_length': {
                'mean': sum(lengths) / len(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0,
                'median': sorted(lengths)[len(lengths)//2] if lengths else 0
            },

            # 单词数统计
            'word_count': {
                'mean': sum(word_counts) / len(word_counts) if word_counts else 0,
                'min': min(word_counts) if word_counts else 0,
                'max': max(word_counts) if word_counts else 0,
                'median': sorted(word_counts)[len(word_counts)//2] if word_counts else 0
            },

            # 行数统计
            'line_count': {
                'mean': sum(line_counts) / len(line_counts) if line_counts else 0,
                'min': min(line_counts) if line_counts else 0,
                'max': max(line_counts) if line_counts else 0,
                'median': sorted(line_counts)[len(line_counts)//2] if line_counts else 0
            }
        }

        # 专家使用统计
        if expert_usage:
            total_usage = sum(expert_usage.values())
            stats['expert_usage'] = {
                'total_calls': total_usage,
                'usage_by_expert': expert_usage,
                'usage_percentage': {
                    expert: (count / total_usage * 100) if total_usage > 0 else 0
                    for expert, count in expert_usage.items()
                }
            }

        logger.info(f"平均字符长度: {stats['char_length']['mean']:.1f}")
        logger.info(f"平均单词数: {stats['word_count']['mean']:.1f}")

        return stats

    def calculate_binary_classification_metrics(
        self,
        predictions: List[str],
        references: List[str],
        format_threshold: float = None,
        rouge_threshold: float = None,
        bertscore_threshold: float = None,
        use_and_logic: bool = None
    ) -> Dict[str, Any]:
        """
        计算二分类指标：TP, TN, FP, FN

        定义：
        - 有效指令（正类）= 格式完整 AND 语义相似度达标
        - 格式完整 = 三段式结构完整（Definition + Emphasis & Caution + Things to Avoid）
        - 语义相似度达标 = (ROUGE-L >= rouge_threshold) AND/OR (BERTScore F1 >= bertscore_threshold)

        分类：
        - TP (True Positive): 格式正确 + 语义达标
        - FP (False Positive): 格式正确 + 语义不达标（生成了错误的指令）
        - FN (False Negative): 格式不正确 或 语义不达标
        - TN (True Negative): 在当前场景中不适用（所有输入都需要生成指令）

        Args:
            predictions: 生成的指令列表
            references: 参考指令列表
            format_threshold: 格式分数阈值（默认使用配置值）
            rouge_threshold: ROUGE-L阈值（默认使用配置值）
            bertscore_threshold: BERTScore F1阈值（默认使用配置值）
            use_and_logic: 是否使用AND逻辑组合ROUGE和BERTScore（默认使用配置值）

        Returns:
            dict: 包含TP, FP, FN, TN, Precision, Recall, F1, Accuracy的字典
        """
        # 使用配置的默认值
        if format_threshold is None:
            format_threshold = EvaluationThresholds.FORMAT_SCORE_THRESHOLD
        if rouge_threshold is None:
            rouge_threshold = EvaluationThresholds.ROUGE_L_THRESHOLD
        if bertscore_threshold is None:
            bertscore_threshold = EvaluationThresholds.BERTSCORE_F1_THRESHOLD
        if use_and_logic is None:
            use_and_logic = EvaluationThresholds.USE_AND_LOGIC

        logger.info(f"计算二分类指标 - 样本数: {len(predictions)}")
        logger.info(f"阈值配置:")
        logger.info(f"  格式分数阈值: {format_threshold}")
        logger.info(f"  ROUGE-L阈值: {rouge_threshold}")
        logger.info(f"  BERTScore F1阈值: {bertscore_threshold}")
        logger.info(f"  组合逻辑: {'AND (两者都需满足)' if use_and_logic else 'OR (满足一个即可)'}")

        if len(predictions) != len(references):
            raise ValueError(
                f"预测和参考数量不匹配: {len(predictions)} vs {len(references)}"
            )

        # 计算格式指标
        format_results = self.calculate_format_metrics(predictions)

        # 计算ROUGE-L分数（用于语义相似度）
        self._lazy_load_metrics()
        try:
            rouge_result = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            rouge_l_scores = []
            # 获取每个样本的ROUGE-L分数
            for pred, ref in zip(predictions, references):
                sample_rouge = self.rouge_metric.compute(
                    predictions=[pred],
                    references=[ref]
                )
                rouge_l_scores.append(sample_rouge['rougeL'])
        except Exception as e:
            logger.error(f"ROUGE-L计算失败: {e}")
            rouge_l_scores = [0.0] * len(predictions)

        # 可选：使用BERTScore作为额外的语义相似度指标
        bertscore_f1_scores = []
        if self.use_bertscore and self.bertscore_metric is not None:
            try:
                logger.info("使用BERTScore计算语义相似度...")
                bertscore_result = self.bertscore_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang='en'
                )
                bertscore_f1_scores = bertscore_result['f1']
                logger.info(f"BERTScore F1平均值: {sum(bertscore_f1_scores)/len(bertscore_f1_scores):.4f}")
            except Exception as e:
                logger.error(f"BERTScore计算失败: {e}")
                bertscore_f1_scores = [0.0] * len(predictions)

        # 计算每个样本的分类
        tp = 0  # True Positive
        fp = 0  # False Positive
        fn = 0  # False Negative
        tn = 0  # True Negative (在当前场景中为0)

        valid_samples = []  # 记录TP样本索引
        invalid_samples = []  # 记录FP/FN样本索引

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # 检查格式完整性
            format_check = self._check_single_format(pred)
            is_format_valid = (
                format_check['has_definition'] and
                format_check['has_emphasis'] and
                format_check['has_avoid'] and
                format_check['format_score'] >= format_threshold
            )

            # 检查语义相似度
            rouge_l_score = rouge_l_scores[i]
            rouge_valid = rouge_l_score >= rouge_threshold

            # 如果有BERTScore，使用配置的组合逻辑
            if bertscore_f1_scores:
                bertscore_f1 = bertscore_f1_scores[i]
                bertscore_valid = bertscore_f1 >= bertscore_threshold

                # 根据配置使用AND或OR逻辑
                if use_and_logic:
                    is_semantic_valid = rouge_valid and bertscore_valid
                else:
                    is_semantic_valid = rouge_valid or bertscore_valid
            else:
                # 如果没有BERTScore，只使用ROUGE
                is_semantic_valid = rouge_valid

            # 分类逻辑
            if is_format_valid and is_semantic_valid:
                tp += 1
                valid_samples.append(i)
            elif is_format_valid and not is_semantic_valid:
                fp += 1
                invalid_samples.append(i)
            else:
                fn += 1
                invalid_samples.append(i)

        # 计算派生指标
        total = len(predictions)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        results = {
            # 基础分类指标
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,

            # 派生指标
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,

            # 元数据
            'total_samples': total,
            'valid_samples': valid_samples,
            'invalid_samples': invalid_samples,

            # 阈值信息
            'format_threshold': format_threshold,
            'rouge_threshold': rouge_threshold,
            'bertscore_threshold': bertscore_threshold,
            'use_and_logic': use_and_logic,
            'use_bertscore': self.use_bertscore and len(bertscore_f1_scores) > 0
        }

        logger.info(f"二分类指标计算完成:")
        logger.info(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")

        return results

    def _check_single_format(self, instruction: str) -> Dict[str, Any]:
        """
        检查单个指令的格式

        Args:
            instruction: 指令文本

        Returns:
            dict: 格式检查结果
        """
        import re

        result = {
            'has_definition': False,
            'has_emphasis': False,
            'has_avoid': False,
            'definition_has_content': False,
            'emphasis_valid': False,
            'avoid_valid': False,
            'format_score': 0.0
        }

        if not instruction or len(instruction.strip()) < 10:
            return result

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        for line in lines:
            # Definition检查
            if line.startswith('Definition:'):
                result['has_definition'] = True
                content = line[len('Definition:'):].strip()
                if content and content != '-':
                    result['definition_has_content'] = True

            # Emphasis检查
            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                result['has_emphasis'] = True
                prefix_len = len('Emphasis & Caution:') if 'Emphasis & Caution:' in line else len('Emphasis and Caution:')
                content = line[prefix_len:].strip()
                if content:
                    result['emphasis_valid'] = True

            # Avoid检查
            elif line.startswith('Things to Avoid:'):
                result['has_avoid'] = True
                content = line[len('Things to Avoid:'):].strip()
                if content:
                    result['avoid_valid'] = True

        # 计算格式分数
        score = 0.0
        if result['definition_has_content']:
            score += 0.4
        if result['has_emphasis']:
            score += 0.3
        if result['has_avoid']:
            score += 0.3

        result['format_score'] = score

        return result

    def generate_comprehensive_report(
        self,
        predictions: List[str],
        references: List[str],
        expert_usage: Optional[Dict[str, int]] = None,
        save_path: Optional[str] = None,
        include_binary_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        生成综合评估报告

        Args:
            predictions: 生成的指令列表
            references: 参考指令列表
            expert_usage: 专家使用统计
            save_path: 保存路径(可选)
            include_binary_metrics: 是否包含二分类指标（默认True）

        Returns:
            dict: 综合评估报告
        """
        logger.info("=" * 80)
        logger.info("生成综合评估报告")
        logger.info("=" * 80)

        report = {
            'metadata': {
                'total_samples': len(predictions),
                'timestamp': self._get_timestamp()
            }
        }

        # 1. 生成质量指标
        logger.info("\n[1/4] 计算生成质量指标...")
        report['generation_quality'] = self.calculate_generation_quality(
            predictions, references
        )

        # 2. 格式指标
        logger.info("\n[2/4] 计算格式指标...")
        report['format_metrics'] = self.calculate_format_metrics(predictions)

        # 3. 二分类指标（TP/TN/FP/FN）
        if include_binary_metrics:
            logger.info("\n[3/4] 计算二分类指标（TP/TN/FP/FN）...")
            report['binary_classification'] = self.calculate_binary_classification_metrics(
                predictions, references
            )
        else:
            logger.info("\n[3/4] 跳过二分类指标计算")

        # 4. 统计指标
        logger.info("\n[4/4] 计算统计指标...")
        report['statistical_metrics'] = self.calculate_statistical_metrics(
            predictions, expert_usage
        )

        # 保存报告
        if save_path:
            self._save_report(report, save_path)

        logger.info("=" * 80)
        logger.info("综合评估报告生成完成")
        logger.info("=" * 80)

        return report

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _save_report(self, report: Dict, save_path: str):
        """
        保存评估报告

        Args:
            report: 报告字典
            save_path: 保存路径
        """
        import json
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"评估报告已保存至: {save_path}")

    def print_report_summary(self, report: Dict):
        """
        打印报告摘要

        Args:
            report: 评估报告字典
        """
        print("\n" + "=" * 80)
        print("评估报告摘要")
        print("=" * 80)

        # 生成质量
        print("\n[生成质量指标]")
        quality = report['generation_quality']
        print(f"  BLEU:      {quality['bleu']:.4f}")
        print(f"  ROUGE-1:   {quality['rouge1']:.4f}")
        print(f"  ROUGE-2:   {quality['rouge2']:.4f}")
        print(f"  ROUGE-L:   {quality['rougeL']:.4f}")
        print(f"  METEOR:    {quality['meteor']:.4f}")
        if 'bertscore_f1' in quality:
            print(f"  BERTScore P: {quality['bertscore_precision']:.4f}")
            print(f"  BERTScore R: {quality['bertscore_recall']:.4f}")
            print(f"  BERTScore F1: {quality['bertscore_f1']:.4f}")

        # 格式指标
        print("\n[格式指标]")
        format_m = report['format_metrics']
        print(f"  格式验证通过率: {format_m['valid_rate']:.2%}")
        print(f"  平均格式分数:   {format_m['avg_format_score']:.4f}")
        print(f"  Definition有效: {format_m['definition_has_content']:.2%}")
        print(f"  Emphasis有效:   {format_m['emphasis_valid']:.2%}")
        print(f"  Avoid有效:      {format_m['avoid_valid']:.2%}")

        # 二分类指标
        if 'binary_classification' in report:
            print("\n[二分类指标 (TP/TN/FP/FN)]")
            binary = report['binary_classification']
            print(f"  TP (True Positive):  {binary['TP']:4d}  - 格式正确且语义达标")
            print(f"  FP (False Positive): {binary['FP']:4d}  - 格式正确但语义不达标")
            print(f"  FN (False Negative): {binary['FN']:4d}  - 格式错误或语义不达标")
            print(f"  TN (True Negative):  {binary['TN']:4d}  - 不适用")
            print(f"  ---")
            print(f"  Precision (精确率): {binary['precision']:.4f}")
            print(f"  Recall (召回率):    {binary['recall']:.4f}")
            print(f"  F1 Score:           {binary['f1_score']:.4f}")
            print(f"  Accuracy (准确率):  {binary['accuracy']:.4f}")

        # 统计指标
        print("\n[统计指标]")
        stats = report['statistical_metrics']
        print(f"  平均字符长度: {stats['char_length']['mean']:.1f}")
        print(f"  平均单词数:   {stats['word_count']['mean']:.1f}")
        print(f"  平均行数:     {stats['line_count']['mean']:.1f}")

        if 'expert_usage' in stats:
            print("\n[专家使用统计]")
            for expert, pct in stats['expert_usage']['usage_percentage'].items():
                print(f"  {expert}: {pct:.1f}%")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    print("=" * 60)
    print("增强评估指标模块测试")
    print("=" * 60)

    # 创建评估器（默认开启BERTScore）
    metrics = EnhancedMetrics(use_bertscore=True)

    # 测试数据
    predictions = [
        "Definition: In this task, draw bounding boxes around objects.\nEmphasis & Caution: Be accurate.\nThings to Avoid: Do not annotate backgrounds.",
        "Definition: In this task, implement login functionality.\nEmphasis & Caution: -\nThings to Avoid: Do not skip validation.",
        "Definition: -\nEmphasis & Caution: Test thoroughly.\nThings to Avoid: -"
    ]

    references = [
        "Definition: In this task, annotate all visible objects with bounding boxes.\nEmphasis & Caution: Focus on foreground objects.\nThings to Avoid: Avoid partial objects.",
        "Definition: In this task, create user authentication system.\nEmphasis & Caution: Ensure security.\nThings to Avoid: Do not store plain passwords.",
        "Definition: In this task, test the login feature.\nEmphasis & Caution: Cover edge cases.\nThings to Avoid: Do not skip error handling."
    ]

    print("\n测试1: 格式指标")
    print("-" * 60)
    format_results = metrics.calculate_format_metrics(predictions)
    print(f"格式验证通过率: {format_results['valid_rate']:.2%}")
    print(f"平均格式分数: {format_results['avg_format_score']:.4f}")

    print("\n测试2: 生成质量指标")
    print("-" * 60)
    try:
        quality_results = metrics.calculate_generation_quality(predictions, references)
        print(f"BLEU: {quality_results['bleu']:.4f}")
        print(f"ROUGE-L: {quality_results['rougeL']:.4f}")
        print(f"METEOR: {quality_results['meteor']:.4f}")
        if 'bertscore_f1' in quality_results:
            print(f"BERTScore F1: {quality_results['bertscore_f1']:.4f}")
    except Exception as e:
        print(f"生成质量指标计算失败(可能缺少依赖): {e}")

    print("\n测试3: 统计指标")
    print("-" * 60)
    expert_usage = {'text_expert': 1, 'image_expert': 1, 'uml_expert': 1}
    stats_results = metrics.calculate_statistical_metrics(predictions, expert_usage)
    print(f"平均字符长度: {stats_results['char_length']['mean']:.1f}")
    print(f"平均单词数: {stats_results['word_count']['mean']:.1f}")

    print("\n测试4: 二分类指标（TP/TN/FP/FN）")
    print("-" * 60)
    try:
        binary_results = metrics.calculate_binary_classification_metrics(
            predictions, references
        )
        print(f"TP: {binary_results['TP']}, FP: {binary_results['FP']}")
        print(f"FN: {binary_results['FN']}, TN: {binary_results['TN']}")
        print(f"Precision: {binary_results['precision']:.4f}")
        print(f"Recall: {binary_results['recall']:.4f}")
        print(f"F1 Score: {binary_results['f1_score']:.4f}")
    except Exception as e:
        print(f"二分类指标计算失败: {e}")

    print("\n测试完成!")