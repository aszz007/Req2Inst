"""
实验脚本公共工具模块
功能：提供所有实验脚本（exp1–exp11）共用的工具函数，消除跨实验重复代码
包含：缓存检查、数据加载、专家构建、样本构建、参数解析等
"""

import json
import argparse
from pathlib import Path


def is_full_run_cache(cache_dir, filename):
    """
    检查是否存在非test-mode的推理缓存文件。

    用于 --only-missing 逻辑：如果已有完整运行的缓存则跳过重复推理，
    但test-mode缓存视为"缺失"以便后续full-run自动覆盖。

    Args:
        cache_dir: 缓存目录路径
        filename: 缓存文件名

    Returns:
        bool: True 表示存在有效的全量缓存
    """
    filepath = Path(cache_dir) / filename
    if not filepath.exists():
        return False
    try:
        raw = json.loads(filepath.read_text(encoding='utf-8'))
        return not (
            raw.get('test_mode', False)
            or raw.get('metadata', {}).get('test_mode', False)
        )
    except Exception:
        return False


def load_test_data(expert_type):
    """
    加载指定专家类型的测试集数据。

    复用 exp2/exp3/exp9/exp10 中完全相同的 _load_test_data() 实现。

    Args:
        expert_type: 专家类型（'text', 'image', 'uml', 'general'）

    Returns:
        list: 测试集数据列表
    """
    from src.training.data_loader import (
        TextDatasetLoader, ImageDatasetLoader,
        UMLDatasetLoader, GeneralDatasetLoader,
        split_dataset_for_expert,
    )
    if expert_type == 'text':
        data = TextDatasetLoader().load_csv_files()
    elif expert_type == 'image':
        data = ImageDatasetLoader().load_csv_file()
    elif expert_type == 'uml':
        data = UMLDatasetLoader().load_csv_file()
    else:  # general
        data = GeneralDatasetLoader().load_all_data()

    _, _, test_data = split_dataset_for_expert(data, expert_type)
    return test_data


def get_expert(expert_type, lora_path=None, use_4bit=True):
    """
    构建指定类型的专家实例。

    复用 exp3/exp9 中的 _get_expert() 实现。

    Args:
        expert_type: 专家类型（'text', 'image', 'uml', 'general'）
        lora_path: LoRA权重路径（None则使用默认配置）
        use_4bit: 是否使用4bit量化

    Returns:
        Expert实例
    """
    from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert
    cls = {
        'text': TextExpert,
        'image': ImageExpert,
        'uml': UMLExpert,
        'general': GeneralExpert,
    }[expert_type]
    return cls(lora_path=lora_path, use_4bit=use_4bit)


def make_samples(inputs, predictions, references):
    """
    构建标准化的样本列表（index + input + prediction + reference）。

    复用所有实验中反复出现的 samples 列表推导式。

    Args:
        inputs: 输入文本列表
        predictions: 预测文本列表
        references: 参考文本列表

    Returns:
        list[dict]: 样本列表
    """
    return [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]


def truncate_for_test_mode(inputs, references, test_mode, n=10):
    """
    在test-mode下截断数据到前n条。

    复用所有实验中的 `if args.test_mode: inputs, references = inputs[:10], references[:10]`

    Args:
        inputs: 输入列表
        references: 参考输出列表
        test_mode: 是否为test模式
        n: test模式下保留的样本数

    Returns:
        tuple: (inputs, references)
    """
    if test_mode:
        return inputs[:n], references[:n]
    return inputs, references


def add_common_args(parser):
    """
    向 ArgumentParser 添加所有实验共用的命令行参数。

    共用参数：--force-regenerate, --from-cache, --no-bertscore, --test-mode, --only-missing

    Args:
        parser: argparse.ArgumentParser 实例

    Returns:
        parser: 添加参数后的 parser（同一实例，方便链式调用）
    """
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Re-run inference even if cache exists')
    parser.add_argument('--from-cache', action='store_true',
                        help='Skip inference, load from cache only')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='Disable BERTScore for faster evaluation')
    parser.add_argument('--test-mode', action='store_true',
                        help='Use 10 samples only (quick validation)')
    parser.add_argument('--only-missing', action='store_true',
                        help='Skip combinations that already have a full-run cache. '
                             'Test-mode caches are treated as missing and re-run automatically.')
    return parser


def finalize_args(args):
    """
    处理参数间的依赖关系（如 --from-cache 时禁用 --force-regenerate）。

    Args:
        args: argparse.Namespace

    Returns:
        args: 处理后的 args
    """
    if getattr(args, 'from_cache', False):
        args.force_regenerate = False
        if hasattr(args, 'force_retrain'):
            args.force_retrain = False
    return args