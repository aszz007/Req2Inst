"""
scripts/preprocessing/dataset_sampling/sample_text_dataset.py

从文本数据集（data/dataset/text/）中随机采样并展示样本。
文本数据集目录下可能包含若干CSV文件，脚本会合并所有CSV后再采样。

用法:
    conda activate instruction_generator
    python scripts/preprocessing/dataset_sampling/sample_text_dataset.py
    python scripts/preprocessing/dataset_sampling/sample_text_dataset.py --n 5
    python scripts/preprocessing/dataset_sampling/sample_text_dataset.py --n 10 --seed 42
    python scripts/preprocessing/dataset_sampling/sample_text_dataset.py --output outputs/samples/text_samples.csv
"""

import argparse
import os
import sys
import random
import pandas as pd
from pathlib import Path

# 项目根目录（从本脚本位置向上4级）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# 文本数据集目录（可能包含若干CSV文件）
TEXT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "text"

# 训练时使用的字段
TRAIN_FIELD = "Low_Requirements"
OUTPUT_FIELD = "Instruction"

# 数据集总规模参考（来自框架文档）
DATASET_TOTAL = 2472


def load_text_dataset(dataset_dir: Path) -> pd.DataFrame:
    """
    加载文本数据集目录下所有CSV文件并合并。

    Args:
        dataset_dir: 文本数据集目录路径

    Returns:
        合并后的DataFrame
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"文本数据集目录不存在: {dataset_dir}")

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"数据集目录中未找到任何CSV文件: {dataset_dir}")

    print(f"发现 {len(csv_files)} 个CSV文件:")
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="utf-8")
        print(f"  - {csv_file.name}: {len(df)} 条")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"合并后总计: {len(combined)} 条\n")
    return combined


def sample_dataset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    从DataFrame中随机采样n条记录。

    Args:
        df: 源数据集
        n: 采样数量
        seed: 随机种子，保证可复现

    Returns:
        采样结果DataFrame
    """
    if n > len(df):
        print(f"[警告] 请求采样 {n} 条，但数据集只有 {len(df)} 条，将返回全部数据。")
        return df.copy()

    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def display_samples(samples: pd.DataFrame) -> None:
    """
    打印采样结果到终端。

    Args:
        samples: 采样结果DataFrame
    """
    print("=" * 70)
    print(f"随机采样结果（共 {len(samples)} 条）")
    print("=" * 70)

    for idx, row in samples.iterrows():
        print(f"\n【样本 {idx + 1}】")
        print("-" * 50)
        # 展示训练输入字段
        if TRAIN_FIELD in row:
            req = str(row[TRAIN_FIELD]).strip()
            print(f"[{TRAIN_FIELD}]\n{req}")
        elif "High_Requirements" in row:
            # 仅展示，不用于训练
            high = str(row.get("High_Requirements", "")).strip()
            if high:
                print(f"[High_Requirements（仅展示，不用于训练）]\n{high}")
        # 展示指令输出
        if OUTPUT_FIELD in row:
            instr = str(row[OUTPUT_FIELD]).strip()
            print(f"\n[{OUTPUT_FIELD}]\n{instr}")
        # 展示其他可用字段（排除训练无关字段）
        extra_cols = [
            c for c in row.index
            if c not in {TRAIN_FIELD, "High_Requirements", OUTPUT_FIELD}
        ]
        if extra_cols:
            for col in extra_cols:
                val = str(row[col]).strip()
                if val and val != "nan":
                    print(f"\n[{col}]: {val}")
        print("-" * 50)


def save_samples(samples: pd.DataFrame, output_path: str) -> None:
    """
    将采样结果保存到CSV文件。

    Args:
        samples: 采样结果DataFrame
        output_path: 输出文件路径
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out, index=False, encoding="utf-8")
    print(f"\n采样结果已保存至: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从文本数据集（data/dataset/text/）随机采样并展示样本"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="采样数量（默认: 3）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，不指定时每次结果不同（默认: None）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="将采样结果保存到指定路径（可选，例如 outputs/samples/text_samples.csv）"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=f"数据集目录路径（默认: {TEXT_DATASET_DIR}）"
    )
    return parser.parse_args()


def run_sampling(n: int = 3, seed: int = None, output: str = None,
                 dataset_dir: str = None) -> pd.DataFrame:
    """
    执行采样流程，供外部调用或脚本直接运行。

    Args:
        n: 采样数量
        seed: 随机种子
        output: 输出文件路径（可选）
        dataset_dir: 数据集目录路径（可选）

    Returns:
        采样结果DataFrame
    """
    # 确定数据集目录
    dir_path = Path(dataset_dir) if dataset_dir else TEXT_DATASET_DIR

    # 设置随机种子
    actual_seed = seed if seed is not None else random.randint(0, 99999)
    print(f"随机种子: {actual_seed}")

    # 加载数据集
    print(f"加载文本数据集: {dir_path}")
    df = load_text_dataset(dir_path)

    # 采样
    samples = sample_dataset(df, n=n, seed=actual_seed)

    # 展示
    display_samples(samples)

    # 统计字段信息
    print(f"\n数据集列: {list(df.columns)}")
    print(f"数据集规模: {len(df)} 条（框架参考: {DATASET_TOTAL} 条）")

    # 保存（可选）
    if output:
        save_samples(samples, output)

    return samples


if __name__ == "__main__":
    args = parse_args()
    run_sampling(
        n=args.n,
        seed=args.seed,
        output=args.output,
        dataset_dir=args.dataset_dir,
    )