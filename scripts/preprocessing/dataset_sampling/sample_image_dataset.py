"""
scripts/preprocessing/dataset_sampling/sample_image_dataset.py

从图像数据集（data/dataset/image/image_dataset.csv）中随机采样并展示样本。
图像数据集只有一个CSV文件。训练时输入为JSON文本描述，不是图像本身。
采样时仅展示 Description 字段（JSON中的描述内容）及对应指令。

用法:
    conda activate instruction_generator
    python scripts/preprocessing/dataset_sampling/sample_image_dataset.py
    python scripts/preprocessing/dataset_sampling/sample_image_dataset.py --n 5
    python scripts/preprocessing/dataset_sampling/sample_image_dataset.py --n 10 --seed 42
    python scripts/preprocessing/dataset_sampling/sample_image_dataset.py --output outputs/samples/image_samples.csv
"""

import argparse
import ast
import json
import os
import sys
import random
import pandas as pd
from pathlib import Path

# 项目根目录（从本脚本位置向上4级）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# 图像数据集路径（只有一个CSV文件）
IMAGE_DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "image" / "image_dataset.csv"

# 训练时使用的字段
# Image Expert 输入：JSON 文本描述（仅提取 Description 字段）
# 训练时忽略 High_Requirements，只使用 Low_Requirements
DESCRIPTION_FIELD = "Description"       # JSON 描述列（包含整个JSON，训练时仅取 Description 字段）
TRAIN_INPUT_FIELD = "Low_Requirements"  # 训练输入字段（如果有单独列）
OUTPUT_FIELD = "Instruction"            # 训练输出字段

# 数据集总规模参考（来自框架文档）
DATASET_TOTAL = 1000


def load_image_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    加载图像数据集CSV文件。

    Args:
        dataset_path: 数据集CSV文件路径

    Returns:
        加载的DataFrame
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"图像数据集文件不存在: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"加载图像数据集: {dataset_path.name}")
    print(f"总计: {len(df)} 条，列: {list(df.columns)}\n")
    return df


def extract_description(raw_value: str) -> str:
    """
    从 JSON 描述字段中提取 Description 内容。
    训练时仅使用 JSON 中的 Description 字段，其他元数据不参与训练。

    Args:
        raw_value: CSV中的原始字段值（可能是JSON字符串或普通文本）

    Returns:
        提取到的描述文本
    """
    if not isinstance(raw_value, str) or not raw_value.strip():
        return ""

    raw = raw_value.strip()

    # 尝试解析 JSON
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, dict) and "Description" in parsed:
                return str(parsed["Description"]).strip()
        except Exception:
            pass

    # 非 JSON 格式，直接返回原始文本
    return raw


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
    Image Expert 输入是 JSON 文本描述，展示时提取 Description 字段。

    Args:
        samples: 采样结果DataFrame
    """
    print("=" * 70)
    print(f"随机采样结果（共 {len(samples)} 条）")
    print("注意: Image Expert 输入是 JSON 文本描述，不是图像")
    print("=" * 70)

    for idx, row in samples.iterrows():
        print(f"\n【样本 {idx + 1}】")
        print("-" * 50)

        # 展示 JSON 描述字段（训练时仅取 Description 子字段）
        if DESCRIPTION_FIELD in row:
            raw = str(row[DESCRIPTION_FIELD]).strip()
            description = extract_description(raw)
            if description:
                print(f"[Description（训练输入，提取自JSON）]\n{description}")
            else:
                print(f"[{DESCRIPTION_FIELD}（原始）]\n{raw}")
        elif TRAIN_INPUT_FIELD in row:
            val = str(row[TRAIN_INPUT_FIELD]).strip()
            if val and val != "nan":
                print(f"[{TRAIN_INPUT_FIELD}]\n{val}")

        # Low_Requirements（训练输入字段）
        if TRAIN_INPUT_FIELD in row and DESCRIPTION_FIELD in row:
            val = str(row.get(TRAIN_INPUT_FIELD, "")).strip()
            if val and val != "nan":
                print(f"\n[{TRAIN_INPUT_FIELD}]\n{val}")

        # 展示指令输出
        if OUTPUT_FIELD in row:
            instr = str(row[OUTPUT_FIELD]).strip()
            if instr and instr != "nan":
                print(f"\n[{OUTPUT_FIELD}]\n{instr}")

        # High_Requirements（仅展示，不用于训练）
        high = str(row.get("High_Requirements", "")).strip()
        if high and high != "nan":
            print(f"\n[High_Requirements（仅展示，不用于训练）]\n{high}")

        # 其他字段
        skip_cols = {DESCRIPTION_FIELD, TRAIN_INPUT_FIELD, OUTPUT_FIELD, "High_Requirements"}
        for col in row.index:
            if col not in skip_cols:
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
        description="从图像数据集（data/dataset/image/image_dataset.csv）随机采样并展示样本"
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
        help="将采样结果保存到指定路径（可选，例如 outputs/samples/image_samples.csv）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"数据集CSV路径（默认: {IMAGE_DATASET_PATH}）"
    )
    return parser.parse_args()


def run_sampling(n: int = 3, seed: int = None, output: str = None,
                 dataset: str = None) -> pd.DataFrame:
    """
    执行采样流程，供外部调用或脚本直接运行。

    Args:
        n: 采样数量
        seed: 随机种子
        output: 输出文件路径（可选）
        dataset: 数据集CSV路径（可选）

    Returns:
        采样结果DataFrame
    """
    # 确定数据集路径
    csv_path = Path(dataset) if dataset else IMAGE_DATASET_PATH

    # 设置随机种子
    actual_seed = seed if seed is not None else random.randint(0, 99999)
    print(f"随机种子: {actual_seed}")

    # 加载数据集
    df = load_image_dataset(csv_path)

    # 采样
    samples = sample_dataset(df, n=n, seed=actual_seed)

    # 展示
    display_samples(samples)

    # 统计信息
    print(f"\n数据集规模: {len(df)} 条（框架参考: {DATASET_TOTAL} 条）")
    print(f"训练集/验证集/测试集参考: 800 / 100 / 100")

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
        dataset=args.dataset,
    )