"""
基于分组的数据集划分工具

确保所有共享相同输入（Low_Requirements）的样本被分配到同一划分，
防止在基于检索的基线评估（Exp1）中出现数据泄漏。

背景：
  文本数据集中存在大量重复的 Low_Requirements，但对应不同的
  Instruction（一个需求对应多种有效指令变体）。若使用朴素随机划分，
  相同的 Low_Requirements 可能同时出现在训练集（检索索引）和测试集中，
  使 BM25 / LSA 等方法能找到近似精确匹配，从而虚高其评分。

  基于分组的划分将所有共享同一 Low_Requirements 的样本统一分配到
  同一分区，确保测试集只包含训练时从未见过的需求。

用法：
  from src.utils.group_split import group_split_by_input
  train, val, test = group_split_by_input(all_data)
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple


def group_split_by_input(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dedup_identical: bool = True,
    input_key: str = "input",
    output_key: str = "output",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按相同输入将数据集分组后进行划分，确保同一输入的所有样本归属同一分区。

    Args:
        data:             字典列表，每项包含 *input_key* 和 *output_key* 字段。
        train_ratio:      训练集占**分组数**的目标比例。
        val_ratio:        验证集占**分组数**的目标比例。
        test_ratio:       测试集占**分组数**的目标比例。
        seed:             随机种子，用于复现性。
        dedup_identical:  若为 True，则删除输入和输出完全相同的重复行（保留一条）。
                          输入相同但输出不同的行将被保留。
        input_key:        作为分组依据的字典键。
        output_key:       作为输出列的字典键。

    Returns:
        (train_data, val_data, test_data)
    """
    # ------------------------------------------------------------------
    # 1. 可选：删除输入+输出完全相同的重复行
    # ------------------------------------------------------------------
    if dedup_identical:
        seen = set()
        cleaned: List[Dict] = []
        for item in data:
            key = (item[input_key], item[output_key])
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        n_removed = len(data) - len(cleaned)
        if n_removed:
            print(f"[group_split] 已删除 {n_removed} 条完全重复行 "
                  f"({len(data)} → {len(cleaned)})")
        data = cleaned

    # ------------------------------------------------------------------
    # 2. 按输入分组
    # ------------------------------------------------------------------
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in data:
        groups[item[input_key]].append(item)

    # 先排序保证确定性，再随机打乱
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    # ------------------------------------------------------------------
    # 3. 将分组划分为训练集 / 验证集 / 测试集
    # ------------------------------------------------------------------
    n = len(group_keys)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # 测试集取剩余部分
    train_keys = group_keys[:n_train]
    val_keys = group_keys[n_train : n_train + n_val]
    test_keys = group_keys[n_train + n_val :]

    # ------------------------------------------------------------------
    # 4. 将分组展开为样本列表，并在各划分内部随机打乱
    # ------------------------------------------------------------------
    train_data = [item for k in train_keys for item in groups[k]]
    val_data = [item for k in val_keys for item in groups[k]]
    test_data = [item for k in test_keys for item in groups[k]]

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    return train_data, val_data, test_data