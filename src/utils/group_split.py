"""
Group-based dataset splitting utility.

Ensures all samples sharing the same input (Low_Requirements) are assigned
to the same split, preventing data leakage in retrieval-based baseline
evaluations (Exp1).

Background:
  The text datasets contain many duplicated Low_Requirements with *different*
  Instructions (one requirement → multiple valid instruction variants).  Under
  a naive random split, the same Low_Requirements can appear in both the
  training set (= retrieval index) and the test set, allowing BM25 / LSA to
  find near-exact matches and inflating their scores.

  Group-based splitting assigns all samples that share a Low_Requirements to
  the same partition, so the test set contains only *unseen* requirements.

Usage:
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
    """Split dataset so that all samples with the same *input* stay together.

    Args:
        data:             List of dicts, each containing *input_key* and
                          *output_key* fields.
        train_ratio:      Target fraction of **groups** for training.
        val_ratio:        Target fraction of **groups** for validation.
        test_ratio:       Target fraction of **groups** for testing.
        seed:             Random seed for reproducibility.
        dedup_identical:  If True, drop rows where BOTH input AND output are
                          identical (keeps one copy). Rows with the same
                          input but different output are preserved.
        input_key:        Dict key used as the grouping column.
        output_key:       Dict key used as the output column.

    Returns:
        (train_data, val_data, test_data)
    """
    # ------------------------------------------------------------------
    # 1. Optional: remove fully-identical (input + output) duplicates
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
            print(f"[group_split] Removed {n_removed} fully-identical duplicates "
                  f"({len(data)} → {len(cleaned)})")
        data = cleaned

    # ------------------------------------------------------------------
    # 2. Group by input
    # ------------------------------------------------------------------
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in data:
        groups[item[input_key]].append(item)

    # Sort keys first (determinism), then shuffle
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    # ------------------------------------------------------------------
    # 3. Split groups into train / val / test
    # ------------------------------------------------------------------
    n = len(group_keys)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # test gets the remainder
    train_keys = group_keys[:n_train]
    val_keys = group_keys[n_train : n_train + n_val]
    test_keys = group_keys[n_train + n_val :]

    # ------------------------------------------------------------------
    # 4. Flatten groups → sample lists, shuffle within each split
    # ------------------------------------------------------------------
    train_data = [item for k in train_keys for item in groups[k]]
    val_data = [item for k in val_keys for item in groups[k]]
    test_data = [item for k in test_keys for item in groups[k]]

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    return train_data, val_data, test_data