"""
src/routing/learned_router.py

Learned Router: 基于MLP分类器的数据驱动专家路由模块

用途：
  - 替代 expert_router.py 的规则路由（Hard Routing）
  - 从 Qwen3-8B 最后一层 hidden state 提取特征
  - MLP 分类器预测最优专家（text/image/uml/general）
  - 输出的概率分布可直接作为 Output Ensemble 的融合权重

与现有模块的关系：
  - expert_router.py  : Hard Routing（规则），不依赖本模块
  - soft_router.py    : Soft Routing（LoRA参数融合），不依赖本模块
  - learned_router.py : Learned Routing（MLP），本模块
  - exp10 同时使用 learned_router + soft_router 的 logit 融合思路

训练数据来源：
  - outputs/evaluations/experiments/exp9_routing_strategy/phase1_results.json
  - oracle_selections 字段中的逐域最优专家标签

权重保存路径：
  - checkpoints/exp10_learned_router/router_mlp_best.pt

Author: Claude
Date: 2026-03-08
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 专家索引映射
EXPERT_TO_IDX: Dict[str, int] = {
    'text': 0,
    'image': 1,
    'uml': 2,
    'general': 3,
}
IDX_TO_EXPERT: Dict[int, str] = {v: k for k, v in EXPERT_TO_IDX.items()}
ALL_EXPERTS: List[str] = ['text', 'image', 'uml', 'general']


class RouterMLP:
    """
    轻量级 MLP 路由分类器

    架构：
        Linear(input_dim, 512) → LayerNorm → ReLU → Dropout(0.2)
        → Linear(512, 128) → LayerNorm → ReLU → Dropout(0.1)
        → Linear(128, 4) → Softmax

    总参数量约 2.1M（input_dim=4096 时）

    使用示例：
        router = RouterMLP()
        router.load("checkpoints/exp10_learned_router/router_mlp_best.pt")

        # 特征提取
        extractor = HiddenStateExtractor(base_model, tokenizer)
        features = extractor.extract(inputs)   # (N, 4096)

        # 路由预测
        probs = router.predict_proba(features)  # (N, 4)
        experts = router.predict(features)      # (N,) int索引
        expert_names = router.predict_names(features)  # (N,) str名称
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden1: int = 512,
        hidden2: int = 128,
        num_classes: int = 4,
        dropout1: float = 0.2,
        dropout2: float = 0.1,
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise RuntimeError("RouterMLP 需要安装 PyTorch")

        self.input_dim = input_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden1),
                    nn.LayerNorm(hidden1),
                    nn.ReLU(),
                    nn.Dropout(dropout1),
                    nn.Linear(hidden1, hidden2),
                    nn.LayerNorm(hidden2),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(hidden2, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        self.model = _MLP().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"RouterMLP 初始化完成 | 参数量: {total_params:,} | 设备: {self.device}")

    # ──────────────────────────────────────
    # 权重管理
    # ──────────────────────────────────────

    def save(self, path) -> None:
        """保存模型权重（含 input_dim 元数据，供 load 时自动适配）"""
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {'state_dict': self.model.state_dict(), 'input_dim': self.input_dim},
            path,
        )
        logger.info(f"Router 已保存: {path} (input_dim={self.input_dim})")

    def load(self, path) -> bool:
        """
        加载模型权重

        同时支持新格式（dict 含 state_dict + input_dim）和旧格式（裸 state_dict）。
        当 checkpoint 中的 input_dim 与当前实例不一致时，自动重建 MLP 以匹配维度。

        Returns:
            bool: 加载是否成功
        """
        import torch
        path = Path(path)
        if not path.exists():
            logger.error(f"Router 权重文件不存在: {path}")
            return False
        try:
            ckpt = torch.load(path, map_location=self.device)

            # 区分新格式（dict with metadata）和旧格式（裸 state_dict）
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                ckpt_input_dim = ckpt.get('input_dim', self.input_dim)
            else:
                # 旧格式兼容：直接是 state_dict
                state_dict = ckpt
                ckpt_input_dim = self.input_dim
                logger.warning(
                    f"Router 权重为旧格式（不含 input_dim 元数据），"
                    f"假设 input_dim={self.input_dim}"
                )

            # 若 input_dim 不符，重建 MLP 以匹配 checkpoint 维度
            if ckpt_input_dim != self.input_dim:
                logger.warning(
                    f"input_dim 不匹配（当前={self.input_dim}，"
                    f"checkpoint={ckpt_input_dim}），重建 MLP..."
                )
                self.__init__(input_dim=ckpt_input_dim)

            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Router 已加载: {path} (input_dim={self.input_dim})")
            return True
        except Exception as e:
            logger.error(f"Router 加载失败: {e}")
            return False

    # ──────────────────────────────────────
    # 推理接口
    # ──────────────────────────────────────

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        返回各专家的路由概率分布

        Args:
            features: np.ndarray, shape (N, input_dim)，L2归一化后的hidden state

        Returns:
            probs: np.ndarray, shape (N, 4)，各专家概率（text/image/uml/general）
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        返回最优专家的整数索引

        Args:
            features: np.ndarray, shape (N, input_dim)

        Returns:
            indices: np.ndarray, shape (N,)，值为 0~3
        """
        probs = self.predict_proba(features)
        return np.argmax(probs, axis=1)

    def predict_names(self, features: np.ndarray) -> List[str]:
        """
        返回最优专家的名称列表

        Args:
            features: np.ndarray, shape (N, input_dim)

        Returns:
            names: List[str]，如 ['text', 'general', 'uml', ...]
        """
        indices = self.predict(features)
        return [IDX_TO_EXPERT[int(i)] for i in indices]

    def predict_top2(
        self, features: np.ndarray, collapse_threshold: float = 0.85
    ) -> List[Tuple[str, str, float, float]]:
        """
        返回 top-2 专家及其归一化权重，供 Output Ensemble 使用

        当 top-1 概率 >= collapse_threshold 时退化为单专家（w1=1.0, w2=0.0）

        Args:
            features: np.ndarray, shape (N, input_dim)
            collapse_threshold: float，退化为单专家的阈值（默认 0.85）

        Returns:
            List of (expert1_name, expert2_name, w1, w2)
            当退化时 expert2_name = expert1_name, w1 = 1.0, w2 = 0.0
        """
        probs = self.predict_proba(features)
        results = []

        for prob in probs:
            top2_idxs = np.argsort(prob)[::-1][:2]
            e1, e2 = IDX_TO_EXPERT[top2_idxs[0]], IDX_TO_EXPERT[top2_idxs[1]]
            w1, w2 = float(prob[top2_idxs[0]]), float(prob[top2_idxs[1]])

            if w1 >= collapse_threshold:
                results.append((e1, e1, 1.0, 0.0))
            else:
                w_sum = w1 + w2
                results.append((e1, e2, w1 / w_sum, w2 / w_sum))

        return results

    def get_routing_stats(self, features: np.ndarray) -> Dict[str, int]:
        """统计各专家被路由到的次数"""
        names = self.predict_names(features)
        stats = {e: 0 for e in ALL_EXPERTS}
        for name in names:
            stats[name] += 1
        return stats


class HiddenStateExtractor:
    """
    从 Qwen3-8B 提取最后一层 hidden state 作为路由特征

    提取策略：取每条序列最后一个有效（非padding）token 的隐状态，L2 归一化

    使用示例：
        from models.language_model import LanguageModel
        lm = LanguageModel(use_4bit=True)
        extractor = HiddenStateExtractor(lm.model, lm.tokenizer)
        features = extractor.extract(inputs, batch_size=4)
        # features.shape == (N, 4096)
    """

    def __init__(self, base_model, tokenizer, max_length: int = 512):
        """
        Args:
            base_model: 已加载的 Qwen3-8B 模型（frozen，不加载LoRA）
            tokenizer: 对应的 tokenizer
            max_length: 输入截断长度
        """
        self.model = base_model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def extract(
        self,
        inputs: List[str],
        batch_size: int = 4,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        批量提取 hidden states

        Args:
            inputs: 输入文本列表
            batch_size: 批次大小
            normalize: 是否 L2 归一化（建议开启，MLP 输入更稳定）

        Returns:
            features: np.ndarray, shape (N, hidden_size)
        """
        import torch

        self.model.eval()
        all_features = []
        total = len(inputs)

        for i in range(0, total, batch_size):
            batch = inputs[i: i + batch_size]
            batch_features = self._extract_batch(batch)
            all_features.append(batch_features)

            if (i // batch_size) % 20 == 0:
                logger.info(f"  特征提取进度: {min(i + batch_size, total)}/{total}")

        features = np.concatenate(all_features, axis=0)

        if normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            features = features / norms

        logger.info(f"特征提取完成: shape={features.shape}, normalized={normalize}")
        return features

    def _extract_batch(self, batch: List[str]) -> np.ndarray:
        """提取单个 batch 的 hidden states"""
        import torch

        try:
            encoded = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encoded['input_ids'].to(self.model.device)
            attention_mask = encoded['attention_mask'].to(self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # 最后一层 hidden state: (B, seq_len, hidden_size)
                last_hidden = outputs.hidden_states[-1]

                # 每条序列最后一个有效 token 的索引
                seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
                batch_features = last_hidden[
                    torch.arange(len(batch)), seq_lens, :
                ].cpu().float().numpy()   # (B, hidden_size)

            return batch_features

        except Exception as e:
            logger.error(f"  batch 特征提取失败: {e}")
            hidden_size = self.model.config.hidden_size
            return np.zeros((len(batch), hidden_size), dtype=np.float32)

    def extract_and_save(
        self,
        inputs: List[str],
        save_path,
        labels: Optional[List[int]] = None,
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        提取特征并保存到 .npz 文件

        Args:
            inputs: 输入文本列表
            save_path: 保存路径（.npz 格式）
            labels: 可选标签列表（Oracle专家索引）
            batch_size: 批次大小

        Returns:
            features: np.ndarray
        """
        features = self.extract(inputs, batch_size=batch_size)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if labels is not None:
            np.savez(save_path, features=features, labels=np.array(labels, dtype=np.int64))
        else:
            np.savez(save_path, features=features)

        logger.info(f"特征已保存: {save_path} (features={features.shape})")
        return features


class LearnedRouterInference:
    """
    完整的学习路由推理流程封装

    整合 HiddenStateExtractor + RouterMLP，对外提供一个高层接口，
    供 exp10 和未来推理脚本直接调用。

    使用示例：
        router_inf = LearnedRouterInference(
            base_model=lm.model,
            tokenizer=lm.tokenizer,
            router_ckpt="checkpoints/exp10_learned_router/router_mlp_best.pt",
        )
        # 单条推理
        expert_name = router_inf.route_single("需求描述文本")

        # 批量路由
        expert_names = router_inf.route_batch(inputs)

        # 获取 Ensemble 权重
        top2_list = router_inf.get_ensemble_weights(inputs)
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        router_ckpt,
        input_dim: int = 4096,
        max_length: int = 512,
        collapse_threshold: float = 0.85,
        feature_cache_path: Optional[str] = None,
    ):
        """
        Args:
            base_model: Qwen3-8B 模型（frozen）
            tokenizer: 对应 tokenizer
            router_ckpt: RouterMLP 权重路径
            input_dim: MLP 输入维度（默认 4096，与 Qwen3-8B hidden size 一致）
            max_length: 特征提取时的最大输入长度
            collapse_threshold: top-1 概率超过此值时退化为单专家
            feature_cache_path: 特征缓存路径（可选，避免重复提取）
        """
        self.extractor = HiddenStateExtractor(base_model, tokenizer, max_length)
        # input_dim 作为初始猜测值；若 checkpoint 中记录了不同的 input_dim，
        # RouterMLP.load() 会自动检测并重建模型，无需调用方手动传入正确维度。
        self.router = RouterMLP(input_dim=input_dim)
        self.collapse_threshold = collapse_threshold
        self.feature_cache_path = Path(feature_cache_path) if feature_cache_path else None

        if not self.router.load(router_ckpt):
            raise RuntimeError(f"Router 权重加载失败: {router_ckpt}")

        # 加载后同步实际使用的 input_dim（可能因 checkpoint 而被重建）
        logger.info(
            f"LearnedRouterInference 初始化完成 | "
            f"input_dim={self.router.input_dim} | "
            f"collapse_threshold={collapse_threshold}"
        )

    def route_single(self, input_text: str) -> str:
        """
        单条输入路由，返回专家名称

        Args:
            input_text: 原始输入文本

        Returns:
            expert_name: 'text' / 'image' / 'uml' / 'general'
        """
        features = self.extractor.extract([input_text], batch_size=1)
        return self.router.predict_names(features)[0]

    def route_batch(self, inputs: List[str], batch_size: int = 4) -> List[str]:
        """
        批量路由，返回每条输入对应的专家名称

        Args:
            inputs: 输入文本列表
            batch_size: 特征提取批次大小

        Returns:
            expert_names: List[str]
        """
        features = self._get_features(inputs, batch_size)
        return self.router.predict_names(features)

    def get_ensemble_weights(
        self, inputs: List[str], batch_size: int = 4
    ) -> List[Tuple[str, str, float, float]]:
        """
        获取 Output Ensemble 所需的 top-2 专家权重

        Args:
            inputs: 输入文本列表
            batch_size: 特征提取批次大小

        Returns:
            List of (expert1, expert2, w1, w2)
        """
        features = self._get_features(inputs, batch_size)
        return self.router.predict_top2(features, self.collapse_threshold)

    def get_routing_probs(
        self, inputs: List[str], batch_size: int = 4
    ) -> np.ndarray:
        """
        返回完整的概率分布矩阵

        Returns:
            probs: np.ndarray, shape (N, 4)
        """
        features = self._get_features(inputs, batch_size)
        return self.router.predict_proba(features)

    def _get_features(self, inputs: List[str], batch_size: int) -> np.ndarray:
        """从缓存加载或重新提取特征"""
        if self.feature_cache_path and self.feature_cache_path.exists():
            try:
                data = np.load(self.feature_cache_path)
                features = data['features']
                if len(features) == len(inputs):
                    logger.debug(f"特征从缓存加载: {self.feature_cache_path}")
                    return features
            except Exception:
                pass
        return self.extractor.extract(inputs, batch_size=batch_size)


def load_router_from_checkpoint(
    base_model,
    tokenizer,
    ckpt_path: str = "checkpoints/exp10_learned_router/router_mlp_best.pt",
    **kwargs,
) -> LearnedRouterInference:
    """
    便捷工厂函数，快速构建 LearnedRouterInference 实例

    Args:
        base_model: 已加载的 Qwen3-8B（frozen）
        tokenizer: 对应 tokenizer
        ckpt_path: Router 权重路径
        **kwargs: 传给 LearnedRouterInference 的其他参数

    Returns:
        LearnedRouterInference 实例
    """
    return LearnedRouterInference(
        base_model=base_model,
        tokenizer=tokenizer,
        router_ckpt=ckpt_path,
        **kwargs,
    )