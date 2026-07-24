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


class FocalLoss:
    """
    Focal Loss：聚焦难分样本，专项解决 text-general 边界模糊导致的 general recall 低问题

    FL(p_t) = -alpha_t · (1 - p_t)^gamma · log(p_t)

    ── 为什么 CrossEntropyLoss + 类权重还不够 ─────────────────────────────────
    CrossEntropyLoss 对所有样本同等惩罚（经类权重缩放后）。
    然而 general 样本被误分为 text 时，模型对 text 的置信度往往已经较高（0.5~0.7），
    常规 CE 对这类"中等置信度误分"的惩罚力度仍然不足：
      - CE 损失 ≈ -log(0.3) = 1.2（general 被以 70% 置信度错判为 text）
      - 类权重 1.86 × 1.2 = 2.2，梯度信号仍偏弱

    Focal Loss 在此基础上乘以 (1-p_t)^gamma：
      - p_t=0.3 时，focal weight = (1-0.3)^2 = 0.49，损失 = 0.49 × 1.2 = 0.59
      - p_t=0.1 时，focal weight = (1-0.1)^2 = 0.81，损失 = 0.81 × 2.3 = 1.86
    即越难分的样本，gamma 项越大，梯度越强（而非越小）。

    ── 实现为纯函数包装而非 nn.Module ──────────────────────────────────────────
    避免在 exp10._train_router 中引入额外的 Module 注册，保持调用侧简洁。
    FocalLoss 实例可直接替换 nn.CrossEntropyLoss() 使用：
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
        loss = criterion(logits, targets)
    """

    def __init__(self, alpha=None, gamma: float = 2.0):
        """
        Args:
            alpha: 类别权重张量 (num_classes,)，已移至正确 device；
                   传入前应已做逆频率归一化，与 CrossEntropyLoss(weight=...) 语义一致
            gamma: 聚焦参数，值越大对难分样本的权重放大越强，推荐 1.5~2.5
                   gamma=0 退化为加权 CrossEntropyLoss
        """
        self.alpha = alpha  # torch.Tensor or None
        self.gamma = gamma

    def __call__(self, logits, targets):
        """
        Args:
            logits: (N, num_classes)，模型原始输出（未经 softmax）
            targets: (N,)，整数类别标签

        Returns:
            scalar loss
        """
        import torch
        import torch.nn.functional as F

        # 移动 alpha 到当前 logits 所在设备（首次 __call__ 时自动对齐）
        alpha = self.alpha
        if alpha is not None and alpha.device != logits.device:
            alpha = alpha.to(logits.device)
            self.alpha = alpha  # 缓存，避免每步重复搬运

        # per-sample 标准 CE 损失（含类权重）
        ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction='none')

        # p_t：模型对正确类别的预测概率
        # 利用 CE=-log(p_t) 反推，避免重复 softmax
        with torch.no_grad():
            p_t = torch.exp(-ce_loss)

        # Focal weight：难分样本（p_t 低）自动获得更大权重
        focal_weight = (1.0 - p_t) ** self.gamma

        return (focal_weight * ce_loss).mean()


class RouterMLP:
    """
    带残差连接的 MLP 路由分类器

    ── 架构说明 ──────────────────────────────────────────────────────────────
    Block1: Linear(input_dim→512) → LayerNorm → GELU → Dropout(0.15)
    Block2: Linear(512→256)       → LayerNorm → GELU → Dropout(0.10)
    Skip:   Linear(512→256, bias=False)   ← 残差跳跃投影
    Add:    Block2_out + Skip_out
    Head:   Linear(256→4)

    相比原始 Sequential 架构的改进：
    1. hidden2: 128 → 256
       原来 128 维决策层对 text/general 的特征边界（高维、连续）表达能力不足；
       256 维提供更充裕的分类空间，捕捉两类之间的细粒度差异。
    2. 残差 Skip 连接（512→256）
       Block1 提取的粗粒度域特征可以绕过 Block2 直接传入分类头；
       允许 Head 同时参考"粗粒度原始信号"和"Block2 的精炼信号"，
       类似 ResNet 里 skip 有助于训练稳定、防止梯度消失。
    3. ReLU → GELU
       GELU 在零点附近的平滑梯度对小样本（64条 general）更友好，
       避免 ReLU 的"死神经元"问题导致 general 方向的梯度永久消失。

    总参数量约 2.2M（input_dim=4096 时，略多于原 2.1M）

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
        hidden2: int = 256,
        num_classes: int = 4,
        dropout1: float = 0.15,
        dropout2: float = 0.10,
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise RuntimeError("RouterMLP 需要安装 PyTorch")

        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                # Block1：高维压缩，提取粗粒度域特征
                self.block1 = nn.Sequential(
                    nn.Linear(input_dim, hidden1),
                    nn.LayerNorm(hidden1),
                    nn.GELU(),
                    nn.Dropout(dropout1),
                )
                # Block2：特征精炼，学习类间细粒度边界
                self.block2 = nn.Sequential(
                    nn.Linear(hidden1, hidden2),
                    nn.LayerNorm(hidden2),
                    nn.GELU(),
                    nn.Dropout(dropout2),
                )
                # Skip：残差跳跃投影，无 bias 避免额外偏置干扰 LayerNorm
                # 梯度可绕过 Block2 直接回传，缓解 text-general 边界的梯度消失
                self.skip = nn.Linear(hidden1, hidden2, bias=False)
                # 分类头
                self.head = nn.Linear(hidden2, num_classes)

            def forward(self, x):
                h1 = self.block1(x)
                # 残差融合：精炼特征 + 跳跃连接，共同送入分类头
                h2 = self.block2(h1) + self.skip(h1)
                return self.head(h2)

        self.model = _MLP().to(self.device)
        # 后处理校准偏置：对每个类别的 logit 添加一个标量偏置，
        # 在验证集上通过坐标下降搜索，使 macro-F1 最大化。
        # 零向量 = 无校准（默认行为与未校准一致）。
        self.calibration_offsets: np.ndarray = np.zeros(num_classes, dtype=np.float32)
        self.num_classes = num_classes
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"RouterMLP 初始化完成 | 参数量: {total_params:,} | 设备: {self.device} | "
            f"架构: {input_dim}→{hidden1}→(residual)→{hidden2}→{num_classes}"
        )

    # ──────────────────────────────────────
    # 权重管理
    # ──────────────────────────────────────

    def save(self, path) -> None:
        """保存模型权重（含架构元数据，供 load 时精确重建）"""
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden1': self.hidden1,
                'hidden2': self.hidden2,
                'calibration_offsets': self.calibration_offsets,
            },
            path,
        )
        cal_nonzero = np.any(self.calibration_offsets != 0)
        logger.info(
            f"Router 已保存: {path} "
            f"(input_dim={self.input_dim}, hidden1={self.hidden1}, hidden2={self.hidden2}, "
            f"calibrated={'yes' if cal_nonzero else 'no'})"
        )

    def load(self, path) -> bool:
        """
        加载模型权重

        支持三种格式：
          1. 新格式（含 input_dim + hidden1 + hidden2）：精确重建架构后加载
          2. 中间格式（含 input_dim，不含 hidden1/hidden2）：
             假设旧默认值 hidden1=512, hidden2=128 重建
          3. 旧格式（裸 state_dict）：使用当前实例参数，并给出警告

        Returns:
            bool: 加载是否成功
        """
        import torch
        path = Path(path)
        if not path.exists():
            logger.error(f"Router 权重文件不存在: {path}")
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)

            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                ckpt_input_dim = ckpt.get('input_dim', self.input_dim)
                ckpt_hidden1 = ckpt.get('hidden1', 512)   # 旧checkpoint默认值
                ckpt_hidden2 = ckpt.get('hidden2', 128)   # 旧checkpoint默认值
            else:
                state_dict = ckpt
                ckpt_input_dim = self.input_dim
                ckpt_hidden1, ckpt_hidden2 = self.hidden1, self.hidden2
                logger.warning(
                    f"Router 权重为旧格式（裸 state_dict），"
                    f"假设 input_dim={self.input_dim}, hidden1={self.hidden1}, hidden2={self.hidden2}"
                )

            # 若架构参数不匹配，重建 MLP
            if (ckpt_input_dim != self.input_dim
                    or ckpt_hidden1 != self.hidden1
                    or ckpt_hidden2 != self.hidden2):
                logger.warning(
                    f"架构不匹配（当前: {self.input_dim}/{self.hidden1}/{self.hidden2}，"
                    f"checkpoint: {ckpt_input_dim}/{ckpt_hidden1}/{ckpt_hidden2}），重建 MLP..."
                )
                self.__init__(
                    input_dim=ckpt_input_dim,
                    hidden1=ckpt_hidden1,
                    hidden2=ckpt_hidden2,
                )

            self.model.load_state_dict(state_dict)
            self.model.eval()

            # 恢复校准偏置（旧 checkpoint 不含此字段时保持零向量）
            if isinstance(ckpt, dict) and 'calibration_offsets' in ckpt:
                self.calibration_offsets = np.array(
                    ckpt['calibration_offsets'], dtype=np.float32
                )
            else:
                self.calibration_offsets = np.zeros(self.num_classes, dtype=np.float32)

            cal_nonzero = np.any(self.calibration_offsets != 0)
            logger.info(
                f"Router 已加载: {path} "
                f"(input_dim={self.input_dim}, hidden1={self.hidden1}, hidden2={self.hidden2}, "
                f"calibrated={'yes, offsets=' + str(self.calibration_offsets.round(2)) if cal_nonzero else 'no'})"
            )
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

            # 应用后处理校准偏置（坐标下降在验证集上搜索的最优 per-class 偏移）
            # 零向量时等价于无校准，不影响未设置校准时的行为
            if np.any(self.calibration_offsets != 0):
                offsets_t = torch.tensor(
                    self.calibration_offsets, dtype=torch.float32, device=logits.device
                )
                logits = logits + offsets_t

            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def set_calibration_offsets(self, offsets: np.ndarray) -> None:
        """
        设置后处理校准偏置（由 exp10._calibrate_class_offsets 计算后调用）

        Args:
            offsets: np.ndarray, shape (4,)，每个类别的 logit 偏置值
                     正值 → 提高该类别的预测概率（提升 recall）
                     负值 → 降低该类别的预测概率（提升 precision）
        """
        self.calibration_offsets = np.array(offsets, dtype=np.float32)
        logger.info(
            f"校准偏置已设置: "
            f"text={offsets[0]:+.2f}, image={offsets[1]:+.2f}, "
            f"uml={offsets[2]:+.2f}, general={offsets[3]:+.2f}"
        )

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

                # 加权平均池化（attention mask 作权重）
                # 优于单取最后一个 token：最后 token 的表示受位置偏置影响大，
                # 而平均池化利用了整条序列的信息，对变长输入更鲁棒。
                # 这也是文本分类任务中最常用的句向量提取方式。
                mask = attention_mask.unsqueeze(-1).float()      # (B, seq_len, 1)
                sum_hidden = (last_hidden * mask).sum(dim=1)     # (B, hidden_size)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)       # (B, 1)
                batch_features = (sum_hidden / sum_mask).cpu().float().numpy()  # (B, hidden_size)

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
