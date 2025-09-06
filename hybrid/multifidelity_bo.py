"""Multi-fidelity Bayesian optimization (lightweight, dependency‑free).

设计目标:
1. 仅依赖 numpy / torch, 不额外引入 heavy BO 框架。
2. 低保真 (cheap) 与 高保真 (expensive) 之间用一个简化的线性共模: y_H(x) ≈ a + b * y_L(x) + δ(x)
3. δ(x) 用一个简化 RBF 核的核岭回归 (Kernel Ridge) 拟合 residual。
4. 采集策略: 在候选池里基于 (预测均值 + 探索项 κ * σ_pred) 选择下一个高保真点 (最大化目标)。
5. 适应 维度稀疏: 只在已筛出的 support set 子空间中做建模与采集, 其它维度保持固定或默认值。

注意: 这里的接口统一为 “最大化” 形式。如果外部是最小化, 传入之前请自行取负。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict
import numpy as np
import torch

ArrayLike = np.ndarray


def _normalize(X: ArrayLike, bounds: ArrayLike) -> ArrayLike:
    lo = bounds[:, 0]; hi = bounds[:, 1]
    return (X - lo) / (hi - lo + 1e-12)


def _rbf_kernel(X1: ArrayLike, X2: ArrayLike, lengthscale: float) -> ArrayLike:
    # (n1,d),(n2,d) -> (n1,n2)
    d2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * d2 / (lengthscale ** 2 + 1e-12))


@dataclass
class SurrogateModel:
    """简单核岭回归 surrogate (RBF)。"""
    X: ArrayLike  # normalized
    y: ArrayLike  # (n,)
    alpha: float  # (n,) dual 权重
    K_inv: ArrayLike  # (n,n)
    lengthscale: float
    noise: float

    def predict(self, Xq: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        Kx = _rbf_kernel(Xq, self.X, self.lengthscale)  # (m,n)
        mean = Kx @ self.alpha
        # var = k(x,x) - Kx K_inv Kx^T (对角)
        K_inv_KxT = self.K_inv @ Kx.T  # (n,m)
        diag = np.ones(Xq.shape[0])  # k(x,x)=1 for normalized RBF
        var = np.maximum(0.0, diag - np.sum(Kx * K_inv_KxT.T, axis=1))
        return mean, var


def fit_surrogate(X: ArrayLike, y: ArrayLike, noise: float = 1e-6) -> SurrogateModel:
    # 估计 lengthscale: median pairwise distance
    if X.shape[0] < 2:
        ls = 0.5
    else:
        # 采样一部分对减少 O(n^2)
        idx = np.random.choice(X.shape[0], size=min(200, X.shape[0]), replace=False)
        Xs = X[idx]
        d2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
        dists = np.sqrt(d2 + 1e-12)
        ls = np.median(dists[dists > 0])
        if not np.isfinite(ls) or ls <= 0:
            ls = 0.5
    K = _rbf_kernel(X, X, ls)
    K[np.diag_indices_from(K)] += noise
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        # fallback: add jitter
        K[np.diag_indices_from(K)] += 1e-6
        K_inv = np.linalg.pinv(K)
    alpha = K_inv @ y
    return SurrogateModel(X=X, y=y, alpha=alpha, K_inv=K_inv, lengthscale=ls, noise=noise)


@dataclass
class MFBOConfig:
    bounds: List[Tuple[float, float]]
    support_dims: List[int]
    cheap_pool_size: int = 2048          # 低保真池大小
    initial_expensive: int = 8           # 初始高保真点数
    max_expensive: int = 64             # 总高保真预算
    kappa: float = 2.0                  # UCB 探索系数
    random_exp_frac: float = 0.1        # 随机探索比例
    refine_topk: int = 128              # 候选池中挑选 topK 做采集打分
    seed: Optional[int] = None
    # 新增性能相关参数
    max_low_model_points: int = 1024     # 训练低保真 surrogate 的最大点数 (随机子集); -1 使用全部
    improve_patience: int = 5            # 连续若干次未显著提升则提前停止
    improve_tol: float = 1e-4            # 提升阈值 (y 增加不足此值视为未改进)


@dataclass
class MFBOResult:
    best_x: ArrayLike
    best_y_expensive: float
    history: Dict[str, ArrayLike]


class MultiFidelityBO:
    def __init__(self, config: MFBOConfig, cheap_eval: Callable[[ArrayLike], ArrayLike], expensive_eval: Callable[[ArrayLike], ArrayLike]):
        self.cfg = config
        self.cheap_eval = cheap_eval
        self.expensive_eval = expensive_eval
        self.bounds = np.array(config.bounds, dtype=float)
        self.d = self.bounds.shape[0]
        self.support_dims = sorted(config.support_dims)
        if config.seed is not None:
            np.random.seed(config.seed)
        # 固定支撑维度之外的其它维度取中点 (可后续暴露)
        self.fixed_x = self.bounds.mean(axis=1)

    def _sample_uniform(self, n: int) -> ArrayLike:
        lo = self.bounds[:, 0]; hi = self.bounds[:, 1]
        return lo + np.random.rand(n, self.d) * (hi - lo)

    def run(self) -> MFBOResult:
        cfg = self.cfg
        # 1) 构建低保真候选池
        pool = self._sample_uniform(cfg.cheap_pool_size)
        y_low_pool = self.cheap_eval(pool)  # (N,)
        assert y_low_pool.shape[0] == pool.shape[0]
        # 2) 选初始高保真点: 低保真排序 + 随机探索
        n_init = min(cfg.initial_expensive, cfg.max_expensive)
        order_low = np.argsort(-y_low_pool)
        n_rank = int(n_init * (1 - cfg.random_exp_frac))
        topk_idx = order_low[:n_rank]
        remain = n_init - n_rank
        if remain > 0:
            rand_idx = np.random.choice(order_low[n_rank:], size=remain, replace=False)
            init_idx = np.concatenate([topk_idx, rand_idx])
        else:
            init_idx = topk_idx
        X_exp = pool[init_idx]
        y_low_exp = y_low_pool[init_idx]
        y_exp = self.expensive_eval(X_exp)

        # 3) 预构建低保真 surrogate (一次性)
        norm_pool_full = _normalize(pool[:, self.support_dims], self.bounds[self.support_dims])
        if cfg.max_low_model_points > 0 and norm_pool_full.shape[0] > cfg.max_low_model_points:
            sub_sel = np.random.choice(norm_pool_full.shape[0], cfg.max_low_model_points, replace=False)
            low_model = fit_surrogate(norm_pool_full[sub_sel], y_low_pool[sub_sel])
        else:
            low_model = fit_surrogate(norm_pool_full, y_low_pool)
        # 缓存对整个 pool 的预测 (减少重复 kernel)
        low_mu_pool, low_var_pool = low_model.predict(norm_pool_full)

        # 4) 主循环 (仅 residual + acquisition 更新)
        visited = np.zeros(pool.shape[0], dtype=bool)
        visited[init_idx] = True
        best_idx_local = int(np.argmax(y_exp))
        best_val_local = float(y_exp[best_idx_local])
        no_improve = 0
        history = {
            'X_exp': [X_exp.copy()],
            'y_exp': [y_exp.copy()],
            'y_low_exp': [y_low_exp.copy()],
        }

        while X_exp.shape[0] < cfg.max_expensive:
            # 4.1) residual surrogate
            y_low_e = y_low_exp
            y_e = y_exp
            A = np.vstack([np.ones_like(y_low_e), y_low_e]).T
            coef, *_ = np.linalg.lstsq(A, y_e, rcond=None)
            a, b = coef
            residual = y_e - (a + b * y_low_e)
            if residual.shape[0] >= 2:
                norm_Xe = _normalize(X_exp[:, self.support_dims], self.bounds[self.support_dims])
                res_model = fit_surrogate(norm_Xe, residual, noise=1e-5)
            else:
                res_model = None

            # 4.2) 候选子集索引 (未访问)
            cand_idx = np.where(~visited)[0]
            if cand_idx.size == 0:
                break
            if cand_idx.size > cfg.refine_topk:
                # 先按低保真值筛一个窗口 (快速) — 选出 top refine_topk
                order = np.argsort(-y_low_pool[cand_idx])[:cfg.refine_topk]
                cand_idx = cand_idx[order]
            # 取缓存的低保真预测
            mu_low = low_mu_pool[cand_idx]
            var_low = low_var_pool[cand_idx]
            # residual 预测
            if res_model is not None:
                norm_sub = norm_pool_full[cand_idx][:, :]  # already normalized subset
                mu_res, var_res = res_model.predict(norm_sub)
            else:
                mu_res = np.zeros_like(mu_low); var_res = np.full_like(var_low, 0.05)
            mu_high = a + b * mu_low + mu_res
            var_high = (b ** 2) * var_low + var_res
            std_high = np.sqrt(np.maximum(1e-12, var_high))
            acq = mu_high + cfg.kappa * std_high
            if np.random.rand() < cfg.random_exp_frac:
                pick_global = np.random.choice(cand_idx)
            else:
                pick_global = cand_idx[int(np.argmax(acq))]
            x_new = pool[pick_global:pick_global+1]
            y_low_new = y_low_pool[pick_global:pick_global+1]
            y_exp_new = self.expensive_eval(x_new)
            # 更新
            X_exp = np.vstack([X_exp, x_new])
            y_low_exp = np.concatenate([y_low_exp, y_low_new])
            y_exp = np.concatenate([y_exp, y_exp_new])
            visited[pick_global] = True
            history['X_exp'].append(x_new.copy())
            history['y_exp'].append(y_exp_new.copy())
            history['y_low_exp'].append(y_low_new.copy())
            cur_best_iter = float(np.max(y_exp))
            if cur_best_iter > best_val_local + cfg.improve_tol:
                best_val_local = cur_best_iter
                best_idx_local = int(np.argmax(y_exp))
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= cfg.improve_patience:
                break

        return MFBOResult(best_x=X_exp[best_idx_local], best_y_expensive=float(best_val_local), history=history)


def coordinate_refine(x0: ArrayLike, bounds: List[Tuple[float, float]], eval_func: Callable[[ArrayLike], ArrayLike], support_dims: List[int], n_passes: int = 3, points_per_dim: int = 9) -> Tuple[ArrayLike, float]:
    """简单坐标搜索 (只在 support_dims 上) 进一步精化。

    Args:
        x0: 初始解 (d,)
        bounds: 原始 bounds
        eval_func: 高保真评估 (batch)
        support_dims: 待优化维度索引
    Returns: (best_x, best_val)
    """
    x_best = x0.copy();
    best_val = float(eval_func(x_best[None, :])[0])
    b_arr = np.array(bounds, dtype=float)
    for _ in range(n_passes):
        improved = False
        for d_idx in support_dims:
            lo, hi = b_arr[d_idx]
            grid = np.linspace(lo, hi, points_per_dim)
            Xc = np.tile(x_best, (points_per_dim, 1))
            Xc[:, d_idx] = grid
            vals = eval_func(Xc)
            m = int(np.argmax(vals))
            if float(vals[m]) > best_val + 1e-12:
                best_val = float(vals[m])
                x_best = Xc[m].copy()
                improved = True
        if not improved:
            break
    return x_best, best_val
