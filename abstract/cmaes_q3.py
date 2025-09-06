from __future__ import annotations
"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Problem 3
=====================================================================

目标:
  优化题目3 (单无人机多枚炸弹遮蔽最大化), 决策向量:
      x = [ speed, azimuth, (t1,d1), (t2,d2), ... (tB,dB) ]
  目标最大化 M1 遮蔽时长 -> 这里最小化:  f(x) = -occluded_time["M1"]

特性:
  1. 自适应协方差矩阵演化策略 (CMA-ES)
  2. 自动适应步长控制 (sigma adaptation)
  3. 权重累积演化路径 (evolution paths)
  4. 协方差矩阵更新 (rank-μ 和 rank-one update)
  5. 自动重启机制 (当收敛停滞时)
  6. 边界处理 (mirror 反射)
  7. 可选 LHS 初始化
  8. 完整日志记录
  9. 断点续传支持

CMA-ES 优势:
  - 对多模态、病态条件等复杂优化问题表现优异
  - 自适应步长，无需手动调参
  - 旋转不变性，适应参数间耦合
  - 理论基础扎实，广泛验证

依赖:
  - evaluate_problem3
  - bounds_for_problem

使用:
  from abstract.cmaes_q3 import solve_q3_cmaes
  res = solve_q3_cmaes()
  print(res.best_value, -res.best_value_occlusion_seconds)

命令行测试:
  uv run -m abstract.cmaes_q3

返回:
  CMAESQ3Result:
      best_x: numpy.ndarray
      best_value: float (已经是最小化形式 = -occlusion)
      best_eval: evaluate_problem3 返回的完整结构
      history: list[IterLog]
      meta: dict

说明:
  本实现基于标准 CMA-ES 算法，包含完整的自适应机制和边界处理。
  可与其他优化器进行混合使用或作为多保真优化的基础算法。
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math
import random
import numpy as np
import json
from pathlib import Path

from abstract.occlusion_eval import evaluate_q3_router
from optimizer.spec import bounds_for_problem

# ---------------------------- 数据结构 ----------------------------

@dataclass
class IterLog:
    iter: int
    best_value: float
    mean_value: float
    sigma: float
    condition_number: float
    axis_ratio: float
    evaluations: int

@dataclass
class CMAESQ3Result:
    best_x: np.ndarray
    best_value: float                 # minimized objective = -occlusion
    best_eval: Dict[str, Any]
    history: List[IterLog] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_value_occlusion_seconds(self) -> float:
        # occlusion = -best_value
        return -float(self.best_value)

# ---------------------------- 工具函数 ----------------------------

def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling"""
    if n <= 0:
        return np.zeros((0, d), dtype=np.float64)
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, d))
    a = cut[:-1][:, None]
    b = cut[1:][:, None]
    pts = a + (b - a) * u
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts

def _mirror_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """边界镜像反射处理"""
    x_new = x.copy()

    # 处理下边界
    mask_lo = x_new < lo
    if np.any(mask_lo):
        x_new[mask_lo] = 2 * lo[mask_lo] - x_new[mask_lo]

    # 处理上边界
    mask_hi = x_new > hi
    if np.any(mask_hi):
        x_new[mask_hi] = 2 * hi[mask_hi] - x_new[mask_hi]

    # 如果反射后仍越界，则截断
    x_new = np.clip(x_new, lo, hi)
    return x_new

# ---------------------------- CMA-ES 核心类 ----------------------------

class CMAES:
    def __init__(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        population_size: Optional[int] = None,
        sigma0: float = 0.3,
        mean0: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        self.dim = dim
        self.bounds = bounds
        self.lo = np.array([b[0] for b in bounds], dtype=np.float64)
        self.hi = np.array([b[1] for b in bounds], dtype=np.float64)
        self.span = self.hi - self.lo

        # 设置随机数生成器
        self.rng = np.random.default_rng(seed)

        # 种群大小 (标准设置)
        if population_size is None:
            self.lambda_ = 4 + int(3 * np.log(dim))
        else:
            self.lambda_ = population_size

        self.mu = self.lambda_ // 2  # 选择的父代数量

        # 权重设置
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # 步长控制参数
        self.sigma = sigma0
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.cs

        # 协方差矩阵适应参数
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff))

        # 期望值
        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # 初始化状态变量
        if mean0 is not None:
            self.mean = mean0.copy()
        else:
            self.mean = self.lo + 0.5 * self.span

        self.ps = np.zeros(dim)  # 步长演化路径
        self.pc = np.zeros(dim)  # 协方差演化路径
        self.C = np.eye(dim)     # 协方差矩阵
        self.invsqrtC = np.eye(dim)  # C^(-1/2)

        # 统计变量
        self.eigeneval = 0
        self.counteval = 0
        self.best_value = np.inf
        self.best_x = None

    def ask(self) -> np.ndarray:
        """生成新的候选解"""
        # 特征分解 (定期更新)
        if self.counteval - self.eigeneval > self.lambda_ / (self.c1 + self.cmu) / self.dim / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # 强制对称
            D, B = np.linalg.eigh(self.C)
            D = np.sqrt(np.maximum(D, 1e-14))  # 避免负特征值
            self.invsqrtC = B @ np.diag(1.0 / D) @ B.T

        # 生成候选解
        population = np.zeros((self.lambda_, self.dim))
        for i in range(self.lambda_):
            z = self.rng.standard_normal(self.dim)
            y = self.invsqrtC @ z
            x = self.mean + self.sigma * y
            # 边界处理
            x = _mirror_bounds(x, self.lo, self.hi)
            population[i] = x

        return population

    def tell(self, population: np.ndarray, fitness: np.ndarray):
        """更新CMA-ES状态"""
        self.counteval += len(fitness)

        # 排序选择
        indices = np.argsort(fitness)
        selected = population[indices[:self.mu]]

        # 更新最优解
        if fitness[indices[0]] < self.best_value:
            self.best_value = fitness[indices[0]]
            self.best_x = population[indices[0]].copy()

        # 更新均值
        mean_old = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * selected, axis=0)

        # 演化路径更新
        y = (self.mean - mean_old) / self.sigma
        z = self.invsqrtC @ y

        # 步长演化路径
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z

        # 协方差演化路径
        hsig = (np.linalg.norm(self.ps) /
                np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lambda_)) /
                self.chiN < 1.4 + 2 / (self.dim + 1))

        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y

        # 协方差矩阵更新
        # Rank-one update
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * np.outer(self.pc, self.pc))

        # Rank-μ update
        for i in range(self.mu):
            yi = (selected[i] - mean_old) / self.sigma
            self.C += self.cmu * self.weights[i] * np.outer(yi, yi)

        # 步长更新
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

        # 确保协方差矩阵对称性
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

    def get_condition_number(self) -> float:
        """获取协方差矩阵条件数"""
        eigenvals = np.linalg.eigvals(self.C)
        eigenvals = np.maximum(eigenvals, 1e-14)
        return np.max(eigenvals) / np.min(eigenvals)

    def get_axis_ratio(self) -> float:
        """获取轴比 (最大/最小标准差)"""
        eigenvals = np.linalg.eigvals(self.C)
        eigenvals = np.maximum(eigenvals, 1e-14)
        return np.sqrt(np.max(eigenvals)) / np.sqrt(np.min(eigenvals))

# ---------------------------- 主要函数 ----------------------------

def solve_q3_cmaes(
    *,
    bombs_count: int = 3,
    # CMA-ES 参数
    population_size: Optional[int] = None,
    sigma0: float = 0.25,
    max_evaluations: int = 10000,
    target_fitness: float = -np.inf,
    tolerance: float = 1e-12,
    # 重启机制
    restart_on_stagnation: bool = True,
    restart_patience: int = 100,
    max_restarts: int = 3,
    # 评估控制
    method: str = "rough_caps_torch",
    dt: float = 0.02,
    # 初始化
    lhs_init: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
    save_path: Optional[str] = None,
    resume_path: Optional[str] = None,
) -> CMAESQ3Result:
    """
    CMA-ES 优化 Problem 3 (单无人机，多炸弹)。最小化目标函数 -occlusion。
    返回 CMAESQ3Result (包含历史记录 / 终局评估)。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ----------------- 构建边界 -----------------
    bounds = bounds_for_problem(3, bombs_count)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    span = hi - lo

    # ----------------- 初始化中心点 -----------------
    mean0 = None
    if resume_path:
        try:
            p = Path(resume_path)
            if p.is_file():
                data = json.loads(p.read_text())
                bx = data.get("best_x")
                if isinstance(bx, list) and len(bx) == dim:
                    mean0 = np.array(bx, dtype=np.float64)
                    mean0 = np.clip(mean0, lo, hi)
                    if verbose:
                        print(f"[CMA-ES-Q3][resume] loaded previous best from {resume_path}")
        except Exception as e:
            if verbose:
                print(f"[CMA-ES-Q3][resume][warn] failed loading {resume_path}: {e}")

    if mean0 is None:
        if lhs_init:
            lhs_points = _lhs(1, dim, rng)
            mean0 = lo + lhs_points[0] * span
        else:
            mean0 = lo + rng.random(dim) * span

    # 方法映射: 旧的 occlusion method (e.g. 'rough_caps_torch', 'judge_caps_torch', 'rough_caps', 'judge_caps')
    # -> router judge keys ('torch_rough', 'torch_exact', 'numpy_rough', 'numpy_exact')
    def _map_method_to_judge(m: str) -> str:
        m = (m or "").lower()
        if "torch" in m:
            if "rough" in m:
                return "torch_rough"
            if "newton" in m:
                return "torch_newton"
            return "torch_exact"
        # numpy side
        if "rough" in m:
            return "numpy_rough"
        return "numpy_exact"

    # 评估函数 (最小化 -occlusion)
    def _decode_and_eval(vec: np.ndarray) -> float:
        speed = float(vec[0])
        az = float(vec[1])
        bombs: List[Tuple[float, float]] = []
        valid = True
        # 解码时间&延迟对
        for i in range(bombs_count):
            t = float(vec[2 + 2 * i])
            d = float(vec[2 + 2 * i + 1])
            bombs.append((t, d))
        # 简单有效性：速度范围、炸弹时间间隔排序
        if not (70.0 <= speed <= 140.0):
            valid = False
        bombs_sorted = sorted(bombs, key=lambda x: x[0])
        for j in range(1, len(bombs_sorted)):
            if bombs_sorted[j][0] - bombs_sorted[j-1][0] < 1.0 - 1e-9:
                valid = False
                break
        if not valid:
            return 1e12  # 惩罚（最小化）
        try:
            judge_key = _map_method_to_judge(method)
            res = evaluate_q3_router(
                bombs=bombs_sorted,
                speed=speed,
                azimuth=az,
                dt=dt,
                judge=judge_key,
                vectorized=True,
                return_details=False,
                penalize_invalid=False,
                verbose=False,
            )
            occ = float(res["occluded_time"]["M1"])
        except Exception:
            return 1e12
        return -occ  # 最小化

    # 主循环变量
    history: List[IterLog] = []
    global_best_value = np.inf
    global_best_x = None
    global_best_eval = None
    total_evaluations = 0
    restart_count = 0
    stagnation_count = 0
    last_improvement = 0

    if verbose:
        print(f"[CMA-ES-Q3] Starting optimization with {dim}D, bombs_count={bombs_count}")

    # 重启循环
    while restart_count <= max_restarts and total_evaluations < max_evaluations:
        if restart_count > 0 and verbose:
            print(f"[CMA-ES-Q3] Restart #{restart_count}")

        # 创建CMA-ES实例
        if restart_count == 0:
            cmaes = CMAES(dim, bounds, population_size, sigma0, mean0, seed)
        else:
            # 重启时在全局最优附近
            if global_best_x is not None:
                restart_mean = global_best_x + rng.normal(0, 0.1 * sigma0, dim) * span
                restart_mean = np.clip(restart_mean, lo, hi)
            else:
                restart_mean = mean0
            cmaes = CMAES(dim, bounds, population_size, sigma0 * 0.8, restart_mean, seed + restart_count)

        local_stagnation = 0
        local_best = np.inf

        # 当前重启的迭代循环
        while (total_evaluations < max_evaluations and
               local_stagnation < restart_patience and
               global_best_value > target_fitness):

            # 生成候选解
            population = cmaes.ask()

            # 评估
            fitness = np.array([_decode_and_eval(x) for x in population])
            total_evaluations += len(fitness)

            # 更新CMA-ES
            cmaes.tell(population, fitness)

            # 检查改进
            current_best = np.min(fitness)
            if current_best < local_best - tolerance:
                local_best = current_best
                local_stagnation = 0

                if current_best < global_best_value:
                    global_best_value = current_best
                    global_best_x = population[np.argmin(fitness)].copy()
                    last_improvement = total_evaluations
                    stagnation_count = 0
            else:
                local_stagnation += 1
                stagnation_count += 1

            # 记录日志
            iter_num = len(history) + 1
            mean_fitness = np.mean(fitness[np.isfinite(fitness)])
            history.append(IterLog(
                iter=iter_num,
                best_value=global_best_value,
                mean_value=mean_fitness,
                sigma=cmaes.sigma,
                condition_number=cmaes.get_condition_number(),
                axis_ratio=cmaes.get_axis_ratio(),
                evaluations=total_evaluations,
            ))

            if verbose and (iter_num == 1 or iter_num % max(1, 20) == 0):
                print(f"[CMA-ES-Q3] iter {iter_num:4d} best={global_best_value:.6f} "
                      f"sigma={cmaes.sigma:.4f} cond={cmaes.get_condition_number():.2e} "
                      f"evals={total_evaluations}")

        # 检查是否需要重启
        if restart_on_stagnation and local_stagnation >= restart_patience and restart_count < max_restarts:
            restart_count += 1
        else:
            break

    # ----------------- 最终评估 -----------------
    if global_best_x is not None:
        final_eval = evaluate_q3_router(
            bombs=[(float(global_best_x[2 + 2 * i]), float(global_best_x[2 + 2 * i + 1]))
                   for i in range(bombs_count)],
            speed=float(global_best_x[0]),
            azimuth=float(global_best_x[1]),
            dt=dt,
            judge=_map_method_to_judge(method),
            vectorized=True,
            return_details=False,
            penalize_invalid=False,
            verbose=False,
        )
        global_best_eval = final_eval
    else:
        global_best_eval = {}

    result = CMAESQ3Result(
        best_x=global_best_x if global_best_x is not None else mean0,
        best_value=global_best_value,
        best_eval=global_best_eval,
        history=history,
        meta=dict(
            dim=dim,
            population_size=cmaes.lambda_ if 'cmaes' in locals() else population_size,
            max_evaluations=max_evaluations,
            total_evaluations=total_evaluations,
            bombs_count=bombs_count,
            method=method,
            dt=dt,
            seed=seed,
            restart_count=restart_count,
            last_improvement=last_improvement,
        )
    )

    # 保存结果
    if save_path and global_best_x is not None:
        try:
            data_out = {
                "best_x": result.best_x.tolist(),
                "best_value": float(result.best_value),
                "meta": result.meta,
            }
            Path(save_path).write_text(json.dumps(data_out, ensure_ascii=False, indent=2))
            if verbose:
                print(f"[CMA-ES-Q3] saved result to {save_path}")
        except Exception as e:
            if verbose:
                print(f"[CMA-ES-Q3][warn] failed to save result: {e}")

    if verbose:
        print("\n[CMA-ES-Q3] DONE")
        print("  best minimized value (=-occlusion):", result.best_value)
        print("  occlusion seconds:", -result.best_value if result.best_value != np.inf else 0)
        print("  total evaluations:", total_evaluations)
        print("  restarts:", restart_count)
        if global_best_x is not None:
            print("  best decision vector:", global_best_x.tolist())

    return result


# ---------------------------- CLI 演示 ----------------------------

def _demo():
    print("Running CMA-ES-Q3 demo ...")
    res = solve_q3_cmaes(
        bombs_count=3,
        population_size=20,
        max_evaluations=1000,
        method="rough_caps_torch",
        dt=0.02,
        seed=42,
        verbose=True,
    )
    print("\n=== Summary ===")
    print("Best x:", res.best_x.tolist())
    print("Best occlusion (s):", -res.best_value if res.best_value != np.inf else 0)
    print("Eval full:", res.best_eval.get("occluded_time", {}))
    print("Meta:", res.meta)
    print("First 5 logs:")
    for log in res.history[:5]:
        print(log)
    print("Last 5 logs:")
    for log in res.history[-5:]:
        print(log)

if __name__ == "__main__":
    _demo()
