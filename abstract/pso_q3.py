from __future__ import annotations
"""
PSO with Partial Restart & Advanced Heuristics for Problem 3
===========================================================

目标:
  优化 题目3 (单无人机多枚炸弹遮蔽最大化), 决策向量:
      x = [ speed, azimuth, (t1,d1), (t2,d2), ... (tB,dB) ]
  目标最大化 M1 遮蔽时长 -> 这里最小化:  f(x) = -occluded_time["M1"]

特性 (相比 baseline PSO 增强):
  1. 分段惯性权重 (线性衰减 + 停滞自适应 boost)
  2. 部分粒子随机重启 (Partial Restart)：
       - 每次迭代按 restart_ratio 选取非精英随机子集重置
       - 停滞或多样性不足时增加重启强度
  3. 多样性度量 & 自适应：
       - diversity = 平均 ||x_i - centroid|| / 初始平均值
       - 若低于 diversity_min_frac -> 提高本轮重启比例 / 引入全局探索
  4. 精英保留：
       - 前 elite_k (按个人最好 pbest 值) 永不被重置
  5. 混合重启策略：
       - 80% 全局均匀重置，20% 围绕当前全局最优高斯微扰 (局部放大搜索)
  6. 速度控制：
       - vmax = vmax_frac * (hi-lo)
       - 重启粒子速度置零或随机（可配置）
  7. 可选 LHS 初始化 (lhs_init=True)
  8. 迭代中周期性本地微扰 (micro_exploit)：
       - 每隔 exploit_interval 代，对当前全局最优附近生成若干试探点 (不增加粒子数)
  9. 自适应加速系数保守调节 (可关闭)
 10. 完整日志 (history) 追踪：gbest, diversity, restart_count, stagnation

依赖:
  - evaluate_problem3
  - bounds_for_problem

使用:
  from abstract.pso_q3 import solve_q3_pso
  res = solve_q3_pso()
  print(res.best_value, -res.best_value_occlusion_seconds)

命令行测试:
  uv run -m abstract.pso_q3

返回:
  PSOQ3Result:
      best_x: numpy.ndarray
      best_value: float (已经是最小化形式 = -occlusion)
      best_eval: evaluate_problem3 返回的完整结构
      history: list[IterLog]

说明:
  本实现偏工程化 + 注释充分，方便后续继续接入混合/多保真/代理模型。
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
    gbest_value: float
    mean_pbest: float
    diversity: float
    restart_particles: int
    adaptive_restart_ratio: float
    stagnation: int
    w: float

@dataclass
class PSOQ3Result:
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

# ---------------------------- 主要函数 ----------------------------

def solve_q3_pso(
    *,
    bombs_count: int = 3,
    pop: int = 120,
    iters: int = 20000,
    # PSO 基本参数
    w_max: float = 0.78,
    w_min: float = 0.40,
    c1: float = 1.55,
    c2: float = 1.55,
    adaptive_accel: bool = True,          # 自适应调节 c1/c2 (轻量)
    accel_decay: float = 0.92,
    accel_min: float = 1.25,
    vmax_frac: float = 0.55,
    # 重启策略
    restart_ratio: float = 0.10,          # 基础每代随机重启比例
    restart_boost: float = 0.25,          # 停滞或多样性低时增加量
    stagnation_patience: int = 25,        # 没有 gbest 改善的迭代阈值
    elite_k: int = 3,                     # 精英不被随机重启
    diversity_min_frac: float = 0.35,     # 多样性归一值低于该阈值视为聚集
    gaussian_local_frac: float = 0.20,    # 重启粒子中使用局部高斯扰动比例
    local_gauss_scale: float = 0.15,      # 相对 span 的局部扰动
    restart_reset_velocity: bool = True,
    # 微扰探索
    micro_exploit: bool = True,
    exploit_interval: int = 40,
    exploit_points: int = 5,
    exploit_scale: float = 0.08,
    # 评估控制
    method: str = "rough_caps_torch",
    dt: float = 0.02,
    # 初始化 & 持久化
    lhs_init: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
    save_path: Optional[str] = None,          # 若提供则保存最优解 (JSON)
    resume_path: Optional[str] = None,        # 若提供则从该 JSON 中读取 best_x 作为种子重新开始
    resume_inject_ratio: float = 0.15,        # 恢复时将历史最优周围生成该比例的新粒子
) -> PSOQ3Result:
    """
    PSO 优化 Problem 3 (单无人机，多炸弹)。最小化目标函数 -occlusion。
    返回 PSOQ3Result (包含史记录 / 终局评估)。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ----------------- 构建边界 -----------------
    bounds = bounds_for_problem(3, bombs_count)   # [(low,high), ...]
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    span = hi - lo
    vmax = span * vmax_frac

    # ----------------- 初始化种群 -----------------
    if lhs_init:
        base = _lhs(pop, dim, rng)
        X = lo + base * span
    else:
        X = lo + rng.random((pop, dim)) * span
    V = (rng.random((pop, dim)) - 0.5) * 2.0 * vmax

    # -------------- Resume 注入 --------------
    resume_best: Optional[np.ndarray] = None
    if resume_path:
        try:
            p = Path(resume_path)
            if p.is_file():
                data = json.loads(p.read_text())
                bx = data.get("best_x")
                if isinstance(bx, list) and len(bx) == dim:
                    resume_best = np.array(bx, dtype=np.float64)
                    if verbose:
                        print(f"[PSO-Q3][resume] loaded previous best from {resume_path}")
        except Exception as e:
            if verbose:
                print(f"[PSO-Q3][resume][warn] failed loading {resume_path}: {e}")

    if resume_best is not None:
        # 放在第一个粒子
        X[0] = np.clip(resume_best, lo, hi)
        # 在其附近生成若干扰动粒子 (不超过种群大小)
        k_inject = max(1, int(pop * resume_inject_ratio))
        noise_scale = 0.05
        for i in range(1, min(1 + k_inject, pop)):
            noise = rng.normal(0.0, 1.0, size=dim) * (noise_scale * span)
            X[i] = np.clip(resume_best + noise, lo, hi)
        # 速度清零以稳定
        V[:1 + k_inject] = 0.0

    # 方法映射: occlusion method -> router judge key
    def _map_method_to_judge(m: str) -> str:
        m = (m or "").lower()
        if "torch" in m:
            if "rough" in m:
                return "torch_rough"
            if "newton" in m:
                return "torch_newton"
            return "torch_exact"
        if "rough" in m:
            return "numpy_rough"
        return "numpy_exact"

    # 评估函数
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

    def eval_batch(M: np.ndarray, phase: str = "init") -> np.ndarray:
        """
        批量评估辅助函数。
        新增 heartbeat 日志: 在初始化阶段 (phase='init') 按进度 10% 输出一次，便于观察长时间
        沉默是否仍在工作 (比如 torch 版本 rough 判定较慢时)。
        """
        out = np.empty((M.shape[0],), dtype=np.float64)
        n = M.shape[0]
        for i in range(n):
            out[i] = _decode_and_eval(M[i])
            if verbose and phase == "init":
                # 打印首个、每 10% 以及最后一个
                if i == 0 or (i + 1) == n or ((i + 1) % max(1, n // 10) == 0):
                    print(f"[PSO-Q3][init] evaluated {i + 1}/{n}")
        return out

    # 初始化个体最优
    P = X.copy()
    # 初始种群评估使用 phase="init" 触发 heartbeat 输出
    pbest = eval_batch(P, phase="init")
    order = np.argsort(pbest)
    g_idx = int(order[0])
    g = P[g_idx].copy()
    gbest = float(pbest[g_idx])

    if verbose:
        print(f"[PSO-Q3] init gbest={gbest:.6f}  x={g.tolist()}")

    # 多样性测量：初始参考
    def _diversity(pop_matrix: np.ndarray) -> float:
        centroid = pop_matrix.mean(axis=0)
        dists = np.linalg.norm(pop_matrix - centroid, axis=1)
        return float(np.mean(dists))

    div0 = _diversity(X)
    if div0 <= 1e-12:
        div0 = 1.0

    history: List[IterLog] = []
    stagnation = 0
    last_improve_iter = 0

    # 迭代主循环
    for it in range(1, iters + 1):
        # 动态惯性权重
        progress = it / iters
        w = w_max - (w_max - w_min) * progress
        # 停滞提升惯性（加大探索）
        if stagnation >= stagnation_patience // 2:
            w = min(w_max + 0.10, w * 1.05)

        # 自适应调节加速系数 (轻量)
        if adaptive_accel and it % 20 == 0 and stagnation > 0:
            c1 = max(accel_min, c1 * accel_decay)
            c2 = max(accel_min, c2 * accel_decay)

        # 更新速度 / 位置
        r1 = rng.random((pop, dim))
        r2 = rng.random((pop, dim))
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        # 限幅
        V = np.clip(V, -vmax, vmax)
        X = X + V

        # 边界 + 简单反弹 (可换更严谨 reflect)
        below = X < lo
        above = X > hi
        if below.any() or above.any():
            X = np.where(below, lo, X)
            X = np.where(above, hi, X)
            V[below] *= -0.4
            V[above] *= -0.4

        # 评估
        vals = eval_batch(X)

        # pbest 更新
        improved = vals < pbest
        if np.any(improved):
            pbest[improved] = vals[improved]
            P[improved] = X[improved]
            # gbest 更新
            new_best_idx = int(np.argmin(pbest))
            new_best_val = float(pbest[new_best_idx])
            if new_best_val < gbest - 1e-15:
                gbest = new_best_val
                g = P[new_best_idx].copy()
                stagnation = 0
                last_improve_iter = it
            else:
                stagnation += 1
        else:
            stagnation += 1

        # 统计多样性
        div_curr = _diversity(X)
        div_ratio = div_curr / div0

        # 自适应重启比例
        adapt_restart_ratio = restart_ratio
        if stagnation >= stagnation_patience:
            adapt_restart_ratio += restart_boost
        if div_ratio < diversity_min_frac:
            adapt_restart_ratio += restart_boost * (1.0 - div_ratio / max(1e-9, diversity_min_frac))
        adapt_restart_ratio = min(0.85, adapt_restart_ratio)

        # 执行部分随机重启
        restart_count = 0
        if adapt_restart_ratio > 1e-9:
            n_restart = int(round(pop * adapt_restart_ratio))
            if n_restart > 0:
                # 精英保护: 排序 pbest
                elite_k_eff = min(elite_k, pop // 3)
                elite_idx = np.argsort(pbest)[:elite_k_eff]
                elite_mask = np.zeros(pop, dtype=bool)
                elite_mask[elite_idx] = True

                candidates_idx = np.where(~elite_mask)[0]
                if len(candidates_idx) > 0:
                    if n_restart > len(candidates_idx):
                        n_restart = len(candidates_idx)
                    chosen = rng.choice(candidates_idx, size=n_restart, replace=False)
                    # 局部/全局混合
                    n_local = int(round(len(chosen) * gaussian_local_frac))
                    if n_local > 0:
                        local_subset = chosen[:n_local]
                        global_subset = chosen[n_local:]
                    else:
                        local_subset = np.array([], dtype=int)
                        global_subset = chosen

                    # 全局重启
                    if global_subset.size > 0:
                        R = lo + rng.random((global_subset.size, dim)) * span
                        X[global_subset] = R
                        if restart_reset_velocity:
                            V[global_subset] = 0.0
                        else:
                            V[global_subset] = (rng.random((global_subset.size, dim)) - 0.5) * 2 * vmax
                        vals_global = eval_batch(X[global_subset])
                        pbest[global_subset] = vals_global
                        P[global_subset] = X[global_subset]

                    # 局部高斯（围绕当前 g）
                    if local_subset.size > 0:
                        base = g
                        noise = rng.normal(0.0, 1.0, size=(local_subset.size, dim)) * (local_gauss_scale * span)
                        local_pos = base + noise
                        local_pos = np.clip(local_pos, lo, hi)
                        X[local_subset] = local_pos
                        if restart_reset_velocity:
                            V[local_subset] = 0.0
                        else:
                            V[local_subset] = (rng.random((local_subset.size, dim)) - 0.5) * 2 * vmax
                        vals_local = eval_batch(X[local_subset])
                        pbest[local_subset] = vals_local
                        P[local_subset] = X[local_subset]

                    restart_count = n_restart
                    # 全局最优可能改善
                    new_best_idx2 = int(np.argmin(pbest))
                    new_best_val2 = float(pbest[new_best_idx2])
                    if new_best_val2 < gbest - 1e-15:
                        gbest = new_best_val2
                        g = P[new_best_idx2].copy()
                        stagnation = 0
                        last_improve_iter = it

        # 微扰探索：不改变粒子数量，只尝试更新 gbest
        if micro_exploit and exploit_interval > 0 and (it % exploit_interval == 0):
            trial_scale = exploit_scale * (0.5 + 0.5 * (1.0 - div_ratio))
            for _ in range(exploit_points):
                noise = rng.normal(0.0, 1.0, size=dim) * (trial_scale * span)
                cand = np.clip(g + noise, lo, hi)
                v_c = _decode_and_eval(cand)
                if v_c < gbest - 1e-15:
                    gbest = v_c
                    g = cand.copy()
                    stagnation = 0
                    last_improve_iter = it

        # 记录日志
        mean_pbest = float(np.mean(pbest[np.isfinite(pbest)]))
        history.append(IterLog(
            iter=it,
            gbest_value=gbest,
            mean_pbest=mean_pbest,
            diversity=div_ratio,
            restart_particles=restart_count,
            adaptive_restart_ratio=adapt_restart_ratio,
            stagnation=stagnation,
            w=w,
        ))

        if verbose and (it == 1 or it == iters or it % max(5, iters // 10) == 0):
            print(f"[PSO-Q3] iter {it:4d} gbest={gbest:.6f} div={div_ratio:.3f} "
                  f"rest={restart_count} w={w:.3f} stag={stagnation}")

    # ----------------- 最终评估 (router 单次向量化) -----------------
    final_eval = evaluate_q3_router(
        bombs=[(float(g[2 + 2 * i]), float(g[2 + 2 * i + 1])) for i in range(bombs_count)],
        speed=float(g[0]),
        azimuth=float(g[1]),
        dt=dt,
        judge=_map_method_to_judge(method),
        vectorized=True,
        return_details=False,
        penalize_invalid=False,
        verbose=False,
    )

    result = PSOQ3Result(
        best_x=g.copy(),
        best_value=gbest,
        best_eval=final_eval,
        history=history,
        meta=dict(
            dim=dim,
            population=pop,
            iterations=iters,
            bombs_count=bombs_count,
            method=method,
            dt=dt,
            seed=seed,
            last_improve_iter=last_improve_iter,
        )
    )
    if save_path:
        try:
            data_out = {
                "best_x": result.best_x.tolist(),
                "best_value": float(result.best_value),
                "meta": result.meta,
            }
            Path(save_path).write_text(json.dumps(data_out, ensure_ascii=False, indent=2))
            if verbose:
                print(f"[PSO-Q3] saved result to {save_path}")
        except Exception as e:
            if verbose:
                print(f"[PSO-Q3][warn] failed to save result: {e}")

    if verbose:
        print("\n[PSO-Q3] DONE")
        print("  best minimized value (=-occlusion):", result.best_value)
        print("  occlusion seconds:", -result.best_value)
        print("  best decision vector:", result.best_x.tolist())
    return result


# ---------------------------- CLI 演示 ----------------------------

def _demo():
    print("Running PSO-Q3 demo ...")
    res = solve_q3_pso(
        bombs_count=3,
        pop=60,
        iters=120,
        method="rough_caps_torch",
        dt=0.02,
        seed=42,
        verbose=True,
    )
    print("\n=== Summary ===")
    print("Best x:", res.best_x.tolist())
    print("Best occlusion (s):", -res.best_value)
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
