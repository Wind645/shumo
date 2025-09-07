"""
Staged / Blocked Dimension PSO Scheduler.

核心目标:
  在高维（或存在强烈非线性耦合）但单次仿真昂贵的情境下，通过“分批只动部分维度、其它维度冻结”的方式
  降低瞬时搜索空间规模，改善早期探索质量，并缓解粒子群早收敛到局部最优的问题。

当前实现覆盖 设计思路(一) + 维度分批策略(二) 的基础版本：
  1. 维护全局向量 full_best
  2. 选择一个维度子集 active_dims
  3. 构造子目标 f_sub(x_sub) => 嵌入 full_best 得到 full_x => 调原始 objective
  4. 调用已有 ParallelPSO 优化该子空间
  5. 把子结果写回全局，进入下一个维度 batch
  6. 可选最后做一轮全维精修 (refine_full=True)

支持的分批策略 (mode):
  - 'fixed'      : 固定窗口切分（策略 A）
  - 'shuffle'    : 每个 epoch/循环随机洗牌维度再按 group_size 切分（策略 B）
  - 'overlap'    : 重叠滑动窗口 (window_size, stride)（策略 C）
  - 'sensitivity': 先做轻量敏感度估计（单变量扰动）按影响排序再分组（策略 D）
  - 'custom'     : 直接传入 stage_dims（外部构造）

策略 E（自适应再分组）尚未实现，留接口位。

依赖:
  - optimizer.pso.ParallelPSO

使用示例:
    from optimizer.pso import build_problem_objective
    from optimizer.staged_pso import StagedDimPSO

    encoder, objective = build_problem_objective(problem_id=5)
    dim = encoder.dim
    staged = StagedDimPSO(
        dim=dim,
        objective=objective,
        mode='overlap',
        window_size=5,
        stride=3,
        iterations_per_stage=30,
        swarm_size=48,
        refine_full=True,
        refine_iterations=60
    )
    result = staged.run()
    print("BEST", result.best_fitness)
    print("Stages:", len(result.stage_logs))
    for lg in result.stage_logs[:3]:
        print(lg.as_dict())

扩展方向（未在该基础版本实现）:
  - 自适应动态重组 (策略 E)
  - Teleport / 停滞检测（此处留了简单 hook，可扩展）
  - 多轮循环重复 shuffle/overlap 序列直到预算耗尽
  - 引入局部二阶近似 / CMA-ES 小范围精修

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, List, Optional, Tuple, Dict, Any, Literal
import numpy as np

try:
    # 优先相对导入（包内调用）
    from .pso import ParallelPSO
except ImportError:  # pragma: no cover
    # 允许独立脚本调试
    from optimizer.pso import ParallelPSO  # type: ignore

ModeLiteral = Literal['fixed', 'shuffle', 'overlap', 'sensitivity', 'custom']


# --------------------------------------------------------------------------- #
# 日志与结果数据结构
# --------------------------------------------------------------------------- #
@dataclass
class StageLog:
    stage_id: int
    active_dims: List[int]
    sub_dim: int
    iterations: int
    sub_best_fitness: float
    global_best_after: float
    improved_global: bool
    sub_history: List[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "active_dims": self.active_dims,
            "sub_dim": self.sub_dim,
            "iterations": self.iterations,
            "sub_best_fitness": self.sub_best_fitness,
            "global_best_after": self.global_best_after,
            "improved_global": self.improved_global,
            "sub_history": self.sub_history,
        }


@dataclass
class StagedDimPSOResult:
    best_position: np.ndarray
    best_fitness: float
    stage_logs: List[StageLog]
    eval_count: int


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class StagedDimPSO:
    """
    Staged Dimension Scheduler 封装。

    参数:
      dim                : 全局维度
      objective          : f(x_full) -> fitness
      mode               : 'fixed' | 'shuffle' | 'overlap' | 'sensitivity' | 'custom'
      stage_dims         : 当 mode='custom' 时提供 List[List[int]]
      group_size         : shuffle 模式下单组大小
      window_size/stride : overlap 模式窗口与步幅；fixed 模式用 window_size 且 stride=window_size
      sensitivity_perturb: sensitivity 模式单维扰动幅度（相对 or 绝对）
      sensitivity_samples: 每维扰动采样次数
      max_stages         : 限制最大阶段数 (None 表示不裁剪)
      iterations_per_stage: 子 PSO 迭代
      swarm_size         : 子 PSO 粒子数
      maximize           : True=最大化
      bounds             : (lo, hi) 全维 bound
      seed               : 随机种子
      refine_full        : 是否阶段结束后做全维微调
      refine_iterations  : 全维微调迭代数
      verbose            : 打印进度
      reuse_final_swarm  : refine 时尝试以 full_best 为中心初始化
    """
    def __init__(self,
                 dim: int,
                 objective: Callable[[Sequence[float]], float],
                 mode: ModeLiteral = 'fixed',
                 stage_dims: Optional[List[List[int]]] = None,
                 group_size: int = 5,
                 window_size: int = 5,
                 stride: Optional[int] = None,
                 sensitivity_perturb: float = 0.1,
                 sensitivity_samples: int = 3,
                 max_stages: Optional[int] = None,
                 iterations_per_stage: int = 30,
                 swarm_size: int = 40,
                 maximize: bool = True,
                 bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                 seed: Optional[int] = None,
                 refine_full: bool = True,
                 refine_iterations: int = 50,
                 verbose: bool = True,
                 reuse_final_swarm: bool = True):
        self.dim = int(dim)
        self.objective = objective
        self.mode = mode
        self.iterations_per_stage = iterations_per_stage
        self.swarm_size = swarm_size
        self.maximize = maximize
        self.verbose = verbose
        self.refine_full = refine_full
        self.refine_iterations = refine_iterations
        self.reuse_final_swarm = reuse_final_swarm

        self.random = np.random.RandomState(seed or 12345)

        # Bounds
        if bounds is not None:
            lo = np.asarray(bounds[0], dtype=float)
            hi = np.asarray(bounds[1], dtype=float)
            if lo.shape != (self.dim,) or hi.shape != (self.dim,):
                raise ValueError("bounds must be shape (dim,)")
            if np.any(hi <= lo):
                raise ValueError("All hi > lo required")
            self.bounds = (lo, hi)
        else:
            self.bounds = None

        # 全局 best 初始化
        if self.bounds is None:
            self.full_best = self.random.uniform(-1, 1, self.dim)
        else:
            lo, hi = self.bounds
            self.full_best = lo + (hi - lo) * self.random.rand(self.dim)
        self.full_best_fitness = -np.inf if maximize else np.inf

        self.stage_logs: List[StageLog] = []
        self.eval_count = 0

        # 分组参数
        self._fixed_window_size = window_size
        self._fixed_stride = stride if stride is not None else window_size
        self._group_size = group_size
        self._sensitivity_perturb = float(sensitivity_perturb)
        self._sensitivity_samples = int(sensitivity_samples)
        self._max_stages = max_stages
        self._custom_stage_dims = stage_dims

        if self.mode == 'custom' and not self._custom_stage_dims:
            raise ValueError("mode='custom' 需要提供 stage_dims")

    # ---------------------------- 公共入口 ---------------------------- #
    def run(self) -> StagedDimPSOResult:
        if self.verbose:
            print(f"[StagedPSO] Mode={self.mode}")

        stage_groups = self._build_stage_groups()
        if self._max_stages is not None:
            stage_groups = stage_groups[:self._max_stages]
            if self.verbose:
                print(f"[StagedPSO] Trimmed to max_stages={self._max_stages}")

        # 初始评估
        # 说明：
        #   - 当前无条件评估 self.full_best 以得到 base_fit。
        #   - 如果调用方（例如 Hybrid 管线）已经提前计算并设置 full_best_fitness，
        #     可以通过添加条件跳过重复评估以节省一次模拟调用。
        #   - 可选逻辑示例（未启用）：
        #       if (self.maximize and self.full_best_fitness != -np.inf) or \
        #          ((not self.maximize) and self.full_best_fitness != np.inf):
        #           base_fit = self.full_best_fitness
        #       else:
        #           base_fit = self.objective(self.full_best); self.eval_count += 1
        base_fit = self.objective(self.full_best)
        self.eval_count += 1
        self.full_best_fitness = base_fit
        if self.verbose:
            print(f"[StagedPSO] Initial fitness={base_fit:.6f}")

        for sid, dims in enumerate(stage_groups):
            self._run_one_stage(stage_id=sid, active_dims=dims)

        if self.refine_full:
            self._full_refine()

        return StagedDimPSOResult(
            best_position=self.full_best.copy(),
            best_fitness=float(self.full_best_fitness),
            stage_logs=self.stage_logs,
            eval_count=self.eval_count
        )

    # ---------------------------- 分组构造 ---------------------------- #
    def _build_stage_groups(self) -> List[List[int]]:
        if self.mode == 'custom':
            return [list(g) for g in self._custom_stage_dims]

        if self.mode == 'fixed':
            groups = []
            for i in range(0, self.dim, self._fixed_window_size):
                groups.append(list(range(i, min(self.dim, i + self._fixed_window_size))))
            if self.verbose:
                print(f"[StagedPSO] Fixed groups={len(groups)} window={self._fixed_window_size}")
            return groups

        if self.mode == 'shuffle':
            idx = np.arange(self.dim)
            self.random.shuffle(idx)
            groups = []
            for i in range(0, self.dim, self._group_size):
                groups.append(idx[i:i + self._group_size].tolist())
            if self.verbose:
                print(f"[StagedPSO] Shuffle groups={len(groups)} group_size={self._group_size}")
            return groups

        if self.mode == 'overlap':
            groups = []
            w = self._fixed_window_size
            s = self._fixed_stride
            i = 0
            while i < self.dim:
                groups.append(list(range(i, min(self.dim, i + w))))
                i += s
            if self.verbose:
                print(f"[StagedPSO] Overlap groups={len(groups)} window={w} stride={s}")
            return groups

        if self.mode == 'sensitivity':
            order = self._estimate_sensitivity_order()
            groups = []
            for i in range(0, self.dim, self._group_size):
                groups.append(order[i:i + self._group_size])
            if self.verbose:
                print(f"[StagedPSO] Sensitivity groups={len(groups)} group_size={self._group_size}")
            return groups

        raise ValueError(f"Unknown mode={self.mode}")

    # ---------------------------- 敏感度估计 ---------------------------- #
    def _estimate_sensitivity_order(self) -> List[int]:
        """
        简单单变量扰动：对每个维度做 N 次 ±delta 扰动，记录 fitness 变化绝对值均值。
        复杂耦合未考虑，只做排序。
        """
        base = self.full_best.copy()
        base_fit = self.objective(base)
        self.eval_count += 1

        impacts = np.zeros(self.dim)
        if self.bounds is None:
            scale = np.ones(self.dim)
        else:
            lo, hi = self.bounds
            scale = (hi - lo)

        for d in range(self.dim):
            vals = []
            span = scale[d]
            delta = self._sensitivity_perturb * (span if np.isfinite(span) else 1.0)
            for _ in range(self._sensitivity_samples):
                for sign in (+1, -1):
                    trial = base.copy()
                    trial[d] += sign * delta
                    if self.bounds is not None:
                        lo, hi = self.bounds
                        trial[d] = np.clip(trial[d], lo[d], hi[d])
                    fit = self.objective(trial)
                    self.eval_count += 1
                    vals.append(abs(fit - base_fit))
            impacts[d] = np.mean(vals) if vals else 0.0
        order = list(np.argsort(-impacts))  # 大到小
        if self.verbose:
            topk = min(5, self.dim)
            print(f"[StagedPSO] Sensitivity top {topk} dims:",
                  [(int(i), float(impacts[i])) for i in order[:topk]])
        return order

    # ---------------------------- 单阶段运行 ---------------------------- #
    def _run_one_stage(self, stage_id: int, active_dims: List[int]):
        sub_dim = len(active_dims)
        if self.verbose:
            print(f"[StagedPSO] Stage {stage_id} dims={active_dims} (k={sub_dim})")

        # 构造子 objective
        def sub_obj(x_sub: Sequence[float]) -> float:
            vec = self.full_best.copy()
            vec[active_dims] = x_sub
            return self.objective(vec)

        # 子 bounds
        sub_bounds = None
        if self.bounds is not None:
            lo, hi = self.bounds
            sub_bounds = (lo[active_dims], hi[active_dims])

        # 初始化子 swarm 以当前 best 的该子向量为中心
        center = self.full_best[active_dims].copy()
        if sub_bounds is None:
            init_positions = center + 0.10 * np.random.randn(self.swarm_size, sub_dim)
        else:
            lo_s, hi_s = sub_bounds
            span = hi_s - lo_s
            init_positions = center + 0.10 * span * np.random.randn(self.swarm_size, sub_dim)
            init_positions = np.clip(init_positions, lo_s, hi_s)
        # 保留精确中心（历史全局最优子向量）为粒子0，避免初始化噪声把最优“冲掉”
        init_positions[0] = center

        # 构造 ParallelPSO
        pso = ParallelPSO(
            dim=sub_dim,
            objective=sub_obj,
            swarm_size=self.swarm_size,
            iterations=self.iterations_per_stage,
            maximize=self.maximize,
            bounds=sub_bounds,
            reset_prob=0.02,
        )
        # 覆盖初始
        pso.positions = init_positions.copy()
        pso.personal_best_positions = pso.positions.copy()
        if self.maximize:
            pso.personal_best_fitness[:] = -np.inf
            pso.global_best_fitness = -np.inf
        else:
            pso.personal_best_fitness[:] = np.inf
            pso.global_best_fitness = np.inf
        pso.global_best_position = pso.positions[0].copy()

        # 先单独评估粒子0（精确中心），作为基准 personal/global best，避免必须重新“撞回”该点
        center_fit = sub_obj(center)
        self.eval_count += 1
        pso.personal_best_fitness[0] = center_fit
        pso.personal_best_positions[0] = center.copy()
        pso.global_best_fitness = center_fit
        pso.global_best_position = center.copy()

        # 自定义迭代（直接复用其内部 _evaluate_batch，避免频繁建模多进程池的开销由类本身承担）
        history: List[float] = []
        stage_eval = 0
        for it in range(self.iterations_per_stage):
            fitness = pso._evaluate_batch(pso.positions)
            stage_eval += len(fitness)

            # 更新个体 / 全局
            for i, fit in enumerate(fitness):
                better = fit > pso.personal_best_fitness[i] if self.maximize else fit < pso.personal_best_fitness[i]
                if better:
                    pso.personal_best_fitness[i] = fit
                    pso.personal_best_positions[i] = pso.positions[i].copy()
            best_idx = int(np.argmax(pso.personal_best_fitness) if self.maximize else
                           np.argmin(pso.personal_best_fitness))
            best_fit = pso.personal_best_fitness[best_idx]
            better_global = best_fit > pso.global_best_fitness if self.maximize else best_fit < pso.global_best_fitness
            if better_global:
                pso.global_best_fitness = best_fit
                pso.global_best_position = pso.personal_best_positions[best_idx].copy()

            history.append(float(pso.global_best_fitness))
            # 基础打印
            if self.verbose and (it + 1) % max(1, self.iterations_per_stage // 5) == 0:
                print(f"  [Stage {stage_id}] iter {it+1}/{self.iterations_per_stage} "
                      f"sub_best={pso.global_best_fitness:.6f}")

            # 速度更新
            r1 = np.random.rand(pso.swarm_size, sub_dim)
            r2 = np.random.rand(pso.swarm_size, sub_dim)
            cog = pso.c1 * r1 * (pso.personal_best_positions - pso.positions)
            soc = pso.c2 * r2 * (pso.global_best_position - pso.positions)
            pso.velocities = pso.w * pso.velocities + cog + soc
            if pso.velocity_clamp is not None:
                vmin, vmax = pso.velocity_clamp
                np.clip(pso.velocities, vmin, vmax, out=pso.velocities)
            pso.positions += pso.velocities
            if pso.bounds is not None:
                lo_s, hi_s = pso.bounds
                np.clip(pso.positions, lo_s, hi_s, out=pso.positions)

        self.eval_count += stage_eval

        # 写回全局并重新评估整向量（必要：子目标忽略其它维度潜在非线性耦合影响）
        candidate = self.full_best.copy()
        candidate[active_dims] = pso.global_best_position
        full_fit = self.objective(candidate)
        self.eval_count += 1

        improved_global = self._is_better(full_fit, self.full_best_fitness)
        if improved_global:
            self.full_best = candidate
            self.full_best_fitness = full_fit

        if self.verbose:
            print(f"[StagedPSO] Stage {stage_id} done sub_best={pso.global_best_fitness:.6f} "
                  f"full_fit={full_fit:.6f} improved={improved_global}")

        self.stage_logs.append(StageLog(
            stage_id=stage_id,
            active_dims=list(active_dims),
            sub_dim=sub_dim,
            iterations=self.iterations_per_stage,
            sub_best_fitness=float(pso.global_best_fitness),
            global_best_after=float(self.full_best_fitness),
            improved_global=improved_global,
            sub_history=history
        ))

    # ---------------------------- 最终全维精修 ---------------------------- #
    def _full_refine(self):
        if self.verbose:
            print("[StagedPSO] Starting full-dimensional refine...")
        bounds = self.bounds
        pso = ParallelPSO(
            dim=self.dim,
            objective=self.objective,
            swarm_size=max(self.swarm_size, 40),
            iterations=self.refine_iterations,
            maximize=self.maximize,
            bounds=bounds,
            reset_prob=0.03,
        )
        # 以 full_best 为中心 (注入精确最优粒子0，降低噪声并直接评估)
        if self.reuse_final_swarm:
            if bounds is None:
                base = self.full_best
                pso.positions = base + 0.12 * np.random.randn(pso.swarm_size, self.dim)
            else:
                lo, hi = bounds
                span = hi - lo
                base = self.full_best
                pso.positions = np.clip(base + 0.12 * span * np.random.randn(pso.swarm_size, self.dim),
                                        lo, hi)
            # 保留精确全局最优向量为粒子0，避免被噪声冲掉
            pso.positions[0] = self.full_best
            pso.personal_best_positions = pso.positions.copy()
            if self.maximize:
                pso.personal_best_fitness[:] = -np.inf
                pso.global_best_fitness = -np.inf
            else:
                pso.personal_best_fitness[:] = np.inf
                pso.global_best_fitness = np.inf
            # 直接评估粒子0并设定其为个人 / 全局最优
            best0 = self.objective(self.full_best)
            self.eval_count += 1
            pso.personal_best_fitness[0] = best0
            pso.global_best_fitness = best0
            pso.global_best_position = self.full_best.copy()

        res = pso.run()
        self.eval_count += res.eval_count
        if self._is_better(res.best_fitness, self.full_best_fitness):
            if self.verbose:
                print(f"[StagedPSO] Refine improved {self.full_best_fitness:.6f} -> {res.best_fitness:.6f}")
            self.full_best = res.best_position
            self.full_best_fitness = res.best_fitness
        else:
            if self.verbose:
                print("[StagedPSO] Refine did not improve global best.")

    # ---------------------------- 辅助 ---------------------------- #
    def _is_better(self, a: float, b: float) -> bool:
        return a > b if self.maximize else a < b

    # 可预留扩展接口：自适应再分组 / Teleport / 停滞检测
    # def _adaptive_regroup(self): pass
    # def _teleport_if_stalled(self): pass


__all__ = [
    "StagedDimPSO",
    "StagedDimPSOResult",
    "StageLog",
]
