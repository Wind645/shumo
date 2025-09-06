"""
Simplified Block-based PSO (BlockPSO) with mandatory shared process pool.

Key changes vs 旧版本:
  - 永远使用单一共享 multiprocessing.Pool；彻底移除每块内部独立进程池与可选开关。
  - 为每个块分配独立 seed (base_seed + block_id) 以降低初始 swarm 相关性。
  - 仅保留核心“多块 + 淘汰”机制，去掉向后兼容分支与冗余逻辑。
  - 仍然使用 ParallelPSO（来自 pso.py）作为块内基础优化器，但其 _evaluate_batch 已被 monkey patch
    到共享进程池，避免大量进程反复创建/销毁。
  - 代码从零重写，接口保持直观化；不考虑旧版本兼容。

注意：
  - 仍然是一次性笛卡尔划分 [-1,1]^dim -> (blocks_per_dim ** dim) 个 axis-aligned 子块；
    维度高时指数爆炸。务必控制 blocks_per_dim^dim 的规模。
  - 如果块数非常大（> 50k）会直接抛出异常保护（可按需放宽）。
  - 不做层次细分/自适应再切分；可以在后续扩展（例如对最优块递归 subdivision）。

用法示例：
    from optimizer.pso import build_problem_objective
    from optimizer.block_pso import BlockPSO

    encoder, objective = build_problem_objective(problem_id=3)
    bpso = BlockPSO(
        dim=encoder.dim,
        objective=objective,
        blocks_per_dim=2,
        total_iterations=120,
        block_iterations=8,
        elimination_fraction=0.4,
        min_blocks=2,
        swarm_size=48,
        inertia=0.72,
        cognitive=1.49,
        social=1.49,
        reset_prob=0.05,
        velocity_clamp=(-0.5, 0.5),
        processes=6,
        seed=2025,
        maximize=True,
    )
    result = bpso.run()
    print("BEST", result.best_fitness, result.best_position)

扩展建议（未内置）：
  - 块层次递归细分：对最优块再划分并继续淘汰。
  - 全局最佳跨块共享：每轮结束后把全局 best 写入其他块 swarm。
  - 自适应淘汰率：前期高淘汰，加速压缩；后期降低保持多样性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, List, Dict, Any, Optional, Tuple
import numpy as np
import multiprocessing as mp
import time
import math
import types

from .pso import ParallelPSO, _worker_init, _worker_eval_vector  # type: ignore


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class BlockPSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: List[float]                 # Global best fitness after each round
    block_histories: List[Dict[str, Any]]  # Per-round summaries
    elapsed: float
    eval_count: int


# --------------------------------------------------------------------------- #
# BlockPSO
# --------------------------------------------------------------------------- #
class BlockPSO:
    """
    多块并行 + 淘汰 的粗到细全局搜索器（简化版，强制共享进程池）。

    参数:
      dim                  : 维度
      objective            : 目标函数 vector -> fitness
      blocks_per_dim       : 每维划分份数 (>=1)，总块数 = blocks_per_dim ** dim
      total_iterations     : 总的迭代预算（所有轮合计）
      block_iterations     : 每轮每存活块运行的迭代次数
      elimination_fraction : 每轮结束淘汰比例 (0~1)，最后一轮不淘汰
      min_blocks           : 至少保留块数
      swarm_size           : 每块内部 PSO 粒子数
      maximize             : 是否为最大化
      processes            : 共享进程池进程数；None 自动 (cpu_count - 1)
      seed                 : 基础随机种子；块内会加 block_id 偏移
      inertia / cognitive / social / reset_prob / velocity_clamp: 直接透传给 ParallelPSO

    运行:
      - 预生成所有块（警惕指数膨胀）。
      - 创建一个共享进程池，并设置 worker initializer。
      - Monkey patch 每个块内部 ParallelPSO._evaluate_batch => 使用共享池 map。
      - 多轮循环：各块追加迭代 -> 汇总 -> 淘汰 -> 下一轮。
      - 返回最终最优块的全局 best。

    限制:
      - 不适合高维配大 blocks_per_dim。
    """
    def __init__(self,
                 dim: int,
                 objective: Callable[[Sequence[float]], float],
                 blocks_per_dim: int = 2,
                 total_iterations: int = 200,
                 block_iterations: int = 10,
                 elimination_fraction: float = 0.30,
                 min_blocks: int = 1,
                 swarm_size: int = 30,
                 maximize: bool = True,
                 processes: Optional[int] = None,
                 seed: Optional[int] = None,
                 inertia: float = 0.72,
                 cognitive: float = 1.49,
                 social: float = 1.49,
                 reset_prob: float = 0.05,
                 velocity_clamp: Optional[Tuple[float, float]] = None):
        # 参数校验
        if blocks_per_dim < 1:
            raise ValueError("blocks_per_dim must be >= 1")
        if total_iterations <= 0:
            raise ValueError("total_iterations must be > 0")
        if block_iterations <= 0:
            raise ValueError("block_iterations must be > 0")
        if not (0.0 <= elimination_fraction < 1.0):
            raise ValueError("elimination_fraction must be in [0,1)")
        if min_blocks < 1:
            raise ValueError("min_blocks must be >= 1")

        self.dim = dim
        self.objective = objective
        self.blocks_per_dim = blocks_per_dim
        self.total_iterations = total_iterations
        self.block_iterations = block_iterations
        self.elimination_fraction = elimination_fraction
        self.min_blocks = min_blocks
        self.swarm_size = swarm_size
        self.maximize = maximize
        self.processes = processes
        self.base_seed = seed if seed is not None else int(time.time())
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.reset_prob = reset_prob
        self.velocity_clamp = velocity_clamp

        # 生成块
        total_blocks = blocks_per_dim ** dim
        # 简单保护：防止意外指数爆炸
        if total_blocks > 50_000:
            raise ValueError(
                f"Block count {total_blocks} too large; reduce blocks_per_dim or dim."
            )

        edges = np.linspace(-1.0, 1.0, blocks_per_dim + 1)
        self.blocks: List[Dict[str, Any]] = []

        def _recurse(idx_prefix: List[int], depth: int):
            if depth == dim:
                lows = [edges[i] for i in idx_prefix]
                highs = [edges[i + 1] for i in idx_prefix]
                lo = np.array(lows)
                hi = np.array(highs)
                block_id = len(self.blocks)
                pso = ParallelPSO(
                    dim=dim,
                    objective=self.objective,
                    swarm_size=self.swarm_size,
                    iterations=self.block_iterations,   # 将在 run() 里按轮重写
                    inertia=self.inertia,
                    cognitive=self.cognitive,
                    social=self.social,
                    reset_prob=self.reset_prob,
                    velocity_clamp=self.velocity_clamp,
                    maximize=self.maximize,
                    processes=1,         # 占位：不使用内部进程池
                    seed=self.base_seed + block_id,
                    bounds=(lo, hi)
                )
                self.blocks.append({
                    "lo": lo,
                    "hi": hi,
                    "pso": pso,
                    "best_fitness": -np.inf if self.maximize else np.inf,
                    "history": []
                })
                return
            for c in range(blocks_per_dim):
                _recurse(idx_prefix + [c], depth + 1)

        _recurse([], 0)

    # --------------------------------------------------------------------- #
    # 运行
    # --------------------------------------------------------------------- #
    def run(self) -> BlockPSOResult:
        start = time.time()
        rounds = (self.total_iterations + self.block_iterations - 1) // self.block_iterations
        remaining = self.total_iterations

        global_history: List[float] = []
        round_summaries: List[Dict[str, Any]] = []
        eval_count = 0

        # 构建共享进程池并 patch 所有块的 ParallelPSO._evaluate_batch
        proc = self.processes or max(1, mp.cpu_count() - 1)
        with mp.Pool(
            processes=proc,
            initializer=_worker_init,
            initargs=(self.base_seed, self.objective, self.maximize)
        ) as pool:

            def _shared_eval(self_ref, batch, _pool=pool):
                return _pool.map(_worker_eval_vector, batch)

            for blk in self.blocks:
                pso_obj = blk["pso"]
                pso_obj._evaluate_batch = types.MethodType(_shared_eval, pso_obj)  # type: ignore

            # 轮循环
            for r in range(rounds):
                iters = min(self.block_iterations, remaining)
                remaining -= iters

                # 每个块追加迭代
                for blk in self.blocks:
                    res = blk["pso"].run(iterations=iters)
                    blk["history"].extend(res.history)
                    improved = (
                        (self.maximize and res.best_fitness > blk["best_fitness"]) or
                        ((not self.maximize) and res.best_fitness < blk["best_fitness"])
                    )
                    if improved:
                        blk["best_fitness"] = res.best_fitness
                    eval_count += res.eval_count

                # 本轮全局最好
                if self.maximize:
                    best_blk = max(self.blocks, key=lambda b: b["best_fitness"])
                else:
                    best_blk = min(self.blocks, key=lambda b: b["best_fitness"])
                global_history.append(best_blk["best_fitness"])

                round_summaries.append({
                    "round": r,
                    "active_blocks": len(self.blocks),
                    "best_fitness": best_blk["best_fitness"],
                    "iterations_used": iters
                })
                print(f"[BlockPSO] Round {r+1}/{rounds} "
                      f"blocks={len(self.blocks)} "
                      f"best={best_blk['best_fitness']:.6f}",
                      flush=True)

                # 淘汰（最后一轮跳过）
                if r < rounds - 1 and len(self.blocks) > self.min_blocks:
                    drop = int(math.floor(len(self.blocks) * self.elimination_fraction))
                    if len(self.blocks) - drop < self.min_blocks:
                        drop = len(self.blocks) - self.min_blocks
                    if drop > 0:
                        sorted_blocks = sorted(
                            self.blocks,
                            key=lambda b: b["best_fitness"],
                            reverse=self.maximize
                        )
                        if self.maximize:
                            # 丢弃最差 drop 个（尾部）
                            self.blocks = sorted_blocks[:-drop]
                        else:
                            # 最小化：丢弃最差（前部）
                            self.blocks = sorted_blocks[drop:]

        # 最终最好块
        if self.maximize:
            best_blk = max(self.blocks, key=lambda b: b["best_fitness"])
        else:
            best_blk = min(self.blocks, key=lambda b: b["best_fitness"])

        elapsed = time.time() - start
        return BlockPSOResult(
            best_position=best_blk["pso"].global_best_position.copy(),
            best_fitness=float(best_blk["best_fitness"]),
            history=global_history,
            block_histories=round_summaries,
            elapsed=elapsed,
            eval_count=eval_count
        )


__all__ = [
    "BlockPSO",
    "BlockPSOResult",
]
