"""Hybrid Evolutionary Pipeline: DE -> StagedDimPSO -> PushPSO (No Crowding)

This refactored hybrid optimizer replaces the previous (DE -> DPSO -> optional PushPSO)
design with a new three-stage continuous pipeline:

  Stage 1: Differential Evolution (large population, few generations)
           - Provides broad global exploration and diverse elite set.
  Stage 2: StagedDimPSO (dimension batching / overlap refinement)
           - Sequential or overlapping subspace optimization to refine coordinate-wise.
  Stage 3: PushPSO (crowding disabled, only repeller / stagnation escape)
           - Final global polishing & controlled diversification around the refined best.

Motivation
----------
1. Removes discrete recombination (DPSO) complexity when parameters are inherently
   continuous (raw vector later decoded by encoders).
2. Staged dimension refinement reduces effective search dimensionality per phase
   aiding escape from weak local optima.
3. PushPSO (with crowding disabled) supplies adaptive repulsion / stagnation-based
   exploration without the O(N^2) classic crowding penalty.

Compatibility
-------------
For backward compatibility with existing callers (e.g., optimize.py expecting the
old hybrid output keys), this implementation still exposes a class named
`HybridDE_DPSO` and includes a synthetic "dpso" section in the result dict.
The "dpso" section now proxies the StagedDimPSO stage:

  result["dpso"]["best_values"]  => staged best continuous vector
  result["dpso"]["history"]      => staged aggregated global-best history
  (best_indices is an empty list placeholder)

Result Structure (dictionary) returned by run():
{
  "config": {...},
  "de": {...},
  "staged": { "best_fitness": ..., "best_position": [...], "stage_logs": [...] },
  "dpso": { ... synthetic alias to staged ... },
  "push": { ... }          # only if push_refine=True
  "final": {
      "best_fitness": ...,
      "best_position_continuous": [...],
      "source": "push" | "staged",
      "per_missile_times": [...]
  },
  "total_elapsed_sec": ...
}

Key Parameters (see __init__ signature for full list):
  problem_id: 2..5
  de_population / de_generations / de_F / de_CR
  staged_mode: fixed|shuffle|overlap|sensitivity|custom
  staged_window / staged_stride / staged_group_size
  staged_iterations_per_stage / staged_swarm_size
  push_enable (bool) – turn on/off PushPSO stage
  push_iterations / push_swarm / push_repeller_radius / push_repeller_strength
  maximize (bool)
  show_times (bool) compute per-missile time breakdown
  seed

Dependencies (internal project modules):
  - optimizer.de.DifferentialEvolution
  - optimizer.staged_pso.StagedDimPSO
  - optimizer.ppso.PushPSO
  - optimizer.pso (simulate_fitness utilities & judge switching)

Potential Extensions (not implemented here):
  - Adaptive scheduling between staged groups based on per-dimension gain.
  - Multi-cycle repetition: (DE small refresh) -> staged -> push -> staged ...
  - CMA-ES local insertion inside final PushPSO plateau.

Usage Example
-------------
    from optimizer.hybrid import HybridDE_DPSO
    hybrid = HybridDE_DPSO(
        problem_id=5,
        de_population=1024,
        de_generations=25,
        staged_mode="overlap",
        staged_window=5,
        staged_stride=3,
        push_enable=True,
        push_iterations=200,
        show_times=True,
        seed=2025,
        verbose=True,
    )
    result = hybrid.run()
    print("Final best:", result["final"]["best_fitness"])

"""

from __future__ import annotations

import os
import time
import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import multiprocessing as mp

# ---------------------------------------------------------------------------
# Project-local imports (guarded for standalone testing)
# ---------------------------------------------------------------------------
try:
    from .encoders import ENCODERS, StrategyEncoder  # type: ignore
    from .pso import simulate_fitness, select_judge  # type: ignore
    from .de import DifferentialEvolution  # type: ignore
    from .staged_pso import StagedDimPSO, StagedDimPSOResult, StageLog  # type: ignore
except ImportError:  # pragma: no cover
    from encoders import ENCODERS, StrategyEncoder  # type: ignore
    from pso import simulate_fitness, select_judge  # type: ignore
    from de import DifferentialEvolution  # type: ignore
    from staged_pso import StagedDimPSO, StagedDimPSOResult, StageLog  # type: ignore

# PushPSO optional import (only when push_enable=True)
_PushPSO = None
try:
    from .ppso import PushPSO  # type: ignore
    _PushPSO = PushPSO
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Helper: unified objective builder with fixed dt
# ---------------------------------------------------------------------------
def _build_objective(problem_id: int,
                     aggregate: str,
                     weights: Optional[Sequence[float]],
                     dt: float) -> Callable[[Sequence[float]], float]:
    def _obj(vec: Sequence[float]) -> float:
        return simulate_fitness(problem_id, vec, dt=dt, aggregate=aggregate, weights=weights)
    return _obj


# ---------------------------------------------------------------------------
# Dataclasses for stage summaries
# ---------------------------------------------------------------------------
@dataclass
class _StageSummary:
    best_fitness: float
    best_position: List[float]
    history: List[float]
    elapsed_sec: float
    eval_count: int
    history_stage_marks: Optional[List[float]] = None  # optional list of global best after each stage


# ---------------------------------------------------------------------------
# Main Hybrid Class (DE -> StagedDimPSO -> optional PushPSO)
# ---------------------------------------------------------------------------
class HybridDE_DPSO:  # name kept for backward compatibility with optimize.py
    def __init__(self,
                 problem_id: int,
                 aggregate: str = "sum",
                 weights: Optional[Sequence[float]] = None,
                 maximize: bool = True,
                 judge: str = "rough",
                 force_judge_on_init: bool = True,
                 sample_K: int = 32,
                 # --- DE stage ---
                 de_population: int = 1024,
                 de_generations: int = 30,
                 de_F: float = 0.75,
                 de_CR: float = 0.9,
                 de_seed: Optional[int] = None,
                 de_processes: Optional[int] = None,
                 de_bounds: Tuple[float, float] = (-1.0, 1.0),
                 de_warm_model: Optional[str] = None,
                 # --- StagedDimPSO stage ---
                 staged_mode: str = "overlap",            # fixed|shuffle|overlap|sensitivity|custom
                 staged_group_size: int = 5,
                 staged_window: int = 5,
                 staged_stride: int = 3,
                 staged_iterations_per_stage: int = 30,
                 staged_swarm_size: int = 48,
                 staged_max_stages: Optional[int] = None,
                 staged_refine_full: bool = True,
                 staged_refine_iterations: int = 60,
                 staged_seed: Optional[int] = None,
                 # --- PushPSO final stage ---
                 push_enable: bool = True,
                 push_iterations: int = 200,
                 push_swarm: int = 512,
                 push_inertia: float = 0.72,
                 push_cognitive: float = 1.49,
                 push_social: float = 1.49,
                 push_reset_prob: float = 0.03,
                 push_repeller_radius: float = 0.6,
                 push_repeller_strength: float = 1.2,
                 push_repeller_decay: float = 0.985,
                 push_seed: Optional[int] = None,
                 push_velocity_clamp: Optional[Tuple[float, float]] = None,
                 # --- Common / misc ---
                 dt: float = 0.05,
                 show_times: bool = True,
                 verbose: bool = True,
                 seed: Optional[int] = None,
                 models_dir: Optional[str] = None):
        """
        Constructor sets configuration for all three stages.
        """
        if problem_id not in ENCODERS:
            raise ValueError("Supported problem ids: 2,3,4,5.")
        if aggregate not in {"sum", "min", "weighted"}:
            raise ValueError("aggregate must be 'sum' | 'min' | 'weighted'")
        if aggregate == "weighted" and weights is None:
            raise ValueError("weights required when aggregate='weighted'")
        if staged_mode not in {"fixed", "shuffle", "overlap", "sensitivity", "custom"}:
            raise ValueError("staged_mode invalid")

        self.problem_id = problem_id
        self.encoder: StrategyEncoder = ENCODERS[problem_id]
        self.dim = self.encoder.dim
        self.aggregate = aggregate
        self.weights = list(weights) if weights is not None else None
        self.maximize = maximize
        self.judge = judge
        self.force_judge_on_init = force_judge_on_init
        self.sample_K = sample_K

        # Seeds
        base_seed = seed or int(time.time())
        self.random = random.Random(base_seed)
        self.seed = base_seed
        self.de_seed = de_seed or (base_seed + 101)
        self.staged_seed = staged_seed or (base_seed + 202)
        self.push_seed = push_seed or (base_seed + 303)

        # Stage configs
        self.de_population = de_population
        self.de_generations = de_generations
        self.de_F = de_F
        self.de_CR = de_CR
        self.de_processes = de_processes
        self.de_bounds = de_bounds
        self.de_warm_model = de_warm_model

        self.staged_mode = staged_mode
        self.staged_group_size = staged_group_size
        self.staged_window = staged_window
        self.staged_stride = staged_stride
        self.staged_iterations_per_stage = staged_iterations_per_stage
        self.staged_swarm_size = staged_swarm_size
        self.staged_max_stages = staged_max_stages
        self.staged_refine_full = staged_refine_full
        self.staged_refine_iterations = staged_refine_iterations

        self.push_enable = push_enable and (_PushPSO is not None)
        self.push_iterations = push_iterations
        self.push_swarm = push_swarm
        self.push_inertia = push_inertia
        self.push_cognitive = push_cognitive
        self.push_social = push_social
        self.push_reset_prob = push_reset_prob
        self.push_repeller_radius = push_repeller_radius
        self.push_repeller_strength = push_repeller_strength
        self.push_repeller_decay = push_repeller_decay
        self.push_velocity_clamp = push_velocity_clamp

        self.dt = dt
        self.show_times = show_times
        self.verbose = verbose
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )

        if self.force_judge_on_init:
            try:
                select_judge(
                    self.judge,
                    K=self.sample_K if self.judge == "sample" else 32,
                    verbose=self.verbose
                )
            except Exception as e:  # pragma: no cover
                if self.verbose:
                    print(f"[Hybrid] Judge selection failed ({e}); continuing.")

    # -----------------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        de_summary = self._run_de()
        staged_summary = self._run_staged(de_summary.best_position)
        push_summary: Optional[_StageSummary] = None

        best_fitness = staged_summary.best_fitness
        best_position = list(staged_summary.best_position)
        source = "staged"

        if self.push_enable:
            try:
                p_summary = self._run_push(best_position)
                push_summary = p_summary
                if self._better(p_summary.best_fitness, best_fitness):
                    best_fitness = p_summary.best_fitness
                    best_position = p_summary.best_position
                    source = "push"
            except Exception as e:  # pragma: no cover
                if self.verbose:
                    print(f"[Hybrid] Push stage skipped ({e})")

        per_missile_times = None
        if self.show_times:
            try:
                from .pso import build_problem_times_fn  # type: ignore
                times_fn = build_problem_times_fn(self.problem_id, dt=self.dt)
                per_missile_times = times_fn(best_position)
            except Exception:
                pass

        total_elapsed = time.time() - t0

        # Synthetic "dpso" alias for backward compatibility (maps to staged)
        dpso_alias = {
            "best_fitness": staged_summary.best_fitness,
            "best_indices": [],  # placeholder
            "best_values": list(staged_summary.best_position),
            "history": list(staged_summary.history),
            "elapsed_sec": staged_summary.elapsed_sec,
            "eval_count": staged_summary.eval_count,
        }

        result: Dict[str, Any] = {
            "config": self._config_dict(),
            "de": {
                "best_fitness": de_summary.best_fitness,
                "best_position": de_summary.best_position,
                "history": de_summary.history,
                "elapsed_sec": de_summary.elapsed_sec,
                "eval_count": de_summary.eval_count,
            },
            "staged": {
                "best_fitness": staged_summary.best_fitness,
                "best_position": staged_summary.best_position,
                "history": staged_summary.history,
                "stage_count": len(staged_summary.history_stage_marks) if staged_summary.history_stage_marks is not None else 0,
                "stage_marks": staged_summary.history_stage_marks if staged_summary.history_stage_marks is not None else [],
            },
            "dpso": dpso_alias,  # compatibility
            "final": {
                "best_fitness": best_fitness,
                "best_position_continuous": best_position,
                "source": source,
                "per_missile_times": per_missile_times,
            },
            "total_elapsed_sec": total_elapsed,
        }
        if push_summary is not None:
            result["push"] = {
                "best_fitness": push_summary.best_fitness,
                "best_position": push_summary.best_position,
                "history": push_summary.history,
                "elapsed_sec": push_summary.elapsed_sec,
                "eval_count": push_summary.eval_count,
            }

        if self.verbose:
            msg = "[Hybrid] Finished. DE {:.6f} -> Staged {:.6f}".format(
                de_summary.best_fitness, staged_summary.best_fitness
            )
            if push_summary:
                msg += " -> Push {:.6f}".format(push_summary.best_fitness)
            msg += f" | Final({source})={best_fitness:.6f}"
            print(msg)
        # Persist into unified model JSON (similar to optimize.run_hybrid behavior)
        try:
            from .optimize import merge_optimizer_section  # local import to avoid circular at module load
            persist_section = {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "best_fitness": float(best_fitness),
                "best_position": list(best_position),
                "de_best_fitness": float(de_summary.best_fitness),
                "staged_best_fitness": float(staged_summary.best_fitness),
                "push_best_fitness": float(push_summary.best_fitness) if push_summary else None,
                "source": source,
                "history_de": list(de_summary.history),
                "history_staged": list(staged_summary.history),
                "history_push": list(push_summary.history) if push_summary else None,
            }
            merge_optimizer_section(self.problem_id, "hybrid", persist_section, maximize=self.maximize)
        except Exception as e:
            if self.verbose:
                print(f"[Hybrid] Persist failed: {e}")
        return result

    # -----------------------------------------------------------------------
    # Stage 1: Differential Evolution
    # -----------------------------------------------------------------------
    def _run_de(self) -> _StageSummary:
        if self.verbose:
            print(f"[Hybrid:DE] pop={self.de_population} gens={self.de_generations}")
        # Use existing DifferentialEvolution for clean parallel evaluation
        objective = _build_objective(
            self.problem_id,
            aggregate=self.aggregate,
            weights=self.weights,
            dt=self.dt
        )
        de = DifferentialEvolution(
            dim=self.dim,
            objective=objective,
            population_size=self.de_population,
            generations=self.de_generations,
            F=self.de_F,
            CR=self.de_CR,
            maximize=self.maximize,
            processes=self.de_processes,
            seed=self.de_seed,
            init_model=self.de_warm_model,
            perturb_std=0.15,
            save_on_exit=None,
            include_swarm_on_save=False,
        )
        start = time.time()
        res = de.run()
        elapsed = time.time() - start
        return _StageSummary(
            best_fitness=float(res.best_fitness),
            best_position=res.best_position.tolist(),
            history=list(res.history),
            elapsed_sec=elapsed,
            eval_count=int(res.eval_count),
        )

    # -----------------------------------------------------------------------
    # Stage 2: StagedDimPSO refinement
    # -----------------------------------------------------------------------
    def _run_staged(self, init_best: List[float]):
        if self.verbose:
            print("[Hybrid:Staged] start mode={} window={} stride={} group_size={}".format(
                self.staged_mode, self.staged_window, self.staged_stride, self.staged_group_size
            ))
        objective = _build_objective(
            self.problem_id,
            aggregate=self.aggregate,
            weights=self.weights,
            dt=self.dt
        )
        lo = np.full(self.dim, -1.0)
        hi = np.full(self.dim, 1.0)
        staged = StagedDimPSO(
            dim=self.dim,
            objective=objective,
            mode=self.staged_mode,  # type: ignore[arg-type]
            group_size=self.staged_group_size,
            window_size=self.staged_window,
            stride=self.staged_stride,
            iterations_per_stage=self.staged_iterations_per_stage,
            swarm_size=self.staged_swarm_size,
            maximize=self.maximize,
            bounds=(lo, hi),
            seed=self.staged_seed,
            refine_full=self.staged_refine_full,
            refine_iterations=self.staged_refine_iterations,
            max_stages=self.staged_max_stages,
            verbose=self.verbose,
        )
        # Override initial full_best with DE best
        staged.full_best = np.asarray(init_best, dtype=float).copy()
        # NOTE: 不在这里立即调用 objective()，避免与 StagedDimPSO.run() 内部
        #       初始 base_fit 评估重复。让 run() 自己第一次评估 full_best。
        #       这样可以减少一次 simulate_fitness 调用。
        # 如果后续需要复用已有 fitness，可在 StagedDimPSO 中加入跳过初评估的 flag。
        # staged.full_best_fitness 保持为初始化值 (-inf / +inf)，run() 会覆盖。

        s_start = time.time()
        s_res: StagedDimPSOResult = staged.run()
        s_elapsed = time.time() - s_start

        # Aggregate a simple global history: global best after each stage
        stage_marks = [lg.global_best_after for lg in s_res.stage_logs]
        # Build synthetic history: refine by concatenating stage sub-histories' last values
        # Or simply store the piecewise global improvements (stage_marks).
        synthetic_history = stage_marks.copy()

        # Collect
        summary = _StageSummary(
            best_fitness=float(s_res.best_fitness),
            best_position=s_res.best_position.tolist(),
            history=synthetic_history,
            elapsed_sec=s_elapsed,
            eval_count=int(s_res.eval_count),
        )
        # Attach extra for later referencing
        summary.history_stage_marks = stage_marks  # type: ignore
        return summary

    # -----------------------------------------------------------------------
    # Stage 3: PushPSO final polishing (no crowding)
    # -----------------------------------------------------------------------
    def _run_push(self, init_position: List[float]) -> _StageSummary:
        if not self.push_enable or _PushPSO is None:
            raise RuntimeError("PushPSO not available or disabled.")

        if self.verbose:
            print(f"[Hybrid:Push] swarm={self.push_swarm} iters={self.push_iterations} crowding=OFF")

        objective = _build_objective(
            self.problem_id,
            aggregate=self.aggregate,
            weights=self.weights,
            dt=self.dt
        )
        push = _PushPSO(
            dim=self.dim,
            objective=objective,
            swarm_size=self.push_swarm,
            iterations=self.push_iterations,
            inertia=self.push_inertia,
            cognitive=self.push_cognitive,
            social=self.push_social,
            reset_prob=self.push_reset_prob,
            velocity_clamp=self.push_velocity_clamp,
            maximize=self.maximize,
            processes=max(1, mp.cpu_count() - 2),
            seed=self.push_seed,
            crowd_radius=None,
            density_threshold=6,
            push_strength=0.8,
            adaptive_push=True,
            enable_crowding=False,  # IMPORTANT: disable O(N^2) crowd cost
            stagnation_iter_threshold=20,
            cluster_eps=0.15,
            cluster_min_size=5,
            cluster_max_size=12,
            repeller_radius=self.push_repeller_radius,
            repeller_strength=self.push_repeller_strength,
            repeller_decay=self.push_repeller_decay,
            save_on_exit=None,
            include_swarm_on_save=False,
        )

        # Warm start around staged best (preserve exact best as particle 0 and seed fitness to avoid losing it)
        try:
            center = np.asarray(init_position, dtype=float)
            noise = np.random.normal(0, 0.15, (self.push_swarm, self.dim))
            push.positions = np.clip(center[None, :] + noise, -1.0, 1.0)
            # Force exact best into particle 0 (overwrite any noise)
            push.positions[0] = center
            push.personal_best_positions = push.positions.copy()
            if self.maximize:
                push.personal_best_fitness[:] = -np.inf
                push.global_best_fitness = -np.inf
            else:
                push.personal_best_fitness[:] = np.inf
                push.global_best_fitness = np.inf
            # Evaluate exact best once to seed global / personal best (prevents it from being "forgotten")
            seeded_fit = objective(center)
            push.personal_best_fitness[0] = seeded_fit
            push.global_best_fitness = seeded_fit
            push.global_best_position = center.copy()
        except Exception:  # pragma: no cover
            if self.verbose:
                print("[Hybrid:Push] Warm start failed, using default init.")

        p_start = time.time()
        pres = push.run()
        p_elapsed = time.time() - p_start

        return _StageSummary(
            best_fitness=float(pres.best_fitness),
            best_position=pres.best_position.tolist(),
            history=list(pres.history),
            elapsed_sec=p_elapsed,
            eval_count=int(pres.eval_count),
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _better(self, a: float, b: float) -> bool:
        return a > b if self.maximize else a < b

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "dim": self.dim,
            "aggregate": self.aggregate,
            "maximize": self.maximize,
            "judge": self.judge,
            "de_population": self.de_population,
            "de_generations": self.de_generations,
            "staged_mode": self.staged_mode,
            "staged_window": self.staged_window,
            "staged_stride": self.staged_stride,
            "staged_group_size": self.staged_group_size,
            "push_enable": self.push_enable,
            "push_iterations": self.push_iterations,
            "dt": self.dt,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Simple CLI demo
# ---------------------------------------------------------------------------
def _demo():
    hyb = HybridDE_DPSO(
        problem_id=2,
        de_population=128,
        de_generations=12,
        staged_mode="overlap",
        staged_window=4,
        staged_stride=2,
        staged_iterations_per_stage=15,
        push_enable=True,
        push_iterations=80,
        show_times=False,
        verbose=True,
        seed=2025,
    )
    out = hyb.run()
    print("Hybrid final best:", out["final"]["best_fitness"])


if __name__ == "__main__":  # pragma: no cover
    _demo()
