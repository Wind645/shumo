"""
Hybrid DE + DPSO Optimizer
==========================

This module implements a two-stage hybrid optimization pipeline tailored for
the projectile / drone strategy search problems (problems 2–5):

Stage 1 (Continuous Exploration - Differential Evolution):
  - Large-population Differential Evolution (DE) with dynamic simulation step
    size dt annealed linearly from dt_start (e.g. 0.10) down to dt_end (e.g. 0.05).
  - Uses the continuous encoders (same as PSO/DE in existing code) to explore
    broadly in the raw vector space [-1,1]^dim.
  - Population can be extremely large (default 1024), evaluated in parallel.
  - Batch evaluation supported to reduce peak memory / scheduling overhead.
  - Collects a pool of high-quality elite solutions for discretization.

Stage 2 (Combinatorial Refinement - Discrete PSO, DPSO):
  - Builds per-dimension categorical sets from the elite solutions discovered
    in Stage 1 (unique values, optionally padded to a minimum size).
  - Runs a DiscretePSO that recombines these elite per-dimension values to
    potentially reach better fitness through cross-dimension mixing that
    standard DE might not have sampled jointly.

Motivation:
  Large continuous evolutionary search (DE) is good at discovering promising
  *value levels* per dimension, while combinatorial recombination (DPSO)
  can exploit these levels more aggressively without needing gradient info.

Key Features:
  - Clean API with a single HybridDE_DPSO class.
  - JSON warm start support delegated to underlying components if needed.
  - Judge selection (rough vs sample) optionally forced to rough for speed.
  - Flexible elite selection & category capping.
  - Graceful guards against category explosion.

Dependencies (already present in project):
  - optimizer.de.DifferentialEvolution
  - optimizer.dpso.DiscretePSO
  - optimizer.pso (simulate_fitness helpers / select_judge)
  - optimizer.encoders (ENCODERS mapping)

Usage Example
-------------
    from optimizer.hybrid import HybridDE_DPSO

    hybrid = HybridDE_DPSO(
        problem_id=4,
        de_generations=40,
        de_population=1024,
        dt_start=0.10,
        dt_end=0.05,
        dpso_iterations=60,
        dpso_swarm_size=180,
        elite_fraction=0.08,
        per_dim_category_cap=32,
        min_categories_per_dim=6,
        judge="rough",
        show_times=True,
        seed=2025,
    )

    result = hybrid.run()
    print("Hybrid best fitness:", result['final']['best_fitness'])

Result Structure (dict):
{
  "config": {...},     # resolved configuration
  "de": {
      "best_fitness": ...,
      "best_position": [...],
      "history": [...],
      "elapsed_sec": ...,
      "eval_count": ...,
      "elite_count": ...,
  },
  "dpso": {
      "best_fitness": ...,
      "best_indices": [...],
      "best_values": [...],
      "history": [...],
      "elapsed_sec": ...,
      "eval_count": ...
  },
  "final": {           # alias to dpso stage summary for convenience
      "best_fitness": ...,
      "best_values": [...],
      "best_position_continuous": [...],   # decoded from discrete pick
  }
}

Notes:
  - Continuous best (DE) and discrete best (DPSO recombination) may differ.
  - DPSO categories are raw continuous values from DE (still in [-1,1]).
  - The encoder mapping (angle, speed raw, etc.) is applied consistently only
    inside objective evaluations (no need to pre-transform categories).

Potential Extensions (not implemented):
  - Adaptive clustering (k-means) before DPSO to reduce redundancy.
  - Multi-pass hybridization (DE -> DPSO -> local PSO).
  - Mixed continuous-discrete metaheuristics (some dims continuous, some discrete).
"""

from __future__ import annotations

import os
import math
import time
import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import multiprocessing as mp

# Reuse existing project modules
try:
    from .encoders import ENCODERS, StrategyEncoder  # type: ignore
    from .pso import simulate_fitness, select_judge  # type: ignore
    from .dpso import DiscretePSO, build_integer_categories, DiscretePSOResult  # type: ignore
except ImportError:  # pragma: no cover
    from encoders import ENCODERS, StrategyEncoder  # type: ignore
    from pso import simulate_fitness, select_judge  # type: ignore
    from dpso import DiscretePSO, build_integer_categories, DiscretePSOResult  # type: ignore

try:
    from .pso import _worker_init, _worker_eval_vector  # type: ignore
except ImportError:  # pragma: no cover
    from pso import _worker_init, _worker_eval_vector  # type: ignore

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _linear_schedule(step: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return end
    alpha = step / (total - 1)
    return start + (end - start) * alpha


def _build_dynamic_objective(problem_id: int,
                             aggregate: str,
                             weights: Optional[Sequence[float]],
                             dt_getter: Callable[[], float]) -> Callable[[Sequence[float]], float]:
    """
    Returns objective(vec) that uses up-to-date dt via dt_getter().
    """
    def _obj(vec: Sequence[float]) -> float:
        dt = dt_getter()
        return simulate_fitness(problem_id, vec, dt=dt, aggregate=aggregate, weights=weights)
    return _obj


@dataclass
class _DEState:
    population: np.ndarray
    fitness: np.ndarray
    best_fitness: float
    best_position: np.ndarray
    history: List[float]
    eval_count: int
    elapsed: float


# -----------------------------------------------------------------------------
# Hybrid main class
# -----------------------------------------------------------------------------
class HybridDE_DPSO:
    """
    Two-stage hybrid: Differential Evolution (continuous) followed by Discrete PSO.

    Parameters:
      problem_id                : int, in {2,3,4,5}
      aggregate                 : "sum" | "min" | "weighted"
      weights                   : list of floats if aggregate == "weighted"
      maximize                  : maximize objective (True) else minimize
      judge                     : "rough" | "sample" (rough recommended for DE stage)
      select_judge_on_init      : if True, calls select_judge(judge, ...)
      sample_K                  : K for sampling judge (only if judge == "sample")
      # --- DE Stage ---
      de_population             : population size (large, e.g., 1024)
      de_generations            : total generations
      de_F                      : differential weight F
      de_CR                     : crossover probability
      dt_start                  : initial simulation step (e.g., 0.10)
      dt_end                    : final simulation step (e.g., 0.05)
      de_batch_size             : evaluate population in chunks (None=all)
      de_processes              : worker processes (None => cpu_count-1)
      # --- Elite Extraction ---
      elite_fraction            : fraction (0,1] of final DE population as elite
      elite_top_k               : explicit cap on number of elite solutions (optional)
      per_dim_category_cap      : maximum category count per dimension post-merge
      min_categories_per_dim    : ensure at least this many categories (pad via sampling)
      category_pad_noise_std    : stddev for padding new category values
      # --- DPSO Stage ---
      dpso_swarm_size           : number of discrete particles
      dpso_iterations           : iterations
      dpso_inertia              : DPSO w
      dpso_cognitive            : DPSO c1
      dpso_social               : DPSO c2
      dpso_reset_prob           : per-particle reset probability
      dpso_noise_std            : Gaussian score noise
      dpso_temperature_decay    : optional multiplicative decay for (c1,c2)
      # --- Misc ---
      seed                      : random seed
      show_times                : if True, print per-missile times for final hybrids
      verbose                   : progress printing
      models_dir                : base models dir (future extension / logging)
    """
    def __init__(self,
                 problem_id: int,
                 aggregate: str = "sum",
                 weights: Optional[Sequence[float]] = None,
                 maximize: bool = True,
                 judge: str = "rough",
                 select_judge_on_init: bool = True,
                 sample_K: int = 32,
                 # DE Stage
                 de_population: int = 1024,
                 de_generations: int = 40,
                 de_F: float = 0.8,
                 de_CR: float = 0.9,
                 dt_start: float = 0.10,
                 dt_end: float = 0.05,
                 de_batch_size: Optional[int] = None,
                 de_processes: Optional[int] = None,
                 # Elite extraction
                 elite_fraction: float = 0.05,
                 elite_top_k: Optional[int] = None,
                 per_dim_category_cap: int = 48,
                 min_categories_per_dim: int = 4,
                 category_pad_noise_std: float = 0.05,
                 # DPSO Stage
                 dpso_swarm_size: int = 200,
                 dpso_iterations: int = 60,
                 dpso_inertia: float = 0.65,
                 dpso_cognitive: float = 1.4,
                 dpso_social: float = 1.4,
                 dpso_reset_prob: float = 0.03,
                 dpso_noise_std: float = 0.01,
                 dpso_temperature_decay: Optional[float] = None,
                 # Misc
                 seed: Optional[int] = None,
                 show_times: bool = True,
                 verbose: bool = True,
                 models_dir: Optional[str] = None):
        if problem_id not in ENCODERS:
            raise ValueError("Supported problem ids: 2,3,4,5")
        if aggregate not in {"sum", "min", "weighted"}:
            raise ValueError("aggregate must be sum|min|weighted")
        if aggregate == "weighted" and weights is None:
            raise ValueError("weights must be provided when aggregate='weighted'")
        if elite_fraction <= 0 or elite_fraction > 1:
            raise ValueError("elite_fraction must be in (0,1]")
        if per_dim_category_cap < 1:
            raise ValueError("per_dim_category_cap must be >=1")
        if min_categories_per_dim < 1:
            raise ValueError("min_categories_per_dim must be >=1")
        if dt_start <= 0 or dt_end <= 0:
            raise ValueError("dt values must be positive")

        self.problem_id = problem_id
        self.encoder: StrategyEncoder = ENCODERS[problem_id]
        self.dim = self.encoder.dim
        self.aggregate = aggregate
        self.weights = list(weights) if weights is not None else None
        self.maximize = maximize
        self.judge = judge
        self.select_judge_on_init = select_judge_on_init
        self.sample_K = sample_K

        # DE config
        self.de_population = de_population
        self.de_generations = de_generations
        self.de_F = de_F
        self.de_CR = de_CR
        self.dt_start = dt_start
        self.dt_end = dt_end
        self.de_batch_size = de_batch_size
        self.de_processes = de_processes or max(1, (mp.cpu_count() - 1))

        # Elite
        self.elite_fraction = elite_fraction
        self.elite_top_k = elite_top_k
        self.per_dim_category_cap = per_dim_category_cap
        self.min_categories_per_dim = min_categories_per_dim
        self.category_pad_noise_std = category_pad_noise_std

        # DPSO
        self.dpso_swarm_size = dpso_swarm_size
        self.dpso_iterations = dpso_iterations
        self.dpso_inertia = dpso_inertia
        self.dpso_cognitive = dpso_cognitive
        self.dpso_social = dpso_social
        self.dpso_reset_prob = dpso_reset_prob
        self.dpso_noise_std = dpso_noise_std
        self.dpso_temperature_decay = dpso_temperature_decay

        self.seed = seed or int(time.time())
        self.show_times = show_times
        self.verbose = verbose
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.select_judge_on_init:
            select_judge(
                self.judge,
                K=self.sample_K if self.judge == "sample" else 32,
                verbose=self.verbose
            )

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        t0 = time.time()

        # Stage 1: Differential Evolution with dt annealing
        de_state = self._run_de_stage()

        # Construct categories from DE elites
        categories = self._build_discrete_categories(de_state.population, de_state.fitness)

        # Stage 2: DPSO over constructed categories
        dpso_result = self._run_dpso_stage(categories)

        # Build final continuous best (mapping discrete pick -> vector)
        # Here discrete categories already are raw continuous dimension values.
        best_continuous = list(map(float, dpso_result.best_values))

        if self.show_times:
            # Evaluate final times (simulate_fitness already sums; we want the individual missile times)
            # Reuse the encoder and simulator path (avoid duplication).
            from .pso import build_problem_times_fn  # type: ignore
            times_fn = build_problem_times_fn(self.problem_id, dt=self.dt_end)
            per_missile_times = times_fn(best_continuous)
        else:
            per_missile_times = None

        final_summary = {
            "best_fitness": float(dpso_result.best_fitness),
            "best_values": [float(v) for v in dpso_result.best_values],
            "best_position_continuous": best_continuous,
            "per_missile_times": per_missile_times,
        }

        total_elapsed = time.time() - t0
        out = {
            "config": self._config_dict(),
            "de": {
                "best_fitness": float(de_state.best_fitness),
                "best_position": de_state.best_position.tolist(),
                "history": de_state.history,
                "elapsed_sec": float(de_state.elapsed),
                "eval_count": int(de_state.eval_count),
                "elite_count": self._elite_count(de_state.population.shape[0]),
                "dt_start": self.dt_start,
                "dt_end": self.dt_end,
            },
            "dpso": {
                "best_fitness": float(dpso_result.best_fitness),
                "best_indices": dpso_result.best_indices.tolist(),
                "best_values": [float(v) for v in dpso_result.best_values],
                "history": dpso_result.history,
                "elapsed_sec": float(dpso_result.elapsed),
                "eval_count": int(dpso_result.eval_count),
            },
            "final": final_summary,
            "total_elapsed_sec": total_elapsed,
        }

        if self.verbose:
            print("[Hybrid] Completed. DE best={:.6f} DPSO best={:.6f}".format(
                de_state.best_fitness, dpso_result.best_fitness
            ))
        return out

    # -------------------------------------------------------------------------
    # Internal: DE Stage
    # -------------------------------------------------------------------------
    def _run_de_stage(self) -> _DEState:
        """
        Custom DE loop with dynamic dt scheduling.
        Mirrors classic DE/rand/1/bin with shared multiprocess pool.
        """
        dim = self.dim
        pop_size = self.de_population
        F = self.de_F
        CR = self.de_CR
        generations = self.de_generations
        maximize = self.maximize

        # Initialize population in [-1,1]
        population = np.random.uniform(-1.0, 1.0, (pop_size, dim))

        # dt schedule getter (mutable generation index)
        current_gen = {"g": 0}

        def dt_getter():
            return _linear_schedule(current_gen["g"], generations, self.dt_start, self.dt_end)

        objective = _build_dynamic_objective(
            self.problem_id,
            self.aggregate,
            self.weights,
            dt_getter
        )

        fitness = np.full(pop_size, -np.inf if maximize else np.inf)
        best_fitness = -np.inf if maximize else np.inf
        best_position = population[0].copy()
        history: List[float] = []
        eval_count = 0

        start = time.time()
        processes = self.de_processes

        if self.verbose:
            print(f"[Hybrid:DE] Start: pop={pop_size} gens={generations} dt={self.dt_start}->{self.dt_end} processes={processes}")

        # Shared pool
        with mp.Pool(
            processes=processes,
            initializer=_worker_init,
            initargs=(self.seed, objective, maximize)
        ) as pool:

            # Evaluate initial
            init_fit = self._eval_population(population, pool)
            eval_count += len(init_fit)
            for i, f in enumerate(init_fit):
                fitness[i] = f

            if maximize:
                idx = int(np.argmax(fitness))
                best_fitness = fitness[idx]
            else:
                idx = int(np.argmin(fitness))
                best_fitness = fitness[idx]
            best_position = population[idx].copy()
            history.append(float(best_fitness))

            # Main generations
            for g in range(generations):
                current_gen["g"] = g  # update dt schedule
                dt_now = dt_getter()

                if self.verbose and (g % max(1, generations // 10) == 0):
                    print(f"[Hybrid:DE] Gen {g+1}/{generations} dt={dt_now:.4f} best={best_fitness:.6f}")

                trials = np.empty_like(population)

                # Mutation + Crossover (DE/rand/1/bin)
                for i in range(pop_size):
                    idxs = list(range(pop_size))
                    idxs.remove(i)
                    r1, r2, r3 = random.sample(idxs, 3)
                    x1 = population[r1]
                    x2 = population[r2]
                    x3 = population[r3]
                    mutant = x1 + F * (x2 - x3)

                    # (Optional) Bound correction -> clip to [-1,1]
                    np.clip(mutant, -1.5, 1.5, out=mutant)  # slightly generous, then soft clip
                    mutant = np.clip(mutant, -1.0, 1.0)

                    target = population[i]
                    trial = target.copy()
                    j_rand = random.randrange(dim)
                    for j in range(dim):
                        if random.random() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trials[i] = trial

                # Evaluate trials
                trial_fit = self._eval_population(trials, pool)
                eval_count += len(trial_fit)

                # Selection
                improved_any = False
                for i in range(pop_size):
                    tf = trial_fit[i]
                    better = tf > fitness[i] if maximize else tf < fitness[i]
                    if better:
                        population[i] = trials[i]
                        fitness[i] = tf
                        improved_any = True
                if improved_any:
                    if maximize:
                        idx = int(np.argmax(fitness))
                        bf = fitness[idx]
                        if bf > best_fitness:
                            best_fitness = bf
                            best_position = population[idx].copy()
                    else:
                        idx = int(np.argmin(fitness))
                        bf = fitness[idx]
                        if bf < best_fitness:
                            best_fitness = bf
                            best_position = population[idx].copy()

                history.append(float(best_fitness))

            if self.verbose:
                print(f"[Hybrid:DE] Finished gens={generations} best={best_fitness:.6f}")

        elapsed = time.time() - start
        return _DEState(
            population=population,
            fitness=fitness,
            best_fitness=float(best_fitness),
            best_position=best_position.copy(),
            history=history,
            eval_count=eval_count,
            elapsed=elapsed
        )

    def _eval_population(self, population: np.ndarray, pool) -> List[float]:
        """
        Evaluate a population using existing worker mechanism with optional batching.
        """
        if self.de_batch_size is None or self.de_batch_size >= len(population):
            return pool.map(_worker_eval_vector, population.tolist())
        out: List[float] = []
        bs = self.de_batch_size
        for start in range(0, len(population), bs):
            chunk = population[start:start+bs]
            out.extend(pool.map(_worker_eval_vector, chunk.tolist()))
        return out

    # -------------------------------------------------------------------------
    # Elite extraction -> categories for DPSO
    # -------------------------------------------------------------------------
    def _elite_count(self, pop_size: int) -> int:
        count = max(1, int(math.ceil(pop_size * self.elite_fraction)))
        if self.elite_top_k is not None:
            count = min(count, self.elite_top_k)
        return count

    def _build_discrete_categories(self,
                                   population: np.ndarray,
                                   fitness: np.ndarray) -> List[List[float]]:
        pop_size, dim = population.shape
        assert dim == self.dim
        elite_count = self._elite_count(pop_size)

        # Sort indices by fitness
        if self.maximize:
            order = np.argsort(-fitness)
        else:
            order = np.argsort(fitness)
        elite_indices = order[:elite_count]
        elite_vectors = population[elite_indices]

        if self.verbose:
            print(f"[Hybrid] Elite extraction: selected top {elite_count} / {pop_size}")

        categories: List[List[float]] = []
        for d in range(dim):
            vals = elite_vectors[:, d]
            uniq = sorted(set(float(v) for v in vals))
            # If too many categories, down-sample (e.g., via quantile selection)
            if len(uniq) > self.per_dim_category_cap:
                # Uniform pick in sorted order by index stride
                stride = len(uniq) / self.per_dim_category_cap
                reduced = []
                for k in range(self.per_dim_category_cap):
                    idx = int(round(k * stride))
                    if idx >= len(uniq):
                        idx = len(uniq) - 1
                    reduced.append(uniq[idx])
                uniq = sorted(set(reduced))
            # Pad if too few
            while len(uniq) < self.min_categories_per_dim:
                # Sample noise around mean (or 0) to diversify
                mu = statistics.fmean(uniq) if uniq else 0.0
                new_val = float(np.clip(
                    mu + np.random.normal(0, self.category_pad_noise_std),
                    -1.0, 1.0
                ))
                uniq.append(new_val)
                uniq = sorted(set(uniq))
            categories.append(uniq)

        if self.verbose:
            cat_sizes = [len(c) for c in categories]
            total_combinations = 1
            for s in cat_sizes:
                total_combinations *= s
                if total_combinations > 1e9:  # soft guard
                    break
            print(f"[Hybrid] Category sizes per dim: {cat_sizes}")
            if total_combinations > 1e8:
                print("[Hybrid] Warning: DPSO search space is large "
                      f"({total_combinations} combinations) – may be slow.")

        return categories

    # -------------------------------------------------------------------------
    # DPSO Stage
    # -------------------------------------------------------------------------
    def _run_dpso_stage(self, categories: List[List[float]]) -> DiscretePSOResult:
        # Objective wrapper: categories already raw continuous parameters in [-1,1]
        # We just feed into simulator via encoder after adding them as vector.
        def obj_from_raw(raw_vec: Sequence[float]) -> float:
            return simulate_fitness(
                self.problem_id,
                raw_vec,
                dt=self.dt_end,
                aggregate=self.aggregate,
                weights=self.weights
            )

        dpso = DiscretePSO(
            categories=categories,
            objective=obj_from_raw,
            swarm_size=self.dpso_swarm_size,
            iterations=self.dpso_iterations,
            inertia=self.dpso_inertia,
            cognitive=self.dpso_cognitive,
            social=self.dpso_social,
            reset_prob=self.dpso_reset_prob,
            noise_std=self.dpso_noise_std,
            maximize=self.maximize,
            processes=max(1, self.de_processes // 2),
            seed=self.seed + 1337,
            pass_indices=False,
            init_model=None,
            perturb_prob=0.15,
            models_dir=self.models_dir,
            save_on_exit=None,
            include_swarm_on_save=False,
            temperature_decay=self.dpso_temperature_decay,
            min_score_clip=-15.0,
            max_score_clip=15.0
        )
        if self.verbose:
            print(f"[Hybrid:DPSO] Start categories_dim={len(categories)} swarm={self.dpso_swarm_size} iters={self.dpso_iterations}")
        result = dpso.run()
        if self.verbose:
            print(f"[Hybrid:DPSO] Done best={result.best_fitness:.6f}")
        return result

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _config_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "dim": self.dim,
            "aggregate": self.aggregate,
            "maximize": self.maximize,
            "de_population": self.de_population,
            "de_generations": self.de_generations,
            "dt_start": self.dt_start,
            "dt_end": self.dt_end,
            "elite_fraction": self.elite_fraction,
            "elite_top_k": self.elite_top_k,
            "per_dim_category_cap": self.per_dim_category_cap,
            "min_categories_per_dim": self.min_categories_per_dim,
            "dpso_swarm_size": self.dpso_swarm_size,
            "dpso_iterations": self.dpso_iterations,
            "seed": self.seed,
            "judge": self.judge,
        }


# -----------------------------------------------------------------------------
# Standalone execution demo
# -----------------------------------------------------------------------------
def _demo():
    # Simple demonstration for problem_id=2 (quick run).
    hybrid = HybridDE_DPSO(
        problem_id=2,
        de_population=128,
        de_generations=12,
        dt_start=0.10,
        dt_end=0.05,
        dpso_swarm_size=80,
        dpso_iterations=20,
        elite_fraction=0.15,
        per_dim_category_cap=16,
        min_categories_per_dim=5,
        show_times=False,
        verbose=True,
        seed=1234
    )
    res = hybrid.run()
    print("[Hybrid Demo] Final best fitness:", res["final"]["best_fitness"])
    print("[Hybrid Demo] Best discrete values:", res["final"]["best_values"])

if __name__ == "__main__":  # pragma: no cover
    _demo()
