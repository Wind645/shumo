# Particle Swarm Optimization framework with parallel simulation evaluation + JSON model initialization & persistence.
from __future__ import annotations

# import math removed (unused after refactor)
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Optional, Union, Any, Dict
import numpy as np
import multiprocessing as mp
import functools
import os
import sys
import json
from datetime import datetime

try:
    from simulator import Simulator
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from simulator import Simulator

"""
Design goals:
 - Generic PSO core (dimension-agnostic).
 - Parallel evaluation for expensive physics simulations (problem 2-5).
 - Random reset mechanism to mitigate premature convergence (escape local optima).
 - Strategy encoders/decoders translating flat particle vectors into Simulator strategy objects.
 - Flexible objective aggregation (sum / min / weighted) over multiple missiles.
 - NEW: Ability to bootstrap swarm from previously saved JSON model (warm start) and
        automatically save new results for iterative refinement.

JSON warm start file (placed under ../models):
{
  "schema": 1,
  "problem_id": 2,
  "dim": 4,
  "saved_at": "2025-09-06T12:34:56Z",
  "best_fitness": 4.321,
  "best_position": [...],
  "swarm_positions": [[...], [...], ...],          # optional
  "notes": "optional free-form text"
}

Loading rules:
 - If 'swarm_positions' matches the requested dimension it seeds the swarm directly.
 - Else if only 'best_position' is present, the whole swarm is initialized as
   best_position plus Gaussian noise (std=perturb_std).
 - Dimension mismatches raise ValueError (protect against accidental misuse).

Saving:
 - Call ParallelPSO.save_model(name) manually OR pass save_on_exit="run_name"
   into the constructor to auto-save after run().
 - When include_swarm=True we also persist full swarm positions.
 - Files are written to ../models/{name}.json ('.json' appended if missing).

Encoding overview (all continuous variables; angles in radians):

Problem 2 (single drone, 1 bomb)
  vector = [angle, speed_raw, release_raw, fuse_raw]
  speed = 70 + sigmoid(speed_raw)*70  -> [70,140]
  release_time = max(0, release_raw)  (upper bounded later)
  fuse_delay  = clamp(fuse_raw, 0.1, 15)

Problem 3 (single drone, 3 bombs)
  vector = [angle, speed_raw,
            r1_raw, f1_raw,
            gap2_raw, f2_raw,
            gap3_raw, f3_raw]
  release times constructed enforcing ordering & 1 s separation:
    t1 = positive(r1_raw)
    t2 = t1 + 1 + positive(gap2_raw)
    t3 = t2 + 1 + positive(gap3_raw)

Problem 4 (3 drones, each 1 bomb) => 3 * Problem2

Problem 5 (5 drones, each up to 3 bombs):
  For each drone:
    [angle, speed_raw,
     act1, r1_raw, f1_raw,
     act2, gap2_raw, f2_raw,
     act3, gap3_raw, f3_raw]
  actX in (0,1) via sigmoid -> active if > 0.5
  Release schedule constructed like problem 3 when active.
  Inactive bombs skipped (not passed to simulator).

NOTE: Simulator expects tuple of per-drone entries:
   (direction_vec, speed_float, [[release_time, fuse_delay], ...])
where direction_vec is np.array([dx, dy, 0]) already normalized.

The PSO below is maximization by default (higher fitness better).
"""

# ----------------------------- Utility functions -----------------------------
# sigmoid imported from encoders.sigmoid

# positive imported from encoders.positive

# clamp imported from encoders.clamp


# ----------------------------- Strategy encoders (moved) -----------------------------
# Refactored: encoder classes have been moved to encoders.py.
# We keep backward compatibility by importing the public mapping and base class.
from .encoders import (
    StrategyEncoder,
    ENCODERS,
    Problem2Encoder,
    Problem3Encoder,
    Problem4Encoder,
    Problem5Encoder,
    sigmoid,
    positive,
    clamp,
)

# ----------------------------- Judge selection -----------------------------
CURRENT_JUDGE = "rough"
def select_judge(name: str, *, K: int = 32, verbose: bool = True):
    """
    选择遮挡判定函数 (judge)。必须在创建 Simulator / 运行 PSO 前调用。

    可选:
      "rough" / "batch_rough"   : 角度解析法 (judges.batch_rough.is_sphere_blocked_vectorized)
      "sample" / "batch_sample" : 采样法 (judges.batch_sample.is_cylinder_blocked_vectorized) — 用上下底面各 K 个点。

    两者对 Simulator 的接口统一为:
        fn(data: ndarray shape (N,7)) -> ndarray(bool shape (N,))

    参数:
      name: 选择名称
      K:    仅当选择 sample 时生效，表示每个底面采样点数 (默认 32)
      verbose: 是否打印切换信息

    用法:
      from optimizer.pso import select_judge
      select_judge("sample", K=48)
      # 之后 build_problem_objective / ParallelPSO 即使用采样判定

    注意:
      - 多进程下 (PSO 使用进程池) 每个子进程初始化时会重新导入模块，
        因此在创建 PSO 之前调用即可；子进程中会看到修改后的 simulator.is_sphere_blocked_vectorized。
      - 若在运行中途再次切换，对已经运行中的仿真不追溯，但后续新建 Simulator 会使用新 judge。
    """
    global CURRENT_JUDGE
    import simulator as _sim
    if name in {"rough", "batch_rough"}:
        from judges.batch_rough import is_sphere_blocked_vectorized as _fn
        _sim.is_sphere_blocked_vectorized = _fn
        CURRENT_JUDGE = "rough"
        if verbose:
            print("[Judge] Switched to rough analytic judge.")
    elif name in {"sample", "batch_sample"}:
        from judges.batch_sample import is_cylinder_blocked_vectorized
        def _wrapped(data, _K=K):
            return is_cylinder_blocked_vectorized(data, K=_K)
        _wrapped.__name__ = "is_sphere_blocked_vectorized"
        _sim.is_sphere_blocked_vectorized = _wrapped
        CURRENT_JUDGE = f"sample(K={K})"
        if verbose:
           print(f"[Judge] Switched to sampling judge with K={K}.")
    else:
        raise ValueError("Unknown judge name. Use 'rough' or 'sample'.")


# ----------------------------- Fitness / Objective wrappers -----------------------------
def simulate_fitness(problem_id: int,
                     strategy_vector: Sequence[float],
                     dt: float = 0.05,
                     aggregate: str = "sum",
                     weights: Optional[Sequence[float]] = None) -> float:
    encoder = ENCODERS[problem_id]
    strategy = encoder.encode(strategy_vector)
    sim = Simulator(problem_id=problem_id, dt=dt, strategy=strategy)
    sim.run_until_end()
    times = sim.compute_batch_occlusions()
    if not times:
        return 0.0
    penalty_factor = 1.0
    if problem_id in (2, 3):
        drones = [strategy]
    else:
        drones = list(strategy)  # type: ignore
    try:
        for d in drones:
            bombs = d[2]
            if not bombs or len(bombs) < 2:
                continue
            release_times = sorted(b[0] for b in bombs)
            for a, b in zip(release_times, release_times[1:]):
                if (b - a) < 1.0 - 1e-9:
                    penalty_factor = 0.3
                    break
            if penalty_factor < 1.0:
                break
    except Exception:
        pass
    if aggregate == "sum":
        base = float(sum(times))
    elif aggregate == "min":
        base = float(min(times))
    elif aggregate == "weighted":
        if weights is None:
           raise ValueError("weights required for weighted aggregation")
        if len(weights) != len(times):
           raise ValueError("weights length mismatch")
        base = float(sum(w*t for w, t in zip(weights, times)))
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")
    return penalty_factor * base


_GLOBAL_OBJECTIVE = None
_GLOBAL_MAXIMIZE = True

def _worker_init(seed_base: int, objective, maximize: bool):
    global _GLOBAL_OBJECTIVE, _GLOBAL_MAXIMIZE
    _GLOBAL_OBJECTIVE = objective
    _GLOBAL_MAXIMIZE = maximize
    pid = mp.current_process().pid or 0
    base_seed = seed_base + pid
    random.seed(base_seed)
    np.random.seed(base_seed)

def _worker_eval_vector(x: Sequence[float]):
    try:
        return _GLOBAL_OBJECTIVE(x)  # type: ignore
    except Exception:
        return -1e9 if _GLOBAL_MAXIMIZE else 1e9


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: List[float]
    eval_count: int
    elapsed: float


class ParallelPSO:
    """
    Parallel Particle Swarm Optimizer with optional warm start from JSON model
    and automatic persistence.

    Parameters (additions):
      init_model: Optional base name of JSON file in models directory used to seed swarm.
      perturb_std: Stddev for Gaussian noise added around best_position when only
                   a single vector is available in the JSON warm start.
      models_dir: Override models directory (default ../models relative to this file).
      save_on_exit: If provided, automatically save model with this base name after run().
      include_swarm_on_save: Whether to persist full swarm when auto-saving.
    """
    def __init__(self,
                 dim: int,
                 objective: Callable[[Sequence[float]], float],
                 swarm_size: int = 40,
                 iterations: int = 100,
                 inertia: float = 0.72,
                 cognitive: float = 1.49,
                 social: float = 1.49,
                 reset_prob: float = 0.02,
                 velocity_clamp: Optional[Tuple[float, float]] = None,
                 maximize: bool = True,
                 processes: Optional[int] = None,
                 seed: Optional[int] = None,
                 best_times_fn: Optional[Callable[[Sequence[float]], List[float]]] = None,
                 init_model: Optional[str] = None,
                 perturb_std: float = 0.1,
                 models_dir: Optional[str] = None,
                 save_on_exit: Optional[str] = None,
                 include_swarm_on_save: bool = True,
                 bounds: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None):
        self.dim = dim
        self.objective = objective
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = inertia
        self.c1 = cognitive
        self.c2 = social
        self.reset_prob = reset_prob
        self.velocity_clamp = velocity_clamp
        self.maximize = maximize
        self.processes = processes or max(1, mp.cpu_count() - 1)
        self.seed = seed or int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.best_times_fn = best_times_fn
        self.init_model = init_model
        self.perturb_std = float(perturb_std)
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.save_on_exit = save_on_exit
        self.include_swarm_on_save = include_swarm_on_save

        # Bounds (optional)
        if bounds is not None:
            lo = np.asarray(bounds[0], dtype=float)
            hi = np.asarray(bounds[1], dtype=float)
            if lo.shape != (dim,) or hi.shape != (dim,):
                raise ValueError("bounds arrays must have shape (dim,)")
            if np.any(hi <= lo):
                raise ValueError("All hi bounds must be > lo bounds")
            self.bounds: Optional[Tuple[np.ndarray, np.ndarray]] = (lo, hi)
        else:
            self.bounds = None

        # Initialize swarm (random first; may be overwritten by warm start)
        if self.bounds is None:
            self.positions = np.random.uniform(-1, 1, (swarm_size, dim))
        else:
            lo, hi = self.bounds
            self.positions = lo + (hi - lo) * np.random.rand(swarm_size, dim)
        self.velocities = np.zeros((swarm_size, dim))
        self.personal_best_positions = self.positions.copy()
        if self.maximize:
            self.personal_best_fitness = np.full(swarm_size, -np.inf)
            self.global_best_fitness = -np.inf
        else:
            self.personal_best_fitness = np.full(swarm_size, np.inf)
            self.global_best_fitness = np.inf
        self.global_best_position = self.positions[0].copy()

        # Attempt warm start
        if self.init_model:
            try:
                self._load_warm_start(self.init_model)
            except Exception as e:
                print(f"[PSO] Warm start load failed ({e}); falling back to random initialization.", flush=True)

    # ----------------- Warm Start Loader -----------------
    def _resolve_model_path(self, name: str) -> str:
        if not name.endswith(".json"):
            name = name + ".json"
        return os.path.join(self.models_dir, name)

    def _load_warm_start(self, name: str):
        path = self._resolve_model_path(name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Warm start JSON root must be object")
        if "dim" in data and data["dim"] != self.dim:
            raise ValueError(f"Warm start dim mismatch: file={data['dim']} expected={self.dim}")
        if "swarm_positions" in data:
            swarm = data["swarm_positions"]
            if (not isinstance(swarm, list)) or len(swarm) == 0:
                raise ValueError("swarm_positions must be non-empty list")
            arr = np.asarray(swarm, dtype=float)
            if arr.shape[1] != self.dim:
                raise ValueError(f"swarm_positions inner dim mismatch: {arr.shape[1]} != {self.dim}")
            # Resize or pad/crop to swarm_size
            if arr.shape[0] >= self.swarm_size:
                self.positions = arr[:self.swarm_size].copy()
            else:
                # Tile to reach desired size
                reps = (self.swarm_size + arr.shape[0] - 1) // arr.shape[0]
                tiled = np.tile(arr, (reps, 1))[:self.swarm_size]
                self.positions = tiled.copy()
            print(f"[PSO] Warm start: loaded {arr.shape[0]} swarm positions from {os.path.basename(path)}", flush=True)
        elif "best_position" in data:
            best_pos = np.asarray(data["best_position"], dtype=float)
            if best_pos.shape[0] != self.dim:
                raise ValueError(f"best_position dim mismatch: {best_pos.shape[0]} != {self.dim}")
            noise = np.random.normal(0.0, self.perturb_std, (self.swarm_size, self.dim))
            self.positions = best_pos[None, :] + noise
            print(f"[PSO] Warm start: seeded swarm around best_position with std={self.perturb_std}", flush=True)
        else:
            raise ValueError("Warm start JSON must contain 'swarm_positions' or 'best_position'")
        # Reset dependent state
        self.velocities = np.zeros_like(self.positions)
        self.personal_best_positions = self.positions.copy()
        if self.maximize:
            self.personal_best_fitness.fill(-np.inf)
            self.global_best_fitness = -np.inf
        else:
            self.personal_best_fitness.fill(np.inf)
            self.global_best_fitness = np.inf
        self.global_best_position = self.positions[0].copy()
        print(f"[PSO] Warm start complete. Swarm size={self.swarm_size} dim={self.dim}", flush=True)

    # ----------------- Persistence -----------------
    def save_model(self,
                   name: str,
                   include_swarm: bool = True,
                   notes: Optional[str] = None,
                   extra: Optional[Dict[str, Any]] = None):
        os.makedirs(self.models_dir, exist_ok=True)
        path = self._resolve_model_path(name)
        payload: Dict[str, Any] = {
            "schema": 1,
            "dim": self.dim,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "best_fitness": float(self.global_best_fitness),
            "best_position": self.global_best_position.tolist(),
        }
        if include_swarm:
            payload["swarm_positions"] = self.positions.tolist()
        if notes:
            payload["notes"] = notes
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[PSO] Model saved to {path}", flush=True)

    # ----------------- Core -----------------
    def _evaluate_batch(self, batch: np.ndarray) -> List[float]:
        with mp.Pool(processes=self.processes,
                     initializer=_worker_init,
                     initargs=(self.seed, self.objective, self.maximize)) as pool:
            fitness = pool.map(_worker_eval_vector, batch)
        return fitness

    def _objective_wrapper(self, x: Sequence[float]) -> float:
        return self.objective(x)

    def run(self, iterations: Optional[int] = None) -> PSOResult:
        start = time.time()
        history: List[float] = []
        eval_count = 0

        iter_count = self.iterations if iterations is None else int(iterations)
        for it in range(iter_count):
            reset_mask = np.random.rand(self.swarm_size) < self.reset_prob
            if reset_mask.any():
                if self.bounds is None:
                    self.positions[reset_mask] = np.random.uniform(-1, 1,
                                                                    (reset_mask.sum(), self.dim))
                else:
                    lo, hi = self.bounds
                    self.positions[reset_mask] = lo + (hi - lo) * np.random.rand(reset_mask.sum(), self.dim)
                self.velocities[reset_mask] = 0.0

            fitness = self._evaluate_batch(self.positions)
            eval_count += len(fitness)

            for i, fit in enumerate(fitness):
                better = fit > self.personal_best_fitness[i] if self.maximize else fit < self.personal_best_fitness[i]
                if better:
                    self.personal_best_fitness[i] = fit
                    self.personal_best_positions[i] = self.positions[i].copy()
            best_idx = int(np.argmax(self.personal_best_fitness) if self.maximize else
                           np.argmin(self.personal_best_fitness))
            best_fit = self.personal_best_fitness[best_idx]
            better_global = best_fit > self.global_best_fitness if self.maximize else best_fit < self.global_best_fitness
            if better_global:
                self.global_best_fitness = best_fit
                self.global_best_position = self.personal_best_positions[best_idx].copy()

            history.append(self.global_best_fitness)
            if self.best_times_fn is not None:
                best_times = self.best_times_fn(self.global_best_position)
                print(f"[PSO] Iter {it+1}/{iter_count} best_fitness={self.global_best_fitness:.6f} per_missile={best_times}", flush=True)
            else:
                print(f"[PSO] Iter {it+1}/{iter_count} best_fitness={self.global_best_fitness:.6f}", flush=True)

            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            cognitive_term = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_term = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive_term + social_term
            if self.velocity_clamp is not None:
                vmin, vmax = self.velocity_clamp
                np.clip(self.velocities, vmin, vmax, out=self.velocities)
            self.positions += self.velocities
            if self.bounds is not None:
                lo, hi = self.bounds
                np.clip(self.positions, lo, hi, out=self.positions)

        elapsed = time.time() - start
        result = PSOResult(
            best_position=self.global_best_position.copy(),
            best_fitness=float(self.global_best_fitness),
            history=history,
            eval_count=eval_count,
            elapsed=elapsed
        )
        if self.save_on_exit:
            try:
                self.save_model(self.save_on_exit, include_swarm=self.include_swarm_on_save)
            except Exception as e:
                print(f"[PSO] Auto-save failed: {e}", flush=True)
        return result


# ----------------- Block-based PSO moved -----------------
# Optional re-export: if unavailable, expose None placeholders.
try:
    from .block_pso import BlockPSO, BlockPSOResult  # type: ignore
except Exception:  # pragma: no cover
    BlockPSO = None  # type: ignore
    BlockPSOResult = None  # type: ignore


def _objective_dispatch(vec: Sequence[float],
                        problem_id: int,
                        dt: float,
                        aggregate: str,
                        weights: Optional[Sequence[float]]):
    return simulate_fitness(problem_id, vec, dt=dt, aggregate=aggregate, weights=weights)

def build_problem_objective(problem_id: int,
                            dt: float = 0.05,
                            aggregate: str = "sum",
                            weights: Optional[Sequence[float]] = None) -> Tuple[StrategyEncoder, Callable[[Sequence[float]], float]]:
    if problem_id not in ENCODERS:
        raise ValueError("PSO only needed for problems 2-5")
    encoder = ENCODERS[problem_id]
    objective = functools.partial(_objective_dispatch,
                                  problem_id=problem_id,
                                  dt=dt,
                                  aggregate=aggregate,
                                  weights=weights)
    return encoder, objective

def build_problem_times_fn(problem_id: int,
                           dt: float = 0.05) -> Callable[[Sequence[float]], List[float]]:
    if problem_id not in ENCODERS:
        raise ValueError("PSO only needed for problems 2-5")
    encoder = ENCODERS[problem_id]
    def _times(vec: Sequence[float]) -> List[float]:
        strategy = encoder.encode(vec)
        sim = Simulator(problem_id=problem_id, dt=dt, strategy=strategy)
        sim.run_until_end()
        return sim.compute_batch_occlusions()
    return _times


if __name__ == "__main__":
    # Example usage moved to optimize.py (see optimizer/optimize.py for a runnable demo using
    # both ParallelPSO and BlockPSO).
    pass
