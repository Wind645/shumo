# Particle Swarm Optimization framework with parallel simulation evaluation.
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Optional, Union
import numpy as np
import multiprocessing as mp
import functools
import os
import sys

# Local import (lightweight) - handle script/module execution path issues
try:
    from simulator import Simulator
except ImportError:
    # When running as "python optimizer/pso.py", add project root to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from simulator import Simulator

"""
Design goals:
 - Generic PSO core (dimension-agnostic).
 - Parallel evaluation for expensive physics simulations (problem 2-5).
 - Random reset mechanism to mitigate premature convergence (escape local optima).
 - Strategy encoders/decoders translating flat particle vectors into Simulator strategy objects.
 - Flexible objective aggregation (sum / min / weighted) over multiple missiles.

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
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def positive(x: float) -> float:
    # Smooth-ish positivity
    return math.log1p(math.exp(x))  # softplus

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ----------------------------- Strategy encoders -----------------------------
class StrategyEncoder:
    """
    Base class: map particle vector -> simulator strategy object.
    Subclasses define dimension (dim) and implement encode(position) -> strategy.
    """
    dim: int

    def encode(self, position: Sequence[float]):
       # To be overridden
       raise NotImplementedError

    def initial_position(self) -> np.ndarray:
       # Random initial vector in [-1,1]
       return np.random.uniform(-1, 1, self.dim)


class Problem2Encoder(StrategyEncoder):
    dim = 4  # angle, speed_raw, release_raw, fuse_raw

    def encode(self, position: Sequence[float]):
        angle, speed_raw, release_raw, fuse_raw = position
        # angle -> direction on XY plane
        dir_vec = np.array([math.cos(angle), math.sin(angle), 0.0])
        speed = 70.0 + sigmoid(speed_raw)*70.0  # 70-140
        release_time = positive(release_raw)   # >=0
        fuse = clamp(positive(fuse_raw), 0.1, 15.0)
        strat = (dir_vec, float(speed), [[float(release_time), float(fuse)]])
        return strat


class Problem3Encoder(StrategyEncoder):
    dim = 8  # angle, speed_raw, r1,f1,gap2,f2,gap3,f3

    def encode(self, position: Sequence[float]):
        angle, speed_raw, r1_raw, f1_raw, gap2_raw, f2_raw, gap3_raw, f3_raw = position
        dir_vec = np.array([math.cos(angle), math.sin(angle), 0.0])
        speed = 70.0 + sigmoid(speed_raw)*70.0
        t1 = positive(r1_raw)
        t2 = t1 + 1.0 + positive(gap2_raw)
        t3 = t2 + 1.0 + positive(gap3_raw)
        bombs = [
            [float(t1), clamp(positive(f1_raw), 0.1, 15.0)],
            [float(t2), clamp(positive(f2_raw), 0.1, 15.0)],
            [float(t3), clamp(positive(f3_raw), 0.1, 15.0)],
        ]
        return (dir_vec, float(speed), bombs)


class Problem4Encoder(StrategyEncoder):
    # 3 drones * problem2 dims (4) = 12
    dim = 12

    def encode(self, position: Sequence[float]):
        # Split into 3 chunks
        chunks = [position[i:i+4] for i in range(0, 12, 4)]
        encoder = Problem2Encoder()
        drones = [encoder.encode(c) for c in chunks]
        # Return tuple expected by Simulator
        return tuple(drones)  # type: ignore


class Problem5Encoder(StrategyEncoder):
    # Each drone: angle, speed_raw, (act1,r1,f1),(act2,gap2,f2),(act3,gap3,f3) = 2 + 9 = 11
    dim = 11 * 5  # 55

    def encode_single(self, vec: Sequence[float]):
        angle, speed_raw, act1, r1_raw, f1_raw, act2, gap2_raw, f2_raw, act3, gap3_raw, f3_raw = vec
        dir_vec = np.array([math.cos(angle), math.sin(angle), 0.0])
        speed = 70.0 + sigmoid(speed_raw)*70.0
        # Activation
        a1 = sigmoid(act1)
        a2 = sigmoid(act2)
        a3 = sigmoid(act3)
        bombs = []
        # Build times sequentially for active bombs
        if a1 > 0.5:
            t1 = positive(r1_raw)
            bombs.append([float(t1), clamp(positive(f1_raw), 0.1, 15.0)])
            base_t = t1
        else:
            base_t = 0.0
        if a2 > 0.5:
            t2 = base_t + 1.0 + positive(gap2_raw)
            bombs.append([float(t2), clamp(positive(f2_raw), 0.1, 15.0)])
            base_t = t2
        if a3 > 0.5:
            t3 = base_t + 1.0 + positive(gap3_raw)
            bombs.append([float(t3), clamp(positive(f3_raw), 0.1, 15.0)])
        return (dir_vec, float(speed), bombs)

    def encode(self, position: Sequence[float]):
        drones = [self.encode_single(position[i:i+11]) for i in range(0, self.dim, 11)]
        return tuple(drones)  # type: ignore


# Map problem id -> encoder class helper
ENCODERS = {
    2: Problem2Encoder(),
    3: Problem3Encoder(),
    4: Problem4Encoder(),
    5: Problem5Encoder(),
}


# ----------------------------- Fitness / Objective wrappers -----------------------------
def simulate_fitness(problem_id: int,
                     strategy_vector: Sequence[float],
                     dt: float = 0.05,
                     aggregate: str = "sum",
                     weights: Optional[Sequence[float]] = None) -> float:
    """
    Convert a flat vector to simulator strategy using encoder,
    run simulation, compute occlusion metric.

    aggregate: 'sum' (default) sums occlusion times over missiles
               'min' uses minimum occlusion time (balance across missiles)
               'weighted' uses provided weights (len=missile_count)
    """
    encoder = ENCODERS[problem_id]
    strategy = encoder.encode(strategy_vector)
    sim = Simulator(problem_id=problem_id, dt=dt, strategy=strategy)
    sim.run_until_end()
    times = sim.compute_batch_occlusions()
    if not times:
        return 0.0
    # Penalty: if any drone releases two bombs with time separation < 1s multiply fitness by 0.2
    penalty_factor = 1.0
    # Normalize strategy form to iterable of drone triplets (dir, speed, bombs)
    if problem_id in (2, 3):
        drones = [strategy]
    else:
        drones = list(strategy)  # tuple -> list
    try:
        for d in drones:
            bombs = d[2]
            if not bombs or len(bombs) < 2:
                continue
            release_times = sorted(b[0] for b in bombs)
            for a, b in zip(release_times, release_times[1:]):
                if (b - a) < 1.0 - 1e-9:
                    penalty_factor = 0.2
                    break
            if penalty_factor < 1.0:
                break
    except Exception:
        # If structure unexpected, fall back without penalty
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
    # Store objective & config in globals for workers (pickle-safe)
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


# ----------------------------- Parallel PSO Core -----------------------------
@dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: List[float]
    eval_count: int
    elapsed: float


class ParallelPSO:
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
                 best_times_fn: Optional[Callable[[Sequence[float]], List[float]]] = None):
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

        # Swarm state
        self.positions = np.random.uniform(-1, 1, (swarm_size, dim))
        self.velocities = np.zeros((swarm_size, dim))
        self.personal_best_positions = self.positions.copy()
        # Initialize personal best fitness with -inf for maximization or +inf for minimization
        if self.maximize:
            self.personal_best_fitness = np.full(swarm_size, -np.inf)
            self.global_best_fitness = -np.inf
        else:
            self.personal_best_fitness = np.full(swarm_size, np.inf)
            self.global_best_fitness = np.inf
        self.global_best_position = self.positions[0].copy()

    def _evaluate_batch(self, batch: np.ndarray) -> List[float]:
        # Parallel map using global objective inside workers
        with mp.Pool(processes=self.processes,
                     initializer=_worker_init,
                     initargs=(self.seed, self.objective, self.maximize)) as pool:
            fitness = pool.map(_worker_eval_vector, batch)
        return fitness

    def _objective_wrapper(self, x: Sequence[float]) -> float:
        # Legacy (unused) kept for reference; evaluation handled in worker via globals
        return self.objective(x)

    def run(self) -> PSOResult:
        start = time.time()
        history = []
        eval_count = 0

        for it in range(self.iterations):
            # Random resets to escape local optima
            reset_mask = np.random.rand(self.swarm_size) < self.reset_prob
            if reset_mask.any():
                self.positions[reset_mask] = np.random.uniform(-1, 1,
                                                               (reset_mask.sum(), self.dim))
                self.velocities[reset_mask] = 0.0

            fitness = self._evaluate_batch(self.positions)
            eval_count += len(fitness)

            for i, fit in enumerate(fitness):
                better = fit > self.personal_best_fitness[i] if self.maximize else fit < self.personal_best_fitness[i]
                if better:
                    self.personal_best_fitness[i] = fit
                    self.personal_best_positions[i] = self.positions[i].copy()
            # Global best
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
                print(f"[PSO] Iter {it+1}/{self.iterations} best_fitness={self.global_best_fitness:.6f} per_missile={best_times}", flush=True)
            else:
                print(f"[PSO] Iter {it+1}/{self.iterations} best_fitness={self.global_best_fitness:.6f}", flush=True)

            # Update velocities & positions
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            cognitive_term = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_term = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive_term + social_term
            if self.velocity_clamp is not None:
                vmin, vmax = self.velocity_clamp
                np.clip(self.velocities, vmin, vmax, out=self.velocities)
            self.positions += self.velocities

        elapsed = time.time() - start
        return PSOResult(
            best_position=self.global_best_position.copy(),
            best_fitness=float(self.global_best_fitness),
            history=history,
            eval_count=eval_count,
            elapsed=elapsed
        )


# ----------------------------- Convenience factory -----------------------------
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
    # Use functools.partial so the callable is picklable for multiprocessing
    objective = functools.partial(_objective_dispatch,
                                  problem_id=problem_id,
                                  dt=dt,
                                  aggregate=aggregate,
                                  weights=weights)
    return encoder, objective

def build_problem_times_fn(problem_id: int,
                           dt: float = 0.05) -> Callable[[Sequence[float]], List[float]]:
    """
    Build a callable that returns the per-missile occlusion times (list[float]) for a
    given flat strategy vector, without applying aggregation or penalties.

    This is useful to pass into ParallelPSO as best_times_fn so each iteration
    can report the raw occlusion distribution for the current global best.

    Parameters
    ----------
    problem_id : int
        Problem identifier (2-5).
    dt : float
        Simulator timestep.

    Returns
    -------
    Callable[[Sequence[float]], List[float]]
        Function mapping a flat vector to per-missile occlusion seconds.
    """
    if problem_id not in ENCODERS:
        raise ValueError("PSO only needed for problems 2-5")
    encoder = ENCODERS[problem_id]

    def _times(vec: Sequence[float]) -> List[float]:
        strategy = encoder.encode(vec)
        sim = Simulator(problem_id=problem_id, dt=dt, strategy=strategy)
        sim.run_until_end()
        return sim.compute_batch_occlusions()

    return _times


# ----------------------------- Example usage -----------------------------
if __name__ == "__main__":
    # Example: optimize Problem 2 (single drone, one bomb) using strict full-coverage occlusion (now the default)
    problem_id = 2
    encoder, obj = build_problem_objective(problem_id, dt=0.01, aggregate="sum")
    times_fn = build_problem_times_fn(problem_id, dt=0.01)
    pso = ParallelPSO(dim=encoder.dim,
                      objective=obj,
                      swarm_size=72,
                      iterations=128,
                      reset_prob=0.05,
                      velocity_clamp=(-0.5, 0.5),
                      processes=4,
                      seed=42,
                      best_times_fn=times_fn)
    result = pso.run()
    print("Best fitness:", result.best_fitness)
    print("Best position vector:", result.best_position)
    # Decode to human-readable strategy
    strategy = encoder.encode(list(result.best_position))
    print("Decoded strategy:", strategy)
    print("History (best so far):", result.history)

    # You can similarly optimize problems 3-5 by changing problem_id, though
    # higher dimensions will require more iterations and larger swarm sizes.
