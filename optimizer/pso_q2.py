from __future__ import annotations
"""
Simple PSO (Particle Swarm Optimization) for Problem 2.
Decision vector x = [speed, azimuth, release_time, explode_delay]
Objective: minimize objective_q2_vector(x) which equals -occluded_time.

Constants and bounds are kept simple and adjustable.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import math
import random
import numpy as np

from api.objectives import objective_q2_vector
from api.problems import evaluate_problem2

# Default bounds (can be overridden in solve_q2_pso args)
BOUNDS_DEFAULT: Tuple[Tuple[float, float], ...] = (
    (70.0, 140.0),   # speed
    (-math.pi, math.pi),  # azimuth (rad)
    (0.0, 30.0),     # release_time (s)
    (0.2, 12.0),     # explode_delay (s)
)

@dataclass
class PSOResult:
    best_x: np.ndarray
    best_value: float
    best_eval: Dict


def solve_q2_pso(
    *,
    pop: int = 70,
    iters: int = 200,
    w: float = 0.65,
    c1: float = 1.6,
    c2: float = 1.6,
    vmax_frac: float = 0.6,  # max velocity as fraction of (hi-lo)
    method: str = "judge_caps",  # or "sampling"
    dt: float = 0.02,
    seed: Optional[int] = None,
    bounds: Tuple[Tuple[float, float], ...] = BOUNDS_DEFAULT,
    verbose: bool = False,
) -> PSOResult:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dim = 4
    assert len(bounds) == dim
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo
    vmax = vmax_frac * span

    # Initialize swarm
    X = lo + np.random.rand(pop, dim) * span
    V = (np.random.rand(pop, dim) - 0.5) * 2.0 * vmax

    def eval_obj(x: Iterable[float]) -> float:
        return float(objective_q2_vector(x, method=method, dt=dt))

    # Personal and global bests
    P = X.copy()
    pbest = np.array([eval_obj(P[i]) for i in range(pop)], dtype=float)
    g_idx = int(np.argmin(pbest))
    g = P[g_idx].copy()
    gbest = float(pbest[g_idx])

    if verbose:
        print(f"init gbest={gbest:.6f} x={g}")

    for t in range(iters):
        r1 = np.random.rand(pop, dim)
        r2 = np.random.rand(pop, dim)
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        # Clamp velocity
        V = np.clip(V, -vmax, vmax)
        X = X + V
        # Handle bounds by clipping and reflecting velocity where clipped
        for i in range(pop):
            xi = X[i]
            for d in range(dim):
                if xi[d] < lo[d]:
                    xi[d] = lo[d]
                    V[i, d] = -0.5 * V[i, d]
                elif xi[d] > hi[d]:
                    xi[d] = hi[d]
                    V[i, d] = -0.5 * V[i, d]
        # Evaluate and update bests
        for i in range(pop):
            val = eval_obj(X[i])
            if val < pbest[i]:
                pbest[i] = val
                P[i] = X[i].copy()
                if val < gbest:
                    gbest = float(val)
                    g = X[i].copy()
        if verbose and (t % max(1, iters // 10) == 0 or t == iters - 1):
            print(f"iter {t+1}/{iters} gbest={gbest:.6f}")

    best_eval = evaluate_problem2(
        speed=float(g[0]),
        azimuth=float(g[1]),
        release_time=float(g[2]),
        explode_delay=float(g[3]),
        occlusion_method=method,
        dt=dt,
    )
    return PSOResult(best_x=g, best_value=gbest, best_eval=best_eval)


if __name__ == "__main__":
    # Tiny demo run
    out = solve_q2_pso(pop=12, iters=20, method="sampling", seed=42, verbose=True)
    bx = out.best_x
    print("Best x:", bx)
    print("Best value (minimized):", out.best_value)
    print("M1 遮蔽时长(s):", out.best_eval["occluded_time"]["M1"], "总计(s):", out.best_eval["total"])
