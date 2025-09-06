from __future__ import annotations
"""
Simple PSO (Particle Swarm Optimization) for Problem 2.
Decision vector x = [speed, azimuth, release_time, explode_delay]
Objective: minimize objective_q2_vector(x) which equals -occluded_time.

Constants and bounds are kept simple and adjustable.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List
import math
import random
import numpy as np

from api.objectives import objective_q2_vector
from api.problems import evaluate_problem2
from vectorized_judge import vectorized_circle_fully_occluded_by_sphere
from simcore.constants import (
    C_BASE_DEFAULT as _C_BASE,
    R_CYL_DEFAULT as _R_CYL,
    H_CYL_DEFAULT as _H_CYL,
    SMOKE_LIFETIME_DEFAULT as _SMOKE_LIFE,
    SMOKE_DESCENT_DEFAULT as _SMOKE_DESCENT,
    SMOKE_RADIUS_DEFAULT as _SMOKE_RADIUS,
)
from simcore.constants import G as _G

# Missile spec (copied from api/decision.MISSILES_DEF['M1'])
_MISSILE_POS0 = np.array([20000.0, 0.0, 2000.0], dtype=np.float64)
_MISSILE_SPEED = 300.0
_MISSILE_TARGET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

def _missile_dir_and_flight_time():
    d = _MISSILE_TARGET - _MISSILE_POS0
    n = np.linalg.norm(d)
    return d / n, n / _MISSILE_SPEED

_MISSILE_DIR, _MISSILE_T_FLIGHT = _missile_dir_and_flight_time()

def _compute_cloud_center0(drone_pos0: np.ndarray, drone_dir: np.ndarray, speed: np.ndarray,
                           release_time: np.ndarray, explode_delay: np.ndarray) -> np.ndarray:
    """Vectorized bomb explosion position (cloud initial center)."""
    # position at release
    rel_pos = drone_pos0 + drone_dir * speed[:, np.newaxis] * release_time[:, np.newaxis]
    # velocity (horizontal only)
    vel = drone_dir * speed[:, np.newaxis]
    dt = explode_delay[:, np.newaxis]
    a = np.array([0.0, 0.0, -_G], dtype=rel_pos.dtype)
    center0 = rel_pos + vel * dt + 0.5 * a * (dt * dt)
    return center0

def _batch_occluded_time(params: np.ndarray, *, dt: float) -> np.ndarray:
    """Compute occluded time (seconds) for each candidate parameter vector.

    params: (N,4) -> speed, azimuth, release_time, explode_delay
    Returns array (N,) occluded_time for missile M1 using judge_caps semantics (both caps occluded).
    Uses vectorized_judge for per-time-step cap occlusion tests. Timeline loop over t, batch over candidates.
    """
    N = params.shape[0]
    dtype = np.float64
    p = params.astype(dtype)
    speed = p[:, 0]
    az = p[:, 1]
    t_rel = p[:, 2]
    dly = p[:, 3]

    # Basic validity masks (mirror objective_q2_vector constraints)
    valid = (speed >= 70.0) & (speed <= 140.0) & (t_rel >= 0.0) & (dly > 0.0)
    occluded_time = np.zeros(N, dtype=dtype)
    if not valid.any():
        return occluded_time  # all zero

    # Drone spec (fixed as in evaluate_problem2)
    drone_pos0 = np.array([17800.0, 0.0, 1800.0], dtype=dtype)[np.newaxis, :].repeat(N, axis=0)
    drone_dir = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)

    # Precompute cloud initial centers
    center0 = _compute_cloud_center0(drone_pos0, drone_dir, speed, t_rel, dly)  # (N,3)
    explode_time = t_rel + dly
    cloud_end_time = explode_time + _SMOKE_LIFE

    # Missile path times
    T_f = float(_MISSILE_T_FLIGHT)
    n_steps = int(T_f / dt) + 1
    t_grid = np.linspace(0.0, T_f, n_steps, dtype=dtype)
    missile_pos = _MISSILE_POS0[np.newaxis, :] + _MISSILE_DIR * (_MISSILE_SPEED * t_grid[:, np.newaxis])
    # Cylinder caps constants
    Cb = _C_BASE.astype(dtype)
    Ct = Cb + np.array([0.0, 0.0, float(_H_CYL)], dtype=dtype)
    r_cyl = np.array(float(_R_CYL), dtype=dtype)

    # Iterate timeline; vectorize across active candidates
    for ti, t in enumerate(t_grid):
        # active candidates with cloud present and valid
        act_mask = valid & (t >= explode_time) & (t <= cloud_end_time)
        if not act_mask.any():
            continue
        idx = np.nonzero(act_mask)[0]
        V = missile_pos[ti][np.newaxis, :].repeat(len(idx), axis=0)  # (k,3)
        # Cloud center at time t: center0 + [0,0, -desc * (t - explode_time)]
        tau = np.maximum(0.0, t - explode_time[idx])
        center_t = center0[idx].copy()
        center_t[:, 2] = center0[idx][:, 2] - _SMOKE_DESCENT * tau
        R_cloud = np.full((len(idx),), float(_SMOKE_RADIUS), dtype=dtype)

        # Bottom cap transform (already z=0 plane)
        Vb = V.copy()
        Cb_flat = Cb[np.newaxis, :].repeat(len(idx), axis=0)
        Sb = center_t.copy()
        rb = np.full((len(idx),), float(r_cyl), dtype=dtype)

        res_b = vectorized_circle_fully_occluded_by_sphere(Vb, Cb_flat, rb, Sb, R_cloud)
        oc_b = np.array(res_b.occluded)

        # Top cap transform: shift by Ct.z
        shift_z = Ct[2]
        Vt = V.copy(); Vt[:, 2] = V[:, 2] - shift_z
        Ct_flat = Ct.copy(); Ct_flat[2] = 0.0
        Ct_batch = Ct_flat[np.newaxis, :].repeat(len(idx), axis=0)
        St = center_t.copy(); St[:, 2] = St[:, 2] - shift_z
        rt = rb  # same radius
        res_t = vectorized_circle_fully_occluded_by_sphere(Vt, Ct_batch, rt, St, R_cloud)
        oc_t = np.array(res_t.occluded)

        oc_both = oc_b & oc_t
        if oc_both.any():
            occluded_time[idx[oc_both]] += dt
    return occluded_time


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
    vmax_frac: float = 0.6,
    method: str = "judge_caps",
    dt: float = 0.02,
    seed: Optional[int] = None,
    bounds: Tuple[Tuple[float, float], ...] = BOUNDS_DEFAULT,
    verbose: bool = False,
    use_vectorized: bool = True,
    # ---- 泛化支持 ----
    problem: int = 2,
    bombs_count: int = 2,
) -> PSOResult:
    """PSO for Q2 with optional numpy-parallel batch evaluation via vectorized_judge.

    When use_vectorized=True and method == 'judge_caps', a custom batched objective
    replaces the slower Python simulation for swarm evaluation. The returned
    best_eval is still computed with the canonical simulator for consistency.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if problem == 2:
        dim = 4
        assert len(bounds) == dim
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
    else:
        dim = 2 + 2 * bombs_count
        if bounds is BOUNDS_DEFAULT or len(bounds) == 4:
            b_list: List[Tuple[float, float]] = [
                (70.0, 140.0),
                (-math.pi, math.pi),
            ]
            for _ in range(bombs_count):
                b_list.append((0.0, 60.0))
                b_list.append((0.2, 12.0))
            lo = np.array([b[0] for b in b_list], dtype=np.float64)
            hi = np.array([b[1] for b in b_list], dtype=np.float64)
        else:
            lo = np.array([b[0] for b in bounds], dtype=np.float64)
            hi = np.array([b[1] for b in bounds], dtype=np.float64)
    span = hi - lo
    vmax = vmax_frac * span

    # Initialize swarm
    X = lo + np.random.rand(pop, dim).astype(np.float64) * span
    V = (np.random.rand(pop, dim).astype(np.float64) - 0.5) * 2.0 * vmax

    def _decode_eval(vec: np.ndarray) -> float:
        if problem == 2:
            return float(objective_q2_vector(vec, method=method, dt=dt))
        speed = float(vec[0]); az = float(vec[1])
        bombs = []
        for i in range(bombs_count):
            t = float(vec[2 + 2 * i]); d = float(vec[2 + 2 * i + 1])
            bombs.append((t, d))
        try:
            if problem == 3:
                from api.problems import evaluate_problem3
                res = evaluate_problem3(bombs=bombs, speed=speed, azimuth=az, dt=dt, occlusion_method=method)
            elif problem == 4:
                from api.problems import evaluate_problem4
                drones_spec = [{
                    'pos0': [17800.0, 0.0, 1800.0],
                    'speed': speed,
                    'azimuth': az,
                    'bombs': [{'deploy_time': t, 'explode_delay': d} for (t, d) in bombs],
                }]
                res = evaluate_problem4(drones_spec=drones_spec, dt=dt, occlusion_method=method)
            elif problem == 5:
                from api.problems import evaluate_problem5
                drones_spec = [{
                    'pos0': [17800.0, 0.0, 1800.0],
                    'speed': speed,
                    'azimuth': az,
                    'bombs': [{'deploy_time': t, 'explode_delay': d} for (t, d) in bombs],
                }]
                res = evaluate_problem5(drones_spec=drones_spec, dt=dt, occlusion_method=method)
            else:
                return 0.0
        except Exception:
            return 0.0
        oc = res.get('occluded_time', {})
        if isinstance(oc, dict):
            v = oc.get('M1', res.get('total', 0.0))
        else:
            v = res.get('total', 0.0)
        return -float(v)

    def eval_obj_single(x: Iterable[float]) -> float:
        return _decode_eval(np.asarray(list(x), dtype=np.float64))

    def eval_batch(Xb: np.ndarray) -> np.ndarray:
        if problem == 2 and use_vectorized and method == "judge_caps" and Xb.shape[1] == 4:
            occluded_time = _batch_occluded_time(Xb, dt=dt)
            return -occluded_time.astype(np.float64)
        vals = [eval_obj_single(Xb[i]) for i in range(Xb.shape[0])]
        return np.array(vals, dtype=np.float64)

    # Personal and global bests
    P = X.copy()
    pbest = eval_batch(P)
    g_idx = int(np.argmin(pbest))
    g = P[g_idx].copy()
    gbest = float(pbest[g_idx])

    if verbose:
        print(f"init gbest={gbest:.6f} x={g.tolist()}")

    for it in range(iters):
        r1 = np.random.rand(pop, dim).astype(np.float64)
        r2 = np.random.rand(pop, dim).astype(np.float64)
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        V = np.maximum(np.minimum(V, vmax), -vmax)  # clamp
        X = X + V
        # Bounds handling
        below = X < lo
        above = X > hi
        if below.any() or above.any():
            X = np.maximum(np.minimum(X, hi), lo)
            V[below] *= -0.5
            V[above] *= -0.5
        # Batch evaluation
        vals = eval_batch(X)
        improved = vals < pbest
        if improved.any():
            pbest[improved] = vals[improved]
            P[improved] = X[improved]
            best_val_iter = np.min(pbest)
            idx_iter = np.argmin(pbest)
            if float(best_val_iter) < gbest - 1e-15:
                gbest = float(best_val_iter)
                g = P[idx_iter].copy()
        if verbose and (it % max(1, iters // 10) == 0 or it == iters - 1):
            print(f"iter {it+1}/{iters} gbest={gbest:.6f}")

    # Final authoritative evaluation via full simulator (objective value already minimized form)
    if problem == 2:
        best_eval = evaluate_problem2(
            speed=float(g[0]),
            azimuth=float(g[1]),
            release_time=float(g[2]),
            explode_delay=float(g[3]),
            occlusion_method=method,
            dt=dt,
        )
    else:
        speed = float(g[0]); az = float(g[1])
        bombs = []
        for i in range(bombs_count):
            t = float(g[2 + 2 * i]); d = float(g[2 + 2 * i + 1])
            bombs.append((t, d))
        if problem == 3:
            from api.problems import evaluate_problem3
            best_eval = evaluate_problem3(bombs=bombs, speed=speed, azimuth=az, dt=dt, occlusion_method=method)
        elif problem == 4:
            from api.problems import evaluate_problem4
            drones_spec = [{
                'pos0': [17800.0, 0.0, 1800.0],
                'speed': speed,
                'azimuth': az,
                'bombs': [{'deploy_time': t, 'explode_delay': d} for (t, d) in bombs],
            }]
            best_eval = evaluate_problem4(drones_spec=drones_spec, dt=dt, occlusion_method=method)
        elif problem == 5:
            from api.problems import evaluate_problem5
            drones_spec = [{
                'pos0': [17800.0, 0.0, 1800.0],
                'speed': speed,
                'azimuth': az,
                'bombs': [{'deploy_time': t, 'explode_delay': d} for (t, d) in bombs],
            }]
            best_eval = evaluate_problem5(drones_spec=drones_spec, dt=dt, occlusion_method=method)
        else:
            best_eval = {'occluded_time': {'M1': -float(gbest)}, 'total': -float(gbest)}
    return PSOResult(best_x=g, best_value=gbest, best_eval=best_eval)


if __name__ == "__main__":
    # Tiny demo run
    out = solve_q2_pso(pop=12, iters=20, method="sampling", seed=42, verbose=True)
    bx = out.best_x
    print("Best x:", bx)
    print("Best value (minimized):", out.best_value)
    print("M1 遮蔽时长(s):", out.best_eval["occluded_time"]["M1"], "总计(s):", out.best_eval["total"])
