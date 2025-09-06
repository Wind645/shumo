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
import torch

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
_MISSILE_POS0 = torch.tensor([20000.0, 0.0, 2000.0], dtype=torch.float64)
_MISSILE_SPEED = 300.0
_MISSILE_TARGET = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

def _missile_dir_and_flight_time():
    d = _MISSILE_TARGET - _MISSILE_POS0
    n = torch.linalg.norm(d)
    return d / n, n / _MISSILE_SPEED

_MISSILE_DIR, _MISSILE_T_FLIGHT = _missile_dir_and_flight_time()

def _compute_cloud_center0(drone_pos0: torch.Tensor, drone_dir: torch.Tensor, speed: torch.Tensor,
                           release_time: torch.Tensor, explode_delay: torch.Tensor) -> torch.Tensor:
    """Vectorized bomb explosion position (cloud initial center)."""
    # position at release
    rel_pos = drone_pos0 + drone_dir * speed.unsqueeze(1) * release_time.unsqueeze(1)
    # velocity (horizontal only)
    vel = drone_dir * speed.unsqueeze(1)
    dt = explode_delay.unsqueeze(1)
    a = torch.tensor([0.0, 0.0, -_G], dtype=rel_pos.dtype, device=rel_pos.device)
    center0 = rel_pos + vel * dt + 0.5 * a * (dt * dt)
    return center0

def _batch_occluded_time(params: torch.Tensor, *, dt: float, device: torch.device) -> torch.Tensor:
    """Compute occluded time (seconds) for each candidate parameter vector.

    params: (N,4) -> speed, azimuth, release_time, explode_delay
    Returns tensor (N,) occluded_time for missile M1 using judge_caps semantics (both caps occluded).
    Uses vectorized_judge for per-time-step cap occlusion tests. Timeline loop over t, batch over candidates.
    """
    with torch.no_grad():
        N = params.shape[0]
        dtype = torch.float64
        p = params.to(device=device, dtype=dtype)
        speed = p[:, 0]
        az = p[:, 1]
        t_rel = p[:, 2]
        dly = p[:, 3]

        # Basic validity masks (mirror objective_q2_vector constraints)
        valid = (speed >= 70.0) & (speed <= 140.0) & (t_rel >= 0.0) & (dly > 0.0)
        occluded_time = torch.zeros(N, dtype=dtype, device=device)
        if not valid.any():
            return occluded_time  # all zero

        # Drone spec (fixed as in evaluate_problem2)
        drone_pos0 = torch.tensor([17800.0, 0.0, 1800.0], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)
        drone_dir = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], dim=1)

        # Precompute cloud initial centers
        center0 = _compute_cloud_center0(drone_pos0, drone_dir, speed, t_rel, dly)  # (N,3)
        explode_time = t_rel + dly
        cloud_end_time = explode_time + _SMOKE_LIFE

        # Missile path times
        T_f = float(_MISSILE_T_FLIGHT)
        n_steps = int(T_f / dt) + 1
        t_grid = torch.linspace(0.0, T_f, n_steps, device=device, dtype=dtype)
        missile_pos = _MISSILE_POS0.to(device=device, dtype=dtype).unsqueeze(0) + _MISSILE_DIR.to(device) * (_MISSILE_SPEED * t_grid).unsqueeze(1)
        # Cylinder caps constants
        Cb = _C_BASE.to(device=device, dtype=dtype)
        Ct = Cb + torch.tensor([0.0, 0.0, float(_H_CYL)], dtype=dtype, device=device)
        r_cyl = torch.tensor(float(_R_CYL), dtype=dtype, device=device)

        # Iterate timeline; vectorize across active candidates
        for ti, t in enumerate(t_grid):
            # active candidates with cloud present and valid
            act_mask = valid & (t >= explode_time) & (t <= cloud_end_time)
            if not act_mask.any():
                continue
            idx = torch.nonzero(act_mask, as_tuple=False).squeeze(1)
            V = missile_pos[ti].unsqueeze(0).repeat(idx.numel(), 1)  # (k,3)
            # Cloud center at time t: center0 + [0,0, -desc * (t - explode_time)]
            tau = (t - explode_time[idx]).clamp(min=0.0)
            center_t = center0[idx].clone()
            center_t[:, 2] = center0[idx][:, 2] - _SMOKE_DESCENT * tau
            R_cloud = torch.full((idx.numel(),), float(_SMOKE_RADIUS), device=device, dtype=dtype)

            # Bottom cap transform (already z=0 plane)
            Vb = V.clone()
            Cb_flat = Cb.clone().unsqueeze(0).repeat(idx.numel(), 1)
            Sb = center_t.clone()
            rb = torch.full((idx.numel(),), float(r_cyl), device=device, dtype=dtype)

            res_b = vectorized_circle_fully_occluded_by_sphere(Vb, Cb_flat, rb, Sb, R_cloud)
            oc_b = torch.as_tensor(res_b.occluded, device=device)

            # Top cap transform: shift by Ct.z
            shift_z = Ct[2]
            Vt = V.clone(); Vt[:, 2] = V[:, 2] - shift_z
            Ct_flat = Ct.clone(); Ct_flat = torch.tensor([Ct_flat[0], Ct_flat[1], 0.0], dtype=dtype, device=device)
            Ct_batch = Ct_flat.unsqueeze(0).repeat(idx.numel(), 1)
            St = center_t.clone(); St[:, 2] = St[:, 2] - shift_z
            rt = rb  # same radius
            res_t = vectorized_circle_fully_occluded_by_sphere(Vt, Ct_batch, rt, St, R_cloud)
            oc_t = torch.as_tensor(res_t.occluded, device=device)

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
    best_x: torch.Tensor
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
    device: Optional[torch.device] = None,
    use_vectorized: bool = True,
) -> PSOResult:
    """PSO for Q2 with optional torch-parallel batch evaluation via vectorized_judge.

    When use_vectorized=True and method == 'judge_caps', a custom batched objective
    replaces the slower Python simulation for swarm evaluation. The returned
    best_eval is still computed with the canonical simulator for consistency.
    """
    with torch.no_grad():
        dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        dim = 4
        assert len(bounds) == dim
        lo = torch.tensor([b[0] for b in bounds], dtype=torch.float64, device=dev)
        hi = torch.tensor([b[1] for b in bounds], dtype=torch.float64, device=dev)
        span = hi - lo
        vmax = vmax_frac * span

        # Initialize swarm
        X = lo + torch.rand(pop, dim, device=dev, dtype=torch.float64) * span
        V = (torch.rand(pop, dim, device=dev, dtype=torch.float64) - 0.5) * 2.0 * vmax

        def eval_obj_single(x: Iterable[float]) -> float:
            return float(objective_q2_vector(x, method=method, dt=dt))

        def eval_batch(Xb: torch.Tensor) -> torch.Tensor:
            if use_vectorized and method == "judge_caps":
                occluded_time = _batch_occluded_time(Xb, dt=dt, device=dev)
                # Objective is -occluded_time['M1'] (maximize occlusion => minimize negative)
                return -occluded_time.to(dtype=torch.float64)
            # Fallback: loop slow path
            vals = [eval_obj_single(Xb[i]) for i in range(Xb.shape[0])]
            return torch.tensor(vals, device=dev, dtype=torch.float64)

        # Personal and global bests
        P = X.clone()
        pbest = eval_batch(P)
        g_idx = int(torch.argmin(pbest))
        g = P[g_idx].clone()
        gbest = float(pbest[g_idx])

        if verbose:
            print(f"init gbest={gbest:.6f} x={g.cpu().tolist()}")

        for it in range(iters):
            r1 = torch.rand(pop, dim, device=dev, dtype=torch.float64)
            r2 = torch.rand(pop, dim, device=dev, dtype=torch.float64)
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
            V = torch.maximum(torch.minimum(V, vmax), -vmax)  # clamp
            X = X + V
            # Bounds handling
            below = X < lo
            above = X > hi
            if below.any() or above.any():
                X = torch.maximum(torch.minimum(X, hi), lo)
                V[below] *= -0.5
                V[above] *= -0.5
            # Batch evaluation
            vals = eval_batch(X)
            improved = vals < pbest
            if improved.any():
                pbest[improved] = vals[improved]
                P[improved] = X[improved]
                best_val_iter, idx_iter = torch.min(pbest, dim=0)
                if float(best_val_iter) < gbest - 1e-15:
                    gbest = float(best_val_iter)
                    g = P[int(idx_iter)].clone()
            if verbose and (it % max(1, iters // 10) == 0 or it == iters - 1):
                print(f"iter {it+1}/{iters} gbest={gbest:.6f}")

    # Final authoritative evaluation via full simulator (objective value already minimized form)
    best_eval = evaluate_problem2(
        speed=float(g[0].cpu()),
        azimuth=float(g[1].cpu()),
        release_time=float(g[2].cpu()),
        explode_delay=float(g[3].cpu()),
        occlusion_method=method,
        dt=dt,
    )
    return PSOResult(best_x=g.cpu(), best_value=gbest, best_eval=best_eval)


if __name__ == "__main__":
    # Tiny demo run
    out = solve_q2_pso(pop=12, iters=20, method="sampling", seed=42, verbose=True)
    bx = out.best_x
    print("Best x:", bx)
    print("Best value (minimized):", out.best_value)
    print("M1 遮蔽时长(s):", out.best_eval["occluded_time"]["M1"], "总计(s):", out.best_eval["total"])
