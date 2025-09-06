from __future__ import annotations
"""
Simulated Annealing (SA) for Problem 2 (Q2).

Decision vector x = [speed, azimuth, release_time, explode_delay]
Objective: minimize objective_q2_vector(x) which equals -occluded_time.

GPU-accelerated batch evaluation: we keep a working population (neighbors per step)
so we can leverage the vectorized occlusion judge (`vectorized_judge`). This makes
SA more efficient on GPU vs single-candidate scalar evaluations.

Strategy
--------
At each temperature T we:
 1. Generate K neighbor candidates around the current solution using Gaussian
    perturbations scaled by an adaptive step size per dimension.
 2. Evaluate all K neighbors in a single batched call (if method == 'judge_caps').
 3. Accept one neighbor according to standard Metropolis criterion.
We also track a global best and perform occasional "restarts" if stuck.

The objective function value = -occluded_time, so lower is better.

Bounds (mirroring `pso_q2`):
 speed: [70, 140]
 azimuth: [-pi, pi]
 release_time: [0, 30]
 explode_delay: [0.2, 12]

Returned best_eval is produced via canonical simulator for consistency.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict
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
    rel_pos = drone_pos0 + drone_dir * speed.unsqueeze(1) * release_time.unsqueeze(1)
    vel = drone_dir * speed.unsqueeze(1)
    dt = explode_delay.unsqueeze(1)
    a = torch.tensor([0.0, 0.0, -_G], dtype=rel_pos.dtype, device=rel_pos.device)
    center0 = rel_pos + vel * dt + 0.5 * a * (dt * dt)
    return center0


def _batch_occluded_time(params: torch.Tensor, *, dt: float, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        N = params.shape[0]
        dtype = torch.float64
        p = params.to(device=device, dtype=dtype)
        speed = p[:, 0]
        az = p[:, 1]
        t_rel = p[:, 2]
        dly = p[:, 3]
        valid = (speed >= 70.0) & (speed <= 140.0) & (t_rel >= 0.0) & (dly > 0.0)
        occluded_time = torch.zeros(N, dtype=dtype, device=device)
        if not valid.any():
            return occluded_time
        drone_pos0 = torch.tensor([17800.0, 0.0, 1800.0], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)
        drone_dir = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], dim=1)
        center0 = _compute_cloud_center0(drone_pos0, drone_dir, speed, t_rel, dly)
        explode_time = t_rel + dly
        cloud_end_time = explode_time + _SMOKE_LIFE
        T_f = float(_MISSILE_T_FLIGHT)
        n_steps = int(T_f / dt) + 1
        t_grid = torch.linspace(0.0, T_f, n_steps, device=device, dtype=dtype)
        missile_pos = _MISSILE_POS0.to(device=device, dtype=dtype).unsqueeze(0) + _MISSILE_DIR.to(device) * (_MISSILE_SPEED * t_grid).unsqueeze(1)
        Cb = _C_BASE.to(device=device, dtype=dtype)
        Ct = Cb + torch.tensor([0.0, 0.0, float(_H_CYL)], dtype=dtype, device=device)
        r_cyl = torch.tensor(float(_R_CYL), dtype=dtype, device=device)
        for ti, t in enumerate(t_grid):
            act_mask = valid & (t >= explode_time) & (t <= cloud_end_time)
            if not act_mask.any():
                continue
            idx = torch.nonzero(act_mask, as_tuple=False).squeeze(1)
            V = missile_pos[ti].unsqueeze(0).repeat(idx.numel(), 1)
            tau = (t - explode_time[idx]).clamp(min=0.0)
            center_t = center0[idx].clone()
            center_t[:, 2] = center0[idx][:, 2] - _SMOKE_DESCENT * tau
            R_cloud = torch.full((idx.numel(),), float(_SMOKE_RADIUS), device=device, dtype=dtype)
            Vb = V.clone()
            Cb_flat = Cb.clone().unsqueeze(0).repeat(idx.numel(), 1)
            Sb = center_t.clone()
            rb = torch.full((idx.numel(),), float(r_cyl), device=device, dtype=dtype)
            res_b = vectorized_circle_fully_occluded_by_sphere(Vb, Cb_flat, rb, Sb, R_cloud)
            oc_b = torch.as_tensor(res_b.occluded, device=device)
            shift_z = Ct[2]
            Vt = V.clone(); Vt[:, 2] = V[:, 2] - shift_z
            Ct_flat = Ct.clone(); Ct_flat = torch.tensor([Ct_flat[0], Ct_flat[1], 0.0], dtype=dtype, device=device)
            Ct_batch = Ct_flat.unsqueeze(0).repeat(idx.numel(), 1)
            St = center_t.clone(); St[:, 2] = St[:, 2] - shift_z
            rt = rb
            res_t = vectorized_circle_fully_occluded_by_sphere(Vt, Ct_batch, rt, St, R_cloud)
            oc_t = torch.as_tensor(res_t.occluded, device=device)
            oc_both = oc_b & oc_t
            if oc_both.any():
                occluded_time[idx[oc_both]] += dt
        return occluded_time

BOUNDS_DEFAULT: Tuple[Tuple[float, float], ...] = (
    (70.0, 140.0),
    (-math.pi, math.pi),
    (0.0, 30.0),
    (0.2, 12.0),
)

@dataclass
class SAResult:
    best_x: torch.Tensor
    best_value: float
    best_eval: Dict


def solve_q2_sa(
    *,
    iters: int = 8000,
    neighbor_batch: int = 128,
    init_temp: float = 2.0,
    final_temp: float = 1e-3,
    cooling: str = "exp",  # 'exp' | 'linear'
    step_scale: float = 0.15,
    method: str = "judge_caps",
    dt: float = 0.02,
    seed: Optional[int] = None,
    bounds: Tuple[Tuple[float, float], ...] = BOUNDS_DEFAULT,
    device: Optional[torch.device] = None,
    use_vectorized: bool = True,
    verbose: bool = False,
    restart_every: Optional[int] = 3000,
) -> SAResult:
    """Simulated annealing with batched neighbor evaluation.

    objective = -occluded_time (minimize). Accept worse with probability exp(-delta / T).
    """
    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    dim = 4
    assert len(bounds) == dim
    lo = torch.tensor([b[0] for b in bounds], dtype=torch.float64, device=dev)
    hi = torch.tensor([b[1] for b in bounds], dtype=torch.float64, device=dev)
    span = hi - lo
    # initial solution uniformly random
    cur_x = lo + torch.rand(dim, device=dev, dtype=torch.float64) * span

    def eval_single(x: Iterable[float]) -> float:
        return float(objective_q2_vector(x, method=method, dt=dt))

    def eval_batch(Xb: torch.Tensor) -> torch.Tensor:
        if use_vectorized and method == "judge_caps":
            occluded_time = _batch_occluded_time(Xb, dt=dt, device=dev)
            return -occluded_time.to(dtype=torch.float64)
        vals = [eval_single(Xb[i]) for i in range(Xb.shape[0])]
        return torch.tensor(vals, device=dev, dtype=torch.float64)

    cur_val = eval_batch(cur_x.unsqueeze(0))[0]
    best_x = cur_x.clone()
    best_val = float(cur_val)

    # adaptive step std per dimension (relative to span)
    base_sigma = step_scale * span

    def temperature(k: int) -> float:
        if cooling == 'exp':
            return init_temp * (final_temp / init_temp) ** (k / max(1, iters - 1))
        else:  # linear
            return init_temp + (final_temp - init_temp) * (k / max(1, iters - 1))

    accept_count = 0
    for k in range(iters):
        T = temperature(k)
        # generate neighbor batch around current x
        noise = torch.randn(neighbor_batch, dim, device=dev, dtype=torch.float64) * base_sigma
        candidates = cur_x.unsqueeze(0) + noise
        # reflect / clamp to bounds
        candidates = torch.maximum(torch.minimum(candidates, hi), lo)
        vals = eval_batch(candidates)
        # Choose one candidate via Metropolis among batch:
        # compute acceptance probabilities
        delta = vals - cur_val  # (B,)
        # Always accept strictly better ones if any; else probabilistic
        better_mask = delta < 0.0
        chosen_idx = None
        if better_mask.any():
            # pick best of better
            bvals = vals[better_mask]
            rel_idx = torch.argmin(bvals)
            chosen_idx = torch.nonzero(better_mask, as_tuple=False).squeeze(1)[rel_idx]
        else:
            # compute probabilities p_i = exp(-delta_i / T)
            probs = torch.exp(-delta / max(T, 1e-12))
            probs = probs / probs.sum()
            # sample
            chosen_idx = torch.multinomial(probs, 1)[0]
        new_x = candidates[chosen_idx]
        new_val = vals[chosen_idx]
        # Metropolis acceptance (redundant for better case but keeps logic uniform)
        d = new_val - cur_val
        if d < 0 or random.random() < math.exp(-float(d) / max(T, 1e-12)):
            cur_x = new_x
            cur_val = new_val
            accept_count += 1
            if float(cur_val) < best_val - 1e-15:
                best_val = float(cur_val)
                best_x = cur_x.clone()
        # Optional restart if stagnating
        if restart_every and (k > 0) and (k % restart_every == 0):
            # reinitialize near global best (small noise)
            cur_x = best_x + 0.05 * span * torch.randn(dim, device=dev, dtype=torch.float64)
            cur_x = torch.maximum(torch.minimum(cur_x, hi), lo)
            cur_val = eval_batch(cur_x.unsqueeze(0))[0]
        if verbose and (k % max(1, iters // 20) == 0 or k == iters - 1):
            acc_rate = accept_count / (k + 1)
            print(f"iter {k+1}/{iters} T={T:.4f} cur={float(cur_val):.6f} best={best_val:.6f} acc_rate={acc_rate:.3f}")

    best_eval = evaluate_problem2(
        speed=float(best_x[0].cpu()),
        azimuth=float(best_x[1].cpu()),
        release_time=float(best_x[2].cpu()),
        explode_delay=float(best_x[3].cpu()),
        occlusion_method=method,
        dt=dt,
    )
    return SAResult(best_x=best_x.cpu(), best_value=best_val, best_eval=best_eval)


if __name__ == "__main__":
    # 使用 judge_caps + 向量化评估以充分利用 GPU
    res = solve_q2_sa(iters=1000, neighbor_batch=64, seed=42, verbose=True, method="judge_caps")
    print("Best x:", res.best_x)
    print("Best value (minimized):", res.best_value)
    print("M1 遮蔽时长(s):", res.best_eval["occluded_time"]["M1"], "总计(s):", res.best_eval["total"])
