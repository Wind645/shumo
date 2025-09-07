from __future__ import annotations
"""
CMA-ES (separable) for Problem 2
  Optimize FY1 (single UAV, single smoke bomb) to maximize M1 occlusion time.
Decision variables (x = [speed, azimuth, release_time, explode_delay]):
  - speed ∈ [70, 140] m/s
  - azimuth ∈ [0, 2π)
  - release_time ∈ [0, rel_max]
  - explode_delay ∈ [0, delay_max]

Strategy:
  - Use separable CMA-ES (diagonal covariance) with angle-aware ops
  - Domain contraction: multi-stage shrinking around current best on {speed, release, delay}
  - Local restarts: if stagnation detected, re-sample covariance and slightly perturb mean

Usage:
  python cma_problem2.py
  python cma_problem2.py --use-cli --iters-per-stage 200 --stages 2 --method judge_caps --seed 42

Notes:
  - "judge_caps" is fast and conservative; "sampling" is slower but more accurate.
  - You can raise rel_max / delay_max to enlarge the search space.
  - Periodic checkpoint of the best solution is supported.
"""
import json
import math
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Tuple, List

import numpy as np

from optimizer_api import evaluate_problem2

# ===================== User-configurable block =====================
CONFIG = dict(
    # simulation / objective
    method="judge_caps",   # 'judge_caps' or 'sampling'
    dt=0.01,
    rel_max=66.0,
    delay_max=20.0,

    # CMA-ES hyper-parameters
    pop_size=None,           # None -> auto: 4 + floor(3*ln(n))
    iters_per_stage=200,
    stages=2,                # number of contraction stages (>=1)
    contraction_fracs=[1.0, 0.5, 0.25],  # applied to non-angle spans per stage (will be truncated to stages)
    init_sigma_frac=0.30,    # initial sigma as fraction of span (per-dim)

    # stagnation / restart
    stall_window=40,         # iterations without significant improvement to trigger restart
    stall_eps=1e-3,          # improvement threshold in seconds
    restart_noise_frac=0.05, # perturb mean by this fraction of current span at restart

    # misc
    seed=42,                 # None for system random

    # logging / checkpoint
    progress=True,
    progress_interval=1.0,   # seconds
    verbose=False,
    checkpoint_file="best_p2_cma.json",
    ckpt_every_iter=10,
)
# ==================================================================

SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi

# Provided good initial solution (optional warm start of mean)
INIT_SEED = dict(
    speed=118.33654509070709,
    azimuth=0.1173679648999947,
    release_time=0.13445172037229525,
    explode_delay=0.6956187202253264,
    value=4.589999999999947,
)


@dataclass
class Candidate:
    x: np.ndarray
    f: float

    def as_dict(self) -> Dict:
        return dict(
            speed=float(self.x[0]),
            azimuth=float(self.x[1]),
            release_time=float(self.x[2]),
            explode_delay=float(self.x[3]),
            value=float(self.f),
        )


def _atomic_save_json(obj: Dict, path: str):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def wrap_angle(a: float) -> float:
    twopi = 2.0 * math.pi
    return a % twopi


def ang_diff(a: float, b: float) -> float:
    d = (a - b) % (2.0 * math.pi)
    if d >= math.pi:
        d -= 2.0 * math.pi
    return d


def vec_add_angle(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    y = x + d
    y = y.copy()
    y[1] = wrap_angle(y[1])
    return y


def _bounds(rel_max: float, delay_max: float) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array([SPEED_MIN, AZIM_MIN, 0.0, 0.0], dtype=float)
    hi = np.array([SPEED_MAX, AZIM_MAX, rel_max, delay_max], dtype=float)
    return lo, hi


def _clip_and_wrap(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    y = np.minimum(np.maximum(x, lo), hi)
    y = y.copy()
    y[1] = wrap_angle(y[1])
    return y


def _eval_vec(x: np.ndarray, method: str, dt: float) -> float:
    res = evaluate_problem2(
        speed=float(x[0]),
        azimuth=float(x[1]),
        release_time=float(x[2]),
        explode_delay=float(x[3]),
        occlusion_method=method,
        dt=dt,
    )
    return float(res["occluded_time"]["M1"])  # maximize


def _weighted_circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return math.atan2(s, c) % (2.0 * math.pi)


class SepCMA:
    def __init__(self, rng: np.random.Generator, m0: np.ndarray, span: np.ndarray,
                 lo: np.ndarray, hi: np.ndarray, *,
                 pop_size: int | None, init_sigma_frac: float):
        self.rng = rng
        self.n = len(m0)
        self.lo = lo.copy()
        self.hi = hi.copy()
        self.span = span.copy()

        # mean
        self.m = m0.copy()

        # diag covariance and step-size
        self.diagC = (np.maximum(1e-12, (init_sigma_frac * span) ** 2)).astype(float)
        self.sigma = 1.0  # we absorb scale into diagC; keep sigma as global scalar

        # strategy parameters
        n = self.n
        lam = pop_size if pop_size is not None else int(4 + math.floor(3 * math.log(n)))
        lam = max(lam, 4)
        self.lam = lam
        self.mu = lam // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = (weights / np.sum(weights)).astype(float)
        self.mueff = float(1.0 / np.sum(self.weights ** 2))

        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)

        # expectation of ||N(0,I)||
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * (n ** 2)))

    def _sample(self) -> np.ndarray:
        # Sample z ~ N(0, I), then y = sigma * sqrt(diagC) * z, x = m (+ angle wrap), clip to bounds
        z = self.rng.standard_normal(self.n)
        step = self.sigma * np.sqrt(self.diagC) * z
        x = vec_add_angle(self.m, step)
        x = _clip_and_wrap(x, self.lo, self.hi)
        return x

    def _diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # angle-aware difference (a - b)
        d = a - b
        d = d.copy()
        d[1] = ang_diff(a[1], b[1])
        return d

    def ask(self, k: int | None = None) -> List[np.ndarray]:
        k = self.lam if k is None else k
        return [self._sample() for _ in range(k)]

    def tell(self, X: List[np.ndarray], fvals: List[float]):
        # select top mu by maximizing f
        idx = np.argsort(-np.asarray(fvals))
        elites = [X[i] for i in idx[: self.mu]]
        elites_arr = np.stack(elites, axis=0)

        # weighted mean with circular dim for angle
        m_new = self.m.copy()
        m_new[0] = float(np.sum(self.weights * elites_arr[:, 0]))
        m_new[1] = _weighted_circular_mean(elites_arr[:, 1], self.weights)
        m_new[2] = float(np.sum(self.weights * elites_arr[:, 2]))
        m_new[3] = float(np.sum(self.weights * elites_arr[:, 3]))

        # y_w: weighted step in parameter space from old mean
        diffs = np.array([self._diff(e, self.m) for e in elites])  # shape (mu, n)
        y_w = np.sum(diffs * self.weights[:, None], axis=0)

        # evolution paths
        inv_sqrt_diag = 1.0 / (np.sqrt(self.diagC) + 1e-12)
        z_w = y_w * inv_sqrt_diag / max(1e-12, self.sigma)

        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_w
        norm_ps = float(np.linalg.norm(self.ps))
        hsig = 1.0 if norm_ps / math.sqrt(1 - (1 - self.cs) ** (2)) < (1.4 + 2 / (self.n + 1)) * self.chiN else 0.0

        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

        # covariance (diagonal)
        # rank-one and rank-mu updates
        y_sq = np.sum((diffs ** 2) * self.weights[:, None], axis=0)
        self.diagC = (1 - self.c1 - self.cmu) * self.diagC + self.c1 * (self.pc ** 2) + self.cmu * y_sq
        self.diagC = np.maximum(self.diagC, 1e-18)

        # step-size
        self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))

        # set new mean (angle wrapped and clipped)
        self.m = _clip_and_wrap(m_new, self.lo, self.hi)

    def set_bounds(self, lo: np.ndarray, hi: np.ndarray):
        # adjust bounds (e.g., contraction)
        self.lo = lo.copy()
        self.hi = hi.copy()
        self.span = hi - lo
        # keep mean inside
        self.m = _clip_and_wrap(self.m, self.lo, self.hi)

    def restart(self, center: np.ndarray, noise_frac: float = 0.05):
        # reset paths
        self.pc[:] = 0.0
        self.ps[:] = 0.0
        # reinit diagC (scaled by current span)
        self.diagC = (np.maximum(1e-12, (0.25 * self.span) ** 2)).astype(float)
        self.sigma = 1.0
        # mean to center with small noise (angle-aware)
        noise = (self.rng.standard_normal(self.n) * (noise_frac * self.span))
        noise[1] = self.rng.standard_normal() * (noise_frac * math.pi)
        self.m = _clip_and_wrap(vec_add_angle(center, noise), self.lo, self.hi)


def _init_mean(lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # try seed
    sx = np.array([
        float(np.clip(INIT_SEED["speed"], SPEED_MIN, SPEED_MAX)),
        wrap_angle(float(INIT_SEED["azimuth"])),
        float(np.clip(INIT_SEED["release_time"], 0.0, hi[2])),
        float(np.clip(INIT_SEED["explode_delay"], 0.0, hi[3])),
    ], dtype=float)
    if np.all(np.isfinite(sx)):
        return _clip_and_wrap(sx, lo, hi)
    # fallback to middle
    return _clip_and_wrap((lo + hi) * 0.5, lo, hi)


def _contract_bounds_around_best(lo0: np.ndarray, hi0: np.ndarray, best_x: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    frac = float(max(1e-6, min(1.0, frac)))
    span0 = hi0 - lo0
    # keep azimuth full range to avoid wrap complications
    lo = lo0.copy()
    hi = hi0.copy()
    # speed, release, delay indices: 0,2,3
    for j in (0, 2, 3):
        width = span0[j] * frac
        center = float(best_x[j])
        lo[j] = max(lo0[j], center - 0.5 * width)
        hi[j] = min(hi0[j], center + 0.5 * width)
        if hi[j] - lo[j] < 1e-6:
            lo[j] = max(lo0[j], center - 1e-3)
            hi[j] = min(hi0[j], center + 1e-3)
    return lo, hi


def cma_solve(args: SimpleNamespace) -> Candidate:
    rng = np.random.default_rng(args.seed)
    lo0, hi0 = _bounds(args.rel_max, args.delay_max)
    span0 = hi0 - lo0

    # derive stage contraction factors length = stages
    cf = list(CONFIG["contraction_fracs"]) if not hasattr(args, 'contraction_fracs') else list(args.contraction_fracs)
    if len(cf) < args.stages:
        while len(cf) < args.stages:
            cf.append(cf[-1] if cf else 1.0)
    cf = cf[: args.stages]

    # initial mean
    m0 = _init_mean(lo0, hi0, rng)

    # init optimizer on stage 0 bounds
    opt = SepCMA(rng, m0, span0, lo0, hi0, pop_size=args.pop_size, init_sigma_frac=args.init_sigma_frac)

    best = Candidate(x=m0.copy(), f=_eval_vec(m0, args.method, args.dt))

    start = time.time()
    last_prog = start

    # checkpoint initial
    if args.checkpoint_file:
        _atomic_save_json(best.as_dict() | dict(iter=0, stage=0, timestamp=time.time(),
                                                azimuth_deg=math.degrees(best.x[1]),
                                                explode_at=float(best.x[2] + best.x[3])),
                          args.checkpoint_file)

    # multi-stage with contraction
    for stage in range(args.stages):
        # update bounds by contraction around current best
        if cf[stage] < 1.0 or stage > 0:
            lo_s, hi_s = _contract_bounds_around_best(lo0, hi0, best.x, cf[stage])
            opt.set_bounds(lo_s, hi_s)
        # reset sigma scale to span of current stage
        opt.restart(center=best.x, noise_frac=0.01)

        stall_counter = 0
        last_best = best.f

        total_iters = int(args.iters_per_stage)
        for it in range(1, total_iters + 1):
            X = opt.ask()
            fvals = [
                _eval_vec(_clip_and_wrap(x, opt.lo, opt.hi), args.method, args.dt)
                for x in X
            ]
            # update best
            for x, f in zip(X, fvals):
                if f > best.f:
                    best = Candidate(x=x.copy(), f=f)
            # stagnation tracking
            if best.f > last_best + args.stall_eps:
                last_best = best.f
                stall_counter = 0
            else:
                stall_counter += 1

            # CMA update
            opt.tell(X, fvals)

            # progress
            now = time.time()
            if args.progress and (now - last_prog >= args.progress_interval):
                elapsed = now - start
                print(f"[prog] stage={stage+1}/{args.stages} iter={it}/{total_iters} best={best.f:.4f}s "
                      f"spd={best.x[0]:.2f} az={best.x[1]:.3f} rel={best.x[2]:.2f} dly={best.x[3]:.2f} "
                      f"elapsed={elapsed:.1f}s", flush=True)
                last_prog = now

            # checkpoint
            glob_iter = stage * args.iters_per_stage + it
            if args.ckpt_every_iter and (glob_iter % args.ckpt_every_iter == 0) and args.checkpoint_file:
                _atomic_save_json(best.as_dict() | dict(iter=int(glob_iter), stage=int(stage), timestamp=time.time(),
                                                        azimuth_deg=math.degrees(best.x[1]),
                                                        explode_at=float(best.x[2] + best.x[3])),
                                  args.checkpoint_file)

            # restart on stagnation
            if stall_counter >= args.stall_window:
                opt.restart(center=best.x, noise_frac=args.restart_noise_frac)
                stall_counter = 0

    # final checkpoint
    if args.checkpoint_file:
        _atomic_save_json(best.as_dict() | dict(iter=int(args.stages * args.iters_per_stage), stage=int(args.stages-1),
                                                timestamp=time.time(),
                                                azimuth_deg=math.degrees(best.x[1]),
                                                explode_at=float(best.x[2] + best.x[3])),
                          args.checkpoint_file)

    return best


def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        method=cfg['method'], dt=cfg['dt'], rel_max=cfg['rel_max'], delay_max=cfg['delay_max'],
        pop_size=cfg['pop_size'], iters_per_stage=cfg['iters_per_stage'], stages=cfg['stages'],
        contraction_fracs=cfg['contraction_fracs'], init_sigma_frac=cfg['init_sigma_frac'],
        stall_window=cfg['stall_window'], stall_eps=cfg['stall_eps'], restart_noise_frac=cfg['restart_noise_frac'],
        seed=cfg['seed'], progress=cfg['progress'], progress_interval=cfg['progress_interval'],
        verbose=cfg['verbose'], checkpoint_file=cfg['checkpoint_file'], ckpt_every_iter=cfg['ckpt_every_iter']
    )


def main():
    import sys
    use_cli = '--use-cli' in sys.argv
    if not use_cli:
        args = _build_args_via_config()
    else:
        import argparse
        ap = argparse.ArgumentParser(description="CMA-ES for Problem 2 (maximize M1 occlusion)")
        ap.add_argument('--use-cli', action='store_true')
        ap.add_argument('--method', choices=['judge_caps', 'sampling'], default='judge_caps')
        ap.add_argument('--dt', type=float, default=0.01)
        ap.add_argument('--rel-max', type=float, default=66.0)
        ap.add_argument('--delay-max', type=float, default=20.0)
        ap.add_argument('--pop-size', type=int, default=None)
        ap.add_argument('--iters-per-stage', type=int, default=200)
        ap.add_argument('--stages', type=int, default=2)
        ap.add_argument('--contraction-fracs', type=float, nargs='*', default=None,
                        help='Stage-wise contraction fractions for non-angle dims (e.g., 1.0 0.5 0.25)')
        ap.add_argument('--init-sigma-frac', type=float, default=0.30)
        ap.add_argument('--stall-window', type=int, default=40)
        ap.add_argument('--stall-eps', type=float, default=1e-3)
        ap.add_argument('--restart-noise-frac', type=float, default=0.05)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--progress', action='store_true')
        ap.add_argument('--progress-interval', type=float, default=1.0)
        ap.add_argument('--verbose', action='store_true')
        ap.add_argument('--checkpoint-file', type=str, default='best_p2_cma.json')
        ap.add_argument('--ckpt-every-iter', type=int, default=10)
        args = ap.parse_args()
        if args.contraction_fracs is None:
            args.contraction_fracs = CONFIG['contraction_fracs']

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t0 = time.time()
    best = cma_solve(args)
    t1 = time.time()

    print("=== CMA-ES Result (Problem 2) ===")
    print(f"Best occluded time: {best.f:.4f} s")
    print(f"Speed: {best.x[0]:.3f} m/s")
    print(f"Azimuth: {best.x[1]:.6f} rad  (deg={math.degrees(best.x[1]):.2f})")
    print(f"Release time: {best.x[2]:.3f} s")
    print(f"Explode delay: {best.x[3]:.3f} s (explode @ {best.x[2] + best.x[3]:.3f} s)")

    if args.method == 'judge_caps':
        try:
            res_sampling = evaluate_problem2(
                speed=float(best.x[0]), azimuth=float(best.x[1]),
                release_time=float(best.x[2]), explode_delay=float(best.x[3]),
                occlusion_method='sampling', dt=args.dt
            )
            v2 = float(res_sampling['occluded_time']['M1'])
            print(f"(Sampling verification) Occluded time: {v2:.4f} s")
        except Exception as e:
            print(f"Sampling verification failed: {e}")

    print(f"Runtime: {t1 - t0:.2f} s | Iters/Stage: {args.iters_per_stage} | Stages: {args.stages} | Pop: {args.pop_size or (4 + int(3*math.log(4)))}")


if __name__ == "__main__":
    main()
