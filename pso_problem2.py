from __future__ import annotations
"""
Particle Swarm Optimization (PSO) for Problem 2
  Optimize FY1 (single UAV, single smoke bomb) to maximize M1 occlusion time.
Decision variables (x = [speed, azimuth, release_time, explode_delay]):
  - speed ∈ [70, 140] m/s
  - azimuth ∈ [0, 2π)
  - release_time ∈ [0, rel_max]
  - explode_delay ∈ [0, delay_max]

Usage:
  python pso_problem2.py
  python pso_problem2.py --use-cli --iters 300 --swarm 50 --method judge_caps --seed 42

Notes:
  - "judge_caps" is fast and conservative; "sampling" is slower but more accurate.
  - You can raise rel_max / delay_max to enlarge the search space.
  - Periodic checkpoint of the best solution is supported.
"""
import math
import time
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np

from optimizer_api import evaluate_problem2

# ===================== User-configurable block =====================
CONFIG = dict(
    # simulation / objective
    method="judge_caps",   # 'judge_caps' or 'sampling'
    dt=0.01,
    rel_max=66.0,
    delay_max=20.0,

    # PSO hyper-parameters
    swarm_size=50,
    iters=300,
    w_start=0.9,            # inertia start
    w_end=0.4,              # inertia end
    c1=1.8,                 # cognitive
    c2=1.8,                 # social
    v_clamp_frac=0.25,      # velocity clamp as fraction of range
    reset_fraction=0.10,    # fraction of particles to randomly reset each iter (>=1 if possible)

    # misc
    seed=42,                # None for system random

    # logging / checkpoint
    progress=True,
    progress_interval=1.0,  # seconds
    verbose=False,
    checkpoint_file="best_p2_pso.json",
    ckpt_every_iter=10,
)
# ==================================================================

SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi

# Provided good initial solution (will be injected as a particle)
INIT_SEED = dict(
    speed=118.33654509070709,
    azimuth=0.1173679648999947,
    release_time=0.13445172037229525,
    explode_delay=0.6956187202253264,
    value=4.589999999999947,
)


@dataclass
class Particle:
    x: np.ndarray   # position [speed, azimuth, release_time, explode_delay]
    v: np.ndarray   # velocity
    f: float        # fitness
    pbest_x: np.ndarray
    pbest_f: float

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
    """Shortest signed angular difference a-b mapped to [-pi, pi)."""
    d = (a - b) % (2.0 * math.pi)
    if d >= math.pi:
        d -= 2.0 * math.pi
    return d


def vec_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    d = d.copy()
    # use shortest angular difference on azimuth dim (idx=1)
    d[1] = ang_diff(a[1], b[1])
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


def _init_swarm(rng: np.random.Generator, swarm_size: int, lo: np.ndarray, hi: np.ndarray,
                method: str, dt: float) -> Tuple[list[Particle], Particle]:
    swarm: list[Particle] = []
    gbest: Particle | None = None

    base = np.array([120.0, math.pi, 2.0, 4.0], dtype=float)
    span = hi - lo
    vspan = span.copy()
    vspan[1] = math.pi  # reasonable angular velocity span

    # Try to inject seed as first particle
    injected = False
    if swarm_size >= 1:
        sx = np.array([
            float(np.clip(INIT_SEED["speed"], SPEED_MIN, SPEED_MAX)),
            wrap_angle(float(INIT_SEED["azimuth"])) ,
            float(np.clip(INIT_SEED["release_time"], 0.0, hi[2])),
            float(np.clip(INIT_SEED["explode_delay"], 0.0, hi[3])),
        ], dtype=float)
        sx = _clip_and_wrap(sx, lo, hi)
        # small random velocity
        sv = (rng.random(4) * 2.0 - 1.0) * 0.05 * vspan
        sv[1] = (rng.random() * 2.0 - 1.0) * (0.05 * vspan[1])
        try:
            sf = _eval_vec(sx, method, dt)
        except Exception:
            sf = float(INIT_SEED.get("value", 0.0))
        sp = Particle(x=sx, v=sv, f=sf, pbest_x=sx.copy(), pbest_f=sf)
        swarm.append(sp)
        gbest = Particle(x=sp.x.copy(), v=sp.v.copy(), f=sp.f, pbest_x=sp.pbest_x.copy(), pbest_f=sp.pbest_f)
        injected = True

    for i in range(1 if injected else 0, swarm_size):
        if rng.random() < 0.3:
            x = base + rng.normal(0.0, 0.12, size=4) * span
        else:
            x = lo + rng.random(4) * span
        x[1] = wrap_angle(x[1])
        x = _clip_and_wrap(x, lo, hi)
        # init v within +/- 10% of range
        v = (rng.random(4) * 2.0 - 1.0) * 0.10 * vspan
        v[1] = (rng.random() * 2.0 - 1.0) * (0.10 * vspan[1])
        f = _eval_vec(x, method, dt)
        p = Particle(x=x, v=v, f=f, pbest_x=x.copy(), pbest_f=f)
        swarm.append(p)
        if gbest is None or p.f > gbest.f:
            gbest = Particle(x=p.x.copy(), v=p.v.copy(), f=p.f, pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

    assert gbest is not None
    return swarm, gbest


def pso(args: SimpleNamespace) -> Particle:
    rng = np.random.default_rng(args.seed)
    lo, hi = _bounds(args.rel_max, args.delay_max)
    span = hi - lo
    base = np.array([120.0, math.pi, 2.0, 4.0], dtype=float)
    vspan = span.copy()
    vspan[1] = math.pi

    swarm, gbest = _init_swarm(rng, args.swarm_size, lo, hi, args.method, args.dt)

    v_max = args.v_clamp_frac * span
    v_max[1] = args.v_clamp_frac * math.pi

    start = time.time()
    last_prog = start

    if args.checkpoint_file:
        _atomic_save_json(gbest.as_dict() | dict(iter=0, timestamp=time.time(),
                                                 azimuth_deg=math.degrees(gbest.x[1]),
                                                 explode_at=float(gbest.x[2] + gbest.x[3])),
                          args.checkpoint_file)

    for it in range(1, args.iters + 1):
        w = args.w_end + (args.w_start - args.w_end) * max(0.0, (args.iters - it) / max(1, args.iters - 1))

        gbest_x = gbest.x.copy()
        for p in swarm:
            r1 = rng.random(4)
            r2 = rng.random(4)
            cognitive = r1 * vec_sub(p.pbest_x, p.x)
            social = r2 * vec_sub(gbest_x, p.x)
            p.v = w * p.v + args.c1 * cognitive + args.c2 * social
            # velocity clamp
            p.v = np.clip(p.v, -v_max, v_max)
            # position update (with angle wrap)
            p.x = vec_add_angle(p.x, p.v)
            p.x = _clip_and_wrap(p.x, lo, hi)

            # evaluate
            p.f = _eval_vec(p.x, args.method, args.dt)
            if p.f >= p.pbest_f:
                p.pbest_x = p.x.copy()
                p.pbest_f = p.f
                if p.f > gbest.f:
                    gbest = Particle(x=p.x.copy(), v=p.v.copy(), f=p.f,
                                     pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

        # Randomly reset a few particles (avoid resetting the current best)
        avail = list(range(len(swarm)))
        if len(avail) > 1:  # there is something other than best to reset
            best_idx = int(np.argmax([p.f for p in swarm]))
            if best_idx in avail:
                avail.remove(best_idx)
            if len(avail) > 0:
                m = max(1, int(args.reset_fraction * args.swarm_size))
                k = min(m, len(avail))
                reset_idxs = list(rng.choice(avail, size=k, replace=False))
                for i in reset_idxs:
                    if rng.random() < 0.3:
                        x_new = base + rng.normal(0.0, 0.12, size=4) * span
                    else:
                        x_new = lo + rng.random(4) * span
                    x_new[1] = wrap_angle(x_new[1])
                    x_new = _clip_and_wrap(x_new, lo, hi)
                    v_new = (rng.random(4) * 2.0 - 1.0) * 0.10 * vspan
                    v_new[1] = (rng.random() * 2.0 - 1.0) * (0.10 * vspan[1])
                    f_new = _eval_vec(x_new, args.method, args.dt)
                    p = swarm[i]
                    p.x = x_new
                    p.v = v_new
                    p.f = f_new
                    p.pbest_x = x_new.copy()
                    p.pbest_f = f_new
                    if f_new > gbest.f:
                        gbest = Particle(x=x_new.copy(), v=v_new.copy(), f=f_new,
                                         pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

        now = time.time()
        if args.progress and (now - last_prog >= args.progress_interval):
            elapsed = now - start
            print(f"[prog] iter={it}/{args.iters} best={gbest.f:.4f}s "
                  f"spd={gbest.x[0]:.2f} az={gbest.x[1]:.3f} rel={gbest.x[2]:.2f} dly={gbest.x[3]:.2f} "
                  f"w={w:.2f} elapsed={elapsed:.1f}s", flush=True)
            last_prog = now

        if args.ckpt_every_iter and (it % args.ckpt_every_iter == 0) and args.checkpoint_file:
            _atomic_save_json(gbest.as_dict() | dict(iter=it, timestamp=time.time(),
                                                     azimuth_deg=math.degrees(gbest.x[1]),
                                                     explode_at=float(gbest.x[2] + gbest.x[3])),
                              args.checkpoint_file)

    if args.checkpoint_file:
        _atomic_save_json(gbest.as_dict() | dict(iter=args.iters, timestamp=time.time(),
                                                 azimuth_deg=math.degrees(gbest.x[1]),
                                                 explode_at=float(gbest.x[2] + gbest.x[3])),
                          args.checkpoint_file)

    return gbest


def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        method=cfg['method'], dt=cfg['dt'], rel_max=cfg['rel_max'], delay_max=cfg['delay_max'],
        swarm_size=cfg['swarm_size'], iters=cfg['iters'], w_start=cfg['w_start'], w_end=cfg['w_end'],
        c1=cfg['c1'], c2=cfg['c2'], v_clamp_frac=cfg['v_clamp_frac'], reset_fraction=cfg['reset_fraction'],
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
        ap = argparse.ArgumentParser(description="PSO for Problem 2 (maximize M1 occlusion)")
        ap.add_argument('--use-cli', action='store_true')
        ap.add_argument('--method', choices=['judge_caps', 'sampling'], default='judge_caps')
        ap.add_argument('--dt', type=float, default=0.01)
        ap.add_argument('--rel-max', type=float, default=66.0)
        ap.add_argument('--delay-max', type=float, default=20.0)
        ap.add_argument('--swarm', dest='swarm_size', type=int, default=50)
        ap.add_argument('--iters', type=int, default=300)
        ap.add_argument('--w-start', type=float, default=0.9)
        ap.add_argument('--w-end', type=float, default=0.4)
        ap.add_argument('--c1', type=float, default=1.8)
        ap.add_argument('--c2', type=float, default=1.8)
        ap.add_argument('--v-clamp-frac', type=float, default=0.25)
        ap.add_argument('--reset-fraction', type=float, default=0.10)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--progress', action='store_true')
        ap.add_argument('--progress-interval', type=float, default=1.0)
        ap.add_argument('--verbose', action='store_true')
        ap.add_argument('--checkpoint-file', type=str, default='best_p2_pso.json')
        ap.add_argument('--ckpt-every-iter', type=int, default=10)
        args = ap.parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t0 = time.time()
    best = pso(args)
    t1 = time.time()

    print("=== PSO Result (Problem 2) ===")
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

    print(f"Runtime: {t1 - t0:.2f} s | Iters: {args.iters} | Swarm: {args.swarm_size}")


if __name__ == "__main__":
    main()
