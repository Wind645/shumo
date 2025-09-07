from __future__ import annotations
"""
Particle Swarm Optimization (PSO) for Problem 3
  Optimize FY1 (single UAV, three smoke bombs) to maximize M1 occlusion time on M1.

Decision vector x (len=8):
  [speed, azimuth, t1, t2, t3, d1, d2, d3]
    - speed ∈ [70, 140] m/s
    - azimuth ∈ [0, 2π)
    - t1,t2,t3 ∈ [0, rel_max] with t2 >= t1 + 1, t3 >= t2 + 1
    - d1,d2,d3 ∈ [0, delay_max]

Usage:
  python pso_problem3.py
  python pso_problem3.py --use-cli --iters 250 --swarm 60 --method judge_caps --seed 42

Notes:
  - judge_caps is faster but conservative; sampling is slower but more accurate.
  - Results exported to result1.xlsx (install pandas/openpyxl if missing).
"""
import math
import os
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np

from optimizer_api import evaluate_problem3

# ===================== User-configurable block =====================
CONFIG = dict(
    # simulation / objective
    method="judge_caps",   # 'judge_caps' or 'sampling'
    dt=0.05,
    rel_max=66.0,
    delay_max=20.0,
    min_gap=1.0,

    # PSO hyper-parameters
    swarm_size=50,
    iters=300,
    w_start=0.85,
    w_end=0.45,
    c1=1.7,
    c2=1.7,
    v_clamp_frac=0.25,   # velocity clamp as fraction of range per dim
    reset_fraction=0.10, # fraction of particles to randomly reset per iter

    # misc
    seed=42,

    # logging / checkpoint
    progress=True,
    progress_interval=1.0,  # seconds
    verbose=False,
    checkpoint_file="best_p3_pso.json",
    ckpt_every_iter=20,
)
# ==================================================================

SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi
DELAY_MIN = 0.0


# ---------------- Utilities ----------------

def _atomic_save_json(obj: Dict, path: str):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def wrap_angle(a: float) -> float:
    return a % (2.0 * math.pi)


def ang_diff(a: float, b: float) -> float:
    """Shortest signed angular difference a-b mapped to [-pi, pi)."""
    d = (a - b) % (2.0 * math.pi)
    if d >= math.pi:
        d -= 2.0 * math.pi
    return d


def clip(x, lo, hi):
    return min(max(x, lo), hi)


def project_times(times: List[float], rel_max: float, min_gap: float) -> List[float]:
    """Project 3 release times to [0, rel_max] with minimum spacing min_gap, sorted."""
    n = len(times)
    t = sorted(float(x) for x in times)
    for i in range(1, n):
        t[i] = max(t[i], t[i - 1] + min_gap)
    if t[-1] > rel_max:
        t[-1] = rel_max
        for i in range(n - 2, -1, -1):
            t[i] = min(t[i], t[i + 1] - min_gap)
        if t[0] < 0.0:
            shift = -t[0]
            t = [ti + shift for ti in t]
            for i in range(1, n):
                t[i] = max(t[i], t[i - 1] + min_gap)
    t[0] = clip(t[0], 0.0, rel_max)
    for i in range(1, n):
        t[i] = clip(max(t[i], t[i - 1] + min_gap), 0.0, rel_max)
    if t[-1] > rel_max + 1e-9:
        # fallback: evenly spaced inside bounds
        base_hi = max(0.0, rel_max - (n - 1) * min_gap)
        base = min(max(t[0], 0.0), base_hi)
        t = [base + i * min_gap for i in range(n)]
    return t


def decode_position(x: np.ndarray, cfg: SimpleNamespace) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Map raw vector to feasible decision variables."""
    speed = clip(float(x[0]), SPEED_MIN, SPEED_MAX)
    az = wrap_angle(float(x[1]))
    t_raw = [float(x[2]), float(x[3]), float(x[4])]
    d_raw = [float(x[5]), float(x[6]), float(x[7])]
    times = project_times([clip(t, 0.0, cfg.rel_max) for t in t_raw], cfg.rel_max, cfg.min_gap)
    delays = [clip(d, DELAY_MIN, cfg.delay_max) for d in d_raw]
    bombs = list(zip(times, delays))
    return speed, az, bombs


def encode_feasible(speed: float, az: float, bombs: List[Tuple[float, float]]) -> np.ndarray:
    """Pack feasible decision variables back into vector form (8-d)."""
    t1, t2, t3 = bombs[0][0], bombs[1][0], bombs[2][0]
    d1, d2, d3 = bombs[0][1], bombs[1][1], bombs[2][1]
    return np.array([speed, az, t1, t2, t3, d1, d2, d3], dtype=float)


def evaluate_vec(x: np.ndarray, cfg: SimpleNamespace) -> float:
    speed, az, bombs = decode_position(x, cfg)
    res = evaluate_problem3(bombs=bombs, speed=speed, azimuth=az, dt=cfg.dt, occlusion_method=cfg.method)
    return float(res["occluded_time"]["M1"])  # maximize


def random_vec(cfg: SimpleNamespace, rng: np.random.Generator) -> np.ndarray:
    speed = rng.uniform(SPEED_MIN, SPEED_MAX)
    az = rng.uniform(0.0, 2.0 * math.pi)
    t1 = rng.uniform(0.0, min(5.0, cfg.rel_max))
    t2 = t1 + cfg.min_gap + rng.uniform(0.0, 2.0)
    t3 = t2 + cfg.min_gap + rng.uniform(0.0, 2.0)
    times = project_times([t1, t2, t3], cfg.rel_max, cfg.min_gap)
    d1 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    d2 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    d3 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    return np.array([speed, az, times[0], times[1], times[2], d1, d2, d3], dtype=float)


# ---------------- PSO core ----------------

@dataclass
class Particle:
    x: np.ndarray   # position (8,)
    v: np.ndarray   # velocity (8,)
    f: float        # fitness
    pbest_x: np.ndarray
    pbest_f: float

    def as_dict(self, val_only: bool = False) -> Dict:
        d = dict(
            speed=float(self.x[0]), azimuth=float(self.x[1]),
            t1=float(self.x[2]), t2=float(self.x[3]), t3=float(self.x[4]),
            d1=float(self.x[5]), d2=float(self.x[6]), d3=float(self.x[7]),
            value=float(self.f),
        )
        if val_only:
            return dict(value=float(self.f))
        return d


def _bounds(cfg: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array([SPEED_MIN, AZIM_MIN, 0.0, 0.0, 0.0, DELAY_MIN, DELAY_MIN, DELAY_MIN], dtype=float)
    hi = np.array([SPEED_MAX, AZIM_MAX, cfg.rel_max, cfg.rel_max, cfg.rel_max,
                   cfg.delay_max, cfg.delay_max, cfg.delay_max], dtype=float)
    return lo, hi


def _clip_and_wrap_raw(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    y = np.minimum(np.maximum(x, lo), hi)
    y[1] = wrap_angle(y[1])
    return y


def vec_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    d = d.copy()
    d[1] = ang_diff(a[1], b[1])
    return d


def vec_add_angle(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    y = x + d
    y = y.copy()
    y[1] = wrap_angle(y[1])
    return y


def _init_swarm(cfg: SimpleNamespace, rng: np.random.Generator) -> Tuple[List[Particle], Particle]:
    lo, hi = _bounds(cfg)
    span = hi - lo
    vspan = span.copy()
    vspan[1] = math.pi

    swarm: List[Particle] = []
    gbest: Particle | None = None

    # base around reasonable heuristic
    base = np.array([120.0, math.pi, 2.0, 3.5, 5.0, 4.0, 4.0, 4.0], dtype=float)

    for _ in range(cfg.swarm_size):
        if rng.random() < 0.35:
            x = base + rng.normal(0.0, 0.12, size=8) * span
        else:
            x = lo + rng.random(8) * span
        x[1] = wrap_angle(x[1])
        x = _clip_and_wrap_raw(x, lo, hi)
        # project to feasible manifold via decode/encode
        spd, az, bombs = decode_position(x, cfg)
        x = encode_feasible(spd, az, bombs)
        v = (rng.random(8) * 2.0 - 1.0) * 0.10 * vspan
        v[1] = (rng.random() * 2.0 - 1.0) * (0.10 * vspan[1])
        f = evaluate_vec(x, cfg)
        p = Particle(x=x, v=v, f=f, pbest_x=x.copy(), pbest_f=f)
        swarm.append(p)
        if gbest is None or p.f > gbest.f:
            gbest = Particle(x=p.x.copy(), v=p.v.copy(), f=p.f, pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

    assert gbest is not None
    return swarm, gbest


def pso(cfg: SimpleNamespace) -> Particle:
    rng = np.random.default_rng(cfg.seed)
    lo, hi = _bounds(cfg)
    span = hi - lo

    swarm, gbest = _init_swarm(cfg, rng)

    # velocity clamp per-dimension
    v_max = cfg.v_clamp_frac * span
    v_max[1] = cfg.v_clamp_frac * math.pi

    start = time.time()
    last_prog = start

    # initial checkpoint
    if cfg.checkpoint_file:
        _atomic_save_json(dict(iter=0, timestamp=time.time(), azimuth_deg=math.degrees(gbest.x[1])) | gbest.as_dict(), cfg.checkpoint_file)

    for it in range(1, cfg.iters + 1):
        w = cfg.w_end + (cfg.w_start - cfg.w_end) * max(0.0, (cfg.iters - it) / max(1, cfg.iters - 1))
        gbest_x = gbest.x.copy()

        for p in swarm:
            r1 = rng.random(8)
            r2 = rng.random(8)
            cognitive = r1 * vec_sub(p.pbest_x, p.x)
            social = r2 * vec_sub(gbest_x, p.x)
            p.v = w * p.v + cfg.c1 * cognitive + cfg.c2 * social
            # clamp
            p.v = np.clip(p.v, -v_max, v_max)
            # update
            p.x = vec_add_angle(p.x, p.v)
            # raw clip then project via decode
            p.x = _clip_and_wrap_raw(p.x, lo, hi)
            spd, az, bombs = decode_position(p.x, cfg)
            p.x = encode_feasible(spd, az, bombs)

            # evaluate
            p.f = evaluate_vec(p.x, cfg)
            if p.f >= p.pbest_f:
                p.pbest_x = p.x.copy()
                p.pbest_f = p.f
                if p.f > gbest.f:
                    gbest = Particle(x=p.x.copy(), v=p.v.copy(), f=p.f, pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

        # random resets (avoid resetting the current best)
        avail = list(range(len(swarm)))
        if len(avail) > 1:
            best_idx = int(np.argmax([p.f for p in swarm]))
            if best_idx in avail:
                avail.remove(best_idx)
            if len(avail) > 0:
                m = max(1, int(cfg.reset_fraction * cfg.swarm_size))
                k = min(m, len(avail))
                reset_idxs = list(rng.choice(avail, size=k, replace=False))
                for i in reset_idxs:
                    x_new = random_vec(cfg, rng)
                    # small random velocity
                    vspan = span.copy(); vspan[1] = math.pi
                    v_new = (rng.random(8) * 2.0 - 1.0) * 0.10 * vspan
                    v_new[1] = (rng.random() * 2.0 - 1.0) * (0.10 * vspan[1])
                    f_new = evaluate_vec(x_new, cfg)
                    p = swarm[i]
                    p.x, p.v, p.f = x_new, v_new, f_new
                    p.pbest_x, p.pbest_f = x_new.copy(), f_new
                    if f_new > gbest.f:
                        gbest = Particle(x=x_new.copy(), v=v_new.copy(), f=f_new,
                                         pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

        # progress
        now = time.time()
        if cfg.progress and (now - last_prog >= cfg.progress_interval):
            elapsed = now - start
            spd, az, bombs = decode_position(gbest.x, cfg)
            print(f"[prog] iter={it}/{cfg.iters} best={gbest.f:.4f}s spd={spd:.2f} az={az:.3f} "
                  f"t=({bombs[0][0]:.2f},{bombs[1][0]:.2f},{bombs[2][0]:.2f}) "
                  f"d=({bombs[0][1]:.2f},{bombs[1][1]:.2f},{bombs[2][1]:.2f}) w={w:.2f} elapsed={elapsed:.1f}s",
                  flush=True)
            last_prog = now

        # checkpoint
        if cfg.ckpt_every_iter and (it % cfg.ckpt_every_iter == 0) and cfg.checkpoint_file:
            _atomic_save_json(dict(iter=it, timestamp=time.time(), azimuth_deg=math.degrees(gbest.x[1])) | gbest.as_dict(), cfg.checkpoint_file)

    # final checkpoint
    if cfg.checkpoint_file:
        _atomic_save_json(dict(iter=cfg.iters, timestamp=time.time(), azimuth_deg=math.degrees(gbest.x[1])) | gbest.as_dict(), cfg.checkpoint_file)

    return gbest


# ---------------- Export helpers ----------------

def export_excel(x: np.ndarray, val: float, path: str):
    try:
        import pandas as pd
    except Exception as e:
        print(f"[warn] pandas not available, cannot write {path}. Please: pip install pandas openpyxl. Error: {e}")
        return
    speed = float(x[0])
    az = float(x[1])
    t1, t2, t3 = float(x[2]), float(x[3]), float(x[4])
    d1, d2, d3 = float(x[5]), float(x[6]), float(x[7])
    rows = [{
        "speed(m/s)": speed,
        "azimuth(deg)": math.degrees(az),
        "bomb1_deploy(s)": t1,
        "bomb1_explode(s)": t1 + d1,
        "bomb2_deploy(s)": t2,
        "bomb2_explode(s)": t2 + d2,
        "bomb3_deploy(s)": t3,
        "bomb3_explode(s)": t3 + d3,
        "total_occluded_time(s)": float(val),
    }]
    df = pd.DataFrame(rows)
    try:
        df.to_excel(path, index=False)
        print(f"Saved result to {path}")
    except Exception as e:
        print(f"[warn] failed to save {path}: {e}")


# ---------------- Driver ----------------

def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        method=cfg['method'], dt=cfg['dt'], rel_max=cfg['rel_max'], delay_max=cfg['delay_max'], min_gap=cfg['min_gap'],
        swarm_size=cfg['swarm_size'], iters=cfg['iters'], w_start=cfg['w_start'], w_end=cfg['w_end'],
        c1=cfg['c1'], c2=cfg['c2'], v_clamp_frac=cfg['v_clamp_frac'], reset_fraction=cfg['reset_fraction'],
        seed=cfg['seed'], progress=cfg['progress'], progress_interval=cfg['progress_interval'],
        verbose=cfg['verbose'], checkpoint_file=cfg['checkpoint_file'], ckpt_every_iter=cfg['ckpt_every_iter'],
    )


def main():
    import sys
    use_cli = '--use-cli' in sys.argv
    if not use_cli:
        args = _build_args_via_config()
    else:
        import argparse
        ap = argparse.ArgumentParser(description="PSO for Problem 3 (maximize M1 occlusion, 3 bombs)")
        ap.add_argument('--use-cli', action='store_true')
        ap.add_argument('--method', choices=['judge_caps', 'sampling'], default='judge_caps')
        ap.add_argument('--dt', type=float, default=0.05)
        ap.add_argument('--rel-max', type=float, default=66.0)
        ap.add_argument('--delay-max', type=float, default=20.0)
        ap.add_argument('--min-gap', type=float, default=1.0)
        ap.add_argument('--swarm', dest='swarm_size', type=int, default=50)
        ap.add_argument('--iters', type=int, default=300)
        ap.add_argument('--w-start', type=float, default=0.85)
        ap.add_argument('--w-end', type=float, default=0.45)
        ap.add_argument('--c1', type=float, default=1.7)
        ap.add_argument('--c2', type=float, default=1.7)
        ap.add_argument('--v-clamp-frac', type=float, default=0.25)
        ap.add_argument('--reset-fraction', type=float, default=0.10)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--progress', action='store_true')
        ap.add_argument('--progress-interval', type=float, default=1.0)
        ap.add_argument('--verbose', action='store_true')
        ap.add_argument('--checkpoint-file', type=str, default='best_p3_pso.json')
        ap.add_argument('--ckpt-every-iter', type=int, default=20)
        args_cli = ap.parse_args()
        args = SimpleNamespace(
            method=args_cli.method, dt=args_cli.dt, rel_max=args_cli.rel_max, delay_max=args_cli.delay_max, min_gap=args_cli.min_gap,
            swarm_size=args_cli.swarm_size, iters=args_cli.iters, w_start=args_cli.w_start, w_end=args_cli.w_end,
            c1=args_cli.c1, c2=args_cli.c2, v_clamp_frac=args_cli.v_clamp_frac, reset_fraction=args_cli.reset_fraction,
            seed=args_cli.seed, progress=args_cli.progress, progress_interval=args_cli.progress_interval,
            verbose=args_cli.verbose, checkpoint_file=args_cli.checkpoint_file, ckpt_every_iter=args_cli.ckpt_every_iter,
        )

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t0 = time.time()
    best = pso(args)
    t1 = time.time()

    spd, az, bombs = decode_position(best.x, args)
    print("=== PSO Result (Problem 3) ===")
    print(f"Best occluded time: {best.f:.4f} s")
    print(f"Speed: {spd:.3f} m/s")
    print(f"Azimuth: {az:.6f} rad  (deg={math.degrees(az):.2f})")
    for i, (t_rel, dly) in enumerate(bombs, 1):
        print(f"Bomb{i}: deploy={t_rel:.3f} s, explode_delay={dly:.3f} s (explode @ {t_rel + dly:.3f} s)")
    print(f"Runtime: {t1 - t0:.2f} s | Iters: {args.iters} | Swarm: {args.swarm_size}")

    # verify with sampling if used judge_caps
    if args.method == 'judge_caps':
        try:
            res_sampling = evaluate_problem3(bombs=bombs, speed=spd, azimuth=az, dt=args.dt, occlusion_method='sampling')
            v2 = float(res_sampling['occluded_time']['M1'])
            print(f"(Sampling verification) Occluded time: {v2:.4f} s")
        except Exception as e:
            print(f"Sampling verification failed: {e}")

    # export to Excel
    try:
        export_excel(best.x, best.f, 'result1.xlsx')
    except Exception as e:
        print(f"Export failed: {e}")


if __name__ == '__main__':
    main()
