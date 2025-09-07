from __future__ import annotations
"""
PSO + Differential Evolution hybrid solver for Problem 3:
  Optimize FY1 (single UAV, 3 smoke bombs) to maximize occlusion time on M1.

Decision variables (encoded vector x):
  x = [speed, azimuth, t1, t2, t3, d1, d2, d3]
    - speed ∈ [70, 140]
    - azimuth ∈ [0, 2π)
    - t1,t2,t3 ∈ [0, rel_max] with t2 >= t1 + 1, t3 >= t2 + 1
    - d1,d2,d3 ∈ [0, delay_max]

Objective:
  Maximize occlusion time for M1 (evaluate via optimizer_api.evaluate_problem3).

Run:
  python pso_de_problem3.py  # uses CONFIG below
  python pso_de_problem3.py --use-cli --iters 400 --pop 60 --method judge_caps

Notes:
  - judge_caps is fast but conservative; sampling is slower but more accurate.
  - Results exported to result1.xlsx. JSON checkpoint saved if enabled.
"""
import math
import time
import json
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple, List, Dict

import numpy as np

from optimizer_api import evaluate_problem3

# --------------- User config ---------------
CONFIG = dict(
    # search space
    rel_max=66.0,
    delay_max=20.0,
    min_gap=1.0,

    # evaluation
    method="judge_caps",  # 'judge_caps' | 'sampling'
    dt=0.05,

    # PSO params
    pop_size=40,
    iters=250,
    w=0.68,          # inertia
    c1=1.6,          # cognitive
    c2=1.6,          # social
    vmax=None,       # optional velocity cap (None = auto)

    # DE injection (performed each iteration on a subset)
    de_frac=0.30,    # fraction of population to attempt DE on
    de_strategy="current_to_best", # 'rand1' | 'current_to_best'
    F=0.55,          # DE mutation factor
    CR=0.8,          # DE crossover rate

    # restarts / noise
    elite_keep=3,
    jitter_std=0.02,  # small noise injected to avoid stagnation

    # reproducibility
    seed=42,

    # logging
    progress=True,
    progress_interval=0.5,
    checkpoint_file="best_p3_pso_de.json",
    # optional wall-clock time budget (seconds). 0 = disabled
    time_budget_sec=0,
)

# --------------- Bounds ---------------
SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZ_MIN, AZ_MAX = 0.0, 2.0 * math.pi
RELEASE_MAX_DEFAULT = 66.0
DELAY_MIN, DELAY_MAX_DEFAULT = 0.0, 20.0

# --------------- Helpers ---------------

def clip(x, lo, hi):
    return min(max(x, lo), hi)


def wrap_angle(a: float) -> float:
    return a % (2.0 * math.pi)


def project_times(times: List[float], rel_max: float, min_gap: float) -> List[float]:
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
        # fallback to equally spaced inside bounds
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


def evaluate_vec(x: np.ndarray, cfg: SimpleNamespace) -> float:
    speed, az, bombs = decode_position(x, cfg)
    res = evaluate_problem3(bombs=bombs, speed=speed, azimuth=az, dt=cfg.dt, occlusion_method=cfg.method)
    return float(res["occluded_time"]["M1"])


def random_vec(cfg: SimpleNamespace) -> np.ndarray:
    speed = random.uniform(SPEED_MIN, SPEED_MAX)
    az = random.uniform(0.0, 2.0 * math.pi)
    # start early is often helpful
    t1 = random.uniform(0.0, min(5.0, cfg.rel_max))
    t2 = t1 + cfg.min_gap + random.uniform(0.0, 2.0)
    t3 = t2 + cfg.min_gap + random.uniform(0.0, 2.0)
    times = project_times([t1, t2, t3], cfg.rel_max, cfg.min_gap)
    d1 = random.uniform(2.0, min(6.0, cfg.delay_max))
    d2 = random.uniform(2.0, min(6.0, cfg.delay_max))
    d3 = random.uniform(2.0, min(6.0, cfg.delay_max))
    return np.array([speed, az, times[0], times[1], times[2], d1, d2, d3], dtype=float)


@dataclass
class Particle:
    x: np.ndarray
    v: np.ndarray
    pbest_x: np.ndarray
    pbest_val: float


def pso_de_hybrid(cfg: SimpleNamespace):
    rng = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    pop: List[Particle] = []
    for _ in range(cfg.pop_size):
        x0 = random_vec(cfg)
        # init velocity scale heuristics
        v0 = rng.normal(0.0, [5.0, 0.3, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]).astype(float)
        val = evaluate_vec(x0, cfg)
        pop.append(Particle(x=x0.copy(), v=v0.copy(), pbest_x=x0.copy(), pbest_val=val))

    # global best
    g_idx = int(np.argmax([p.pbest_val for p in pop]))
    gbest_x = pop[g_idx].pbest_x.copy()
    gbest_val = pop[g_idx].pbest_val

    start = time.time()
    last_prog = start
    deadline = None
    if hasattr(cfg, 'time_budget_sec') and cfg.time_budget_sec and cfg.time_budget_sec > 0:
        deadline = start + float(cfg.time_budget_sec)

    for it in range(1, cfg.iters + 1):
        # time budget check
        if deadline is not None and time.time() >= deadline:
            print("[time-budget] reached, stopping early.")
            break
        # PSO update
        for p in pop:
            r1 = rng.random(len(p.x))
            r2 = rng.random(len(p.x))
            cognitive = cfg.c1 * r1 * (p.pbest_x - p.x)
            social = cfg.c2 * r2 * (gbest_x - p.x)
            p.v = cfg.w * p.v + cognitive + social
            # optional velocity cap
            if cfg.vmax is not None:
                vmax = np.asarray(cfg.vmax, dtype=float)
                p.v = np.clip(p.v, -vmax, vmax)
            # position update
            p.x = p.x + p.v
            # repair feasible region
            speed, az, bombs = decode_position(p.x, cfg)
            t1, t2, t3 = bombs[0][0], bombs[1][0], bombs[2][0]
            d1, d2, d3 = bombs[0][1], bombs[1][1], bombs[2][1]
            p.x = np.array([speed, az, t1, t2, t3, d1, d2, d3], dtype=float)

        # Evaluate and update pbests
        for p in pop:
            val = evaluate_vec(p.x, cfg)
            if val > p.pbest_val:
                p.pbest_val = val
                p.pbest_x = p.x.copy()
                if val > gbest_val:
                    gbest_val = val
                    gbest_x = p.x.copy()

        # DE injection on a subset
        k = max(1, int(cfg.de_frac * cfg.pop_size))
        idxs = rng.choice(len(pop), size=k, replace=False)
        for idx in idxs:
            trial = de_trial(pop, idx, gbest_x, cfg, rng)
            if trial is not None:
                trial_val = evaluate_vec(trial, cfg)
                if trial_val > pop[idx].pbest_val:
                    pop[idx].pbest_val = trial_val
                    pop[idx].pbest_x = trial.copy()
                    pop[idx].x = trial.copy()  # move to trial
                    if trial_val > gbest_val:
                        gbest_val = trial_val
                        gbest_x = trial.copy()

        # inject small jitter to non-elite to avoid collapse
        elites = set(np.argsort([-p.pbest_val for p in pop])[:cfg.elite_keep])
        for i, p in enumerate(pop):
            if i in elites:
                continue
            p.x += rng.normal(0.0, cfg.jitter_std, size=p.x.shape)
            # reproject
            speed, az, bombs = decode_position(p.x, cfg)
            t1, t2, t3 = bombs[0][0], bombs[1][0], bombs[2][0]
            d1, d2, d3 = bombs[0][1], bombs[1][1], bombs[2][1]
            p.x = np.array([speed, az, t1, t2, t3, d1, d2, d3], dtype=float)

        # progress
        now = time.time()
        if cfg.progress and (now - last_prog >= cfg.progress_interval):
            elapsed = now - start
            print(f"[prog] iter={it}/{cfg.iters} best={gbest_val:.4f}s elapsed={elapsed:.1f}s", flush=True)
            last_prog = now

        # checkpoint
        if cfg.checkpoint_file and (it % max(50, cfg.iters // 10) == 0 or it == cfg.iters):
            save_checkpoint(cfg.checkpoint_file, gbest_x, gbest_val, it, time.time() - start)

    return gbest_x, gbest_val


def de_trial(pop: List[Particle], idx: int, gbest_x: np.ndarray, cfg: SimpleNamespace, rng: np.random.Generator) -> np.ndarray | None:
    NP = len(pop)
    i = idx
    # choose distinct r1, r2, r3
    candidates = [j for j in range(NP) if j != i]
    if len(candidates) < 3:
        return None
    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

    x_i = pop[i].x
    if cfg.de_strategy == "rand1":
        x_r1 = pop[r1].x; x_r2 = pop[r2].x; x_r3 = pop[r3].x
        v = x_r1 + cfg.F * (x_r2 - x_r3)
    else:  # current_to_best/1
        x_best = gbest_x
        x_r1 = pop[r1].x; x_r2 = pop[r2].x
        v = x_i + cfg.F * (x_best - x_i) + cfg.F * (x_r1 - x_r2)

    # binomial crossover
    d = len(x_i)
    j_rand = rng.integers(0, d)
    u = np.array([v[j] if (rng.random() < cfg.CR or j == j_rand) else x_i[j] for j in range(d)], dtype=float)

    # reproject to feasible manifold
    speed, az, bombs = decode_position(u, cfg)
    t1, t2, t3 = bombs[0][0], bombs[1][0], bombs[2][0]
    d1, d2, d3 = bombs[0][1], bombs[1][1], bombs[2][1]
    u = np.array([speed, az, t1, t2, t3, d1, d2, d3], dtype=float)
    return u


def save_checkpoint(path: str, gbest_x: np.ndarray, gbest_val: float, iters: int, elapsed: float):
    data = vector_to_dict(gbest_x, gbest_val)
    data.update(dict(iters=iters, elapsed_sec=elapsed, timestamp=time.time()))
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[warn] checkpoint save failed: {e}")


def vector_to_dict(x: np.ndarray, val: float) -> Dict:
    speed = float(x[0])
    az = float(x[1])
    bombs = [
        dict(deploy_time=float(x[2]), explode_delay=float(x[5])),
        dict(deploy_time=float(x[3]), explode_delay=float(x[6])),
        dict(deploy_time=float(x[4]), explode_delay=float(x[7])),
    ]
    return dict(speed=speed, azimuth=az, bombs=bombs, value=val)


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


def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        rel_max=cfg['rel_max'], delay_max=cfg['delay_max'], min_gap=cfg['min_gap'],
        method=cfg['method'], dt=cfg['dt'],
        pop_size=cfg['pop_size'], iters=cfg['iters'], w=cfg['w'], c1=cfg['c1'], c2=cfg['c2'], vmax=cfg['vmax'],
        de_frac=cfg['de_frac'], de_strategy=cfg['de_strategy'], F=cfg['F'], CR=cfg['CR'],
        elite_keep=cfg['elite_keep'], jitter_std=cfg['jitter_std'], seed=cfg['seed'],
        progress=cfg['progress'], progress_interval=cfg['progress_interval'], checkpoint_file=cfg['checkpoint_file'],
        time_budget_sec=cfg.get('time_budget_sec', 0),
    )


def main():
    import argparse
    import sys

    use_cli = '--use-cli' in sys.argv
    if not use_cli:
        args = _build_args_via_config()
    else:
        ap = argparse.ArgumentParser(description="PSO+DE hybrid solver for Problem 3 (3 bombs, maximize M1 occlusion)")
        ap.add_argument('--use-cli', action='store_true')
        ap.add_argument('--rel-max', type=float, default=RELEASE_MAX_DEFAULT)
        ap.add_argument('--delay-max', type=float, default=DELAY_MAX_DEFAULT)
        ap.add_argument('--min-gap', type=float, default=1.0)
        ap.add_argument('--method', choices=['judge_caps', 'sampling'], default='judge_caps')
        ap.add_argument('--dt', type=float, default=0.05)
        ap.add_argument('--pop', type=int, default=40)
        ap.add_argument('--iters', type=int, default=250)
        ap.add_argument('--w', type=float, default=0.68)
        ap.add_argument('--c1', type=float, default=1.6)
        ap.add_argument('--c2', type=float, default=1.6)
        ap.add_argument('--F', type=float, default=0.55)
        ap.add_argument('--CR', type=float, default=0.8)
        ap.add_argument('--de-frac', type=float, default=0.30)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--progress', action='store_true')
        ap.add_argument('--progress-interval', type=float, default=0.5)
        ap.add_argument('--checkpoint-file', type=str, default='')
        ap.add_argument('--time-budget-sec', type=float, default=0.0)
        args_cli = ap.parse_args()
        args = SimpleNamespace(
            rel_max=args_cli.rel_max, delay_max=args_cli.delay_max, min_gap=args_cli.min_gap,
            method=args_cli.method, dt=args_cli.dt,
            pop_size=args_cli.pop, iters=args_cli.iters, w=args_cli.w, c1=args_cli.c1, c2=args_cli.c2, vmax=None,
            de_frac=args_cli.de_frac, de_strategy='current_to_best', F=args_cli.F, CR=args_cli.CR,
            elite_keep=3, jitter_std=0.02, seed=args_cli.seed,
            progress=args_cli.progress, progress_interval=args_cli.progress_interval, checkpoint_file=args_cli.checkpoint_file,
            time_budget_sec=args_cli.time_budget_sec,
        )

    # run
    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t0 = time.time()
    gbest_x, gbest_val = pso_de_hybrid(args)
    t1 = time.time()

    # print result
    speed, az, bombs = decode_position(gbest_x, args)
    print("=== PSO+DE Result (Problem 3) ===")
    print(f"Best occluded time: {gbest_val:.4f} s")
    print(f"Speed: {speed:.3f} m/s")
    print(f"Azimuth: {az:.6f} rad  (deg={math.degrees(az):.2f})")
    for i, (t_rel, dly) in enumerate(bombs, 1):
        print(f"Bomb{i}: deploy={t_rel:.3f} s, explode_delay={dly:.3f} s (explode @ {t_rel + dly:.3f} s)")
    print(f"Runtime: {t1 - t0:.2f} s | iters={args.iters} pop={args.pop_size}")

    # verify with sampling if search used judge_caps
    if args.method == 'judge_caps':
        try:
            res_sampling = evaluate_problem3(bombs=bombs, speed=speed, azimuth=az, dt=args.dt, occlusion_method='sampling')
            v2 = float(res_sampling['occluded_time']['M1'])
            print(f"(Sampling verification) Occluded time: {v2:.4f} s")
        except Exception as e:
            print(f"Sampling verification failed: {e}")

    # export
    try:
        export_excel(gbest_x, gbest_val, 'result1.xlsx')
    except Exception as e:
        print(f"Export failed: {e}")

    # save json
    if args.checkpoint_file:
        save_checkpoint(args.checkpoint_file, gbest_x, gbest_val, args.iters, t1 - t0)


if __name__ == '__main__':
    main()
