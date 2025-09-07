from __future__ import annotations
"""
PSO + Differential Evolution hybrid solver for Problem 4:
  3 UAVs (FY1,FY2,FY3), each with 1 smoke bomb. Maximize occlusion time on M1.

Decision vector (len=12):
  x = [s1, a1, t1, d1,  s2, a2, t2, d2,  s3, a3, t3, d3]
    - si ∈ [70, 140]
    - ai ∈ [0, 2π)
    - ti ∈ [0, rel_max]
    - di ∈ [0, delay_max]

Run:
  python pso_de_problem4.py  # uses CONFIG below
  python pso_de_problem4.py --use-cli --iters 300 --pop 60 --method judge_caps

Notes:
  - judge_caps is fast but conservative; sampling is slower but more accurate.
  - Results exported to result2.xlsx. JSON checkpoint saved if enabled.
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

from optimizer_api import evaluate_problem4

# --------------- User config ---------------
CONFIG = dict(
    # search space
    rel_max=66.0,
    delay_max=20.0,

    # evaluation
    method="judge_caps",  # 'judge_caps' | 'sampling'
    dt=0.05,

    # PSO params
    pop_size=60,
    iters=300,
    w=0.70,          # inertia
    c1=1.7,          # cognitive
    c2=1.7,          # social
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
    checkpoint_file="best_p4_pso_de.json",
    # optional wall-clock time budget (seconds). 0 = disabled
    time_budget_sec=0,
)

# --------------- Bounds ---------------
SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZ_MIN, AZ_MAX = 0.0, 2.0 * math.pi
DELAY_MIN, DELAY_MAX_DEFAULT = 0.0, 20.0

# UAV initial positions (for reference/export; evaluation uses evaluate_problem4)
FY1_POS = np.array([17800.0,     0.0, 1800.0])
FY2_POS = np.array([12000.0,  1400.0, 1400.0])
FY3_POS = np.array([ 6000.0, -3000.0,  700.0])

AZ_IDXS = (1, 5, 9)

# Optional default seed file path
DEFAULT_SEED_FILE = "best_p4_seed.json"

# --------------- Helpers ---------------

def clip(x, lo, hi):
    return min(max(x, lo), hi)


def wrap_angle(a: float) -> float:
    return a % (2.0 * math.pi)


def decode_position(x: np.ndarray, cfg: SimpleNamespace):
    s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3 = [float(v) for v in x]
    s1 = clip(s1, SPEED_MIN, SPEED_MAX); s2 = clip(s2, SPEED_MIN, SPEED_MAX); s3 = clip(s3, SPEED_MIN, SPEED_MAX)
    a1 = wrap_angle(a1); a2 = wrap_angle(a2); a3 = wrap_angle(a3)
    t1 = clip(t1, 0.0, cfg.rel_max); t2 = clip(t2, 0.0, cfg.rel_max); t3 = clip(t3, 0.0, cfg.rel_max)
    d1 = clip(d1, DELAY_MIN, cfg.delay_max); d2 = clip(d2, DELAY_MIN, cfg.delay_max); d3 = clip(d3, DELAY_MIN, cfg.delay_max)
    return [s1, s2, s3], [a1, a2, a3], [t1, t2, t3], [d1, d2, d3]


def encode_position(speeds, azs, times, delays) -> np.ndarray:
    return np.array([speeds[0], azs[0], times[0], delays[0],
                     speeds[1], azs[1], times[1], delays[1],
                     speeds[2], azs[2], times[2], delays[2]], dtype=float)


def evaluate_vec(x: np.ndarray, cfg: SimpleNamespace) -> float:
    speeds, azs, times, delays = decode_position(x, cfg)
    drones_spec = [
        {"pos0": FY1_POS.tolist(), "speed": speeds[0], "azimuth": azs[0], "bombs": [{"deploy_time": times[0], "explode_delay": delays[0]}]},
        {"pos0": FY2_POS.tolist(), "speed": speeds[1], "azimuth": azs[1], "bombs": [{"deploy_time": times[1], "explode_delay": delays[1]}]},
        {"pos0": FY3_POS.tolist(), "speed": speeds[2], "azimuth": azs[2], "bombs": [{"deploy_time": times[2], "explode_delay": delays[2]}]},
    ]
    res = evaluate_problem4(drones_spec=drones_spec, dt=cfg.dt, occlusion_method=cfg.method)
    return float(res["occluded_time"]["M1"])  # maximize


def random_vec(cfg: SimpleNamespace) -> np.ndarray:
    def base_az(pos):
        return math.atan2(-pos[1], -pos[0])
    a1 = wrap_angle(base_az(FY1_POS) + random.gauss(0.0, 0.30))
    a2 = wrap_angle(base_az(FY2_POS) + random.gauss(0.0, 0.30))
    a3 = wrap_angle(base_az(FY3_POS) + random.gauss(0.0, 0.35))
    s1 = random.uniform(90.0, 130.0)
    s2 = random.uniform(90.0, 130.0)
    s3 = random.uniform(90.0, 130.0)
    t1 = random.uniform(0.0, min(6.0, cfg.rel_max))
    t2 = random.uniform(0.0, min(6.0, cfg.rel_max))
    t3 = random.uniform(0.0, min(6.0, cfg.rel_max))
    d1 = random.uniform(2.0, min(6.0, cfg.delay_max))
    d2 = random.uniform(2.0, min(6.0, cfg.delay_max))
    d3 = random.uniform(2.0, min(6.0, cfg.delay_max))
    return np.array([s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3], dtype=float)


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

    # Try to load optional seed
    seed_path = getattr(cfg, 'init_seed_file', None)
    if seed_path is None and os.path.isfile(DEFAULT_SEED_FILE):
        seed_path = DEFAULT_SEED_FILE
    if seed_path and os.path.isfile(seed_path):
        try:
            with open(seed_path, 'r', encoding='utf-8') as f:
                sd = json.load(f)
            seed_x = np.array([
                float(sd.get('s1')), float(sd.get('a1')), float(sd.get('t1')), float(sd.get('d1')),
                float(sd.get('s2')), float(sd.get('a2')), float(sd.get('t2')), float(sd.get('d2')),
                float(sd.get('s3')), float(sd.get('a3')), float(sd.get('t3')), float(sd.get('d3')),
            ], dtype=float)
            # repair/clip
            speeds, azs, times, delays = decode_position(seed_x, cfg)
            seed_x = encode_position(speeds, azs, times, delays)
            v0 = rng.normal(0.0, [5.0, 0.3, 2.0, 1.0] * 3).astype(float) * 0.25
            val = evaluate_vec(seed_x, cfg)
            pop.append(Particle(x=seed_x.copy(), v=v0.copy(), pbest_x=seed_x.copy(), pbest_val=val))
            if getattr(cfg, 'progress', False):
                print(f"[seed] loaded {seed_path} with value≈{val:.4f}s")
        except Exception as e:
            print(f"[warn] failed to load seed from {seed_path}: {e}")

    for _ in range(len(pop), cfg.pop_size):
        x0 = random_vec(cfg)
        # init velocity scale heuristics per dim: [spd, ang, time, delay] x3
        v0 = rng.normal(0.0, [5.0, 0.3, 2.0, 1.0] * 3).astype(float)
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
            speeds, azs, times, delays = decode_position(p.x, cfg)
            p.x = encode_position(speeds, azs, times, delays)

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
            speeds, azs, times, delays = decode_position(p.x, cfg)
            p.x = encode_position(speeds, azs, times, delays)

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
    speeds, azs, times, delays = decode_position(u, cfg)
    u = encode_position(speeds, azs, times, delays)
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
    s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3 = [float(z) for z in x]
    bombs = [
        dict(drone="FY1", deploy_time=t1, explode_delay=d1),
        dict(drone="FY2", deploy_time=t2, explode_delay=d2),
        dict(drone="FY3", deploy_time=t3, explode_delay=d3),
    ]
    return dict(FY1_speed=s1, FY1_azimuth=a1, FY2_speed=s2, FY2_azimuth=a2, FY3_speed=s3, FY3_azimuth=a3,
                bombs=bombs, value=val)


def export_excel(x: np.ndarray, val: float, path: str):
    try:
        import pandas as pd
    except Exception as e:
        print(f"[warn] pandas not available, cannot write {path}. Please: pip install pandas openpyxl. Error: {e}")
        return
    s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3 = [float(z) for z in x]
    rows = [{
        "FY1_speed(m/s)": s1,
        "FY1_azimuth(deg)": math.degrees(a1),
        "FY1_bomb_deploy(s)": t1,
        "FY1_bomb_explode(s)": t1 + d1,
        "FY2_speed(m/s)": s2,
        "FY2_azimuth(deg)": math.degrees(a2),
        "FY2_bomb_deploy(s)": t2,
        "FY2_bomb_explode(s)": t2 + d2,
        "FY3_speed(m/s)": s3,
        "FY3_azimuth(deg)": math.degrees(a3),
        "FY3_bomb_deploy(s)": t3,
        "FY3_bomb_explode(s)": t3 + d3,
        "total_occluded_time(s)": float(val),
    }]
    try:
        df = pd.DataFrame(rows)
        df.to_excel(path, index=False)
        print(f"Saved result to {path}")
    except Exception as e:
        print(f"[warn] failed to save {path}: {e}")


def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        rel_max=cfg['rel_max'], delay_max=cfg['delay_max'],
        method=cfg['method'], dt=cfg['dt'],
        pop_size=cfg['pop_size'], iters=cfg['iters'], w=cfg['w'], c1=cfg['c1'], c2=cfg['c2'], vmax=cfg['vmax'],
        de_frac=cfg['de_frac'], de_strategy=cfg['de_strategy'], F=cfg['F'], CR=cfg['CR'],
        elite_keep=cfg['elite_keep'], jitter_std=cfg['jitter_std'], seed=cfg['seed'],
        progress=cfg['progress'], progress_interval=cfg['progress_interval'], checkpoint_file=cfg['checkpoint_file'],
        time_budget_sec=cfg.get('time_budget_sec', 0), init_seed_file=None,
    )


def main():
    import argparse
    import sys

    use_cli = '--use-cli' in sys.argv
    if not use_cli:
        args = _build_args_via_config()
    else:
        ap = argparse.ArgumentParser(description="PSO+DE hybrid solver for Problem 4 (3 UAVs, 1 bomb each, maximize M1 occlusion)")
        ap.add_argument('--use-cli', action='store_true')
        ap.add_argument('--rel-max', type=float, default=66.0)
        ap.add_argument('--delay-max', type=float, default=20.0)
        ap.add_argument('--method', choices=['judge_caps', 'sampling'], default='judge_caps')
        ap.add_argument('--dt', type=float, default=0.05)
        ap.add_argument('--pop', type=int, default=60)
        ap.add_argument('--iters', type=int, default=300)
        ap.add_argument('--w', type=float, default=0.70)
        ap.add_argument('--c1', type=float, default=1.7)
        ap.add_argument('--c2', type=float, default=1.7)
        ap.add_argument('--F', type=float, default=0.55)
        ap.add_argument('--CR', type=float, default=0.8)
        ap.add_argument('--de-frac', type=float, default=0.30)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--progress', action='store_true')
        ap.add_argument('--progress-interval', type=float, default=0.5)
        ap.add_argument('--checkpoint-file', type=str, default='')
        ap.add_argument('--time-budget-sec', type=float, default=0.0)
        ap.add_argument('--init-seed-file', type=str, default='')
        args_cli = ap.parse_args()
        args = SimpleNamespace(
            rel_max=args_cli.rel_max, delay_max=args_cli.delay_max,
            method=args_cli.method, dt=args_cli.dt,
            pop_size=args_cli.pop, iters=args_cli.iters, w=args_cli.w, c1=args_cli.c1, c2=args_cli.c2, vmax=None,
            de_frac=args_cli.de_frac, de_strategy='current_to_best', F=args_cli.F, CR=args_cli.CR,
            elite_keep=3, jitter_std=0.02, seed=args_cli.seed,
            progress=args_cli.progress, progress_interval=args_cli.progress_interval, checkpoint_file=args_cli.checkpoint_file,
            time_budget_sec=args_cli.time_budget_sec, init_seed_file=(args_cli.init_seed_file or None),
        )

    # run
    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t0 = time.time()
    gbest_x, gbest_val = pso_de_hybrid(args)
    t1 = time.time()

    # print result
    speeds, azs, times, delays = decode_position(gbest_x, args)
    print("=== PSO+DE Result (Problem 4) ===")
    print(f"Best occluded time: {gbest_val:.4f} s")
    print(f"FY1: speed={speeds[0]:.2f} m/s, az={azs[0]:.6f} rad (deg={math.degrees(azs[0]):.2f}), release={times[0]:.3f} s, delay={delays[0]:.3f} s")
    print(f"FY2: speed={speeds[1]:.2f} m/s, az={azs[1]:.6f} rad (deg={math.degrees(azs[1]):.2f}), release={times[1]:.3f} s, delay={delays[1]:.3f} s")
    print(f"FY3: speed={speeds[2]:.2f} m/s, az={azs[2]:.6f} rad (deg={math.degrees(azs[2]):.2f}), release={times[2]:.3f} s, delay={delays[2]:.3f} s")
    print(f"Runtime: {t1 - t0:.2f} s | iters={args.iters} pop={args.pop_size}")

    # verify with sampling if search used judge_caps
    if args.method == 'judge_caps':
        try:
            from optimizer_api import evaluate_problem4 as eval_p4
            drones_spec = [
                {"pos0": FY1_POS.tolist(), "speed": speeds[0], "azimuth": azs[0], "bombs": [{"deploy_time": times[0], "explode_delay": delays[0]}]},
                {"pos0": FY2_POS.tolist(), "speed": speeds[1], "azimuth": azs[1], "bombs": [{"deploy_time": times[1], "explode_delay": delays[1]}]},
                {"pos0": FY3_POS.tolist(), "speed": speeds[2], "azimuth": azs[2], "bombs": [{"deploy_time": times[2], "explode_delay": delays[2]}]},
            ]
            v2 = float(eval_p4(drones_spec=drones_spec, dt=args.dt, occlusion_method='sampling')["occluded_time"]["M1"]) 
            print(f"(Sampling verification) Occluded time: {v2:.4f} s")
        except Exception as e:
            print(f"Sampling verification failed: {e}")

    # export
    try:
        export_excel(gbest_x, gbest_val, 'result2.xlsx')
    except Exception as e:
        print(f"Export failed: {e}")

    # save json
    if args.checkpoint_file:
        save_checkpoint(args.checkpoint_file, gbest_x, gbest_val, args.iters, t1 - t0)


if __name__ == '__main__':
    main()
