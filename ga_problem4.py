from __future__ import annotations
"""
Genetic Algorithm (GA) for Problem 4
  3 UAVs (FY1, FY2, FY3), each carries one smoke bomb. Optimize each UAV's
  speed, azimuth, and its bomb's release time and explode delay to maximize
  occluded time on M1.

Decision vector x (len=12):
  [s1, a1, t1, d1,  s2, a2, t2, d2,  s3, a3, t3, d3]
    - si ∈ [70, 140] m/s
    - ai ∈ [0, 2π) (wrap-around)
    - ti ∈ [0, rel_max]
    - di ∈ [0, delay_max]

Usage:
  python ga_problem4.py
  python ga_problem4.py --use-cli --pop 20000 --gens 150 --workers 8 --init-seed-file best_p4_seed.json

Notes:
  - judge_caps is faster but conservative; sampling is slower but more accurate.
  - For very large populations, consider increasing --workers to use more CPU cores.
  - Results exported to result2.xlsx (install pandas/openpyxl if missing).
"""
import math
import os
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, Dict, Optional

import numpy as np

from optimizer_api import evaluate_problem4

# ===================== User-configurable block =====================
CONFIG = dict(
    # simulation / objective
    method="judge_caps",   # 'judge_caps' or 'sampling'
    dt=0.1,
    rel_max=66.0,
    delay_max=20.0,

    # GA hyper-parameters
    population_size=2000,   # you can push to tens of thousands if CPU allows
    generations=100,
    elite_fraction=0.01,     # keep top 1% as elites
    tournament_k=3,
    crossover_rate=0.9,
    mutation_rate=0.15,      # per-gene mutation probability
    mutation_sigma_frac=0.08, # mutation std as fraction of variable span (angles use fraction of pi)
    jitter_seed_fraction=0.01, # fraction of population initialized as seed jitters if seed is provided

    # misc
    seed=42,

    # logging / checkpoint
    progress=True,
    progress_interval=2.0,  # seconds
    verbose=False,
    checkpoint_file="best_p4_ga.json",
    ckpt_every_gen=1,

    # evaluation batching / parallel
    batch_size=1024,
    workers=1,              # >1 uses multiprocessing Pool
)
# ==================================================================

# UAV initial positions (fixed heights)
FY1_POS = np.array([17800.0,     0.0, 1800.0])
FY2_POS = np.array([12000.0,  1400.0, 1400.0])
FY3_POS = np.array([ 6000.0, -3000.0,  700.0])

# Allow optional seed injection file (default path)
DEFAULT_SEED_FILE = "best_p4_seed.json"

SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi
DELAY_MIN = 0.0
AZ_IDXS = (1, 5, 9)  # indices of azimuth dims in the 12-d vector


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


def _bounds(cfg: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray]:
    lo_block = np.array([SPEED_MIN, AZIM_MIN, 0.0, DELAY_MIN], dtype=float)
    hi_block = np.array([SPEED_MAX, AZIM_MAX, cfg.rel_max, cfg.delay_max], dtype=float)
    lo = np.tile(lo_block, 3)
    hi = np.tile(hi_block, 3)
    return lo, hi


def _clip_and_wrap_raw(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    y = np.minimum(np.maximum(x, lo), hi)
    # wrap azimuth dims
    for k in AZ_IDXS:
        y[k] = wrap_angle(y[k])
    return y


def _base_az_for(pos_xy: Tuple[float, float]) -> float:
    x0, y0 = pos_xy
    # azimuth pointing roughly towards fake target (0,0) in XY plane
    return math.atan2(-y0, -x0)


def decode_position(x: np.ndarray, cfg: SimpleNamespace):
    """Map raw vector to feasible decision variables for 3 UAVs.
    Returns (speeds[3], azimuths[3], times[3], delays[3]).
    """
    y = x.astype(float).copy()
    lo, hi = _bounds(cfg)
    y = _clip_and_wrap_raw(y, lo, hi)
    s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3 = y.tolist()
    speeds = [float(np.clip(s1, SPEED_MIN, SPEED_MAX)), float(np.clip(s2, SPEED_MIN, SPEED_MAX)), float(np.clip(s3, SPEED_MIN, SPEED_MAX))]
    azs = [wrap_angle(a1), wrap_angle(a2), wrap_angle(a3)]
    times = [float(np.clip(t1, 0.0, cfg.rel_max)), float(np.clip(t2, 0.0, cfg.rel_max)), float(np.clip(t3, 0.0, cfg.rel_max))]
    delays = [float(np.clip(d1, DELAY_MIN, cfg.delay_max)), float(np.clip(d2, DELAY_MIN, cfg.delay_max)), float(np.clip(d3, DELAY_MIN, cfg.delay_max))]
    return speeds, azs, times, delays


def encode_feasible(speeds: List[float], azs: List[float], times: List[float], delays: List[float]) -> np.ndarray:
    return np.array([
        float(speeds[0]), float(azs[0]), float(times[0]), float(delays[0]),
        float(speeds[1]), float(azs[1]), float(times[1]), float(delays[1]),
        float(speeds[2]), float(azs[2]), float(times[2]), float(delays[2]),
    ], dtype=float)


def evaluate_vec(x: np.ndarray, cfg: SimpleNamespace) -> float:
    speeds, azs, times, delays = decode_position(x, cfg)
    drones_spec = [
        {"pos0": FY1_POS.tolist(), "speed": speeds[0], "azimuth": azs[0], "bombs": [{"deploy_time": times[0], "explode_delay": delays[0]}]},
        {"pos0": FY2_POS.tolist(), "speed": speeds[1], "azimuth": azs[1], "bombs": [{"deploy_time": times[1], "explode_delay": delays[1]}]},
        {"pos0": FY3_POS.tolist(), "speed": speeds[2], "azimuth": azs[2], "bombs": [{"deploy_time": times[2], "explode_delay": delays[2]}]},
    ]
    res = evaluate_problem4(drones_spec=drones_spec, dt=cfg.dt, occlusion_method=cfg.method)
    return float(res["occluded_time"]["M1"])  # maximize


def random_vec(cfg: SimpleNamespace, rng: np.random.Generator) -> np.ndarray:
    # heuristic base around pointing to origin
    base_az = [
        _base_az_for((FY1_POS[0], FY1_POS[1])),
        _base_az_for((FY2_POS[0], FY2_POS[1])),
        _base_az_for((FY3_POS[0], FY3_POS[1])),
    ]
    s1 = rng.uniform(90.0, 130.0); a1 = wrap_angle(base_az[0] + rng.normal(0.0, 0.3))
    s2 = rng.uniform(90.0, 130.0); a2 = wrap_angle(base_az[1] + rng.normal(0.0, 0.3))
    s3 = rng.uniform(90.0, 130.0); a3 = wrap_angle(base_az[2] + rng.normal(0.0, 0.35))
    # early releases are often helpful
    t1 = rng.uniform(0.0, min(6.0, cfg.rel_max))
    t2 = rng.uniform(0.0, min(6.0, cfg.rel_max))
    t3 = rng.uniform(0.0, min(6.0, cfg.rel_max))
    d1 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    d2 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    d3 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    x = np.array([s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3], dtype=float)
    lo, hi = _bounds(cfg)
    return _clip_and_wrap_raw(x, lo, hi)


# ---------------- GA core ----------------

@dataclass
class Individual:
    x: np.ndarray   # position (12,)
    f: float        # fitness

    def as_dict(self) -> Dict:
        return dict(
            s1=float(self.x[0]), a1=float(self.x[1]), t1=float(self.x[2]), d1=float(self.x[3]),
            s2=float(self.x[4]), a2=float(self.x[5]), t2=float(self.x[6]), d2=float(self.x[7]),
            s3=float(self.x[8]), a3=float(self.x[9]), t3=float(self.x[10]), d3=float(self.x[11]),
            value=float(self.f),
        )


def _init_population(cfg: SimpleNamespace, rng: np.random.Generator) -> List[np.ndarray]:
    lo, hi = _bounds(cfg)
    pop: List[np.ndarray] = []

    # Try to load optional seed from cfg.init_seed_file or default path
    seed_x: Optional[np.ndarray] = None
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
            seed_x = _clip_and_wrap_raw(seed_x, lo, hi)
            if getattr(cfg, 'progress', False):
                print(f"[seed] loaded {seed_path}")
        except Exception as e:
            print(f"[warn] failed to load seed from {seed_path}: {e}")

    # base heuristic
    base_az = [
        _base_az_for((FY1_POS[0], FY1_POS[1])),
        _base_az_for((FY2_POS[0], FY2_POS[1])),
        _base_az_for((FY3_POS[0], FY3_POS[1])),
    ]
    base = np.array([
        120.0, base_az[0], 2.0, 3.5,
        120.0, base_az[1], 4.0, 4.0,
        120.0, base_az[2], 4.0, 4.0,
    ], dtype=float)

    # how many seed jitters
    n = int(cfg.population_size)
    n_seed_jitter = 0
    if seed_x is not None and cfg.jitter_seed_fraction > 0.0:
        n_seed_jitter = max(1, int(cfg.jitter_seed_fraction * n))

    for i in range(n):
        if i == 0 and seed_x is not None:
            x = seed_x.copy()
        elif i < 1 + n_seed_jitter and seed_x is not None:
            span = (_bounds(cfg)[1] - _bounds(cfg)[0])
            jitter = np.zeros(12, dtype=float)
            jitter[0::4] = rng.normal(0.0, 0.05 * span[0], size=3)  # speeds
            for k in AZ_IDXS:
                jitter[k] = (rng.random() * 2.0 - 1.0) * 0.05 * math.pi
            jitter[2::4] = rng.normal(0.0, 0.05 * span[2], size=3)  # release times
            jitter[3::4] = rng.normal(0.0, 0.05 * span[3], size=3)  # delays
            x = _clip_and_wrap_raw(seed_x + jitter, *_bounds(cfg))
        else:
            if rng.random() < 0.35:
                x = base + rng.normal(0.0, 0.12, size=12) * (_bounds(cfg)[1] - _bounds(cfg)[0])
            else:
                lo, hi = _bounds(cfg)
                x = lo + rng.random(12) * (hi - lo)
            x = _clip_and_wrap_raw(x, *_bounds(cfg))
        pop.append(x)

    return pop


def _evaluate_population(pop: List[np.ndarray], cfg: SimpleNamespace) -> np.ndarray:
    # Sequential by default; optionally parallel
    vals = np.empty(len(pop), dtype=float)
    if getattr(cfg, 'workers', 1) and cfg.workers > 1:
        try:
            import multiprocessing as mp
            def _eval_one(x_arr: np.ndarray) -> float:
                return evaluate_vec(x_arr, cfg)
            with mp.Pool(processes=int(cfg.workers)) as pool:
                for i, f in enumerate(pool.imap(_eval_one, pop, chunksize=16)):
                    vals[i] = float(f)
        except Exception as e:
            print(f"[warn] multiprocessing disabled due to: {e}; falling back to sequential")
            for i, x in enumerate(pop):
                vals[i] = evaluate_vec(x, cfg)
    else:
        for i, x in enumerate(pop):
            vals[i] = evaluate_vec(x, cfg)
    return vals


def _tournament_select(fitness: np.ndarray, rng: np.random.Generator, k: int) -> int:
    n = len(fitness)
    idxs = rng.integers(0, n, size=k)
    best = int(idxs[0])
    best_f = fitness[best]
    for j in idxs[1:]:
        if fitness[int(j)] > best_f:
            best = int(j)
            best_f = fitness[best]
    return best


def _crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, cfg: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray]:
    # Blend-like crossover with circular handling for angles
    lo, hi = _bounds(cfg)
    span = hi - lo
    c1 = p1.copy()
    c2 = p2.copy()
    if rng.random() < cfg.crossover_rate:
        u = rng.random(12)
        for i in range(12):
            if i in AZ_IDXS:
                diff = ang_diff(p2[i], p1[i])
                c1[i] = wrap_angle(p1[i] + u[i] * diff)
                c2[i] = wrap_angle(p2[i] - u[i] * diff)
            else:
                di = p2[i] - p1[i]
                c1[i] = p1[i] + u[i] * di
                c2[i] = p2[i] - u[i] * di
    c1 = _clip_and_wrap_raw(c1, lo, hi)
    c2 = _clip_and_wrap_raw(c2, lo, hi)
    return c1, c2


def _mutate(x: np.ndarray, rng: np.random.Generator, cfg: SimpleNamespace) -> np.ndarray:
    lo, hi = _bounds(cfg)
    span = hi - lo
    y = x.copy()
    for i in range(12):
        if rng.random() < cfg.mutation_rate:
            if i in AZ_IDXS:
                y[i] = wrap_angle(y[i] + (rng.random() * 2.0 - 1.0) * (cfg.mutation_sigma_frac * math.pi))
            else:
                y[i] = y[i] + rng.normal(0.0, cfg.mutation_sigma_frac * span[i])
    y = _clip_and_wrap_raw(y, lo, hi)
    return y


def ga(cfg: SimpleNamespace) -> Individual:
    rng = np.random.default_rng(cfg.seed)
    pop = _init_population(cfg, rng)
    fitness = _evaluate_population(pop, cfg)

    # initial best & checkpoint
    best_idx = int(np.argmax(fitness))
    best = Individual(x=pop[best_idx].copy(), f=float(fitness[best_idx]))
    if cfg.checkpoint_file:
        _atomic_save_json(dict(gen=0, timestamp=time.time()) | best.as_dict(), cfg.checkpoint_file)

    start = time.time()
    last_prog = start

    n = len(pop)
    elite = max(1, int(cfg.elite_fraction * n))

    for gen in range(1, cfg.generations + 1):
        # Elitism
        order = np.argsort(-fitness)
        elites_idx = order[:elite]
        elites = [pop[i].copy() for i in elites_idx]

        # Mating pool via tournament selection
        children: List[np.ndarray] = []
        while len(children) < n - elite:
            i1 = _tournament_select(fitness, rng, cfg.tournament_k)
            i2 = _tournament_select(fitness, rng, cfg.tournament_k)
            if i1 == i2:
                i2 = (i2 + 1) % n
            p1, p2 = pop[i1], pop[i2]
            c1, c2 = _crossover(p1, p2, rng, cfg)
            c1 = _mutate(c1, rng, cfg)
            c2 = _mutate(c2, rng, cfg)
            children.append(c1)
            if len(children) < n - elite:
                children.append(c2)

        # New population
        pop = elites + children[: n - elite]
        fitness = _evaluate_population(pop, cfg)

        # Track best
        best_idx = int(np.argmax(fitness))
        cur_best = Individual(x=pop[best_idx].copy(), f=float(fitness[best_idx]))
        if cur_best.f > best.f:
            best = cur_best

        # progress
        now = time.time()
        if cfg.progress and (now - last_prog >= cfg.progress_interval):
            elapsed = now - start
            print(
                f"[prog] gen={gen}/{cfg.generations} best={best.f:.4f}s "
                f"FY1(s={best.x[0]:.1f},az={best.x[1]:.2f},t={best.x[2]:.2f},d={best.x[3]:.2f}) "
                f"FY2(s={best.x[4]:.1f},az={best.x[5]:.2f},t={best.x[6]:.2f},d={best.x[7]:.2f}) "
                f"FY3(s={best.x[8]:.1f},az={best.x[9]:.2f},t={best.x[10]:.2f},d={best.x[11]:.2f}) "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            last_prog = now

        # checkpoint
        if cfg.ckpt_every_gen and (gen % cfg.ckpt_every_gen == 0) and cfg.checkpoint_file:
            _atomic_save_json(dict(gen=gen, timestamp=time.time()) | best.as_dict(), cfg.checkpoint_file)

    # final checkpoint
    if cfg.checkpoint_file:
        _atomic_save_json(dict(gen=cfg.generations, timestamp=time.time()) | best.as_dict(), cfg.checkpoint_file)

    return best


# ---------------- Export helpers ----------------

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
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(path, index=False)
        print(f"Saved result to {path}")
    except Exception as e:
        print(f"[warn] failed to save {path}: {e}")


# ---------------- Driver ----------------

def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        method=cfg['method'], dt=cfg['dt'], rel_max=cfg['rel_max'], delay_max=cfg['delay_max'],
        population_size=cfg['population_size'], generations=cfg['generations'], elite_fraction=cfg['elite_fraction'],
        tournament_k=cfg['tournament_k'], crossover_rate=cfg['crossover_rate'], mutation_rate=cfg['mutation_rate'],
        mutation_sigma_frac=cfg['mutation_sigma_frac'], jitter_seed_fraction=cfg['jitter_seed_fraction'],
        seed=cfg['seed'], progress=cfg['progress'], progress_interval=cfg['progress_interval'],
        verbose=cfg['verbose'], checkpoint_file=cfg['checkpoint_file'], ckpt_every_gen=cfg['ckpt_every_gen'],
        batch_size=cfg['batch_size'], workers=cfg['workers'], init_seed_file=None,
    )


def main():
    import sys
    use_cli = '--use-cli' in sys.argv
    if not use_cli:
        args = _build_args_via_config()
    else:
        import argparse
        ap = argparse.ArgumentParser(description="Genetic Algorithm for Problem 4 (3 UAVs, 1 bomb each, maximize M1 occlusion)")
        ap.add_argument('--use-cli', action='store_true')
        ap.add_argument('--method', choices=['judge_caps', 'sampling'], default='judge_caps')
        ap.add_argument('--dt', type=float, default=0.05)
        ap.add_argument('--rel-max', type=float, default=66.0)
        ap.add_argument('--delay-max', type=float, default=20.0)
        ap.add_argument('--pop', dest='population_size', type=int, default=12000)
        ap.add_argument('--gens', dest='generations', type=int, default=150)
        ap.add_argument('--elite-frac', type=float, default=0.01)
        ap.add_argument('--tournament-k', type=int, default=3)
        ap.add_argument('--crossover-rate', type=float, default=0.9)
        ap.add_argument('--mutation-rate', type=float, default=0.15)
        ap.add_argument('--mutation-sigma-frac', type=float, default=0.08)
        ap.add_argument('--jitter-seed-fraction', type=float, default=0.01)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--progress', action='store_true')
        ap.add_argument('--progress-interval', type=float, default=2.0)
        ap.add_argument('--verbose', action='store_true')
        ap.add_argument('--checkpoint-file', type=str, default='best_p4_ga.json')
        ap.add_argument('--ckpt-every-gen', type=int, default=1)
        ap.add_argument('--batch-size', type=int, default=1024)
        ap.add_argument('--workers', type=int, default=1)
        ap.add_argument('--init-seed-file', type=str, default='')
        args_cli = ap.parse_args()
        args = SimpleNamespace(
            method=args_cli.method, dt=args_cli.dt, rel_max=args_cli.rel_max, delay_max=args_cli.delay_max,
            population_size=args_cli.population_size, generations=args_cli.generations, elite_fraction=args_cli.elite_frac,
            tournament_k=args_cli.tournament_k, crossover_rate=args_cli.crossover_rate, mutation_rate=args_cli.mutation_rate,
            mutation_sigma_frac=args_cli.mutation_sigma_frac, jitter_seed_fraction=args_cli.jitter_seed_fraction,
            seed=args_cli.seed, progress=args_cli.progress, progress_interval=args_cli.progress_interval,
            verbose=args_cli.verbose, checkpoint_file=args_cli.checkpoint_file, ckpt_every_gen=args_cli.ckpt_every_gen,
            batch_size=args_cli.batch_size, workers=args_cli.workers, init_seed_file=(args_cli.init_seed_file or None),
        )

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t0 = time.time()
    best = ga(args)
    t1 = time.time()

    print("=== GA Result (Problem 4) ===")
    print(f"Best occluded time: {best.f:.4f} s")
    s1, a1, t1r, d1, s2, a2, t2r, d2, s3, a3, t3r, d3 = [float(z) for z in best.x]
    print(f"FY1: speed={s1:.2f} m/s, az={a1:.6f} rad (deg={math.degrees(a1):.2f}), release={t1r:.3f} s, delay={d1:.3f} s")
    print(f"FY2: speed={s2:.2f} m/s, az={a2:.6f} rad (deg={math.degrees(a2):.2f}), release={t2r:.3f} s, delay={d2:.3f} s")
    print(f"FY3: speed={s3:.2f} m/s, az={a3:.6f} rad (deg={math.degrees(a3):.2f}), release={t3r:.3f} s, delay={d3:.3f} s")
    print(f"Runtime: {time.time() - t0:.2f} s | Generations: {args.generations} | Pop: {args.population_size}")

    # verify with sampling if search used judge_caps
    if args.method == 'judge_caps':
        try:
            from optimizer_api import evaluate_problem4 as eval_p4
            speeds, azs, times, delays = decode_position(best.x, args)
            drones_spec = [
                {"pos0": FY1_POS.tolist(), "speed": speeds[0], "azimuth": azs[0], "bombs": [{"deploy_time": times[0], "explode_delay": delays[0]}]},
                {"pos0": FY2_POS.tolist(), "speed": speeds[1], "azimuth": azs[1], "bombs": [{"deploy_time": times[1], "explode_delay": delays[1]}]},
                {"pos0": FY3_POS.tolist(), "speed": speeds[2], "azimuth": azs[2], "bombs": [{"deploy_time": times[2], "explode_delay": delays[2]}]},
            ]
            v2 = float(eval_p4(drones_spec=drones_spec, dt=args.dt, occlusion_method='sampling')["occluded_time"]["M1"]) 
            print(f"(Sampling verification) Occluded time: {v2:.4f} s")
        except Exception as e:
            print(f"Sampling verification failed: {e}")

    # export to Excel
    try:
        export_excel(best.x, best.f, 'result2.xlsx')
    except Exception as e:
        print(f"Export failed: {e}")


if __name__ == '__main__':
    main()
