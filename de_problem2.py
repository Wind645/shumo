from __future__ import annotations
"""
Differential Evolution (DE) solver for Problem 2:
  Optimize FY1 (single UAV, single smoke bomb) parameters to maximize M1 occlusion time.
Decision variables (x = [speed, azimuth, release_time, explode_delay]):
  - speed ∈ [70, 140] m/s
  - azimuth ∈ [0, 2π)  (horizontal heading, x-axis = 0, CCW positive)
  - release_time ∈ [0, rel_max]
  - explode_delay ∈ [0, delay_max]
Objective:
  Maximize total occluded time (seconds) returned by evaluate_problem2().

Usage:
  python de_problem2.py
  python de_problem2.py --use-cli --gens 400 --pop 60 --method judge_caps --seed 42

Notes:
  - "judge_caps" is fast and conservative; "sampling" is slower but more accurate.
  - You can raise rel_max / delay_max to enlarge the search space.
  - Supports periodic checkpointing of the current best solution.
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

    # DE hyper-parameters
    pop_size=60,            # population size (>= 4)
    gens=400,               # number of generations
    F=0.7,                  # differential weight (0.4~0.9)
    CR=0.9,                 # crossover probability (0.2~0.95)
    strategy="rand1bin",   # 'rand1bin' | 'best1bin'
    seed=42,                # None for system random

    # logging / checkpoint
    progress=True,
    progress_interval=1.0,  # seconds
    verbose=False,
    checkpoint_file="best_p2_de.json",
    ckpt_every_gen=20,
)
# ==================================================================

SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi

@dataclass
class Individual:
    x: np.ndarray  # [speed, azimuth, release_time, explode_delay]
    f: float       # objective value (occluded time)

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


def _bounds(rel_max: float, delay_max: float) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array([SPEED_MIN, AZIM_MIN, 0.0, 0.0], dtype=float)
    hi = np.array([SPEED_MAX, AZIM_MAX, rel_max, delay_max], dtype=float)
    return lo, hi


def _clip_and_wrap(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    y = np.minimum(np.maximum(x, lo), hi)
    # wrap azimuth
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


def _init_population(rng: np.random.Generator, pop_size: int, lo: np.ndarray, hi: np.ndarray,
                     method: str, dt: float) -> Tuple[list[Individual], Individual]:
    pop: list[Individual] = []
    best: Individual | None = None

    # Heuristic center near a reasonable baseline
    base = np.array([120.0, math.pi, 2.0, 4.0], dtype=float)
    span = hi - lo
    for i in range(pop_size):
        # 70% uniform in bounds, 30% around baseline
        if rng.random() < 0.3:
            x = base + rng.normal(0.0, 0.12, size=4) * span
        else:
            x = lo + rng.random(4) * span
        x[1] = wrap_angle(x[1])
        x = _clip_and_wrap(x, lo, hi)
        f = _eval_vec(x, method, dt)
        ind = Individual(x=x, f=f)
        pop.append(ind)
        if best is None or ind.f > best.f:
            best = ind
    assert best is not None
    return pop, best


def _mutate_rand1(rng: np.random.Generator, X: np.ndarray, F: float) -> np.ndarray:
    r1, r2, r3 = rng.choice(len(X), size=3, replace=False)
    return X[r1] + F * (X[r2] - X[r3])


def _mutate_best1(rng: np.random.Generator, X: np.ndarray, best: np.ndarray, F: float) -> np.ndarray:
    r1, r2 = rng.choice(len(X), size=2, replace=False)
    return best + F * (X[r1] - X[r2])


def _crossover_bin(rng: np.random.Generator, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
    D = target.shape[0]
    j_rand = rng.integers(0, D)
    trial = target.copy()
    mask = rng.random(D) < CR
    mask[j_rand] = True
    trial[mask] = mutant[mask]
    return trial


def differential_evolution(args: SimpleNamespace) -> Individual:
    rng = np.random.default_rng(args.seed)
    lo, hi = _bounds(args.rel_max, args.delay_max)

    pop, best = _init_population(rng, args.pop_size, lo, hi, args.method, args.dt)
    X = np.stack([ind.x for ind in pop], axis=0)
    F = float(args.F)
    CR = float(args.CR)

    start = time.time()
    last_prog = start

    if args.checkpoint_file:
        _atomic_save_json(best.as_dict() | dict(gen=0, timestamp=time.time()), args.checkpoint_file)

    for gen in range(1, args.gens + 1):
        for i in range(args.pop_size):
            xi = X[i]
            if args.strategy == "best1bin":
                mutant = _mutate_best1(rng, X, best.x, F)
            else:  # rand1bin
                mutant = _mutate_rand1(rng, X, F)

            trial = _crossover_bin(rng, xi, mutant, CR)
            trial = _clip_and_wrap(trial, lo, hi)

            f_trial = _eval_vec(trial, args.method, args.dt)
            if f_trial >= pop[i].f:  # maximize
                X[i] = trial
                pop[i] = Individual(x=trial, f=f_trial)
                if f_trial > best.f:
                    best = pop[i]
                    if args.checkpoint_file:
                        _atomic_save_json(best.as_dict() | dict(gen=gen, timestamp=time.time()), args.checkpoint_file)

        now = time.time()
        if args.progress and (now - last_prog >= args.progress_interval):
            elapsed = now - start
            print(f"[prog] gen={gen}/{args.gens} best={best.f:.4f}s "
                  f"spd={best.x[0]:.2f} az={best.x[1]:.3f} rel={best.x[2]:.2f} dly={best.x[3]:.2f} "
                  f"elapsed={elapsed:.1f}s", flush=True)
            last_prog = now

        if args.ckpt_every_gen and (gen % args.ckpt_every_gen == 0) and args.checkpoint_file:
            _atomic_save_json(best.as_dict() | dict(gen=gen, timestamp=time.time()), args.checkpoint_file)

    if args.checkpoint_file:
        _atomic_save_json(best.as_dict() | dict(gen=args.gens, timestamp=time.time()), args.checkpoint_file)
    return best


def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        method=cfg['method'], dt=cfg['dt'], rel_max=cfg['rel_max'], delay_max=cfg['delay_max'],
        pop_size=cfg['pop_size'], gens=cfg['gens'], F=cfg['F'], CR=cfg['CR'], strategy=cfg['strategy'],
        seed=cfg['seed'], progress=cfg['progress'], progress_interval=cfg['progress_interval'],
        verbose=cfg['verbose'], checkpoint_file=cfg['checkpoint_file'], ckpt_every_gen=cfg['ckpt_every_gen']
    )


def main():
    import sys
    use_cli = '--use-cli' in sys.argv
    if not use_cli:
        args = _build_args_via_config()
    else:
        import argparse
        ap = argparse.ArgumentParser(description="Differential Evolution for Problem 2 (maximize M1 occlusion)")
        ap.add_argument('--use-cli', action='store_true')
        ap.add_argument('--method', choices=['judge_caps', 'sampling'], default='judge_caps')
        ap.add_argument('--dt', type=float, default=0.01)
        ap.add_argument('--rel-max', type=float, default=66.0)
        ap.add_argument('--delay-max', type=float, default=20.0)
        ap.add_argument('--pop', dest='pop_size', type=int, default=60)
        ap.add_argument('--gens', type=int, default=400)
        ap.add_argument('--F', type=float, default=0.7)
        ap.add_argument('--CR', type=float, default=0.9)
        ap.add_argument('--strategy', choices=['rand1bin', 'best1bin'], default='rand1bin')
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--progress', action='store_true')
        ap.add_argument('--progress-interval', type=float, default=1.0)
        ap.add_argument('--verbose', action='store_true')
        ap.add_argument('--checkpoint-file', type=str, default='best_p2_de.json')
        ap.add_argument('--ckpt-every-gen', type=int, default=20)
        args = ap.parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t0 = time.time()
    best = differential_evolution(args)
    t1 = time.time()

    print("=== Differential Evolution Result (Problem 2) ===")
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

    print(f"Runtime: {t1 - t0:.2f} s | Generations: {args.gens} | Pop: {args.pop_size}")


if __name__ == "__main__":
    main()
