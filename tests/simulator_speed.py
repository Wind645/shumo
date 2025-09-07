#!/usr/bin/env python3
"""
Benchmark the performance of running randomly generated Simulator scenarios.

Spec:
  - Generate 10,000 random scenarios.
  - Each scenario uses dt = 0.01.
  - For each scenario:
        * Randomly pick a problem_id in {1,2,3,4,5}.
        * Generate required random strategy data (if needed).
        * Run until missiles are gone (or max_time reached).
        * Compute occlusion times (to include post‑processing cost).
  - Report:
        * Total wall time
        * Average seconds per simulation
        * Median seconds per simulation
        * Simulations per second

Adjust the constants (N_RUNS, MAX_TIME, etc.) if you want a lighter benchmark.

Run:
    python -m shumo.tests.simulator_speed
or:
    python shumo/tests/simulator_speed.py
"""

from __future__ import annotations

import os
import time
import random
import statistics
import math
from typing import List, Tuple
import numpy as np

from simulator import Simulator


# =========================
# Configuration constants
# =========================
N_RUNS: int = 1000
DT: float = 0.01
MAX_TIME: float = 120.0          # Safety cap per scenario
PROGRESS_INTERVAL: int = 10    # Print progress every N scenarios
MAX_BOMBS_PER_DRONE: int = 3     # Keep small for speed
MIN_RELEASE_TIME: float = 0.3
MAX_RELEASE_TIME: float = 15.0
MIN_FUSE: float = 0.5
MAX_FUSE: float = 9.0
MIN_SPEED: float = 60.0
MAX_SPEED: float = 180.0


# =========================
# Random generation helpers
# =========================
def random_direction_xy() -> np.ndarray:
    """Return a normalized 3D vector lying in the XY plane (z=0)."""
    theta = random.uniform(0, 2 * math.pi)
    return np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)


def random_bomb_strategy() -> List[List[float]]:
    """
    Produce a (possibly empty) list of scheduled bomb releases.
    Each entry: [release_time_since_start, fuse_delay]
    Ensures strictly increasing release times.
    """
    n = random.randint(0, MAX_BOMBS_PER_DRONE)
    if n == 0:
        return []
    releases = sorted(random.uniform(MIN_RELEASE_TIME, MAX_RELEASE_TIME) for _ in range(n))
    strategy: List[List[float]] = []
    for r in releases:
        fuse = random.uniform(MIN_FUSE, MAX_FUSE)
        strategy.append([float(r), float(fuse)])
    return strategy


def random_drone_spec():
    """
    Return a tuple (direction, speed, strategy) matching Simulator expectations.
    """
    direction = random_direction_xy()
    speed = random.uniform(MIN_SPEED, MAX_SPEED)
    strategy = random_bomb_strategy()
    return (direction, speed, strategy)


def random_strategy_for_problem(problem_id: int):
    """
    Build the 'strategy' argument expected by Simulator for problem_ids 2..5.
    For:
      2: single drone tuple
      3: single drone tuple
      4: triple of drone tuples
      5: quintuple of drone tuples
    """
    if problem_id in (2, 3):
        return random_drone_spec()
    elif problem_id == 4:
        return (random_drone_spec(), random_drone_spec(), random_drone_spec())
    elif problem_id == 5:
        return (
            random_drone_spec(),
            random_drone_spec(),
            random_drone_spec(),
            random_drone_spec(),
            random_drone_spec(),
        )
    else:
        # problem_id 1 does not require a strategy
        return None


def run_one(problem_id: int):
    """
    Run a single simulation with a random (or fixed) strategy and return elapsed wall time.
    """
    strategy = None
    if problem_id != 1:
        strategy = random_strategy_for_problem(problem_id)

    t0 = time.perf_counter()
    sim = Simulator(problem_id=problem_id, dt=DT, strategy=strategy)
    sim.run_until_end(max_time=MAX_TIME)
    sim.compute_batch_occlusions()
    t1 = time.perf_counter()
    return t1 - t0


def main():
    print(f"Running {N_RUNS} simulations (dt={DT}) ...")
    timings: List[float] = []
    for i in range(1, N_RUNS + 1):
        pid = random.randint(1, 5)
        elapsed = run_one(pid)
        timings.append(elapsed)
        if (i % PROGRESS_INTERVAL) == 0:
            avg = sum(timings) / len(timings)
            print(f"[{i}/{N_RUNS}] avg_per_sim={avg:.6f}s latest={elapsed:.6f}s")

    total = sum(timings)
    avg = total / len(timings)
    median = statistics.median(timings)
    sims_per_sec = len(timings) / total if total > 0 else float('inf')

    print("\n=== Benchmark Results ===")
    print(f"Total simulations : {len(timings)}")
    print(f"Total wall time   : {total:.3f} s")
    print(f"Average / sim     : {avg:.6f} s")
    print(f"Median  / sim     : {median:.6f} s")
    print(f"Simulations / sec : {sims_per_sec:.2f}")
    print("=========================")


if __name__ == "__main__":
    # Optional reproducibility; comment out for fully varying runs
    seed = os.environ.get("SIM_SPEED_SEED")
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
    main()
