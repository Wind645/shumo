"""Problem presets constructing entities & schedules compatible with Simulator.

Provides a single function `build_problem(problem_id, strategy=None)` returning:
  missile: Missile (or list of Missile if multiple) -> we normalize to list
  drones:  list[Drone]
  schedules: list[(drone_index, deploy_time, explode_delay)]

`strategy` semantics (follow q3_sim original imperative version):
  problem 1: ignored; fixed drone 1 direction [1,0,0], speed 120, strategy [[1.5,3.6]]
  problem 2: strategy = (direction, speed, [[deploy, delay], ...])
  problem 3: strategy = (direction, speed, [[deploy, delay], ...]) expects 3 bombs
  problem 4: strategy = [ (dir1, speed1, strat1), (dir2, speed2, strat2), (dir3, speed3, strat3) ]
  problem 5: strategy = list of 5 tuples (dir, speed, strat) for drones 1..5

Each drone strategy entry: [deploy_time, explode_delay]
Converted to schedules = list of (drone_index, deploy_time, explode_delay)
where drone_index is 0-based index in returned drones list (Simulator expectation).
"""
from __future__ import annotations
from typing import List, Tuple, Sequence
import numpy as np
from .entities import Drone, Missile

Schedule = Tuple[int, float, float]


def _to_schedule(drone_idx: int, strat: Sequence[Sequence[float]]) -> List[Schedule]:
    out: List[Schedule] = []
    for item in strat:
        if len(item) != 2:
            raise ValueError("strategy item must be [deploy_time, explode_delay]")
        deploy, delay = float(item[0]), float(item[1])
        out.append((drone_idx, deploy, delay))
    return out


def build_problem(problem_id: int, strategy=None):
    drones: List[Drone] = []
    schedules: List[Schedule] = []

    if problem_id == 1:
        # Fixed single drone preset
        direction = np.array([1.0, 0.0, 0.0])
        speed = 120.0
        strat = [[1.5, 3.6]]
        drones.append(Drone(1, direction, speed, strat))
        schedules.extend(_to_schedule(0, strat))
        missiles = [Missile(1)]
    elif problem_id == 2:
        if strategy is None:
            raise ValueError("strategy required for problem 2: (direction, speed, strat)")
        direction, speed, strat = strategy
        drones.append(Drone(1, np.array(direction, dtype=float), float(speed), strat))
        schedules.extend(_to_schedule(0, strat))
        missiles = [Missile(1)]
    elif problem_id == 3:
        if strategy is None:
            raise ValueError("strategy required for problem 3: (direction, speed, strat-with-3-bombs)")
        direction, speed, strat = strategy
        if len(strat) < 3:
            raise ValueError("problem 3 expects at least 3 bombs in strategy")
        drones.append(Drone(1, np.array(direction, dtype=float), float(speed), strat))
        schedules.extend(_to_schedule(0, strat))
        missiles = [Missile(1)]
    elif problem_id == 4:
        if strategy is None:
            raise ValueError("strategy required for problem 4: list of 3 (dir,speed,strat)")
        if len(strategy) != 3:
            raise ValueError("problem 4 requires exactly 3 drone strategies")
        for idx, (direction, speed, strat) in enumerate(strategy):
            drones.append(Drone(idx+1, np.array(direction, dtype=float), float(speed), strat))
            schedules.extend(_to_schedule(idx, strat))
        missiles = [Missile(1)]
    elif problem_id == 5:
        if strategy is None:
            raise ValueError("strategy required for problem 5: list of 5 (dir,speed,strat)")
        if len(strategy) != 5:
            raise ValueError("problem 5 requires exactly 5 drone strategies")
        for idx, (direction, speed, strat) in enumerate(strategy):
            drones.append(Drone(idx+1, np.array(direction, dtype=float), float(speed), strat))
            schedules.extend(_to_schedule(idx, strat))
        missiles = [Missile(1), Missile(2), Missile(3)]
    else:
        raise ValueError(f"Unknown problem_id {problem_id}")

    return missiles, drones, schedules

__all__ = ["build_problem"]
