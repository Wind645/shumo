from __future__ import annotations
"""Unified optimization encoding for problems 2~5.

Encoding summary (decision vector x):
  Problem 2: [speed, azimuth, release_time, explode_delay]
  Problem 3: [speed, azimuth, (t1,d1), (t2,d2), ... up to bombs_count]
  Problem 4: For FY1,FY2,FY3 each 1 bomb ->
     [s1,a1,t1,d1, s2,a2,t2,d2, s3,a3,t3,d3]
  Problem 5: For FY1..FY5 each bombs_count bombs -> per drone block:
     [speed, azimuth, (t1,d1), ... (tB,dB)], concatenated for 5 drones.

Bounds (default):
  speed: [70,140]; azimuth: [-pi,pi]; deploy_time: [0,60]; explode_delay: [0.2,12]

Return helpers ensure bombs within min interval (1s). If violation -> invalid flag.
"""
from math import pi
from typing import List, Tuple, Dict, Any
from simcore.constants import DRONES_POS0, MIN_BOMB_INTERVAL, DRONE_SPEED_MIN, DRONE_SPEED_MAX

DeployBound = (0.0, 60.0)
DelayBound = (0.2, 12.0)
SpeedBound = (DRONE_SPEED_MIN, DRONE_SPEED_MAX)
AzimuthBound = (-pi, pi)

PROB4_DRONES = ["FY1", "FY2", "FY3"]
PROB5_DRONES = ["FY1", "FY2", "FY3", "FY4", "FY5"]

class DecodedDecision:
    def __init__(self, problem: int, drones_spec: List[Dict], invalid: bool=False):
        self.problem = problem
        self.drones_spec = drones_spec
        self.invalid = invalid

    def to_eval_kwargs(self) -> Dict[str, Any]:
        if self.problem == 2:
            d = self.drones_spec[0]
            b = d['bombs'][0]
            return dict(speed=d['speed'], azimuth=d.get('azimuth'), release_time=b['deploy_time'], explode_delay=b['explode_delay'])
        elif self.problem == 3:
            d = self.drones_spec[0]
            bombs = [(b['deploy_time'], b['explode_delay']) for b in d['bombs']]
            return dict(bombs=bombs, speed=d['speed'], azimuth=d.get('azimuth'))
        elif self.problem == 4:
            return dict(drones_spec=self.drones_spec)
        elif self.problem == 5:
            return dict(drones_spec=self.drones_spec)
        else:
            return {}

# ----------------------------------------------------------------------------
# Bounds construction
# ----------------------------------------------------------------------------

def bounds_for_problem(problem: int, bombs_count: int) -> List[Tuple[float,float]]:
    if problem == 2:
        return [SpeedBound, AzimuthBound, DeployBound, DelayBound]
    if problem == 3:
        out = [SpeedBound, AzimuthBound]
        for _ in range(bombs_count):
            out.append(DeployBound); out.append(DelayBound)
        return out
    if problem == 4:
        # 3 drones each 1 bomb
        out: List[Tuple[float,float]] = []
        for _ in PROB4_DRONES:
            out.extend([SpeedBound, AzimuthBound, DeployBound, DelayBound])
        return out
    if problem == 5:
        out = []
        for _ in PROB5_DRONES:
            out.extend([SpeedBound, AzimuthBound])
            for _b in range(bombs_count):
                out.append(DeployBound); out.append(DelayBound)
        return out
    raise ValueError(f"Unsupported problem {problem}")

# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------

def _validate_and_sort_bombs(bombs: List[Tuple[float,float]]) -> Tuple[List[Tuple[float,float]], bool]:
    bombs_sorted = sorted(bombs, key=lambda x: x[0])
    invalid = False
    for i in range(1, len(bombs_sorted)):
        if bombs_sorted[i][0] - bombs_sorted[i-1][0] < MIN_BOMB_INTERVAL - 1e-9:
            invalid = True
            break
    return bombs_sorted, invalid

def decode_vector(problem: int, x: List[float], bombs_count: int) -> DecodedDecision:
    xv = list(map(float, x))
    idx = 0
    drones_spec: List[Dict] = []
    invalid = False
    if problem == 2:
        speed = xv[idx]; az = xv[idx+1]; t_rel = xv[idx+2]; dly = xv[idx+3]
        bombs = [(t_rel, dly)]
        b_sorted, inv = _validate_and_sort_bombs(bombs)
        invalid = invalid or inv
        drones_spec.append(dict(pos0=DRONES_POS0['FY1'].tolist(), speed=speed, azimuth=az, bombs=[{'deploy_time':t,'explode_delay':d} for (t,d) in b_sorted]))
        return DecodedDecision(problem, drones_spec, invalid)
    if problem == 3:
        speed = xv[idx]; az = xv[idx+1]; idx += 2
        bombs: List[Tuple[float,float]] = []
        for _ in range(bombs_count):
            t = xv[idx]; d = xv[idx+1]; idx += 2
            bombs.append((t,d))
        bombs_sorted, inv = _validate_and_sort_bombs(bombs)
        invalid = invalid or inv
        drones_spec.append(dict(pos0=DRONES_POS0['FY1'].tolist(), speed=speed, azimuth=az, bombs=[{'deploy_time':t,'explode_delay':d} for (t,d) in bombs_sorted]))
        return DecodedDecision(problem, drones_spec, invalid)
    if problem == 4:
        for name in PROB4_DRONES:
            speed = xv[idx]; az = xv[idx+1]; t = xv[idx+2]; d = xv[idx+3]; idx += 4
            bombs_sorted, inv = _validate_and_sort_bombs([(t,d)])
            invalid = invalid or inv
            drones_spec.append(dict(pos0=DRONES_POS0[name].tolist(), speed=speed, azimuth=az, bombs=[{'deploy_time':t,'explode_delay':d} for (t,d) in bombs_sorted]))
        return DecodedDecision(problem, drones_spec, invalid)
    if problem == 5:
        for name in PROB5_DRONES:
            speed = xv[idx]; az = xv[idx+1]; idx += 2
            bombs: List[Tuple[float,float]] = []
            for _ in range(bombs_count):
                t = xv[idx]; d = xv[idx+1]; idx += 2
                bombs.append((t,d))
            bombs_sorted, inv = _validate_and_sort_bombs(bombs)
            invalid = invalid or inv
            drones_spec.append(dict(pos0=DRONES_POS0[name].tolist(), speed=speed, azimuth=az, bombs=[{'deploy_time':t,'explode_delay':d} for (t,d) in bombs_sorted]))
        return DecodedDecision(problem, drones_spec, invalid)
    raise ValueError(f"Unsupported problem {problem}")

__all__ = [
    'bounds_for_problem', 'decode_vector', 'DecodedDecision'
]
