from __future__ import annotations
from typing import Iterable, List, Tuple, Optional
from .problems import evaluate_problem2, evaluate_problem3, evaluate_problem4

SPEED_MIN = 70.0
SPEED_MAX = 140.0

def _penalty_value(base: float, factor: float = 1e6) -> float:
    return base - factor

def objective_q2_vector(x: Iterable[float], *, method: str = "sampling", dt: float = 0.05) -> float:
    xs = list(map(float, x))
    if len(xs) != 4:
        return _penalty_value(0.0)
    speed, azim, t_rel, dly = xs
    if not (SPEED_MIN <= speed <= SPEED_MAX) or t_rel < 0.0 or dly <= 0.0:
        return _penalty_value(0.0)
    res = evaluate_problem2(speed=speed, azimuth=azim, release_time=t_rel, explode_delay=dly, occlusion_method=method, dt=dt)
    return -float(res["occluded_time"]["M1"])  # minimize

def objective_q3_vector(x: Iterable[float], *, method: str = "sampling", dt: float = 0.05) -> float:
    xs = list(map(float, x))
    if len(xs) != 8:
        return _penalty_value(0.0)
    speed, azim, t1, d1, t2, d2, t3, d3 = xs
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        return _penalty_value(0.0)
    ts_ds = [(t1, d1), (t2, d2), (t3, d3)]
    ts_ds.sort(key=lambda p: p[0])
    for i, (t, d) in enumerate(ts_ds):
        if t < 0.0 or d <= 0.0:
            return _penalty_value(0.0)
        if i > 0 and (t - ts_ds[i-1][0] < 1.0 - 1e-9):
            return _penalty_value(0.0)
    res = evaluate_problem3(bombs=ts_ds, speed=speed, azimuth=azim, dt=dt, occlusion_method=method)
    return -float(res["occluded_time"]["M1"])  # minimize

def objective_q4_vector(x: Iterable[float], *, method: str = "sampling", dt: float = 0.05) -> float:
    xs = list(map(float, x))
    if len(xs) != 12:
        return _penalty_value(0.0)
    s1,a1,t1,d1, s2,a2,t2,d2, s3,a3,t3,d3 = xs
    for s in (s1,s2,s3):
        if not (SPEED_MIN <= s <= SPEED_MAX):
            return _penalty_value(0.0)
    for t,d in ((t1,d1),(t2,d2),(t3,d3)):
        if t < 0.0 or d <= 0.0:
            return _penalty_value(0.0)
    drones = [
        {"pos0": [17800.0, 0.0, 1800.0], "speed": s1, "azimuth": a1, "bombs": [{"deploy_time": t1, "explode_delay": d1}]},
        {"pos0": [12000.0, 1400.0, 1400.0], "speed": s2, "azimuth": a2, "bombs": [{"deploy_time": t2, "explode_delay": d2}]},
        {"pos0": [6000.0, -3000.0, 700.0], "speed": s3, "azimuth": a3, "bombs": [{"deploy_time": t3, "explode_delay": d3}]},
    ]
    res = evaluate_problem4(drones_spec=drones, dt=dt, occlusion_method=method)
    return -float(res["occluded_time"]["M1"])  # minimize
