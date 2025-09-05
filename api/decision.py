from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional, Union
import numpy as np
from simcore import (
    Missile, Drone, Cylinder, Simulator, OcclusionEvaluator,
    C_BASE_DEFAULT, R_CYL_DEFAULT, H_CYL_DEFAULT,
    SMOKE_DESCENT_DEFAULT, SMOKE_LIFETIME_DEFAULT, SMOKE_RADIUS_DEFAULT,
)

Vec3 = np.ndarray
FAKE_TARGET = np.array([0.0, 0.0, 0.0], dtype=float)
SPEED_MIN = 70.0
SPEED_MAX = 140.0

def _as_vec3(x) -> Vec3:
    return np.asarray(x, dtype=float).reshape(3)

def _norm(v: Vec3) -> Vec3:
    v = _as_vec3(v)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n

def azimuth_to_dir(azim_rad: float) -> Vec3:
    return np.array([np.cos(azim_rad), np.sin(azim_rad), 0.0], dtype=float)

def _validate_bombs(bombs: List[Dict]):
    times = sorted(float(b["deploy_time"]) for b in bombs)
    for i in range(1, len(times)):
        if times[i] - times[i-1] < 1.0 - 1e-9:
            raise ValueError(f"同一无人机投弹间隔不足1秒: {times[i-1]} -> {times[i]}")

def build_drones_and_schedules(decision: Dict) -> Tuple[List[Drone], List[Tuple[int, float, float]]]:
    drones: List[Drone] = []
    schedules: List[Tuple[int, float, float]] = []
    if "drones" not in decision or not isinstance(decision["drones"], list):
        raise ValueError("decision['drones'] 必须为列表")
    for di, d in enumerate(decision["drones"]):
        pos0 = _as_vec3(d["pos0"])
        v = float(d["speed"])
        if not (SPEED_MIN - 1e-9 <= v <= SPEED_MAX + 1e-9):
            raise ValueError(f"无人机速度超范围[70,140]: {v}")
        if "direction" in d:
            dir_vec = _norm(d["direction"])
        elif "azimuth" in d:
            dir_vec = _norm(azimuth_to_dir(float(d["azimuth"])))
        elif d.get("aim_fake_target", False):
            target_h = FAKE_TARGET.copy(); target_h[2] = pos0[2]
            dir_vec = _norm(target_h - pos0)
        else:
            raise ValueError("必须提供 direction 或 azimuth 或 aim_fake_target 之一作为航向")
        drones.append(Drone(pos0=pos0, direction=dir_vec, speed=v))
        bombs = d.get("bombs", [])
        _validate_bombs(bombs)
        for b in bombs:
            schedules.append((di, float(b["deploy_time"]), float(b["explode_delay"])) )
    schedules.sort(key=lambda x: x[1])
    return drones, schedules

MISSILES_DEF = {
    "M1": dict(pos0=np.array([20000.0, 0.0, 2000.0]), speed=300.0, target=FAKE_TARGET),
    "M2": dict(pos0=np.array([19000.0, 600.0, 2100.0]), speed=300.0, target=FAKE_TARGET),
    "M3": dict(pos0=np.array([18000.0, -600.0, 1900.0]), speed=300.0, target=FAKE_TARGET),
}

def _build_missiles(which: Union[str, List[str]]) -> Dict[str, Missile]:
    if isinstance(which, str):
        keys = ["M1", "M2", "M3"] if which == "M1M2M3" else [which]
    else:
        keys = list(which)
    res: Dict[str, Missile] = {}
    for k in keys:
        spec = MISSILES_DEF[k]
        res[k] = Missile(pos0=spec["pos0"], speed=spec["speed"], target=spec["target"])
    return res

def _max_flight_time(missiles: Dict[str, Missile]) -> float:
    return max(m.flight_time for m in missiles.values())

def _eval_occlusion_over_timeline(missiles: Dict[str, Missile], timeline: List[Dict], cyl: Cylinder, method: str) -> Dict[str, float]:
    evaluator = OcclusionEvaluator(cyl, method=method)
    occluded_time: Dict[str, float] = {k: 0.0 for k in missiles.keys()}
    dt = float(timeline[1]["t"] - timeline[0]["t"]) if len(timeline) >= 2 else 0.05
    for rec in timeline:
        t = float(rec["t"])
        spheres = [(S, R) for (S, R) in rec.get("clouds", [])]
        for name, m in missiles.items():
            if t > m.flight_time + 1e-12:
                continue
            if not spheres:
                continue
            V = m.position(t)
            ok, _ = evaluator.fully_occluded(V, spheres)
            if ok:
                occluded_time[name] += dt
    return occluded_time

def simulate_with_decision(
    decision: Dict,
    which: Union[str, List[str]] = "M1",
    *, dt: float = 0.05, occlusion_method: str = "judge_caps", verbose: bool = False, return_timeline: bool = False,
) -> Dict:
    drones, schedules = build_drones_and_schedules(decision)
    missiles = _build_missiles(which)
    T_max = _max_flight_time(missiles)
    first_missile: Missile = list(missiles.values())[0]
    cyl = Cylinder(C_base=C_BASE_DEFAULT.copy(), r=R_CYL_DEFAULT, h=H_CYL_DEFAULT)
    sim = Simulator(
        missile=first_missile,
        drones=drones,
        cylinder=cyl,
        n_theta=48, n_h=16, n_cap_radial=6, check_caps=True,
        smoke_radius=SMOKE_RADIUS_DEFAULT, smoke_lifetime=SMOKE_LIFETIME_DEFAULT, smoke_descent=SMOKE_DESCENT_DEFAULT,
        occlusion_method=occlusion_method,
        schedules=schedules,
    )
    sim_out = sim.run(dt=dt, t_max=T_max, verbose=verbose)
    occluded_time = _eval_occlusion_over_timeline(missiles, sim_out["timeline"], cyl, occlusion_method)
    result = {
        "occluded_time": occluded_time,
        "total": float(sum(occluded_time.values())),
        "missile_flight_time": {k: float(v.flight_time) for k, v in missiles.items()},
        "dt": float(dt),
    }
    if return_timeline:
        result["timeline"] = sim_out["timeline"]
    return result
