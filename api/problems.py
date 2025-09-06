from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
from .decision import simulate_with_decision
from simcore.vectorized_pack import package_timeline_single_missile, occluded_time_vectorized_sampling
from simcore import Missile, Drone, C_BASE_DEFAULT, R_CYL_DEFAULT, H_CYL_DEFAULT, SMOKE_RADIUS_DEFAULT, SMOKE_LIFETIME_DEFAULT, SMOKE_DESCENT_DEFAULT
from .decision import _build_missiles, build_drones_and_schedules

def _simulate_any(decision, which: str, dt: float, occlusion_method: str):
    if occlusion_method == 'vectorized_sampling':
        # Currently only supports single missile names (M1/M2/M3)
        missiles = _build_missiles(which)
        if len(missiles) != 1:
            raise ValueError("vectorized_sampling 目前仅支持单枚导弹情形。")
        (name, missile) = list(missiles.items())[0]
        drones, schedules = build_drones_and_schedules(decision)
        packed = package_timeline_single_missile(missile, drones, schedules, dt=dt)
        occ = occluded_time_vectorized_sampling(packed)
        return {
            'occluded_time': {name: occ},
            'total': float(occ),
            'missile_flight_time': {name: float(missile.flight_time)},
            'dt': float(dt),
        }
    else:
        return simulate_with_decision(decision, which=which, dt=dt, occlusion_method=occlusion_method)

def evaluate_problem1(
    *, dt: float = 0.05, occlusion_method: str = 'sampling'
) -> Dict:
    """问题1: FY1 以 120 m/s 朝假目标方向, 1.5 s 后投放, 3.6 s 后起爆, 求对 M1 遮蔽时长。"""
    fy1 = {
        'pos0': [17800.0, 0.0, 1800.0],
        'speed': 120.0,
        'bombs': [{ 'deploy_time': 1.5, 'explode_delay': 3.6 }],
        'aim_fake_target': True,
    }
    decision = {'drones': [fy1]}
    return _simulate_any(decision, which='M1', dt=dt, occlusion_method=occlusion_method)

def evaluate_problem2(
    speed: float,
    direction: Optional[Iterable[float]] = None,
    azimuth: Optional[float] = None,
    explode_delay: float = 3.6,
    release_time: float = 1.5,
    occlusion_method: str = "sampling",
    dt: float = 0.05,
) -> Dict:
    fy1 = {
        "pos0": [17800.0, 0.0, 1800.0],
        "speed": float(speed),
        "bombs": [{"deploy_time": float(release_time), "explode_delay": float(explode_delay)}],
    }
    if direction is not None:
        fy1["direction"] = list(direction)
    elif azimuth is not None:
        fy1["azimuth"] = float(azimuth)
    else:
        fy1["aim_fake_target"] = True
    decision = {"drones": [fy1]}
    return _simulate_any(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

def evaluate_problem3(
    bombs: List[Tuple[float, float]],
    speed: float = 120.0,
    azimuth: Optional[float] = None,
    direction: Optional[Iterable[float]] = None,
    dt: float = 0.05,
    occlusion_method: str = "sampling",
) -> Dict:
    fy1 = {
        "pos0": [17800.0, 0.0, 1800.0],
        "speed": float(speed),
        "bombs": [{"deploy_time": float(t), "explode_delay": float(d)} for (t, d) in bombs],
    }
    if direction is not None:
        fy1["direction"] = list(direction)
    elif azimuth is not None:
        fy1["azimuth"] = float(azimuth)
    else:
        fy1["aim_fake_target"] = True
    decision = {"drones": [fy1]}
    return _simulate_any(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

def evaluate_problem4(drones_spec: List[Dict], dt: float = 0.05, occlusion_method: str = "sampling") -> Dict:
    decision = {"drones": drones_spec}
    return _simulate_any(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

def evaluate_problem5(drones_spec: List[Dict], dt: float = 0.05, occlusion_method: str = "sampling") -> Dict:
    decision = {"drones": drones_spec}
    return _simulate_any(decision, which="M1M2M3", dt=dt, occlusion_method=occlusion_method)
