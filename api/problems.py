from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
from .decision import simulate_with_decision

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
    return simulate_with_decision(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

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
    return simulate_with_decision(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

def evaluate_problem4(drones_spec: List[Dict], dt: float = 0.05, occlusion_method: str = "sampling") -> Dict:
    decision = {"drones": drones_spec}
    return simulate_with_decision(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

def evaluate_problem5(drones_spec: List[Dict], dt: float = 0.05, occlusion_method: str = "sampling") -> Dict:
    decision = {"drones": drones_spec}
    return simulate_with_decision(decision, which="M1M2M3", dt=dt, occlusion_method=occlusion_method)
