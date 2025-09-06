"""
Router-based occlusion evaluation helper for Problem 3 (single drone, multiple bombs).

This module provides a lightweight alternative to repeatedly calling the high-level
API evaluation path when optimizing Problem 3. It leverages the newly introduced
`judges.occlusion_time_router` to batch all circle–sphere occlusion queries
(bottom & top cylinder caps) into ONE vectorized judge pass (per cap) instead
of performing per-frame per-sphere loops.

Key Features
------------
1. Single entry point `evaluate_q3_router(...)` returning a result dict analogous to
   api.problems.evaluate_problem3 (so it can be swapped in minimal changes).
2. Supports both NumPy and Torch judge variants:
      judge='torch_exact'   -> vectorized_circle_fully_occluded_by_sphere_torch
      judge='torch_newton'  -> vectorized_circle_fully_occluded_by_sphere_torch_newton
      judge='torch_rough'   -> rough torch adapter (uses TorchRoughVectorizedOcclusionJudge)
      judge='numpy_exact'   -> vectorized_circle_fully_occluded_by_sphere
      judge='numpy_rough'   -> rough numpy adapter (RoughVectorizedOcclusionJudge)
   Or pass a custom `judge_fn` with the compatible signature returning an object
   with `.occluded` (boolean array / tensor).
3. Basic parameter validation & penalties for invalid decisions (optionally).
4. Can optionally return detailed router diagnostics (frame masks, etc.).

Usage
-----
    from abstract.occlusion_eval import evaluate_q3_router

    res = evaluate_q3_router(
        bombs=[(1.2, 5.0), (3.0, 4.0), (5.0, 6.0)],
        speed=110.0,
        azimuth=0.3,
        dt=0.02,
        judge='torch_exact',
        return_details=True,
    )
    print(res['occluded_time']['M1'], res['details'].keys())

Why Not Reuse Simulator.occlusion_method?
-----------------------------------------
The simulator currently performs per-frame occlusion via `cylinder_caps_fully_occluded`.
To obtain the full frame timeline we still call `sim.run()`, but then we discard its
internal per-frame occlusion flags and re-compute in a single vectorized pass. This
avoids modifying core simulator logic while enabling a clean performance path.

If future refactors add a "timeline only" mode to Simulator, this module can switch
to that to skip the intermediate per-frame judge cost entirely.

Return Structure
----------------
Matches the structure of evaluate_problem3:
{
  'occluded_time': {'M1': float_seconds},
  'total': float_seconds,
  'missile_flight_time': {'M1': float_seconds},
  'dt': dt_value,
  'details': {...}  # present only if return_details=True
}

"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Callable, Any
import math
import numpy as np

# Entities / Simulator
from simcore.entities import Missile, Drone, Cylinder
from simcore.simulator import Simulator

# Router + base judges
from judges import (
    occlusion_time_router,
    vectorized_circle_fully_occluded_by_sphere,
    vectorized_circle_fully_occluded_by_sphere_torch,
    vectorized_circle_fully_occluded_by_sphere_torch_newton,
    TorchRoughVectorizedOcclusionJudge,
    RoughVectorizedOcclusionJudge,
)

# ----------------------------------------------------------------------------------------------------------------------
# Internal judge adapters (rough variants do not expose the same functional interface as exact vectorized functions)
# ----------------------------------------------------------------------------------------------------------------------

def _numpy_rough_adapter(V: np.ndarray, C: np.ndarray, r: np.ndarray, S: np.ndarray, R: np.ndarray):
    """
    Adapter object to mimic the exact judge result interface using RoughVectorizedOcclusionJudge.
    """
    judge = RoughVectorizedOcclusionJudge()
    res = judge.judge_batch(V, C, r, S, R)
    return res  # already has .occluded, .f_min,... (NumPy arrays)


def _torch_rough_adapter(V, C, r, S, R):
    """
    Adapter for torch rough judge to match expected return (with .occluded).
    """
    # All inputs must be torch tensors (router enforces torch conversion before calling this)
    tr_judge = TorchRoughVectorizedOcclusionJudge(device=str(V.device), dtype=V.dtype)
    # The rough judge expects numpy-like arrays but handles torch tensors inside .judge_batch via _to_tensor
    res = tr_judge.judge_batch(V, C, r, S, R)
    return res  # has .occluded etc.


# Mapping from textual key to judge (callable)
_JUDGE_REGISTRY: Dict[str, Callable] = {
    "numpy_exact": vectorized_circle_fully_occluded_by_sphere,
    "numpy_rough": _numpy_rough_adapter,
    "torch_exact": vectorized_circle_fully_occluded_by_sphere_torch,
    "torch_newton": vectorized_circle_fully_occluded_by_sphere_torch_newton,
    "torch_rough": _torch_rough_adapter,
}

# ----------------------------------------------------------------------------------------------------------------------
# Utility: Build simulator for Problem 3 (single drone, multiple bombs)
# ----------------------------------------------------------------------------------------------------------------------

def _build_simulator_q3(
    bombs: List[Tuple[float, float]],
    speed: float,
    azimuth: float,
    dt: float,
) -> Simulator:
    """
    Construct a Simulator instance for Problem 3 with one missile (M1) and one drone (ID=1).

    bombs: list of (deploy_time, explode_delay)
    speed: drone speed (70~140 typical)
    azimuth: horizontal direction angle in radians
    """
    missile = Missile(1)  # M1
    direction = np.array([math.cos(azimuth), math.sin(azimuth), 0.0], dtype=np.float32)
    # Drone strategy not used by Simulator (it uses schedules); keep empty
    drone = Drone(1, direction=direction, speed=float(speed), strategy=[])
    schedules = [(0, float(t), float(d)) for (t, d) in bombs]
    sim = Simulator(
        missile=missile,
        drones=[drone],
        cylinder=Cylinder(),  # defaults
        schedules=schedules,
        occlusion_method="judge_caps",  # placeholder; router overrides computation
        dt=dt,
    )
    return sim


# ----------------------------------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------------------------------

def _validate_decision(
    bombs: List[Tuple[float, float]],
    speed: float,
    min_interval: float = 1.0,
) -> bool:
    """
    Basic feasibility checks (mirrors specs used elsewhere):
      - speed within [70, 140]
      - deploy times non-negative & delays > 0
      - chronological order (after sorting) with at least min_interval spacing
    """
    if not (70.0 <= speed <= 140.0):
        return False
    bombs_sorted = sorted(bombs, key=lambda x: x[0])
    last_t = None
    for t, d in bombs_sorted:
        if t < 0.0 or d <= 0.0:
            return False
        if last_t is not None and (t - last_t) < (min_interval - 1e-9):
            return False
        last_t = t
    return True


# ----------------------------------------------------------------------------------------------------------------------
# Public Evaluation Function
# ----------------------------------------------------------------------------------------------------------------------

def evaluate_q3_router(
    *,
    bombs: List[Tuple[float, float]],
    speed: float,
    azimuth: float,
    dt: float = 0.02,
    judge: str = "torch_exact",
    judge_fn: Optional[Callable] = None,
    vectorized: bool = True,
    return_details: bool = False,
    penalize_invalid: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate Problem 3 occlusion using the router-based single-pass vectorized judge.

    Parameters
    ----------
    bombs : List[(deploy_time, explode_delay)]
    speed : float
    azimuth : float
    dt : float
        Simulation step size.
    judge : str
        One of: 'torch_exact', 'torch_newton', 'torch_rough',
                'numpy_exact', 'numpy_rough'
        Ignored if judge_fn is provided.
    judge_fn : callable | None
        Custom judge with signature judge_fn(V,C,r,S,R) returning object w/ `.occluded`.
    vectorized : bool
        If False, falls back to simulator loop (mainly for debugging/perf comparison).
    return_details : bool
        Include router diagnostic masks.
    penalize_invalid : bool
        If True and decision invalid, returns a structure with large penalty values.
    verbose : bool
        Print minor progress info.

    Returns
    -------
    Dict with keys:
        occluded_time -> {'M1': seconds}
        total -> seconds
        missile_flight_time -> {'M1': seconds}
        dt -> float
        (optional) details -> router diagnostics
    """
    if judge_fn is None:
        if judge not in _JUDGE_REGISTRY:
            raise ValueError(f"Unknown judge key: {judge}")
        judge_fn = _JUDGE_REGISTRY[judge]

    valid = _validate_decision(bombs, speed)
    if (not valid) and penalize_invalid:
        penalty = 0.0  # Could also choose -inf semantics; keep 0 for clarity here
        result = {
            "occluded_time": {"M1": penalty},
            "total": penalty,
            "missile_flight_time": {"M1": 0.0},
            "dt": float(dt),
            "invalid": True,
        }
        if return_details:
            result["details"] = {"reason": "invalid_decision", "bombs": bombs, "speed": speed}
        return result

    bombs_sorted = sorted(bombs, key=lambda x: x[0])
    sim = _build_simulator_q3(bombs_sorted, speed=float(speed), azimuth=float(azimuth), dt=float(dt))

    # Run router
    occ_time, details = occlusion_time_router(
        sim,
        judge_fn=judge_fn,
        dt=float(dt),
        vectorized=vectorized,
        return_details=return_details,
        verbose=verbose,
    )

    res = {
        "occluded_time": {"M1": float(occ_time)},
        "total": float(occ_time),
        "missile_flight_time": {"M1": float(sim.missile.flight_time)},
        "dt": float(dt),
    }
    if return_details:
        res["details"] = details
    return res


# ----------------------------------------------------------------------------------------------------------------------
# CLI / Demonstration
# ----------------------------------------------------------------------------------------------------------------------

def _demo():
    bombs = [(1.5, 4.0), (3.0, 5.0), (5.0, 4.5)]
    speed = 120.0
    az = 0.2
    dt = 0.02
    for j in ["torch_exact", "torch_newton", "torch_rough", "numpy_exact", "numpy_rough"]:
        try:
            out = evaluate_q3_router(
                bombs=bombs,
                speed=speed,
                azimuth=az,
                dt=dt,
                judge=j,
                return_details=False,
                verbose=False,
            )
            print(f"[demo] judge={j:12s} occluded_time={out['occluded_time']['M1']:.4f}s flight={out['missile_flight_time']['M1']:.4f}s")
        except Exception as e:
            print(f"[demo] judge={j:12s} ERROR: {e}")


if __name__ == "__main__":
    _demo()


__all__ = [
    "evaluate_q3_router",
]
