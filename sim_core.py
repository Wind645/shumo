from __future__ import annotations
from typing import Dict
import numpy as np

# Backward-compatible shim that re-exports the new package API in smaller files.
from simcore import (
    G,
    SMOKE_RADIUS_DEFAULT,
    SMOKE_LIFETIME_DEFAULT,
    SMOKE_DESCENT_DEFAULT,
    C_BASE_DEFAULT,
    R_CYL_DEFAULT,
    H_CYL_DEFAULT,
    Missile,
    Drone,
    Cylinder,
    Simulator,
)

def run_problem1(
    dt: float = 0.05,
    n_theta: int = 48,
    n_h: int = 16,
    n_cap_radial: int = 6,
    check_caps: bool = True,
    occlusion_method: str = "sampling",
    verbose: bool = False,
) -> Dict:
    fake_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    m1 = Missile(pos0=np.array([20000.0, 0.0, 2000.0]), speed=300.0, target=fake_target)
    fy1_pos0 = np.array([17800.0, 0.0, 1800.0], dtype=np.float32)
    dir_h = fake_target.copy(); dir_h[2] = fy1_pos0[2]
    drone_dir = dir_h - fy1_pos0
    fy1 = Drone(pos0=fy1_pos0, direction=drone_dir, speed=120.0)
    cyl = Cylinder(C_base=C_BASE_DEFAULT.copy(), r=R_CYL_DEFAULT, h=H_CYL_DEFAULT)
    sim = Simulator(
        missile=m1,
        drones=[fy1],
        cylinder=cyl,
        n_theta=n_theta,
        n_h=n_h,
        n_cap_radial=n_cap_radial,
        check_caps=check_caps,
        smoke_radius=SMOKE_RADIUS_DEFAULT,
        smoke_lifetime=SMOKE_LIFETIME_DEFAULT,
        smoke_descent=SMOKE_DESCENT_DEFAULT,
        occlusion_method=occlusion_method,
        schedules=[(0, 1.5, 3.6)],
    )
    res = sim.run(dt=dt, t_max=None, verbose=verbose)
    if verbose:
        print(f"有效遮蔽时长: {res['occluded_time']:.5f} s / 导弹总飞行 {res['missile_flight_time']:.3f} s")
    return res

if __name__ == "__main__":
    out = run_problem1(dt=0.0005, verbose=True, occlusion_method="judge_caps")
    print(f"[Q1] 有效遮蔽总时长 ≈ {out['occluded_time']:.10f} s")
