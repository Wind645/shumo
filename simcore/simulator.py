from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

# Inlined constants (simcore.constants removed)
SMOKE_RADIUS_DEFAULT = 10.0
SMOKE_LIFETIME_DEFAULT = 20.0
SMOKE_DESCENT_DEFAULT = 3.0

from .entities import Missile, Drone, Bomb, SmokeCloud, Cylinder
# Legacy OcclusionEvaluator removed in favor of direct cap-only occlusion dispatcher.
# Side surface occlusion is intentionally ignored per new requirements.
from judges.cylinder_occlusion import cylinder_caps_fully_occluded

Vec3 = np.ndarray

@dataclass
class Simulator:
    missile: Missile
    drones: List[Drone]
    cylinder: Cylinder = field(default_factory=Cylinder)
    n_theta: int = 48
    n_h: int = 16
    n_cap_radial: int = 6
    check_caps: bool = True
    smoke_radius: float = SMOKE_RADIUS_DEFAULT
    smoke_lifetime: float = SMOKE_LIFETIME_DEFAULT
    smoke_descent: float = SMOKE_DESCENT_DEFAULT
    occlusion_method: str = "sampling"
    schedules: List[Tuple[int, float, float]] = field(default_factory=list)

    def run(self, dt: float, t_max: Optional[float] = None, verbose: bool = False) -> Dict:
        if t_max is None:
            t_max = self.missile.flight_time
        emitted = [False] * len(self.schedules)
        bombs: List[Bomb] = []
        clouds: List[SmokeCloud] = []
        # Removed OcclusionEvaluator construction (side surface no longer evaluated)
        # (n_theta, n_h, n_cap_radial, check_caps retained as legacy config but unused here)
        # evaluator = OcclusionEvaluator(
        #     self.cylinder, n_theta=self.n_theta, n_h=self.n_h,
        #     n_cap_radial=self.n_cap_radial, check_caps=self.check_caps,
        #     method=self.occlusion_method,
        # )
        t = 0.0
        occluded_time = 0.0
        timeline = []
        while t <= t_max + 1e-9:
            for idx, (di, deploy_time, explode_delay) in enumerate(self.schedules):
                if (not emitted[idx]) and (t >= deploy_time):
                    drone = self.drones[di]
                    pos = drone.position(deploy_time)
                    vel = drone.dir * drone.speed
                    bombs.append(Bomb(release_time=deploy_time, release_pos=pos, release_vel=vel, explode_delay=explode_delay))
                    emitted[idx] = True
                    if verbose:
                        print(f"[t={deploy_time:.2f}] Drone#{di} 投放烟幕弹")
            for b in bombs:
                te = b.explode_time()
                if (t >= te) and (not any(abs(c.start_time - te) < 1e-9 for c in clouds)):
                    center = b.position(te)
                    clouds.append(SmokeCloud(
                        start_time=te, center0=center,
                        radius=self.smoke_radius, life_time=self.smoke_lifetime, descent_speed=self.smoke_descent,
                    ))
                    if verbose:
                        print(f"[t={t:.2f}] 烟幕弹起爆，云团形成于 {center}")
            active_spheres: List[Tuple[np.ndarray, float]] = []
            for c in clouds:
                if c.active(t):
                    active_spheres.append((c.center(t), c.radius))
            V = self.missile.position(t)
            if len(active_spheres) == 0:
                occluded = False
                # For cap-only logic, when no spheres active we report both caps not covered.
                stats = dict(mode=self.occlusion_method, bottom=False, top=False, bottom_hits=[], top_hits=[])
            else:
                # Map legacy sampling methods to exact cap judge since side surface is ignored now.
                method = self.occlusion_method
                if method in ("sampling", "sampling_torch"):
                    method = "judge_caps"
                occluded, info = cylinder_caps_fully_occluded(
                    V,
                    active_spheres,
                    self.cylinder.C_base,
                    self.cylinder.r,
                    self.cylinder.h,
                    method=method,
                )
                stats = info
            if occluded:
                occluded_time += dt
            timeline.append(dict(
                t=float(t),
                occluded=bool(occluded),
                missile_pos=V.copy(),
                clouds=[(S.copy(), float(R)) for (S, R) in active_spheres],
                stats=stats,
            ))
            t += dt
        return dict(occluded_time=float(occluded_time), timeline=timeline, missile_flight_time=float(self.missile.flight_time))
