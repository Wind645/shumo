from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import torch

from .constants import SMOKE_DESCENT_DEFAULT, SMOKE_LIFETIME_DEFAULT, SMOKE_RADIUS_DEFAULT
from .entities import Missile, Drone, Bomb, SmokeCloud, Cylinder
from .occlusion import OcclusionEvaluator

Vec3 = torch.Tensor

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
        with torch.no_grad():
            if t_max is None:
                t_max = self.missile.flight_time
            emitted = [False] * len(self.schedules)
            bombs: List[Bomb] = []
            clouds: List[SmokeCloud] = []
            evaluator = OcclusionEvaluator(
                self.cylinder, n_theta=self.n_theta, n_h=self.n_h,
                n_cap_radial=self.n_cap_radial, check_caps=self.check_caps,
                method=self.occlusion_method,
            )
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
                active_spheres: List[Tuple[torch.Tensor, float]] = []
                for c in clouds:
                    if c.active(t):
                        active_spheres.append((c.center(t), c.radius))
                V = self.missile.position(t)
                if len(active_spheres) == 0:
                    occluded = False
                    stats = dict(total_points=0, blocked_points=0, uncovered_indices=[], mode=self.occlusion_method)
                else:
                    occluded, stats = evaluator.fully_occluded(V, active_spheres)
                if occluded:
                    occluded_time += dt
                timeline.append(dict(
                    t=float(t),
                    occluded=bool(occluded),
                    missile_pos=V.clone(),
                    clouds=[(S.clone(), float(R)) for (S, R) in active_spheres],
                    stats=stats,
                ))
                t += dt
            return dict(occluded_time=float(occluded_time), timeline=timeline, missile_flight_time=float(self.missile.flight_time))
