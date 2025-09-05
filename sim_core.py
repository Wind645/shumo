from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterable
import numpy as np

# 复用采样法的采样分布与线段-球相交测试
from baolijiefa import CylinderOcclusionJudge
# 新增：解析法（端面两圆）
from judge import OcclusionJudge

Vec3 = np.ndarray

# 常量
G = 9.8  # m/s^2
SMOKE_RADIUS_DEFAULT = 10.0
SMOKE_LIFETIME_DEFAULT = 20.0
SMOKE_DESCENT_DEFAULT = 3.0

# 圆柱（真目标）默认参数
C_BASE_DEFAULT = np.array([0.0, 200.0, 0.0], dtype=float)
R_CYL_DEFAULT = 7.0
H_CYL_DEFAULT = 10.0

def _as_vec3(x) -> Vec3:
    return np.asarray(x, dtype=float).reshape(3)

@dataclass
class Missile:
    """
    匀速直线飞行的导弹，指向目标点 target。
    """
    pos0: Vec3
    speed: float
    target: Vec3
    t0: float = 0.0

    def __post_init__(self):
        d = _as_vec3(self.target) - _as_vec3(self.pos0)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Missile init: target equals pos0.")
        self.dir = d / n
        self.flight_time = n / float(self.speed)

    def position(self, t: float) -> Vec3:
        tau = max(0.0, t - self.t0)
        tau = min(tau, self.flight_time)  # 到达目标后停止
        return _as_vec3(self.pos0) + self.dir * self.speed * tau

@dataclass
class Drone:
    """
    等高度匀速直线飞行的无人机，方向与速度固定。
    """
    pos0: Vec3
    direction: Vec3
    speed: float
    t0: float = 0.0

    def __post_init__(self):
        d = _as_vec3(self.direction)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Drone direction cannot be zero.")
        self.dir = d / n

    def position(self, t: float) -> Vec3:
        tau = max(0.0, t - self.t0)
        return _as_vec3(self.pos0) + self.dir * self.speed * tau

@dataclass
class Bomb:
    """
    烟幕弹：投放后仅受重力（z方向）作用的抛体运动，到时起爆。
    """
    release_time: float
    release_pos: Vec3
    release_vel: Vec3        # 一般取无人机投放瞬间速度
    explode_delay: float
    g: float = G

    def explode_time(self) -> float:
        return self.release_time + self.explode_delay

    def position(self, t: float) -> Vec3:
        """
        t < explode_time 时有效：p(t) = p0 + v0*dt + 0.5*a*dt^2, a=(0,0,-g)
        """
        dt = t - self.release_time
        if dt < 0:
            return _as_vec3(self.release_pos)
        a = np.array([0.0, 0.0, -self.g], dtype=float)
        return _as_vec3(self.release_pos) + _as_vec3(self.release_vel) * dt + 0.5 * a * (dt * dt)

@dataclass
class SmokeCloud:
    """
    烟幕云团：起爆后瞬时形成球，匀速下沉，有限寿命。
    """
    start_time: float
    center0: Vec3
    radius: float = SMOKE_RADIUS_DEFAULT
    life_time: float = SMOKE_LIFETIME_DEFAULT
    descent_speed: float = SMOKE_DESCENT_DEFAULT

    def active(self, t: float) -> bool:
        tau = t - self.start_time
        return (tau >= 0.0) and (tau <= self.life_time)

    def center(self, t: float) -> Vec3:
        tau = max(0.0, t - self.start_time)
        # 匀速下沉（-z方向）
        return _as_vec3(self.center0) + np.array([0.0, 0.0, -self.descent_speed * tau], dtype=float)

@dataclass
class Cylinder:
    """
    圆柱体（真目标）：底面圆心 C_base，半径 r，高 h（沿 z）。
    """
    C_base: Vec3 = field(default_factory=lambda: C_BASE_DEFAULT.copy())
    r: float = R_CYL_DEFAULT
    h: float = H_CYL_DEFAULT

class OcclusionEvaluator:
    """
    遮蔽评估：从观察点 V 看，多球体(烟幕云团)对圆柱是否“完全遮蔽”。
    method:
      - 'sampling'   使用 CylinderOcclusionJudge 的采样法（支持多球并集）
      - 'judge_caps' 使用解析法：对底/顶端面两圆，若分别被“任一球”完全遮蔽，则认为圆柱被遮蔽
                    （与 compare.py/new_compare.py 的解析方案一致，偏保守，不考虑侧面与多球拼接遮蔽）
    """
    def __init__(self, cyl: Cylinder,
                 n_theta: int = 48, n_h: int = 16, n_cap_radial: int = 6, check_caps: bool = True,
                 method: str = "sampling"):
        self.cyl = cyl
        self.n_theta = int(max(4, n_theta))
        self.n_h = int(max(2, n_h))
        self.n_cap_radial = int(max(1, n_cap_radial))
        self.check_caps = bool(check_caps)
        self.method = method

        # 仅 sampling 模式下需要采样点
        self._pts = None
        if self.method == "sampling":
            self._sampler = CylinderOcclusionJudge(
                V=np.array([0.0, 0.0, 1.0]),  # 占位
                S=np.array([0.0, 0.0, 2.0]),  # 占位
                R=1.0,
                C_base=self.cyl.C_base,
                r_cyl=self.cyl.r,
                h_cyl=self.cyl.h,
                n_theta=self.n_theta,
                n_h=self.n_h,
                n_cap_radial=self.n_cap_radial,
                check_caps=self.check_caps
            )
            self._pts = self._sampler._sample_cylinder_points()

    @staticmethod
    def _hit_any_sphere(V: Vec3, P: Vec3, spheres: Iterable[Tuple[Vec3, float]]) -> bool:
        for S, R in spheres:
            if CylinderOcclusionJudge._segment_intersects_sphere(V, P, S, R):
                return True
        return False

    def _judge_cap_by_union(self, V: Vec3, spheres: Iterable[Tuple[Vec3, float]], C_cap: Vec3, r_cap: float) -> Tuple[bool, List[int]]:
        """
        给定观察点 V 与端面圆 (C_cap, r_cap)，判断是否被“任一球”完全遮蔽。
        OcclusionJudge 要求圆位于 z=0 平面，因此做平移：
          V' = V - (0,0,C_cap.z), S' = S - (0,0,C_cap.z), C' = (C_cap.x, C_cap.y, 0)。
        返回 (ok, hit_indices)。
        """
        Vp = np.array([V[0], V[1], V[2] - C_cap[2]])
        C_flat = np.array([C_cap[0], C_cap[1], 0.0])
        hits: List[int] = []
        for k, (S, R) in enumerate(spheres):
            Sp = np.array([S[0], S[1], S[2] - C_cap[2]])
            j = OcclusionJudge(Vp, C_flat, r_cap, Sp, R)
            res = j.is_fully_occluded()
            if bool(res.occluded):
                hits.append(k)
        return (len(hits) > 0), hits

    def _fully_occluded_sampling(self, V: Vec3, spheres: Iterable[Tuple[Vec3, float]]):
        pts = self._pts
        total = int(pts.shape[0]) if pts is not None else 0
        blocked = 0
        uncovered = []
        for i, P in enumerate(pts):
            if self._hit_any_sphere(V, P, spheres):
                blocked += 1
            else:
                uncovered.append(i)
        return (blocked == total), dict(total_points=total, blocked_points=blocked, uncovered_indices=uncovered[:16])

    def _fully_occluded_judge_caps(self, V: Vec3, spheres: Iterable[Tuple[Vec3, float]]):
        spheres = list(spheres)
        if len(spheres) == 0:
            return False, dict(mode="judge_caps", bottom=False, top=False, bottom_hits=[], top_hits=[])

        Cb = self.cyl.C_base
        Ct = self.cyl.C_base + np.array([0.0, 0.0, self.cyl.h])
        bottom_ok, bottom_hits = self._judge_cap_by_union(V, spheres, Cb, self.cyl.r)
        top_ok, top_hits = self._judge_cap_by_union(V, spheres, Ct, self.cyl.r)
        ok = bool(bottom_ok and top_ok)
        return ok, dict(mode="judge_caps", bottom=bool(bottom_ok), top=bool(top_ok),
                        bottom_hits=bottom_hits[:8], top_hits=top_hits[:8])

    def fully_occluded(self, V: Vec3, spheres: Iterable[Tuple[Vec3, float]]):
        if self.method == "judge_caps":
            return self._fully_occluded_judge_caps(V, spheres)
        # 默认采样法
        return self._fully_occluded_sampling(V, spheres)

@dataclass
class Simulator:
    """
    统一仿真器：推进时间轴、生成烟幕云团、统计对真目标的遮蔽时长。
    """
    missile: Missile
    drones: List[Drone]
    cylinder: Cylinder = field(default_factory=Cylinder)
    # 采样遮蔽参数
    n_theta: int = 48
    n_h: int = 16
    n_cap_radial: int = 6
    check_caps: bool = True
    # 烟幕参数
    smoke_radius: float = SMOKE_RADIUS_DEFAULT
    smoke_lifetime: float = SMOKE_LIFETIME_DEFAULT
    smoke_descent: float = SMOKE_DESCENT_DEFAULT
    # 遮蔽判定方法：'sampling' 或 'judge_caps'
    occlusion_method: str = "sampling"

    # 投弹计划：每个元素为 (drone_index, deploy_time, explode_delay)
    schedules: List[Tuple[int, float, float]] = field(default_factory=list)

    def run(self, dt: float, t_max: Optional[float] = None, verbose: bool = False) -> Dict:
        """
        推进仿真并返回统计：
          - occluded_time: 有效遮蔽累计时长（秒）
          - timeline: 每步记录字典数组（t, occluded, missile_pos, clouds=[(center,radius), ...]）
        """
        # 时间终止：默认到导弹抵达假目标为止
        if t_max is None:
            t_max = self.missile.flight_time

        # 投弹触发状态
        emitted = [False] * len(self.schedules)
        bombs: List[Bomb] = []
        clouds: List[SmokeCloud] = []

        evaluator = OcclusionEvaluator(
            self.cylinder, n_theta=self.n_theta, n_h=self.n_h,
            n_cap_radial=self.n_cap_radial, check_caps=self.check_caps,
            method=self.occlusion_method
        )

        t = 0.0
        occluded_time = 0.0
        timeline = []

        while t <= t_max + 1e-9:
            # 1) 触发投弹
            for idx, (di, deploy_time, explode_delay) in enumerate(self.schedules):
                if (not emitted[idx]) and (t >= deploy_time):
                    drone = self.drones[di]
                    pos = drone.position(deploy_time)
                    vel = drone.dir * drone.speed  # 与无人机同水平速度（等高度飞行）
                    bomb = Bomb(release_time=deploy_time, release_pos=pos, release_vel=vel, explode_delay=explode_delay)
                    bombs.append(bomb)
                    emitted[idx] = True
                    if verbose:
                        print(f"[t={deploy_time:.2f}] Drone#{di} 投放烟幕弹")

            # 2) 生成/维护云团
            for b in bombs:
                te = b.explode_time()
                if (t >= te) and (not any(abs(c.start_time - te) < 1e-9 for c in clouds)):
                    # 起爆，生成云团
                    center = b.position(te)
                    clouds.append(SmokeCloud(
                        start_time=te, center0=center,
                        radius=self.smoke_radius, life_time=self.smoke_lifetime, descent_speed=self.smoke_descent
                    ))
                    if verbose:
                        print(f"[t={t:.2f}] 烟幕弹起爆，云团形成于 {center}")

            # 3) 计算当前生效云团
            active_spheres: List[Tuple[Vec3, float]] = []
            for c in clouds:
                if c.active(t):
                    active_spheres.append((c.center(t), c.radius))

            # 4) 遮蔽判定
            V = self.missile.position(t)
            if len(active_spheres) == 0:
                occluded = False
                stats = dict(total_points=0, blocked_points=0, uncovered_indices=[], mode=self.occlusion_method)
            else:
                occluded, stats = evaluator.fully_occluded(V, active_spheres)

            # 5) 统计
            if occluded:
                occluded_time += dt

            timeline.append(dict(
                t=float(t),
                occluded=bool(occluded),
                missile_pos=V.copy(),
                clouds=[(S.copy(), float(R)) for (S, R) in active_spheres],
                stats=stats
            ))

            t += dt

        return dict(
            occluded_time=float(occluded_time),
            timeline=timeline,
            missile_flight_time=float(self.missile.flight_time)
        )

# --------- 题目问题1：一键复现函数 ---------

def run_problem1(
    dt: float = 0.05,
    n_theta: int = 48, n_h: int = 16, n_cap_radial: int = 6, check_caps: bool = True,
    occlusion_method: str = "sampling",
    verbose: bool = False
) -> Dict:
    """
    问题1：FY1 以 120 m/s 朝向假目标方向等高度直飞；1.5s 投放 1 枚烟幕弹，3.6s 后起爆。
    计算对 M1 的有效遮蔽总时长（导弹在飞行期间内）。
    occlusion_method: 'sampling'（默认）或 'judge_caps'（解析法，仅端面两圆，偏保守）。
    """
    fake_target = np.array([0.0, 0.0, 0.0], dtype=float)

    # 导弹 M1
    m1 = Missile(
        pos0=np.array([20000.0, 0.0, 2000.0], dtype=float),
        speed=300.0,
        target=fake_target
    )

    # 无人机 FY1：朝向假目标，但保持等高度（方向只在水平面取向）
    fy1_pos0 = np.array([17800.0, 0.0, 1800.0], dtype=float)
    dir_h = fake_target.copy()
    dir_h[2] = fy1_pos0[2]  # 等高度投影
    drone_dir = dir_h - fy1_pos0
    fy1 = Drone(
        pos0=fy1_pos0,
        direction=drone_dir,
        speed=120.0
    )

    cyl = Cylinder(C_base=C_BASE_DEFAULT.copy(), r=R_CYL_DEFAULT, h=H_CYL_DEFAULT)

    sim = Simulator(
        missile=m1,
        drones=[fy1],
        cylinder=cyl,
        n_theta=n_theta, n_h=n_h, n_cap_radial=n_cap_radial, check_caps=check_caps,
        smoke_radius=SMOKE_RADIUS_DEFAULT,
        smoke_lifetime=SMOKE_LIFETIME_DEFAULT,
        smoke_descent=SMOKE_DESCENT_DEFAULT,
        occlusion_method=occlusion_method,
        schedules=[(0, 1.5, 3.6)]  # (drone_index, deploy_time, explode_delay)
    )

    res = sim.run(dt=dt, t_max=None, verbose=verbose)
    if verbose:
        print(f"有效遮蔽时长: {res['occluded_time']:.3f} s / 导弹总飞行 {res['missile_flight_time']:.3f} s")
    return res

if __name__ == "__main__":
    # 简单自测：输出问题1的有效遮蔽时长
    out = run_problem1(dt=0.000005, verbose=True, occlusion_method="judge_caps")
    print(f"[Q1] 有效遮蔽总时长 ≈ {out['occluded_time']:.3f} s")
