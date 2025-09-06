from __future__ import annotations
"""Entity definitions (analytic + stepwise update compatible).

Rewritten to follow the stepwise 可更新(update) 风格 (参考 `q3_sim.py`) while
remaining 100% backward compatible with existing analytic API relied on by
`simcore.simulator` 与 `simcore.vectorized_pack`:

Compatibility guarantees kept:
 - Missile.position(t), Drone.position(t) 依旧可用 (解析公式 + 常量速度直线飞行)
 - Bomb.position(t), Bomb.explode_time() 不变
 - SmokeCloud.active(t), SmokeCloud.center(t), attributes: radius / life_time / descent_speed
 - Cylinder unchanged

新增能力:
 - 基类 Object: 维护 all_objects, 提供 update(dt) & remove()
 - 每个实体增加 stepwise update 支持 (可在需要时做逐帧推进, 与解析接口并存)
 - Drone/Missile/Bomb/SmokeCloud 继承 Object (SmokeCloud 也能被逐帧下沉)

说明:
 - 解析接口是权威; stepwise 更新只更新当前内部 pos (与 position(t) 对应),
   不影响现有算法使用解析公式。
 - 若使用 stepwise update, 仍可随时调用 position(t_now) 获得解析结果进行校验。
"""
import numpy as np
from typing import List

Vec3 = np.ndarray

def _as_vec3(x) -> np.ndarray:
    return np.array(x, dtype=np.float32).reshape(3)


# ----------------------------------------------------------------------------
# Base Object with registry
# ----------------------------------------------------------------------------
class Object:
    all_objects: List['Object'] = []

    def __init__(self, pos=None):
        self.pos = _as_vec3(pos if pos is not None else np.zeros(3))
        Object.all_objects.append(self)
        self._removed = False

    def update(self, dt: float):  # override in subclasses
        pass

    def remove(self):
        if (not self._removed) and self in Object.all_objects:
            Object.all_objects.remove(self)
            self._removed = True


# ----------------------------------------------------------------------------
# Missile (constant speed straight line to origin, compatible analytic form)
# ----------------------------------------------------------------------------
class Missile(Object):
    def __init__(self, id: int, *, target: Vec3 = None, speed: float = 300.0):
        if id == 1:
            start = np.array([20000, 0, 2000], dtype=np.float32)
        elif id == 2:
            start = np.array([19000, 600, 2100], dtype=np.float32)
        else:  # id == 3
            start = np.array([18000, -600, 1900], dtype=np.float32)
        super().__init__(start)
        self.target = _as_vec3(target if target is not None else np.array([0, 0, 0], dtype=np.float32))
        self.speed = float(speed)
        self.t0 = 0.0
        d = self.target - self.pos
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Missile start equals target.")
        self.dir = d / n
        self.flight_time = n / self.speed
        # stepwise state
        self._age = 0.0
        self.occluded_time: float = 0.0  # 给批量遮挡统计使用

    # 解析位置
    def position(self, t: float) -> np.ndarray:
        tau = max(0.0, t - self.t0)
        tau = min(tau, self.flight_time)
        return _as_vec3(self.pos) + self.dir * self.speed * tau

    def update(self, dt: float):
        if self._age >= self.flight_time:
            return
        step = min(dt, self.flight_time - self._age)
        self.pos += self.dir * self.speed * step
        self._age += step
        # 不自动 remove; 由外部根据需要处理 (保持兼容不改变现有模拟器逻辑)


# ----------------------------------------------------------------------------
# Drone
# ----------------------------------------------------------------------------
class Drone(Object):
    def __init__(self, id: int, direction: Vec3, speed: float, strategy: list):
        poses = {
            1: np.array([17800, 0, 1800], dtype=np.float32),
            2: np.array([12000, 1400, 1400], dtype=np.float32),
            3: np.array([6000, -3000, 700], dtype=np.float32),
            4: np.array([11000, 2000, 1800], dtype=np.float32),
            5: np.array([13000, -2000, 1300], dtype=np.float32)
        }
        super().__init__(poses[id])
        self.direction = _as_vec3(direction)
        n = np.linalg.norm(self.direction)
        if n == 0:
            raise ValueError("Drone direction cannot be zero.")
        self.dir = self.direction / n
        self.speed = float(speed)
        # strategy: List[[deploy_time, explode_delay]] (与 q3_sim 对齐)
        self.strategy = list(strategy)
        self.t0 = 0.0
        self._age = 0.0

    # 解析位置
    def position(self, t: float) -> np.ndarray:
        tau = max(0.0, t - self.t0)
        return _as_vec3(self.pos) + self.dir * self.speed * tau

    def update(self, dt: float):
        self.pos += self.dir * self.speed * dt
        self._age += dt
        # 在 update 模式下可实时投放 (但保持独立, 不影响旧模拟器, 旧模拟器仍使用 schedules)
        for item in list(self.strategy):
            deploy_time, explode_delay = item
            if abs(deploy_time - self._age) < dt:
                # 投放炸弹 (release velocity = 当前无人机速度)
                Bomb(
                    release_time=self._age,  # 本地 age 时间
                    release_pos=self.pos.copy(),
                    release_vel=self.dir * self.speed,
                    explode_delay=explode_delay,
                )
                self.strategy.remove(item)


# ----------------------------------------------------------------------------
# Bomb (ballistic until explode -> becomes SmokeCloud)
# ----------------------------------------------------------------------------
class Bomb(Object):
    def __init__(self, release_time: float, release_pos: np.ndarray, release_vel: np.ndarray, explode_delay: float, g: float = 9.8):
        super().__init__(release_pos)
        self.release_time = float(release_time)
        self.release_pos = _as_vec3(release_pos)
        self.release_vel = _as_vec3(release_vel)
        self.explode_delay = float(explode_delay)
        self.g = float(g)
        self._age = 0.0  # stepwise age since release
        self._exploded = False

    def explode_time(self) -> float:
        return self.release_time + self.explode_delay

    # 解析位置
    def position(self, t: float) -> np.ndarray:
        dt = t - self.release_time
        if dt < 0:
            return _as_vec3(self.release_pos)
        a = np.array([0.0, 0.0, -self.g], dtype=np.float32)
        return _as_vec3(self.release_pos) + _as_vec3(self.release_vel) * dt + 0.5 * a * (dt * dt)

    def update(self, dt: float):
        # ballistic integration
        self._age += dt
        a = np.array([0.0, 0.0, -self.g], dtype=np.float32)
        self.pos += _as_vec3(self.release_vel) * dt + 0.5 * a * (dt * dt)
        self.release_vel = self.release_vel + a * dt
        if (not self._exploded) and self._age >= self.explode_delay:
            # transform into smoke cloud
            SmokeCloud(
                start_time=self.release_time + self.explode_delay,
                center0=self.pos.copy(),
            )
            self._exploded = True
            self.remove()


# ----------------------------------------------------------------------------
# SmokeCloud (downward drift for life_time)
# ----------------------------------------------------------------------------
class SmokeCloud(Object):
    def __init__(self, start_time: float, center0: np.ndarray, radius: float = 10.0, life_time: float = 20.0, descent_speed: float = 3.0):
        super().__init__(center0)
        self.start_time = float(start_time)
        self.center0 = _as_vec3(center0)
        self.radius = float(radius)
        self.life_time = float(life_time)
        self.descent_speed = float(descent_speed)
        self._age = 0.0

    def active(self, t: float) -> bool:
        tau = t - self.start_time
        return (tau >= 0.0) and (tau <= self.life_time)

    def center(self, t: float) -> np.ndarray:
        tau = max(0.0, t - self.start_time)
        return _as_vec3(self.center0) + np.array([0.0, 0.0, -self.descent_speed * tau], dtype=np.float32)

    def update(self, dt: float):
        self._age += dt
        if self._age <= self.life_time:
            self.pos[2] -= self.descent_speed * dt
        else:
            self.remove()


# ----------------------------------------------------------------------------
# Cylinder (unchanged)
# ----------------------------------------------------------------------------
class Cylinder:
    def __init__(self, C_base: np.ndarray = np.array([0.0, 200.0, 0.0], dtype=np.float32), r: float = 7.0, h: float = 10.0):
        self.C_base = _as_vec3(C_base)
        self.r = float(r)
        self.h = float(h)

__all__ = [
    'Object', 'Missile', 'Drone', 'Bomb', 'SmokeCloud', 'Cylinder'
]
