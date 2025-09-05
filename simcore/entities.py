from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np

from .constants import G, C_BASE_DEFAULT, R_CYL_DEFAULT, H_CYL_DEFAULT

Vec3 = np.ndarray

def _as_vec3(x) -> Vec3:
    return np.asarray(x, dtype=float).reshape(3)

@dataclass
class Missile:
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
        tau = min(tau, self.flight_time)
        return _as_vec3(self.pos0) + self.dir * self.speed * tau

@dataclass
class Drone:
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
    release_time: float
    release_pos: Vec3
    release_vel: Vec3
    explode_delay: float
    g: float = G

    def explode_time(self) -> float:
        return self.release_time + self.explode_delay

    def position(self, t: float) -> Vec3:
        dt = t - self.release_time
        if dt < 0:
            return _as_vec3(self.release_pos)
        a = np.array([0.0, 0.0, -self.g], dtype=float)
        return _as_vec3(self.release_pos) + _as_vec3(self.release_vel) * dt + 0.5 * a * (dt * dt)

@dataclass
class SmokeCloud:
    start_time: float
    center0: Vec3
    radius: float = 10.0
    life_time: float = 20.0
    descent_speed: float = 3.0

    def active(self, t: float) -> bool:
        tau = t - self.start_time
        return (tau >= 0.0) and (tau <= self.life_time)

    def center(self, t: float) -> Vec3:
        tau = max(0.0, t - self.start_time)
        return _as_vec3(self.center0) + np.array([0.0, 0.0, -self.descent_speed * tau], dtype=float)

@dataclass
class Cylinder:
    C_base: Vec3 = field(default_factory=lambda: C_BASE_DEFAULT.copy())
    r: float = R_CYL_DEFAULT
    h: float = H_CYL_DEFAULT
