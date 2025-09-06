from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np

from .constants import G, C_BASE_DEFAULT, R_CYL_DEFAULT, H_CYL_DEFAULT

Vec3 = np.ndarray

def _as_vec3(x) -> np.ndarray:
    return np.array(x, dtype=np.float32).reshape(3)

@dataclass
class Missile:
    pos0: np.ndarray
    speed: float
    target: np.ndarray
    t0: float = 0.0

    def __post_init__(self):
        d = _as_vec3(self.target) - _as_vec3(self.pos0)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Missile init: target equals pos0.")
        self.dir = d / n
        self.flight_time = n / float(self.speed)

    def position(self, t: float) -> np.ndarray:
        tau = max(0.0, t - self.t0)
        tau = min(tau, self.flight_time)
        return _as_vec3(self.pos0) + self.dir * self.speed * tau

@dataclass
class Drone:
    pos0: np.ndarray
    direction: np.ndarray
    speed: float
    t0: float = 0.0

    def __post_init__(self):
        d = _as_vec3(self.direction)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Drone direction cannot be zero.")
        self.dir = d / n

    def position(self, t: float) -> np.ndarray:
        tau = max(0.0, t - self.t0)
        return _as_vec3(self.pos0) + self.dir * self.speed * tau

@dataclass
class Bomb:
    release_time: float
    release_pos: np.ndarray
    release_vel: np.ndarray
    explode_delay: float
    g: float = G

    def explode_time(self) -> float:
        return self.release_time + self.explode_delay

    def position(self, t: float) -> np.ndarray:
        dt = t - self.release_time
        if dt < 0:
            return _as_vec3(self.release_pos)
        a = np.array([0.0, 0.0, -self.g], dtype=np.float32)
        return _as_vec3(self.release_pos) + _as_vec3(self.release_vel) * dt + 0.5 * a * (dt * dt)

@dataclass
class SmokeCloud:
    start_time: float
    center0: np.ndarray
    radius: float = 10.0
    life_time: float = 20.0
    descent_speed: float = 3.0

    def active(self, t: float) -> bool:
        tau = t - self.start_time
        return (tau >= 0.0) and (tau <= self.life_time)

    def center(self, t: float) -> np.ndarray:
        tau = max(0.0, t - self.start_time)
        return _as_vec3(self.center0) + np.array([0.0, 0.0, -self.descent_speed * tau], dtype=np.float32)

@dataclass
class Cylinder:
    C_base: np.ndarray = field(default_factory=lambda: C_BASE_DEFAULT.copy())
    r: float = R_CYL_DEFAULT
    h: float = H_CYL_DEFAULT
