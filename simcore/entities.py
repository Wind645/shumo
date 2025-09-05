from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import torch

from .constants import G, C_BASE_DEFAULT, R_CYL_DEFAULT, H_CYL_DEFAULT

Vec3 = torch.Tensor

def _as_vec3(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float).reshape(3)

@dataclass
class Missile:
    pos0: torch.Tensor
    speed: float
    target: torch.Tensor
    t0: float = 0.0

    def __post_init__(self):
        with torch.no_grad():
            d = _as_vec3(self.target) - _as_vec3(self.pos0)
            n = torch.norm(d)
            if n == 0:
                raise ValueError("Missile init: target equals pos0.")
            self.dir = d / n
            self.flight_time = n / float(self.speed)

    def position(self, t: float) -> torch.Tensor:
        with torch.no_grad():
            tau = max(0.0, t - self.t0)
            tau = min(tau, self.flight_time)
            return _as_vec3(self.pos0) + self.dir * self.speed * tau

@dataclass
class Drone:
    pos0: torch.Tensor
    direction: torch.Tensor
    speed: float
    t0: float = 0.0

    def __post_init__(self):
        with torch.no_grad():
            d = _as_vec3(self.direction)
            n = torch.norm(d)
            if n == 0:
                raise ValueError("Drone direction cannot be zero.")
            self.dir = d / n

    def position(self, t: float) -> torch.Tensor:
        with torch.no_grad():
            tau = max(0.0, t - self.t0)
            return _as_vec3(self.pos0) + self.dir * self.speed * tau

@dataclass
class Bomb:
    release_time: float
    release_pos: torch.Tensor
    release_vel: torch.Tensor
    explode_delay: float
    g: float = G

    def explode_time(self) -> float:
        return self.release_time + self.explode_delay

    def position(self, t: float) -> torch.Tensor:
        with torch.no_grad():
            dt = t - self.release_time
            if dt < 0:
                return _as_vec3(self.release_pos)
            a = torch.tensor([0.0, 0.0, -self.g], dtype=torch.float)
            return _as_vec3(self.release_pos) + _as_vec3(self.release_vel) * dt + 0.5 * a * (dt * dt)

@dataclass
class SmokeCloud:
    start_time: float
    center0: torch.Tensor
    radius: float = 10.0
    life_time: float = 20.0
    descent_speed: float = 3.0

    def active(self, t: float) -> bool:
        tau = t - self.start_time
        return (tau >= 0.0) and (tau <= self.life_time)

    def center(self, t: float) -> torch.Tensor:
        with torch.no_grad():
            tau = max(0.0, t - self.start_time)
            return _as_vec3(self.center0) + torch.tensor([0.0, 0.0, -self.descent_speed * tau], dtype=torch.float)

@dataclass
class Cylinder:
    C_base: torch.Tensor = field(default_factory=lambda: C_BASE_DEFAULT.clone())
    r: float = R_CYL_DEFAULT
    h: float = H_CYL_DEFAULT
