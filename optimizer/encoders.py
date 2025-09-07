"""
Encoders translating a flat particle vector into a Simulator strategy
for problems 2–5.

Each encoder exposes:
  - dim: int
  - encode(position: Sequence[float]) -> strategy
  - initial_position() -> np.ndarray in [-1,1]^dim

Returned strategy shapes (expected by Simulator):

Problem 2 (single drone, 1 bomb):
  (direction_vec: np.ndarray(3,),
   speed: float,
   [[release_time, fuse_delay]])

Problem 3 (single drone, 3 bombs):
  (direction_vec, speed, [
     [t1, fuse1],
     [t2, fuse2],
     [t3, fuse3],
  ])

Problem 4 (3 drones, 1 bomb each):
  (
    (dir1, speed1, [[t1,f1]]),
    (dir2, speed2, [[t2,f2]]),
    (dir3, speed3, [[t3,f3]]),
  )

Problem 5 (up to 5 drones, each up to 3 bombs with activation flags):
  (
    (dirA, speedA, [[t1,f1], [t2,f2], ...]),
    ...
    (dirE, speedE, [...])
  )

Utility transforms:
  sigmoid(x)  -> (0,1)
  positive(x) -> smooth >=0 (softplus)
  clamp(v, lo, hi)

Speed mapping:
  speed = 70 + sigmoid(raw_speed)*70  in [70,140]

Fuse delays clamped to [0.1, 15].

Use:
  from optimizer.encoders import ENCODERS, get_encoder
"""
from __future__ import annotations

import math
from typing import Sequence, Dict
import numpy as np

__all__ = [
    "sigmoid",
    "positive",
    "clamp",
    "StrategyEncoder",
    "Problem2Encoder",
    "Problem3Encoder",
    "Problem4Encoder",
    "Problem5Encoder",
    "ENCODERS",
    "get_encoder",
]

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def positive(x: float) -> float:
    # Softplus for smooth non-negative mapping
    return math.log1p(math.exp(x))


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #

class StrategyEncoder:
    dim: int

    def encode(self, position: Sequence[float]):
        raise NotImplementedError

    def initial_position(self) -> np.ndarray:
        return np.random.uniform(-1, 1, self.dim)


# --------------------------------------------------------------------------- #
# Problem-specific encoders
# --------------------------------------------------------------------------- #

class Problem2Encoder(StrategyEncoder):
    dim = 4
    # vector: [angle, speed_raw, release_raw, fuse_raw]

    def encode(self, position: Sequence[float]):
        angle, speed_raw, release_raw, fuse_raw = position
        dir_vec = np.array([math.cos(angle * 2 * math.pi), math.sin(angle * 2 * math.pi), 0.0])
        speed = 70.0 + sigmoid(speed_raw) * 70.0
        release_time = positive(release_raw)
        fuse = clamp(positive(fuse_raw), 0.1, 15.0)
        return (dir_vec, float(speed), [[float(release_time), float(fuse)]])


class Problem3Encoder(StrategyEncoder):
    dim = 8
    # vector: [angle, speed_raw, r1_raw, f1_raw, gap2_raw, f2_raw, gap3_raw, f3_raw]

    def encode(self, position: Sequence[float]):
        (angle, speed_raw, r1_raw, f1_raw,
         gap2_raw, f2_raw, gap3_raw, f3_raw) = position
        theta = angle * 2 * math.pi
        dir_vec = np.array([math.cos(theta), math.sin(theta), 0.0])
        speed = 70.0 + sigmoid(speed_raw) * 70.0
        t1 = positive(r1_raw)
        t2 = t1 + 1.0 + positive(gap2_raw)
        t3 = t2 + 1.0 + positive(gap3_raw)
        bombs = [
            [float(t1), clamp(positive(f1_raw), 0.1, 15.0)],
            [float(t2), clamp(positive(f2_raw), 0.1, 15.0)],
            [float(t3), clamp(positive(f3_raw), 0.1, 15.0)],
        ]
        return (dir_vec, float(speed), bombs)


class Problem4Encoder(StrategyEncoder):
    dim = 12
    # vector: 3 * Problem2 (3 drones, 1 bomb each)

    def encode(self, position: Sequence[float]):
        chunks = [position[i:i+4] for i in range(0, 12, 4)]
        encoder = Problem2Encoder()
        drones = [encoder.encode(c) for c in chunks]
        return tuple(drones)  # type: ignore


class Problem5Encoder(StrategyEncoder):
    dim = 55
    # 5 drones; each:
    # [angle, speed_raw,
    #  act1, r1_raw, f1_raw,
    #  act2, gap2_raw, f2_raw,
    #  act3, gap3_raw, f3_raw]  (11 per) => 5*11=55

    def _encode_single(self, vec: Sequence[float]):
        (angle, speed_raw,
         act1, r1_raw, f1_raw,
         act2, gap2_raw, f2_raw,
         act3, gap3_raw, f3_raw) = vec
        theta = angle * 2 * math.pi
        dir_vec = np.array([math.cos(theta), math.sin(theta), 0.0])
        speed = 70.0 + sigmoid(speed_raw) * 70.0
        a1 = sigmoid(act1)
        a2 = sigmoid(act2)
        a3 = sigmoid(act3)
        bombs = []
        base_t = 0.0
        if a1 > 0.5:
            t1 = positive(r1_raw)
            bombs.append([float(t1), clamp(positive(f1_raw), 0.1, 15.0)])
            base_t = t1
        if a2 > 0.5:
            t2 = base_t + 1.0 + positive(gap2_raw)
            bombs.append([float(t2), clamp(positive(f2_raw), 0.1, 15.0)])
            base_t = t2
        if a3 > 0.5:
            t3 = base_t + 1.0 + positive(gap3_raw)
            bombs.append([float(t3), clamp(positive(f3_raw), 0.1, 15.0)])
        return (dir_vec, float(speed), bombs)

    def encode(self, position: Sequence[float]):
        drones = [self._encode_single(position[i:i+11])
                  for i in range(0, self.dim, 11)]
        return tuple(drones)  # type: ignore


# Mapping problem_id -> encoder instance
ENCODERS: Dict[int, StrategyEncoder] = {
    2: Problem2Encoder(),
    3: Problem3Encoder(),
    4: Problem4Encoder(),
    5: Problem5Encoder(),
}


def get_encoder(problem_id: int) -> StrategyEncoder:
    if problem_id not in ENCODERS:
        raise ValueError("Supported problem ids: 2,3,4,5")
    return ENCODERS[problem_id]
