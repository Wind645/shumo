"""
Reduced dimensional encoder for Problem 5 (multi‑drone, up to 3 bombs each).

WHY
----
The original Problem5 encoding uses 11 raw parameters per drone:
[angle, speed_raw,
 act1, r1_raw, f1_raw,
 act2, gap2_raw, f2_raw,
 act3, gap3_raw, f3_raw]  => 5 drones * 11 = 55 dims.

These treat (release times, gaps, fuses) as fully independent positive
variables. In practice, we can couple timing and fuse structure to shrink
the effective search space while retaining flexibility:

New per‑drone reduced parameter set (10 dims, no act1 because first bomb
is always meaningful; you can still deactivate later ones):

  1. heading_raw          (maps to angle θ = raw * 2π)
  2. speed_raw
  3. first_release_raw    (t1 = softplus(first_release_raw))
  4. gap_scale_raw        (gap_scale = softplus(...))
  5. pattern_shape_raw    (factor = 1 + sigmoid(pattern_shape_raw) in (1,2))
  6. fuse_base_raw        (f1 = clamp(softplus(fuse_base_raw), 0.1, 15))
  7. fuse_delta1_raw      (f2 = clamp(softplus(fuse_base_raw + fuse_delta1_raw)))
  8. fuse_delta2_raw      (f3 = clamp(softplus(fuse_base_raw + fuse_delta1_raw + fuse_delta2_raw)))
  9. act2_raw             (bomb2 active if sigmoid(act2_raw) > 0.5)
  10. act3_raw            (bomb3 active if bomb2 active AND sigmoid(act3_raw) > 0.5)

Derived schedule:
  t1 = positive(first_release_raw)
  gap2 = 1 + gap_scale
  gap3 = 1 + gap_scale * factor   (factor in (1,2))
  t2 = t1 + gap2
  t3 = t2 + gap3
This guarantees t2 >= t1 + 1 and t3 >= t2 + 1 automatically.

Result bombs list (per drone):
  Always include bomb1 (t1, f1).
  Include bomb2 if act2_active -> (t2, f2).
  Include bomb3 if act2_active and act3_active -> (t3, f3).

Total reduced dimension: 5 drones * 10 = 50 (vs original 55).

We also provide:
  - ReducedProblem5Encoder.encode(reduced_vector) -> strategy tuple expected by Simulator.
  - reduced_to_full55(reduced_vector) -> approximate back‑projection into the
    original 55‑dim raw vector space (so you can still use existing
    Problem5Encoder logic or persist unified model files). Because of
    coupled structure, this mapping is not exact inverse—but it is
    consistent with original semantics (softplus for times, etc.).
  - full55_to_reduced(best_full) helper (best effort) if you want to
    migrate an existing 55‑dim solution into reduced space (heuristic).

Mathematical helpers:
  positive(x) = softplus(x) = log(1+exp(x))
  inverse_positive(y) = log(exp(y)-1)  (stable for moderate y)

NOTE:
  This module does NOT modify global ENCODERS map automatically—you can
  elect to use this reduced encoder in a custom optimization run, or
  wrap it to produce a full 55‑dim vector for compatibility with existing
  objective pipeline.

USAGE EXAMPLE
-------------
    from optimizer.reduced_p5 import ReducedProblem5Encoder, reduced_to_full55
    enc = ReducedProblem5Encoder()
    vec = enc.initial_position()
    strategy = enc.encode(vec)
    full55 = reduced_to_full55(vec)   # if needed by legacy objective

You can then plug (enc.encode -> strategy) directly into Simulator
(problem_id=5) or adapt optimizer.optimize to branch on a flag.

"""

from __future__ import annotations
import math
from typing import Sequence, List, Tuple
import numpy as np

try:
    # Re-use existing transformations to stay consistent
    from .encoders import sigmoid, positive, clamp
except ImportError:
    # Fallback minimal re-implementation
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
    def positive(x: float) -> float:
        return math.log1p(math.exp(x))
    def clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v


G = 9.81  # gravity (may be useful if later you decide to tie fuse to altitude)


# ---------------- Inverse softplus (positive) ----------------
def inverse_positive(y: float) -> float:
    """
    Inverts softplus when y>0: y = log(1+e^x) => x = log(e^y - 1)
    Safe for moderate y; for large y softplus(x) ~ x so inverse ~ y.
    """
    if y < 1e-12:
        return -20.0
    # For very large y, exp(y) may overflow—use approximation.
    if y > 30:
        return y  # softplus ~ identity
    return math.log(math.expm1(y))


# ---------------- Reduced Encoder ----------------
class ReducedProblem5Encoder:
    """
    Reduced dimension Problem 5 encoder.

    dim = 50 (5 drones * 10 params)
    Drone slice layout (10 values):
      0: heading_raw
      1: speed_raw
      2: first_release_raw
      3: gap_scale_raw
      4: pattern_shape_raw
      5: fuse_base_raw
      6: fuse_delta1_raw
      7: fuse_delta2_raw
      8: act2_raw
      9: act3_raw
    """

    dim: int = 5 * 10

    def initial_position(self) -> np.ndarray:
        return np.random.uniform(-1, 1, self.dim)

    def encode(self, position: Sequence[float]):
        if len(position) != self.dim:
            raise ValueError(f"Expected reduced vector length {self.dim}, got {len(position)}")
        drones = []
        for d in range(5):
            off = d * 10
            (heading_raw,
             speed_raw,
             first_release_raw,
             gap_scale_raw,
             pattern_shape_raw,
             fuse_base_raw,
             fuse_delta1_raw,
             fuse_delta2_raw,
             act2_raw,
             act3_raw) = position[off:off+10]

            # Direction
            theta = heading_raw * 2 * math.pi
            dir_vec = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)

            # Speed
            speed = 70.0 + sigmoid(speed_raw) * 70.0

            # Times
            t1 = positive(first_release_raw)
            gap_scale = positive(gap_scale_raw)
            factor = 1.0 + sigmoid(pattern_shape_raw)  # in (1,2)
            gap2 = 1.0 + gap_scale
            gap3 = 1.0 + gap_scale * factor
            t2 = t1 + gap2
            t3 = t2 + gap3

            # Fuses
            f1 = clamp(positive(fuse_base_raw), 0.1, 15.0)
            f2 = clamp(positive(fuse_base_raw + fuse_delta1_raw), 0.1, 15.0)
            f3 = clamp(positive(fuse_base_raw + fuse_delta1_raw + fuse_delta2_raw), 0.1, 15.0)

            # Activations
            act2 = sigmoid(act2_raw) > 0.5
            act3 = act2 and (sigmoid(act3_raw) > 0.5)

            bombs: List[List[float]] = [[float(t1), float(f1)]]
            if act2:
                bombs.append([float(t2), float(f2)])
            if act3:
                bombs.append([float(t3), float(f3)])

            drones.append((dir_vec, float(speed), bombs))
        return tuple(drones)  # type: ignore


# ---------------- Back-projection to full 55 raw vector ----------------
def reduced_to_full55(reduced: Sequence[float]) -> List[float]:
    """
    Approximate mapping from reduced 50-dim vector to original 55 raw vector layout
    used by Problem5Encoder:

      Per drone original 11 layout:
        [angle_raw, speed_raw,
         act1, r1_raw, f1_raw,
         act2, gap2_raw, f2_raw,
         act3, gap3_raw, f3_raw]

    We reconstruct:
      angle_raw    = heading_raw   (identical meaning)
      speed_raw    = speed_raw
      act1         = +6 (always active -> sigmoid(~0.9975) ≈ 0.9975)
      r1_raw       = first_release_raw
      f1_raw       = fuse_base_raw
      act2         = act2_raw
      gap2_raw     = inverse_positive( gap_scale )   where gap_scale = softplus(gap_scale_raw)
      f2_raw       = fuse_base_raw + fuse_delta1_raw
      act3         = act3_raw
      gap3_raw     = inverse_positive( gap_scale * (1 + sigmoid(pattern_shape_raw)) )
      f3_raw       = fuse_base_raw + fuse_delta1_raw + fuse_delta2_raw

    NOTE: Because the original model expects positive(gap*_raw) directly as an
    *additive offset* (in the encoder: positive(gap2_raw)), this reversal
    is consistent.

    Returns a list[float] of length 55.
    """
    if len(reduced) != 50:
        raise ValueError("Reduced vector must have length 50.")
    full: List[float] = []
    for d in range(5):
        off = d * 10
        (heading_raw,
         speed_raw,
         first_release_raw,
         gap_scale_raw,
         pattern_shape_raw,
         fuse_base_raw,
         fuse_delta1_raw,
         fuse_delta2_raw,
         act2_raw,
         act3_raw) = reduced[off:off+10]

        gap_scale = positive(gap_scale_raw)
        factor = 1.0 + sigmoid(pattern_shape_raw)
        gap2_raw = inverse_positive(gap_scale)  # positive(gap2_raw)=gap_scale
        gap3_raw = inverse_positive(gap_scale * factor)

        # Force first bomb active
        act1_raw = 6.0  # sigmoid(6) ~ 0.9975

        r1_raw = first_release_raw
        f1_raw = fuse_base_raw
        f2_raw = fuse_base_raw + fuse_delta1_raw
        f3_raw = fuse_base_raw + fuse_delta1_raw + fuse_delta2_raw

        full.extend([
            heading_raw, speed_raw,
            act1_raw, r1_raw, f1_raw,
            act2_raw, gap2_raw, f2_raw,
            act3_raw, gap3_raw, f3_raw
        ])
    return full


def full55_to_reduced(full: Sequence[float]) -> List[float]:
    """
    Best-effort heuristic to map a 55-dim original raw vector into the reduced 50-dim space.
    Assumes original ordering and semantics. Because the reduced form couples
    some originally independent degrees of freedom, this is lossy.

    Strategy:
      heading_raw       = angle_raw
      speed_raw         = speed_raw
      first_release_raw = r1_raw
      gap_scale_raw     ≈ inverse_positive( max( positive(gap2_raw) - 1, 1e-6 ) ) ??? Not needed.
        Actually: original gap2 = positive(gap2_raw)
                  We define gap_scale = max(positive(gap2_raw) - 1, 1e-6) ? WAIT:
        In original Problem5: t2 = t1 + 1 + positive(gap2_raw)
        Our reduced: gap2 = 1 + gap_scale  => we want gap_scale = positive(gap2_raw)
        So gap_scale_raw = gap2_raw (directly) is fine (since positive(gap_scale_raw) = gap_scale)
      pattern_shape_raw chosen so that:
        gap3_original_offset = positive(gap3_raw)
        gap3_reduced = 1 + gap_scale * factor  with factor=(1+sigmoid(pattern_shape_raw))
        => positive(gap3_raw) ≈ 1 + gap_scale * factor
        factor ≈ (positive(gap3_raw)-1)/gap_scale
        pattern_shape_raw ≈ sigmoid^-1(factor - 1)
      fuse chain:
        fuse_base_raw = f1_raw
        fuse_delta1_raw = (f2_raw - f1_raw)
        fuse_delta2_raw = (f3_raw - f2_raw)
      act2_raw, act3_raw copied

    Returns a length-50 reduced vector.
    """
    if len(full) != 55:
        raise ValueError("Full raw vector must have length 55.")
    reduced: List[float] = []
    for d in range(5):
        base = d * 11
        (angle_raw, speed_raw,
         act1_raw, r1_raw, f1_raw,
         act2_raw, gap2_raw, f2_raw,
         act3_raw, gap3_raw, f3_raw) = full[base:base+11]

        # First release
        first_release_raw = r1_raw

        # gap_scale_raw: want positive(gap_scale_raw)=gap_scale=positive(gap2_raw)
        # So we can just re-use gap2_raw (since positive(gap2_raw) is original gap value)
        gap_scale_raw = gap2_raw

        gap_scale_val = positive(gap2_raw)
        g3_val = positive(gap3_raw)
        # Solve factor ~ (g3_val - 1)/gap_scale_val
        if gap_scale_val < 1e-9:
            factor = 1.0
        else:
            # Ensure >1 range
            factor = max(1.0 + 1e-6, min(2.0 - 1e-6, (g3_val - 1.0) / gap_scale_val))
        # pattern_shape_raw from factor = 1 + sigmoid(ps_raw)
        # => sigmoid(ps_raw) = factor - 1
        sig_target = factor - 1.0
        sig_target = min(1 - 1e-8, max(1e-8, sig_target))
        pattern_shape_raw = math.log(sig_target / (1 - sig_target))

        fuse_base_raw = f1_raw
        fuse_delta1_raw = f2_raw - f1_raw
        fuse_delta2_raw = f3_raw - f2_raw

        reduced.extend([
            angle_raw, speed_raw,
            first_release_raw, gap_scale_raw, pattern_shape_raw,
            fuse_base_raw, fuse_delta1_raw, fuse_delta2_raw,
            act2_raw, act3_raw
        ])
    return reduced


# ---------------- Convenience objective builder ----------------
def build_reduced_problem5_objective(dt: float,
                                     aggregate: str = "sum",
                                     weights: Sequence[float] | None = None):
    """
    Returns (encoder, objective_fn) for reduced encoding usage directly with
    a PSO / DE style optimizer.

    The objective_fn expects a reduced 50-dim vector.

    Internally converts to standard Simulator strategy (bypasses the legacy
    raw->full55->Problem5Encoder path).
    """
    from .pso import simulate_fitness  # reuse existing simulation logic

    encoder = ReducedProblem5Encoder()

    def objective(vec: Sequence[float]) -> float:
        # Directly evaluate using simulate_fitness by passing the *full* raw vector
        # OR build a custom evaluate that bypasses Problem5Encoder. Here we leverage
        # existing pipeline for consistency:
        full_raw = reduced_to_full55(vec)
        return simulate_fitness(5, full_raw, dt=dt, aggregate=aggregate, weights=weights)

    return encoder, objective


__all__ = [
    "ReducedProblem5Encoder",
    "reduced_to_full55",
    "full55_to_reduced",
    "build_reduced_problem5_objective",
]

if __name__ == "__main__":
    # Simple self-test
    enc = ReducedProblem5Encoder()
    vec = enc.initial_position()
    strat = enc.encode(vec)
    full = reduced_to_full55(vec)
    back = full55_to_reduced(full)
    diff = np.linalg.norm(np.asarray(vec) - np.asarray(back))
    print("Reduced vector norm diff after round trip (heuristic):", diff)
    print("Strategy example (first drone bombs):", strat[0][2])
