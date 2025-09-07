from __future__ import annotations
"""
Brute-force enumerate UAV azimuth and speed, and for each pair solve for best
(release_time, explode_delay) quickly.

Fast inner solver idea (fixed speed v and azimuth theta):
  - Optimize over explosion time T = release_time + explode_delay (outer 1D)
  - For each T, optimize explode_delay d in domain [max(0, T-rel_max), min(delay_max, T)] (inner 1D)
    with release_time r = T - d.
This reduces a 2D search (r,d) with linear constraint to two nested 1D searches.

We use:
  - Coarse grid over T centered at geometric equal-x time T_eqx = 2200 / (300 + v*cos(theta)).
  - Golden-section 1D refinement on both T and d.
  - Fast evaluator method='judge_caps' by default.

Output: per (azimuth, speed) best r,d and occlusion seconds in JSONL.
"""
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

from optimizer_api import evaluate_problem2

SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi

# Default global bounds
RELEASE_MAX_DEFAULT = 66.0
DELAY_MAX_DEFAULT = 20.0
DT_DEFAULT = 0.01
METHOD_DEFAULT = "judge_caps"  # 'judge_caps' | 'sampling'

# Search defaults
T_WINDOW_BEFORE = 8.0       # search window before T_eqx
T_WINDOW_AFTER = 12.0       # search window after T_eqx
T_COARSE_SAMPLES = 41
T_REFINE_ITERS = 12         # golden iterations for T
DLY_COARSE_SAMPLES = 9
DLY_REFINE_ITERS = 12       # golden iterations for dly
TOP_K_T_FOR_REFINE = 3

# Optional geometric angle filter (skip very unlikely headings)
USE_GEOM_FILTER = True
GEOM_R = 10.0  # lateral tolerance (m) used by filter


# ---------------- helper math ----------------

def wrap_angle(a: float) -> float:
    return a % (2.0 * math.pi)


def clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def equal_x_time(v: float, theta: float) -> float:
    # Solve 17800 + v*cos(theta)*t = 20000 - 300*t -> t = 2200 / (300 + v*cos(theta))
    den = 300.0 + v * math.cos(theta)
    if den <= 1e-9:
        return 0.0
    return 2200.0 / den


def lateral_at_equal_x(v: float, theta: float) -> float:
    # |y| at equal-x time
    t = equal_x_time(v, theta)
    return abs(v * math.sin(theta) * t)


def geometric_heading_ok(theta: float, R: float = GEOM_R) -> bool:
    # From derivation: forward intersection near radius band requires cos>0 and |tan| <= R/200
    c = math.cos(theta)
    if c <= 0:
        return False
    return abs(math.tan(theta)) <= (R / 200.0)


def evaluate_fixed(speed: float, azimuth: float, release_time: float, explode_delay: float,
                   method: str, dt: float) -> float:
    res = evaluate_problem2(
        speed=float(speed),
        azimuth=float(azimuth),
        release_time=float(release_time),
        explode_delay=float(explode_delay),
        occlusion_method=method,
        dt=dt,
    )
    return float(res["occluded_time"]["M1"])  # seconds


@dataclass
class PairResult:
    release_time: float
    explode_delay: float
    value: float

    def as_dict(self) -> Dict:
        return dict(release_time=self.release_time, explode_delay=self.explode_delay, value=self.value,
                    explode_at=self.release_time + self.explode_delay)

# New: clamp to local windows
def clamp_local(r: float, d: float, r_lo: float, r_hi: float, d_lo: float, d_hi: float) -> Tuple[float, float]:
    return clip(r, r_lo, r_hi), clip(d, d_lo, d_hi)


# ---------------- Simulated Annealing for fixed (speed, azimuth) ----------------

def _init_guess_rd(speed: float, azim: float, rel_max: float, delay_max: float,
                   method: str, dt: float) -> PairResult:
    # Heuristic: center around equal-x time
    T0 = equal_x_time(speed, azim)
    a = max(0.0, T0 - rel_max)
    b = min(delay_max, T0)
    if b < a:
        # fallback to mid of bounds
        d0 = 0.5 * delay_max
    else:
        d0 = 0.5 * (a + b)
    r0 = clip(T0 - d0, 0.0, rel_max)
    d0 = clip(d0, 0.0, delay_max)
    v0 = evaluate_fixed(speed, azim, r0, d0, method, dt)

    # also try a random sample
    rr = random.random() * rel_max
    dd = random.random() * delay_max
    vr = evaluate_fixed(speed, azim, rr, dd, method, dt)

    if vr > v0:
        return PairResult(rr, dd, vr)
    return PairResult(r0, d0, v0)


def _neighbor_rd(speed: float, azim: float, cur: PairResult, rel_max: float, delay_max: float,
                 method: str, dt: float, temp: float,
                 scale_r: float = 1.2, scale_d: float = 0.8) -> PairResult:
    s = max(temp, 1e-3)
    r = clip(cur.release_time + random.gauss(0.0, scale_r * s), 0.0, rel_max)
    d = clip(cur.explode_delay + random.gauss(0.0, scale_d * s), 0.0, delay_max)
    val = evaluate_fixed(speed, azim, r, d, method, dt)
    return PairResult(r, d, val)


def sa_opt_rd(speed: float, azim: float, rel_max: float, delay_max: float,
              method: str, dt: float,
              t0: float = 0.8, t_end: float = 1e-3, alpha: float = 0.988,
              steps_per_t: int = 40, max_steps: int = 300,
              global_jump_prob: float = 0.03,
              base: Optional[PairResult] = None,
              r_lo: Optional[float] = None, r_hi: Optional[float] = None,
              d_lo: Optional[float] = None, d_hi: Optional[float] = None) -> PairResult:
    # Local bounds fallback to global
    r_lo_eff = 0.0 if r_lo is None else max(0.0, r_lo)
    r_hi_eff = rel_max if r_hi is None else min(rel_max, r_hi)
    d_lo_eff = 0.0 if d_lo is None else max(0.0, d_lo)
    d_hi_eff = delay_max if d_hi is None else min(delay_max, d_hi)

    if base is not None:
        r0, d0 = clamp_local(base.release_time, base.explode_delay, r_lo_eff, r_hi_eff, d_lo_eff, d_hi_eff)
        v0 = evaluate_fixed(speed, azim, r0, d0, method, dt)
        cur = PairResult(r0, d0, v0)
    else:
        cur = _init_guess_rd(speed, azim, rel_max, delay_max, method, dt)
        r0, d0 = clamp_local(cur.release_time, cur.explode_delay, r_lo_eff, r_hi_eff, d_lo_eff, d_hi_eff)
        if (r0 != cur.release_time) or (d0 != cur.explode_delay):
            v0 = evaluate_fixed(speed, azim, r0, d0, method, dt)
            cur = PairResult(r0, d0, v0)

    best = cur
    t = t0
    steps = 0
    while t > t_end and steps < max_steps:
        for _ in range(steps_per_t):
            steps += 1
            # occasional global jump inside local box
            if global_jump_prob > 0 and random.random() < global_jump_prob:
                rj = r_lo_eff + random.random() * max(1e-12, (r_hi_eff - r_lo_eff))
                dj = d_lo_eff + random.random() * max(1e-12, (d_hi_eff - d_lo_eff))
                vj = evaluate_fixed(speed, azim, rj, dj, method, dt)
                j = PairResult(rj, dj, vj)
                if j.value >= cur.value:
                    cur = j
                    if j.value > best.value:
                        best = j
                    continue
            # neighbor with local bounds
            s = max(t, 1e-3)
            r_prop = cur.release_time + random.gauss(0.0, 1.2 * s)
            d_prop = cur.explode_delay + random.gauss(0.0, 0.8 * s)
            r_prop, d_prop = clamp_local(r_prop, d_prop, r_lo_eff, r_hi_eff, d_lo_eff, d_hi_eff)
            v_prop = evaluate_fixed(speed, azim, r_prop, d_prop, method, dt)
            delta = v_prop - cur.value
            if delta >= 0 or math.exp(delta / max(t, 1e-9)) > random.random():
                cur = PairResult(r_prop, d_prop, v_prop)
                if cur.value > best.value:
                    best = cur
            if steps >= max_steps:
                break
        t *= alpha
    return best


# ------------------------ Previous nested 1D optimizer (kept) -------------------

def inner_opt_delay(speed: float, azim: float, T: float, rel_max: float, delay_max: float,
                    method: str, dt: float) -> PairResult:
    # Domain for delay given T: d in [max(0,T-rel_max), min(delay_max, T)]
    a = max(0.0, T - rel_max)
    b = min(delay_max, T)
    if b - a <= 1e-9:
        r = clip(T - a, 0.0, rel_max)
        val = evaluate_fixed(speed, azim, r, a, method, dt)
        return PairResult(r, a, val)

    # coarse scan
    best = None
    for i in range(DLY_COARSE_SAMPLES):
        u = i / max(1, DLY_COARSE_SAMPLES - 1)
        d = a * (1 - u) + b * u
        r = T - d
        val = evaluate_fixed(speed, azim, r, d, method, dt)
        if (best is None) or (val > best.value):
            best = PairResult(r, d, val)

    # golden-section refine around domain using maximize
    phi = (1 + 5 ** 0.5) / 2
    invphi = 1 / phi
    # initialize interior points
    x1 = b - (b - a) * invphi
    x2 = a + (b - a) * invphi
    f1 = evaluate_fixed(speed, azim, T - x1, x1, method, dt)
    f2 = evaluate_fixed(speed, azim, T - x2, x2, method, dt)

    for _ in range(DLY_REFINE_ITERS):
        if f1 < f2:  # move left bound up
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * invphi
            f2 = evaluate_fixed(speed, azim, T - x2, x2, method, dt)
        else:        # move right bound down
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) * invphi
            f1 = evaluate_fixed(speed, azim, T - x1, x1, method, dt)

    # pick best among the two interior and coarse best
    cand = [best,
            PairResult(T - x1, x1, f1),
            PairResult(T - x2, x2, f2)]
    cand_best = max(cand, key=lambda z: z.value)
    return cand_best


def outer_opt_T(speed: float, azim: float, rel_max: float, delay_max: float,
                method: str, dt: float) -> PairResult:
    # T bounds: [0, min(rel_max+delay_max, ~70s)]
    T_lo = 0.0
    T_hi = min(rel_max + delay_max, 70.0)

    # center around equal-x time
    T0 = equal_x_time(speed, azim)
    a = clip(T0 - T_WINDOW_BEFORE, T_lo, T_hi)
    b = clip(T0 + T_WINDOW_AFTER, T_lo, T_hi)
    if b - a < 1e-6:
        a, b = T_lo, T_hi

    # coarse scan over T
    T_candidates = []
    for i in range(T_COARSE_SAMPLES):
        u = i / max(1, T_COARSE_SAMPLES - 1)
        T = a * (1 - u) + b * u
        pr = inner_opt_delay(speed, azim, T, rel_max, delay_max, method, dt)
        T_candidates.append((T, pr))
    T_candidates.sort(key=lambda tp: tp[1].value, reverse=True)
    best = T_candidates[0][1]

    # refine top-K using golden on T while solving inner delay each time
    phi = (1 + 5 ** 0.5) / 2
    invphi = 1 / phi

    # initial bracket for refine use the overall [a,b]
    L, R = a, b
    x1 = R - (R - L) * invphi
    x2 = L + (R - L) * invphi
    f1 = inner_opt_delay(speed, azim, x1, rel_max, delay_max, method, dt)
    f2 = inner_opt_delay(speed, azim, x2, rel_max, delay_max, method, dt)

    if f2.value > best.value:
        best = f2
    if f1.value > best.value:
        best = f1

    for _ in range(T_REFINE_ITERS):
        if f1.value < f2.value:
            L = x1
            x1 = x2
            f1 = f2
            x2 = L + (R - L) * invphi
            f2 = inner_opt_delay(speed, azim, x2, rel_max, delay_max, method, dt)
            if f2.value > best.value:
                best = f2
        else:
            R = x2
            x2 = x1
            f2 = f1
            x1 = R - (R - L) * invphi
            f1 = inner_opt_delay(speed, azim, x1, rel_max, delay_max, method, dt)
            if f1.value > best.value:
                best = f1

    return best


# New: coarse grid scan to find first non-zero, optionally focused around base

def coarse_scan_rd(speed: float, azim: float, rel_max: float, delay_max: float,
                   method: str, dt: float,
                   r_points: int = 7, d_points: int = 7,
                   center: Optional[PairResult] = None,
                   r_window: float = 3.0, d_window: float = 2.0) -> PairResult:
    if center is None:
        # derive a center from equal-x heuristic
        guess = _init_guess_rd(speed, azim, rel_max, delay_max, method, dt)
        rc, dc = guess.release_time, guess.explode_delay
    else:
        rc, dc = center.release_time, center.explode_delay

    r_lo = clip(rc - r_window, 0.0, rel_max)
    r_hi = clip(rc + r_window, 0.0, rel_max)
    d_lo = clip(dc - d_window, 0.0, delay_max)
    d_hi = clip(dc + d_window, 0.0, delay_max)

    best = None
    for i in range(max(1, r_points)):
        u = 0.0 if r_points == 1 else i / (r_points - 1)
        r = r_lo * (1 - u) + r_hi * u
        for j in range(max(1, d_points)):
            v = 0.0 if d_points == 1 else j / (d_points - 1)
            d = d_lo * (1 - v) + d_hi * v
            val = evaluate_fixed(speed, azim, r, d, method, dt)
            if (best is None) or (val > best.value):
                best = PairResult(r, d, val)
            # early exit on first non-zero reasonably good
            if val > 1e-6 and (i + j) <= 2:
                return best
    return best if best is not None else PairResult(0.0, 0.0, 0.0)


# ------------------------------- main -------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Enumerate (azimuth, speed) and solve best (release, delay) quickly")
    ap.add_argument("--angle-step", type=float, default=math.radians(1.0), help="angle step in radians")
    ap.add_argument("--speed-step", type=float, default=5.0, help="speed step in m/s")
    ap.add_argument("--start", type=float, default=0.0, help="angle start (rad)")
    ap.add_argument("--end", type=float, default=2*math.pi, help="angle end (rad, inclusive scan)")

    ap.add_argument("--method", choices=["judge_caps", "sampling"], default=METHOD_DEFAULT)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--rel-max", type=float, default=RELEASE_MAX_DEFAULT)
    ap.add_argument("--delay-max", type=float, default=DELAY_MAX_DEFAULT)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use-geom-filter", action="store_true", help="skip angles failing geometric filter")

    ap.add_argument("--out-jsonl", type=str, default="best_p2_angle_speed.jsonl",
                    help="write best per (angle,speed) row to JSONL")

    # Solver selection
    ap.add_argument("--solver", choices=["sa", "nested", "hybrid"], default="hybrid",
                    help="inner optimizer for (release, delay)")
    ap.add_argument("--dt-coarse", type=float, default=None,
                    help="coarse dt for nested/hybrid (defaults to --dt if None)")
    ap.add_argument("--hybrid-sa-steps", type=int, default=60,
                    help="SA steps used in hybrid refine")

    # Skip pair if lateral at equal-x exceeds threshold (fast prefilter)
    ap.add_argument("--skip-if-lateral-gt", type=float, default=30.0,
                    help="skip (angle,speed) if |y| at equal-x > threshold meters (0 to disable)")

    # SA parameters for (release, delay)
    ap.add_argument("--sa-t0", type=float, default=0.8)
    ap.add_argument("--sa-t-end", type=float, default=1e-3)
    ap.add_argument("--sa-alpha", type=float, default=0.988)
    ap.add_argument("--sa-steps-per-t", type=int, default=40)
    ap.add_argument("--sa-max-steps", type=int, default=300)
    ap.add_argument("--sa-global-jump-prob", type=float, default=0.03)

    # progress
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress-interval", type=float, default=1.0)

    args = ap.parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000
    random.seed(args.seed)

    if args.dt_coarse is None:
        args.dt_coarse = args.dt

    # prepare output
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            f.write("")

    angle_step = max(1e-6, args.angle_step)
    speed_step = max(1e-6, args.speed_step)

    total_angles = int(round((args.end - args.start) / angle_step)) + 1

    # build valid angle list (respecting filter)
    angles: list[float] = []
    for ai in range(total_angles):
        az = wrap_angle(args.start + ai * angle_step)
        if args.use_geom_filter and USE_GEOM_FILTER:
            if not geometric_heading_ok(az, GEOM_R):
                continue
        angles.append(az)

    speed_vals = []
    v = SPEED_MIN
    while v <= SPEED_MAX + 1e-9:
        speed_vals.append(round(v, 6))
        v += speed_step

    total_pairs = len(angles) * len(speed_vals)

    t0 = time.time()
    last_prog = t0
    processed = 0
    global_best = None
    prev_nonzero: Optional[PairResult] = None

    for ai, az in enumerate(angles, 1):
        for sp in speed_vals:
            # optional lateral prefilter
            if args.skip_if_lateral_gt and lateral_at_equal_x(sp, az) > args.skip_if_lateral_gt:
                processed += 1
                continue

            # choose inner solver
            if args.solver == "nested":
                pr = outer_opt_T(sp, az, args.rel_max, args.delay_max, args.method, args.dt_coarse)
            elif args.solver == "sa":
                # coarse scan first to place SA near non-zero
                pr_seed = coarse_scan_rd(sp, az, args.rel_max, args.delay_max, args.method, args.dt_coarse,
                                         r_points=5, d_points=5, center=prev_nonzero, r_window=3.0, d_window=2.0)
                pr = sa_opt_rd(
                    sp, az, args.rel_max, args.delay_max, args.method, args.dt,
                    t0=args.sa_t0, t_end=args.sa_t_end, alpha=args.sa_alpha,
                    steps_per_t=args.sa_steps_per_t, max_steps=args.sa_max_steps,
                    global_jump_prob=args.sa_global_jump_prob,
                    base=pr_seed,
                    r_lo=clip(pr_seed.release_time - 3.0, 0.0, args.rel_max),
                    r_hi=clip(pr_seed.release_time + 3.0, 0.0, args.rel_max),
                    d_lo=clip(pr_seed.explode_delay - 2.0, 0.0, args.delay_max),
                    d_hi=clip(pr_seed.explode_delay + 2.0, 0.0, args.delay_max),
                )
            else:  # hybrid
                # coarse seed around previous nonzero
                pr_seed = coarse_scan_rd(sp, az, args.rel_max, args.delay_max, args.method, args.dt_coarse,
                                         r_points=5, d_points=5, center=prev_nonzero, r_window=3.0, d_window=2.0)
                if pr_seed.value <= 1e-9:
                    pr_seed = outer_opt_T(sp, az, args.rel_max, args.delay_max, args.method, args.dt_coarse)
                pr = sa_opt_rd(
                    sp, az, args.rel_max, args.delay_max, args.method, args.dt,
                    t0=args.sa_t0, t_end=args.sa_t_end, alpha=args.sa_alpha,
                    steps_per_t=args.sa_steps_per_t, max_steps=args.hybrid_sa_steps,
                    global_jump_prob=args.sa_global_jump_prob,
                    base=pr_seed,
                    r_lo=clip(pr_seed.release_time - 3.0, 0.0, args.rel_max),
                    r_hi=clip(pr_seed.release_time + 3.0, 0.0, args.rel_max),
                    d_lo=clip(pr_seed.explode_delay - 2.0, 0.0, args.delay_max),
                    d_hi=clip(pr_seed.explode_delay + 2.0, 0.0, args.delay_max),
                )

            rec = dict(
                azimuth=az,
                azimuth_deg=math.degrees(az),
                speed=sp,
                release_time=pr.release_time,
                explode_delay=pr.explode_delay,
                value=pr.value,
                explode_at=pr.release_time + pr.explode_delay,
                method=args.method,
                dt=args.dt,
                seed=args.seed,
            )
            if args.out_jsonl:
                with open(args.out_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (global_best is None) or (rec["value"] > global_best["value"]):
                global_best = rec
            if pr.value > 1e-9:
                prev_nonzero = pr
            processed += 1
            now = time.time()
            if args.progress and (now - last_prog >= args.progress_interval):
                elapsed = now - t0
                frac = processed / max(1, total_pairs)
                eta = elapsed * (1 - frac) / max(frac, 1e-9)
                best_val = (0.0 if global_best is None else global_best["value"])
                print(f"[prog] pairs={processed}/{total_pairs} {frac:6.2%} best={best_val:.4f}s elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                      flush=True)
                last_prog = now
        print(f"[angle {az:.3f} rad | {math.degrees(az):6.2f} deg] done speeds={len(speed_vals)}")

    t1 = time.time()
    print("=== Angle+Speed enumeration finished ===")
    if global_best is not None:
        print(f"Best overall: val={global_best['value']:.4f}s | az={global_best['azimuth']:.4f} rad ({global_best['azimuth_deg']:.2f} deg) "
              f"spd={global_best['speed']:.2f} rel={global_best['release_time']:.2f} "
              f"dly={global_best['explode_delay']:.2f} expl@{global_best['explode_at']:.2f}")
    print(f"Runtime: {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
