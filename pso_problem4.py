from __future__ import annotations
"""
Particle Swarm Optimization (PSO) for Problem 4
  3 UAVs (FY1, FY2, FY3), each carries one smoke bomb. Optimize each UAV's
  speed, azimuth, and its bomb's release time and explode delay to maximize
  occluded time on M1.

Decision vector x (len=12):
  [s1, a1, t1, d1,  s2, a2, t2, d2,  s3, a3, t3, d3]
    - si ∈ [70, 140] m/s
    - ai ∈ [0, 2π) (wrap-around)
    - ti ∈ [0, rel_max]
    - di ∈ [0, delay_max]

Usage:
  python pso_problem4.py
  python pso_problem4.py --use-cli --iters 300 --swarm 60 --method judge_caps --seed 42

Notes:
  - judge_caps is faster but conservative; sampling is slower but more accurate.
  - Results exported to result2.xlsx (install pandas/openpyxl if missing).
"""
import math
import os
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np

from optimizer_api import evaluate_problem4

# ===================== User-configurable block =====================
CONFIG = dict(
    # simulation / objective
    method="judge_caps",   # 'judge_caps' or 'sampling'
    dt=0.05,
    rel_max=66.0,
    delay_max=20.0,

    # PSO hyper-parameters
    swarm_size=60,
    iters=300,
    w_start=0.9,
    w_end=0.45,
    c1=1.8,
    c2=1.8,
    v_clamp_frac=0.25,   # velocity clamp as fraction of range per dim
    reset_fraction=0.12, # fraction of particles to randomly reset per iter

    # misc
    seed=42,

    # logging / checkpoint
    progress=True,
    progress_interval=1.0,  # seconds
    verbose=False,
    checkpoint_file="best_p4_pso.json",
    ckpt_every_iter=20,
)
# ==================================================================

# UAV initial positions (fixed heights)
FY1_POS = np.array([17800.0,     0.0, 1800.0])
FY2_POS = np.array([12000.0,  1400.0, 1400.0])
FY3_POS = np.array([ 6000.0, -3000.0,  700.0])

# Allow optional seed injection file (default path)
DEFAULT_SEED_FILE = "best_p4_seed.json"

SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi
DELAY_MIN = 0.0
AZ_IDXS = (1, 5, 9)  # indices of azimuth dims in the 12-d vector


# ---------------- Utilities ----------------

def _atomic_save_json(obj: Dict, path: str):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def wrap_angle(a: float) -> float:
    return a % (2.0 * math.pi)


def ang_diff(a: float, b: float) -> float:
    """Shortest signed angular difference a-b mapped to [-pi, pi)."""
    d = (a - b) % (2.0 * math.pi)
    if d >= math.pi:
        d -= 2.0 * math.pi
    return d


def clip(x, lo, hi):
    return min(max(x, lo), hi)


def _bounds(cfg: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray]:
    lo_block = np.array([SPEED_MIN, AZIM_MIN, 0.0, DELAY_MIN], dtype=float)
    hi_block = np.array([SPEED_MAX, AZIM_MAX, cfg.rel_max, cfg.delay_max], dtype=float)
    lo = np.tile(lo_block, 3)
    hi = np.tile(hi_block, 3)
    return lo, hi


def _clip_and_wrap_raw(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    y = np.minimum(np.maximum(x, lo), hi)
    # wrap azimuth dims
    for k in AZ_IDXS:
        y[k] = wrap_angle(y[k])
    return y


def vec_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    d = d.copy()
    for k in AZ_IDXS:
        d[k] = ang_diff(a[k], b[k])
    return d


def vec_add_angle(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    y = x + d
    y = y.copy()
    for k in AZ_IDXS:
        y[k] = wrap_angle(y[k])
    return y


def _base_az_for(pos_xy: Tuple[float, float]) -> float:
    x0, y0 = pos_xy
    # azimuth pointing roughly towards fake target (0,0) in XY plane
    return math.atan2(-y0, -x0)


def decode_position(x: np.ndarray, cfg: SimpleNamespace):
    """Map raw vector to feasible decision variables for 3 UAVs.
    Returns (speeds[3], azimuths[3], times[3], delays[3]).
    """
    y = x.astype(float).copy()
    # clip & wrap
    lo, hi = _bounds(cfg)
    y = _clip_and_wrap_raw(y, lo, hi)
    s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3 = y.tolist()
    speeds = [clip(s1, SPEED_MIN, SPEED_MAX), clip(s2, SPEED_MIN, SPEED_MAX), clip(s3, SPEED_MIN, SPEED_MAX)]
    azs = [wrap_angle(a1), wrap_angle(a2), wrap_angle(a3)]
    times = [clip(t1, 0.0, cfg.rel_max), clip(t2, 0.0, cfg.rel_max), clip(t3, 0.0, cfg.rel_max)]
    delays = [clip(d1, DELAY_MIN, cfg.delay_max), clip(d2, DELAY_MIN, cfg.delay_max), clip(d3, DELAY_MIN, cfg.delay_max)]
    return speeds, azs, times, delays


def encode_feasible(speeds: List[float], azs: List[float], times: List[float], delays: List[float]) -> np.ndarray:
    return np.array([
        float(speeds[0]), float(azs[0]), float(times[0]), float(delays[0]),
        float(speeds[1]), float(azs[1]), float(times[1]), float(delays[1]),
        float(speeds[2]), float(azs[2]), float(times[2]), float(delays[2]),
    ], dtype=float)


def evaluate_vec(x: np.ndarray, cfg: SimpleNamespace) -> float:
    speeds, azs, times, delays = decode_position(x, cfg)
    drones_spec = [
        {"pos0": FY1_POS.tolist(), "speed": speeds[0], "azimuth": azs[0], "bombs": [{"deploy_time": times[0], "explode_delay": delays[0]}]},
        {"pos0": FY2_POS.tolist(), "speed": speeds[1], "azimuth": azs[1], "bombs": [{"deploy_time": times[1], "explode_delay": delays[1]}]},
        {"pos0": FY3_POS.tolist(), "speed": speeds[2], "azimuth": azs[2], "bombs": [{"deploy_time": times[2], "explode_delay": delays[2]}]},
    ]
    res = evaluate_problem4(drones_spec=drones_spec, dt=cfg.dt, occlusion_method=cfg.method)
    return float(res["occluded_time"]["M1"])  # maximize


def random_vec(cfg: SimpleNamespace, rng: np.random.Generator) -> np.ndarray:
    # heuristic base around pointing to origin
    base_az = [
        _base_az_for((FY1_POS[0], FY1_POS[1])),
        _base_az_for((FY2_POS[0], FY2_POS[1])),
        _base_az_for((FY3_POS[0], FY3_POS[1])),
    ]
    s1 = rng.uniform(90.0, 130.0); a1 = wrap_angle(base_az[0] + rng.normal(0.0, 0.3))
    s2 = rng.uniform(90.0, 130.0); a2 = wrap_angle(base_az[1] + rng.normal(0.0, 0.3))
    s3 = rng.uniform(90.0, 130.0); a3 = wrap_angle(base_az[2] + rng.normal(0.0, 0.35))
    # early releases are often helpful
    t1 = rng.uniform(0.0, min(6.0, cfg.rel_max))
    t2 = rng.uniform(0.0, min(6.0, cfg.rel_max))
    t3 = rng.uniform(0.0, min(6.0, cfg.rel_max))
    d1 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    d2 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    d3 = rng.uniform(2.0, min(6.0, cfg.delay_max))
    return np.array([s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3], dtype=float)


# ---------------- PSO core ----------------

@dataclass
class Particle:
    x: np.ndarray   # position (12,)
    v: np.ndarray   # velocity (12,)
    f: float        # fitness
    pbest_x: np.ndarray
    pbest_f: float

    def as_dict(self) -> Dict:
        return dict(
            s1=float(self.x[0]), a1=float(self.x[1]), t1=float(self.x[2]), d1=float(self.x[3]),
            s2=float(self.x[4]), a2=float(self.x[5]), t2=float(self.x[6]), d2=float(self.x[7]),
            s3=float(self.x[8]), a3=float(self.x[9]), t3=float(self.x[10]), d3=float(self.x[11]),
            value=float(self.f),
        )


def _init_swarm(cfg: SimpleNamespace, rng: np.random.Generator) -> Tuple[List[Particle], Particle]:
    lo, hi = _bounds(cfg)
    span = hi - lo
    vspan = span.copy()
    for k in AZ_IDXS:
        vspan[k] = math.pi

    swarm: List[Particle] = []
    gbest: Particle | None = None

    # Try to load optional seed from cfg.init_seed_file or default path
    seed_x: np.ndarray | None = None
    seed_path = getattr(cfg, 'init_seed_file', None)
    if seed_path is None and os.path.isfile(DEFAULT_SEED_FILE):
        seed_path = DEFAULT_SEED_FILE
    if seed_path and os.path.isfile(seed_path):
        try:
            with open(seed_path, 'r', encoding='utf-8') as f:
                sd = json.load(f)
            seed_x = np.array([
                float(sd.get('s1')), float(sd.get('a1')), float(sd.get('t1')), float(sd.get('d1')),
                float(sd.get('s2')), float(sd.get('a2')), float(sd.get('t2')), float(sd.get('d2')),
                float(sd.get('s3')), float(sd.get('a3')), float(sd.get('t3')), float(sd.get('d3')),
            ], dtype=float)
            # clip & wrap, then evaluate
            seed_x = _clip_and_wrap_raw(seed_x, lo, hi)
            f0 = evaluate_vec(seed_x, cfg)
            v0 = (rng.random(12) * 2.0 - 1.0) * 0.05 * vspan
            for k in AZ_IDXS:
                v0[k] = (rng.random() * 2.0 - 1.0) * (0.05 * vspan[k])
            p0 = Particle(x=seed_x.copy(), v=v0.copy(), f=f0, pbest_x=seed_x.copy(), pbest_f=f0)
            swarm.append(p0)
            gbest = Particle(x=p0.x.copy(), v=p0.v.copy(), f=p0.f, pbest_x=p0.pbest_x.copy(), pbest_f=p0.pbest_f)
            if getattr(cfg, 'progress', False):
                print(f"[seed] loaded {seed_path} with value≈{f0:.4f}s")
        except Exception as e:
            print(f"[warn] failed to load seed from {seed_path}: {e}")

    # base around reasonable heuristic (point roughly to origin, moderate speeds)
    base_az = [
        _base_az_for((FY1_POS[0], FY1_POS[1])),
        _base_az_for((FY2_POS[0], FY2_POS[1])),
        _base_az_for((FY3_POS[0], FY3_POS[1])),
    ]
    base = np.array([
        120.0, base_az[0], 2.0, 3.5,
        120.0, base_az[1], 4.0, 4.0,
        120.0, base_az[2], 4.0, 4.0,
    ], dtype=float)

    for _ in range(len(swarm), cfg.swarm_size):
        if rng.random() < 0.35:
            # jitter around base within a small fraction of full range
            x = base + rng.normal(0.0, 0.12, size=12) * span
        else:
            x = lo + rng.random(12) * span
        x = _clip_and_wrap_raw(x, lo, hi)
        # project to feasible manifold via decode/encode
        speeds, azs, times, delays = decode_position(x, cfg)
        x = encode_feasible(speeds, azs, times, delays)
        v = (rng.random(12) * 2.0 - 1.0) * 0.10 * vspan
        for k in AZ_IDXS:
            v[k] = (rng.random() * 2.0 - 1.0) * (0.10 * vspan[k])
        f = evaluate_vec(x, cfg)
        p = Particle(x=x, v=v, f=f, pbest_x=x.copy(), pbest_f=f)
        swarm.append(p)
        if gbest is None or p.f > gbest.f:
            gbest = Particle(x=p.x.copy(), v=p.v.copy(), f=p.f, pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

    assert gbest is not None
    return swarm, gbest


def pso(cfg: SimpleNamespace) -> Particle:
    rng = np.random.default_rng(cfg.seed)
    lo, hi = _bounds(cfg)
    span = hi - lo

    swarm, gbest = _init_swarm(cfg, rng)

    # velocity clamp per-dimension
    v_max = cfg.v_clamp_frac * span
    for k in AZ_IDXS:
        v_max[k] = cfg.v_clamp_frac * math.pi

    start = time.time()
    last_prog = start

    # initial checkpoint
    if cfg.checkpoint_file:
        _atomic_save_json(dict(iter=0, timestamp=time.time()) | gbest.as_dict(), cfg.checkpoint_file)

    for it in range(1, cfg.iters + 1):
        w = cfg.w_end + (cfg.w_start - cfg.w_end) * max(0.0, (cfg.iters - it) / max(1, cfg.iters - 1))
        gbest_x = gbest.x.copy()

        for p in swarm:
            r1 = rng.random(12)
            r2 = rng.random(12)
            cognitive = r1 * vec_sub(p.pbest_x, p.x)
            social = r2 * vec_sub(gbest_x, p.x)
            p.v = w * p.v + cfg.c1 * cognitive + cfg.c2 * social
            # clamp
            p.v = np.clip(p.v, -v_max, v_max)
            # update
            p.x = vec_add_angle(p.x, p.v)
            # raw clip
            p.x = _clip_and_wrap_raw(p.x, lo, hi)

            # evaluate
            p.f = evaluate_vec(p.x, cfg)
            if p.f >= p.pbest_f:
                p.pbest_x = p.x.copy()
                p.pbest_f = p.f
                if p.f > gbest.f:
                    gbest = Particle(x=p.x.copy(), v=p.v.copy(), f=p.f, pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

        # random resets (avoid resetting the current best)
        avail = list(range(len(swarm)))
        if len(avail) > 1:
            best_idx = int(np.argmax([p.f for p in swarm]))
            if best_idx in avail:
                avail.remove(best_idx)
            if len(avail) > 0:
                m = max(1, int(cfg.reset_fraction * cfg.swarm_size))
                k = min(m, len(avail))
                reset_idxs = list(rng.choice(avail, size=k, replace=False))
                for i in reset_idxs:
                    x_new = random_vec(cfg, rng)
                    # small random velocity
                    vspan = span.copy()
                    for k2 in AZ_IDXS:
                        vspan[k2] = math.pi
                    v_new = (rng.random(12) * 2.0 - 1.0) * 0.10 * vspan
                    for k2 in AZ_IDXS:
                        v_new[k2] = (rng.random() * 2.0 - 1.0) * (0.10 * vspan[k2])
                    f_new = evaluate_vec(x_new, cfg)
                    p = swarm[i]
                    p.x, p.v, p.f = x_new, v_new, f_new
                    p.pbest_x, p.pbest_f = x_new.copy(), f_new
                    if f_new > gbest.f:
                        gbest = Particle(x=x_new.copy(), v=v_new.copy(), f=f_new,
                                         pbest_x=p.pbest_x.copy(), pbest_f=p.pbest_f)

        # progress
        now = time.time()
        if cfg.progress and (now - last_prog >= cfg.progress_interval):
            elapsed = now - start
            print(
                f"[prog] iter={it}/{cfg.iters} best={gbest.f:.4f}s "
                f"FY1(s={gbest.x[0]:.1f},az={gbest.x[1]:.2f},t={gbest.x[2]:.2f},d={gbest.x[3]:.2f}) "
                f"FY2(s={gbest.x[4]:.1f},az={gbest.x[5]:.2f},t={gbest.x[6]:.2f},d={gbest.x[7]:.2f}) "
                f"FY3(s={gbest.x[8]:.1f},az={gbest.x[9]:.2f},t={gbest.x[10]:.2f},d={gbest.x[11]:.2f}) "
                f"w={w:.2f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            last_prog = now

        # checkpoint
        if cfg.ckpt_every_iter and (it % cfg.ckpt_every_iter == 0) and cfg.checkpoint_file:
            _atomic_save_json(dict(iter=it, timestamp=time.time()) | gbest.as_dict(), cfg.checkpoint_file)

    # final checkpoint
    if cfg.checkpoint_file:
        _atomic_save_json(dict(iter=cfg.iters, timestamp=time.time()) | gbest.as_dict(), cfg.checkpoint_file)

    return gbest


# ---------------- Export helpers ----------------

def export_excel(x: np.ndarray, val: float, path: str):
    try:
        import pandas as pd
    except Exception as e:
        print(f"[warn] pandas not available, cannot write {path}. Please: pip install pandas openpyxl. Error: {e}")
        return
    s1, a1, t1, d1, s2, a2, t2, d2, s3, a3, t3, d3 = [float(z) for z in x]
    rows = [{
        "FY1_speed(m/s)": s1,
        "FY1_azimuth(deg)": math.degrees(a1),
        "FY1_bomb_deploy(s)": t1,
        "FY1_bomb_explode(s)": t1 + d1,
        "FY2_speed(m/s)": s2,
        "FY2_azimuth(deg)": math.degrees(a2),
        "FY2_bomb_deploy(s)": t2,
        "FY2_bomb_explode(s)": t2 + d2,
        "FY3_speed(m/s)": s3,
        "FY3_azimuth(deg)": math.degrees(a3),
        "FY3_bomb_deploy(s)": t3,
        "FY3_bomb_explode(s)": t3 + d3,
        "total_occluded_time(s)": float(val),
    }]
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(path, index=False)
        print(f"Saved result to {path}")
    except Exception as e:
        print(f"[warn] failed to save {path}: {e}")


# ---------------- Driver ----------------

def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        method=cfg['method'], dt=cfg['dt'], rel_max=cfg['rel_max'], delay_max=cfg['delay_max'],
        swarm_size=cfg['swarm_size'], iters=cfg['iters'], w_start=cfg['w_start'], w_end=cfg['w_end'],
        c1=cfg['c1'], c2=cfg['c2'], v_clamp_frac=cfg['v_clamp_frac'], reset_fraction=cfg['reset_fraction'],
        seed=cfg['seed'], progress=cfg['progress'], progress_interval=cfg['progress_interval'],
        verbose=cfg['verbose'], checkpoint_file=cfg['checkpoint_file'], ckpt_every_iter=cfg['ckpt_every_iter'],
        init_seed_file=None,
    )


def main():
    import sys
    use_cli = '--use-cli' in sys.argv
    if not use_cli:
        args = _build_args_via_config()
    else:
        import argparse
        ap = argparse.ArgumentParser(description="PSO for Problem 4 (3 UAVs, 1 bomb each, maximize M1 occlusion)")
        ap.add_argument('--use-cli', action='store_true')
        ap.add_argument('--method', choices=['judge_caps', 'sampling'], default='judge_caps')
        ap.add_argument('--dt', type=float, default=0.05)
        ap.add_argument('--rel-max', type=float, default=66.0)
        ap.add_argument('--delay-max', type=float, default=20.0)
        ap.add_argument('--swarm', dest='swarm_size', type=int, default=60)
        ap.add_argument('--iters', type=int, default=300)
        ap.add_argument('--w-start', type=float, default=0.9)
        ap.add_argument('--w-end', type=float, default=0.45)
        ap.add_argument('--c1', type=float, default=1.8)
        ap.add_argument('--c2', type=float, default=1.8)
        ap.add_argument('--v-clamp-frac', type=float, default=0.25)
        ap.add_argument('--reset-fraction', type=float, default=0.12)
        ap.add_argument('--seed', type=int, default=42)
        ap.add_argument('--progress', action='store_true')
        ap.add_argument('--progress-interval', type=float, default=1.0)
        ap.add_argument('--verbose', action='store_true')
        ap.add_argument('--checkpoint-file', type=str, default='best_p4_pso.json')
        ap.add_argument('--ckpt-every-iter', type=int, default=20)
        ap.add_argument('--init-seed-file', type=str, default='')
        args_cli = ap.parse_args()
        args = SimpleNamespace(
            method=args_cli.method, dt=args_cli.dt, rel_max=args_cli.rel_max, delay_max=args_cli.delay_max,
            swarm_size=args_cli.swarm_size, iters=args_cli.iters, w_start=args_cli.w_start, w_end=args_cli.w_end,
            c1=args_cli.c1, c2=args_cli.c2, v_clamp_frac=args_cli.v_clamp_frac, reset_fraction=args_cli.reset_fraction,
            seed=args_cli.seed, progress=args_cli.progress, progress_interval=args_cli.progress_interval,
            verbose=args_cli.verbose, checkpoint_file=args_cli.checkpoint_file, ckpt_every_iter=args_cli.ckpt_every_iter,
            init_seed_file=(args_cli.init_seed_file or None),
        )

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t0 = time.time()
    best = pso(args)
    t1 = time.time()

    print("=== PSO Result (Problem 4) ===")
    print(f"Best occluded time: {best.f:.4f} s")
    s1, a1, t1r, d1, s2, a2, t2r, d2, s3, a3, t3r, d3 = [float(z) for z in best.x]
    print(f"FY1: speed={s1:.2f} m/s, az={a1:.6f} rad (deg={math.degrees(a1):.2f}), release={t1r:.3f} s, delay={d1:.3f} s")
    print(f"FY2: speed={s2:.2f} m/s, az={a2:.6f} rad (deg={math.degrees(a2):.2f}), release={t2r:.3f} s, delay={d2:.3f} s")
    print(f"FY3: speed={s3:.2f} m/s, az={a3:.6f} rad (deg={math.degrees(a3):.2f}), release={t3r:.3f} s, delay={d3:.3f} s")
    print(f"Runtime: {time.time() - t0:.2f} s | Iters: {args.iters} | Swarm: {args.swarm_size}")

    # verify with sampling if search used judge_caps
    if args.method == 'judge_caps':
        try:
            from optimizer_api import evaluate_problem4 as eval_p4
            speeds, azs, times, delays = decode_position(best.x, args)
            drones_spec = [
                {"pos0": FY1_POS.tolist(), "speed": speeds[0], "azimuth": azs[0], "bombs": [{"deploy_time": times[0], "explode_delay": delays[0]}]},
                {"pos0": FY2_POS.tolist(), "speed": speeds[1], "azimuth": azs[1], "bombs": [{"deploy_time": times[1], "explode_delay": delays[1]}]},
                {"pos0": FY3_POS.tolist(), "speed": speeds[2], "azimuth": azs[2], "bombs": [{"deploy_time": times[2], "explode_delay": delays[2]}]},
            ]
            v2 = float(eval_p4(drones_spec=drones_spec, dt=args.dt, occlusion_method='sampling')["occluded_time"]["M1"]) 
            print(f"(Sampling verification) Occluded time: {v2:.4f} s")
        except Exception as e:
            print(f"Sampling verification failed: {e}")

    # export to Excel
    try:
        export_excel(best.x, best.f, 'result2.xlsx')
    except Exception as e:
        print(f"Export failed: {e}")


if __name__ == '__main__':
    main()
