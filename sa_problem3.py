from __future__ import annotations
"""
Simulated Annealing solver for Problem 3:
  Optimize FY1 (single UAV, three smoke bombs) parameters to maximize M1 occlusion time.
Decision variables:
  - speed ∈ [70, 140] m/s (fixed throughout flight)
  - azimuth ∈ [0, 2π)  (horizontal heading, x-axis = 0, CCW positive)
  - For each of 3 bombs i ∈ {1,2,3}:
      - deploy_time_i ∈ [0, rel_max]
      - explode_delay_i ∈ [0, delay_max]
    Subject to: deploy_time_{i+1} - deploy_time_i ≥ 1.0 seconds
Objective:
  Maximize total occluded time (seconds) on M1 (overlaps not double-counted) using evaluate_problem3().

Usage:
  python sa_problem3.py  # use CONFIG below
  python sa_problem3.py --use-cli --iters 40000 --method judge_caps --seed 42  # optional CLI

Notes:
  - judge_caps 解析法较快（仅端面两圆, 较保守）；sampling 更精确但慢。
  - 支持周期性保存最优解 (--checkpoint-file, --ckpt-steps / --ckpt-seconds)
  - 使用与问题2一致的高级探索（Cauchy / mixed 邻域、重热、全局跳跃、重启等）
  - 结果将导出到 result1.xlsx（若缺少 pandas/openpyxl 会提示安装）
"""
import argparse
import math
import os
import json
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np

from optimizer_api import evaluate_problem3

# ===================== 用户可直接修改的配置区域 =====================
CONFIG = dict(
    # 基础仿真 / 目标
    method="judge_caps",          # 遮蔽判定方法: 'judge_caps' (快, 保守) | 'sampling' (慢, 精细)
    dt=0.02,                       # 仿真时间步 (s) 适中:0.05  精细:0.02  粗略:0.1
    rel_max=66.0,                  # 投放时间上界 (s)
    delay_max=20.0,                # 起爆延迟上界 (s)

    # 初始随机解与退火主控
    t0=1.2,                        # 初始温度
    t_end=1e-3,                    # 终止温度阈值
    alpha=0.988,                   # 每轮降温因子
    steps_per_t=60,                # 每个温度水平的 Metropolis 迭代次数
    max_steps=200000,              # 总步数上限
    seed=42,                       # 随机种子 (None=使用系统随机)

    # 邻域与探索强度
    neighbor_mode="mixed",        # 'gauss' | 'cauchy' | 'mixed'
    cauchy_scale=1.2,              # Cauchy 重尾尺度 (mixed 或 cauchy 模式生效)
    mixed_gauss_prob=0.45,         # mixed 模式下使用高斯的概率
    temp_scale=1.25,               # 额外放大 (温度*temp_scale) 以增大步长

    # 全局跳跃 / 重启 / 重热
    global_jump_prob=0.05,         # 全局跳跃概率（尝试全新随机解）
    accept_worse_jump=True,        # 全局跳跃是否允许更差也替换当前位置
    restart_every=15000,           # 硬重启周期 (0=关闭)
    reheat_every=5000,             # 周期性重热 (0=关闭)
    reheat_factor=1.8,             # 重热时温度乘以该因子 (上限 t0)
    auto_reheat=True,              # 是否启用停滞自动重热
    stag_reheat_steps=1500,        # 连续未提升多少步触发自动重热

    # 日志/进度/保存
    progress=True,                 # 是否显示周期进度行
    progress_interval=1.0,         # 进度输出时间间隔 (秒)
    verbose=False,                 # 输出详细 step 级日志 (配合 log_every)
    log_every=500,                 # 每多少步输出一次 verbose 行
    checkpoint_file="best_p3.json", # 最优解保存文件 (空字符串代表不保存)
    ckpt_steps=200000,             # 每 N 步保存 (0=关闭)
    ckpt_seconds=60,               # 每 N 秒保存 (0=关闭)
    ckpt_on_improve=True,          # 一旦提升立即保存
)
# ================== 结束：只需修改上面 CONFIG ======================

# Bounds / defaults
SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi
RELEASE_MAX_DEFAULT = 66.0
DELAY_MIN, DELAY_MAX_DEFAULT = 0.0, 20.0
MIN_GAP = 1.0  # 相邻两枚投放间隔 ≥ 1 s

# Annealing defaults
T0_DEFAULT = 1.0
T_END_DEFAULT = 1e-3
ALPHA_DEFAULT = 0.98
STEPS_PER_T_DEFAULT = 50
NO_IMPROVE_STOP = 200000

# ---------------- 工具函数 ----------------

def _rand_cauchy(scale: float) -> float:
    u = random.random()
    return math.tan(math.pi * (u - 0.5)) * scale


def _atomic_save_json(obj: Dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _save_checkpoint(best: 'Solution', steps: int, elapsed: float, path: str):
    data = best.as_dict()
    data.update(dict(steps=steps, elapsed_sec=elapsed, timestamp=time.time()))
    try:
        _atomic_save_json(data, path)
    except Exception as e:
        print(f"[warn] checkpoint save failed: {e}")


def clip(x, lo, hi):
    return min(max(x, lo), hi)


def wrap_angle(a: float) -> float:
    return a % (2.0 * math.pi)


def _sample_release_times(n: int, rel_max: float, min_gap: float = MIN_GAP) -> List[float]:
    """Sample n nondecreasing times within [0, rel_max] with minimum gap min_gap.
    Construction: choose base in [0, rel_max - (n-1)*min_gap], then distribute slack
    as a nondecreasing cumulative sequence.
    """
    if rel_max <= 0:
        return [0.0] * n
    base_hi = max(0.0, rel_max - (n - 1) * min_gap)
    base = random.uniform(0.0, base_hi)
    slack = max(0.0, rel_max - (base + (n - 1) * min_gap))
    # Build nondecreasing cumulative extras within [0, slack]
    us = sorted(random.random() for _ in range(n))
    cum = [0.0 if us[-1] == 0 else slack * (u / us[-1]) for u in us]
    return [base + i * min_gap + cum[i] for i in range(n)]


def _project_times(times: List[float], rel_max: float, min_gap: float = MIN_GAP) -> List[float]:
    """Project times to satisfy 0<=t<=rel_max, sorted, and min gaps.
    Forward-backward pass, then shift inside bounds if necessary.
    """
    n = len(times)
    t = sorted(float(x) for x in times)
    # Forward pass: enforce gaps
    for i in range(1, n):
        t[i] = max(t[i], t[i - 1] + min_gap)
    # Clamp last to rel_max then backward pass if overflow
    if t[-1] > rel_max:
        t[-1] = rel_max
        for i in range(n - 2, -1, -1):
            t[i] = min(t[i], t[i + 1] - min_gap)
        # If earliest < 0, shift right and re-enforce
        if t[0] < 0.0:
            shift = -t[0]
            t = [ti + shift for ti in t]
            for i in range(1, n):
                t[i] = max(t[i], t[i - 1] + min_gap)
            if t[-1] > rel_max:
                # As a fallback, resample feasible times
                return _sample_release_times(n, rel_max, min_gap)
    # Final clamp to [0, rel_max]
    t[0] = clip(t[0], 0.0, rel_max)
    for i in range(1, n):
        t[i] = clip(max(t[i], t[i - 1] + min_gap), 0.0, rel_max)
    # If still infeasible due to rounding, resample
    if t[-1] > rel_max + 1e-9:
        return _sample_release_times(n, rel_max, min_gap)
    return t


@dataclass
class Solution:
    speed: float
    azimuth: float
    bombs: List[Tuple[float, float]]  # [(deploy_time, explode_delay)] length=3
    value: float  # objective value (occluded time)

    def as_dict(self) -> Dict:
        d = {
            "speed": self.speed,
            "azimuth": self.azimuth,
            "bombs": [dict(deploy_time=t, explode_delay=dd) for (t, dd) in self.bombs],
            "value": self.value,
        }
        return d


# ---------------- 评价与解生成 ----------------

def evaluate(speed: float, azimuth: float, bombs: List[Tuple[float, float]], method: str, dt: float) -> float:
    res = evaluate_problem3(bombs=bombs, speed=speed, azimuth=azimuth, dt=dt, occlusion_method=method)
    return float(res["occluded_time"]["M1"])


def random_initial(rel_max: float, delay_max: float, method: str, dt: float) -> Solution:
    base_speed = 120.0
    base_az = math.pi  # 指向假目标的大致反向
    speed = clip(random.gauss(base_speed, 10.0), SPEED_MIN, SPEED_MAX)
    azimuth = wrap_angle(base_az + random.gauss(0.0, 0.15))
    deploys = _sample_release_times(3, rel_max, MIN_GAP)
    delays = [random.uniform(2.0, min(6.0, delay_max)) for _ in range(3)]
    bombs = list(zip(deploys, delays))
    val = evaluate(speed, azimuth, bombs, method, dt)
    return Solution(speed, azimuth, bombs, val)


def neighbor(sol: Solution, rel_max: float, delay_max: float, method: str, dt: float, temp: float,
            mode: str = "mixed", cauchy_base: float = 1.0, temp_scale: float = 1.0, mix_p: float = 0.5) -> Solution:
    """Enhanced neighbor with selectable distribution for triple-bomb schedule.
    mode: 'gauss' | 'cauchy' | 'mixed'
    """
    scale = max(temp * temp_scale, 1e-4)

    def d_gauss(s):
        return random.gauss(0.0, s)

    def d_cauchy(s):
        return _rand_cauchy(s)

    def pick(s_short: float, s_long: float):
        if mode == "gauss":
            return d_gauss(s_short)
        elif mode == "cauchy":
            return d_cauchy(s_long)
        else:
            return d_gauss(s_short) if random.random() < mix_p else d_cauchy(s_long)

    spd = clip(sol.speed + pick(8.0 * scale, cauchy_base * 15.0 * scale), SPEED_MIN, SPEED_MAX)
    az = wrap_angle(sol.azimuth + pick(0.5 * scale, cauchy_base * 1.2 * scale))

    times = [t for (t, _) in sol.bombs]
    delays = [d for (_, d) in sol.bombs]

    # 时间与延时的扰动（不同维度不同尺度）
    times = [t + pick(2.0 * scale, cauchy_base * 4.0 * scale) for t in times]
    delays = [clip(d + pick(1.0 * scale, cauchy_base * 2.5 * scale), DELAY_MIN, delay_max) for d in delays]

    # 约束投放时间：区间与最小间隔
    times = _project_times(times, rel_max, MIN_GAP)

    bombs = list(zip(times, delays))
    val = evaluate(spd, az, bombs, method, dt)
    return Solution(spd, az, bombs, val)


# ---------------- 退火主程序 ----------------

def simulated_annealing(args) -> Solution:
    random.seed(args.seed)
    np.random.seed(args.seed)

    best = current = random_initial(args.rel_max, args.delay_max, args.method, args.dt)
    t = args.t0
    steps = 0
    last_improve_step = 0
    stagnation_counter = 0

    log_every = max(1, args.log_every)

    start_time = time.time()
    last_progress_time = start_time
    accepted_moves_period = 0
    total_moves_period = 0

    last_ckpt_steps = 0
    last_ckpt_time = start_time

    if args.checkpoint_file:
        _save_checkpoint(best, steps, 0.0, args.checkpoint_file)

    while t > args.t_end and steps < args.max_steps:
        for _ in range(args.steps_per_t):
            steps += 1

            # 重启
            if args.restart_every and steps % args.restart_every == 0 and steps > 0:
                if args.verbose or args.progress:
                    print(f"[restart] step={steps} keep best={best.value:.4f}")
                current = random_initial(args.rel_max, args.delay_max, args.method, args.dt)

            # 全局跳跃
            if args.global_jump_prob > 0 and random.random() < args.global_jump_prob:
                nj = random_initial(args.rel_max, args.delay_max, args.method, args.dt)
                if args.accept_worse_jump or nj.value >= current.value:
                    current = nj

            cand = neighbor(
                current,
                args.rel_max,
                args.delay_max,
                args.method,
                args.dt,
                t,
                mode=args.neighbor_mode,
                cauchy_base=args.cauchy_scale,
                temp_scale=args.temp_scale,
                mix_p=args.mixed_gauss_prob,
            )

            delta = cand.value - current.value
            accepted = False
            if delta >= 0 or math.exp(delta / max(t, 1e-9)) > random.random():
                current = cand
                accepted = True

            if cand.value > best.value + 1e-12:
                best = cand
                last_improve_step = steps
                stagnation_counter = 0
                if args.checkpoint_file and args.ckpt_on_improve:
                    _save_checkpoint(best, steps, time.time() - start_time, args.checkpoint_file)
            else:
                stagnation_counter += 1

            # 自适应重热
            if args.auto_reheat and stagnation_counter >= args.stag_reheat_steps:
                old_t = t
                t = min(t * args.reheat_factor, args.t0)
                stagnation_counter = 0
                if args.verbose or args.progress:
                    print(f"[reheat] step={steps} T {old_t:.4g} -> {t:.4g}")

            # 周期重热
            if args.reheat_every and (steps % args.reheat_every == 0):
                old_t = t
                t = min(t * args.reheat_factor, args.t0)
                if args.verbose or args.progress:
                    print(f"[periodic reheat] step={steps} T {old_t:.4g} -> {t:.4g}")

            total_moves_period += 1
            if accepted:
                accepted_moves_period += 1

            # 进度输出
            now = time.time()
            if args.progress and (now - last_progress_time >= args.progress_interval):
                elapsed = now - start_time
                frac = steps / args.max_steps if args.max_steps > 0 else 0.0
                eta = elapsed * (1 - frac) / frac if frac > 1e-6 else float('nan')
                acc_rate = (accepted_moves_period / max(1, total_moves_period))
                print(
                    f"[prog] {steps}/{args.max_steps} {frac:6.2%} T={t:.4g} best={best.value:.4f}s "
                    f"cur={current.value:.4f}s acc={acc_rate:5.1%} elapsed={elapsed:6.1f}s ETA={eta:6.1f}s",
                    flush=True,
                )
                last_progress_time = now
                accepted_moves_period = 0
                total_moves_period = 0

            # 周期 checkpoint
            if args.checkpoint_file:
                do_ckpt = False
                if args.ckpt_steps and (steps - last_ckpt_steps) >= args.ckpt_steps:
                    do_ckpt = True
                if args.ckpt_seconds and (time.time() - last_ckpt_time) >= args.ckpt_seconds:
                    do_ckpt = True
                if do_ckpt:
                    _save_checkpoint(best, steps, time.time() - start_time, args.checkpoint_file)
                    last_ckpt_steps = steps
                    last_ckpt_time = time.time()

            if steps >= args.max_steps:
                break

        t *= args.alpha
        if (steps - last_improve_step) >= NO_IMPROVE_STOP:
            if args.verbose or args.progress:
                print(f"Early stop: {NO_IMPROVE_STOP} steps no improvement.")
            break

    if args.checkpoint_file:
        _save_checkpoint(best, steps, time.time() - start_time, args.checkpoint_file)
    return best


def _build_args_via_config() -> SimpleNamespace:
    cfg = CONFIG.copy()
    return SimpleNamespace(
        method=cfg['method'], dt=cfg['dt'], rel_max=cfg['rel_max'], delay_max=cfg['delay_max'],
        t0=cfg['t0'], t_end=cfg['t_end'], alpha=cfg['alpha'], steps_per_t=cfg['steps_per_t'],
        max_steps=cfg['max_steps'], seed=cfg['seed'], neighbor_mode=cfg['neighbor_mode'],
        cauchy_scale=cfg['cauchy_scale'], mixed_gauss_prob=cfg['mixed_gauss_prob'],
        temp_scale=cfg['temp_scale'], global_jump_prob=cfg['global_jump_prob'],
        accept_worse_jump=cfg['accept_worse_jump'], restart_every=cfg['restart_every'],
        reheat_every=cfg['reheat_every'], reheat_factor=cfg['reheat_factor'],
        auto_reheat=cfg['auto_reheat'], stag_reheat_steps=cfg['stag_reheat_steps'],
        progress=cfg['progress'], progress_interval=cfg['progress_interval'],
        verbose=cfg['verbose'], log_every=cfg['log_every'], checkpoint_file=cfg['checkpoint_file'],
        ckpt_steps=cfg['ckpt_steps'], ckpt_seconds=cfg['ckpt_seconds'], ckpt_on_improve=cfg['ckpt_on_improve'],
        iters=None,
    )


def _export_excel(best: Solution, path: str, total_time: float):
    """Export the best strategy to an Excel file.
    Columns include speed, azimuth(deg), for each bomb deploy/explode times, and total occluded time.
    If pandas/openpyxl is unavailable, print a hint.
    """
    try:
        import pandas as pd
    except Exception as e:
        print(f"[warn] pandas not available, cannot write {path}. Please: pip install pandas openpyxl. Error: {e}")
        return

    az_deg = math.degrees(best.azimuth)
    rows = [{
        "speed(m/s)": best.speed,
        "azimuth(deg)": az_deg,
        "bomb1_deploy(s)": best.bombs[0][0],
        "bomb1_explode(s)": best.bombs[0][0] + best.bombs[0][1],
        "bomb2_deploy(s)": best.bombs[1][0],
        "bomb2_explode(s)": best.bombs[1][0] + best.bombs[1][1],
        "bomb3_deploy(s)": best.bombs[2][0],
        "bomb3_explode(s)": best.bombs[2][0] + best.bombs[2][1],
        "total_occluded_time(s)": total_time,
    }]
    df = pd.DataFrame(rows)
    try:
        df.to_excel(path, index=False)
        print(f"Saved result to {path}")
    except Exception as e:
        print(f"[warn] failed to save {path}: {e}")


def main():
    import sys
    use_cli = '--use-cli' in sys.argv
    if not use_cli:
        args = _build_args_via_config()
    else:
        ap = argparse.ArgumentParser(description="Simulated Annealing solver for Problem 3 (maximize M1 occlusion time, 3 bombs)")
        ap.add_argument('--use-cli', action='store_true', help='explicitly use CLI args (internal)')
        ap.add_argument("--method", choices=["judge_caps", "sampling"], default="judge_caps")
        ap.add_argument("--dt", type=float, default=0.05)
        ap.add_argument("--rel-max", type=float, default=RELEASE_MAX_DEFAULT)
        ap.add_argument("--delay-max", type=float, default=DELAY_MAX_DEFAULT)
        ap.add_argument("--t0", type=float, default=T0_DEFAULT)
        ap.add_argument("--t-end", type=float, default=T_END_DEFAULT)
        ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
        ap.add_argument("--steps-per-t", type=int, default=STEPS_PER_T_DEFAULT)
        ap.add_argument("--max-steps", type=int, default=200000)
        ap.add_argument("--iters", type=int, default=None)
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--verbose", action="store_true")
        ap.add_argument("--log-every", type=int, default=200)
        ap.add_argument("--progress", action="store_true")
        ap.add_argument("--progress-interval", type=float, default=1.0)
        ap.add_argument("--checkpoint-file", type=str, default="")
        ap.add_argument("--ckpt-steps", type=int, default=0)
        ap.add_argument("--ckpt-seconds", type=float, default=0.0)
        ap.add_argument("--ckpt-on-improve", action="store_true")
        ap.add_argument("--neighbor-mode", choices=["gauss", "cauchy", "mixed"], default="mixed")
        ap.add_argument("--cauchy-scale", type=float, default=1.0)
        ap.add_argument("--mixed-gauss-prob", type=float, default=0.5)
        ap.add_argument("--temp-scale", type=float, default=1.0)
        ap.add_argument("--global-jump-prob", type=float, default=0.01)
        ap.add_argument("--accept-worse-jump", action="store_true")
        ap.add_argument("--reheat-every", type=int, default=0)
        ap.add_argument("--reheat-factor", type=float, default=1.5)
        ap.add_argument("--auto-reheat", action="store_true")
        ap.add_argument("--stag-reheat-steps", type=int, default=1500)
        ap.add_argument("--restart-every", type=int, default=0)
        args = ap.parse_args()
        if args.iters is not None:
            args.max_steps = args.iters
        if not args.checkpoint_file:
            args.checkpoint_file = ""

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t_start = time.time()
    best = simulated_annealing(args)
    t_end = time.time()

    print("=== Simulated Annealing Result (Problem 3, 3 bombs) ===")
    print(f"Best occluded time: {best.value:.4f} s")
    print(f"Speed: {best.speed:.3f} m/s")
    print(f"Azimuth: {best.azimuth:.6f} rad  (deg={math.degrees(best.azimuth):.2f})")
    for i, (t_rel, dly) in enumerate(best.bombs, 1):
        print(f"Bomb{i}: deploy={t_rel:.3f} s, explode_delay={dly:.3f} s (explode @ {t_rel + dly:.3f} s)")

    # 验证（如采用 judge_caps 则用 sampling 再评一次）
    if args.method == "judge_caps":
        try:
            res_sampling = evaluate_problem3(
                bombs=best.bombs, speed=best.speed, azimuth=best.azimuth, dt=args.dt, occlusion_method="sampling"
            )
            v2 = float(res_sampling["occluded_time"]["M1"])
            print(f"(Sampling verification) Occluded time: {v2:.4f} s")
        except Exception as e:
            print(f"Sampling verification failed: {e}")

    print(f"Runtime: {t_end - t_start:.2f} s | Steps: {args.max_steps}")

    # 导出 Excel 结果
    try:
        total_time = float(best.value)
        _export_excel(best, path="result1.xlsx", total_time=total_time)
    except Exception as e:
        print(f"Export failed: {e}")


if __name__ == "__main__":
    main()
