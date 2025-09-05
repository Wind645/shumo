from __future__ import annotations
"""
Simulated Annealing solver for Problem 2:
  Optimize FY1 (single UAV, single smoke bomb) parameters to maximize M1 occlusion time.
Decision variables:
  - speed ∈ [70, 140] m/s
  - azimuth ∈ [0, 2π)  (horizontal heading, x-axis = 0, CCW positive)
  - release_time ∈ [0, T_rel_max]
  - explode_delay ∈ [delay_min, delay_max]
Objective:
  Maximize total occluded time (seconds) returned by evaluate_problem2().

Usage:
  python sa_problem2.py --iters 4000 --method judge_caps --seed 42

Notes:
  - judge_caps 解析法较快（仅端面两圆, 较保守）；sampling 更精确但慢。
  - You can raise --rel-max / --delay-max to enlarge search space.
  - 支持周期性保存最优解 (--checkpoint-file, --ckpt-steps / --ckpt-seconds)
  - 新增高级探索参数：重热 / Cauchy 邻域 / 随机跳跃 / 重启
  - 现在支持直接在代码顶部配置参数 (CONFIG)，无需命令行。
    你只需修改 CONFIG 字典即可调整优化行为。
"""
import argparse  # 仍保留，但默认不再使用命令行
import math
import random
import time
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple, Dict

import numpy as np

from optimizer_api import evaluate_problem2

# ===================== 用户可直接修改的配置区域 =====================
# 每个键后面给出含义与推荐范围。修改后直接: python3 sa_problem2.py 运行即可。
CONFIG = dict(
    # 基础仿真 / 目标
    method="judge_caps",          # 遮蔽判定方法: 'judge_caps' (快, 保守) 或 'sampling' (慢, 精细)
    dt=0.05,                       # 仿真时间步 (s) 适中:0.05  精细:0.02  粗略:0.1
    rel_max=66.0,                  # 投放时间搜索上界 (s)
    delay_max=20.0,                # 起爆延迟搜索上界 (s)

    # 初始随机解与退火主控
    t0=1.2,                        # 初始温度 (越大越容易接受差解)
    t_end=1e-3,                    # 终止温度阈值
    alpha=0.998,                   # 每轮降温因子 (越接近1 降温越慢)
    steps_per_t=60,                # 每个温度水平的 Metropolis 迭代次数
    max_steps=25000,               # 总步数上限 (主控迭代预算)
    seed=42,                       # 随机种子 (改为 None 使用系统随机)

    # 邻域与探索强度
    neighbor_mode="mixed",        # 'gauss' | 'cauchy' | 'mixed' (推荐 mixed)
    cauchy_scale=1.2,              # Cauchy 重尾尺度 (mixed 或 cauchy 模式生效)
    mixed_gauss_prob=0.45,         # mixed 模式下使用高斯的概率
    temp_scale=1.25,               # 额外放大 (温度*temp_scale) 以增大步长

    # 全局跳跃 / 重启 / 重热
    global_jump_prob=0.05,         # 每步触发一次“全新随机解”尝试的概率
    accept_worse_jump=True,        # 全局跳跃是否允许更差也替换当前位置
    restart_every=15000,           # 硬重启周期 (0=关闭)
    reheat_every=5000,             # 周期性重热 (0=关闭)
    reheat_factor=1.8,             # 重热时温度乘以该因子 (上限 t0)
    auto_reheat=True,              # 是否启用停滞自动重热
    stag_reheat_steps=1500,        # 连续未提升多少步触发自动重热

    # 早停判据 (继承 NO_IMPROVE_STOP 逻辑)
    # NO_IMPROVE_STOP 在代码下方常量处，可按需改。

    # 日志/进度/保存
    progress=True,                 # 是否显示周期进度行
    progress_interval=1.0,         # 进度输出时间间隔 (秒)
    verbose=False,                 # 输出详细 step 级日志 (配合 log_every)
    log_every=500,                 # 每多少步输出一次 verbose 行
    checkpoint_file="best_p2.json", # 最优解保存文件 (空字符串代表不保存)
    ckpt_steps=2000,               # 每 N 步保存 (0=关闭)
    ckpt_seconds=0,                # 每 N 秒保存 (0=关闭)
    ckpt_on_improve=True,          # 一旦提升立即保存
)
# ================== 结束：只需修改上面 CONFIG ======================

# Bounds / defaults
SPEED_MIN, SPEED_MAX = 70.0, 140.0
AZIM_MIN, AZIM_MAX = 0.0, 2.0 * math.pi
RELEASE_MAX_DEFAULT = 66.0          # 可调: 最大投放时间 (s)
DELAY_MIN, DELAY_MAX_DEFAULT = 0.0, 20.0   # 起爆延迟范围 (s)

# Annealing defaults
T0_DEFAULT = 1.0        # 初始温度 (对单位=秒的 occlusion_time, 一般 0~若干秒)
T_END_DEFAULT = 1e-3
ALPHA_DEFAULT = 0.985   # 降温因子
STEPS_PER_T_DEFAULT = 50
NO_IMPROVE_STOP = 200000  # 早停: 若超此步无提升

# ---------------- 高级扰动 / 退火策略辅助 ----------------

def _rand_cauchy(scale: float) -> float:
    # 标准 Cauchy 变量: tan(pi*(U-0.5))，再缩放
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

@dataclass
class Solution:
    speed: float
    azimuth: float
    release_time: float
    explode_delay: float
    value: float  # objective (occluded time)

    def as_dict(self) -> Dict:
        return dict(speed=self.speed, azimuth=self.azimuth, release_time=self.release_time,
                    explode_delay=self.explode_delay, value=self.value)


def clip(x, lo, hi):
    return min(max(x, lo), hi)


def wrap_angle(a: float) -> float:
    twopi = 2.0 * math.pi
    return a % twopi


def evaluate(speed: float, azimuth: float, release_time: float, explode_delay: float, method: str, dt: float) -> float:
    """Return occluded time for M1 under given decision."""
    res = evaluate_problem2(
        speed=speed,
        azimuth=azimuth,
        release_time=release_time,
        explode_delay=explode_delay,
        occlusion_method=method,
        dt=dt,
    )
    return float(res["occluded_time"]["M1"])


def random_initial(rel_max: float, delay_max: float, method: str, dt: float) -> Solution:
    # Heuristic: start near baseline (120 m/s, heading to fake target -> pi) with slight perturbation
    base_speed = 120.0
    base_az = math.pi  # FY1 initial pos (17800,0,1800) -> fake target (0,0,0) projection ≈ negative x
    speed = clip(random.gauss(base_speed, 10.0), SPEED_MIN, SPEED_MAX)
    azimuth = wrap_angle(base_az + random.gauss(0.0, 0.15))
    release_time = random.uniform(0.5, min(5.0, rel_max))  # early releases often good
    explode_delay = random.uniform(2.0, min(6.0, delay_max))
    val = evaluate(speed, azimuth, release_time, explode_delay, method, dt)
    return Solution(speed, azimuth, release_time, explode_delay, val)


def neighbor(sol: Solution, rel_max: float, delay_max: float, method: str, dt: float, temp: float,
            mode: str = "gauss", cauchy_base: float = 1.0, temp_scale: float = 1.0,
            mix_p: float = 0.5) -> Solution:
    """Enhanced neighbor with selectable distribution.
    mode: 'gauss' | 'cauchy' | 'mixed'
    cauchy_base: base scale for Cauchy heavy-tail
    temp_scale: multiplier on temperature for step size
    mix_p: probability of choosing gaussian in mixed mode
    """
    # 温度缩放 + 用户外部缩放
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
        else:  # mixed
            return d_gauss(s_short) if random.random() < mix_p else d_cauchy(s_long)

    # 不同维度设置不同的基准尺度（经验）
    spd = sol.speed + pick(8.0 * scale, cauchy_base * 15.0 * scale)
    az  = sol.azimuth + pick(0.5 * scale, cauchy_base * 1.2 * scale)
    rel = sol.release_time + pick(2.0 * scale, cauchy_base * 4.0 * scale)
    dly = sol.explode_delay + pick(1.0 * scale, cauchy_base * 2.5 * scale)

    speed = clip(spd, SPEED_MIN, SPEED_MAX)
    azimuth = wrap_angle(az)
    release_time = clip(rel, 0.0, rel_max)
    explode_delay = clip(dly, DELAY_MIN, delay_max)

    val = evaluate(speed, azimuth, release_time, explode_delay, method, dt)
    return Solution(speed, azimuth, release_time, explode_delay, val)

# 旧 neighbor 保留兼容
def neighbor_old(sol: Solution, rel_max: float, delay_max: float, method: str, dt: float, temp: float) -> Solution:
    # Temperature-scaled perturbations
    scale = max(temp, 1e-3)
    speed = clip(sol.speed + random.gauss(0.0, 8.0 * scale), SPEED_MIN, SPEED_MAX)
    azimuth = wrap_angle(sol.azimuth + random.gauss(0.0, 0.5 * scale))
    release_time = clip(sol.release_time + random.gauss(0.0, 2.0 * scale), 0.0, rel_max)
    explode_delay = clip(sol.explode_delay + random.gauss(0.0, 1.0 * scale), DELAY_MIN, delay_max)
    # Ensure explosion occurs within some reasonable horizon (missile flight ~67 s) - implicit by bounds
    val = evaluate(speed, azimuth, release_time, explode_delay, method, dt)
    return Solution(speed, azimuth, release_time, explode_delay, val)


def simulated_annealing(args) -> Solution:
    random.seed(args.seed)
    np.random.seed(args.seed)

    best = current = random_initial(args.rel_max, args.delay_max, args.method, args.dt)
    t = args.t0
    steps = 0
    last_improve_step = 0
    last_best_value = best.value
    stagnation_counter = 0

    log_every = max(1, args.log_every)

    # 进度&接受率统计
    start_time = time.time()
    last_progress_time = start_time
    accepted_moves_period = 0
    total_moves_period = 0

    last_ckpt_steps = 0
    last_ckpt_time = start_time

    # 初始立即保存一次（若指定）
    if args.checkpoint_file:
        _save_checkpoint(best, steps, 0.0, args.checkpoint_file)

    while t > args.t_end and steps < args.max_steps:
        for _ in range(args.steps_per_t):
            steps += 1

            # 触发重启（硬重启: 重新随机当前位置，保留 best）
            if args.restart_every and steps % args.restart_every == 0 and steps > 0:
                if args.verbose or args.progress:
                    print(f"[restart] step={steps} keep best={best.value:.4f}")
                current = random_initial(args.rel_max, args.delay_max, args.method, args.dt)

            # 偶发全局跳跃（soft jump 基于概率）
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
                mix_p=args.mixed_gauss_prob
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

            # 自适应重热: 连续停滞
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

            # 更新统计
            total_moves_period += 1
            if accepted:
                accepted_moves_period += 1

            # 常规 verbose 日志
            if steps % log_every == 0 and args.verbose:
                print(f"[step {steps:05d}] T={t:.4f} cur={current.value:.4f} best={best.value:.4f} "
                      f"(spd={best.speed:.2f}, az={best.azimuth:.3f}, rel={best.release_time:.2f}, delay={best.explode_delay:.2f})")

            # 进度输出
            now = time.time()
            if args.progress and (now - last_progress_time >= args.progress_interval):
                elapsed = now - start_time
                frac = steps / args.max_steps if args.max_steps > 0 else 0.0
                eta = elapsed * (1 - frac) / frac if frac > 1e-6 else float('nan')
                acc_rate = (accepted_moves_period / max(1, total_moves_period))
                print(f"[prog] {steps}/{args.max_steps} {frac:6.2%} T={t:.4g} best={best.value:.4f}s cur={current.value:.4f}s "
                      f"acc={acc_rate:5.1%} elapsed={elapsed:6.1f}s ETA={eta:6.1f}s", flush=True)
                last_progress_time = now
                accepted_moves_period = 0
                total_moves_period = 0

            # 周期性 checkpoint
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
        # 正常降温
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
    # 兼容旧字段名称 -> args 属性名
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
        # 兼容旧接口中未使用但代码访问的占位
        iters=None
    )


def main():
    import sys
    use_cli = '--use-cli' in sys.argv  # 若命令行包含 --use-cli 则启用原 argparse
    if not use_cli:
        args = _build_args_via_config()
    else:
        # 原 argparse 流程 (保留以防需要临时实验)
        ap = argparse.ArgumentParser(description="Simulated Annealing solver for Problem 2 (maximize M1 occlusion time)")
        ap.add_argument('--use-cli', action='store_true', help='explicitly use CLI args (internal)')
        ap.add_argument("--method", choices=["judge_caps", "sampling"], default="judge_caps")
        ap.add_argument("--dt", type=float, default=0.05)
        ap.add_argument("--rel-max", type=float, default=RELEASE_MAX_DEFAULT)
        ap.add_argument("--delay-max", type=float, default=DELAY_MAX_DEFAULT)
        ap.add_argument("--t0", type=float, default=T0_DEFAULT)
        ap.add_argument("--t-end", type=float, default=T_END_DEFAULT)
        ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
        ap.add_argument("--steps-per-t", type=int, default=STEPS_PER_T_DEFAULT)
        ap.add_argument("--max-steps", type=int, default=20000)
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

    # 兼容: 若配置 seed 为 None, 则使用当前时间随机
    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000

    t_start = time.time()
    best = simulated_annealing(args)
    t_end = time.time()

    print("=== Simulated Annealing Result (Problem 2) ===")
    print(f"Best occluded time: {best.value:.4f} s")
    print(f"Speed: {best.speed:.3f} m/s")
    print(f"Azimuth: {best.azimuth:.6f} rad  (deg={math.degrees(best.azimuth):.2f})")
    print(f"Release time: {best.release_time:.3f} s")
    print(f"Explode delay: {best.explode_delay:.3f} s (explode @ {best.release_time + best.explode_delay:.3f} s)")

    if args.method == "judge_caps":
        try:
            res_sampling = evaluate_problem2(
                speed=best.speed, azimuth=best.azimuth,
                release_time=best.release_time, explode_delay=best.explode_delay,
                occlusion_method="sampling", dt=args.dt
            )
            v2 = float(res_sampling["occluded_time"]["M1"])
            print(f"(Sampling verification) Occluded time: {v2:.4f} s")
        except Exception as e:
            print(f"Sampling verification failed: {e}")

    print(f"Runtime: {t_end - t_start:.2f} s | Steps: {args.max_steps}")

if __name__ == "__main__":
    main()
