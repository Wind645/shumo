from __future__ import annotations
import math
import random
import time
import json
from dataclasses import dataclass
from typing import Dict, Optional

from optimizer_api import evaluate_problem2

# --------------------------- 边界和默认值 ---------------------------
SPEED_MIN, SPEED_MAX = 70.0, 140.0
DELAY_MIN = 0.0

# 搜索空间的默认值
RELEASE_MAX_DEFAULT = 66.0
DELAY_MAX_DEFAULT = 20.0
DT_DEFAULT = 0.01
METHOD_DEFAULT = "judge_caps"  # 'judge_caps' | 'sampling'

# 每个角度局部搜索的 SA 默认值（小波动）
T0_LOCAL = 0.6
T_END_LOCAL = 1e-3
ALPHA_LOCAL = 0.98
STEPS_PER_T_LOCAL = 40
MAX_STEPS_PER_ANGLE = 1  # 每个角度的默认总 SA 步数
GLOBAL_JUMP_PROB = 0.02   # 偶尔在边界内进行更广泛的跳跃

# 如果在此角度找不到任何非零遮挡，则重启策略
MAX_RESTARTS_NO_NONZERO = 1

# --------------------------- 解析角度带 ---------------------------
# 严格标准：在同一时间 t，使 x_u(t)=x_m(t)。在那时，侧向偏移是 y = v t sin(theta)，t = 2200 / (300 + v cos(theta))。对于给定的 theta，v∈[vmin,vmax] 上的最小可能 |y| 在 v=vmin 时达到（因为 v/(300+v cosθ) 在 v 上严格增加）。因此，存在一些速度产生侧向偏差 ≤ R 在等 x 时间上的必要且充分条件是：
#   y_min(theta) = 2200*vmin*|sinθ| / (300 + vmin*cosθ) ≤ R。
# 我们只枚举满足此不等式的角度。

def equal_x_lateral_band_indices(start_ang: float, end_ang: float, step: float,
                                 R: float = 10.0, vmin: float = SPEED_MIN):
    total = int(round((end_ang - start_ang) / max(step, 1e-6))) + 1
    idx = []
    for i in range(total):
        th = (start_ang + i * step) % (2.0 * math.pi)
        D = 300.0 + vmin * math.cos(th)
        # D>0 对于 vmin<=140
        y_min = 2200.0 * vmin * abs(math.sin(th)) / D
        if y_min <= R:
            idx.append(i)
    return total, idx


# --------------------------- 几何角度过滤器 ---------------------------
# 理由（水平几何，严格且保守）：
# - 导弹水平轨迹：M(t) = (20000 - 300 t, 0)
# - UAV 水平射线：U(s) = (17800 + s cosθ, s sinθ), s ≥ 0
# - 我们寻找与导弹轨迹的前向交点（相同 x 和 |y| ≤ R）在某些 s>0。
#   前向射线上的 x 匹配发生在 s* = 200 / cosθ（需要 cosθ>0）。那里的侧向偏移是 |y| = |s* sinθ| = |200 tanθ|。
# - 因此，前向交点在侧向半径 R 内的必要且充分条件是：
#     cosθ > 0  和  |tanθ| ≤ R / 200。
# - R = 10 m 时，这产生 |tanθ| ≤ 0.05，即 θ ∈ (-atan(0.05), +atan(0.05)) 围绕 0（以及 2π 通过周期性）。
# - 这也保证了可行的时间排序（爆炸在遮挡之前，并在 20s 窗口内），
#   因为导弹在 t=0 后 ~0.667 s 到达交叉 x，云窗口是 20 s。

GEOM_R = 10.0  # 侧向容差 (m)

def geometric_indices(start_ang: float, end_ang: float, step: float, R: float = GEOM_R):
    total = int(round((end_ang - start_ang) / max(step, 1e-6))) + 1
    tan_lim = R / 200.0
    idx = []
    for i in range(total):
        th = (start_ang + i * step) % (2.0 * math.pi)
        c = math.cos(th)
        if c <= 0:
            continue
        if abs(math.tan(th)) <= tan_lim:
            idx.append(i)
    return total, idx, tan_lim


@dataclass
class Sol3:
    speed: float
    release_time: float
    explode_delay: float
    value: float

    def as_dict(self) -> Dict:
        return dict(speed=self.speed, release_time=self.release_time,
                    explode_delay=self.explode_delay, value=self.value)


def clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def wrap_angle(a: float) -> float:
    return a % (2.0 * math.pi)


def evaluate_fixed_angle(speed: float, release_time: float, explode_delay: float,
                         azimuth: float, method: str, dt: float) -> float:
    res = evaluate_problem2(
        speed=speed,
        azimuth=azimuth,
        release_time=release_time,
        explode_delay=explode_delay,
        occlusion_method=method,
        dt=dt,
    )
    return float(res["occluded_time"]["M1"])  # 秒


# ------------------------ 初始化策略 ------------------------

def random_initial(rel_max: float, delay_max: float, azimuth: float, method: str, dt: float) -> Sol3:
    # 启发式接近 120 m/s 和较早释放，适度延迟
    base_speed = 120.0
    speed = clip(random.gauss(base_speed, 10.0), SPEED_MIN, SPEED_MAX)
    release_time = random.uniform(0.5, min(5.0, rel_max))
    explode_delay = random.uniform(2.0, min(6.0, delay_max))
    val = evaluate_fixed_angle(speed, release_time, explode_delay, azimuth, method, dt)
    return Sol3(speed, release_time, explode_delay, val)


def jitter_from_base(base: Sol3, rel_max: float, delay_max: float, azimuth: float, method: str, dt: float,
                     scale_speed: float = 3.0, scale_rel: float = 1.0, scale_dly: float = 0.8) -> Sol3:
    speed = clip(random.gauss(base.speed, scale_speed), SPEED_MIN, SPEED_MAX)
    release_time = clip(random.gauss(base.release_time, scale_rel), 0.0, rel_max)
    explode_delay = clip(random.gauss(base.explode_delay, scale_dly), DELAY_MIN, delay_max)
    val = evaluate_fixed_angle(speed, release_time, explode_delay, azimuth, method, dt)
    return Sol3(speed, release_time, explode_delay, val)


# ----------------------------- 局部邻域 -----------------------------

def neighbor(sol: Sol3, rel_max: float, delay_max: float, azimuth: float, method: str, dt: float,
             temp: float, temp_scale: float = 1.0) -> Sol3:
    # 小局部高斯移动；温度略微缩放幅度
    s = max(temp * temp_scale, 1e-3)
    speed = clip(sol.speed + random.gauss(0.0, 1.8 * s), SPEED_MIN, SPEED_MAX)
    release_time = clip(sol.release_time + random.gauss(0.0, 0.6 * s), 0.0, rel_max)
    explode_delay = clip(sol.explode_delay + random.gauss(0.0, 0.5 * s), DELAY_MIN, delay_max)
    val = evaluate_fixed_angle(speed, release_time, explode_delay, azimuth, method, dt)
    return Sol3(speed, release_time, explode_delay, val)


def broad_jump(rel_max: float, delay_max: float, azimuth: float, method: str, dt: float,
               around: Optional[Sol3] = None) -> Sol3:
    # 偶尔进行更广泛的移动以逃离零盆地
    if around is None:
        return random_initial(rel_max, delay_max, azimuth, method, dt)
    speed = clip(random.gauss(around.speed, 8.0), SPEED_MIN, SPEED_MAX)
    release_time = clip(random.gauss(around.release_time, 3.0), 0.0, rel_max)
    explode_delay = clip(random.gauss(around.explode_delay, 2.0), DELAY_MIN, delay_max)
    val = evaluate_fixed_angle(speed, release_time, explode_delay, azimuth, method, dt)
    return Sol3(speed, release_time, explode_delay, val)


# -------------------------- 每个固定角度的 SA 运行 --------------------------

def sa_for_angle(azimuth: float, rel_max: float, delay_max: float, method: str, dt: float,
                 base: Optional[Sol3],
                 t0: float = T0_LOCAL, t_end: float = T_END_LOCAL,
                 alpha: float = ALPHA_LOCAL, steps_per_t: int = STEPS_PER_T_LOCAL,
                 max_steps: int = MAX_STEPS_PER_ANGLE,
                 temp_scale: float = 1.0,
                 global_jump_prob: float = GLOBAL_JUMP_PROB) -> Sol3:
    # 起始点
    if base is not None:
        current = jitter_from_base(base, rel_max, delay_max, azimuth, method, dt)
    else:
        current = random_initial(rel_max, delay_max, azimuth, method, dt)
    best = current

    t = t0
    steps = 0
    while t > t_end and steps < max_steps:
        for _ in range(steps_per_t):
            steps += 1

            # 偶尔进行更广泛的跳跃
            if global_jump_prob > 0 and random.random() < global_jump_prob:
                j = broad_jump(rel_max, delay_max, azimuth, method, dt, around=current)
                if j.value >= current.value:
                    current = j
                    if j.value > best.value:
                        best = j
                    continue

            cand = neighbor(current, rel_max, delay_max, azimuth, method, dt, temp=t, temp_scale=temp_scale)
            delta = cand.value - current.value
            if delta >= 0 or math.exp(delta / max(t, 1e-9)) > random.random():
                current = cand
                if cand.value > best.value:
                    best = cand

            if steps >= max_steps:
                break
        t *= alpha
    return best


# ------------------------------- 主扫描 -------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="角度扫描 + 局部 SA 用于问题 2（最大化 M1 遮挡时间）")
    ap.add_argument("--angle-step", type=float, default=0.1, help="角度步长（弧度，默认 0.01）")
    ap.add_argument("--start", type=float, default=0.0, help="起始角度（弧度）")
    ap.add_argument("--end", type=float, default=6.28, help="结束角度（弧度，包含扫描到此）")

    ap.add_argument("--method", choices=["judge_caps", "sampling"], default=METHOD_DEFAULT)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--rel-max", type=float, default=RELEASE_MAX_DEFAULT)
    ap.add_argument("--delay-max", type=float, default=DELAY_MAX_DEFAULT)

    ap.add_argument("--seed", type=int, default=42)

    # 每个角度的 SA 控制
    ap.add_argument("--t0", type=float, default=T0_LOCAL)
    ap.add_argument("--t-end", type=float, default=T_END_LOCAL)
    ap.add_argument("--alpha", type=float, default=ALPHA_LOCAL)
    ap.add_argument("--steps-per-t", type=int, default=STEPS_PER_T_LOCAL)
    ap.add_argument("--max-steps-per-angle", type=int, default=MAX_STEPS_PER_ANGLE)
    ap.add_argument("--temp-scale", type=float, default=1.0)
    ap.add_argument("--global-jump-prob", type=float, default=GLOBAL_JUMP_PROB)

    # 重启策略
    ap.add_argument("--max-restarts", type=int, default=MAX_RESTARTS_NO_NONZERO,
                    help="如果在此角度找不到非零遮挡，则最大重启次数")

    # 输出
    ap.add_argument("--out-jsonl", type=str, default="best_p2_angle_sweep.jsonl",
                    help="写入每个角度最佳结果的路径（JSONL）")

    args = ap.parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2_000_000_000
    random.seed(args.seed)

    step = args.angle_step
    start_ang = args.start
    end_ang = args.end

    # 准备 JSONL 输出（截断现有文件）
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            f.write("")

    # 只枚举围绕 0 和 π 的两个解析带（无模拟预扫描）
    total_angles = int(round((end_ang - start_ang) / max(step, 1e-6))) + 1

    # 使用 vmin=70 从 y_min(θ) = 2200*vmin*|sinθ|/(300+vmin*cosθ) ≤ R 导出的半宽度
    R = 10.0
    vmin = SPEED_MIN
    # 使用更大的容差以获得更宽的角度带，例如 45 度
    d0 = math.radians(45.0)   # 围绕 0（以及 2π）
    dp = math.radians(45.0)   # 围绕 π

    def angdiff(a: float, b: float) -> float:
        # 最小有符号角度差
        return math.atan2(math.sin(a - b), math.cos(a - b))

    indices = []
    for i in range(total_angles):
        a = (start_ang + i * step) % (2.0 * math.pi)
        if abs(angdiff(a, 0.0)) <= d0 or abs(angdiff(a, math.pi)) <= dp:
            indices.append(i)

    if not indices:
        print("[analytic] 没有角度落在围绕 0 或 π 的导带内。")
        return

    print(
        f"[analytic] 带：围绕 0（以及 2π）的 |θ|≤{math.degrees(d0):.1f}°，围绕 π 的 |θ-π|≤{math.degrees(dp):.1f}°；选择了 {len(indices)}/{total_angles} 个角度。"
    )

    prev_best_nonzero: Optional[Sol3] = None
    global_best: Optional[Dict] = None

    t_sweep_start = time.time()

    total_iter = len(indices)

    for idx_pos, i in enumerate(indices, 1):
        az = wrap_angle(start_ang + i * step)
        # 决定基础：使用上一个角度非零最佳；否则 None -> 随机搜索
        base = prev_best_nonzero

        tries = 0
        best_for_angle: Optional[Sol3] = None
        found_nonzero = False

        while True:
            sol = sa_for_angle(
                azimuth=az,
                rel_max=args.rel_max,
                delay_max=args.delay_max,
                method=args.method,
                dt=args.dt,
                base=base if tries == 0 else None,  # 只有第一次尝试使用上一个角度基础
                t0=args.t0,
                t_end=args.t_end,
                alpha=args.alpha,
                steps_per_t=args.steps_per_t,
                max_steps=args.max_steps_per_angle,
                temp_scale=args.temp_scale,
                global_jump_prob=args.global_jump_prob,
            )
            if best_for_angle is None or sol.value > best_for_angle.value:
                best_for_angle = sol

            if sol.value > 1e-12:
                found_nonzero = True
                break

            tries += 1
            if tries > args.max_restarts:
                break
            # 后续尝试：强制从头重新搜索
            base = None

        # 打印每个角度预览
        deg = math.degrees(az)
        if found_nonzero and best_for_angle is not None:
            print(f"[angle {az:.2f} rad | {deg:6.2f} deg] best={best_for_angle.value:.4f}s "
                  f"spd={best_for_angle.speed:.2f} rel={best_for_angle.release_time:.2f} "
                  f"dly={best_for_angle.explode_delay:.2f} expl@{best_for_angle.release_time + best_for_angle.explode_delay:.2f}")
        else:
            best_val = 0.0 if best_for_angle is None else best_for_angle.value
            print(f"[angle {az:.2f} rad | {deg:6.2f} deg] 找不到非零（尝试={tries}）；best={best_val:.4f}s")

        # 持久化 JSONL 行
        if args.out_jsonl and best_for_angle is not None:
            record = dict(
                azimuth=az,
                azimuth_deg=deg,
                speed=best_for_angle.speed,
                release_time=best_for_angle.release_time,
                explode_delay=best_for_angle.explode_delay,
                value=best_for_angle.value,
                explode_at=best_for_angle.release_time + best_for_angle.explode_delay,
                found_nonzero=bool(found_nonzero),
                seed=args.seed,
                method=args.method,
                dt=args.dt,
            )
            with open(args.out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 跟踪全局最佳
            if global_best is None or record["value"] > global_best["value"]:
                global_best = record

        # 更新下一个角度链接规则的 prev_best_nonzero
        prev_best_nonzero = best_for_angle if (found_nonzero and best_for_angle is not None) else None

    t_sweep_end = time.time()

    print("=== 角度扫描完成 ===")
    print(f"迭代角度：{total_iter} | 运行时间：{t_sweep_end - t_sweep_start:.1f}s")
    if global_best is not None:
        print("最佳整体：")
        print(
            f"val={global_best['value']:.4f}s | az={global_best['azimuth']:.4f} rad ({global_best['azimuth_deg']:.2f} deg) "
            f"spd={global_best['speed']:.2f} rel={global_best['release_time']:.2f} "
            f"dly={global_best['explode_delay']:.2f} expl@{global_best['explode_at']:.2f}"
        )
    else:
        print("在任何角度都没有找到非零遮挡。")


if __name__ == "__main__":
    main()
