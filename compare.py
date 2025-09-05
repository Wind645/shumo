from __future__ import annotations

import sys
import random
import numpy as np

from baolijiefa import CylinderOcclusionJudge
from judge import OcclusionJudge


# 目标圆柱参数（与题面一致）
C_BASE = np.array([0.0, 200.0, 0.0], dtype=float)
R_CYL = 7.0
H_CYL = 10.0

# 烟幕球半径（题目指定中心10m范围内有效）
R_SPHERE = 10.0

# 随机范围设置
XY_RANGE = 2000.0
Z_MAX = 2000.0

# ANSI 颜色
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def random_sphere_center(R: float) -> np.ndarray:
    """随机生成球心 S，保证 S.z > R。"""
    while True:
        x = random.uniform(-XY_RANGE, XY_RANGE)
        y = random.uniform(-XY_RANGE, XY_RANGE)
        z = random.uniform(R + 1e-3, Z_MAX)
        return np.array([x, y, z], dtype=float)


def random_observer() -> np.ndarray:
    """随机生成观察点 V，保证 V.z > 0。"""
    while True:
        x = random.uniform(-XY_RANGE, XY_RANGE)
        y = random.uniform(-XY_RANGE, XY_RANGE)
        z = random.uniform(1e-3, Z_MAX)
        return np.array([x, y, z], dtype=float)


def inside_cylinder(V: np.ndarray, C_base: np.ndarray, r_cyl: float, h_cyl: float) -> bool:
    """判断点 V 是否位于以 z 轴为轴线、底心 C_base、半径 r_cyl、高度 h_cyl 的直圆柱体内部（含边界）。"""
    dx, dy = V[0] - C_base[0], V[1] - C_base[1]
    rho2 = dx * dx + dy * dy
    z0 = C_base[2]
    z1 = z0 + h_cyl
    z_min, z_max = (z0, z1) if z0 <= z1 else (z1, z0)
    return (rho2 <= r_cyl * r_cyl + 1e-12) and (z_min - 1e-12 <= V[2] <= z_max + 1e-12)


def inside_sphere(V: np.ndarray, S: np.ndarray, R: float) -> bool:
    return np.linalg.norm(V - S) <= R + 1e-12


def judge_caps_by_analytic(V: np.ndarray, S: np.ndarray, R: float, C_base: np.ndarray, r_cyl: float, h_cyl: float) -> bool:
    """
    使用精确圆-球遮蔽判定（judge.py）分别判断底面与顶面两圆是否被完全遮蔽，
    并以两者都被遮蔽作为圆柱被遮蔽的判定结果。

    注意：judge 假设圆位于 z=0 平面，因此对顶面需要做坐标平移：
      V' = V - (0,0,h)，S' = S - (0,0,h)，C'=(C_base.x, C_base.y, 0)。
    """
    # 底面（已在 z=0）
    judge_bottom = OcclusionJudge(V, C_base, r_cyl, S, R)
    res_bottom = judge_bottom.is_fully_occluded()

    # 顶面：将坐标整体平移 -h，使顶面落在 z=0 平面
    C_top = C_base + np.array([0.0, 0.0, h_cyl])
    V_top = V - np.array([0.0, 0.0, h_cyl])
    S_top = S - np.array([0.0, 0.0, h_cyl])
    C_top_flat = np.array([C_top[0], C_top[1], 0.0])
    judge_top = OcclusionJudge(V_top, C_top_flat, r_cyl, S_top, R)
    res_top = judge_top.is_fully_occluded()

    return bool(res_bottom.occluded and res_top.occluded)


def single_trial(rng_seed: int | None = None):
    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed)

    # 生成合法的 S, V
    for _ in range(10000):  # 防止极少数情况下难以采到合格样本
        S = random_sphere_center(R_SPHERE)
        V = random_observer()

        if inside_sphere(V, S, R_SPHERE):
            continue
        if inside_cylinder(V, C_BASE, R_CYL, H_CYL):
            continue
        break
    else:
        raise RuntimeError("Failed to sample valid V,S in 10000 attempts")

    # 方法1：采样法（圆柱体侧面+端面）
    judge_sample = CylinderOcclusionJudge(
        V=V,
        S=S,
        R=R_SPHERE,
        C_base=C_BASE,
        r_cyl=R_CYL,
        h_cyl=H_CYL,
        n_theta=48,
        n_h=16,
        n_cap_radial=6,
        check_caps=True,
    )
    res1 = judge_sample.is_fully_occluded()
    ans1 = bool(res1.occluded)

    # 方法2：精确法（仅端面两圆）
    ans2 = judge_caps_by_analytic(V, S, R_SPHERE, C_BASE, R_CYL, H_CYL)

    return ans1, ans2, dict(V=V, S=S)


def main():
    MAXN = 1000
    ok = 0
    diff = 0

    for i in range(1, MAXN + 1):
        ans1, ans2, _dbg = single_trial()
        same = (ans1 == ans2)
        if same:
            ok += 1
            print(f"[{i:04d}] {GREEN}一致{RESET}  ans1(采样法)={ans1}  ans2(端面解析)={ans2}")
        else:
            diff += 1
            print(f"[{i:04d}] {RED}不一致{RESET} ans1(采样法)={ans1}  ans2(端面解析)={ans2}")

    print()
    color = GREEN if diff == 0 else (YELLOW if diff < MAXN * 0.05 else RED)
    print(f"结果汇总：一致 {ok} / {MAXN}，不一致 {diff} / {MAXN}")
    print(color + ("全部一致" if diff == 0 else ("少量不一致" if diff < MAXN * 0.05 else "存在不一致")) + RESET)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断退出。")

