from __future__ import annotations
"""采样法 vs 解析法对比 (去除 argparse 版)

通过顶部常量直接配置，避免命令行解析。
"""
import time
import random
from typing import Dict, Tuple

import numpy as np

from baolijiefa import CylinderOcclusionJudge
from utils import circle_fmin_cos, circle_fully_occluded_by_sphere

# 目标圆柱参数（默认与 compare.py 保持一致）
C_BASE = np.array([0.0, 200.0, 0.0], dtype=float)
R_CYL = 7.0
H_CYL = 10.0

# 遮挡球半径
R_SPHERE = 10.0

# 随机范围
XY_RANGE = 2000.0
Z_MAX = 2000.0

# 颜色
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def random_sphere_center(R: float) -> np.ndarray:
    x = random.uniform(-XY_RANGE, XY_RANGE)
    y = random.uniform(-XY_RANGE, XY_RANGE)
    z = random.uniform(R + 1e-3, Z_MAX)
    return np.array([x, y, z], dtype=float)


def random_observer() -> np.ndarray:
    x = random.uniform(-XY_RANGE, XY_RANGE)
    y = random.uniform(-XY_RANGE, XY_RANGE)
    z = random.uniform(1e-3, Z_MAX)
    return np.array([x, y, z], dtype=float)


def inside_cylinder(P: np.ndarray, C_base: np.ndarray, r_cyl: float, h_cyl: float) -> bool:
    dx, dy = P[0] - C_base[0], P[1] - C_base[1]
    rho2 = dx * dx + dy * dy
    z0, z1 = C_base[2], C_base[2] + h_cyl
    zmin, zmax = (z0, z1) if z0 <= z1 else (z1, z0)
    return (rho2 <= r_cyl * r_cyl + 1e-12) and (zmin - 1e-12 <= P[2] <= zmax + 1e-12)


def inside_sphere(P: np.ndarray, S: np.ndarray, R: float) -> bool:
    return np.linalg.norm(P - S) <= R + 1e-12


def segment_min_dist_to_point(V: np.ndarray, P: np.ndarray, S: np.ndarray) -> Tuple[float, float]:
    """
    返回球心 S 到线段 VP 的最小距离及对应参数 t∈[0,1]。
    """
    d = P - V
    a = float(np.dot(d, d))
    if a == 0.0:
        return float(np.linalg.norm(V - S)), 0.0
    t = -float(np.dot(V - S, d)) / a
    t = max(0.0, min(1.0, t))
    closest = V + t * d
    return float(np.linalg.norm(closest - S)), t


def analytic_caps(V: np.ndarray, S: np.ndarray, R: float, C_base: np.ndarray, r_cyl: float, h_cyl: float) -> Dict:
    """
    用解析法分别判断底/顶端面圆，并返回综合信息和裕量。
    margin_cap = f_min - cos_alpha_s；综合 margin = min(margin_bottom, margin_top)。
    """
    # 底面（已在 z=0）
    ok_b, dbg_b = circle_fully_occluded_by_sphere(V, C_base, r_cyl, S, R, return_debug=True)
    fmin_b = dbg_b.get("f_min", None) if isinstance(dbg_b, dict) else None
    cosa_b = dbg_b.get("cos_alpha_s", None) if isinstance(dbg_b, dict) else None
    margin_b = None if (fmin_b is None or cosa_b is None) else float(fmin_b - cosa_b)

    # 顶面：整体平移 -h，使顶面落在 z=0
    C_top = C_base + np.array([0.0, 0.0, h_cyl])
    V_top = V - np.array([0.0, 0.0, h_cyl])
    S_top = S - np.array([0.0, 0.0, h_cyl])
    C_top_flat = np.array([C_top[0], C_top[1], 0.0])
    ok_t, dbg_t = circle_fully_occluded_by_sphere(V_top, C_top_flat, r_cyl, S_top, R, return_debug=True)
    fmin_t = dbg_t.get("f_min", None) if isinstance(dbg_t, dict) else None
    cosa_t = dbg_t.get("cos_alpha_s", None) if isinstance(dbg_t, dict) else None
    margin_t = None if (fmin_t is None or cosa_t is None) else float(fmin_t - cosa_t)

    if (margin_b is None) or (margin_t is None):
        combined = bool(ok_b and ok_t)
        return dict(
            ok=combined,
            bottom=dict(ok=bool(ok_b), f_min=fmin_b, cos_alpha_s=cosa_b, margin=margin_b),
            top=dict(ok=bool(ok_t), f_min=fmin_t, cos_alpha_s=cosa_t, margin=margin_t),
            margin=None,
        )

    return dict(
        ok=bool(ok_b and ok_t),
        bottom=dict(ok=bool(ok_b), f_min=fmin_b, cos_alpha_s=cosa_b, margin=margin_b),
        top=dict(ok=bool(ok_t), f_min=fmin_t, cos_alpha_s=cosa_t, margin=margin_t),
        margin=float(min(margin_b, margin_t)),
    )


def sampling_cylinder_with_margin(
    V: np.ndarray, S: np.ndarray, R: float, C_base: np.ndarray, r_cyl: float, h_cyl: float,
    n_theta: int, n_h: int, n_cap_radial: int, check_caps: bool
) -> Dict:
    """
    调用采样判定器，同时计算“采样裕量”：min_P (R - d_min(VP,S))。
    """
    judge = CylinderOcclusionJudge(
        V=V, S=S, R=R, C_base=C_base, r_cyl=r_cyl, h_cyl=h_cyl,
        n_theta=n_theta, n_h=n_h, n_cap_radial=n_cap_radial, check_caps=check_caps
    )
    res = judge.is_fully_occluded()

    # 重用其采样点以获得一致的“最小裕量”
    try:
        pts = judge._sample_cylinder_points()  # 直接调用其内部采样，确保一致
    except Exception:
        # 退化兜底：调用一次 is_fully_occluded() 后再尝试
        pts = judge._sample_cylinder_points()

    best_margin = float("+inf")
    best_idx = -1
    best_t = None

    for i, P in enumerate(pts):
        dmin, t = segment_min_dist_to_point(V, P, S)
        margin = R - dmin
        if margin < best_margin:
            best_margin = margin
            best_idx = i
            best_t = t

    return dict(
        ok=bool(res.occluded),
        total_points=int(res.total_points),
        blocked_points=int(res.blocked_points),
        uncovered_indices=list(res.uncovered_indices),
        min_margin=float(best_margin),
        worst_point_index=int(best_idx),
        worst_t=float(best_t) if best_t is not None else None
    )


def run_one_trial(
    n_theta: int, n_h: int, n_cap_radial: int, check_caps: bool, refine_on_mismatch: bool, seed: int | None = None
) -> Tuple[bool, Dict]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 采样合法 V,S
    for _ in range(10000):
        S = random_sphere_center(R_SPHERE)
        V = random_observer()
        if inside_sphere(V, S, R_SPHERE):
            continue
        if inside_cylinder(V, C_BASE, R_CYL, H_CYL):
            continue
        break
    else:
        raise RuntimeError("Failed to sample valid V,S pair.")

    t0 = time.time()
    samp = sampling_cylinder_with_margin(V, S, R_SPHERE, C_BASE, R_CYL, H_CYL, n_theta, n_h, n_cap_radial, check_caps)
    t1 = time.time()
    ana = analytic_caps(V, S, R_SPHERE, C_BASE, R_CYL, H_CYL)
    t2 = time.time()

    same = (samp["ok"] == ana["ok"])
    detail = dict(V=V, S=S, sampling=samp, analytic=ana, time_us=dict(sampling=(t1-t0)*1e6, analytic=(t2-t1)*1e6))

    # 不一致则可选加密采样复核
    if (not same) and refine_on_mismatch:
        samp_ref = sampling_cylinder_with_margin(
            V, S, R_SPHERE, C_BASE, R_CYL, H_CYL,
            n_theta=n_theta * 2, n_h=n_h * 2, n_cap_radial=max(2, n_cap_radial * 2), check_caps=check_caps
        )
        detail["sampling_refined"] = samp_ref
        same_ref = (samp_ref["ok"] == ana["ok"])
        detail["same_after_refine"] = same_ref

    return same, detail


# ================== 可编辑常量区域 ==================
CONF_TRIALS = 200
CONF_SEED: int | None = 123
CONF_N_THETA = 48
CONF_N_H = 16
CONF_N_CAP_RADIAL = 6
CONF_CHECK_CAPS = True  # True=采样端面; False=只侧面
CONF_REFINE_ON_MISMATCH = False
CONF_VERBOSE = False
# ================== 可编辑常量区域 END ==============


def run_from_constants():
    if CONF_SEED is not None:
        random.seed(CONF_SEED)
        np.random.seed(CONF_SEED)

    ok = 0
    diff = 0
    t_samp = 0.0
    t_ana = 0.0

    for i in range(1, CONF_TRIALS + 1):
        same, info = run_one_trial(
            n_theta=CONF_N_THETA,
            n_h=CONF_N_H,
            n_cap_radial=CONF_N_CAP_RADIAL,
            check_caps=CONF_CHECK_CAPS,
            refine_on_mismatch=CONF_REFINE_ON_MISMATCH,
        )
        t_samp += info["time_us"]["sampling"]
        t_ana += info["time_us"]["analytic"]

        if same:
            ok += 1
            if CONF_VERBOSE:
                print(f"[{i:04d}] {GREEN}一致{RESET}  samp={info['sampling']['ok']}  ana={info['analytic']['ok']}  "
                      f"samp_margin={info['sampling']['min_margin']:.6g}  "
                      f"ana_margin={info['analytic']['margin'] if info['analytic']['margin'] is not None else None}")
        else:
            diff += 1
            ana = info["analytic"]
            samp = info["sampling"]
            print(f"[{i:04d}] {RED}不一致{RESET}  samp={samp['ok']}  ana={ana['ok']}")
            print(f"  采样裕量 min(R - dmin): {samp['min_margin']:.6g}  "
                  f"(worst_idx={samp['worst_point_index']}, t*={samp['worst_t']})")
            print(f"  解析裕量 min(f_min - cosα): {ana['margin'] if ana['margin'] is not None else None}")
            print(f"  底面: ok={ana['bottom']['ok']}  margin={ana['bottom']['margin']}")
            print(f"  顶面: ok={ana['top']['ok']}     margin={ana['top']['margin']}")
            if CONF_REFINE_ON_MISMATCH and ("sampling_refined" in info):
                sr = info["sampling_refined"]
                print(f"  加密采样后: ok={sr['ok']}  min_margin={sr['min_margin']:.6g}  same_after_refine={info['same_after_refine']}")

    print()
    print(f"汇总：一致 {ok} / {CONF_TRIALS}，不一致 {diff} / {CONF_TRIALS}")
    print(f"平均耗时：采样 {t_samp/CONF_TRIALS:.1f} μs/例，解析 {t_ana/CONF_TRIALS:.1f} μs/例")


if __name__ == "__main__":
    run_from_constants()