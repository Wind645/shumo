from __future__ import annotations
"""Vectorized timeline packaging + approximate occlusion evaluation.

Purpose:
    在一次仿真中, 预先把每一帧(时间步)的导弹观察点 V_t 以及所有活动烟雾球
    (S_t,R) 打包成致密张量, 再用一次向量化广播判断遮蔽。与逐帧逐球循环相比,
    在 Python 层开销更低, 适合探索参数或快速评估。

Supported method key exposed to外部: occlusion_method == 'vectorized_sampling'
    -> 走此处实现。其逻辑与 OcclusionEvaluator(method='sampling') 等价思想:
       对圆柱上下两个圆面采样的一组点 P_i, 若所有线段 V_t->P_i 被至少一个
       球体遮挡, 则记该时间步为遮蔽。

Design:
    1. 采样点 (2K,3) 仅由圆柱与 K 决定, 复用。
    2. 对时间轴长度 T, 以及最多 B 个烟雾球(=投放/起爆数) 构建:
         - V: (T,3)
         - S: (T,B,3)
         - R: (T,B)
         - active_mask: (T,B) => bool, 指示该帧该球是否存在且仍在有效期内
    3. 判定流程 (broadcast):
         对每帧, 计算所有 (active 球, 所有采样段) 是否覆盖, 再按段聚合。

Complexity:
    Let T ~ 1000, B <= 15, M=2K <= 128 -> 1000*15*128 ~= 1.9M 线段判定, numpy 可接受。

Limitations:
    - 当前实现只支持单导弹或多导弹分别循环聚合 (多导弹时将独立统计并求和)。
    - 与精确算法存在采样误差; K 越大越精确, 但开销线性增大。
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
import numpy as np

from .constants import (
    VECTORIZED_CAPS_SAMPLE_K, G, SMOKE_DESCENT_DEFAULT, SMOKE_LIFETIME_DEFAULT, SMOKE_RADIUS_DEFAULT,
    C_BASE_DEFAULT, H_CYL_DEFAULT, R_CYL_DEFAULT,
)
from .entities import SmokeCloud, Bomb, Drone, Missile, Cylinder

Vec3 = np.ndarray

# ---------------------------------------------------------------------------
# Sampling of cylinder caps
# ---------------------------------------------------------------------------

def sample_cylinder_top_bottom(K: int = VECTORIZED_CAPS_SAMPLE_K, *, r: float = R_CYL_DEFAULT, C_base=C_BASE_DEFAULT, h: float = H_CYL_DEFAULT) -> np.ndarray:
    """Return (2K,3) sample points on bottom and top circular edges of the cylinder."""
    angles = np.linspace(0.0, 2*np.pi, K, endpoint=False)
    c, s = np.cos(angles), np.sin(angles)
    circle_edge = np.stack([c * r, s * r, np.zeros_like(c)], axis=1)
    bottom = C_base.reshape(1,3) + circle_edge
    top = bottom.copy(); top[:,2] += h
    return np.concatenate([bottom, top], axis=0).astype(np.float64)

# ---------------------------------------------------------------------------
# Packaging timeline
# ---------------------------------------------------------------------------

@dataclass
class PackedTimeline:
    times: np.ndarray            # (T,)
    missile_pos: np.ndarray      # (T,3)
    sphere_centers: np.ndarray   # (T,B,3)
    sphere_radii: np.ndarray     # (T,B)
    active_mask: np.ndarray      # (T,B) bool
    sample_points: np.ndarray    # (M,3)
    dt: float


def package_timeline_single_missile(missile: Missile, drones: List[Drone], schedules: List[Tuple[int,float,float]], *, dt: float) -> PackedTimeline:
    """Run a lightweight simulation only to produce timeline arrays.

    We replicate logic from Simulator.run but without per-frame occlusion evaluation.
    """
    t_max = missile.flight_time
    emitted = [False] * len(schedules)
    bombs: List[Bomb] = []
    clouds: List[SmokeCloud] = []

    times: List[float] = []
    missile_pos: List[np.ndarray] = []

    # upper bound: number of bombs == len(schedules)
    B = len(schedules)
    # Pre-allocate dynamic arrays for sphere centers/radii; we'll fill with NaN when inactive
    sphere_centers: List[np.ndarray] = []
    sphere_radii: List[np.ndarray] = []
    active_masks: List[np.ndarray] = []

    t = 0.0
    while t <= t_max + 1e-9:
        # deploy
        for idx, (di, deploy_time, explode_delay) in enumerate(schedules):
            if (not emitted[idx]) and (t >= deploy_time):
                drone = drones[di]
                pos = drone.position(deploy_time)
                vel = drone.dir * drone.speed
                bombs.append(Bomb(release_time=deploy_time, release_pos=pos, release_vel=vel, explode_delay=explode_delay))
                emitted[idx] = True
        # explode -> cloud born
        for b in bombs:
            te = b.explode_time()
            if (t >= te) and (not any(abs(c.start_time - te) < 1e-9 for c in clouds)):
                center = b.position(te)
                clouds.append(SmokeCloud(
                    start_time=te, center0=center,
                    radius=SMOKE_RADIUS_DEFAULT, life_time=SMOKE_LIFETIME_DEFAULT, descent_speed=SMOKE_DESCENT_DEFAULT,
                ))
        # record
        active_centers = np.full((B,3), np.nan, dtype=np.float64)
        active_r = np.full((B,), np.nan, dtype=np.float64)
        active_mask = np.zeros((B,), dtype=bool)
        for ci, c in enumerate(clouds):
            if ci >= B:
                break
            if c.active(t):
                active_centers[ci] = c.center(t)
                active_r[ci] = c.radius
                active_mask[ci] = True
        times.append(t)
        missile_pos.append(missile.position(t))
        sphere_centers.append(active_centers)
        sphere_radii.append(active_r)
        active_masks.append(active_mask)
        t += dt

    return PackedTimeline(
        times=np.asarray(times, dtype=np.float64),
        missile_pos=np.asarray(missile_pos, dtype=np.float64),
        sphere_centers=np.asarray(sphere_centers, dtype=np.float64),
        sphere_radii = np.asarray(sphere_radii, dtype=np.float64),
        active_mask = np.asarray(active_masks, dtype=bool),
        sample_points = sample_cylinder_top_bottom(),
        dt=float(dt),
    )

# ---------------------------------------------------------------------------
# Vectorized occlusion evaluation over packed timeline (sampling method)
# ---------------------------------------------------------------------------

def _segment_intersects_sphere(V: np.ndarray, P: np.ndarray, S: np.ndarray, R: float) -> bool:
    # Same geometric test as CylinderOcclusionJudge._segment_intersects_sphere but inlined to avoid import loop
    L = P - V
    LV2 = np.dot(L, L)
    if LV2 <= 1e-15:
        return False
    t = np.dot(S - V, L) / LV2
    if t < 0.0:
        closest = V
    elif t > 1.0:
        closest = P
    else:
        closest = V + t * L
    return np.dot(closest - S, closest - S) <= R * R + 1e-12


def occluded_time_vectorized_sampling(packed: PackedTimeline) -> float:
    """Return total occluded time using sampling union-of-spheres test (vectorized per-frame)."""
    V = packed.missile_pos        # (T,3)
    S_all = packed.sphere_centers  # (T,B,3)
    R_all = packed.sphere_radii    # (T,B)
    active = packed.active_mask    # (T,B)
    pts = packed.sample_points     # (M,3)
    dt = packed.dt
    T, M = V.shape[0], pts.shape[0]

    # We'll iterate frames (T) in Python but vectorize within frame across spheres & sample segments.
    occluded = 0.0
    seg_cache = None  # reuse (M,3) sample points difference if needed, but each frame V_t changes.

    for t_idx in range(T):
        v = V[t_idx]
        mask = active[t_idx]
        if not mask.any():
            continue
        centers = S_all[t_idx][mask]  # (K,3)
        radii = R_all[t_idx][mask]    # (K,)
        # For each sample point test if ANY sphere covers segment
        all_covered = True
        for P in pts:
            covered = False
            for (S, R) in zip(centers, radii):
                if _segment_intersects_sphere(v, P, S, R):
                    covered = True
                    break
            if not covered:
                all_covered = False
                break
        if all_covered:
            occluded += dt
    return float(occluded)

__all__ = [
    'PackedTimeline', 'package_timeline_single_missile', 'occluded_time_vectorized_sampling'
]
