from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

def circle_fmin_cos(V, C, r, S):
    """
    返回 f_min = min_t cosθ(t) 及达到该值的 t（弧度）。
    参数:
      - V: 观察点（3,）
      - C: 圆心（3,），要求 C[2]=0（圆位于 z=0 平面）
      - r: 圆半径
      - S: 球心（3,）
    返回:
      f_min, t_at_min（若无法评估返回 (None, None)）
    """
    consts = _f_of_t_constants(V, C, r, S)
    A, B, D, E, F, G = consts['A'], consts['B'], consts['D'], consts['E'], consts['F'], consts['G']

    # r=0 退化：圆退化为点，此时 f(t) 常数
    if abs(r) <= 0.0 + 0.0:
        f0 = _evaluate_f_at_t(0.0, A, B, D, E, F, G, r=0.0)
        return f0, 0.0

    # 构造四次多项式
    coeffs = _quartic_coeffs_for_stationary(A, B, D, E, F, G, r)
    roots = _poly_real_roots_desc(coeffs, tol=1e-10)

    candidates = []

    if roots is None:
        # 驻点条件恒成立：f(t) 为常函数
        f0 = _evaluate_f_at_t(0.0, A, B, D, E, F, G, r)
        return f0, 0.0

    # 用实根生成候选 t 并评估 f
    for z in roots:
        f, t = _evaluate_f_from_z(z, A, B, D, E, F, G, r)
        if f is not None and np.isfinite(f):
            candidates.append((f, t))

    # 极少数情况下（例如观测点恰在圆上导致 M(t)=0）可能没有可用候选
    # 为了得到 f_min，我们在有限代表点上评估（这不是网格采样，只是退化兜底）
    if not candidates:
        for t in (0.0, np.pi/2, np.pi, 3*np.pi/2):
            f = _evaluate_f_at_t(t, A, B, D, E, F, G, r)
            if f is not None and np.isfinite(f):
                candidates.append((f, t))

    if not candidates:
        return None, None

    # 取最小 f
    f_vals = np.array([ft[0] for ft in candidates], dtype=float)
    idx = int(np.argmin(f_vals))
    f_min, t_min = candidates[idx]
    return f_min, t_min

def circle_fully_occluded_by_sphere(V, C, r, S, R, return_debug=False):
    """
    判定：从观察点 V 看，位于 z=0 的半径 r 的圆（圆心 C），是否被球（球心 S，半径 R）完全遮挡。
    精确必要充分条件（无采样）：
      max_t angle(u, X(t)) <= alpha_s
      等价于 min_t cosθ(t) >= cos(alpha_s)
    参数:
      - V: 观察点（3,）
      - C: 圆心（3,），要求 C[2]=0
      - r: 圆半径
      - S: 球心（3,）
      - R: 球半径
    返回:
      - occluded: bool
      - 若 return_debug=True，额外返回字典，包含 f_min、cos_alpha_s、t_min 等。
    """
    V = np.asarray(V, dtype=float).reshape(3)
    C = np.asarray(C, dtype=float).reshape(3)
    S = np.asarray(S, dtype=float).reshape(3)
    R = float(R)
    r = float(r)

    # 球的观测几何退化：观测点在/内球 => 球遮挡一切方向
    S_prime = S - V
    dS = np.linalg.norm(S_prime)
    if dS <= 0.0 + 0.0 or R >= dS:
        if return_debug:
            return True, dict(reason="observer_in_or_at_sphere", f_min=None, cos_alpha_s=None, t_min=None)
        return True

    # 计算 f_min
    f_min, t_min = circle_fmin_cos(V, C, r, S)
    if f_min is None:
        # 无法稳健评估（极端退化）
        if return_debug:
            return False, dict(reason="degenerate_no_eval", f_min=None, cos_alpha_s=None, t_min=None)
        return False

    # 球锥阈值 cos(alpha_s) = sqrt(1 - (R/dS)^2)
    ratio = R / dS
    if ratio >= 1.0:
        cos_alpha_s = 0.0
    else:
        cos_alpha_s = np.sqrt(max(0.0, 1.0 - ratio * ratio))

    occluded = (f_min >= cos_alpha_s - 1e-12)

    if return_debug:
        return occluded, dict(f_min=float(f_min), cos_alpha_s=float(cos_alpha_s), t_min=float(t_min))
    return occluded


@dataclass
class OcclusionResult:
	"""
	结果载体：是否完全遮蔽及关键中间量。

	字段:
	  - occluded: 是否完全被遮挡（充分且必要，角域判定）
	  - f_min: min_t cos(theta(t))（观察方向与球轴方向的最小余弦）
	  - cos_alpha_s: 球切线锥半顶角的余弦阈值
	  - t_min: 达到 f_min 的圆周参数 t（弧度）
	  - extra: 其他可选调试信息
	"""

	occluded: bool
	f_min: Optional[float]
	cos_alpha_s: Optional[float]
	t_min: Optional[float]
	extra: Optional[Dict] = None


class OcclusionJudge:
	"""
	单圆-球体遮挡判定器。

	几何设定：
	- 圆位于 z=0 平面，圆心 C=(x_c, y_c, 0)，半径 r。
	- 观察点 V=(x_v, y_v, z_v) 且 z_v>0。
	- 球心 S=(x_s, y_s, z_s) 且 z_s>0，球半径 R。

	判定准则（必要且充分）：
	设 u = (S-V)/||S-V||，球切线锥半顶角 alpha_s = arcsin(R/||S-V||)。
	令 psi_max = max_{X∈圆} angle(u, X-V)。
	则圆被球完全遮挡 当且仅当 psi_max ≤ alpha_s。
	等价为 min_t cos(theta(t)) ≥ cos(alpha_s)。
	"""

	def __init__(self, V: np.ndarray, C: np.ndarray, r: float, S: np.ndarray, R: float):
		self.V = np.asarray(V, dtype=float).reshape(3)
		self.C = np.asarray(C, dtype=float).reshape(3)
		self.r = float(r)
		self.S = np.asarray(S, dtype=float).reshape(3)
		self.R = float(R)

	def fmin_cos(self) -> Tuple[Optional[float], Optional[float]]:
		"""
		计算 f_min = min_t cos(theta(t)) 及达到该值的 t_min（弧度）。
		若退化无法评估，返回 (None, None)。
		"""
		return circle_fmin_cos(self.V, self.C, self.r, self.S)

	def is_fully_occluded(self) -> OcclusionResult:
		"""
		判定圆是否被球完全遮挡，并返回关键中间量。
		"""
		occluded, dbg = circle_fully_occluded_by_sphere(
			self.V, self.C, self.r, self.S, self.R, return_debug=True
		)

		# dbg 可能在极端退化时只包含 reason
		f_min = dbg.get("f_min") if isinstance(dbg, dict) else None
		cos_alpha_s = dbg.get("cos_alpha_s") if isinstance(dbg, dict) else None
		t_min = dbg.get("t_min") if isinstance(dbg, dict) else None

		return OcclusionResult(
			occluded=bool(occluded),
			f_min=f_min,
			cos_alpha_s=cos_alpha_s,
			t_min=t_min,
			extra=dbg if isinstance(dbg, dict) else None,
		)
        
# ================== 新增：圆柱完全遮挡解析(角域)批量接口 ==================
def is_cylinder_blocked_analytic(data,
                                 use_midpoint: bool = True) -> np.ndarray:
    """
    批量解析判定（无采样）：烟幕球是否完全遮挡圆柱目标。
    与 batch_rough 的公式一致，但这里对圆柱使用多圆心并判:
      设:
        球心 S, 半径 R_s
        观测点 M
        圆柱半径 r_cyl = 7, 高 10, 底/顶/中点圆心:
          B=(0,200,0), T=(0,200,10), M_c=(0,200,5)
      定义:
        θ_s = arcsin(R_s / d_s),  d_s = |S-M|
        θ_i = arcsin(r_cyl / d_i), d_i = |C_i - M|
        φ_i = angle( (C_i - M), (S - M) )
      条件:
        对所有选取的 C_i, 均满足 θ_s >= θ_i + φ_i
      返回 True.

    参数:
      data: shape (N,7):
            [missile_x, missile_y, missile_z,
             smoke_x, smoke_y, smoke_z,
             R_smoke]
      use_midpoint: 是否包含中点 (0,200,5) 判定 (默认 True 更严格)

    返回:
      ndarray(bool) shape (N,)
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError("data must have shape (N,7)")

    # 常量
    r_cyl = 7.0
    centers = [
        np.array([0.0, 200.0, 0.0]),
        np.array([0.0, 200.0, 10.0]),
    ]
    if use_midpoint:
        centers.append(np.array([0.0, 200.0, 5.0]))
    centers = np.stack(centers, axis=0)        # (Kc,3)

    M = arr[:, :3]        # (N,3)
    S = arr[:, 3:6]       # (N,3)
    R_s = arr[:, 6]       # (N,)

    v_s = S - M
    d_s = np.linalg.norm(v_s, axis=1)
    eps = 1e-12
    valid = d_s > eps
    θ_s = np.zeros_like(d_s)
    θ_s[valid] = np.arcsin(np.clip(R_s[valid] / d_s[valid], -1.0, 1.0))

    # 逐中心并判
    all_ok = np.ones(len(arr), dtype=bool)
    for C in centers:
        v_c = C[None,:] - M          # (N,3)
        d_c = np.linalg.norm(v_c, axis=1)
        θ_c = np.zeros_like(d_c)
        good = d_c > eps
        θ_c[good] = np.arcsin(np.clip(r_cyl / d_c[good], -1.0, 1.0))

        # 夹角 φ
        dot = np.sum(v_c * v_s, axis=1)
        denom = (d_c * d_s) + eps
        cosφ = np.clip(dot / denom, -1.0, 1.0)
        φ = np.arccos(cosφ)

        cond = θ_s >= (θ_c + φ)
        all_ok &= cond

    return all_ok

# ================== /新增 ==================

if __name__ == "__main__":
	# 简单自检示例
	V = np.array([0.0, 0.0, 2.0])
	C = np.array([0.5, 0.0, 0.0])
	r = 0.3
	S = np.array([0.2, 0.1, 1.0])
	R = 0.5

	judge = OcclusionJudge(V, C, r, S, R)
	res = judge.is_fully_occluded()
	print("occluded:", res.occluded)
	print("f_min:", res.f_min, "cos_alpha_s:", res.cos_alpha_s, "t_min:", res.t_min)
	# 新增简单批量 API 自检
	demo = np.array([
		# 期望 True: 大球包住
		[100.0, -50.0, 30.0, 0.0, 200.0, 5.0, 80.0],
		# 期望 False: 球太小
		[100.0, -50.0, 30.0, 0.0, 200.0, 5.0, 5.0],
	])
	print("analytic batch:", is_cylinder_blocked_analytic(demo))