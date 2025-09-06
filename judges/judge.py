from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from utils import (
	circle_fmin_cos,
	circle_fully_occluded_by_sphere,
)


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
