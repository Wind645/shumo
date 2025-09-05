from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

@dataclass
class CylinderOcclusionResult:
    """
    采样法判定结果（近似）：
      - occluded: 是否全部采样点都被遮挡（True 视为看不到）
      - total_points: 采样总点数
      - blocked_points: 被遮挡的采样点数
      - uncovered_indices: 未被遮挡的采样点（索引列表，最多返回若干个）
      - extra: 其他信息（参数、用时等可扩展）
    """
    occluded: bool
    total_points: int
    blocked_points: int
    uncovered_indices: List[int]
    extra: Optional[Dict] = None


class CylinderOcclusionJudge:
    """
    采样法：从观察点 V 看，遮挡球 (S, R) 是否能遮住给定直圆柱的所有可见连线。

    几何设定（默认轴平行 z 轴）：
      - 圆柱底面圆心 C_base=(x, y, z0)，半径 r_cyl，高度 h_cyl（沿 z 轴，可正可负）。
      - 观察点 V=(x_v, y_v, z_v)。
      - 遮挡球心 S=(x_s, y_s, z_s)，半径 R。

    判定准则（采样近似）：
      - 在圆柱侧面与两个端面上采样若干点 P。
      - 若对所有 P，线段 VP 与球 (S,R) 相交（相交点参数 t∈[0,1]），
        则视为“完全被遮挡”，返回 True；否则 False。
      - 注意：这是近似（由采样密度决定），想更稳妥就加密采样。
    """

    def __init__(
        self,
        V: torch.Tensor,
        S: torch.Tensor,
        R: float,
        C_base: torch.Tensor,
        r_cyl: float,
        h_cyl: float,
        n_theta: int = 48,       # 周向采样数（侧面与端面）
        n_h: int = 16,           # 高度方向采样数（侧面）
        n_cap_radial: int = 6,   # 端面径向环数（含中心）
        check_caps: bool = True, # 是否采样端面
        max_uncovered_report: int = 16,  # 最多记录多少未遮挡样本索引
    ):
        with torch.no_grad():
            self.V = torch.tensor(V, dtype=torch.float).reshape(3)
            self.S = torch.tensor(S, dtype=torch.float).reshape(3)
            self.R = float(R)

            self.C_base = torch.tensor(C_base, dtype=torch.float).reshape(3)
            self.r_cyl = float(r_cyl)
            self.h_cyl = float(h_cyl)

            self.n_theta = int(max(4, n_theta))
            self.n_h = int(max(2, n_h))
            self.n_cap_radial = int(max(1, n_cap_radial))
            self.check_caps = bool(check_caps)
            self.max_uncovered_report = int(max_uncovered_report)

    @staticmethod
    def _segment_intersects_sphere(V: torch.Tensor, P: torch.Tensor, S: torch.Tensor, R: float) -> bool:
        """
        线段 VP 与球(S,R) 是否相交（存在交点且在 t∈[0,1]）。
        方向方程 |V + t(P-V) - S|^2 = R^2 => a t^2 + b t + c = 0
        """
        with torch.no_grad():
            V = torch.tensor(V, dtype=torch.float)
            P = torch.tensor(P, dtype=torch.float)
            S = torch.tensor(S, dtype=torch.float)
            d = P - V
            a = torch.dot(d, d)
            if a == 0.0:
                return torch.dot(V - S, V - S) <= R * R
            b = 2.0 * torch.dot(d, V - S)
            c = torch.dot(V - S, V - S) - R * R
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return False
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

    def _sample_cylinder_points(self) -> torch.Tensor:
        """
        生成圆柱表面与端面的采样点（N,3）：
          - 侧面：n_theta * (n_h+1)
          - 端面：若启用，每个端面 n_theta * n_cap_radial（含中心环）
        """
        with torch.no_grad():
            x0, y0, z0 = self.C_base
            z1 = z0 + self.h_cyl
            zs = torch.linspace(z0, z1, self.n_h + 1)
            thetas = torch.linspace(0.0, 2.0 * torch.pi, self.n_theta, endpoint=False)
            pts = []
            for z in zs:
                for th in thetas:
                    x = x0 + self.r_cyl * torch.cos(th)
                    y = y0 + self.r_cyl * torch.sin(th)
                    pts.append((x, y, z))
            if self.check_caps:
                if self.n_cap_radial == 1:
                    radii = [0.0]
                else:
                    radii = [0.0] + [self.r_cyl * torch.sqrt(torch.tensor(i / (self.n_cap_radial - 1))) for i in range(1, self.n_cap_radial)]
                for z_cap in (z0, z1):
                    for r in radii:
                        if r == 0.0:
                            pts.append((x0, y0, z_cap))
                        else:
                            for th in thetas:
                                x = x0 + r * torch.cos(th)
                                y = y0 + r * torch.sin(th)
                                pts.append((x, y, z_cap))
            return torch.tensor(pts, dtype=torch.float)

    def is_fully_occluded(self) -> CylinderOcclusionResult:
        with torch.no_grad():
            if torch.norm(self.S - self.V) <= self.R:
                pts = self._sample_cylinder_points()
                return CylinderOcclusionResult(
                    occluded=True,
                    total_points=pts.shape[0],
                    blocked_points=pts.shape[0],
                    uncovered_indices=[],
                    extra=dict(reason="observer_inside_sphere"),
                )
            pts = self._sample_cylinder_points()
            blocked = 0
            uncovered_idx: List[int] = []
            for i, P in enumerate(pts):
                hit = self._segment_intersects_sphere(self.V, P, self.S, self.R)
                if hit:
                    blocked += 1
                else:
                    if len(uncovered_idx) < self.max_uncovered_report:
                        uncovered_idx.append(i)
            occluded = (blocked == pts.shape[0])
            return CylinderOcclusionResult(
                occluded=occluded,
                total_points=int(pts.shape[0]),
                blocked_points=int(blocked),
                uncovered_indices=uncovered_idx,
                extra=dict(
                    n_theta=self.n_theta,
                    n_h=self.n_h,
                    n_cap_radial=self.n_cap_radial,
                    check_caps=self.check_caps,
                ),
            )
