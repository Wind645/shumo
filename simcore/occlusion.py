from __future__ import annotations
from typing import Iterable, List, Tuple
import numpy as np

from .baolijiefa import CylinderOcclusionJudge
from judge import OcclusionJudge
from .entities import Cylinder

Vec3 = np.ndarray

class OcclusionEvaluator:
    def __init__(self, cyl: Cylinder, n_theta: int = 48, n_h: int = 16,
                 n_cap_radial: int = 6, check_caps: bool = True, method: str = "sampling"):
        self.cyl = cyl
        self.n_theta = int(max(4, n_theta))
        self.n_h = int(max(2, n_h))
        self.n_cap_radial = int(max(1, n_cap_radial))
        self.check_caps = bool(check_caps)
        self.method = method
        self._pts = None
        if self.method == "sampling":
            self._sampler = CylinderOcclusionJudge(
                V=np.array([0.0, 0.0, 1.0]),
                S=np.array([0.0, 0.0, 2.0]),
                R=1.0,
                C_base=self.cyl.C_base,
                r_cyl=self.cyl.r,
                h_cyl=self.cyl.h,
                n_theta=self.n_theta,
                n_h=self.n_h,
                n_cap_radial=self.n_cap_radial,
                check_caps=self.check_caps,
            )
            self._pts = self._sampler._sample_cylinder_points()

    @staticmethod
    def _hit_any_sphere(V: np.ndarray, P: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]) -> bool:
        for S, R in spheres:
            if CylinderOcclusionJudge._segment_intersects_sphere(V, P, S, R):
                return True
        return False

    def _judge_cap_by_union(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]], C_cap: np.ndarray, r_cap: float):
        Vp = np.array([V[0], V[1], V[2] - C_cap[2]])
        C_flat = np.array([C_cap[0], C_cap[1], 0.0])
        hits: List[int] = []
        for k, (S, R) in enumerate(spheres):
            Sp = np.array([S[0], S[1], S[2] - C_cap[2]])
            j = OcclusionJudge(Vp, C_flat, r_cap, Sp, R)
            res = j.is_fully_occluded()
            if bool(res.occluded):
                hits.append(k)
        return (len(hits) > 0), hits

    def _fully_occluded_sampling(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        pts = self._pts
        total = int(pts.shape[0]) if pts is not None else 0
        blocked = 0
        uncovered = []
        for i, P in enumerate(pts):
            if self._hit_any_sphere(V, P, spheres):
                blocked += 1
            else:
                uncovered.append(i)
        return (blocked == total), dict(total_points=total, blocked_points=blocked, uncovered_indices=uncovered[:16])

    def _fully_occluded_judge_caps(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        spheres = list(spheres)
        if len(spheres) == 0:
            return False, dict(mode="judge_caps", bottom=False, top=False, bottom_hits=[], top_hits=[])
        Cb = self.cyl.C_base
        Ct = self.cyl.C_base + np.array([0.0, 0.0, self.cyl.h])
        bottom_ok, bottom_hits = self._judge_cap_by_union(V, spheres, Cb, self.cyl.r)
        top_ok, top_hits = self._judge_cap_by_union(V, spheres, Ct, self.cyl.r)
        ok = bool(bottom_ok and top_ok)
        return ok, dict(mode="judge_caps", bottom=bool(bottom_ok), top=bool(top_ok), bottom_hits=bottom_hits[:8], top_hits=top_hits[:8])

    def fully_occluded(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        if self.method == "judge_caps":
            return self._fully_occluded_judge_caps(V, spheres)
        return self._fully_occluded_sampling(V, spheres)
