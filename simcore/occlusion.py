from __future__ import annotations
from typing import Iterable, List, Tuple
import numpy as np

# 原先依赖 baolijiefa.CylinderOcclusionJudge 提供的线段-球相交与采样逻辑。
# 该文件已移除, 这里内联最小必要函数, 避免额外抽象。
# NOTE (migration):
#   新的统一圆柱遮挡判定/采样工具已迁移到 judges.cylinder_occlusion 模块中
#   (提供 sampling / judge_caps / rough_caps 以及对应 torch 版本)。
#   后续新代码若仅需判定，可直接:
#       from judges.cylinder_occlusion import cylinder_caps_fully_occluded_exact
#   或使用:
#       from judges.cylinder_occlusion import (
#           cylinder_fully_occluded_sampling,
#           cylinder_caps_fully_occluded_exact,
#           cylinder_caps_fully_occluded_rough,
#           cylinder_fully_occluded_sampling_torch,
#           cylinder_caps_fully_occluded_exact_torch,
#           cylinder_caps_fully_occluded_rough_torch,
#       )
#   本文件仍保留以兼容现有 simcore 内部调用；清理完成后可考虑删除，统一走 judges/。
def _segment_intersects_sphere(V: np.ndarray, P: np.ndarray, S: np.ndarray, R: float) -> bool:
    """线段 VP 与球(S,R) 是否相交 (存在 t∈[0,1] 使得 |V + t(P-V) - S| <= R)。"""
    V = np.asarray(V, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    d = P - V
    a = float(np.dot(d, d))
    if a == 0.0:  # 退化为点
        return float(np.dot(V - S, V - S)) <= R * R + 1e-12
    b = 2.0 * float(np.dot(d, V - S))
    c = float(np.dot(V - S, V - S) - R * R)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sqrt_disc = disc ** 0.5
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

def _sample_cylinder_points(C_base: np.ndarray, r: float, h: float, *,
                            n_theta: int, n_h: int, n_cap_radial: int, check_caps: bool) -> np.ndarray:
    """生成圆柱(含端面)采样点集合 (N,3)。

    侧面: n_theta * (n_h+1)
    端面: 若启用, 两端各 n_theta * (n_cap_radial-1) + 1 (中心)
    """
    C_base = np.asarray(C_base, dtype=np.float64)
    x0, y0, z0 = C_base
    z1 = z0 + h
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(z0, z1, n_h + 1)
    pts: List[Tuple[float,float,float]] = []
    for z in zs:
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        xs = x0 + r * cos_t
        ys = y0 + r * sin_t
        for x, y in zip(xs, ys):
            pts.append((x, y, z))
    if check_caps:
        if n_cap_radial <= 1:
            radii = [0.0]
        else:
            # 与原实现一致: sqrt(i/(n-1)) 生成近似均匀面积环
            radii = [0.0] + [r * (i / (n_cap_radial - 1)) ** 0.5 for i in range(1, n_cap_radial)]
        for z_cap in (z0, z1):
            for rr in radii:
                if rr == 0.0:
                    pts.append((x0, y0, z_cap))
                else:
                    cos_t = np.cos(thetas)
                    sin_t = np.sin(thetas)
                    xs = x0 + rr * cos_t
                    ys = y0 + rr * sin_t
                    for x, y in zip(xs, ys):
                        pts.append((x, y, z_cap))
    return np.asarray(pts, dtype=np.float64)

from judges.judge import OcclusionJudge
from judges.rough_judge import RoughOcclusionJudge  # 新增: 近似判定
from .entities import Cylinder
# ---- Torch support (optional) ----
try:
    from judges import TORCH_AVAILABLE
except Exception:
    TORCH_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        from judges import TorchRoughVectorizedOcclusionJudge  # rough_caps_torch
        from judges.vectorized_judge_torch import vectorized_circle_fully_occluded_by_sphere_torch  # judge_caps_torch
    except Exception:
        TORCH_AVAILABLE = False

Vec3 = np.ndarray
# --- Guarantee optional torch symbols exist (avoid “possibly unbound” diagnostics) ---
try:
    TorchRoughVectorizedOcclusionJudge  # type: ignore[name-defined]
except NameError:
    TorchRoughVectorizedOcclusionJudge = None  # type: ignore
try:
    vectorized_circle_fully_occluded_by_sphere_torch  # type: ignore[name-defined]
except NameError:
    vectorized_circle_fully_occluded_by_sphere_torch = None  # type: ignore

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
        if self.method in ("sampling", "sampling_torch"):
            # 直接预采样 (sampling_torch 也需要点集供 GPU 向量化或回退 CPU)
            self._pts = _sample_cylinder_points(
                self.cyl.C_base, self.cyl.r, self.cyl.h,
                n_theta=self.n_theta, n_h=self.n_h,
                n_cap_radial=self.n_cap_radial, check_caps=self.check_caps
            )

    @staticmethod
    def _hit_any_sphere(V: np.ndarray, P: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]) -> bool:
        for S, R in spheres:
            if _segment_intersects_sphere(V, P, S, R):
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

    def _judge_cap_by_union_rough(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]], C_cap: np.ndarray, r_cap: float):
        """粗糙近似版本: 使用 RoughOcclusionJudge (只会漏判, 不会误判)。"""
        Vp = np.array([V[0], V[1], V[2] - C_cap[2]])
        C_flat = np.array([C_cap[0], C_cap[1], 0.0])
        hits: List[int] = []
        for k, (S, R) in enumerate(spheres):
            Sp = np.array([S[0], S[1], S[2] - C_cap[2]])
            j = RoughOcclusionJudge(Vp, C_flat, r_cap, Sp, R)
            res = j.is_fully_occluded()
            if bool(res.occluded):
                hits.append(k)
        return (len(hits) > 0), hits

    def _fully_occluded_sampling(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        pts = self._pts
        total = int(pts.shape[0]) if pts is not None else 0
        blocked = 0
        uncovered = []
        if pts is None:
            return False, dict(total_points=0, blocked_points=0, uncovered_indices=[], note="no_points")
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

    def _fully_occluded_rough_caps(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        spheres = list(spheres)
        if len(spheres) == 0:
            return False, dict(mode="rough_caps", bottom=False, top=False, bottom_hits=[], top_hits=[])
        Cb = self.cyl.C_base
        Ct = self.cyl.C_base + np.array([0.0, 0.0, self.cyl.h])
        bottom_ok, bottom_hits = self._judge_cap_by_union_rough(V, spheres, Cb, self.cyl.r)
        top_ok, top_hits = self._judge_cap_by_union_rough(V, spheres, Ct, self.cyl.r)
        ok = bool(bottom_ok and top_ok)
        return ok, dict(mode="rough_caps", bottom=bool(bottom_ok), top=bool(top_ok), bottom_hits=bottom_hits[:8], top_hits=top_hits[:8])

    # ---- Torch accelerated helpers ----
    def _cap_union_judge_caps_torch(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]], C_cap: np.ndarray, r_cap: float):
        """Check if cap (center C_cap, radius r_cap) fully occluded by union of spheres using torch vectorized judge."""
        if not TORCH_AVAILABLE:
            return False, []
        spheres_list = list(spheres)
        if not spheres_list:
            return False, []
        import torch
        n = len(spheres_list)
        V_arr = np.repeat(V.reshape(1, 3), n, axis=0)
        C_arr = np.repeat(C_cap.reshape(1, 3), n, axis=0)
        r_arr = np.full(n, r_cap, dtype=np.float64)
        S_arr = np.stack([s[0] for s in spheres_list]).astype(np.float64)
        R_arr = np.array([s[1] for s in spheres_list], dtype=np.float64)
        V_t = torch.as_tensor(V_arr, dtype=torch.float64)
        C_t = torch.as_tensor(C_arr, dtype=torch.float64)
        r_t = torch.as_tensor(r_arr, dtype=torch.float64)
        S_t = torch.as_tensor(S_arr, dtype=torch.float64)
        R_t = torch.as_tensor(R_arr, dtype=torch.float64)
        if 'vectorized_circle_fully_occluded_by_sphere_torch' not in globals() or vectorized_circle_fully_occluded_by_sphere_torch is None:
            return False, []
        res = vectorized_circle_fully_occluded_by_sphere_torch(V_t, C_t, r_t, S_t, R_t)
        hits_idx = res.occluded.nonzero().flatten().tolist()
        return (len(hits_idx) > 0), hits_idx[:8]

    def _cap_union_rough_caps_torch(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]], C_cap: np.ndarray, r_cap: float):
        """Rough torch version using TorchRoughVectorizedOcclusionJudge (sufficient condition)."""
        if not TORCH_AVAILABLE:
            return False, []
        spheres_list = list(spheres)
        if not spheres_list:
            return False, []
        import torch
        n = len(spheres_list)
        V_arr = np.repeat(V.reshape(1, 3), n, axis=0)
        C_arr = np.repeat(C_cap.reshape(1, 3), n, axis=0)
        r_arr = np.full(n, r_cap, dtype=np.float32)
        S_arr = np.stack([s[0] for s in spheres_list]).astype(np.float32)
        R_arr = np.array([s[1] for s in spheres_list], dtype=np.float32)
        if 'TorchRoughVectorizedOcclusionJudge' not in globals() or TorchRoughVectorizedOcclusionJudge is None:
            return False, []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # if not hasattr(self, '_rough_caps_torch_device_logged'):
        #     print(f"[OcclusionTorch] rough_caps_torch using device={device}")
        #     self._rough_caps_torch_device_logged = True
        judge = TorchRoughVectorizedOcclusionJudge(device=device, dtype=torch.float32)
        res = judge.judge_batch(V_arr, C_arr, r_arr, S_arr, R_arr)
        hits_idx = res.occluded.nonzero().flatten().tolist()
        return (len(hits_idx) > 0), hits_idx[:8]

    def _fully_occluded_judge_caps_torch(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        spheres_list = list(spheres)
        if len(spheres_list) == 0:
            return False, dict(mode="judge_caps_torch", bottom=False, top=False, bottom_hits=[], top_hits=[])
        Cb = self.cyl.C_base
        Ct = self.cyl.C_base + np.array([0.0, 0.0, self.cyl.h])
        bottom_ok, bottom_hits = self._cap_union_judge_caps_torch(V, spheres_list, Cb, self.cyl.r)
        top_ok, top_hits = self._cap_union_judge_caps_torch(V, spheres_list, Ct, self.cyl.r)
        ok = bool(bottom_ok and top_ok)
        return ok, dict(mode="judge_caps_torch", bottom=bool(bottom_ok), top=bool(top_ok),
                        bottom_hits=bottom_hits, top_hits=top_hits)

    def _fully_occluded_rough_caps_torch(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        spheres_list = list(spheres)
        if len(spheres_list) == 0:
            return False, dict(mode="rough_caps_torch", bottom=False, top=False, bottom_hits=[], top_hits=[])
        Cb = self.cyl.C_base
        Ct = self.cyl.C_base + np.array([0.0, 0.0, self.cyl.h])
        bottom_ok, bottom_hits = self._cap_union_rough_caps_torch(V, spheres_list, Cb, self.cyl.r)
        top_ok, top_hits = self._cap_union_rough_caps_torch(V, spheres_list, Ct, self.cyl.r)
        ok = bool(bottom_ok and top_ok)
        return ok, dict(mode="rough_caps_torch", bottom=bool(bottom_ok), top=bool(top_ok),
                        bottom_hits=bottom_hits, top_hits=top_hits)

    def _fully_occluded_sampling_torch(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        """Torch vectorized segment-sphere test for pre-sampled cylinder surface points (fallback to numpy if torch unavailable)."""
        if not TORCH_AVAILABLE:
            # 回退前若尚未采样则补采
            if self._pts is None:
                self._pts = _sample_cylinder_points(
                    self.cyl.C_base, self.cyl.r, self.cyl.h,
                    n_theta=self.n_theta, n_h=self.n_h,
                    n_cap_radial=self.n_cap_radial, check_caps=self.check_caps
                )
            return self._fully_occluded_sampling(V, spheres)
        spheres_list = list(spheres)
        if len(spheres_list) == 0:
            return False, dict(mode="sampling_torch", total_points=int(self._pts.shape[0]) if self._pts is not None else 0,
                               blocked_points=0, uncovered_indices=[])
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pts = self._pts
        if pts is None or pts.shape[0] == 0:
            # 动态补采 (理论上不应发生, 防御性处理)
            self._pts = _sample_cylinder_points(
                self.cyl.C_base, self.cyl.r, self.cyl.h,
                n_theta=self.n_theta, n_h=self.n_h,
                n_cap_radial=self.n_cap_radial, check_caps=self.check_caps
            )
            pts = self._pts
            if pts is None or pts.shape[0] == 0:
                return False, dict(mode="sampling_torch", total_points=0, blocked_points=0, uncovered_indices=[])
        Vv = torch.as_tensor(V.reshape(1, 3), dtype=torch.float64, device=device)
        P = torch.as_tensor(pts, dtype=torch.float64, device=device)  # (M,3)
        S = torch.as_tensor(np.stack([s[0] for s in spheres_list]), dtype=torch.float64, device=device)  # (K,3)
        R = torch.as_tensor([s[1] for s in spheres_list], dtype=torch.float64, device=device).view(-1, 1)  # (K,1)

        # Segments: from V to each P => L = P - V (M,3)
        L = P - Vv  # (M,3)
        a = (L * L).sum(-1).clamp(min=1e-18)  # (M,)
        # Expand for spheres: (K,M,3)
        Lk = L.unsqueeze(0)
        Vrep = Vv.unsqueeze(0).expand(S.shape[0], -1, -1)  # (K,1,3)->(K,M,3) after broadcast
        Vrep = Vrep.expand(-1, L.shape[0], -1)
        Sexp = S.unsqueeze(1).expand(-1, L.shape[0], -1)   # (K,M,3)
        d = Lk  # reuse
        a_exp = a.unsqueeze(0)                             # (1,M)
        b = 2.0 * (d * (Vrep - Sexp)).sum(-1)              # (K,M)
        c = ((Vrep - Sexp) * (Vrep - Sexp)).sum(-1) - (R * R)  # (K,M)
        disc = b * b - 4.0 * a_exp * c
        hit = disc >= 0.0
        # Need t roots inside [0,1]
        sqrt_disc = torch.zeros_like(disc, device=device)
        sqrt_disc[hit] = torch.sqrt(disc[hit])
        t1 = (-b - sqrt_disc) / (2.0 * a_exp)
        t2 = (-b + sqrt_disc) / (2.0 * a_exp)
        seg_hit = hit & ((t1 >= 0.0) & (t1 <= 1.0) | (t2 >= 0.0) & (t2 <= 1.0))
        blocked_points = seg_hit.any(dim=0)  # (M,)
        blocked = int(blocked_points.sum().item())
        total = int(pts.shape[0])
        uncovered_idx = torch.nonzero(~blocked_points).flatten().tolist()[:16]
        return (blocked == total), dict(mode="sampling_torch", total_points=total,
                                        blocked_points=blocked,
                                        uncovered_indices=uncovered_idx,
                                        device=str(device))

    def fully_occluded(self, V: np.ndarray, spheres: Iterable[Tuple[np.ndarray, float]]):
        # Torch-dispatch first
        if self.method == "judge_caps_torch":
            return self._fully_occluded_judge_caps_torch(V, spheres)
        if self.method == "rough_caps_torch":
            return self._fully_occluded_rough_caps_torch(V, spheres)
        if self.method == "sampling_torch":
            return self._fully_occluded_sampling_torch(V, spheres)
        # Existing CPU methods
        if self.method == "judge_caps":
            return self._fully_occluded_judge_caps(V, spheres)
        if self.method == "rough_caps":
            return self._fully_occluded_rough_caps(V, spheres)
        return self._fully_occluded_sampling(V, spheres)
