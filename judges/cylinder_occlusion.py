"""
Cylinder occlusion algorithms (sampling / analytic cap union) placed inside judges package.

Goal:
    Provide self-contained cylinder occlusion utilities here (independent of simcore/)
    so higher-level optimization / simulation code can import from `judges` only.

Provided:
    - sample_cylinder_surface_points(...)
    - cylinder_fully_occluded_sampling(...)
    - cylinder_caps_fully_occluded_exact(...)
    - cylinder_caps_fully_occluded_rough(...)
    - Torch accelerated counterparts (if torch is available):
        * cylinder_caps_fully_occluded_exact_torch(...)
        * cylinder_caps_fully_occluded_rough_torch(...)
        * cylinder_fully_occluded_sampling_torch(...)

Concepts:
    A vertical cylinder defined by:
        base center C_base = (x0, y0, z0)
        radius r
        height h     (top center at C_base + (0,0,h))

    We test whether ALL points on the cylinder surface (or a sampled subset) are fully
    occluded by the UNION of a set of spheres (S_i, R_i) from a viewpoint V.

    Three strategies:

    1. sampling:
        Discretize the side surface (n_theta * (n_h+1) points) plus (optionally)
        two caps (radial rings) -> total M points. Each sample point P tests if
        segment VP intersects ANY sphere (sufficient for that sample).
        Full occlusion if every sample is covered.
        (May produce false negatives if sampling resolution is low; no false positives.)

    2. judge_caps (exact caps union + implicit side omission):
        Treat bottom and top circular caps as circles (in planes z = z0 and z=z0+h).
        For each cap, check if it is fully occluded by at least one sphere (i.e.,
        there exists some sphere whose tangent cone fully covers the entire circle).
        Uses exact circle-sphere judge batch function.

        NOTE: This only certifies the two end caps are covered; if you need full
        lateral surface guarantee, you'd combine with side sampling or extend
        geometry. In many original use cases, the lateral surface was NOT required
        (only need the “hole” up & down blocked). Kept semantics identical to the
        previous `judge_caps` naming.

    3. rough_caps (approximate caps union):
        Same as judge_caps but using rough (sufficient) occlusion judge which may
        yield additional false negatives but is cheaper.

Torch variants mirror (2) and (3) using torch-based vectorized circle judge to offload
heavy lifting to GPU if desired.

Return Format:
    All functions return (occluded: bool, info: dict). The dict TRY to share common keys:
        mode: str, one of {'sampling', 'judge_caps', 'rough_caps', 'sampling_torch',
                           'judge_caps_torch', 'rough_caps_torch'}
        For sampling:
            total_points, blocked_points, uncovered_indices (<=16)
        For cap-based:
            bottom: bool, top: bool
            bottom_hits: list[int]  (indices of spheres that individually cover bottom cap)
            top_hits:    list[int]
        Additional fields may be added in the future.

Dependencies:
    - Pure NumPy path depends on:
        vectorized_circle_fully_occluded_by_sphere
        RoughOcclusionJudge (for rough single-case fallback if needed)
    - Torch path depends on:
        vectorized_circle_fully_occluded_by_sphere_torch (import guarded)

Usage Example:
    from judges.cylinder_occlusion import (
        sample_cylinder_surface_points,
        cylinder_fully_occluded_sampling,
        cylinder_caps_fully_occluded_exact,
    )

    Cb = np.array([0.0, 200.0, 0.0])
    V  = np.array([1000.0, 0.0, 300.0])
    spheres = [(np.array([50.0, 210.0, 40.0]), 25.0)]
    ok, stats = cylinder_caps_fully_occluded_exact(V, spheres, Cb, 7.0, 10.0)
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Optional
import numpy as np

# Public re-exports (import locally to avoid circulars when this file imported very early)
from .vectorized_judge import (
    vectorized_circle_fully_occluded_by_sphere,
)
from .rough_judge import RoughOcclusionJudge
try:
    from .vectorized_judge_torch import (
        vectorized_circle_fully_occluded_by_sphere_torch,
    )
    from .rough_judge_torch import (
        TorchRoughVectorizedOcclusionJudge,
    )
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


# -------------------------------------------------------------------------------------------------
# Sampling Utilities
# -------------------------------------------------------------------------------------------------

def sample_cylinder_surface_points(
    C_base: np.ndarray,
    r: float,
    h: float,
    *,
    n_theta: int = 48,
    n_h: int = 16,
    n_cap_radial: int = 6,
    check_caps: bool = True,
) -> np.ndarray:
    """
    Generate sample points (N,3) for the cylinder surface (side + optional caps).

    Side:
        For each of (n_h+1) z-rings (including top & bottom) place n_theta points.

    Caps (if check_caps):
        Two caps each with radial layers:
            radii = [0.0] + [r * sqrt(i/(n_cap_radial-1)) for i=1..n_cap_radial-1]
        Each non-center layer has n_theta points.

    Args:
        C_base: (3,) base center
        r: radius
        h: height
        n_theta: angular resolution
        n_h: vertical segmentation resolution
        n_cap_radial: radial layers including center (>=1)
        check_caps: include tops/bottoms

    Returns:
        pts: (N,3) numpy float64
    """
    C_base = np.asarray(C_base, dtype=np.float64).reshape(3)
    n_theta = int(max(4, n_theta))
    n_h = int(max(1, n_h))
    n_cap_radial = int(max(1, n_cap_radial))

    z0 = C_base[2]
    z1 = z0 + float(h)

    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    pts: List[Tuple[float, float, float]] = []

    # Side surface
    z_levels = np.linspace(z0, z1, n_h + 1)
    for z in z_levels:
        xs = C_base[0] + r * cos_t
        ys = C_base[1] + r * sin_t
        for x, y in zip(xs, ys):
            pts.append((x, y, z))

    if check_caps:
        if n_cap_radial == 1:
            radii = [0.0]
        else:
            radii = [0.0] + [r * np.sqrt(i / (n_cap_radial - 1)) for i in range(1, n_cap_radial)]
        for z_cap in (z0, z1):
            for rr in radii:
                if rr == 0.0:
                    pts.append((C_base[0], C_base[1], z_cap))
                else:
                    xs = C_base[0] + rr * cos_t
                    ys = C_base[1] + rr * sin_t
                    for x, y in zip(xs, ys):
                        pts.append((x, y, z_cap))

    return np.asarray(pts, dtype=np.float64)


def _segment_intersects_sphere(V: np.ndarray, P: np.ndarray, S: np.ndarray, R: float) -> bool:
    """
    Basic geometric test: does segment V->P intersect sphere (S,R)?
    Equivalent to solving |V + t(P-V) - S|^2 = R^2 for any t in [0,1].
    """
    V = np.asarray(V, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    d = P - V
    a = float(np.dot(d, d))
    if a <= 1e-18:  # degenerate
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


def cylinder_fully_occluded_sampling(
    V: np.ndarray,
    spheres: Iterable[Tuple[np.ndarray, float]],
    C_base: np.ndarray,
    r: float,
    h: float,
    *,
    n_theta: int = 48,
    n_h: int = 16,
    n_cap_radial: int = 6,
    check_caps: bool = True,
    points: Optional[np.ndarray] = None,
) -> Tuple[bool, Dict]:
    """
    Sampling-based test: a cylinder is considered fully occluded if ALL sampled
    surface points are occluded (the segment from V to that point intersects at
    least one sphere).

    Args:
        V: (3,) viewpoint
        spheres: iterable of (S, R)
        C_base, r, h: cylinder spec
        n_theta, n_h, n_cap_radial, check_caps: sampling resolution
        points: optional precomputed sample points (override generation)

    Returns:
        (occluded, info)
    """
    V = np.asarray(V, dtype=np.float64).reshape(3)
    spheres_list = list(spheres)
    if points is None:
        pts = sample_cylinder_surface_points(
            C_base, r, h,
            n_theta=n_theta,
            n_h=n_h,
            n_cap_radial=n_cap_radial,
            check_caps=check_caps,
        )
    else:
        pts = np.asarray(points, dtype=np.float64)
    total = int(pts.shape[0])
    if total == 0:
        return False, dict(mode="sampling", total_points=0, blocked_points=0, uncovered_indices=[], note="no_points")

    blocked = 0
    uncovered: List[int] = []
    for i, P in enumerate(pts):
        covered = False
        for (S, R) in spheres_list:
            if _segment_intersects_sphere(V, P, S, R):
                covered = True
                break
        if covered:
            blocked += 1
        else:
            uncovered.append(i)
        # early exit if any uncovered and no need to enumerate all?  -> we still gather stats
    occluded = (blocked == total)
    return occluded, dict(
        mode="sampling",
        total_points=total,
        blocked_points=blocked,
        uncovered_indices=uncovered[:16],
    )


# -------------------------------------------------------------------------------------------------
# Cap-based (exact) union approach
# -------------------------------------------------------------------------------------------------

def _cap_occlusion_by_sphere_union_exact(
    V: np.ndarray,
    spheres: List[Tuple[np.ndarray, float]],
    C_cap: np.ndarray,
    r_cap: float,
) -> Tuple[bool, List[int]]:
    """
    Determine if circular cap (center C_cap, radius r_cap in its own plane) is fully
    occluded by the union of spheres. We require that at least ONE sphere fully
    covers the cap (sufficient + necessary for that sphere alone).

    Implementation detail:
        Flatten plane by translating so cap plane is z=0 (subtract C_cap.z from V and sphere centers).
        Then apply vectorized circle-sphere occlusion (NumPy) with each sphere.

    Returns:
        (covered, hit_indices) where hit_indices are sphere indices that alone fully cover the cap.
    """
    if len(spheres) == 0:
        return False, []
    Vp = np.array([V[0], V[1], V[2] - C_cap[2]], dtype=np.float64)
    C_flat = np.array([C_cap[0], C_cap[1], 0.0], dtype=np.float64)

    n = len(spheres)
    V_batch = np.repeat(Vp.reshape(1, 3), n, axis=0)
    C_batch = np.repeat(C_flat.reshape(1, 3), n, axis=0)
    r_batch = np.full(n, r_cap, dtype=np.float64)
    S_batch = np.zeros((n, 3), dtype=np.float64)
    R_batch = np.zeros(n, dtype=np.float64)
    for i, (S, R) in enumerate(spheres):
        S_batch[i] = [S[0], S[1], S[2] - C_cap[2]]
        R_batch[i] = R

    res = vectorized_circle_fully_occluded_by_sphere(V_batch, C_batch, r_batch, S_batch, R_batch)
    hit_idx = np.nonzero(res.occluded)[0].tolist()
    return (len(hit_idx) > 0), hit_idx


def _cap_occlusion_by_sphere_union_rough(
    V: np.ndarray,
    spheres: List[Tuple[np.ndarray, float]],
    C_cap: np.ndarray,
    r_cap: float,
) -> Tuple[bool, List[int]]:
    """
    Rough version: use RoughOcclusionJudge per sphere (loop). No false positives,
    may have false negatives.
    """
    if len(spheres) == 0:
        return False, []
    Vp = np.array([V[0], V[1], V[2] - C_cap[2]], dtype=np.float64)
    C_flat = np.array([C_cap[0], C_cap[1], 0.0], dtype=np.float64)
    hits: List[int] = []
    for idx, (S, R) in enumerate(spheres):
        Sp = np.array([S[0], S[1], S[2] - C_cap[2]], dtype=np.float64)
        judge = RoughOcclusionJudge(Vp, C_flat, r_cap, Sp, R)
        if judge.is_fully_occluded().occluded:
            hits.append(idx)
    return (len(hits) > 0), hits


def cylinder_caps_fully_occluded_exact(
    V: np.ndarray,
    spheres: Iterable[Tuple[np.ndarray, float]],
    C_base: np.ndarray,
    r: float,
    h: float,
) -> Tuple[bool, Dict]:
    """
    Exact (per-cap) occlusion: bottom & top caps each must be fully occluded by
    AT LEAST one sphere individually.

    Side surface is NOT considered. Equivalent semantics to historical 'judge_caps'.

    Returns:
        (ok, info)
    """
    spheres_list = list(spheres)
    if len(spheres_list) == 0:
        return False, dict(mode="judge_caps", bottom=False, top=False, bottom_hits=[], top_hits=[])
    Cb = np.asarray(C_base, dtype=np.float64).reshape(3)
    Ct = Cb + np.array([0.0, 0.0, float(h)], dtype=np.float64)
    bottom_ok, bottom_hits = _cap_occlusion_by_sphere_union_exact(V, spheres_list, Cb, r)
    top_ok, top_hits = _cap_occlusion_by_sphere_union_exact(V, spheres_list, Ct, r)
    ok = bool(bottom_ok and top_ok)
    return ok, dict(
        mode="judge_caps",
        bottom=bool(bottom_ok),
        top=bool(top_ok),
        bottom_hits=bottom_hits[:8],
        top_hits=top_hits[:8],
    )


def cylinder_caps_fully_occluded_rough(
    V: np.ndarray,
    spheres: Iterable[Tuple[np.ndarray, float]],
    C_base: np.ndarray,
    r: float,
    h: float,
) -> Tuple[bool, Dict]:
    """
    Rough (approximate) per-cap union test using RoughOcclusionJudge.
    """
    spheres_list = list(spheres)
    if len(spheres_list) == 0:
        return False, dict(mode="rough_caps", bottom=False, top=False, bottom_hits=[], top_hits=[])
    Cb = np.asarray(C_base, dtype=np.float64).reshape(3)
    Ct = Cb + np.array([0.0, 0.0, float(h)], dtype=np.float64)
    bottom_ok, bottom_hits = _cap_occlusion_by_sphere_union_rough(V, spheres_list, Cb, r)
    top_ok, top_hits = _cap_occlusion_by_sphere_union_rough(V, spheres_list, Ct, r)
    ok = bool(bottom_ok and top_ok)
    return ok, dict(
        mode="rough_caps",
        bottom=bool(bottom_ok),
        top=bool(top_ok),
        bottom_hits=bottom_hits[:8],
        top_hits=top_hits[:8],
    )


# -------------------------------------------------------------------------------------------------
# Torch Accelerated Variants (Optional)
# -------------------------------------------------------------------------------------------------

def _cap_occlusion_by_sphere_union_exact_torch(
    V: np.ndarray,
    spheres: List[Tuple[np.ndarray, float]],
    C_cap: np.ndarray,
    r_cap: float,
    *,
    device: str = "cuda",
    dtype="float64",
):
    """
    Torch batch test analogous to _cap_occlusion_by_sphere_union_exact
    but uses vectorized_circle_fully_occluded_by_sphere_torch.
    """
    if not _TORCH_OK:
        return False, []
    import torch
    if len(spheres) == 0:
        return False, []
    Vp = np.array([V[0], V[1], V[2] - C_cap[2]], dtype=np.float64)
    C_flat = np.array([C_cap[0], C_cap[1], 0.0], dtype=np.float64)
    n = len(spheres)
    V_batch = np.repeat(Vp.reshape(1, 3), n, axis=0)
    C_batch = np.repeat(C_flat.reshape(1, 3), n, axis=0)
    r_batch = np.full(n, r_cap, dtype=np.float64)
    S_batch = np.zeros((n, 3), dtype=np.float64)
    R_batch = np.zeros(n, dtype=np.float64)
    for i, (S, R) in enumerate(spheres):
        S_batch[i] = [S[0], S[1], S[2] - C_cap[2]]
        R_batch[i] = R
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    tV = torch.as_tensor(V_batch, dtype=getattr(torch, dtype), device=dev)
    tC = torch.as_tensor(C_batch, dtype=getattr(torch, dtype), device=dev)
    tr = torch.as_tensor(r_batch, dtype=getattr(torch, dtype), device=dev)
    tS = torch.as_tensor(S_batch, dtype=getattr(torch, dtype), device=dev)
    tR = torch.as_tensor(R_batch, dtype=getattr(torch, dtype), device=dev)
    res = vectorized_circle_fully_occluded_by_sphere_torch(tV, tC, tr, tS, tR)
    hits = res.occluded.nonzero().flatten().tolist()
    return (len(hits) > 0), hits


def _cap_occlusion_by_sphere_union_rough_torch(
    V: np.ndarray,
    spheres: List[Tuple[np.ndarray, float]],
    C_cap: np.ndarray,
    r_cap: float,
    *,
    device: str = "cuda",
    dtype="float32",
):
    """
    Torch rough batch test using TorchRoughVectorizedOcclusionJudge.

    Performance improvement:
      - Cache TorchRoughVectorizedOcclusionJudge instances per (device,dtype) to avoid
        repeated construction for every frame & every cap check.
    """
    if not _TORCH_OK:
        return False, []
    import torch
    if len(spheres) == 0:
        return False, []
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    # ----- Judge cache (attached as function attribute to avoid global clutter) -----
    if not hasattr(_cap_occlusion_by_sphere_union_rough_torch, "_judge_cache"):
        _cap_occlusion_by_sphere_union_rough_torch._judge_cache = {}
    judge_cache = _cap_occlusion_by_sphere_union_rough_torch._judge_cache  # type: ignore[attr-defined]
    cache_key = (str(dev), dtype)
    judge = judge_cache.get(cache_key)
    if judge is None:
        judge = TorchRoughVectorizedOcclusionJudge(device=str(dev), dtype=getattr(torch, dtype))
        judge_cache[cache_key] = judge

    n = len(spheres)
    Vp = np.array([V[0], V[1], V[2] - C_cap[2]], dtype=np.float32)
    C_flat = np.array([C_cap[0], C_cap[1], 0.0], dtype=np.float32)
    V_arr = np.repeat(Vp.reshape(1, 3), n, axis=0)
    C_arr = np.repeat(C_flat.reshape(1, 3), n, axis=0)
    r_arr = np.full(n, r_cap, dtype=np.float32)
    S_arr = np.zeros((n, 3), dtype=np.float32)
    R_arr = np.zeros(n, dtype=np.float32)
    for i, (S, R) in enumerate(spheres):
        S_arr[i] = [S[0], S[1], S[2] - C_cap[2]]
        R_arr[i] = R

    res = judge.judge_batch(V_arr, C_arr, r_arr, S_arr, R_arr)
    hits = res.occluded.nonzero().flatten().tolist()
    return (len(hits) > 0), hits


def cylinder_caps_fully_occluded_exact_torch(
    V: np.ndarray,
    spheres: Iterable[Tuple[np.ndarray, float]],
    C_base: np.ndarray,
    r: float,
    h: float,
    *,
    device: str = "cuda",
    dtype: str = "float64",
) -> Tuple[bool, Dict]:
    """
    Torch-accelerated version of cylinder_caps_fully_occluded_exact.
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available for exact_torch method.")
    spheres_list = list(spheres)
    if len(spheres_list) == 0:
        return False, dict(mode="judge_caps_torch", bottom=False, top=False, bottom_hits=[], top_hits=[])
    Cb = np.asarray(C_base, dtype=np.float64).reshape(3)
    Ct = Cb + np.array([0.0, 0.0, float(h)], dtype=np.float64)
    bottom_ok, bottom_hits = _cap_occlusion_by_sphere_union_exact_torch(V, spheres_list, Cb, r, device=device, dtype=dtype)
    top_ok, top_hits = _cap_occlusion_by_sphere_union_exact_torch(V, spheres_list, Ct, r, device=device, dtype=dtype)
    ok = bool(bottom_ok and top_ok)
    return ok, dict(
        mode="judge_caps_torch",
        bottom=bool(bottom_ok),
        top=bool(top_ok),
        bottom_hits=bottom_hits[:8],
        top_hits=top_hits[:8],
        device=device,
    )


def cylinder_caps_fully_occluded_rough_torch(
    V: np.ndarray,
    spheres: Iterable[Tuple[np.ndarray, float]],
    C_base: np.ndarray,
    r: float,
    h: float,
    *,
    device: str = "cuda",
    dtype: str = "float32",
) -> Tuple[bool, Dict]:
    """
    Torch-accelerated rough version of cylinder_caps_fully_occluded_rough.
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available for rough_caps_torch method.")
    spheres_list = list(spheres)
    if len(spheres_list) == 0:
        return False, dict(mode="rough_caps_torch", bottom=False, top=False, bottom_hits=[], top_hits=[])
    Cb = np.asarray(C_base, dtype=np.float32).reshape(3)
    Ct = Cb + np.array([0.0, 0.0, float(h)], dtype=np.float32)
    bottom_ok, bottom_hits = _cap_occlusion_by_sphere_union_rough_torch(V, spheres_list, Cb, r, device=device, dtype=dtype)
    top_ok, top_hits = _cap_occlusion_by_sphere_union_rough_torch(V, spheres_list, Ct, r, device=device, dtype=dtype)
    ok = bool(bottom_ok and top_ok)
    return ok, dict(
        mode="rough_caps_torch",
        bottom=bool(bottom_ok),
        top=bool(top_ok),
        bottom_hits=bottom_hits[:8],
        top_hits=top_hits[:8],
        device=device,
    )


def cylinder_fully_occluded_sampling_torch(
    V: np.ndarray,
    spheres: Iterable[Tuple[np.ndarray, float]],
    C_base: np.ndarray,
    r: float,
    h: float,
    *,
    n_theta: int = 48,
    n_h: int = 16,
    n_cap_radial: int = 6,
    check_caps: bool = True,
    device: str = "cuda",
    dtype: str = "float64",
    points: Optional[np.ndarray] = None,
    chunk: int = 8192,
) -> Tuple[bool, Dict]:
    """
    Torch vectorized segment-sphere test for all sampled points simultaneously.

    If torch not available falls back to CPU sampling function.

    Args:
        chunk: process sample points in blocks to control GPU memory

    Returns:
        (occluded, info)
    """
    if not _TORCH_OK:
        # fallback
        return cylinder_fully_occluded_sampling(
            V, spheres, C_base, r, h,
            n_theta=n_theta, n_h=n_h, n_cap_radial=n_cap_radial, check_caps=check_caps, points=points
        )
    import torch
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    spheres_list = list(spheres)
    if points is None:
        pts = sample_cylinder_surface_points(
            C_base, r, h,
            n_theta=n_theta, n_h=n_h,
            n_cap_radial=n_cap_radial,
            check_caps=check_caps,
        )
    else:
        pts = np.asarray(points, dtype=np.float64)
    total = pts.shape[0]
    if total == 0:
        return False, dict(mode="sampling_torch", total_points=0, blocked_points=0, uncovered_indices=[], note="no_points")
    if len(spheres_list) == 0:
        return False, dict(mode="sampling_torch", total_points=total, blocked_points=0, uncovered_indices=list(range(min(16, total))))
    V_t = torch.as_tensor(V.reshape(1, 3), dtype=getattr(torch, dtype), device=dev)
    P_t = torch.as_tensor(pts, dtype=getattr(torch, dtype), device=dev)  # (M,3)
    S_t = torch.as_tensor(np.stack([s[0] for s in spheres_list]), dtype=getattr(torch, dtype), device=dev)  # (K,3)
    R_t = torch.as_tensor([s[1] for s in spheres_list], dtype=getattr(torch, dtype), device=dev)  # (K,)
    # We'll test (M,K) segments in manageable chunks
    M = P_t.shape[0]
    K = S_t.shape[0]
    blocked_mask = torch.zeros(M, dtype=torch.bool, device=dev)
    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        # segments for current slice
        seg = P_t[start:end] - V_t  # (m,3)
        a = (seg * seg).sum(-1).clamp(min=1e-18)  # (m,)
        # Expand across spheres => (K,m,3)
        seg_exp = seg.unsqueeze(0).expand(K, -1, -1)
        V_rep = V_t.unsqueeze(0).expand(K, -1, -1)  # (K,1,3)
        V_rep = V_rep.expand(-1, seg.shape[0], -1)
        S_exp = S_t.unsqueeze(1).expand(-1, seg.shape[0], -1)
        a_exp = a.unsqueeze(0)  # (1,m)
        b = 2.0 * (seg_exp * (V_rep - S_exp)).sum(-1)  # (K,m)
        c = ((V_rep - S_exp) * (V_rep - S_exp)).sum(-1) - R_t.view(-1, 1) ** 2
        disc = b * b - 4.0 * a_exp * c
        hit = disc >= 0.0
        sqrt_disc = torch.zeros_like(disc)
        sqrt_disc[hit] = torch.sqrt(disc[hit])
        t1 = (-b - sqrt_disc) / (2.0 * a_exp)
        t2 = (-b + sqrt_disc) / (2.0 * a_exp)
        seg_hit = hit & ((t1 >= 0.0) & (t1 <= 1.0) | (t2 >= 0.0) & (t2 <= 1.0))
        any_hit = seg_hit.any(dim=0)  # (m,)
        blocked_mask[start:end] = any_hit
        # Early exit if already found an uncovered and don't need full stats?
        # We still continue to gather uncovered indices for debug parity.
    blocked_points = int(blocked_mask.sum().item())
    uncovered_idx = torch.nonzero(~blocked_mask).flatten().tolist()
    occluded = (blocked_points == total)
    return occluded, dict(
        mode="sampling_torch",
        total_points=total,
        blocked_points=blocked_points,
        uncovered_indices=uncovered_idx[:16],
        spheres_count=K,
        device=str(dev),
    )


# -------------------------------------------------------------------------------------------------
# __all__
# -------------------------------------------------------------------------------------------------

def cylinder_caps_fully_occluded(
    V: np.ndarray,
    spheres: Iterable[Tuple[np.ndarray, float]],
    C_base: np.ndarray,
    r: float,
    h: float,
    *,
    method: str = "judge_caps",
    torch_device: str = "cuda",
    dtype_exact: str = "float64",
    dtype_rough: str = "float32",
) -> Tuple[bool, Dict]:
    """
    Unified dispatcher for cap-only cylinder occlusion (we explicitly ignore the lateral/surface side).

    Parameters:
        V: (3,) viewpoint
        spheres: iterable of (center(np.ndarray (3,)), radius(float))
        C_base, r, h: cylinder spec
        method:
            "judge_caps" | "exact" | "caps"                -> numpy exact
            "rough_caps" | "rough"                         -> numpy rough
            "judge_caps_torch" | "exact_torch" | "caps_torch" -> torch exact
            "rough_caps_torch" | "rough_torch"             -> torch rough
        torch_device: preferred torch device when torch variants selected
        dtype_exact / dtype_rough: dtypes passed to torch variants

    Returns:
        (ok, info_dict)
        info_dict at least contains:
            mode: underlying implementation id
            dispatch: the original method argument
            any method-specific fields (see underlying functions)

    Notes:
        - Side surface is intentionally not tested (requirement: “we do not care about side”).
        - Raises ValueError on unknown method.
    """
    m = method.lower()
    if m in ("judge_caps", "exact", "caps"):
        ok, info = cylinder_caps_fully_occluded_exact(V, spheres, C_base, r, h)
    elif m in ("rough_caps", "rough"):
        ok, info = cylinder_caps_fully_occluded_rough(V, spheres, C_base, r, h)
    elif m in ("judge_caps_torch", "exact_torch", "caps_torch"):
        ok, info = cylinder_caps_fully_occluded_exact_torch(
            V, spheres, C_base, r, h, device=torch_device, dtype=dtype_exact
        )
    elif m in ("rough_caps_torch", "rough_torch"):
        ok, info = cylinder_caps_fully_occluded_rough_torch(
            V, spheres, C_base, r, h, device=torch_device, dtype=dtype_rough
        )
    else:
        raise ValueError(f"Unknown cylinder caps occlusion method: {method}")
    # Annotate dispatch origin
    if "mode" not in info:
        info["mode"] = m
    info["dispatch"] = method
    return ok, info


__all__ = [
    # Sampling
    "sample_cylinder_surface_points",
    "cylinder_fully_occluded_sampling",
    "cylinder_fully_occluded_sampling_torch",
    # Cap methods (exact / rough)
    "cylinder_caps_fully_occluded_exact",
    "cylinder_caps_fully_occluded_exact_torch",
    "cylinder_caps_fully_occluded_rough",
    "cylinder_caps_fully_occluded_rough_torch",
    # Dispatcher
    "cylinder_caps_fully_occluded",
]
