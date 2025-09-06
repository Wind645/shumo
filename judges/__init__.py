"""
Hard dependency version of judges package initializer.

Environment guarantee: torch and all torch-based judge modules ARE available.
(You requested a 100% torch-present environment, so no optional / fallback logic.)

Public API (import surface):

NumPy (exact + rough):
    OcclusionJudge
    OcclusionResult
    VectorizedOcclusionJudge
    VectorizedOcclusionResult
    vectorized_circle_fully_occluded_by_sphere
    RoughOcclusionJudge
    RoughVectorizedOcclusionJudge

Torch accelerated (exact + variants + rough + sampled):
    VectorizedOcclusionResultTorch
    vectorized_circle_fully_occluded_by_sphere_torch
    vectorized_circle_fully_occluded_by_sphere_torch_newton
    batch_occluded_time_caps_torch
    batch_occluded_time_caps_torch_newton
    batch_occluded_time_caps_torch_sampled
    TorchRoughVectorizedOcclusionJudge
    TorchVectorizedOcclusionResult

Meta:
    TORCH_AVAILABLE  (always True here)

Notes:
- This file intentionally raises ImportError immediately if any torch
  extension modules are missing, providing fail-fast behavior.
- If later you want a graceful (optional) torch path, reintroduce a try/except
  guard around the torch imports.
"""

from .judge import (
    OcclusionJudge,
    OcclusionResult,
)
from .vectorized_judge import (
    VectorizedOcclusionJudge,
    VectorizedOcclusionResult,
    vectorized_circle_fully_occluded_by_sphere,
)
from .rough_judge import (
    RoughOcclusionJudge,
    RoughVectorizedOcclusionJudge,
)

# Torch-backed implementations (hard dependency)
from .vectorized_judge_torch import (
    VectorizedOcclusionResultTorch,
    vectorized_circle_fully_occluded_by_sphere_torch,
    vectorized_circle_fully_occluded_by_sphere_torch_newton,
    batch_occluded_time_caps_torch,
    batch_occluded_time_caps_torch_newton,
)
from .rough_judge_torch import (
    TorchRoughVectorizedOcclusionJudge,
    TorchVectorizedOcclusionResult,
)
from .vectorized_judge_torch_sampled import (
    batch_occluded_time_caps_torch_sampled,
)
# Cylinder occlusion APIs (pure numpy + torch variants) – self contained inside judges/
from .cylinder_occlusion import (
    sample_cylinder_surface_points,
    cylinder_fully_occluded_sampling,
    cylinder_caps_fully_occluded,
    cylinder_caps_fully_occluded_exact,
    cylinder_caps_fully_occluded_rough,
    cylinder_fully_occluded_sampling_torch,
    cylinder_caps_fully_occluded_exact_torch,
    cylinder_caps_fully_occluded_rough_torch,
)
from .router import occlusion_time_router

TORCH_AVAILABLE = True  # Hard guarantee in this variant

__all__ = [
    # Base / NumPy
    "OcclusionJudge",
    "OcclusionResult",
    "VectorizedOcclusionJudge",
    "VectorizedOcclusionResult",
    "vectorized_circle_fully_occluded_by_sphere",
    "RoughOcclusionJudge",
    "RoughVectorizedOcclusionJudge",
    # Torch exact / accelerated
    "VectorizedOcclusionResultTorch",
    "vectorized_circle_fully_occluded_by_sphere_torch",
    "vectorized_circle_fully_occluded_by_sphere_torch_newton",
    "batch_occluded_time_caps_torch",
    "batch_occluded_time_caps_torch_newton",
    "batch_occluded_time_caps_torch_sampled",
    # Torch rough
    "TorchRoughVectorizedOcclusionJudge",
    "TorchVectorizedOcclusionResult",
    # Backward-compat wrapper classes
    "TorchVectorizedOcclusionJudge",
    "TorchVectorizedOcclusionJudgeSampled",
    # Cylinder occlusion (sampling + caps exact/rough) + torch variants
    "sample_cylinder_surface_points",
    "cylinder_fully_occluded_sampling",
    "cylinder_caps_fully_occluded",
    "cylinder_caps_fully_occluded_exact",
    "cylinder_caps_fully_occluded_rough",
    "cylinder_fully_occluded_sampling_torch",
    "cylinder_caps_fully_occluded_exact_torch",
    "cylinder_caps_fully_occluded_rough_torch",
    # High-level router
    "occlusion_time_router",
    # Meta
    "TORCH_AVAILABLE",
]

# ---------------------------------------------------------------------------
# Backward compatibility wrapper classes expected by legacy code (e.g. verify.py)
# These provide the older OO interface:
#   judge = VectorizedOcclusionJudge(V, circles_array, spheres_array)
#   mat = judge.compute_occlusion_matrix()
#
# circles_array shape: (Nc, 3) -> (x_c, y_c, r)
# spheres_array shape: (Ns, 4) -> (x_s, y_s, z_s, R)
# ---------------------------------------------------------------------------

# Wrap NumPy version only if the class does not already provide the legacy interface.
try:
    import numpy as _np
    if not hasattr(VectorizedOcclusionJudge, "compute_occlusion_matrix"):
        class _VectorizedOcclusionJudgeLegacy(VectorizedOcclusionJudge):  # type: ignore
            def __init__(self, V, circles_array, spheres_array):
                super().__init__()
                self._V = _np.asarray(V, dtype=_np.float64).reshape(3)
                ca = _np.asarray(circles_array, dtype=_np.float64)
                sa = _np.asarray(spheres_array, dtype=_np.float64)
                if ca.ndim != 2 or ca.shape[1] != 3:
                    raise ValueError("circles_array must be (Nc,3) with columns (x,y,r)")
                if sa.ndim != 2 or sa.shape[1] != 4:
                    raise ValueError("spheres_array must be (Ns,4) with columns (x,y,z,R)")
                self._C = _np.column_stack([ca[:, 0], ca[:, 1], _np.zeros(ca.shape[0])])
                self._r = ca[:, 2]
                self._S_full = sa[:, :3]
                self._R_full = sa[:, 3]

            def compute_occlusion_matrix(self):
                Nc = self._C.shape[0]
                Ns = self._S_full.shape[0]
                out = _np.zeros((Nc, Ns), dtype=bool)
                V_batch_base = _np.repeat(self._V[_np.newaxis, :], Nc, axis=0)
                for j in range(Ns):
                    V_batch = V_batch_base
                    C_batch = self._C
                    r_batch = self._r
                    S_batch = _np.repeat(self._S_full[j][_np.newaxis, :], Nc, axis=0)
                    R_batch = _np.full(Nc, self._R_full[j], dtype=_np.float64)
                    res = vectorized_circle_fully_occluded_by_sphere(V_batch, C_batch, r_batch, S_batch, R_batch)
                    out[:, j] = res.occluded
                return out

        VectorizedOcclusionJudge = _VectorizedOcclusionJudgeLegacy  # type: ignore
except Exception:
    pass  # Do not block import if legacy wrapper creation fails unexpectedly.

# Torch wrapper classes
import torch as _torch

class TorchVectorizedOcclusionJudge:
    """
    Backward-compatible Torch accelerated vectorized judge with legacy API.

    Usage:
        judge = TorchVectorizedOcclusionJudge(V, circles_array, spheres_array)
        mat = judge.compute_occlusion_matrix()  # (Nc, Ns) bool numpy array
    """
    def __init__(self, V, circles_array, spheres_array, *, device=None, dtype=_torch.float64):
        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        self.device = _torch.device(device)
        self.dtype = dtype
        import numpy as _np
        self._V = _np.asarray(V, dtype=_np.float64).reshape(3)
        ca = _np.asarray(circles_array, dtype=_np.float64)
        sa = _np.asarray(spheres_array, dtype=_np.float64)
        if ca.ndim != 2 or ca.shape[1] != 3:
            raise ValueError("circles_array must be (Nc,3) (x,y,r)")
        if sa.ndim != 2 or sa.shape[1] != 4:
            raise ValueError("spheres_array must be (Ns,4) (x,y,z,R)")
        self._C = _np.column_stack([ca[:, 0], ca[:, 1], _np.zeros(ca.shape[0])])
        self._r = ca[:, 2]
        self._S_full = sa[:, :3]
        self._R_full = sa[:, 3]

    def compute_occlusion_matrix(self):
        from .vectorized_judge_torch import vectorized_circle_fully_occluded_by_sphere_torch as _torch_eval
        V_np = self._V
        import numpy as _np
        Nc = self._C.shape[0]
        Ns = self._S_full.shape[0]
        out = _np.zeros((Nc, Ns), dtype=bool)
        V_batch_base = _np.repeat(V_np[_np.newaxis, :], Nc, axis=0)
        for j in range(Ns):
            Vb = _torch.as_tensor(V_batch_base, dtype=self.dtype, device=self.device)
            Cb = _torch.as_tensor(self._C, dtype=self.dtype, device=self.device)
            rb = _torch.as_tensor(self._r, dtype=self.dtype, device=self.device)
            Sb = _torch.as_tensor(_np.repeat(self._S_full[j][_np.newaxis, :], Nc, axis=0), dtype=self.dtype, device=self.device)
            Rb = _torch.full((Nc,), float(self._R_full[j]), dtype=self.dtype, device=self.device)
            res = _torch_eval(Vb, Cb, rb, Sb, Rb)
            out[:, j] = res.occluded.detach().cpu().numpy().astype(bool)
        return out


class TorchVectorizedOcclusionJudgeSampled(TorchVectorizedOcclusionJudge):
    """
    Approximate sampled version (currently identical to exact torch wrapper for
    per-cap occlusion matrix; kept for backward compatibility).
    """
    # Could override compute_occlusion_matrix with a sampled approximation if desired.
    pass
