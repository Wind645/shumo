"""
Torch GPU 加速的 rough (近似) 遮挡判定。

核心思想
========
将“烟雾圆”(半径 disk_radius，圆心 O) 视为其外接最小球 (同心，半径相同)；
目标物 (在本项目中是固定圆柱的外界球近似) 由球 (center=S, radius=R) 近似。
从观察点 V 出发，看两个球的视场锥(以 V 为顶点、包住各球的最小圆锥)。
若在角度意义上，小球(烟雾)的锥形完全覆盖目标球的锥形，则目标被完全遮挡。

严格充分条件(不产生误报，只可能漏判“部分遮挡”):
    gamma + beta <= alpha
其中:
    gamma = angle( u_S , u_C ) = 观察方向到目标中心与到烟雾中心方向夹角
    beta  = arcsin( r / ||C - V|| ),  烟雾球对应的“半角”
    alpha = arcsin( R / ||S - V|| ),  目标球对应的“半角”

如果 gamma + beta <= alpha，则可断定：从视点 V 看，烟雾圆至少完全遮住目标球(完全遮挡)。

本文件提供两类接口：
1. 类 TorchRoughVectorizedOcclusionJudge: 复刻与扩展单批 (N,) 判定（与用户示例风格一致）。
2. 针对模拟需求的 pairwise/broadcast 判定函数：
      occlusion_full_cover_matrix(posV, posO, ...)  -> (Nv, No) bool
      occlusion_partial_matrix(posV, posO, ...)     -> (Nv, No) bool (使用线性投影近似的“部分遮挡”条件)

注意：
- full cover 判定严格(不误报)，但只回答“是否完全遮住”，忽略部分遮挡。
- partial 判定使用 earlier rough.py 中的线性投影近似，可认为是“有交叠即遮挡”。
- 你可以根据任务需求选择何种判定。
- 所有函数/类都支持在 GPU (CUDA) 或 CPU 上运行；自动选择 GPU (若可用)。

依赖：
    PyTorch

示例用法 (pairwise):
    import torch
    from rough_torch import occlusion_partial_matrix
    missiles = torch.tensor([[20000.,0.,2000.]], device='cuda')
    smokes   = torch.tensor([[12000.,100.,1500.],
                             [15000.,-50.,1700.]], device='cuda')
    occ_any = occlusion_partial_matrix(missiles, smokes).any(dim=1)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

try:
    import torch
except Exception as e:  # pragma: no cover
    raise ImportError("需要安装 PyTorch 才能使用 rough_torch 模块") from e


# ====== 全局常量 (与 rough.py 保持一致) ======
SPHERE_CENTER = torch.tensor([0.0, 200.0, 5.0])   # 目标外接球中心
SPHERE_RADIUS = math.sqrt(74.0)                   # 目标外接球半径
DISK_RADIUS = 10.0                                # 烟雾圆半径
_EPS = 1e-12


# ==================== Dataclass for single-batch API ====================

@dataclass
class TorchVectorizedOcclusionResult:
    """
    与示例保持一致的结果容器：
        occluded: (N,) bool   - 是否完全遮挡 (full cover 判定)
        f_min: (N,) float     - 近似 cos(gamma + beta)；observer_in 时 NaN
        cos_alpha_s: (N,) float  - cos(alpha)；observer_in 时 NaN
        t_min: (N,) NaN (占位)
        valid: (N,) bool
    """
    occluded: torch.Tensor
    f_min: torch.Tensor
    cos_alpha_s: torch.Tensor
    t_min: torch.Tensor
    valid: torch.Tensor

    def to_numpy(self):
        return {
            "occluded": self.occluded.detach().cpu().numpy(),
            "f_min": self.f_min.detach().cpu().numpy(),
            "cos_alpha_s": self.cos_alpha_s.detach().cpu().numpy(),
            "t_min": self.t_min.detach().cpu().numpy(),
            "valid": self.valid.detach().cpu().numpy(),
        }


class TorchRoughVectorizedOcclusionJudge:
    """
    单批 (N,) 版本：
        输入：
            V: (N,3) 观察点
            C: (N,3) 烟雾圆心(小球中心)
            r: (N,)  烟雾半径 (可全常数)
            S: (N,3) 目标球心
            R: (N,)  目标球半径
        输出：TorchVectorizedOcclusionResult
    """
    def __init__(self, device: Optional[str] = None, dtype: torch.dtype = torch.float32):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, self.dtype)
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def judge_batch(
        self,
        V,
        C,
        r,
        S,
        R,
        stable: bool = True,
    ) -> TorchVectorizedOcclusionResult:
        Vt = self._to_tensor(V)
        Ct = self._to_tensor(C)
        St = self._to_tensor(S)
        rt = self._to_tensor(r).reshape(-1)
        Rt = self._to_tensor(R).reshape(-1)

        if Vt.ndim != 2 or Vt.shape[-1] != 3:
            raise ValueError("V must have shape (N,3)")
        N = Vt.shape[0]
        for name, T in (("C", Ct), ("S", St)):
            if T.shape != Vt.shape:
                raise ValueError(f"{name} must have shape (N,3)")
        if rt.shape != (N,) or Rt.shape != (N,):
            raise ValueError("r,R must have shape (N,)")

        eps = torch.tensor(_EPS, device=self.device, dtype=self.dtype)

        VS = St - Vt
        VC = Ct - Vt
        dS = torch.linalg.norm(VS, dim=1)
        dC = torch.linalg.norm(VC, dim=1)

        # 单位方向
        uS = torch.zeros_like(VS)
        maskS = dS > eps
        uS[maskS] = VS[maskS] / dS[maskS].unsqueeze(-1)

        uC = torch.zeros_like(VC)
        maskC = dC > eps
        uC[maskC] = VC[maskC] / dC[maskC].unsqueeze(-1)
        # 若 dC=0，用 uS 替代
        rep = (~maskC) & maskS
        uC[rep] = uS[rep]

        # 比例 (clamp 避免数值炸)
        ratio_s = torch.clamp(Rt / torch.where(dS > 0, dS, torch.ones_like(dS)), 0.0, 1.0)
        ratio_c = torch.clamp(rt / torch.where(dC > 0, dC, torch.ones_like(dC)), 0.0, 1.0)

        # 半角
        alpha = torch.arcsin(ratio_s)
        beta = torch.arcsin(ratio_c)

        cos_alpha = torch.sqrt(torch.clamp(1.0 - ratio_s * ratio_s, min=0.0))

        cos_gamma = torch.sum(uS * uC, dim=1).clamp(-1.0, 1.0)
        gamma = torch.arccos(cos_gamma)
        sin_gamma = torch.sqrt(torch.clamp(1.0 - cos_gamma * cos_gamma, min=0.0))
        sin_beta = ratio_c
        cos_beta = torch.sqrt(torch.clamp(1.0 - sin_beta * sin_beta, min=0.0))

        # 近似 cos(gamma+beta)
        f_min = cos_gamma * cos_beta - sin_gamma * sin_beta

        # 观察点在目标球内 -> 特殊处理
        observer_in = (dS <= eps) | (Rt >= dS)
        occluded = observer_in | ((gamma + beta) <= (alpha + 1e-12))

        nan = torch.full((1,), float("nan"), device=self.device, dtype=self.dtype)
        f_min = torch.where(observer_in, nan.expand_as(f_min), f_min)
        cos_alpha = torch.where(observer_in, nan.expand_as(cos_alpha), cos_alpha)
        t_min = torch.full((N,), float("nan"), device=self.device, dtype=self.dtype)
        valid = torch.ones((N,), dtype=torch.bool, device=self.device)

        return TorchVectorizedOcclusionResult(
            occluded=occluded.bool(),
            f_min=f_min,
            cos_alpha_s=cos_alpha,
            t_min=t_min,
            valid=valid,
        )


# ==================== Pairwise / Broadcast Helpers ====================

def _auto_device_dtype(posV, posO, device, dtype):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float32
    posV = torch.as_tensor(posV, device=device, dtype=dtype)
    posO = torch.as_tensor(posO, device=device, dtype=dtype)
    return posV, posO, device, dtype


@torch.no_grad()
def occlusion_full_cover_matrix(
    posV,
    posO,
    sphere_center: Tuple[float, float, float] = (0.0, 200.0, 5.0),
    sphere_radius: float = SPHERE_RADIUS,
    disk_radius: float = DISK_RADIUS,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    计算 (Nv, No) 的完全遮挡判定矩阵：
        True 表示该烟雾圆(视作球) 在该观察点下完全遮住目标球(保守充分条件)
        False 包含“未遮挡”或“部分遮挡”。

    参数:
        posV: (Nv,3) 观察点
        posO: (No,3) 烟雾圆心
    返回:
        (Nv, No) bool
    """
    posV, posO, device, dtype = _auto_device_dtype(posV, posO, device, dtype)
    sphere_center = torch.as_tensor(sphere_center, device=device, dtype=dtype)

    # 形状对齐
    V = posV[:, None, :]          # (Nv,1,3)
    O = posO[None, :, :]          # (1,No,3)
    S = sphere_center.view(1, 1, 3).expand_as(V)  # (Nv,1,3)

    VS = S - V
    VO = O - V

    dS = torch.linalg.norm(VS, dim=-1)  # (Nv,1)
    dO = torch.linalg.norm(VO, dim=-1)  # (Nv,No)

    eps = torch.tensor(_EPS, device=device, dtype=dtype)

    # 单位方向
    uS = torch.zeros_like(VS)
    mask_dS = dS > eps
    uS[mask_dS] = VS[mask_dS] / dS[mask_dS].unsqueeze(-1)

    uO = torch.zeros_like(VO)
    mask_dO = dO > eps
    uO[mask_dO] = VO[mask_dO] / dO[mask_dO].unsqueeze(-1)
    # dO=0 的点用 uS 的方向(广播)
    rep = (~mask_dO) & mask_dS.expand_as(mask_dO)
    uO[rep] = uS[rep // 1]  # rep 与 uS broadcast 兼容

    ratio_s = torch.clamp(sphere_radius / torch.where(dS > 0, dS, torch.ones_like(dS)), 0.0, 1.0)  # (Nv,1)
    ratio_o = torch.clamp(disk_radius / torch.where(dO > 0, dO, torch.ones_like(dO)), 0.0, 1.0)    # (Nv,No)

    alpha = torch.arcsin(ratio_s)  # (Nv,1)
    beta = torch.arcsin(ratio_o)   # (Nv,No)

    # cos gamma
    cos_gamma = (uS * uO).sum(dim=-1).clamp(-1.0, 1.0)  # (Nv,No)
    gamma = torch.arccos(cos_gamma)

    # 完全遮挡条件
    occluded_full = (gamma + beta) <= (alpha + 1e-12)  # broadcast alpha (Nv,1) across No
    return occluded_full


@torch.no_grad()
def occlusion_partial_matrix(
    posV,
    posO,
    sphere_center: Tuple[float, float, float] = (0.0, 200.0, 5.0),
    sphere_radius: float = SPHERE_RADIUS,
    disk_radius: float = DISK_RADIUS,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    部分遮挡(有交叠即 True) 的近似矩阵版本 (Nv,No)，与 rough.py 中思路一致：
      - 对每个 (V_i, O_j):
            t = ( (O - V)·(C - V) ) / ||C - V||^2  (C 为目标球心)
            0 < t < 1 表示 O 在视线段上
            d = 到视线的最小距离
            r_proj = sphere_radius * t  (线性缩放近似)
            若 d <= disk_radius + r_proj 则记为部分遮挡
    """
    posV, posO, device, dtype = _auto_device_dtype(posV, posO, device, dtype)
    sphere_center = torch.as_tensor(sphere_center, device=device, dtype=dtype)

    V = posV[:, None, :]            # (Nv,1,3)
    O = posO[None, :, :]            # (1,No,3)
    C = sphere_center.view(1, 1, 3)

    VC = C - V                      # (Nv,1,3)
    VO = O - V                      # (Nv,No,3)

    vc_norm_sq = (VC * VC).sum(dim=-1)  # (Nv,1)
    near = vc_norm_sq < _EPS

    # 防止除零
    vc_norm_sq_safe = torch.where(near, torch.ones_like(vc_norm_sq), vc_norm_sq)

    t = (VO * VC).sum(dim=-1) / vc_norm_sq_safe  # (Nv,No)
    between = (t > 0.0) & (t < 1.0)

    # 垂距
    v_perp = VO - t.unsqueeze(-1) * VC  # (Nv,No,3)
    d = torch.linalg.norm(v_perp, dim=-1)

    r_proj = sphere_radius * t
    threshold = disk_radius + r_proj
    cover = d <= threshold
    occluded = (~near).expand_as(cover) & between & cover
    return occluded


# ==================== Convenience Wrappers ====================

def occluded_any_partial(posV, posO, **kwargs) -> torch.Tensor:
    """
    返回 (Nv,) bool：每个观察点是否被任一烟雾部分遮挡。
    """
    mat = occlusion_partial_matrix(posV, posO, **kwargs)
    return mat.any(dim=1)


def occluded_any_full(posV, posO, **kwargs) -> torch.Tensor:
    """
    返回 (Nv,) bool：每个观察点是否被任一烟雾完全遮挡。
    """
    mat = occlusion_full_cover_matrix(posV, posO, **kwargs)
    return mat.any(dim=1)


# ==================== __all__ ====================
__all__ = [
    "SPHERE_CENTER",
    "SPHERE_RADIUS",
    "DISK_RADIUS",
    "TorchRoughVectorizedOcclusionJudge",
    "TorchVectorizedOcclusionResult",
    "occlusion_full_cover_matrix",
    "occlusion_partial_matrix",
    "occluded_any_partial",
    "occluded_any_full",
]


# ==================== Self-test ====================

if __name__ == "__main__":
    print("=== rough_torch self test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    missiles = torch.tensor([
        [20000.0,   0.0, 2000.0],
        [18000.0, -600.0, 1900.0],
    ], device=device)
    smokes = torch.tensor([
        [15000.0,   100.0, 1800.0],
        [12000.0,  -200.0, 1500.0],
        [ 5000.0,   300.0,  800.0],
    ], device=device)

    part_mat = occlusion_partial_matrix(missiles, smokes, device=device)
    full_mat = occlusion_full_cover_matrix(missiles, smokes, device=device)
    print("Partial matrix:\n", part_mat)
    print("Full-cover matrix:\n", full_mat)
    print("Any partial:", part_mat.any(dim=1))
    print("Any full   :", full_mat.any(dim=1))

    # Single-batch style
    judge = TorchRoughVectorizedOcclusionJudge(device=device)
    N = smokes.shape[0]
    V = missiles[0:1].repeat(N, 1)          # (N,3)
    C = smokes                              # (N,3)
    r = torch.full((N,), DISK_RADIUS, device=device)
    S = SPHERE_CENTER.to(device).view(1, 3).repeat(N, 1)
    R = torch.full((N,), SPHERE_RADIUS, device=device)
    res = judge.judge_batch(V, C, r, S, R)
    print("Single-batch occluded:", res.occluded)
