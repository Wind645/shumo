"""Torch GPU 加速的 rough (近似) 遮挡判定。

近似思想同 `rough_judge.py`: 使用包围圆的最小球 (圆心 C, 半径 r) 的视角锥
代替原圆，判定该小球是否完全被大球 (S,R) 覆盖。

充分条件 (保守, 无误报): gamma + beta <= alpha
  gamma = angle( u_S , u_C )
  beta  = arcsin( r / ||C-V|| )
  alpha = arcsin( R / ||S-V|| )

近似 f_min ≈ cos(gamma + beta) = cosγ cosβ - sinγ sinβ

提供: `TorchRoughVectorizedOcclusionJudge.judge_batch(...)`，返回
`TorchVectorizedOcclusionResult`，字段与精确版本保持一致语义：
  - occluded: bool tensor (N,)
  - f_min: 近似 cos(gamma+beta)
  - cos_alpha_s: cos(alpha)
  - t_min: 全 NaN (不可用)
  - valid: bool tensor (N,) (全部 True，除观察点在球内情形 f_min=NaN)

说明: 该方法可能产生 False Negative (漏判)；不会产生 False Positive (误判为遮挡)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

try:
    import torch
except Exception as e:  # pragma: no cover - import guard
    raise ImportError("PyTorch 未安装，无法使用 rough_judge_torch") from e


@dataclass
class TorchVectorizedOcclusionResult:
    occluded: torch.Tensor      # (N,) bool
    f_min: torch.Tensor         # (N,) float (approx or NaN)
    cos_alpha_s: torch.Tensor   # (N,) float (or NaN)
    t_min: torch.Tensor         # (N,) float (NaN)
    valid: torch.Tensor         # (N,) bool

    def to_numpy(self):  # 便捷转换
        return dict(
            occluded=self.occluded.detach().cpu().numpy(),
            f_min=self.f_min.detach().cpu().numpy(),
            cos_alpha_s=self.cos_alpha_s.detach().cpu().numpy(),
            t_min=self.t_min.detach().cpu().numpy(),
            valid=self.valid.detach().cpu().numpy(),
        )


class TorchRoughVectorizedOcclusionJudge:
    def __init__(self, device: Optional[str] = None, dtype: torch.dtype = torch.float32):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype

    def _to_tensor(self, arr) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device, self.dtype)
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def judge_batch(self, V, C, r, S, R) -> TorchVectorizedOcclusionResult:
        Vt = self._to_tensor(V)
        Ct = self._to_tensor(C)
        rt = self._to_tensor(r).view(-1)
        St = self._to_tensor(S)
        Rt = self._to_tensor(R).view(-1)

        if Vt.ndim != 2 or Vt.size(-1) != 3:
            raise ValueError("V must be (N,3)")
        N = Vt.shape[0]
        for T in (Ct, St):
            if T.shape != Vt.shape:
                raise ValueError("C,S shape mismatch V")
        if rt.shape != (N,) or Rt.shape != (N,):
            raise ValueError("r,R must be shape (N,)")

        eps = torch.tensor(1e-12, device=self.device, dtype=self.dtype)

        SC = St - Vt  # (N,3)
        CC = Ct - Vt
        dS = torch.linalg.norm(SC, dim=1)  # (N,)
        dC = torch.linalg.norm(CC, dim=1)

        # 单位向量 (避免除零)
        uS = torch.zeros_like(SC)
        mask_dS = dS > eps
        uS[mask_dS] = SC[mask_dS] / dS[mask_dS].unsqueeze(-1)

        uC = torch.zeros_like(CC)
        mask_dC = dC > eps
        uC[mask_dC] = CC[mask_dC] / dC[mask_dC].unsqueeze(-1)
        # dC=0 用 uS 方向
        rep = (~mask_dC) & mask_dS
        uC[rep] = uS[rep]

        # 角度分量
        ratio_s = torch.clamp(Rt / torch.where(dS > 0, dS, torch.ones_like(dS)), 0.0, 1.0)
        ratio_c = torch.clamp(rt / torch.where(dC > 0, dC, torch.ones_like(dC)), 0.0, 1.0)

        alpha = torch.arcsin(ratio_s)
        beta = torch.arcsin(ratio_c)
        cos_alpha = torch.sqrt(torch.clamp(1.0 - ratio_s * ratio_s, min=0.0))

        cos_gamma = torch.sum(uS * uC, dim=1).clamp(-1.0, 1.0)
        gamma = torch.arccos(cos_gamma)
        sin_gamma = torch.sqrt(torch.clamp(1.0 - cos_gamma * cos_gamma, min=0.0))
        sin_beta = ratio_c
        cos_beta = torch.sqrt(torch.clamp(1.0 - sin_beta * sin_beta, min=0.0))

        f_min = cos_gamma * cos_beta - sin_gamma * sin_beta  # cos(gamma+beta)

        observer_in = (dS <= eps) | (Rt >= dS)
        occluded = observer_in | ((gamma + beta) <= (alpha + 1e-12))

        # observer_in 情况下设 NaN
        nan = torch.full((1,), float('nan'), device=self.device, dtype=self.dtype)
        f_min = torch.where(observer_in, nan.expand_as(f_min), f_min)
        cos_alpha = torch.where(observer_in, nan.expand_as(cos_alpha), cos_alpha)

        t_min = torch.full((N,), float('nan'), device=self.device, dtype=self.dtype)
        valid = torch.ones((N,), dtype=torch.bool, device=self.device)

        return TorchVectorizedOcclusionResult(
            occluded=occluded.bool(),
            f_min=f_min,
            cos_alpha_s=cos_alpha,
            t_min=t_min,
            valid=valid,
        )


if __name__ == "__main__":  # 简单自检
    judge = TorchRoughVectorizedOcclusionJudge()
    V = [[0.0, 0.0, 2.0], [0.1, -0.2, 1.5]]
    C = [[0.5, 0.0, 0.0], [0.2, 0.1, 0.0]]
    r = [0.3, 0.6]
    S = [[0.2, 0.1, 1.0], [0.0, 0.0, 0.8]]
    R = [0.5, 0.4]
    res = judge.judge_batch(V, C, r, S, R)
    print("occluded:", res.occluded)
    print("f_min:", res.f_min)
    print("cos_alpha_s:", res.cos_alpha_s)