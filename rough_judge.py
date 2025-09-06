"""Rough (approximate) occlusion judges.

思路: 用包围圆 (位于 z=0 平面, 圆心 C, 半径 r) 的球体 B=(C,r) 代替真实圆的角域。
再判定此小球 B 在观察点 V 下是否被大球 A=(S,R) 的视锥完全覆盖。

严格条件 (原算法): max_t angle(u_S, X(t)-V) \le alpha_S.
近似替换: 令 u_S = (S-V)/||S-V||, u_C = (C-V)/||C-V||.
小球 B 的角半径 beta = arcsin(r/||C-V||); 大球 A 的角半径 alpha = arcsin(R/||S-V||).
则充分条件 (也常较紧) : gamma + beta \le alpha, 其中 gamma = angle(u_S, u_C).

若成立, 原圆一定被完全遮挡; 若不成立, 原圆可能仍被遮挡 (因此该判定可能产生 False Negative, 不会产生 False Positive)。
为保持与精确 Judge API 一致, 返回字段 f_min, cos_alpha_s 等, 但 f_min 为近似: f_min_approx = cos(gamma + beta)。

提供:
  - RoughOcclusionJudge: 单例 API, 与 judge.OcclusionJudge 接口兼容 (is_fully_occluded())
  - RoughVectorizedOcclusionJudge: 批量 API, 与 vectorized_judge.VectorizedOcclusionJudge.judge_batch 兼容
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np

from judge import OcclusionResult  # 复用结果数据结构
from vectorized_judge import VectorizedOcclusionResult  # 复用批量结果结构

_TOL = 1e-12


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _clamp(x: np.ndarray, lo: float, hi: float):
    return np.clip(x, lo, hi)


class RoughOcclusionJudge:
    """近似遮挡判定 (单例)。

    仅使用简单三角函数, O(1) 运算, 不求解多项式。
    f_min 返回的是近似 cos(gamma+beta), 其中 beta 为包围球角半径。
    """

    def __init__(self, V: np.ndarray, C: np.ndarray, r: float, S: np.ndarray, R: float):
        self.V = np.asarray(V, dtype=float).reshape(3)
        self.C = np.asarray(C, dtype=float).reshape(3)
        self.r = float(r)
        self.S = np.asarray(S, dtype=float).reshape(3)
        self.R = float(R)

    def _compute(self) -> Tuple[bool, Dict]:
        V, C, S = self.V, self.C, self.S
        r, R = self.r, self.R

        SC = S - V
        CC = C - V
        dS = np.linalg.norm(SC)
        dC = np.linalg.norm(CC)

        # 球内/观测点在球心: 直接视为遮挡全部
        if dS <= _TOL or R >= dS:
            return True, dict(reason="observer_in_or_at_sphere", f_min=None, cos_alpha_s=None, t_min=None, approx=True)

        # 圆退化为点 => beta=0
        if dC <= _TOL:
            # 若圆心与观察点重合, 只能依据方向对齐 (gamma=0, beta ~ 0)
            beta = 0.0
            uS = SC / dS
            uC = uS  # 视为对齐
        else:
            uS = SC / dS
            uC = CC / dC
            # beta = arcsin( r / dC ) (截断在 [0,1])
            ratio_c = min(1.0, max(0.0, r / dC))
            beta = float(np.arcsin(ratio_c))

        # alpha = arcsin(R / dS)
        ratio_s = min(1.0, max(0.0, R / dS))
        alpha = float(np.arcsin(ratio_s))

        # gamma = angle between uS, uC
        cos_gamma = float(np.dot(uS, uC))
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        gamma = float(np.arccos(cos_gamma))

        # 近似 f_min = cos(gamma + beta)
        # 使用三角展开避免显式加角后再 cos
        # cos(gamma+beta) = cosγ cosβ - sinγ sinβ
        sin_gamma = np.sqrt(max(0.0, 1.0 - cos_gamma * cos_gamma))
        # cosβ, sinβ
        if dC <= _TOL:
            cos_beta = 1.0
            sin_beta = 0.0
        else:
            sin_beta = min(1.0, max(0.0, r / dC))
            cos_beta = np.sqrt(max(0.0, 1.0 - sin_beta * sin_beta))
        f_min_approx = cos_gamma * cos_beta - sin_gamma * sin_beta

        cos_alpha = np.sqrt(max(0.0, 1.0 - ratio_s * ratio_s)) if ratio_s < 1.0 else 0.0

        occluded = (gamma + beta) <= (alpha + 1e-12)

        dbg = dict(
            f_min=float(f_min_approx),
            cos_alpha_s=float(cos_alpha),
            t_min=None,  # 不可用
            gamma=gamma,
            beta=beta,
            alpha=alpha,
            approx=True,
        )
        return occluded, dbg

    def is_fully_occluded(self) -> OcclusionResult:
        occluded, dbg = self._compute()
        return OcclusionResult(
            occluded=bool(occluded),
            f_min=dbg.get("f_min"),
            cos_alpha_s=dbg.get("cos_alpha_s"),
            t_min=None,
            extra=dbg,
        )


class RoughVectorizedOcclusionJudge:
    """批量近似判定，与 VectorizedOcclusionJudge API 对齐: judge_batch(...)->VectorizedOcclusionResult.

    返回 VectorizedOcclusionResult, 其中:
      - f_min: 近似 cos(gamma+beta)
      - cos_alpha_s: 精确 cos(alpha)
      - t_min: 全 NaN (无参数化最优点)
      - valid: 皆为 True (除非 dS=0 情况下 f_min NaN)
    """

    def judge_batch(self, V: np.ndarray, C: np.ndarray, r: np.ndarray, S: np.ndarray, R: np.ndarray) -> VectorizedOcclusionResult:
        V = np.asarray(V, dtype=float)
        C = np.asarray(C, dtype=float)
        r = np.asarray(r, dtype=float)
        S = np.asarray(S, dtype=float)
        R = np.asarray(R, dtype=float)

        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("V must be (N,3)")
        N = V.shape[0]
        for arr in (C, S):
            if arr.shape != V.shape:
                raise ValueError("C,S shape mismatch")
        if r.shape != (N,) or R.shape != (N,):
            raise ValueError("r,R must be (N,)")

        SC = S - V
        CC = C - V
        dS = np.linalg.norm(SC, axis=1)
        dC = np.linalg.norm(CC, axis=1)

        # 单位向量 (避免除 0)
        uS = np.zeros_like(SC)
        mask_dS = dS > _TOL
        uS[mask_dS] = SC[mask_dS] / dS[mask_dS, None]

        uC = np.zeros_like(CC)
        mask_dC = dC > _TOL
        uC[mask_dC] = CC[mask_dC] / dC[mask_dC, None]
        # 若 dC=0, 直接用 uS 方向
        rep = (~mask_dC) & mask_dS
        uC[rep] = uS[rep]

        # 角度分量
        ratio_s = np.clip(R / np.where(dS > 0, dS, 1.0), 0.0, 1.0)
        ratio_c = np.clip(r / np.where(dC > 0, dC, 1.0), 0.0, 1.0)

        alpha = np.arcsin(ratio_s)
        beta = np.arcsin(ratio_c)
        cos_alpha = np.sqrt(np.maximum(0.0, 1.0 - ratio_s * ratio_s))

        cos_gamma = np.sum(uS * uC, axis=1)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        gamma = np.arccos(cos_gamma)
        sin_gamma = np.sqrt(np.maximum(0.0, 1.0 - cos_gamma * cos_gamma))
        sin_beta = ratio_c
        cos_beta = np.sqrt(np.maximum(0.0, 1.0 - sin_beta * sin_beta))

        f_min = cos_gamma * cos_beta - sin_gamma * sin_beta  # cos(gamma+beta)

        # 观测点落在球内/球面 => 直接遮挡
        observer_in = (dS <= _TOL) | (R >= dS)
        occluded = observer_in | ((gamma + beta) <= (alpha + 1e-12))

        # 修正 observer_in 情况: f_min, cos_alpha_s 可设为 NaN/None
        f_min[observer_in] = np.nan
        cos_alpha[observer_in] = np.nan

        t_min = np.full(N, np.nan)
        valid = np.ones(N, dtype=bool)

        return VectorizedOcclusionResult(
            occluded=occluded.astype(bool),
            f_min=f_min,
            cos_alpha_s=cos_alpha,
            t_min=t_min,
            valid=valid,
        )


if __name__ == "__main__":
    # 简单自检
    V = np.array([[0.0, 0.0, 2.0]])
    C = np.array([[0.5, 0.0, 0.0]])
    r = np.array([0.3])
    S = np.array([[0.2, 0.1, 1.0]])
    R = np.array([0.5])
    judge = RoughVectorizedOcclusionJudge()
    res = judge.judge_batch(V, C, r, S, R)
    print("rough occluded:", res.occluded, "f_min:", res.f_min, "cos_alpha_s:", res.cos_alpha_s)