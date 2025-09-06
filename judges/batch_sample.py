from __future__ import annotations
import numpy as np

"""
批量采样判定：判断给定几何配置下烟幕球是否在视线方向上完全遮蔽圆柱目标。

圆柱参数:
  半径 r = 7 m
  高度 h = 10 m
  下底面圆心 B = (0, 200, 0)
  上底面圆心 T = (0, 200, 10)

输入 data: shape (N,7)
  [missile_x, missile_y, missile_z,
   smoke_x,   smoke_y,   smoke_z,
   R_smoke]

判定逻辑 (采样保守判定):
  - 在上下底面圆周各均匀采样 K 个点 (共 2K).
  - 对每个采样点构造视线段 (M->Q), 若线段与球 (C,R) 相交(含端点在内) 则认为该点被遮挡。
  - 2K 个点全部遮挡 => 认为圆柱在该视角下被完全遮蔽 (返回 True)。
  - 否则返回 False。False 可能表示确实未完全遮蔽，或 K 不足导致漏检，可增大 K 提高严格性。

复杂度: O(N*K). 采用向量化以减少 Python 循环开销。

函数:
  is_cylinder_blocked_vectorized(data, K=32) -> ndarray(bool shape (N,))

使用:
  from judges.batch_sample import is_cylinder_blocked_vectorized
  mask = is_cylinder_blocked_vectorized(data, K=48)

该实现仅保留批量接口，内部仍含必要的私有辅助函数。
"""

__all__ = ["is_cylinder_blocked_vectorized"]

# 圆柱几何常量
_CYL_R = 7.0
_CYL_H = 10.0
_BOTTOM = np.array([0.0, 200.0, 0.0])
_TOP = np.array([0.0, 200.0, 10.0])

def _circle_samples(K: int) -> np.ndarray:
    """
    生成上下两个底面圆周采样点: shape (2K,3)
    角度使用 (i+0.5)*2π/K 避免落在主轴方向上 (轻微数值稳定性优化)。
    """
    if K <= 0:
        raise ValueError("K must be positive")
    angles = (np.arange(K) + 0.5) * (2.0 * np.pi / K)
    ca = np.cos(angles)
    sa = np.sin(angles)
    bottom = np.empty((K,3), dtype=float)
    bottom[:,0] = _CYL_R * ca + _BOTTOM[0]
    bottom[:,1] = _CYL_R * sa + _BOTTOM[1]
    bottom[:,2] = _BOTTOM[2]
    top = np.empty((K,3), dtype=float)
    top[:,0] = _CYL_R * ca + _TOP[0]
    top[:,1] = _CYL_R * sa + _TOP[1]
    top[:,2] = _TOP[2]
    return np.vstack([bottom, top])

def _segments_intersect_sphere(missile: np.ndarray,
                               points: np.ndarray,
                               center: np.ndarray,
                               radius: np.ndarray) -> np.ndarray:
    """
    计算所有线段 (M->Q) 是否与球相交。

    missile: (N,1,3)
    points:  (P,3)
    center:  (N,1,3)
    radius:  (N,1,1)
    返回: (N,P) bool

    判定: 线段到球心最近点投影参数 t in [0,1] 且最近距离 <= R
    """
    d = points[None,:,:] - missile          # (N,P,3)
    d_norm2 = np.sum(d*d, axis=2)           # (N,P)
    f = missile - center                    # (N,1,3)
    df = np.sum(d * f, axis=2)              # (N,P)
    eps = 1e-12
    t = - df / (d_norm2 + eps)              # (N,P)
    on_seg = (t >= 0.0) & (t <= 1.0)
    closest = f + t[...,None]*d             # (N,P,3)
    dist2 = np.sum(closest*closest, axis=2) # (N,P)
    intersects = on_seg & (dist2 <= (radius*radius)[:,:,0])
    # 若采样点与导弹重合 -> 视为被遮挡 (不影响严格性)
    intersects |= d_norm2 < eps
    return intersects

def is_cylinder_blocked_vectorized(data, K: int = 32) -> np.ndarray:
    """
    批量判定：上下底面各 K 采样点的视线均与球相交 -> 圆柱完全遮蔽。

    参数:
      data: ndarray shape (N,7)
            [missile_x, missile_y, missile_z,
             smoke_x, smoke_y, smoke_z,
             R_smoke]
      K:    采样点数 (默认 32)，越大越严格但计算更慢。

    返回:
      ndarray(bool) shape (N,)
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError("data must have shape (N,7)")
    if K <= 0:
        raise ValueError("K must be positive")

    samples = _circle_samples(K)          # (2K,3)
    M = arr[:, :3][:,None,:]              # (N,1,3)
    C = arr[:, 3:6][:,None,:]             # (N,1,3)
    R = arr[:, 6][:,None,None]            # (N,1,1)

    mask = _segments_intersect_sphere(M, samples, C, R)  # (N,2K)
    return np.all(mask, axis=1)

if __name__ == "__main__":
    # 自检: 大球居中 -> True
    missile = np.array([[20000.0, 0.0, 2000.0]])
    smoke_center = np.array([[0.0, 200.0, 5.0]])
    big_R = np.array([[50.0]])
    data_ok = np.hstack([missile, smoke_center, big_R])
    print("Expect True :", is_cylinder_blocked_vectorized(data_ok, K=16))

    # 自检: 小球偏移 -> False
    smoke_far = np.array([[500.0, 200.0, 5.0]])
    small_R = np.array([[5.0]])
    data_no = np.hstack([missile, smoke_far, small_R])
    print("Expect False:", is_cylinder_blocked_vectorized(data_no, K=16))
