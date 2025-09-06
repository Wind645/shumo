import numpy as np

def is_sphere_blocked_vectorized(data):
    """
    严格“完全覆盖”版遮挡判定（已移除旧的“部分相交即可”逻辑）：
    判定烟幕球在视角上是否完全覆盖目标。

    条件（充分必要在球面近似下）:
        θ_s >= θ_t + φ
      其中:
        θ_s = arcsin(R_smoke / d_s)
        θ_t = arcsin(R_target / d_t)
        φ   = arccos( (v_t · v_s) / (|v_t||v_s|) )

    目标采用真实圆柱的水平半径 7m（不再使用外接球 sqrt(74)），
    目标中心依旧取 (0, 200, 5) 作为近似几何中心。

    参数:
        data: ndarray shape (N,7):
              [missile_x, missile_y, missile_z,
               smoke_x, smoke_y, smoke_z,
               R_smoke]

    返回:
        bool ndarray shape (N,) ：True 表示在该几何配置下烟幕球完全遮挡目标。
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.zeros(0, dtype=bool)

    target_center = np.array([0.0, 200.0, 5.0])
    target_radius = 7.0  # 严格使用圆柱水平半径

    pos   = data[:, :3]   # 导弹位置
    smoke = data[:, 3:6]  # 烟幕中心
    R_smoke = data[:, 6]

    v_t = target_center - pos
    v_s = smoke - pos
    d_t = np.linalg.norm(v_t, axis=1)
    d_s = np.linalg.norm(v_s, axis=1)

    eps = 1e-9
    valid = (d_t > eps) & (d_s > eps)
    out = np.zeros(len(data), dtype=bool)
    if not np.any(valid):
        return out

    vt_u = v_t[valid] / d_t[valid, None]
    vs_u = v_s[valid] / d_s[valid, None]

    θ_t = np.arcsin(np.clip(target_radius / d_t[valid], -1.0, 1.0))
    θ_s = np.arcsin(np.clip(R_smoke[valid] / d_s[valid], -1.0, 1.0))

    cosφ = np.sum(vt_u * vs_u, axis=1)
    cosφ = np.clip(cosφ, -1.0, 1.0)
    φ = np.arccos(cosφ)

    blocked = θ_s >= (θ_t + φ)
    out[np.where(valid)[0]] = blocked
    return out
