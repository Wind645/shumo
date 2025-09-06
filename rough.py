import numpy as np

"""
rough.py

功能:
判断目标外接球(中心在 (0,200,5), 半径 sqrt(74)) 是否被位于 posO 的半径为 10 的圆(视为与视线垂直的遮挡圆盘)从观察点 posV 方向遮挡。
支持向量化：posV, posO 可以是任意可广播(broadcast)的大张量，最后返回对应位置的布尔结果。

几何近似说明:
1. 目标真实为一个固定圆柱，这里在 judger 中被近似为一个外接球 (center = (0,200,5), R = sqrt(74)).
2. 遮挡物被近似成位于 posO 的半径为 10 的圆盘(法向量近似取为观察方向的反向, 即仅仅使用“与视线方向正交”的投影测试)。
3. 判定思路（部分遮挡判定）:
   - 取观察点 posV 到目标球心 C 的向量 vC。
   - 取观察点 posV 到遮挡圆心 O 的向量 vO。
   - 计算 t = (vO · vC) / |vC|^2，表示 O 在观察点到球心连线方向上的投影参数。
       若 0 < t < 1，则 O 位于观察点与球心之间（在同一直线方向上）。
   - 计算遮挡圆心到该视线直线的最小距离 d。
   - 目标球在该位置的“等效投影半径”近似为 r_proj = sphere_radius * t
     （因为在分数 t 处，圆锥放缩近似，球的角半径基本按线性缩放）。
   - 若 d <= disk_radius + r_proj，则认为存在遮挡（即圆与球的投影重叠，允许部分遮挡）。
   - 可选全遮挡(full=True)时: 使用 d <= disk_radius - r_proj （且阈值非负）来要求遮挡圆完全盖住球的投影。

你可以根据需要选择 partial / full 判定，本文件默认提供 partial 判定接口 `occluded_partial` 和全遮挡判定接口 `occluded_full`。
在 simulator 中通常需要“是否被遮蔽”用于统计不可见时间，是否使用 partial 或 full 取决于评测定义。
默认导出的 `occluded` 使用 partial 判定。

向量化:
- posV, posO 都可以是 (..., 3) 形状，可广播。
- 返回值形状为广播后的前置维度（去掉最后一个 size=3 维）。

数值稳定:
- 若观察点恰好与球心重合(|vC| ~ 0)，直接返回未被遮挡 False。

"""

SPHERE_CENTER = np.array([0.0, 200.0, 5.0], dtype=float)
SPHERE_RADIUS = np.sqrt(74.0)
DISK_RADIUS = 10.0
_EPS = 1e-12


def _prepare(posV, posO):
    posV = np.asarray(posV, dtype=float)
    posO = np.asarray(posO, dtype=float)
    if posV.shape[-1] != 3 or posO.shape[-1] != 3:
        raise ValueError("posV 与 posO 的最后一维必须为 3 (x,y,z).")
    return posV, posO


def _compute_core(posV, posO, sphere_center, sphere_radius, disk_radius, full: bool):
    """
    内部核心计算，返回布尔数组（遮挡与否）。
    full=False: 允许部分遮挡 (d <= R_disk + r_proj)
    full=True : 要求完全遮挡 (d <= R_disk - r_proj, 且阈值>=0)
    """
    posV, posO = _prepare(posV, posO)

    # 向量
    vC = sphere_center - posV          # 观察点到球心
    vO = posO - posV                   # 观察点到遮挡圆心

    norm_vC_sq = np.sum(vC * vC, axis=-1, keepdims=True)
    # 处理观察点与球心一致情况
    near_mask = norm_vC_sq < _EPS
    # 避免除零
    norm_vC_sq_safe = np.where(near_mask, 1.0, norm_vC_sq)

    # 投影参数 t
    t = np.sum(vO * vC, axis=-1, keepdims=True) / norm_vC_sq_safe

    # 遮挡要在 0 < t < 1 之间
    between_mask = (t > 0.0) & (t < 1.0)

    # 到视线的垂直距离
    v_perp = vO - t * vC
    d = np.linalg.norm(v_perp, axis=-1)

    # 球在该 t 处的放缩投影半径（线性近似）
    r_proj = sphere_radius * t[..., 0]

    if full:
        # 完全遮挡，需要圆盘半径足够覆盖球投影
        thr = disk_radius - r_proj
        # 仅 thr >= 0 有意义
        cover_mask = (thr >= 0.0) & (d <= thr)
    else:
        # 部分遮挡（只要有交叠）
        thr = disk_radius + r_proj
        cover_mask = d <= thr

    # 合并全部条件
    occluded_mask = (~near_mask[..., 0]) & between_mask[..., 0] & cover_mask
    return occluded_mask


def occluded_partial(posV, posO,
                     sphere_center=SPHERE_CENTER,
                     sphere_radius=SPHERE_RADIUS,
                     disk_radius=DISK_RADIUS):
    """
    部分遮挡判定（常用）:
    当遮挡圆与球投影有重叠即可认为“被遮挡”。

    参数:
        posV: (...,3) 观察点
        posO: (...,3) 遮挡圆心
    返回:
        Bool 数组 (广播后的形状，不含最后一维)
    """
    return _compute_core(posV, posO, sphere_center, sphere_radius, disk_radius, full=False)


def occluded_full(posV, posO,
                  sphere_center=SPHERE_CENTER,
                  sphere_radius=SPHERE_RADIUS,
                  disk_radius=DISK_RADIUS):
    """
    完全遮挡判定:
    只有当遮挡圆完全覆盖球的投影时返回 True。
    """
    return _compute_core(posV, posO, sphere_center, sphere_radius, disk_radius, full=True)


# 默认导出函数: 使用部分遮挡逻辑
def occluded(posV, posO):
    """
    默认接口：部分遮挡（圆盘投影近似）。
    """
    return occluded_partial(posV, posO)


def occluded_line_sphere(posV,
                         smoke_center,
                         target_center=SPHERE_CENTER,
                         target_radius=SPHERE_RADIUS,
                         smoke_radius=DISK_RADIUS):
    """
    线段-球体遮挡快速判定（球形烟雾 + 目标外接球 叠加成膨胀球模型）
    场景对应物理含义：
      - 目标真实为半径 7 m、高 10 m 的圆柱；此处仍用其外接球近似 (半径 sqrt(7^2 + 5^2))
      - 烟雾在起爆后 20 s 内为一个有效半径 ~10 m 的球形云团（下沉不影响其半径）
      - 判定“是否遮挡”采用：导弹观察点 V 到目标外接球心 C 的线段，与
        (膨胀后球体) 是否相交。膨胀后球体半径 = smoke_radius + target_radius。
        这等价于检测烟雾球心 S 到线段 VC 的最小距离是否 <= smoke_radius + target_radius，
        且最近点参数 t 位于 [0,1] 之间。

    数学步骤（广播向量化）：
        v = C - V
        w = S - V
        t = (w·v)/|v|^2
        若 t∉[0,1] 则不在段内（烟雾不在 V 与 C 之间）
        最近点 P = V + t v
        d = |P - S|
        遮挡条件: d <= smoke_radius + target_radius

    参数:
        posV: (...,3) 观察点（导弹位置，可广播）
        smoke_center: (...,3) 烟雾球心（可广播）
        target_center: (3,) 目标外接球心
        target_radius: float 目标外接球半径
        smoke_radius: float 烟雾球半径（有效遮蔽半径）

    返回:
        bool 数组，广播后形状 = broadcast(posV[...,-1 removed], smoke_center[...,-1 removed])

    说明:
        与原 disk 投影视锥近似相比，此函数更贴近“烟雾为体积球体、遮挡是线段穿过球体并允许
        目标外接球膨胀”的直接物理表述；若竞赛参考答案 ~1.39 s 与此模型匹配，可用它替换。
    """
    posV = np.asarray(posV, dtype=float)
    smoke_center = np.asarray(smoke_center, dtype=float)
    if posV.shape[-1] != 3 or smoke_center.shape[-1] != 3:
        raise ValueError("posV 与 smoke_center 的最后一维必须为 3")
    v = target_center - posV              # (...,3)
    w = smoke_center - posV               # (...,3)
    vv = np.sum(v * v, axis=-1, keepdims=True)
    near = vv < _EPS
    vv_safe = np.where(near, 1.0, vv)
    t = np.sum(w * v, axis=-1, keepdims=True) / vv_safe   # (...,1)
    seg_mask = (t >= 0.0) & (t <= 1.0)
    p = posV + t * v                      # 最近点
    d = np.linalg.norm(p - smoke_center, axis=-1)  # (...,)
    thresh = smoke_radius + target_radius
    mask = (~near[..., 0]) & seg_mask[..., 0] & (d <= thresh)
    return mask


__all__ = [
    "SPHERE_CENTER",
    "SPHERE_RADIUS",
    "DISK_RADIUS",
    "occluded",
    "occluded_partial",
    "occluded_full",
    "occluded_line_sphere",
]


if __name__ == "__main__":
    # 简单自测示例
    viewer = np.array([20000.0, 0.0, 2000.0])
    # 构造一些遮挡点：一个在直线上，一个偏移，一个在目标后面
    posO_list = np.array([
        [10000.0, 100.0, 1000.0],   # 大致朝向目标
        [10000.0, 500.0, 1000.0],   # 侧向偏移较大
        [0.0, 200.0, 5.0],          # 与球心重合（t=1 不算在 between 范围）
        [5000.0, 210.0, 600.0],     # 近距离侧偏
    ])
    res_partial = occluded_partial(viewer, posO_list)
    res_full = occluded_full(viewer, posO_list)

    print("Partial:", res_partial)
    print("Full   :", res_full)
