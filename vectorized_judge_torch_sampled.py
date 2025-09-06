import torch

"""Approximate occlusion time using simple cylinder-cap sampling.

思路 / Idea:
1. 圆柱上下两个圆面各均匀采样 K 个点 (总 2K)。
2. 在时间离散网格上遍历，计算导弹观察点 V_t。
3. 对每个活动烟雾球: 取球心 S_t (含竖直下落), 半径 Rsm。
4. 判断从观察点 V_t 到所有采样点 P_i 的线段是否与球体相交。
   线段最近点到球心距离 d <= R 即判定该采样点被遮挡。
5. 若 2K 采样点全部被遮挡 => 该时间步记入该参数组合的遮挡时间 Δt。

向量化要点:
 - 所有采样点恒定 (固定圆柱)，预先 (2K,3) 复用。
 - 线段-球最近点利用投影参数 t=dot(SV,L)/||L||^2，裁剪到 [0,1]。
 - 批内对所有采样点同时判断，然后 all(dim=1)。
"""


def _sample_cylinder_circles(K: int, radius: float, base_center: torch.Tensor, height: float) -> torch.Tensor:
    """Return (2K,3) sample points on bottom & top circle edges of a vertical cylinder.
    Cylinder axis assumed aligned with +Z. Bottom center = base_center.
    """
    device = base_center.device
    angles = torch.linspace(0, 2 * torch.pi, K + 1, device=device)[:-1]
    c, s = torch.cos(angles), torch.sin(angles)
    circle_edge = torch.stack([c, s, torch.zeros_like(c)], 1) * radius  # (K,3)
    bottom = base_center.unsqueeze(0) + circle_edge
    top = bottom + torch.tensor([0.0, 0.0, height], device=device)
    return torch.cat([bottom, top], 0)  # (2K,3)


def _segment_sphere_cover(view_pts: torch.Tensor, sample_pts: torch.Tensor, sphere_centers: torch.Tensor, sphere_radius: float) -> torch.Tensor:
    """Return boolean (B,) whether each sphere center occludes ALL sample segments.

    view_pts: (B,3)   观察点 V
    sample_pts: (M,3) 采样点 P_i（与批无关）
    sphere_centers: (B,3) 球心 S
    sphere_radius: 标量 R
    算法: 最近点距离法判定线段是否与球相交；对同一 B 的 M 条线段取 all。
    """
    # Broadcast to (B,M,3)
    seg_vec = sample_pts.unsqueeze(0) - view_pts.unsqueeze(1)          # L = P - V
    sv_vec = sphere_centers.unsqueeze(1) - view_pts.unsqueeze(1)       # S - V
    seg_len2 = (seg_vec * seg_vec).sum(-1)                              # (B,M)
    # 投影参数 t (未裁剪)
    t = (sv_vec * seg_vec).sum(-1) / seg_len2
    t = torch.clamp(t, 0.0, 1.0)
    closest = view_pts.unsqueeze(1) + t.unsqueeze(-1) * seg_vec        # 最近点
    dist2 = (closest - sphere_centers.unsqueeze(1)).pow(2).sum(-1)     # (B,M)
    covered_each = dist2 <= (sphere_radius * sphere_radius)
    return covered_each.all(dim=1)  # (B,)


def batch_occluded_time_caps_torch_sampled(
    params: torch.Tensor, *, dt: float, device: torch.device, K: int = 24
) -> torch.Tensor:
    """Compute approximate total occluded time for each parameter row.

    params: (N,4) -> [speed, azimuth, t_rel, delay]
    返回: (N,) 每个参数组合的累计遮挡时间 (秒)
    约束: speed∈[70,140], t_rel>=0, delay>0 之外直接视为 0。
    """
    if params.numel() == 0:
        return torch.zeros(0, device=device, dtype=torch.float64)

    p = params.to(device=device, dtype=torch.float64)
    speed, azimuth, t_rel, delay = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    valid = (speed >= 70) & (speed <= 140) & (t_rel >= 0) & (delay > 0)
    out = torch.zeros(p.size(0), device=device, dtype=torch.float64)
    if not valid.any():
        return out

    # --- Missile straight-line flight (保持与原逻辑一致) ---
    MISSILE_POS0 = torch.tensor([20000.0, 0.0, 2000.0], device=device)
    MISSILE_TARGET = torch.zeros(3, device=device)
    MISSILE_SPEED = 300.0
    to_target = MISSILE_TARGET - MISSILE_POS0
    dist = torch.linalg.norm(to_target)
    direction = to_target / dist
    T_final = dist / MISSILE_SPEED
    n_steps = int(T_final / dt) + 1
    t_grid = torch.linspace(0.0, T_final, n_steps, device=device)
    missile_pos = MISSILE_POS0 + MISSILE_SPEED * t_grid.unsqueeze(1) * direction  # (n_steps,3)

    # --- Drone / smoke initial state ---
    drone_start = torch.tensor([17800.0, 0.0, 1800.0], device=device).expand_as(p[:, :3])
    dir_drone = torch.stack([torch.cos(azimuth), torch.sin(azimuth), torch.zeros_like(azimuth)], 1)
    gravity = torch.tensor([0.0, 0.0, -9.8], device=device)
    rel_origin = drone_start + dir_drone * speed.unsqueeze(1) * t_rel.unsqueeze(1)
    vel = dir_drone * speed.unsqueeze(1)
    c0 = rel_origin + vel * delay.unsqueeze(1) + 0.5 * gravity * (delay.unsqueeze(1) ** 2)
    explode_t = t_rel + delay

    # --- Smoke & cylinder scene constants ---
    SMOKE_LIFE = 20.0
    SMOKE_DESCENT = 3.0
    SMOKE_RADIUS = 10.0
    CYL_RADIUS = 7.0
    CYL_HEIGHT = 10.0
    CYL_BASE = torch.tensor([0.0, 200.0, 0.0], device=device)
    # Precompute 2K sampling points on cylinder top & bottom circles
    sample_points = _sample_cylinder_circles(K, CYL_RADIUS, CYL_BASE, CYL_HEIGHT)  # (2K,3)

    # 活动时间区间索引 (离散步)
    start_idx = torch.clamp((explode_t / dt).ceil().long(), 0, n_steps - 1)
    end_idx = torch.clamp(((explode_t + SMOKE_LIFE) / dt).floor().long(), 0, n_steps - 1)
    global_start = start_idx[valid].min().item()
    global_end = end_idx[valid].max().item()

    for ti in range(global_start, global_end + 1):
        active = valid & (ti >= start_idx) & (ti <= end_idx)
        if not active.any():
            continue
        ids = active.nonzero().squeeze(1)
        V_t = missile_pos[ti].expand(ids.numel(), 3)
        t_now = t_grid[ti]
        tau = torch.clamp(t_now - explode_t[ids], min=0.0)
        smoke_center = c0[ids].clone()
        smoke_center[:, 2] -= SMOKE_DESCENT * tau  # 下降
        covered = _segment_sphere_cover(V_t, sample_points, smoke_center, SMOKE_RADIUS)
        out[ids[covered]] += dt

    return out


__all__ = ["batch_occluded_time_caps_torch_sampled"]
