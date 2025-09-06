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


_SAMPLE_CACHE = {}

def _sample_cylinder_circles(K: int, radius: float, base_center: torch.Tensor, height: float, *, dtype=torch.float32) -> torch.Tensor:
    """Return cached (2K,3) sample points on cylinder circle edges (bottom & top)."""
    key = (K, radius, height, base_center.device, dtype)
    if key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[key]
    device = base_center.device
    angles = torch.linspace(0, 2 * torch.pi, K, device=device, dtype=dtype)
    c, s = torch.cos(angles), torch.sin(angles)
    circle_edge = torch.stack([c, s, torch.zeros_like(c)], 1) * radius  # (K,3)
    base_center = base_center.to(device=device, dtype=dtype)
    bottom = base_center.unsqueeze(0) + circle_edge
    top = bottom.clone(); top[:, 2] += height
    samples = torch.cat([bottom, top], 0).contiguous()
    _SAMPLE_CACHE[key] = samples
    return samples


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
    params: torch.Tensor,
    *, dt: float, device: torch.device,
    K: int = 24,
    chunk_pairs: int = 8192,
    use_mask_expand: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute approximate total occluded time for each parameter row.

    params: (N,4) -> [speed, azimuth, t_rel, delay]
    返回: (N,) 每个参数组合的累计遮挡时间 (秒)
    约束: speed∈[70,140], t_rel>=0, delay>0 之外直接视为 0。
    """
    if params.numel() == 0:
        return torch.zeros(0, device=device, dtype=torch.float64)

    p = params.to(device=device, dtype=dtype)
    speed, azimuth, t_rel, delay = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    valid = (speed >= 70) & (speed <= 140) & (t_rel >= 0) & (delay > 0)
    out = torch.zeros(p.size(0), device=device, dtype=dtype)
    if not valid.any():
        return out

    # --- Missile straight-line flight (保持与原逻辑一致) ---
    MISSILE_POS0 = torch.tensor([20000.0, 0.0, 2000.0], device=device, dtype=dtype)
    MISSILE_TARGET = torch.zeros(3, device=device, dtype=dtype)
    MISSILE_SPEED = 300.0
    to_target = MISSILE_TARGET - MISSILE_POS0
    dist = torch.linalg.norm(to_target)
    direction = to_target / dist
    T_final = dist / MISSILE_SPEED
    n_steps = int(T_final / dt) + 1
    t_grid = torch.linspace(0.0, T_final, n_steps, device=device, dtype=dtype)
    missile_pos = MISSILE_POS0 + MISSILE_SPEED * t_grid.unsqueeze(1) * direction  # (n_steps,3)

    # --- Drone / smoke initial state ---
    drone_start = torch.tensor([17800.0, 0.0, 1800.0], device=device, dtype=dtype).expand_as(p[:, :3])
    dir_drone = torch.stack([torch.cos(azimuth), torch.sin(azimuth), torch.zeros_like(azimuth)], 1)
    gravity = torch.tensor([0.0, 0.0, -9.8], device=device, dtype=dtype)
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
    sample_points = _sample_cylinder_circles(K, CYL_RADIUS, CYL_BASE.to(device=device, dtype=dtype), CYL_HEIGHT, dtype=dtype)  # (2K,3)

    # 活动时间区间索引 (离散步)
    start_idx = torch.clamp((explode_t / dt).ceil().long(), 0, n_steps - 1)
    end_idx = torch.clamp(((explode_t + SMOKE_LIFE) / dt).floor().long(), 0, n_steps - 1)
    # ---- 向量化时间维: 展开所有 (candidate, time_step) 活动对 ----
    if use_mask_expand:
        # 掩码方式生成 (candidate, time) 对，避免 Python 循环
        t_idx = torch.arange(n_steps, device=device)
        # (N,T) mask
        mask = (t_idx.unsqueeze(0) >= start_idx.unsqueeze(1)) & (t_idx.unsqueeze(0) <= end_idx.unsqueeze(1)) & valid.unsqueeze(1)
        cand_ids, time_ids = mask.nonzero(as_tuple=True)
    else:
        valid_idx = valid.nonzero().squeeze(1)
        if valid_idx.numel()==0:
            return out
        lengths = (end_idx - start_idx + 1)
        lengths_valid = lengths[valid_idx]
        pos_mask = lengths_valid > 0
        if not pos_mask.all():
            valid_idx = valid_idx[pos_mask]
            lengths_valid = lengths_valid[pos_mask]
        cand_ids = torch.repeat_interleave(valid_idx, lengths_valid)
        time_index_list = [
            torch.arange(start_idx[i], end_idx[i] + 1, device=device)
            for i in valid_idx.tolist()
        ]
        time_ids = torch.cat(time_index_list)
    # 计算每个 pair 的观察点 / 烟雾中心 等
    V_all = missile_pos[time_ids]
    t_now_all = t_grid[time_ids]
    tau_all = torch.clamp(t_now_all - explode_t[cand_ids], min=0.0)
    smoke_center_all = c0[cand_ids].clone()
    smoke_center_all[:, 2] -= SMOKE_DESCENT * tau_all
    B = V_all.size(0)
    R_smoke = SMOKE_RADIUS
    # 分块以控制显存 (chunk_pairs)
    for s in range(0, B, chunk_pairs):
        e = min(s + chunk_pairs, B)
        V_chunk = V_all[s:e]
        S_chunk = smoke_center_all[s:e]
        covered = _segment_sphere_cover(V_chunk, sample_points, S_chunk, R_smoke)
        if covered.any():
            out.index_add_(0, cand_ids[s:e][covered], torch.full((int(covered.sum()),), dt, device=device, dtype=out.dtype))

    return out.to(dtype=torch.float32)


__all__ = ["batch_occluded_time_caps_torch_sampled"]
