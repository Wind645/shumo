import numpy as np

def is_sphere_blocked_vectorized(data):
    """
    向量化版本：判断从给定点pos看向球(0,200,5) r=sqrt(74) 是否被给定球 poss R 遮蔽

    Args:
        data: numpy array of shape (N, 7), where each row is [pos_x, pos_y, pos_z, poss_x, poss_y, poss_z, R]

    Returns:
        numpy array of shape (N,): bool array where True表示被遮蔽，False表示未被遮蔽
    """
    data = np.array(data)
    N = data.shape[0]

    # 目标球参数
    target_center = np.array([0, 200, 5])
    target_radius = np.sqrt(74)

    # 提取观察点、遮蔽球心和半径
    pos = data[:, :3]  # (N, 3)
    poss = data[:, 3:6]  # (N, 3)
    R = data[:, 6]  # (N,)

    # 从观察点到目标中心的向量
    view_direction = target_center[np.newaxis, :] - pos  # (N, 3)
    view_distance = np.linalg.norm(view_direction, axis=1)  # (N,)

    # 处理观察点在目标中心的情况
    valid_mask = view_distance > 1e-10
    result = np.zeros(N, dtype=bool)

    if not np.any(valid_mask):
        return result

    # 只处理有效的情况
    view_direction_valid = view_direction[valid_mask]
    view_distance_valid = view_distance[valid_mask]
    pos_valid = pos[valid_mask]
    poss_valid = poss[valid_mask]
    R_valid = R[valid_mask]

    # 单位视线向量
    view_unit = view_direction_valid / view_distance_valid[:, np.newaxis]  # (N_valid, 3)

    # 从观察点到遮蔽球心的向量
    to_blocker = poss_valid - pos_valid  # (N_valid, 3)

    # 计算观察点到遮蔽球心在视线方向上的投影距离
    projection_distance = np.sum(to_blocker * view_unit, axis=1)  # (N_valid,)

    # 遮蔽球在观察点后方或目标球后方的情况
    valid_projection_mask = (projection_distance > 0) & (projection_distance < view_distance_valid)

    if not np.any(valid_projection_mask):
        return result

    # 进一步筛选有效情况
    projection_distance_valid = projection_distance[valid_projection_mask]
    to_blocker_valid = to_blocker[valid_projection_mask]
    R_valid_final = R_valid[valid_projection_mask]
    view_distance_valid_final = view_distance_valid[valid_projection_mask]

    # 计算观察点到遮蔽球心的距离
    distance_to_blocker = np.linalg.norm(to_blocker_valid, axis=1)  # (N_final,)

    # 计算视线到遮蔽球心的垂直距离
    perpendicular_distance = np.sqrt(np.maximum(0,
        distance_to_blocker**2 - projection_distance_valid**2))  # (N_final,)

    # 在投影点处，计算从观察点到目标球的视线锥半径
    angle_to_target_edge = np.arcsin(target_radius / view_distance_valid_final)  # (N_final,)
    view_cone_radius_at_blocker = projection_distance_valid * np.tan(angle_to_target_edge)  # (N_final,)

    # 判断是否遮蔽：遮蔽球是否与视线锥相交
    blocked_final = (R_valid_final + view_cone_radius_at_blocker) > perpendicular_distance

    # 将结果映射回原始索引
    valid_indices = np.where(valid_mask)[0]
    final_indices = valid_indices[np.where(valid_projection_mask)[0]]
    result[final_indices] = blocked_final

    return result
