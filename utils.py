import numpy as np
def calculate_distance(point1, point2, point3):
    """
    这个方法接受三个参数，三个参数都是numpy array 要求是1X3的坐标，前
    两个点用来确定一条直线，然后计算第三个点到这条直线的距离
    """
    # 计算方向向量
    direction = point2 - point1
    # 计算向量从point1到point3
    vector_to_point = point3 - point1
    # 计算叉积
    cross_product = np.cross(vector_to_point, direction)
    # 计算距离
    distance = np.linalg.norm(cross_product) / np.linalg.norm(direction)
    return distance

def is_sight_hidden(r, distance):
    """
    返回bool类型变量，输入云团半径r，还有云团中心到视线直线的距离，如果成功遮蔽
    返回TRUE，反之False"""
    return r > distance

def _poly_real_roots_desc(coeffs, tol=1e-10):
    """
    求解实根工具：输入按降幂排列的多项式系数 coeffs。
    返回实数根（按升序），复根虚部小于 tol 视为实根。
    自动裁掉前导零系数以适配降阶情形。
    """
    coeffs = np.array(coeffs, dtype=float)
    # 去除前导近零
    i = 0
    while i < len(coeffs) and abs(coeffs[i]) <= tol:
        i += 1
    if i == len(coeffs):
        # 多项式恒为 0：无特定根（驻点处处成立）
        return None  # 用 None 表示恒等式
    coeffs = coeffs[i:]
    deg = len(coeffs) - 1
    if deg <= 0:
        # 常数项（非零）==> 无根
        return []
    # 求根
    roots = np.roots(coeffs)
    real_roots = []
    for z in roots:
        if abs(z.imag) <= tol:
            real_roots.append(z.real)
    # 去重（应对重根数值抖动）
    real_roots = np.array(real_roots, dtype=float)
    if real_roots.size == 0:
        return []
    real_roots.sort()
    dedup = [real_roots[0]]
    for val in real_roots[1:]:
        if abs(val - dedup[-1]) > 1e-8:
            dedup.append(val)
    return dedup

def _f_of_t_constants(V, C, r, S):
    """
    构造 f(t)=cosθ(t)= (D + r(A cos t + B sin t)) / sqrt(E + 2r(F cos t + G sin t))
    所需常数 A,B,D,E,F,G 以及单位向量 u 和平移后的 C', S'。
    约定：圆在原坐标系 z=0 平面，C.z = 0。
    """
    V = np.asarray(V, dtype=float).reshape(3)
    C = np.asarray(C, dtype=float).reshape(3)
    S = np.asarray(S, dtype=float).reshape(3)

    C_prime = C - V         # (X_c, Y_c, -z_v)
    S_prime = S - V
    dS = np.linalg.norm(S_prime)
    if dS == 0.0:
        # 观测点在球心（几何上球遮挡一切方向）
        u = np.array([0.0, 0.0, 1.0])  # 任意
    else:
        u = S_prime / dS

    X_c, Y_c, Z_c = C_prime  # 注意 Z_c = -V.z
    u_x, u_y, u_z = u

    D = u_x * X_c + u_y * Y_c + u_z * Z_c
    A, B = u_x, u_y
    E = (X_c * X_c + Y_c * Y_c + Z_c * Z_c) + r * r
    F, G = X_c, Y_c

    return dict(A=A, B=B, D=D, E=E, F=F, G=G, u=u, C_prime=C_prime, S_prime=S_prime)

def _quartic_coeffs_for_stationary(A, B, D, E, F, G, r):
    """
    由导数驻点条件 2 N'(t) M(t) - N(t) M'(t) = 0 展开三角式，
    经 Weierstrass 代换 z=tan(t/2) 后得到的四次多项式系数（降幂）：
      alpha4 z^4 + alpha3 z^3 + alpha2 z^2 + alpha1 z + alpha0 = 0
    系数显式为：
      设
        P1 = (-A E + D F)          (sin t 一次项)
        Q1 = (B E - D G)           (cos t 一次项)
        P2 = r (-2 A G + B F)      (sin^2 t 项)
        Q2 = r (2 B F - A G)       (cos^2 t 项)
        R  = r (-A F + B G)        (sin t cos t 项)
      则
        alpha4 = Q2 - Q1
        alpha3 = 2 (P1 - R)
        alpha2 = 4 P2 - 2 Q2
        alpha1 = 2 (P1 + R)
        alpha0 = Q1 + Q2
    """
    P1 = (-A * E + D * F)
    Q1 = (B * E - D * G)
    P2 = r * (-2.0 * A * G + B * F)
    Q2 = r * ( 2.0 * B * F - A * G)
    R  = r * (-A * F + B * G)

    alpha4 = Q2 - Q1
    alpha3 = 2.0 * (P1 - R)
    alpha2 = 4.0 * P2 - 2.0 * Q2
    alpha1 = 2.0 * (P1 + R)
    alpha0 = Q1 + Q2

    return np.array([alpha4, alpha3, alpha2, alpha1, alpha0], dtype=float)

def _evaluate_f_from_z(z, A, B, D, E, F, G, r, eps=1e-12):
    """
    使用 z=tan(t/2) 的有理形式直接评估 f(t)。
    返回 f, t（弧度）。若 M(t) 非正或过小，返回 None, t。
    """
    den = 1.0 + z * z
    c = (1.0 - z * z) / den
    s = (2.0 * z) / den
    N = D + r * (A * c + B * s)
    M = E + 2.0 * r * (F * c + G * s)
    if M <= eps:
        t = 2.0 * np.arctan(z)
        return None, t
    f = N / np.sqrt(M)
    t = 2.0 * np.arctan(z)
    return f, t

def _evaluate_f_at_t(t, A, B, D, E, F, G, r, eps=1e-12):
    """
    直接用 t 评估 f(t)（用于退化/常函数情形）。
    """
    c, s = np.cos(t), np.sin(t)
    N = D + r * (A * c + B * s)
    M = E + 2.0 * r * (F * c + G * s)
    if M <= eps:
        return None
    return N / np.sqrt(M)

def circle_fmin_cos(V, C, r, S):
    """
    返回 f_min = min_t cosθ(t) 及达到该值的 t（弧度）。
    参数:
      - V: 观察点（3,）
      - C: 圆心（3,），要求 C[2]=0（圆位于 z=0 平面）
      - r: 圆半径
      - S: 球心（3,）
    返回:
      f_min, t_at_min（若无法评估返回 (None, None)）
    """
    consts = _f_of_t_constants(V, C, r, S)
    A, B, D, E, F, G = consts['A'], consts['B'], consts['D'], consts['E'], consts['F'], consts['G']

    # r=0 退化：圆退化为点，此时 f(t) 常数
    if abs(r) <= 0.0 + 0.0:
        f0 = _evaluate_f_at_t(0.0, A, B, D, E, F, G, r=0.0)
        return f0, 0.0

    # 构造四次多项式
    coeffs = _quartic_coeffs_for_stationary(A, B, D, E, F, G, r)
    roots = _poly_real_roots_desc(coeffs, tol=1e-10)

    candidates = []

    if roots is None:
        # 驻点条件恒成立：f(t) 为常函数
        f0 = _evaluate_f_at_t(0.0, A, B, D, E, F, G, r)
        return f0, 0.0

    # 用实根生成候选 t 并评估 f
    for z in roots:
        f, t = _evaluate_f_from_z(z, A, B, D, E, F, G, r)
        if f is not None and np.isfinite(f):
            candidates.append((f, t))

    # 极少数情况下（例如观测点恰在圆上导致 M(t)=0）可能没有可用候选
    # 为了得到 f_min，我们在有限代表点上评估（这不是网格采样，只是退化兜底）
    if not candidates:
        for t in (0.0, np.pi/2, np.pi, 3*np.pi/2):
            f = _evaluate_f_at_t(t, A, B, D, E, F, G, r)
            if f is not None and np.isfinite(f):
                candidates.append((f, t))

    if not candidates:
        return None, None

    # 取最小 f
    f_vals = np.array([ft[0] for ft in candidates], dtype=float)
    idx = int(np.argmin(f_vals))
    f_min, t_min = candidates[idx]
    return f_min, t_min

def circle_fully_occluded_by_sphere(V, C, r, S, R, return_debug=False):
    """
    判定：从观察点 V 看，位于 z=0 的半径 r 的圆（圆心 C），是否被球（球心 S，半径 R）完全遮挡。
    精确必要充分条件（无采样）：
      max_t angle(u, X(t)) <= alpha_s
      等价于 min_t cosθ(t) >= cos(alpha_s)
    参数:
      - V: 观察点（3,）
      - C: 圆心（3,），要求 C[2]=0
      - r: 圆半径
      - S: 球心（3,）
      - R: 球半径
    返回:
      - occluded: bool
      - 若 return_debug=True，额外返回字典，包含 f_min、cos_alpha_s、t_min 等。
    """
    V = np.asarray(V, dtype=float).reshape(3)
    C = np.asarray(C, dtype=float).reshape(3)
    S = np.asarray(S, dtype=float).reshape(3)
    R = float(R)
    r = float(r)

    # 球的观测几何退化：观测点在/内球 => 球遮挡一切方向
    S_prime = S - V
    dS = np.linalg.norm(S_prime)
    if dS <= 0.0 + 0.0 or R >= dS:
        if return_debug:
            return True, dict(reason="observer_in_or_at_sphere", f_min=None, cos_alpha_s=None, t_min=None)
        return True

    # 计算 f_min
    f_min, t_min = circle_fmin_cos(V, C, r, S)
    if f_min is None:
        # 无法稳健评估（极端退化）
        if return_debug:
            return False, dict(reason="degenerate_no_eval", f_min=None, cos_alpha_s=None, t_min=None)
        return False

    # 球锥阈值 cos(alpha_s) = sqrt(1 - (R/dS)^2)
    ratio = R / dS
    if ratio >= 1.0:
        cos_alpha_s = 0.0
    else:
        cos_alpha_s = np.sqrt(max(0.0, 1.0 - ratio * ratio))

    occluded = (f_min >= cos_alpha_s - 1e-12)

    if return_debug:
        return occluded, dict(f_min=float(f_min), cos_alpha_s=float(cos_alpha_s), t_min=float(t_min))
    return occluded