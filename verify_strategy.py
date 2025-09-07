import numpy as np
from typing import List, Tuple, Dict, Iterable
from sim_core import Missile, Drone, Cylinder, Simulator, C_BASE_DEFAULT, R_CYL_DEFAULT, H_CYL_DEFAULT, SMOKE_RADIUS_DEFAULT, SMOKE_LIFETIME_DEFAULT, SMOKE_DESCENT_DEFAULT

def run_custom_strategy(
    drone_dir: Iterable[float],
    drone_speed: float,
    bomb_params: List[Tuple[float, float]],  # [(deploy_time, explode_delay), ...]
    dt: float = 0.05,
    occlusion_method: str = "sampling",
    n_theta: int = 48, n_h: int = 16, n_cap_radial: int = 6, check_caps: bool = True,
    verbose: bool = False
) -> Dict:
    """
    用自定义无人机方向/速度和投弹计划验证遮蔽效果。
    bomb_params: [(投放时刻, 起爆延迟), ...]
    返回与 Simulator.run 相同的统计。
    """
    fake_target = np.array([0.0, 0.0, 0.0], dtype=float)

    # 导弹保持与问题1一致
    m1 = Missile(
        pos0=np.array([20000.0, 0.0, 2000.0], dtype=float),
        speed=300.0,
        target=fake_target
    )

    # 无人机初始状态保持与问题1一致
    fy1_pos0 = np.array([17800.0, 0.0, 1800.0], dtype=float)
    fy1 = Drone(
        pos0=fy1_pos0,
        direction=np.asarray(drone_dir, dtype=float),
        speed=float(drone_speed)
    )

    # 组装投弹计划（单机索引固定为 0）
    schedules = [(0, float(t_dep), float(t_delay)) for (t_dep, t_delay) in bomb_params]

    sim = Simulator(
        missile=m1,
        drones=[fy1],
        cylinder=Cylinder(C_base=C_BASE_DEFAULT.copy(), r=R_CYL_DEFAULT, h=H_CYL_DEFAULT),
        n_theta=n_theta, n_h=n_h, n_cap_radial=n_cap_radial, check_caps=check_caps,
        smoke_radius=SMOKE_RADIUS_DEFAULT,
        smoke_lifetime=SMOKE_LIFETIME_DEFAULT,
        smoke_descent=SMOKE_DESCENT_DEFAULT,
        occlusion_method=occlusion_method,
        schedules=schedules
    )
    return sim.run(dt=dt, t_max=None, verbose=verbose)

if __name__ == "__main__":
    # 你的解参数
    drone_dir = [0.99628969, 0.08606313, 0.0]
    drone_speed = 107.70993563166081
    bombs = [(0.8196645799210821, 0.3107517042702438)]  # 一枚：在0.8197s投，延迟0.3108s起爆

    res = run_custom_strategy(
        drone_dir=drone_dir,
        drone_speed=drone_speed,
        bomb_params=bombs,
        dt=0.01,                 # 精度需求可调；更小更精细更慢
        occlusion_method="judge_caps",  # 或 "sampling"（更全面但更慢）
        verbose=True
    )
    print("有效遮蔽总时长:", res["occluded_time"], "s (导弹总飞行:", res["missile_flight_time"], "s)")
