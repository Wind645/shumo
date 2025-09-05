from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Union

import numpy as np

from sim_core import (
    Missile, Drone, Cylinder, Simulator, OcclusionEvaluator,
    C_BASE_DEFAULT, R_CYL_DEFAULT, H_CYL_DEFAULT,
    SMOKE_DESCENT_DEFAULT, SMOKE_LIFETIME_DEFAULT, SMOKE_RADIUS_DEFAULT,
)

Vec3 = np.ndarray

FAKE_TARGET = np.array([0.0, 0.0, 0.0], dtype=float)

MISSILES_DEF = {
    "M1": dict(pos0=np.array([20000.0,     0.0, 2000.0]), speed=300.0, target=FAKE_TARGET),
    "M2": dict(pos0=np.array([19000.0,   600.0, 2100.0]), speed=300.0, target=FAKE_TARGET),
    "M3": dict(pos0=np.array([18000.0,  -600.0, 1900.0]), speed=300.0, target=FAKE_TARGET),
}

SPEED_MIN = 70.0
SPEED_MAX = 140.0

def _as_vec3(x) -> Vec3:
    return np.asarray(x, dtype=float).reshape(3)

def _norm(v: Vec3) -> Vec3:
    v = _as_vec3(v)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n

def azimuth_to_dir(azim_rad: float) -> Vec3:
    """
    将平面方位角(弧度，x轴为0，逆时针为正)转为水平单位方向。
    """
    return np.array([np.cos(azim_rad), np.sin(azim_rad), 0.0], dtype=float)

def _validate_bombs(bombs: List[Dict]):
    times = sorted(float(b["deploy_time"]) for b in bombs)
    # 间隔 >= 1s
    for i in range(1, len(times)):
        if times[i] - times[i-1] < 1.0 - 1e-9:
            raise ValueError(f"同一无人机投弹间隔不足1秒: {times[i-1]} -> {times[i]}")

def build_drones_and_schedules(decision: Dict) -> Tuple[List[Drone], List[Tuple[int, float, float]]]:
    """
    根据决策变量组装 Drone 列表与 schedules:
      decision = {
        "drones": [
          {
            "pos0": [x,y,z],
            "speed": v (70~140),
            // 三选一指定航向:
            "direction": [dx,dy,dz] (任意向量，将归一化)
            or "azimuth": rad
            or "aim_fake_target": true  // 指向(0,0,z0)
            "bombs": [
              {"deploy_time": t1, "explode_delay": tau1},
              ...
            ]
          },
          ...
        ]
      }
    返回: (drones, schedules) 其中 schedules 为 [(di, deploy_time, explode_delay), ...]
    """
    drones: List[Drone] = []
    schedules: List[Tuple[int, float, float]] = []

    if "drones" not in decision or not isinstance(decision["drones"], list):
        raise ValueError("decision['drones'] 必须为列表")

    for di, d in enumerate(decision["drones"]):
        pos0 = _as_vec3(d["pos0"])
        v = float(d["speed"])
        if not (SPEED_MIN - 1e-9 <= v <= SPEED_MAX + 1e-9):
            raise ValueError(f"无人机速度超范围[70,140]: {v}")

        # 航向解析（优先顺序：direction > azimuth > aim_fake_target）
        if "direction" in d:
            dir_vec = _norm(d["direction"])
        elif "azimuth" in d:
            dir_vec = _norm(azimuth_to_dir(float(d["azimuth"])))
        elif d.get("aim_fake_target", False):
            target_h = FAKE_TARGET.copy()
            target_h[2] = pos0[2]
            dir_vec = _norm(target_h - pos0)
        else:
            raise ValueError("必须提供 direction 或 azimuth 或 aim_fake_target 之一作为航向")

        # 组装 Drone
        drones.append(Drone(pos0=pos0, direction=dir_vec, speed=v))

        # 校验与收集投弹计划
        bombs = d.get("bombs", [])
        _validate_bombs(bombs)
        for b in bombs:
            schedules.append((di, float(b["deploy_time"]), float(b["explode_delay"])))

    # 全局按时间排序，保证投放触发一致
    schedules.sort(key=lambda x: x[1])
    return drones, schedules

def _build_missiles(which: Union[str, List[str]]) -> Dict[str, Missile]:
    """
    which: 'M1' / 'M1M2M3' / ['M1','M3',...]
    """
    if isinstance(which, str):
        if which == "M1M2M3":
            keys = ["M1", "M2", "M3"]
        else:
            keys = [which]
    else:
        keys = list(which)

    res: Dict[str, Missile] = {}
    for k in keys:
        if k not in MISSILES_DEF:
            raise ValueError(f"未知导弹标识: {k}")
        spec = MISSILES_DEF[k]
        res[k] = Missile(pos0=spec["pos0"], speed=spec["speed"], target=spec["target"])
    return res

def _max_flight_time(missiles: Dict[str, Missile]) -> float:
    return max(m.flight_time for m in missiles.values())

def _eval_occlusion_over_timeline(
    missiles: Dict[str, Missile],
    timeline: List[Dict],
    cyl: Cylinder,
    method: str
) -> Dict[str, float]:
    """
    复用同一时间线的云团状态，对不同导弹分别判定遮蔽并累积时长。
    """
    evaluator = OcclusionEvaluator(cyl, method=method)
    occluded_time: Dict[str, float] = {k: 0.0 for k in missiles.keys()}

    # 时间步长可从相邻时间戳估计（更稳妥），也可以要求调用处统一 dt
    dt = None
    if len(timeline) >= 2:
        dt = float(timeline[1]["t"] - timeline[0]["t"])
    else:
        dt = 0.05

    for rec in timeline:
        t = float(rec["t"])
        spheres = [(S, R) for (S, R) in rec.get("clouds", [])]  # [(center(np.ndarray), radius), ...]

        for name, m in missiles.items():
            if t > m.flight_time + 1e-12:
                continue  # 超过飞行时间不再统计
            if len(spheres) == 0:
                continue
            V = m.position(t)
            ok, _ = evaluator.fully_occluded(V, spheres)
            if ok:
                occluded_time[name] += dt

    return occluded_time

def simulate_with_decision(
    decision: Dict,
    which: Union[str, List[str]] = "M1",
    *,
    dt: float = 0.05,
    occlusion_method: str = "sampling",  # 'sampling' or 'judge_caps'
    verbose: bool = False,
    return_timeline: bool = False
) -> Dict:
    """
    统一入口：给定决策变量，自动仿真并返回有效遮蔽时长。
    - decision: 见 build_drones_and_schedules() 的 schema
    - which: 'M1' / 'M1M2M3' / ['M1','M2',...]
    - 返回:
        {
          "occluded_time": {"M1": t1, "M2": t2, ...},
          "total": Σti,
          "missile_flight_time": {"M1": T1, ...},
          "dt": dt,
          "timeline": [...],  # 可选
        }
    """
    # 1) 构造无人机与投弹计划
    drones, schedules = build_drones_and_schedules(decision)

    # 2) 要评估的导弹集合
    missiles = _build_missiles(which)
    T_max = _max_flight_time(missiles)

    # 3) 用任意导弹（或新建一个）驱动 Simulator 生成云团时间线
    #    选择集合中的第一个导弹；仅用于推进时间和生成云团，遮蔽统计用我们自己的 evaluator。
    first_missile: Missile = list(missiles.values())[0]

    cyl = Cylinder(C_base=C_BASE_DEFAULT.copy(), r=R_CYL_DEFAULT, h=H_CYL_DEFAULT)

    sim = Simulator(
        missile=first_missile,
        drones=drones,
        cylinder=cyl,
        n_theta=48, n_h=16, n_cap_radial=6, check_caps=True,
        smoke_radius=SMOKE_RADIUS_DEFAULT,
        smoke_lifetime=SMOKE_LIFETIME_DEFAULT,
        smoke_descent=SMOKE_DESCENT_DEFAULT,
        occlusion_method=occlusion_method,
        schedules=schedules
    )
    sim_out = sim.run(dt=dt, t_max=T_max, verbose=verbose)

    # 4) 基于时间线对各导弹单独判定遮蔽并累积
    occluded_time = _eval_occlusion_over_timeline(missiles, sim_out["timeline"], cyl, occlusion_method)

    result = {
        "occluded_time": occluded_time,
        "total": float(sum(occluded_time.values())),
        "missile_flight_time": {k: float(v.flight_time) for k, v in missiles.items()},
        "dt": float(dt),
    }
    if return_timeline:
        result["timeline"] = sim_out["timeline"]
    return result

# ------------ 便捷封装（面向题目各问题） -----------------

def evaluate_problem2(
    speed: float,
    direction: Optional[Iterable[float]] = None,
    azimuth: Optional[float] = None,
    explode_delay: float = 3.6,
    release_time: float = 1.5,
    occlusion_method: str = "sampling",
    dt: float = 0.05,
) -> Dict:
    """
    问题2：FY1 投放1枚烟幕弹干扰 M1，未知数为 FY1 航向/速度/投放与起爆。
    传入 speed + (direction | azimuth | aim_fake_target)，返回 M1 的遮蔽时长。
    """
    fy1 = {
        "pos0": [17800.0, 0.0, 1800.0],
        "speed": float(speed),
        "bombs": [{"deploy_time": float(release_time), "explode_delay": float(explode_delay)}]
    }
    if direction is not None:
        fy1["direction"] = list(direction)
    elif azimuth is not None:
        fy1["azimuth"] = float(azimuth)
    else:
        fy1["aim_fake_target"] = True  # 默认指向假目标

    decision = {"drones": [fy1]}
    return simulate_with_decision(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

def evaluate_problem3(
    bombs: List[Tuple[float, float]],
    speed: float = 120.0,
    azimuth: Optional[float] = None,
    direction: Optional[Iterable[float]] = None,
    dt: float = 0.05,
    occlusion_method: str = "sampling",
) -> Dict:
    """
    问题3：FY1 投放3枚烟幕弹干扰 M1。
    bombs: [(deploy_time, explode_delay), ...] 长度≤3
    """
    fy1 = {
        "pos0": [17800.0, 0.0, 1800.0],
        "speed": float(speed),
        "bombs": [{"deploy_time": float(t), "explode_delay": float(d)} for (t, d) in bombs]
    }
    if direction is not None:
        fy1["direction"] = list(direction)
    elif azimuth is not None:
        fy1["azimuth"] = float(azimuth)
    else:
        fy1["aim_fake_target"] = True

    decision = {"drones": [fy1]}
    return simulate_with_decision(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

def evaluate_problem4(
    drones_spec: List[Dict],
    dt: float = 0.05,
    occlusion_method: str = "sampling",
) -> Dict:
    """
    问题4：3架无人机（FY1,FY2,FY3 各1弹）干扰 M1。
    drones_spec 采用 build_drones_and_schedules 所需的单机 schema（只放三架机）。
    """
    decision = {"drones": drones_spec}
    return simulate_with_decision(decision, which="M1", dt=dt, occlusion_method=occlusion_method)

def evaluate_problem5(
    drones_spec: List[Dict],
    dt: float = 0.05,
    occlusion_method: str = "sampling",
) -> Dict:
    """
    问题5：5架无人机（每架≤3弹）干扰 3 枚导弹（M1,M2,M3）。
    drones_spec 采用 build_drones_and_schedules 所需的单机 schema。
    返回每枚导弹遮蔽时长与总和。
    """
    decision = {"drones": drones_spec}
    return simulate_with_decision(decision, which="M1M2M3", dt=dt, occlusion_method=occlusion_method)

# ------------- 示例 ---------------

if __name__ == "__main__":
    # 问题1复现（相当于问题2的特例）
    res = evaluate_problem2(
        speed=120.0,
        azimuth=np.arctan2(0.0, -1.0),  # 指向假目标的水平方位（FY1 在 x>0, y=0）
        release_time=1.5,
        explode_delay=3.6,
        occlusion_method="judge_caps",
        dt=0.05
    )
    print("M1 遮蔽时长(s):", res["occluded_time"]["M1"], "总计(s):", res["total"])
