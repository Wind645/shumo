from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from simcore import Drone
from .decision import build_drones_and_schedules

G = 9.8

def summarize_decision(decision: Dict) -> List[Dict]:
    drones, schedules = build_drones_and_schedules(decision)
    rows: List[Dict] = []
    for di, t_dep, dly in schedules:
        d: Drone = drones[di]
        pos_rel = d.position(t_dep)
        vel_rel = d.dir * d.speed
        t_exp = t_dep + dly
        a = np.array([0.0, 0.0, -G], dtype=float)
        pos_exp = pos_rel + vel_rel * dly + 0.5 * a * (dly * dly)
        rows.append(dict(
            drone_index=int(di),
            drone_pos0_x=float(d.pos0[0]), drone_pos0_y=float(d.pos0[1]), drone_pos0_z=float(d.pos0[2]),
            speed=float(d.speed),
            dir_x=float(d.dir[0]), dir_y=float(d.dir[1]), dir_z=float(d.dir[2]),
            deploy_time=float(t_dep), explode_delay=float(dly), explode_time=float(t_exp),
            release_x=float(pos_rel[0]), release_y=float(pos_rel[1]), release_z=float(pos_rel[2]),
            explode_x=float(pos_exp[0]), explode_y=float(pos_exp[1]), explode_z=float(pos_exp[2]),
        ))
    return rows

def export_strategy_to_excel(file_path: str, rows: List[Dict]) -> None:
    try:
        from openpyxl import Workbook  # type: ignore
    except Exception as e:
        raise RuntimeError("需要安装 openpyxl 才能导出 .xlsx。请先安装：pip install openpyxl") from e
    wb = Workbook()
    ws = wb.active
    ws.title = "strategy"
    headers = list(rows[0].keys()) if rows else [
        "drone_index","drone_pos0_x","drone_pos0_y","drone_pos0_z","speed","dir_x","dir_y","dir_z",
        "deploy_time","explode_delay","explode_time","release_x","release_y","release_z","explode_x","explode_y","explode_z",
    ]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    wb.save(file_path)

def save_result_q3(file_path: str, bombs: List[Tuple[float,float]], speed: float = 120.0, azimuth: float | None = None, direction = None) -> None:
    fy1 = {"pos0": [17800.0, 0.0, 1800.0], "speed": float(speed), "bombs": [{"deploy_time": float(t), "explode_delay": float(d)} for (t,d) in bombs]}
    if direction is not None:
        fy1["direction"] = list(direction)
    elif azimuth is not None:
        fy1["azimuth"] = float(azimuth)
    else:
        fy1["aim_fake_target"] = True
    rows = summarize_decision({"drones": [fy1]})
    export_strategy_to_excel(file_path, rows)

def save_result_q4(file_path: str, drones_spec: List[Dict]) -> None:
    rows = summarize_decision({"drones": drones_spec})
    export_strategy_to_excel(file_path, rows)

def save_result_q5(file_path: str, drones_spec: List[Dict]) -> None:
    rows = summarize_decision({"drones": drones_spec})
    export_strategy_to_excel(file_path, rows)
