"""
Multi-fidelity wrapper to optimize Problem 3 (单无人机多炸弹遮蔽最大化)

该文件将抽象多保真优化器 (abstract.multi_fidelity.optimize_multi_fidelity)
应用到题目 3：变量含义 ( bombs_count=3 示例 ):
    x = [ speed, azimuth,
          t1, d1,
          t2, d2,
          t3, d3 ]

目标: 最大化对 M1 的遮蔽时长 (occluded_time["M1"])
实现方式: 把 maximization 转换成 minimize(-occlusion)

多保真层级示例 (默认 3 层):
  fidelity 0: 粗时间步 dt=0.10, method='rough_caps'
  fidelity 1: 中等 dt=0.05,  method='rough_caps'
  fidelity 2: 精细 dt=0.02,  method='judge_caps'

可根据需求调整 FIDELITY_CONFIG 列表顺序或数量。

运行:
    uv run -m abstract.mf_q3

输出:
  - 每个保真层日志
  - 最终最优参数 / 最高保真真实评估
  - 还会列出 approximate true occlusion (直接从最终评估结构中读取)

如需修改:
  - 调整 BOMBS_COUNT
  - 调整 FIDELITY_CONFIG
  - 调整 INIT_CANDIDATES / ETA / PERTURB_SCALE 等参数

依赖:
  - optimize_multi_fidelity (抽象多保真优化器)
  - bounds_for_problem (获取编码边界)
  - evaluate_problem3 (实际评估)
"""

from __future__ import annotations
from typing import List, Tuple, Sequence, Dict
import math

from abstract.multi_fidelity import optimize_multi_fidelity
from optimizer.spec import bounds_for_problem
from api.problems import evaluate_problem3

# ---------------------- 可调参数区 ----------------------
BOMBS_COUNT = 3                # 炸弹数 (与题 3 需求一致, 可更改)
FIDELITY_CONFIG: List[Dict] = [
    # 每个字典定义一个 fidelity 等级的评估配置
    {"dt": 0.10, "method": "rough_caps_torch"},
    {"dt": 0.05, "method": "rough_caps_torch"},
    {"dt": 0.02, "method": "rough_caps_torch"},  # 最高保真
]
INIT_CANDIDATES = 96           # 初始随机点
ETA = 1.0                      # successive halving 比例 越大越狠
PERTURB_SCALE = 0.18           # 局部扰动幅度(相对各维跨度)
EXPAND_RATIO = 0.6             # 每层扩展比例
SEED = 42
VERBOSE = True
MAX_HIGH_FIDELITY = None       # 可设为整数限制最终层候选
# -------------------------------------------------------

# 惩罚: 不合法参数返回一个很大的正数 (因为在做最小化)
_PENALTY = 1e12

def _decode_q3_vector(x: Sequence[float]) -> Tuple[float, float, List[Tuple[float,float]]]:
    """
    简单解码: 假设固定 BOMBS_COUNT=3 -> len(x) 应为 2 + 2*3 = 8
    返回: (speed, azimuth, [(t1,d1),(t2,d2),(t3,d3)])
    """
    xs = list(map(float, x))
    expected_len = 2 + 2 * BOMBS_COUNT
    if len(xs) != expected_len:
        raise ValueError(f"维度不符: 期望 {expected_len}, 得到 {len(xs)}")
    speed = xs[0]; azim = xs[1]
    bombs = []
    idx = 2
    for _ in range(BOMBS_COUNT):
        t = xs[idx]; d = xs[idx+1]; idx += 2
        bombs.append((t, d))
    return speed, azim, bombs

def _is_invalid(speed: float, bombs: List[Tuple[float,float]]) -> bool:
    # 速度范围 (和 spec 中保持一致: 70~140)
    if not (70.0 <= speed <= 140.0):
        return True
    # 时间/延迟约束 + 最小间隔 (>=1s)
    # 排序以检查间隔
    bombs_sorted = sorted(bombs, key=lambda p: p[0])
    last_t = None
    for t, d in bombs_sorted:
        if t < 0.0 or d <= 0.0:
            return True
        if last_t is not None and t - last_t < 1.0 - 1e-9:
            return True
        last_t = t
    return False

def _objective_problem3(x: Sequence[float], fidelity: int) -> float:
    """
    黑盒接口: 返回需要最小化的值 (这里为 -occlusion 或 惩罚).
    fidelity 通过 FIDELITY_CONFIG 取得 (dt, method).
    """
    speed, azim, bombs = _decode_q3_vector(x)
    if _is_invalid(speed, bombs):
        return _PENALTY
    # 当前 fidelity 配置
    cfg = FIDELITY_CONFIG[fidelity]
    res = evaluate_problem3(
        bombs=bombs,
        speed=speed,
        azimuth=azim,
        dt=cfg["dt"],
        occlusion_method=cfg["method"],
    )
    occ = float(res["occluded_time"]["M1"])
    # 最大化 occ => 最小化 -occ
    return -occ

def run() -> None:
    dim_bounds = bounds_for_problem(3, BOMBS_COUNT)  # [(speed_min,max), (az_min,max), (t1_min,max), (d1_min,max), ...]
    fidelity_levels = len(FIDELITY_CONFIG)

    result = optimize_multi_fidelity(
        dim_bounds=dim_bounds,
        evaluate=_objective_problem3,
        fidelity_levels=fidelity_levels,
        init_candidates=INIT_CANDIDATES,
        eta=ETA,
        max_high_fidelity=MAX_HIGH_FIDELITY,
        perturb_scale=PERTURB_SCALE,
        local_expand=True,
        expand_quota_ratio=EXPAND_RATIO,
        seed=SEED,
        verbose=VERBOSE,
    )

    best_x = result.best_x
    best_speed, best_az, best_bombs = _decode_q3_vector(best_x)
    # 用最高保真配置重新做一次权威评估
    final_cfg = FIDELITY_CONFIG[-1]
    final_eval = evaluate_problem3(
        bombs=best_bombs,
        speed=best_speed,
        azimuth=best_az,
        dt=final_cfg["dt"],
        occlusion_method=final_cfg["method"],
    )
    final_occ = float(final_eval["occluded_time"]["M1"])

    print("\n=== Multi-Fidelity Problem 3 Result ===")
    print("Best decision vector:", best_x)
    print("Decoded:")
    print("  speed:", best_speed)
    print("  azimuth:", best_az)
    print("  bombs (t, delay):", best_bombs)
    print("Highest fidelity re-eval occlusion:", final_occ)
    print("Internal minimized value (best_value):", result.best_value, "(= -occlusion)")

    print("\nLevel summary:")
    for log in result.history:
        print(f"  Level {log.fidelity}: evaluated={log.evaluated}, kept={log.kept}, "
              f"expanded={log.expanded}, time={log.elapsed_sec:.3f}s")

    print("\nMeta:", result.meta)

if __name__ == "__main__":
    run()
