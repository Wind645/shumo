"""场景/物理/任务相关常量 (根据题目文字描述统一集中于此模块)。

说明:
 - 所有坐标单位: m
 - 时间单位: s
 - 题干中文描述的固定数值不要再在其它模块硬编码, 统一从这里导入。
"""

import numpy as np

# ============================= 物理常量 =============================
G = 9.8  # 重力加速度 (m/s^2)

# ============================= 烟幕参数 =============================
# 烟幕弹起爆后瞬时形成球状烟幕云团: 半径 10 m 内有效遮蔽, 持续 20 s, 以 3 m/s 匀速下沉
SMOKE_RADIUS_DEFAULT = 10.0          # 球形云团有效遮蔽半径 (m)
SMOKE_LIFETIME_DEFAULT = 20.0        # 有效遮蔽持续时间 (s)
SMOKE_DESCENT_DEFAULT = 3.0          # 云团匀速下沉速度 (m/s)

# ============================= 目标/圆柱体 =============================
# 圆柱形真目标: 半径 7 m, 高 10 m ; 圆柱底面圆心 C_base=(0,200,0)
C_BASE_DEFAULT = np.array([0.0, 200.0, 0.0], dtype=np.float32)
R_CYL_DEFAULT = 7.0
H_CYL_DEFAULT = 10.0

# 假目标 (题干: 以假目标为原点, 水平面为 xy 平面)
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# ============================= 导弹参数 =============================
MISSILE_SPEED = 300.0  # 空地导弹速度 (m/s)
MISSILES_POS0 = {
	"M1": np.array([20000.0,      0.0, 2000.0], dtype=np.float32),
	"M2": np.array([19000.0,    600.0, 2100.0], dtype=np.float32),
	"M3": np.array([18000.0,   -600.0, 1900.0], dtype=np.float32),
}

# ============================= 无人机初始状态 =============================
DRONES_POS0 = {
	# FY1 ... FY5 初始坐标
	"FY1": np.array([17800.0,     0.0, 1800.0], dtype=np.float32),
	"FY2": np.array([12000.0,  1400.0, 1400.0], dtype=np.float32),
	"FY3": np.array([ 6000.0, -3000.0,  700.0], dtype=np.float32),
	"FY4": np.array([11000.0,  2000.0, 1800.0], dtype=np.float32),
	"FY5": np.array([13000.0, -2000.0, 1300.0], dtype=np.float32),
}

# 无人机速度约束 (m/s)
DRONE_SPEED_MIN = 70.0
DRONE_SPEED_MAX = 140.0

# 同一无人机投放两枚烟幕干扰弹最小时间间隔 (s)
MIN_BOMB_INTERVAL = 1.0

# ============================= 采样/向量化默认参数 =============================
DEFAULT_CYL_SAMPLE_N_THETA = 48
DEFAULT_CYL_SAMPLE_N_H = 16
DEFAULT_CYL_SAMPLE_CAP_RADIAL = 6

# 顶/底圆面向量化快速判定时的简单采样 (只在自定义 vectorized_sampling 时使用)
VECTORIZED_CAPS_SAMPLE_K = 48  # 48 个底面点 + 48 个顶面点

__all__ = [
	# 物理
	"G",
	# 烟幕
	"SMOKE_RADIUS_DEFAULT", "SMOKE_LIFETIME_DEFAULT", "SMOKE_DESCENT_DEFAULT",
	# 圆柱
	"C_BASE_DEFAULT", "R_CYL_DEFAULT", "H_CYL_DEFAULT",
	# 目标/导弹
	"FAKE_TARGET_ORIGIN", "MISSILE_SPEED", "MISSILES_POS0",
	# 无人机
	"DRONES_POS0", "DRONE_SPEED_MIN", "DRONE_SPEED_MAX", "MIN_BOMB_INTERVAL",
	# 采样参数
	"DEFAULT_CYL_SAMPLE_N_THETA", "DEFAULT_CYL_SAMPLE_N_H", "DEFAULT_CYL_SAMPLE_CAP_RADIAL",
	"VECTORIZED_CAPS_SAMPLE_K",
]
