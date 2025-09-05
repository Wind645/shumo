import numpy as np
import time
from utils import calculate_distance, is_sight_hidden, circle_fmin_cos, circle_fully_occluded_by_sphere

# 定义测试数据
V = np.array([0.0, 0.0, 2.0])  # 观察点
C = np.array([0.5, 0.0, 0.0])  # 圆心 (z=0)
r = 0.3  # 圆半径
S = np.array([0.2, 0.1, 1.0])  # 球心
R = 0.5  # 球半径

# 额外测试数据（用于 calculate_distance 和 is_sight_hidden）
point1 = np.array([0.0, 0.0, 0.0])
point2 = np.array([1.0, 0.0, 0.0])
point3 = np.array([0.5, 0.5, 0.0])
distance = calculate_distance(point1, point2, point3)  # 预计算距离

# 测试次数
num_tests = 1000

def time_function(func, *args, num_runs=num_tests):
    """测量函数运行时间的辅助函数"""
    start = time.time()
    for _ in range(num_runs):
        func(*args)
    end = time.time()
    return (end - start) / num_runs  # 平均时间

# 测试 calculate_distance
avg_time_dist = time_function(calculate_distance, point1, point2, point3)
print(f"calculate_distance 平均运行时间: {avg_time_dist:.6f} 秒")

# 测试 is_sight_hidden
avg_time_hidden = time_function(is_sight_hidden, r, distance)
print(f"is_sight_hidden 平均运行时间: {avg_time_hidden:.6f} 秒")

# 测试 circle_fmin_cos
avg_time_fmin = time_function(circle_fmin_cos, V, C, r, S)
print(f"circle_fmin_cos 平均运行时间: {avg_time_fmin:.6f} 秒")

# 测试 circle_fully_occluded_by_sphere
avg_time_occluded = time_function(circle_fully_occluded_by_sphere, V, C, r, S, R)
print(f"circle_fully_occluded_by_sphere 平均运行时间: {avg_time_occluded:.6f} 秒")

# 可选：运行一次并打印结果（验证正确性）
print("\n--- 一次运行结果 ---")
dist = calculate_distance(point1, point2, point3)
hidden = is_sight_hidden(r, dist)
f_min, t_min = circle_fmin_cos(V, C, r, S)
occluded = circle_fully_occluded_by_sphere(V, C, r, S, R, return_debug=True)
print(f"距离: {dist}")
print(f"是否遮挡: {hidden}")
print(f"f_min: {f_min}, t_min: {t_min}")
print(f"完全遮挡: {occluded[0]}, 调试: {occluded[1]}")