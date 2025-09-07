import numpy as np
from batch_sample import is_cylinder_blocked_vectorized
from batch_rough import is_sphere_blocked_vectorized
from jiexi import is_cylinder_blocked_analytic  # 新增导入

def compare_methods(N=50000, K_values=[8,16,24,32,48,64,96,128]):
    """
    生成 N 个随机测试数据，比较两个方法的输出一致性。
    测试多个 K 值（K: 采样点数，上下底面各 K 个，共 2K 个点，用于判断圆柱遮挡的严格性）。
    数据生成偏向 True 判定附近：缩小导弹范围，烟幕靠近目标，增大 R_smoke 下限，以聚焦遮挡场景。
    """
    # 生成随机数据：导弹位置、烟幕位置、烟幕半径（调整范围以偏向 True）
    np.random.seed(42)  # 固定种子以便重现
    missile = np.random.uniform(-1000, 1000, (N, 3))  # 缩小范围
    smoke = np.random.uniform(-500, 500, (N, 3))  # 靠近目标 (0,200,5)
    smoke[:, 1] += 200  # 偏移 y 轴靠近目标
    smoke[:, 2] += 5    # 偏移 z 轴靠近目标
    R_smoke = np.random.uniform(20, 100, (N, 1))  # 增大下限
    data = np.hstack([missile, smoke, R_smoke])

    print(f"测试样本数: {N}")
    print(f"导弹位置范围: (-1000, 1000)")
    print(f"烟幕位置范围: 靠近目标 (0,200,5)")
    print(f"烟幕半径范围: (20, 100)")
    print(f"测试 K 值: {K_values}")

    # 计算 rough 方法的结果（一次）
    result_rough = is_sphere_blocked_vectorized(data)
    true_rough = np.sum(result_rough)
    print(f"Rough 方法: True={true_rough}, False={N - true_rough}")

    # 对每个 K 值测试 sample 方法
    for K in K_values:
        print(f"\nK={K}:")
        result_sample = is_cylinder_blocked_vectorized(data, K=K)
        true_sample = np.sum(result_sample)
        print(f"  Sample: True={true_sample}, False={N - true_sample}")

        # 计算一致性
        matches = np.sum(result_sample == result_rough)
        consistency = matches / N * 100
        print(f"  一致性: {consistency:.2f}% ({matches}/{N})")

        # 显示不一致样本统计（简化）
        diff_indices = np.where(result_sample != result_rough)[0]
        if len(diff_indices) > 0:
            sample_true_rough_false = np.sum((result_sample[diff_indices] == True) & (result_rough[diff_indices] == False))
            sample_false_rough_true = np.sum((result_sample[diff_indices] == False) & (result_rough[diff_indices] == True))
            print(f"  不一致: {len(diff_indices)} (Sample T/R F: {sample_true_rough_false}, Sample F/R T: {sample_false_rough_true})")
        else:
            print("  不一致: 0")

        # 简洁分析
        if K == K_values[0]:
            print("  分析: 初始一致性较低，可能因采样不足。")
        elif consistency > 90:
            print("  分析: 一致性高，方法趋同。")
        else:
            print("  分析: 仍存在差异，考虑增大 K 或调整范围。")

def compare_three_methods(
    N_list = (50000,),
    K_values = (8,16,24,32,48,64,96,128),
    seeds = (0, 1, 2, 3),
    midpoint=True
):
    """
    精简输出版本：
      每行：seed N K  A(T/F)  S(T/F)  R(T/F)  same diff same_rate(%)
    """
    print("seed  N      K  A(T/F)        S(T/F)        R(T/F)        same   diff  same_rate(%)")
    for seed in seeds:
        for N in N_list:
            rng = np.random.default_rng(seed)
            # 生成数据
            missile = rng.uniform(-800, 800, (N, 3))
            smoke = rng.uniform(-400, 400, (N, 3))
            smoke[:,1] += 200
            smoke[:,2] += 5
            R_smoke = rng.uniform(15, 110, (N,1))
            data = np.hstack([missile, smoke, R_smoke])

            # 先算 rough 与 analytic
            rough = is_sphere_blocked_vectorized(data)
            analytic = is_cylinder_blocked_analytic(data, use_midpoint=midpoint)

            aT = int(analytic.sum()); aF = N - aT
            rT = int(rough.sum());   rF = N - rT

            for K in K_values:
                sample = is_cylinder_blocked_vectorized(data, K=K)
                sT = int(sample.sum()); sF = N - sT

                same_mask = (analytic == sample) & (analytic == rough)
                same = int(same_mask.sum())
                diff = N - same
                same_rate = same / N * 100.0

                print(f"{seed:<5}{N:<7}{K:<3} "
                      f"{aT}/{aF:<10} {sT}/{sF:<10} {rT}/{rF:<10} "
                      f"{same:<6}{diff:<6}{same_rate:>8.2f}")



if __name__ == "__main__":
    # 展示随 K 增大一致性趋势（注意 50000 样本 + 大 K 计算量较大）
    compare_methods(N=50000, K_values=[8,16,24,32,48,64,96,128])


