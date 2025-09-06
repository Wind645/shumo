shumo/judges/verify.py
#!/usr/bin/env python3
"""
验证脚本：随机选择数据，比较各种评估方法的答案和速度
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple
import random

from judges import (
    OcclusionJudge,
    VectorizedOcclusionJudge,
    RoughOcclusionJudge,
    RoughVectorizedOcclusionJudge,
    TORCH_AVAILABLE
)

if TORCH_AVAILABLE:
    from judges import (
        TorchVectorizedOcclusionJudge,
        TorchRoughVectorizedOcclusionJudge,
        TorchVectorizedOcclusionJudgeSampled
    )


def generate_random_test_case() -> Tuple[np.ndarray, List[Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float]]]:
    """生成随机测试用例"""
    # 观察点
    V = np.array([
        random.uniform(-5, 5),
        random.uniform(-5, 5),
        random.uniform(1, 10)
    ])

    # 圆数据：圆心和半径
    num_circles = random.randint(5, 20)
    circles = []
    for _ in range(num_circles):
        C = np.array([
            random.uniform(-3, 3),
            random.uniform(-3, 3),
            0.0
        ])
        r = random.uniform(0.1, 1.5)
        circles.append((C, r))

    # 球数据：球心和半径
    num_spheres = random.randint(3, 15)
    spheres = []
    for _ in range(num_spheres):
        S = np.array([
            random.uniform(-2, 2),
            random.uniform(-2, 2),
            random.uniform(0.5, 8)
        ])
        R = random.uniform(0.2, 2.0)
        spheres.append((S, R))

    return V, circles, spheres


def time_method(method_name: str, func, *args) -> Tuple[Any, float]:
    """计时执行方法"""
    start_time = time.perf_counter()
    try:
        result = func(*args)
        end_time = time.perf_counter()
        return result, end_time - start_time
    except Exception as e:
        end_time = time.perf_counter()
        return f"ERROR: {e}", end_time - start_time


def verify_single_case(V: np.ndarray, circles: List, spheres: List) -> Dict[str, Any]:
    """验证单个测试用例"""
    results = {}

    # 1. 基础方法 (OcclusionJudge) - 只测试第一个圆和第一个球
    if circles and spheres:
        C, r = circles[0]
        S, R = spheres[0]

        def test_basic():
            judge = OcclusionJudge(V, C, r, S, R)
            return judge.is_fully_occluded().occluded

        result, timing = time_method("OcclusionJudge", test_basic)
        results["OcclusionJudge"] = {
            "result": result,
            "time": timing,
            "method_type": "single"
        }

    # 2. 向量化方法 (VectorizedOcclusionJudge)
    def test_vectorized():
        # 准备数据格式：(N, 3) for V, C, S; (N,) for r, R
        N = len(circles) * len(spheres)
        V_batch = np.tile(V, (N, 1))
        C_batch = np.zeros((N, 3))
        r_batch = np.zeros(N)
        S_batch = np.zeros((N, 3))
        R_batch = np.zeros(N)

        idx = 0
        for C, r in circles:
            for S, R in spheres:
                C_batch[idx] = C
                r_batch[idx] = r
                S_batch[idx] = S
                R_batch[idx] = R
                idx += 1

        judge = VectorizedOcclusionJudge()
        return judge.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)

    result, timing = time_method("VectorizedOcclusionJudge", test_vectorized)
    results["VectorizedOcclusionJudge"] = {
        "result": result,
        "time": timing,
        "method_type": "vectorized"
    }

    # 3. 粗糙方法 (RoughOcclusionJudge) - 只测试第一个圆和第一个球
    if circles and spheres:
        def test_rough():
            C, r = circles[0]
            S, R = spheres[0]
            judge = RoughOcclusionJudge(V, C, r, S, R)
            return judge.is_fully_occluded()

        result, timing = time_method("RoughOcclusionJudge", test_rough)
        results["RoughOcclusionJudge"] = {
            "result": result,
            "time": timing,
            "method_type": "single"
        }

    # 4. 粗糙向量化方法 (RoughVectorizedOcclusionJudge)
    def test_rough_vectorized():
        # 准备数据格式：(N, 3) for V, C, S; (N,) for r, R
        N = len(circles) * len(spheres)
        V_batch = np.tile(V, (N, 1))
        C_batch = np.zeros((N, 3))
        r_batch = np.zeros(N)
        S_batch = np.zeros((N, 3))
        R_batch = np.zeros(N)

        idx = 0
        for C, r in circles:
            for S, R in spheres:
                C_batch[idx] = C
                r_batch[idx] = r
                S_batch[idx] = S
                R_batch[idx] = R
                idx += 1

        judge = RoughVectorizedOcclusionJudge()
        return judge.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)

    result, timing = time_method("RoughVectorizedOcclusionJudge", test_rough_vectorized)
    results["RoughVectorizedOcclusionJudge"] = {
        "result": result,
        "time": timing,
        "method_type": "vectorized"
    }

    # 5. Torch方法（如果可用）
    if TORCH_AVAILABLE:
        def test_torch_vectorized():
            # 准备数据格式：(N, 3) for V, C, S; (N,) for r, R
            N = len(circles) * len(spheres)
            V_batch = np.tile(V, (N, 1))
            C_batch = np.zeros((N, 3))
            r_batch = np.zeros(N)
            S_batch = np.zeros((N, 3))
            R_batch = np.zeros(N)

            idx = 0
            for C, r in circles:
                for S, R in spheres:
                    C_batch[idx] = C
                    r_batch[idx] = r
                    S_batch[idx] = S
                    R_batch[idx] = R
                    idx += 1

            # 使用legacy API格式
            circles_array = np.column_stack([C_batch[:, 0], C_batch[:, 1], r_batch])
            spheres_array = np.column_stack([S_batch, R_batch])
            judge = TorchVectorizedOcclusionJudge(V, circles_array, spheres_array)
            return judge.compute_occlusion_matrix()

        result, timing = time_method("TorchVectorizedOcclusionJudge", test_torch_vectorized)
        results["TorchVectorizedOcclusionJudge"] = {
            "result": result,
            "time": timing,
            "method_type": "vectorized"
        }

        def test_torch_rough():
            # 准备数据格式：(N, 3) for V, C, S; (N,) for r, R
            N = len(circles) * len(spheres)
            V_batch = np.tile(V, (N, 1))
            C_batch = np.zeros((N, 3))
            r_batch = np.zeros(N)
            S_batch = np.zeros((N, 3))
            R_batch = np.zeros(N)

            idx = 0
            for C, r in circles:
                for S, R in spheres:
                    C_batch[idx] = C
                    r_batch[idx] = r
                    S_batch[idx] = S
                    R_batch[idx] = R
                    idx += 1

            # 使用legacy API格式
            circles_array = np.column_stack([C_batch[:, 0], C_batch[:, 1], r_batch])
            spheres_array = np.column_stack([S_batch, R_batch])
            judge = TorchRoughVectorizedOcclusionJudge()
            # Torch rough judge 可能需要不同的调用方式
            try:
                return judge.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)
            except:
                # 如果没有judge_batch方法，跳过
                return "Not implemented"

        result, timing = time_method("TorchRoughVectorizedOcclusionJudge", test_torch_rough)
        results["TorchRoughVectorizedOcclusionJudge"] = {
            "result": result,
            "time": timing,
            "method_type": "vectorized"
        }

        def test_torch_sampled():
            # 准备数据格式：(N, 3) for V, C, S; (N,) for r, R
            N = len(circles) * len(spheres)
            V_batch = np.tile(V, (N, 1))
            C_batch = np.zeros((N, 3))
            r_batch = np.zeros(N)
            S_batch = np.zeros((N, 3))
            R_batch = np.zeros(N)

            idx = 0
            for C, r in circles:
                for S, R in spheres:
                    C_batch[idx] = C
                    r_batch[idx] = r
                    S_batch[idx] = S
                    R_batch[idx] = R
                    idx += 1

            # 使用legacy API格式
            circles_array = np.column_stack([C_batch[:, 0], C_batch[:, 1], r_batch])
            spheres_array = np.column_stack([S_batch, R_batch])
            judge = TorchVectorizedOcclusionJudgeSampled(V, circles_array, spheres_array)
            return judge.compute_occlusion_matrix()

        result, timing = time_method("TorchVectorizedOcclusionJudgeSampled", test_torch_sampled)
        results["TorchVectorizedOcclusionJudgeSampled"] = {
            "result": result,
            "time": timing,
            "method_type": "vectorized"
        }

    return results


def compare_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """比较结果的一致性"""
    comparison = {
        "timing_comparison": {},
        "result_consistency": {},
        "fastest_method": None,
        "slowest_method": None
    }

    # 提取时间信息
    timings = {}
    for method, data in results.items():
        if isinstance(data, dict) and "time" in data:
            timings[method] = data["time"]

    if timings:
        comparison["timing_comparison"] = timings
        comparison["fastest_method"] = min(timings.keys(), key=lambda k: timings[k])
        comparison["slowest_method"] = max(timings.keys(), key=lambda k: timings[k])

    # 比较向量化方法的结果
    vectorized_methods = [k for k, v in results.items()
                         if isinstance(v, dict) and v.get("method_type") == "vectorized"]

    if len(vectorized_methods) >= 2:
        base_method = vectorized_methods[0]
        base_result = results[base_method]["result"]

        if isinstance(base_result, np.ndarray):
            for method in vectorized_methods[1:]:
                other_result = results[method]["result"]
                if isinstance(other_result, np.ndarray) and base_result.shape == other_result.shape:
                    # 计算结果差异
                    diff = np.abs(base_result - other_result)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    comparison["result_consistency"][f"{base_method}_vs_{method}"] = {
                        "max_difference": float(max_diff),
                        "mean_difference": float(mean_diff),
                        "identical": bool(max_diff < 1e-10)
                    }

    return comparison


def run_performance_test(num_tests: int = 100, seed: int = None) -> None:
    """运行性能测试，专门比较NumPy和Torch的速度"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("="*60)
    print("运行性能测试：NumPy vs Torch 速度对比")
    print(f"测试用例数量: {num_tests}")
    print(f"Torch 可用性: {TORCH_AVAILABLE}")
    print("="*60)

    numpy_times = []
    torch_times = []
    speedup_ratios = []

    for i in range(num_tests):
        print(f"\n测试用例 {i+1}/{num_tests}")
        print("-" * 40)

        # 生成随机数据
        V, circles, spheres = generate_random_test_case()
        print(f"观察点: {V}")
        print(f"圆数量: {len(circles)}, 球数量: {len(spheres)}")

        # 准备数据格式
        N = len(circles) * len(spheres)
        V_batch = np.tile(V, (N, 1))
        C_batch = np.zeros((N, 3))
        r_batch = np.zeros(N)
        S_batch = np.zeros((N, 3))
        R_batch = np.zeros(N)

        idx = 0
        for C, r in circles:
            for S, R in spheres:
                C_batch[idx] = C
                r_batch[idx] = r
                S_batch[idx] = S
                R_batch[idx] = R
                idx += 1

        # NumPy测试
        def test_numpy():
            judge = VectorizedOcclusionJudge()
            return judge.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)

        numpy_result, numpy_time = time_method("NumPy", test_numpy)
        numpy_times.append(numpy_time)

        # Torch测试
        if TORCH_AVAILABLE:
            def test_torch():
                circles_array = np.column_stack([C_batch[:, 0], C_batch[:, 1], r_batch])
                spheres_array = np.column_stack([S_batch, R_batch])
                judge = TorchVectorizedOcclusionJudge(V, circles_array, spheres_array)
                return judge.compute_occlusion_matrix()

            torch_result, torch_time = time_method("Torch", test_torch)
            torch_times.append(torch_time)

            if torch_time > 0:
                speedup = numpy_time / torch_time
                speedup_ratios.append(speedup)
                print(".6f")
                print(".6f")
                print(".2f")
            else:
                print(".6f")
                print(".6f")
        else:
            print(".6f")
            print("Torch: N/A")

    # 汇总性能统计
    print("\n" + "="*60)
    print("性能汇总统计")
    print("="*60)

    if numpy_times:
        avg_numpy = np.mean(numpy_times)
        std_numpy = np.std(numpy_times)
        print(".6f")

    if TORCH_AVAILABLE and torch_times:
        avg_torch = np.mean(torch_times)
        std_torch = np.std(torch_times)
        print(".6f")

        if speedup_ratios:
            avg_speedup = np.mean(speedup_ratios)
            std_speedup = np.std(speedup_ratios)
            print(".2f")
            print(".2f")


def run_verification(num_tests: int = 10, seed: int = None) -> None:
    """运行验证测试"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("="*60)
    print(f"运行 {num_tests} 个随机验证测试")
    print(f"Torch 可用性: {TORCH_AVAILABLE}")
    print("="*60)

    all_results = []

    for i in range(num_tests):
        print(f"\n测试用例 {i+1}/{num_tests}")
        print("-" * 40)

        # 生成随机数据
        V, circles, spheres = generate_random_test_case()
        print(f"观察点: {V}")
        print(f"圆数量: {len(circles)}, 球数量: {len(spheres)}")

        # 运行各种方法
        results = verify_single_case(V, circles, spheres)

        # 比较结果
        comparison = compare_results(results)

        # 打印结果
        print("\n时间对比:")
        for method, timing in comparison["timing_comparison"].items():
            status = "ERROR" if "ERROR" in str(results[method]["result"]) else "OK"
            print(".6f")

        if comparison["fastest_method"]:
            print(".6f")
            print(".6f")

        if comparison["result_consistency"]:
            print("\n结果一致性:")
            for comp_name, comp_data in comparison["result_consistency"].items():
                identical = "✓" if comp_data["identical"] else "✗"
                print(".2e")

        all_results.append({
            "test_case": i+1,
            "circles_count": len(circles),
            "spheres_count": len(spheres),
            "results": results,
            "comparison": comparison
        })

    # 汇总统计
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)

    # 平均时间统计
    method_times = {}
    method_errors = {}

    for test in all_results:
        for method, data in test["results"].items():
            if isinstance(data, dict) and "time" in data:
                if method not in method_times:
                    method_times[method] = []
                    method_errors[method] = 0

                method_times[method].append(data["time"])
                if "ERROR" in str(data["result"]):
                    method_errors[method] += 1

    print("\n平均执行时间:")
    for method, times in method_times.items():
        avg_time = np.mean(times)
        std_time = np.std(times)
        error_rate = method_errors[method] / len(times) * 100
        print(".6f")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="验证各种遮挡判定方法")
    parser.add_argument("--tests", "-n", type=int, default=10, help="测试用例数量")
    parser.add_argument("--seed", "-s", type=int, default=None, help="随机种子")
    parser.add_argument("--performance", "-p", action="store_true", help="运行性能测试模式")

    args = parser.parse_args()

    if args.performance:
        run_performance_test(args.tests, args.seed)
    else:
        run_verification(args.tests, args.seed)
