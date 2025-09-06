"""Performance comparison & benchmarking utilities.

默认模式 (无参数 / 兼容旧行为):
    - (可选) sampling 近似
    - original 精确算法 (逐个)
    - vectorized 精确 (NumPy)
    - rough 近似 (单例 / 向量化)
    - rough (torch 向量化, 若可用)

新增功能 (CLI 参数):
    --only-rough           仅比较 rough 相关方法 (便于放大 N 测试 GPU/CPU 拐点)
    --sizes N1,N2,...      多组样本规模；仅与 --only-rough 搭配 (否则忽略)
    --cases N              单一规模 (旧 CASENUM 的替代)
    --repeats R            正式计时重复次数 (默认 1)
    --warmup W             预热次数 (默认 1; 设 0 可关闭)
    --skip-rough-single    在 only-rough 大规模时跳过单例版本 (避免过慢)
    --no-sampling          跳过 sampling (旧行为保持)

计时改进:
    - 预热 (不计入统计)
    - 多次重复取平均/最小值
    - 若使用 GPU, 在计时前后 torch.cuda.synchronize()，避免异步影响

示例:
    uv run performance_comparison.py --only-rough --sizes 20000,200000,1000000 --repeats 3 --warmup 1
    uv run performance_comparison.py --cases 50000 --no-sampling
"""

import numpy as np
import time
from judge import OcclusionJudge
from vectorized_judge import VectorizedOcclusionJudge
from rough_judge import RoughOcclusionJudge, RoughVectorizedOcclusionJudge
try:  # Torch 可选
    from rough_judge_torch import TorchRoughVectorizedOcclusionJudge  # type: ignore
    import torch  # type: ignore
    TORCH_ROUGH_AVAILABLE = True
except Exception:  # pragma: no cover - 环境下无 torch
    TORCH_ROUGH_AVAILABLE = False
    torch = None  # 占位，避免类型引用报错
import sys
from typing import Callable, List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Configuration flag: 设为 True 时不运行 sampling 方法 (更快)
# 也可在命令行运行: uv run performance_comparison.py --no-sampling
# ---------------------------------------------------------------------------
SKIP_SAMPLING = True  # 保持旧脚本行为 (未提供 --no-sampling 时仍跳过)
CASENUM = 20000       # 默认单一规模 (可被 --cases 覆盖)


# --------------------------------------------------------------------------------------
# 计时辅助
# --------------------------------------------------------------------------------------
def _maybe_cuda_sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_callable(fn: Callable[[], Any], *, warmup: int = 1, repeats: int = 1,
                  label: str = "", cuda_sync: bool = True) -> Dict[str, Any]:
    """对可调用对象进行预热 + 多次重复计时。

    返回: dict(avg, best, repeats, warmup, last, label)
    """
    # 预热
    for _ in range(max(0, warmup)):
        if cuda_sync:
            _maybe_cuda_sync()
        fn()
        if cuda_sync:
            _maybe_cuda_sync()

    times: List[float] = []
    for _ in range(max(1, repeats)):
        if cuda_sync:
            _maybe_cuda_sync()
        t0 = time.perf_counter()
        fn()
        if cuda_sync:
            _maybe_cuda_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times, dtype=float)
    return dict(
        label=label,
        warmup=warmup,
        repeats=repeats,
        avg=float(arr.mean()),
        best=float(arr.min()),
        last=float(arr[-1]),
        all=times,
    )


def sampling_method(V, C, r, S, R, n_samples=1000):
    """
    Naive sampling method for occlusion detection.
    Sample points on the circle and check if any ray intersects the sphere.
    """
    # Generate sample points on circle
    t_samples = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    # Circle points
    circle_points = []
    for t in t_samples:
        x = C[0] + r * np.cos(t)
        y = C[1] + r * np.sin(t) 
        z = C[2]  # Should be 0
        circle_points.append([x, y, z])
    
    circle_points = np.array(circle_points)
    
    # Check each ray from V to circle points
    occluded_count = 0
    
    for point in circle_points:
        ray_dir = point - V
        ray_len = np.linalg.norm(ray_dir)
        ray_dir = ray_dir / ray_len
        
        # Check intersection with sphere
        # Ray: V + t * ray_dir, Sphere: ||p - S||^2 = R^2
        oc = V - S
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - R * R
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            # Ray intersects sphere, check if intersection is before circle point
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            
            # Check if intersection occurs before reaching circle point
            if (t1 > 0 and t1 < ray_len) or (t2 > 0 and t2 < ray_len):
                occluded_count += 1
    
    # Consider fully occluded if most rays are blocked
    return occluded_count >= n_samples * 0.95  # 95% threshold


def test_performance_comparison(skip_sampling: bool = False,
                                only_rough: bool = False,
                                N: int = CASENUM,
                                repeats: int = 1,
                                warmup: int = 1,
                                skip_rough_single: bool = False) -> Dict[str, Any]:
    """Compare performance of occlusion methods.

    Args:
        skip_sampling: 如果为 True, 不执行耗时的 sampling 方法。
    """
    print("=== Performance Comparison Test ===\n")
    if skip_sampling:
        print("[模式] 比较 Original / Vectorized / Rough / Rough(Vectorized) (跳过 Sampling)\n")
    else:
        print("[模式] 比较 Sampling / Original / Vectorized / Rough / Rough(Vectorized)\n")
    print(f"Generating {N} random test cases...")

    np.random.seed(42)
    
    # Random observers (all above z=0)
    V_batch = np.random.randn(N, 3)
    V_batch[:, 2] = np.abs(V_batch[:, 2]) + 0.5
    
    # Random circles (all at z=0)
    C_batch = np.random.randn(N, 3) * 2
    C_batch[:, 2] = 0.0
    r_batch = np.random.uniform(0.1, 1.0, N)
    
    # Random spheres (all above z=0)
    S_batch = np.random.randn(N, 3)
    S_batch[:, 2] = np.abs(S_batch[:, 2]) + 0.1
    R_batch = np.random.uniform(0.1, 1.5, N)
    
    results = {}
    
    if (not skip_sampling) and (not only_rough):
        # Test 1: Sampling Method
        print("1. Testing sampling method...")
        start_time = time.time()
        sampling_results = []
        for i in range(N):
            result = sampling_method(V_batch[i], C_batch[i], r_batch[i], S_batch[i], R_batch[i])
            sampling_results.append(result)
        sampling_time = time.time() - start_time
        results['sampling'] = {
            'time': sampling_time,
            'results': sampling_results
        }
        print(f"   Time: {sampling_time:.4f} seconds ({N/sampling_time:.1f} cases/sec)")
    
    # Test 2: Original Judge Method
    if not only_rough:
        label_original = "1." if skip_sampling else "2."
        print(f"{label_original} Testing original judge method...")
        t_res = time_callable(
            lambda: [OcclusionJudge(V_batch[i], C_batch[i], r_batch[i], S_batch[i], R_batch[i]).is_fully_occluded().occluded for i in range(N)],
            warmup=warmup,
            repeats=repeats,
            label="original",
            cuda_sync=False,
        )
        original_results = []
        # 只执行一次真正取结果 (避免重复 O(N^2) 影响) - 已在 time_callable 内重复执行
        for i in range(N):
            original_results.append(OcclusionJudge(V_batch[i], C_batch[i], r_batch[i], S_batch[i], R_batch[i]).is_fully_occluded().occluded)
        original_time = t_res['best']
        results['original'] = {
            'time': original_time,
            'avg_time': t_res['avg'],
            'results': original_results
        }
        print(f"   Best: {original_time:.4f}s  Avg: {t_res['avg']:.4f}s  ({N/original_time:.1f} cases/sec best)")
    
    # Test 3: Vectorized Method
    if not only_rough:
        label_vectorized = "2." if skip_sampling else "3."
        print(f"{label_vectorized} Testing vectorized method...")
        judge_vec = VectorizedOcclusionJudge()
        def _run_vec():
            return judge_vec.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)
        t_res = time_callable(_run_vec, warmup=warmup, repeats=repeats, label="vectorized", cuda_sync=False)
        vec_result = _run_vec()
        vectorized_results = vec_result.occluded.tolist()
        vectorized_time = t_res['best']
        results['vectorized'] = {
            'time': vectorized_time,
            'avg_time': t_res['avg'],
            'results': vectorized_results
        }
        print(f"   Best: {vectorized_time:.4f}s  Avg: {t_res['avg']:.4f}s  ({N/vectorized_time:.1f} cases/sec best)")

    # Test 4: Rough (single) method
    rough_single_results = []
    rough_single_time = None
    if (not skip_rough_single):
        label_rough_single = ("1." if only_rough else ("3." if skip_sampling else "4."))
        print(f"{label_rough_single} Testing rough (single) method...")
        t_res = time_callable(
            lambda: [RoughOcclusionJudge(V_batch[i], C_batch[i], r_batch[i], S_batch[i], R_batch[i]).is_fully_occluded().occluded for i in range(N)],
            warmup=warmup,
            repeats=repeats,
            label="rough_single",
            cuda_sync=False,
        )
        # 取一次真实结果集
        for i in range(N):
            rough_single_results.append(RoughOcclusionJudge(V_batch[i], C_batch[i], r_batch[i], S_batch[i], R_batch[i]).is_fully_occluded().occluded)
        rough_single_time = t_res['best']
        results['rough_single'] = {
            'time': rough_single_time,
            'avg_time': t_res['avg'],
            'results': rough_single_results
        }
        print(f"   Best: {rough_single_time:.4f}s  Avg: {t_res['avg']:.4f}s  ({N/(rough_single_time or 1):.1f} cases/sec best)")

    # Test 5: Rough Vectorized method
    label_rough_vec = ("1." if (only_rough and skip_rough_single) else ("2." if only_rough else ("4." if skip_sampling else "5.")))
    print(f"{label_rough_vec} Testing rough vectorized method...")
    rough_vec_judge = RoughVectorizedOcclusionJudge()
    def _run_rv():
        return rough_vec_judge.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)
    t_res = time_callable(_run_rv, warmup=warmup, repeats=repeats, label="rough_vectorized", cuda_sync=False)
    rough_vec_res = _run_rv()
    rough_vec_results = rough_vec_res.occluded.tolist()
    rough_vec_time = t_res['best']
    results['rough_vectorized'] = {
        'time': rough_vec_time,
        'avg_time': t_res['avg'],
        'results': rough_vec_results
    }
    print(f"   Best: {rough_vec_time:.4f}s  Avg: {t_res['avg']:.4f}s  ({N/rough_vec_time:.1f} cases/sec best)")

    # Test 6: Rough Torch Vectorized (if available)
    rough_torch_time = None
    rough_torch_time = None
    if TORCH_ROUGH_AVAILABLE:
        label_rough_torch = ("2." if (only_rough and skip_rough_single) else ("3." if only_rough else ("5." if skip_sampling else "6.")))
        print(f"{label_rough_torch} Testing rough torch vectorized method...")
        rough_torch_judge = TorchRoughVectorizedOcclusionJudge()
        def _run_rtv():
            return rough_torch_judge.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)
        t_res = time_callable(_run_rtv, warmup=warmup, repeats=repeats, label="rough_torch_vectorized", cuda_sync=True)
        rough_torch_res = _run_rtv()
        rough_torch_results = rough_torch_res.occluded.detach().cpu().numpy().tolist()
        rough_torch_time = t_res['best']
        results['rough_torch_vectorized'] = {
            'time': rough_torch_time,
            'avg_time': t_res['avg'],
            'results': rough_torch_results
        }
        print(f"   Best: {rough_torch_time:.4f}s  Avg: {t_res['avg']:.4f}s  ({N/rough_torch_time:.1f} cases/sec best)")
    
    # Performance Analysis
    print(f"\n=== Performance Summary ===")
    print(f"Test cases: {N}")
    if (not skip_sampling) and (not only_rough):
        print(f"Sampling method:       {sampling_time:.4f}s  ({N/sampling_time:6.1f} cases/sec)" )
    if not only_rough:
        print(f"Original method:       {original_time:.4f}s  ({N/(original_time):6.1f} cases/sec)")
        print(f"Vectorized (NumPy):    {vectorized_time:.4f}s  ({N/(vectorized_time):6.1f} cases/sec)")
    if rough_single_time is not None:
        print(f"Rough (single):        {rough_single_time:.4f}s  ({N/(rough_single_time):6.1f} cases/sec)")
    print(f"Rough (vectorized):    {rough_vec_time:.4f}s  ({N/(rough_vec_time):6.1f} cases/sec)")
    if rough_torch_time is not None:
        print(f"Rough (torch vec):     {rough_torch_time:.4f}s  ({N/(rough_torch_time):6.1f} cases/sec)")

    print(f"\nSpeedup factors:")
    if not only_rough:
        if (not skip_sampling):
            print(f"Vectorized vs Sampling:        {sampling_time/vectorized_time:6.1f}x")
            print(f"Original  vs Sampling:         {sampling_time/original_time:6.1f}x")
        print(f"Vectorized vs Original:        {original_time/vectorized_time:6.1f}x")
        if rough_single_time is not None:
            print(f"Rough(single) vs Original:     {original_time/(rough_single_time):6.1f}x")
        print(f"Rough(vectorized) vs Original: {original_time/(rough_vec_time):6.1f}x")
        print(f"Rough(vectorized) vs Vectorized:{vectorized_time/(rough_vec_time):6.1f}x")
        if rough_torch_time is not None:
            print(f"Rough(torch vec) vs Original: {original_time/(rough_torch_time):6.1f}x")
            print(f"Rough(torch vec) vs Vectorized:{vectorized_time/(rough_torch_time):6.1f}x")
            print(f"Rough(torch vec) vs Rough(vec):{rough_vec_time/(rough_torch_time):6.1f}x")
    else:
        # only rough 模式下的内部对比
        if rough_single_time is not None:
            print(f"Rough(vec) vs Rough(single):   {rough_single_time/(rough_vec_time):6.1f}x")
        if rough_torch_time is not None:
            print(f"Torch(vec) vs Rough(vec):      {rough_vec_time/(rough_torch_time):6.1f}x")
            if rough_single_time is not None:
                print(f"Torch(vec) vs Rough(single):   {rough_single_time/(rough_torch_time):6.1f}x")
    
    # Accuracy Analysis
    print(f"\n=== Accuracy Analysis ===")
    
    # Accuracy Analysis
    if not skip_sampling:
        sampling_correct = sum(1 for i in range(N) if sampling_results[i] == original_results[i])
        sampling_accuracy = sampling_correct / N
        print(f"Sampling vs Original accuracy:        {sampling_correct}/{N} ({sampling_accuracy:.1%})")
    if not only_rough:
        vec_correct = sum(1 for i in range(N) if vectorized_results[i] == original_results[i])
        vec_accuracy = vec_correct / N
        print(f"Vectorized(NumPy) vs Original:        {vec_correct}/{N} ({vec_accuracy:.1%})")
    if not only_rough and rough_single_time is not None:
        rough_single_correct = sum(1 for i in range(N) if rough_single_results[i] == original_results[i])
        rough_single_accuracy = rough_single_correct / N
        print(f"Rough(single) vs Original:           {rough_single_correct}/{N} ({rough_single_accuracy:.1%})")
    if not only_rough:
        rough_vec_correct = sum(1 for i in range(N) if rough_vec_results[i] == original_results[i])
        rough_vec_accuracy = rough_vec_correct / N
        print(f"Rough(vectorized) vs Original:       {rough_vec_correct}/{N} ({rough_vec_accuracy:.1%})")
        if 'rough_torch_vectorized' in results:
            rough_torch_correct = sum(1 for i in range(N) if results['rough_torch_vectorized']['results'][i] == original_results[i])
            rough_torch_accuracy = rough_torch_correct / N
            print(f"Rough(torch vectorized) vs Original: {rough_torch_correct}/{N} ({rough_torch_accuracy:.1%})")

    # Occlusion statistics
    if not only_rough:
        original_occluded = sum(original_results)
        vec_occluded = sum(vectorized_results)
        print(f"\nOcclusion counts:")
        if not skip_sampling:
            sampling_occluded = sum(sampling_results)
            print(f"Sampling method:       {sampling_occluded:3d}/{N} ({sampling_occluded/N:.1%})")
        print(f"Original method:       {original_occluded:3d}/{N} ({original_occluded/N:.1%})")
        print(f"Vectorized (NumPy):    {vec_occluded:3d}/{N} ({vec_occluded/N:.1%})")
        if rough_single_time is not None:
            rough_single_occluded = sum(rough_single_results)
            print(f"Rough (single):        {rough_single_occluded:3d}/{N} ({rough_single_occluded/N:.1%})")
        rough_vec_occluded = sum(rough_vec_results)
        print(f"Rough (vectorized):    {rough_vec_occluded:3d}/{N} ({rough_vec_occluded/N:.1%})")
        if 'rough_torch_vectorized' in results:
            rough_torch_occluded = sum(results['rough_torch_vectorized']['results'])
            print(f"Rough (torch vec):     {rough_torch_occluded:3d}/{N} ({rough_torch_occluded/N:.1%})")
    if not skip_sampling and sampling_accuracy < 1.0:
        print(f"\nSample disagreements (Sampling vs Original):")
        disagreement_count = 0
        for i in range(N):
            if sampling_results[i] != original_results[i] and disagreement_count < 3:
                print(f"  Case {i}: Sampling={sampling_results[i]}, Original={original_results[i]}")
                disagreement_count += 1
    
    return results


def _parse_sizes(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if part.lower().endswith('k'):
            out.append(int(float(part[:-1]) * 1000))
        elif part.lower().endswith('m'):
            out.append(int(float(part[:-1]) * 1_000_000))
        else:
            out.append(int(part))
    return out


def run_sizes_for_rough(sizes: List[int], repeats: int, warmup: int,
                        skip_rough_single: bool):
    print("=== Rough-only multi-size benchmark ===")
    header = ["N", "rough_vec_best(s)", "rough_vec_Mc/s",]
    if TORCH_ROUGH_AVAILABLE:
        header += ["torch_vec_best(s)", "torch_vec_Mc/s"]
    if not skip_rough_single:
        header[1:1] = ["rough_single_best(s)", "rough_single_kc/s"]
    print(" | ".join(header))
    print("-" * 80)
    for N in sizes:
        res = test_performance_comparison(skip_sampling=True, only_rough=True, N=N,
                                          repeats=repeats, warmup=warmup,
                                          skip_rough_single=skip_rough_single)
        row = [str(N)]
        if not skip_rough_single and 'rough_single' in res:
            t_rs = res['rough_single']['time']
            row += [f"{t_rs:.4f}", f"{(N/1000)/t_rs:,.1f}"]
        t_rv = res['rough_vectorized']['time']
        row += [f"{t_rv:.4f}", f"{(N/1_000_000)/t_rv:,.3f}"]
        if 'rough_torch_vectorized' in res:
            t_rt = res['rough_torch_vectorized']['time']
            row += [f"{t_rt:.4f}", f"{(N/1_000_000)/t_rt:,.3f}"]
        print(" | ".join(row))
    print("\n注: kc/s=千案例/秒, Mc/s=百万案例/秒 (best run).")


"""去除 argparse 的版本: 直接通过顶部常量配置运行性能比较。

编辑下方 CONFIG_* 常量后直接 `uv run performance_comparison.py` 即可。
"""

# ================== 可编辑常量区域 ==================
# 是否仅 Rough 系列测试
CONFIG_ONLY_ROUGH = False

# 是否跳过 sampling (True 推荐, 采样很慢且只是近似)
CONFIG_SKIP_SAMPLING_FORCE = True  # 若 True 则无视下面 skip_sampling 逻辑

# 单一规模 (若 CONFIG_SIZES 非空则忽略)
CONFIG_CASES: int | None = 50000

# 多规模列表字符串 (例如 "20k,200k,1m"), 仅当 CONFIG_ONLY_ROUGH=True 时生效
CONFIG_SIZES: str | None = None

# 预热与重复次数
CONFIG_WARMUP = 1
CONFIG_REPEATS = 1

# 是否跳过 rough 单例版本 (大规模测试提升速度)
CONFIG_SKIP_ROUGH_SINGLE = False
# ================== 可编辑常量区域 END ==============


def _run_from_constants():
    if CONFIG_SIZES and not CONFIG_ONLY_ROUGH:
        print("[警告] 只有在 ONLY_ROUGH 模式才使用多规模; 已忽略 CONFIG_SIZES。")
    if CONFIG_SIZES and CONFIG_ONLY_ROUGH:
        sizes = _parse_sizes(CONFIG_SIZES)
        run_sizes_for_rough(
            sizes,
            repeats=CONFIG_REPEATS,
            warmup=CONFIG_WARMUP,
            skip_rough_single=CONFIG_SKIP_ROUGH_SINGLE,
        )
        return

    N = CONFIG_CASES if CONFIG_CASES is not None else CASENUM
    arg_skip = CONFIG_SKIP_SAMPLING_FORCE or SKIP_SAMPLING
    res = test_performance_comparison(
        skip_sampling=arg_skip,
        only_rough=CONFIG_ONLY_ROUGH,
        N=N,
        repeats=CONFIG_REPEATS,
        warmup=CONFIG_WARMUP,
        skip_rough_single=CONFIG_SKIP_ROUGH_SINGLE,
    )
    print("\n🎉 Performance test completed!")
    if CONFIG_ONLY_ROUGH:
        print("• 已在 only-rough 模式下完成; 可设置 CONFIG_SIZES 做多规模扫描")
    else:
        if not arg_skip:
            print("• Vectorized 精确法 通常最快 (除非 rough vectorized 更快)")
            print("• Rough 系列给出保守判定: 不会误报, 可能漏报")
            print("• Sampling 最慢且仅近似")
        else:
            print("• 跳过 Sampling: 聚焦 精确 vs 近似 vs GPU")
        if TORCH_ROUGH_AVAILABLE:
            print("• 可以配置 ONLY_ROUGH + SIZES 进行大规模 GPU 拐点测试")


if __name__ == "__main__":
    _run_from_constants()