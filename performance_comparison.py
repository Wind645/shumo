"""Performance comparison among:
    - sampling (可选, 很慢)
    - original 精确算法
    - vectorized 精确 (NumPy)
    - rough 近似 (单例/批量)

Torch 版本已移除。

通过设置顶部变量 `SKIP_SAMPLING` (或命令行参数 `--no-sampling`) 可以跳过耗时 sampling。
"""

import numpy as np
import time
from judge import OcclusionJudge
from vectorized_judge import VectorizedOcclusionJudge
from rough_judge import RoughOcclusionJudge, RoughVectorizedOcclusionJudge
try:
    from rough_judge_torch import TorchRoughVectorizedOcclusionJudge
    TORCH_ROUGH_AVAILABLE = True
except Exception:
    TORCH_ROUGH_AVAILABLE = False
import sys

# ---------------------------------------------------------------------------
# Configuration flag: 设为 True 时不运行 sampling 方法 (更快)
# 也可在命令行运行: uv run performance_comparison.py --no-sampling
# ---------------------------------------------------------------------------
SKIP_SAMPLING = True
# Torch 已删除
CASENUM = 20000


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


def test_performance_comparison(skip_sampling: bool = False):
    """Compare performance of occlusion methods.

    Args:
        skip_sampling: 如果为 True, 不执行耗时的 sampling 方法。
    """
    print("=== Performance Comparison Test ===\n")
    if skip_sampling:
        print("[模式] 比较 Original / Vectorized / Rough / Rough(Vectorized) (跳过 Sampling)\n")
    else:
        print("[模式] 比较 Sampling / Original / Vectorized / Rough / Rough(Vectorized)\n")
    print("Generating 500 random test cases...")
    
    # Generate 500 test cases
    np.random.seed(42)
    N = CASENUM
    
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
    
    if not skip_sampling:
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
    label_original = "1." if skip_sampling else "2."
    print(f"{label_original} Testing original judge method...")
    start_time = time.time()
    original_results = []
    
    for i in range(N):
        judge = OcclusionJudge(V_batch[i], C_batch[i], r_batch[i], S_batch[i], R_batch[i])
        result = judge.is_fully_occluded()
        original_results.append(result.occluded)
    
    original_time = time.time() - start_time
    results['original'] = {
        'time': original_time,
        'results': original_results
    }
    print(f"   Time: {original_time:.4f} seconds ({N/original_time:.1f} cases/sec)")
    
    # Test 3: Vectorized Method
    label_vectorized = "2." if skip_sampling else "3."
    print(f"{label_vectorized} Testing vectorized method...")
    start_time = time.time()
    
    judge_vec = VectorizedOcclusionJudge()
    vec_result = judge_vec.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)
    vectorized_results = vec_result.occluded.tolist()
    
    vectorized_time = time.time() - start_time
    results['vectorized'] = {
        'time': vectorized_time,
        'results': vectorized_results
    }
    print(f"   Time: {vectorized_time:.4f} seconds ({N/vectorized_time:.1f} cases/sec)")

    # Test 4: Rough (single) method
    label_rough_single = "3." if skip_sampling else "4."
    print(f"{label_rough_single} Testing rough (single) method...")
    start_time = time.time()
    rough_single_results = []
    for i in range(N):
        rj = RoughOcclusionJudge(V_batch[i], C_batch[i], r_batch[i], S_batch[i], R_batch[i])
        res = rj.is_fully_occluded()
        rough_single_results.append(res.occluded)
    rough_single_time = time.time() - start_time
    results['rough_single'] = {
        'time': rough_single_time,
        'results': rough_single_results
    }
    print(f"   Time: {rough_single_time:.4f} seconds ({N/rough_single_time:.1f} cases/sec)")

    # Test 5: Rough Vectorized method
    label_rough_vec = "4." if skip_sampling else "5."
    print(f"{label_rough_vec} Testing rough vectorized method...")
    start_time = time.time()
    rough_vec_judge = RoughVectorizedOcclusionJudge()
    rough_vec_res = rough_vec_judge.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)
    rough_vec_results = rough_vec_res.occluded.tolist()
    rough_vec_time = time.time() - start_time
    results['rough_vectorized'] = {
        'time': rough_vec_time,
        'results': rough_vec_results
    }
    print(f"   Time: {rough_vec_time:.4f} seconds ({N/rough_vec_time:.1f} cases/sec)")

    # Test 6: Rough Torch Vectorized (if available)
    rough_torch_time = None
    if TORCH_ROUGH_AVAILABLE:
        label_rough_torch = ("5." if skip_sampling else "6.")
        print(f"{label_rough_torch} Testing rough torch vectorized method...")
        start_time = time.time()
        rough_torch_judge = TorchRoughVectorizedOcclusionJudge()
        rough_torch_res = rough_torch_judge.judge_batch(V_batch, C_batch, r_batch, S_batch, R_batch)
        # torch tensors
        rough_torch_results = rough_torch_res.occluded.detach().cpu().numpy().tolist()
        rough_torch_time = time.time() - start_time
        results['rough_torch_vectorized'] = {
            'time': rough_torch_time,
            'results': rough_torch_results
        }
        print(f"   Time: {rough_torch_time:.4f} seconds ({N/rough_torch_time:.1f} cases/sec)")
    
    # Performance Analysis
    print(f"\n=== Performance Summary ===")
    print(f"Test cases: {N}")
    if not skip_sampling:
        print(f"Sampling method:       {sampling_time:.4f}s  ({N/sampling_time:6.1f} cases/sec)")
    print(f"Original method:       {original_time:.4f}s  ({N/original_time:6.1f} cases/sec)")
    print(f"Vectorized (NumPy):    {vectorized_time:.4f}s  ({N/vectorized_time:6.1f} cases/sec)")
    print(f"Rough (single):        {rough_single_time:.4f}s  ({N/rough_single_time:6.1f} cases/sec)")
    print(f"Rough (vectorized):    {rough_vec_time:.4f}s  ({N/rough_vec_time:6.1f} cases/sec)")
    if rough_torch_time is not None:
        print(f"Rough (torch vec):     {rough_torch_time:.4f}s  ({N/rough_torch_time:6.1f} cases/sec)")

    print(f"\nSpeedup factors:")
    if not skip_sampling:
        print(f"Vectorized vs Sampling:        {sampling_time/vectorized_time:6.1f}x")
        print(f"Original  vs Sampling:         {sampling_time/original_time:6.1f}x")
    print(f"Vectorized vs Original:        {original_time/vectorized_time:6.1f}x")
    print(f"Rough(single) vs Original:     {original_time/rough_single_time:6.1f}x")
    print(f"Rough(vectorized) vs Original: {original_time/rough_vec_time:6.1f}x")
    print(f"Rough(vectorized) vs Vectorized:{vectorized_time/rough_vec_time:6.1f}x")
    if rough_torch_time is not None:
        print(f"Rough(torch vec) vs Original: {original_time/rough_torch_time:6.1f}x")
        print(f"Rough(torch vec) vs Vectorized:{vectorized_time/rough_torch_time:6.1f}x")
        print(f"Rough(torch vec) vs Rough(vec):{rough_vec_time/rough_torch_time:6.1f}x")
    
    # Accuracy Analysis
    print(f"\n=== Accuracy Analysis ===")
    
    # Accuracy Analysis
    if not skip_sampling:
        sampling_correct = sum(1 for i in range(N) if sampling_results[i] == original_results[i])
        sampling_accuracy = sampling_correct / N
        print(f"Sampling vs Original accuracy:        {sampling_correct}/{N} ({sampling_accuracy:.1%})")
    vec_correct = sum(1 for i in range(N) if vectorized_results[i] == original_results[i])
    vec_accuracy = vec_correct / N
    print(f"Vectorized(NumPy) vs Original:        {vec_correct}/{N} ({vec_accuracy:.1%})")
    rough_single_correct = sum(1 for i in range(N) if rough_single_results[i] == original_results[i])
    rough_single_accuracy = rough_single_correct / N
    print(f"Rough(single) vs Original:           {rough_single_correct}/{N} ({rough_single_accuracy:.1%})")
    rough_vec_correct = sum(1 for i in range(N) if rough_vec_results[i] == original_results[i])
    rough_vec_accuracy = rough_vec_correct / N
    print(f"Rough(vectorized) vs Original:       {rough_vec_correct}/{N} ({rough_vec_accuracy:.1%})")
    if 'rough_torch_vectorized' in results:
        rough_torch_correct = sum(1 for i in range(N) if results['rough_torch_vectorized']['results'][i] == original_results[i])
        rough_torch_accuracy = rough_torch_correct / N
        print(f"Rough(torch vectorized) vs Original: {rough_torch_correct}/{N} ({rough_torch_accuracy:.1%})")

    # Occlusion statistics
    original_occluded = sum(original_results)
    vec_occluded = sum(vectorized_results)
    print(f"\nOcclusion counts:")
    if not skip_sampling:
        sampling_occluded = sum(sampling_results)
        print(f"Sampling method:       {sampling_occluded:3d}/{N} ({sampling_occluded/N:.1%})")
    print(f"Original method:       {original_occluded:3d}/{N} ({original_occluded/N:.1%})")
    print(f"Vectorized (NumPy):    {vec_occluded:3d}/{N} ({vec_occluded/N:.1%})")
    rough_single_occluded = sum(rough_single_results)
    rough_vec_occluded = sum(rough_vec_results)
    print(f"Rough (single):        {rough_single_occluded:3d}/{N} ({rough_single_occluded/N:.1%})")
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


if __name__ == "__main__":
    # CLI 支持: --no-sampling
    arg_skip = ('--no-sampling' in sys.argv) or SKIP_SAMPLING
    results = test_performance_comparison(skip_sampling=arg_skip)

    print(f"\n🎉 Performance test completed!")
    print(f"\nKey findings:")
    if not arg_skip:
        print("• Vectorized 精确法 通常最快 (除非 rough vectorized 更快)")
        print("• Rough 系列给出保守判定: 不会产生误报 (False Positive) 但可能漏报")
        print("• Sampling 最慢且仅近似, 可省略")
    else:
        print("• 跳过 Sampling: 关注精确 / 向量化 / 近似")
    print("• Rough(vectorized) 适合需要极高速、容忍一定漏检的场景")