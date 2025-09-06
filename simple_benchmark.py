"""Simple performance comparison for occlusion judges."""

import time
import numpy as np
from judges import (
    OcclusionJudge, VectorizedOcclusionJudge, 
    RoughOcclusionJudge, RoughVectorizedOcclusionJudge
)

try:
    from judges import TorchRoughVectorizedOcclusionJudge
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def time_it(func, *args, repeats=3):
    """Simple timing utility."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def generate_test_data(n=1000):
    """Generate random test data."""
    np.random.seed(42)
    
    # Circle data
    circle_centers = np.random.randn(n, 3) * 100
    circle_radii = np.random.uniform(1, 10, n)
    
    # Sphere data  
    sphere_centers = np.random.randn(n, 3) * 50
    sphere_radii = np.random.uniform(5, 20, n)
    
    # Viewpoints
    viewpoints = np.random.randn(n, 3) * 200
    
    return circle_centers, circle_radii, sphere_centers, sphere_radii, viewpoints


def benchmark(n=10000):
    """Run simple benchmark."""
    print(f"Performance comparison with {n} test cases:")
    print("-" * 50)
    
    # Generate data
    cc, cr, sc, sr, vp = generate_test_data(n)
    
    # Test single precision judge  
    def test_single():
        results = []
        for i in range(min(100, n)):  # Only test first 100 for single
            judge = OcclusionJudge(vp[i], cc[i], cr[i], sc[i], sr[i])
            result = judge.is_fully_occluded()
            results.append(result.occluded)
        return results
    
    single_time = time_it(test_single)
    print(f"Single judge (100 cases): {single_time:.4f}s")
    
    # Test vectorized judge
    vec_judge = VectorizedOcclusionJudge()
    
    def test_vectorized():
        return vec_judge.judge_batch(vp, cc, cr, sc, sr)
    
    vec_time = time_it(test_vectorized)
    print(f"Vectorized judge ({n} cases): {vec_time:.4f}s")
    
    # Test rough judge
    rough_judge = RoughVectorizedOcclusionJudge()
    
    def test_rough():
        return rough_judge.judge_batch(vp, cc, cr, sc, sr)
    
    rough_time = time_it(test_rough)
    print(f"Rough judge ({n} cases): {rough_time:.4f}s")
    
    # Test torch rough judge if available
    if HAS_TORCH:
        torch_judge = TorchRoughVectorizedOcclusionJudge()
        
        def test_torch():
            return torch_judge.judge_batch(vp, cc, cr, sc, sr)
        
        torch_time = time_it(test_torch)
        print(f"Torch rough judge ({n} cases): {torch_time:.4f}s")
        
        # Performance ratios
        print("\nSpeedup ratios:")
        print(f"Vectorized vs Single: {single_time/vec_time*100:.1f}x")
        print(f"Rough vs Vectorized: {vec_time/rough_time:.1f}x") 
        print(f"Torch vs Rough: {rough_time/torch_time:.1f}x")
    else:
        print("PyTorch not available - skipping torch tests")
        print(f"\nSpeedup ratios:")
        print(f"Vectorized vs Single: {single_time/vec_time*100:.1f}x")
        print(f"Rough vs Vectorized: {vec_time/rough_time:.1f}x")


if __name__ == "__main__":
    import sys
    
    # Simple CLI
    n = 10000
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Usage: python simple_benchmark.py [num_cases]")
            sys.exit(1)
    
    benchmark(n)