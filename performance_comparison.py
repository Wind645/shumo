"""Simplified performance comparison for occlusion methods.

Usage:
    python performance_comparison.py [--cases N] [--only-rough] [--repeats R]
"""

import sys
import time
import numpy as np
from typing import Dict, Any

from judges import (
    OcclusionJudge, VectorizedOcclusionJudge,
    RoughOcclusionJudge, RoughVectorizedOcclusionJudge
)

try:
    from judges import TorchRoughVectorizedOcclusionJudge
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def parse_args():
    """Simple argument parsing."""
    args = {"cases": 20000, "only_rough": False, "repeats": 1}
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--cases" and i + 1 < len(sys.argv):
            args["cases"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--only-rough":
            args["only_rough"] = True
            i += 1
        elif sys.argv[i] == "--repeats" and i + 1 < len(sys.argv):
            args["repeats"] = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    return args


def time_function(func, *args, repeats=1):
    """Time a function with multiple repeats."""
    times = []
    for _ in range(repeats):
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args)
        
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append(time.perf_counter() - start)
    
    return min(times), result


def generate_data(n):
    """Generate test data."""
    np.random.seed(42)
    
    viewpoints = np.random.uniform(-200, 200, (n, 3))
    circle_centers = np.random.uniform(-100, 100, (n, 3))
    circle_radii = np.random.uniform(1, 10, n)
    sphere_centers = np.random.uniform(-150, 150, (n, 3))
    sphere_radii = np.random.uniform(5, 30, n)
    
    return viewpoints, circle_centers, circle_radii, sphere_centers, sphere_radii


def run_comparison(cases=20000, only_rough=False, repeats=1):
    """Run performance comparison."""
    print(f"Performance comparison with {cases} cases (repeats: {repeats})")
    print("=" * 60)
    
    # Generate data
    V, C, r, S, R = generate_data(cases)
    
    results = {}
    
    if not only_rough:
        # Single judge
        sample_size = min(1000, cases)
        
        def single_test():
            count = 0
            for i in range(sample_size):
                judge = OcclusionJudge(V[i], C[i], r[i], S[i], R[i])
                result = judge.is_fully_occluded()
                if result.occluded:
                    count += 1
            return count
        
        elapsed, occluded_count = time_function(single_test, repeats=repeats)
        results["Single"] = {
            "time": elapsed,
            "rate": sample_size / elapsed,
            "occluded": occluded_count
        }
        
        # Vectorized judge
        vec_judge = VectorizedOcclusionJudge()
        
        def vectorized_test():
            return vec_judge.judge_batch(V, C, r, S, R)
        
        elapsed, result = time_function(vectorized_test, repeats=repeats)
        results["Vectorized"] = {
            "time": elapsed,
            "rate": cases / elapsed,
            "occluded": result["occluded"].sum()
        }
    
    # Rough judges
    rough_judge = RoughVectorizedOcclusionJudge()
    
    def rough_test():
        return rough_judge.judge_batch(V, C, r, S, R)
    
    elapsed, result = time_function(rough_test, repeats=repeats)
    results["Rough"] = {
        "time": elapsed,
        "rate": cases / elapsed,
        "occluded": result["occluded"].sum()
    }
    
    # Torch rough judge
    if HAS_TORCH:
        torch_judge = TorchRoughVectorizedOcclusionJudge()
        
        def torch_test():
            return torch_judge.judge_batch(V, C, r, S, R)
        
        elapsed, result = time_function(torch_test, repeats=repeats)
        results["Torch Rough"] = {
            "time": elapsed,
            "rate": cases / elapsed,
            "occluded": result["occluded"].sum()
        }
    
    # Print results
    for method, data in results.items():
        rate_k = data["rate"] / 1000
        print(f"{method:15}: {data['time']:8.4f}s  {rate_k:8.1f}K/s  {data['occluded']:6d} occluded")
    
    # Print speedups
    if len(results) > 1:
        print("\nSpeedups:")
        methods = list(results.keys())
        for i in range(1, len(methods)):
            base = results[methods[0]]["rate"]
            current = results[methods[i]]["rate"]
            speedup = current / base
            print(f"{methods[i]} vs {methods[0]}: {speedup:.1f}x")


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        run_comparison(
            cases=args["cases"],
            only_rough=args["only_rough"],
            repeats=args["repeats"]
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()