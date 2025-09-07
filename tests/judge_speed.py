
import sys
import time
import numpy as np

# Add the parent directory to the path to import from judges
sys.path.append('../')

from judges.jiexi import is_cylinder_blocked_analytic
from judges.batch_sample import is_cylinder_blocked_vectorized

def generate_test_data(N=1000):
    """
    Generate test data for speed comparison.
    N: number of test cases
    Returns: ndarray of shape (N, 7)
    """
    np.random.seed(42)  # For reproducibility
    missile_x = np.random.uniform(-1000, 1000, N)
    missile_y = np.random.uniform(-1000, 1000, N)
    missile_z = np.random.uniform(100, 1000, N)
    smoke_x = np.random.uniform(-500, 500, N)
    smoke_y = np.random.uniform(150, 250, N)
    smoke_z = np.random.uniform(0, 10, N)
    R_smoke = np.random.uniform(10, 100, N)

    data = np.column_stack([missile_x, missile_y, missile_z, smoke_x, smoke_y, smoke_z, R_smoke])
    return data

def measure_time(func, *args, **kwargs):
    """
    Measure the execution time of a function.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def main():
    print("Speed Comparison: jiexi.py vs batch_sample.py")
    print("=" * 50)

    # Generate test data
    N = 1000  # Number of test cases
    data = generate_test_data(N)
    print(f"Generated {N} test cases.")

    # Test jiexi.py (analytic method)
    print("\nTesting jiexi.py (analytic method)...")
    result_analytic, time_analytic = measure_time(is_cylinder_blocked_analytic, data)
    print(f"Time taken: {time_analytic:.4f} seconds")
    print(f"Results: {np.sum(result_analytic)} True out of {N}")

    # Test batch_sample.py with different K values
    K_values = [16, 32, 64, 128]
    for K in K_values:
        print(f"\nTesting batch_sample.py with K={K}...")
        result_sample, time_sample = measure_time(is_cylinder_blocked_vectorized, data, K=K)
        print(f"Time taken: {time_sample:.4f} seconds")
        print(f"Results: {np.sum(result_sample)} True out of {N}")

        # Compare times
        speedup = time_sample / time_analytic if time_analytic > 0 else float('inf')
        print(f"Speedup factor (analytic vs sample): {speedup:.2f}x (analytic is faster)")

        # Check if results are consistent (they might not be due to sampling)
        consistency = np.mean(result_analytic == result_sample)
        print(f"Result consistency: {consistency:.2%}")

if __name__ == "__main__":
    main()
