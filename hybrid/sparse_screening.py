"""
Sparse dimension screening using cheap function evaluations.
Estimates which dimensions are most important for the objective function.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import warnings

@dataclass
class ScreeningResult:
    """Result of sparse dimension screening."""
    important_dims: List[int]  # Indices of important dimensions
    sensitivity_scores: np.ndarray  # Sensitivity score for each dimension
    support_set_size: int  # Number of dimensions deemed important
    cheap_evals_used: int  # Number of cheap evaluations consumed


class SobolSensitivityAnalysis:
    """Sobol sensitivity analysis for identifying important dimensions."""
    
    def __init__(self, bounds: List[Tuple[float, float]], cheap_eval_func):
        """
        Args:
            bounds: List of (min, max) for each dimension
            cheap_eval_func: Function that takes array of shape (n_samples, n_dims) 
                           and returns array of shape (n_samples,) with cheap evaluations
        """
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.cheap_eval_func = cheap_eval_func
        
    def generate_sobol_samples(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Sobol samples for first-order sensitivity analysis."""
        try:
            from scipy.stats import qmc
            sobol_gen = qmc.Sobol(d=self.n_dims, scramble=True)
            # Generate base samples
            base_samples = sobol_gen.random(n_samples)
            # Generate resampled matrix for each dimension
            resample_matrices = []
            for i in range(self.n_dims):
                resample = base_samples.copy()
                # Replace i-th column with new samples
                new_col = sobol_gen.random(n_samples)[:, i:i+1]
                resample[:, i:i+1] = new_col
                resample_matrices.append(resample)
            
            # Scale to bounds
            def scale_to_bounds(samples):
                return self.bounds[:, 0] + samples * (self.bounds[:, 1] - self.bounds[:, 0])
            
            base_scaled = scale_to_bounds(base_samples)
            resample_scaled = [scale_to_bounds(rs) for rs in resample_matrices]
            
            return base_scaled, resample_scaled
            
        except ImportError:
            # Fallback to Latin Hypercube if scipy not available
            warnings.warn("scipy not available, using Latin Hypercube sampling instead of Sobol")
            return self._latin_hypercube_fallback(n_samples)
    
    def _latin_hypercube_fallback(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback Latin Hypercube sampling."""
        np.random.seed(42)  # For reproducibility
        
        # Generate Latin Hypercube samples
        def lhs_samples(n, d):
            # Simple LHS implementation
            samples = np.zeros((n, d))
            for i in range(d):
                samples[:, i] = (np.random.permutation(n) + np.random.rand(n)) / n
            return samples
        
        base_samples = lhs_samples(n_samples, self.n_dims)
        resample_matrices = []
        
        for i in range(self.n_dims):
            resample = base_samples.copy()
            # Generate new column for dimension i
            new_col = np.random.rand(n_samples)
            resample[:, i] = new_col
            resample_matrices.append(resample)
        
        # Scale to bounds
        def scale_to_bounds(samples):
            return self.bounds[:, 0] + samples * (self.bounds[:, 1] - self.bounds[:, 0])
        
        base_scaled = scale_to_bounds(base_samples)
        resample_scaled = [scale_to_bounds(rs) for rs in resample_matrices]
        
        return base_scaled, resample_scaled
    
    def compute_first_order_indices(self, n_samples: int = 1000) -> np.ndarray:
        """Compute first-order Sobol indices for sensitivity analysis."""
        base_samples, resample_matrices = self.generate_sobol_samples(n_samples)
        # Batch evaluate (减少函数调用开销)
        stacked = np.vstack([base_samples] + resample_matrices)
        try:
            y_all = self.cheap_eval_func(stacked)
            if y_all.shape[0] != stacked.shape[0]:  # 保险
                raise ValueError('cheap_eval_func returned wrong size')
            y_base = y_all[:n_samples]
            y_resamples = []
            for i in range(self.n_dims):
                y_resamples.append(y_all[(i+1)*n_samples:(i+2)*n_samples])
        except Exception:
            # 回退逐块评估
            y_base = self.cheap_eval_func(base_samples)
            y_resamples = [self.cheap_eval_func(r) for r in resample_matrices]
        
        # Compute Sobol indices
        var_y = np.var(y_base, ddof=1)
        if var_y < 1e-12:
            # If variance is too small, return uniform sensitivity
            return np.ones(self.n_dims) / self.n_dims
        
        first_order_indices = np.zeros(self.n_dims)
        
        for i in range(self.n_dims):
            # First-order index: V_i / V
            v_i = np.mean(y_base * (y_resamples[i] - y_base))
            first_order_indices[i] = max(0, v_i / var_y)
        
        return first_order_indices


class MorrisScreening:
    """Morris screening method for identifying important dimensions."""
    
    def __init__(self, bounds: List[Tuple[float, float]], cheap_eval_func):
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.cheap_eval_func = cheap_eval_func
    
    def generate_morris_paths(self, n_trajectories: int = 50, levels: int = 10) -> List[np.ndarray]:
        """Generate Morris screening trajectories."""
        delta = levels / (2 * (levels - 1))
        
        trajectories = []
        for _ in range(n_trajectories):
            # Generate base point
            base = np.random.rand(self.n_dims)
            
            # Generate trajectory by perturbing one dimension at a time
            trajectory = [base.copy()]
            current = base.copy()
            
            # Random permutation of dimensions
            perm = np.random.permutation(self.n_dims)
            
            for dim in perm:
                # Choose direction (+delta or -delta)
                if current[dim] < 1 - delta:
                    current[dim] += delta
                else:
                    current[dim] -= delta
                trajectory.append(current.copy())
            
            # Scale to bounds
            trajectory_scaled = []
            for point in trajectory:
                scaled = self.bounds[:, 0] + point * (self.bounds[:, 1] - self.bounds[:, 0])
                trajectory_scaled.append(scaled)
            
            trajectories.append(np.array(trajectory_scaled))
        
        return trajectories
    
    def compute_elementary_effects(self, n_trajectories: int = 50, levels: int = 10) -> np.ndarray:
        """Compute Morris elementary effects for each dimension."""
        trajectories = self.generate_morris_paths(n_trajectories, levels)
        all_effects = [[] for _ in range(self.n_dims)]
        # 批量评估所有轨迹点以减少函数调用
        try:
            concat = np.vstack(trajectories)
            y_all = self.cheap_eval_func(concat)
            if y_all.shape[0] != concat.shape[0]:
                raise ValueError('size mismatch')
            offset = 0
            for trajectory in trajectories:
                L = trajectory.shape[0]
                y_values = y_all[offset:offset+L]
                offset += L
                for i in range(self.n_dims):
                    for step in range(L - 1):
                        diff = trajectory[step + 1] - trajectory[step]
                        if np.abs(diff[i]) > 1e-10:
                            delta_x = diff[i]
                            delta_y = y_values[step + 1] - y_values[step]
                            all_effects[i].append(delta_y / max(1e-12, delta_x))
                            break
        except Exception:
            for trajectory in trajectories:
                y_values = self.cheap_eval_func(trajectory)
                for i in range(self.n_dims):
                    for step in range(len(trajectory) - 1):
                        diff = trajectory[step + 1] - trajectory[step]
                        if np.abs(diff[i]) > 1e-10:
                            delta_x = diff[i]
                            delta_y = y_values[step + 1] - y_values[step]
                            all_effects[i].append(delta_y / max(1e-12, delta_x))
                            break
        
        # Compute statistics of elementary effects
        sensitivity_scores = np.zeros(self.n_dims)
        for i in range(self.n_dims):
            if all_effects[i]:
                # Use mean of absolute values as sensitivity measure
                sensitivity_scores[i] = np.mean(np.abs(all_effects[i]))
        
        return sensitivity_scores


def screen_dimensions(
    bounds: List[Tuple[float, float]], 
    cheap_eval_func,
    target_support_size: int = 8,
    max_cheap_evals: int = 2000,
    method: str = 'sobol',
    screening_budget_fraction: float = 0.7
) -> ScreeningResult:
    """
    Screen dimensions to identify the most important ones.
    
    Args:
        bounds: List of (min, max) for each dimension
        cheap_eval_func: Function that takes (n_samples, n_dims) array and returns (n_samples,) array
        target_support_size: Target number of important dimensions to identify
        max_cheap_evals: Maximum number of cheap evaluations to use
        method: 'sobol' for Sobol sensitivity analysis, 'morris' for Morris screening
        screening_budget_fraction: Fraction of budget to use for screening vs refinement
    
    Returns:
        ScreeningResult with identified important dimensions
    """
    n_dims = len(bounds)
    screening_budget = int(max_cheap_evals * screening_budget_fraction)
    
    if method == 'sobol':
        screener = SobolSensitivityAnalysis(bounds, cheap_eval_func)
        n_samples = min(screening_budget // 2, 1000)  # Sobol uses 2*n_samples evals
        sensitivity_scores = screener.compute_first_order_indices(n_samples)
        cheap_evals_used = n_samples * (n_dims + 1)
        
    elif method == 'morris':
        screener = MorrisScreening(bounds, cheap_eval_func)
        n_trajectories = min(screening_budget // (n_dims + 1), 50)
        sensitivity_scores = screener.compute_elementary_effects(n_trajectories)
        cheap_evals_used = n_trajectories * (n_dims + 1)
        
    else:
        raise ValueError(f"Unknown screening method: {method}")
    
    # Identify most important dimensions
    target_support_size = min(target_support_size, n_dims)
    important_indices = np.argsort(sensitivity_scores)[::-1][:target_support_size]
    
    # Filter out dimensions with very low sensitivity
    min_sensitivity_threshold = np.max(sensitivity_scores) * 0.1
    important_indices = [i for i in important_indices if sensitivity_scores[i] >= min_sensitivity_threshold]
    
    # Ensure at least 2 dimensions are selected
    if len(important_indices) < 2:
        important_indices = np.argsort(sensitivity_scores)[::-1][:max(2, target_support_size // 2)]
    
    return ScreeningResult(
        important_dims=list(important_indices),
        sensitivity_scores=sensitivity_scores,
        support_set_size=len(important_indices),
        cheap_evals_used=cheap_evals_used
    )