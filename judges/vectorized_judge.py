"""
Vectorized occlusion judgment for processing millions of circle-sphere pairs at once.

This module provides a highly optimized vectorized implementation that can process
large batches of occlusion queries simultaneously using NumPy's broadcasting.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class VectorizedOcclusionResult:
    """Results for batch occlusion queries."""
    occluded: np.ndarray  # (N,) boolean array
    f_min: np.ndarray     # (N,) minimum cosine values
    cos_alpha_s: np.ndarray  # (N,) cosine thresholds
    t_min: np.ndarray     # (N,) optimal t parameters
    valid: np.ndarray     # (N,) which cases were successfully evaluated


def _vectorized_poly_real_roots_desc(coeffs_batch: np.ndarray, max_roots: int = 8, tol: float = 1e-10) -> np.ndarray:
    """
    Vectorized polynomial real root finding.
    
    Args:
        coeffs_batch: (N, 5) array of quartic polynomial coefficients in descending order
        max_roots: Maximum number of roots to store per polynomial
        tol: Tolerance for considering a complex root as real
    
    Returns:
        roots_batch: (N, max_roots) array of real roots, padded with NaN
    """
    N = coeffs_batch.shape[0]
    roots_batch = np.full((N, max_roots), np.nan, dtype=np.float64)
    
    for i in range(N):
        coeffs = coeffs_batch[i]
        
        # Remove leading zeros
        first_nonzero = 0
        while first_nonzero < len(coeffs) and abs(coeffs[first_nonzero]) <= tol:
            first_nonzero += 1
            
        if first_nonzero == len(coeffs):
            # All zeros - degenerate case
            continue
            
        coeffs = coeffs[first_nonzero:]
        deg = len(coeffs) - 1
        
        if deg <= 0:
            # Constant polynomial - no roots
            continue
            
        try:
            roots = np.roots(coeffs)
            real_roots = []
            
            for z in roots:
                if abs(z.imag) <= tol:
                    real_roots.append(z.real)
                    
            if real_roots:
                real_roots = np.array(real_roots, dtype=np.float64)
                real_roots.sort()
                
                # Remove duplicates
                if len(real_roots) > 1:
                    dedup = [real_roots[0]]
                    for val in real_roots[1:]:
                        if abs(val - dedup[-1]) > 1e-8:
                            dedup.append(val)
                    real_roots = np.array(dedup)
                
                # Store up to max_roots
                n_store = min(len(real_roots), max_roots)
                roots_batch[i, :n_store] = real_roots[:n_store]
                
        except:
            # Root finding failed - leave as NaN
            continue
            
    return roots_batch


def _vectorized_f_of_t_constants(V: np.ndarray, C: np.ndarray, r: np.ndarray, S: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Vectorized computation of constants for f(t) = cos(theta(t)).
    
    Args:
        V: (N, 3) observer points
        C: (N, 3) circle centers (assumed C[:, 2] = 0)
        r: (N,) circle radii
        S: (N, 3) sphere centers
    
    Returns:
        Dictionary of vectorized constants A, B, D, E, F, G, u, etc.
    """
    # Coordinate transformation: translate to V as origin
    C_prime = C - V  # (N, 3)
    S_prime = S - V  # (N, 3)
    
    # Compute unit vector u = S_prime / ||S_prime||
    dS = np.linalg.norm(S_prime, axis=1)  # (N,)
    
    # Handle degenerate case where observer is at sphere center
    u = np.zeros_like(S_prime)
    valid_mask = dS > 0
    u[valid_mask] = S_prime[valid_mask] / dS[valid_mask, np.newaxis]
    u[~valid_mask] = np.array([0.0, 0.0, 1.0])  # Arbitrary direction
    
    # Extract coordinates
    X_c, Y_c, Z_c = C_prime[:, 0], C_prime[:, 1], C_prime[:, 2]  # (N,) each
    u_x, u_y, u_z = u[:, 0], u[:, 1], u[:, 2]  # (N,) each
    
    # Compute constants
    D = u_x * X_c + u_y * Y_c + u_z * Z_c  # (N,)
    A, B = u_x, u_y  # (N,) each
    E = (X_c * X_c + Y_c * Y_c + Z_c * Z_c) + r * r  # (N,)
    F, G = X_c, Y_c  # (N,) each
    
    return {
        'A': A, 'B': B, 'D': D, 'E': E, 'F': F, 'G': G,
        'u': u, 'C_prime': C_prime, 'S_prime': S_prime, 'dS': dS
    }


def _vectorized_quartic_coeffs_for_stationary(A: np.ndarray, B: np.ndarray, D: np.ndarray, 
                                            E: np.ndarray, F: np.ndarray, G: np.ndarray, 
                                            r: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of quartic polynomial coefficients.
    
    Returns:
        coeffs: (N, 5) array of coefficients [alpha4, alpha3, alpha2, alpha1, alpha0]
    """
    P1 = (-A * E + D * F)  # (N,)
    Q1 = (B * E - D * G)   # (N,)
    P2 = r * (-2.0 * A * G + B * F)  # (N,)
    Q2 = r * (2.0 * B * F - A * G)   # (N,)
    R = r * (-A * F + B * G)  # (N,)
    
    alpha4 = Q2 - Q1  # (N,)
    alpha3 = 2.0 * (P1 - R)  # (N,)
    alpha2 = 4.0 * P2 - 2.0 * Q2  # (N,)
    alpha1 = 2.0 * (P1 + R)  # (N,)
    alpha0 = Q1 + Q2  # (N,)
    
    return np.column_stack([alpha4, alpha3, alpha2, alpha1, alpha0])  # (N, 5)


def _vectorized_evaluate_f_from_z(z: np.ndarray, A: np.ndarray, B: np.ndarray, D: np.ndarray,
                                E: np.ndarray, F: np.ndarray, G: np.ndarray, r: np.ndarray,
                                eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of f(t) from z = tan(t/2).
    
    Args:
        z: (N, max_roots) array of z values
        Other args: (N,) arrays of constants
        
    Returns:
        f_vals: (N, max_roots) array of f values (NaN where invalid)
        t_vals: (N, max_roots) array of t values
    """
    N, max_roots = z.shape
    
    # Expand constants to match z shape
    A_exp = A[:, np.newaxis]  # (N, 1)
    B_exp = B[:, np.newaxis]  # (N, 1)
    D_exp = D[:, np.newaxis]  # (N, 1)
    E_exp = E[:, np.newaxis]  # (N, 1)
    F_exp = F[:, np.newaxis]  # (N, 1)
    G_exp = G[:, np.newaxis]  # (N, 1)
    r_exp = r[:, np.newaxis]  # (N, 1)
    
    den = 1.0 + z * z  # (N, max_roots)
    c = (1.0 - z * z) / den  # (N, max_roots)
    s = (2.0 * z) / den  # (N, max_roots)
    
    N_vals = D_exp + r_exp * (A_exp * c + B_exp * s)  # (N, max_roots)
    M_vals = E_exp + 2.0 * r_exp * (F_exp * c + G_exp * s)  # (N, max_roots)
    
    t_vals = 2.0 * np.arctan(z)  # (N, max_roots)
    
    # Only compute f where M > eps and z is not NaN
    valid_mask = (M_vals > eps) & np.isfinite(z)  # (N, max_roots)
    f_vals = np.full_like(z, np.nan)
    f_vals[valid_mask] = N_vals[valid_mask] / np.sqrt(M_vals[valid_mask])
    
    return f_vals, t_vals


def vectorized_circle_fmin_cos(V: np.ndarray, C: np.ndarray, r: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of f_min = min_t cos(theta(t)) for batch inputs.
    
    Args:
        V: (N, 3) observer points
        C: (N, 3) circle centers
        r: (N,) circle radii
        S: (N, 3) sphere centers
    
    Returns:
        f_min: (N,) minimum cosine values
        t_min: (N,) corresponding t parameters
    """
    N = V.shape[0]
    
    # Handle zero radius case
    zero_r_mask = np.abs(r) <= 1e-12
    
    # Compute constants
    consts = _vectorized_f_of_t_constants(V, C, r, S)
    A, B, D, E, F, G = consts['A'], consts['B'], consts['D'], consts['E'], consts['F'], consts['G']
    
    # For zero radius cases, f(t) is constant
    f_min = np.full(N, np.nan)
    t_min = np.full(N, np.nan)
    
    if np.any(zero_r_mask):
        # Evaluate at t=0 for zero radius cases
        den = 1.0 + r * r  # This is essentially E for r=0 case
        N_val = D
        M_val = E
        valid_zero_mask = zero_r_mask & (M_val > 1e-12)
        f_min[valid_zero_mask] = N_val[valid_zero_mask] / np.sqrt(M_val[valid_zero_mask])
        t_min[valid_zero_mask] = 0.0
    
    # Handle non-zero radius cases
    nonzero_r_mask = ~zero_r_mask
    if np.any(nonzero_r_mask):
        # Extract non-zero radius cases
        A_nz = A[nonzero_r_mask]
        B_nz = B[nonzero_r_mask]
        D_nz = D[nonzero_r_mask]
        E_nz = E[nonzero_r_mask]
        F_nz = F[nonzero_r_mask]
        G_nz = G[nonzero_r_mask]
        r_nz = r[nonzero_r_mask]
        
        # Compute quartic coefficients
        coeffs = _vectorized_quartic_coeffs_for_stationary(A_nz, B_nz, D_nz, E_nz, F_nz, G_nz, r_nz)
        
        # Find polynomial roots
        max_roots = 8  # Quartic can have at most 4 real roots, but we allow some extra
        roots = _vectorized_poly_real_roots_desc(coeffs, max_roots)
        
        # Evaluate f(t) at all candidate roots
        f_vals, t_vals = _vectorized_evaluate_f_from_z(roots, A_nz, B_nz, D_nz, E_nz, F_nz, G_nz, r_nz)
        
        # Find minimum f value for each case
        f_min_nz = np.full(len(A_nz), np.nan)
        t_min_nz = np.full(len(A_nz), np.nan)
        
        for i in range(len(A_nz)):
            valid_f = f_vals[i][np.isfinite(f_vals[i])]
            valid_t = t_vals[i][np.isfinite(f_vals[i])]
            
            if len(valid_f) > 0:
                min_idx = np.argmin(valid_f)
                f_min_nz[i] = valid_f[min_idx]
                t_min_nz[i] = valid_t[min_idx]
            else:
                # Fallback: evaluate at representative points
                for t_test in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
                    c, s = np.cos(t_test), np.sin(t_test)
                    N_test = D_nz[i] + r_nz[i] * (A_nz[i] * c + B_nz[i] * s)
                    M_test = E_nz[i] + 2.0 * r_nz[i] * (F_nz[i] * c + G_nz[i] * s)
                    if M_test > 1e-12:
                        f_test = N_test / np.sqrt(M_test)
                        if np.isnan(f_min_nz[i]) or f_test < f_min_nz[i]:
                            f_min_nz[i] = f_test
                            t_min_nz[i] = t_test
        
        # Store results back
        f_min[nonzero_r_mask] = f_min_nz
        t_min[nonzero_r_mask] = t_min_nz
    
    return f_min, t_min


def vectorized_circle_fully_occluded_by_sphere(V: np.ndarray, C: np.ndarray, r: np.ndarray, 
                                             S: np.ndarray, R: np.ndarray) -> VectorizedOcclusionResult:
    """
    Vectorized occlusion judgment for batch processing.
    
    Args:
        V: (N, 3) observer points
        C: (N, 3) circle centers (assumed z=0)
        r: (N,) circle radii
        S: (N, 3) sphere centers
        R: (N,) sphere radii
    
    Returns:
        VectorizedOcclusionResult containing batch results
    """
    N = V.shape[0]
    
    # Initialize result arrays
    occluded = np.full(N, False, dtype=bool)
    f_min = np.full(N, np.nan)
    cos_alpha_s = np.full(N, np.nan)
    t_min = np.full(N, np.nan)
    valid = np.full(N, False, dtype=bool)
    
    # Handle observer in/at sphere cases
    S_prime = S - V
    dS = np.linalg.norm(S_prime, axis=1)
    observer_in_sphere = (dS <= 1e-12) | (R >= dS)
    
    occluded[observer_in_sphere] = True
    valid[observer_in_sphere] = True
    
    # Process normal cases
    normal_mask = ~observer_in_sphere
    if np.any(normal_mask):
        # Extract normal cases
        V_norm = V[normal_mask]
        C_norm = C[normal_mask]
        r_norm = r[normal_mask]
        S_norm = S[normal_mask]
        R_norm = R[normal_mask]
        dS_norm = dS[normal_mask]
        
        # Compute f_min for normal cases
        f_min_norm, t_min_norm = vectorized_circle_fmin_cos(V_norm, C_norm, r_norm, S_norm)
        
        # Compute cos(alpha_s) = sqrt(1 - (R/dS)^2)
        ratio = R_norm / dS_norm
        cos_alpha_s_norm = np.sqrt(np.maximum(0.0, 1.0 - ratio * ratio))
        cos_alpha_s_norm[ratio >= 1.0] = 0.0
        
        # Determine occlusion
        valid_norm = np.isfinite(f_min_norm)
        occluded_norm = valid_norm & (f_min_norm >= cos_alpha_s_norm - 1e-12)
        
        # Store results
        f_min[normal_mask] = f_min_norm
        cos_alpha_s[normal_mask] = cos_alpha_s_norm
        t_min[normal_mask] = t_min_norm
        valid[normal_mask] = valid_norm
        occluded[normal_mask] = occluded_norm
    
    return VectorizedOcclusionResult(
        occluded=occluded,
        f_min=f_min,
        cos_alpha_s=cos_alpha_s,
        t_min=t_min,
        valid=valid
    )


class VectorizedOcclusionJudge:
    """
    Vectorized occlusion judge for batch processing millions of cases.
    """
    
    def __init__(self):
        pass
    
    def judge_batch(self, V: np.ndarray, C: np.ndarray, r: np.ndarray, 
                   S: np.ndarray, R: np.ndarray) -> VectorizedOcclusionResult:
        """
        Judge occlusion for a batch of circle-sphere pairs.
        
        Args:
            V: (N, 3) observer points
            C: (N, 3) circle centers
            r: (N,) circle radii
            S: (N, 3) sphere centers  
            R: (N,) sphere radii
            
        Returns:
            VectorizedOcclusionResult with batch results
        """
        # Ensure proper shapes and types
        V = np.asarray(V, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("V must be (N, 3) array")
        if C.shape != V.shape:
            raise ValueError("C must have same shape as V")
        if S.shape != V.shape:
            raise ValueError("S must have same shape as V")
        if r.shape != (V.shape[0],):
            raise ValueError("r must be (N,) array")
        if R.shape != (V.shape[0],):
            raise ValueError("R must be (N,) array")
            
        return vectorized_circle_fully_occluded_by_sphere(V, C, r, S, R)