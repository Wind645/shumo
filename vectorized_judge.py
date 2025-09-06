"""Torch-accelerated vectorized occlusion judgment.

This version ports the original NumPy implementation to PyTorch so large batches
can leverage GPU acceleration. Public interface remains NumPy-friendly: inputs
may be NumPy arrays or PyTorch tensors; results are returned as NumPy arrays for
backward compatibility. Internally, computations run on ``device`` (CUDA if
available by default).

Design notes:
* Root finding implemented via companion matrices + ``torch.linalg.eig``.
* Works with heterogeneous polynomial degrees (after trimming leading zeros).
* Falls back gracefully when no valid roots (evaluates a few sample angles).
* Uses double precision for numerical stability (modeled after original code).
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import math
import numpy as np  # kept for I/O compatibility
import torch

Tensor = torch.Tensor


@dataclass
class VectorizedOcclusionResult:
    """Results for batch occlusion queries (NumPy arrays for compatibility)."""
    occluded: np.ndarray
    f_min: np.ndarray
    cos_alpha_s: np.ndarray
    t_min: np.ndarray
    valid: np.ndarray

    def to_torch(self, device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        """Return dict of tensors on selected device (convenience helper)."""
        dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return {k: torch.as_tensor(getattr(self, k), device=dev) for k in ['occluded','f_min','cos_alpha_s','t_min','valid']}


def _torch_poly_real_roots_desc(coeffs_batch: Tensor, max_roots: int = 8, tol: float = 1e-10) -> Tensor:
    """Find real roots of many low-degree polynomials (deg<=4) in descending order.

    Args:
        coeffs_batch: (N, 5) coefficients [a4,a3,a2,a1,a0] (some leading may be 0)
        max_roots: maximum roots stored per polynomial
        tol: tolerance for real / leading zero decisions

    Returns:
        (N,max_roots) tensor of real roots sorted ascending (NaN padded)
    """
    device = coeffs_batch.device
    dtype = coeffs_batch.dtype
    N = coeffs_batch.shape[0]
    roots_out = torch.full((N, max_roots), float('nan'), device=device, dtype=dtype)

    # Determine degree per polynomial by first non-zero
    abs_coeffs = coeffs_batch.abs()
    # mask where non-zero (tolerant)
    nonzero_mask = abs_coeffs > tol
    first_nz = torch.where(nonzero_mask.any(dim=1), nonzero_mask.float().argmax(dim=1), torch.full((N,), 5, device=device, dtype=torch.long))

    for deg in range(1, 5):  # degrees 1..4
        # Polynomials whose trimmed degree == deg
        idxs = torch.nonzero((first_nz < 5) & ((5 - first_nz - 1) == deg), as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            continue
        batch_coeffs = coeffs_batch[idxs]
        # Trim and normalize
        trimmed = []
        for i, row in enumerate(batch_coeffs):
            start = first_nz[idxs[i]].item()
            t = row[start:]
            trimmed.append(t / t[0])  # normalize leading to 1
        # Pad each to length deg+1 already trimmed length matches
        norm_coeffs = torch.stack(trimmed, dim=0)  # (B, deg+1)
        # Special cases for deg 1 & 2 with formulas (avoid eig overhead)
        if deg == 1:
            # a x + b => x = -b
            # norm_coeffs = [1, b]
            roots = -norm_coeffs[:, 1]
            roots = roots.unsqueeze(1)
        elif deg == 2:
            # x^2 + p x + q
            p = norm_coeffs[:, 1]
            q = norm_coeffs[:, 2]
            disc = p * p - 4 * q
            real_mask = disc >= -tol
            roots_list = []
            if real_mask.any():
                disc_clamped = torch.clamp(disc[real_mask], min=0)
                sqrt_disc = torch.sqrt(disc_clamped)
                r1 = (-p[real_mask] - sqrt_disc) / 2
                r2 = (-p[real_mask] + sqrt_disc) / 2
                two = torch.stack([r1, r2], dim=1)
                roots_tmp = torch.full((real_mask.shape[0], 2), float('nan'), device=device, dtype=dtype)
                roots_tmp[real_mask] = two
                roots_list.append(roots_tmp)
            else:
                roots_list.append(torch.full((real_mask.shape[0], 2), float('nan'), device=device, dtype=dtype))
            roots = roots_list[0]
        else:
            # Build companion matrices for deg 3 or 4
            # norm_coeffs: [1, b_{deg-1}, ..., b_0]
            c_desc = norm_coeffs[:, 1:]  # (B, deg)
            c_asc = torch.flip(c_desc, dims=[1])  # (B, deg) => c0..c_{deg-1}
            Bsize = c_asc.shape[0]
            C = torch.zeros((Bsize, deg, deg), device=device, dtype=dtype)
            # sub-diagonal ones
            if deg > 1:
                eye_sub = torch.eye(deg - 1, device=device, dtype=dtype)
                C[:, 1:, :-1] = eye_sub
            # last column = -c_j
            C[:, :, -1] = -c_asc
            # Eigenvalues
            eigvals = torch.linalg.eigvals(C)  # (B, deg)
            # Filter real roots
            real_mask = eigvals.imag.abs() <= tol
            real_vals = torch.where(real_mask, eigvals.real, torch.full_like(eigvals.real, float('nan')))
            # Collect real roots row-wise
            roots_list = []
            for i in range(Bsize):
                rv = real_vals[i]
                rvi = rv[torch.isfinite(rv)]
                if rvi.numel() == 0:
                    roots_list.append(torch.full((0,), float('nan'), device=device, dtype=dtype))
                else:
                    roots_list.append(torch.sort(rvi).values)
            # Pad to deg roots (upper bound) for uniform shape
            max_r = max((x.numel() for x in roots_list), default=0)
            if max_r == 0:
                roots = torch.full((Bsize, 0), float('nan'), device=device, dtype=dtype)
            else:
                tmp = torch.full((Bsize, max_r), float('nan'), device=device, dtype=dtype)
                for i, rts in enumerate(roots_list):
                    if rts.numel():
                        tmp[i, :rts.numel()] = rts
                roots = tmp
        # Deduplicate approximately & store
        for bi, global_i in enumerate(idxs.tolist()):
            rts = roots[bi]
            rts = rts[torch.isfinite(rts)]
            if rts.numel() == 0:
                continue
            # sort already sorted mostly; ensure ascending
            rts = torch.sort(rts).values
            # dedup
            if rts.numel() > 1:
                keep = [0]
                last = rts[0].item()
                for j in range(1, rts.numel()):
                    v = rts[j].item()
                    if abs(v - last) > 1e-8:
                        keep.append(j)
                        last = v
                rts = rts[keep]
            n_store = min(rts.numel(), max_roots)
            roots_out[global_i, :n_store] = rts[:n_store]

    return roots_out


def _vectorized_f_of_t_constants(V: Tensor, C: Tensor, r: Tensor, S: Tensor) -> Dict[str, Tensor]:
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
    dS = torch.linalg.norm(S_prime, dim=1)  # (N,)
    
    # Handle degenerate case where observer is at sphere center
    u = torch.zeros_like(S_prime)
    valid_mask = dS > 0
    if valid_mask.any():
        u[valid_mask] = S_prime[valid_mask] / dS[valid_mask].unsqueeze(1)
    if (~valid_mask).any():
        u[~valid_mask] = torch.tensor([0.0, 0.0, 1.0], dtype=V.dtype, device=V.device)
    
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


def _vectorized_quartic_coeffs_for_stationary(A: Tensor, B: Tensor, D: Tensor,
                                              E: Tensor, F: Tensor, G: Tensor,
                                              r: Tensor) -> Tensor:
    """
    Vectorized computation of quartic polynomial coefficients.
    
    Returns:
        coeffs: (N, 5) array of coefficients [alpha4, alpha3, alpha2, alpha1, alpha0]
    """
    P1 = (-A * E + D * F)
    Q1 = (B * E - D * G)
    P2 = r * (-2.0 * A * G + B * F)
    Q2 = r * (2.0 * B * F - A * G)
    Rv = r * (-A * F + B * G)

    alpha4 = Q2 - Q1
    alpha3 = 2.0 * (P1 - Rv)
    alpha2 = 4.0 * P2 - 2.0 * Q2
    alpha1 = 2.0 * (P1 + Rv)
    alpha0 = Q1 + Q2

    return torch.stack([alpha4, alpha3, alpha2, alpha1, alpha0], dim=1)


def _vectorized_evaluate_f_from_z(z: Tensor, A: Tensor, B: Tensor, D: Tensor,
                                  E: Tensor, F: Tensor, G: Tensor, r: Tensor,
                                  eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
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
    A_exp = A.unsqueeze(1)
    B_exp = B.unsqueeze(1)
    D_exp = D.unsqueeze(1)
    E_exp = E.unsqueeze(1)
    F_exp = F.unsqueeze(1)
    G_exp = G.unsqueeze(1)
    r_exp = r.unsqueeze(1)

    den = 1.0 + z * z
    c = (1.0 - z * z) / den
    s = (2.0 * z) / den

    N_vals = D_exp + r_exp * (A_exp * c + B_exp * s)
    M_vals = E_exp + 2.0 * r_exp * (F_exp * c + G_exp * s)

    t_vals = 2.0 * torch.atan(z)
    valid_mask = (M_vals > eps) & torch.isfinite(z)
    f_vals = torch.full_like(z, float('nan'))
    safe = torch.sqrt(M_vals[valid_mask])
    f_vals[valid_mask] = N_vals[valid_mask] / safe
    return f_vals, t_vals


def vectorized_circle_fmin_cos(V: Tensor, C: Tensor, r: Tensor, S: Tensor) -> Tuple[Tensor, Tensor]:
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
    zero_r_mask = r.abs() <= 1e-12
    consts = _vectorized_f_of_t_constants(V, C, r, S)
    A, B, D, E, F, G = consts['A'], consts['B'], consts['D'], consts['E'], consts['F'], consts['G']

    f_min = torch.full((N,), float('nan'), device=V.device, dtype=V.dtype)
    t_min = torch.full((N,), float('nan'), device=V.device, dtype=V.dtype)

    if zero_r_mask.any():
        N_val = D
        M_val = E
        valid_zero_mask = zero_r_mask & (M_val > 1e-12)
        if valid_zero_mask.any():
            f_min[valid_zero_mask] = N_val[valid_zero_mask] / torch.sqrt(M_val[valid_zero_mask])
            t_min[valid_zero_mask] = 0.0

    nonzero_r_mask = ~zero_r_mask
    if nonzero_r_mask.any():
        A_nz = A[nonzero_r_mask]
        B_nz = B[nonzero_r_mask]
        D_nz = D[nonzero_r_mask]
        E_nz = E[nonzero_r_mask]
        F_nz = F[nonzero_r_mask]
        G_nz = G[nonzero_r_mask]
        r_nz = r[nonzero_r_mask]

        coeffs = _vectorized_quartic_coeffs_for_stationary(A_nz, B_nz, D_nz, E_nz, F_nz, G_nz, r_nz)
        roots = _torch_poly_real_roots_desc(coeffs, max_roots=8)
        f_vals, t_vals = _vectorized_evaluate_f_from_z(roots, A_nz, B_nz, D_nz, E_nz, F_nz, G_nz, r_nz)

        valid_mask = torch.isfinite(f_vals)
        # Replace invalid with +inf so min ignores them
        tmp = f_vals.clone()
        tmp[~valid_mask] = torch.inf
        row_min_vals, row_min_idx = tmp.min(dim=1)
        # rows with all invalid -> row_min_vals == inf
        all_invalid = ~torch.isfinite(row_min_vals) | (row_min_vals == torch.inf)
        # assign min values
        f_min_nz = torch.full_like(row_min_vals, float('nan'))
        t_min_nz = torch.full_like(row_min_vals, float('nan'))
        valid_rows = ~all_invalid
        if valid_rows.any():
            f_min_nz[valid_rows] = row_min_vals[valid_rows]
            t_min_nz[valid_rows] = t_vals[valid_rows, row_min_idx[valid_rows]]
        # Fallback evaluations for invalid rows
        if all_invalid.any():
            fallback_ts = torch.tensor([0.0, math.pi/2, math.pi, 3*math.pi/2], device=V.device, dtype=V.dtype)
            for idx_local in torch.nonzero(all_invalid, as_tuple=False).squeeze(-1).tolist():
                best_val = None
                best_t = None
                for t_test in fallback_ts:
                    c_val = torch.cos(t_test)
                    s_val = torch.sin(t_test)
                    N_test = D_nz[idx_local] + r_nz[idx_local] * (A_nz[idx_local] * c_val + B_nz[idx_local] * s_val)
                    M_test = E_nz[idx_local] + 2.0 * r_nz[idx_local] * (F_nz[idx_local] * c_val + G_nz[idx_local] * s_val)
                    if M_test > 1e-12:
                        f_test = N_test / torch.sqrt(M_test)
                        if (best_val is None) or (f_test < best_val):
                            best_val = f_test
                            best_t = t_test
                if best_val is not None:
                    f_min_nz[idx_local] = best_val
                    t_min_nz[idx_local] = best_t

        f_min[nonzero_r_mask] = f_min_nz
        t_min[nonzero_r_mask] = t_min_nz

    return f_min, t_min


def vectorized_circle_fully_occluded_by_sphere(V: Tensor, C: Tensor, r: Tensor,
                                               S: Tensor, R: Tensor) -> VectorizedOcclusionResult:
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
    occluded = torch.zeros(N, dtype=torch.bool, device=V.device)
    f_min = torch.full((N,), float('nan'), dtype=V.dtype, device=V.device)
    cos_alpha_s = torch.full((N,), float('nan'), dtype=V.dtype, device=V.device)
    t_min = torch.full((N,), float('nan'), dtype=V.dtype, device=V.device)
    valid = torch.zeros(N, dtype=torch.bool, device=V.device)

    S_prime = S - V
    dS = torch.linalg.norm(S_prime, dim=1)
    observer_in_sphere = (dS <= 1e-12) | (R >= dS)
    if observer_in_sphere.any():
        occluded[observer_in_sphere] = True
        valid[observer_in_sphere] = True

    normal_mask = ~observer_in_sphere
    if normal_mask.any():
        V_norm = V[normal_mask]
        C_norm = C[normal_mask]
        r_norm = r[normal_mask]
        S_norm = S[normal_mask]
        R_norm = R[normal_mask]
        dS_norm = dS[normal_mask]

        f_min_norm, t_min_norm = vectorized_circle_fmin_cos(V_norm, C_norm, r_norm, S_norm)
        ratio = R_norm / dS_norm
        cos_alpha_s_norm = torch.sqrt(torch.clamp(1.0 - ratio * ratio, min=0.0))
        cos_alpha_s_norm[ratio >= 1.0] = 0.0
        valid_norm = torch.isfinite(f_min_norm)
        occluded_norm = valid_norm & (f_min_norm >= cos_alpha_s_norm - 1e-12)

        f_min[normal_mask] = f_min_norm
        cos_alpha_s[normal_mask] = cos_alpha_s_norm
        t_min[normal_mask] = t_min_norm
        valid[normal_mask] = valid_norm
        occluded[normal_mask] = occluded_norm

    # Convert back to numpy arrays on CPU for compatibility
    return VectorizedOcclusionResult(
        occluded=occluded.cpu().numpy(),
        f_min=f_min.cpu().numpy(),
        cos_alpha_s=cos_alpha_s.cpu().numpy(),
        t_min=t_min.cpu().numpy(),
        valid=valid.cpu().numpy(),
    )


class VectorizedOcclusionJudge:
    """Torch-based vectorized occlusion judge.

    Parameters:
        device: optional torch.device. Defaults to CUDA if available else CPU.
        dtype: floating dtype (default torch.float64 for numerical fidelity).
    """

    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

    def _to_tensor(self, x: Any) -> Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def judge_batch(self, V: Any, C: Any, r: Any, S: Any, R: Any,
                    return_torch: bool = False) -> VectorizedOcclusionResult | Dict[str, Tensor]:
        V_t = self._to_tensor(V)
        C_t = self._to_tensor(C)
        r_t = self._to_tensor(r)
        S_t = self._to_tensor(S)
        R_t = self._to_tensor(R)

        if V_t.ndim != 2 or V_t.shape[1] != 3:
            raise ValueError("V must be (N,3)")
        for arr, name in [(C_t, 'C'), (S_t, 'S')]:
            if arr.shape != V_t.shape:
                raise ValueError(f"{name} must match V shape")
        if r_t.shape != (V_t.shape[0],):
            raise ValueError("r must be (N,)")
        if R_t.shape != (V_t.shape[0],):
            raise ValueError("R must be (N,)")

        result = vectorized_circle_fully_occluded_by_sphere(V_t, C_t, r_t, S_t, R_t)
        if return_torch:
            return result.to_torch(self.device)
        return result