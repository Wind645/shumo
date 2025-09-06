"""Torch-accelerated version of `vectorized_judge`.

This module mirrors the semantics of `vectorized_judge.py` but performs the
bulk of arithmetic in PyTorch so it can leverage GPU acceleration. The only
part that still relies on NumPy is the quartic root finding (per candidate),
since PyTorch does not provide a native polynomial root solver. Coefficients
are transferred to CPU for root solving and the real roots are brought back
to Torch for subsequent vectorized evaluation.

Intended usage: batch geometric occlusion queries (caps fully occluded by a
smoke sphere) inside optimization loops where GPU residency of state reduces
overhead relative to the pure NumPy version.

NOTE: Because root solving remains on CPU, for very large batch sizes where
root finding dominates runtime, speedup may be limited. Future improvement
could implement an analytic quartic solver in Torch or approximate f_min via
sampling to stay purely on GPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import numpy as np

# Reuse the reliable NumPy root finder from existing module
from vectorized_judge import _vectorized_poly_real_roots_desc as _np_roots_batch

# ---------------------------------------------------------------------------
# Optional: GPU batch quartic real root extraction via companion matrix eigen
# decomposition. This avoids CPU <-> GPU transfers for large batches when the
# polynomial degree is truly 4 with non-zero leading coefficient.
# ---------------------------------------------------------------------------
def _batch_real_roots_quartic_torch(coeffs: torch.Tensor, *, tol: float = 1e-12, max_roots: int = 8) -> torch.Tensor:
    """Vectorized real root extraction for quartics using companion matrix eigenvalues.

    Eliminates per-row Python loops: eigen-decomposition is batched and real roots
    are filtered & sorted with tensor ops. Degenerate (|a4|<=tol) rows fall back to the
    existing NumPy routine (still batched) because they may represent lower degree polys.
    """
    device = coeffs.device
    coeffs64 = coeffs.to(torch.float64)
    N = coeffs64.shape[0]
    roots_out = torch.full((N, max_roots), float('nan'), device=device, dtype=torch.float64)
    a4 = coeffs64[:, 0]
    gpu_mask = torch.abs(a4) > tol
    # Fallback (possibly lower degree)
    if (~gpu_mask).any():
        coeffs_np = coeffs64[~gpu_mask].detach().cpu().numpy()
        roots_np = _np_roots_batch(coeffs_np, max_roots=max_roots)
        roots_out[~gpu_mask] = torch.from_numpy(roots_np).to(device)
    if gpu_mask.any():
        c = coeffs64[gpu_mask]
        b = c[:, 1:] / c[:, 0].unsqueeze(1)  # monic normalization
        b3, b2, b1, b0 = b.unbind(1)
        K = c.shape[0]
        M = torch.zeros((K, 4, 4), device=device, dtype=torch.complex128)
        M[:, 1, 0] = 1.0; M[:, 2, 1] = 1.0; M[:, 3, 2] = 1.0
        M[:, 0, 3] = -b0; M[:, 1, 3] = -b1; M[:, 2, 3] = -b2; M[:, 3, 3] = -b3
        eigvals = torch.linalg.eigvals(M)  # (K,4)
        # Keep real eigenvalues
        real_parts = eigvals.real
        imag_abs = eigvals.imag.abs()
        real_parts = torch.where(imag_abs <= 1e-9, real_parts, torch.full_like(real_parts, float('nan')))
        # Sort real roots per row (NaNs pushed to end by replacing temporarily)
        tmp = torch.nan_to_num(real_parts, nan=1e9)
        tmp_sorted, _ = torch.sort(tmp, dim=1)
        roots_sorted = torch.where(tmp_sorted > 1e8, torch.full_like(tmp_sorted, float('nan')), tmp_sorted)
        # Deduplicate approximately by masking near-equal consecutive values
        if roots_sorted.shape[1] > 1:
            diff = torch.abs(roots_sorted[:, 1:] - roots_sorted[:, :-1])
            dup_mask = diff <= 1e-8
            # Shift mask to align with positions to drop; replace duplicates with NaN
            dup_positions = torch.cat([torch.zeros((K,1), dtype=torch.bool, device=device), dup_mask], dim=1)
            roots_sorted = torch.where(dup_positions, torch.full_like(roots_sorted, float('nan')), roots_sorted)
        # Re-pack first max_roots non-NaNs per row
        if max_roots <= roots_sorted.shape[1]:
            # Simple: just take first max_roots columns (already sorted) – acceptable because NaNs at end.
            sel_roots = roots_sorted[:, :max_roots]
        else:
            pad = torch.full((K, max_roots - roots_sorted.shape[1]), float('nan'), device=device)
            sel_roots = torch.cat([roots_sorted, pad], dim=1)
        roots_out[gpu_mask] = sel_roots
    return roots_out


@dataclass
class VectorizedOcclusionResultTorch:
    occluded: torch.Tensor   # (N,) bool
    f_min: torch.Tensor      # (N,) float (cosine minima)
    cos_alpha_s: torch.Tensor  # (N,) float
    t_min: torch.Tensor      # (N,) float (parameter t where min occurs)
    valid: torch.Tensor      # (N,) bool (successful evaluation)


def _f_of_t_constants_torch(V: torch.Tensor, C: torch.Tensor, r: torch.Tensor, S: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute constants for f(t)=cos(theta(t)) analogously to numpy version.

    Args:
        V, C, S: (N,3) tensors
        r: (N,) radii
    Returns dict of tensors (all (N,) except vectors) on same device/dtype.
    """
    C_prime = C - V
    S_prime = S - V
    dS = torch.linalg.norm(S_prime, dim=1)
    # unit vectors u handling degenerate zeros
    u = torch.zeros_like(S_prime)
    mask = dS > 0
    if mask.any():
        u[mask] = S_prime[mask] / dS[mask].unsqueeze(1)
    # arbitrary direction when degenerate
    if (~mask).any():
        u[~mask] = torch.tensor([0.0, 0.0, 1.0], device=V.device, dtype=V.dtype)
    X_c, Y_c, Z_c = C_prime[:, 0], C_prime[:, 1], C_prime[:, 2]
    u_x, u_y, u_z = u[:, 0], u[:, 1], u[:, 2]
    D = u_x * X_c + u_y * Y_c + u_z * Z_c
    A = u_x
    B = u_y
    E = (X_c * X_c + Y_c * Y_c + Z_c * Z_c) + r * r
    F = X_c
    G = Y_c
    return {
        'A': A, 'B': B, 'D': D, 'E': E, 'F': F, 'G': G,
        'u': u, 'C_prime': C_prime, 'S_prime': S_prime, 'dS': dS
    }


def _quartic_coeffs_torch(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, E: torch.Tensor, F: torch.Tensor, G: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
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


def _evaluate_f_from_z_torch(z: torch.Tensor, A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, E: torch.Tensor, F: torch.Tensor, G: torch.Tensor, r: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    # z: (N,R)
    den = 1.0 + z * z
    c = (1.0 - z * z) / den
    s = (2.0 * z) / den
    # expand dims
    A_e = A.unsqueeze(1); B_e = B.unsqueeze(1); D_e = D.unsqueeze(1)
    E_e = E.unsqueeze(1); F_e = F.unsqueeze(1); G_e = G.unsqueeze(1); r_e = r.unsqueeze(1)
    N_vals = D_e + r_e * (A_e * c + B_e * s)
    M_vals = E_e + 2.0 * r_e * (F_e * c + G_e * s)
    t_vals = 2.0 * torch.atan(z)
    valid_mask = (M_vals > eps) & torch.isfinite(z)
    f_vals = torch.full_like(z, float('nan'))
    if valid_mask.any():
        f_vals[valid_mask] = N_vals[valid_mask] / torch.sqrt(M_vals[valid_mask])
    return f_vals, t_vals


def vectorized_circle_fmin_cos_torch(V: torch.Tensor, C: torch.Tensor, r: torch.Tensor, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch analog optimized: removes Python loops for per-row minima selection.

    Strategy: compute roots batch, evaluate f values, use nan-aware reduction.
    Fallback sampling only for rows with no finite root evaluations.
    """
    device = V.device; dtype = V.dtype
    N = V.shape[0]
    zero_r_mask = torch.abs(r) <= 1e-12
    consts = _f_of_t_constants_torch(V, C, r, S)
    A = consts['A']; B = consts['B']; D = consts['D']; E = consts['E']; F = consts['F']; G = consts['G']
    f_min = torch.full((N,), float('nan'), device=device, dtype=dtype)
    t_min = torch.full((N,), float('nan'), device=device, dtype=dtype)
    # Zero radius direct evaluation
    if zero_r_mask.any():
        M0 = E[zero_r_mask]
        Nz = D[zero_r_mask]
        mvalid = M0 > 1e-12
        if mvalid.any():
            vals = Nz[mvalid] / torch.sqrt(M0[mvalid])
            zr_idx = zero_r_mask.nonzero().squeeze(1)[mvalid]
            f_min[zr_idx] = vals
            t_min[zr_idx] = 0.0
    nonzero_mask = ~zero_r_mask
    if nonzero_mask.any():
        idx = nonzero_mask.nonzero().squeeze(1)
        A_n = A[idx]; B_n = B[idx]; D_n = D[idx]; E_n = E[idx]; F_n = F[idx]; G_n = G[idx]; r_n = r[idx]
        coeffs = _quartic_coeffs_torch(A_n, B_n, D_n, E_n, F_n, G_n, r_n)
        # Root finding
        roots = _batch_real_roots_quartic_torch(coeffs, max_roots=8) if V.is_cuda else torch.from_numpy(_np_roots_batch(coeffs.detach().cpu().numpy(), max_roots=8)).to(device, dtype)
        f_vals, t_vals = _evaluate_f_from_z_torch(roots, A_n, B_n, D_n, E_n, F_n, G_n, r_n)
        valid_mask = torch.isfinite(f_vals)
        any_valid = valid_mask.any(dim=1)
        # For valid rows compute nanmin
        if any_valid.any():
            fv = f_vals.clone()
            fv[~valid_mask] = torch.inf
            min_vals, min_pos = torch.min(fv, dim=1)
            # mark rows that had NO valid -> min_vals will be inf; filter
            good_rows = (min_vals < torch.inf) & any_valid
            if good_rows.any():
                tgt = idx[good_rows]
                f_min[tgt] = min_vals[good_rows]
                # gather corresponding t
                pos = min_pos[good_rows]
                t_sel = t_vals[good_rows, :].gather(1, pos.unsqueeze(1)).squeeze(1)
                t_min[tgt] = t_sel
        # Fallback sampling for rows with no valid root
        fallback_rows = idx[~any_valid]
        if fallback_rows.numel():
            tfixed = torch.tensor([0.0, torch.pi/2, torch.pi, 3*torch.pi/2], device=device, dtype=dtype)
            for j, row in enumerate(fallback_rows):
                A_j, B_j, D_j, E_j, F_j, G_j, r_j = A[row], B[row], D[row], E[row], F[row], G[row], r[row]
                best_v = torch.tensor(float('inf'), device=device, dtype=dtype)
                best_t = torch.tensor(float('nan'), device=device, dtype=dtype)
                for t_test in tfixed:
                    c = torch.cos(t_test); s = torch.sin(t_test)
                    N_test = D_j + r_j*(A_j*c + B_j*s)
                    M_test = E_j + 2.0*r_j*(F_j*c + G_j*s)
                    if M_test > 1e-12:
                        f_test = N_test/torch.sqrt(M_test)
                        if f_test < best_v:
                            best_v = f_test; best_t = t_test
                if best_v.isfinite():
                    f_min[row] = best_v; t_min[row] = best_t
    return f_min, t_min


def vectorized_circle_fully_occluded_by_sphere_torch(V: torch.Tensor, C: torch.Tensor, r: torch.Tensor, S: torch.Tensor, R: torch.Tensor) -> VectorizedOcclusionResultTorch:
    """Torch batch occlusion judgment.
    Inputs: V,C,S (N,3); r,R (N,) all same device/dtype (float64 recommended).
    Returns boolean occlusion for each pair plus auxiliary data.
    """
    N = V.shape[0]; device=V.device; dtype=V.dtype
    occluded = torch.zeros(N, dtype=torch.bool, device=device)
    f_min = torch.full((N,), float('nan'), dtype=dtype, device=device)
    cos_alpha_s = torch.full((N,), float('nan'), dtype=dtype, device=device)
    t_min = torch.full((N,), float('nan'), dtype=dtype, device=device)
    valid = torch.zeros(N, dtype=torch.bool, device=device)
    S_prime = S - V
    dS = torch.linalg.norm(S_prime, dim=1)
    observer_in_sphere = (dS <= 1e-12) | (R >= dS)
    if observer_in_sphere.any():
        occluded[observer_in_sphere] = True
        valid[observer_in_sphere] = True
    normal_mask = ~observer_in_sphere
    if normal_mask.any():
        idx = normal_mask.nonzero().squeeze(1)
        f_min_n, t_min_n = vectorized_circle_fmin_cos_torch(V[idx], C[idx], r[idx], S[idx])
        dS_norm = dS[idx]
        ratio = R[idx] / dS_norm
        cos_alpha_s_norm = torch.sqrt(torch.clamp(1.0 - ratio*ratio, min=0.0))
        cos_alpha_s_norm[ratio >= 1.0] = 0.0
        valid_norm = torch.isfinite(f_min_n)
        occluded_norm = valid_norm & (f_min_n >= cos_alpha_s_norm - 1e-12)
        f_min[idx] = f_min_n
        cos_alpha_s[idx] = cos_alpha_s_norm
        t_min[idx] = t_min_n
        valid[idx] = valid_norm
        occluded[idx] = occluded_norm
    return VectorizedOcclusionResultTorch(occluded=occluded, f_min=f_min, cos_alpha_s=cos_alpha_s, t_min=t_min, valid=valid)


def batch_occluded_time_caps_torch(params: torch.Tensor, *, dt: float, device: torch.device) -> torch.Tensor:
    """Torch implementation of cap-based occluded time (judge_caps) mirroring the NumPy exact version.

    Args:
        params: (N,4) speed, azimuth, release_time, explode_delay
        dt: time step
        device: torch device
    Returns:
        (N,) tensor of occluded times (seconds)
    """
    if params.numel()==0:
        return torch.zeros(0, device=device, dtype=torch.float64)
    p = params.to(device=device, dtype=torch.float64)
    speed, az, t_rel, dly = p[:,0], p[:,1], p[:,2], p[:,3]
    N = p.shape[0]
    valid = (speed>=70.0) & (speed<=140.0) & (t_rel>=0.0) & (dly>0.0)
    occluded_time = torch.zeros(N, device=device, dtype=torch.float64)
    if not valid.any():
        return occluded_time
    # Constants (match ga_q2 / hybrid implementation)
    MISSILE_POS0 = torch.tensor([20000.0,0.0,2000.0], device=device, dtype=torch.float64)
    MISSILE_TARGET = torch.tensor([0.0,0.0,0.0], device=device, dtype=torch.float64)
    MISSILE_SPEED = 300.0
    dvec = MISSILE_TARGET - MISSILE_POS0; dist = torch.linalg.norm(dvec); MISSILE_DIR = dvec / dist
    T_f = dist / MISSILE_SPEED
    n_steps = int(T_f/dt) + 1
    t_grid = torch.linspace(0.0, T_f, n_steps, device=device, dtype=torch.float64)
    missile_pos = MISSILE_POS0.unsqueeze(0) + MISSILE_DIR.unsqueeze(0) * (MISSILE_SPEED * t_grid).unsqueeze(1)  # (T,3)
    # Drone + cloud initial state
    drone_pos0 = torch.tensor([17800.0,0.0,1800.0], device=device, dtype=torch.float64).expand(N,3)
    drone_dir = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], dim=1)
    a_vec = torch.tensor([0.0,0.0,-9.8], device=device, dtype=torch.float64)
    rel = drone_pos0 + drone_dir * speed.unsqueeze(1) * t_rel.unsqueeze(1)
    vel = drone_dir * speed.unsqueeze(1)
    c0 = rel + vel * dly.unsqueeze(1) + 0.5 * a_vec * (dly.unsqueeze(1)**2)
    explode = t_rel + dly
    SMOKE_LIFE = 20.0; SMOKE_DESCENT = 3.0; SMOKE_RADIUS = 10.0
    end = explode + SMOKE_LIFE
    # Cylinder constants
    R_CYL = 7.0; H_CYL = 10.0
    Cb = torch.tensor([0.0,200.0,0.0], device=device, dtype=torch.float64)
    Ct = Cb + torch.tensor([0.0,0.0,H_CYL], device=device, dtype=torch.float64)
    # Main loop (iterate only over potentially active global window)
    start_idx = torch.clamp((explode/dt).ceil().long(), 0, n_steps-1)
    end_idx = torch.clamp((end/dt).floor().long(), 0, n_steps-1)
    global_start = start_idx[valid].min().item()
    global_end = end_idx[valid].max().item()
    for ti in range(global_start, global_end+1):
        t_now = t_grid[ti]
        act_mask = valid & (ti >= start_idx) & (ti <= end_idx)
        if not act_mask.any():
            if ti > end_idx[valid].max():
                break
            continue
        idx = act_mask.nonzero().squeeze(1)
        V = missile_pos[ti].unsqueeze(0).expand(idx.shape[0],3)
        tau = torch.clamp(t_now - explode[idx], min=0.0)
        center_t = c0[idx].clone()
        center_t[:,2] = c0[idx][:,2] - SMOKE_DESCENT * tau
        R_cloud = torch.full((idx.shape[0],), SMOKE_RADIUS, device=device, dtype=torch.float64)
        # Bottom cap (already in z=0 plane of Cb)
        Cb_batch = Cb.unsqueeze(0).expand(idx.shape[0],3)
        r_cap = torch.full((idx.shape[0],), R_CYL, device=device, dtype=torch.float64)
        res_b = vectorized_circle_fully_occluded_by_sphere_torch(V, Cb_batch, r_cap, center_t, R_cloud)
        oc_b = res_b.occluded
        # Top cap: shift frame by H_CYL
        Vt = V.clone(); Vt[:,2] -= H_CYL
        Ct_flat = torch.tensor([Ct[0].item(), Ct[1].item(), 0.0], device=device, dtype=torch.float64).unsqueeze(0).expand(idx.shape[0],3)
        St = center_t.clone(); St[:,2] -= H_CYL
        res_t = vectorized_circle_fully_occluded_by_sphere_torch(Vt, Ct_flat, r_cap, St, R_cloud)
        oc_t = res_t.occluded
        both = oc_b & oc_t
        if both.any():
            occluded_time[idx[both]] += dt
    return occluded_time


__all__ = [
    'VectorizedOcclusionResultTorch',
    'vectorized_circle_fully_occluded_by_sphere_torch',
    'batch_occluded_time_caps_torch',
    'batch_occluded_time_caps_torch_newton'
]

# ===================== Newton-based (approximate) variant =====================
def vectorized_circle_fmin_cos_torch_newton(V: torch.Tensor, C: torch.Tensor, r: torch.Tensor, S: torch.Tensor, *, n_seeds: int = 8, iters: int = 6, tol: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate f_min using Newton iterations on H(t)=0 instead of quartic roots.

    H(t) = N'(t) M(t) - 0.5 N(t) M'(t). We iterate t_{k+1} = t_k - H/H'.
    Use multiple uniformly spaced initial seeds in [0, 2π). Keep best f among converged.
    Falls back to simple angle sampling (the seeds themselves) if Newton fails.
    """
    device = V.device; dtype = V.dtype; N = V.shape[0]
    zero_r_mask = torch.abs(r) <= 1e-12
    consts = _f_of_t_constants_torch(V, C, r, S)
    A = consts['A']; B = consts['B']; D = consts['D']; E = consts['E']; F = consts['F']; G = consts['G']
    f_min = torch.full((N,), float('nan'), device=device, dtype=dtype)
    t_min = torch.full((N,), float('nan'), device=device, dtype=dtype)
    # Zero radius: evaluate at t=0
    if zero_r_mask.any():
        M0 = E[zero_r_mask]; Nz = D[zero_r_mask]; mv = M0 > 1e-12
        if mv.any():
            vals = Nz[mv] / torch.sqrt(M0[mv])
            zr = zero_r_mask.nonzero().squeeze(1)[mv]
            f_min[zr] = vals; t_min[zr] = 0.0
    nz_mask = ~zero_r_mask
    if not nz_mask.any():
        return f_min, t_min
    idx = nz_mask.nonzero().squeeze(1)
    A_n, B_n, D_n, E_n, F_n, G_n, r_n = A[idx], B[idx], D[idx], E[idx], F[idx], G[idx], r[idx]
    # Seeds
    seeds = torch.linspace(0, 2*torch.pi, n_seeds+1, device=device, dtype=dtype)[:-1]  # (S,)
    t = seeds.unsqueeze(0).expand(idx.shape[0], -1)  # (K,S)
    # Newton iterations
    for _ in range(iters):
        c = torch.cos(t); s = torch.sin(t)
        N_val = D_n.unsqueeze(1) + r_n.unsqueeze(1)*(A_n.unsqueeze(1)*c + B_n.unsqueeze(1)*s)
        M_val = E_n.unsqueeze(1) + 2*r_n.unsqueeze(1)*(F_n.unsqueeze(1)*c + G_n.unsqueeze(1)*s)
        # Derivatives
        # N' = r(-A s + B c)
        Np = r_n.unsqueeze(1)*(-A_n.unsqueeze(1)*s + B_n.unsqueeze(1)*c)
        # N'' = r(-A c - B s)
        Npp = r_n.unsqueeze(1)*(-A_n.unsqueeze(1)*c - B_n.unsqueeze(1)*s)
        # M' = 2r(-F s + G c)
        Mp = 2*r_n.unsqueeze(1)*(-F_n.unsqueeze(1)*s + G_n.unsqueeze(1)*c)
        # M'' = 2r(-F c - G s)
        Mpp = 2*r_n.unsqueeze(1)*(-F_n.unsqueeze(1)*c - G_n.unsqueeze(1)*s)
        H = Np*M_val - 0.5*N_val*Mp
        Hp = Npp*M_val + 0.5*Np*Mp - 0.5*N_val*Mpp
        denom = Hp.abs() + 1e-12
        delta = H/denom
        t_next = t - delta
        # Wrap to [0,2π)
        t = (t_next + 2*torch.pi) % (2*torch.pi)
        if (delta.abs() < tol).all():
            break
    # Evaluate final f values, mask invalid (M<=eps)
    c = torch.cos(t); s = torch.sin(t)
    N_val = D_n.unsqueeze(1) + r_n.unsqueeze(1)*(A_n.unsqueeze(1)*c + B_n.unsqueeze(1)*s)
    M_val = E_n.unsqueeze(1) + 2*r_n.unsqueeze(1)*(F_n.unsqueeze(1)*c + G_n.unsqueeze(1)*s)
    valid = M_val > 1e-12
    f_vals = torch.full_like(N_val, torch.inf)
    f_vals[valid] = N_val[valid]/torch.sqrt(M_val[valid])
    # Also include seed sampling baseline (same t already) -> f_vals holds both Newton-refined & seeds coincide
    min_vals, pos = torch.min(f_vals, dim=1)
    good = torch.isfinite(min_vals)
    if good.any():
        tgt = idx[good]
        f_min[tgt] = min_vals[good]
        t_min[tgt] = t[good, :].gather(1, pos[good].unsqueeze(1)).squeeze(1)
    return f_min, t_min


def vectorized_circle_fully_occluded_by_sphere_torch_newton(V: torch.Tensor, C: torch.Tensor, r: torch.Tensor, S: torch.Tensor, R: torch.Tensor) -> VectorizedOcclusionResultTorch:
    N = V.shape[0]; device=V.device; dtype=V.dtype
    occluded = torch.zeros(N, dtype=torch.bool, device=device)
    f_min = torch.full((N,), float('nan'), dtype=dtype, device=device)
    cos_alpha_s = torch.full((N,), float('nan'), dtype=dtype, device=device)
    t_min = torch.full((N,), float('nan'), dtype=dtype, device=device)
    valid = torch.zeros(N, dtype=torch.bool, device=device)
    S_prime = S - V; dS = torch.linalg.norm(S_prime, dim=1)
    observer_in = (dS <= 1e-12) | (R >= dS)
    if observer_in.any():
        occluded[observer_in] = True; valid[observer_in] = True
    normal = ~observer_in
    if normal.any():
        idx = normal.nonzero().squeeze(1)
        f_min_n, t_min_n = vectorized_circle_fmin_cos_torch_newton(V[idx], C[idx], r[idx], S[idx])
        dS_n = dS[idx]; ratio = R[idx]/dS_n
        cos_as = torch.sqrt(torch.clamp(1 - ratio*ratio, min=0.0)); cos_as[ratio>=1] = 0.0
        valid_n = torch.isfinite(f_min_n)
        occ_n = valid_n & (f_min_n >= cos_as - 1e-12)
        f_min[idx] = f_min_n; cos_alpha_s[idx] = cos_as; t_min[idx] = t_min_n; valid[idx]=valid_n; occluded[idx]=occ_n
    return VectorizedOcclusionResultTorch(occluded=occluded, f_min=f_min, cos_alpha_s=cos_alpha_s, t_min=t_min, valid=valid)


def batch_occluded_time_caps_torch_newton(params: torch.Tensor, *, dt: float, device: torch.device) -> torch.Tensor:
    if params.numel()==0:
        return torch.zeros(0, device=device, dtype=torch.float64)
    p = params.to(device=device, dtype=torch.float64)
    speed, az, t_rel, dly = p[:,0], p[:,1], p[:,2], p[:,3]
    N = p.shape[0]
    valid = (speed>=70.0) & (speed<=140.0) & (t_rel>=0.0) & (dly>0.0)
    out = torch.zeros(N, device=device, dtype=torch.float64)
    if not valid.any():
        return out
    MISSILE_POS0 = torch.tensor([20000.0,0.0,2000.0], device=device, dtype=torch.float64)
    MISSILE_TARGET = torch.tensor([0.0,0.0,0.0], device=device, dtype=torch.float64)
    MISSILE_SPEED = 300.0
    dvec = MISSILE_TARGET - MISSILE_POS0; dist = torch.linalg.norm(dvec); MISSILE_DIR = dvec/dist
    T_f = dist / MISSILE_SPEED
    n_steps = int(T_f/dt)+1
    t_grid = torch.linspace(0.0, T_f, n_steps, device=device, dtype=torch.float64)
    missile_pos = MISSILE_POS0.unsqueeze(0)+MISSILE_DIR.unsqueeze(0)*(MISSILE_SPEED*t_grid).unsqueeze(1)
    drone0 = torch.tensor([17800.0,0.0,1800.0], device=device, dtype=torch.float64).expand(N,3)
    ddir = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)],1)
    a_vec = torch.tensor([0.0,0.0,-9.8], device=device, dtype=torch.float64)
    rel = drone0 + ddir*speed.unsqueeze(1)*t_rel.unsqueeze(1)
    vel = ddir*speed.unsqueeze(1)
    c0 = rel + vel*dly.unsqueeze(1) + 0.5*a_vec*(dly.unsqueeze(1)**2)
    explode = t_rel + dly; SMOKE_LIFE=20.0; SMOKE_DESCENT=3.0; SMOKE_RADIUS=10.0
    end = explode + SMOKE_LIFE
    R_CYL=7.0; H_CYL=10.0
    Cb = torch.tensor([0.0,200.0,0.0], device=device, dtype=torch.float64)
    Ct = Cb + torch.tensor([0.0,0.0,H_CYL], device=device, dtype=torch.float64)
    start_idx = torch.clamp((explode/dt).ceil().long(),0,n_steps-1)
    end_idx = torch.clamp((end/dt).floor().long(),0,n_steps-1)
    g_start = start_idx[valid].min().item(); g_end = end_idx[valid].max().item()
    for ti in range(g_start, g_end+1):
        act = valid & (ti>=start_idx) & (ti<=end_idx)
        if not act.any():
            if ti > end_idx[valid].max():
                break
            continue
        ids = act.nonzero().squeeze(1)
        V = missile_pos[ti].unsqueeze(0).expand(ids.shape[0],3)
        t_now = t_grid[ti]
        tau = torch.clamp(t_now - explode[ids], min=0.0)
        center_t = c0[ids].clone(); center_t[:,2] = c0[ids][:,2]-SMOKE_DESCENT*tau
        R_cloud = torch.full((ids.shape[0],), SMOKE_RADIUS, device=device, dtype=torch.float64)
        Cb_b = Cb.unsqueeze(0).expand(ids.shape[0],3)
        r_cap = torch.full((ids.shape[0],), R_CYL, device=device, dtype=torch.float64)
        res_b = vectorized_circle_fully_occluded_by_sphere_torch_newton(V, Cb_b, r_cap, center_t, R_cloud)
        # top
        Vt = V.clone(); Vt[:,2]-=H_CYL
        Ct_flat = torch.tensor([Ct[0].item(), Ct[1].item(), 0.0], device=device, dtype=torch.float64).unsqueeze(0).expand(ids.shape[0],3)
        St = center_t.clone(); St[:,2]-=H_CYL
        res_t = vectorized_circle_fully_occluded_by_sphere_torch_newton(Vt, Ct_flat, r_cap, St, R_cloud)
        full = res_b.occluded & res_t.occluded
        if full.any():
            out[ids[full]] += dt
    return out
