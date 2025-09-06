shumo/judges/router.py#L1-220
"""
Occlusion routing utilities.

Goal:
    Provide a single high-level function (occlusion_time_router) that, given a
    Simulator instance and a (circle ⟂ sphere) vectorized judge function
    (NumPy or Torch), computes the total occlusion time of the cylinder caps
    in ONE batched evaluation instead of per-frame / per-sphere nested loops.

Key Idea:
    For each simulation frame t and each ACTIVE smoke sphere k we form two
    circle–sphere queries (bottom cap & top cap). A frame is counted as
    occluded iff:
        (∃ sphere k covering bottom cap) AND (∃ sphere k' covering top cap)
    (k and k' are allowed to differ).

    We convert all (frame, sphere) pairs into two flat batches (bottom & top),
    call the provided judge function ONCE per cap set, then fold results
    back to frame level with a logical any() per frame. Finally we combine
    bottom & top results by logical and() and accumulate time.

Supported judge function signatures (duck-typed):
    result = judge_fn(V_batch, C_batch, r_batch, S_batch, R_batch)
    where:
        - V_batch: (N,3)
        - C_batch: (N,3)   (circle centers, z=0 in the flattened local frame)
        - r_batch: (N,)
        - S_batch: (N,3)   (sphere centers after same flatten transform)
        - R_batch: (N,)
    result must expose attribute/result field `.occluded` yielding shape (N,)
    boolean-like (NumPy ndarray or Torch tensor).

Typical judge functions already available in the project:
    - judges.vectorized_circle_fully_occluded_by_sphere          (NumPy)
    - judges.vectorized_circle_fully_occluded_by_sphere_torch    (Torch)
    - judges.vectorized_circle_fully_occluded_by_sphere_torch_newton (Torch Newton approx)
    - Rough variants are also acceptable (they still produce .occluded)

Usage Example:
    from simcore.simulator import Simulator
    from simcore.entities import Missile, Drone, Cylinder
    from judges import vectorized_circle_fully_occluded_by_sphere_torch
    from judges.router import occlusion_time_router
    import numpy as np

    missile = Missile(1)
    drone = Drone(1, direction=np.array([1,0,0]), speed=100.0, strategy=[])
    sim = Simulator(missile=missile, drones=[drone])
    # Provide schedules etc. (omitted)
    occluded_time, details = occlusion_time_router(
        sim,
        judge_fn=vectorized_circle_fully_occluded_by_sphere_torch,
        dt=0.02,
        vectorized=True,
        return_details=True,
    )
    print(occluded_time, details['frames_total'])

Notes:
    - If vectorized=False the function simply delegates to Simulator.run and
      returns its already-computed occluded_time (loop fallback mode).
    - When vectorized=True we still invoke sim.run(dt, verbose=...) ONLY to
      gather missile positions & active spheres per frame; its own per-frame
      occlusion results are ignored to prevent double computation.
    - Torch vs NumPy is auto-detected based on the judge function output type
      (or inferred from its __name__). You can override via force_torch flag.
    - Frames with zero active spheres are automatically considered not occluded.

Performance Considerations:
    Let T be number of frames, B_max maximum simultaneous active spheres.
    We build at most T * B_tot rows (B_tot = total active (frame,sphere) pairs;
    often much less than T * B_max if clouds are sparse). Only those active
    pairs are sent to the judge function, minimizing wasted computation.

Return:
    (occluded_time: float, details: dict | None)

details dict (when return_details=True) contains:
    frames_total
    frame_dt
    active_pairs_bottom
    active_pairs_top
    bottom_occluded_mask (np.ndarray / list[bool])
    top_occluded_mask
    final_frame_mask
    method: 'vectorized' | 'loop'
    backend: 'torch' | 'numpy'
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict, Any, List

# Type alias for judge function (duck typed)
JudgeFn = Callable[..., Any]


def _is_torch_tensor(obj) -> bool:
    return obj.__class__.__module__.startswith("torch") and hasattr(obj, "dtype")


def _maybe_to_numpy(x):
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return x


def occlusion_time_router(
    sim,
    judge_fn: JudgeFn,
    *,
    dt: float,
    vectorized: bool = True,
    timeline: Optional[List[Dict[str, Any]]] = None,
    force_torch: Optional[bool] = None,
    return_details: bool = False,
    verbose: bool = False,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Compute total occlusion time for a Simulator via a supplied circle-sphere judge.
    """
    # ------------------------------------------------------------------
    # Acquire / prepare timeline
    # ------------------------------------------------------------------
    if timeline is None:
        timeline_result = sim.run(dt=dt, verbose=(vectorized or verbose))
        timeline = timeline_result["timeline"]
    frames_total = len(timeline)
    if frames_total == 0:
        return 0.0, (dict(frames_total=0, frame_dt=dt, method="vectorized", backend="none") if return_details else None)

    # ------------------------------------------------------------------
    # Loop fallback (non-vectorized)
    # ------------------------------------------------------------------
    if not vectorized:
        occluded_frames = sum(1 for f in timeline if f.get("occluded"))
        occluded_time = occluded_frames * dt
        details = None
        if return_details:
            details = dict(
                frames_total=frames_total,
                frame_dt=dt,
                method="loop",
                backend="simulator",
                occluded_frames=occluded_frames,
            )
        return float(occluded_time), details

    # ------------------------------------------------------------------
    # Extract cylinder geometry
    # ------------------------------------------------------------------
    cyl = sim.cylinder
    Cb = getattr(cyl, "C_base")
    r_cap = float(getattr(cyl, "r"))
    h_cap = float(getattr(cyl, "h"))
    Ct = Cb + type(Cb)([0.0, 0.0, h_cap])

    # ------------------------------------------------------------------
    # Collect active spheres & viewpoints per frame
    # ------------------------------------------------------------------
    V_list: List[Any] = []
    bottom_rows: List[int] = []
    top_rows: List[int] = []
    S_bottom: List[Any] = []
    R_bottom: List[float] = []
    S_top: List[Any] = []
    R_top: List[float] = []

    Cb_flat = [Cb[0], Cb[1], 0.0]
    Ct_flat = [Ct[0], Ct[1], 0.0]

    for fi, frame in enumerate(timeline):
        Vt = frame["missile_pos"]
        clouds = frame.get("clouds", [])
        V_list.append(Vt)
        if not clouds:
            continue
        for (S, R) in clouds:
            Sb = S.copy(); Sb[2] -= Cb[2]
            bottom_rows.append(fi)
            S_bottom.append(Sb)
            R_bottom.append(R)
            St = S.copy(); St[2] -= Ct[2]
            top_rows.append(fi)
            S_top.append(St)
            R_top.append(R)

    if len(S_bottom) == 0:
        details = None
        if return_details:
            details = dict(
                frames_total=frames_total,
                frame_dt=dt,
                active_pairs_bottom=0,
                active_pairs_top=0,
                bottom_occluded_mask=[False]*frames_total,
                top_occluded_mask=[False]*frames_total,
                final_frame_mask=[False]*frames_total,
                method="vectorized",
                backend="empty",
            )
        return 0.0, details

    # ------------------------------------------------------------------
    # Decide backend (torch / numpy)
    # ------------------------------------------------------------------
    backend = "numpy"
    use_torch = False
    if force_torch is not None:
        use_torch = force_torch
    else:
        name = getattr(judge_fn, "__name__", "")
        if "_torch" in name:
            use_torch = True

    if use_torch:
        try:
            import torch  # type: ignore
            backend = "torch"
        except Exception:
            use_torch = False
            backend = "numpy"

    import numpy as _np
    V_arr = _np.asarray(V_list, dtype=_np.float64)

    def _build_batch(rows: List[int], sphere_centers: List[Any], sphere_radii: List[float], cap_flat_center, cap_z: float):
        n = len(rows)
        Vb = _np.asarray(V_arr[rows], dtype=_np.float64).copy()
        if n:
            Vb[:, 2] -= cap_z   # proper flattening of viewpoint to cap plane
        Cb_arr = _np.repeat(_np.asarray(cap_flat_center, dtype=_np.float64).reshape(1, 3), n, axis=0)
        rb = _np.full(n, r_cap, dtype=_np.float64)
        Sb_arr = _np.asarray(sphere_centers, dtype=_np.float64)
        Rb = _np.asarray(sphere_radii, dtype=_np.float64)
        return Vb, Cb_arr, rb, Sb_arr, Rb

    Vb_bot, Cb_bot, rb_bot, Sb_bot, Rb_bot = _build_batch(bottom_rows, S_bottom, R_bottom, Cb_flat, Cb[2])
    Vb_top, Cb_top, rb_top, Sb_top, Rb_top = _build_batch(top_rows, S_top, R_top, Ct_flat, Ct[2])

    if use_torch:
        import torch  # type: ignore

        def _to_t(x):
            return torch.as_tensor(x, dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")

        tVb_bot = _to_t(Vb_bot); tCb_bot = _to_t(Cb_bot); trb_bot = _to_t(rb_bot)
        tSb_bot = _to_t(Sb_bot); tRb_bot = _to_t(Rb_bot)
        tVb_top = _to_t(Vb_top); tCb_top = _to_t(Cb_top); trb_top = _to_t(rb_top)
        tSb_top = _to_t(Sb_top); tRb_top = _to_t(Rb_top)

        if verbose:
            print(f"[router] Running torch batch: bottom_pairs={tVb_bot.shape[0]}, top_pairs={tVb_top.shape[0]}")
        res_bot = judge_fn(tVb_bot, tCb_bot, trb_bot, tSb_bot, tRb_bot)
        res_top = judge_fn(tVb_top, tCb_top, trb_top, tSb_top, tRb_top)
        occluded_bot_pairs = res_bot.occluded.detach().cpu().numpy().astype(bool)
        occluded_top_pairs = res_top.occluded.detach().cpu().numpy().astype(bool)
    else:
        if verbose:
            print(f"[router] Running numpy batch: bottom_pairs={Vb_bot.shape[0]}, top_pairs={Vb_top.shape[0]}")
        res_bot = judge_fn(Vb_bot, Cb_bot, rb_bot, Sb_bot, Rb_bot)
        res_top = judge_fn(Vb_top, Cb_top, rb_top, Sb_top, Rb_top)
        occluded_bot_pairs = _maybe_to_numpy(res_bot.occluded).astype(bool)
        occluded_top_pairs = _maybe_to_numpy(res_top.occluded).astype(bool)

    bottom_mask = _np.zeros(frames_total, dtype=bool)
    top_mask = _np.zeros(frames_total, dtype=bool)

    for row_idx, fi in enumerate(bottom_rows):
        if occluded_bot_pairs[row_idx]:
            bottom_mask[fi] = True
    for row_idx, fi in enumerate(top_rows):
        if occluded_top_pairs[row_idx]:
            top_mask[fi] = True

    final_mask = bottom_mask & top_mask
    occluded_frames = int(final_mask.sum())
    occluded_time = occluded_frames * dt

    details = None
    if return_details:
        details = dict(
            frames_total=frames_total,
            frame_dt=dt,
            active_pairs_bottom=len(bottom_rows),
            active_pairs_top=len(top_rows),
            bottom_occluded_mask=bottom_mask,
            top_occluded_mask=top_mask,
            final_frame_mask=final_mask,
            method="vectorized",
            backend=backend,
            occluded_frames=occluded_frames,
        )

    return float(occluded_time), details


__all__ = ["occlusion_time_router"]
