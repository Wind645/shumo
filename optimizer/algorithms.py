from __future__ import annotations
"""Unified, minimal optimization algorithms (GA / PSO / SA) for Problems 2–5.

Design goals:
 - Single file; no repetition across algorithms.
 - Simple: focus on core logic; minimal checks / no heavy engineering.
 - Shared encoding via optimizer.spec (bounds + decode).
 - Batch speed-up only for Problem 2 with occlusion_method='judge_caps'.
 - Other cases fall back to scalar simulation calls.

Decision encodings: see optimizer/spec.py
Objective: maximize total occluded time (we minimize f = -occluded_time_total)
Returned best_value is minimized objective; occluded time = -best_value.
"""
import math, random, numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Iterable, Optional
from optimizer.spec import bounds_for_problem, decode_vector
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False
from api import problems
from api.problems import evaluate_problem2, evaluate_problem3, evaluate_problem4, evaluate_problem5
from vectorized_judge import vectorized_circle_fully_occluded_by_sphere
from simcore.constants import (
    R_CYL_DEFAULT as _R_CYL,
    H_CYL_DEFAULT as _H_CYL,
    SMOKE_LIFETIME_DEFAULT as _SMOKE_LIFE,
    SMOKE_DESCENT_DEFAULT as _SMOKE_DESCENT,
    SMOKE_RADIUS_DEFAULT as _SMOKE_RADIUS,
    G as _G,
)

# Missile spec (Problem statement M1)
_MISSILE_POS0 = np.array([20000.0, 0.0, 2000.0], dtype=np.float64)
_MISSILE_SPEED = 300.0
_MISSILE_TARGET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

def _missile_dir_and_flight_time():
    d = _MISSILE_TARGET - _MISSILE_POS0
    n = np.linalg.norm(d)
    return d / n, n / _MISSILE_SPEED
_MISSILE_DIR, _MISSILE_T_FLIGHT = _missile_dir_and_flight_time()

# --------------------- batch occlusion (Problem 2, judge_caps) ---------------------

def _batch_occluded_time_q2_judge_caps(params: np.ndarray, *, dt: float) -> np.ndarray:
    if params.size == 0:
        return np.zeros(0, dtype=np.float64)
    p = params.astype(np.float64)
    speed, az, t_rel, dly = p[:,0], p[:,1], p[:,2], p[:,3]
    N = p.shape[0]
    valid = (speed>=70)&(speed<=140)&(t_rel>=0)&(dly>0)
    out = np.zeros(N, dtype=np.float64)
    if not valid.any():
        return out
    T_f = float(_MISSILE_T_FLIGHT)
    n_steps = int(T_f/dt)+1
    t_grid = np.linspace(0.0, T_f, n_steps, dtype=np.float64)
    missile_pos = _MISSILE_POS0[np.newaxis,:] + _MISSILE_DIR[np.newaxis,:]*(_MISSILE_SPEED*t_grid[:,np.newaxis])
    drone0 = np.array([17800.0,0.0,1800.0], dtype=np.float64)
    drone0_batch = np.tile(drone0,(N,1))
    ddir = np.column_stack([np.cos(az), np.sin(az), np.zeros_like(az)])
    a = np.array([0.0,0.0,-_G], dtype=np.float64)
    rel = drone0_batch + ddir*speed[:,None]*t_rel[:,None]
    vel = ddir*speed[:,None]
    c0 = rel + vel*dly[:,None] + 0.5*a*(dly[:,None]**2)
    explode = t_rel + dly
    end = explode + _SMOKE_LIFE
    r_cyl = float(_R_CYL)
    shift = float(_H_CYL)
    Rsm = float(_SMOKE_RADIUS)
    Cb = np.array([0.0,200.0,0.0], dtype=np.float64)
    Ct = Cb + np.array([0.0,0.0,shift], dtype=np.float64)
    # Precompute step bounds per candidate for loop shrinking
    start_idx = np.clip(np.ceil(explode/dt).astype(int), 0, n_steps-1)
    end_idx   = np.clip(np.floor(end/dt).astype(int), 0, n_steps-1)
    global_start = start_idx[valid].min()
    global_end = end_idx[valid].max()
    for ti in range(global_start, global_end+1):
        act = valid & (ti>=start_idx) & (ti<=end_idx)
        if not act.any():
            continue
        idx = np.nonzero(act)[0]
        V = missile_pos[ti]
        t_now = t_grid[ti]
        tau = np.maximum(0.0, t_now - explode[idx])
        center_t = c0[idx].copy(); center_t[:,2] = c0[idx][:,2] - _SMOKE_DESCENT*tau
        # bottom
        SCb = center_t - V; CCb = Cb - V
        CCb = np.tile(CCb,(len(idx),1))
        dSb = np.linalg.norm(SCb,axis=1); dCb = np.linalg.norm(CCb,axis=1)
        ratio_s = np.clip(Rsm/(dSb+1e-9),0,1); ratio_c = np.clip(r_cyl/(dCb+1e-9),0,1)
        uS = SCb/(dSb[:,None]+1e-9); uC = CCb/(dCb[:,None]+1e-9)
        cosg = np.clip((uS*uC).sum(1), -1,1); g = np.arccos(cosg)
        b = np.arcsin(ratio_c); apha = np.arcsin(ratio_s)
        oc_b = (g+b) <= (apha+1e-6)
        # top
        Vt = V.copy(); Vt[2] -= shift
        Ct_flat = np.array([Ct[0],Ct[1],0.0])
        SCt = center_t.copy(); SCt[:,2]-=shift; SCt -= Vt
        CCt = Ct_flat - Vt; CCt = np.tile(CCt,(len(idx),1))
        dSt = np.linalg.norm(SCt,axis=1); dCt = np.linalg.norm(CCt,axis=1)
        ratio_s2 = np.clip(Rsm/(dSt+1e-9),0,1); ratio_c2 = np.clip(r_cyl/(dCt+1e-9),0,1)
        uS2 = SCt/(dSt[:,None]+1e-9); uC2 = CCt/(dCt[:,None]+1e-9)
        cosg2 = np.clip((uS2*uC2).sum(1), -1,1); g2 = np.arccos(cosg2)
        b2 = np.arcsin(ratio_c2); apha2 = np.arcsin(ratio_s2)
        oc_t = (g2+b2) <= (apha2+1e-6)
        full = oc_b & oc_t
        if full.any():
            out[idx[full]] += dt
    return out

# --------------------- (可选) torch fused batch for Q2 ---------------------
TORCH_Q2_FUSED: bool = False  # 由 router 赋值 True 则尝试使用 GPU (或 CPU torch) 加速 Q2 judge_caps / rough_caps

def _batch_occluded_time_q2_torch(params: np.ndarray, *, dt: float) -> np.ndarray:
    if (not _TORCH_AVAILABLE) or params.size == 0:
        return np.zeros(params.shape[0], dtype=np.float64)
    import math as _m
    p = torch.as_tensor(params, dtype=torch.float32)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = p.to(dev)
    speed, az, t_rel, dly = p[:,0], p[:,1], p[:,2], p[:,3]
    N = p.shape[0]
    valid = (speed>=70)&(speed<=140)&(t_rel>=0)&(dly>0)
    out = torch.zeros(N, device=dev)
    if not valid.any():
        return out.cpu().numpy().astype(float)
    missile_pos0 = torch.tensor([20000.0,0.0,2000.0], device=dev)
    missile_target = torch.tensor([0.0,0.0,0.0], device=dev)
    d = missile_target - missile_pos0
    n = torch.linalg.norm(d)
    miss_dir = d / n
    miss_speed = 300.0
    T_f = float(n / miss_speed)
    n_steps = int(T_f/dt)+1
    t_grid = torch.linspace(0.0, T_f, n_steps, device=dev)
    missile_pos = missile_pos0.unsqueeze(0) + miss_dir.unsqueeze(0)*(miss_speed*t_grid).unsqueeze(1)
    drone0 = torch.tensor([17800.0,0.0,1800.0], device=dev).view(1,3).repeat(N,1)
    ddir = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)],1)
    a = torch.tensor([0.0,0.0,-_G], device=dev)
    rel = drone0 + ddir*speed.unsqueeze(1)*t_rel.unsqueeze(1)
    vel = ddir*speed.unsqueeze(1)
    c0 = rel + vel*dly.unsqueeze(1) + 0.5*a*(dly.unsqueeze(1)**2)
    explode = t_rel + dly
    end = explode + _SMOKE_LIFE
    t_row = t_grid.unsqueeze(0)
    active = valid.unsqueeze(1) & (t_row>=explode.unsqueeze(1)) & (t_row<=end.unsqueeze(1))
    if not active.any():
        return out.cpu().numpy().astype(float)
    tau = torch.clamp(t_row - explode.unsqueeze(1), min=0)
    ct = c0.unsqueeze(1).expand(N,n_steps,3).clone()
    ct[:,:,2] = c0[:,2].unsqueeze(1) - _SMOKE_DESCENT*tau
    V = missile_pos.unsqueeze(0).expand(N,n_steps,3)
    Cb = torch.tensor([0.0,200.0,0.0], device=dev)
    r_cyl = float(_R_CYL); shift = float(_H_CYL); Rsm = float(_SMOKE_RADIUS)
    # bottom
    SCb = ct - V; CCb = Cb.view(1,1,3) - V
    dSb = torch.linalg.norm(SCb, dim=2); dCb = torch.linalg.norm(CCb, dim=2)
    ratio_s = torch.clamp(Rsm/(dSb+1e-9),0,1); ratio_c = torch.clamp(r_cyl/(dCb+1e-9),0,1)
    uS = SCb/(dSb.unsqueeze(2)+1e-9); uC = CCb/(dCb.unsqueeze(2)+1e-9)
    cosg = (uS*uC).sum(2).clamp(-1,1); g = torch.arccos(cosg)
    b = torch.arcsin(ratio_c); apha = torch.arcsin(ratio_s)
    oc_b = (g+b) <= (apha+1e-6)
    # top
    Vt = V.clone(); Vt[:,:,2]-=shift
    Ct_flat = torch.tensor([0.0,200.0,shift], device=dev)  # cylinder top center
    # translate top plane to z=0 technique
    SCt = ct.clone(); SCt[:,:,2]-=shift; SCt -= Vt
    CCt = torch.tensor([0.0,200.0,0.0], device=dev).view(1,1,3) - Vt
    dSt = torch.linalg.norm(SCt, dim=2); dCt = torch.linalg.norm(CCt, dim=2)
    ratio_s2 = torch.clamp(Rsm/(dSt+1e-9),0,1); ratio_c2 = torch.clamp(r_cyl/(dCt+1e-9),0,1)
    uS2 = SCt/(dSt.unsqueeze(2)+1e-9); uC2 = CCt/(dCt.unsqueeze(2)+1e-9)
    cosg2 = (uS2*uC2).sum(2).clamp(-1,1); g2 = torch.arccos(cosg2)
    b2 = torch.arcsin(ratio_c2); apha2 = torch.arcsin(ratio_s2)
    oc_t = (g2+b2) <= (apha2+1e-6)
    full = active & oc_b & oc_t
    out[valid] = full[valid].sum(1)*dt
    return out.detach().cpu().numpy().astype(float)

# --------------------- objective wrapper ---------------------

def _scalar_occluded_time(problem: int, x: List[float], bombs_count: int, method: str, dt: float) -> float:
    dec = decode_vector(problem, x, bombs_count)
    # 若炸弹时间间隔非法，直接返回 0 遮蔽时间 (最差) 而不是抛异常中断优化
    if dec.invalid:
        return 0.0
    try:
        if problem == 2:
            r = evaluate_problem2(**dec.to_eval_kwargs(), occlusion_method=method, dt=dt)
        elif problem == 3:
            r = evaluate_problem3(**dec.to_eval_kwargs(), occlusion_method=method, dt=dt)
        elif problem == 4:
            r = evaluate_problem4(**dec.to_eval_kwargs(), occlusion_method=method, dt=dt)
        elif problem == 5:
            r = evaluate_problem5(**dec.to_eval_kwargs(), occlusion_method=method, dt=dt)
        else:
            return 0.0
    except ValueError as e:
        msg = str(e)
        if '投弹间隔不足' in msg or '间隔不足' in msg:
            return 0.0  # 视为无效方案
        raise
    if 'total' in r:
        return float(r['total']) if problem>2 else float(r['occluded_time'].get('M1', r['total']))
    return 0.0

# Batch (only prob2 & judge_caps) returns occluded time array

def _batch_eval(problem: int, X: np.ndarray, bombs_count: int, method: str, dt: float) -> np.ndarray:
    # Q2 向量化: judge_caps 与 rough_caps
    if problem==2 and method in ('judge_caps','rough_caps') and X.shape[1]==4:
        if TORCH_Q2_FUSED and _TORCH_AVAILABLE:
            return _batch_occluded_time_q2_torch(X, dt=dt)
        return _batch_occluded_time_q2_judge_caps(X, dt=dt)
    return np.array([_scalar_occluded_time(problem, X[i].tolist(), bombs_count, method, dt) for i in range(X.shape[0])], dtype=np.float64)

# ============================= SA =============================
@dataclass
class SAResult:
    best_x: np.ndarray
    best_value: float
    best_eval: Dict

def solve_sa(*, problem: int=2, bombs_count: int=2, iters: int=2000, dt: float=0.02, method: str='judge_caps', neighbor_batch: int=64, step_scale: float=0.2, init_temp: float=2.0, final_temp: float=1e-3, seed: int|None=None, verbose: bool=False, log_interval: int|None=None) -> SAResult:
    if seed is not None: random.seed(seed); np.random.seed(seed)
    bnds = bounds_for_problem(problem, bombs_count)
    lo = np.array([a for (a,_) in bnds]); hi = np.array([b for (_,b) in bnds])
    span = hi - lo; dim = len(bnds)
    cur = lo + np.random.rand(dim)*span
    def temp(k): return init_temp*((final_temp/init_temp)**(k/max(1,iters-1)))
    def obj_vec(X): return -_batch_eval(problem, X, bombs_count, method, dt)  # minimize -occ
    cur_val = obj_vec(cur.reshape(1,-1))[0]
    best_x = cur.copy(); best_val = float(cur_val)
    sigma = step_scale*span
    if log_interval is None:
        log_interval = max(1, iters//20)
    for k in range(iters):
        T = temp(k)
        cand = cur + np.random.randn(neighbor_batch, dim)*sigma
        cand = np.clip(cand, lo, hi)
        vals = obj_vec(cand)
        delta = vals - cur_val
        better = delta < 0
        if better.any():
            idx = np.argmin(vals)
        else:
            probs = np.exp(-delta/max(T,1e-12)); probs /= probs.sum()
            idx = np.random.choice(neighbor_batch, p=probs)
        new_x = cand[idx]; new_val = vals[idx]; d = new_val - cur_val
        if d < 0 or random.random() < math.exp(-float(d)/max(T,1e-12)):
            cur = new_x; cur_val = new_val
            if cur_val < best_val - 1e-15:
                best_val = float(cur_val); best_x = cur.copy()
    if verbose and (k % log_interval==0 or k==iters-1):
            print(f"[SA] {k+1}/{iters} cur={cur_val:.4f} best={best_val:.4f}")
    # authoritative eval
    occ = _scalar_occluded_time(problem, best_x.tolist(), bombs_count, method, dt)
    eval_res = {'total': occ} if problem>2 else evaluate_problem2(**decode_vector(2, best_x.tolist(), bombs_count).to_eval_kwargs(), occlusion_method=method, dt=dt)
    return SAResult(best_x=best_x, best_value=best_val, best_eval=eval_res)

# ============================= GA =============================
@dataclass
class GAResult:
    best_x: np.ndarray
    best_value: float
    best_eval: Dict

def solve_ga(*, problem: int=2, bombs_count: int=2, pop_size: int=40, generations: int=80, pc: float=0.8, pm: float=0.1, eta_c: float=20.0, eta_m: float=20.0, dt: float=0.02, method: str='judge_caps', seed: int|None=None, verbose: bool=False) -> GAResult:
    if seed is not None: random.seed(seed); np.random.seed(seed)
    bnds = bounds_for_problem(problem, bombs_count)
    lo = np.array([a for (a,_) in bnds]); hi = np.array([b for (_,b) in bnds])
    dim = len(bnds)
    def init_pop(): return lo + np.random.rand(pop_size, dim)*(hi-lo)
    def tournament(pop, fit, k=3):
        out = np.zeros_like(pop)
        for i in range(pop.shape[0]):
            idx = np.random.choice(pop.shape[0], k, replace=False)
            j = idx[np.argmin(fit[idx])]
            out[i] = pop[j]
        return out
    def sbx(p1,p2):
        c1=p1.copy(); c2=p2.copy()
        for i in range(dim):
            if random.random()<=pc and abs(p1[i]-p2[i])>1e-14:
                y1=min(p1[i],p2[i]); y2=max(p1[i],p2[i]); lb,ub=lo[i],hi[i]
                beta=1+2*(y1-lb)/(y2-y1); alpha=2-beta**-(eta_c+1)
                u=random.random()
                if u<=1/alpha: betaq=(u*alpha)**(1/(eta_c+1))
                else: betaq=(1/(2-u*alpha))**(1/(eta_c+1))
                c1[i]=0.5*((y1+y2)-betaq*(y2-y1))
                beta=1+2*(ub-y2)/(y2-y1); alpha=2-beta**-(eta_c+1)
                u=random.random()
                if u<=1/alpha: betaq=(u*alpha)**(1/(eta_c+1))
                else: betaq=(1/(2-u*alpha))**(1/(eta_c+1))
                c2[i]=0.5*((y1+y2)+betaq*(y2-y1))
        c1[i]=np.clip(c1[i],lo[i],hi[i]); c2[i]=np.clip(c2[i],lo[i],hi[i])
        return c1,c2
    def mutate(ind):
        for i in range(dim):
            if random.random()<=pm:
                y=ind[i]; lb,ub=lo[i],hi[i]; delta1=(y-lb)/(ub-lb); delta2=(ub-y)/(ub-lb)
                mut_pow=1/(eta_m+1); u=random.random()
                if u<0.5:
                    xy=1-delta1; val=2*u+(1-2*u)*(xy**(eta_m+1)); d=val**mut_pow-1
                else:
                    xy=1-delta2; val=2*(1-u)+2*(u-0.5)*(xy**(eta_m+1)); d=1-val**mut_pow
                ind[i]=np.clip(y+d*(ub-lb),lb,ub)
        return ind
    pop = init_pop(); occ = _batch_eval(problem,pop,bombs_count,method,dt); fit = -occ
    best_x = pop[np.argmin(fit)].copy(); best_val = fit.min()
    for g in range(generations):
        parents = tournament(pop, fit)
        kids=[]
        for i in range(0,pop_size,2):
            p1=parents[i]; p2=parents[(i+1)%pop_size]; c1,c2=sbx(p1,p2); kids.append(mutate(c1)); kids.append(mutate(c2))
        pop = np.vstack(kids)[:pop_size]
        occ = _batch_eval(problem,pop,bombs_count,method,dt); fit = -occ
        if fit.min()<best_val-1e-15:
            best_val=fit.min(); best_x=pop[np.argmin(fit)].copy()
        if verbose and (g%max(1,generations//10)==0 or g==generations-1):
            print(f"[GA] gen {g+1}/{generations} best={best_val:.4f}")
    final_occ = _scalar_occluded_time(problem,best_x.tolist(),bombs_count,method,dt)
    eval_res={'total':final_occ} if problem>2 else evaluate_problem2(**decode_vector(2,best_x.tolist(),bombs_count).to_eval_kwargs(), occlusion_method=method, dt=dt)
    return GAResult(best_x=best_x, best_value=best_val, best_eval=eval_res)

# ============================= PSO =============================
@dataclass
class PSOResult:
    best_x: np.ndarray
    best_value: float
    best_eval: Dict

def solve_pso(*, problem: int=2, bombs_count: int=2, pop: int=60, iters: int=150, w: float=0.65, c1: float=1.6, c2: float=1.6, vmax_frac: float=0.6, dt: float=0.02, method: str='judge_caps', seed: int|None=None, verbose: bool=False) -> PSOResult:
    if seed is not None: random.seed(seed); np.random.seed(seed)
    bnds = bounds_for_problem(problem, bombs_count)
    lo = np.array([a for (a,_) in bnds]); hi = np.array([b for (_,b) in bnds])
    span = hi-lo; dim=len(bnds); vmax=vmax_frac*span
    X = lo + np.random.rand(pop,dim)*span
    V = (np.random.rand(pop,dim)-0.5)*2*vmax
    occ = _batch_eval(problem,X,bombs_count,method,dt); fit=-occ
    P = X.copy(); pbest = fit.copy(); gi=int(np.argmin(fit)); g=X[gi].copy(); gbest=float(fit[gi])
    for it in range(iters):
        r1=np.random.rand(pop,dim); r2=np.random.rand(pop,dim)
        V = w*V + c1*r1*(P-X) + c2*r2*(g-X)
        V = np.clip(V, -vmax, vmax); X = X + V; X = np.clip(X, lo, hi)
        occ = _batch_eval(problem,X,bombs_count,method,dt); fit=-occ
        improved = fit < pbest
        if improved.any():
            pbest[improved]=fit[improved]; P[improved]=X[improved]
            gi = int(np.argmin(pbest));
            if pbest[gi] < gbest - 1e-15:
                gbest=float(pbest[gi]); g=P[gi].copy()
        if verbose and (it%max(1,iters//10)==0 or it==iters-1):
            print(f"[PSO] {it+1}/{iters} best={gbest:.4f}")
    final_occ = _scalar_occluded_time(problem,g.tolist(),bombs_count,method,dt)
    eval_res={'total':final_occ} if problem>2 else evaluate_problem2(**decode_vector(2,g.tolist(),bombs_count).to_eval_kwargs(), occlusion_method=method, dt=dt)
    return PSOResult(best_x=g, best_value=gbest, best_eval=eval_res)

__all__ = ['solve_sa','solve_ga','solve_pso','SAResult','GAResult','PSOResult']
