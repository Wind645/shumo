from __future__ import annotations
"""简化版 SA + (可选) GA。仅保留 fused 粗评估与最小必要逻辑。"""
import os, math, random, numpy as np, torch
from dataclasses import dataclass
from typing import Optional, Dict
from simcore.constants import (
    SMOKE_LIFETIME_DEFAULT as _SMOKE_LIFE,
    SMOKE_DESCENT_DEFAULT as _SMOKE_DESCENT,
    SMOKE_RADIUS_DEFAULT as _SMOKE_RADIUS,
    R_CYL_DEFAULT as _R_CYL,
    H_CYL_DEFAULT as _H_CYL,
    C_BASE_DEFAULT as _C_BASE,
    G as _G,
)
from api.problems import evaluate_problem2
from optimizer.ga_q2 import solve_q2_ga, BOUNDS_DEFAULT, _batch_occluded_time_numpy as _ga_batch_occluded_time_numpy

# ================= 简单可调参数 =================
# 直接修改下面这些变量后保存，再用 `uv run -m optimizer.hybrid_sa_ga_q2` 运行。
SA_ITERS = 40000          # 模拟退火迭代数
SA_BATCH = 1           # 每步候选数量 (GPU 批大小)
SA_STEP_SCALE = 0.20      # 相对步长 (乘以区间 span)
SA_INIT_TEMP = 2.0        # 初始温度
SA_FINAL_TEMP = 1e-3      # 终止温度
SA_DT = 0.005              # 时间步长 (越小越精细, 计算更慢)
SA_LOG_INTERVAL = 10     # 日志间隔
CHECKPOINT_PATH = 'sa_checkpoint.pt'  # 断点文件 (None 关闭)
RESUME = True             # 若存在断点文件是否读取 best_x 继续（重新计数）
USE_GA = False            # 是否在 SA 后跑 GA 微调
GA_GENERATIONS = 200      # GA 迭代次数 (仅当 USE_GA=True)

USE_LOCAL_PSO = True     # 是否启用局部 PSO 细化
LOCAL_PSO_POP = 64        # 局部 PSO 粒子数量
LOCAL_PSO_ITERS = 15      # 局部 PSO 迭代次数 (注意过大会显著变慢)
LOCAL_PSO_TOPK = 16       # 从当前 SA batch 中选取前 K 个作为种子
LOCAL_PSO_INIT_NOISE = 0.02  # 初始扰动幅度 (相对区间 span)
LOCAL_PSO_VMAX_FRAC = 0.2    # 速度上限 (相对区间 span)
LOCAL_PSO_W = 0.5
LOCAL_PSO_C1 = 1.5
LOCAL_PSO_C2 = 1.5
SEED = None               # 随机种子 (None 表示不固定)
DEVICE = None             # 指定 'cuda' 或 'cpu'; None 自动选择
VERBOSE = True            # 是否打印过程
JUDGE_BACKEND = 'vectorized_torch_sampled'   # 遮挡评估后端: rough | vectorized | vectorized_torch | vectorized_torch_newton | vectorized_torch_sampled
# =================================================

_MISSILE_POS0 = torch.tensor([20000.0,0.0,2000.0])
_MISSILE_SPEED = 300.0
_MISSILE_TARGET = torch.tensor([0.0,0.0,0.0])
def _missile_dir_and_flight_time():
    d = _MISSILE_TARGET - _MISSILE_POS0; n = torch.linalg.norm(d); return d/n, n/_MISSILE_SPEED
_MISSILE_DIR, _MISSILE_T_FLIGHT = _missile_dir_and_flight_time()

def _fused_occluded_time(params: torch.Tensor, dt: float, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        if params.numel()==0:
            return torch.zeros(0, device=device)
        p = params.to(device=device, dtype=torch.float32)
        speed, az, t_rel, dly = p[:,0], p[:,1], p[:,2], p[:,3]
        valid = (speed>=70)&(speed<=140)&(t_rel>=0)&(dly>0)
        N = p.shape[0]
        out = torch.zeros(N, device=device)
        if not valid.any():
            return out
        T_f = float(_MISSILE_T_FLIGHT); n_steps = int(T_f/dt)+1
        tg = torch.linspace(0,T_f,n_steps,device=device)
        miss = _MISSILE_POS0.to(device) + _MISSILE_DIR.to(device)*(_MISSILE_SPEED*tg).unsqueeze(1)  # (T,3)
        drone0 = torch.tensor([17800.0,0.0,1800.0], device=device).view(1,3).repeat(N,1)
        ddir = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)],1)
        a = torch.tensor([0,0,-_G], device=device)
        rel = drone0 + ddir*speed.unsqueeze(1)*t_rel.unsqueeze(1)
        vel = ddir*speed.unsqueeze(1)
        c0 = rel + vel*dly.unsqueeze(1) + 0.5*a*(dly.unsqueeze(1)**2)
        explode = t_rel + dly; end = explode + _SMOKE_LIFE
        tg_row = tg.unsqueeze(0)
        active = valid.unsqueeze(1) & (tg_row>=explode.unsqueeze(1)) & (tg_row<=end.unsqueeze(1))
        if not active.any():
            return out
        tau = torch.clamp(tg_row - explode.unsqueeze(1), min=0)
        ct = c0.unsqueeze(1).expand(N,n_steps,3).clone(); ct[:,:,2]=c0[:,2].unsqueeze(1)-_SMOKE_DESCENT*tau
        V = miss.unsqueeze(0).expand(N,n_steps,3)
        Cb = torch.as_tensor(_C_BASE, device=device, dtype=torch.float32)
    r_cyl = float(_R_CYL); shift = float(_H_CYL); R = float(_SMOKE_RADIUS)
    SCb = ct - V; CCb = Cb.view(1,1,3)-V
    dSb = torch.linalg.norm(SCb, dim=2); dCb = torch.linalg.norm(CCb, dim=2)
    ratio_s = torch.clamp(R/(dSb+1e-9),0,1); ratio_c = torch.clamp(r_cyl/(dCb+1e-9),0,1)
    uS = SCb/(dSb.unsqueeze(2)+1e-9); uC = CCb/(dCb.unsqueeze(2)+1e-9)
    cosg = (uS*uC).sum(2).clamp(-1,1); g = torch.arccos(cosg)
    b = torch.arcsin(ratio_c); apha = torch.arcsin(ratio_s)
    oc_b = (g+b)<=(apha+1e-6)
    Vt = V.clone(); Vt[:,:,2]-=shift; Ct = torch.tensor([Cb[0],Cb[1],0.0], device=device)
    SCt = ct.clone(); SCt[:,:,2]-=shift; SCt -= Vt; CCt = Ct.view(1,1,3)-Vt
    dSt = torch.linalg.norm(SCt, dim=2); dCt = torch.linalg.norm(CCt, dim=2)
    ratio_s2 = torch.clamp(R/(dSt+1e-9),0,1); ratio_c2 = torch.clamp(r_cyl/(dCt+1e-9),0,1)
    uS2 = SCt/(dSt.unsqueeze(2)+1e-9); uC2 = CCt/(dCt.unsqueeze(2)+1e-9)
    cosg2 = (uS2*uC2).sum(2).clamp(-1,1); g2 = torch.arccos(cosg2)
    b2 = torch.arcsin(ratio_c2); apha2 = torch.arcsin(ratio_s2)
    oc_t = (g2+b2)<=(apha2+1e-6)
    full = active & oc_b & oc_t
    out[valid] = full[valid].sum(1)*dt
    return out

def _vectorized_occluded_time(params: torch.Tensor, dt: float, device: torch.device) -> torch.Tensor:
    """使用 NumPy 精确 (vectorized_judge) 版本估计遮挡时间。

    为避免重复代码，复用 GA 模块中 `_batch_occluded_time_numpy` 的实现；该实现
    在 CPU 上运行，可能比 GPU rough 版本慢一个数量级，只在需要更精确
    的退火打分时选择。
    """
    if params.numel() == 0:
        return torch.zeros(0, device=device)
    arr = params.detach().cpu().numpy().astype(float)
    occ = _ga_batch_occluded_time_numpy(arr, dt=dt)  # (N,) occluded_time
    return torch.as_tensor(occ, device=device, dtype=torch.float32)

@dataclass
class HybridResult:
    best_x: np.ndarray
    best_value: float
    best_eval: Dict

def solve_q2_hybrid(*, sa_iters=20000, sa_neighbor_batch=1024, sa_step_scale=0.2,
                    sa_init_temp=2.0, sa_final_temp=1e-3, dt=0.02, log_interval=100,
                    checkpoint_path: Optional[str] = 'sa_checkpoint.pt', resume=True,
                    use_ga=False, ga_generations=200, seed: Optional[int]=None,
                    device: Optional[str]=None, verbose=True,
                    judge_backend: str = 'rough',
                    # 局部 PSO 相关配置（与全局常量同名时可覆盖）
                    use_local_pso: Optional[bool]=None,
                    local_pso_pop: int = LOCAL_PSO_POP,
                    local_pso_iters: int = LOCAL_PSO_ITERS,
                    local_pso_topk: int = LOCAL_PSO_TOPK,
                    local_pso_init_noise: float = LOCAL_PSO_INIT_NOISE,
                    local_pso_vmax_frac: float = LOCAL_PSO_VMAX_FRAC,
                    local_pso_w: float = LOCAL_PSO_W,
                    local_pso_c1: float = LOCAL_PSO_C1,
                    local_pso_c2: float = LOCAL_PSO_C2,
                    ) -> HybridResult:
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    lo = torch.tensor([b[0] for b in BOUNDS_DEFAULT], device=dev); hi = torch.tensor([b[1] for b in BOUNDS_DEFAULT], device=dev)
    span = hi - lo
    if resume and checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            ck = torch.load(checkpoint_path, map_location=dev, weights_only=False)
            cur_x = ck['best_x'].to(dev).float()
            if verbose: print(f"[SA] resume best_x prev_val={ck['best_val']:.6f}")
        except Exception:
            cur_x = lo + torch.rand(4, device=dev)*span
    else:
        cur_x = lo + torch.rand(4, device=dev)*span
    def _temp(k):
        a = k/max(1, sa_iters-1); return sa_init_temp*((sa_final_temp/sa_init_temp)**a)
    # 选择遮挡评估后端
    if judge_backend == 'rough':
        _eval_backend = lambda P: _fused_occluded_time(P, dt, dev)
    elif judge_backend == 'vectorized':
        _eval_backend = lambda P: _vectorized_occluded_time(P, dt, dev)
    elif judge_backend == 'vectorized_torch':
        try:
            from vectorized_judge_torch import batch_occluded_time_caps_torch as _batch_caps_torch
        except ImportError as e:
            raise ImportError("需要 vectorized_judge_torch 模块") from e
        _eval_backend = lambda P: _batch_caps_torch(P, dt=dt, device=dev)
    elif judge_backend == 'vectorized_torch_newton':
        try:
            from vectorized_judge_torch import batch_occluded_time_caps_torch_newton as _batch_caps_newton
        except ImportError as e:
            raise ImportError("需要 vectorized_judge_torch 模块 (newton)") from e
        _eval_backend = lambda P: _batch_caps_newton(P, dt=dt, device=dev)
    elif judge_backend == 'vectorized_torch_sampled':
        try:
            from vectorized_judge_torch_sampled import batch_occluded_time_caps_torch_sampled as _batch_caps_sampled
        except ImportError as e:
            raise ImportError("需要 vectorized_judge_torch_sampled 模块 (采样版本)") from e
        _eval_backend = lambda P: _batch_caps_sampled(P, dt=dt, device=dev)
    else:
        raise ValueError(f"未知 JUDGE_BACKEND={judge_backend}, 必须是 'rough'|'vectorized'|'vectorized_torch'|'vectorized_torch_newton'|'vectorized_torch_sampled'")

    val = -_eval_backend(cur_x.unsqueeze(0))[0]
    best_x = cur_x.clone(); best_val = float(val)
    sigma = sa_step_scale * span
    # 允许外部参数覆盖全局是否启用局部 PSO
    if use_local_pso is None:
        use_local_pso = USE_LOCAL_PSO

    def _local_pso_refine(batch_params: torch.Tensor, batch_vals: torch.Tensor, base_sigma: torch.Tensor):
        """对当前 SA 生成的 batch (cand) 进行局部 PSO 细化。

        batch_params: (B,4) 候选参数 (已在 lo-hi 之间)
        batch_vals:   (B,) 目标函数值 (已为最小化形式: -occluded_time)
        base_sigma:   (4,) 该轮使用的步长尺度 (用于设定初始扰动)
        返回: (best_x_refined, best_val_refined)
        """
        B = batch_params.shape[0]
        K = min(local_pso_topk, B)
        # 选出当前 batch 中值最好的 K 个作为中心
        topk_idx = torch.topk(-batch_vals, k=K).indices  # -vals 因为值小=好
        seeds = batch_params[topk_idx].detach().cpu().numpy()  # (K,4)
        span_np = (hi - lo).cpu().numpy().astype(float)
        lo_np = lo.cpu().numpy().astype(float); hi_np = hi.cpu().numpy().astype(float)
        # 构造初始种群：重复 seeds 并加噪声
        reps = int(math.ceil(local_pso_pop / K))
        init = np.tile(seeds, (reps, 1))[:local_pso_pop]
        noise_scale = local_pso_init_noise * span_np
        init = init + np.random.randn(*init.shape) * noise_scale
        init = np.clip(init, lo_np, hi_np)
        # 初始化速度
        vmax = local_pso_vmax_frac * span_np
        V = (np.random.rand(local_pso_pop, 4) - 0.5) * 2.0 * vmax
        X = init.copy()
        # 评估函数 (批量) 复用 _eval_backend
        def eval_batch_np(arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.as_tensor(arr, device=dev, dtype=torch.float32)
                return (-_eval_backend(t)).cpu().numpy().astype(float)
        pbest_pos = X.copy(); pbest_val = eval_batch_np(X)
        gidx = int(np.argmin(pbest_val)); gbest_pos = pbest_pos[gidx].copy(); gbest_val = float(pbest_val[gidx])
        for itp in range(local_pso_iters):
            r1 = np.random.rand(local_pso_pop,4)
            r2 = np.random.rand(local_pso_pop,4)
            V = (local_pso_w * V
                 + local_pso_c1 * r1 * (pbest_pos - X)
                 + local_pso_c2 * r2 * (gbest_pos - X))
            V = np.maximum(np.minimum(V, vmax), -vmax)
            X = X + V
            # 边界处理
            below = X < lo_np; above = X > hi_np
            if below.any() or above.any():
                X = np.maximum(np.minimum(X, hi_np), lo_np)
                V[below] *= -0.5; V[above] *= -0.5
            vals = eval_batch_np(X)
            improved = vals < pbest_val
            if improved.any():
                pbest_val[improved] = vals[improved]; pbest_pos[improved] = X[improved]
                gi = int(np.argmin(pbest_val))
                if float(pbest_val[gi]) < gbest_val - 1e-15:
                    gbest_val = float(pbest_val[gi]); gbest_pos = pbest_pos[gi].copy()
        # 返回 torch 张量形式
        return torch.as_tensor(gbest_pos, device=dev, dtype=torch.float32), gbest_val

    for k in range(sa_iters):
        T=_temp(k)
        cand = cur_x.unsqueeze(0)+torch.randn(sa_neighbor_batch,4,device=dev)*sigma
        cand = torch.minimum(torch.maximum(cand, lo), hi)
        vals = -_eval_backend(cand)
        # 在选择前执行局部 PSO 细化（可选）
        if use_local_pso and k>0:  # 跳过第一轮 (没有足够上下文)
            try:
                pso_best_x, pso_best_val = _local_pso_refine(cand, vals, sigma)
                if pso_best_val < vals.min() - 1e-15:
                    # 把 pso 最优插入 cand 中 (替换最差一个)
                    worst_idx = torch.argmax(vals)
                    cand[worst_idx] = pso_best_x
                    vals[worst_idx] = pso_best_val
                    if verbose:
                        print(f"[LocalPSO] k={k} improved batch best to {pso_best_val:.6f}")
            except Exception as _e:
                if verbose:
                    print(f"[LocalPSO] 失败: {_e}")
        delta = vals - val
        better = delta<0
        if better.any():
            chosen = torch.nonzero(better).squeeze(1)[torch.argmin(vals[better])]
        else:
            probs = torch.softmax(-delta/max(T,1e-12),0); chosen = torch.multinomial(probs,1)[0]
        nv = vals[chosen]; nx = cand[chosen]; d = nv - val
        if d<0 or random.random()<math.exp(-float(d)/max(T,1e-12)):
            cur_x, val = nx, nv
            if float(val) < best_val - 1e-15:
                best_val = float(val); best_x = cur_x.clone()
        if (k % log_interval==0) or (k==sa_iters-1):
            if verbose: print(f"[SA] {k}/{sa_iters} T={T:.4f} cur={float(val):.6f} best={best_val:.6f}")
            if checkpoint_path:
                try: torch.save({'best_x':best_x.cpu(),'best_val':best_val}, checkpoint_path)
                except Exception: pass
    bx = best_x.cpu().numpy().astype(float)
    try:
        bev = evaluate_problem2(speed=float(bx[0]), azimuth=float(bx[1]), release_time=float(bx[2]), explode_delay=float(bx[3]), occlusion_method='judge_caps', dt=dt)
    except Exception:
        bev = {'occluded_time': {'M1': float(-best_val)}, 'total': float(-best_val)}
    if use_ga:
        init_pop = np.tile(bx,(512,1)) + np.random.randn(512,4)*0.01
        init_pop = np.clip(init_pop, lo.cpu().numpy(), hi.cpu().numpy())
        ga_res = solve_q2_ga(
            pop_size=512,
            n_generations=ga_generations,
            initial_population=init_pop,
            dt=dt,
            method='judge_caps',
            use_vectorized=True,
            verbose=False,
            judge_backend=judge_backend,
            device=device,
        )
        bx = ga_res.best_x; best_val = ga_res.best_value; bev = ga_res.best_eval
    if checkpoint_path:
        try: torch.save({'best_x': torch.tensor(bx), 'best_val': best_val}, checkpoint_path)
        except Exception: pass
    if verbose: print('[SA DONE] best', bx, 'val', best_val)
    return HybridResult(best_x=bx, best_value=best_val, best_eval=bev)

if __name__ == '__main__':
    res = solve_q2_hybrid(
        sa_iters=SA_ITERS,
        sa_neighbor_batch=SA_BATCH,
        sa_step_scale=SA_STEP_SCALE,
        sa_init_temp=SA_INIT_TEMP,
        sa_final_temp=SA_FINAL_TEMP,
        dt=SA_DT,
        log_interval=SA_LOG_INTERVAL,
        checkpoint_path=CHECKPOINT_PATH,
        resume=RESUME,
        use_ga=USE_GA,
        ga_generations=GA_GENERATIONS,
        seed=SEED,
        device=DEVICE,
        verbose=VERBOSE,
        judge_backend=JUDGE_BACKEND,
    use_local_pso=USE_LOCAL_PSO,
    local_pso_pop=LOCAL_PSO_POP,
    local_pso_iters=LOCAL_PSO_ITERS,
    local_pso_topk=LOCAL_PSO_TOPK,
    local_pso_init_noise=LOCAL_PSO_INIT_NOISE,
    local_pso_vmax_frac=LOCAL_PSO_VMAX_FRAC,
    local_pso_w=LOCAL_PSO_W,
    local_pso_c1=LOCAL_PSO_C1,
    local_pso_c2=LOCAL_PSO_C2,
    )
    print('BEST_X =', res.best_x, 'BEST_VALUE =', res.best_value)
