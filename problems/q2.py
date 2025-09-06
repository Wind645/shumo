from __future__ import annotations
"""Problem 2 optimization script.
Tune FY1 speed / azimuth / release_time / explode_delay to maximize occluded time for M1.
"""
import json, time
from pathlib import Path
from optimizer.algorithms import solve_sa, solve_ga, solve_pso
from optimizer.spec import decode_vector
from api.problems import evaluate_problem2
from simcore.constants import DRONES_POS0

# ================= 可编辑区：算法/参数选择 =================
ALGO = 'sa'              # 'sa' | 'ga' | 'pso' | 'hybrid'
DT = 0.02                # 时间步长 (sa/ga/pso 共用)
METHOD = 'judge_caps'    # 'sampling' | 'judge_caps' | 'rough_caps' | 'vectorized_sampling'
SEED = None
VERBOSE = True

# --- SA 参数 ---
SA_ITERS = 4000
SA_NEIGHBOR_BATCH = 512
SA_STEP_SCALE = 0.25
SA_INIT_TEMP = 2.0
SA_FINAL_TEMP = 1e-3
SA_LOG_INTERVAL = 200

# --- GA 参数 ---
GA_GENERATIONS = 80
GA_POP = 60

# --- PSO 参数 ---
PSO_ITERS = 200
PSO_POP = 80

# --- HYBRID (仅 Problem 2) 参数 (引用 optimizer.hybrid) ---
HYBRID_SA_ITERS = 20000
HYBRID_SA_BATCH = 256
HYBRID_SA_STEP = 0.20
HYBRID_DT = 0.01
HYBRID_BACKEND = 'vectorized_torch_sampled'  # rough|vectorized|vectorized_torch|vectorized_torch_newton|vectorized_torch_sampled
HYBRID_LOG_INTERVAL = 100
HYBRID_USE_GA = False
HYBRID_GA_GENS = 150
HYBRID_USE_LOCAL_PSO = True
# ==========================================================

LOG_DIR = Path('log'); LOG_DIR.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    # 根据 ALGO 选择算法
    if ALGO == 'sa':
        res = solve_sa(problem=2, bombs_count=1, iters=SA_ITERS, dt=DT, method=METHOD,
                       neighbor_batch=SA_NEIGHBOR_BATCH, step_scale=SA_STEP_SCALE,
                       init_temp=SA_INIT_TEMP, final_temp=SA_FINAL_TEMP, log_interval=SA_LOG_INTERVAL,
                       seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
        dec = decode_vector(2, best_x.tolist(), 1)
        eval_full = evaluate_problem2(**dec.to_eval_kwargs(), dt=DT, occlusion_method=METHOD)
    elif ALGO == 'ga':
        res = solve_ga(problem=2, bombs_count=1, generations=GA_GENERATIONS, pop_size=GA_POP,
                       dt=DT, method=METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
        dec = decode_vector(2, best_x.tolist(), 1)
        eval_full = evaluate_problem2(**dec.to_eval_kwargs(), dt=DT, occlusion_method=METHOD)
    elif ALGO == 'pso':
        res = solve_pso(problem=2, bombs_count=1, iters=PSO_ITERS, pop=PSO_POP,
                        dt=DT, method=METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
        dec = decode_vector(2, best_x.tolist(), 1)
        eval_full = evaluate_problem2(**dec.to_eval_kwargs(), dt=DT, occlusion_method=METHOD)
    elif ALGO == 'hybrid':
        from optimizer.hybrid import solve_q2_hybrid
        hres = solve_q2_hybrid(
            sa_iters=HYBRID_SA_ITERS, sa_neighbor_batch=HYBRID_SA_BATCH, sa_step_scale=HYBRID_SA_STEP,
            sa_init_temp=2.0, sa_final_temp=1e-3, dt=HYBRID_DT, log_interval=HYBRID_LOG_INTERVAL,
            checkpoint_path='sa_checkpoint.pt', resume=True, use_ga=HYBRID_USE_GA,
            ga_generations=HYBRID_GA_GENS, seed=SEED, verbose=VERBOSE,
            judge_backend=HYBRID_BACKEND, use_local_pso=HYBRID_USE_LOCAL_PSO
        )
        best_x = hres.best_x
        dec = decode_vector(2, best_x.tolist(), 1)
        eval_full = evaluate_problem2(**dec.to_eval_kwargs(), dt=HYBRID_DT, occlusion_method='judge_caps')
        # 覆盖 DT 用于输出
        if DT != HYBRID_DT:
            DT_local = HYBRID_DT
        else:
            DT_local = DT
        DT_out = DT_local
    else:
        raise ValueError('未知 ALGO')
    DT_out = locals().get('DT_out', DT)
    out = {
        'algo': ALGO,
        'best_x': best_x.tolist(),
        'decoded': dec.drones_spec,
        'occluded_time': eval_full['occluded_time'],
        'total': eval_full.get('total', eval_full['occluded_time'].get('M1')),
        'dt': DT_out,
        'method': METHOD,
        'seconds_used': time.time()-t0,
    }
    fp = LOG_DIR / 'q2_result.json'
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    if VERBOSE:
        print('Q2 optimization finished. Result saved to', fp)
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
