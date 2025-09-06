from __future__ import annotations
"""Unified configurable runner (replacement for old optimizer.router).

Edit the constants below then run:
  uv run python -m problems.runner

Configuration covers Problems 2–5 and algorithms: 'sa','ga','pso','hybrid' (hybrid only for Q2).
No argparse used; just modify and save.
"""
import json, time, math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from optimizer.algorithms import solve_sa, solve_ga, solve_pso
from optimizer.spec import decode_vector
from api.problems import (
    evaluate_problem2, evaluate_problem3, evaluate_problem4, evaluate_problem5
)

# ======================= 可编辑区 =======================
PROBLEM = 2                 # 2 / 3 / 4 / 5
ALGO = 'sa'                 # 'sa' | 'ga' | 'pso' | 'hybrid'(仅Q2)
BOMBS_COUNT = 3             # 题3/5 时炸弹数量 (Q2 固定1, Q4 固定1)
DT = 0.02                   # 时间步长
OCCLUSION_METHOD = 'judge_caps'  # 'sampling'|'judge_caps'|'rough_caps'|'vectorized_sampling'
SEED = None                 # 随机种子
VERBOSE = True              # 是否打印过程
# ------ SA 参数 ------
SA_ITERS = 4000
SA_NEIGHBOR_BATCH = 512
SA_STEP_SCALE = 0.25
SA_INIT_TEMP = 2.0
SA_FINAL_TEMP = 1e-3
SA_LOG_INTERVAL = 200
# ------ GA 参数 ------
GA_GENERATIONS = 80
GA_POP = 60
# ------ PSO 参数 ------
PSO_ITERS = 200
PSO_POP = 80
# ------ HYBRID (Q2 only, 基于 optimizer.hybrid.solve_q2_hybrid) ------
HYBRID_SA_ITERS = 20000
HYBRID_SA_BATCH = 256
HYBRID_SA_STEP = 0.20
HYBRID_DT = 0.01
HYBRID_BACKEND = 'vectorized_torch_sampled'  # rough|vectorized|vectorized_torch|vectorized_torch_newton|vectorized_torch_sampled
HYBRID_LOG_INTERVAL = 100
HYBRID_USE_GA = False
HYBRID_GA_GENS = 150
HYBRID_USE_LOCAL_PSO = True
# =======================================================

LOG_DIR = Path('log'); LOG_DIR.mkdir(exist_ok=True)


def _eval_problem(dec, *, dt: float, method: str) -> Dict[str, Any]:
    if PROBLEM == 2:
        return evaluate_problem2(**dec.to_eval_kwargs(), dt=dt, occlusion_method=method)
    if PROBLEM == 3:
        return evaluate_problem3(**dec.to_eval_kwargs(), dt=dt, occlusion_method=method)
    if PROBLEM == 4:
        return evaluate_problem4(**dec.to_eval_kwargs(), dt=dt, occlusion_method=method)
    if PROBLEM == 5:
        return evaluate_problem5(**dec.to_eval_kwargs(), dt=dt, occlusion_method=method)
    raise ValueError('Unsupported PROBLEM')


def run():
    global BOMBS_COUNT
    t0 = time.time()
    if PROBLEM == 2:
        BOMBS_COUNT = 1
    elif PROBLEM == 4:
        BOMBS_COUNT = 1
    # 选择算法
    if ALGO == 'sa':
        res = solve_sa(problem=PROBLEM, bombs_count=BOMBS_COUNT, iters=SA_ITERS, dt=DT,
                       method=OCCLUSION_METHOD, neighbor_batch=SA_NEIGHBOR_BATCH,
                       step_scale=SA_STEP_SCALE, init_temp=SA_INIT_TEMP, final_temp=SA_FINAL_TEMP,
                       log_interval=SA_LOG_INTERVAL, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x; dec = decode_vector(PROBLEM, best_x.tolist(), BOMBS_COUNT)
        raw_eval = _eval_problem(dec, dt=DT, method=OCCLUSION_METHOD)
        total_occ = raw_eval.get('total', raw_eval['occluded_time'].get('M1'))
    elif ALGO == 'ga':
        res = solve_ga(problem=PROBLEM, bombs_count=BOMBS_COUNT, generations=GA_GENERATIONS, pop_size=GA_POP,
                       dt=DT, method=OCCLUSION_METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x; dec = decode_vector(PROBLEM, best_x.tolist(), BOMBS_COUNT)
        raw_eval = _eval_problem(dec, dt=DT, method=OCCLUSION_METHOD)
        total_occ = raw_eval.get('total', raw_eval['occluded_time'].get('M1'))
    elif ALGO == 'pso':
        res = solve_pso(problem=PROBLEM, bombs_count=BOMBS_COUNT, iters=PSO_ITERS, pop=PSO_POP,
                        dt=DT, method=OCCLUSION_METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x; dec = decode_vector(PROBLEM, best_x.tolist(), BOMBS_COUNT)
        raw_eval = _eval_problem(dec, dt=DT, method=OCCLUSION_METHOD)
        total_occ = raw_eval.get('total', raw_eval['occluded_time'].get('M1'))
    elif ALGO == 'hybrid':
        if PROBLEM != 2:
            raise ValueError('hybrid 目前仅支持 Problem 2')
        from optimizer.hybrid import solve_q2_hybrid
        hres = solve_q2_hybrid(
            sa_iters=HYBRID_SA_ITERS, sa_neighbor_batch=HYBRID_SA_BATCH, sa_step_scale=HYBRID_SA_STEP,
            sa_init_temp=2.0, sa_final_temp=1e-3, dt=HYBRID_DT, log_interval=HYBRID_LOG_INTERVAL,
            checkpoint_path='sa_checkpoint.pt', resume=True, use_ga=HYBRID_USE_GA,
            ga_generations=HYBRID_GA_GENS, seed=SEED, verbose=VERBOSE,
            judge_backend=HYBRID_BACKEND, use_local_pso=HYBRID_USE_LOCAL_PSO
        )
        best_x = hres.best_x
        from optimizer.spec import decode_vector as _dec
        dec = _dec(2, best_x.tolist(), 1)
        raw_eval = evaluate_problem2(**dec.to_eval_kwargs(), dt=HYBRID_DT, occlusion_method='judge_caps')
        total_occ = raw_eval.get('total', raw_eval['occluded_time'].get('M1'))
    else:
        raise ValueError('未知 ALGO')

    elapsed = time.time() - t0
    out = dict(
        problem=PROBLEM,
        algo=ALGO,
        occlusion_method=OCCLUSION_METHOD,
        dt=DT,
        best_x=best_x.tolist(),
        decoded=dec.drones_spec,
        occluded_time=raw_eval.get('occluded_time'),
        total=total_occ,
        elapsed_sec=elapsed,
    )
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = LOG_DIR / f'problem{PROBLEM}_{ALGO}_{ts}.json'
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    if VERBOSE:
        print(f'[RUNNER] 完成 Problem {PROBLEM} {ALGO} total={total_occ:.4f} (保存 {fp})')
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
