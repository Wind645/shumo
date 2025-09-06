from __future__ import annotations
"""Problem 4: FY1,FY2,FY3 each drop 1 bomb to maximize occlusion for M1."""
import json, time
from pathlib import Path
from optimizer.algorithms import solve_sa, solve_ga, solve_pso
from optimizer.spec import decode_vector
from api.problems import evaluate_problem4

ALGO = 'sa'           # 'sa' | 'ga' | 'pso'
DT = 0.02
METHOD = 'judge_caps'
SEED = None
VERBOSE = True

# SA
SA_ITERS = 8000
SA_NEIGHBOR_BATCH = 640
SA_STEP_SCALE = 0.25
SA_INIT_TEMP = 2.5
SA_FINAL_TEMP = 1e-3
SA_LOG_INTERVAL = 400

# GA
GA_GENERATIONS = 150
GA_POP = 90

# PSO
PSO_ITERS = 400
PSO_POP = 100
LOG_DIR = Path('log'); LOG_DIR.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    if ALGO == 'sa':
        res = solve_sa(problem=4, bombs_count=1, iters=SA_ITERS, dt=DT, method=METHOD,
                       neighbor_batch=SA_NEIGHBOR_BATCH, step_scale=SA_STEP_SCALE,
                       init_temp=SA_INIT_TEMP, final_temp=SA_FINAL_TEMP, log_interval=SA_LOG_INTERVAL,
                       seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    elif ALGO == 'ga':
        res = solve_ga(problem=4, bombs_count=1, generations=GA_GENERATIONS, pop_size=GA_POP,
                       dt=DT, method=METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    elif ALGO == 'pso':
        res = solve_pso(problem=4, bombs_count=1, iters=PSO_ITERS, pop=PSO_POP,
                        dt=DT, method=METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    else:
        raise ValueError('未知 ALGO')
    dec = decode_vector(4, best_x.tolist(), 1)
    eval_full = evaluate_problem4(**dec.to_eval_kwargs(), dt=DT, occlusion_method=METHOD)
    out = {
        'algo': ALGO,
        'best_x': best_x.tolist(),
        'decoded': dec.drones_spec,
        'occluded_time': eval_full['occluded_time'],
        'total': eval_full.get('total', sum(eval_full['occluded_time'].values())),
        'dt': DT,
        'method': METHOD,
        'seconds_used': time.time()-t0,
    }
    fp = LOG_DIR / 'q4_result.json'
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    if VERBOSE:
        print('Q4 optimization finished. Result saved to', fp)
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
