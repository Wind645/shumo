from __future__ import annotations
"""Problem 3: FY1 throws 3 bombs (adjust BOMBS_COUNT) to maximize occlusion for M1.
Result should mirror original problem 3 requirement and can be exported.
"""
import json, time
from pathlib import Path
from optimizer.algorithms import solve_sa, solve_ga, solve_pso
from optimizer.spec import decode_vector
from api.problems import evaluate_problem3

ALGO = 'sa'            # 'sa' | 'ga' | 'pso'
DT = 0.02
METHOD = 'judge_caps'
BOMBS_COUNT = 3
SEED = None
VERBOSE = True

# SA
SA_ITERS = 6000
SA_NEIGHBOR_BATCH = 512
SA_STEP_SCALE = 0.25
SA_INIT_TEMP = 2.0
SA_FINAL_TEMP = 1e-3
SA_LOG_INTERVAL = 300

# GA
GA_GENERATIONS = 120
GA_POP = 80

# PSO
PSO_ITERS = 300
PSO_POP = 90
LOG_DIR = Path('log'); LOG_DIR.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    if ALGO == 'sa':
        res = solve_sa(problem=3, bombs_count=BOMBS_COUNT, iters=SA_ITERS, dt=DT, method=METHOD,
                       neighbor_batch=SA_NEIGHBOR_BATCH, step_scale=SA_STEP_SCALE,
                       init_temp=SA_INIT_TEMP, final_temp=SA_FINAL_TEMP, log_interval=SA_LOG_INTERVAL,
                       seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    elif ALGO == 'ga':
        res = solve_ga(problem=3, bombs_count=BOMBS_COUNT, generations=GA_GENERATIONS, pop_size=GA_POP,
                       dt=DT, method=METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    elif ALGO == 'pso':
        res = solve_pso(problem=3, bombs_count=BOMBS_COUNT, iters=PSO_ITERS, pop=PSO_POP,
                        dt=DT, method=METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    else:
        raise ValueError('未知 ALGO')
    dec = decode_vector(3, best_x.tolist(), BOMBS_COUNT)
    eval_full = evaluate_problem3(**dec.to_eval_kwargs(), dt=DT, occlusion_method=METHOD)
    out = {
        'algo': ALGO,
        'best_x': best_x.tolist(),
        'decoded': dec.drones_spec,
        'occluded_time': eval_full['occluded_time'],
        'total': eval_full.get('total', eval_full['occluded_time'].get('M1')),
        'dt': DT,
        'method': METHOD,
        'seconds_used': time.time()-t0,
    }
    fp = LOG_DIR / 'q3_result.json'
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    if VERBOSE:
        print('Q3 optimization finished. Result saved to', fp)
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
