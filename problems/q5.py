from __future__ import annotations
"""Problem 5: 5 drones, each up to B bombs (default 3) interfering with M1,M2,M3."""
import json, time
from pathlib import Path
from optimizer.algorithms import solve_sa, solve_ga, solve_pso
from optimizer.spec import decode_vector
from api.problems import evaluate_problem5

ALGO = 'sa'              # 'sa' | 'ga' | 'pso'
DT = 0.05                # increase dt to keep runtime reasonable
METHOD = 'sampling'      # multi-missile; keep general accurate method
BOMBS_PER_DRONE = 3
SEED = None
VERBOSE = True

# SA
SA_ITERS = 12000
SA_NEIGHBOR_BATCH = 512
SA_STEP_SCALE = 0.25
SA_INIT_TEMP = 3.0
SA_FINAL_TEMP = 1e-3
SA_LOG_INTERVAL = 600

# GA
GA_GENERATIONS = 180
GA_POP = 100

# PSO
PSO_ITERS = 500
PSO_POP = 120
LOG_DIR = Path('log'); LOG_DIR.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    if ALGO == 'sa':
        res = solve_sa(problem=5, bombs_count=BOMBS_PER_DRONE, iters=SA_ITERS, dt=DT, method=METHOD,
                       neighbor_batch=SA_NEIGHBOR_BATCH, step_scale=SA_STEP_SCALE,
                       init_temp=SA_INIT_TEMP, final_temp=SA_FINAL_TEMP, log_interval=SA_LOG_INTERVAL,
                       seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    elif ALGO == 'ga':
        res = solve_ga(problem=5, bombs_count=BOMBS_PER_DRONE, generations=GA_GENERATIONS, pop_size=GA_POP,
                       dt=DT, method=METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    elif ALGO == 'pso':
        res = solve_pso(problem=5, bombs_count=BOMBS_PER_DRONE, iters=PSO_ITERS, pop=PSO_POP,
                        dt=DT, method=METHOD, seed=SEED, verbose=VERBOSE)
        best_x = res.best_x
    else:
        raise ValueError('未知 ALGO')
    dec = decode_vector(5, best_x.tolist(), BOMBS_PER_DRONE)
    eval_full = evaluate_problem5(**dec.to_eval_kwargs(), dt=DT, occlusion_method=METHOD)
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
    fp = LOG_DIR / 'q5_result.json'
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    if VERBOSE:
        print('Q5 optimization finished. Result saved to', fp)
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
