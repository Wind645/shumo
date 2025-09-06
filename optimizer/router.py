"""简单路由: 通过修改下方常量选择题目与算法 (不使用命令行参数)。

默认: problem=2, algo='hybrid'.

编码 (problem>=3): [speed, azimuth, t1,d1,t2,d2,...]
"""
from __future__ import annotations

# ==================== 可编辑区 ====================
PROBLEM = 3                 # 2 / 3 / 4 / 5
ALGO = 'sa'             # 'hybrid' | 'ga' | 'pso' | 'sa'
BOMBS_COUNT = 2             # problem>=3 时炸弹数 (编码长度会随之变化)
DT = 0.02
BACKEND = 'rough'  # 仅 problem=2 时有效
SEED = None
VERBOSE = False

# 算法迭代/代数 (None 使用下面的默认)
ITERS = None
GA_GENERATIONS = None

# 各算法缺省 (只在对应 ITERS/GA_GENERATIONS 为 None 且被调用时使用)
_DEF_HYBRID_ITERS = 2000
_DEF_GA_GENS = 80
_DEF_PSO_ITERS = 200
_DEF_SA_ITERS = 4000
# ==================================================

def run():
    prob = PROBLEM
    algo = ALGO
    if algo == 'hybrid':
        from optimizer.hybrid import solve_q2_hybrid
        if prob != 2:
            print('[WARN] hybrid 仅支持 problem=2, 已强制使用 2')
            prob = 2
        res = solve_q2_hybrid(
            sa_iters = ITERS or _DEF_HYBRID_ITERS,
            dt = DT,
            judge_backend = BACKEND,
            seed = SEED,
            verbose = VERBOSE,
        )
        print('HYBRID RESULT', res.best_x, res.best_value, res.best_eval.get('total'))
    elif algo == 'ga':
        from optimizer.ga_q2 import solve_q2_ga
        res = solve_q2_ga(
            n_generations = GA_GENERATIONS or ITERS or _DEF_GA_GENS,
            dt = DT,
            judge_backend = (BACKEND if prob==2 else None),
            problem = prob,
            bombs_count = BOMBS_COUNT,
            verbose = VERBOSE,
            seed = SEED,
        )
        print('GA RESULT', res.best_x, res.best_value, res.best_eval.get('total'))
    elif algo == 'pso':
        from optimizer.pso_q2 import solve_q2_pso
        res = solve_q2_pso(
            iters = ITERS or _DEF_PSO_ITERS,
            dt = DT,
            problem = prob,
            bombs_count = BOMBS_COUNT,
            use_vectorized = (prob==2),
            verbose = VERBOSE,
            seed = SEED,
        )
        print('PSO RESULT', res.best_x, res.best_value, res.best_eval.get('total'))
    elif algo == 'sa':
        from optimizer.sa_q2 import solve_q2_sa
        res = solve_q2_sa(
            iters = ITERS or _DEF_SA_ITERS,
            dt = DT,
            problem = prob,
            bombs_count = BOMBS_COUNT,
            use_vectorized = (prob==2),
            verbose = VERBOSE,
            seed = SEED,
        )
        print('SA RESULT', res.best_x, res.best_value, res.best_eval.get('total'))
    else:
        raise SystemExit('未知 ALGO')

if __name__ == '__main__':
    run()
