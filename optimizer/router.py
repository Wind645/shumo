"""统一优化路由入口: 通过修改下方常量选择题目、算法与评估方式。

支持题号: 2,3,4,5 (问题1 为固定策略评估, 通常无需优化, 可按需扩展)。

通用编码参见 `optimizer/spec.py`:
    - Q2: [speed, azimuth, t_rel, explode_delay]
    - Q3: [speed, azimuth, (t1,d1)...]
    - Q4: 3 架无人机各 1 枚: [s1,a1,t1,d1, s2,a2,t2,d2, s3,a3,t3,d3]
    - Q5: 5 架无人机, 每架 bombs_count 枚: per-drone block 依次拼接。

可选择 occlusion_method: 'sampling' | 'judge_caps' | 'rough_caps' | 'vectorized_sampling'
（向量化采样仅当前实现 M1 单导弹, 自动在问题 2,3,4,5 单导弹场景使用）。
"""
from __future__ import annotations

# ==================== 可编辑区 ====================
PROBLEM = 3                  # 2 / 3 / 4 / 5
ALGO = 'sa'                  # 'ga' | 'pso' | 'sa'
BOMBS_COUNT = 3              # problem=3/5 时炸弹数 (编码长度随之变化)
DT = 0.02
OCCLUSION_METHOD = 'rough_caps'  # 'sampling' | 'judge_caps' | 'rough_caps' | 'vectorized_sampling'
BACKEND = 'rough'            # (保留占位, 新统一实现暂不区分 backend)
SEED = None
VERBOSE = True

# ---- SA 额外可调参数（仅 ALGO='sa' 有效） ----
SA_NEIGHBOR_BATCH = 512      # 每轮候选数量 (越大利用批量越快, GPU/NumPy 更明显)
SA_STEP_SCALE = 0.25         # 步长 (相对各维边界跨度)
SA_INIT_TEMP = 2.0           # 初始温度
SA_FINAL_TEMP = 1e-3         # 终止温度
SA_LOG_INTERVAL = 50         # 日志间隔 (迭代次数)

# 是否启用 Q2 torch fused (GPU) 加速 (仅问题2 且 method 属于 judge_caps/rough_caps)
USE_TORCH_Q2 = True

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
    prob = PROBLEM; algo = ALGO
    from optimizer.algorithms import solve_ga, solve_pso, solve_sa
    # 如启用 torch fused, 设置全局开关
    try:
        import optimizer.algorithms as _algs
        _algs.TORCH_Q2_FUSED = bool(USE_TORCH_Q2)
    except Exception:
        pass
    if algo == 'ga':
        res = solve_ga(
            problem=prob,
            bombs_count=BOMBS_COUNT,
            generations=GA_GENERATIONS or ITERS or _DEF_GA_GENS,
            dt=DT,
            method=OCCLUSION_METHOD,
            seed=SEED,
            verbose=VERBOSE,
        ); tag='GA'
    elif algo == 'pso':
        res = solve_pso(
            problem=prob,
            bombs_count=BOMBS_COUNT,
            iters=ITERS or _DEF_PSO_ITERS,
            dt=DT,
            method=OCCLUSION_METHOD,
            seed=SEED,
            verbose=VERBOSE,
        ); tag='PSO'
    elif algo == 'sa':
        res = solve_sa(
            problem=prob,
            bombs_count=BOMBS_COUNT,
            iters=ITERS or _DEF_SA_ITERS,
            dt=DT,
            method=OCCLUSION_METHOD,
            neighbor_batch=SA_NEIGHBOR_BATCH,
            step_scale=SA_STEP_SCALE,
            init_temp=SA_INIT_TEMP,
            final_temp=SA_FINAL_TEMP,
            log_interval=SA_LOG_INTERVAL,
            seed=SEED,
            verbose=VERBOSE,
        ); tag='SA'
    else:
        raise SystemExit('未知 ALGO')
    print(f'{tag} RESULT', res.best_x, res.best_value, res.best_eval.get('total'))

if __name__ == '__main__':
    run()
