from __future__ import annotations
"""统一优化入口（去除 argparse 版）

把原来命令行参数改成文件顶部常量，题目给定的条件本来就是常量，直接在这里改值即可。

说明：
1. 若某项设为 None 则使用 router 中的默认值；否则覆盖。
2. 只需编辑下面这几行常量即可，其他逻辑不动。
"""

from optimizer import router

# ================== 可编辑常量区域 ==================
# 问题编号 (2,3,4,5)；设为 None 则保持 router 默认
PROBLEM_OVERRIDE: int | None = None

# 算法选择: 'sa' | 'ga' | 'pso' | 'hybrid'
ALGO_OVERRIDE: str | None = None

# 每架无人机投放弹数（题 3 / 5 相关）
BOMBS_PER_DRONE_OVERRIDE: int | None = None

# 时间步长
DT_OVERRIDE: float | None = None

# 遮蔽判定方法: 'sampling' | 'judge_caps' | 'vectorized_sampling'
OCCLUSION_METHOD_OVERRIDE: str | None = None

# backend: rough | vectorized* | 其它实现
BACKEND_OVERRIDE: str | None = None

# 总迭代次数 (模拟 / SA / PSO 通用)；若为 GA 可用 GA_GENERATIONS_OVERRIDE
ITERS_OVERRIDE: int | None = None

# GA 代数
GA_GENERATIONS_OVERRIDE: int | None = None

# 随机种子
SEED_OVERRIDE: int | None = 42

# 详细输出
VERBOSE_OVERRIDE: bool | None = True
# ================== 可编辑常量区域 END ==============


def apply_overrides():
    if PROBLEM_OVERRIDE is not None:
        router.PROBLEM = PROBLEM_OVERRIDE
    if ALGO_OVERRIDE is not None:
        router.ALGO = ALGO_OVERRIDE
    if BOMBS_PER_DRONE_OVERRIDE is not None:
        router.BOMBS_COUNT = BOMBS_PER_DRONE_OVERRIDE
    if DT_OVERRIDE is not None:
        router.DT = DT_OVERRIDE
    if OCCLUSION_METHOD_OVERRIDE is not None:
        router.OCCLUSION_METHOD = OCCLUSION_METHOD_OVERRIDE
    if BACKEND_OVERRIDE is not None:
        router.BACKEND = BACKEND_OVERRIDE
    if ITERS_OVERRIDE is not None:
        router.ITERS = ITERS_OVERRIDE
    if GA_GENERATIONS_OVERRIDE is not None:
        router.GA_GENERATIONS = GA_GENERATIONS_OVERRIDE
    if SEED_OVERRIDE is not None:
        router.SEED = SEED_OVERRIDE
    if VERBOSE_OVERRIDE is not None:
        router.VERBOSE = VERBOSE_OVERRIDE


def main():  # 保持接口
    apply_overrides()
    router.run()


if __name__ == '__main__':  # 直接运行本文件即可
    main()
