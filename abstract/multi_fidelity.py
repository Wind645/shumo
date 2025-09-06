"""
多保真抽象优化 (Multi-Fidelity Optimization, MFO)
================================================
目标: 在仅提供
  1) 一个“待优化参数向量” (list[float])
  2) 一个黑盒评分函数 evaluate(x, fidelity:int) -> float
的前提下, 给出一个极简却结构清晰的多保真搜索流程。

设计原则:
  - 纯粹抽象, 不依赖具体领域含义。
  - 只假定 fidelity 等级整数 0..F-1, 等级越高越精确/昂贵。
  - 使用“逐级筛选 + 局部扰动” (类似 Successive Halving) 思路。
  - 不引入复杂模型 (可留接口做 surrogate / BO 扩展)。

核心接口:
  optimize_multi_fidelity(
      dim_bounds: list[tuple[float,float]],
      evaluate: Callable[[list[float], int], float],
      fidelity_levels: int,
      init_candidates: int = 64,
      eta: float = 3.0,
      max_high_fidelity: int | None = None,
      perturb_scale: float = 0.15,
      local_expand: bool = True,
      seed: int | None = None,
  ) -> MultiFidelityResult

调度策略:
  - 初始生成 N 个随机点 (均匀采样).
  - 对每一 fidelity level:
      * 评估剩余候选
      * 排序, 仅保留 top_k = max(1, round(n / eta))
      * 如果启用 local_expand, 对最优若干点做局部高斯扰动补充, 直到候选数 >= min(keep + expand_quota, 原数)
        这样在升阶时兼顾 exploitation 与 exploration。
  - 在最终最高保真层可再次(可选)重复评估或截断数量 (通过 max_high_fidelity)。
  - 返回最高保真层最优点。

可扩展点(留空实现, 方便后续自定义):
  - Surrogate / 模型辅助 (接口: hook_after_level / hook_new_points)
  - 重复评估取均值 (当前简单一次评估)

使用说明:
  参见文件底部 __main__ 示例 (一个带噪 sphere + fidelity 偏差的 toy 函数)。

依赖: 仅标准库 + numpy (numpy 在本项目已是依赖; 如不需要可替换为 random).

Author: Abstract engineering template
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Sequence, Dict, Any
import math
import random
import numpy as np
import time
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    x: List[float]
    value_by_fidelity: Dict[int, float] = field(default_factory=dict)

    def best_value(self) -> float:
        # 返回目前已评估的最高保真度数值 (假设 fidelity 越大越可信)
        if not self.value_by_fidelity:
            return math.inf
        # 直接取最大 fidelity 的值
        f = max(self.value_by_fidelity.keys())
        return self.value_by_fidelity[f]

@dataclass
class LevelLog:
    fidelity: int
    evaluated: int  # 原始进入该层的候选数量 (含上一层保留 + 扩展前)
    kept: int       # 该层结束后进入下一层的数量
    expanded: int   # 局部扩展数量
    elapsed_sec: float
    plateau_injected: int = 0  # 平台期随机/自适应注入的数量

@dataclass
class MultiFidelityResult:
    best_x: List[float]
    best_value: float
    history: List[LevelLog]
    all_final: List[Tuple[List[float], float]]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_x": self.best_x,
            "best_value": self.best_value,
            "history": [log.__dict__ for log in self.history],
            "all_final": [(x, v) for (x, v) in self.all_final],
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def optimize_multi_fidelity(
    dim_bounds: Sequence[Tuple[float, float]],
    evaluate: Callable[[Sequence[float], int], float],
    fidelity_levels: int,
    init_candidates: int = 64,
    eta: float = 3.0,
    max_high_fidelity: int | None = None,
    perturb_scale: float = 0.15,
    local_expand: bool = True,
    expand_quota_ratio: float = 0.5,
    # --- 新增: 平台期(所有 value 相同) 逃逸机制参数 ---
    plateau_inject_ratio: float = 0.5,
    plateau_adaptive_scale: float = 2.5,
    plateau_detect_eps: float = 0.0,
    # --- 新增: 极端稀疏目标支持 ---
    sparse_positive_threshold: float = -1e-12,
    sparse_escalate_factor: float = 1.5,
    sparse_max_inject_per_level: int = 512,
    # --- 新增: LHS 初始化与额外 level0 注入 ---
    use_lhs_init: bool = True,
    level0_lhs_extra_ratio: float = 0.5,
    seed: int | None = None,
    verbose: bool = True,
    # --- 新增: 持久化与恢复 ---
    save_path: str | None = None,
    resume_path: str | None = None,
    resume_inject_ratio: float = 0.15,   # 恢复时在历史最优附近注入 (ratio * init_candidates) 个扰动点
) -> MultiFidelityResult:
    """
    进行多保真优化的主函数 (最小化目标).

    参数:
      dim_bounds: 每一维 (low, high)
      evaluate(x, fidelity): 返回该 fidelity 下的标量损失 (越小越好)
      fidelity_levels: fidelity 等级数量 (整数 >= 1)
      init_candidates: 初始随机候选数量
      eta: 每升一级保真筛选比例因子 (Successive Halving)
      max_high_fidelity: 最终最高保真层最大评估候选数 (None 不截断)
      perturb_scale: 局部扰动幅度 (相对每维区间长度)
      local_expand: 是否在保留后做邻域扩展
      expand_quota_ratio: 扩展上限规模 = kept * ratio
      seed: 随机种子
      verbose: 打印简要日志

    返回:
      MultiFidelityResult
    """
    assert fidelity_levels >= 1, "fidelity_levels 必须 >= 1"
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    dim = len(dim_bounds)
    span = np.array([hi - lo for (lo, hi) in dim_bounds], dtype=float)

    def random_point() -> List[float]:
        return [
            rng.uniform(lo, hi)
            for (lo, hi) in dim_bounds
        ]

    # ---- LHS 采样函数 ----
    def _lhs(n: int, d: int, rng_np: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, d))
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng_np.random((n, d))
        a = cut[:-1][:, None]
        b = cut[1:][:, None]
        pts = a + (b - a) * u  # (n,1) broadcast
        # 每列独立打乱
        for j in range(d):
            rng_np.shuffle(pts[:, j])
        return pts

    # 初始化候选 (LHS 优先)
    if use_lhs_init:
        lhs_raw = _lhs(init_candidates, len(dim_bounds), np_rng)
        population: List[CandidateRecord] = []
        for r in lhs_raw:
            vec = [lo + float(r[j]) * (hi - lo) for j, (lo, hi) in enumerate(dim_bounds)]
            population.append(CandidateRecord(vec))
        if verbose:
            print(f"[MF] init using LHS n={init_candidates}")
    else:
        population = [CandidateRecord(random_point()) for _ in range(init_candidates)]
    # ---------- Resume 支持 ----------
    resume_best: List[float] | None = None
    if resume_path:
        try:
            rp = Path(resume_path)
            if rp.is_file():
                data = json.loads(rp.read_text())
                bx = data.get("best_x")
                if isinstance(bx, list) and len(bx) == len(dim_bounds):
                    resume_best = [float(v) for v in bx]
                    if verbose:
                        print(f"[MF][resume] loaded previous best from {resume_path}")
        except Exception as e:
            if verbose:
                print(f"[MF][resume][warn] load failed: {e}")
    if resume_best:
        # 覆盖第一个
        population[0].x = resume_best
        # 注入局部扰动
        k_inject = max(1, int(init_candidates * resume_inject_ratio))
        span_arr = np.array([hi - lo for (lo, hi) in dim_bounds], dtype=float)
        for i in range(1, min(1 + k_inject, len(population))):
            noise = np_rng.normal(0.0, 1.0, size=len(dim_bounds)) * (0.05 * span_arr)
            vec = [float(np.clip(resume_best[d] + noise[d], dim_bounds[d][0], dim_bounds[d][1])) for d in range(len(dim_bounds))]
            population[i].x = vec
        if verbose:
            print(f"[MF][resume] injected {min(k_inject, len(population)-1)} local variants")

    history: List[LevelLog] = []
    t0_total = time.time()

    # 逐级 fidelity
    for f in range(fidelity_levels):
        t_level_start = time.time()
        # 评估尚未在该 fidelity 计算过的候选 (加入进度输出)
        to_eval = [c for c in population if f not in c.value_by_fidelity]
        total_eval = len(to_eval)
        if verbose:
            print(f"[MF] level={f} start evaluate {total_eval} candidates")
        if total_eval:
            report_every = max(1, total_eval // 5)  # 约 5 个进度点
            for i, cand in enumerate(to_eval, 1):
                val = evaluate(cand.x, f)
                cand.value_by_fidelity[f] = float(val)
                if verbose and (i == 1 or i == total_eval or i % report_every == 0):
                    pct = 100.0 * i / total_eval
                    print(f"[MF] level={f} progress {i}/{total_eval} ({pct:.1f}%)")

        # ------------------ 平台期检测 & 逃逸 ------------------
        plateau_injected = 0
        vmin = float('nan')  # Ensure defined for later plateau logging even if current_vals is empty
        current_vals = [c.value_by_fidelity[f] for c in population]
        if current_vals:
            # 基础 min/max
            vmin = min(current_vals); vmax = max(current_vals)
            is_plateau = (vmax - vmin) <= plateau_detect_eps
            # 额外: 忽略巨大惩罚值 (>=1e9) 只看“有效”子集是否平台
            valid_vals = [v for v in current_vals if v < 1e9]
            if len(valid_vals) >= 2:
                vmin_v = min(valid_vals); vmax_v = max(valid_vals)
                if (vmax_v - vmin_v) <= plateau_detect_eps:
                    # 有效集合平台，且存在至少一个惩罚样本 或 有效集合跨度极小
                    if len(valid_vals) < len(current_vals):
                        is_plateau = True
        else:
            is_plateau = False
        if is_plateau and f < fidelity_levels - 1:  # 只在非最终层做逃逸
            if verbose:
                print(f"[MF] level={f} detected plateau (all values={vmin:.6g}), injecting diversity")
            # 1) 注入随机新点
            inject_n = max(1, int(len(population) * plateau_inject_ratio))
            for _ in range(inject_n):
                new_c = CandidateRecord(random_point())
                # 立即评估本层, 以便参与排序
                new_c.value_by_fidelity[f] = float(evaluate(new_c.x, f))
                population.append(new_c)
            plateau_injected += inject_n
            # 2) 对现有少量优者做强化扰动 (适度)
            # 选取前 k (临时以全部，之后排序再截断)
            k_seed = min(len(population), max(2, int(math.sqrt(len(population)))))
            bases = population[:k_seed]
            adapt_scale = perturb_scale * plateau_adaptive_scale
            for parent in bases:
                base = np.array(parent.x)
                noise = np_rng.normal(0.0, 1.0, size=dim) * (adapt_scale * span)
                child_vec = np.clip(base + noise, [lo for (lo, _) in dim_bounds], [hi for (_, hi) in dim_bounds])
                child = CandidateRecord(list(map(float, child_vec)))
                child.value_by_fidelity[f] = float(evaluate(child.x, f))
                population.append(child)
                plateau_injected += 1
            if verbose:
                print(f"[MF] level={f} plateau injection total={plateau_injected} (random + adaptive)")

        # ------------------ 稀疏: 尚无任何“好”点时全局加采样 ------------------
        # “好”点定义: value < sparse_positive_threshold (例如 -occ < 0 => occ>0)
        has_positive = any(v < sparse_positive_threshold for v in current_vals)
        if (not has_positive) and f < fidelity_levels - 1:
            # 仍未发现有效区域：扩大随机探索覆盖
            target_size = int(len(population) * sparse_escalate_factor)
            # 上限控制: 不超过 初始 * (sparse_escalate_factor ** (f+1))
            theoretical_cap = int(init_candidates * (sparse_escalate_factor ** (f + 1)))
            target_size = min(target_size, theoretical_cap)
            need = target_size - len(population)
            if need > 0:
                need = min(need, sparse_max_inject_per_level)
                if verbose:
                    print(f"[MF] level={f} sparse escalate: injecting {need} fresh (no good sample yet)")
                for _ in range(need):
                    nc = CandidateRecord(random_point())
                    nc.value_by_fidelity[f] = float(evaluate(nc.x, f))
                    population.append(nc)
                plateau_injected += need  # 复用字段统计注入 (虽然语义不同)

        # ---- Level0 结束前额外 LHS 注入：仍无命中且未触发 plateau_injected 足够 ----
        if f == 0 and (not has_positive) and level0_lhs_extra_ratio > 0 and len(population) < int(init_candidates * (1.0 + level0_lhs_extra_ratio * 1.5)):
            extra_n = int(init_candidates * level0_lhs_extra_ratio)
            if extra_n > 0:
                lhs_extra = _lhs(extra_n, dim, np_rng)
                injected = 0
                for r in lhs_extra:
                    vec = [lo + float(r[j]) * (hi - lo) for j, (lo, hi) in enumerate(dim_bounds)]
                    c_new = CandidateRecord(vec)
                    c_new.value_by_fidelity[f] = float(evaluate(c_new.x, f))
                    population.append(c_new)
                    injected += 1
                plateau_injected += injected
                if verbose:
                    print(f"[MF] level=0 extra LHS inject={injected} (no positive yet)")

        # 排序 — 使用当前 fidelity 的值
        population.sort(key=lambda c: c.value_by_fidelity[f])

        n_before = len(population)
        # 计算保留数量
        if f < fidelity_levels - 1:
            keep = max(1, round(n_before / eta))
        else:
            keep = n_before  # 最后一级暂不强制筛 (可再截断)
        kept = population[:keep]

        expanded = 0
        if local_expand and f < fidelity_levels - 1:
            # 允许在升阶前围绕 top 部分做局部搜索，增加多样性
            expand_target = min(
                n_before,  # 不超过原数量
                keep + max(1, int(keep * expand_quota_ratio))
            )
            while len(kept) < expand_target:
                # 选一个母体 (偏向更优)
                parent = kept[rng.randrange(len(kept))]
                base = np.array(parent.x)
                noise = np_rng.normal(0.0, 1.0, size=dim) * (perturb_scale * span)
                child_vec = np.clip(base + noise, [lo for (lo, _) in dim_bounds], [hi for (_, hi) in dim_bounds])
                kept.append(CandidateRecord(list(map(float, child_vec))))
                expanded += 1

        population = kept

        # 如果是最后一级可以截断数量
        if f == fidelity_levels - 1 and max_high_fidelity is not None:
            population = population[:max_high_fidelity]

        elapsed = time.time() - t_level_start
        history.append(LevelLog(
            fidelity=f,
            evaluated=n_before,
            kept=len(population),
            expanded=expanded,
            elapsed_sec=elapsed,
            plateau_injected=plateau_injected,
        ))
        if verbose:
            best_val = population[0].value_by_fidelity[f]
            print(f"[MF] level={f} evaluated={n_before} kept={len(population)} "
                  f"expanded={expanded} plateau_injected={plateau_injected} best={best_val:.6g} time={elapsed:.3f}s")

    # 汇总最高保真 (即 f = fidelity_levels-1)
    f_final = fidelity_levels - 1
    population.sort(key=lambda c: c.value_by_fidelity[f_final])
    best_cand = population[0]
    best_value = best_cand.value_by_fidelity[f_final]

    result = MultiFidelityResult(
        best_x=list(best_cand.x),
        best_value=best_value,
        history=history,
        all_final=[(c.x, c.value_by_fidelity[f_final]) for c in population],
        meta=dict(
            total_time=time.time() - t0_total,
            dim=dim,
            fidelity_levels=fidelity_levels,
            init_candidates=init_candidates,
            eta=eta,
            seed=seed,
        )
    )
    if save_path:
        try:
            out = {
                "best_x": result.best_x,
                "best_value": float(result.best_value),
                "meta": result.meta,
            }
            Path(save_path).write_text(json.dumps(out, ensure_ascii=False, indent=2))
            if verbose:
                print(f"[MF] saved result to {save_path}")
        except Exception as e:
            if verbose:
                print(f"[MF][warn] failed to save result: {e}")
    if verbose:
        print(f"[MF] DONE best_value={best_value:.6g} time={result.meta['total_time']:.3f}s")
    return result


# ---------------------------------------------------------------------------
# 可选: 一个超简单的“多保真函数”示例
# ---------------------------------------------------------------------------

def _toy_multifidelity_function(x: Sequence[float], fidelity: int) -> float:
    """
    一个演示用的 toy objective:
      真目标: sphere(x) = sum(x_i^2)
      低保真层添加系统性偏差 + 噪声, fidelity 越高偏差越小, 噪声越少。

    fidelity = 0: bias = 5.0, noise σ=1.0
    fidelity = 1: bias = 1.0, noise σ=0.3
    fidelity = 2: bias = 0.0, noise σ=0.05  (视为最高精度)

    你可以替换为任意 evaluate(x, fidelity) 实现。
    """
    sphere = sum(v * v for v in x)
    if fidelity == 0:
        bias, sigma = 5.0, 1.0
    elif fidelity == 1:
        bias, sigma = 1.0, 0.3
    else:
        bias, sigma = 0.0, 0.05
    noise = random.gauss(0.0, sigma)
    return sphere + bias + noise


# ---------------------------------------------------------------------------
# 主入口示例 (可删除)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 维度与边界 (5 维 [-3,3])
    bounds = [(-3.0, 3.0)] * 5

    res = optimize_multi_fidelity(
        dim_bounds=bounds,
        evaluate=_toy_multifidelity_function,
        fidelity_levels=3,
        init_candidates=80,
        eta=3.0,
        perturb_scale=0.12,
        seed=42,
        verbose=True,
    )

    print("\n=== Result (Toy) ===")
    print("Best x:", res.best_x)
    print("Best value (approx true sphere):", res.best_value)
    print("Levels:")
    for log in res.history:
        print(log)
    # 真实目标值(无 bias 噪声) 估算
    true_val = sum(v * v for v in res.best_x)
    print("Approx true sphere:", true_val)
