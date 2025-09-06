"""End-to-end hybrid optimization pipeline.

Stage 1: Cheap function sparse screening  -> support dims
Stage 2: Multi-fidelity BO (low+high)     -> global search in reduced subspace
Stage 3: Local refinement (coordinate / optional CMA placeholder)

Target problem characteristics (as描述):
  - 20D 原问题 (可扩展) / 稀疏 (有效维度远小于全部维度)
  - cheap + expensive 双层目标, 昂贵预算有限
  - 目标: 最大化 (例如 occluded_time)

公共接口: run_hybrid_pipeline(config) -> dict
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np
import time

from .sparse_screening import screen_dimensions
from .multifidelity_bo import (
    MultiFidelityBO, MFBOConfig, coordinate_refine
)


@dataclass
class HybridPipelineConfig:
    bounds: List[Tuple[float, float]]
    # Screening
    screening_method: str = 'sobol'
    screening_budget: int = 4000
    target_support: int = 8
    screening_budget_fraction: float = 0.6
    use_screening: bool = True  # Option3: 允许关闭筛选, 直接全维做多保真
    # MF BO
    cheap_pool_size: int = 4096
    initial_expensive: int = 12
    max_expensive: int = 80
    kappa: float = 2.2
    random_exp_frac: float = 0.12
    refine_topk: int = 256
    # Local refine
    coord_passes: int = 4
    coord_points_per_dim: int = 11
    # Misc
    seed: Optional[int] = None
    verbose: bool = True


def run_hybrid_pipeline(cfg: HybridPipelineConfig,
                        cheap_eval: Callable[[np.ndarray], np.ndarray],
                        expensive_eval: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
    t0 = time.time()
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    # 1) Screening
    if cfg.use_screening:
        scr_res = screen_dimensions(
            bounds=cfg.bounds,
            cheap_eval_func=cheap_eval,
            target_support_size=cfg.target_support,
            max_cheap_evals=cfg.screening_budget,
            method=cfg.screening_method,
            screening_budget_fraction=cfg.screening_budget_fraction,
        )
        support_dims = scr_res.important_dims
        if cfg.verbose:
            print(f"[Screening] support={support_dims} size={len(support_dims)} scores={np.round(scr_res.sensitivity_scores,4)}")
    else:
        # 构造一个虚拟的 screening 结果 (不耗费 cheap 预算)
        d = len(cfg.bounds)
        support_dims = list(range(d))
        scr_res = type('Dummy', (), {
            '__dict__': {
                'important_dims': support_dims,
                'sensitivity_scores': np.ones(d)/d,
                'support_set_size': d,
                'cheap_evals_used': 0
            }
        })()
        if cfg.verbose:
            print(f"[Screening:SKIPPED] use all dims = {support_dims}")

    # 2) MF BO
    mf_conf = MFBOConfig(
        bounds=cfg.bounds,
        support_dims=support_dims,
        cheap_pool_size=cfg.cheap_pool_size,
        initial_expensive=cfg.initial_expensive,
        max_expensive=cfg.max_expensive,
        kappa=cfg.kappa,
        random_exp_frac=cfg.random_exp_frac,
        refine_topk=cfg.refine_topk,
        seed=cfg.seed,
    )
    mfbo = MultiFidelityBO(mf_conf, cheap_eval, expensive_eval)
    mf_res = mfbo.run()
    if cfg.verbose:
        print(f"[MFBO] best_expensive={mf_res.best_y_expensive:.6f} x={np.round(mf_res.best_x,4)}")

    # 3) Coordinate refine
    x_ref, val_ref = coordinate_refine(
        x0=mf_res.best_x,
        bounds=cfg.bounds,
        eval_func=expensive_eval,
        support_dims=support_dims,
        n_passes=cfg.coord_passes,
        points_per_dim=cfg.coord_points_per_dim,
    )
    if cfg.verbose:
        print(f"[Refine] improved={val_ref - mf_res.best_y_expensive:+.6f} final={val_ref:.6f}")

    elapsed = time.time() - t0
    return {
        'config': asdict(cfg),
        'screening': scr_res.__dict__,
        'mfbo_best_x': mf_res.best_x,
        'mfbo_best_val': mf_res.best_y_expensive,
        'refined_x': x_ref,
        'refined_val': val_ref,
        'elapsed_sec': elapsed,
    }


# 简单可运行示例 (占位) -----------------------------------------------------------------
def _demo():  # pragma: no cover
    d = 12
    bounds = [(-2, 2)] * d
    # 构造稀疏真实目标: 只有前 3 维有效 (Branin-like + sin)
    def cheap_f(X: np.ndarray) -> np.ndarray:
        # noisy low-fidelity (缩放 + 噪声)
        core = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 1]) - 0.3 * (X[:, 2] ** 2)
        return core + 0.1 * np.random.randn(X.shape[0])

    def expensive_f(X: np.ndarray) -> np.ndarray:
        core = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 1]) - 0.3 * (X[:, 2] ** 2)
        penalty = -0.02 * np.sum(X[:, 3:] ** 2, axis=1)
        return core + penalty

    cfg = HybridPipelineConfig(bounds=bounds, seed=0, target_support=5, max_expensive=40)
    out = run_hybrid_pipeline(cfg, cheap_f, expensive_f)
    print(out['refined_val'], out['refined_x'])

if __name__ == '__main__':  # pragma: no cover
    _demo()
