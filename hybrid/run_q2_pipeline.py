"""Run the new hybrid (screening + multi-fidelity BO + refinement) pipeline for Problem 2.

Problem 2 参数 (4 维):
 0 speed           [70, 140]
 1 azimuth         [-pi, pi]
 2 release_time    [0.0, 8.0]   (假设最大 8s 航迹飞行窗口)
 3 explode_delay   [0.1, 12.0]  (烟幕寿命相关, 给宽一些)

低保真 (cheap) 评价: 使用粗粒度 dt=0.05 + occlusion_method='rough_caps' (或 vectorized_sampling 较快版本)
高保真 (expensive) 评价: 使用 dt=0.01 + occlusion_method='judge_caps'

最大化 M1 遮蔽时间 (返回 total 字段)。
"""
from __future__ import annotations
import numpy as np
from math import pi
from typing import Dict
from api.problems import evaluate_problem2
from .pipeline import HybridPipelineConfig, run_hybrid_pipeline


BOUNDS_Q2 = [
    (70.0, 140.0),      # speed
    (-pi, pi),          # azimuth
    (0.0, 8.0),         # release_time
    (0.1, 12.0),        # explode_delay
]


def _cheap_eval(X: np.ndarray) -> np.ndarray:
    vals = []
    for row in X:
        r = evaluate_problem2(
            speed=float(row[0]), azimuth=float(row[1]),
            release_time=float(row[2]), explode_delay=float(row[3]),
            dt=0.05, occlusion_method='rough_caps'
        )
        vals.append(r.get('total', r['occluded_time'].get('M1')))
    return np.array(vals, dtype=float)


def _expensive_eval(X: np.ndarray) -> np.ndarray:
    vals = []
    for row in X:
        r = evaluate_problem2(
            speed=float(row[0]), azimuth=float(row[1]),
            release_time=float(row[2]), explode_delay=float(row[3]),
            dt=0.01, occlusion_method='judge_caps'
        )
        vals.append(r.get('total', r['occluded_time'].get('M1')))
    return np.array(vals, dtype=float)


def main():  # pragma: no cover
    import argparse, json, time
    ap = argparse.ArgumentParser(description='Hybrid pipeline for Problem 2')
    ap.add_argument('--screening-budget', type=int, default=1200)
    ap.add_argument('--max-expensive', type=int, default=60)
    ap.add_argument('--initial-expensive', type=int, default=10)
    ap.add_argument('--cheap-pool', type=int, default=3000)
    ap.add_argument('--target-support', type=int, default=4)
    ap.add_argument('--kappa', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outfile', type=str, default='log/q2_hybrid_pipeline.json')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    cfg = HybridPipelineConfig(
        bounds=BOUNDS_Q2,
        target_support=args.target_support,
        screening_budget=args.screening_budget,
        screening_budget_fraction=0.6,
        cheap_pool_size=args.cheap_pool,
        initial_expensive=args.initial_expensive,
        max_expensive=args.max_expensive,
        kappa=args.kappa,
        random_exp_frac=0.15,
        refine_topk=200,
        coord_passes=3,
        coord_points_per_dim=13,
        seed=args.seed,
        verbose=args.verbose or True,
    )
    t0 = time.time()
    res = run_hybrid_pipeline(cfg, _cheap_eval, _expensive_eval)
    res['seconds_total'] = time.time() - t0
    import os, pathlib
    pathlib.Path('log').mkdir(exist_ok=True)
    with open(args.outfile, 'w', encoding='utf-8') as f:
        json.dump({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in res.items()}, f, ensure_ascii=False, indent=2)
    print('[HybridPipeline] saved to', args.outfile)
    print('best_refined_val', res['refined_val'])
    print('best_refined_x', res['refined_x'])


if __name__ == '__main__':  # pragma: no cover
    main()
