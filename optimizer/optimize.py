"""Constant-based optimization script for PSO, BlockPSO, Differential Evolution (DE),
Discrete PSO (DPSO), Hybrid DE+DPSO, PushPSO (crowding + stagnation repellers),
and a Reduced mode (Problem 5 reduced-dimensional encoder).

Run with:
    python -m optimizer.optimize

Modes:
  MODE = "parallel"  -> single ParallelPSO run
  MODE = "block"     -> block-based multi-PSO (regional elimination)
  MODE = "de"        -> Differential Evolution
  MODE = "dpso"      -> Discrete PSO over categorical sets
  MODE = "hybrid"    -> Two-stage DE (continuous) + DPSO (discrete recombination)
  MODE = "push"      -> PushPSO (crowding + stagnation cluster repellers)
  MODE = "reduced"   -> Reduced Problem 5 (50-dim) encoder + standard ParallelPSO

Configuration sections:
  GLOBAL_*           : judge & shared simulation controls
  PARALLEL_*         : parameters for ParallelPSO
  BLOCK_*            : parameters for BlockPSO
  DE_*               : parameters for DifferentialEvolution
  DPSO_*             : parameters for DiscretePSO (MODE == "dpso")
  HYBRID_*           : parameters for Hybrid (MODE == "hybrid")

Produced outputs:
  - Console concise logs (per iteration / round / generation).
  - Summary JSON-like dict printed at the end.
  - Optional model auto-save (all optimizers except currently hybrid/DPSO which you
    can extend similarly).
"""
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Sequence, Optional
import numpy as np
from .save_system import UniversalSaveSystem, SaveState

from .pso import (
    ParallelPSO,
    build_problem_objective,
    build_problem_times_fn,
    select_judge,
)
try:
    from .staged_pso import StagedDimPSO  # type: ignore
except Exception:  # pragma: no cover
    StagedDimPSO = None  # type: ignore

try:
    from .block_pso import BlockPSO  # type: ignore
except Exception:  # pragma: no cover
    BlockPSO = None  # type: ignore

try:
    from .de import DifferentialEvolution  # type: ignore
except Exception:  # pragma: no cover
    DifferentialEvolution = None  # type: ignore

try:
    from .dpso import DiscretePSO  # type: ignore
except Exception:  # pragma: no cover
    DiscretePSO = None  # type: ignore

try:
    from .hybrid import HybridDE_DPSO  # type: ignore
except Exception:  # pragma: no cover
    HybridDE_DPSO = None  # type: ignore

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Mode: "parallel" | "block" | "de" | "dpso" | "hybrid" | "push" | "reduced" | "staged"
MODE: str = "de"

# Problem id (2..5)
PROBLEM_ID: int = 4
# Unified save/load base name per problem (independent of optimizer method)
UNIFIED_MODEL_BASENAME: str = f"problem{PROBLEM_ID}_latest"

# Judge: "rough" or "sample"
GLOBAL_JUDGE: str = "rough"
# Sampling K (only used when GLOBAL_JUDGE == "sample")
GLOBAL_JUDGE_SAMPLE_K: int = 8

# Simulation dt (used for continuous methods & DPSO objective)
GLOBAL_DT: float = 0.01

# Objective aggregation: "sum" | "min" | "weighted"
GLOBAL_AGGREGATE: str = "sum"
# Weights (only used if GLOBAL_AGGREGATE == "weighted")
GLOBAL_WEIGHTS: Optional[List[float]] = None  # e.g. [1.0, 0.5, 0.5]

# Whether we treat problem as maximization (default True)
GLOBAL_MAXIMIZE: bool = True

# Print per-iteration / per-round progress
GLOBAL_VERBOSE: bool = True

# Show per-missile times for final best position
GLOBAL_SHOW_TIMES: bool = True

# ---------------- Parallel PSO specific ----------------
PARALLEL_ITERATIONS: int = 50
PARALLEL_SWARM_SIZE: int = 64
PARALLEL_INERTIA: float = 0.72
PARALLEL_COGNITIVE: float = 1.49
PARALLEL_SOCIAL: float = 1.49
PARALLEL_RESET_PROB: float = 0.05
PARALLEL_VELOCITY_CLAMP: Optional[tuple[float, float]] = (-0.5, 0.5)
PARALLEL_PROCESSES: Optional[int] = None
PARALLEL_SEED: Optional[int] = 14
PARALLEL_INIT_MODEL: Optional[str] = UNIFIED_MODEL_BASENAME
PARALLEL_PERTURB_STD: float = 0.15
PARALLEL_SAVE_MODEL: Optional[str] = UNIFIED_MODEL_BASENAME
PARALLEL_INCLUDE_SWARM_ON_SAVE: bool = True

# ---------------- Push PSO specific ----------------
PUSH_ITERATIONS: int = 2000
PUSH_SWARM_SIZE: int = 512
PUSH_INERTIA: float = 0.72
PUSH_COGNITIVE: float = 1.49
PUSH_SOCIAL: float = 1.49
PUSH_RESET_PROB: float = 0.02
PUSH_VELOCITY_CLAMP: Optional[tuple[float, float]] = (-0.5, 0.5)
PUSH_PROCESSES: Optional[int] = None
PUSH_SEED: Optional[int] = 12
PUSH_CROWD_RADIUS: Optional[float] = None          # set None to use grid mode
PUSH_DENSITY_THRESHOLD: int = 6
PUSH_PUSH_STRENGTH: float = 0.8
PUSH_ADAPTIVE_PUSH: bool = True
PUSH_STAGNATION_ITERS: int = 3                    # (legacy per-particle stagnation threshold kept for compatibility)
PUSH_CLUSTER_EPS: float = 0.15
PUSH_CLUSTER_MIN_SIZE: int = 3
PUSH_CLUSTER_MAX_SIZE: int = 10
PUSH_REPELLER_RADIUS: float = 0.6
PUSH_REPELLER_STRENGTH: float = 1.2
PUSH_REPELLER_DECAY: float = 0.95
# ---- New region stagnation (local repeller) controls ----
PUSH_REGION_STAGNATION_ITERS: int = 20            # increased: require longer local stagnation
PUSH_REGION_MIN_GROUP: int = 5                    # larger group to form local repeller
PUSH_REGION_REPELLER_RADIUS_SCALE: float = 1.0    # region_repeller_radius_scale
# ---- Global stagnation escape controls ----
PUSH_GLOBAL_STAGNATION_ITERS: int = 50            # allow exploitation before global escape
PUSH_GLOBAL_STAGNATION_REINIT_FRACTION: float = 0.15
PUSH_GLOBAL_STAGNATION_REPELLER_STRENGTH_SCALE: float = 1.2
PUSH_GLOBAL_STAGNATION_REPELLER_RADIUS_SCALE: float = 1.0
# ---- Repeller population / behavior caps ----
PUSH_MAX_ACTIVE_REPELLERS: int = 120              # tighter cap to prevent explosion
PUSH_REPELLER_INVERSE_DISTANCE: bool = False      # toggle inverse-distance extra scaling
PUSH_SAVE_MODEL: Optional[str] = UNIFIED_MODEL_BASENAME
PUSH_INCLUDE_SWARM_ON_SAVE: bool = True

# --- Push PSO region stagnation (local repeller) params ---
# 连续多少轮单粒子未改进视为“区域停滞”候选
PUSH_REGION_STAGNATION_ITERS: int = 20
# 形成一个区域 repeller 至少需要的停滞粒子数量
PUSH_REGION_MIN_GROUP: int = 5
# 区域分组时使用的半径缩放（相对于 repeller_radius）
PUSH_REGION_REPELLER_RADIUS_SCALE: float = 1.0

# --- Push PSO global stagnation (whole-swarm escape) params ---
# 全局最优连续多少轮无改进触发全局逃逸
PUSH_GLOBAL_STAGNATION_ITERS: int = 50
# 触发时重新随机初始化的粒子比例
PUSH_GLOBAL_STAGNATION_REINIT_FRACTION: float = 0.15
# 触发时在全局最优处放置的强力 repeller 强度倍率 (基于当前 repeller_strength)
PUSH_GLOBAL_STAGNATION_REPELLER_STRENGTH_MULT: float = 1.2
# 触发时强力 repeller 半径倍率 (基于当前 repeller_radius)
PUSH_GLOBAL_STAGNATION_REPELLER_RADIUS_MULT: float = 1.0

# ---------------- Block PSO specific ----------------
BLOCK_BLOCKS_PER_DIM: int = 2
BLOCK_TOTAL_ITERATIONS: int = 150
BLOCK_BLOCK_ITERATIONS: int = 10
BLOCK_ELIMINATION_FRACTION: float = 0.30
BLOCK_MIN_BLOCKS: int = 1
BLOCK_SWARM_SIZE: int = 64
BLOCK_INERTIA: float = 0.72
BLOCK_COGNITIVE: float = 1.49
BLOCK_SOCIAL: float = 1.49
BLOCK_RESET_PROB: float = 0.08
BLOCK_VELOCITY_CLAMP: Optional[tuple[float, float]] = (-0.4, 0.4)
BLOCK_PROCESSES: Optional[int] = 4
BLOCK_SEED: Optional[int] = 2025
BLOCK_SAVE_MODEL: Optional[str] = UNIFIED_MODEL_BASENAME

# ---------------- Differential Evolution specific ----------------
# Large-pop configuration: larger population improves coverage; tune generations & F accordingly.
# Population raised (>=2000 as requested). F lowered for stability with large diversity.
DE_GENERATIONS: int = 250         # more total evaluations with large population
DE_POPULATION_SIZE: int = 5000     # large population as per requirement (>=2000)
DE_F: float = 0.9                  # slightly smaller differential weight to reduce overshoot in large pop
DE_CR: float = 0.9                 # keep crossover rate
DE_PROCESSES: Optional[int] = None
DE_SEED: Optional[int] = 111
DE_INIT_MODEL: Optional[str] = UNIFIED_MODEL_BASENAME
DE_PERTURB_STD: float = 0.7      # slightly lower perturb around best when warm starting large population
DE_SAVE_MODEL: Optional[str] = UNIFIED_MODEL_BASENAME
DE_INCLUDE_POP_ON_SAVE: bool = True

# ---------------- StagedDimPSO specific ----------------
# 维度分批搜索相关配置
STAGED_MODE_TYPE: str = "overlap"      # 'fixed' | 'shuffle' | 'overlap' | 'sensitivity' | 'custom'
STAGED_GROUP_SIZE: int = 5             # for shuffle / sensitivity
STAGED_WINDOW_SIZE: int = 5            # for fixed / overlap
STAGED_STRIDE: int = 3                 # for overlap (ignored for fixed => stride=window)
STAGED_ITERATIONS_PER_STAGE: int = 30
STAGED_SWARM_SIZE: int = 48
STAGED_REFINE_FULL: bool = True
STAGED_REFINE_ITERATIONS: int = 60
STAGED_MAX_STAGES: int | None = None   # limit number of stage groups (None = all)
STAGED_SEED: int | None = 2025

# ---------------- Discrete PSO (DPSO) specific ----------------
# DPSO operates on a discretized categorical set taken from a seed sampling procedure.
# For simplicity here we just sample initial continuous points randomly and discretize.
DPSO_SAMPLE_POP: int = 256          # number of random continuous samples to derive categories
DPSO_CATEGORY_CAP: int = 24          # max categories per dimension
DPSO_MIN_CATEGORIES: int = 6
DPSO_SWARM_SIZE: int = 160
DPSO_ITERATIONS: int = 50
DPSO_INERTIA: float = 0.65
DPSO_COGNITIVE: float = 1.4
DPSO_SOCIAL: float = 1.4
DPSO_RESET_PROB: float = 0.04
DPSO_NOISE_STD: float = 0.015
DPSO_TEMPERATURE_DECAY: Optional[float] = None
DPSO_SEED: Optional[int] = 555

# ---------------- Hybrid (DE -> Staged -> Push) specific ----------------
# Stage 1: Differential Evolution (global exploration)
HYBRID_DE_POPULATION: int = 10000
HYBRID_DE_GENERATIONS: int = 200
HYBRID_DE_F: float = 0.9
HYBRID_DE_CR: float = 0.9
# Optional DE elitism / selection tuning (GA-style)
HYBRID_DE_ELITE_FRACTION: float = 0.01  # keep top 1% each generation (if supported)
HYBRID_DE_TOURNAMENT_K: int = 2         # tournament size for selection (if supported)
# Optional GA stage (if enabled, replaces DE in hybrid Stage 1)
HYBRID_GA_ENABLE: bool = True
HYBRID_GA_POPULATION: int = 10000
HYBRID_GA_GENERATIONS: int = 200
HYBRID_GA_ELITE_FRACTION: float = 0.01
HYBRID_GA_TOURNAMENT_K: int = 2
HYBRID_GA_CROSSOVER_RATE: float = 0.9
HYBRID_GA_MUTATION_RATE: float = 0.15
HYBRID_GA_MUTATION_SIGMA_FRAC: float = 0.08
HYBRID_GA_JITTER_SEED_FRACTION: float = 0.8
HYBRID_GA_STAGNATION_PATIENCE: int = 10
# Unified simulation dt for hybrid pipeline
HYBRID_DT_END: float = 0.05
# Seed (shared base for sub-stages)
HYBRID_SEED: Optional[int] = 2025
# Stage 3 (optional): PushPSO polish (crowding disabled)
HYBRID_ENABLE_PUSH: bool = True
HYBRID_PUSH_ITERS: int = 160
HYBRID_PUSH_SWARM: int = 512
HYBRID_PUSH_REPELLER_RADIUS: float = 0.6
HYBRID_PUSH_REPELLER_STRENGTH: float = 1.2
HYBRID_PUSH_REPELLER_DECAY: float = 0.985
HYBRID_PUSH_RESET_PROB: float = 0.04
HYBRID_PUSH_INERTIA: float = 0.72
HYBRID_PUSH_COG: float = 1.49
HYBRID_PUSH_SOC: float = 1.49

# =============================================================================
# OPTIONAL ENVIRONMENT OVERRIDES
# =============================================================================
_ENV_OVERRIDES = {
    "MODE": ("MODE", str),
    "PROBLEM_ID": ("PROBLEM_ID", int),
    "PARALLEL_ITERATIONS": ("PARALLEL_ITERATIONS", int),
    "PARALLEL_SWARM_SIZE": ("PARALLEL_SWARM_SIZE", int),
    "BLOCK_TOTAL_ITERATIONS": ("BLOCK_TOTAL_ITERATIONS", int),
    "BLOCK_BLOCK_ITERATIONS": ("BLOCK_BLOCK_ITERATIONS", int),
    "DE_GENERATIONS": ("DE_GENERATIONS", int),
    "DE_POPULATION_SIZE": ("DE_POPULATION_SIZE", int),
    "DPSO_ITERATIONS": ("DPSO_ITERATIONS", int),
    "DPSO_SWARM_SIZE": ("DPSO_SWARM_SIZE", int),
    "HYBRID_DE_GENERATIONS": ("HYBRID_DE_GENERATIONS", int),
    "HYBRID_DE_POPULATION": ("HYBRID_DE_POPULATION", int),
}
for _k, (env_name, cast) in _ENV_OVERRIDES.items():
    if env_name in os.environ:
        try:
            globals()[_k] = cast(os.environ[env_name])  # type: ignore
        except Exception:
            pass

# =============================================================================
# Helpers
# =============================================================================
def _maybe_weights() -> Optional[List[float]]:
    return GLOBAL_WEIGHTS if GLOBAL_AGGREGATE == "weighted" else None

def _safe_strategy_to_json(strategy_obj):
    if isinstance(strategy_obj, (list, tuple)):
        return [_safe_strategy_to_json(x) for x in strategy_obj]
    if hasattr(strategy_obj, "tolist"):
        try:
            return strategy_obj.tolist()
        except Exception:
            pass
    if isinstance(strategy_obj, (int, float, str)) or strategy_obj is None:
        return strategy_obj
    return str(strategy_obj)

# =============================================================================
# Run functions
# =============================================================================
def run_parallel() -> Dict[str, Any]:
    encoder, objective = build_problem_objective(
        PROBLEM_ID, dt=GLOBAL_DT, aggregate=GLOBAL_AGGREGATE, weights=_maybe_weights()
    )
    best_times_fn = build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT) if GLOBAL_SHOW_TIMES else None
    pso = ParallelPSO(
        dim=encoder.dim,
        objective=objective,
        swarm_size=PARALLEL_SWARM_SIZE,
        iterations=PARALLEL_ITERATIONS,
        inertia=PARALLEL_INERTIA,
        cognitive=PARALLEL_COGNITIVE,
        social=PARALLEL_SOCIAL,
        reset_prob=PARALLEL_RESET_PROB,
        velocity_clamp=PARALLEL_VELOCITY_CLAMP,
        maximize=GLOBAL_MAXIMIZE,
        processes=PARALLEL_PROCESSES,
        seed=PARALLEL_SEED,
        best_times_fn=best_times_fn,
        init_model=PARALLEL_INIT_MODEL,
        perturb_std=PARALLEL_PERTURB_STD,
        # Disable internal auto-save; we handle unified merge below
        save_on_exit=None,
        include_swarm_on_save=PARALLEL_INCLUDE_SWARM_ON_SAVE,
    )
    result = pso.run()
    strategy = encoder.encode(list(result.best_position))
    final_times = None
    if GLOBAL_SHOW_TIMES:
        final_times = best_times_fn(list(result.best_position)) if best_times_fn else build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT)(list(result.best_position))
    # Unified per-problem merge
    section = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dim": encoder.dim,
        "best_fitness": float(result.best_fitness),
        "best_position": result.best_position.tolist(),
        "history": result.history,
        "strategy": _safe_strategy_to_json(strategy),
    }
    if final_times is not None:
        section["times"] = final_times
    # Include swarm positions for richer future warm starts
    try:
        section["swarm_positions"] = pso.positions.tolist()
    except Exception:
        pass
    merge_optimizer_section(PROBLEM_ID, "parallel", section, maximize=GLOBAL_MAXIMIZE)
    if GLOBAL_VERBOSE:
        print(f"[ParallelPSO] Done iters={PARALLEL_ITERATIONS} best={result.best_fitness:.6f}")
    return {
        "mode": "parallel",
        "problem": PROBLEM_ID,
        "best_fitness": float(result.best_fitness),
        "best_position": result.best_position.tolist(),
        "history": result.history,
        "eval_count": int(result.eval_count),
        "elapsed_sec": float(result.elapsed),
        "strategy": _safe_strategy_to_json(strategy),
        "times": final_times,
    }


def run_push() -> Dict[str, Any]:
    # Local import to avoid mandatory dependency if file not present
    from .ppso import PushPSO
    encoder, objective = build_problem_objective(
        PROBLEM_ID, dt=GLOBAL_DT, aggregate=GLOBAL_AGGREGATE, weights=_maybe_weights()
    )
    best_times_fn = build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT) if GLOBAL_SHOW_TIMES else None
    pso = PushPSO(
        dim=encoder.dim,
        objective=objective,
        swarm_size=PUSH_SWARM_SIZE,
        iterations=PUSH_ITERATIONS,
        inertia=PUSH_INERTIA,
        cognitive=PUSH_COGNITIVE,
        social=PUSH_SOCIAL,
        reset_prob=PUSH_RESET_PROB,
        velocity_clamp=PUSH_VELOCITY_CLAMP,
        maximize=GLOBAL_MAXIMIZE,
        processes=PUSH_PROCESSES,
        seed=PUSH_SEED,
        crowd_radius=PUSH_CROWD_RADIUS,
        density_threshold=PUSH_DENSITY_THRESHOLD,
        push_strength=PUSH_PUSH_STRENGTH,
        adaptive_push=PUSH_ADAPTIVE_PUSH,
        enable_crowding=False,  # explicitly disable O(N^2) crowding; rely on repellers
        stagnation_iter_threshold=PUSH_STAGNATION_ITERS,
        cluster_eps=PUSH_CLUSTER_EPS,
        cluster_min_size=PUSH_CLUSTER_MIN_SIZE,
        cluster_max_size=PUSH_CLUSTER_MAX_SIZE,
        repeller_radius=PUSH_REPELLER_RADIUS,
        repeller_strength=PUSH_REPELLER_STRENGTH,
        repeller_decay=PUSH_REPELLER_DECAY,
        save_on_exit=PUSH_SAVE_MODEL,
        include_swarm_on_save=PUSH_INCLUDE_SWARM_ON_SAVE,
    )
    # ---- Apply extended PushPSO tuning constants (post-construction to keep backward compatibility) ----
    # Local region stagnation repeller parameters
    pso.region_stagnation_iters = PUSH_REGION_STAGNATION_ITERS
    pso.region_min_group = PUSH_REGION_MIN_GROUP
    pso.region_repeller_radius_scale = PUSH_REGION_REPELLER_RADIUS_SCALE
    # Repeller capacity & behavior
    pso.max_active_repellers = PUSH_MAX_ACTIVE_REPELLERS
    pso.repeller_use_inverse_distance = PUSH_REPELLER_INVERSE_DISTANCE
    # Advanced lifecycle controls (currently hard-coded; promote to constants/env if needed)
    pso.repeller_max_age = 150
    pso.repeller_radius_decay = 0.997
    pso.max_new_repellers_per_iter = 3
    pso.duplicate_distance_scale = 1.2
    # Global stagnation escape overrides
    pso.global_stagnation_iters = PUSH_GLOBAL_STAGNATION_ITERS
    pso.global_stagnation_reinit_fraction = PUSH_GLOBAL_STAGNATION_REINIT_FRACTION
    # Support both *_SCALE (new) and *_MULT (legacy) constant names
    if 'PUSH_GLOBAL_STAGNATION_REPELLER_STRENGTH_SCALE' in globals():
        pso.global_stagnation_repeller_strength = pso.repeller_strength * PUSH_GLOBAL_STAGNATION_REPELLER_STRENGTH_SCALE
    else:
        pso.global_stagnation_repeller_strength = pso.repeller_strength * PUSH_GLOBAL_STAGNATION_REPELLER_STRENGTH_MULT
    if 'PUSH_GLOBAL_STAGNATION_REPELLER_RADIUS_SCALE' in globals():
        pso.global_stagnation_repeller_radius = pso.repeller_radius * PUSH_GLOBAL_STAGNATION_REPELLER_RADIUS_SCALE
    else:
        pso.global_stagnation_repeller_radius = pso.repeller_radius * PUSH_GLOBAL_STAGNATION_REPELLER_RADIUS_MULT
    # Run optimizer after parameter overrides
    result = pso.run()
    strategy = encoder.encode(list(result.best_position))
    final_times = None
    if GLOBAL_SHOW_TIMES:
        final_times = best_times_fn(list(result.best_position)) if best_times_fn else build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT)(list(result.best_position))
    if GLOBAL_VERBOSE:
        print(f"[PushPSO] Done iters={PUSH_ITERATIONS} best={result.best_fitness:.6f} repellers={len(result.repeller_events)}")
    return {
        "mode": "push",
        "problem": PROBLEM_ID,
        "best_fitness": float(result.best_fitness),
        "best_position": result.best_position.tolist(),
        "history": result.history,
        "eval_count": int(result.eval_count),
        "elapsed_sec": float(result.elapsed),
        "strategy": _safe_strategy_to_json(strategy),
        "times": final_times,
        "repeller_events": int(len(result.repeller_events)),
    }

def run_staged() -> Dict[str, Any]:
    if StagedDimPSO is None:
        raise RuntimeError("StagedDimPSO module not available.")
    encoder, objective = build_problem_objective(
        PROBLEM_ID, dt=GLOBAL_DT, aggregate=GLOBAL_AGGREGATE, weights=_maybe_weights()
    )
    # Build bounds (reuse [-1,1] if no explicit provided; real encoder内部自己再映射)
    lo = np.full(encoder.dim, -1.0)
    hi = np.full(encoder.dim, 1.0)
    staged = StagedDimPSO(
        dim=encoder.dim,
        objective=objective,
        mode=STAGED_MODE_TYPE,
        group_size=STAGED_GROUP_SIZE,
        window_size=STAGED_WINDOW_SIZE,
        stride=STAGED_STRIDE,
        iterations_per_stage=STAGED_ITERATIONS_PER_STAGE,
        swarm_size=STAGED_SWARM_SIZE,
        maximize=GLOBAL_MAXIMIZE,
        bounds=(lo, hi),
        seed=STAGED_SEED,
        refine_full=STAGED_REFINE_FULL,
        refine_iterations=STAGED_REFINE_ITERATIONS,
        max_stages=STAGED_MAX_STAGES,
        verbose=GLOBAL_VERBOSE,
    )
    result = staged.run()
    strategy = encoder.encode(list(result.best_position))
    # Build per-stage summary subset (avoid huge dump)
    stage_summaries = [{
        "id": lg.stage_id,
        "k": lg.sub_dim,
        "improved": lg.improved_global,
        "after": lg.global_best_after
    } for lg in result.stage_logs]
    section = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dim": encoder.dim,
        "best_fitness": float(result.best_fitness),
        "best_position": result.best_position.tolist(),
        "stage_count": len(result.stage_logs),
        "stage_summaries": stage_summaries,
    }
    merge_optimizer_section(PROBLEM_ID, "staged", section, maximize=GLOBAL_MAXIMIZE)
    if GLOBAL_VERBOSE:
        print(f"[StagedDimPSO] Done stages={len(result.stage_logs)} best={result.best_fitness:.6f}")
    return {
        "mode": "staged",
        "problem": PROBLEM_ID,
        "best_fitness": float(result.best_fitness),
        "best_position": result.best_position.tolist(),
        "stage_count": len(result.stage_logs),
        "eval_count": int(result.eval_count),
        "strategy": _safe_strategy_to_json(strategy),
        "elapsed_sec": None,
    }

def run_block() -> Dict[str, Any]:
    if BlockPSO is None:
        raise RuntimeError("BlockPSO module not available.")
    encoder, objective = build_problem_objective(
        PROBLEM_ID, dt=GLOBAL_DT, aggregate=GLOBAL_AGGREGATE, weights=_maybe_weights()
    )
    bpso = BlockPSO(
        dim=encoder.dim,
        objective=objective,
        blocks_per_dim=BLOCK_BLOCKS_PER_DIM,
        total_iterations=BLOCK_TOTAL_ITERATIONS,
        block_iterations=BLOCK_BLOCK_ITERATIONS,
        elimination_fraction=BLOCK_ELIMINATION_FRACTION,
        min_blocks=BLOCK_MIN_BLOCKS,
        swarm_size=BLOCK_SWARM_SIZE,
        maximize=GLOBAL_MAXIMIZE,
        inertia=BLOCK_INERTIA,
        cognitive=BLOCK_COGNITIVE,
        social=BLOCK_SOCIAL,
        reset_prob=BLOCK_RESET_PROB,
        velocity_clamp=BLOCK_VELOCITY_CLAMP,
        processes=BLOCK_PROCESSES,
        seed=BLOCK_SEED,
    )
    result = bpso.run()
    strategy = encoder.encode(list(result.best_position))
    final_times = None
    if GLOBAL_SHOW_TIMES:
        final_times = build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT)(list(result.best_position))
    if BLOCK_SAVE_MODEL:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"{BLOCK_SAVE_MODEL}.json")

        # Merge block optimizer results into a unified per-problem file.
        existing: dict = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}

        # Optimizer-specific section aggregation
        optimizers = existing.get("optimizers", {})

        block_section = {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dim": encoder.dim,
            "best_fitness": float(result.best_fitness),
            "best_position": result.best_position.tolist(),
            "history": result.history,
            "round_history": result.block_histories,
        }
        if final_times is not None:
            block_section["times"] = final_times

        optimizers["block"] = block_section

        # Maintain / update backward-compatible top-level best fields.
        current_best = existing.get("best_fitness")
        if current_best is None:
            existing["best_fitness"] = float(result.best_fitness)
            existing["best_position"] = result.best_position.tolist()
        else:
            better = (result.best_fitness > current_best) if GLOBAL_MAXIMIZE else (result.best_fitness < current_best)
            if better:
                existing["best_fitness"] = float(result.best_fitness)
                existing["best_position"] = result.best_position.tolist()

        existing["optimizers"] = optimizers
        existing.setdefault("schema", 1)
        existing["problem"] = PROBLEM_ID

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[BlockPSO] Model merged into {path}", flush=True)
        # (legacy block save code removed; unified merge already written above)
    if GLOBAL_VERBOSE:
        print(f"[BlockPSO] Done rounds={len(result.history)} best={result.best_fitness:.6f}")
    return {
        "mode": "block",
        "problem": PROBLEM_ID,
        "best_fitness": float(result.best_fitness),
        "best_position": result.best_position.tolist(),
        "history_per_round": result.history,
        "block_round_summaries": result.block_histories,
        "elapsed_sec": float(result.elapsed),
        "eval_count": int(result.eval_count),
        "strategy": _safe_strategy_to_json(strategy),
        "times": final_times,
    }

def run_de() -> Dict[str, Any]:
    if DifferentialEvolution is None:
        raise RuntimeError("DifferentialEvolution module not available.")
    encoder, objective = build_problem_objective(
        PROBLEM_ID, dt=GLOBAL_DT, aggregate=GLOBAL_AGGREGATE, weights=_maybe_weights()
    )
    best_times_fn = build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT) if GLOBAL_SHOW_TIMES else None
    de = DifferentialEvolution(
        dim=encoder.dim,
        objective=objective,
        population_size=DE_POPULATION_SIZE,
        generations=DE_GENERATIONS,
        F=DE_F,
        CR=DE_CR,
        maximize=GLOBAL_MAXIMIZE,
        processes=DE_PROCESSES,
        seed=DE_SEED,
        best_times_fn=best_times_fn,
        init_model=DE_INIT_MODEL,
        perturb_std=DE_PERTURB_STD,
        save_on_exit=DE_SAVE_MODEL,
        include_swarm_on_save=DE_INCLUDE_POP_ON_SAVE,
    )
    result = de.run()
    strategy = encoder.encode(list(result.best_position))
    final_times = None
    if GLOBAL_SHOW_TIMES:
        final_times = best_times_fn(list(result.best_position)) if best_times_fn else build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT)(list(result.best_position))
    if GLOBAL_VERBOSE:
        print(f"[DE] Done generations={DE_GENERATIONS} best={result.best_fitness:.6f}")
    return {
        "mode": "de",
        "problem": PROBLEM_ID,
        "best_fitness": float(result.best_fitness),
        "best_position": result.best_position.tolist(),
        "history": result.history,
        "eval_count": int(result.eval_count),
        "elapsed_sec": float(result.elapsed),
        "strategy": _safe_strategy_to_json(strategy),
        "times": final_times,
    }

def run_dpso() -> Dict[str, Any]:
    if DiscretePSO is None:
        raise RuntimeError("DiscretePSO module not available.")
    # Build categories from random sampling in continuous space
    encoder, objective = build_problem_objective(
        PROBLEM_ID, dt=GLOBAL_DT, aggregate=GLOBAL_AGGREGATE, weights=_maybe_weights()
    )
    dim = encoder.dim
    samples = []
    for _ in range(DPSO_SAMPLE_POP):
        samples.append(list((2 * (os.urandom(1)[0] / 255.0) - 1.0) for _ in range(dim)))
    samples_arr = [[row[d] for row in samples] for d in range(dim)]
    categories: List[List[float]] = []
    for d in range(dim):
        uniq = sorted(set(float(v) for v in samples_arr[d]))
        if len(uniq) > DPSO_CATEGORY_CAP:
            stride = len(uniq) / DPSO_CATEGORY_CAP
            reduced = []
            for k in range(DPSO_CATEGORY_CAP):
                idx = int(round(k * stride))
                if idx >= len(uniq):
                    idx = len(uniq) - 1
                reduced.append(uniq[idx])
            uniq = sorted(set(reduced))
        while len(uniq) < DPSO_MIN_CATEGORIES:
            uniq.append(0.0)
            uniq = sorted(set(uniq))
        categories.append(uniq)
    def obj(raw_vec: Sequence[float]) -> float:
        return objective(raw_vec)
    dpso = DiscretePSO(
        categories=categories,
        objective=obj,
        swarm_size=DPSO_SWARM_SIZE,
        iterations=DPSO_ITERATIONS,
        inertia=DPSO_INERTIA,
        cognitive=DPSO_COGNITIVE,
        social=DPSO_SOCIAL,
        reset_prob=DPSO_RESET_PROB,
        noise_std=DPSO_NOISE_STD,
        maximize=GLOBAL_MAXIMIZE,
        processes=None,
        seed=DPSO_SEED or int(time.time()),
        pass_indices=False,
        init_model=None,
        perturb_prob=0.15,
        models_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
        save_on_exit=None,
        include_swarm_on_save=False,
        temperature_decay=DPSO_TEMPERATURE_DECAY,
    )
    res = dpso.run()
    if GLOBAL_SHOW_TIMES:
        times_fn = build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT)
        times = times_fn(res.best_values)
    else:
        times = None
    return {
        "mode": "dpso",
        "problem": PROBLEM_ID,
        "best_fitness": float(res.best_fitness),
        "best_indices": res.best_indices.tolist(),
        "best_values": [float(v) for v in res.best_values],
        "history": res.history,
        "eval_count": int(res.eval_count),
        "elapsed_sec": float(res.elapsed),
        "times": times,
    }

def run_hybrid() -> Dict[str, Any]:
    """
    New hybrid pipeline (inline):
        Stage 1: Differential Evolution (large population, few generations)
        Stage 2: StagedDimPSO refinement (uses STAGED_* configuration constants)
        Stage 3: Optional PushPSO polish (crowding disabled)

    Keeps outward contract (mode='hybrid') while removing legacy DPSO usage.
    A synthetic 'dpso' section is still written for backward compatibility,
    mapping to the staged refinement results.
    """
    # Safe initial placeholders for static/type analyzers; real values set in Stage 1
    stage1_algo = "de"
    de_res = None  # will hold GA or DE result object
    # -------- Stage 0: Build objective & encoder ----------
    encoder, base_objective = build_problem_objective(
        PROBLEM_ID, dt=HYBRID_DT_END, aggregate=GLOBAL_AGGREGATE, weights=_maybe_weights()
    )

    # -------- Stage 1: Differential Evolution -------------
    if DifferentialEvolution is None:
        raise RuntimeError("DifferentialEvolution module not available (needed for hybrid).")

    use_ga = 'HYBRID_GA_ENABLE' in globals() and HYBRID_GA_ENABLE
    if use_ga:
        try:
            from .ga import GeneticAlgorithm
            ga = GeneticAlgorithm(
                dim=encoder.dim,
                objective=base_objective,
                population_size=HYBRID_GA_POPULATION,
                generations=HYBRID_GA_GENERATIONS,
                elite_fraction=HYBRID_GA_ELITE_FRACTION,
                tournament_k=HYBRID_GA_TOURNAMENT_K,
                crossover_rate=HYBRID_GA_CROSSOVER_RATE,
                mutation_rate=HYBRID_GA_MUTATION_RATE,
                mutation_sigma_frac=HYBRID_GA_MUTATION_SIGMA_FRAC,
                jitter_seed_fraction=HYBRID_GA_JITTER_SEED_FRACTION,
                stagnation_patience=HYBRID_GA_STAGNATION_PATIENCE,
                maximize=GLOBAL_MAXIMIZE,
                processes=None,
                seed=HYBRID_SEED,
                init_model=None,
                perturb_std=0.15,
                save_on_exit=None,
                include_swarm_on_save=False,
            )
            de_res = ga.run()  # alias to reuse downstream variable names
            stage1_algo = "ga"
            if GLOBAL_VERBOSE:
                print(f"[Hybrid] GA stage done best={de_res.best_fitness:.6f}")
        except Exception as _e:
            if GLOBAL_VERBOSE:
                print(f"[Hybrid] GA unavailable ({_e}); falling back to DE.")
            use_ga = False
    if not use_ga:
        de = DifferentialEvolution(
            dim=encoder.dim,
            objective=base_objective,
            population_size=HYBRID_DE_POPULATION,
            generations=HYBRID_DE_GENERATIONS,
            F=HYBRID_DE_F,
            CR=HYBRID_DE_CR,
            elite_fraction=HYBRID_DE_ELITE_FRACTION,
            tournament_k=HYBRID_DE_TOURNAMENT_K,
            maximize=GLOBAL_MAXIMIZE,
            processes=None,
            seed=HYBRID_SEED,
            init_model=None,
            perturb_std=0.15,
            save_on_exit=None,
            include_swarm_on_save=False,
        )
        de_res = de.run()
        stage1_algo = "de"
        if GLOBAL_VERBOSE:
            print(f"[Hybrid] DE stage done best={de_res.best_fitness:.6f}")

    # -------- Stage 2: StagedDimPSO Refinement ------------
    if StagedDimPSO is None:
        raise RuntimeError("StagedDimPSO module not available (needed for hybrid).")

    lo = np.full(encoder.dim, -1.0)
    hi = np.full(encoder.dim,  1.0)

    staged = StagedDimPSO(
        dim=encoder.dim,
        objective=base_objective,
        mode=STAGED_MODE_TYPE,
        group_size=STAGED_GROUP_SIZE,
        window_size=STAGED_WINDOW_SIZE,
        stride=STAGED_STRIDE,
        iterations_per_stage=STAGED_ITERATIONS_PER_STAGE,
        swarm_size=STAGED_SWARM_SIZE,
        maximize=GLOBAL_MAXIMIZE,
        bounds=(lo, hi),
        seed=STAGED_SEED,
        refine_full=STAGED_REFINE_FULL,
        refine_iterations=STAGED_REFINE_ITERATIONS,
        max_stages=STAGED_MAX_STAGES,
        verbose=GLOBAL_VERBOSE,
    )
    # Warm start staged with Stage 1 best
    if de_res is None:
        raise RuntimeError("Stage 1 (DE/GA) failed to produce a result.")
    staged.full_best = de_res.best_position.copy()
    staged.full_best_fitness = float(de_res.best_fitness)
    staged.eval_count += 1  # account for reused fitness as "known"
    staged_res = staged.run()
    if GLOBAL_VERBOSE:
        print(f"[Hybrid] Staged refinement done best={staged_res.best_fitness:.6f}")

    # Synthetic history for compatibility
    staged_history = [lg.global_best_after for lg in staged_res.stage_logs]

    # -------- Stage 3: Optional PushPSO polish ------------
    push_section = None
    final_best_fitness = float(staged_res.best_fitness)
    final_best_position = staged_res.best_position.copy()
    final_source = "staged"

    if HYBRID_ENABLE_PUSH:
        try:
            from .ppso import PushPSO
            push = PushPSO(
                dim=encoder.dim,
                objective=base_objective,
                swarm_size=HYBRID_PUSH_SWARM,
                iterations=HYBRID_PUSH_ITERS,
                inertia=HYBRID_PUSH_INERTIA,
                cognitive=HYBRID_PUSH_COG,
                social=HYBRID_PUSH_SOC,
                reset_prob=HYBRID_PUSH_RESET_PROB,
                velocity_clamp=None,
                maximize=GLOBAL_MAXIMIZE,
                processes=None,
                seed=(HYBRID_SEED or 0) + 4242,
                crowd_radius=None,
                density_threshold=6,
                push_strength=HYBRID_PUSH_REPELLER_STRENGTH,
                adaptive_push=True,
                enable_crowding=False,   # 禁用昂贵 crowding
                stagnation_iter_threshold=20,
                cluster_eps=0.15,
                cluster_min_size=5,
                cluster_max_size=12,
                repeller_radius=HYBRID_PUSH_REPELLER_RADIUS,
                repeller_strength=HYBRID_PUSH_REPELLER_STRENGTH,
                repeller_decay=HYBRID_PUSH_REPELLER_DECAY,
                save_on_exit=None,
                include_swarm_on_save=False,
            )
            # Warm start around staged best
            center = staged_res.best_position
            noise = np.random.normal(0, 0.18, (HYBRID_PUSH_SWARM, center.shape[0]))
            push.positions = np.clip(center[None, :] + noise, -1.0, 1.0)
            push.personal_best_positions = push.positions.copy()
            if GLOBAL_MAXIMIZE:
                push.personal_best_fitness[:] = -np.inf
                push.global_best_fitness = -np.inf
            else:
                push.personal_best_fitness[:] = np.inf
                push.global_best_fitness = np.inf
            push.global_best_position = push.positions[0].copy()

            push_res = push.run()
            push_section = {
                "best_fitness": float(push_res.best_fitness),
                "best_position": push_res.best_position.tolist(),
                "history": push_res.history,
                "elapsed_sec": float(push_res.elapsed),
                "eval_count": int(push_res.eval_count),
            }
            if GLOBAL_VERBOSE:
                print(f"[Hybrid] Push polish done best={push_res.best_fitness:.6f}")
            improved = (push_res.best_fitness > final_best_fitness) if GLOBAL_MAXIMIZE else (push_res.best_fitness < final_best_fitness)
            if improved:
                final_best_fitness = float(push_res.best_fitness)
                final_best_position = push_res.best_position.copy()
                final_source = "push"
        except Exception as e:
            if GLOBAL_VERBOSE:
                print(f"[Hybrid] Push stage skipped ({e})")

    # -------- Per-missile times (optional) ---------------
    final_times = None
    if GLOBAL_SHOW_TIMES:
        try:
            times_fn = build_problem_times_fn(PROBLEM_ID, dt=HYBRID_DT_END)
            final_times = times_fn(final_best_position.tolist())
        except Exception:
            pass

    # -------- Unified model merge ------------------------
    section: Dict[str, Any] = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage1_algo": stage1_algo,
        "de_best_fitness": float(de_res.best_fitness),
        "staged_best_fitness": float(staged_res.best_fitness),
        "final_best_fitness": float(final_best_fitness),
        "source": final_source,
        "history_de": de_res.history,
        "history_staged": staged_history,
    }
    if push_section:
        section["push_best_fitness"] = push_section["best_fitness"]
        section["history_push"] = push_section["history"]

    merge_optimizer_section(PROBLEM_ID, "hybrid", section, maximize=GLOBAL_MAXIMIZE)

    return {
        "mode": "hybrid",
        "problem": PROBLEM_ID,
        "best_fitness": float(final_best_fitness),
        "best_position": final_best_position.tolist(),
        "elapsed_sec": None,
        "stage1_algo": stage1_algo,
        "history_de": de_res.history,
        "history_staged": staged_history,
        "history_push": push_section["history"] if push_section else None,
        "times": final_times,
        "source": final_source,
    }

# =============================================================================
# Main
# =============================================================================
def run_reduced() -> Dict[str, Any]:
    """
    Reduced-dimension optimization for Problem 5 using ReducedProblem5Encoder (50 dims).
    Falls back to normal problem 5 fitness pipeline by converting to full 55-dim raw vector.
    """
    if PROBLEM_ID != 5:
        raise ValueError("Reduced mode only supported when PROBLEM_ID == 5.")
    try:
        from .reduced_p5 import build_reduced_problem5_objective, reduced_to_full55
        from .encoders import Problem5Encoder
    except Exception as e:
        raise RuntimeError(f"Reduced encoder module not available: {e}")
    encoder, objective = build_reduced_problem5_objective(
        dt=GLOBAL_DT,
        aggregate=GLOBAL_AGGREGATE,
        weights=_maybe_weights()
    )
    # Reuse ParallelPSO core (could also allow PushPSO later)
    pso = ParallelPSO(
        dim=encoder.dim,
        objective=objective,
        swarm_size=PARALLEL_SWARM_SIZE,
        iterations=PARALLEL_ITERATIONS,
        inertia=PARALLEL_INERTIA,
        cognitive=PARALLEL_COGNITIVE,
        social=PARALLEL_SOCIAL,
        reset_prob=PARALLEL_RESET_PROB,
        velocity_clamp=PARALLEL_VELOCITY_CLAMP,
        maximize=GLOBAL_MAXIMIZE,
        processes=PARALLEL_PROCESSES,
        seed=PARALLEL_SEED,
        best_times_fn=None,
        init_model=None,
        perturb_std=PARALLEL_PERTURB_STD,
        save_on_exit=None,
        include_swarm_on_save=False,
    )
    res = pso.run()
    # Convert reduced best vector to full 55 then to a readable strategy
    full_raw = reduced_to_full55(res.best_position.tolist())
    # Use standard Problem5Encoder for final readable strategy
    std_enc = Problem5Encoder()
    strategy = std_enc.encode(full_raw)
    # (Optional) compute times
    if GLOBAL_SHOW_TIMES:
        from .pso import build_problem_times_fn
        times_fn = build_problem_times_fn(5, dt=GLOBAL_DT)
        times = times_fn(full_raw)
    else:
        times = None
    return {
        "mode": "reduced",
        "problem": 5,
        "best_fitness": float(res.best_fitness),
        "best_position": res.best_position.tolist(),  # reduced space vector
        "history": res.history,
        "elapsed_sec": float(res.elapsed),
        "eval_count": int(res.eval_count),
        "strategy": _safe_strategy_to_json(strategy),
        "times": times,
    }


def main():
    # Robust forced sampling K override: ensure later calls (or defaults elsewhere) cannot
    # silently revert to a larger K. If judge != "sample" this is a cheap no-op.
    if GLOBAL_JUDGE == "sample":
        try:
            from .pso import force_sample_K  # local import to avoid unused symbol when not sampling
            force_sample_K(GLOBAL_JUDGE_SAMPLE_K, verbose=GLOBAL_VERBOSE)
        except Exception as e:
            if GLOBAL_VERBOSE:
                print(f"[Judge] Failed to force sampling K={GLOBAL_JUDGE_SAMPLE_K} ({e}); continuing.", flush=True)
    select_judge(
        GLOBAL_JUDGE,
        K=GLOBAL_JUDGE_SAMPLE_K if GLOBAL_JUDGE == "sample" else 32,
        verbose=GLOBAL_VERBOSE
    )
    if MODE == "parallel":
        result = run_parallel()
    elif MODE == "block":
        result = run_block()
    elif MODE == "de":
        result = run_de()
    elif MODE == "dpso":
        result = run_dpso()
    elif MODE == "hybrid":
        result = run_hybrid()
    elif MODE == "push":
        result = run_push()
    elif MODE == "reduced":
        result = run_reduced()
    elif MODE == "staged":
        result = run_staged()
    else:
        raise ValueError(f"Unknown MODE={MODE}. Use 'parallel' | 'block' | 'de' | 'dpso' | 'hybrid' | 'push' | 'reduced' | 'staged'.")
    # ---------------- Universal final save snapshot ----------------
    try:
        save_sys = UniversalSaveSystem(problem_id=PROBLEM_ID, save_interval=1)
        # Derive a generic history (different optimizers use different keys)
        history = (
            result.get("history")
            or result.get("history_de")
            or result.get("history_staged")
            or result.get("history_push")
            or result.get("history_per_round")
            or []
        )
        iteration = len(history)
        # Best position may be absent for discrete mode; fall back to best_values
        best_position = result.get("best_position")
        if best_position is None and "best_values" in result:
            try:
                best_position = [float(v) for v in result["best_values"]]
            except Exception:
                best_position = []
        if best_position is None:
            best_position = []
        eval_count = int(result.get("eval_count", 0))
        elapsed_sec = result.get("elapsed_sec")
        if elapsed_sec is None:
            # Some modes (hybrid/staged) put elapsed inside sub-sections or omit; use 0 fallback
            elapsed_sec = 0.0
        save_state = SaveState(
            problem_id=PROBLEM_ID,
            optimizer_type=result.get("mode", MODE),
            iteration=iteration,
            best_fitness=float(result.get("best_fitness") if result.get("best_fitness") is not None else float("nan")),
            best_position=best_position,
            fitness_history=history,
            eval_count=eval_count,
            elapsed_time=float(elapsed_sec) if isinstance(elapsed_sec, (int, float)) else 0.0,
            optimizer_state={},  # Placeholder; extend by exposing internal state if needed
            save_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_iterations=iteration,   # If desired, replace with configured target iterations per mode
            dimensions=len(best_position),
            best_strategy=result.get("strategy"),
            best_times=result.get("times"),
        )
        save_sys.save_progress(save_state, force=True)
    except Exception as e:
        if GLOBAL_VERBOSE:
            print(f"[SaveSystem] Final save failed: {e}", flush=True)
    summary_keys = ["mode", "problem", "best_fitness", "elapsed_sec"]
    summary = {k: result[k] for k in summary_keys if k in result}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result

# ---------------------------------------------------------------------------
# Unified per-problem model save/load helpers
# ---------------------------------------------------------------------------
def _unified_model_path(problem_id: int) -> str:
    """
    Resolve (and create if needed) the unified per-problem model JSON path.
    Filename pattern: problem{problem_id}_latest.json
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, f"problem{problem_id}_latest.json")


def load_unified_model(problem_id: int) -> dict:
    """
    Load the unified per-problem JSON (if exists). Returns {} on any failure.
    Structure (evolving):
    {
      "schema": 1,
      "problem": <int>,
      "best_fitness": <float>,
      "best_position": [...],
      "optimizers": {
         "parallel": {...},
         "block": {...},
         "de": {...},
         "dpso": {...},
         "hybrid": {...},
         "push": {...}
      }
    }
    """
    path = _unified_model_path(problem_id)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def merge_optimizer_section(problem_id: int,
                            optimizer: str,
                            section: dict,
                            maximize: bool = True) -> str:
    """
    Merge (or insert) an optimizer-specific result section into the unified
    per-problem JSON. Also maintains top-level best_fitness / best_position
    for quick interchange across optimizers.

    Args:
      problem_id : problem identifier
      optimizer  : short key ("parallel", "block", "de", "dpso", "hybrid", "push", ...)
      section    : dict containing at least best_fitness (and optionally best_position, history, etc.)
      maximize   : whether larger fitness is better (default True)

    Returns:
      Path to the unified JSON file.
    """
    data = load_unified_model(problem_id)
    optimizers = data.get("optimizers", {})
    optimizers[optimizer] = section
    data["optimizers"] = optimizers
    data.setdefault("schema", 1)
    data["problem"] = problem_id

    new_best = section.get("best_fitness")
    if new_best is not None:
        current_best = data.get("best_fitness")
        if current_best is None:
            data["best_fitness"] = new_best
            if "best_position" in section:
                data["best_position"] = section["best_position"]
        else:
            better = (new_best > current_best) if maximize else (new_best < current_best)
            if better:
                data["best_fitness"] = new_best
                if "best_position" in section:
                    data["best_position"] = section["best_position"]

    path = _unified_model_path(problem_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if GLOBAL_VERBOSE:
        print(f"[UNIFIED] Merged optimizer='{optimizer}' into {os.path.basename(path)}", flush=True)
    return path

if __name__ == "__main__":
    main()
