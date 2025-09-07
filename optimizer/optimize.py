"""Constant-based optimization script for PSO, BlockPSO, and Differential Evolution (DE).

Run with:
    python -m optimizer.optimize

Modes:
  MODE = "parallel"  -> single ParallelPSO run
  MODE = "block"     -> block-based multi-PSO (regional elimination)
  MODE = "de"        -> Differential Evolution

Configuration sections:
  GLOBAL_*           : judge & shared simulation controls
  PARALLEL_*         : parameters for ParallelPSO
  BLOCK_*            : parameters for BlockPSO (only when MODE == "block")
  DE_*               : parameters for DifferentialEvolution (only when MODE == "de")

Produced outputs:
  - Console concise logs (per iteration / round / generation).
  - Summary JSON-like dict printed at the end.
  - Optional model auto-save (all optimizers); BlockPSO saves summary, DE/PSO save full JSON.

"""
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Sequence, Optional

from .pso import (
    ParallelPSO,
    build_problem_objective,
    build_problem_times_fn,
    select_judge,
)

try:
    from .block_pso import BlockPSO  # type: ignore
except Exception:  # pragma: no cover
    BlockPSO = None  # type: ignore

try:
    from .de import DifferentialEvolution  # type: ignore
except Exception:  # pragma: no cover
    DifferentialEvolution = None  # type: ignore

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Mode: "parallel" | "block" | "de"
MODE: str = "de"

# Problem id (2..5)
PROBLEM_ID: int = 4

# Judge: "rough" or "sample"
GLOBAL_JUDGE: str = "rough"
# Sampling K (only used when GLOBAL_JUDGE == "sample")
GLOBAL_JUDGE_SAMPLE_K: int = 24

# Simulation dt
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
PARALLEL_INIT_MODEL: Optional[str] = "problem4_latest"
PARALLEL_PERTURB_STD: float = 0.15
PARALLEL_SAVE_MODEL: Optional[str] = "problem4_latest"
PARALLEL_INCLUDE_SWARM_ON_SAVE: bool = True

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
BLOCK_SAVE_MODEL: Optional[str] = "block_problem4_latest"

# ---------------- Differential Evolution specific ----------------
DE_GENERATIONS: int = 60
DE_POPULATION_SIZE: int = 80
DE_F: float = 0.8
DE_CR: float = 0.9
DE_PROCESSES: Optional[int] = None
DE_SEED: Optional[int] = 123
DE_INIT_MODEL: Optional[str] = "problem4_de_latest"
DE_PERTURB_STD: float = 0.15
DE_SAVE_MODEL: Optional[str] = "problem4_de_latest"
DE_INCLUDE_POP_ON_SAVE: bool = True

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
        save_on_exit=PARALLEL_SAVE_MODEL,
        include_swarm_on_save=PARALLEL_INCLUDE_SWARM_ON_SAVE,
    )
    result = pso.run()
    strategy = encoder.encode(list(result.best_position))
    final_times = None
    if GLOBAL_SHOW_TIMES:
        final_times = best_times_fn(list(result.best_position)) if best_times_fn else build_problem_times_fn(PROBLEM_ID, dt=GLOBAL_DT)(list(result.best_position))
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
        payload = {
            "schema": 1,
            "mode": "block",
            "problem": PROBLEM_ID,
            "dim": encoder.dim,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "best_fitness": float(result.best_fitness),
            "best_position": result.best_position.tolist(),
            "round_history": result.block_histories,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if GLOBAL_VERBOSE:
                print(f"[BlockPSO] Summary saved {path}")
        except Exception as e:
            print(f"[BlockPSO] Save failed ({e})")
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

# =============================================================================
# Main
# =============================================================================
def main():
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
    else:
        raise ValueError(f"Unknown MODE={MODE}. Use 'parallel' | 'block' | 'de'.")
    summary_keys = ["mode", "problem", "best_fitness", "elapsed_sec"]
    summary = {k: result[k] for k in summary_keys}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result

if __name__ == "__main__":
    main()
