"""
Discrete Particle Swarm Optimization (DPSO) with warm start & JSON persistence.

Design goals:
  - Support generic per-dimension categorical (discrete) domains.
  - Reuse the multiprocessing evaluation helpers from pso.py for consistency.
  - API style mirrors ParallelPSO / DifferentialEvolution for interchangeability.
  - Warm start from JSON (either full swarm indices or only best indices).
  - Clean, easily extensible core (probability / score based discrete transitions).
  - Fully self‑contained: no other module modifications required.

Core concept (multi-category extension of Binary PSO):
  Each particle maintains, for every dimension, a score vector over that dimension's
  categories (akin to logits). After updating scores, a softmax yields sampling
  probabilities for the next categorical choice.

Score update:
    S = w * S
    S[i,d, personal_best_idx] += c1 * r1
    S[i,d, global_best_idx]   += c2 * r2

Where:
  - w  : inertia (decays past preference)
  - c1 : cognitive weight
  - c2 : social weight
  - r1, r2 ~ U(0,1) per (particle, dim)
  - Optionally add small Gaussian noise to maintain exploration (noise_std).

After update:
  probs = softmax(S[i,d, :Ld]) over the Ld categories (Ld may differ by dim)
  new_index sampled from probs.

Random reset:
  With probability 'reset_prob' per particle per iteration, all dimension choices
  are reinitialized uniformly (scores zeroed to avoid bias).

Bounds / Domain:
  Provided as 'categories': List[List[Any]], length = dim, each inner list is the
  set of possible values for that dimension (>=1 element). Internally, we track
  only the index for efficiency; to evaluate objective we either:
    - pass indices directly (pass_indices=True), or
    - map to the actual values (pass_indices=False; default).

JSON persistence schema (schema = 1):
{
  "schema": 1,
  "dim": <int>,
  "categories": [[...], [...], ...],          # all per-dimension categories
  "saved_at": "2025-09-07T12:34:56Z",
  "best_fitness": 123.456,
  "best_indices": [0, 2, 5, ...],
  "swarm_indices": [[...], [...], ...],       # optional full swarm
  "notes": "optional free-form text"
}

Warm start logic:
  - If 'swarm_indices' present and shape matches (N, dim), adapt to swarm_size
    by truncation or tiling.
  - Else if only 'best_indices' present, replicate plus random mutations controlled
    by 'perturb_prob' (each dimension independently mutated).
  - Dimension / cardinality mismatch raises ValueError.

Usage example:

    from optimizer.dpso import DiscretePSO, build_integer_categories

    # Suppose we need 5 integer dimensions with cardinalities [3,4,5,2,6]
    categories = build_integer_categories([3,4,5,2,6])
    def objective(vec):
        # vec is list of actual integers by default
        return -sum(v*v for v in vec)  # maximize negative square (toy)

    dpso = DiscretePSO(
        categories=categories,
        objective=objective,
        swarm_size=60,
        iterations=80,
        inertia=0.6,
        cognitive=1.2,
        social=1.4,
        reset_prob=0.03,
        noise_std=0.05,
        maximize=True,
        processes=None,
        seed=42,
        init_model="toy_integer_dpso",   # optional warm start
        save_on_exit="toy_integer_dpso"  # auto-save after run
    )
    result = dpso.run()
    print(result.best_fitness, result.best_indices, result.best_values)

Notes:
  - Objective must be pickleable for multiprocessing.
  - If categories contain non-numeric values and you pass_indices=False (default),
    they are forwarded verbatim to objective.
  - Set pass_indices=True if objective operates on integer indices directly.

Potential future extensions (not implemented):
  - Per-dimension adaptive cooling of inertia / noise.
  - Hybrid local search on the global best.
  - Elitist injection (force keep top-k unchanged each iteration).
"""

from __future__ import annotations

import os
import json
import time
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import multiprocessing as mp

# Reuse evaluation helpers from PSO
try:
    from .pso import _worker_init, _worker_eval_vector  # type: ignore
except ImportError:  # Fallback path if relative import context differs
    from pso import _worker_init, _worker_eval_vector  # type: ignore


__all__ = [
    "DiscretePSO",
    "DiscretePSOResult",
    "build_integer_categories",
]


# --------------------------------------------------------------------------- #
# Dataclass for results
# --------------------------------------------------------------------------- #
@dataclass
class DiscretePSOResult:
    best_indices: np.ndarray             # (dim,)
    best_values: List[Any]               # mapped values (len=dim)
    best_fitness: float
    history: List[float]                 # global best fitness by iteration
    eval_count: int
    elapsed: float


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_integer_categories(cardinalities: Sequence[int]) -> List[List[int]]:
    """
    Convenience helper: given cardinalities like [3,4,5], produce
      [[0,1,2], [0,1,2,3], [0,1,2,3,4]]

    Args:
      cardinalities: sequence of positive ints

    Returns:
      list-of-lists categories
    """
    out: List[List[int]] = []
    for c in cardinalities:
        if c <= 0:
            raise ValueError("Cardinality must be positive")
        out.append(list(range(c)))
    return out


# --------------------------------------------------------------------------- #
# Main Discrete PSO
# --------------------------------------------------------------------------- #
class DiscretePSO:
    """
    Discrete / Categorical Particle Swarm Optimization.

    Parameters:
      categories         : list[list[Any]] per dimension; each inner list length >=1
      objective          : Callable(vector_or_indices) -> fitness (float)
      swarm_size         : number of particles
      iterations         : number of iterations
      inertia            : w (float >=0)
      cognitive          : c1
      social             : c2
      reset_prob         : probability (per particle per iteration) of random re-init
      noise_std          : stddev of Gaussian noise added to scores each update
      maximize           : True (maximize) or False (minimize)
      processes          : number of worker processes (None => cpu_count-1)
      seed               : base random seed
      pass_indices       : if True, objective receives list[int] (indices);
                           else it receives actual mapped values categories[d][idx]
      init_model         : optional base name for warm start JSON
      perturb_prob       : used ONLY when warm-start has only best_indices;
                           probability of mutating each dimension in replication
      models_dir         : override models directory (default ../models)
      save_on_exit       : auto-save JSON model with this base name after run
      include_swarm_on_save: whether to include full swarm_indices in JSON
      temperature_decay  : optional multiplicative decay for (c1,c2) each iteration
                           e.g. 0.995 -> gradually reduce exploitation
      min_score_clip     : numeric lower clip for scores (avoid runaway magnitude)
      max_score_clip     : numeric upper clip for scores
    """
    def __init__(self,
                 categories: List[List[Any]],
                 objective: Callable[[Sequence[Any]], float],
                 swarm_size: int = 50,
                 iterations: int = 100,
                 inertia: float = 0.7,
                 cognitive: float = 1.5,
                 social: float = 1.5,
                 reset_prob: float = 0.02,
                 noise_std: float = 0.0,
                 maximize: bool = True,
                 processes: Optional[int] = None,
                 seed: Optional[int] = None,
                 pass_indices: bool = False,
                 init_model: Optional[str] = None,
                 perturb_prob: float = 0.1,
                 models_dir: Optional[str] = None,
                 save_on_exit: Optional[str] = None,
                 include_swarm_on_save: bool = True,
                 temperature_decay: Optional[float] = None,
                 min_score_clip: float = -20.0,
                 max_score_clip: float = 20.0):
        if not categories or not isinstance(categories, list):
            raise ValueError("categories must be a non-empty list of lists")
        dim = len(categories)
        for i, c in enumerate(categories):
            if not isinstance(c, list) or len(c) == 0:
                raise ValueError(f"categories[{i}] must be non-empty list")
        self.categories = categories
        self.dim = dim
        self.objective = objective
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = float(inertia)
        self.c1 = float(cognitive)
        self.c2 = float(social)
        self.reset_prob = float(reset_prob)
        self.noise_std = float(noise_std)
        self.maximize = maximize
        self.processes = processes or max(1, mp.cpu_count() - 1)
        self.seed = seed or int(time.time())
        self.pass_indices = pass_indices
        self.init_model = init_model
        self.perturb_prob = float(perturb_prob)
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.save_on_exit = save_on_exit
        self.include_swarm_on_save = include_swarm_on_save
        self.temperature_decay = temperature_decay
        self.min_score_clip = float(min_score_clip)
        self.max_score_clip = float(max_score_clip)

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Precompute cardinalities
        self.cardinalities = [len(c) for c in self.categories]
        self.max_card = max(self.cardinalities)
        # Mask (dim, max_card): True where valid
        self.valid_mask = np.zeros((self.dim, self.max_card), dtype=bool)
        for d, L in enumerate(self.cardinalities):
            self.valid_mask[d, :L] = True

        # Swarm: indices shape (swarm_size, dim)
        self.indices = np.empty((self.swarm_size, self.dim), dtype=np.int32)
        for i in range(self.swarm_size):
            for d, L in enumerate(self.cardinalities):
                self.indices[i, d] = random.randrange(L)

        # Scores tensor (swarm_size, dim, max_card) -> logits before softmax
        self.scores = np.zeros((self.swarm_size, self.dim, self.max_card), dtype=float)

        # Personal best
        self.personal_best_indices = self.indices.copy()
        if self.maximize:
            self.personal_best_fitness = np.full(self.swarm_size, -np.inf)
            self.global_best_fitness = -np.inf
        else:
            self.personal_best_fitness = np.full(self.swarm_size, np.inf)
            self.global_best_fitness = np.inf
        self.global_best_indices = self.indices[0].copy()

        # Warm start (if requested)
        if self.init_model:
            try:
                self._warm_start(self.init_model)
            except Exception as e:
                print(f"[DPSO] Warm start failed ({e}); using random initialization.", flush=True)

    # --------------------------------------------------------------------- #
    # Warm Start
    # --------------------------------------------------------------------- #
    def _resolve_model_path(self, name: str) -> str:
        if not name.endswith(".json"):
            name += ".json"
        return os.path.join(self.models_dir, name)

    def _warm_start(self, name: str):
        path = self._resolve_model_path(name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Warm start JSON root must be an object")
        if data.get("dim") != self.dim:
            raise ValueError(f"Warm start dim mismatch file={data.get('dim')} expected={self.dim}")
        # Validate category shapes
        wcats = data.get("categories")
        if wcats is not None:
            if len(wcats) != self.dim:
                raise ValueError("categories length mismatch in warm start file")
            for d, (a, b) in enumerate(zip(wcats, self.categories)):
                if len(a) != len(b):
                    raise ValueError(f"categories[{d}] cardinality mismatch warm={len(a)} current={len(b)}")
        if "swarm_indices" in data:
            arr = np.asarray(data["swarm_indices"], dtype=int)
            if arr.ndim != 2 or arr.shape[1] != self.dim:
                raise ValueError("swarm_indices shape mismatch")
            if arr.shape[0] >= self.swarm_size:
                self.indices = arr[:self.swarm_size].copy()
            else:
                reps = (self.swarm_size + arr.shape[0] - 1) // arr.shape[0]
                self.indices = np.tile(arr, (reps, 1))[:self.swarm_size].copy()
            print(f"[DPSO] Warm start: loaded {arr.shape[0]} swarm indices.", flush=True)
        elif "best_indices" in data:
            best = np.asarray(data["best_indices"], dtype=int)
            if best.shape != (self.dim,):
                raise ValueError("best_indices shape mismatch")
            base = np.tile(best, (self.swarm_size, 1))
            # Mutate each dimension with probability perturb_prob
            for i in range(self.swarm_size):
                for d, L in enumerate(self.cardinalities):
                    if random.random() < self.perturb_prob:
                        base[i, d] = random.randrange(L)
            self.indices = base
            print(f"[DPSO] Warm start: replicated best_indices with perturb_prob={self.perturb_prob}.", flush=True)
        else:
            raise ValueError("Warm start JSON must contain 'swarm_indices' or 'best_indices'")
        # Reset dependent state
        self.personal_best_indices = self.indices.copy()
        if self.maximize:
            self.personal_best_fitness[:] = -np.inf
            self.global_best_fitness = -np.inf
        else:
            self.personal_best_fitness[:] = np.inf
            self.global_best_fitness = np.inf
        self.global_best_indices = self.indices[0].copy()
        self.scores.fill(0.0)
        print(f"[DPSO] Warm start complete swarm_size={self.swarm_size} dim={self.dim}", flush=True)

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save_model(self,
                   name: str,
                   include_swarm: bool = True,
                   notes: Optional[str] = None,
                   extra: Optional[Dict[str, Any]] = None):
        os.makedirs(self.models_dir, exist_ok=True)
        path = self._resolve_model_path(name)
        payload: Dict[str, Any] = {
            "schema": 1,
            "dim": self.dim,
            "categories": self.categories,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "best_fitness": float(self.global_best_fitness),
            "best_indices": self.global_best_indices.tolist(),
        }
        if include_swarm:
            payload["swarm_indices"] = self.indices.tolist()
        if notes:
            payload["notes"] = notes
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[DPSO] Model saved to {path}", flush=True)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _indices_to_values(self, idx_row: np.ndarray) -> List[Any]:
        return [self.categories[d][int(idx_row[d])] for d in range(self.dim)]

    def _evaluate_batch(self, batch_indices: np.ndarray) -> List[float]:
        """
        Evaluate a batch of index-vectors using multiprocessing pool.
        batch_indices: (B, dim) int array
        """
        # Prepare inputs (list of Sequence[Any] or indices)
        if self.pass_indices:
            feed = [list(map(int, row)) for row in batch_indices]
        else:
            feed = [self._indices_to_values(row) for row in batch_indices]

        # We'll serialize each vector as a tuple and re-map inside worker via a closure,
        # but simpler is to build a local objective wrapper referencing self.objective.
        # However we intentionally reuse global worker pattern: we pass feed elements directly.
        # So objective must accept the vector shape we chose.
        # To keep consistent with _worker_eval_vector which only accepts a single vector,
        # we spawn pool.map over feed (the private global objective is set in initializer).
        with mp.Pool(processes=self.processes,
                     initializer=_worker_init,
                     initargs=(self.seed, self.objective, self.maximize)) as pool:
            fitness = pool.map(_worker_eval_vector, feed)
        return fitness

    def _softmax_rows(self, logits: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        logits shape: (swarm_size, dim, max_card)
        valid_mask shape: (dim, max_card)
        Returns probabilities same shape; invalid entries = 0.
        """
        # Mask invalid logits with large negative
        masked = np.where(valid_mask[None, :, :], logits, -1e9)
        # To improve numerical stability:
        max_logits = masked.max(axis=2, keepdims=True)
        exps = np.exp(masked - max_logits)
        exps *= valid_mask[None, :, :]
        sums = exps.sum(axis=2, keepdims=True)
        # Avoid divide by zero by fallback uniform among valid if sums==0
        zero_mask = (sums <= 0)
        probs = np.divide(exps, sums, out=np.zeros_like(exps), where=(sums > 0))
        if zero_mask.any():
            # For rows with all zero, assign uniform over valid
            valid_counts = valid_mask.sum(axis=1)
            for i in range(self.swarm_size):
                for d in range(self.dim):
                    if zero_mask[i, d, 0]:  # sums==0 for this (i,d)
                        L = valid_counts[d]
                        probs[i, d, :L] = 1.0 / L
        return probs

    def _sample_new_indices(self, probs: np.ndarray):
        # probs shape (swarm_size, dim, max_card)
        for i in range(self.swarm_size):
            for d, L in enumerate(self.cardinalities):
                p = probs[i, d, :L]
                # Numerical guard
                s = p.sum()
                if s <= 0:
                    p = np.full(L, 1.0 / L)
                else:
                    p = p / s
                self.indices[i, d] = np.random.choice(L, p=p)

    # --------------------------------------------------------------------- #
    # Core run
    # --------------------------------------------------------------------- #
    def run(self, iterations: Optional[int] = None) -> DiscretePSOResult:
        start = time.time()
        iter_count = self.iterations if iterations is None else int(iterations)
        history: List[float] = []
        eval_count = 0

        # Initial evaluation (optional: could skip if personal_best_fitness all inf)
        init_fitness = self._evaluate_batch(self.indices)
        eval_count += len(init_fitness)
        for i, fit in enumerate(init_fitness):
            self.personal_best_fitness[i] = fit
            self.personal_best_indices[i] = self.indices[i].copy()
        # Global best
        if self.maximize:
            best_idx = int(np.argmax(self.personal_best_fitness))
        else:
            best_idx = int(np.argmin(self.personal_best_fitness))
        self.global_best_fitness = float(self.personal_best_fitness[best_idx])
        self.global_best_indices = self.personal_best_indices[best_idx].copy()
        history.append(self.global_best_fitness)

        c1 = self.c1
        c2 = self.c2

        for it in range(iter_count):
            # Optional temperature decay
            if self.temperature_decay:
                c1 *= self.temperature_decay
                c2 *= self.temperature_decay

            # Random resets
            reset_mask = np.random.rand(self.swarm_size) < self.reset_prob
            if reset_mask.any():
                for idx in np.where(reset_mask)[0]:
                    for d, L in enumerate(self.cardinalities):
                        self.indices[idx, d] = random.randrange(L)
                    self.scores[idx, :, :] = 0.0  # Reset scores for that particle

            # Update scores / probabilities
            # Apply inertia
            self.scores *= self.w

            # Add cognitive & social components
            # r1, r2 per (particle, dim)
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            for i in range(self.swarm_size):
                for d in range(self.dim):
                    pb = int(self.personal_best_indices[i, d])
                    gb = int(self.global_best_indices[d])
                    self.scores[i, d, pb] += c1 * r1[i, d]
                    self.scores[i, d, gb] += c2 * r2[i, d]

            # Optional noise (exploration)
            if self.noise_std > 0:
                noise = np.random.normal(0.0, self.noise_std, self.scores.shape)
                noise *= self.valid_mask[None, :, :]
                self.scores += noise

            # Clip scores to avoid overflow
            np.clip(self.scores, self.min_score_clip, self.max_score_clip, out=self.scores)

            # Convert to probabilities & sample
            probs = self._softmax_rows(self.scores, self.valid_mask)
            self._sample_new_indices(probs)

            # Evaluate new population
            fitness = self._evaluate_batch(self.indices)
            eval_count += len(fitness)

            # Personal / global best updates
            for i, fit in enumerate(fitness):
                better = fit > self.personal_best_fitness[i] if self.maximize else fit < self.personal_best_fitness[i]
                if better:
                    self.personal_best_fitness[i] = fit
                    self.personal_best_indices[i] = self.indices[i].copy()

            if self.maximize:
                best_idx = int(np.argmax(self.personal_best_fitness))
                best_fit = float(self.personal_best_fitness[best_idx])
                if best_fit > self.global_best_fitness:
                    self.global_best_fitness = best_fit
                    self.global_best_indices = self.personal_best_indices[best_idx].copy()
            else:
                best_idx = int(np.argmin(self.personal_best_fitness))
                best_fit = float(self.personal_best_fitness[best_idx])
                if best_fit < self.global_best_fitness:
                    self.global_best_fitness = best_fit
                    self.global_best_indices = self.personal_best_indices[best_idx].copy()

            history.append(self.global_best_fitness)
            print(f"[DPSO] Iter {it+1}/{iter_count} best_fitness={self.global_best_fitness:.6f}", flush=True)

        elapsed = time.time() - start
        best_values = self._indices_to_values(self.global_best_indices)
        result = DiscretePSOResult(
            best_indices=self.global_best_indices.copy(),
            best_values=best_values,
            best_fitness=float(self.global_best_fitness),
            history=history,
            eval_count=eval_count,
            elapsed=elapsed
        )
        if self.save_on_exit:
            try:
                self.save_model(self.save_on_exit, include_swarm=self.include_swarm_on_save)
            except Exception as e:
                print(f"[DPSO] Auto-save failed: {e}", flush=True)
        return result


# --------------------------------------------------------------------------- #
# Example self-test (only runs when module executed directly)
# --------------------------------------------------------------------------- #
def _example():
    # Simple categorical optimization: choose letters to maximize custom score.
    categories = [
        list("ABCDE"),          # dim 0
        list("XYZ"),            # dim 1
        list("KL"),             # dim 2
        list("MNOP"),           # dim 3
    ]
    target = ("D", "X", "L", "P")

    def objective(values: Sequence[str]) -> float:
        # Score = +1 for each match + small random noise
        score = sum(v == t for v, t in zip(values, target))
        return score + random.random() * 0.1  # tie-breaking

    dpso = DiscretePSO(
        categories=categories,
        objective=objective,
        swarm_size=40,
        iterations=30,
        inertia=0.6,
        cognitive=1.3,
        social=1.4,
        reset_prob=0.05,
        noise_std=0.02,
        maximize=True,
        seed=123,
        init_model=None,
        save_on_exit=None
    )
    result = dpso.run()
    print("[DPSO Example] Best fitness:", result.best_fitness)
    print("[DPSO Example] Best values:", result.best_values)

if __name__ == "__main__":  # pragma: no cover
    _example()
