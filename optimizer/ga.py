from __future__ import annotations

import os
import json
import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence, List, Optional, Tuple, Union, Any, Dict

import numpy as np  # type: ignore
import multiprocessing as mp

# Reuse worker initialization helpers from PSO / DE for consistent parallel evaluation
from .pso import _worker_init, _worker_eval_vector  # type: ignore


@dataclass
class GAResult:
    best_position: np.ndarray
    best_fitness: float
    history: List[float]
    eval_count: int
    elapsed: float
    generations: int
    early_stopped: bool


class GeneticAlgorithm:
    """
    Parallel Genetic Algorithm for continuous domains in [-1,1] (or provided bounds).

    Features:
      - Uniform / tournament selection with elitism (elite_fraction).
      - Uniform crossover (per-gene selection between two parents) with probability crossover_rate; otherwise gene copied from first parent.
      - Gaussian mutation with per-gene probability mutation_rate and sigma = mutation_sigma_frac * span (span = hi-lo).
      - Warm start from JSON model (swarm_positions / best_position) similar to PSO & DE.
      - Jitter seeding: portion (jitter_seed_fraction) of population replaced by noisy copies of best seed solution.
      - Multiprocessing evaluation identical to DE / PSO helpers for consistency.
      - Early stopping on stagnation_patience (no global improvement).

    Parameters:
      dim: dimension
      objective: Callable[[vector], float]
      population_size
      generations
      elite_fraction: fraction of population copied unchanged (>=1 individual if >0)
      tournament_k: tournament size for parent selection
      crossover_rate
      mutation_rate
      mutation_sigma_frac
      jitter_seed_fraction
      stagnation_patience
      maximize: True to maximize objective
      processes: worker processes (None -> cpu_count - 1)
      seed: RNG seed
      best_times_fn: optional callback(vector)->List[float] for verbose per-generation logging
      init_model: optional JSON file base name (under ../models) for warm start
      perturb_std: stddev when seeding around best_position if no population saved
      bounds: (lo, hi) arrays or lists; if None defaults to [-1,1]
      save_on_exit: if provided, save model with best + population
      include_swarm_on_save

    JSON schema (compatible subset):
      {
        "schema": 1,
        "dim": ...,
        "saved_at": "...Z",
        "best_fitness": ...,
        "best_position": [...],
        "swarm_positions": [[...], ...]
      }
    """
    def __init__(self,
                 dim: int,
                 objective: Callable[[Sequence[float]], float],
                 population_size: int = 1024,
                 generations: int = 200,
                 elite_fraction: float = 0.01,
                 tournament_k: int = 2,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.15,
                 mutation_sigma_frac: float = 0.08,
                 jitter_seed_fraction: float = 0.8,
                 stagnation_patience: int = 10,
                 maximize: bool = True,
                 processes: Optional[int] = None,
                 seed: Optional[int] = None,
                 best_times_fn: Optional[Callable[[Sequence[float]], List[float]]] = None,
                 init_model: Optional[str] = None,
                 perturb_std: float = 0.15,
                 bounds: Optional[Tuple[Union[Sequence[float], np.ndarray], Union[Sequence[float], np.ndarray]]] = None,
                 save_on_exit: Optional[str] = None,
                 include_swarm_on_save: bool = True,
                 models_dir: Optional[str] = None):
        self.dim = dim
        self.objective = objective
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.elite_fraction = max(0.0, min(1.0, float(elite_fraction)))
        self.tournament_k = max(2, int(tournament_k))
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.mutation_sigma_frac = float(mutation_sigma_frac)
        self.jitter_seed_fraction = max(0.0, min(1.0, float(jitter_seed_fraction)))
        self.stagnation_patience = max(1, int(stagnation_patience))
        self.maximize = maximize
        self.processes = processes or max(1, mp.cpu_count() - 1)
        self.seed = seed or int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.best_times_fn = best_times_fn
        self.init_model = init_model
        self.perturb_std = float(perturb_std)
        self.save_on_exit = save_on_exit
        self.include_swarm_on_save = include_swarm_on_save
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

        if bounds is not None:
            lo = np.asarray(bounds[0], dtype=float)
            hi = np.asarray(bounds[1], dtype=float)
        else:
            lo = np.full(dim, -1.0)
            hi = np.full(dim, 1.0)
        self.bounds = (lo, hi)
        self.span = hi - lo

        # Initialize population uniformly
        self.population = lo + (hi - lo) * np.random.rand(self.population_size, dim)

        if self.maximize:
            self.fitness = np.full(self.population_size, -np.inf)
            self.best_fitness = -np.inf
        else:
            self.fitness = np.full(self.population_size, np.inf)
            self.best_fitness = np.inf
        self.best_position = self.population[0].copy()

        if self.init_model:
            self._warm_start(self.init_model)

    # Warm start similar to DE
    def _resolve_model_path(self, name: str) -> str:
        if not name.endswith(".json"):
            name += ".json"
        return os.path.join(self.models_dir, name)

    def _warm_start(self, name: str):
        path = self._resolve_model_path(name)
        if not os.path.isfile(path):
            # create stub
            os.makedirs(self.models_dir, exist_ok=True)
            stub = {
                "schema": 1,
                "dim": self.dim,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "best_fitness": float(-np.inf if self.maximize else np.inf),
                "best_position": self.population[0].tolist()
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(stub, f, ensure_ascii=False, indent=2)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        lo, hi = self.bounds
        if "swarm_positions" in data:
           arr = np.asarray(data["swarm_positions"], dtype=float)
           if arr.shape[1] == self.dim:
               # tile or truncate
               if arr.shape[0] >= self.population_size:
                   self.population = arr[:self.population_size].copy()
               else:
                   reps = (self.population_size + arr.shape[0] - 1)//arr.shape[0]
                   self.population = np.tile(arr, (reps,1))[:self.population_size].copy()
        elif "best_position" in data:
           bp = np.asarray(data["best_position"], dtype=float)
           noise = np.random.normal(0.0, self.perturb_std, (self.population_size, self.dim))
           self.population = bp[None,:] + noise
        np.clip(self.population, lo, hi, out=self.population)

        # Jitter seed fraction exploitation
        if self.jitter_seed_fraction > 0.0:
            count = int(round(self.jitter_seed_fraction * self.population_size))
            if count > 0:
                base = self.population[0].copy()
                jitter = np.random.normal(0.0, self.perturb_std, (count, self.dim))
                self.population[:count] = np.clip(base[None,:] + jitter, lo, hi)

    def save_model(self, name: str, include_swarm: bool = True, notes: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        os.makedirs(self.models_dir, exist_ok=True)
        path = self._resolve_model_path(name)
        payload: Dict[str, Any] = {
            "schema": 1,
            "dim": self.dim,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "best_fitness": float(self.best_fitness),
            "best_position": self.best_position.tolist()
        }
        if include_swarm:
            payload["swarm_positions"] = self.population.tolist()
        if notes:
            payload["notes"] = notes
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[GA] Model saved to {path}", flush=True)

    def _evaluate_population(self, pool, pop: np.ndarray) -> List[float]:
        return pool.map(_worker_eval_vector, pop)

    # Tournament parent selection returning index
    def _tournament(self, fit: np.ndarray) -> int:
        k = self.tournament_k
        idxs = random.sample(range(self.population_size), k)
        if self.maximize:
            return max(idxs, key=lambda i: fit[i])
        else:
            return min(idxs, key=lambda i: fit[i])

    def _uniform_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        mask = np.random.rand(self.dim) < self.crossover_rate
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, child: np.ndarray):
        lo, hi = self.bounds
        span = self.span
        mut_mask = np.random.rand(self.dim) < self.mutation_rate
        if not mut_mask.any():
            return
        sigma = self.mutation_sigma_frac * span
        child[mut_mask] += np.random.normal(0.0, sigma[mut_mask])
        np.clip(child, lo, hi, out=child)

    def run(self, generations: Optional[int] = None) -> GAResult:
        start = time.time()
        gen_target = self.generations if generations is None else int(generations)
        eval_count = 0
        history: List[float] = []
        lo, hi = self.bounds

        with mp.Pool(processes=self.processes, initializer=_worker_init, initargs=(self.seed, self.objective, self.maximize)) as pool:
            # initial eval
            fit_vals = self._evaluate_population(pool, self.population)
            eval_count += len(fit_vals)
            self.fitness[:] = fit_vals
            if self.maximize:
                best_i = int(np.argmax(self.fitness))
            else:
                best_i = int(np.argmin(self.fitness))
            self.best_fitness = float(self.fitness[best_i])
            self.best_position = self.population[best_i].copy()
            history.append(float(self.best_fitness))

            stagnation = 0

            for g in range(gen_target):
                # Determine elite indices
                elite_count = max(1, int(round(self.elite_fraction * self.population_size))) if self.elite_fraction > 0.0 else 0
                if elite_count > 0:
                    if self.maximize:
                        elite_order = np.argsort(-self.fitness)
                    else:
                        elite_order = np.argsort(self.fitness)
                    elites = self.population[elite_order[:elite_count]].copy()
                    elites_fit = self.fitness[elite_order[:elite_count]].copy()
                else:
                    elites = np.empty((0, self.dim))
                    elites_fit = np.empty((0,))

                # Generate offspring
                offspring_count = self.population_size - elite_count
                new_pop = []
                while len(new_pop) < offspring_count:
                    p1 = self.population[self._tournament(self.fitness)]
                    p2 = self.population[self._tournament(self.fitness)]
                    if p1 is p2:
                        # re-draw
                        continue
                    child = self._uniform_crossover(p1, p2)
                    self._mutate(child)
                    new_pop.append(child)
                new_pop_arr = np.asarray(new_pop, dtype=float)
                if elite_count > 0:
                    self.population = np.vstack([elites, new_pop_arr])
                else:
                    self.population = new_pop_arr

                # Evaluate
                fit_vals = self._evaluate_population(pool, self.population)
                eval_count += len(fit_vals)
                self.fitness[:] = fit_vals

                # Update global best
                if self.maximize:
                    best_i = int(np.argmax(self.fitness))
                else:
                    best_i = int(np.argmin(self.fitness))
                best_val = float(self.fitness[best_i])
                improved = (best_val > self.best_fitness) if self.maximize else (best_val < self.best_fitness)
                if improved:
                    self.best_fitness = best_val
                    self.best_position = self.population[best_i].copy()
                    stagnation = 0
                else:
                    stagnation += 1

                history.append(float(self.best_fitness))

                if self.best_times_fn is not None:
                    times = self.best_times_fn(self.best_position)
                    print(f"[GA] Gen {g+1}/{gen_target} best={self.best_fitness:.6f} stagn={stagnation} per_missile={times}", flush=True)
                else:
                    print(f"[GA] Gen {g+1}/{gen_target} best={self.best_fitness:.6f} stagn={stagnation}", flush=True)

                if stagnation >= self.stagnation_patience:
                    print(f"[GA] Early stop at generation {g+1} (stagnation {stagnation} >= {self.stagnation_patience})", flush=True)
                    gen_executed = g + 1
                    early = True
                    break
            else:
                gen_executed = gen_target
                early = False

        elapsed = time.time() - start
        result = GAResult(
            best_position=self.best_position.copy(),
            best_fitness=float(self.best_fitness),
            history=history,
            eval_count=eval_count,
            elapsed=elapsed,
            generations=gen_executed,
            early_stopped=early
        )
        if self.save_on_exit:
            self.save_model(self.save_on_exit, include_swarm=self.include_swarm_on_save)
        return result

__all__ = ["GeneticAlgorithm", "GAResult"]
