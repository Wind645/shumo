# Differential Evolution optimizer with optional JSON warm start & persistence.
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

# Reuse worker initialization helpers from PSO for consistent parallel evaluation
from .pso import _worker_init, _worker_eval_vector  # type: ignore


@dataclass
class DEResult:
    best_position: np.ndarray
    best_fitness: float
    history: List[float]
    eval_count: int
    elapsed: float


class DifferentialEvolution:
    """
    Parallel Differential Evolution (DE) - clean implementation.

    Strategy: classic DE/rand/1/bin
      v = x_r1 + F * (x_r2 - x_r3)
      u_j = v_j (if rand<CR or j==j_rand) else x_target_j

    Features mirrored from ParallelPSO for API symmetry:
      - Warm start from JSON model (swarm_positions or best_position).
      - Optional bounds.
      - Parallel objective evaluation via multiprocessing.
      - Optional auto-save on exit (same JSON layout).
      - Optional best_times callback for logging per-generation extra info.

    Parameters:
      dim                 : vector dimension
      objective           : function(vector) -> fitness (float)
      population_size     : number of candidates
      generations         : number of DE generations
      F                   : differential weight (0 < F <= 2)
      CR                  : crossover probability (0..1)
      maximize            : True = maximize (default), False = minimize
      processes           : worker processes (None -> cpu_count-1)
      seed                : base random seed (int or None)
      best_times_fn       : optional fn(vector)->List[float] for logging
      init_model          : optional JSON base name (under ../models)
      perturb_std         : stddev for Gaussian noise if only best_position
      models_dir          : override models directory
      save_on_exit        : auto-save model name (JSON)
      include_swarm_on_save: include full population in JSON
      bounds              : (lo, hi) arrays length dim, elementwise

    JSON schema matches PSO save for interchangeability:
      {
        "schema": 1,
        "dim": ...,
        "saved_at": "...Z",
        "best_fitness": ...,
        "best_position": [...],
        "swarm_positions": [[...], ...]   # if include_swarm_on_save True
      }
    """
    def __init__(self,
                 dim: int,
                 objective: Callable[[Sequence[float]], float],
                 population_size: int = 80,
                 generations: int = 100,
                 F: float = 0.8,
                 CR: float = 0.9,
                 maximize: bool = True,
                 processes: Optional[int] = None,
                 seed: Optional[int] = None,
                 best_times_fn: Optional[Callable[[Sequence[float]], List[float]]] = None,
                 init_model: Optional[str] = None,
                 perturb_std: float = 0.15,
                 models_dir: Optional[str] = None,
                 save_on_exit: Optional[str] = None,
                 include_swarm_on_save: bool = True,
                 bounds: Optional[Tuple[Union[Sequence[float], np.ndarray],
                                        Union[Sequence[float], np.ndarray]]] = None):
        self.dim = dim
        self.objective = objective
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.maximize = maximize
        self.processes = processes or max(1, mp.cpu_count() - 1)
        self.seed = seed or int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.best_times_fn = best_times_fn
        self.init_model = init_model
        self.perturb_std = float(perturb_std)
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.save_on_exit = save_on_exit
        self.include_swarm_on_save = include_swarm_on_save

        # Bounds
        if bounds is not None:
            lo = np.asarray(bounds[0], dtype=float)
            hi = np.asarray(bounds[1], dtype=float)
            self.bounds: Optional[Tuple[np.ndarray, np.ndarray]] = (lo, hi)
        else:
            self.bounds = None

        # Initialize population
        if self.bounds is None:
            self.population = np.random.uniform(-1.0, 1.0, (population_size, dim))
        else:
            lo, hi = self.bounds
            self.population = lo + (hi - lo) * np.random.rand(population_size, dim)

        if self.maximize:
            self.fitness = np.full(population_size, -np.inf)
            self.best_fitness = -np.inf
        else:
            self.fitness = np.full(population_size, np.inf)
            self.best_fitness = np.inf
        self.best_position = self.population[0].copy()

        if self.init_model:
            self._warm_start(self.init_model)

    # ---------------- Warm Start ----------------
    def _resolve_model_path(self, name: str) -> str:
        if not name.endswith(".json"):
            name += ".json"
        return os.path.join(self.models_dir, name)

    def _warm_start(self, name: str):
        path = self._resolve_model_path(name)
        if not os.path.isfile(path):
            # Create stub model file
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
            print(f"[DE] Warm start file missing. Created stub {path}", flush=True)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "swarm_positions" in data:
            arr = np.asarray(data["swarm_positions"], dtype=float)
            if arr.shape[1] == self.dim:
                if arr.shape[0] >= self.population_size:
                    self.population = arr[:self.population_size].copy()
                else:
                    reps = (self.population_size + arr.shape[0] - 1) // arr.shape[0]
                    self.population = np.tile(arr, (reps, 1))[:self.population_size].copy()
                print(f"[DE] Warm start: loaded {arr.shape[0]} population vectors.", flush=True)
        elif "best_position" in data:
            bp = np.asarray(data["best_position"], dtype=float)
            noise = np.random.normal(0.0, self.perturb_std, (self.population_size, self.dim))
            self.population = bp[None, :] + noise
            print(f"[DE] Warm start: seeded around best_position std={self.perturb_std}.", flush=True)
        else:
            print("[DE] Warm start: no usable keys; ignored.", flush=True)
        if self.bounds is not None:
            lo, hi = self.bounds
            np.clip(self.population, lo, hi, out=self.population)

    # ---------------- Persistence ----------------
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
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "best_fitness": float(self.best_fitness),
            "best_position": self.best_position.tolist(),
        }
        if include_swarm:
            payload["swarm_positions"] = self.population.tolist()
        if notes:
            payload["notes"] = notes
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[DE] Model saved to {path}", flush=True)

    # ---------------- Internal Helpers ----------------
    def _evaluate(self, batch: np.ndarray) -> List[float]:
        with mp.Pool(processes=self.processes,
                     initializer=_worker_init,
                     initargs=(self.seed, self.objective, self.maximize)) as pool:
            out = pool.map(_worker_eval_vector, batch)
        return out

    # ---------------- Run ----------------
    def run(self, generations: Optional[int] = None) -> DEResult:
        start = time.time()
        eval_count = 0
        gen_count = self.generations if generations is None else int(generations)

        # Build one shared pool (mirrors PSO pattern)
        with mp.Pool(processes=self.processes,
                     initializer=_worker_init,
                     initargs=(self.seed, self.objective, self.maximize)) as pool:

            # Initial evaluation
            init_fit = pool.map(_worker_eval_vector, self.population)
            eval_count += len(init_fit)
            for i, fit in enumerate(init_fit):
                self.fitness[i] = fit
            if self.maximize:
                idx = int(np.argmax(self.fitness))
                self.best_fitness = self.fitness[idx]
            else:
                idx = int(np.argmin(self.fitness))
                self.best_fitness = self.fitness[idx]
            self.best_position = self.population[idx].copy()

            history: List[float] = [float(self.best_fitness)]

            # Generation loop
            for g in range(gen_count):
                # Create all trial vectors first
                trials = np.empty_like(self.population)
                for i in range(self.population_size):
                    # Mutation indices
                    idxs = list(range(self.population_size))
                    idxs.remove(i)
                    r1, r2, r3 = random.sample(idxs, 3)
                    x1 = self.population[r1]
                    x2 = self.population[r2]
                    x3 = self.population[r3]
                    mutant = x1 + self.F * (x2 - x3)

                    # Binomial crossover
                    target = self.population[i]
                    trial = target.copy()
                    j_rand = random.randrange(self.dim)
                    for j in range(self.dim):
                        if random.random() < self.CR or j == j_rand:
                            trial[j] = mutant[j]

                    # Bounds
                    if self.bounds is not None:
                        lo, hi = self.bounds
                        np.clip(trial, lo, hi, out=trial)
                    trials[i] = trial

                # Batch evaluate trials
                trial_fitness = pool.map(_worker_eval_vector, trials)
                eval_count += len(trial_fitness)

                # Selection
                for i in range(self.population_size):
                    tf = trial_fitness[i]
                    better = tf > self.fitness[i] if self.maximize else tf < self.fitness[i]
                    if better:
                        self.population[i] = trials[i]
                        self.fitness[i] = tf
                        if (tf > self.best_fitness and self.maximize) or (tf < self.best_fitness and (not self.maximize)):
                            self.best_fitness = tf
                            self.best_position = trials[i].copy()

                history.append(float(self.best_fitness))
                if self.best_times_fn is not None:
                    times = self.best_times_fn(self.best_position)
                    print(f"[DE] Gen {g+1}/{gen_count} best_fitness={self.best_fitness:.6f} per_missile={times}", flush=True)
                else:
                    print(f"[DE] Gen {g+1}/{gen_count} best_fitness={self.best_fitness:.6f}", flush=True)

        elapsed = time.time() - start
        result = DEResult(
            best_position=self.best_position.copy(),
            best_fitness=float(self.best_fitness),
            history=history,
            eval_count=eval_count,
            elapsed=elapsed
        )
        if self.save_on_exit:
            self.save_model(self.save_on_exit, include_swarm=self.include_swarm_on_save)
        return result


__all__ = [
    "DifferentialEvolution",
    "DEResult",
]
