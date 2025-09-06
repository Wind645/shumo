from __future__ import annotations
"""
Genetic Algorithm (GA) for Problem 2 (Q2).

Decision vector x = [speed, azimuth, release_time, explode_delay]
Objective: minimize objective_q2_vector(x) which equals -occluded_time.

Pure NumPy implementation using vectorized occlusion judge for efficient
batch evaluation. The GA uses tournament selection, simulated binary crossover (SBX), 
and polynomial mutation for real-valued optimization.

Strategy
--------
1. Initialize random population within bounds
2. For each generation:
   - Select parents using tournament selection
   - Apply crossover (SBX) and mutation (polynomial)
   - Evaluate offspring using batch occlusion calculation
   - Select survivors (elitism + best individuals)
3. Return best solution found across all generations

The objective function value = -occluded_time, so lower is better.

Bounds (same as PSO/SA):
 speed: [70, 140]
 azimuth: [-pi, pi]
 release_time: [0, 30]
 explode_delay: [0.2, 12]

Returned best_eval is produced via canonical simulator for consistency.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict
import math
import random
import numpy as np

from api.objectives import objective_q2_vector
from api.problems import evaluate_problem2
from vectorized_judge import vectorized_circle_fully_occluded_by_sphere
from simcore.constants import (
    R_CYL_DEFAULT as _R_CYL,
    H_CYL_DEFAULT as _H_CYL,
    SMOKE_LIFETIME_DEFAULT as _SMOKE_LIFE,
    SMOKE_DESCENT_DEFAULT as _SMOKE_DESCENT,
    SMOKE_RADIUS_DEFAULT as _SMOKE_RADIUS,
    G as _G
)

# Convert torch constants to numpy
import torch
_C_BASE_TORCH = torch.tensor([0.0, 200.0, 0.0], dtype=torch.float)
_C_BASE = np.array([0.0, 200.0, 0.0], dtype=np.float64)

_MISSILE_POS0 = np.array([20000.0, 0.0, 2000.0], dtype=np.float64)
_MISSILE_SPEED = 300.0
_MISSILE_TARGET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

def _missile_dir_and_flight_time():
    d = _MISSILE_TARGET - _MISSILE_POS0
    n = np.linalg.norm(d)
    return d / n, n / _MISSILE_SPEED

_MISSILE_DIR, _MISSILE_T_FLIGHT = _missile_dir_and_flight_time()


def _compute_cloud_center0_numpy(drone_pos0: np.ndarray, drone_dir: np.ndarray, speed: np.ndarray,
                                 release_time: np.ndarray, explode_delay: np.ndarray) -> np.ndarray:
    """Numpy version of cloud center computation."""
    rel_pos = drone_pos0 + drone_dir * speed[:, np.newaxis] * release_time[:, np.newaxis]
    vel = drone_dir * speed[:, np.newaxis]
    dt = explode_delay[:, np.newaxis]
    a = np.array([0.0, 0.0, -_G], dtype=rel_pos.dtype)
    center0 = rel_pos + vel * dt + 0.5 * a * (dt * dt)
    return center0


def _batch_occluded_time_numpy(params: np.ndarray, *, dt: float) -> np.ndarray:
    """Compute occluded time (seconds) for each candidate parameter vector using pure numpy.
    
    params: (N,4) -> speed, azimuth, release_time, explode_delay
    Returns array (N,) occluded_time for missile M1 using judge_caps semantics.
    """
    N = params.shape[0]
    p = params.astype(np.float64)
    speed = p[:, 0]
    az = p[:, 1]
    t_rel = p[:, 2]
    dly = p[:, 3]

    # Basic validity masks
    valid = (speed >= 70.0) & (speed <= 140.0) & (t_rel >= 0.0) & (dly > 0.0)
    occluded_time = np.zeros(N, dtype=np.float64)
    if not valid.any():
        return occluded_time

    # Drone spec
    drone_pos0 = np.tile(np.array([17800.0, 0.0, 1800.0], dtype=np.float64), (N, 1))
    drone_dir = np.column_stack([np.cos(az), np.sin(az), np.zeros_like(az)])

    # Precompute cloud initial centers
    center0 = _compute_cloud_center0_numpy(drone_pos0, drone_dir, speed, t_rel, dly)  # (N,3)
    explode_time = t_rel + dly
    cloud_end_time = explode_time + _SMOKE_LIFE

    # Missile path times
    T_f = float(_MISSILE_T_FLIGHT)
    n_steps = int(T_f / dt) + 1
    t_grid = np.linspace(0.0, T_f, n_steps, dtype=np.float64)
    missile_pos = _MISSILE_POS0[np.newaxis, :] + _MISSILE_DIR[np.newaxis, :] * (_MISSILE_SPEED * t_grid[:, np.newaxis])
    
    # Cylinder caps constants (convert to numpy)
    Cb = np.array(_C_BASE, dtype=np.float64)
    Ct = Cb + np.array([0.0, 0.0, float(_H_CYL)], dtype=np.float64)
    r_cyl = float(_R_CYL)

    # Iterate timeline; vectorize across active candidates
    for ti, t in enumerate(t_grid):
        # active candidates with cloud present and valid
        act_mask = valid & (t >= explode_time) & (t <= cloud_end_time)
        if not act_mask.any():
            continue
        idx = np.nonzero(act_mask)[0]
        V = np.tile(missile_pos[ti], (len(idx), 1))  # (k,3)
        
        # Cloud center at time t
        tau = np.maximum(0.0, t - explode_time[idx])
        center_t = center0[idx].copy()
        center_t[:, 2] = center0[idx][:, 2] - _SMOKE_DESCENT * tau
        R_cloud = np.full(len(idx), float(_SMOKE_RADIUS), dtype=np.float64)

        # Bottom cap transform (already z=0 plane)
        Vb = V.copy()
        Cb_flat = np.tile(Cb, (len(idx), 1))
        Sb = center_t.copy()
        rb = np.full(len(idx), r_cyl, dtype=np.float64)

        res_b = vectorized_circle_fully_occluded_by_sphere(Vb, Cb_flat, rb, Sb, R_cloud)
        oc_b = res_b.occluded

        # Top cap transform: shift by Ct.z
        shift_z = Ct[2]
        Vt = V.copy()
        Vt[:, 2] = V[:, 2] - shift_z
        Ct_flat = np.array([Ct[0], Ct[1], 0.0], dtype=np.float64)
        Ct_batch = np.tile(Ct_flat, (len(idx), 1))
        St = center_t.copy()
        St[:, 2] = St[:, 2] - shift_z
        rt = rb  # same radius
        res_t = vectorized_circle_fully_occluded_by_sphere(Vt, Ct_batch, rt, St, R_cloud)
        oc_t = res_t.occluded

        oc_both = oc_b & oc_t
        if oc_both.any():
            occluded_time[idx[oc_both]] += dt
    return occluded_time


BOUNDS_DEFAULT: Tuple[Tuple[float, float], ...] = (
    (70.0, 140.0),
    (-math.pi, math.pi),
    (0.0, 30.0),
    (0.2, 12.0),
)


@dataclass
class GAResult:
    best_x: np.ndarray
    best_value: float
    best_eval: Dict


def tournament_selection(population: np.ndarray, fitness: np.ndarray, tournament_size: int = 3) -> np.ndarray:
    """Tournament selection for parent selection."""
    n_pop = population.shape[0]
    selected = np.zeros_like(population)
    
    for i in range(n_pop):
        tournament_idx = np.random.choice(n_pop, tournament_size, replace=False)
        tournament_fitness = fitness[tournament_idx]
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        selected[i] = population[winner_idx]
    
    return selected


def simulated_binary_crossover(parent1: np.ndarray, parent2: np.ndarray, bounds: np.ndarray, 
                               eta_c: float = 20.0, pc: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX) for real-valued optimization."""
    n_vars = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    for i in range(n_vars):
        if np.random.random() <= pc:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                y1 = min(parent1[i], parent2[i])
                y2 = max(parent1[i], parent2[i])
                lb = bounds[i, 0]
                ub = bounds[i, 1]
                
                # Calculate beta values
                beta1 = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                beta2 = 1.0 + (2.0 * (ub - y2) / (y2 - y1))
                
                # Calculate alpha values
                alpha1 = 2.0 - beta1 ** -(eta_c + 1.0)
                alpha2 = 2.0 - beta2 ** -(eta_c + 1.0)
                
                u1 = np.random.random()
                u2 = np.random.random()
                
                # Calculate beta_q values
                if u1 <= (1.0 / alpha1):
                    betaq1 = (u1 * alpha1) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq1 = (1.0 / (2.0 - u1 * alpha1)) ** (1.0 / (eta_c + 1.0))
                    
                if u2 <= (1.0 / alpha2):
                    betaq2 = (u2 * alpha2) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq2 = (1.0 / (2.0 - u2 * alpha2)) ** (1.0 / (eta_c + 1.0))
                
                # Create offspring
                child1[i] = 0.5 * ((y1 + y2) - betaq1 * (y2 - y1))
                child2[i] = 0.5 * ((y1 + y2) + betaq2 * (y2 - y1))
                
                # Ensure bounds
                child1[i] = np.clip(child1[i], lb, ub)
                child2[i] = np.clip(child2[i], lb, ub)
    
    return child1, child2


def polynomial_mutation(individual: np.ndarray, bounds: np.ndarray, eta_m: float = 20.0, pm: float = 0.1) -> np.ndarray:
    """Polynomial mutation for real-valued optimization."""
    n_vars = len(individual)
    mutant = individual.copy()
    
    for i in range(n_vars):
        if np.random.random() <= pm:
            y = individual[i]
            lb = bounds[i, 0]
            ub = bounds[i, 1]
            delta1 = (y - lb) / (ub - lb)
            delta2 = (ub - y) / (ub - lb)
            
            mut_pow = 1.0 / (eta_m + 1.0)
            rnd = np.random.random()
            
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow
            
            y_new = y + deltaq * (ub - lb)
            mutant[i] = np.clip(y_new, lb, ub)
    
    return mutant


def solve_q2_ga(
    *,
    pop_size: int = 50,
    n_generations: int = 100,
    pc: float = 0.8,
    pm: float = 0.1,
    tournament_size: int = 3,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
    elitism: int = 2,
    method: str = "judge_caps",
    dt: float = 0.02,
    seed: Optional[int] = None,
    bounds: Tuple[Tuple[float, float], ...] = BOUNDS_DEFAULT,
    use_vectorized: bool = True,
    verbose: bool = False,
) -> GAResult:
    """Genetic Algorithm for Q2 with vectorized batch evaluation.

    Args:
        pop_size: Population size
        n_generations: Number of generations
        pc: Crossover probability
        pm: Mutation probability
        tournament_size: Tournament size for selection
        eta_c: Distribution index for crossover
        eta_m: Distribution index for mutation
        elitism: Number of elite individuals to preserve
        method: Evaluation method ("judge_caps" or "sampling")
        dt: Time step for simulation
        seed: Random seed
        bounds: Variable bounds
        use_vectorized: Use vectorized evaluation
        verbose: Print progress
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    dim = 4
    assert len(bounds) == dim
    bounds_array = np.array(bounds, dtype=np.float64)
    lo = bounds_array[:, 0]
    hi = bounds_array[:, 1]
    
    def eval_single(x: Iterable[float]) -> float:
        return float(objective_q2_vector(x, method=method, dt=dt))

    def eval_batch(pop: np.ndarray) -> np.ndarray:
        if use_vectorized and method == "judge_caps":
            occluded_time = _batch_occluded_time_numpy(pop, dt=dt)
            return -occluded_time.astype(np.float64)
        vals = [eval_single(pop[i]) for i in range(pop.shape[0])]
        return np.array(vals, dtype=np.float64)

    # Initialize population
    population = lo + np.random.random((pop_size, dim)) * (hi - lo)
    fitness = eval_batch(population)
    
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    if verbose:
        print(f"Generation 0: Best fitness = {best_fitness:.6f}")

    for gen in range(n_generations):
        # Selection
        parents = tournament_selection(population, fitness, tournament_size)
        
        # Crossover and Mutation
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:
                parent1 = parents[i]
                parent2 = parents[i + 1]
            else:
                parent1 = parents[i]
                parent2 = parents[np.random.randint(pop_size)]
            
            child1, child2 = simulated_binary_crossover(parent1, parent2, bounds_array, eta_c, pc)
            child1 = polynomial_mutation(child1, bounds_array, eta_m, pm)
            child2 = polynomial_mutation(child2, bounds_array, eta_m, pm)
            
            offspring.extend([child1, child2])
        
        offspring = np.array(offspring[:pop_size])
        offspring_fitness = eval_batch(offspring)
        
        # Elitism: combine population and offspring, select best
        combined_pop = np.vstack([population, offspring])
        combined_fitness = np.hstack([fitness, offspring_fitness])
        
        # Select survivors
        sorted_indices = np.argsort(combined_fitness)
        population = combined_pop[sorted_indices[:pop_size]]
        fitness = combined_fitness[sorted_indices[:pop_size]]
        
        # Update best
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_x = population[0].copy()
        
        if verbose and (gen % max(1, n_generations // 10) == 0 or gen == n_generations - 1):
            print(f"Generation {gen + 1}: Best fitness = {best_fitness:.6f}")

    # Final evaluation with canonical simulator
    try:
        best_eval = evaluate_problem2(
            speed=float(best_x[0]),
            azimuth=float(best_x[1]),
            release_time=float(best_x[2]),
            explode_delay=float(best_x[3]),
            occlusion_method=method,
            dt=dt,
        )
    except RuntimeError as e:
        if "device" in str(e):
            # Fallback for device issues - use sampling method
            best_eval = evaluate_problem2(
                speed=float(best_x[0]),
                azimuth=float(best_x[1]),
                release_time=float(best_x[2]),
                explode_delay=float(best_x[3]),
                occlusion_method="sampling",
                dt=dt,
            )
        else:
            raise
    
    return GAResult(best_x=best_x, best_value=best_fitness, best_eval=best_eval)


if __name__ == "__main__":
    # Demo run with pure numpy implementation
    res = solve_q2_ga(pop_size=30, n_generations=50, seed=42, verbose=True, method="judge_caps")
    print("Best x:", res.best_x)
    print("Best value (minimized):", res.best_value)
    print("M1 遮蔽时长(s):", res.best_eval["occluded_time"]["M1"], "总计(s):", res.best_eval["total"])