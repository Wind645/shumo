from __future__ import annotations
# NOTE: Refactored: removed O(N^2) radius-based crowding. An optional lightweight
# grid crowding can be enabled via enable_crowding flag (False by default).

"""
PushPSO (Crowding / Repulsion + Stagnation Repeller PSO)
=======================================================

在经典 PSO 基础上实现 3 类“多样性保持 / 防早熟”机制:
1. 拥挤排斥 (Crowding Repulsion)
   - 半径模式 (radius)
   - 网格模式 (grid)

2. 随机重置 (Random Reset)

3. 停滞小群落驱散 (Stagnation Cluster Repeller)
   - 当一组粒子（规模较小）在相近区域连续若干迭代 (stagnation_iter_threshold) 未获得个体最优改进，
     则判定为“停滞小群落”，在其中心生成一个“排斥点 (repeller)”，该点持续在其半径范围内对粒子
     施加斥力，强制群体离开局部最优陷阱。
   - 斥力可逐步衰减 (repeller_decay)，强度低于 repeller_min_strength 时删除。
   - 避免重复放置：若已有排斥点中心距离新群落中心较近 (< repeller_radius/2) 则不再新增。

使用示例
--------
    def sphere(x):
        return -sum(v*v for v in x)  # 最大化 -||x||^2 => 期望收敛到 0

    pso = PushPSO(
        dim=5,
        objective=sphere,
        swarm_size=60,
        iterations=300,
        crowd_radius=0.35,         # 使用半径模式（不传则自动网格模式）
        density_threshold=6,
        push_strength=0.8,
        adaptive_push=True,
        bounds=([-2]*5, [2]*5),
        maximize=True,
        # 停滞小群落驱散参数
        stagnation_iter_threshold=50,
        cluster_eps=0.15,
        cluster_min_size=3,
        cluster_max_size=10,
        repeller_radius=0.6,
        repeller_strength=1.2,
        repeller_decay=0.985
    )
    result = pso.run()
    print("Best fitness:", result.best_fitness)
    print("Best position:", result.best_position)
    print("Repellers created:", len(result.repeller_events))

主要参数
--------
(标准 PSO)
- dim, objective, swarm_size, iterations, inertia(w), cognitive(c1), social(c2)
- reset_prob: 迭代中粒子随机重置概率
- velocity_clamp: (vmin, vmax)
- bounds: (lo, hi) 可选；缺省时 crowd grid 模式使用 [-1,1]

(拥挤排斥)
- crowd_radius: 半径模式；不提供则使用网格模式
- density_threshold: 拥挤阈值
- push_strength: 推开基础强度
- cells_per_dim: 网格模式划分份数
- max_push_per_iter: 每迭代最多应用推开的粒子数量
- adaptive_push: True 时按 (拥挤度/阈值) 线性放大推开
- normalize_push: 是否单位化方向

(停滞小群落驱散)
- stagnation_iter_threshold: 判定停滞需要的连续未改进迭代数
- cluster_eps: 判定小群落的空间半径 (欧式距离)
- cluster_min_size / cluster_max_size: 小群落规模上下界
- repeller_radius: 排斥点影响半径
- repeller_strength: 排斥点初始强度 (对速度的附加)
- repeller_decay: 每迭代乘以衰减系数
- repeller_min_strength: 强度低于该值移除
- max_active_repellers: 活动排斥点数量上限（超出时移除最早/最弱）
- repeller_use_inverse_distance: 若 True，强度乘以 (1 / (d + eps))，增强近距离效果
- cluster_recheck_interval: 每多少迭代才触发一次聚类检测（>1 可降低开销）

返回结果 PushPSOResult
----------------------
- best_position, best_fitness
- history: 每次迭代全局最优
- eval_count: 评估次数
- elapsed: 用时 (seconds)
- crowd_stats: 拥挤统计
- repeller_events: 列表，记录排斥点创建的迭代、位置、群落规模等
- final_repellers: 结束时剩余排斥点的描述

复杂度
------
- 半径拥挤检测: O(N^2 * dim)
- 网格拥挤检测: O(N)
- 停滞聚类（贪心半径合并）: O(M^2) (M = 停滞候选数量)；可通过 cluster_recheck_interval 减低频率

扩展方向
--------
- 使用 KD-Tree 代替 O(N^2) 半径搜索
- 更先进聚类 (DBSCAN) / 层次聚类
- 自适应调节 repeller_strength / radius
- 与分块 (BlockPSO) 结合，在块内应用停滞驱散
"""

import os
import json
import time
import random
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# 复用 ParallelPSO 的 worker 初始化逻辑
from .pso import _worker_init, _worker_eval_vector  # type: ignore


# --------------------------------------------------------------------------- #
# Result DataClass
# --------------------------------------------------------------------------- #
@dataclass
class PushPSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: List[float]
    eval_count: int
    elapsed: float
    crowd_stats: List[Dict[str, Any]]
    repeller_events: List[Dict[str, Any]]
    final_repellers: List[Dict[str, Any]]


# --------------------------------------------------------------------------- #
# PushPSO
# --------------------------------------------------------------------------- #
class PushPSO:
    def __init__(self,
                 dim: int,
                 objective: Callable[[Sequence[float]], float],
                 swarm_size: int = 40,
                 iterations: int = 100,
                 inertia: float = 0.72,
                 cognitive: float = 1.49,
                 social: float = 1.49,
                 reset_prob: float = 0.02,
                 velocity_clamp: Optional[Tuple[float, float]] = None,
                 maximize: bool = True,
                 processes: Optional[int] = None,
                 seed: Optional[int] = None,
                 bounds: Optional[Tuple[Union[Sequence[float], np.ndarray],
                                        Union[Sequence[float], np.ndarray]]] = None,
                 # Crowding
                 crowd_radius: Optional[float] = None,  # deprecated: radius crowding removed
                 density_threshold: int = 8,
                 push_strength: float = 0.7,
                 cells_per_dim: int = 8,
                 max_push_per_iter: Optional[int] = None,
                 normalize_push: bool = True,
                 adaptive_push: bool = True,
                 # Stagnation Cluster Repeller
                 stagnation_iter_threshold: int = 50,
                 cluster_eps: float = 0.15,
                 cluster_min_size: int = 3,
                 cluster_max_size: Optional[int] = 12,
                 cluster_recheck_interval: int = 1,
                 repeller_radius: float = 0.5,
                 repeller_strength: float = 1.0,
                 repeller_decay: float = 0.99,
                 repeller_min_strength: float = 1e-4,
                 max_active_repellers: Optional[int] = 50,
                 repeller_use_inverse_distance: bool = False,
                 # Persistence (optional)
                 init_model: Optional[str] = None,
                 enable_crowding: bool = False,
                 models_dir: Optional[str] = None,
                 save_on_exit: Optional[str] = None,
                 include_swarm_on_save: bool = True,
                 perturb_std: float = 0.1):
        # ------------------------------------------------------------------
        # Adaptive clustering related (auto‑initialized lazily later):
        #   self.stagnation_improve_epsilon : float  -> minimal improvement threshold (default 1e-6)
        #   self._current_cluster_eps       : float  -> current working eps for clustering (starts at cluster_eps)
        #   self.cluster_eps_growth         : float  -> multiplicative growth factor when no new repeller forms
        #   self.cluster_eps_max            : float  -> hard cap for adaptive eps enlargement
        #   self.cluster_eps_adaptive       : bool   -> master switch for adaptive eps enlargement
        #
        # These are NOT constructor arguments (kept out to avoid breaking older
        # saved configs). They are created on first call to
        # _detect_and_create_repellers(). If you need to tune them externally
        # you can set the attributes manually after constructing PushPSO:
        #
        #   pso.cluster_eps_growth = 1.3
        #   pso.cluster_eps_max = 0.5
        #   pso.cluster_eps_adaptive = True
        #
        # This preserves backward compatibility while allowing advanced control.
        # ------------------------------------------------------------------
        # --- Basic validation ---
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if swarm_size <= 1:
            raise ValueError("swarm_size must be > 1")
        if iterations <= 0:
            raise ValueError("iterations must be > 0")
        if density_threshold < 1:
            raise ValueError("density_threshold must be >= 1")
        if crowd_radius is not None and crowd_radius <= 0:
            raise ValueError("crowd_radius must be > 0")
        if cells_per_dim < 1:
            raise ValueError("cells_per_dim must be >= 1")
        if stagnation_iter_threshold < 1:
            raise ValueError("stagnation_iter_threshold must be >=1")
        if cluster_eps <= 0:
            raise ValueError("cluster_eps must be > 0")
        if cluster_min_size < 1:
            raise ValueError("cluster_min_size must be >=1")
        if cluster_max_size is not None and cluster_max_size < cluster_min_size:
            raise ValueError("cluster_max_size must be >= cluster_min_size")
        if repeller_radius <= 0:
            raise ValueError("repeller_radius must be > 0")
        if repeller_strength <= 0:
            raise ValueError("repeller_strength must be > 0")
        if not (0 < repeller_decay <= 1.0):
            raise ValueError("repeller_decay must be in (0,1]")
        if cluster_recheck_interval < 1:
            raise ValueError("cluster_recheck_interval must be >=1")

        self.dim = dim
        self.objective = objective
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = float(inertia)
        self.c1 = float(cognitive)
        self.c2 = float(social)
        self.reset_prob = float(reset_prob)
        self.velocity_clamp = velocity_clamp
        self.maximize = maximize
        self.processes = processes or max(1, mp.cpu_count() - 1)
        self.seed = seed or int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Bounds
        if bounds is not None:
            lo = np.asarray(bounds[0], dtype=float)
            hi = np.asarray(bounds[1], dtype=float)
            if lo.shape != (dim,) or hi.shape != (dim,):
                raise ValueError("bounds arrays must have shape (dim,)")
            if np.any(hi <= lo):
                raise ValueError("All hi bounds must be > lo bounds")
            self.bounds: Optional[Tuple[np.ndarray, np.ndarray]] = (lo, hi)
        else:
            self.bounds = None

        # Swarm initialization
        if self.bounds is None:
            self.positions = np.random.uniform(-1.0, 1.0, (swarm_size, dim))
        else:
            lo, hi = self.bounds
            self.positions = lo + (hi - lo) * np.random.rand(swarm_size, dim)
        self.velocities = np.zeros((swarm_size, dim), dtype=float)
        self.personal_best_positions = self.positions.copy()
        if self.maximize:
            self.personal_best_fitness = np.full(swarm_size, -np.inf)
            self.global_best_fitness = -np.inf
        else:
            self.personal_best_fitness = np.full(swarm_size, np.inf)
            self.global_best_fitness = np.inf
        self.global_best_position = self.positions[0].copy()

        # Crowding params
        self.crowd_radius = crowd_radius
        self.density_threshold = int(density_threshold)
        self.push_strength = float(push_strength)
        self.cells_per_dim = int(cells_per_dim)
        self.max_push_per_iter = max_push_per_iter
        self.normalize_push = normalize_push
        self.adaptive_push = adaptive_push
        # Deprecated: radius / grid auto mode replaced by explicit flag
        self.enable_crowding = enable_crowding
        if crowd_radius is not None:
            # Keep value only if someone still passes it; we ignore radius algorithm now.
            self.crowd_radius = crowd_radius  # for backward compatibility in saved model

        # Stagnation / Repeller params
        self.stagnation_iter_threshold = stagnation_iter_threshold
        self.cluster_eps = cluster_eps
        self.cluster_min_size = cluster_min_size
        self.cluster_max_size = cluster_max_size
        self.cluster_recheck_interval = cluster_recheck_interval
        self.repeller_radius = repeller_radius
        self.repeller_strength = repeller_strength
        self.repeller_decay = repeller_decay
        self.repeller_min_strength = repeller_min_strength
        self.max_active_repellers = max_active_repellers
        self.repeller_use_inverse_distance = repeller_use_inverse_distance
        # New lifecycle / control attributes (can be overridden after construction)
        self.repeller_max_age = 150              # max iterations a repeller can live
        self.repeller_radius_decay = 0.997       # multiplicative decay applied each iter
        self.max_new_repellers_per_iter = 3      # hard cap per iteration
        self.duplicate_distance_scale = 1.2      # duplicate if center distance < scale * radius
        # -------- Global stagnation escape parameters (simple, deterministic) --------
        # 连续多少轮全局最优未提升触发一次全局逃逸:
        self.global_stagnation_iters = max(25, iterations // 40)  # 可调整
        # 触发时重新随机初始化的粒子比例:
        self.global_stagnation_reinit_fraction = 0.25
        # 触发时在当前全局最优处投放一个更强的 repeller（若附近不存在）:
        self.global_stagnation_repeller_strength = self.repeller_strength * 1.5
        self.global_stagnation_repeller_radius = self.repeller_radius * 1.2
        # 计数器
        self.global_no_improve_iters = 0

        # Per-particle stagnation counters
        self.no_improve_iters = np.zeros(swarm_size, dtype=np.int32)

        # Active repellers: list of dicts
        # dict keys: center (np.ndarray), strength (float), radius (float),
        #            age (int), created_iter (int), origin_size (int)
        self.repellers: List[Dict[str, Any]] = []
        self.repeller_events: List[Dict[str, Any]] = []

        # Persistence
        self.init_model = init_model
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.save_on_exit = save_on_exit
        self.include_swarm_on_save = include_swarm_on_save
        self.perturb_std = float(perturb_std)

        if self.init_model:
            try:
                self._load_warm_start(self.init_model)
            except Exception as e:
                print(f"[PushPSO] Warm start failed ({e}); fallback to random init.", flush=True)

    # ---------------- Persistence (similar to ParallelPSO minimal) ------------
    def _resolve_model_path(self, name: str) -> str:
        if not name.endswith(".json"):
            name += ".json"
        return os.path.join(self.models_dir, name)

    def _load_warm_start(self, name: str):
        path = self._resolve_model_path(name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Warm start JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("dim") != self.dim:
            raise ValueError("dim mismatch in warm start file")
        if "swarm_positions" in data:
            arr = np.asarray(data["swarm_positions"], dtype=float)
            if arr.ndim != 2 or arr.shape[1] != self.dim:
                raise ValueError("swarm_positions shape mismatch")
            if arr.shape[0] >= self.swarm_size:
                self.positions = arr[:self.swarm_size].copy()
            else:
                reps = (self.swarm_size + arr.shape[0] - 1) // arr.shape[0]
                self.positions = np.tile(arr, (reps, 1))[:self.swarm_size].copy()
            print(f"[PushPSO] Warm start loaded {arr.shape[0]} positions.", flush=True)
        elif "best_position" in data:
            bp = np.asarray(data["best_position"], dtype=float)
            if bp.shape != (self.dim,):
                raise ValueError("best_position shape mismatch")
            noise = np.random.normal(0.0, self.perturb_std, (self.swarm_size, self.dim))
            self.positions = bp[None, :] + noise
            print(f"[PushPSO] Warm start seeded around best_position std={self.perturb_std}.", flush=True)
        else:
            raise ValueError("Warm start JSON must contain 'swarm_positions' or 'best_position'")
        # Reset dependent state
        self.velocities.fill(0.0)
        self.personal_best_positions = self.positions.copy()
        if self.maximize:
            self.personal_best_fitness.fill(-np.inf)
            self.global_best_fitness = -np.inf
        else:
            self.personal_best_fitness.fill(np.inf)
            self.global_best_fitness = np.inf
        self.global_best_position = self.positions[0].copy()

    def save_model(self,
                   name: str,
                   include_swarm: bool = True,
                   notes: Optional[str] = None,
                   extra: Optional[Dict[str, Any]] = None):
        os.makedirs(self.models_dir, exist_ok=True)
        path = self._resolve_model_path(name)
        payload: Dict[str, Any] = {
            "schema": 2,
            "dim": self.dim,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "best_fitness": float(self.global_best_fitness),
            "best_position": self.global_best_position.tolist(),
            "crowd_radius": self.crowd_radius,
            "density_threshold": self.density_threshold,
            "push_strength": self.push_strength,
            "repellers_active": len(self.repellers),
        }
        if include_swarm:
            payload["swarm_positions"] = self.positions.tolist()
        if extra:
            payload.update(extra)
        if notes:
            payload["notes"] = notes
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[PushPSO] Model saved to {path}", flush=True)

    # ---------------- Evaluation ----------------
    def _evaluate_batch(self, batch: np.ndarray) -> List[float]:
        with mp.Pool(processes=self.processes,
                     initializer=_worker_init,
                     initargs=(self.seed, self.objective, self.maximize)) as pool:
            return pool.map(_worker_eval_vector, batch)

    # ---------------- Crowding (radius) ----------
    # Radius crowding removed (O(N^2) cost); kept as stub for backward compatibility
    def _crowding_radius_push(self) -> Dict[str, Any]:
        return {"mode": "none"}
        N = self.swarm_size
        pos = self.positions
        r2 = (self.crowd_radius ** 2) if self.crowd_radius is not None else 0.0
        diff = pos[:, None, :] - pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        neighbor_mask = dist2 <= r2
        counts = neighbor_mask.sum(axis=1)
        overcrowded = counts > self.density_threshold
        num_over = int(overcrowded.sum())
        pushes = np.zeros_like(pos)

        if num_over > 0:
            idxs = np.where(overcrowded)[0]
            for i in idxs:
                neigh_idx = np.where(neighbor_mask[i])[0]
                centroid = pos[neigh_idx].mean(axis=0)
                direction = pos[i] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-12 and self.normalize_push:
                    direction /= norm
                scale = counts[i] / max(1.0, self.density_threshold) if self.adaptive_push else 1.0
                pushes[i] = self.push_strength * scale * direction
            if self.max_push_per_iter is not None and num_over > self.max_push_per_iter:
                order = np.argsort(-counts[idxs])
                keep = idxs[order[:self.max_push_per_iter]]
                keep_mask = np.zeros(N, dtype=bool)
                keep_mask[keep] = True
                pushes[~keep_mask] = 0.0
            self.velocities += pushes

        return {
            "mode": "radius",
            "mean_neighbors": float(np.mean(counts)),
            "max_neighbors": int(np.max(counts)),
            "overcrowded": int(num_over),
        }

    # ---------------- Crowding (grid) ------------
    def _crowding_grid_push(self) -> Dict[str, Any]:
        pos = self.positions
        if self.bounds is None:
            lo = np.full(self.dim, -1.0)
            hi = np.full(self.dim, 1.0)
        else:
            lo, hi = self.bounds
        span = hi - lo
        normed = (pos - lo) / span
        np.clip(normed, 0.0, 0.999999, out=normed)
        idx = (normed * self.cells_per_dim).astype(int)
        cell_keys = [tuple(row.tolist()) for row in idx]
        counts: Dict[Tuple[int, ...], int] = {}
        for k in cell_keys:
            counts[k] = counts.get(k, 0) + 1

        overcrowded_mask = np.zeros(self.swarm_size, dtype=bool)
        for i, k in enumerate(cell_keys):
            if counts[k] > self.density_threshold:
                overcrowded_mask[i] = True

        pushes = np.zeros_like(pos)
        num_over = int(overcrowded_mask.sum())
        if num_over > 0:
            center_cache: Dict[Tuple[int, ...], np.ndarray] = {}
            for i, k in enumerate(cell_keys):
                if not overcrowded_mask[i]:
                    continue
                if k not in center_cache:
                    center_norm = (np.array(k) + 0.5) / self.cells_per_dim
                    center_cache[k] = lo + center_norm * span
                direction = pos[i] - center_cache[k]
                norm = np.linalg.norm(direction)
                if norm > 1e-12 and self.normalize_push:
                    direction /= norm
                scale = counts[k] / max(1.0, self.density_threshold) if self.adaptive_push else 1.0
                pushes[i] = self.push_strength * scale * direction
            if self.max_push_per_iter is not None and num_over > self.max_push_per_iter:
                idxs = np.where(overcrowded_mask)[0]
                scores = np.array([counts[cell_keys[i]] for i in idxs])
                order = np.argsort(-scores)
                keep = idxs[order[:self.max_push_per_iter]]
                keep_mask = np.zeros(self.swarm_size, dtype=bool)
                keep_mask[keep] = True
                pushes[~keep_mask] = 0.0
            self.velocities += pushes

        return {
            "mode": "grid",
            "unique_cells": len(counts),
            "overcrowded": int(num_over),
            "max_cell_pop": int(max(counts.values()) if counts else 0),
            "mean_cell_pop": float(np.mean(list(counts.values())) if counts else 0.0),
        }

    # ---------------- Stagnation Region Repeller (Simplified) ----------------
    def _detect_and_create_repellers(self, iteration: int):
        """
        改进版局部停滞驱散：
        - 更严格的触发：需要较多停滞迭代与更大的局部群体
        - 限制单轮新增数量，防止 repeller 爆炸
        - 使用更宽松的重复判定距离，减少密集覆盖
        - 若已接近全局上限，提前退出
        """
        # ---- Lazy init of simple region parameters (allow external override) ----
        if not hasattr(self, "region_stagnation_iters"):
            self.region_stagnation_iters = 20
        if not hasattr(self, "region_min_group"):
            self.region_min_group = 5
        if not hasattr(self, "region_repeller_radius_scale"):
            self.region_repeller_radius_scale = 1.0
        if not hasattr(self, "max_new_repellers_per_iter"):
            self.max_new_repellers_per_iter = 3
        if not hasattr(self, "duplicate_distance_scale"):
            self.duplicate_distance_scale = 1.2

        # 提前全局容量控制
        if self.max_active_repellers is not None and len(self.repellers) >= self.max_active_repellers:
            return

        # 选出达到停滞阈值的粒子
        stagnating = np.where(self.no_improve_iters >= self.region_stagnation_iters)[0]
        if stagnating.size < self.region_min_group:
            return

        pos = self.positions
        rad = self.repeller_radius * self.region_repeller_radius_scale
        rad2 = rad * rad
        taken = np.zeros(self.swarm_size, dtype=bool)

        new_count = 0
        # 遍历停滞粒子（顺序随机化，避免空间偏置）
        stagnating_shuffled = stagnating.copy()
        np.random.shuffle(stagnating_shuffled)

        for idx in stagnating_shuffled:
            if taken[idx]:
                continue
            if new_count >= self.max_new_repellers_per_iter:
                break
            if self.max_active_repellers is not None and len(self.repellers) >= self.max_active_repellers:
                break

            center = pos[idx]

            # 检查现有 repeller 重复（用更大的 duplicate 距离）
            duplicate = False
            dup_thresh = self.repeller_radius * self.duplicate_distance_scale
            for rep in self.repellers:
                if np.linalg.norm(rep["center"] - center) <= dup_thresh:
                    duplicate = True
                    break
            if duplicate:
                continue

            # 寻找该中心附近其它停滞粒子
            diff = pos[stagnating] - center
            d2 = np.sum(diff * diff, axis=1)
            local_mask = d2 <= rad2
            local_indices = stagnating[local_mask]
            local_indices = [i for i in local_indices if not taken[i]]
            if len(local_indices) < self.region_min_group:
                continue

            cluster_center = pos[local_indices].mean(axis=0)

            # 质心重复检查
            duplicate = False
            for rep in self.repellers:
                if np.linalg.norm(rep["center"] - cluster_center) <= dup_thresh:
                    duplicate = True
                    break
            if duplicate:
                for i in local_indices:
                    taken[i] = True
                continue

            rep = {
                "center": cluster_center.copy(),
                "strength": float(self.repeller_strength),
                "radius": float(self.repeller_radius),
                "age": 0,
                "created_iter": iteration,
                "origin_size": len(local_indices),
            }
            self.repellers.append(rep)
            self.repeller_events.append({
                "iter": iteration,
                "center": cluster_center.tolist(),
                "size": len(local_indices),
                "strength": rep["strength"],
                "radius": rep["radius"],
            })
            new_count += 1
            for i in local_indices:
                taken[i] = True

        if new_count > 0:
            print(f"[PushPSO] Region repellers added={new_count} "
                  f"(local_stag_iters>={self.region_stagnation_iters})", flush=True)

        # 若超过容量上限，按强度和新鲜度裁剪
        if self.max_active_repellers is not None and len(self.repellers) > self.max_active_repellers:
            # 排序：优先保留强且新的
            self.repellers.sort(key=lambda r: (r["strength"], -r["age"]), reverse=True)
            removed = len(self.repellers) - self.max_active_repellers
            if removed > 0:
                self.repellers = self.repellers[:self.max_active_repellers]
                print(f"[PushPSO] Pruned {removed} repellers (capacity)", flush=True)

    # ---------------- Apply Repellers ----------------
    def _apply_repellers(self):
        if not self.repellers:
            return
        pos = self.positions

        total_push_cap = 5.0  # 防止叠加后速度爆炸（可调）
        accumulated_push = np.zeros_like(self.velocities)

        for rep in self.repellers:
            center = rep["center"]
            radius = rep["radius"]
            strength = rep["strength"]
            if strength <= 0.0 or radius <= 1e-12:
                continue
            diff = pos - center
            dist = np.linalg.norm(diff, axis=1)
            mask = dist < radius
            if not np.any(mask):
                continue

            directions = np.zeros_like(diff)
            sub = diff[mask]
            dsub = dist[mask]
            nz = dsub > 1e-12
            if np.any(nz):
                sub_norm = sub[nz] / dsub[nz, None]
                mapped_idx = np.where(mask)[0][nz]
                directions[mapped_idx] = sub_norm

            # 线性衰减 + 可选逆距离
            influence = (1.0 - (dist[mask] / radius))
            if self.repeller_use_inverse_distance:
                influence *= 1.0 / (dist[mask] + 1e-9)

            push_vec = (strength * influence)[:, None] * directions[mask]
            accumulated_push[mask] += push_vec

        # 归一/裁剪累积 push，避免破坏 PSO 原有探索方向
        norms = np.linalg.norm(accumulated_push, axis=1)
        over = norms > total_push_cap
        if np.any(over):
            accumulated_push[over] = (accumulated_push[over] /
                                      norms[over, None]) * total_push_cap

        self.velocities += accumulated_push

        # Decay, radius shrink & prune
        kept: List[Dict[str, Any]] = []
        for rep in self.repellers:
            rep["age"] += 1
            rep["strength"] *= self.repeller_decay
            # 半径也随时间轻微衰减
            rep["radius"] *= getattr(self, "repeller_radius_decay", 1.0)
            if rep["age"] > getattr(self, "repeller_max_age", 10**9):
                continue
            if rep["strength"] >= self.repeller_min_strength and rep["radius"] > 1e-6:
                kept.append(rep)

        removed = len(self.repellers) - len(kept)
        if removed > 0:
            print(f"[PushPSO] Pruned {removed} expired/weak repeller(s)", flush=True)
        self.repellers = kept

        # 容量再保证（以防外部调参导致未及时裁剪）
        if self.max_active_repellers is not None and len(self.repellers) > self.max_active_repellers:
            self.repellers.sort(key=lambda r: (r["strength"], -r["age"]), reverse=True)
            extra = len(self.repellers) - self.max_active_repellers
            self.repellers = self.repellers[:self.max_active_repellers]
            print(f"[PushPSO] Capacity prune removed {extra} repellers", flush=True)

    # ---------------- Core Run ----------------
    def run(self, iterations: Optional[int] = None) -> PushPSOResult:
        start = time.time()
        iter_count = self.iterations if iterations is None else int(iterations)
        history: List[float] = []
        eval_count = 0
        crowd_stats: List[Dict[str, Any]] = []

        for it in range(iter_count):
            # Random reset
            reset_mask = np.random.rand(self.swarm_size) < self.reset_prob
            if reset_mask.any():
                if self.bounds is None:
                    self.positions[reset_mask] = np.random.uniform(-1.0, 1.0,
                                                                   (reset_mask.sum(), self.dim))
                else:
                    lo, hi = self.bounds
                    self.positions[reset_mask] = lo + (hi - lo) * np.random.rand(reset_mask.sum(), self.dim)
                self.velocities[reset_mask] = 0.0
                # Reset stagnation counters for those
                self.no_improve_iters[reset_mask] = 0

            # Evaluate
            fitness = self._evaluate_batch(self.positions)
            eval_count += len(fitness)

            # Update personal/global best + stagnation counters
            for i, fit in enumerate(fitness):
                better = fit > self.personal_best_fitness[i] if self.maximize else fit < self.personal_best_fitness[i]
                if better:
                    self.personal_best_fitness[i] = fit
                    self.personal_best_positions[i] = self.positions[i].copy()
                    self.no_improve_iters[i] = 0
                else:
                    self.no_improve_iters[i] += 1

            improved_global = False
            if self.maximize:
                best_idx = int(np.argmax(self.personal_best_fitness))
                best_fit = self.personal_best_fitness[best_idx]
                if best_fit > self.global_best_fitness:
                    self.global_best_fitness = best_fit
                    self.global_best_position = self.personal_best_positions[best_idx].copy()
                    improved_global = True
            else:
                best_idx = int(np.argmin(self.personal_best_fitness))
                best_fit = self.personal_best_fitness[best_idx]
                if best_fit < self.global_best_fitness:
                    self.global_best_fitness = best_fit
                    self.global_best_position = self.personal_best_positions[best_idx].copy()
                    improved_global = True
            # 更新全局停滞计数
            if improved_global:
                self.global_no_improve_iters = 0
            else:
                self.global_no_improve_iters += 1
            # 触发全局逃逸: 放置强力 repeller + 重置部分粒子
            if self.global_no_improve_iters >= self.global_stagnation_iters:
                center = self.global_best_position.copy()
                # 避免与已有 repeller 重复
                duplicate = any(np.linalg.norm(rp["center"] - center) <= self.repeller_radius * 0.6 for rp in self.repellers)
                if not duplicate:
                    rep = {
                        "center": center.copy(),
                        "strength": float(self.global_stagnation_repeller_strength),
                        "radius": float(self.global_stagnation_repeller_radius),
                        "age": 0,
                        "created_iter": it,
                        "origin_size": -1,
                    }
                    self.repellers.append(rep)
                    self.repeller_events.append({
                        "iter": it,
                        "center": center.tolist(),
                        "size": -1,
                        "strength": rep["strength"],
                        "radius": rep["radius"],
                        "global": True,
                        "reason": "global_stagnation"
                    })
                    print(f"[PushPSO][GLOBAL] Stagnation escape: added repeller strength={rep['strength']:.3f} radius={rep['radius']:.3f} after {self.global_no_improve_iters} stagnant iters", flush=True)
                # 重新随机化一部分粒子
                k = int(self.swarm_size * self.global_stagnation_reinit_fraction)
                if k > 0:
                    idx = np.random.choice(self.swarm_size, k, replace=False)
                    if self.bounds is None:
                        self.positions[idx] = np.random.uniform(-1.0, 1.0, (k, self.dim))
                    else:
                        lo, hi = self.bounds
                        self.positions[idx] = lo + (hi - lo) * np.random.rand(k, self.dim)
                    self.velocities[idx] = 0.0
                    self.no_improve_iters[idx] = 0
                    # 重置这些粒子的个人最好
                    if self.maximize:
                        self.personal_best_fitness[idx] = -np.inf
                    else:
                        self.personal_best_fitness[idx] = np.inf
                    print(f"[PushPSO][GLOBAL] Reinitialized {k} particles to escape stagnation.", flush=True)
                self.global_no_improve_iters = 0  # 重置计数

            history.append(float(self.global_best_fitness))

            # Standard PSO velocity update
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            cognitive_term = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_term = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive_term + social_term

            # Crowding push
            if self.enable_crowding:
                stat = self._crowding_grid_push()  # only O(N) grid crowding retained
                crowd_stats.append(stat)
            else:
                stat = {}

            # Repellers (cluster detection & application)
            if (it % self.cluster_recheck_interval) == 0:
                self._detect_and_create_repellers(it)
            self._apply_repellers()

            # Velocity clamp
            if self.velocity_clamp is not None:
                vmin, vmax = self.velocity_clamp
                np.clip(self.velocities, vmin, vmax, out=self.velocities)

            # Position update
            self.positions += self.velocities
            if self.bounds is not None:
                lo, hi = self.bounds
                np.clip(self.positions, lo, hi, out=self.positions)

            # Logging
            if it % 1 == 0:
                rep_info = f"repellers={len(self.repellers)}"
                gstag = f"g_stag={self.global_no_improve_iters}/{self.global_stagnation_iters}"
                if self.enable_crowding:
                    print(f"[PushPSO] Iter {it+1}/{iter_count} best={self.global_best_fitness:.6f} "
                          f"cells={stat.get('unique_cells',0)} overcrowded={stat.get('overcrowded',0)} "
                          f"{rep_info} {gstag}",
                          flush=True)
                else:
                    print(f"[PushPSO] Iter {it+1}/{iter_count} best={self.global_best_fitness:.6f} "
                          f"{rep_info} {gstag}",
                          flush=True)

        elapsed = time.time() - start
        result = PushPSOResult(
            best_position=self.global_best_position.copy(),
            best_fitness=float(self.global_best_fitness),
            history=history,
            eval_count=eval_count,
            elapsed=elapsed,
            crowd_stats=crowd_stats,
            repeller_events=self.repeller_events.copy(),
            final_repellers=[
                {
                    "center": rep["center"].tolist(),
                    "strength": rep["strength"],
                    "radius": rep["radius"],
                    "age": rep["age"],
                    "created_iter": rep["created_iter"],
                    "origin_size": rep["origin_size"],
                }
                for rep in self.repellers
            ]
        )
        if self.save_on_exit:
            try:
                self.save_model(self.save_on_exit, include_swarm=self.include_swarm_on_save,
                                extra={"repeller_events": len(self.repeller_events)})
            except Exception as e:
                print(f"[PushPSO] Auto-save failed: {e}", flush=True)
        return result


__all__ = [
    "PushPSO",
    "PushPSOResult",
]
