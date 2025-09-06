# Simulator with batch occlusion using judges.batch_rough
from abc import abstractmethod
from typing import List, cast
import numpy as np

# Import the vectorized visibility judge
from judges.batch_rough import is_sphere_blocked_vectorized

Vec3 = np.ndarray
G = 9.81

# ======== BASE OBJECT SYSTEM ========
class Object:
    all_objects = []

    def __init__(self, pos: Vec3 = np.zeros(3)):
        # Force float dtype to avoid numpy casting issues during in‑place ops
        self.pos = np.array(pos, dtype=float)
        Object.all_objects.append(self)

    @abstractmethod
    def update(self, dt: float):
        ...

    def remove(self):
        Object.all_objects.remove(self)


# ======== DOMAIN OBJECTS ========
class Drone(Object):
    def __init__(self, id: int, direction: Vec3, speed: float, strategy: List[List[float]]):
        poses = {
            1: np.array([17800, 0, 1800]),
            2: np.array([12000, 1400, 1400]),
            3: np.array([6000, -3000, 700]),
            4: np.array([11000, 2000, 1800]),
            5: np.array([13000, -2000, 1300]),
        }
        super().__init__(poses[id])
        self.direction = np.array(direction, dtype=float)  # expect normalized, z=0
        self.speed = float(speed)
        self.age = 0.0
        # strategy entries: [release_time (after start), fuse_delay]
        self.strategy = list(strategy)

    def update(self, dt: float):
        velocity = self.direction * self.speed
        self.pos += velocity * dt
        self.age += dt
        # Drop bombs whose scheduled release time is reached
        for entry in list(self.strategy):
            release_time, fuse = entry
            if self.age + 1e-9 >= release_time:
                Bomb(self.pos.copy(), fuse, velocity.copy())
                self.strategy.remove(entry)


class Missile(Object):
    def __init__(self, id: int):
        if id == 1:
            super().__init__(np.array([20000, 0, 2000]))
        elif id == 2:
            super().__init__(np.array([19000, 600, 2100]))
        else:
            super().__init__(np.array([18000, -600, 1900]))
        # Missile should move toward the false target (origin), hence negative direction
        dir_vec = self.pos / np.linalg.norm(self.pos)
        self.velocity = (-1.0) * dir_vec * 300.0
        self.occluded_time: float = 0.0

    def update(self, dt: float):
        self.pos += self.velocity * dt
        # Remove once it hits ground
        if self.pos[2] <= 0:
            self.remove()


class Smoke(Object):
    def __init__(self, pos: Vec3):
        super().__init__(pos)
        self.age = 0.0

    def update(self, dt: float):
        # Sink at 3 m/s
        self.pos[2] -= 3.0 * dt
        self.age += dt
        # Lifetime 20 s
        if self.age >= 20.0:
            self.remove()


class Bomb(Object):
    def __init__(self, pos: Vec3, timing: float, init_velocity: Vec3):
        super().__init__(pos)
        self.timing = float(timing)  # time to detonation (fuse)
        self.velocity = np.array(init_velocity, dtype=float)

    def update(self, dt: float):
        # Update position (no drag)
        self.pos += self.velocity * dt
        # Gravity on z
        self.velocity[2] -= G * dt
        # If hit ground before detonation, discard
        if self.pos[2] <= 0:
            self.remove()
            return
        self.timing -= dt
        if self.timing <= 0:
            # Spawn smoke sphere (radius handled in judge)
            Smoke(self.pos.copy())
            self.remove()


# ======== SIMULATOR ========
class Simulator:
    """
    Runs scenarios for problems 1–5 and post-processes occlusion time.

    Occlusion rule (STRICT full coverage version):
      - Each smoke is a sphere of radius 10 m, lifetime 20 s, sinking at 3 m/s.
      - Target lateral geometry is approximated with radius 7.0 m (cylinder horizontal radius),
        center at (0, 200, 5). (Previous looser outer bounding sphere logic removed.)
      - A missile is 'occluded' in a timestep ONLY if there exists at least one smoke sphere
        whose angular disc (as seen from the missile) fully covers the target's angular disc.
        Mathematically: θ_smoke >= θ_target + φ, where
            θ_smoke = arcsin(R_smoke / d_smoke),
            θ_target = arcsin(R_target / d_target),
            φ = angular separation between target center direction and smoke center direction.
      - Partial / grazing overlap no longer counts; legacy 'partial' mode has been removed.
    """

    def __init__(self, problem_id: int, dt: float = 0.01, strategy=None):
        # Reset global object registry for a clean run
        Object.all_objects = []
        self.time = 0.0
        self.dt = dt
        self.end = False

        if problem_id == 1:
            # Q1 fixed parameters: drone 1 flies toward false target (origin) along -x at 120 m/s
            # Release at 1.5 s, fuse 3.6 s
            Drone(1, np.array([-1, 0, 0], dtype=float), 120.0, [[1.5, 3.6]])
            Missile(1)
        elif problem_id == 2:
            if strategy is None:
                raise ValueError("strategy must be provided when problem_id == 2")
            direction, speed, st = strategy
            Drone(1, direction, speed, st)
            Missile(1)
        elif problem_id == 3:
            if strategy is None:
                raise ValueError("strategy must be provided when problem_id == 3")
            direction, speed, st = strategy
            Drone(1, direction, speed, st)
            Missile(1)
        elif problem_id == 4:
            if strategy is None:
                raise ValueError("strategy must be provided when problem_id == 4")
            (d1, d2, d3) = strategy
            Drone(1, *d1)
            Drone(2, *d2)
            Drone(3, *d3)
            Missile(1)
        elif problem_id == 5:
            if strategy is None:
                raise ValueError("strategy must be provided when problem_id == 5")
            (d1, d2, d3, d4, d5) = strategy
            Drone(1, *d1)
            Drone(2, *d2)
            Drone(3, *d3)
            Drone(4, *d4)
            Drone(5, *d5)
            Missile(1)
            Missile(2)
            Missile(3)
        else:
            raise ValueError("Unsupported problem_id")

        # Cache missiles (will not change in count after init)
        self.missiles = [o for o in Object.all_objects if isinstance(o, Missile)]

        # Storage for trajectories (per step snapshots)
        self._missile_traj: list[list[Vec3]] = []
        self._smoke_traj: list[list[Vec3]] = []

    def update(self):
        self.time += self.dt
        missile_count = 0
        # Iterate over copy because objects can be removed during update
        for obj in list(Object.all_objects):
            obj.update(self.dt)
            if isinstance(obj, Missile):
                missile_count += 1
        # Snapshot after updates
        smokes = [o for o in Object.all_objects if isinstance(o, Smoke)]
        self._missile_traj.append(
            [np.array(cast(np.ndarray, m.pos), dtype=float) for m in self.missiles]
        )
        self._smoke_traj.append(
            [np.array(cast(np.ndarray, s.pos), dtype=float) for s in smokes]
        )
        if missile_count == 0:
            self.end = True

    def run_until_end(self, max_time: float = 300.0):
        while not self.end and self.time < max_time:
            self.update()

    def compute_batch_occlusions(self, smoke_radius: float = 10.0):
        """
        Batch-evaluate LOS blocking for every (t, missile, smoke) combination
        in ONE call to the vectorized judge function.

        Returns list[float]: occluded seconds per missile.
        """
        if not self.missiles:
            return []

        T = len(self._missile_traj)
        if T == 0:
            return [0.0] * len(self.missiles)

        # Build batch rows: each row = [missile_xyz, smoke_xyz, R]
        rows = []
        # Mapping from (t, missile_index) -> list of row indices
        index_groups: list[list[list[int]]] = [
            [ [] for _ in range(len(self.missiles)) ] for _ in range(T)
        ]

        for t in range(T):
            missile_positions = self._missile_traj[t]
            smoke_positions = self._smoke_traj[t]
            if not smoke_positions:
                continue
            for mi, mpos in enumerate(missile_positions):
                for spos in smoke_positions:
                    idx = len(rows)
                    rows.append([mpos[0], mpos[1], mpos[2],
                                 spos[0], spos[1], spos[2],
                                 smoke_radius])
                    index_groups[t][mi].append(idx)

        if not rows:
            # No smoke at all
            for m in self.missiles:
                m.occluded_time = 0.0
            return [0.0] * len(self.missiles)

        data = np.asarray(rows, dtype=float)
        blocked = is_sphere_blocked_vectorized(data)  # boolean shape (N,)

        # Accumulate per missile per frame
        M = len(self.missiles)
        occluded_counts = np.zeros(M, dtype=int)
        for t in range(T):
            for mi in range(M):
                group = index_groups[t][mi]
                if not group:
                    continue
                # Occluded this frame if ANY associated row is True
                if blocked[group].any():
                    occluded_counts[mi] += 1

        occluded_time = occluded_counts * self.dt
        for m, tval in zip(self.missiles, occluded_time):
            m.occluded_time = float(tval)
        return [float(x) for x in occluded_time]


if __name__ == "__main__":
    # Example: run Q1 and report occlusion
    sim = Simulator(problem_id=1, dt=0.01)
    sim.run_until_end()
    times = sim.compute_batch_occlusions()
    print("Occluded times per missile (s):", times)
    for i, m in enumerate(sim.missiles, start=1):
        print(f"Missile {i}: occluded_time = {m.occluded_time:.3f} s")
