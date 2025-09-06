from typing_extensions import Dict
from turtledemo.clock import dtfont
from abc import abstractmethod
import numpy as np
from typing import List
from posix import DirEntry

Vec3 = np.ndarray
G = 9.81

# ======== ALL OBJECTS ========

class Object:
    all_objects = []

    def __init__(self, pos: Vec3 = np.zeros(3)):
        self.pos = pos
        Object.all_objects.append(self)

    @abstractmethod
    def update(self, dt):
        pass

class Drone(Object):
    def __init__(self, id : int, direction : Vec3, speed : float, strategy: List):
        if id == 1:
            super().__init__(np.array([17800, 0, 1800]))
        elif id == 2:
            super().__init__(np.array([12000, 1400, 1400]))
        elif id == 3:
            super().__init__(np.array([6000, -3000, 700]))
        elif id == 4:
            super().__init__(np.array([11000, 2000, 1800]))
        else: # id = 5
            super().__init__(np.array([13000, -2000, 1300]))

        self.direction = direction # remember to make sure norm = 1 and z = 0
        self.speed = speed
        self.age = 0
        self.strategy = strategy

    def update(self, dt): # strategy : [(time, timing)]
        velocity = self.direction * self.speed
        self.pos += velocity * dt
        self.age += dt
        for each in list(self.strategy):
            time, timing = each
            if abs(time - self.age) < dt:
                Bomb(self.pos.copy(), timing, velocity.copy())
                self.strategy.remove(each)

class Missile(Object):
    def __init__(self, id : int):
        if id == 1:
            super().__init__(np.array([20000, 0, 2000]))
        elif id == 2:
            super().__init__(np.array([19000, 600, 2100]))
        else: # id == 3
            super().__init__(np.array([18000, -600, 1900]))

        self.velocity = self.pos / np.linalg.norm(self.pos) * 300

    def update(self, dt):
        self.pos += self.velocity * dt

class Smoke(Object):
    def __init__(self, pos: Vec3):
        super().__init__(pos)
        self.radius = 10
        self.age = 0

    def update(self, dt):
        self.pos += np.array([0, 0, -3]) * dt
        self.age += dt
        if self.age >= 20:
            if self in Object.all_objects:
                Object.all_objects.remove(self)

class Bomb(Object):
    def __init__(self, pos: Vec3, timing : float, init_velocity : Vec3):
        super().__init__(pos)
        self.timing = timing
        self.velocity = init_velocity

    def update(self, dt):
        self.pos += self.velocity * dt
        self.velocity += np.array([0, 0, -G]) * dt
        if self.pos[2] <= 0:
            if self in Object.all_objects:
                Object.all_objects.remove(self)
            return
        self.timing -= dt
        if self.timing <= 0:
            Smoke(self.pos.copy()) # already added to Object.all in __init__
            if self in Object.all_objects:
                Object.all_objects.remove(self)

class Simulator:
    def __init__(self, problem_id : int, dt = 0.01):
        self.time = 0
        self.dt = dt

    def update(self):
        self.time += self.dt
        # Iterate over a static copy to allow safe removal inside updates
        for each in list(Object.all_objects):
            each.update(self.dt)

        # if problem_id == 1:
        #     direction = np.array([ 0.99486373, -0.01117824,  0.1006042 ])
        #     Drone(1, direction, 120, [[1.5, 3.6]])
        # elif problem_id == 2:


# =========================
# === APPENDED EXTENSIONS (Non-intrusive) FOR PROBLEM 1 ANALYSIS ===
# =========================
# The original code above is preserved verbatim. Below are helper utilities to
# run Problem 1 scenario and compute the effective遮蔽 (obscuration) time
# between missile M1 and the real target using the generated smoke cloud.

def reset_objects():
    """
    Clear all existing simulation objects so a new scenario can be cleanly executed.
    """
    Object.all_objects.clear()

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero-length vector.")
    return v / n

def distance_point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    Returns the minimal distance from point p to segment a-b and the projection parameter s in [0,1].
    """
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 == 0:
        return norm(p - a), 0.0, a
    s = np.dot(p - a, ab) / ab_len2
    s_clamped = max(0.0, min(1.0, s))
    closest = a + ab * s_clamped
    return norm(p - closest), s_clamped, closest

def setup_problem1(drone_speed: float = 120.0):
    """
    Create objects for Problem 1 as specified:
    - Drone FY1 at (17800, 0, 1800), heading horizontally toward fake target (origin projected horizontally).
    - Missile M1.
    - Drone releases bomb at t=1.5 s with a fuse time 3.6 s (explosion at 5.1 s).
    """
    reset_objects()
    # Horizontal direction: keep altitude (z=0 component in direction)
    direction = np.array([-1.0, 0.0, 0.0])  # purely toward negative x, constant altitude
    Drone(1, direction, drone_speed, [[1.5, 3.6]])
    Missile(1)

def get_objects_by_type(cls):
    return [o for o in Object.all_objects if isinstance(o, cls)]

def compute_problem1_effective_time(
    dt: float = 0.01,
    max_time: float = 26.0,
    target_center: np.ndarray = np.array([0.0, 200.0, 5.0]),
    smoke_radius: float = 10.0
):
    """
    Run the simulation loop for Problem 1 and compute the effective遮蔽 time.

    Obscuration criterion:
      The smoke sphere (center S, radius R) intersects the line-of-sight segment between
      missile M and target T (point chosen at center height). We check geometric intersection:
      distance from S to segment M-T <= R.

    Returns a dictionary with:
      effective_time: total accumulated time of LOS obstruction
      explosion_time: expected explosion time (5.1)
      sample_count: number of simulation steps
      explosion_position: bomb position at explosion moment
      dt: time step used
    """
    # Known scheduled explosion time
    explosion_time = 1.5 + 3.6  # 5.1 s
    simulator = Simulator(problem_id=1, dt=dt)

    # To capture explosion position, we monitor when a Smoke is created
    captured_explosion_pos = None
    effective_time = 0.0
    steps = 0

    # Run until either max_time or smoke fully disappeared
    while simulator.time <= max_time:
        simulator.update()
        steps += 1

        # Capture explosion position (first smoke)
        smokes = get_objects_by_type(Smoke)
        if captured_explosion_pos is None and smokes:
            # The first Smoke object center at creation is explosion point
            captured_explosion_pos = smokes[0].pos.copy()

        # Need missile and smoke for obscuration
        missiles = get_objects_by_type(Missile)
        if not missiles:
            break
        missile = missiles[0]
        if not smokes:
            continue  # no smoke yet or expired

        smoke = smokes[0]

        # Line segment M -> target_center
        dist, s, _ = distance_point_segment(smoke.pos, missile.pos, target_center)
        if dist <= smoke_radius and 0.0 <= s <= 1.0:
            effective_time += dt

        # Optional early exit: after smoke lifetime expected end (explosion_time + 20s)
        if simulator.time > explosion_time + 20.2:  # slight buffer
            break

    return {
        "effective_time": effective_time,
        "explosion_time": explosion_time,
        "sample_count": steps,
        "explosion_position": captured_explosion_pos,
        "dt": dt
    }

def run_problem1(dt: float = 0.01):
    """
    High-level convenience function:
      1. Sets up scenario
      2. Runs simulation and computes effective遮蔽 time
      3. Returns result dict
    """
    setup_problem1()
    result = compute_problem1_effective_time(dt=dt)
    return result

def format_problem1_report(result: dict) -> str:
    exp_pos = result.get("explosion_position", None)
    if exp_pos is not None:
        pos_str = f"({exp_pos[0]:.3f}, {exp_pos[1]:.3f}, {exp_pos[2]:.3f})"
    else:
        pos_str = "N/A"
    return (
        "Problem 1 Simulation Report\n"
        "---------------------------\n"
        f"Time step (dt): {result['dt']}\n"
        f"Samples: {result['sample_count']}\n"
        f"Explosion time (s): {result['explosion_time']:.3f}\n"
        f"Explosion position: {pos_str}\n"
        f"Effective LOS遮蔽 time (s): {result['effective_time']:.6f}\n"
        f"Fraction of theoretical max (20 s): {result['effective_time']/20.0*100:.3f}%\n"
    )

# Allow module to be run directly for a quick Problem 1 check without altering original logic
if __name__ == "__main__":
    res = run_problem1(dt=0.005)  # moderate resolution
    print(format_problem1_report(res))
