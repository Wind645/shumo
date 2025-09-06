# from itertools import count  # removed unused import
from abc import abstractmethod
import numpy as np
from typing import List, cast
from rough import occluded as rough_occluded

Vec3 = np.ndarray # or None
G = 9.81
batch_calc_set = []

# ======== ALL OBJECTS ========

class Object:
    all_objects = []

    def __init__(self, pos: Vec3 = np.zeros(3)):
        # 强制转成 float，避免后续 += 浮点位移时触发 int -> float 的 UFuncOutputCastingError
        self.pos = np.array(pos, dtype=float)
        Object.all_objects.append(self)

    @abstractmethod
    def update(self, dt):
        pass

    def remove(self):
        Object.all_objects.remove(self)

class Drone(Object):
    def __init__(self, id : int, direction : Vec3, speed : float, strategy: List):
        poses = {
            1: np.array([17800, 0, 1800]),
            2: np.array([12000, 1400, 1400]),
            3: np.array([6000, -3000, 700]),
            4: np.array([11000, 2000, 1800]),
            5: np.array([13000, -2000, 1300])
        }
        super().__init__(poses[id])

        self.direction = np.array(direction, dtype=float)  # ensure float dtype; remember to make sure norm = 1 and z = 0
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
        self.occluded_time: float = 0.0

    def update(self, dt):
        self.pos += self.velocity * dt

        if self.pos[2] <= 0:
            self.remove()

class Smoke(Object):
    def __init__(self, pos: Vec3):
        super().__init__(pos)
        self.age = 0

    def update(self, dt):
        self.pos += np.array([0, 0, -3]) * dt
        self.age += dt
        if self.age >= 20:
            self.remove()
        if batch_calc_set:
            batch_calc_set.append((self.age, self.pos, ))

class Bomb(Object):
    def __init__(self, pos: Vec3, timing : float, init_velocity : Vec3):
        super().__init__(pos)
        self.timing = timing
        self.velocity = np.array(init_velocity, dtype=float)

    def update(self, dt):
        self.pos += self.velocity * dt
        self.velocity += np.array([0, 0, -G]) * dt
        if self.pos[2] <= 0:
            self.remove()
            return
        self.timing -= dt
        if self.timing <= 0:
            Smoke(self.pos.copy()) # already added to Object.all in __init__
            self.remove()


class Simulator:
    def __init__(self, problem_id : int, dt = 0.01, strategy=None):
        self.time = 0
        self.dt = dt
        self.end = False

        if problem_id == 1:
            Drone(1, np.array([1, 0, 0], dtype=float), 120.0, [[1.5, 3.6]])
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
            direction, speed, st = strategy # strategy 需要有三个烟雾弹
            Drone(1, direction, speed, st)
            Missile(1)
        elif problem_id == 4:
            if strategy is None:
                raise ValueError("strategy must be provided when problem_id == 4")
            [(direction1, speed1, st1), (direction2, speed2, st2), (direction3, speed3, st3)] = strategy
            Drone(1, direction1, speed1, st1)
            Drone(2, direction2, speed2, st2)
            Drone(3, direction3, speed3, st3)
            Missile(1)
        else:
            if strategy is None:
                raise ValueError("strategy must be provided when problem_id == 5")
            [(direction1, speed1, st1), (direction2, speed2, st2), (direction3, speed3, st3), (direction4, speed4, st4), (direction5, speed5, st5)] = strategy
            Drone(1, direction1, speed1, st1)
            Drone(2, direction2, speed2, st2)
            Drone(3, direction3, speed3, st3)
            Drone(4, direction4, speed4, st4)
            Drone(5, direction5, speed5, st5)
            Missile(1)
            Missile(2)
            Missile(3)

        # Cache missile list (assumed fixed count after init)
        self.missiles = [o for o in Object.all_objects if isinstance(o, Missile)]

        # Trajectory accumulation for batch occlusion:
        # _missile_traj: list[ list[Vec3] ] length T, inner length = M
        # _smoke_traj:   list[ list[Vec3] ] length T, inner length variable
        self._missile_traj = []
        self._smoke_traj = []
        self._max_smokes = 0  # track padding size for batch tensor

    def update(self):
        self.time += self.dt
        # Iterate over a static copy to allow safe removal inside updates
        count_missles = 0
        for each in list(Object.all_objects):
            each.update(self.dt)
            if isinstance(each, Missile):
                count_missles += 1
        # Snapshot positions for later batch occlusion computation
        smokes = [o for o in Object.all_objects if isinstance(o, Smoke)]
        # missiles list fixed -> use self.missiles
        self._missile_traj.append([np.array(cast(np.ndarray, m.pos), dtype=float) for m in self.missiles])  # type: ignore
        self._smoke_traj.append([np.array(cast(np.ndarray, s.pos), dtype=float) for s in smokes])  # type: ignore
        if len(smokes) > self._max_smokes:
            self._max_smokes = len(smokes)
        if count_missles == 0:
            print("No missiles left")
            self.end = True

    def compute_batch_occlusions(self):
        """
        一次性批量计算所有帧的导弹被烟雾遮挡总时长。
        使用 rough.occluded 在一个大张量上做一次广播，而不是逐帧调用。
        调用时机：在外部模拟主循环结束后调用。
        返回：list[float] 每个导弹的累计遮挡时长(秒)。
        """
        if not self.missiles:
            return []
        T = len(self._missile_traj)
        if T == 0:
            return [0.0] * len(self.missiles)
        M = len(self.missiles)
        Smax = self._max_smokes
        missile_arr = np.asarray(self._missile_traj, dtype=float)  # (T,M,3)
        if Smax == 0:
            # 没有烟雾
            for m in self.missiles:
                m.occluded_time = 0.0
            return [0.0] * M

        smoke_arr = np.zeros((T, Smax, 3), dtype=float)
        active_mask = np.zeros((T, Smax), dtype=bool)
        for t, smokes in enumerate(self._smoke_traj):
            for j, pos in enumerate(smokes):
                smoke_arr[t, j] = pos
                active_mask[t, j] = True

        # Batch occlusion: (T,M,1,3) vs (T,1,Smax,3)
        occ = rough_occluded(missile_arr[:, :, None, :], smoke_arr[:, None, :, :])  # (T,M,Smax)
        occ &= active_mask[:, None, :]  # mask out padded smokes
        occ_any = occ.any(axis=2)  # (T,M)
        occluded_time = occ_any.sum(axis=0) * self.dt
        for m, tval in zip(self.missiles, occluded_time):
            m.occluded_time = float(tval)
        return [float(x) for x in occluded_time]

if __name__ == "__main__":
    # 简单测试：运行第一题场景，结束后批量计算遮挡时间
    sim = Simulator(problem_id=1, dt=0.01)
    max_time = 300.0  # 安全上限
    while not sim.end and sim.time < max_time:
        sim.update()
    occluded_times = sim.compute_batch_occlusions()
    print("Occluded times per missile (s):", occluded_times)
    for idx, m in enumerate(sim.missiles):
        print(f"Missile {idx} final position={m.pos}, occluded_time={m.occluded_time:.3f}s")
