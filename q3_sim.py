import numpy as np
from simcore.entities import Drone, Missile
from simcore.simulator import Simulator

def simulate_q3():
    # Create FY1 drone
    direction = np.array([0, 0, -1])  # example direction
    speed = 100  # example speed
    strategy = []  # empty for now
    drone = Drone(1, direction, speed, strategy)

    # Create M1 missile
    missile = Missile(1)

    # Schedules for 3 bombs: (drone_index, deploy_time, explode_delay)
    schedules = [
        (0, 10.0, 5.0),  # bomb 1
        (0, 15.0, 5.0),  # bomb 2
        (0, 20.0, 5.0),  # bomb 3
    ]

    # Create simulator
    sim = Simulator(missile=missile, drones=[drone], schedules=schedules)

    # Run simulation
    result = sim.run(dt=0.1, verbose=True)

    print(f"Total occluded time: {result['occluded_time']}")

if __name__ == '__main__':
    simulate_q3()
