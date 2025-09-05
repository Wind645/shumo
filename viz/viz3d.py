import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def create_dynamic_simulation():
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2000, 22000); ax.set_ylim(-4000, 4000); ax.set_zlim(0, 2500)
    fake_target = np.array([0, 0, 0])
    missiles_data = {
        'M1': {'pos': np.array([20000, 0, 2000]), 'color': 'orange'},
        'M2': {'pos': np.array([19000, 600, 2100]), 'color': 'orangered'},
        'M3': {'pos': np.array([18000, -600, 1900]), 'color': 'red'},
    }
    missile_speed = 300
    dt = 0.1
    total_time = 60
    for name, data in missiles_data.items():
        direction = fake_target - data['pos']
        distance = np.linalg.norm(direction)
        data['direction'] = direction / distance
        data['flight_time'] = distance / missile_speed
    missile_trajectories = {name: [data['pos'].copy()] for name, data in missiles_data.items()}

    def animate(frame):
        ax.cla()
        ax.set_xlim(-2000, 22000); ax.set_ylim(-4000, 4000); ax.set_zlim(0, 2500)
        ax.set_title(f'3D仿真 - t={frame*dt:.1f}s')
        for name, data in missiles_data.items():
            t = frame * dt
            if t <= data['flight_time']:
                p = data['pos'] + data['direction'] * missile_speed * t
                missile_trajectories[name].append(p.copy())
                tr = np.array(missile_trajectories[name])
                ax.plot(tr[:,0], tr[:,1], tr[:,2], color=data['color'], linewidth=2)
                ax.scatter(*p, color=data['color'], s=50)
    frames = int(total_time / dt)
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=120, repeat=True)
    return fig, ani
