import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_dynamic_top_view():
    fig, ax = plt.subplots(figsize=(12, 9))
    fake_target = np.array([0, 0])
    real_target = np.array([0, 200])
    missiles_data = {
        'M1': {'pos': np.array([20000, 0]), 'color': 'orange'},
        'M2': {'pos': np.array([19000, 600]), 'color': 'orangered'},
        'M3': {'pos': np.array([18000, -600]), 'color': 'red'},
    }
    missile_speed = 300
    dt = 0.1
    total_time = 80
    for name, data in missiles_data.items():
        direction = fake_target - data['pos']
        distance = np.linalg.norm(direction)
        data['direction'] = direction / distance
        data['flight_time'] = distance / missile_speed
    missile_trajectories = {name: [data['pos'].copy()] for name, data in missiles_data.items()}

    def animate(frame):
        ax.clear()
        current_time = frame * dt
        ax.set_xlim(-2000, 22000); ax.set_ylim(-4000, 4000)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_aspect('equal')
        ax.set_title(f'俯视图 - 时间: {current_time:.1f}s')
        ax.grid(True, alpha=0.3)
        ax.scatter(*fake_target, color='red', s=200, marker='*')
        circle = plt.Circle(real_target, 7, color='blue', fill=False, linewidth=2)
        ax.add_patch(circle)
        for name, data in missiles_data.items():
            if current_time <= data['flight_time']:
                new_pos = data['pos'] + data['direction'] * missile_speed * current_time
                missile_trajectories[name].append(new_pos.copy())
                ax.scatter(*new_pos, color=data['color'], s=120, marker='^', edgecolors='black', linewidth=1)
                traj = np.array(missile_trajectories[name])
                ax.plot(traj[:, 0], traj[:, 1], color=data['color'], alpha=0.7, linewidth=2)
    frames = int(total_time / dt)
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=200, repeat=True)
    return fig, ani
